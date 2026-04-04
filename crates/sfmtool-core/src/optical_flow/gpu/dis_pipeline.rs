// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated DIS inverse search and densification.
//!
//! Three compute shaders form the DIS per-level pipeline:
//! 1. **Gradients** — central differences on the reference image
//! 2. **Inverse Search** — one thread per patch (Option B), with Hessian
//!    precomputation, iterative bilinear sampling, and outlier rejection
//! 3. **Densify** — gather-based weighted averaging (Option B), one thread per
//!    output pixel

use super::super::{DisFlowParams, FlowField, GrayImage};
use super::{
    buf_entry, create_compute_pipeline, create_pool_storage, create_pool_uniform, read_buffer,
    storage_ro_entry, storage_rw_entry, uniform_params_layout, GpuContext, WG_SIZE,
};
const GRADIENT_SHADER: &str = include_str!("shaders/compute_gradients.wgsl");
const INVERSE_SEARCH_SHADER: &str = include_str!("shaders/inverse_search.wgsl");
const DENSIFY_SHADER: &str = include_str!("shaders/densify.wgsl");

/// Workgroup size for the 1D inverse search dispatch.
const IS_WG_SIZE: u32 = 64;

// Uniform parameter structs. Must match WGSL struct layouts exactly.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GradientParams {
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct InverseSearchParams {
    pub width: u32,
    pub height: u32,
    pub patch_size: u32,
    pub stride: u32,
    pub num_patches_x: u32,
    pub num_patches_y: u32,
    pub grad_descent_iterations: u32,
    pub normalize: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DensifyParams {
    pub width: u32,
    pub height: u32,
    pub patch_size: u32,
    pub stride: u32,
    pub num_patches_x: u32,
    pub num_patches_y: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Pre-allocated GPU buffers and bind groups for DIS inverse search.
pub(super) struct DisBufferPool {
    // Data buffers (pixel-sized)
    ref_buf: wgpu::Buffer,
    tgt_buf: wgpu::Buffer,
    flow_u_buf: wgpu::Buffer,
    flow_v_buf: wgpu::Buffer,
    // These buffers are referenced by bind groups but not accessed directly.
    _grad_x_buf: wgpu::Buffer,
    _grad_y_buf: wgpu::Buffer,
    out_flow_u_buf: wgpu::Buffer,
    out_flow_v_buf: wgpu::Buffer,
    // Data buffers (patch-sized)
    _patch_flow_u_buf: wgpu::Buffer,
    _patch_flow_v_buf: wgpu::Buffer,
    // Uniform buffers
    gradient_params_buf: wgpu::Buffer,
    is_params_buf: wgpu::Buffer,
    densify_params_buf: wgpu::Buffer,
    // Bind groups
    gradient_data_bg: wgpu::BindGroup,
    gradient_params_bg: wgpu::BindGroup,
    is_data_bg: wgpu::BindGroup,
    is_params_bg: wgpu::BindGroup,
    densify_data_bg: wgpu::BindGroup,
    densify_params_bg: wgpu::BindGroup,
}

impl DisBufferPool {
    /// Access the input flow_u buffer (for creating upsample bind groups).
    pub(super) fn flow_u_buf(&self) -> &wgpu::Buffer {
        &self.flow_u_buf
    }

    /// Access the input flow_v buffer (for creating upsample bind groups).
    pub(super) fn flow_v_buf(&self) -> &wgpu::Buffer {
        &self.flow_v_buf
    }

    /// Encode GPU buffer copies of DIS output flow and input images to destination buffers.
    ///
    /// Used by the combined DIS+variational path to transfer data between buffer pools
    /// without a CPU round-trip.
    pub(super) fn encode_copies_to(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst_ref: &wgpu::Buffer,
        dst_tgt: &wgpu::Buffer,
        dst_flow_u: &wgpu::Buffer,
        dst_flow_v: &wgpu::Buffer,
        n_pixels: usize,
    ) {
        let byte_size = (n_pixels * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out_flow_u_buf, 0, dst_flow_u, 0, byte_size);
        encoder.copy_buffer_to_buffer(&self.out_flow_v_buf, 0, dst_flow_v, 0, byte_size);
        encoder.copy_buffer_to_buffer(&self.ref_buf, 0, dst_ref, 0, byte_size);
        encoder.copy_buffer_to_buffer(&self.tgt_buf, 0, dst_tgt, 0, byte_size);
    }

    fn new(
        device: &wgpu::Device,
        pipeline: &GpuDisPipeline,
        pixel_capacity: usize,
        patch_capacity: usize,
    ) -> Self {
        let pixel_bytes = (pixel_capacity * 4) as u64;
        let patch_bytes = (patch_capacity * 4) as u64;

        let ref_buf = create_pool_storage(device, "pool dis ref", pixel_bytes);
        let tgt_buf = create_pool_storage(device, "pool dis tgt", pixel_bytes);
        let flow_u_buf = create_pool_storage(device, "pool dis flow_u", pixel_bytes);
        let flow_v_buf = create_pool_storage(device, "pool dis flow_v", pixel_bytes);
        let grad_x_buf = create_pool_storage(device, "pool dis grad_x", pixel_bytes);
        let grad_y_buf = create_pool_storage(device, "pool dis grad_y", pixel_bytes);
        let out_flow_u_buf = create_pool_storage(device, "pool dis out_flow_u", pixel_bytes);
        let out_flow_v_buf = create_pool_storage(device, "pool dis out_flow_v", pixel_bytes);
        let patch_flow_u_buf = create_pool_storage(device, "pool dis patch_flow_u", patch_bytes);
        let patch_flow_v_buf = create_pool_storage(device, "pool dis patch_flow_v", patch_bytes);

        let gradient_params_buf = create_pool_uniform(device, "pool dis gradient params", 16);
        let is_params_buf = create_pool_uniform(
            device,
            "pool dis is params",
            std::mem::size_of::<InverseSearchParams>() as u64,
        );
        let densify_params_buf = create_pool_uniform(
            device,
            "pool dis densify params",
            std::mem::size_of::<DensifyParams>() as u64,
        );

        // Create bind groups
        let gradient_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool gradient data bg"),
            layout: &pipeline.gradient_data_layout,
            entries: &[
                buf_entry(0, &ref_buf),
                buf_entry(1, &grad_x_buf),
                buf_entry(2, &grad_y_buf),
            ],
        });
        let gradient_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool gradient params bg"),
            layout: &pipeline.gradient_params_layout,
            entries: &[buf_entry(0, &gradient_params_buf)],
        });

        let is_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool is data bg"),
            layout: &pipeline.inverse_search_data_layout,
            entries: &[
                buf_entry(0, &ref_buf),
                buf_entry(1, &tgt_buf),
                buf_entry(2, &grad_x_buf),
                buf_entry(3, &grad_y_buf),
                buf_entry(4, &flow_u_buf),
                buf_entry(5, &flow_v_buf),
                buf_entry(6, &patch_flow_u_buf),
                buf_entry(7, &patch_flow_v_buf),
            ],
        });
        let is_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool is params bg"),
            layout: &pipeline.inverse_search_params_layout,
            entries: &[buf_entry(0, &is_params_buf)],
        });

        let densify_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool densify data bg"),
            layout: &pipeline.densify_data_layout,
            entries: &[
                buf_entry(0, &ref_buf),
                buf_entry(1, &tgt_buf),
                buf_entry(2, &patch_flow_u_buf),
                buf_entry(3, &patch_flow_v_buf),
                buf_entry(4, &out_flow_u_buf),
                buf_entry(5, &out_flow_v_buf),
            ],
        });
        let densify_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool densify params bg"),
            layout: &pipeline.densify_params_layout,
            entries: &[buf_entry(0, &densify_params_buf)],
        });

        Self {
            ref_buf,
            tgt_buf,
            flow_u_buf,
            flow_v_buf,
            _grad_x_buf: grad_x_buf,
            _grad_y_buf: grad_y_buf,
            out_flow_u_buf,
            out_flow_v_buf,
            _patch_flow_u_buf: patch_flow_u_buf,
            _patch_flow_v_buf: patch_flow_v_buf,
            gradient_params_buf,
            is_params_buf,
            densify_params_buf,
            gradient_data_bg,
            gradient_params_bg,
            is_data_bg,
            is_params_bg,
            densify_data_bg,
            densify_params_bg,
        }
    }
}

/// GPU compute pipelines for DIS inverse search and densification.
///
/// Create once, reuse across calls. Pipelines are independent of image dimensions.
/// Buffers are allocated per-call and passed through the pipeline.
pub(crate) struct GpuDisPipeline {
    gradient_pipeline: wgpu::ComputePipeline,
    gradient_data_layout: wgpu::BindGroupLayout,
    pub(super) gradient_params_layout: wgpu::BindGroupLayout,

    inverse_search_pipeline: wgpu::ComputePipeline,
    inverse_search_data_layout: wgpu::BindGroupLayout,
    pub(super) inverse_search_params_layout: wgpu::BindGroupLayout,

    densify_pipeline: wgpu::ComputePipeline,
    densify_data_layout: wgpu::BindGroupLayout,
    pub(super) densify_params_layout: wgpu::BindGroupLayout,
}

impl GpuDisPipeline {
    /// Create all GPU compute pipelines for DIS.
    pub(crate) fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        // --- Gradient pipeline ---
        let gradient_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gradient data"),
                entries: &[
                    storage_ro_entry(0), // image
                    storage_rw_entry(1), // grad_x
                    storage_rw_entry(2), // grad_y
                ],
            });
        let gradient_params_layout = uniform_params_layout(device, "gradient params");
        let gradient_pipeline = create_compute_pipeline(
            device,
            GRADIENT_SHADER,
            &[&gradient_data_layout, &gradient_params_layout],
        );

        // --- Inverse search pipeline ---
        let inverse_search_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inverse search data"),
                entries: &[
                    storage_ro_entry(0), // ref_image
                    storage_ro_entry(1), // tgt_image
                    storage_ro_entry(2), // grad_x
                    storage_ro_entry(3), // grad_y
                    storage_ro_entry(4), // flow_u
                    storage_ro_entry(5), // flow_v
                    storage_rw_entry(6), // patch_flow_u
                    storage_rw_entry(7), // patch_flow_v
                ],
            });
        let inverse_search_params_layout = uniform_params_layout(device, "inverse search params");
        let inverse_search_pipeline = create_compute_pipeline(
            device,
            INVERSE_SEARCH_SHADER,
            &[&inverse_search_data_layout, &inverse_search_params_layout],
        );

        // --- Densify pipeline ---
        let densify_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("densify data"),
                entries: &[
                    storage_ro_entry(0), // ref_image
                    storage_ro_entry(1), // tgt_image
                    storage_ro_entry(2), // patch_flow_u
                    storage_ro_entry(3), // patch_flow_v
                    storage_rw_entry(4), // flow_u (output)
                    storage_rw_entry(5), // flow_v (output)
                ],
            });
        let densify_params_layout = uniform_params_layout(device, "densify params");
        let densify_pipeline = create_compute_pipeline(
            device,
            DENSIFY_SHADER,
            &[&densify_data_layout, &densify_params_layout],
        );

        Self {
            gradient_pipeline,
            gradient_data_layout,
            gradient_params_layout,
            inverse_search_pipeline,
            inverse_search_data_layout,
            inverse_search_params_layout,
            densify_pipeline,
            densify_data_layout,
            densify_params_layout,
        }
    }

    /// Create a buffer pool with enough capacity for the given pixel and patch counts.
    pub(super) fn create_pool(
        &self,
        device: &wgpu::Device,
        n: usize,
        total_patches: usize,
    ) -> DisBufferPool {
        DisBufferPool::new(device, self, n, total_patches)
    }

    /// Upload input data and encode DIS compute commands into the given encoder.
    ///
    /// The pool must already be created via [`create_pool`]. This encodes gradient
    /// computation, inverse search, and densification dispatches but does NOT submit
    /// or read back results.
    ///
    /// When `flow` is `Some`, the flow field is uploaded from CPU. When `None`, the
    /// flow is assumed to already be in the pool's flow buffers (e.g., from a GPU
    /// upsample dispatch).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upload_and_encode(
        &self,
        pool: &DisBufferPool,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ref_image: &GrayImage,
        tgt_image: &GrayImage,
        flow: Option<&FlowField>,
        params: &DisFlowParams,
    ) {
        let w = ref_image.width();
        let h = ref_image.height();
        let ps = params.patch_size;
        let stride = params.patch_stride();

        let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
        let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };
        if num_patches_x == 0 || num_patches_y == 0 {
            return;
        }
        let total_patches = (num_patches_x * num_patches_y) as usize;

        // Upload data into pooled buffers
        queue.write_buffer(&pool.ref_buf, 0, bytemuck::cast_slice(ref_image.data()));
        queue.write_buffer(&pool.tgt_buf, 0, bytemuck::cast_slice(tgt_image.data()));
        if let Some(flow) = flow {
            queue.write_buffer(&pool.flow_u_buf, 0, bytemuck::cast_slice(flow.u_slice()));
            queue.write_buffer(&pool.flow_v_buf, 0, bytemuck::cast_slice(flow.v_slice()));
        }

        // Update uniform params
        queue.write_buffer(
            &pool.gradient_params_buf,
            0,
            bytemuck::bytes_of(&GradientParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &pool.is_params_buf,
            0,
            bytemuck::bytes_of(&InverseSearchParams {
                width: w,
                height: h,
                patch_size: ps,
                stride,
                num_patches_x,
                num_patches_y,
                grad_descent_iterations: params.grad_descent_iterations,
                normalize: u32::from(params.normalize_patches),
            }),
        );
        queue.write_buffer(
            &pool.densify_params_buf,
            0,
            bytemuck::bytes_of(&DensifyParams {
                width: w,
                height: h,
                patch_size: ps,
                stride,
                num_patches_x,
                num_patches_y,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        // Encode compute passes
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DIS inverse search"),
                timestamp_writes: None,
            });

            let wg_x = w.div_ceil(WG_SIZE);
            let wg_y = h.div_ceil(WG_SIZE);
            pass.set_pipeline(&self.gradient_pipeline);
            pass.set_bind_group(0, Some(&pool.gradient_data_bg), &[]);
            pass.set_bind_group(1, Some(&pool.gradient_params_bg), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);

            let is_workgroups = (total_patches as u32).div_ceil(IS_WG_SIZE);
            pass.set_pipeline(&self.inverse_search_pipeline);
            pass.set_bind_group(0, Some(&pool.is_data_bg), &[]);
            pass.set_bind_group(1, Some(&pool.is_params_bg), &[]);
            pass.dispatch_workgroups(is_workgroups, 1, 1);

            pass.set_pipeline(&self.densify_pipeline);
            pass.set_bind_group(0, Some(&pool.densify_data_bg), &[]);
            pass.set_bind_group(1, Some(&pool.densify_params_bg), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }

    /// Copy images from GPU pyramid buffers and encode DIS compute commands
    /// using externally-provided params bind groups.
    ///
    /// This method does NOT write uniform
    /// params via `queue.write_buffer`. The caller must write params to the
    /// per-level uniform buffers before the single `queue.submit()`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_pyramid_and_encode_with_bind_groups(
        &self,
        pool: &DisBufferPool,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ref_pyr_buf: &wgpu::Buffer,
        tgt_pyr_buf: &wgpu::Buffer,
        w: u32,
        h: u32,
        flow: Option<&FlowField>,
        params: &DisFlowParams,
        gradient_params_bg: &wgpu::BindGroup,
        is_params_bg: &wgpu::BindGroup,
        densify_params_bg: &wgpu::BindGroup,
    ) {
        let ps = params.patch_size;
        let stride = params.patch_stride();

        let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
        let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };
        if num_patches_x == 0 || num_patches_y == 0 {
            return;
        }
        let total_patches = (num_patches_x * num_patches_y) as usize;
        let n = (w as usize) * (h as usize);
        let byte_size = (n * 4) as u64;

        // GPU-side copy of images from pyramid buffers
        encoder.copy_buffer_to_buffer(ref_pyr_buf, 0, &pool.ref_buf, 0, byte_size);
        encoder.copy_buffer_to_buffer(tgt_pyr_buf, 0, &pool.tgt_buf, 0, byte_size);

        if let Some(flow) = flow {
            queue.write_buffer(&pool.flow_u_buf, 0, bytemuck::cast_slice(flow.u_slice()));
            queue.write_buffer(&pool.flow_v_buf, 0, bytemuck::cast_slice(flow.v_slice()));
        }

        // Encode compute passes (using caller-provided params bind groups)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DIS inverse search"),
                timestamp_writes: None,
            });

            let wg_x = w.div_ceil(WG_SIZE);
            let wg_y = h.div_ceil(WG_SIZE);
            pass.set_pipeline(&self.gradient_pipeline);
            pass.set_bind_group(0, Some(&pool.gradient_data_bg), &[]);
            pass.set_bind_group(1, Some(gradient_params_bg), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);

            let is_workgroups = (total_patches as u32).div_ceil(IS_WG_SIZE);
            pass.set_pipeline(&self.inverse_search_pipeline);
            pass.set_bind_group(0, Some(&pool.is_data_bg), &[]);
            pass.set_bind_group(1, Some(is_params_bg), &[]);
            pass.dispatch_workgroups(is_workgroups, 1, 1);

            pass.set_pipeline(&self.densify_pipeline);
            pass.set_bind_group(0, Some(&pool.densify_data_bg), &[]);
            pass.set_bind_group(1, Some(densify_params_bg), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }

    /// Run GPU-accelerated DIS inverse search and densification (standalone).
    ///
    /// Replaces the CPU gradient computation, inverse search, and densification
    /// stages of `refine_flow_at_level`. The flow field is updated in place with
    /// the dense result. Variational refinement is NOT included here.
    pub(crate) fn run(
        &self,
        ctx: &GpuContext,
        ref_image: &GrayImage,
        tgt_image: &GrayImage,
        flow: &mut FlowField,
        params: &DisFlowParams,
    ) {
        let w = ref_image.width();
        let h = ref_image.height();
        let ps = params.patch_size;
        let stride = params.patch_stride();

        let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
        let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };
        if num_patches_x == 0 || num_patches_y == 0 {
            return;
        }
        let total_patches = (num_patches_x * num_patches_y) as usize;
        let n = (w as usize) * (h as usize);

        let device = &ctx.device;
        let queue = &ctx.queue;

        let pool = self.create_pool(device, n, total_patches);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.upload_and_encode(
            &pool,
            queue,
            &mut encoder,
            ref_image,
            tgt_image,
            Some(flow),
            params,
        );

        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read back the dense flow
        let flow_u_data = read_buffer(device, queue, &pool.out_flow_u_buf, n);
        let flow_v_data = read_buffer(device, queue, &pool.out_flow_v_buf, n);

        flow.u_slice_mut().copy_from_slice(&flow_u_data);
        flow.v_slice_mut().copy_from_slice(&flow_v_data);
    }
}