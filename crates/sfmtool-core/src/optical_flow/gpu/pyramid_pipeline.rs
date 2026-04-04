// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated Gaussian pyramid construction.
//!
//! Two-pass separable convolution: horizontal blur+downsample, then vertical
//! blur+downsample. Each pyramid level is built by dispatching both passes.
//! All levels can be built in a single command submission.

use super::{
    buf_entry, create_pool_storage, create_pool_uniform, storage_ro_entry, storage_rw_entry,
    uniform_params_layout, GpuContext, WG_SIZE,
};

const BLUR_DOWNSAMPLE_SHADER: &str = include_str!("shaders/blur_downsample.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

/// Pre-allocated GPU buffers for pyramid construction.
///
/// We need buffers for:
/// - Each pyramid level for both images (ref and tgt)
/// - An intermediate buffer for the horizontal pass output
/// - A params uniform buffer
pub(crate) struct PyramidBufferPool {
    /// Per-level buffers for the reference image pyramid.
    /// Level 0 is full resolution, level i is 2^i downsampled.
    pub(super) ref_level_bufs: Vec<wgpu::Buffer>,
    /// Per-level buffers for the target image pyramid.
    pub(super) tgt_level_bufs: Vec<wgpu::Buffer>,
    /// Intermediate buffer for horizontal pass output (needs to hold up to
    /// full_height * half_width pixels — the largest intermediate size).
    intermediate_buf: wgpu::Buffer,
    /// Uniform params buffer for the horizontal pass.
    horiz_params_buf: wgpu::Buffer,
    /// Uniform params buffer for the vertical pass.
    vert_params_buf: wgpu::Buffer,
}

impl PyramidBufferPool {
    fn new(device: &wgpu::Device, full_w: u32, full_h: u32, num_levels: usize) -> Self {
        let mut ref_level_bufs = Vec::with_capacity(num_levels);
        let mut tgt_level_bufs = Vec::with_capacity(num_levels);

        let mut w = full_w;
        let mut h = full_h;
        let mut max_intermediate = 0usize;

        for i in 0..num_levels {
            let n = (w as usize) * (h as usize);
            let byte_size = (n * 4) as u64;
            ref_level_bufs.push(create_pool_storage(
                device,
                &format!("pool pyr ref L{i}"),
                byte_size,
            ));
            tgt_level_bufs.push(create_pool_storage(
                device,
                &format!("pool pyr tgt L{i}"),
                byte_size,
            ));

            if i + 1 < num_levels {
                // Intermediate after horizontal pass: (w/2) * h
                let inter_n = (w as usize / 2) * (h as usize);
                max_intermediate = max_intermediate.max(inter_n);
            }

            w /= 2;
            h /= 2;
        }

        let intermediate_buf = create_pool_storage(
            device,
            "pool pyr intermediate",
            (max_intermediate.max(1) * 4) as u64,
        );

        let param_size = std::mem::size_of::<BlurParams>() as u64;
        let horiz_params_buf = create_pool_uniform(device, "pool pyr horiz params", param_size);
        let vert_params_buf = create_pool_uniform(device, "pool pyr vert params", param_size);

        Self {
            ref_level_bufs,
            tgt_level_bufs,
            intermediate_buf,
            horiz_params_buf,
            vert_params_buf,
        }
    }
}

/// GPU compute pipelines for Gaussian pyramid construction.
pub(crate) struct GpuPyramidPipeline {
    horiz_pipeline: wgpu::ComputePipeline,
    vert_pipeline: wgpu::ComputePipeline,
    data_layout: wgpu::BindGroupLayout,
    params_layout: wgpu::BindGroupLayout,
}

impl GpuPyramidPipeline {
    pub(crate) fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur downsample data"),
            entries: &[
                storage_ro_entry(0), // input
                storage_rw_entry(1), // output
            ],
        });
        let params_layout = uniform_params_layout(device, "blur downsample params");

        // Two entry points in the same shader module
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur_downsample"),
            source: wgpu::ShaderSource::Wgsl(BLUR_DOWNSAMPLE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&data_layout, &params_layout],
            push_constant_ranges: &[],
        });

        let horiz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur horiz"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("horiz"),
            compilation_options: Default::default(),
            cache: None,
        });
        let vert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur vert"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("vert"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            horiz_pipeline,
            vert_pipeline,
            data_layout,
            params_layout,
        }
    }

    /// Create a buffer pool for the given full resolution and number of levels.
    pub(super) fn create_pool(
        &self,
        device: &wgpu::Device,
        full_w: u32,
        full_h: u32,
        num_levels: usize,
    ) -> PyramidBufferPool {
        PyramidBufferPool::new(device, full_w, full_h, num_levels)
    }

    /// Upload full-resolution images and build pyramid levels on GPU.
    ///
    /// Uploads level-0 images via `queue.write_buffer`, then builds each subsequent
    /// level with a submit+wait per level (params differ between levels).
    pub(super) fn upload_and_build(
        &self,
        pool: &PyramidBufferPool,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ref_image: &super::super::GrayImage,
        tgt_image: &super::super::GrayImage,
        num_levels: usize,
    ) {
        // Upload level 0 (full resolution)
        queue.write_buffer(
            &pool.ref_level_bufs[0],
            0,
            bytemuck::cast_slice(ref_image.data()),
        );
        queue.write_buffer(
            &pool.tgt_level_bufs[0],
            0,
            bytemuck::cast_slice(tgt_image.data()),
        );

        let mut w = ref_image.width();
        let mut h = ref_image.height();

        // Build each subsequent level. Ref and tgt at the same level share
        // identical dimensions, so they can share a single params write and
        // submission. Different levels need separate submissions because
        // queue.write_buffer is staged for the next submit.
        for level in 1..num_levels {
            let out_w = w / 2;
            let out_h = h / 2;

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // Build ref and tgt pyramid levels (same dimensions, same params)
            self.encode_one_level(
                pool,
                device,
                queue,
                &mut encoder,
                &pool.ref_level_bufs[level - 1],
                &pool.ref_level_bufs[level],
                w,
                h,
                out_w,
                out_h,
            );

            self.encode_one_level(
                pool,
                device,
                queue,
                &mut encoder,
                &pool.tgt_level_bufs[level - 1],
                &pool.tgt_level_bufs[level],
                w,
                h,
                out_w,
                out_h,
            );

            queue.submit(Some(encoder.finish()));
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            w = out_w;
            h = out_h;
        }
    }

    /// Encode horizontal + vertical blur+downsample for one pyramid level.
    #[allow(clippy::too_many_arguments)]
    fn encode_one_level(
        &self,
        pool: &PyramidBufferPool,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        in_w: u32,
        in_h: u32,
        out_w: u32,
        out_h: u32,
    ) {
        // --- Horizontal pass ---
        // Input: in_w × in_h, Output: out_w × in_h (into intermediate)
        queue.write_buffer(
            &pool.horiz_params_buf,
            0,
            bytemuck::bytes_of(&BlurParams {
                in_width: in_w,
                in_height: in_h,
                out_width: out_w,
                out_height: in_h, // not used by horiz, but kept consistent
            }),
        );

        // --- Vertical pass ---
        // Input: out_w × in_h (intermediate), Output: out_w × out_h
        queue.write_buffer(
            &pool.vert_params_buf,
            0,
            bytemuck::bytes_of(&BlurParams {
                in_width: out_w,
                in_height: in_h,
                out_width: out_w,
                out_height: out_h,
            }),
        );

        let horiz_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyr horiz data"),
            layout: &self.data_layout,
            entries: &[
                buf_entry(0, input_buf),
                buf_entry(1, &pool.intermediate_buf),
            ],
        });
        let horiz_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyr horiz params"),
            layout: &self.params_layout,
            entries: &[buf_entry(0, &pool.horiz_params_buf)],
        });

        let vert_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyr vert data"),
            layout: &self.data_layout,
            entries: &[
                buf_entry(0, &pool.intermediate_buf),
                buf_entry(1, output_buf),
            ],
        });
        let vert_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyr vert params"),
            layout: &self.params_layout,
            entries: &[buf_entry(0, &pool.vert_params_buf)],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pyramid horiz"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.horiz_pipeline);
            pass.set_bind_group(0, Some(&horiz_data_bg), &[]);
            pass.set_bind_group(1, Some(&horiz_params_bg), &[]);
            pass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), in_h.div_ceil(WG_SIZE), 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pyramid vert"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vert_pipeline);
            pass.set_bind_group(0, Some(&vert_data_bg), &[]);
            pass.set_bind_group(1, Some(&vert_params_bg), &[]);
            pass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
        }
    }
}