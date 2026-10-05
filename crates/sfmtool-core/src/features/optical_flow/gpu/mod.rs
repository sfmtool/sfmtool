// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated DIS optical flow using wgpu compute shaders.
//!
//! This module runs the per-level stages of DIS optical flow on the GPU: the
//! Gaussian pyramid, DIS inverse search and densification, variational
//! refinement, and the 2x flow upsampling between levels and to full
//! resolution. The variational Jacobi solver is the largest CPU cost (~57% of
//! per-scale CPU time) and maps directly to GPU compute: each pixel is
//! independent within an iteration, with simple stencil access patterns.
//!
//! # Architecture
//!
//! The pipeline is split across sibling modules:
//! - `context` — wgpu plumbing: the [`GpuContext`], POD uniform-parameter
//!   structs, and bind-group / pipeline / buffer helpers.
//! - `variational` — the four-shader variational refinement pipeline
//!   ([`GpuVariationalRefiner`]) and its buffer pool.
//! - `dis_pipeline` / `pyramid_pipeline` — DIS inverse search + densification
//!   and image-pyramid construction.
//!
//! [`GpuFlowContext`] (below) bundles them and drives the per-level DIS +
//! variational orchestration.

mod context;
mod dis_pipeline;
mod pyramid_pipeline;
mod variational;

use super::variational::VariationalParams;

pub use context::GpuContext;
pub use variational::GpuVariationalRefiner;

pub(crate) use dis_pipeline::GpuDisPipeline;
pub(crate) use pyramid_pipeline::{GpuPyramidPipeline, PyramidBufferPool};

use context::{
    buf_entry, create_compute_pipeline, create_pool_storage, create_pool_uniform, read_buffer,
    storage_ro_entry, storage_rw_entry, uniform_params_layout, CoeffParams, JacobiParams,
    UpdateParams, UpsampleParams, WarpParams, WG_SIZE,
};

const UPSAMPLE_SHADER: &str = include_str!("shaders/upsample_flow.wgsl");

/// Bundled GPU context and pipelines for optical flow computation.
///
/// Create once at startup via [`GpuFlowContext::new()`], then pass as
/// `Option<&GpuFlowContext>` to [`compute_optical_flow`](super::compute_optical_flow).
/// When present, the variational refinement stage runs on the GPU.
pub struct GpuFlowContext {
    ctx: GpuContext,
    refiner: GpuVariationalRefiner,
    dis_pipeline: GpuDisPipeline,
    pyramid_pipeline: GpuPyramidPipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    upsample_data_layout: wgpu::BindGroupLayout,
    upsample_params_layout: wgpu::BindGroupLayout,
}

impl GpuFlowContext {
    /// Create a GPU flow context. Returns `None` if no GPU is available.
    pub fn new() -> Option<Self> {
        let ctx = GpuContext::new()?;
        let refiner = GpuVariationalRefiner::new(&ctx);
        let dis_pipeline = GpuDisPipeline::new(&ctx);
        let pyramid_pipeline = GpuPyramidPipeline::new(&ctx);

        let device = &ctx.device;

        // Upsample pipeline
        let upsample_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("upsample flow data"),
                entries: &[
                    storage_ro_entry(0), // flow_u_in
                    storage_ro_entry(1), // flow_v_in
                    storage_rw_entry(2), // flow_u_out
                    storage_rw_entry(3), // flow_v_out
                ],
            });
        let upsample_params_layout = uniform_params_layout(device, "upsample flow params");
        let upsample_pipeline = create_compute_pipeline(
            device,
            UPSAMPLE_SHADER,
            &[&upsample_data_layout, &upsample_params_layout],
        );
        Some(Self {
            ctx,
            refiner,
            dis_pipeline,
            pyramid_pipeline,
            upsample_pipeline,
            upsample_data_layout,
            upsample_params_layout,
        })
    }

    /// Run DIS and variational refinement for one pyramid level as a single GPU
    /// submission, updating `flow` in place.
    ///
    /// `flow` is left unchanged when the level is smaller than one patch or
    /// than 3 pixels on a side. See "Minimizing CPU↔GPU Transfers" in
    /// `specs/core/features/gpu-optical-flow.md`.
    pub(crate) fn run_dis_and_variational(
        &self,
        ref_image: &super::GrayImage,
        tgt_image: &super::GrayImage,
        flow: &mut super::FlowField,
        dis_params: &super::DisFlowParams,
        var_params: &super::variational::VariationalParams,
    ) {
        let w = ref_image.width();
        let h = ref_image.height();
        let n = (w as usize) * (h as usize);

        let ps = dis_params.patch_size;
        let stride = dis_params.patch_stride();
        let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
        let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };
        if num_patches_x == 0 || num_patches_y == 0 {
            return;
        }
        let total_patches = (num_patches_x * num_patches_y) as usize;

        if w < 3 || h < 3 {
            return;
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let dis_pool = self.dis_pipeline.create_pool(device, n, total_patches);
        let var_pool = self.refiner.create_pool(device, n);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Upload input data and encode DIS commands
        self.dis_pipeline.upload_and_encode(
            &dis_pool,
            queue,
            &mut encoder,
            ref_image,
            tgt_image,
            Some(flow),
            dis_params,
        );

        // GPU-side copy: DIS outputs → variational inputs (no CPU round-trip)
        dis_pool.encode_copies_to(
            &mut encoder,
            &var_pool.ref_buf,
            &var_pool.tgt_buf,
            &var_pool.flow_u_buf,
            &var_pool.flow_v_buf,
            n,
        );

        // Upload variational params and encode variational commands
        self.refiner
            .encode_variational(&var_pool, queue, &mut encoder, w, h, var_params);

        // Single submit + single sync
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Single readback of the final refined flow
        let flow_u_data = read_buffer(device, queue, &var_pool.flow_u_buf, n);
        let flow_v_data = read_buffer(device, queue, &var_pool.flow_v_buf, n);

        flow.u_slice_mut().copy_from_slice(&flow_u_data);
        flow.v_slice_mut().copy_from_slice(&flow_v_data);
    }

    /// Run GPU-accelerated DIS inverse search and densification.
    ///
    /// Replaces the CPU gradient computation, inverse search, and densification.
    /// The flow field is updated in place with the dense result.
    pub(crate) fn run_dis_level(
        &self,
        ref_image: &super::GrayImage,
        tgt_image: &super::GrayImage,
        flow: &mut super::FlowField,
        params: &super::DisFlowParams,
    ) {
        self.dis_pipeline
            .run(&self.ctx, ref_image, tgt_image, flow, params);
    }

    /// Build GPU pyramids and optionally read back a level for CPU seeding.
    ///
    /// Returns `(pyramid_pool, seed_images)`. The pyramid pool must be passed
    /// to [`Self::run_gpu_levels_prebuilt`]. Seed images are returned if
    /// `cpu_seed_level` is provided.
    #[allow(clippy::type_complexity)]
    pub(crate) fn build_gpu_pyramid(
        &self,
        ref_image: &super::GrayImage,
        tgt_image: &super::GrayImage,
        num_pyramid_levels: usize,
        cpu_seed_level: Option<(usize, u32, u32)>,
    ) -> (
        PyramidBufferPool,
        Option<(super::GrayImage, super::GrayImage)>,
    ) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let pyr_pool = self.pyramid_pipeline.create_pool(
            device,
            ref_image.width(),
            ref_image.height(),
            num_pyramid_levels,
        );

        self.pyramid_pipeline.upload_and_build(
            &pyr_pool,
            device,
            queue,
            ref_image,
            tgt_image,
            num_pyramid_levels,
        );

        let seed_images = cpu_seed_level.map(|(level, w, h)| {
            let n = (w as usize) * (h as usize);
            let ref_data = read_buffer(device, queue, &pyr_pool.ref_level_bufs[level], n);
            let tgt_data = read_buffer(device, queue, &pyr_pool.tgt_level_bufs[level], n);
            (
                super::GrayImage::new(w, h, ref_data),
                super::GrayImage::new(w, h, tgt_data),
            )
        });

        (pyr_pool, seed_images)
    }

    /// Process GPU levels using a pre-built GPU pyramid, with optional final
    /// upsample to full resolution.
    ///
    /// The GPU pyramid must have been built by a prior call to
    /// [`Self::build_gpu_pyramid`].
    /// `gpu_scales` contains `(scale_index, width, height)` in coarse-to-fine order.
    ///
    /// If `final_upsample_steps > 0`, the flow is upsampled on the GPU by 2x that
    /// many times before readback, avoiding the costly CPU upsample. The returned
    /// flow will be at the upsampled resolution (not resized to exact target dims —
    /// the caller should apply `resize_flow_to` if needed).
    ///
    /// All uniform buffers and bind groups are created before encoding, so every
    /// level and the final upsample go into one command buffer with a single
    /// submit and wait, followed by one readback.
    pub(crate) fn run_gpu_levels_prebuilt(
        &self,
        pyr_pool: &PyramidBufferPool,
        gpu_scales: &[(u32, u32, u32)],
        flow: super::FlowField,
        params: &super::DisFlowParams,
        final_upsample_steps: u32,
    ) -> super::FlowField {
        if gpu_scales.is_empty() {
            return flow;
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Find maximum pixel and patch counts across GPU levels for pool allocation.
        let max_n = gpu_scales
            .iter()
            .map(|&(_, w, h)| (w as usize) * (h as usize))
            .max()
            .unwrap();
        let max_patches = gpu_scales
            .iter()
            .map(|&(_, w, h)| {
                let (npx, npy) = patch_grid(w, h, params);
                (npx * npy) as usize
            })
            .max()
            .unwrap();

        // Create DIS and variational buffer pools for this call.
        let dis_pool = self.dis_pipeline.create_pool(device, max_n, max_patches);
        let var_pool = self.refiner.create_pool(device, max_n);

        // Inter-level upsample: variational output flow → DIS input flow (shared data bg).
        let upsample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upsample flow bg"),
            layout: &self.upsample_data_layout,
            entries: &[
                buf_entry(0, &var_pool.flow_u_buf),
                buf_entry(1, &var_pool.flow_v_buf),
                buf_entry(2, dis_pool.flow_u_buf()),
                buf_entry(3, dis_pool.flow_v_buf()),
            ],
        });

        let level_bgs: Vec<LevelBindGroups> = gpu_scales
            .iter()
            .enumerate()
            .map(|(i, &(_, w, h))| {
                let prev_dims = (i > 0).then(|| {
                    let (_, prev_w, prev_h) = gpu_scales[i - 1];
                    (prev_w, prev_h)
                });
                self.create_level_bind_groups(w, h, prev_dims, params)
            })
            .collect();

        let (_, last_w, last_h) = *gpu_scales.last().unwrap();
        let final_upsample =
            self.create_final_upsample(&var_pool, last_w, last_h, final_upsample_steps);

        // Encode all levels + final upsample into a single command buffer.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.encode_levels(
            &mut encoder,
            pyr_pool,
            &dis_pool,
            &var_pool,
            &upsample_bg,
            &level_bgs,
            gpu_scales,
            &flow,
            params,
        );
        self.encode_final_upsample(&mut encoder, &final_upsample.steps);

        // Single submit and wait for all levels + final upsample.
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        self.read_back_final_flow(&var_pool, &final_upsample, last_w, last_h)
    }

    /// Create the uniform buffers for one GPU level, write its parameters, and
    /// build the bind groups that reference them.
    ///
    /// `prev_dims` is the previous (coarser) GPU level's `(width, height)`, or
    /// `None` for the first GPU level, which has no inter-level upsample.
    fn create_level_bind_groups(
        &self,
        w: u32,
        h: u32,
        prev_dims: Option<(u32, u32)>,
        params: &super::DisFlowParams,
    ) -> LevelBindGroups {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let ps = params.patch_size;
        let stride = params.patch_stride();
        let (num_patches_x, num_patches_y) = patch_grid(w, h, params);

        // DIS uniform buffers
        let gradient_buf = create_pool_uniform(device, "lvl gradient params", 16);
        queue.write_buffer(
            &gradient_buf,
            0,
            bytemuck::bytes_of(&dis_pipeline::GradientParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let is_buf = create_pool_uniform(
            device,
            "lvl is params",
            std::mem::size_of::<dis_pipeline::InverseSearchParams>() as u64,
        );
        queue.write_buffer(
            &is_buf,
            0,
            bytemuck::bytes_of(&dis_pipeline::InverseSearchParams {
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
        let densify_buf = create_pool_uniform(
            device,
            "lvl densify params",
            std::mem::size_of::<dis_pipeline::DensifyParams>() as u64,
        );
        queue.write_buffer(
            &densify_buf,
            0,
            bytemuck::bytes_of(&dis_pipeline::DensifyParams {
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

        // Variational uniform buffers. The outer iteration count is passed to
        // the encoder, not stored in a uniform.
        let warp_buf = create_pool_uniform(device, "lvl warp params", 16);
        queue.write_buffer(
            &warp_buf,
            0,
            bytemuck::bytes_of(&WarpParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let coeff_buf = create_pool_uniform(device, "lvl coeff params", 16);
        queue.write_buffer(
            &coeff_buf,
            0,
            bytemuck::bytes_of(&CoeffParams {
                width: w,
                height: h,
                delta: params.variational_delta,
                gamma: params.variational_gamma,
            }),
        );
        let jacobi_buf = create_pool_uniform(device, "lvl jacobi params", 16);
        queue.write_buffer(
            &jacobi_buf,
            0,
            bytemuck::bytes_of(&JacobiParams {
                width: w,
                height: h,
                alpha: params.variational_alpha,
                _pad: 0,
            }),
        );
        let update_buf = create_pool_uniform(device, "lvl update params", 16);
        queue.write_buffer(
            &update_buf,
            0,
            bytemuck::bytes_of(&UpdateParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        // Upsample uniform buffer (for levels after first)
        let upsample_params_bg = prev_dims.map(|(prev_w, prev_h)| {
            let up_buf = create_pool_uniform(device, "lvl upsample params", 16);
            queue.write_buffer(
                &up_buf,
                0,
                bytemuck::bytes_of(&UpsampleParams {
                    out_width: w,
                    out_height: h,
                    in_width: prev_w,
                    in_height: prev_h,
                }),
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lvl upsample params bg"),
                layout: &self.upsample_params_layout,
                entries: &[buf_entry(0, &up_buf)],
            })
        });

        // Create bind groups referencing the per-level uniform buffers
        let gradient_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl gradient params bg"),
            layout: &self.dis_pipeline.gradient_params_layout,
            entries: &[buf_entry(0, &gradient_buf)],
        });
        let is_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl is params bg"),
            layout: &self.dis_pipeline.inverse_search_params_layout,
            entries: &[buf_entry(0, &is_buf)],
        });
        let densify_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl densify params bg"),
            layout: &self.dis_pipeline.densify_params_layout,
            entries: &[buf_entry(0, &densify_buf)],
        });
        let warp_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl warp params bg"),
            layout: &self.refiner.warp_params_layout,
            entries: &[buf_entry(0, &warp_buf)],
        });
        let coeff_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl coeff params bg"),
            layout: &self.refiner.coeff_params_layout,
            entries: &[buf_entry(0, &coeff_buf)],
        });
        let jacobi_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl jacobi params bg"),
            layout: &self.refiner.jacobi_params_layout,
            entries: &[buf_entry(0, &jacobi_buf)],
        });
        let update_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lvl update params bg"),
            layout: &self.refiner.update_params_layout,
            entries: &[buf_entry(0, &update_buf)],
        });

        LevelBindGroups {
            gradient_params_bg,
            is_params_bg,
            densify_params_bg,
            warp_params_bg,
            coeff_params_bg,
            jacobi_params_bg,
            update_params_bg,
            upsample_params_bg,
        }
    }

    /// Allocate the ping-pong buffers and build the per-step bind groups for
    /// upsampling the finest GPU level's flow (`last_w` x `last_h`, read from
    /// `var_pool`) by 2x `steps` times.
    ///
    /// Step 0 reads `var_pool`; even steps write buffer A and odd steps write
    /// buffer B, which is only allocated when there are at least two steps.
    /// With `steps == 0` nothing is allocated.
    fn create_final_upsample(
        &self,
        var_pool: &variational::VariationalBufferPool,
        last_w: u32,
        last_h: u32,
        steps: u32,
    ) -> FinalUpsample {
        if steps == 0 {
            return FinalUpsample {
                buf_a: None,
                buf_b: None,
                steps: Vec::new(),
            };
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let full_w = last_w << steps;
        let full_h = last_h << steps;
        let full_bytes = ((full_w as usize) * (full_h as usize) * 4) as u64;

        let buf_a = (
            create_pool_storage(device, "final upsample a_u", full_bytes),
            create_pool_storage(device, "final upsample a_v", full_bytes),
        );
        let buf_b = (steps > 1).then(|| {
            (
                create_pool_storage(device, "final upsample b_u", full_bytes),
                create_pool_storage(device, "final upsample b_v", full_bytes),
            )
        });

        let mut src_w = last_w;
        let mut src_h = last_h;
        let mut step_bgs = Vec::new();

        for step in 0..steps {
            let out_w = src_w * 2;
            let out_h = src_h * 2;

            let (src_u_buf, src_v_buf): (&wgpu::Buffer, &wgpu::Buffer) = if step == 0 {
                (&var_pool.flow_u_buf, &var_pool.flow_v_buf)
            } else if step % 2 == 1 {
                (&buf_a.0, &buf_a.1)
            } else {
                let (u, v) = buf_b.as_ref().unwrap();
                (u, v)
            };
            let (dst_u_buf, dst_v_buf): (&wgpu::Buffer, &wgpu::Buffer) = if step % 2 == 0 {
                (&buf_a.0, &buf_a.1)
            } else {
                let (u, v) = buf_b.as_ref().unwrap();
                (u, v)
            };

            let data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("final upsample data bg"),
                layout: &self.upsample_data_layout,
                entries: &[
                    buf_entry(0, src_u_buf),
                    buf_entry(1, src_v_buf),
                    buf_entry(2, dst_u_buf),
                    buf_entry(3, dst_v_buf),
                ],
            });

            let up_buf = create_pool_uniform(device, "final upsample params", 16);
            queue.write_buffer(
                &up_buf,
                0,
                bytemuck::bytes_of(&UpsampleParams {
                    out_width: out_w,
                    out_height: out_h,
                    in_width: src_w,
                    in_height: src_h,
                }),
            );
            let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("final upsample params bg"),
                layout: &self.upsample_params_layout,
                entries: &[buf_entry(0, &up_buf)],
            });

            step_bgs.push(FinalUpsampleStep {
                data_bg,
                params_bg,
                out_w,
                out_h,
            });

            src_w = out_w;
            src_h = out_h;
        }

        FinalUpsample {
            buf_a: Some(buf_a),
            buf_b,
            steps: step_bgs,
        }
    }

    /// Encode DIS and variational refinement for every GPU level, coarse to
    /// fine, into `encoder`.
    ///
    /// The first level uploads `flow` from the CPU; each later level instead
    /// upsamples the previous level's variational output into the DIS input
    /// flow on the GPU.
    #[allow(clippy::too_many_arguments)]
    fn encode_levels(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pyr_pool: &PyramidBufferPool,
        dis_pool: &dis_pipeline::DisBufferPool,
        var_pool: &variational::VariationalBufferPool,
        upsample_bg: &wgpu::BindGroup,
        level_bgs: &[LevelBindGroups],
        gpu_scales: &[(u32, u32, u32)],
        flow: &super::FlowField,
        params: &super::DisFlowParams,
    ) {
        let queue = &self.ctx.queue;

        for (i, &(scale, w, h)) in gpu_scales.iter().enumerate() {
            let n = (w as usize) * (h as usize);
            let lvl = &level_bgs[i];

            let var_params = VariationalParams {
                delta: params.variational_delta,
                gamma: params.variational_gamma,
                alpha: params.variational_alpha,
                jacobi_iterations: params.variational_jacobi_iterations,
                outer_iterations: params.variational_outer_iterations_base * (scale + 1),
            };

            if i == 0 {
                // First GPU level: upload flow from CPU, copy images from pyramid.
                self.dis_pipeline.copy_pyramid_and_encode_with_bind_groups(
                    dis_pool,
                    queue,
                    encoder,
                    &pyr_pool.ref_level_bufs[scale as usize],
                    &pyr_pool.tgt_level_bufs[scale as usize],
                    w,
                    h,
                    Some(flow),
                    params,
                    &lvl.gradient_params_bg,
                    &lvl.is_params_bg,
                    &lvl.densify_params_bg,
                );
            } else {
                // GPU upsample: previous level's variational output → DIS input flow.
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("flow upsample"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.upsample_pipeline);
                    pass.set_bind_group(0, Some(upsample_bg), &[]);
                    pass.set_bind_group(1, Some(lvl.upsample_params_bg.as_ref().unwrap()), &[]);
                    pass.dispatch_workgroups(w.div_ceil(WG_SIZE), h.div_ceil(WG_SIZE), 1);
                }

                // Copy images from pyramid and encode DIS (flow already in DIS pool).
                self.dis_pipeline.copy_pyramid_and_encode_with_bind_groups(
                    dis_pool,
                    queue,
                    encoder,
                    &pyr_pool.ref_level_bufs[scale as usize],
                    &pyr_pool.tgt_level_bufs[scale as usize],
                    w,
                    h,
                    None,
                    params,
                    &lvl.gradient_params_bg,
                    &lvl.is_params_bg,
                    &lvl.densify_params_bg,
                );
            }

            // GPU-side copy: DIS outputs → variational inputs.
            dis_pool.encode_copies_to(
                encoder,
                &var_pool.ref_buf,
                &var_pool.tgt_buf,
                &var_pool.flow_u_buf,
                &var_pool.flow_v_buf,
                n,
            );

            // Encode variational refinement.
            self.refiner.encode_variational_with_bind_groups(
                var_pool,
                encoder,
                w,
                h,
                &var_params,
                &lvl.warp_params_bg,
                &lvl.coeff_params_bg,
                &lvl.jacobi_params_bg,
                &lvl.update_params_bg,
            );
        }
    }

    /// Encode one compute pass per final upsample step into `encoder`.
    fn encode_final_upsample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        steps: &[FinalUpsampleStep],
    ) {
        for step in steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("final flow upsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, Some(&step.data_bg), &[]);
            pass.set_bind_group(1, Some(&step.params_bg), &[]);
            pass.dispatch_workgroups(
                step.out_w.div_ceil(WG_SIZE),
                step.out_h.div_ceil(WG_SIZE),
                1,
            );
        }
    }

    /// Read the result flow back to the CPU after the submission has completed.
    ///
    /// Without final upsample steps this is the finest GPU level's variational
    /// output (`last_w` x `last_h`); otherwise it is the buffer the last step
    /// wrote, at that step's output size.
    fn read_back_final_flow(
        &self,
        var_pool: &variational::VariationalBufferPool,
        final_upsample: &FinalUpsample,
        last_w: u32,
        last_h: u32,
    ) -> super::FlowField {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let Some(last_step) = final_upsample.steps.last() else {
            let final_n = (last_w as usize) * (last_h as usize);
            let flow_u_data = read_buffer(device, queue, &var_pool.flow_u_buf, final_n);
            let flow_v_data = read_buffer(device, queue, &var_pool.flow_v_buf, final_n);

            return super::FlowField::from_split(last_w, last_h, flow_u_data, flow_v_data);
        };

        let final_w = last_step.out_w;
        let final_h = last_step.out_h;
        let final_n = (final_w as usize) * (final_h as usize);

        // The last step wrote to buf_a (even step) or buf_b (odd step).
        let last_step_index = final_upsample.steps.len() - 1;
        let (final_u_buf, final_v_buf) = if last_step_index.is_multiple_of(2) {
            let (u, v) = final_upsample.buf_a.as_ref().unwrap();
            (u, v)
        } else {
            let (u, v) = final_upsample.buf_b.as_ref().unwrap();
            (u, v)
        };

        let flow_u_data = read_buffer(device, queue, final_u_buf, final_n);
        let flow_v_data = read_buffer(device, queue, final_v_buf, final_n);

        super::FlowField::from_split(final_w, final_h, flow_u_data, flow_v_data)
    }
}

/// Number of DIS patches along x and y for a `w` x `h` level; zero along an
/// axis shorter than one patch.
fn patch_grid(w: u32, h: u32, params: &super::DisFlowParams) -> (u32, u32) {
    let ps = params.patch_size;
    let stride = params.patch_stride();
    let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
    let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };
    (num_patches_x, num_patches_y)
}

/// Per-level uniform-parameter bind groups, created before encoding so that all
/// GPU levels fit in one command buffer.
struct LevelBindGroups {
    // DIS params bind groups
    gradient_params_bg: wgpu::BindGroup,
    is_params_bg: wgpu::BindGroup,
    densify_params_bg: wgpu::BindGroup,
    // Variational params bind groups
    warp_params_bg: wgpu::BindGroup,
    coeff_params_bg: wgpu::BindGroup,
    jacobi_params_bg: wgpu::BindGroup,
    update_params_bg: wgpu::BindGroup,
    // Upsample params bind group (None for first level)
    upsample_params_bg: Option<wgpu::BindGroup>,
}

/// One 2x step of the final upsample to full resolution.
struct FinalUpsampleStep {
    data_bg: wgpu::BindGroup,
    params_bg: wgpu::BindGroup,
    out_w: u32,
    out_h: u32,
}

/// Ping-pong buffers and per-step bind groups for the final upsample. The
/// buffers are kept here so they stay alive until readback.
struct FinalUpsample {
    buf_a: Option<(wgpu::Buffer, wgpu::Buffer)>,
    buf_b: Option<(wgpu::Buffer, wgpu::Buffer)>,
    steps: Vec<FinalUpsampleStep>,
}

#[cfg(test)]
mod tests;
