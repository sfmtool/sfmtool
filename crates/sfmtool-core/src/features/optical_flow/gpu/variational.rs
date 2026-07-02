// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Variational refinement compute pipelines and their buffer pool.
//!
//! Four compute shaders form the variational refinement pipeline:
//! 1. **Warp** — bilinear warp of target image by current flow
//! 2. **Coefficients** — fused gradient computation + coefficient precomputation
//! 3. **Jacobi** — double-buffered Jacobi iteration (ping-pong)
//! 4. **Update** — apply accumulated du/dv to the flow field
//!
//! Coefficients are packed as `vec4(a11, a12, a22, b1)` + separate `b2` to stay
//! within the default 8-storage-buffer-per-stage limit.

use super::super::variational::VariationalParams;
use super::context::{
    buf_entry, create_compute_pipeline, create_pool_storage, create_pool_uniform, storage_ro_entry,
    storage_rw_entry, uniform_params_layout, CoeffParams, GpuContext, JacobiParams, UpdateParams,
    WarpParams, WG_SIZE,
};

const WARP_SHADER: &str = include_str!("shaders/warp_by_flow.wgsl");
const COEFF_SHADER: &str = include_str!("shaders/precompute_coefficients.wgsl");
const JACOBI_SHADER: &str = include_str!("shaders/jacobi_step.wgsl");
const UPDATE_SHADER: &str = include_str!("shaders/apply_flow_update.wgsl");

/// Pre-allocated GPU buffers and bind groups for variational refinement.
///
/// Created lazily on first use and grown as needed. Reused across calls to avoid
/// per-call buffer and bind group creation overhead.
pub(super) struct VariationalBufferPool {
    // Data buffers
    pub(super) ref_buf: wgpu::Buffer,
    pub(super) tgt_buf: wgpu::Buffer,
    pub(super) flow_u_buf: wgpu::Buffer,
    pub(super) flow_v_buf: wgpu::Buffer,
    // These buffers are referenced by bind groups but not accessed directly.
    _warped_buf: wgpu::Buffer,
    _coeff_buf: wgpu::Buffer,
    _b2_buf: wgpu::Buffer,
    du_buf_a: wgpu::Buffer,
    dv_buf_a: wgpu::Buffer,
    du_buf_b: wgpu::Buffer,
    dv_buf_b: wgpu::Buffer,
    // Uniform buffers (updated via write_buffer each call)
    warp_params_buf: wgpu::Buffer,
    coeff_params_buf: wgpu::Buffer,
    jacobi_params_buf: wgpu::Buffer,
    update_params_buf: wgpu::Buffer,
    // Bind groups (valid as long as buffer objects don't change)
    warp_data_bg: wgpu::BindGroup,
    warp_params_bg: wgpu::BindGroup,
    coeff_data_bg: wgpu::BindGroup,
    coeff_params_bg: wgpu::BindGroup,
    jacobi_bg_a: wgpu::BindGroup,
    jacobi_bg_b: wgpu::BindGroup,
    jacobi_params_bg: wgpu::BindGroup,
    update_bg_even: wgpu::BindGroup,
    update_bg_odd: wgpu::BindGroup,
    update_params_bg: wgpu::BindGroup,
}

impl VariationalBufferPool {
    fn new(device: &wgpu::Device, refiner: &GpuVariationalRefiner, capacity: usize) -> Self {
        let f32_bytes = (capacity * 4) as u64;
        let coeff_bytes = (capacity * 16) as u64; // vec4 per pixel
        let uniform_bytes = 16u64; // all variational uniform structs are 16 bytes

        let ref_buf = create_pool_storage(device, "pool var ref", f32_bytes);
        let tgt_buf = create_pool_storage(device, "pool var tgt", f32_bytes);
        let flow_u_buf = create_pool_storage(device, "pool var flow_u", f32_bytes);
        let flow_v_buf = create_pool_storage(device, "pool var flow_v", f32_bytes);
        let warped_buf = create_pool_storage(device, "pool var warped", f32_bytes);
        let coeff_buf = create_pool_storage(device, "pool var coeff", coeff_bytes);
        let b2_buf = create_pool_storage(device, "pool var b2", f32_bytes);
        let du_buf_a = create_pool_storage(device, "pool var du_a", f32_bytes);
        let dv_buf_a = create_pool_storage(device, "pool var dv_a", f32_bytes);
        let du_buf_b = create_pool_storage(device, "pool var du_b", f32_bytes);
        let dv_buf_b = create_pool_storage(device, "pool var dv_b", f32_bytes);

        let warp_params_buf = create_pool_uniform(device, "pool var warp params", uniform_bytes);
        let coeff_params_buf = create_pool_uniform(device, "pool var coeff params", uniform_bytes);
        let jacobi_params_buf =
            create_pool_uniform(device, "pool var jacobi params", uniform_bytes);
        let update_params_buf =
            create_pool_uniform(device, "pool var update params", uniform_bytes);

        // Create bind groups referencing pool buffers
        let warp_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool warp data bg"),
            layout: &refiner.warp_data_layout,
            entries: &[
                buf_entry(0, &tgt_buf),
                buf_entry(1, &flow_u_buf),
                buf_entry(2, &flow_v_buf),
                buf_entry(3, &warped_buf),
            ],
        });
        let warp_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool warp params bg"),
            layout: &refiner.warp_params_layout,
            entries: &[buf_entry(0, &warp_params_buf)],
        });

        let coeff_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool coeff data bg"),
            layout: &refiner.coeff_data_layout,
            entries: &[
                buf_entry(0, &ref_buf),
                buf_entry(1, &warped_buf),
                buf_entry(2, &coeff_buf),
                buf_entry(3, &b2_buf),
            ],
        });
        let coeff_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool coeff params bg"),
            layout: &refiner.coeff_params_layout,
            entries: &[buf_entry(0, &coeff_params_buf)],
        });

        let jacobi_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool jacobi bg A"),
            layout: &refiner.jacobi_data_layout,
            entries: &[
                buf_entry(0, &du_buf_a),
                buf_entry(1, &dv_buf_a),
                buf_entry(2, &du_buf_b),
                buf_entry(3, &dv_buf_b),
                buf_entry(4, &flow_u_buf),
                buf_entry(5, &flow_v_buf),
                buf_entry(6, &coeff_buf),
                buf_entry(7, &b2_buf),
            ],
        });
        let jacobi_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool jacobi bg B"),
            layout: &refiner.jacobi_data_layout,
            entries: &[
                buf_entry(0, &du_buf_b),
                buf_entry(1, &dv_buf_b),
                buf_entry(2, &du_buf_a),
                buf_entry(3, &dv_buf_a),
                buf_entry(4, &flow_u_buf),
                buf_entry(5, &flow_v_buf),
                buf_entry(6, &coeff_buf),
                buf_entry(7, &b2_buf),
            ],
        });
        let jacobi_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool jacobi params bg"),
            layout: &refiner.jacobi_params_layout,
            entries: &[buf_entry(0, &jacobi_params_buf)],
        });

        let update_bg_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool update bg even"),
            layout: &refiner.update_data_layout,
            entries: &[
                buf_entry(0, &flow_u_buf),
                buf_entry(1, &flow_v_buf),
                buf_entry(2, &du_buf_a),
                buf_entry(3, &dv_buf_a),
            ],
        });
        let update_bg_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool update bg odd"),
            layout: &refiner.update_data_layout,
            entries: &[
                buf_entry(0, &flow_u_buf),
                buf_entry(1, &flow_v_buf),
                buf_entry(2, &du_buf_b),
                buf_entry(3, &dv_buf_b),
            ],
        });
        let update_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool update params bg"),
            layout: &refiner.update_params_layout,
            entries: &[buf_entry(0, &update_params_buf)],
        });

        Self {
            ref_buf,
            tgt_buf,
            flow_u_buf,
            flow_v_buf,
            _warped_buf: warped_buf,
            _coeff_buf: coeff_buf,
            _b2_buf: b2_buf,
            du_buf_a,
            dv_buf_a,
            du_buf_b,
            dv_buf_b,
            warp_params_buf,
            coeff_params_buf,
            jacobi_params_buf,
            update_params_buf,
            warp_data_bg,
            warp_params_bg,
            coeff_data_bg,
            coeff_params_bg,
            jacobi_bg_a,
            jacobi_bg_b,
            jacobi_params_bg,
            update_bg_even,
            update_bg_odd,
            update_params_bg,
        }
    }
}

/// GPU compute pipelines for variational refinement.
///
/// Create once, reuse across calls. Pipelines are independent of image dimensions.
/// Buffers are allocated per-call and passed through the pipeline.
pub struct GpuVariationalRefiner {
    warp_pipeline: wgpu::ComputePipeline,
    warp_data_layout: wgpu::BindGroupLayout,
    pub(super) warp_params_layout: wgpu::BindGroupLayout,

    coeff_pipeline: wgpu::ComputePipeline,
    coeff_data_layout: wgpu::BindGroupLayout,
    pub(super) coeff_params_layout: wgpu::BindGroupLayout,

    jacobi_pipeline: wgpu::ComputePipeline,
    jacobi_data_layout: wgpu::BindGroupLayout,
    pub(super) jacobi_params_layout: wgpu::BindGroupLayout,

    update_pipeline: wgpu::ComputePipeline,
    update_data_layout: wgpu::BindGroupLayout,
    pub(super) update_params_layout: wgpu::BindGroupLayout,
}

impl GpuVariationalRefiner {
    /// Create all GPU compute pipelines.
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        // --- Warp pipeline ---
        let warp_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp data"),
            entries: &[
                storage_ro_entry(0), // tgt_image
                storage_ro_entry(1), // flow_u
                storage_ro_entry(2), // flow_v
                storage_rw_entry(3), // warped
            ],
        });
        let warp_params_layout = uniform_params_layout(device, "warp params");
        let warp_pipeline = create_compute_pipeline(
            device,
            WARP_SHADER,
            &[&warp_data_layout, &warp_params_layout],
        );

        // --- Coefficient pipeline ---
        let coeff_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coeff data"),
            entries: &[
                storage_ro_entry(0), // ref_image
                storage_ro_entry(1), // warped_image
                storage_rw_entry(2), // coefficients (vec4)
                storage_rw_entry(3), // b2
            ],
        });
        let coeff_params_layout = uniform_params_layout(device, "coeff params");
        let coeff_pipeline = create_compute_pipeline(
            device,
            COEFF_SHADER,
            &[&coeff_data_layout, &coeff_params_layout],
        );

        // --- Jacobi pipeline ---
        let jacobi_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("jacobi data"),
                entries: &[
                    storage_ro_entry(0), // du_old
                    storage_ro_entry(1), // dv_old
                    storage_rw_entry(2), // du_new
                    storage_rw_entry(3), // dv_new
                    storage_ro_entry(4), // flow_u
                    storage_ro_entry(5), // flow_v
                    storage_ro_entry(6), // coefficients (vec4)
                    storage_ro_entry(7), // b2
                ],
            });
        let jacobi_params_layout = uniform_params_layout(device, "jacobi params");
        let jacobi_pipeline = create_compute_pipeline(
            device,
            JACOBI_SHADER,
            &[&jacobi_data_layout, &jacobi_params_layout],
        );

        // --- Update pipeline ---
        let update_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("update data"),
                entries: &[
                    storage_rw_entry(0), // flow_u
                    storage_rw_entry(1), // flow_v
                    storage_ro_entry(2), // du
                    storage_ro_entry(3), // dv
                ],
            });
        let update_params_layout = uniform_params_layout(device, "update params");
        let update_pipeline = create_compute_pipeline(
            device,
            UPDATE_SHADER,
            &[&update_data_layout, &update_params_layout],
        );

        Self {
            warp_pipeline,
            warp_data_layout,
            warp_params_layout,
            coeff_pipeline,
            coeff_data_layout,
            coeff_params_layout,
            jacobi_pipeline,
            jacobi_data_layout,
            jacobi_params_layout,
            update_pipeline,
            update_data_layout,
            update_params_layout,
        }
    }

    /// Create a buffer pool with at least `n` pixels of capacity.
    pub(super) fn create_pool(&self, device: &wgpu::Device, n: usize) -> VariationalBufferPool {
        VariationalBufferPool::new(device, self, n)
    }

    /// Upload uniform params and encode variational refinement commands into the
    /// given encoder. Data buffers (ref, tgt, flow) must already be populated.
    ///
    /// Used by both standalone [`refine`] and the combined DIS+variational path.
    pub(super) fn encode_variational(
        &self,
        pool: &VariationalBufferPool,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        w: u32,
        h: u32,
        params: &VariationalParams,
    ) {
        let n = (w as usize) * (h as usize);

        // Upload uniform params
        queue.write_buffer(
            &pool.warp_params_buf,
            0,
            bytemuck::bytes_of(&WarpParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &pool.coeff_params_buf,
            0,
            bytemuck::bytes_of(&CoeffParams {
                width: w,
                height: h,
                delta: params.delta,
                gamma: params.gamma,
            }),
        );
        queue.write_buffer(
            &pool.jacobi_params_buf,
            0,
            bytemuck::bytes_of(&JacobiParams {
                width: w,
                height: h,
                alpha: params.alpha,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &pool.update_params_buf,
            0,
            bytemuck::bytes_of(&UpdateParams {
                width: w,
                height: h,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        let wg_x = w.div_ceil(WG_SIZE);
        let wg_y = h.div_ceil(WG_SIZE);
        let buf_size = (n * 4) as u64;
        let jacobi_bgs = [&pool.jacobi_bg_a, &pool.jacobi_bg_b];

        for _outer in 0..params.outer_iterations {
            encoder.clear_buffer(&pool.du_buf_a, 0, Some(buf_size));
            encoder.clear_buffer(&pool.dv_buf_a, 0, Some(buf_size));
            encoder.clear_buffer(&pool.du_buf_b, 0, Some(buf_size));
            encoder.clear_buffer(&pool.dv_buf_b, 0, Some(buf_size));

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("variational refinement"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.warp_pipeline);
                pass.set_bind_group(0, Some(&pool.warp_data_bg), &[]);
                pass.set_bind_group(1, Some(&pool.warp_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pass.set_pipeline(&self.coeff_pipeline);
                pass.set_bind_group(0, Some(&pool.coeff_data_bg), &[]);
                pass.set_bind_group(1, Some(&pool.coeff_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pass.set_pipeline(&self.jacobi_pipeline);
                pass.set_bind_group(1, Some(&pool.jacobi_params_bg), &[]);
                for i in 0..params.jacobi_iterations {
                    pass.set_bind_group(0, Some(jacobi_bgs[(i % 2) as usize]), &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }

                pass.set_pipeline(&self.update_pipeline);
                if params.jacobi_iterations.is_multiple_of(2) {
                    pass.set_bind_group(0, Some(&pool.update_bg_even), &[]);
                } else {
                    pass.set_bind_group(0, Some(&pool.update_bg_odd), &[]);
                }
                pass.set_bind_group(1, Some(&pool.update_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
    }
    /// Encode variational refinement using externally-provided params bind groups.
    ///
    /// Unlike [`encode_variational`], this method does NOT write uniform params
    /// via `queue.write_buffer`. The caller must write params to per-level uniform
    /// buffers before the single `queue.submit()`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_variational_with_bind_groups(
        &self,
        pool: &VariationalBufferPool,
        encoder: &mut wgpu::CommandEncoder,
        w: u32,
        h: u32,
        params: &VariationalParams,
        warp_params_bg: &wgpu::BindGroup,
        coeff_params_bg: &wgpu::BindGroup,
        jacobi_params_bg: &wgpu::BindGroup,
        update_params_bg: &wgpu::BindGroup,
    ) {
        let n = (w as usize) * (h as usize);
        let wg_x = w.div_ceil(WG_SIZE);
        let wg_y = h.div_ceil(WG_SIZE);
        let buf_size = (n * 4) as u64;
        let jacobi_bgs = [&pool.jacobi_bg_a, &pool.jacobi_bg_b];

        for _outer in 0..params.outer_iterations {
            encoder.clear_buffer(&pool.du_buf_a, 0, Some(buf_size));
            encoder.clear_buffer(&pool.dv_buf_a, 0, Some(buf_size));
            encoder.clear_buffer(&pool.du_buf_b, 0, Some(buf_size));
            encoder.clear_buffer(&pool.dv_buf_b, 0, Some(buf_size));

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("variational refinement"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.warp_pipeline);
                pass.set_bind_group(0, Some(&pool.warp_data_bg), &[]);
                pass.set_bind_group(1, Some(warp_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pass.set_pipeline(&self.coeff_pipeline);
                pass.set_bind_group(0, Some(&pool.coeff_data_bg), &[]);
                pass.set_bind_group(1, Some(coeff_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pass.set_pipeline(&self.jacobi_pipeline);
                pass.set_bind_group(1, Some(jacobi_params_bg), &[]);
                for i in 0..params.jacobi_iterations {
                    pass.set_bind_group(0, Some(jacobi_bgs[(i % 2) as usize]), &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }

                pass.set_pipeline(&self.update_pipeline);
                if params.jacobi_iterations.is_multiple_of(2) {
                    pass.set_bind_group(0, Some(&pool.update_bg_even), &[]);
                } else {
                    pass.set_bind_group(0, Some(&pool.update_bg_odd), &[]);
                }
                pass.set_bind_group(1, Some(update_params_bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
    }
}
