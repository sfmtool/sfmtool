// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated variational refinement using wgpu compute shaders.
//!
//! This module implements the variational refinement stage of DIS optical flow
//! on the GPU. The Jacobi solver is the primary bottleneck (~57% of per-scale
//! CPU time) and maps perfectly to GPU compute: each pixel is independent within
//! an iteration, with simple stencil access patterns.
//!
//! # Architecture
//!
//! Four compute shaders form the variational refinement pipeline:
//! 1. **Warp** — bilinear warp of target image by current flow
//! 2. **Coefficients** — fused gradient computation + coefficient precomputation
//! 3. **Jacobi** — double-buffered Jacobi iteration (ping-pong)
//! 4. **Update** — apply accumulated du/dv to the flow field
//!
//! Coefficients are packed as `vec4(a11, a12, a22, b1)` + separate `b2` to stay
//! within the default 8-storage-buffer-per-stage limit.

mod dis_pipeline;
mod pyramid_pipeline;

use super::variational::VariationalParams;

pub(crate) use dis_pipeline::GpuDisPipeline;
pub(crate) use pyramid_pipeline::{GpuPyramidPipeline, PyramidBufferPool};

const WARP_SHADER: &str = include_str!("shaders/warp_by_flow.wgsl");
const COEFF_SHADER: &str = include_str!("shaders/precompute_coefficients.wgsl");
const JACOBI_SHADER: &str = include_str!("shaders/jacobi_step.wgsl");
const UPDATE_SHADER: &str = include_str!("shaders/apply_flow_update.wgsl");
const UPSAMPLE_SHADER: &str = include_str!("shaders/upsample_flow.wgsl");

/// Workgroup size matching the WGSL declarations.
const WG_SIZE: u32 = 16;

// Uniform parameter structs. Must match WGSL struct layouts exactly.
// All padded to 16 bytes for uniform buffer alignment.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CoeffParams {
    width: u32,
    height: u32,
    delta: f32,
    gamma: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JacobiParams {
    width: u32,
    height: u32,
    alpha: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpsampleParams {
    out_width: u32,
    out_height: u32,
    in_width: u32,
    in_height: u32,
}

/// GPU context holding a wgpu device and queue for compute operations.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a new GPU context from the default adapter.
    /// Returns `None` if no suitable GPU is available.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sfmtool gpu compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()?;

        Some(Self { device, queue })
    }

    /// Create from an existing wgpu device and queue (e.g., shared with GUI).
    pub fn from_existing(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }
}

/// Pre-allocated GPU buffers and bind groups for variational refinement.
///
/// Created lazily on first use and grown as needed. Reused across calls to avoid
/// per-call buffer and bind group creation overhead.
struct VariationalBufferPool {
    // Data buffers
    ref_buf: wgpu::Buffer,
    tgt_buf: wgpu::Buffer,
    flow_u_buf: wgpu::Buffer,
    flow_v_buf: wgpu::Buffer,
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
    warp_params_layout: wgpu::BindGroupLayout,

    coeff_pipeline: wgpu::ComputePipeline,
    coeff_data_layout: wgpu::BindGroupLayout,
    coeff_params_layout: wgpu::BindGroupLayout,

    jacobi_pipeline: wgpu::ComputePipeline,
    jacobi_data_layout: wgpu::BindGroupLayout,
    jacobi_params_layout: wgpu::BindGroupLayout,

    update_pipeline: wgpu::ComputePipeline,
    update_data_layout: wgpu::BindGroupLayout,
    update_params_layout: wgpu::BindGroupLayout,
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
    fn create_pool(&self, device: &wgpu::Device, n: usize) -> VariationalBufferPool {
        VariationalBufferPool::new(device, self, n)
    }

    /// Upload uniform params and encode variational refinement commands into the
    /// given encoder. Data buffers (ref, tgt, flow) must already be populated.
    ///
    /// Used by both standalone [`refine`] and the combined DIS+variational path.
    fn encode_variational(
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
    fn encode_variational_with_bind_groups(
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

// --- Helper functions ---

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_params_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_pool_storage(device: &wgpu::Device, label: &str, byte_size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_pool_uniform(device: &wgpu::Device, label: &str, byte_size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Read a GPU buffer back to CPU as a Vec<f32>.
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    num_f32s: usize,
) -> Vec<f32> {
    let size = (num_f32s * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

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

    /// Run DIS and variational refinement as a single GPU submission.
    ///
    /// This eliminates redundant per-level CPU↔GPU transfers: images are uploaded
    /// once, DIS output flow is copied to variational input via GPU-side buffer
    /// copy (no CPU round-trip), and only the final refined flow is read back.
    /// Compared to calling `run_dis_level` + `variational_refine` separately,
    /// this saves 2 image uploads, 1 flow readback, 1 flow upload, and 1
    /// device.poll synchronization point per level.
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
    /// to [`run_gpu_levels_prebuilt`]. Seed images are returned if
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
    /// The GPU pyramid must have been built by a prior call to [`build_gpu_pyramid`].
    /// `gpu_scales` contains `(scale_index, width, height)` in coarse-to-fine order.
    ///
    /// If `final_upsample_steps > 0`, the flow is upsampled on the GPU by 2x that
    /// many times before readback, avoiding the costly CPU upsample. The returned
    /// flow will be at the upsampled resolution (not resized to exact target dims —
    /// the caller should apply `resize_flow_to` if needed).
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
                let ps = params.patch_size;
                let stride = params.patch_stride();
                let npx = if w >= ps { (w - ps) / stride + 1 } else { 0 };
                let npy = if h >= ps { (h - ps) / stride + 1 } else { 0 };
                (npx * npy) as usize
            })
            .max()
            .unwrap();

        // Create DIS and variational buffer pools for this call.
        let dis_pool = self.dis_pipeline.create_pool(device, max_n, max_patches);
        let var_pool = self.refiner.create_pool(device, max_n);

        // --- Create per-level uniform buffers, write params, and build bind groups ---
        // This allows all levels to be encoded into a single command buffer with
        // one submit+wait, eliminating per-level synchronization overhead.

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

        let level_bgs: Vec<LevelBindGroups> = gpu_scales
            .iter()
            .enumerate()
            .map(|(i, &(scale, w, h))| {
                let ps = params.patch_size;
                let stride = params.patch_stride();
                let num_patches_x = if w >= ps { (w - ps) / stride + 1 } else { 0 };
                let num_patches_y = if h >= ps { (h - ps) / stride + 1 } else { 0 };

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

                // Variational uniform buffers
                let outer_iterations = params.variational_outer_iterations_base * (scale + 1);
                let _ = outer_iterations; // used by encode, not stored in uniform

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
                let upsample_params_bg = if i > 0 {
                    let (_, prev_w, prev_h) = gpu_scales[i - 1];
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
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("lvl upsample params bg"),
                        layout: &self.upsample_params_layout,
                        entries: &[buf_entry(0, &up_buf)],
                    }))
                } else {
                    None
                };

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
            })
            .collect();

        // --- Final upsample buffers and bind groups (if needed) ---
        let (_, last_w, last_h) = *gpu_scales.last().unwrap();

        struct FinalUpsampleStep {
            data_bg: wgpu::BindGroup,
            params_bg: wgpu::BindGroup,
            out_w: u32,
            out_h: u32,
        }

        // Buffers for final upsample ping-pong. Kept alive for readback.
        let mut final_buf_a: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
        let mut final_buf_b: Option<(wgpu::Buffer, wgpu::Buffer)> = None;

        let final_upsample_bufs: Vec<FinalUpsampleStep> = if final_upsample_steps > 0 {
            let full_w = last_w << final_upsample_steps;
            let full_h = last_h << final_upsample_steps;
            let full_bytes = ((full_w as usize) * (full_h as usize) * 4) as u64;

            final_buf_a = Some((
                create_pool_storage(device, "final upsample a_u", full_bytes),
                create_pool_storage(device, "final upsample a_v", full_bytes),
            ));
            if final_upsample_steps > 1 {
                final_buf_b = Some((
                    create_pool_storage(device, "final upsample b_u", full_bytes),
                    create_pool_storage(device, "final upsample b_v", full_bytes),
                ));
            }

            let (buf_a_u, buf_a_v) = final_buf_a.as_ref().unwrap();

            let mut src_w = last_w;
            let mut src_h = last_h;
            let mut steps = Vec::new();

            for step in 0..final_upsample_steps {
                let out_w = src_w * 2;
                let out_h = src_h * 2;

                let (src_u_buf, src_v_buf): (&wgpu::Buffer, &wgpu::Buffer) = if step == 0 {
                    (&var_pool.flow_u_buf, &var_pool.flow_v_buf)
                } else if step % 2 == 1 {
                    (buf_a_u, buf_a_v)
                } else {
                    let (u, v) = final_buf_b.as_ref().unwrap();
                    (u, v)
                };
                let (dst_u_buf, dst_v_buf): (&wgpu::Buffer, &wgpu::Buffer) = if step % 2 == 0 {
                    (buf_a_u, buf_a_v)
                } else {
                    let (u, v) = final_buf_b.as_ref().unwrap();
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

                steps.push(FinalUpsampleStep {
                    data_bg,
                    params_bg,
                    out_w,
                    out_h,
                });

                src_w = out_w;
                src_h = out_h;
            }
            steps
        } else {
            Vec::new()
        };

        // --- Encode all levels + final upsample into a single command buffer ---
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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
                    &dis_pool,
                    queue,
                    &mut encoder,
                    &pyr_pool.ref_level_bufs[scale as usize],
                    &pyr_pool.tgt_level_bufs[scale as usize],
                    w,
                    h,
                    Some(&flow),
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
                    pass.set_bind_group(0, Some(&upsample_bg), &[]);
                    pass.set_bind_group(1, Some(lvl.upsample_params_bg.as_ref().unwrap()), &[]);
                    pass.dispatch_workgroups(w.div_ceil(WG_SIZE), h.div_ceil(WG_SIZE), 1);
                }

                // Copy images from pyramid and encode DIS (flow already in DIS pool).
                self.dis_pipeline.copy_pyramid_and_encode_with_bind_groups(
                    &dis_pool,
                    queue,
                    &mut encoder,
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
                &mut encoder,
                &var_pool.ref_buf,
                &var_pool.tgt_buf,
                &var_pool.flow_u_buf,
                &var_pool.flow_v_buf,
                n,
            );

            // Encode variational refinement.
            self.refiner.encode_variational_with_bind_groups(
                &var_pool,
                &mut encoder,
                w,
                h,
                &var_params,
                &lvl.warp_params_bg,
                &lvl.coeff_params_bg,
                &lvl.jacobi_params_bg,
                &lvl.update_params_bg,
            );
        }

        // Encode final upsample steps (if any).
        for step in &final_upsample_bufs {
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

        // Single submit and wait for all levels + final upsample.
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read back from the appropriate buffer.
        if final_upsample_steps == 0 {
            let final_n = (last_w as usize) * (last_h as usize);
            let flow_u_data = read_buffer(device, queue, &var_pool.flow_u_buf, final_n);
            let flow_v_data = read_buffer(device, queue, &var_pool.flow_v_buf, final_n);

            super::FlowField::from_split(last_w, last_h, flow_u_data, flow_v_data)
        } else {
            let last_step = final_upsample_bufs.last().unwrap();
            let final_w = last_step.out_w;
            let final_h = last_step.out_h;
            let final_n = (final_w as usize) * (final_h as usize);

            // The last step wrote to buf_a (even step) or buf_b (odd step).
            let (final_u_buf, final_v_buf) = if (final_upsample_steps - 1).is_multiple_of(2) {
                let (u, v) = final_buf_a.as_ref().unwrap();
                (u, v)
            } else {
                let (u, v) = final_buf_b.as_ref().unwrap();
                (u, v)
            };

            let flow_u_data = read_buffer(device, queue, final_u_buf, final_n);
            let flow_v_data = read_buffer(device, queue, final_v_buf, final_n);

            super::FlowField::from_split(final_w, final_h, flow_u_data, flow_v_data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FlowField, GrayImage};
    use super::*;

    /// Derive the image subdirectory from the filename (e.g. "seoul_bull_sculpture_08.jpg" -> "seoul_bull_sculpture").
    fn image_subdir(filename: &str) -> &str {
        let stem = filename.rsplit_once('.').map_or(filename, |(s, _)| s);
        stem.rsplit_once('_').map_or(stem, |(prefix, _)| prefix)
    }

    /// Load a grayscale image from the test data directory.
    fn load_test_image(filename: &str) -> GrayImage {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(manifest)
            .join("../../test-data/images")
            .join(image_subdir(filename))
            .join(filename);
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e))
            .into_luma8();
        let (w, h) = img.dimensions();
        GrayImage::from_u8(w, h, img.as_raw())
    }

    #[test]
    fn test_gpu_variational_identical_images() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Use the combined DIS+variational path on identical images
        let img = GrayImage::checkerboard(32, 32);
        let params = super::super::DisFlowParams {
            variational_refinement: true,
            gpu_min_pixels: 0, // force GPU even for small images
            ..super::super::DisFlowParams::default_quality()
        };

        let flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

        // For identical images, flow should remain near zero
        let mut max_flow = 0.0f32;
        for row in 0..32 {
            for col in 0..32 {
                let (dx, dy) = flow.get(col, row);
                max_flow = max_flow.max(dx.abs()).max(dy.abs());
            }
        }
        assert!(
            max_flow < 0.5,
            "Expected near-zero flow for identical images, got max_flow={}",
            max_flow
        );
    }

    #[test]
    fn test_gpu_vs_cpu_variational() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Test GPU vs CPU with variational refinement enabled
        let ref_img = GrayImage::checkerboard(64, 64);
        let tgt_img = GrayImage::shifted(&ref_img, 2.0, 1.0);

        let params = super::super::DisFlowParams {
            variational_refinement: true,
            gpu_min_pixels: 0,
            finest_scale: Some(0),
            coarsest_scale: Some(3),
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.001,
            "GPU vs CPU RMSE too large: {} (max_diff={})",
            rmse,
            max_diff
        );
    }

    /// Helper to compute RMSE between two flow fields.
    fn flow_rmse(a: &FlowField, b: &FlowField) -> (f64, f32) {
        let w = a.width();
        let h = a.height();
        let n = (w * h) as usize;
        let mut max_diff = 0.0f32;
        let mut sum_diff_sq = 0.0f64;
        for row in 0..h {
            for col in 0..w {
                let (au, av) = a.get(col, row);
                let (bu, bv) = b.get(col, row);
                let du = (au - bu).abs();
                let dv = (av - bv).abs();
                max_diff = max_diff.max(du).max(dv);
                sum_diff_sq += (du as f64).powi(2) + (dv as f64).powi(2);
            }
        }
        let rmse = (sum_diff_sq / (2 * n) as f64).sqrt();
        (rmse, max_diff)
    }

    #[test]
    fn test_gpu_vs_cpu_full_pipeline_shifted() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        let ref_img = GrayImage::checkerboard(128, 128);
        let tgt_img = GrayImage::shifted(&ref_img, 3.0, 1.5);
        let params = super::super::DisFlowParams {
            finest_scale: Some(0),
            coarsest_scale: Some(3),
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.001,
            "Full pipeline GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_vs_cpu_full_pipeline_identical() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        let img = GrayImage::checkerboard(64, 64);
        let params = super::super::DisFlowParams::default_quality();

        let cpu_flow = super::super::compute_optical_flow(&img, &img, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.001,
            "Identical images GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    // --- Real image tests using dataset images checked into the repo ---

    #[test]
    fn test_gpu_vs_cpu_seoul_bull_consecutive() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Seoul bull: 270x480, horizontal orbit — consecutive pair has modest motion
        let img_a = load_test_image("seoul_bull_sculpture_08.jpg");
        let img_b = load_test_image("seoul_bull_sculpture_09.jpg");
        let params = super::super::DisFlowParams::default_quality();

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.01,
            "Seoul bull consecutive GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_vs_cpu_seoul_bull_wider_baseline() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Skip 2 frames for a wider baseline with more motion
        let img_a = load_test_image("seoul_bull_sculpture_05.jpg");
        let img_b = load_test_image("seoul_bull_sculpture_08.jpg");
        let params = super::super::DisFlowParams::default_quality();

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.01,
            "Seoul bull wide baseline GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_vs_cpu_seattle_backyard_consecutive() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Seattle backyard: 360x640, forward push — consecutive pair
        let img_a = load_test_image("seattle_backyard_10.jpg");
        let img_b = load_test_image("seattle_backyard_11.jpg");
        let params = super::super::DisFlowParams::default_quality();

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.01,
            "Seattle backyard consecutive GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    #[ignore] // slow diagnostic test — run manually with: cargo test -p sfmtool-core --lib gpu::tests::diagnostic -- --ignored --nocapture
    fn test_gpu_vs_cpu_seattle_backyard_panning_diagnostic() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        let img_a = load_test_image("seattle_backyard_20.jpg");
        let img_b = load_test_image("seattle_backyard_22.jpg");

        let w = img_a.width();
        let h = img_a.height();
        eprintln!("Image size: {w}x{h}");

        let base_params = super::super::DisFlowParams::default_quality();
        // Replicate compute_coarsest_scale logic
        let coarsest = (2.0 * w as f64 / (5.0 * base_params.patch_size as f64))
            .log2()
            .floor() as u32;
        let finest = coarsest.saturating_sub(2);
        eprintln!("Pyramid: coarsest={coarsest}, finest={finest}");

        // Test each scale independently to isolate divergence
        for test_scale in (finest..=coarsest).rev() {
            let params = super::super::DisFlowParams {
                finest_scale: Some(test_scale),
                coarsest_scale: Some(test_scale),
                ..super::super::DisFlowParams::default_quality()
            };

            let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
            let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

            let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
            eprintln!(
                "  Scale {test_scale} only: RMSE={rmse:.6}, max_diff={max_diff:.4}, \
                 flow_size={}x{}",
                cpu_flow.width(),
                cpu_flow.height()
            );
        }

        // Test cumulative: coarsest down to each finer level
        for stop_scale in (finest..coarsest).rev() {
            let params = super::super::DisFlowParams {
                finest_scale: Some(stop_scale),
                coarsest_scale: Some(coarsest),
                ..super::super::DisFlowParams::default_quality()
            };

            let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
            let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

            let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
            eprintln!("  Scales {coarsest}→{stop_scale}: RMSE={rmse:.6}, max_diff={max_diff:.4}");
        }

        // Full pipeline — find where the worst pixels are
        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &base_params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &base_params, Some(&gpu));

        // Find top-5 worst pixels
        let total = cpu_flow.width() as usize * cpu_flow.height() as usize;
        let mut diffs: Vec<(u32, u32, f32)> = Vec::new();
        for row in 0..cpu_flow.height() {
            for col in 0..cpu_flow.width() {
                let (cu, cv) = cpu_flow.get(col, row);
                let (gu, gv) = gpu_flow.get(col, row);
                let d = (cu - gu).abs().max((cv - gv).abs());
                diffs.push((col, row, d));
            }
        }
        diffs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        eprintln!("  Top-5 worst pixels:");
        for &(col, row, d) in diffs.iter().take(5) {
            let (cu, cv) = cpu_flow.get(col, row);
            let (gu, gv) = gpu_flow.get(col, row);
            eprintln!(
                "    ({col}, {row}): diff={d:.4}, cpu=({cu:.3},{cv:.3}), gpu=({gu:.3},{gv:.3})"
            );
        }

        // RMSE excluding pixels whose flow points outside the image (occluded regions)
        let fw = cpu_flow.width() as f32;
        let fh = cpu_flow.height() as f32;
        let mut inbounds_sum_sq = 0.0f64;
        let mut inbounds_count = 0usize;
        for row in 0..cpu_flow.height() {
            for col in 0..cpu_flow.width() {
                let (cu, cv) = cpu_flow.get(col, row);
                let tx = col as f32 + cu;
                let ty = row as f32 + cv;
                if tx >= 0.0 && tx < fw && ty >= 0.0 && ty < fh {
                    let (gu, gv) = gpu_flow.get(col, row);
                    let du = (cu - gu) as f64;
                    let dv = (cv - gv) as f64;
                    inbounds_sum_sq += du * du + dv * dv;
                    inbounds_count += 1;
                }
            }
        }
        if inbounds_count > 0 {
            let inbounds_rmse = (inbounds_sum_sq / (2 * inbounds_count) as f64).sqrt();
            eprintln!(
                "  In-bounds RMSE={inbounds_rmse:.6} ({inbounds_count}/{total} pixels, \
                 {:.1}% excluded)",
                100.0 * (1.0 - inbounds_count as f64 / total as f64)
            );
        }

        // Count pixels above various thresholds
        let thresholds = [0.01, 0.1, 1.0, 5.0];
        for &t in &thresholds {
            let count = diffs.iter().filter(|d| d.2 > t).count();
            eprintln!(
                "    pixels with diff > {t}: {count} ({:.2}%)",
                100.0 * count as f64 / total as f64
            );
        }

        // Spatial distribution: which image quadrant are the diff>1 pixels in?
        let hw = cpu_flow.width() / 2;
        let hh = cpu_flow.height() / 2;
        let mut quadrants = [0u32; 4]; // TL, TR, BL, BR
                                       // Also check how many are within 8 pixels of any edge
        let mut edge_count = 0u32;
        for &(col, row, d) in &diffs {
            if d > 1.0 {
                let qi = if row < hh { 0 } else { 2 } + if col < hw { 0 } else { 1 };
                quadrants[qi as usize] += 1;
                if col < 8 || col >= cpu_flow.width() - 8 || row < 8 || row >= cpu_flow.height() - 8
                {
                    edge_count += 1;
                }
            }
        }
        eprintln!(
            "  Pixels with diff>1 by quadrant: TL={} TR={} BL={} BR={}",
            quadrants[0], quadrants[1], quadrants[2], quadrants[3]
        );
        eprintln!(
            "  Of those, within 8px of edge: {edge_count}/{}",
            quadrants.iter().sum::<u32>()
        );

        // Check if divergence is purely in the DIS patch phase by running
        // the full pipeline WITHOUT variational refinement
        // Check if divergence is purely in the variational refinement
        let no_var_params = super::super::DisFlowParams {
            variational_refinement: false,
            ..super::super::DisFlowParams::default_quality()
        };
        let cpu_novar = super::super::compute_optical_flow(&img_a, &img_b, &no_var_params, None);
        let gpu_novar =
            super::super::compute_optical_flow(&img_a, &img_b, &no_var_params, Some(&gpu));
        let (rmse_nv, max_nv) = flow_rmse(&cpu_novar, &gpu_novar);
        eprintln!("  Without variational: RMSE={rmse_nv:.6}, max_diff={max_nv:.4}");

        // Test DIS+variational via the combined path with more outer iterations,
        // starting from the identical DIS-only flow. This isolates whether
        // variational refinement diverges on this image pair.
        let var_3_params = super::super::DisFlowParams {
            variational_outer_iterations_base: 3,
            gpu_min_pixels: 0,
            ..super::super::DisFlowParams::default_quality()
        };
        let cpu_var_flow = super::super::compute_optical_flow(&img_a, &img_b, &var_3_params, None);
        let gpu_var_flow =
            super::super::compute_optical_flow(&img_a, &img_b, &var_3_params, Some(&gpu));
        let (rmse_v, max_v) = flow_rmse(&cpu_var_flow, &gpu_var_flow);
        eprintln!("  DIS+Variational (3 outer): RMSE={rmse_v:.6}, max_diff={max_v:.4}");
    }

    #[test]
    fn test_gpu_vs_cpu_seattle_backyard_panning() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Seattle backyard: frames in the panning section (camera stops and pans right).
        // This pair has higher GPU/CPU divergence (~0.45 RMSE) than others (<0.001).
        // Investigation shows this is NOT a GPU variational bug:
        //   - DIS patch matching without variational: bit-exact (RMSE=0.0)
        //   - Variational from same init flow: near-exact (RMSE=0.00006)
        //   - Per-scale independently: near-exact (<0.00002)
        // The divergence comes from the non-convex DIS inverse search at scale 2:
        // tiny variational differences at coarser scales (RMSE=0.0001) change the
        // initial flow enough to push some patches into different local minima.
        // All affected pixels are in the top-left quadrant where flow vectors exceed
        // 40px (pointing off-frame), a poorly-conditioned occluded region.
        let img_a = load_test_image("seattle_backyard_20.jpg");
        let img_b = load_test_image("seattle_backyard_22.jpg");
        let params = super::super::DisFlowParams::default_quality();

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.5,
            "Seattle backyard panning GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    // --- DIS inverse search + densification tests ---

    #[test]
    fn test_gpu_dis_identical_images() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Without variational refinement, DIS on identical images should give ~0 flow
        let img = GrayImage::checkerboard(64, 64);
        let params = super::super::DisFlowParams {
            variational_refinement: false,
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&img, &img, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img, &img, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.01,
            "DIS-only identical GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_dis_shifted_image() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // DIS without variational on a known shift
        let ref_img = GrayImage::checkerboard(128, 128);
        let tgt_img = GrayImage::shifted(&ref_img, 3.0, 1.5);
        let params = super::super::DisFlowParams {
            variational_refinement: false,
            finest_scale: Some(0),
            coarsest_scale: Some(3),
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&ref_img, &tgt_img, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.1,
            "DIS-only shifted GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_dis_real_images() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        // Real images without variational — isolates DIS inverse search + densification
        let img_a = load_test_image("seoul_bull_sculpture_08.jpg");
        let img_b = load_test_image("seoul_bull_sculpture_09.jpg");
        let params = super::super::DisFlowParams {
            variational_refinement: false,
            ..super::super::DisFlowParams::default_quality()
        };

        let cpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, None);
        let gpu_flow = super::super::compute_optical_flow(&img_a, &img_b, &params, Some(&gpu));

        let (rmse, max_diff) = flow_rmse(&cpu_flow, &gpu_flow);
        assert!(
            rmse < 0.1,
            "DIS-only real images GPU vs CPU RMSE too large: {rmse:.4} (max_diff={max_diff:.4})"
        );
    }

    #[test]
    fn test_gpu_pyramid_vs_cpu_pyramid() {
        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };

        let img = load_test_image("seoul_bull_sculpture_08.jpg");
        let num_levels = 4;

        // Build CPU pyramid
        let cpu_pyr = super::super::pyramid::ImagePyramid::build(&img, num_levels);

        // Build GPU pyramid
        let device = &gpu.ctx.device;
        let queue = &gpu.ctx.queue;

        let pyr_pool = gpu.pyramid_pipeline.create_pool(
            device,
            img.width(),
            img.height(),
            num_levels as usize,
        );

        // Use a dummy image for tgt (we only validate ref)
        let dummy = GrayImage::new_constant(img.width(), img.height(), 0.0);
        gpu.pyramid_pipeline.upload_and_build(
            &pyr_pool,
            device,
            queue,
            &img,
            &dummy,
            num_levels as usize,
        );

        // Read back each GPU level and compare to CPU
        for level in 0..num_levels as usize {
            let cpu_level = cpu_pyr.level(level);
            let n = (cpu_level.width() as usize) * (cpu_level.height() as usize);
            let gpu_data = read_buffer(device, queue, &pyr_pool.ref_level_bufs[level], n);

            let mut max_diff = 0.0f32;
            let mut sum_diff_sq = 0.0f64;
            for (cpu, gpu) in cpu_level.data().iter().zip(gpu_data.iter()) {
                let d = (cpu - gpu).abs();
                max_diff = max_diff.max(d);
                sum_diff_sq += (d as f64).powi(2);
            }
            let rmse = (sum_diff_sq / n as f64).sqrt();

            assert!(
                rmse < 1e-5,
                "Pyramid level {level} RMSE too large: {rmse:.8} (max_diff={max_diff:.8})"
            );
        }
    }

    /// Benchmark: GPU vs CPU optical flow pipeline.
    ///
    /// Run with: cargo test -p sfmtool-core --release --lib gpu::tests::bench_gpu_vs_cpu -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_gpu_vs_cpu() {
        use std::time::Instant;

        let Some(gpu) = GpuFlowContext::new() else {
            eprintln!("Skipping GPU benchmark: no GPU available");
            return;
        };

        let pairs: &[(&str, &str, &str)] = &[
            (
                "seoul_bull",
                "seoul_bull_sculpture_08.jpg",
                "seoul_bull_sculpture_09.jpg",
            ),
            (
                "seattle_backyard",
                "seattle_backyard_10.jpg",
                "seattle_backyard_11.jpg",
            ),
            ("dino_dog_toy", "dino_dog_toy_42.jpg", "dino_dog_toy_43.jpg"),
        ];

        let presets: &[(&str, super::super::DisFlowParams)] = &[
            ("default", super::super::DisFlowParams::default_quality()),
            ("high_quality", super::super::DisFlowParams::high_quality()),
        ];

        for &(pair_name, file_a, file_b) in pairs {
            let img_a = load_test_image(file_a);
            let img_b = load_test_image(file_b);
            eprintln!("\n{pair_name} ({}x{}):", img_a.width(), img_a.height());

            for (preset_name, params) in presets {
                // Warmup
                let _ = super::super::compute_optical_flow(&img_a, &img_b, params, Some(&gpu));
                let _ = super::super::compute_optical_flow(&img_a, &img_b, params, None);

                let n = 5;
                let mut gpu_times = Vec::new();
                let mut cpu_times = Vec::new();

                for _ in 0..n {
                    let t = Instant::now();
                    let _ = super::super::compute_optical_flow(&img_a, &img_b, params, Some(&gpu));
                    gpu_times.push(t.elapsed().as_secs_f64());
                }
                for _ in 0..n {
                    let t = Instant::now();
                    let _ = super::super::compute_optical_flow(&img_a, &img_b, params, None);
                    cpu_times.push(t.elapsed().as_secs_f64());
                }

                gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let gpu_med = gpu_times[n / 2];
                let cpu_med = cpu_times[n / 2];

                eprintln!(
                    "  {preset_name}: GPU={:.0}ms CPU={:.0}ms speedup={:.2}x",
                    gpu_med * 1000.0,
                    cpu_med * 1000.0,
                    cpu_med / gpu_med,
                );
            }
        }
    }
}
