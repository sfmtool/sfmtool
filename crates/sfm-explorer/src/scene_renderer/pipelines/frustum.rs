// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;

const SHADER: &str = include_str!("../../shaders/frustum.wgsl");

/// Resources created by the frustum wireframe pipeline.
pub(in crate::scene_renderer) struct FrustumPipelineResources {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub(in crate::scene_renderer) fn create(device: &wgpu::Device) -> FrustumPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("frustum shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frustum bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: this reconstruction's ReconUniforms
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("frustum pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("frustum pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                // Slot 0: quad corners (per-vertex), reuses point quad buffer
                Some(QUAD_VERTEX_LAYOUT),
                // Slot 1: frustum edge instances (per-instance)
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<FrustumEdge>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1, // endpoint_a
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 16,
                            shader_location: 2, // endpoint_b
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 28,
                            shader_location: 3, // frustum_index
                        },
                    ],
                }),
            ],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(GBUFFER_DEPTH_STATE),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            // Frustums write 0.0 to @location(1), actively clearing any point
            // depth they occlude so EDL skips those pixels.
            targets: &gbuffer_targets(Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)),
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    // Frustum uniform buffer
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frustum uniforms"),
        size: std::mem::size_of::<FrustumUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    FrustumPipelineResources {
        pipeline,
        uniform_buffer,
        bind_group_layout,
    }
}
