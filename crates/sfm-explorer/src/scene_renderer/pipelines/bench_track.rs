// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;

const SHADER: &str = include_str!("../../shaders/bench_track.wgsl");

/// Resources created by the bench track pipelines.
pub(in crate::scene_renderer) struct BenchTrackPipelineResources {
    /// Instanced ribbon-quad edges: the frame, the normal, the barbs and the
    /// marks.
    pub edge_pipeline: wgpu::RenderPipeline,
    /// The filled centre disc, a triangle list.
    pub disc_pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub(in crate::scene_renderer) fn create(device: &wgpu::Device) -> BenchTrackPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bench track shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bench track bind group layout"),
        entries: &[
            // binding 0: uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: hardware depth texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bench track pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    // Ordinary alpha rather than the compass's additive blend: the three
    // violets carry meaning, and additive light over a bright point cloud
    // washes them all to white.
    let color_target = [Some(wgpu::ColorTargetState {
        format: EDL_OUTPUT_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bench track edge pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_edge"),
            buffers: &[
                // Slot 0: quad corners (per-vertex), reuses the point quad buffer
                Some(QUAD_VERTEX_LAYOUT),
                // Slot 1: one edge of the figure (per-instance)
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BenchEdgeInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1, // endpoint_a (xyz + w)
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 2, // endpoint_b (xyz + w)
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 3, // color
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
        depth_stencil: None, // depth occlusion is done in the fragment shader
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &color_target,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let disc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bench track disc pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_disc"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<BenchDiscVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0, // position (xyz + w)
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 1, // color
                    },
                ],
            })],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &color_target,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench track uniforms"),
        size: std::mem::size_of::<BenchTrackUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    BenchTrackPipelineResources {
        edge_pipeline,
        disc_pipeline,
        uniform_buffer,
        bind_group_layout,
    }
}
