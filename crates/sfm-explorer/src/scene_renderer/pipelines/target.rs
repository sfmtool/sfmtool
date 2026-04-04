// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("../../shaders/target_indicator.wgsl");

/// Resources created by the target indicator pipeline.
pub(in crate::scene_renderer) struct TargetPipelineResources {
    /// Wireframe edge pipeline (vertical axis + circular ring).
    pub edge_pipeline: wgpu::RenderPipeline,
    pub edge_buffer: wgpu::Buffer,
    pub edge_count: u32,
    /// Filled star polygon pipeline.
    pub star_pipeline: wgpu::RenderPipeline,
    pub star_buffer: wgpu::Buffer,
    pub star_vertex_count: u32,
    /// Shared resources.
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Additive blend state shared by both edge and star pipelines.
const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

pub(in crate::scene_renderer) fn create(device: &wgpu::Device) -> TargetPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("target indicator shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("target indicator bind group layout"),
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
        label: Some("target indicator pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let color_target = [Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: Some(ADDITIVE_BLEND),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    // ── Wireframe edge pipeline (vertical axis + circular ring) ──

    let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("target indicator edge pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                // Slot 0: quad corners (per-vertex), reuses point quad buffer
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                // Slot 1: compass edge instances (per-instance, with width factors)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CompassEdgeInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1, // endpoint_a (xyz + width)
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 2, // endpoint_b (xyz + width)
                        },
                    ],
                },
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &color_target,
            compilation_options: Default::default(),
        }),
        multiview: None,
        cache: None,
    });

    // ── Filled star polygon pipeline ──

    let star_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("target indicator star pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_star"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
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
            entry_point: Some("fs_star"),
            targets: &color_target,
            compilation_options: Default::default(),
        }),
        multiview: None,
        cache: None,
    });

    // ── Vertex buffers ──

    // Wireframe edges (vertical axis + circular ring)
    let edge_instances = create_compass_edge_instances();
    let edge_count = edge_instances.len() as u32;
    let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compass edges"),
        contents: bytemuck::cast_slice(&edge_instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Filled star polygon
    let star_mesh = create_compass_star_mesh();
    let star_vertex_count = star_mesh.len() as u32;
    let star_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compass star"),
        contents: bytemuck::cast_slice(&star_mesh),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Shared uniform buffer
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("target indicator uniforms"),
        size: std::mem::size_of::<TargetIndicatorUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    TargetPipelineResources {
        edge_pipeline,
        edge_buffer,
        edge_count,
        star_pipeline,
        star_buffer,
        star_vertex_count,
        uniform_buffer,
        bind_group_layout,
    }
}