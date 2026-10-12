// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;
use wgpu::util::DeviceExt;

const SHADER: &str = concat!(
    include_str!("../../shaders/common.wgsl"),
    include_str!("../../shaders/points.wgsl")
);

/// Resources created by the point splat pipeline.
///
/// No bind group: it pairs the global uniform buffer below with a per-recon
/// `ReconUniforms` slice, so one is built per loaded reconstruction (see
/// [`ReconResources::new`](crate::scene_renderer::recon::ReconResources::new)).
pub(in crate::scene_renderer) struct PointPipelineResources {
    pub pipeline: wgpu::RenderPipeline,
    /// The same splats, read through the camera model proposal's preview
    /// buffer, blended toward each point's target by the node's `preview_s`.
    pub preview_pipeline: wgpu::RenderPipeline,
    pub quad_vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub(in crate::scene_renderer) fn create(device: &wgpu::Device) -> PointPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("point shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("point bind group layout"),
        entries: &[
            // 0: global camera/selection uniforms
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
            // 1: this reconstruction's ReconUniforms
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
        label: Some("point pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let instance_layouts = [
        // Slot 0: quad vertices (per-vertex)
        Some(QUAD_VERTEX_LAYOUT),
        // Slot 1: point instances (per-instance)
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        }),
        // Slot 2: per-point liveness (per-instance). Its own buffer so the
        // overlay's deleted set can be written into it without rewriting the
        // instance buffer beside it.
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 0,
                shader_location: 3,
            }],
        }),
        // Slot 3, read by the preview variant alone: the camera model
        // proposal's target per point, a `vec4` whose `w` is the preview
        // flag. See `upload::preview`.
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 4]>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 5,
            }],
        }),
    ];

    let make = |label: &str, entry_point: &str, buffers: &[Option<wgpu::VertexBufferLayout>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(entry_point),
                buffers,
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // billboards face the camera, don't cull
                ..Default::default()
            },
            depth_stencil: Some(GBUFFER_DEPTH_STATE),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // Splats are the geometry EDL actually shades, so @location(1)
                // carries their real positive view-space depth.
                targets: &gbuffer_targets(Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)),
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipeline = make("point pipeline", "vs_main", &instance_layouts[..3]);
    // The node a camera model proposal's preview is on is drawn with this
    // variant, which reads the preview buffer in slot 3; every other node
    // with the one above.
    let preview_pipeline = make("point preview pipeline", "vs_preview", &instance_layouts);

    // Static quad vertices (triangle strip)
    let quad_vertices = [
        QuadVertex {
            position: [-1.0, -1.0],
        },
        QuadVertex {
            position: [1.0, -1.0],
        },
        QuadVertex {
            position: [-1.0, 1.0],
        },
        QuadVertex {
            position: [1.0, 1.0],
        },
    ];
    let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quad vertices"),
        contents: bytemuck::cast_slice(&quad_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Point uniform buffer
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("point uniforms"),
        size: std::mem::size_of::<PointUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    PointPipelineResources {
        pipeline,
        preview_pipeline,
        quad_vertex_buffer,
        uniform_buffer,
        bind_group_layout,
    }
}
