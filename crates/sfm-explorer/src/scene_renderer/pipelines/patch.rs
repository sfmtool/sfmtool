// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;

const SHADER: &str = concat!(
    include_str!("../../shaders/common.wgsl"),
    include_str!("../../shaders/patch.wgsl")
);

/// Resources created by the patch surfel pipeline.
///
/// No uniform buffer: `PatchUniforms` carries the atlas grid, which is
/// per-recon, so each bundle allocates its own alongside its atlas.
pub(in crate::scene_renderer) struct PatchPipelineResources {
    pub pipeline: wgpu::RenderPipeline,
    /// The same surfels, moved toward the camera model proposal's targets.
    pub preview_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub(in crate::scene_renderer) fn create(device: &wgpu::Device) -> PatchPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("patch shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("patch bind group layout"),
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
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 3: this reconstruction's ReconUniforms
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("patch pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let instance_layouts = [
        // Slot 0: quad corners (per-vertex), reuses point quad buffer
        Some(QUAD_VERTEX_LAYOUT),
        // Slot 1: patch instances (per-instance)
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PatchInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 1, // center
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 2, // w
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: 3, // u_halfvec
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 32,
                    shader_location: 4, // v_halfvec
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 44,
                    shader_location: 5, // atlas_layer
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 48,
                    shader_location: 6, // point_index
                },
            ],
        }),
        // Slot 2: per-patch liveness (per-instance), the deleted mask's
        // patch half. See `upload::overlay`.
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 0,
                shader_location: 7,
            }],
        }),
        // Slot 3, read by the preview variant alone: the camera model
        // proposal's target centre and flag, and the size factor, per slot.
        // See `upload::preview`.
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfelPreview>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 8,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 16,
                    shader_location: 9,
                },
            ],
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
                cull_mode: None, // double-sided: stored normals can be unreliable
                ..Default::default()
            },
            depth_stencil: Some(GBUFFER_DEPTH_STATE),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // Patches write 0.0 to @location(1): a textured surface should
                // not take EDL edge-darkening.
                targets: &gbuffer_targets(Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)),
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipeline = make("patch pipeline", "vs_main", &instance_layouts[..3]);
    // The node a camera model proposal's preview is on draws its surfels with
    // this variant, which reads the surfel preview buffer in slot 3.
    let preview_pipeline = make("patch preview pipeline", "vs_preview", &instance_layouts);

    PatchPipelineResources {
        pipeline,
        preview_pipeline,
        bind_group_layout,
    }
}
