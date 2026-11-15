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
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// The same surfels, moved toward the camera model proposal's targets.
    pub(crate) preview_pipeline: wgpu::RenderPipeline,
    /// The hidden part of each highlighted surfel's outline, drawn after the
    /// EDL pass onto its output: the surfels of `pipeline`, outlined ones
    /// alone, tested against the scene's depth and writing none.
    pub(crate) hidden_outline_pipeline: wgpu::RenderPipeline,
    /// The same, for the node a camera model proposal's preview is on.
    pub(crate) hidden_outline_preview_pipeline: wgpu::RenderPipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
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
        // Slot 0: the splat corners (per-vertex), the point pipeline's buffer
        Some(SPLAT_CORNER_LAYOUT),
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

    let make = |label: &str,
                entry_point: &str,
                buffers: &[Option<wgpu::VertexBufferLayout>],
                fragment_entry_point: &str,
                targets: &[Option<wgpu::ColorTargetState>],
                depth_stencil: wgpu::DepthStencilState| {
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
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry_point),
                targets,
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    // Patches write 0.0 to @location(1): a textured surface should not take
    // EDL edge-darkening.
    let gbuffer = gbuffer_targets(Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING));
    let pipeline = make(
        "patch pipeline",
        "vs_main",
        &instance_layouts[..3],
        "fs_main",
        &gbuffer,
        GBUFFER_DEPTH_STATE,
    );
    // The node a camera model proposal's preview is on draws its surfels with
    // this variant, which reads the surfel preview buffer in slot 3.
    let preview_pipeline = make(
        "patch preview pipeline",
        "vs_preview",
        &instance_layouts,
        "fs_main",
        &gbuffer,
        GBUFFER_DEPTH_STATE,
    );

    // The hidden part of a highlighted surfel's outline, after the EDL pass:
    // the EDL output alone as its target, in straight alpha as the bench
    // figure blends, so it writes no linear depth and no pick id; and the
    // scene's depth read-only, tested `Less`, which under reversed-Z passes
    // only where something nearer than the outline was drawn.
    let edl_output = [Some(wgpu::ColorTargetState {
        format: EDL_OUTPUT_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let hidden_outline_pipeline = make(
        "patch hidden outline pipeline",
        "vs_hidden_outline",
        &instance_layouts[..3],
        "fs_hidden_outline",
        &edl_output,
        HIDDEN_OUTLINE_DEPTH_STATE,
    );
    let hidden_outline_preview_pipeline = make(
        "patch hidden outline preview pipeline",
        "vs_hidden_outline_preview",
        &instance_layouts,
        "fs_hidden_outline",
        &edl_output,
        HIDDEN_OUTLINE_DEPTH_STATE,
    );

    PatchPipelineResources {
        pipeline,
        preview_pipeline,
        hidden_outline_pipeline,
        hidden_outline_preview_pipeline,
        bind_group_layout,
    }
}
