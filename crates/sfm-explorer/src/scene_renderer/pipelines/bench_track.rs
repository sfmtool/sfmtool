// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;

const SHADER: &str = concat!(
    include_str!("../../shaders/common.wgsl"),
    include_str!("../../shaders/bench_track.wgsl")
);

/// The blend of every bench track pipeline, the measurements' included.
///
/// Ordinary alpha rather than the compass's additive blend: the three violets
/// carry meaning, and additive light over a bright point cloud washes them all
/// to white.
pub(in crate::scene_renderer) const BLEND: wgpu::BlendState = wgpu::BlendState::ALPHA_BLENDING;

/// Resources created by the bench track pipelines.
pub(in crate::scene_renderer) struct BenchTrackPipelineResources {
    /// The figure's pieces out of the overlay ribbon store: the frame, the
    /// normal, the barbs and the marks, each a segment.
    pub(crate) edge_pipeline: wgpu::RenderPipeline,
    /// The same pieces into the measurements' layer
    /// ([`MEASUREMENT_TARGET_FORMAT`]): the drawn measurements' rings, dots and
    /// lines, which egui composites as it composites its own painting.
    pub(crate) measurement_pipeline: wgpu::RenderPipeline,
    /// The filled centre disc, a triangle list.
    pub(crate) disc_pipeline: wgpu::RenderPipeline,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
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

    let target = |format| {
        [Some(wgpu::ColorTargetState {
            format,
            blend: Some(BLEND),
            write_mask: wgpu::ColorWrites::ALL,
        })]
    };
    let color_target = target(EDL_OUTPUT_FORMAT);

    // One overlay pipeline per target format: the figure's onto the sRGB view of
    // the EDL output, the measurements' into their encoded layer.
    let edge = |label, targets: &[Option<wgpu::ColorTargetState>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_overlay"),
                buffers: &[
                    // Slot 0: quad corners (per-vertex), reuses the point quad buffer
                    Some(QUAD_VERTEX_LAYOUT),
                    // Slot 1: one piece of the figure or of a measurement (per-instance)
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<OverlayInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 1, // a (xyz + w)
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2, // b (xyz + w)
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 3, // color
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 48,
                                shader_location: 4, // half_width, in points
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 52,
                                shader_location: 5, // radius, in points
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 56,
                                shader_location: 6, // kind
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 60,
                                shader_location: 7, // flags
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
                entry_point: Some("fs_overlay"),
                targets,
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let edge_pipeline = edge("bench track edge pipeline", &color_target);
    let measurement_pipeline = edge(
        "measurement edge pipeline",
        &target(MEASUREMENT_TARGET_FORMAT),
    );

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
        measurement_pipeline,
        disc_pipeline,
        uniform_buffer,
        bind_group_layout,
    }
}
