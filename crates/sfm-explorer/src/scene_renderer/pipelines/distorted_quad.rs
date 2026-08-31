// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::gpu_types::*;

const SHADER: &str = include_str!("../../shaders/distorted_quad.wgsl");

pub(in crate::scene_renderer) fn create(
    device: &wgpu::Device,
    image_quad_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("distorted quad shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    // Reuses the same bind group layout as image_quad (uniforms + texture array + sampler)
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("distorted quad pipeline layout"),
        bind_group_layouts: &[Some(image_quad_bind_group_layout)],
        ..Default::default()
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("distorted quad pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<DistortedQuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0, // position
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 12,
                        shader_location: 1, // frustum_index
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 16,
                        shader_location: 2, // uv
                    },
                ],
            })],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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
            targets: &gbuffer_targets(None),
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}
