// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-creation tests against a headless `wgpu` device.
//!
//! Every pipeline in this module compiles a WGSL shader. wgpu-core runs naga's
//! full validation on the noop backend, so building them here is a real check
//! that the shaders parse and type-check — the one thing an edit to a `.wgsl`
//! file cannot otherwise fail on until the GUI is launched on a real GPU.

use super::super::gpu_types::{
    PointUniforms, ReconUniforms, GBUFFER_COLOR_FORMAT, GBUFFER_LINEAR_DEPTH_FORMAT,
    GBUFFER_PICK_FORMAT, HW_DEPTH_FORMAT,
};

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("noop device")
}

#[test]
fn every_shader_compiles() {
    let (device, _queue) = device();
    super::points::create(&device);
    super::edl::create(&device);
    super::frustum::create(&device);
    let image_quad = super::image_quad::create(&device);
    super::patch::create(&device);
    super::target::create(&device);
    super::track_ray::create(&device);
    let bg_image = super::bg_image::create(&device);
    // These two reuse the bind-group layout of the pipeline they extend.
    super::bg_distorted::create(&device, &bg_image.bind_group_layout);
    super::distorted_quad::create(&device, &image_quad.bind_group_layout);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_point_uniforms_struct_fits_the_buffer_the_pipeline_allocates() {
    // The buffer is sized from `size_of::<PointUniforms>()`, so a write of the
    // whole struct must land exactly. wgpu validates the bounds, which is what
    // catches a field added on the Rust side without the padding a uniform
    // buffer's 16-byte alignment demands.
    let (device, queue) = device();
    let resources = super::points::create(&device);
    queue.write_buffer(
        &resources.uniform_buffer,
        0,
        bytemuck::bytes_of(&PointUniforms {
            view_proj: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            camera_right: [1.0, 0.0, 0.0],
            _pad0: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            selected_point_index: u32::MAX,
            hovered_point_index: u32::MAX,
            screen_width: 800.0,
            screen_height: 600.0,
            infinity_point_px: 3.0,
            _pad: [0.0; 4],
        }),
    );
    assert_eq!(
        std::mem::size_of::<PointUniforms>() % 16,
        0,
        "a WGSL uniform struct is rounded up to a 16-byte multiple; the Rust \
         side has to match or the tail of the buffer is garbage"
    );
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_recon_uniforms_struct_matches_its_wgsl_layout() {
    // Five shaders declare this block; all five must agree with the Rust
    // definition, and wgpu validates the write against the buffer size.
    let (device, queue) = device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("recon uniforms"),
        size: std::mem::size_of::<ReconUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &buffer,
        0,
        bytemuck::bytes_of(&ReconUniforms {
            model: [[0.0; 4]; 4],
            point_size: 1.0,
            point_pick_base: 0,
            image_pick_base: 0,
            pickable: 1,
            tint_color: [0.0; 4],
            show_infinity: 1.0,
            _pad: [0.0; 3],
        }),
    );
    assert_eq!(std::mem::size_of::<ReconUniforms>() % 16, 0);
    // mat4 (64) + four scalars (16) + vec4 (16) + a scalar rounded up to the
    // struct's 16-byte alignment (16). The four scalars before the vec4 are
    // sized to satisfy *its* alignment; the trailing pad satisfies the struct's.
    assert_eq!(std::mem::size_of::<ReconUniforms>(), 112);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_gbuffer_pipelines_match_the_textures_sizing_allocates() {
    // The structural tie between `gpu_types`' render-target constants, the
    // textures `scene_renderer::sizing` builds from them, and the pipelines
    // that declare their targets from them. wgpu checks a pipeline's target
    // formats and depth state against the pass it is set on, so binding all
    // five pass-1 pipelines inside a pass assembled the way `ensure_size` +
    // `render` assemble the real one is the whole contract, checked. Without
    // it, a format changed in one place and missed in another is a validation
    // error at the first frame on a real GPU — which no test here has.
    let (device, queue) = device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let attachment = |label, format, usage| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let target = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let color = attachment("splat color", GBUFFER_COLOR_FORMAT, target);
    let linear_depth = attachment("linear depth", GBUFFER_LINEAR_DEPTH_FORMAT, target);
    let pick = attachment("pick buffer", GBUFFER_PICK_FORMAT, target);
    let hw_depth = attachment("hw depth", HW_DEPTH_FORMAT, target);

    let points = super::points::create(&device);
    let frustum = super::frustum::create(&device);
    let image_quad = super::image_quad::create(&device);
    let patch = super::patch::create(&device);
    let distorted_quad = super::distorted_quad::create(&device, &image_quad.bind_group_layout);

    let attach = |view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer contract"),
            color_attachments: &[attach(&color), attach(&linear_depth), attach(&pick)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &hw_depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0), // reversed-Z: 0 = far
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&points.pipeline);
        pass.set_pipeline(&frustum.pipeline);
        pass.set_pipeline(&image_quad.pipeline);
        pass.set_pipeline(&patch.pipeline);
        pass.set_pipeline(&distorted_quad);
    }
    queue.submit([encoder.finish()]);

    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "a pass-1 pipeline disagrees with the G-buffer `sizing` allocates: {error:?}"
    );
}
