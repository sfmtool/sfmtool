// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-creation tests against a headless `wgpu` device.
//!
//! Every pipeline in this module compiles a WGSL shader. wgpu-core runs naga's
//! full validation on the noop backend, so building them here is a real check
//! that the shaders parse and type-check — the one thing an edit to a `.wgsl`
//! file cannot otherwise fail on until the GUI is launched on a real GPU.

use super::super::gpu_types::PointUniforms;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions { enable: true },
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
            point_size: 1.0,
            camera_up: [0.0, 1.0, 0.0],
            selected_point_index: u32::MAX,
            hovered_point_index: u32::MAX,
            screen_width: 800.0,
            screen_height: 600.0,
            infinity_point_px: 3.0,
            show_infinity: 1.0,
            _pad: [0.0; 3],
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
