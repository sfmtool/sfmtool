// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the bench track pass puts on the GPU, against a headless `wgpu`
//! device.
//!
//! The device comes from wgpu's `noop` backend, as
//! [the upload tests](super::super::upload) do: wgpu-core still runs its full
//! validation -- limits, buffer sizes, pipeline and shader checks -- while
//! wgpu-hal stubs the driver calls. So a vertex layout that disagrees with the
//! instance struct, or a uniform block whose Rust side has drifted from the
//! WGSL, is a validation error here rather than a blank pass on a real GPU.

use sfmtool_core::bench::EditableTrack;

use super::super::gpu_types::{BenchDiscVertex, BenchEdgeInstance, BenchTrackUniforms};
use super::super::SceneRenderer;
use crate::scene::ImageRef;
use crate::viewer_3d::bench_track::{self, Figure};

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

/// The figure the staged demo track draws.
fn staged_figure() -> Figure {
    let (state, id, label) = bench_track::tests::staged();
    let track = (**state.bench_track(id, &label).expect("the staged track")).clone();
    bench_track::tests::figure_of(&state, id, &track).expect("a track-stage track draws a figure")
}

#[test]
fn an_upload_holds_one_instance_per_stroke_and_the_disc_beside_it() {
    let (device, _queue) = device();
    let figure = staged_figure();
    let mut r = SceneRenderer::new();

    r.upload_bench_track(&device, Some(&figure));

    assert_eq!(r.bench_track_edge_count, figure.strokes().count() as u32);
    assert_eq!(r.bench_track_disc_count, figure.disc.len() as u32);
    assert!(r.bench_track_edge_buffer.is_some());
    assert!(r.bench_track_disc_buffer.is_some());
    assert_eq!(r.bench_track_fog_distance, figure.fog_distance);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn a_cluster_stage_item_uploads_nothing() {
    let (device, _queue) = device();
    let (mut state, id) = crate::bench::tests::state();
    let seeded = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a cluster started at a pixel of image 0");
    let track = (**state.bench_track(id, &seeded.label).expect("the item")).clone();
    assert!(matches!(
        track.stage,
        sfmtool_core::bench::Stage::Cluster(_)
    ));
    let figure = bench_track::tests::figure_of(&state, id, &track);
    assert!(figure.is_none(), "a cluster has no geometry to draw");

    let mut r = SceneRenderer::new();
    r.upload_bench_track(&device, figure.as_ref());

    assert_eq!(r.bench_track_edge_count, 0);
    assert_eq!(r.bench_track_disc_count, 0);
    assert!(r.bench_track_edge_buffer.is_none());
    assert!(r.bench_track_disc_buffer.is_none());
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn an_upload_after_a_figure_is_gone_drops_the_buffers() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    r.upload_bench_track(&device, Some(&staged_figure()));
    assert!(r.bench_track_edge_count > 0);

    r.upload_bench_track(&device, None);

    assert_eq!(r.bench_track_edge_count, 0);
    assert_eq!(r.bench_track_disc_count, 0);
    assert!(r.bench_track_edge_buffer.is_none());
    assert!(r.bench_track_disc_buffer.is_none());
}

#[test]
fn a_track_with_nothing_to_draw_uploads_nothing() {
    // An `EditableTrack` cannot be at the track stage with no observations and
    // no frame, so the empty figure is stated directly: the guard is about the
    // upload refusing a zero-length buffer, which wgpu rejects.
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    let empty = EditableTrack::empty_cluster();
    let (state, id, _) = bench_track::tests::staged();

    r.upload_bench_track(
        &device,
        bench_track::tests::figure_of(&state, id, &empty).as_ref(),
    );

    assert_eq!(r.bench_track_edge_count, 0);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_pass_pipelines_agree_with_the_buffers_the_upload_writes() {
    // The structural tie between the instance structs, the vertex layouts the
    // pipelines declare from them, and the EDL output the pass draws onto.
    // wgpu checks a pipeline's target format against the pass it is set on and
    // a draw's vertex buffers against its layouts, so encoding the real draw
    // over a real figure is the whole contract, checked.
    let (device, queue) = device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let resources = super::super::pipelines::bench_track::create(&device);
    let output = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("edl output"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::super::gpu_types::EDL_OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("hw depth"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::super::gpu_types::HW_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bench track bind group"),
        layout: &resources.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resources.uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&depth),
            },
        ],
    });

    // The uniform block, written whole: wgpu validates the write against the
    // buffer the pipeline sized from the Rust struct, which is what catches a
    // field added without the padding a uniform buffer's alignment demands.
    queue.write_buffer(
        &resources.uniform_buffer,
        0,
        bytemuck::bytes_of(&BenchTrackUniforms {
            view_proj: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            screen_size: [64.0, 64.0],
            line_half_width: 1.25,
            near: 0.1,
            fog_distance: 1.0,
            _pad: [0.0; 3],
        }),
    );
    assert_eq!(std::mem::size_of::<BenchTrackUniforms>() % 16, 0);

    let figure = staged_figure();
    let edges: Vec<BenchEdgeInstance> = figure
        .strokes()
        .map(|stroke| BenchEdgeInstance {
            endpoint_a: stroke.a,
            endpoint_b: stroke.b,
            color: stroke.color,
        })
        .collect();
    let disc: Vec<BenchDiscVertex> = figure
        .disc
        .iter()
        .map(|vertex| BenchDiscVertex {
            position: vertex.at,
            color: vertex.color,
        })
        .collect();
    let quad = super::super::pipelines::points::create(&device).quad_vertex_buffer;
    let buffer = |label, contents: &[u8]| {
        wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX,
            },
        )
    };
    let edge_buffer = buffer("bench track edges", bytemuck::cast_slice(&edges));
    let disc_buffer = buffer("bench track disc", bytemuck::cast_slice(&disc));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench track pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&resources.disc_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, disc_buffer.slice(..));
        pass.draw(0..disc.len() as u32, 0..1);

        pass.set_pipeline(&resources.edge_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad.slice(..));
        pass.set_vertex_buffer(1, edge_buffer.slice(..));
        pass.draw(0..4, 0..edges.len() as u32);
    }
    queue.submit([encoder.finish()]);

    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "the bench track pass is invalid: {error:?}"
    );
}
