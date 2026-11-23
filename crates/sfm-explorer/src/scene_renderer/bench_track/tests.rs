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

use super::super::gpu_types::{
    BenchDiscVertex, BenchEdgeInstance, BenchTrackUniforms, BENCH_LINE_HALF_WIDTH,
};
use super::super::SceneRenderer;
use crate::scene::ImageRef;
use crate::viewer_3d::bench_track::{self, Figure};
use crate::viewer_3d::measurements::Ribbon;

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
            None,
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
    // The measurements draw into a layer of their own, made as
    // `SceneRenderer::ensure_size` makes it.
    let encoded = device
        .create_texture(&super::super::sizing::measurement_layer_descriptor(64, 64))
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
            near: 0.1,
            fog_distance: 1.0,
        }),
    );
    assert_eq!(std::mem::size_of::<BenchTrackUniforms>() % 16, 0);

    let figure = staged_figure();
    let edges = super::figure_instances(&figure);
    let measurements = super::measurement_instances(&[ribbon(1.0)]);
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
    let measurement_buffer = buffer("measurement edges", bytemuck::cast_slice(&measurements));

    let mut encoder = device.create_command_encoder(&Default::default());
    let begin = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
        encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bench track pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            })
            .forget_lifetime()
    };
    {
        let mut pass = begin(&mut encoder, &encoded);
        pass.set_pipeline(&resources.measurement_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, quad.slice(..));
        pass.set_vertex_buffer(1, measurement_buffer.slice(..));
        pass.draw(0..4, 0..measurements.len() as u32);
    }
    {
        let mut pass = begin(&mut encoder, &output);
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

/// A measurement ribbon from the origin to `x` along the X axis, in a
/// half-transparent orange, 3 pixels wide.
fn ribbon(x: f32) -> Ribbon {
    Ribbon {
        a: [0.0, 0.0, 0.0, 1.0],
        b: [x, 0.0, 0.0, 1.0],
        color: [1.0, 0.5, 0.0, 0.55],
        half_width: 1.5,
    }
}

#[test]
fn the_figure_fades_behind_the_scene_and_the_measurements_do_not() {
    let figure = staged_figure();
    let edges = super::figure_instances(&figure);
    assert_eq!(edges.len(), figure.strokes().count());
    for (edge, stroke) in edges.iter().zip(figure.strokes()) {
        assert_eq!(
            (edge.endpoint_a, edge.endpoint_b, edge.color),
            (stroke.a, stroke.b, stroke.color)
        );
        assert_eq!(edge.half_width, BENCH_LINE_HALF_WIDTH);
        assert_eq!(edge.fade, 1.0, "the figure fades with distance behind");
    }

    let ribbons = [ribbon(1.0), ribbon(2.0)];
    let measurements = super::measurement_instances(&ribbons);
    assert_eq!(
        measurements,
        vec![
            BenchEdgeInstance {
                endpoint_a: ribbons[0].a,
                endpoint_b: ribbons[0].b,
                color: ribbons[0].color,
                half_width: 1.5,
                fade: 0.0,
            },
            BenchEdgeInstance {
                endpoint_a: ribbons[1].a,
                endpoint_b: ribbons[1].b,
                color: ribbons[1].color,
                half_width: 1.5,
                fade: 0.0,
            },
        ],
        "a measurement holds at BEHIND_OPACITY behind the scene"
    );
}

#[test]
fn an_upload_of_measurements_holds_one_instance_per_ribbon_and_nothing_for_none() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();

    r.upload_measurements(&device, &[ribbon(1.0), ribbon(2.0), ribbon(3.0)]);
    assert_eq!(r.measurement_edge_count, 3);
    assert!(r.measurement_edge_buffer.is_some());
    assert_eq!(r.bench_track_edge_count, 0, "the figure's own edges apart");
    assert_eq!(
        r.bench_pass_count(),
        3,
        "the pass draws them with no figure"
    );

    r.upload_measurements(&device, &[]);
    assert_eq!(r.measurement_edge_count, 0);
    assert!(r.measurement_edge_buffer.is_none());
    assert_eq!(r.bench_pass_count(), 0);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

/// The measurements' layer, as the pass fills it and egui paints it, gives
/// what egui's own painting of the same pieces gave, to within one level in
/// 255: a translucent line over its halo, the dimmed others of *Show all
/// measurements* and a dimmed tint, over black, dark greys, a mid grey and
/// white, on a window surface in sRGB and on one in encoded values.
///
/// The layer is a texture of its own in an encoded format with no second view
/// format, opened cleared to transparent: drawing the measurements into an
/// `Rgba8Unorm` view of the `Rgba8UnormSrgb` scene texture instead read the
/// scene back decoded to linear light on a real GPU, and a background of 35
/// came back as 4 in every tile a ribbon touched. The pass's blend is straight
/// alpha for colour and `One, OneMinusSrcAlpha` for alpha, which over a
/// transparent layer leaves egui's premultiplied colour in it. egui then
/// composites the layer exactly as it composites a colour it paints itself:
/// `p + dst · (1 − a)` on encoded values onto an encoded surface, and
/// `linear(p) + dst · (1 − a)` in linear light onto an sRGB one. Blending the
/// colour in linear light onto the scene texture, as the pass first did, drew
/// the dimmed grey over black at 188 where egui painted 135.
#[test]
fn a_translucent_measurement_composites_as_egui_painted_it() {
    use super::super::gpu_types::{EDL_OUTPUT_FORMAT, MEASUREMENT_TARGET_FORMAT};
    use crate::measurement_marks::{HALO, OTHER_OPACITY, STORED_RGB};
    use crate::viewer_3d::measurements::encoded_rgba;
    use egui::ecolor::{gamma_u8_from_linear_f32, linear_f32_from_gamma_u8};
    use egui::Color32;

    let layer = super::super::sizing::measurement_layer_descriptor(8, 8);
    assert_eq!(layer.format, MEASUREMENT_TARGET_FORMAT);
    assert!(!layer.format.is_srgb(), "the layer holds encoded values");
    assert!(layer.view_formats.is_empty(), "the layer has one format");
    assert_ne!(layer.format, EDL_OUTPUT_FORMAT, "it is not the scene's");
    assert_eq!(
        super::MEASUREMENT_LAYER_LOAD,
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
    );
    let blend = wgpu::BlendState::ALPHA_BLENDING;
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Add);

    let level = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    // One channel and the alpha of the layer, as the pass leaves it in 8 bits.
    let fill = |pieces: &[Color32]| {
        let (mut c, mut a) = (0.0_f32, 0.0_f32);
        for piece in pieces {
            let src = encoded_rgba(*piece);
            c = f32::from(level(src[0] * src[3] + c * (1.0 - src[3]))) / 255.0;
            a = f32::from(level(src[3] + a * (1.0 - src[3]))) / 255.0;
        }
        (c, a)
    };
    // egui compositing one premultiplied colour `p` of alpha `a` over the
    // surface, which holds `dst` as the screen shows it, on either surface.
    let composite = |srgb_surface: bool, p: f32, a: f32, dst: f32| {
        if srgb_surface {
            let lin = |v: f32| linear_f32_from_gamma_u8(level(v));
            f32::from(gamma_u8_from_linear_f32(lin(p) + lin(dst) * (1.0 - a))) / 255.0
        } else {
            p + dst * (1.0 - a)
        }
    };
    // egui painting the pieces one after another, as it did before.
    let painted = |srgb_surface: bool, pieces: &[Color32], dst: f32| {
        pieces.iter().fold(dst, |dst, piece| {
            let [p, _, _, a] = piece.to_array().map(|v| f32::from(v) / 255.0);
            composite(srgb_surface, p, a, dst)
        })
    };

    let grey = Color32::from_rgb(STORED_RGB[0], STORED_RGB[1], STORED_RGB[2]);
    let dimmed = grey.gamma_multiply(OTHER_OPACITY);
    let tint = Color32::from_rgb(220, 120, 40).gamma_multiply(OTHER_OPACITY);
    for srgb_surface in [true, false] {
        for pieces in [
            &[HALO, grey][..],
            &[HALO, dimmed],
            &[dimmed],
            &[HALO],
            &[tint],
        ] {
            for background in [0u8, 30, 35, 128, 255] {
                let dst = f32::from(background) / 255.0;
                let (c, a) = fill(pieces);
                let drawn = composite(srgb_surface, c, a, dst);
                let expected = painted(srgb_surface, pieces, dst);
                assert!(
                    (drawn - expected).abs() <= 1.0 / 255.0,
                    "{pieces:?} over {background}, sRGB surface {srgb_surface}: {} != {}",
                    drawn * 255.0,
                    expected * 255.0
                );
            }
        }
    }
    // The first pass: the colour in linear light, blended onto the sRGB scene
    // texture with straight alpha, over black.
    let [r, _, _, a] = dimmed.to_srgba_unmultiplied();
    let first = gamma_u8_from_linear_f32(linear_f32_from_gamma_u8(r) * f32::from(a) / 255.0);
    let before = level(painted(true, &[dimmed], 0.0));
    assert!(first > before + 40, "{first} against {before}");
}
