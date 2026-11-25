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
    mat4_to_cols, BenchTrackUniforms, OverlayInstance, BENCH_LINE_HALF_WIDTH,
};
use super::super::overlay_ribbons::{OverlayTarget, ProducerId};
use super::super::SceneRenderer;
use super::OverlayView;
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
    let (device, queue) = device();
    let figure = staged_figure();
    let mut r = SceneRenderer::new();

    r.upload_bench_track(&device, &queue, Some(&figure));

    assert_eq!(
        r.overlay_ribbons.drawn_count(OverlayTarget::Scene),
        figure.strokes().count() as u32 + 2,
        "a segment per stroke and the arrowhead's two barbs"
    );
    assert_eq!(r.bench_track_disc_count, figure.disc.len() as u32);
    assert!(r.overlay_ribbons.buffer().is_some());
    assert!(r.bench_track_disc_buffer.is_some());
    assert_eq!(r.bench_track_fog_distance, figure.fog_distance);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn a_cluster_stage_item_uploads_nothing() {
    let (device, queue) = device();
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
    r.upload_bench_track(&device, &queue, figure.as_ref());

    assert_eq!(r.overlay_ribbons.drawn_count(OverlayTarget::Scene), 0);
    assert_eq!(r.bench_track_disc_count, 0);
    assert!(
        r.overlay_ribbons.buffer().is_none(),
        "nothing was written, so the store made no buffer"
    );
    assert!(r.bench_track_disc_buffer.is_none());
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn an_upload_after_a_figure_is_gone_draws_nothing_and_keeps_the_room() {
    let (device, queue) = device();
    let mut r = SceneRenderer::new();
    let figure = staged_figure();
    let figure_edges = super::figure_instances(&figure).len() as u32;
    r.upload_bench_track(&device, &queue, Some(&figure));
    assert_eq!(
        r.overlay_ribbons.drawn_count(OverlayTarget::Scene),
        figure_edges
    );
    assert!(figure_edges > 0);

    r.upload_bench_track(&device, &queue, None);

    assert_eq!(r.overlay_ribbons.drawn_count(OverlayTarget::Scene), 0);
    assert_eq!(r.bench_track_disc_count, 0);
    let (_, count, capacity) = r
        .overlay_ribbons
        .region(ProducerId::BenchFigure)
        .expect("the figure's region");
    assert_eq!(count, 0);
    assert!(capacity >= figure_edges, "the region keeps its room");
    assert!(r.bench_track_disc_buffer.is_none());
}

#[test]
fn a_track_with_nothing_to_draw_uploads_nothing() {
    // An `EditableTrack` cannot be at the track stage with no observations and
    // no frame, so the empty figure is stated directly: the guard is about the
    // upload refusing a zero-length buffer, which wgpu rejects.
    let (device, queue) = device();
    let mut r = SceneRenderer::new();
    let empty = EditableTrack::empty_cluster();
    let (state, id, _) = bench_track::tests::staged();

    r.upload_bench_track(
        &device,
        &queue,
        bench_track::tests::figure_of(&state, id, &empty).as_ref(),
    );

    assert_eq!(r.overlay_ribbons.drawn_count(OverlayTarget::Scene), 0);
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
            proj: [[0.0; 4]; 4],
            screen_size: [64.0, 64.0],
            near: 0.1,
            fog_distance: 1.0,
            pixels_per_point: 2.0,
            pull_fraction: 0.01,
            _pad: [0.0; 2],
        }),
    );
    assert_eq!(std::mem::size_of::<BenchTrackUniforms>() % 16, 0);

    let figure = staged_figure();
    let edges = super::figure_instances(&figure);
    assert!(
        edges.iter().any(|edge| edge.kind == OverlayInstance::BARB),
        "the figure's barbs go through the pipeline too"
    );
    // Every kind of piece: a segment, a ring and a disc.
    let measurements = [segment(1.0), ring(), disc()];
    // A track ray to a finite point and one to a point at infinity, whose far
    // end is a direction.
    let rays = [
        OverlayInstance::segment(
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [255, 165, 0, 153],
            0.75,
            OverlayInstance::HOLD,
        ),
        OverlayInstance::segment(
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [255, 165, 0, 153],
            0.75,
            OverlayInstance::HOLD,
        ),
    ];

    // The renderer's own pass, over the pipelines, targets and bind group as
    // `ensure_pipelines` and `ensure_size` make them.
    let mut r = SceneRenderer::new();
    r.bench_track_edge_pipeline = Some(resources.edge_pipeline);
    r.measurement_pipeline = Some(resources.measurement_pipeline);
    r.bench_track_disc_pipeline = Some(resources.disc_pipeline);
    r.quad_vertex_buffer =
        Some(super::super::pipelines::points::create(&device).quad_vertex_buffer);
    r.bench_track_bind_group = Some(bind_group);
    r.edl_output_view = Some(output);
    r.measurement_layer_view = Some(encoded);

    // All three producers' instances in the one store, the measurements
    // written first so that the later writes move their region along: each
    // pass then draws ranges of the shared buffer away from its start.
    r.upload_measurements(&device, &queue, &measurements);
    r.upload_bench_track(&device, &queue, Some(&figure));
    r.overlay_ribbons.write(
        &device,
        &queue,
        ProducerId::TrackRays,
        OverlayTarget::Scene,
        &rays,
    );
    assert!(r.bench_track_disc_count > 0, "the figure's disc is drawn");
    let scene: Vec<_> = r
        .overlay_ribbons
        .draws(OverlayTarget::Scene)
        .map(|(producer, range)| (producer, range.len()))
        .collect();
    assert_eq!(
        scene,
        [
            (ProducerId::TrackRays, rays.len()),
            (ProducerId::BenchFigure, edges.len())
        ],
        "the rays are drawn first, under the figure"
    );
    let layer: Vec<_> = r
        .overlay_ribbons
        .draws(OverlayTarget::MeasurementLayer)
        .collect();
    assert_eq!(layer.len(), 1);
    assert!(layer[0].1.start > 0, "{layer:?}");
    assert_eq!(layer[0].1.len(), measurements.len());

    // With the rays, and with the preview leaving them out.
    for show_track_rays in [true, false] {
        let mut encoder = device.create_command_encoder(&Default::default());
        r.render_bench_track(&mut encoder, show_track_rays);
        queue.submit([encoder.finish()]);
    }

    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "the bench track pass is invalid: {error:?}"
    );
}

/// How a measurement's pieces are drawn.
const MEASUREMENT: u32 = OverlayInstance::HOLD | OverlayInstance::PULL_TOWARD_EYE;

/// A measurement's line from the origin to `x` along the X axis, in a
/// half-transparent orange, 3 points wide.
fn segment(x: f32) -> OverlayInstance {
    OverlayInstance::segment(
        [0.0, 0.0, 0.0, 1.0],
        [x, 0.0, 0.0, 1.0],
        [140, 70, 0, 140],
        1.5,
        MEASUREMENT,
    )
}

/// A measurement's ring at the origin, 5 points to the middle of its stroke.
fn ring() -> OverlayInstance {
    OverlayInstance::ring([0.0, 0.0, 0.0, 1.0], [255; 4], 5.0, 1.0, MEASUREMENT)
}

/// A measurement's dot at the origin.
fn disc() -> OverlayInstance {
    OverlayInstance::disc([0.0, 0.0, 0.0, 1.0], [255; 4], 1.5, MEASUREMENT)
}

/// The figure's pieces are segments, and the arrowhead's two barbs, at the
/// bench's width in points, fading behind the scene and not pulled toward the
/// eye: the strokes, then the barbs.
#[test]
fn the_figure_is_segments_and_barbs_that_fade_behind_the_scene() {
    let figure = staged_figure();
    let edges = super::figure_instances(&figure);
    let normal = figure
        .normal
        .expect("a finite patch frame carries its normal");
    assert_eq!(edges.len(), figure.strokes().count() + 2);
    let (segments, barbs) = edges.split_at(edges.len() - 2);
    for (edge, stroke) in segments.iter().zip(figure.strokes()) {
        assert_eq!(
            (edge.a, edge.b, edge.color.to_le_bytes()),
            (stroke.a, stroke.b, stroke.color)
        );
        assert_eq!(edge.kind, OverlayInstance::SEGMENT);
        assert_eq!((edge.half_width, edge.radius), (BENCH_LINE_HALF_WIDTH, 0.0));
    }
    assert_eq!(segments[4].b, normal.shaft.b, "the normal's segment");
    for (edge, barb) in barbs.iter().zip(normal.barbs) {
        assert_eq!(
            (edge.a, edge.b, edge.color.to_le_bytes(), edge.radius),
            (barb.tip, barb.foot, barb.color, barb.tan_angle)
        );
        assert_eq!(edge.kind, OverlayInstance::BARB);
        assert_eq!(edge.half_width, BENCH_LINE_HALF_WIDTH);
    }
    for edge in &edges {
        assert_eq!(
            edge.flags,
            OverlayInstance::FADE,
            "the figure fades with distance behind, and is not pulled"
        );
    }
}

/// An upload of a figure equal to the one last written writes nothing: not
/// its region of the store, and not its disc, whose buffer is kept. A frame
/// that only moves the camera hands the upload such a figure
/// (`viewer_3d::tests::a_camera_move_leaves_the_bench_figure_as_it_was`). A
/// figure that differs is written, and so is the figure going away, once.
#[test]
fn an_upload_of_the_figure_last_written_writes_nothing() {
    let (device, queue) = device();
    let mut r = SceneRenderer::new();
    let figure = staged_figure();
    let writes = |r: &SceneRenderer| {
        r.overlay_ribbons
            .issued
            .iter()
            .filter(|write| write.producer == ProducerId::BenchFigure)
            .count()
    };

    r.upload_measurements(&device, &queue, &[]);
    r.upload_bench_track(&device, &queue, None);
    assert_eq!(writes(&r), 0, "nothing was written and nothing is there");

    r.upload_bench_track(&device, &queue, Some(&figure));
    let written = writes(&r);
    assert!(written > 0);
    let disc = r.bench_track_disc_buffer.clone().expect("the disc");

    r.upload_bench_track(&device, &queue, Some(&figure.clone()));
    assert_eq!(writes(&r), written, "the same figure was written again");
    assert_eq!(
        r.bench_track_disc_buffer.as_ref(),
        Some(&disc),
        "the disc was made again"
    );

    let mut moved = figure.clone();
    moved.square[0].a[0] += 1.0;
    r.upload_bench_track(&device, &queue, Some(&moved));
    assert_eq!(writes(&r), written + 1, "a changed figure is written");
    assert_ne!(r.bench_track_disc_buffer.as_ref(), Some(&disc));

    r.upload_bench_track(&device, &queue, None);
    assert_eq!(r.overlay_ribbons.drawn_count(OverlayTarget::Scene), 0);
    assert!(r.bench_track_disc_buffer.is_none());
    r.upload_bench_track(&device, &queue, Some(&moved));
    assert_eq!(
        r.overlay_ribbons.drawn_count(OverlayTarget::Scene),
        super::figure_instances(&moved).len() as u32,
        "a figure after none is written again"
    );
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

/// Where `bench_track.wgsl`'s `barb_end` puts a barb's far end, in view space:
/// that function line for line, in `f64`.
fn barb_end(
    tip: nalgebra::Vector3<f64>,
    foot: nalgebra::Vector3<f64>,
    tan_angle: f64,
) -> nalgebra::Vector3<f64> {
    const BARB_MIN_SINE: f64 = 1.0e-5;
    let shaft = tip - foot;
    let back = shaft.norm();
    if back == 0.0 {
        return foot;
    }
    let mut side = shaft.cross(&-tip);
    if side.norm() <= BARB_MIN_SINE * back * tip.norm() {
        side = shaft.cross(&nalgebra::Vector3::y());
        if side.norm() <= BARB_MIN_SINE * back {
            side = shaft.cross(&nalgebra::Vector3::x());
        }
    }
    foot + side.normalize() * (tan_angle * back)
}

/// The barbs the shader draws are the barbs Image Detail's construction,
/// `geometry::arrow`, squares to the same eye, from every view but the one
/// straight down the normal, where the shader stands them across the screen's
/// vertical rather than along the patch's own `u`.
///
/// The shader is mirrored by [`barb_end`] rather than run: the `noop` device
/// draws nothing to read back. The lines of `barb_end` and of the barb's branch
/// of `vs_overlay` are pinned in the shader's text, so a change to either that
/// the mirror does not follow fails here; what the pin cannot catch is a change
/// to the mirror and the shader together that `geometry::arrow` would not
/// agree with, which the comparison below does catch.
#[test]
fn the_shader_s_barbs_are_geometry_s_arrow_squared_to_the_eye() {
    use crate::bench::geometry;
    use nalgebra::{Point3, Vector3};

    let bench = include_str!("../../shaders/bench_track.wgsl");
    for line in [
        "const KIND_BARB: u32 = 3u;",
        "const BARB_MIN_SINE: f32 = 1.0e-5;",
        "if in.kind == KIND_RING || in.kind == KIND_DISC {",
        "if in.kind == KIND_BARB {",
        "vb = vec4<f32>(barb_end(va.xyz, vb.xyz, in.radius), 1.0);",
        "out.shape = vec3<f32>(f32(KIND_SEGMENT), 0.0, half_px);",
        "let shaft = tip - foot;",
        "let back = length(shaft);",
        "var side = cross(shaft, -tip);",
        "if length(side) <= BARB_MIN_SINE * back * length(tip) {",
        "side = cross(shaft, vec3<f32>(0.0, 1.0, 0.0));",
        "if length(side) <= BARB_MIN_SINE * back {",
        "side = cross(shaft, vec3<f32>(1.0, 0.0, 0.0));",
        "return foot + normalize(side) * (tan_angle * back);",
    ] {
        assert!(
            bench.lines().any(|l| l.trim() == line),
            "bench_track.wgsl does not declare `{line}`"
        );
    }
    assert_eq!(OverlayInstance::BARB, 3);

    // The staged figure's arrow, against the construction it was built from.
    let figure = staged_figure();
    let normal = figure
        .normal
        .expect("a finite patch frame carries its normal");
    let v3 = |p: [f32; 4]| Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
    let (tail, tip) = (v3(normal.shaft.a), v3(normal.shaft.b));
    let length = (tip - tail).norm();
    let n = (tip - tail) / length;
    let u = n.cross(&Vector3::z()).normalize();
    let up = n.cross(&u);

    // Off to the side, a little off the normal, and nearly down it. The
    // last is so nearly down it that the `f32` places the instance holds
    // turn the side by a few degrees about the shaft from where `f64` puts
    // it, so there the barb is held to what the side is rather than to
    // `geometry::arrow`'s own: square to the shaft and to the line of sight,
    // at the barb's length.
    for (degrees, roll) in [
        (70.0_f64, 0.0_f64),
        (25.0, 1.0),
        (3.0, 2.0),
        (0.5, 3.0),
        (0.01, 0.5),
    ] {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let towards = n * cos + (u * roll.cos() + up * roll.sin()) * sin;
        let eye = Point3::from(tip + towards * (6.0 * length));
        let camera = crate::viewer_3d::ViewportCamera {
            camera: sfmtool_core::Camera::look_at(eye, Point3::from(tail), up),
            ..Default::default()
        };
        let view = camera.view_matrix();
        let to_view = |p: Vector3<f64>| (view * p.push(1.0)).xyz();
        let drawn = geometry::arrow(Point3::from(tail), n, length, eye, u);
        for (k, barb) in normal.barbs.iter().enumerate() {
            let end = barb_end(
                to_view(v3(barb.tip)),
                to_view(v3(barb.foot)),
                f64::from(barb.tan_angle),
            );
            let expected = to_view(drawn.barbs[k].coords);
            // The `f32` places' rounding turns the side by an angle that
            // grows as the view comes round to the shaft.
            if degrees >= 0.5 {
                let tolerance = 1e-3 * 0.1 * length / degrees.to_radians().sin();
                assert!(
                    (end - expected).norm() < tolerance,
                    "{degrees} degrees off the normal, barb {k}: {end:?} != {expected:?}"
                );
            }
            let (tip, foot) = (to_view(v3(barb.tip)), to_view(v3(barb.foot)));
            let side = end - foot;
            assert!(
                (side.norm() - 0.1 * length).abs() < 1e-5 * length,
                "{degrees} degrees off the normal, barb {k} is {} long",
                side.norm()
            );
            assert!(side.normalize().dot(&(tip - foot).normalize()).abs() < 1e-4);
            assert!(side.normalize().dot(&tip.normalize()).abs() < 1e-4);
        }
    }

    // Straight down the normal: the barbs lie across it, along the screen's
    // horizontal, the same length as from any other view.
    let eye = Point3::from(tip + n * (6.0 * length));
    let camera = crate::viewer_3d::ViewportCamera {
        camera: sfmtool_core::Camera::look_at(eye, Point3::from(tail), up),
        ..Default::default()
    };
    let view = camera.view_matrix();
    let to_view = |p: Vector3<f64>| (view * p.push(1.0)).xyz();
    let ends: Vec<_> = normal
        .barbs
        .iter()
        .map(|barb| {
            barb_end(
                to_view(v3(barb.tip)),
                to_view(v3(barb.foot)),
                f64::from(barb.tan_angle),
            )
        })
        .collect();
    let foot = to_view(v3(normal.barbs[0].foot));
    let across = ends[0] - ends[1];
    assert!(across.y.abs() < 1e-6 * length && across.z.abs() < 1e-6 * length);
    assert!((across.norm() - 2.0 * 0.1 * length).abs() < 1e-5 * length);
    assert!(((ends[0] + ends[1]) / 2.0 - foot).norm() < 1e-6 * length);
}

/// The camera block carries the projection on its own, so the shader can
/// project a piece after pulling it toward the eye in view space, and the
/// frame's pixels per point and pull fraction as the 3D view handed them.
#[test]
fn the_camera_block_carries_the_frame_s_scale_and_pull() {
    let camera = crate::viewer_3d::ViewportCamera {
        fov: 0.9,
        ..Default::default()
    };
    let view = OverlayView {
        pixels_per_point: 1.5,
        pull_fraction: 0.0125,
    };
    let uniforms = super::bench_track_uniforms(&camera, (300, 200), 2.0, view);
    let proj = camera.projection_matrix(1.5);
    assert_eq!(uniforms.proj, mat4_to_cols(&proj));
    assert_eq!(uniforms.view, mat4_to_cols(&camera.view_matrix()));
    assert_eq!(
        uniforms.view_proj,
        mat4_to_cols(&(proj * camera.view_matrix()))
    );
    assert_eq!(uniforms.screen_size, [300.0, 200.0]);
    assert_eq!(uniforms.fog_distance, 2.0);
    assert_eq!(uniforms.pixels_per_point, 1.5);
    assert_eq!(uniforms.pull_fraction, 0.0125);
}

#[test]
fn an_upload_of_measurements_holds_its_instances_and_nothing_for_none() {
    let (device, queue) = device();
    let mut r = SceneRenderer::new();

    r.upload_measurements(&device, &queue, &[segment(1.0), ring(), disc()]);
    assert_eq!(
        r.overlay_ribbons
            .drawn_count(OverlayTarget::MeasurementLayer),
        3
    );
    assert!(r.overlay_ribbons.buffer().is_some());
    assert_eq!(
        r.overlay_ribbons.drawn_count(OverlayTarget::Scene),
        0,
        "the figure's own edges apart"
    );
    assert_eq!(
        r.bench_pass_count(),
        3,
        "the pass draws them with no figure"
    );

    r.upload_measurements(&device, &queue, &[]);
    assert_eq!(
        r.overlay_ribbons
            .drawn_count(OverlayTarget::MeasurementLayer),
        0
    );
    let (_, count, capacity) = r
        .overlay_ribbons
        .region(ProducerId::Measurements)
        .expect("the measurements' region");
    assert_eq!(
        (count, capacity >= 3),
        (0, true),
        "emptied, with its room kept"
    );
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
/// came back as 4 in every tile a ribbon touched. The pass's blend is
/// premultiplied alpha, `One, OneMinusSrcAlpha`, and the pieces carry egui's
/// premultiplied bytes as they are, so over a transparent layer one piece
/// leaves exactly those bytes in it. egui then composites the layer exactly as
/// it composites a colour it paints itself:
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
    let blend = super::super::pipelines::bench_track::BLEND;
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
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
            // What `vs_measurement` hands on: the bytes over 255, not decoded.
            let [p, _, _, pa] = encoded_rgba(*piece).map(|v| f32::from(v) / 255.0);
            c = f32::from(level(p + c * (1.0 - pa))) / 255.0;
            a = f32::from(level(pa + a * (1.0 - pa))) / 255.0;
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
    // One translucent piece leaves egui's own bytes in the layer, exactly.
    for piece in [dimmed, tint, HALO] {
        let [p, _, _, a] = piece.to_array();
        assert_eq!(
            fill(&[piece]),
            (f32::from(p) / 255.0, f32::from(a) / 255.0),
            "{piece:?}"
        );
    }
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
