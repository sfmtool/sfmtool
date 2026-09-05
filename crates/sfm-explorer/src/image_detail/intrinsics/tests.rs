// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the intrinsics overlay layer.
//!
//! Two kinds. What the layer *decides* — the trustworthy split of the
//! displacement field, the auto scale, the tick ladder — is tested without a
//! frame, because a painted string cannot show any of it. What the layer *says*
//! comes off a real headless egui frame's galleys
//! (`test_support::painted_texts`, the `point_track_detail/tests.rs` pattern).
//!
//! And one test that is neither: every glyph the layer writes has to be in
//! egui's bundled fonts. A glyph that is not renders as a replacement box
//! rather than failing, and asserting a galley's *string* passes happily on
//! characters the font cannot draw — which is how a `──`/`╌╌` legend shipped
//! in phase 5's first draft.

use sfmtool_core::camera::{CameraIntrinsics, CameraModel};

use super::{controls, CameraLayer};
use crate::state::IntrinsicsDisplaySettings;

/// `kerry_park`'s real intrinsics: an `OPENCV_FISHEYE` on a 480 × 480 frame,
/// whose image rectangle's corners are outside the lens's image circle.
fn kerry_park_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.1499937015594,
            focal_length_y: 129.2573627423474,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.038113353966529886,
            radial_distortion_k2: -0.00800851799065643,
            radial_distortion_k3: 0.008329720504707577,
            radial_distortion_k4: -0.0026901578801066814,
        },
        width: 480,
        height: 480,
    }
}

/// An ordinary `SIMPLE_RADIAL`, the `seoul_bull_sculpture` shape: trustworthy
/// at every angle its projective divide accepts.
fn simple_radial_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 344.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.035,
        },
        width: 270,
        height: 480,
    }
}

/// A plain pinhole: its own ideal map, so the field is identically zero.
fn pinhole_camera() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 900.0,
            focal_length_y: 900.0,
            principal_point_x: 640.0,
            principal_point_y: 360.0,
        },
        width: 1280,
        height: 720,
    }
}

#[test]
fn the_layers_glyphs_are_available_in_the_bundled_fonts() {
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(super::LABEL_SIZE);
    for glyph in [
        super::DELTA,
        super::MINUS,
        super::MIDDOT,
        super::DEGREE,
        controls::TIMES,
        controls::GEAR,
    ] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

/// The layer's numbers are [`report::distortion_extent`]'s, not a second
/// summary of the same field.
///
/// What that function computes — the square-celled grid, the maximum over the
/// trusted nodes, the excluded count — is pinned in `camera/report/tests.rs`,
/// on both a bounded fisheye and an unbounded radial. What is the layer's own
/// is this plumbing, and that the arrows are the extent's field split by its
/// own trust predicate rather than by a second reading of the bound.
#[test]
fn the_layer_reports_the_extent_the_core_computed() {
    use sfmtool_core::camera::report;

    for (camera, cols) in [
        (kerry_park_camera(), 16),
        (simple_radial_camera(), 16),
        (pinhole_camera(), 12),
    ] {
        let extent = report::distortion_extent(&camera, cols);
        let layer = CameraLayer::compute(&camera, cols);
        assert_eq!(layer.grid, extent.grid);
        assert_eq!(layer.limit_deg, extent.limit_deg);
        assert_eq!(layer.max_px, extent.max_px);
        assert_eq!(layer.extrapolated, (extent.excluded, extent.total()));
        assert_eq!(layer.arrows.len(), extent.field.len());
        let marked = layer.arrows.iter().filter(|a| !a.trusted).count();
        assert_eq!(marked, extent.excluded);
    }
}

#[test]
fn a_centred_principal_point_gets_no_offset_label() {
    // Both checked-in fixtures put the principal point exactly at the image
    // centre, which is the case where a connector and a `Δ 0.0 px` would be
    // pure noise on top of the marker that already names the spot.
    assert_eq!(super::offset_label(&kerry_park_camera()), None);
    assert_eq!(super::offset_label(&pinhole_camera()), None);
}

#[test]
fn an_offset_principal_point_is_labelled_in_pixels_and_in_percent() {
    // 135 − 270/2 = 0, 240 − 480/2 = 0; move it off centre to see the label.
    let mut camera = simple_radial_camera();
    let CameraModel::SimpleRadial {
        principal_point_x,
        principal_point_y,
        ..
    } = &mut camera.model
    else {
        unreachable!("fixture is a SIMPLE_RADIAL")
    };
    *principal_point_x -= 12.4;
    *principal_point_y += 3.1;

    let label = super::offset_label(&camera).expect("off centre by 12.8 px");
    assert!(
        label.contains("−12.4") && label.contains("+3.1"),
        "signed pixel offsets, got {label:?}"
    );
    // 12.78 px against a half-diagonal of 275.4 px.
    assert!(
        label.ends_with("4.6%"),
        "percent of half-diagonal: {label:?}"
    );
}

// ── The angular grid ────────────────────────────────────────────────────

/// The grid as the panel would sample it at `scale` panel pixels per image
/// pixel, with the rings on so the contour paths are exercised too.
fn grid(camera: &CameraIntrinsics, scale: f32) -> super::axes::AxisGeometry {
    super::axes::AxisGeometry::compute(
        camera,
        scale,
        true,
        sfmtool_core::camera::report::trustworthy_max_theta_deg(camera),
    )
}

/// The ladder step the grid would draw at `scale`.
fn ladder(camera: &CameraIntrinsics, scale: f32) -> f64 {
    super::axes::tick_ladder(camera, scale, super::axes::axis_reaches(camera, scale))
}

#[test]
fn the_tick_ladder_gets_finer_as_the_panel_zooms_in() {
    let camera = simple_radial_camera();
    // Fitted to a panel (about 1.4 panel px per image px) against 8× into it.
    let fitted = ladder(&camera, 1.4);
    let zoomed = ladder(&camera, 1.4 * 8.0);
    assert!(
        zoomed < fitted,
        "8× in should not still be on the {fitted}° ladder"
    );
    // And every step is one the ladder actually offers.
    for step in [fitted, zoomed] {
        assert!(
            [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0].contains(&step),
            "{step} is not on the ladder"
        );
    }
    // A 129 px/rad fisheye fitted to a panel cannot carry a fine ladder, and a
    // 344 px/rad lens at the same scale can: the ladder is per camera as much
    // as per zoom.
    assert!(ladder(&kerry_park_camera(), 1.4) > fitted);
}

#[test]
fn a_radial_model_keeps_its_axes_straight_and_bunches_their_ticks() {
    // Sagitta of the vertical axis: how far it departs from the straight line
    // between its ends. Radial distortion moves every point along its own
    // radius from the principal point, so an axis *through* the principal point
    // is straight however violent the lens — this is the spec's "they are not
    // straight" corrected against the real projection, and the fixtures are
    // both radial.
    let bend = |camera: &CameraIntrinsics| {
        let axis: Vec<[f64; 2]> = grid(camera, 1.4).vertical.iter().map(|(p, _)| *p).collect();
        assert!(axis.len() > 8, "the axis should be densely sampled");
        let (first, last) = (axis[0], axis[axis.len() - 1]);
        axis.iter()
            .map(|p| {
                let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
                let length = dx.hypot(dy).max(1e-9);
                ((p[0] - first[0]) * dy - (p[1] - first[1]) * dx).abs() / length
            })
            .fold(0.0_f64, f64::max)
    };
    assert!(bend(&pinhole_camera()) < 1e-9);
    assert!(bend(&kerry_park_camera()) < 0.01, "a fisheye's too");

    // What the fisheye's distortion does show is in the tick spacing: equal
    // angular steps landing at unequal pixel steps.
    let camera = kerry_park_camera();
    let ticks: Vec<f64> = grid(&camera, 1.4)
        .ticks
        .iter()
        .filter(|t| t.axis == super::axes::Axis::Vertical)
        .map(|t| t.at[1])
        .collect();
    let gaps: Vec<f64> = ticks.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let (min, max) = gaps
        .iter()
        .fold((f64::MAX, 0.0_f64), |(lo, hi), g| (lo.min(*g), hi.max(*g)));
    assert!(
        max > 1.02 * min,
        "equal angles should not land at equal pixels on a fisheye: {gaps:?}"
    );
}

#[test]
fn the_grid_stops_where_the_frame_does() {
    // Every sample of every part of the grid is on the frame, give or take the
    // 5% of the diagonal the sweep is allowed to overshoot by.
    let camera = kerry_park_camera();
    let grid = grid(&camera, 1.4);
    let margin = 0.05 * f64::from(camera.width).hypot(f64::from(camera.height));
    let inside = |p: &[f64; 2]| {
        p[0] >= -margin
            && p[1] >= -margin
            && p[0] <= f64::from(camera.width) + margin
            && p[1] <= f64::from(camera.height) + margin
    };
    assert!(grid.horizontal.iter().all(|(p, _)| inside(p)));
    assert!(grid.vertical.iter().all(|(p, _)| inside(p)));
    assert!(grid
        .rings
        .iter()
        .flat_map(|r| &r.runs)
        .flatten()
        .all(inside));
}

#[test]
fn the_axes_stop_where_the_projection_folds_and_go_dashed_before_that() {
    // `kerry_park`'s polynomial turns over near 130°: past the turn the radius
    // crashes from 191 px back to 6 px, and an unguarded sweep draws the axis
    // back through everything it has already drawn, putting a confident `−120°`
    // tick between the `−60°` and `−30°` ones. So the axis must be monotone in
    // radius from the principal point, all the way out.
    let camera = kerry_park_camera();
    let axis = grid(&camera, 1.4).horizontal;
    let (cx, cy) = camera.principal_point();
    let radius = |p: &[f64; 2]| (p[0] - cx).hypot(p[1] - cy);
    let turn = axis.len() / 2;
    for half in [&axis[..=turn], &axis[turn..]] {
        let radii: Vec<f64> = half.iter().map(|(p, _)| radius(p)).collect();
        let monotone = radii.windows(2).all(|w| w[0] >= w[1] - 1e-9)
            || radii.windows(2).all(|w| w[0] <= w[1] + 1e-9);
        assert!(monotone, "an axis is a scale only where it is monotone");
    }

    // And the part outside the trustworthy bound is flagged for dashing rather
    // than dropped, so the reader can still see how much frame is out there.
    assert!(axis.iter().any(|(_, trusted)| *trusted));
    assert!(axis.iter().any(|(_, trusted)| !*trusted));
    // An unbounded model has no dashed part at all.
    assert!(grid(&simple_radial_camera(), 1.4)
        .horizontal
        .iter()
        .all(|(_, trusted)| *trusted));
}

#[test]
fn the_ticks_and_rings_stop_at_the_trustworthy_bound() {
    // A tick is a claim that this pixel is N degrees off axis, and the rings
    // carry the same claim; neither is made where the model has stopped
    // describing a lens. `kerry_park` is bounded at 84.1°, and its frame's
    // mid-edge is past 100°.
    let camera = kerry_park_camera();
    let grid = grid(&camera, 1.4);
    let limit = camera_limit(&camera);
    for tick in &grid.ticks {
        let angle =
            sfmtool_core::camera::report::off_axis_angle_deg(&camera, tick.at[0], tick.at[1]);
        assert!(
            angle <= limit + 0.5,
            "a {} tick sits at {angle}°, past the {limit}° bound",
            tick.label
        );
    }
    for ring in &grid.rings {
        let angle: f64 = ring.label.trim_end_matches(super::DEGREE).parse().unwrap();
        assert!(angle <= limit, "a {angle}° ring is past the bound");
    }
}

/// The angle past which `camera` is extrapolating.
fn camera_limit(camera: &CameraIntrinsics) -> f64 {
    sfmtool_core::camera::report::trustworthy_max_theta_deg(camera).expect("a bounded fixture")
}

#[test]
fn a_bounded_model_gets_a_contour_where_it_stops_describing_a_lens() {
    let edge = grid(&kerry_park_camera(), 1.4)
        .trustworthy_edge
        .expect("kerry_park's OPENCV_FISHEYE is bounded, and the contour is on frame");
    assert!(edge.label.starts_with("extrapolated past 84."));
    assert!(!edge.runs.is_empty());

    // And an unbounded model has nothing to mark.
    assert!(grid(&simple_radial_camera(), 1.4)
        .trustworthy_edge
        .is_none());
    assert!(grid(&simple_radial_camera(), 1.4).spline_domain.is_none());
}

#[test]
fn the_tick_labels_are_signed_and_carry_their_degree_sign() {
    let ticks = grid(&simple_radial_camera(), 1.4).ticks;
    let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
    assert!(labels.contains(&"0°"), "{labels:?}");
    assert!(
        labels.iter().any(|l| l.starts_with('+')),
        "positive angles are marked as such: {labels:?}"
    );
    assert!(
        labels.iter().any(|l| l.starts_with(super::MINUS)),
        "negative angles use the typographic minus: {labels:?}"
    );
    // Zero is labelled once, not once per axis, because both axes cross there
    // and the principal-point reticle is already sitting on it.
    assert_eq!(labels.iter().filter(|l| **l == "0°").count(), 1);
}

// ── The displacement field ──────────────────────────────────────────────

#[test]
fn the_auto_scale_is_fitted_to_the_lens_and_not_to_the_fold() {
    // The spec's rule is "the smallest exaggeration that brings the largest
    // displacement in the grid to at least 8 panel pixels". On `kerry_park` the
    // largest displacement in the grid is a 273 px fold in the black corners,
    // which would pick ×1 and leave every real arrow invisible.
    let camera = kerry_park_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let cell = f64::from(camera.width) / 16.0;
    // A 480 px frame in a narrow Image Detail panel, where the lens's own 13 px
    // of displacement does not yet reach 8 panel pixels and the fold's 273 px
    // does.
    let fit = 0.3_f32;

    let fitted_to_the_lens = super::field::auto_scale(layer.max_px, cell, fit);
    let unfiltered = sfmtool_core::camera::report::distortion_field(&camera, 16, 16)
        .iter()
        .map(|s| (s.pixel[0] - s.reference[0]).hypot(s.pixel[1] - s.reference[1]))
        .fold(0.0_f64, f64::max);
    let fitted_to_the_fold = super::field::auto_scale(unfiltered, cell, fit);

    assert_eq!(
        fitted_to_the_fold, 1.0,
        "the fold's {unfiltered:.0} px clears the floor on its own, so the \
         spec's unfiltered rule exaggerates nothing"
    );
    assert!(
        fitted_to_the_lens > fitted_to_the_fold,
        "the lens's {:.1} px needs exaggerating, and got {fitted_to_the_lens}",
        layer.max_px
    );
}

#[test]
fn no_arrow_outgrows_its_own_grid_cell() {
    // A mild lens would otherwise be exaggerated until the field is a tangle.
    for cols in [8, 16, 32] {
        let camera = simple_radial_camera();
        let layer = CameraLayer::compute(&camera, cols);
        let cell = f64::from(camera.width) / cols as f64;
        let scale = super::field::auto_scale(layer.max_px, cell, 1.4);
        assert!(
            f64::from(scale) * layer.max_px <= cell || scale == 1.0,
            "at {cols} across, {scale} times {:.2} px exceeds the {cell:.1} px cell",
            layer.max_px
        );
    }
}

#[test]
fn the_auto_scale_does_not_move_when_the_panel_zooms() {
    // It is fitted at the panel's *fit* scale, which is why the legend's
    // exaggeration holds still while the user pans and zooms.
    let camera = simple_radial_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let cell = f64::from(camera.width) / 16.0;
    let at_fit = super::field::auto_scale(layer.max_px, cell, 1.4);
    for zoom in [2.0, 8.0, 32.0] {
        assert_eq!(
            super::field::auto_scale(layer.max_px, cell, 1.4),
            at_fit,
            "the scale is a function of the fit, not of the {zoom}× zoom"
        );
    }
}

#[test]
fn the_correction_points_the_way_the_sign_of_k1_says_it_should() {
    // The property, not the numbers: an expected-value assertion is satisfied
    // by a sign-flipped constant and would have passed on the field pointing
    // backwards, which is what shipped until someone looked at it.
    //
    // `SIMPLE_RADIAL` is `r_d = f·ρ·(1 + k1·ρ²)`, so `k1 > 0` puts the model's
    // pixel *outside* the ideal one and the correction — from the real pixel
    // toward where its content belongs — points **inward**; `k1 < 0` puts it
    // inside and the correction points **outward**. Both signs are checked, so
    // the test pins the convention against the optics rather than against one
    // fixture that happens to have a sign.
    for (k1, inward) in [(0.035, true), (-0.035, false)] {
        let camera = CameraIntrinsics {
            model: CameraModel::SimpleRadial {
                focal_length: 344.0,
                principal_point_x: 135.0,
                principal_point_y: 240.0,
                radial_distortion_k1: k1,
            },
            width: 270,
            height: 480,
        };
        let (cx, cy) = camera.principal_point();
        let layer = CameraLayer::compute(&camera, 16);

        let mut checked = 0;
        for arrow in layer.arrows.iter().filter(|a| a.trusted) {
            // The tail is the grid node itself — a real pixel of the image the
            // field is drawn on.
            let step = (
                arrow.reference[0] - arrow.pixel[0],
                arrow.reference[1] - arrow.pixel[1],
            );
            if step.0.hypot(step.1) < 1e-9 {
                continue; // the node at the principal point has nowhere to point
            }
            let out = (arrow.pixel[0] - cx, arrow.pixel[1] - cy);
            let radial = out.0 * step.0 + out.1 * step.1;
            assert_eq!(
                radial < 0.0,
                inward,
                "k1 = {k1}: the arrow at {:?} points the wrong way",
                arrow.pixel
            );
            checked += 1;
        }
        assert!(checked > 0, "k1 = {k1}: no arrow had a direction to check");
    }
}

#[test]
fn the_field_is_split_into_measurements_and_marked_nodes() {
    let layer = CameraLayer::compute(&kerry_park_camera(), 16);
    let drawn = layer.arrows.iter().filter(|a| a.trusted).count();
    let marked = layer.arrows.iter().filter(|a| !a.trusted).count();
    assert!(drawn > 0 && marked > 0);
    assert_eq!(marked, layer.extrapolated.0);
    assert_eq!(drawn + marked, layer.extrapolated.1);
}

// ── Composition with the feature layer ──────────────────────────────────
//
// The property the whole design rests on: this is a layer, not an eighth
// `OverlayMode`. So a whole panel frame is run with each mode in turn, twice,
// and what the feature layer painted must not move when the intrinsics layer is
// switched on underneath it.

/// One whole `ImageDetail::show` frame, returning every shape it painted.
///
/// The panel needs real CPU pixels to build its texture from and a real SIFT
/// cache to draw features from, so both are synthesized: egui allocates
/// textures on the CPU, and `run_frame_headless` discards the delta.
fn panel_frame(
    node: &crate::scene::SceneNode,
    sift: &crate::state::CachedSiftFeatures,
    image: &sfmtool_core::camera::remap::ImageU8,
    feature_display: &crate::state::FeatureDisplaySettings,
    intrinsics: &mut IntrinsicsDisplaySettings,
) -> Vec<egui::Shape> {
    panel_frame_with_input(
        node,
        sift,
        image,
        feature_display,
        intrinsics,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            ..Default::default()
        },
    )
}

/// The same frame, with the caller's own input events — a pointer position and
/// a keystroke, for the bindings the panel owns.
fn panel_frame_with_input(
    node: &crate::scene::SceneNode,
    sift: &crate::state::CachedSiftFeatures,
    image: &sfmtool_core::camera::remap::ImageU8,
    feature_display: &crate::state::FeatureDisplaySettings,
    intrinsics: &mut IntrinsicsDisplaySettings,
    input: egui::RawInput,
) -> Vec<egui::Shape> {
    let mut detail = super::super::ImageDetail::new();
    let ctx = egui::Context::default();

    // A warm-up pass carrying the pointer but no keystrokes. egui resolves
    // hover against the *previous* pass's widget rects, so on a fresh context's
    // first frame nothing is hovered and the panel's own key bindings — `Z`,
    // and now `I` — would never fire.
    let warm_up = egui::RawInput {
        screen_rect: input.screen_rect,
        events: input
            .events
            .iter()
            .filter(|event| !matches!(event, egui::Event::Key { .. }))
            .cloned()
            .collect(),
        ..Default::default()
    };
    let mut warm = ctx.run_ui(warm_up, |ui| {
        detail.show(
            ui,
            &node.recon,
            node.id,
            Some(0),
            None,
            None,
            &[],
            &crate::platform::ScrollInput::default(),
            Some(sift),
            Some(image),
            feature_display,
            intrinsics,
        );
    });
    warm.textures_delta.clear();

    let mut output = ctx.run_ui(input, |ui| {
        detail.show(
            ui,
            &node.recon,
            node.id,
            Some(0),
            None,
            None,
            &[],
            &crate::platform::ScrollInput::default(),
            Some(sift),
            Some(image),
            feature_display,
            intrinsics,
        );
    });
    output.textures_delta.clear();
    output
        .shapes
        .into_iter()
        .map(|clipped| clipped.shape)
        .collect()
}

/// Every shape in a frame, flattened out of the nested `Shape::Vec`s and
/// reduced to a comparable key.
///
/// Text is keyed by its position and its string rather than by its whole debug
/// form: a galley carries the atlas UVs of its glyphs, and the intrinsics layer
/// rasterizes `°`, `·` and `×` into that atlas before the colorbar's own labels
/// reach it, which moves every later glyph's UVs without moving a pixel of the
/// colorbar on screen.
fn flatten(shapes: &[egui::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::Shape::Vec(inner) => flatten(inner, out),
            egui::Shape::Text(text) => {
                out.push(format!("Text({:?}, {:?})", text.pos, text.galley.text()))
            }
            other => out.push(format!("{other:?}")),
        }
    }
}

/// A node, a SIFT cache and a stand-in photograph the panel can really run on.
///
/// One reconstruction per test rather than one per frame: `demo` relaxes its
/// points onto a sphere from a random start, so two calls do not agree and the
/// heatmap modes' value ranges would differ for reasons having nothing to do
/// with the layer.
fn demo_panel_fixture() -> (
    crate::scene::SceneNode,
    crate::state::CachedSiftFeatures,
    sfmtool_core::camera::remap::ImageU8,
) {
    use sfmtool_core::camera::remap::ImageU8;

    let node = crate::scene::SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        sfmtool_core::SfmrReconstruction::demo(64),
    );
    let camera = &node.recon.cameras[0];
    let (w, h) = (
        f32::from(camera.width as u16),
        f32::from(camera.height as u16),
    );
    // A tiny stand-in for the photograph: the panel only uploads it.
    let image = ImageU8::new(8, 8, 3, vec![90u8; 8 * 8 * 3]);
    // Features spread over the frame, one per tracked keypoint index.
    let count = node.recon.image_feature_to_point[0].len().max(1);
    let sift = crate::state::CachedSiftFeatures {
        positions_xy: (0..count)
            .map(|i| {
                [
                    (i % 8) as f32 * w / 8.0 + 40.0,
                    (i / 8) as f32 * h / 8.0 + 30.0,
                ]
            })
            .collect(),
        affine_shapes: vec![[[6.0, 0.0], [0.0, 6.0]]; count],
        read_count: count,
    };
    (node, sift, image)
}

#[test]
fn the_layer_composes_with_every_feature_mode_without_disturbing_it() {
    use crate::state::{FeatureDisplaySettings, OverlayMode};

    let (node, sift, image) = demo_panel_fixture();
    for mode in OverlayMode::ALL {
        let feature_display = FeatureDisplaySettings {
            overlay_mode: mode,
            tracked_only: false,
            ..Default::default()
        };

        let mut off = IntrinsicsDisplaySettings {
            enabled: false,
            ..Default::default()
        };
        let mut on = IntrinsicsDisplaySettings::default();

        let (mut without, mut with) = (Vec::new(), Vec::new());
        flatten(
            &panel_frame(&node, &sift, &image, &feature_display, &mut off),
            &mut without,
        );
        flatten(
            &panel_frame(&node, &sift, &image, &feature_display, &mut on),
            &mut with,
        );

        assert!(
            with.len() > without.len(),
            "{mode:?}: the layer should put ink on the image"
        );
        // Every shape the mode drew on its own is still there, unmoved: the
        // layer draws beneath the features and takes nothing away.
        for shape in &without {
            assert!(
                with.contains(shape),
                "{mode:?}: enabling the layer changed what the feature layer \
                 drew — missing {shape}"
            );
        }
    }
}

#[test]
fn i_toggles_the_layer_while_the_pointer_is_over_the_panel() {
    let (node, sift, image) = demo_panel_fixture();
    let feature_display = crate::state::FeatureDisplaySettings::default();

    let mut settings = IntrinsicsDisplaySettings::default();
    assert!(
        settings.enabled,
        "on by default: it is the reference frame the features sit in"
    );
    panel_frame_with_input(
        &node,
        &sift,
        &image,
        &feature_display,
        &mut settings,
        press_i(egui::pos2(450.0, 350.0)),
    );
    assert!(!settings.enabled, "`I` over the panel turns the layer off");

    // And it is a toggle, not a latch.
    panel_frame_with_input(
        &node,
        &sift,
        &image,
        &feature_display,
        &mut settings,
        press_i(egui::pos2(450.0, 350.0)),
    );
    assert!(settings.enabled);

    // With the pointer outside the panel the key belongs to whatever is under
    // it instead, exactly as the panel's existing `Z` behaves.
    panel_frame_with_input(
        &node,
        &sift,
        &image,
        &feature_display,
        &mut settings,
        press_i(egui::pos2(-40.0, -40.0)),
    );
    assert!(settings.enabled);
}

/// The human's `I` is one `Display` entry, in the words
/// `set_image_detail_display` produces for the same control — and a frame
/// nobody touched is not a change.
///
/// The differ and not the widget decides, which is what keeps the Action Log
/// free of the phantom rows [`crate::action_log::ActionLog::changed`]
/// documents: a value nobody touched is equal to the one before it.
#[test]
fn a_frame_that_toggles_the_layer_records_it_and_a_quiet_one_does_not() {
    let (node, sift, image) = demo_panel_fixture();
    let feature_display = crate::state::FeatureDisplaySettings::default();
    let mut settings = IntrinsicsDisplaySettings::default();
    let mut log = crate::action_log::ActionLog::new();

    // A frame with the pointer over the panel and no keystroke: the toolbar's
    // own re-derivations run, and none of them is a change.
    logged_panel_frame(
        &node,
        &sift,
        &image,
        &feature_display,
        &mut settings,
        &mut log,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            events: vec![egui::Event::PointerMoved(egui::pos2(450.0, 350.0))],
            ..Default::default()
        },
    );
    assert_eq!(
        log.entries().count(),
        0,
        "a frame nobody touched recorded {:?}",
        log.entries().collect::<Vec<_>>()
    );

    logged_panel_frame(
        &node,
        &sift,
        &image,
        &feature_display,
        &mut settings,
        &mut log,
        press_i(egui::pos2(450.0, 350.0)),
    );
    let entries: Vec<_> = log.entries().collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].text, "Intrinsics off");
    assert_eq!(entries[0].actor, crate::action_log::Actor::User);
    assert_eq!(entries[0].kind, crate::action_log::Kind::Display);
}

/// One `I` press over the panel, as the key test spells it.
fn press_i(at: egui::Pos2) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(900.0, 700.0),
        )),
        events: vec![
            egui::Event::PointerMoved(at),
            egui::Event::Key {
                key: egui::Key::I,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        ..Default::default()
    }
}

/// One panel frame with the dock's before/after snapshot around it — the same
/// pair `TabViewer::ui` takes, so what this records is what the viewer records.
fn logged_panel_frame(
    node: &crate::scene::SceneNode,
    sift: &crate::state::CachedSiftFeatures,
    image: &sfmtool_core::camera::remap::ImageU8,
    feature_display: &crate::state::FeatureDisplaySettings,
    intrinsics: &mut IntrinsicsDisplaySettings,
    log: &mut crate::action_log::ActionLog,
    input: egui::RawInput,
) {
    let before = crate::state::ImageDetailDisplay::snapshot(feature_display, intrinsics);
    panel_frame_with_input(node, sift, image, feature_display, intrinsics, input);
    let after = crate::state::ImageDetailDisplay::snapshot(feature_display, intrinsics);
    crate::state::record_image_detail_changes(log, &before, &after);
}

// ── The hover readout ───────────────────────────────────────────────────

#[test]
fn the_readout_turns_the_image_into_a_protractor() {
    let camera = simple_radial_camera();
    let limit = sfmtool_core::camera::report::trustworthy_max_theta_deg(&camera);
    let text = super::hover::readout(&camera, limit, [200.0, 120.0]).expect("on the image");
    let lines: Vec<&str> = text.lines().collect();

    assert_eq!(lines[0], "pixel  (200.0, 120.0)");
    assert!(lines[1].starts_with("ray    ("), "{:?}", lines[1]);
    assert!(
        lines[2].contains("off-axis ") && lines[2].contains("azimuth "),
        "{:?}",
        lines[2]
    );
    assert!(lines[3].starts_with("distortion  "), "{:?}", lines[3]);
    assert!(lines[3].ends_with(" px"), "{:?}", lines[3]);

    // (200, 120) is right of the principal point (135, 240) and, since `v`
    // grows downward, above it. The canonical frame has +X right and +Y up and
    // azimuth is measured from +X, so that is the first quadrant — the sign
    // convention the whole layer's labels rest on, checked once here.
    let azimuth: f64 = lines[2]
        .rsplit_once("azimuth ")
        .and_then(|(_, tail)| tail.trim_end_matches(super::DEGREE).parse().ok())
        .expect("an azimuth");
    assert!(
        (0.0..90.0).contains(&azimuth),
        "right of and above the principal point is the first quadrant, got {azimuth}"
    );
}

#[test]
fn the_readout_is_only_for_pixels_that_exist() {
    let camera = simple_radial_camera();
    assert!(super::hover::readout(&camera, None, [-1.0, 100.0]).is_none());
    assert!(super::hover::readout(&camera, None, [100.0, 999.0]).is_none());
    assert!(super::hover::readout(&camera, None, [0.0, 0.0]).is_some());
}

#[test]
fn the_readout_names_no_displacement_where_the_model_is_extrapolating() {
    let camera = kerry_park_camera();
    let limit = camera_limit(&camera);

    // The frame's corner is past the bound; its centre is not.
    let corner = super::hover::readout(&camera, Some(limit), [2.0, 2.0]).expect("on the image");
    let corner_line = corner.lines().last().expect("a distortion line");
    assert!(
        corner_line.starts_with("distortion  not modelled past 84."),
        "{corner_line:?}"
    );
    assert!(
        !corner_line.contains(" px"),
        "no number where there is no measurement: {corner_line:?}"
    );

    let middle = super::hover::readout(&camera, Some(limit), [250.0, 250.0]).expect("on the image");
    assert!(
        middle.lines().last().is_some_and(|l| l.ends_with(" px")),
        "{middle:?}"
    );
}

#[test]
fn a_model_with_no_distortion_gets_no_distortion_line() {
    let text =
        super::hover::readout(&pinhole_camera(), None, [400.0, 300.0]).expect("on the image");
    assert!(!text.contains("distortion"), "{text:?}");
    assert_eq!(text.lines().count(), 3);
}

#[test]
fn the_feature_tooltip_is_untouched_with_the_layer_off() {
    // The regression a composed tooltip most plausibly breaks: with no
    // intrinsics readout, the panel must draw exactly what it always has.
    let feature = "Point3D #42 | err: 0.412px | tracklen: 7";
    let tooltip = |top: Option<&str>, readout: Option<&str>| {
        painted(|ui| {
            let panel = ui.available_rect_before_wrap();
            let painter = ui.painter().clone();
            super::super::overlay::draw_tooltip(
                &painter,
                egui::pos2(120.0, 160.0),
                panel,
                top,
                readout,
            );
        })
    };

    assert_eq!(
        tooltip(Some(feature), None),
        vec![feature.to_owned()],
        "with the layer off the tooltip is the feature line and nothing else"
    );
    assert_eq!(
        tooltip(Some(feature), Some("off-axis 17.3°")),
        vec![feature.to_owned(), "off-axis 17.3°".to_owned()],
        "with it on, the readout goes below the feature line"
    );
    assert_eq!(
        tooltip(None, Some("off-axis 17.3°")),
        vec!["off-axis 17.3°".to_owned()],
        "and off a feature it is the readout alone"
    );
    assert!(tooltip(None, None).is_empty());
}

// ── The toolbar and its popup ───────────────────────────────────────────

/// Every string one headless frame of `run_ui` painted.
fn painted(run_ui: impl FnMut(&mut egui::Ui)) -> Vec<String> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(900.0, 600.0),
        )),
        ..Default::default()
    };
    crate::test_support::painted_texts(&ctx, input, run_ui)
}

#[test]
fn the_toolbar_offers_the_checkbox_and_the_gear_whatever_the_feature_mode() {
    let mut settings = IntrinsicsDisplaySettings::default();
    let texts = painted(|ui| {
        ui.horizontal(|ui| {
            super::show_intrinsics_controls(ui, &mut settings, None, None);
        });
    });
    assert!(texts.iter().any(|t| t == "Intrinsics"), "{texts:?}");
    assert!(texts.iter().any(|t| t == controls::GEAR), "{texts:?}");
}

#[test]
fn the_popup_offers_a_distortion_row_only_when_there_is_distortion() {
    let camera = simple_radial_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let mut settings = IntrinsicsDisplaySettings::default();
    let texts = painted(|ui| {
        controls::settings_popup(ui, &mut settings, Some(&camera), Some(&layer));
    });
    assert!(texts.iter().any(|t| t == "Axes"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Iso-angle rings"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Distortion field"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Grid density"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "No distortion"), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.starts_with("max displacement")),
        "{texts:?}"
    );

    // A pinhole is its own ideal map, so the control that would do nothing is
    // replaced by the statement of why.
    let pinhole = pinhole_camera();
    let layer = CameraLayer::compute(&pinhole, 16);
    let texts = painted(|ui| {
        controls::settings_popup(ui, &mut settings, Some(&pinhole), Some(&layer));
    });
    assert!(texts.iter().any(|t| t == "No distortion"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Distortion field"), "{texts:?}");
}

#[test]
fn the_popups_footer_says_which_domain_its_number_is_about() {
    let camera = kerry_park_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let mut settings = IntrinsicsDisplaySettings::default();
    let texts = painted(|ui| {
        controls::settings_popup(ui, &mut settings, Some(&camera), Some(&layer));
    });
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("inside 84.") && t.ends_with("nodes extrapolated")),
        "a bounded model should qualify its maximum: {texts:?}"
    );

    // An unbounded one has nothing to qualify.
    let camera = simple_radial_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let texts = painted(|ui| {
        controls::settings_popup(ui, &mut settings, Some(&camera), Some(&layer));
    });
    assert!(!texts.iter().any(|t| t.starts_with("inside ")), "{texts:?}");
}

#[test]
fn an_exaggeration_is_spelled_without_its_decimal_point() {
    assert_eq!(controls::scale_text(3.0), "3");
    assert_eq!(controls::scale_text(1.5), "1.5");
}
