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
        controls::TIMES,
        controls::GEAR,
    ] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

#[test]
fn the_grid_keeps_its_cells_square() {
    let layer = CameraLayer::compute(&simple_radial_camera(), 16);
    // 270 × 480, so a 16-wide grid is 28 rows deep.
    assert_eq!(layer.grid, (16, 28));

    let layer = CameraLayer::compute(&kerry_park_camera(), 12);
    assert_eq!(layer.grid, (12, 12));
}

#[test]
fn a_model_that_is_its_own_ideal_map_measures_no_displacement() {
    let layer = CameraLayer::compute(&pinhole_camera(), 16);
    assert_eq!(layer.max_px, 0.0);
    assert_eq!(layer.extrapolated, (0, 0));
    assert_eq!(layer.limit_deg, None);
}

#[test]
fn the_circular_fisheyes_folded_corners_are_excluded_from_the_maximum() {
    let camera = kerry_park_camera();
    let layer = CameraLayer::compute(&camera, 16);
    let limit = layer.limit_deg.expect("OPENCV_FISHEYE is a bounded model");
    assert!(
        (84.0..85.0).contains(&limit),
        "kerry_park's camera 0 stops describing a lens at 84.1°, got {limit}"
    );

    let (outside, total) = layer.extrapolated;
    assert!(outside > 0, "the frame's corners are outside the bound");
    assert!(outside < total, "and its centre is not");

    // The whole point: the unfiltered maximum is 273 px of folded polynomial,
    // the filtered one is the 13 px the lens actually displaces anything.
    let unfiltered = sfmtool_core::camera::report::distortion_field(&camera, 16, 16)
        .iter()
        .map(|s| (s.pixel[0] - s.reference[0]).hypot(s.pixel[1] - s.reference[1]))
        .fold(0.0_f64, f64::max);
    assert!(
        layer.max_px < 20.0,
        "trustworthy maximum should be the lens's own, got {}",
        layer.max_px
    );
    assert!(
        unfiltered > 10.0 * layer.max_px,
        "the fold should dwarf the lens, got {unfiltered} against {}",
        layer.max_px
    );
}

#[test]
fn an_unbounded_model_excludes_nothing() {
    let layer = CameraLayer::compute(&simple_radial_camera(), 16);
    assert_eq!(layer.limit_deg, None);
    assert_eq!(layer.extrapolated.0, 0);
    assert_eq!(layer.extrapolated.1, 16 * 28);
    assert!(layer.max_px > 0.0);
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

#[test]
fn an_exaggeration_is_spelled_without_its_decimal_point() {
    assert_eq!(controls::scale_text(3.0), "3");
    assert_eq!(controls::scale_text(1.5), "1.5");
}
