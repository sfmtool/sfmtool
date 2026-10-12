// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The plot's arithmetic, tested without an egui frame.
//!
//! What the panel *says* is asserted a level up, in
//! `intrinsics_detail/tests.rs`, off the frame's galleys. What lives here is
//! the part a painted string cannot show: which rules the plot decided to
//! draw, where its axes decided to end, and whether the ladder its ticks walk
//! is one a person would have chosen.

use sfmtool_core::camera::{CameraIntrinsics, CameraModel};

use super::*;

/// `kerry_park`'s `OPENCV_FISHEYE`: the bounded case, on a square frame.
fn fisheye() -> CameraIntrinsics {
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

/// `seoul_bull_sculpture`'s `SIMPLE_RADIAL`: the ordinary case, on a portrait
/// frame whose two mid-edges are very different angles.
fn simple_radial() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 336.370993,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.026604,
        },
        width: 270,
        height: 480,
    }
}

fn plot_rect() -> Rect {
    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(400.0, 150.0))
}

fn labels(markers: &[Marker]) -> Vec<&str> {
    markers.iter().map(|m| m.label.as_str()).collect()
}

// ── The labelled rules ──────────────────────────────────────────────────

/// A square frame with a centred principal point reaches the same angle at
/// every mid-edge, so there is one edge rule, not two on the same line.
#[test]
fn a_square_frame_gets_one_edge_marker() {
    let camera = fisheye();
    let derived = Derived::compute(&camera);
    let x_max = derived.fov.unwrap().max_off_axis * X_MARGIN;
    let markers = markers(&camera, &derived, x_max);
    assert_eq!(labels(&markers), ["edge", "corner"]);
    // Ascending, which is what keeps the two label rows alternating usefully.
    assert!(markers[0].theta_deg < markers[1].theta_deg);
    assert!((markers[1].theta_deg - derived.fov.unwrap().max_off_axis).abs() < 1e-9);
}

/// A portrait frame does not, and the two rules are named for their axes.
#[test]
fn a_non_square_frame_marks_both_mid_edges() {
    let camera = simple_radial();
    let derived = Derived::compute(&camera);
    let x_max = derived.fov.unwrap().max_off_axis * X_MARGIN;
    let markers = markers(&camera, &derived, x_max);
    assert_eq!(labels(&markers), ["h edge", "v edge", "corner"]);
    // 270 × 480 with `f = 336`: the short axis reaches half the angle the long
    // one does.
    assert!(markers[0].theta_deg < markers[1].theta_deg);
}

/// The 90° rule is a fact about a perspective model's projective divide, and
/// it is drawn only when the frame gets near enough for that to matter — a 38°
/// lens would otherwise spend most of the axis reaching a ray it never sees.
#[test]
fn the_ninety_degree_asymptote_is_marked_only_when_the_frame_reaches_it() {
    let camera = simple_radial();
    let derived = Derived::compute(&camera);
    let corner = derived.fov.unwrap().max_off_axis;
    assert!(corner < 45.0);
    assert!(!labels(&markers(&camera, &derived, corner * X_MARGIN)).contains(&"90° asymptote"));
    // Same camera, an axis long enough to contain it.
    assert!(labels(&markers(&camera, &derived, 100.0)).contains(&"90° asymptote"));

    // Never for a model with no projective divide to diverge at, however wide
    // the axis: a fisheye is defined past 90° and says nothing special there.
    let fisheye = fisheye();
    let derived = Derived::compute(&fisheye);
    assert!(!labels(&markers(&fisheye, &derived, 180.0)).contains(&"90° asymptote"));
}

/// A spline model's domain end is marked, in the units of the axis: `θ_max`
/// directly for the fisheye spline, `atan(ρ_max)` for the pinhole one, whose
/// domain is an image-plane radius rather than an angle.
#[test]
fn a_spline_model_marks_its_domain_end() {
    let fisheye = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 129.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 1.6,
            bspline: vec![0.0, 0.01, 0.02, 0.0],
        },
        width: 480,
        height: 480,
    };
    let derived = Derived::compute(&fisheye);
    let drawn = markers(&fisheye, &derived, 180.0);
    let spline = drawn
        .iter()
        .find(|m| m.label == "spline domain")
        .expect("no spline domain rule");
    assert!((spline.theta_deg - 1.6_f64.to_degrees()).abs() < 1e-9);

    let pinhole = CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_rho_max: 0.9,
            bspline: vec![0.001, 0.004, 0.01, 0.02],
        },
        width: 640,
        height: 480,
    };
    let derived = Derived::compute(&pinhole);
    let drawn = markers(&pinhole, &derived, 90.0);
    let spline = drawn
        .iter()
        .find(|m| m.label == "spline domain")
        .expect("no spline domain rule");
    assert!((spline.theta_deg - 0.9_f64.atan().to_degrees()).abs() < 1e-9);
}

/// A rule the axis does not reach is not drawn at all, rather than clamped to
/// the border where it would read as a fact about the last angle plotted.
#[test]
fn rules_outside_the_axis_are_dropped() {
    let camera = fisheye();
    let derived = Derived::compute(&camera);
    assert!(markers(&camera, &derived, 30.0).is_empty());
}

// ── The axes ────────────────────────────────────────────────────────────

/// The radial plot's y range comes from the trustworthy samples alone. On
/// `kerry_park` that is the whole difference between a readable plot and a
/// flat line: the ideal map keeps climbing to 340 px at the frame's corner
/// while the trustworthy part of the curve tops out around 200.
#[test]
fn the_radial_scale_is_set_by_the_trustworthy_samples() {
    let camera = fisheye();
    let derived = Derived::compute(&camera);
    let limit = derived.trustworthy_max_theta.unwrap();
    let samples = &derived.profile.samples;
    let trusted = samples.iter().take_while(|s| s.theta_deg <= limit).count();
    assert!(trusted > 8 && trusted < samples.len());

    let bounded = radial_scale(&samples[..trusted], &plot_rect());
    let everything = radial_scale(samples, &plot_rect());
    assert!(bounded.lo == 0.0);
    assert!(
        bounded.hi < everything.hi * 0.8,
        "bounding the domain should have shrunk the y range, {} vs {}",
        bounded.hi,
        everything.hi
    );

    // Bottom of the plot is zero, top is the maximum: y grows upward on
    // screen, which is the flip every hand-painted plot gets wrong once.
    assert_eq!(bounded.at(0.0), plot_rect().bottom());
    assert!(bounded.at(bounded.hi) <= plot_rect().top() + 1e-3);
}

/// The residual's range always contains zero, so the line it is read against
/// is on the plot even for a lens that only ever magnifies.
#[test]
fn the_residual_scale_always_contains_zero() {
    for camera in [fisheye(), simple_radial()] {
        let derived = Derived::compute(&camera);
        let scale = residual_scale(&derived.profile.samples, &plot_rect());
        assert!(scale.lo <= 0.0 && scale.hi >= 0.0);
        let zero = scale.at(0.0);
        assert!(zero >= plot_rect().top() && zero <= plot_rect().bottom());
    }
}

/// A model that is exactly its own reference has no residual at all, and a
/// zero-width range would put every point on one border.
#[test]
fn a_zero_residual_still_gets_a_range() {
    let camera = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let derived = Derived::compute(&camera);
    let scale = residual_scale(&derived.profile.samples, &plot_rect());
    assert!(scale.hi - scale.lo > 1.0);
    assert!((scale.at(0.0) - plot_rect().center().y).abs() < 1e-3);
}

// ── Ticks ───────────────────────────────────────────────────────────────

#[test]
fn ticks_walk_a_one_two_five_ladder() {
    assert_eq!(
        ticks(0.0, 158.0, 8),
        vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    );
    assert_eq!(
        ticks(0.0, 40.0, 8),
        vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    );
    // A range that straddles zero starts at the first multiple of the step
    // inside it — which is where the residual's `-0.0` zero tick comes from.
    let straddling = ticks(-2.4, 6.0, 4);
    assert_eq!(straddling, vec![-0.0, 5.0]);
    assert!(straddling[0].is_sign_negative());
    assert_eq!(format_tick(straddling[0]), "0");
    let below = ticks(-12.0, 6.0, 4);
    assert_eq!(below.first().copied(), Some(-10.0));
    // Degenerate inputs give nothing rather than looping.
    assert!(ticks(1.0, 1.0, 4).is_empty());
    assert!(ticks(0.0, 10.0, 0).is_empty());
    assert!(ticks(0.0, f64::NAN, 4).is_empty());
}

/// The residual's zero tick arrives as `-0.0`, which `{:.0}` renders as `-0`.
#[test]
fn a_tick_at_zero_never_reads_minus_zero() {
    assert_eq!(format_tick(-0.0), "0");
    assert_eq!(format_tick(0.0), "0");
    assert_eq!(format_tick(200.0), "200");
    assert_eq!(format_tick(2.5), "2.5");
    assert_eq!(format_tick(-0.25), "-0.25");
}

// ── The comparison ──────────────────────────────────────────────────────

use super::super::derived::{ChangeProfile, ProposedReport};
use sfmtool_core::camera::refit_intrinsics::{refit_camera_intrinsics, RefitOptions, RefitTarget};

/// `camera` refitted to an eight-coefficient `SFMTOOL_FISHEYE`.
fn spline_refit(
    camera: &CameraIntrinsics,
) -> sfmtool_core::camera::refit_intrinsics::CameraIntrinsicsRefit {
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap();
    refit_camera_intrinsics(camera, &target, &RefitOptions::default()).unwrap()
}

/// A camera compared with itself changes nothing, at any angle or azimuth.
#[test]
fn a_camera_compared_with_itself_changes_nothing() {
    let camera = fisheye();
    let change = ChangeProfile::compute(&camera, &camera);
    assert!(change.samples.len() > 100);
    for s in &change.samples {
        assert_eq!(s.centre_px, 0.0);
        assert_eq!(s.band_px, (0.0, 0.0));
    }
    assert!(!change.band_visible(None));
}

/// An exact fit's change is zero inside the fit's largest angle: an
/// equidistant fisheye is a flat spline.
#[test]
fn an_exact_fits_change_is_zero_inside_the_fit() {
    let camera = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 130.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    };
    let refit = spline_refit(&camera);
    let change = ChangeProfile::compute(&camera, &refit.camera);
    let inside: Vec<_> = change
        .samples
        .iter()
        .filter(|s| s.theta_deg <= refit.theta_fit_deg)
        .collect();
    assert!(inside.len() > 50);
    for s in inside {
        assert!(
            s.centre_px.abs() < 1e-3,
            "{} at {}°",
            s.centre_px,
            s.theta_deg
        );
    }
}

/// On `kerry_park` the radial profile fits to a hundredth of a pixel, so the
/// centre line stays near zero inside the fit, and the dropped `fx ≠ fy` is
/// the band's width.
#[test]
fn a_dropped_aspect_is_the_bands_width() {
    let camera = fisheye();
    let refit = spline_refit(&camera);
    let change = ChangeProfile::compute(&camera, &refit.camera);
    let limit = Derived::compute(&camera).trustworthy_max_theta.unwrap();
    assert!(change.band_visible(Some(limit)));
    let near_fit = change
        .samples
        .iter()
        .rfind(|s| s.theta_deg <= refit.theta_fit_deg)
        .unwrap();
    assert!(near_fit.centre_px.abs() < 0.1, "{}", near_fit.centre_px);
    let width = near_fit.band_px.1 - near_fit.band_px.0;
    // `|fy − fx|` times the distorted angle, about 0.11 × 1.6 rad.
    assert!(width > 0.1 && width < 0.4, "{width}");
}

/// The comparison adds a rule at the fit's largest angle and one at the
/// proposed spline's domain end.
#[test]
fn the_comparison_marks_the_fit_and_the_proposed_domain() {
    let camera = fisheye();
    let refit = spline_refit(&camera);
    let report = ProposedReport::compute(&camera, &refit.camera);
    let compare = Comparison {
        proposed: &refit.camera,
        derived: &report.derived,
        change: &report.change,
        theta_fit_deg: refit.theta_fit_deg,
        monotone_deg: None,
    };
    let drawn = comparison_markers(&compare, 180.0);
    assert_eq!(labels(&drawn), ["fit", "proposed spline domain"]);
    assert!((drawn[0].theta_deg - refit.theta_fit_deg).abs() < 1e-9);
    assert!((drawn[1].theta_deg - refit.spline_domain_deg.unwrap()).abs() < 1e-6);
}

/// The rug bins its ticks a screen column at a time, keeps every
/// observation, and flags the ones past the bound.
#[test]
fn the_rug_keeps_every_observation_and_flags_those_past_the_bound() {
    let x = Scale {
        lo: 0.0,
        hi: 100.0,
        screen_lo: 0.0,
        screen_hi: 100.0,
    };
    let angles = [1.0, 1.2, 1.4, 50.0, 84.5, 90.0, 90.3];
    let columns = rug_columns(&angles, Some(84.0), &x);
    assert_eq!(columns.iter().map(|c| c.1).sum::<usize>(), angles.len());
    assert_eq!(columns[0], (1, 3, false));
    assert_eq!(
        columns.iter().filter(|c| c.2).map(|c| c.1).sum::<usize>(),
        3
    );
    assert_eq!(past_bound(&angles, Some(84.0)), 3);
    assert_eq!(past_bound(&angles, None), 0);
}

/// Two rule labels at the same angle go on two rows, and one near the right
/// edge is put left of its rule so it stays on the plot.
#[test]
fn rule_labels_take_the_first_row_with_room_and_stay_on_the_plot() {
    let mut placed = Vec::new();
    let size = egui::vec2(60.0, 10.0);
    let a = place_label(&mut placed, size, 100.0, 20.0, (50.0, 300.0)).unwrap();
    let b = place_label(&mut placed, size, 100.0, 20.0, (50.0, 300.0)).unwrap();
    assert_eq!(a.x, b.x);
    assert!(b.y > a.y);
    let c = place_label(&mut placed, size, 290.0, 20.0, (50.0, 300.0)).unwrap();
    assert!(c.x + size.x <= 300.0);
    assert!(c.x + size.x <= 290.0);
}

/// Where the monotonicity constraint bound, the change plot says so.
#[test]
fn the_change_plot_marks_where_the_monotone_constraint_bound() {
    let camera = fisheye();
    let refit = spline_refit(&camera);
    let report = ProposedReport::compute(&camera, &refit.camera);
    let derived = Derived::compute(&camera);
    for (monotone_deg, drawn) in [(Some([110.0, 112.0]), true), (None, false)] {
        let compare = Comparison {
            proposed: &refit.camera,
            derived: &report.derived,
            change: &report.change,
            theta_fit_deg: refit.theta_fit_deg,
            monotone_deg,
        };
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(500.0, 900.0))),
            ..Default::default()
        };
        let texts = crate::test_support::painted_texts(&ctx, input, |ui| {
            show_projection_plot(ui, &camera, &derived, Some(&compare), None);
        });
        assert_eq!(
            texts.iter().any(|t| t == "monotone bound"),
            drawn,
            "{texts:?}"
        );
        assert!(texts.iter().any(|t| t == "Change (px)"));
    }
}
