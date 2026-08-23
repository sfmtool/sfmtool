// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use approx::assert_relative_eq;

use super::*;
use crate::camera::intrinsics::MODEL_COUNT;
use crate::camera::CameraModel;

// -----------------------------------------------------------------------
// Fixtures
//
// Two corpora over the same 640 x 480 image with a centred principal point:
// one with every distortion coefficient zeroed, one with live coefficients.
// The undistorted corpus is the one that has to be complete against the
// registry, since the family properties are asserted on it.
//
// Both take the equidistant focal length as a parameter, because the two
// things the fisheye fixtures are wanted for pull in opposite directions:
// F_FISH_WIDE puts the image corner past 90° off-axis, which is where the
// projection is interesting and where an end-to-end field-of-view measurement
// folds; F_FISH_NARROW keeps every sample inside the 80° at which
// `undistort_to_ray` starts blending the polynomial fisheye models toward the
// identity, which is where their round trip stops being exact.
// -----------------------------------------------------------------------

const W: u32 = 640;
const H: u32 = 480;
const CX: f64 = 320.0;
const CY: f64 = 240.0;
/// Perspective focal length: the image corner lands at 38.7° off-axis.
const F_PERSP: f64 = 500.0;
/// The image corner lands at 114.6° off-axis.
const F_FISH_WIDE: f64 = 200.0;
/// The furthest distortion-field node lands at 49.2° off-axis.
const F_FISH_NARROW: f64 = 400.0;

fn cam(model: CameraModel) -> CameraIntrinsics {
    CameraIntrinsics {
        model,
        width: W,
        height: H,
    }
}

/// One camera per registered model, every distortion coefficient zero.
///
/// `has_distortion()` is `false` for all of them, so each *is* its own family's
/// ideal map and every residual this module computes for them must vanish.
fn undistorted_cameras(f_fish: f64) -> Vec<CameraIntrinsics> {
    vec![
        cam(CameraModel::Pinhole {
            focal_length_x: F_PERSP,
            focal_length_y: F_PERSP + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
        }),
        cam(CameraModel::SimplePinhole {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
        }),
        cam(CameraModel::SimpleRadial {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
        }),
        cam(CameraModel::Radial {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
        }),
        cam(CameraModel::OpenCV {
            focal_length_x: F_PERSP,
            focal_length_y: F_PERSP + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            tangential_distortion_p1: 0.0,
            tangential_distortion_p2: 0.0,
        }),
        cam(CameraModel::FullOpenCV {
            focal_length_x: F_PERSP,
            focal_length_y: F_PERSP + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            tangential_distortion_p1: 0.0,
            tangential_distortion_p2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            radial_distortion_k5: 0.0,
            radial_distortion_k6: 0.0,
        }),
        cam(CameraModel::SfmtoolPinhole {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            bspline_rho_max: 0.9,
            bspline: Vec::new(),
        }),
        cam(CameraModel::OpenCVFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        }),
        cam(CameraModel::SimpleRadialFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
        }),
        cam(CameraModel::RadialFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
        }),
        cam(CameraModel::ThinPrismFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            tangential_distortion_p1: 0.0,
            tangential_distortion_p2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            thin_prism_sx1: 0.0,
            thin_prism_sy1: 0.0,
        }),
        cam(CameraModel::RadTanThinPrismFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k0: 0.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            radial_distortion_k5: 0.0,
            tangential_distortion_p0: 0.0,
            tangential_distortion_p1: 0.0,
            thin_prism_s0: 0.0,
            thin_prism_s1: 0.0,
            thin_prism_s2: 0.0,
            thin_prism_s3: 0.0,
        }),
        cam(CameraModel::EquidistantFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
        }),
        cam(CameraModel::SfmtoolFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            bspline_theta_max: 2.0,
            bspline: Vec::new(),
        }),
        cam(CameraModel::Equirectangular {
            focal_length_x: f64::from(W) / (2.0 * PI),
            focal_length_y: f64::from(H) / PI,
            principal_point_x: CX,
            principal_point_y: CY,
        }),
    ]
}

/// The models that can carry distortion, carrying some.
///
/// Deliberately not complete against the registry — four models have no
/// coefficients to give — so it is used only where a live lens is the point.
fn distorted_cameras(f_fish: f64) -> Vec<CameraIntrinsics> {
    vec![
        cam(CameraModel::SimpleRadial {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: -0.15,
        }),
        cam(CameraModel::Radial {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
        }),
        cam(CameraModel::OpenCV {
            focal_length_x: F_PERSP,
            focal_length_y: F_PERSP + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
        }),
        cam(CameraModel::FullOpenCV {
            focal_length_x: F_PERSP,
            focal_length_y: F_PERSP + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
            radial_distortion_k5: 0.002,
            radial_distortion_k6: -0.001,
        }),
        cam(CameraModel::SfmtoolPinhole {
            focal_length: F_PERSP,
            principal_point_x: CX,
            principal_point_y: CY,
            bspline_rho_max: 0.9,
            bspline: vec![
                0.0008, 0.0031, 0.0075, 0.0142, 0.0236, 0.0361, 0.052, 0.0718,
            ],
        }),
        cam(CameraModel::OpenCVFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
        }),
        cam(CameraModel::SimpleRadialFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.05,
        }),
        cam(CameraModel::RadialFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
        }),
        cam(CameraModel::ThinPrismFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
            thin_prism_sx1: 0.002,
            thin_prism_sy1: -0.001,
        }),
        cam(CameraModel::RadTanThinPrismFisheye {
            focal_length_x: f_fish,
            focal_length_y: f_fish + 2.0,
            principal_point_x: CX,
            principal_point_y: CY,
            radial_distortion_k0: 0.05,
            radial_distortion_k1: -0.02,
            radial_distortion_k2: 0.01,
            radial_distortion_k3: -0.005,
            radial_distortion_k4: 0.002,
            radial_distortion_k5: -0.001,
            tangential_distortion_p0: 0.001,
            tangential_distortion_p1: -0.002,
            thin_prism_s0: 0.002,
            thin_prism_s1: -0.001,
            thin_prism_s2: 0.0015,
            thin_prism_s3: -0.0005,
        }),
        cam(CameraModel::SfmtoolFisheye {
            focal_length: f_fish,
            principal_point_x: CX,
            principal_point_y: CY,
            bspline_theta_max: 2.0,
            bspline: vec![-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09],
        }),
    ]
}

/// A camera the distortion field's grid can be reproduced against, so a test
/// can say *which* pixel a sample came from. Mirrors [`distortion_field`]'s
/// documented cell-centre layout, row-major.
fn grid_nodes(cols: usize, rows: usize) -> Vec<(f64, f64)> {
    (0..rows)
        .flat_map(|j| (0..cols).map(move |i| (i, j)))
        .map(|(i, j)| {
            (
                (i as f64 + 0.5) * f64::from(W) / cols as f64,
                (j as f64 + 0.5) * f64::from(H) / rows as f64,
            )
        })
        .collect()
}

// -----------------------------------------------------------------------
// The corpus is complete
// -----------------------------------------------------------------------

/// A newly registered `CameraModel` variant cannot be left untested here.
///
/// The same guard `all_cameras_covers_every_registered_model` puts on the
/// registry's own corpus: the compiler forces a new variant into the registry,
/// and this closes the other half.
#[test]
fn undistorted_cameras_covers_every_registered_model() {
    let cameras = undistorted_cameras(F_FISH_WIDE);
    assert_eq!(
        cameras.len(),
        MODEL_COUNT,
        "undistorted_cameras() has {} entries but the registry holds {MODEL_COUNT} models — \
         a new camera model needs a fixture here",
        cameras.len()
    );

    let mut names: Vec<&str> = cameras.iter().map(|c| c.model_name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        MODEL_COUNT,
        "undistorted_cameras() exercises the same model twice and misses another"
    );

    for cam in &cameras {
        assert!(
            !cam.has_distortion(),
            "{} carries live distortion in the undistorted corpus",
            cam.model_name()
        );
    }
}

// -----------------------------------------------------------------------
// field_of_view
// -----------------------------------------------------------------------

#[test]
fn field_of_view_on_a_pinhole_matches_the_closed_form() {
    let cam = cam(CameraModel::Pinhole {
        focal_length_x: F_PERSP,
        focal_length_y: F_PERSP + 2.0,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let fov = field_of_view(&cam).unwrap();

    let w = f64::from(W);
    let h = f64::from(H);
    assert_relative_eq!(
        fov.horizontal,
        2.0 * (w / (2.0 * F_PERSP)).atan().to_degrees(),
        epsilon = 1e-9
    );
    assert_relative_eq!(
        fov.vertical,
        2.0 * (h / (2.0 * (F_PERSP + 2.0))).atan().to_degrees(),
        epsilon = 1e-9
    );
    // The corner is the same half-angle doubled, and the diagonal is the two
    // opposite corners.
    assert_relative_eq!(fov.diagonal, 2.0 * fov.max_off_axis, epsilon = 1e-9);
}

#[test]
fn field_of_view_on_an_equidistant_fisheye_is_180_degrees_at_f_equals_w_over_pi() {
    let cam = cam(CameraModel::EquidistantFisheye {
        focal_length: f64::from(W) / PI,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let fov = field_of_view(&cam).unwrap();
    assert_relative_eq!(fov.horizontal, 180.0, epsilon = 1e-9);
}

/// The spans are swept through the image centre, so they keep climbing past
/// 180° instead of folding back — see [`FieldOfView`].
#[test]
fn field_of_view_does_not_fold_past_180_degrees() {
    // f = 200 puts the image edge at 320/200 = 1.6 rad = 91.67° off-axis, so
    // edge to edge is 183.35°. The angle *between* the two edge rays is the
    // 176.65° complement, which is the wrong answer and the one a naive
    // measurement gives.
    let cam = cam(CameraModel::EquidistantFisheye {
        focal_length: F_FISH_WIDE,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let fov = field_of_view(&cam).unwrap();
    assert_relative_eq!(
        fov.horizontal,
        2.0 * (CX / F_FISH_WIDE).to_degrees(),
        epsilon = 1e-9
    );
    assert!(fov.horizontal > 180.0, "{} folded back", fov.horizontal);
    // Likewise the diagonal: the corner is at 400/200 = 2 rad = 114.59°.
    assert_relative_eq!(fov.diagonal, 2.0 * 2.0_f64.to_degrees(), epsilon = 1e-9);
    assert_relative_eq!(fov.max_off_axis, 2.0_f64.to_degrees(), epsilon = 1e-9);
}

/// A full panorama's two vertical edges are the *same* ray, so the angle
/// between them is zero and only a swept measurement reports the 360° the
/// image actually covers.
#[test]
fn field_of_view_on_a_full_panorama_is_360_by_180() {
    let cam = cam(CameraModel::Equirectangular {
        focal_length_x: f64::from(W) / (2.0 * PI),
        focal_length_y: f64::from(H) / PI,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let fov = field_of_view(&cam).unwrap();
    assert_relative_eq!(fov.horizontal, 360.0, epsilon = 1e-9);
    assert_relative_eq!(fov.vertical, 180.0, epsilon = 1e-9);
}

#[test]
fn field_of_view_is_none_without_an_image() {
    let mut cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    cam.width = 0;
    assert!(field_of_view(&cam).is_none());
    cam.width = W;
    cam.height = 0;
    assert!(field_of_view(&cam).is_none());
}

// -----------------------------------------------------------------------
// The camera frame: −Z forward, +Y up, +X right
//
// Every sign in this module rides on it, and getting it backwards produces
// plausible numbers rather than an obvious failure.
// -----------------------------------------------------------------------

#[test]
fn an_upward_ray_projects_above_the_principal_point() {
    let eps = 10.0_f64.to_radians();
    let up = [0.0, eps.sin(), -eps.cos()];
    let right = [eps.sin(), 0.0, -eps.cos()];

    for cam in undistorted_cameras(F_FISH_WIDE) {
        let name = cam.model_name();
        let (cx, cy) = cam.principal_point();

        for (label, ray) in [("model", true), ("reference", false)] {
            let project = |r| {
                if ray {
                    cam.ray_to_pixel(r)
                } else {
                    reference_project(&cam, r)
                }
            };
            let (_, v_up) = project(up).unwrap_or_else(|| panic!("{name} {label}: up"));
            assert!(
                v_up < cy,
                "{name} {label}: an upward ray landed at v = {v_up}, at or below cy = {cy}"
            );
            let (u_right, _) = project(right).unwrap_or_else(|| panic!("{name} {label}: right"));
            assert!(
                u_right > cx,
                "{name} {label}: a rightward ray landed at u = {u_right}, at or left of cx = {cx}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// The ideal map: zero residual wherever the model has no distortion
// -----------------------------------------------------------------------

#[test]
fn radial_profile_reference_equals_the_model_without_distortion() {
    for cam in undistorted_cameras(F_FISH_WIDE) {
        let name = cam.model_name();
        for azimuth in [0.0, 37.0, 90.0, 180.0, 271.0] {
            let profile = radial_profile(&cam, azimuth, 32);
            assert!(!profile.is_empty(), "{name}: empty profile at {azimuth}°");
            for sample in &profile {
                assert!(
                    (sample.radius_px - sample.reference_px).abs() <= 1e-12,
                    "{name} at θ = {}°, φ = {azimuth}°: r = {}, r_ref = {}",
                    sample.theta_deg,
                    sample.radius_px,
                    sample.reference_px
                );
            }
        }
    }
}

#[test]
fn distortion_field_is_identically_zero_without_distortion() {
    for cam in undistorted_cameras(F_FISH_WIDE) {
        let name = cam.model_name();
        let field = distortion_field(&cam, 8, 6);
        assert_eq!(field.len(), 48, "{name}: dropped a grid node");
        for sample in field {
            // The last bits of a pixel coordinate, not merely small: what is
            // left is the two paths' multiply order, not a lens.
            let displacement = (sample.pixel[0] - sample.reference[0])
                .hypot(sample.pixel[1] - sample.reference[1]);
            assert!(
                displacement <= 1e-12,
                "{name}: a distortion-free model displaced a pixel by {displacement}, \
                 {:?} against {:?}",
                sample.pixel,
                sample.reference
            );
        }
    }
}

#[test]
fn distortion_field_is_nonzero_and_radially_symmetric_for_simple_radial() {
    let cam = cam(CameraModel::SimpleRadial {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
        radial_distortion_k1: -0.15,
    });
    // An odd grid so that no node lands on the principal point, where the
    // displacement is zero by construction and says nothing.
    let cols = 7;
    let rows = 7;
    let field = distortion_field(&cam, cols, rows);
    assert_eq!(field.len(), cols * rows);

    let worst = field
        .iter()
        .map(|s| (s.pixel[0] - s.reference[0]).hypot(s.pixel[1] - s.reference[1]))
        .fold(0.0_f64, f64::max);
    assert!(worst > 1.0, "k1 = −0.15 displaced at most {worst} px");

    // Radially symmetric: the displacement depends only on the distance from
    // the principal point, and it is purely radial (parallel to that offset).
    for sample in &field {
        let radius = (sample.reference[0] - CX).hypot(sample.reference[1] - CY);
        let displacement =
            (sample.pixel[0] - sample.reference[0]).hypot(sample.pixel[1] - sample.reference[1]);

        // Against the profile along the azimuth this node happens to sit on:
        // one function of θ, sampled two different ways.
        let expected = radius * ((1.0 - 0.15 * (radius / F_PERSP).powi(2)) - 1.0).abs();
        assert_relative_eq!(displacement, expected, epsilon = 1e-9);

        if radius > 1.0 {
            // Purely radial: the cross product of the offset and the
            // displacement vanishes.
            let offset = [sample.reference[0] - CX, sample.reference[1] - CY];
            let delta = [
                sample.pixel[0] - sample.reference[0],
                sample.pixel[1] - sample.reference[1],
            ];
            let cross = offset[0] * delta[1] - offset[1] * delta[0];
            assert_relative_eq!(cross / radius, 0.0, epsilon = 1e-9);
        }
    }

    // Same radius, any azimuth, same displacement.
    let profiles: Vec<f64> = [0.0, 45.0, 123.0, 270.0]
        .into_iter()
        .map(|azimuth| radial_profile(&cam, azimuth, 16)[15].radius_px)
        .collect();
    for r in &profiles {
        assert_relative_eq!(*r, profiles[0], epsilon = 1e-9);
    }
}

// -----------------------------------------------------------------------
// The round trip that makes the arrows mean anything
// -----------------------------------------------------------------------

/// `ray_to_pixel(pixel_to_ray(u)) ≈ u` at every node the distortion field
/// uses, which is what makes a sample's `pixel` the grid node it came from.
///
/// On the narrow fisheye corpus: past 80° off-axis `undistort_to_ray` blends
/// the polynomial fisheye models toward the identity ray on purpose
/// (`blend_fisheye_ray`), so out there it is not their inverse and no round
/// trip can hold. Inside it, every registered model round trips — no model is
/// exempt.
#[test]
fn distortion_field_nodes_round_trip() {
    let nodes = grid_nodes(8, 6);
    let corpus = undistorted_cameras(F_FISH_NARROW)
        .into_iter()
        .chain(distorted_cameras(F_FISH_NARROW));

    for cam in corpus {
        let name = cam.model_name();
        let field = distortion_field(&cam, 8, 6);
        assert_eq!(field.len(), nodes.len(), "{name}: dropped a grid node");
        for (sample, &(u, v)) in field.iter().zip(&nodes) {
            let error = (sample.pixel[0] - u).hypot(sample.pixel[1] - v);
            assert!(
                error <= 1e-6,
                "{name}: ({u}, {v}) came back as ({}, {}), {error} px away",
                sample.pixel[0],
                sample.pixel[1]
            );
        }
    }
}

// -----------------------------------------------------------------------
// radial_profile
// -----------------------------------------------------------------------

#[test]
fn radial_profile_spans_zero_to_the_corner_angle() {
    let cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let fov = field_of_view(&cam).unwrap();
    let profile = radial_profile(&cam, 0.0, 33);

    assert_eq!(profile.len(), 33);
    assert_relative_eq!(profile[0].theta_deg, 0.0);
    assert_relative_eq!(profile[0].radius_px, 0.0, epsilon = 1e-12);
    assert_relative_eq!(profile[32].theta_deg, fov.max_off_axis, epsilon = 1e-12);

    // A pinhole's radial map is f·tan θ, on both curves.
    for sample in &profile {
        let expected = F_PERSP * sample.theta_deg.to_radians().tan();
        assert_relative_eq!(sample.radius_px, expected, epsilon = 1e-9);
        assert_relative_eq!(sample.reference_px, expected, epsilon = 1e-9);
    }

    assert!(radial_profile(&cam, 0.0, 0).is_empty());
    assert_eq!(radial_profile(&cam, 0.0, 1).len(), 1);
}

/// The fisheye ideal is `f·θ`, and a fisheye's actual map departs from it.
#[test]
fn radial_profile_measures_a_fisheye_against_the_equidistant_ideal() {
    let cam = cam(CameraModel::SimpleRadialFisheye {
        focal_length: F_FISH_NARROW,
        principal_point_x: CX,
        principal_point_y: CY,
        radial_distortion_k1: 0.05,
    });
    let profile = radial_profile(&cam, 0.0, 24);
    assert!(!profile.is_empty());

    for sample in &profile {
        let theta = sample.theta_deg.to_radians();
        assert_relative_eq!(sample.reference_px, F_FISH_NARROW * theta, epsilon = 1e-9);
        // θ_d = θ·(1 + k1·θ²).
        assert_relative_eq!(
            sample.radius_px,
            F_FISH_NARROW * theta * (1.0 + 0.05 * theta * theta),
            epsilon = 1e-9
        );
    }
    assert!(profile.last().unwrap().radius_px > profile.last().unwrap().reference_px);
}

// -----------------------------------------------------------------------
// equiv_focal_length_35mm
// -----------------------------------------------------------------------

#[test]
fn equiv_focal_length_35mm_is_none_for_every_pixels_per_radian_model() {
    for cam in undistorted_cameras(F_FISH_WIDE) {
        let name = cam.model_name();
        let equivalent = equiv_focal_length_35mm(&cam);
        if cam.model.is_fisheye() || cam.model.is_equirectangular() {
            assert!(
                equivalent.is_none(),
                "{name} has a pixels-per-radian focal length but reported {equivalent:?} mm"
            );
        } else {
            assert!(equivalent.is_some(), "{name} reported no 35 mm equivalent");
        }
    }
}

#[test]
fn equiv_focal_length_35mm_matches_the_closed_form() {
    let cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let diagonal = f64::from(W).hypot(f64::from(H));
    assert_relative_eq!(
        equiv_focal_length_35mm(&cam).unwrap(),
        F_PERSP * FRAME_35MM_DIAGONAL_MM / diagonal,
        epsilon = 1e-12
    );

    // Independent of the pixel pitch: the same lens over twice the pixel
    // count, at twice the focal length in pixels, is the same equivalent.
    let doubled = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 2.0 * F_PERSP,
            principal_point_x: 2.0 * CX,
            principal_point_y: 2.0 * CY,
        },
        width: 2 * W,
        height: 2 * H,
    };
    assert_relative_eq!(
        equiv_focal_length_35mm(&doubled).unwrap(),
        equiv_focal_length_35mm(&cam).unwrap(),
        epsilon = 1e-12
    );
}

#[test]
fn equiv_focal_length_35mm_is_none_without_an_image() {
    let mut cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    cam.width = 0;
    cam.height = 0;
    assert!(equiv_focal_length_35mm(&cam).is_none());
}

// -----------------------------------------------------------------------
// Degenerate inputs
// -----------------------------------------------------------------------

#[test]
fn distortion_field_is_empty_for_a_degenerate_grid() {
    let cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    assert!(distortion_field(&cam, 0, 6).is_empty());
    assert!(distortion_field(&cam, 8, 0).is_empty());

    let mut no_image = cam.clone();
    no_image.width = 0;
    assert!(distortion_field(&no_image, 8, 6).is_empty());
    assert!(radial_profile(&no_image, 0.0, 8).is_empty());
}
