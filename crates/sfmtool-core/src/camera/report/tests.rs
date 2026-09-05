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
// folds; F_FISH_NARROW keeps every sample inside `FISHEYE_BLEND_START_RAD`,
// the distorted radius at which `undistort_to_ray` starts blending the
// polynomial fisheye models toward the identity ray, which is where their
// round trip stops being exact. That radius is 90° of *distorted* radius, so
// the incidence angle it lands at is per-camera — see
// `trustworthy_max_theta_deg`.
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
/// On the narrow fisheye corpus: past `FISHEYE_BLEND_START_RAD` of distorted
/// radius `undistort_to_ray` blends the polynomial fisheye models toward the
/// identity ray on purpose (`blend_fisheye_ray`), so out there it is not their
/// inverse and no round trip can hold. That is the same boundary
/// [`trustworthy_max_theta_deg`] reports, restated in incidence angle. Inside
/// it, every registered model round trips — no model is exempt.
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
fn displacement_at_is_the_field_sampled_at_one_pixel() {
    // The field is defined in terms of it, so a consumer sampling a pixel of
    // its own — an overlay's hover readout, which wants the value under the
    // cursor rather than at the nearest node — is reading the same arithmetic
    // rather than a second spelling of the ideal map.
    for cam in distorted_cameras(F_FISH_NARROW) {
        let (cols, rows) = (8, 6);
        let field = distortion_field(&cam, cols, rows);
        let nodes = grid_nodes(cols, rows);
        assert_eq!(field.len(), nodes.len(), "{}", cam.model_name());
        for (sample, (u, v)) in field.iter().zip(nodes) {
            let at = displacement_at(&cam, u, v).expect("the field kept this node");
            assert_eq!(at.pixel, sample.pixel, "{}", cam.model_name());
            assert_eq!(at.reference, sample.reference, "{}", cam.model_name());
            assert_eq!(at.theta_deg, sample.theta_deg, "{}", cam.model_name());
        }
    }
}

#[test]
fn displacement_at_is_zero_for_a_model_that_is_its_own_ideal_map() {
    for cam in undistorted_cameras(F_FISH_NARROW) {
        let sample = displacement_at(&cam, 0.4 * f64::from(W), 0.3 * f64::from(H))
            .expect("a pixel well inside the frame");
        let displacement =
            (sample.pixel[0] - sample.reference[0]).hypot(sample.pixel[1] - sample.reference[1]);
        assert!(
            displacement < 1e-12,
            "{} displaced by {displacement}",
            cam.model_name()
        );
    }
}

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

// -----------------------------------------------------------------------
// trustworthy_max_theta_deg
// -----------------------------------------------------------------------

/// The models whose forward map this module will not vouch for all the way out
/// — the three call sites of the wide-angle blend, which is four models
/// because `OPENCV_FISHEYE` and `RADIAL_FISHEYE` share one.
///
/// Spelled by name rather than by predicate so that the classification is a
/// list somebody wrote down: `trustworthy_max_theta_deg`'s own match is
/// exhaustive and will not build without a decision, and this is the other
/// half — the decision made twice, independently, and compared.
const BOUNDED_MODELS: [&str; 4] = [
    "OPENCV_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
    "RAD_TAN_THIN_PRISM_FISHEYE",
];

/// A newly registered model cannot default to "trustworthy everywhere"
/// unnoticed: every registry variant appears in the undistorted corpus, and
/// every model that can carry distortion appears in the distorted one, so
/// between them each of the `MODEL_COUNT` models is classified here.
#[test]
fn trustworthy_domain_is_decided_for_every_registered_model() {
    let undistorted = undistorted_cameras(F_FISH_WIDE);
    assert_eq!(undistorted.len(), MODEL_COUNT);

    // Zero coefficients: the model *is* its family's ideal map, so there is no
    // polynomial to fold and nothing for the blend to blend — including for
    // the four models a live lens would bound.
    for cam in &undistorted {
        assert_eq!(
            trustworthy_max_theta_deg(cam),
            None,
            "{} carries no distortion but reports a trustworthy limit",
            cam.model_name()
        );
    }

    // Live coefficients: exactly the four polynomial fisheye models are
    // bounded, and the bound is a real angle rather than a placeholder.
    let distorted = distorted_cameras(F_FISH_WIDE);
    let mut seen_bounded: Vec<&str> = Vec::new();
    for cam in &distorted {
        let name = cam.model_name();
        let limit = trustworthy_max_theta_deg(cam);
        if BOUNDED_MODELS.contains(&name) {
            let limit = limit.unwrap_or_else(|| panic!("{name} should report a limit"));
            assert!(
                limit > 0.0 && limit < TRUST_SCAN_MAX_DEG,
                "{name} reported an unusable limit of {limit}°"
            );
            seen_bounded.push(name);
        } else {
            assert_eq!(limit, None, "{name} should be trustworthy at every angle");
        }
    }
    seen_bounded.sort_unstable();
    let mut expected = BOUNDED_MODELS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        seen_bounded, expected,
        "the distorted corpus no longer exercises every bounded model"
    );
}

/// The bound really is where the model's own inverse gives up: at the reported
/// angle the distorted radius is `FISHEYE_BLEND_START_RAD`, so the round trip
/// is exact just inside it and drifts well outside.
#[test]
fn the_bound_is_where_the_blend_starts() {
    // A mild, monotone lens: `θ_d = θ·(1 + 0.02·θ² + 0.005·θ⁴)` climbs past
    // 90° of distorted radius without ever turning over, so the blend is what
    // bounds it. (The corpus's `OPENCV_FISHEYE` folds first, which is the
    // other test below.)
    let cam = cam(CameraModel::OpenCVFisheye {
        focal_length_x: 200.0,
        focal_length_y: 200.0,
        principal_point_x: CX,
        principal_point_y: CY,
        radial_distortion_k1: 0.02,
        radial_distortion_k2: 0.005,
        radial_distortion_k3: 0.0,
        radial_distortion_k4: 0.0,
    });
    let limit = trustworthy_max_theta_deg(&cam).unwrap();

    // The distorted radius at the bound is the blend's start radius.
    let radius = |theta_deg: f64| {
        let (s, c) = theta_deg.to_radians().sin_cos();
        let (u, v) = cam.ray_to_pixel([s, 0.0, -c]).unwrap();
        (u - CX).hypot(v - CY) / 200.0
    };
    assert_relative_eq!(radius(limit), FISHEYE_BLEND_START_RAD, epsilon = 1e-9);

    // And it is a real statement about the round trip: exact inside, not
    // outside. (`pixel_to_ray` is the map that blends, so this is the
    // pixel → ray → pixel direction.)
    let round_trip = |theta_deg: f64| {
        let (s, c) = theta_deg.to_radians().sin_cos();
        let (u, v) = cam.ray_to_pixel([s, 0.0, -c]).unwrap();
        let back = cam.ray_to_pixel(cam.pixel_to_ray(u, v)).unwrap();
        (back.0 - u).hypot(back.1 - v)
    };
    assert!(round_trip(limit - 1.0) < 1e-6);
    assert!(round_trip(limit + 12.0) > 1.0);
}

/// A polynomial that peaks before it reaches the blend radius is bounded at
/// its peak instead: past a fold there is no inverse to have, so the blend
/// threshold is not the only way out.
#[test]
fn a_fold_before_the_blend_radius_bounds_the_domain() {
    // k1 < 0 shrinks the rim, so θ_d = θ·(1 + k1·θ²) turns over at
    // θ = 1/√(3·|k1|) rad, well before θ_d could reach 90°.
    let k1 = -0.2_f64;
    let cam = cam(CameraModel::RadialFisheye {
        focal_length: F_FISH_WIDE,
        principal_point_x: CX,
        principal_point_y: CY,
        radial_distortion_k1: k1,
        radial_distortion_k2: 0.0,
    });
    let peak_deg = (1.0 / (3.0 * k1.abs()).sqrt()).to_degrees();
    let limit = trustworthy_max_theta_deg(&cam).unwrap();
    assert_relative_eq!(limit, peak_deg, epsilon = 1e-6);
    assert!(limit < 90.0);
}

/// `kerry_park`'s real `OPENCV_FISHEYE`: a circular fisheye in a square frame,
/// which is the camera this whole function exists for.
///
/// The image corners are 150° off-axis — outside the lens circle — and the
/// displacement field's maximum over the full rectangle is a statement about a
/// folded polynomial, not about a lens. Filtered to the trustworthy domain it
/// becomes a plausible lens number, and the two differ by a factor of twenty.
#[test]
fn the_kerry_park_fisheye_is_bounded_well_inside_its_corners() {
    let cam = CameraIntrinsics {
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
    };
    let limit = trustworthy_max_theta_deg(&cam).unwrap();
    let fov = field_of_view(&cam).unwrap();
    assert!(
        (80.0..90.0).contains(&limit),
        "expected the bound in the low eighties, got {limit}°"
    );
    assert!(fov.max_off_axis > 145.0);

    let field = distortion_field(&cam, 16, 16);
    let displacement = |sample: &DistortionSample| {
        (sample.pixel[0] - sample.reference[0]).hypot(sample.pixel[1] - sample.reference[1])
    };
    let over_the_rectangle = field.iter().map(displacement).fold(0.0_f64, f64::max);
    let inside_the_bound = field
        .iter()
        .filter(|s| s.theta_deg <= limit)
        .map(displacement)
        .fold(0.0_f64, f64::max);

    assert!(over_the_rectangle > 200.0, "{over_the_rectangle}");
    assert!(inside_the_bound < 20.0, "{inside_the_bound}");
    // And the filter really drops something: a third of the frame's grid nodes
    // look at angles the polynomial was never fitted to.
    assert!(field.iter().any(|s| s.theta_deg > limit));
    assert!(field.iter().any(|s| s.theta_deg <= limit));
}

// -----------------------------------------------------------------------
// distortion_extent
// -----------------------------------------------------------------------

/// `kerry_park`'s real `OPENCV_FISHEYE`, the camera the trustworthy bound
/// exists for.
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
/// at every angle its projective divide accepts, on a portrait frame.
fn seoul_bull_camera() -> CameraIntrinsics {
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

/// The rows follow the image aspect, so the sampled cells are square in pixels
/// rather than the grid being square in nodes.
#[test]
fn the_extents_grid_keeps_its_cells_square() {
    // 270 × 480, so a 16-wide grid is 28 rows deep.
    assert_eq!(distortion_extent(&seoul_bull_camera(), 16).grid, (16, 28));
    assert_eq!(distortion_extent(&kerry_park_camera(), 12).grid, (12, 12));
    // A camera with no width has no aspect to preserve.
    let no_image = CameraIntrinsics {
        width: 0,
        ..seoul_bull_camera()
    };
    assert_eq!(distortion_extent(&no_image, 9).grid, (9, 9));
    // And `cols` is clamped rather than producing an empty grid.
    assert_eq!(distortion_extent(&seoul_bull_camera(), 0).grid.0, 1);
}

/// A model that **is** its own ideal map is skipped rather than sampled to a
/// field of zeros: there is nothing to measure and nothing to draw.
#[test]
fn a_model_that_is_its_own_ideal_map_measures_no_displacement() {
    let cam = cam(CameraModel::Pinhole {
        focal_length_x: F_PERSP,
        focal_length_y: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
    });
    let extent = distortion_extent(&cam, 16);
    assert!(extent.field.is_empty());
    assert_eq!(extent.total(), 0);
    assert_eq!(extent.max_px, 0.0);
    assert_eq!(extent.excluded, 0);
    assert_eq!(extent.limit_deg, None);
}

/// The whole reason the summary is not just `field.iter().max()`: on a
/// circular fisheye the unfiltered maximum is 273 px of folded polynomial and
/// the filtered one is the 13 px the lens actually displaces anything.
#[test]
fn the_circular_fisheyes_folded_corners_are_excluded_from_the_maximum() {
    let cam = kerry_park_camera();
    let extent = distortion_extent(&cam, 16);
    let limit = extent.limit_deg.expect("OPENCV_FISHEYE is a bounded model");
    assert!(
        (84.0..85.0).contains(&limit),
        "kerry_park's camera 0 stops describing a lens at 84.1°, got {limit}"
    );

    assert!(
        extent.excluded > 0,
        "the frame's corners are outside the bound"
    );
    assert!(extent.excluded < extent.total(), "and its centre is not");

    let displacement = |sample: &DistortionSample| {
        (sample.pixel[0] - sample.reference[0]).hypot(sample.pixel[1] - sample.reference[1])
    };
    let unfiltered = extent
        .field
        .iter()
        .map(displacement)
        .fold(0.0_f64, f64::max);
    assert!(
        extent.max_px < 20.0,
        "trustworthy maximum should be the lens's own, got {}",
        extent.max_px
    );
    assert!(
        unfiltered > 10.0 * extent.max_px,
        "the fold should dwarf the lens, got {unfiltered} against {}",
        extent.max_px
    );
}

/// An unbounded model is measured over the whole rectangle, and says so by
/// excluding nothing.
#[test]
fn an_unbounded_model_excludes_nothing() {
    let extent = distortion_extent(&seoul_bull_camera(), 16);
    assert_eq!(extent.limit_deg, None);
    assert_eq!(extent.excluded, 0);
    assert_eq!(extent.total(), 16 * 28);
    assert!(extent.max_px > 0.0);
}

/// [`DistortionExtent::trusted`] is the predicate the counts were taken with,
/// so a consumer splitting the field for display splits it the same way.
#[test]
fn the_trust_predicate_partitions_the_field_the_counts_were_taken_over() {
    for cam in [kerry_park_camera(), seoul_bull_camera()] {
        let extent = distortion_extent(&cam, 16);
        let trusted = extent.field.iter().filter(|s| extent.trusted(s)).count();
        assert_eq!(trusted + extent.excluded, extent.total());
    }
}

// -----------------------------------------------------------------------
// off_axis_angle_deg
// -----------------------------------------------------------------------

/// The angle is measured from the optical axis, and the corner angles
/// [`field_of_view`] reports are the same function at the four corners.
#[test]
fn off_axis_angle_agrees_with_the_corner_angles() {
    let cam = cam(CameraModel::SimplePinhole {
        focal_length: F_PERSP,
        principal_point_x: CX + 12.0,
        principal_point_y: CY - 7.0,
    });
    let fov = field_of_view(&cam).unwrap();
    let w = f64::from(W);
    let h = f64::from(H);
    let corner = [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]]
        .into_iter()
        .map(|[u, v]| off_axis_angle_deg(&cam, u, v))
        .fold(0.0_f64, f64::max);
    assert_relative_eq!(corner, fov.max_off_axis, epsilon = 1e-12);

    // Zero at the principal point, not at the image centre.
    assert_relative_eq!(off_axis_angle_deg(&cam, CX + 12.0, CY - 7.0), 0.0);
    assert!(off_axis_angle_deg(&cam, CX, CY) > 1.0);

    // A pinhole's closed form: θ = atan(r / f).
    let theta = off_axis_angle_deg(&cam, CX + 112.0, CY - 7.0);
    assert_relative_eq!(
        theta,
        (100.0_f64 / F_PERSP).atan().to_degrees(),
        epsilon = 1e-12
    );
}

// -----------------------------------------------------------------------
// DistortionSample::theta_deg
// -----------------------------------------------------------------------

/// Every sample carries the incidence angle of the ray its node looks along,
/// which is the field the trustworthy filter is applied to.
#[test]
fn distortion_field_samples_carry_their_incidence_angle() {
    let cam = cam(CameraModel::SimpleRadial {
        focal_length: F_PERSP,
        principal_point_x: CX,
        principal_point_y: CY,
        radial_distortion_k1: -0.15,
    });
    let field = distortion_field(&cam, 8, 6);
    let nodes = grid_nodes(8, 6);
    assert_eq!(field.len(), nodes.len());
    for (sample, &(u, v)) in field.iter().zip(&nodes) {
        assert_relative_eq!(
            sample.theta_deg,
            off_axis_angle_deg(&cam, u, v),
            epsilon = 1e-12
        );
    }
    // Monotone with radius on a radially symmetric model: a node near the
    // middle of the grid looks straighter ahead than the outermost one does.
    assert!(field[0].theta_deg > field[8 * 3 + 4].theta_deg);
}

/// [`angle_between`] holds its relative accuracy down to angles far below one
/// pixel, which is the property that makes it the crate's one spelling of the
/// quantity.
///
/// The alternative — `acos` of a normalized dot product — is what a caller
/// writes when they reach for the one-liner, and it is worst exactly where the
/// degeneracy gates live. Near zero, `cos ε ≈ 1 − ε²/2`: at `ε = 1e-8` the dot
/// product rounds to `1.0` in `f64` and `acos` returns **0**, losing the
/// measurement entirely. This test pins the accurate form against angles a
/// resection gate actually compares (`bearing_span` is tested against one
/// pixel's worth of angle, ~2e-3 rad at f = 500, and refuses below it).
#[test]
fn angle_between_is_accurate_far_below_a_pixel() {
    for &eps in &[1e-3_f64, 1e-5, 1e-8, 1e-11] {
        // Two unit rays separated by exactly `eps` about the x-axis.
        let a = [0.0, 0.0, -1.0];
        let b = [0.0, eps.sin(), -eps.cos()];
        assert_relative_eq!(angle_between(a, b), eps, max_relative = 1e-9);

        // The naive form collapses well before the accurate one does, so a
        // caller that swapped them would silently stop measuring.
        let naive = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .clamp(-1.0, 1.0)
            .acos();
        if eps <= 1e-8 {
            assert!(
                (naive - eps).abs() > 0.1 * eps,
                "acos was expected to lose this angle, but returned {naive} for {eps}"
            );
        }
    }
}

/// Neither input has to be unit length, and the answer does not depend on
/// either magnitude — the other half of why `atan2(|a × b|, a · b)` is the
/// form callers get.
#[test]
fn angle_between_ignores_input_magnitude() {
    let a = [0.0, 0.0, -1.0];
    let b = [0.0, 1.0, -1.0]; // 45° from `a`
    let expected = PI / 4.0;
    assert_relative_eq!(angle_between(a, b), expected, epsilon = 1e-12);
    assert_relative_eq!(
        angle_between([0.0, 0.0, -1e6], [0.0, 1e-6, -1e-6]),
        expected,
        epsilon = 1e-12
    );
}

/// Degrees is the radian primitive scaled, with nothing else between them.
#[test]
fn angle_between_deg_is_the_radian_form_in_degrees() {
    let a = [0.3, -0.5, -1.0];
    let b = [-0.2, 0.1, -1.0];
    assert_relative_eq!(
        angle_between_deg(a, b),
        angle_between(a, b).to_degrees(),
        epsilon = 1e-15
    );
}
