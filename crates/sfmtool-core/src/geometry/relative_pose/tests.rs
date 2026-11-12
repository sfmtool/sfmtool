// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the ray-space two-view estimators.
//!
//! Every fixture plants a scene whose rays reach past the hemisphere
//! (`θ ≥ 90°`, i.e. `ray.z < 0`), because that is exactly the population a
//! `z = 1` normalization would silently drop.

use super::*;
use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

/// Deterministic pseudo-random unit ray, spread over the whole sphere out to
/// `theta_max`.
fn planted_rays(n: usize, theta_max: f64) -> Vec<Vector3<f64>> {
    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let theta = theta_max * t;
            let phi = 2.399_963_23 * i as f64;
            Vector3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect()
}

/// A rotation of `angle` radians about a fixed slanted axis.
fn planted_rotation(angle: f64) -> Matrix3<f64> {
    let axis = Unit::new_normalize(Vector3::new(0.31, -0.87, 0.38));
    Rotation3::from_axis_angle(&axis, angle).into_inner()
}

/// A planted two-view scene: the two ray sets and the relative pose behind
/// them.
type TwoViewScene = (
    Vec<Vector3<f64>>,
    Vec<Vector3<f64>>,
    Matrix3<f64>,
    Vector3<f64>,
);

/// Two-view scene: points at planted depths along the image-1 rays, seen from
/// a second camera at `(R, t)`. Returns the two ray sets and the poses.
fn two_view_scene(n: usize, theta_max: f64, angle: f64, t: Vector3<f64>) -> TwoViewScene {
    let rot = planted_rotation(angle);
    let r1 = planted_rays(n, theta_max);
    let r2 = r1
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let depth = 3.0 + 2.0 * ((0.7 * i as f64).sin() + 1.0);
            (rot * (d * depth) + t).normalize()
        })
        .collect();
    (r1, r2, rot, t)
}

fn n_beyond_hemisphere(rays: &[Vector3<f64>]) -> usize {
    rays.iter().filter(|r| r.z <= 0.0).count()
}

// ── Epipolar cell ───────────────────────────────────────────────────────────

#[test]
fn essential_recovers_a_planted_two_view_geometry_past_the_hemisphere() {
    let (r1, r2, rot, t) = two_view_scene(160, 2.0, 0.35, Vector3::new(0.9, -0.25, 0.4));
    assert!(
        n_beyond_hemisphere(&r1) > 30,
        "fixture must exercise theta >= 90 deg"
    );
    let out = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-6,
            min_inliers: 20,
            ..Default::default()
        },
    )
    .expect("a noise-free scene has an epipolar consensus");
    assert!(out.inliers.iter().all(|&b| b), "every planted ray is exact");
    assert!(
        out.essentialness < 1e-9,
        "the exact geometry is essential: {}",
        out.essentialness
    );
    // E = [t]x R up to sign and scale.
    let tx = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    let truth = (tx * rot).normalize();
    let got = out.e_matrix;
    let d = (got - truth).norm().min((got + truth).norm());
    assert!(d < 1e-9, "epipolar matrix off by {d}");
}

#[test]
fn essential_rejects_outliers_and_keeps_the_planted_inliers() {
    let (mut r1, r2, _rot, _t) = two_view_scene(200, 2.1, 0.3, Vector3::new(0.6, 0.3, -0.5));
    // Corrupt a quarter of the correspondences by rotating one side far away.
    let junk = planted_rotation(0.9);
    for i in (0..r1.len()).step_by(4) {
        r1[i] = junk * r1[i];
    }
    let out = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-4,
            min_inliers: 20,
            samples: 800,
            ..Default::default()
        },
    )
    .expect("three quarters of the population is exact");
    let kept = out.inliers.iter().filter(|&&b| b).count();
    assert!(
        kept >= 145,
        "kept only {kept} of ~150 clean correspondences"
    );
    assert!(
        (0..r1.len()).step_by(4).filter(|&i| out.inliers[i]).count() <= 3,
        "outliers leaked into the consensus"
    );
}

#[test]
fn essential_is_deterministic_in_the_seed() {
    let (r1, r2, _rot, _t) = two_view_scene(120, 1.9, 0.25, Vector3::new(0.4, 0.1, 0.8));
    let opts = RayEssentialOptions {
        max_angle_rad: 1e-5,
        min_inliers: 20,
        seed: 7,
        ..Default::default()
    };
    let a = estimate_essential_rays(&r1, &r2, &opts).unwrap();
    let b = estimate_essential_rays(&r1, &r2, &opts).unwrap();
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.e_matrix, b.e_matrix);
}

#[test]
fn essential_abstains_below_the_inlier_floor() {
    let (r1, mut r2, _rot, _t) = two_view_scene(60, 1.8, 0.3, Vector3::new(0.5, 0.5, 0.2));
    // Scrambled partners: no epipolar geometry explains them.
    r2.rotate_left(17);
    assert!(estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 1e-6,
            min_inliers: 40,
            ..Default::default()
        },
    )
    .is_none());
    // A degenerate tolerance is rejected outright rather than silently taken.
    assert!(estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            max_angle_rad: 0.0,
            ..Default::default()
        },
    )
    .is_none());
}

#[test]
fn one_sided_residuals_are_different_measurements() {
    let (r1, r2, _rot, _t) = two_view_scene(150, 2.0, 0.4, Vector3::new(0.7, -0.4, 0.3));
    let base = RayEssentialOptions {
        max_angle_rad: 3e-3,
        min_inliers: 20,
        ..Default::default()
    };
    let both = estimate_essential_rays(&r1, &r2, &base).unwrap();
    let one = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            side: EpipolarSide::One,
            ..base
        },
    )
    .unwrap();
    let two = estimate_essential_rays(
        &r1,
        &r2,
        &RayEssentialOptions {
            side: EpipolarSide::Two,
            ..base
        },
    )
    .unwrap();
    // The two-sided residual is the max of the one-sided ones, so it never
    // scores a correspondence better than either side does.
    for i in 0..r1.len() {
        assert!(both.residuals_rad[i] >= one.residuals_rad[i] - 1e-12);
        assert!(both.residuals_rad[i] >= two.residuals_rad[i] - 1e-12);
    }
}

// ── Rotation cell ───────────────────────────────────────────────────────────

#[test]
fn rotation_recovers_a_pure_ray_rotation_past_the_hemisphere() {
    let rot = planted_rotation(0.6);
    let r1 = planted_rays(180, 2.2);
    let r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    assert!(n_beyond_hemisphere(&r1) > 50);
    let out = fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-6,
            ..Default::default()
        },
    )
    .expect("a pure rotation is explained exactly");
    assert!(out.inliers.iter().all(|&b| b));
    assert!(out.rms_rad < 1e-7, "rms {}", out.rms_rad);
    assert!((out.rotation - rot).norm() < 1e-12);
}

#[test]
fn rotation_abstains_on_a_pair_with_real_parallax() {
    // Points at widely different depths under a real baseline are not a
    // rotation of each other; no rotation explains enough of them.
    let (r1, r2, _rot, _t) = two_view_scene(200, 2.0, 0.3, Vector3::new(1.6, -0.9, 0.7));
    assert!(fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-4,
            min_inliers: 60,
            ..Default::default()
        },
    )
    .is_none());
}

#[test]
fn rotation_separates_a_far_field_from_its_near_outliers() {
    let rot = planted_rotation(0.45);
    let r1 = planted_rays(200, 2.1);
    let mut r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    // A near-field minority carries genuine parallax and must be excluded.
    let t = Vector3::new(0.8, 0.2, -0.5);
    for i in (0..r1.len()).step_by(5) {
        r2[i] = (rot * (r1[i] * 1.2) + t).normalize();
    }
    let out = fit_ray_rotation(
        &r1,
        &r2,
        &RayRotationOptions {
            max_angle_rad: 1e-3,
            min_inliers: 40,
            ..Default::default()
        },
    )
    .expect("the far field dominates");
    assert!((out.rotation - rot).norm() < 1e-6);
    let near_kept = (0..r1.len()).step_by(5).filter(|&i| out.inliers[i]).count();
    assert!(near_kept <= 2, "{near_kept} near-field rays kept");
}

#[test]
fn rotation_is_deterministic_in_the_seed() {
    let rot = planted_rotation(0.5);
    let r1 = planted_rays(120, 2.0);
    let r2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    let opts = RayRotationOptions {
        max_angle_rad: 1e-5,
        seed: 11,
        ..Default::default()
    };
    let a = fit_ray_rotation(&r1, &r2, &opts).unwrap();
    let b = fit_ray_rotation(&r1, &r2, &opts).unwrap();
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.rotation, b.rotation);
}

// ── Input lengths ───────────────────────────────────────────────────────────

#[test]
fn unequal_lengths_use_the_shorter_prefix() {
    let (r1, r2, _rot, _t) = two_view_scene(150, 2.0, 0.35, Vector3::new(0.8, -0.3, 0.4));
    let rot = planted_rotation(0.5);
    let q2: Vec<Vector3<f64>> = r1.iter().map(|d| rot * d).collect();
    let essential = RayEssentialOptions {
        max_angle_rad: 1e-5,
        min_inliers: 20,
        ..Default::default()
    };
    let rotation = RayRotationOptions {
        max_angle_rad: 1e-6,
        ..Default::default()
    };
    let n = 120;
    // Either side may be the longer one; the extra entries are ignored and the
    // result matches a call on the first `n` correspondences.
    let e_trimmed = estimate_essential_rays(&r1[..n], &r2[..n], &essential).unwrap();
    let r_trimmed = fit_ray_rotation(&r1[..n], &q2[..n], &rotation).unwrap();
    for (a, b, q) in [(&r1[..], &r2[..n], &q2[..n]), (&r1[..n], &r2[..], &q2[..])] {
        let got = estimate_essential_rays(a, b, &essential).unwrap();
        assert_eq!(got.inliers.len(), n);
        assert_eq!(got.residuals_rad.len(), n);
        assert_eq!(got.e_matrix, e_trimmed.e_matrix);
        assert_eq!(got.inliers, e_trimmed.inliers);

        let got = fit_ray_rotation(a, q, &rotation).unwrap();
        assert_eq!(got.inliers.len(), n);
        assert_eq!(got.residuals_rad.len(), n);
        assert_eq!(got.rotation, r_trimmed.rotation);
        assert_eq!(got.inliers, r_trimmed.inliers);
    }
}

// ── Decomposition ───────────────────────────────────────────────────────────

/// Correspondences of points in front of both cameras, `plus` of them seen
/// from `(R, +t)` and `minus` from `(R, −t)`. Every one satisfies the same
/// epipolar matrix `[t]×·R`, so only cheirality tells the two signs apart.
fn mixed_sign_scene(plus: usize, minus: usize) -> TwoViewScene {
    let rot = planted_rotation(0.2);
    let t = Vector3::new(0.6, 0.1, -0.2);
    let r1 = planted_rays(plus + minus, 1.2);
    let r2 = r1
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let depth = 3.0 + 2.0 * ((0.7 * i as f64).sin() + 1.0);
            let sign = if i < plus { 1.0 } else { -1.0 };
            (rot * (d * depth) + t * sign).normalize()
        })
        .collect();
    (r1, r2, rot, t)
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[test]
fn decomposition_keeps_the_pose_most_correspondences_are_in_front_of() {
    for (plus, minus, sign) in [(30usize, 10usize, 1.0), (10, 30, -1.0)] {
        let (r1, r2, rot, t) = mixed_sign_scene(plus, minus);
        let e = skew(&t) * rot;
        let pose = decompose_essential_rays(&(e / e.norm()), &r1, &r2, &vec![true; r1.len()])
            .expect("a pose");
        assert!(
            (pose.rotation - rot).norm() < 1e-9,
            "rotation {:?}",
            pose.rotation
        );
        assert!(
            (pose.translation - sign * t.normalize()).norm() < 1e-9,
            "translation {:?} against {sign} t",
            pose.translation
        );
        assert_eq!(
            pose.in_front.iter().filter(|&&b| b).count(),
            plus.max(minus)
        );
        // Each correspondence in front is one seen from the majority's sign.
        let majority = |i: usize| (i < plus) == (sign > 0.0);
        assert!((0..r1.len()).all(|i| pose.in_front[i] == majority(i)));
    }
}

#[test]
fn decomposition_reads_only_the_considered_correspondences() {
    let (r1, r2, rot, t) = mixed_sign_scene(30, 10);
    let e = skew(&t) * rot;
    // Considering only the ten seen from −t turns the answer to −t.
    let considered: Vec<bool> = (0..40).map(|i| i >= 30).collect();
    let pose = decompose_essential_rays(&e, &r1, &r2, &considered).expect("a pose");
    assert!((pose.translation + t.normalize()).norm() < 1e-9);
    assert_eq!(pose.in_front, considered);
    // Nothing considered, or a matrix that is not finite, gives no pose.
    assert!(decompose_essential_rays(&e, &r1, &r2, &[false; 40]).is_none());
    let nan = Matrix3::from_element(f64::NAN);
    assert!(decompose_essential_rays(&nan, &r1, &r2, &[true; 40]).is_none());
}

// ── Sample bound ────────────────────────────────────────────────────────────

#[test]
fn more_samples_than_the_bound_give_no_estimate_rather_than_a_panic() {
    let (r1, r2, _, _) = two_view_scene(40, 1.2, 0.2, Vector3::new(0.5, 0.0, 0.1));
    for samples in [MAX_RAY_SAMPLES + 1, usize::MAX] {
        let options = RayEssentialOptions {
            samples,
            ..RayEssentialOptions::default()
        };
        assert!(estimate_essential_rays(&r1, &r2, &options).is_none());
        let options = RayRotationOptions {
            samples,
            ..RayRotationOptions::default()
        };
        assert!(fit_ray_rotation(&r1, &r2, &options).is_none());
    }
    // The bound itself is accepted, shown on the cheaper rotation fit of 20
    // rays of a pure rotation.
    let (r1, r2, _, _) = two_view_scene(20, 1.2, 0.2, Vector3::zeros());
    let options = RayRotationOptions {
        samples: MAX_RAY_SAMPLES,
        ..RayRotationOptions::default()
    };
    assert!(fit_ray_rotation(&r1, &r2, &options).is_some());
}

// ── Parallel sample scoring ─────────────────────────────────────────────────

/// The serial loop the samples were once scored with: keep a sample only when
/// its count is strictly larger than the best so far, starting from 0.
fn serial_best<const K: usize, M>(
    samples: &[[usize; K]],
    model: impl Fn(&[usize; K]) -> Option<M>,
    count: impl Fn(&M) -> usize,
) -> Option<(usize, usize, M)> {
    let mut best: Option<(usize, usize, M)> = None;
    for (k, s) in samples.iter().enumerate() {
        let Some(m) = model(s) else { continue };
        let c = count(&m);
        if c > best.as_ref().map_or(0, |b| b.0) {
            best = Some((c, k, m));
        }
    }
    best
}

#[test]
fn best_sample_keeps_the_first_sample_of_the_largest_count() {
    // Counts with many ties at the top, zeros, and samples with no model,
    // over enough samples to be scored on the pool.
    let len = 4096;
    let counts: Vec<usize> = (0..len).map(|k| (k * 7919) % 13).collect();
    let samples: Vec<[usize; 1]> = (0..len).map(|k| [k]).collect();
    let model = |s: &[usize; 1]| (s[0] % 5 != 3).then_some(s[0]);
    assert!(len >= PARALLEL_MIN_SAMPLES);
    let got = best_sample(&samples, model, || (), |&k, _| counts[k]);
    let want = serial_best(&samples, model, |&k| counts[k]).unwrap();
    // The top count is shared by many samples, so the rule decides.
    assert!(
        samples
            .iter()
            .filter(|s| model(s).is_some() && counts[s[0]] == want.0)
            .count()
            > 10
    );
    assert_eq!(got, Some((want.0, want.2)));
    // The same on the calling thread, for fewer samples.
    let few = &samples[..PARALLEL_MIN_SAMPLES - 1];
    let want_few = serial_best(few, model, |&k| counts[k]).unwrap();
    assert!(
        few.iter()
            .filter(|s| model(s).is_some() && counts[s[0]] == want_few.0)
            .count()
            > 1
    );
    assert_eq!(
        best_sample(few, model, || (), |&k, _| counts[k]),
        Some((want_few.0, want_few.2))
    );
    // No sample with an inlier gives none, as the serial loop does.
    assert_eq!(best_sample(&samples, model, || (), |_, _| 0), None);
}

/// A two-view scene with every third correspondence's second ray replaced by
/// an unrelated one, so the clean minimal samples all reach the same count.
fn contaminated_scene() -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let (r1, mut r2, _, _) = two_view_scene(300, 2.0, 0.35, Vector3::new(0.9, -0.25, 0.4));
    let far = planted_rays(300, 3.0);
    for i in (0..300).step_by(3) {
        r2[i] = far[(i * 37 + 11) % 300];
    }
    (r1, r2)
}

#[test]
fn the_essential_fit_takes_the_sample_the_serial_loop_takes() {
    let (r1, r2) = contaminated_scene();
    let options = RayEssentialOptions {
        max_angle_rad: 1e-6,
        min_inliers: 20,
        seed: 5,
        ..Default::default()
    };
    let n = r1.len();
    let sin_tol = options.max_angle_rad.sin();
    let rows = epipolar_rows(&r1, &r2);
    let mut state = options.seed;
    let drawn = draw_samples(&mut state, n, EPI_SAMPLE, options.samples);
    let samples = drawn.as_chunks::<EPI_SAMPLE>().0;
    let model = |s: &[usize; EPI_SAMPLE]| null_from_rows(&rows, s.iter().copied());
    let count = |e: &Matrix3<f64>| {
        let (mut resid, mut scratch) = (vec![0.0; n], vec![0.0; n]);
        residuals_for(e, &r1, &r2, options.side, &mut scratch, &mut resid);
        resid.iter().filter(|&&r| r < sin_tol).count()
    };
    let (want_count, want_k, want_e) = serial_best(samples, model, count).unwrap();
    // Several clean samples tie at the top count, and the first of them wins.
    let tied: Vec<usize> = (0..samples.len())
        .filter(|&k| model(&samples[k]).is_some_and(|e| count(&e) == want_count))
        .collect();
    assert!(tied.len() > 1, "{tied:?}");
    assert_eq!(tied[0], want_k);
    assert!(samples.len() >= PARALLEL_MIN_SAMPLES);
    let got = best_sample(
        samples,
        model,
        || (vec![0.0; n], vec![0.0; n]),
        |e, (resid, scratch)| {
            residuals_for(e, &r1, &r2, options.side, scratch, resid);
            resid.iter().filter(|&&r| r < sin_tol).count()
        },
    );
    assert_eq!(got, Some((want_count, want_e)));
}
