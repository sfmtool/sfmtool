use approx::assert_relative_eq;
use nalgebra::{Point3, Vector3};

use super::consensus::{
    axpy_f32, axpy_f32_scalar, consensus_phi, mean_pairwise_channel, sum_sq_diff,
    sum_sq_diff_scalar,
};
use super::obliquity::{fill_kept_obliquity_priors, fronto_prior, OBLIQUITY_PRIOR_FLOOR};
use super::support::view_render_patch;
use super::znorm::{weighted_moments, weighted_moments_scalar, znorm_write, znorm_write_scalar};
use super::*;
use crate::camera::remap::{remap_bilinear, ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel, WarpMap};
use crate::geometry::RigidTransform;

/// Helper: build a simple pinhole camera.
fn pinhole(width: u32, height: u32, focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: focal,
            focal_length_y: focal,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
        },
        width,
        height,
    }
}

const PLANE_Z: f64 = 4.0;
const IMG_W: u32 = 320;
const IMG_H: u32 = 240;
const FOCAL: f64 = 260.0;

/// Procedural texture on the world plane `z = PLANE_Z`, smooth and
/// multi-frequency so the photometric objective has a well-defined optimum.
fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// Alternate texture for simulating an occluded / wrong-surface view.
fn occluder_texture(x: f64, y: f64) -> f64 {
    127.5 + 60.0 * (y * 13.0 + 1.7).sin() + 40.0 * (x * 29.0 - 0.4).cos()
}

/// Synthesize the image a pinhole camera at `center` (looking down world +z)
/// sees of the textured plane `z = PLANE_Z`.
fn render_plane_view(center: [f64; 3], tex: fn(f64, f64) -> f64) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            // Ray from the camera center through the pixel, intersected with
            // the plane z = PLANE_Z.
            let lambda = (PLANE_Z - center[2]) / 1.0;
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(tex(x, y).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

struct Scene {
    cams: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyrs: Vec<ImageU8Pyramid>,
}

impl Scene {
    fn new(centers: &[[f64; 3]]) -> Self {
        Self::with_textures(
            centers,
            &vec![texture as fn(f64, f64) -> f64; centers.len()],
        )
    }

    fn with_textures(centers: &[[f64; 3]], texs: &[fn(f64, f64) -> f64]) -> Self {
        let cams = centers
            .iter()
            .map(|_| pinhole(IMG_W, IMG_H, FOCAL))
            .collect();
        let poses = centers
            .iter()
            .map(|c| {
                // Rotated 180° about X (canonical −Z-forward camera looking down world
                // +z); cam_from_world translation = R · (-center).
                RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
            })
            .collect();
        let pyrs = centers
            .iter()
            .zip(texs)
            .map(|(c, tex)| ImageU8Pyramid::build(&render_plane_view(*c, *tex), 5))
            .collect();
        Self { cams, poses, pyrs }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cams
            .iter()
            .zip(&self.poses)
            .zip(&self.pyrs)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }
}

/// True surface normal of the synthetic plane (toward the cameras).
fn true_normal() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.0)
}

fn plane_patch(init_n: Vector3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        init_n,
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    )
}

fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.normalize().dot(&b.normalize()).clamp(-1.0, 1.0).acos()
}

fn test_params(objective: Objective) -> NormalRefineParams {
    NormalRefineParams {
        angular_range_deg: 25.0,
        init_steps: 7,
        refine_levels: 3,
        objective,
        window: PatchWindow::GaussianDisk { sigma: 0.6 },
        min_valid_fraction: 0.5,
        min_views: 2,
        sampler: Sampler::Bilinear,
        // The refine tests exercise the confidence stencil (off in production).
        compute_confidence: true,
        ..NormalRefineParams::default()
    }
}

// ---------------------------------------------------------------------------
// Consensus identity
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random zero-mean unit-norm vector.
fn synthetic_normalized(seed: u64, len: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..len)
        .map(|i| ((seed as f64 + 1.3) * (i as f64 * 0.731 + 0.17)).sin())
        .collect();
    let mean = v.iter().sum::<f64>() / len as f64;
    for x in &mut v {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Flatten a nested `[view][channel][pixel]` test stack into the production
/// layout `data[(v*channels + c)*n + k]` as f32, returning the dims.
fn flatten_stack(xs: &[Vec<Vec<f64>>]) -> (Vec<f32>, usize, usize, usize) {
    let views = xs.len();
    let channels = xs[0].len();
    let n = xs[0][0].len();
    let mut data = vec![0f32; views * channels * n];
    for (v, per_channel) in xs.iter().enumerate() {
        for (c, col) in per_channel.iter().enumerate() {
            for (k, &x) in col.iter().enumerate() {
                data[(v * channels + c) * n + k] = x as f32;
            }
        }
    }
    (data, views, channels, n)
}

#[test]
fn sum_sq_diff_avx2_matches_scalar() {
    // The dispatched AVX2 residual reduction agrees with the scalar reference
    // (up to f32 summation order), including a length that exercises the n % 8
    // tail. Mirrors fronto_cache::resample_avx2_matches_scalar.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let row: Vec<f32> = (0..len)
            .map(|i| ((i * 7 % 53) as f32 - 26.0) * 0.13)
            .collect();
        let xbar: Vec<f32> = (0..len)
            .map(|i| ((i * 11 % 47) as f32 - 23.0) * 0.07)
            .collect();
        let want = sum_sq_diff_scalar(&row, &xbar, 0, len);
        let got = sum_sq_diff(&row, &xbar);
        assert!(
            (want - got).abs() <= 1e-3 * (1.0 + want.abs()),
            "len {len}: scalar {want} vs dispatched {got}"
        );
    }
}

#[test]
fn axpy_f32_avx2_matches_scalar() {
    // The dispatched AVX2 weighted-consensus SAXPY agrees with the scalar
    // reference, including a length that exercises the n % 8 tail.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let row: Vec<f32> = (0..len)
            .map(|i| ((i * 5 % 37) as f32 - 18.0) * 0.11)
            .collect();
        let w = 0.137f32;
        let init: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
        let mut want = init.clone();
        axpy_f32_scalar(&mut want, &row, w, 0, len);
        let mut got = init.clone();
        axpy_f32(&mut got, &row, w);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn weighted_moments_avx2_matches_scalar() {
    // The dispatched AVX2 (f64-accumulate) moments agree with the scalar
    // reference, including a length that exercises the n % 4 tail.
    for len in [0usize, 1, 3, 4, 31, 173, 512] {
        let col: Vec<f32> = (0..len)
            .map(|i| ((i * 7 % 53) as f32) * 1.7 + 3.0)
            .collect();
        let w: Vec<f64> = (0..len).map(|i| 0.3 + (i % 5) as f64 * 0.13).collect();
        let (s1s, s2s) = weighted_moments_scalar(&col, &w, 0, len);
        let (s1, s2) = weighted_moments(&col, &w);
        assert!(
            (s1 - s1s).abs() <= 1e-9 * (1.0 + s1s.abs()),
            "s1 {s1s} vs {s1}"
        );
        assert!(
            (s2 - s2s).abs() <= 1e-9 * (1.0 + s2s.abs()),
            "s2 {s2s} vs {s2}"
        );
    }
}

#[test]
fn znorm_write_avx2_matches_scalar() {
    // The dispatched AVX2 normalize write agrees with the scalar reference,
    // including a length that exercises the n % 8 tail.
    for len in [0usize, 1, 7, 8, 31, 173, 512] {
        let src: Vec<f32> = (0..len)
            .map(|i| ((i * 11 % 41) as f32) * 0.9 - 7.0)
            .collect();
        let sqrt_weights: Vec<f32> = (0..len).map(|i| 0.5 + (i % 7) as f32 * 0.05).collect();
        let (mean, inv_norm) = (12.5f32, 0.031f32);
        let mut want = vec![0f32; len];
        znorm_write_scalar(&src, &sqrt_weights, mean, inv_norm, &mut want, 0, len);
        let mut got = vec![0f32; len];
        znorm_write(&src, &sqrt_weights, mean, inv_norm, &mut got);
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn consensus_identity_matches_brute_force_pairwise_mean() {
    for v_count in [2usize, 3, 5, 8] {
        let xs: Vec<Vec<Vec<f64>>> = (0..v_count)
            .map(|i| vec![synthetic_normalized(i as u64, 24)])
            .collect();

        // Brute force: mean ZNCC over all C(V, 2) pairs.
        let mut sum = 0.0;
        let mut pairs = 0.0;
        for i in 0..v_count {
            for j in (i + 1)..v_count {
                sum += xs[i][0]
                    .iter()
                    .zip(&xs[j][0])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                pairs += 1.0;
            }
        }
        let brute = sum / pairs;

        // The consensus reads an f32 stack, so the closed form matches the f64
        // brute force only to f32 precision.
        let (d, vw, ch, nn) = flatten_stack(&xs);
        let mut sc = ConsensusScratch::default();
        let closed = mean_pairwise_channel(&d, vw, ch, nn, 0, &mut sc);
        assert_relative_eq!(closed, brute, epsilon = 1e-5);

        // The full objective averages channels; with one channel it matches.
        let phi = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, None, &mut sc).unwrap();
        assert_relative_eq!(phi, brute, epsilon = 1e-5);
    }
}

/// A unit-norm, zero-mean view that is `base` plus a small distinct perturbation.
fn consistent_view(base: &[f64], seed: u64, amount: f64) -> Vec<f64> {
    let pert = synthetic_normalized(seed, base.len());
    let mut v: Vec<f64> = base
        .iter()
        .zip(&pert)
        .map(|(b, p)| b + amount * p)
        .collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in &mut v {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn robust_consensus_with_uniform_weights_matches_unweighted() {
    // Distinct but mutually-consistent views (a shared signal + small per-view
    // perturbations): with no outlier the Tukey weights stay near-uniform, so the
    // weighted identity ρ̄_w = (‖x̄_w‖² − Σw²)/(1 − Σw²) reduces to the unweighted
    // one. (Identical views would make this trivially 1.0 with zero residuals.)
    let base = synthetic_normalized(7, 24);
    let xs: Vec<Vec<Vec<f64>>> = (0..4)
        .map(|i| vec![consistent_view(&base, 100 + i as u64, 0.12)])
        .collect();
    let (d, vw, ch, nn) = flatten_stack(&xs);
    let mut sc = ConsensusScratch::default();
    let unweighted = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, None, &mut sc).unwrap();
    let robust = consensus_phi(
        &d,
        vw,
        ch,
        nn,
        Objective::RobustWeighted { iters: 3 },
        None,
        &mut sc,
    )
    .unwrap();
    // High (not perfect) agreement, and robust tracks unweighted with no outlier.
    assert!(unweighted > 0.8 && unweighted < 1.0, "phi = {unweighted}");
    assert_relative_eq!(robust, unweighted, epsilon = 1e-2);
}

#[test]
fn robust_consensus_beats_unweighted_with_an_outlier() {
    // Three agreeing views plus one anticorrelated outlier: the unweighted mean is
    // dragged down (x̄ ≈ ½·signal ⇒ ρ̄ ≈ 0), while IRLS drives the outlier's weight
    // toward zero and recovers the agreement of the inliers.
    let good = synthetic_normalized(7, 24);
    let bad: Vec<f64> = good.iter().map(|x| -x).collect();
    let xs: Vec<Vec<Vec<f64>>> = vec![
        vec![good.clone()],
        vec![good.clone()],
        vec![good.clone()],
        vec![bad],
    ];
    let (d, vw, ch, nn) = flatten_stack(&xs);
    let mut sc = ConsensusScratch::default();
    let unweighted = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, None, &mut sc).unwrap();
    let robust = consensus_phi(
        &d,
        vw,
        ch,
        nn,
        Objective::RobustWeighted { iters: 5 },
        None,
        &mut sc,
    )
    .unwrap();
    assert!(
        unweighted.abs() < 1e-6,
        "unweighted should be ~0, got {unweighted}"
    );
    assert!(
        robust > 0.9,
        "robust should reject the outlier and recover inlier agreement, got {robust}"
    );
}

// ---------------------------------------------------------------------------
// Obliquity priors (A: consensus view-weight, B: fronto-parallel prior)
// ---------------------------------------------------------------------------

#[test]
fn kept_obliquity_priors_are_cos_pow_and_off_at_zero() {
    // Three view directions at 0°, 60°, 90° off the normal.
    let n = Vector3::new(0.0, 0.0, 1.0);
    let dirs = vec![
        Vector3::new(0.0, 0.0, 1.0),               // cos = 1
        Vector3::new(3f64.sqrt() / 2.0, 0.0, 0.5), // cos = 0.5 (60°)
        Vector3::new(1.0, 0.0, 0.0),               // cos = 0 (90°, grazing)
    ];
    let kept = [0usize, 1, 2];
    let mut pr = Vec::new();

    // power == 0 disables the prior entirely (consensus runs prior-free): returns
    // false and leaves the buffer empty.
    assert!(!fill_kept_obliquity_priors(&mut pr, &dirs, &kept, &n, 0.0));
    assert!(pr.is_empty());

    // power == 2 is the cos² foreshortening weight, floored for the grazing view.
    assert!(fill_kept_obliquity_priors(&mut pr, &dirs, &kept, &n, 2.0));
    assert_relative_eq!(pr[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(pr[1], 0.25, epsilon = 1e-12);
    assert_relative_eq!(pr[2], OBLIQUITY_PRIOR_FLOOR, epsilon = 1e-12);

    // `kept` selects and orders the priors — only the grazing view, in that order.
    // Refilling reuses the same buffer (clear + extend), so it is exactly the
    // subset with no stale tail.
    let mut sub = pr;
    assert!(fill_kept_obliquity_priors(
        &mut sub,
        &dirs,
        &[2usize, 0],
        &n,
        2.0
    ));
    assert_eq!(sub.len(), 2);
    assert_relative_eq!(sub[0], OBLIQUITY_PRIOR_FLOOR, epsilon = 1e-12);
    assert_relative_eq!(sub[1], 1.0, epsilon = 1e-12);
}

#[test]
fn fronto_prior_rewards_facing_the_cameras() {
    // Two views clustered around +z: the prior is maximized by the normal that
    // faces them and is sign-agnostic (squared), and it is exactly zero when off.
    let dirs = vec![
        Vector3::new(0.1, 0.0, 1.0).normalize(),
        Vector3::new(-0.1, 0.0, 1.0).normalize(),
    ];
    let frontal = Vector3::new(0.0, 0.0, 1.0);
    let tilted = exp_map_normal(&frontal, [35.0f64.to_radians(), 0.0]);

    assert_eq!(fronto_prior(&dirs, &frontal, 0.0), 0.0);
    assert_eq!(fronto_prior(&[], &frontal, 0.5), 0.0);

    let pf = fronto_prior(&dirs, &frontal, 0.5);
    let pt = fronto_prior(&dirs, &tilted, 0.5);
    assert!(pf > pt, "frontal {pf} should score above tilted {pt}");
    // Sign-agnostic: flipping the normal does not change the reward.
    assert_relative_eq!(fronto_prior(&dirs, &(-frontal), 0.5), pf, epsilon = 1e-12);
    // Scales linearly with the weight.
    assert_relative_eq!(
        fronto_prior(&dirs, &frontal, 1.0),
        2.0 * pf,
        epsilon = 1e-12
    );
}

#[test]
fn view_prior_downweights_a_view_in_the_robust_consensus() {
    // Four mutually-consistent views: with a uniform prior (None) the IRLS weights
    // stay near-uniform, but a prior that suppresses view 0 (the obliquity weight
    // for a grazing view) drives its final consensus weight far below the others.
    let base = synthetic_normalized(7, 24);
    let xs: Vec<Vec<Vec<f64>>> = (0..4)
        .map(|i| vec![consistent_view(&base, 100 + i as u64, 0.1)])
        .collect();
    let (d, vw, ch, nn) = flatten_stack(&xs);

    let mut sc = ConsensusScratch::default();
    irls_view_weights(&d, vw, ch, nn, 3, None, &mut sc);
    let uniform = sc.w.clone();
    assert!(
        uniform[0] > 0.15,
        "unpriored view 0 weight {} too low",
        uniform[0]
    );

    // Obliquity prior: view 0 grazing (tiny), the rest frontal.
    let priors = [OBLIQUITY_PRIOR_FLOOR, 1.0, 1.0, 1.0];
    irls_view_weights(&d, vw, ch, nn, 3, Some(&priors), &mut sc);
    assert!(
        sc.w[0] < 1e-3,
        "priored grazing view 0 weight {} should be suppressed",
        sc.w[0]
    );
    assert!(
        sc.w[1] > 0.3,
        "frontal view 1 weight {} should dominate",
        sc.w[1]
    );
}

#[test]
fn fronto_prior_recovers_frontal_normal_at_low_parallax() {
    // Near-zero baseline: tilting the plane shifts every view's patch almost
    // identically, so Φ is flat and photoconsistency alone can't pin the normal —
    // the degeneracy that lets a tilted surfel survive (the distorted-stop-sign
    // failure). Seeded from an 18°-tilted init, the fronto-parallel prior (B)
    // supplies the missing constraint and lands the normal facing the cameras
    // (≈ the true frontal normal).
    let scene = Scene::new(&[[0.002, 0.0, 0.0], [-0.002, 0.0, 0.0], [0.0, 0.002, 0.0]]);
    let views = scene.views();
    let init_n = exp_map_normal(&true_normal(), [18.0f64.to_radians(), 0.0]);
    let patch = plane_patch(init_n);
    let params = NormalRefineParams {
        fronto_prior_weight: 0.3,
        ..test_params(Objective::RobustWeighted { iters: 3 })
    };

    let result = refine_patch_normal(&patch, &views, 15, &params, None);
    let off = angle_between(&result.patch.normal(), &true_normal()).to_degrees();
    assert!(
        off < 6.0,
        "fronto prior should recover the frontal normal at low parallax, got {off} deg off"
    );
}

// ---------------------------------------------------------------------------
// Exp-map and tangent basis
// ---------------------------------------------------------------------------

#[test]
fn exp_map_is_angle_uniform_and_unit() {
    let n0 = Vector3::new(0.3, -0.5, 0.8).normalize();
    assert_relative_eq!(
        (exp_map_normal(&n0, [0.0, 0.0]) - n0).norm(),
        0.0,
        epsilon = 1e-12
    );

    for &(a, b) in &[
        (0.1, 0.0),
        (0.0, 0.2),
        (-0.15, 0.3),
        (0.4, -0.4),
        (1.0, 0.5),
    ] {
        let n = exp_map_normal(&n0, [a, b]);
        assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-12);
        let theta = f64::hypot(a, b);
        assert_relative_eq!(angle_between(&n, &n0), theta, epsilon = 1e-12);
    }

    // Tilting along the first basis axis lands exactly on cos·n0 + sin·u.
    let (u, _) = tangent_basis(&n0);
    let theta = 0.3;
    let n = exp_map_normal(&n0, [theta, 0.0]);
    assert_relative_eq!(
        (n - (n0 * theta.cos() + u * theta.sin())).norm(),
        0.0,
        epsilon = 1e-12
    );
}

#[test]
fn tangent_basis_is_deterministic_and_orthonormal() {
    for n in [
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.3, -0.5, 0.8),
        Vector3::new(-0.7, 0.7, 0.1),
    ] {
        let nn = n.normalize();
        let (u, v) = tangent_basis(&n);
        assert_relative_eq!(u.norm(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(v.norm(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(u.dot(&nn), 0.0, epsilon = 1e-12);
        assert_relative_eq!(v.dot(&nn), 0.0, epsilon = 1e-12);
        assert_relative_eq!(u.dot(&v), 0.0, epsilon = 1e-12);
        // v completes the right-handed frame.
        assert_relative_eq!((nn.cross(&u) - v).norm(), 0.0, epsilon = 1e-12);
        // Pure function of n: a second call returns the identical basis.
        let (u2, v2) = tangent_basis(&n);
        assert_eq!(u, u2);
        assert_eq!(v, v2);
        // Scale-invariant in n.
        let (u3, v3) = tangent_basis(&(n * 2.5));
        assert_relative_eq!((u - u3).norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!((v - v3).norm(), 0.0, epsilon = 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Synthetic refinement
// ---------------------------------------------------------------------------

#[test]
fn confidence_is_nan_when_not_requested() {
    // A patch that refines fine still reports NaN confidence when the stencil is
    // not requested (off by default in production); the score is unaffected.
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    let init_n = exp_map_normal(&true_normal(), [10.0f64.to_radians(), 0.0]);
    let patch = plane_patch(init_n);
    let params = NormalRefineParams {
        compute_confidence: false,
        ..test_params(Objective::MeanPairwise)
    };

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    assert!(result.photoconsistency.is_finite());
    assert!(result.confidence.is_nan());
}

#[test]
fn recovers_fronto_parallel_normal_from_tilted_init() {
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let truth = true_normal();
    let init_n = exp_map_normal(&truth, [15.0f64.to_radians(), 0.0]);
    let patch = plane_patch(init_n);
    let params = test_params(Objective::MeanPairwise);

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    let init_err = angle_between(&init_n, &truth);
    let refined_err = angle_between(&result.patch.normal(), &truth);
    assert!(
        refined_err < init_err && refined_err < 5.0f64.to_radians(),
        "refined normal should move toward truth: init {:.2}°, refined {:.2}°",
        init_err.to_degrees(),
        refined_err.to_degrees()
    );
    assert!(
        result.photoconsistency > result.init_photoconsistency,
        "Φ should improve: init {} -> {}",
        result.init_photoconsistency,
        result.photoconsistency
    );
    assert!(result.photoconsistency > 0.5);
    assert_eq!(result.valid_view_count, 4);
    assert!(result.confidence.is_finite() && result.confidence >= 0.0);

    // Frame conventions preserved: same center / half_extent, u reprojected
    // onto the new plane, v = n × u (right-handed frame; normal = u × v).
    assert_eq!(result.patch.center, patch.center);
    assert_eq!(result.patch.half_extent, patch.half_extent);
    let n = result.patch.normal();
    assert_relative_eq!(result.patch.u_axis.norm(), 1.0, epsilon = 1e-9);
    assert_relative_eq!(result.patch.u_axis.dot(&n), 0.0, epsilon = 1e-9);
    assert_relative_eq!(
        (n.cross(&result.patch.u_axis) - result.patch.v_axis).norm(),
        0.0,
        epsilon = 1e-9
    );
}

#[test]
fn refine_leaves_points_at_infinity_untouched() {
    // A point at infinity has a fixed outward normal (`normalize(-d)`); the
    // refiner must skip it and return its frame byte-for-byte unchanged, even
    // when given enough views that a finite patch would be refined.
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    let patch = OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.02, 0.02],
    );
    let params = test_params(Objective::MeanPairwise);

    let result = refine_patch_normal(&patch, &views, 16, &params, None);

    assert_eq!(result.patch.w, 0.0);
    assert_eq!(result.patch.center, patch.center);
    assert_eq!(result.patch.u_axis, patch.u_axis);
    assert_eq!(result.patch.v_axis, patch.v_axis);
    assert_eq!(result.patch.half_extent, patch.half_extent);
    assert_eq!(result.valid_view_count, 0);
}

#[test]
fn representative_is_none_unless_requested() {
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    // test_params leaves render_bitmap at its default (false).
    let params = test_params(Objective::MeanPairwise);
    let result = refine_patch_normal(&patch, &views, 16, &params, None);
    assert!(result.representative.is_none());
}

#[test]
fn representative_fuses_consistent_views_with_high_agreement() {
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    let mut params = test_params(Objective::RobustWeighted { iters: 3 });
    params.render_bitmap = true;
    let resolution = 16u32;
    let r = resolution as usize;

    let result = refine_patch_normal(&patch, &views, resolution, &params, None);
    let rep = result.representative.expect("bitmap should be rendered");
    assert_eq!(rep.len(), r * r * 4);

    // Center pixel: well inside the gaussian-disk support and covered by every
    // view. Source is single-channel, so the fused RGB is grey.
    let center = (r / 2 * r + r / 2) * 4;
    let (cr, cg, cb, ca) = (
        rep[center],
        rep[center + 1],
        rep[center + 2],
        rep[center + 3],
    );
    assert_eq!(cr, cg);
    assert_eq!(cg, cb);
    // All four views observe the same texture, so cross-view agreement is high
    // (alpha approaches the coverage ceiling of 255·(1 − 1/4) ≈ 191 for 4 views).
    assert!(
        ca > 180,
        "agreement alpha should be high for identical views, got {ca}"
    );

    // Fusion is a (weighted) mean of identical renders, so it matches a single
    // view's render of the same patch at the same pixel.
    let map = WarpMap::from_patch(
        &result.patch,
        views[0].camera,
        views[0].cam_from_world,
        resolution,
    );
    let img0 = remap_bilinear(views[0].pyramid.level(0), &map);
    let v0 = img0.get_pixel((r / 2) as u32, (r / 2) as u32, 0) as i32;
    assert!(
        (cr as i32 - v0).abs() <= 4,
        "fused grey {cr} should match the single-view render {v0}"
    );
}

#[test]
fn representative_is_none_when_too_few_views() {
    let scene = Scene::new(&[[0.0, 0.0, 0.0]]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    let mut params = test_params(Objective::MeanPairwise);
    params.render_bitmap = true; // min_views is 2, so a single view skips the search
    let result = refine_patch_normal(&patch, &views, 16, &params, None);
    assert!(result.representative.is_none());
}

#[test]
fn recovers_normal_when_no_seed_is_at_the_optimum() {
    // Asymmetric cameras (all offset toward +x) so the mean-viewing seed is tilted
    // off the true fronto-parallel normal; combined with a tilted init, *neither*
    // seed sits at the optimum, so the coarse-to-fine search must actually traverse
    // to recover it (the symmetric recovery test above has mean-viewing == truth).
    let centers = [
        [0.5, 0.1, 0.0],
        [1.1, 0.4, 0.0],
        [1.3, -0.5, 0.0],
        [0.7, 0.6, 0.0],
        [0.9, -0.2, 0.0],
    ];
    let scene = Scene::new(&centers);
    let views = scene.views();
    let truth = true_normal();

    let cam_centers: Vec<Point3<f64>> = centers
        .iter()
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();
    let mean_view = mean_viewing_normal(&Point3::new(0.0, 0.0, PLANE_Z), &cam_centers);
    let init_n = exp_map_normal(&truth, [-12.0f64.to_radians(), 10.0f64.to_radians()]);
    // Precondition: neither seed is at the optimum.
    let mean_err = angle_between(&mean_view, &truth);
    let init_err = angle_between(&init_n, &truth);
    assert!(
        mean_err > 5.0f64.to_radians(),
        "mean-view seed too close: {mean_err}"
    );
    assert!(init_err > 10.0f64.to_radians());

    let patch = plane_patch(init_n);
    let params = test_params(Objective::MeanPairwise);
    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    let refined_err = angle_between(&result.patch.normal(), &truth);
    assert!(
        refined_err < 8.0f64.to_radians() && refined_err < mean_err && refined_err < init_err,
        "search should traverse from non-optimal seeds to the true normal: \
         refined {:.2}° (init {:.2}°, mean-view {:.2}°)",
        refined_err.to_degrees(),
        init_err.to_degrees(),
        mean_err.to_degrees(),
    );
    assert!(result.photoconsistency > result.init_photoconsistency);
}

#[test]
fn robust_objective_downweights_occluded_view() {
    // Four agreeing views plus one whose image shows a different surface.
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture, texture, occluder_texture];
    let scene = Scene::with_textures(
        &[
            [0.8, 0.0, 0.0],
            [-0.8, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, -0.7, 0.0],
            [0.4, 0.4, 0.0],
        ],
        &texs,
    );
    let views = scene.views();
    let truth = true_normal();
    let init_n = exp_map_normal(&truth, [0.0, 12.0f64.to_radians()]);
    let patch = plane_patch(init_n);
    let params = test_params(Objective::RobustWeighted { iters: 3 });

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    let refined_err = angle_between(&result.patch.normal(), &truth);
    assert!(
        refined_err < angle_between(&init_n, &truth) && refined_err < 5.0f64.to_radians(),
        "robust refinement should still recover the normal: {:.2}°",
        refined_err.to_degrees()
    );
    assert!(result.photoconsistency >= result.init_photoconsistency);
}

#[test]
fn search_objective_maps_robust_iters() {
    // None defers to `objective`; Some(0) is the cheap mean-pairwise; Some(k) is
    // robust at k iterations. The final / confidence passes always use `objective`.
    let mut p = NormalRefineParams {
        objective: Objective::RobustWeighted { iters: 3 },
        search_robust_iters: None,
        ..NormalRefineParams::default()
    };
    assert_eq!(p.search_objective(), Objective::RobustWeighted { iters: 3 });
    p.search_robust_iters = Some(0);
    assert_eq!(p.search_objective(), Objective::MeanPairwise);
    p.search_robust_iters = Some(2);
    assert_eq!(p.search_objective(), Objective::RobustWeighted { iters: 2 });
}

#[test]
fn cheap_search_objective_still_recovers_normal_with_honest_phi() {
    // A cheaper search objective (mean-pairwise) than the robust final pass still
    // finds the normal, and the reported Φ is the robust final-pass score — never
    // below init — so the search-only knob can't inflate the reported quality.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let truth = true_normal();
    let init_n = exp_map_normal(&truth, [14.0f64.to_radians(), 0.0]);
    let patch = plane_patch(init_n);
    let mut params = test_params(Objective::RobustWeighted { iters: 3 });
    params.search_robust_iters = Some(0);

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    let refined_err = angle_between(&result.patch.normal(), &truth);
    assert!(
        refined_err < angle_between(&init_n, &truth) && refined_err < 5.0f64.to_radians(),
        "cheap-search refinement should still recover the normal: {:.2}°",
        refined_err.to_degrees()
    );
    assert!(result.photoconsistency >= result.init_photoconsistency);
}

#[test]
fn robust_objective_scores_a_min_view_track() {
    // Regression: under the default RobustWeighted objective a track observed by
    // exactly `min_views` views was silently left unrefined, because the
    // effective-view gate `1/Σw² ≥ min_views` fails for clean (near- but not
    // exactly uniform) IRLS weights. A clean 3-view track at min_views = 3 must be
    // scored.
    let scene = Scene::new(&[[0.7, 0.2, 0.0], [-0.6, 0.3, 0.0], [0.1, -0.7, 0.0]]);
    let views = scene.views();
    let mut params = test_params(Objective::RobustWeighted { iters: 3 });
    params.min_views = 3;
    let patch = plane_patch(exp_map_normal(&true_normal(), [8.0f64.to_radians(), 0.0]));

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    assert_eq!(result.valid_view_count, 3);
    assert!(
        result.photoconsistency.is_finite() && result.init_photoconsistency.is_finite(),
        "3-view track must score under RobustWeighted: Φ {} init {}",
        result.photoconsistency,
        result.init_photoconsistency
    );
    assert!(result.photoconsistency >= result.init_photoconsistency);
}

#[test]
fn never_returns_worse_photoconsistency_than_init() {
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    let truth = true_normal();
    for objective in [
        Objective::MeanPairwise,
        Objective::RobustWeighted { iters: 2 },
    ] {
        for delta in [
            [0.0, 0.0],
            [8.0f64.to_radians(), 0.0],
            [0.0, -8.0f64.to_radians()],
            [14.0f64.to_radians(), 10.0f64.to_radians()],
        ] {
            let patch = plane_patch(exp_map_normal(&truth, delta));
            let result = refine_patch_normal(&patch, &views, 13, &test_params(objective), None);
            assert!(result.photoconsistency.is_finite());
            assert!(result.init_photoconsistency.is_finite());
            assert!(
                result.photoconsistency >= result.init_photoconsistency - 1e-12,
                "Φ {} < init Φ {} for delta {:?}",
                result.photoconsistency,
                result.init_photoconsistency,
                delta
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Validity gating
// ---------------------------------------------------------------------------

#[test]
fn too_few_views_skips_search() {
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    let mut params = test_params(Objective::MeanPairwise);
    params.min_views = 3;

    let result = refine_patch_normal(&patch, &views, 15, &params, None);

    // Unrefined: input patch returned verbatim, NaN scores, zero confidence.
    assert_eq!(result.patch.u_axis, patch.u_axis);
    assert_eq!(result.patch.v_axis, patch.v_axis);
    assert!(result.photoconsistency.is_nan());
    assert!(result.init_photoconsistency.is_nan());
    assert_eq!(result.valid_view_count, 0);
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn back_facing_init_normal_is_returned_unrefined() {
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    // Normal pointing away from every camera: the init support is undefined.
    let patch = plane_patch(-true_normal());
    let result = refine_patch_normal(
        &patch,
        &views,
        15,
        &test_params(Objective::MeanPairwise),
        None,
    );
    assert!(result.photoconsistency.is_nan());
    assert_eq!(result.valid_view_count, 0);
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn min_valid_fraction_drops_offscreen_view() {
    // Three good cameras plus one far off-axis whose frame misses the patch.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [8.0, 0.0, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [6.0f64.to_radians(), 0.0]));

    let result = refine_patch_normal(
        &patch,
        &views,
        15,
        &test_params(Objective::MeanPairwise),
        None,
    );
    assert_eq!(result.valid_view_count, 3);
    assert!(result.photoconsistency.is_finite());
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

#[test]
fn confidence_flags_narrow_baseline_degeneracy() {
    // Wide baseline: Φ is peaked, the normal is well constrained.
    let wide = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    // Nearly coincident cameras: tilting the plane shifts all patches
    // identically, Φ is genuinely flat, the confidence must collapse.
    let narrow = Scene::new(&[[0.002, 0.0, 0.0], [-0.002, 0.0, 0.0], [0.0, 0.002, 0.0]]);
    let params = test_params(Objective::MeanPairwise);
    let patch = plane_patch(true_normal());

    let wide_views = wide.views();
    let narrow_views = narrow.views();
    let wide_result = refine_patch_normal(&patch, &wide_views, 15, &params, None);
    let narrow_result = refine_patch_normal(&patch, &narrow_views, 15, &params, None);

    assert!(wide_result.confidence.is_finite() && wide_result.confidence >= 0.0);
    assert!(narrow_result.confidence.is_finite() && narrow_result.confidence >= 0.0);
    assert!(
        narrow_result.confidence < wide_result.confidence,
        "narrow-baseline confidence {} should be below wide-baseline {}",
        narrow_result.confidence,
        wide_result.confidence
    );
    assert!(
        narrow_result.confidence < 0.3,
        "flat Φ must report low confidence, got {}",
        narrow_result.confidence
    );
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

#[test]
fn refine_patch_cloud_normals_refines_in_place() {
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let truth = true_normal();
    let tilts = [
        [10.0f64.to_radians(), 0.0],
        [0.0, -12.0f64.to_radians()],
        [8.0f64.to_radians(), 8.0f64.to_radians()],
    ];
    let mut cloud = PatchCloud {
        patches: tilts
            .iter()
            .map(|&d| plane_patch(exp_map_normal(&truth, d)))
            .collect(),
        point_indexes: vec![0, 1, 2],
    };
    let patch_views: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3]; 3];

    let results = refine_patch_cloud_normals(
        &mut cloud,
        &views,
        &patch_views,
        15,
        &test_params(Objective::MeanPairwise),
        None,
        None,
    );

    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        // The cloud was updated in place with the refined patch.
        assert_eq!(cloud.patches[i].u_axis, r.patch.u_axis);
        assert!(r.photoconsistency >= r.init_photoconsistency - 1e-12);
        let err = angle_between(&cloud.patches[i].normal(), &truth);
        assert!(
            err < 5.0f64.to_radians(),
            "patch {i} refined to {:.2}° from truth",
            err.to_degrees()
        );
    }
}

// ---------------------------------------------------------------------------
// Keypoint-anchored refinement
// ---------------------------------------------------------------------------

/// Project the patch center into a view, returning the source-image pixel — the
/// reprojection `project_i(X_p)` that the keypoint plumbing recenters against.
fn project_center(patch: &OrientedPatch, view: &ProjectedImage<'_>) -> [f64; 2] {
    let pc = view
        .cam_from_world
        .transform_point_homogeneous(patch.center.coords, patch.w);
    let (px, py) = view
        .camera
        .ray_to_pixel([pc.x, pc.y, pc.z])
        .expect("in frame");
    [px, py]
}

#[test]
fn keypoints_at_reprojection_match_no_keypoint_refine() {
    // Invariant: anchoring every view at exactly its own reprojection of the
    // point center (offset ≈ 0) must reproduce the no-keypoint refine to a tight
    // tolerance — proving the plumbing doesn't perturb the no-offset case. Use the
    // exact (cache-off) path on both sides so only the keypoint argument differs.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let truth = true_normal();
    let init_n = exp_map_normal(&truth, [12.0f64.to_radians(), 0.0]);
    let patch = plane_patch(init_n);
    let params = NormalRefineParams {
        cache: CacheMode::Off,
        ..test_params(Objective::RobustWeighted { iters: 3 })
    };

    let baseline = refine_patch_normal(&patch, &views, 15, &params, None);

    // Keypoint = each view's reprojection of the (unrefined) center.
    let kps: Vec<Option<[f64; 2]>> = views
        .iter()
        .map(|v| Some(project_center(&patch, v)))
        .collect();
    let anchored = refine_patch_normal(&patch, &views, 15, &params, Some(&kps));

    let dn = angle_between(&baseline.patch.normal(), &anchored.patch.normal());
    assert!(
        dn < 1e-6,
        "zero-offset keypoints must match no-keypoint normal: Δ {} rad",
        dn
    );
    assert_relative_eq!(
        baseline.photoconsistency,
        anchored.photoconsistency,
        epsilon = 1e-9
    );
    assert_eq!(baseline.valid_view_count, anchored.valid_view_count);
}

#[test]
fn offset_keypoints_change_the_refined_result() {
    // A real in-plane keypoint offset positions the patches off the point center,
    // so the rendered tiles (and hence the refined normal / Φ) differ from the
    // no-keypoint refine. Sanity that the offset is actually applied.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [10.0f64.to_radians(), 0.0]));
    let params = NormalRefineParams {
        cache: CacheMode::Off,
        ..test_params(Objective::MeanPairwise)
    };

    let baseline = refine_patch_normal(&patch, &views, 15, &params, None);

    // Shift each view's keypoint by a sizeable, view-dependent amount in pixels,
    // so the anchored patches sample a different part of the textured plane.
    let kps: Vec<Option<[f64; 2]>> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let c = project_center(&patch, v);
            let s = 1.0 + i as f64;
            Some([c[0] + 14.0 + 3.0 * s, c[1] - 11.0 - 2.0 * s])
        })
        .collect();
    let anchored = refine_patch_normal(&patch, &views, 15, &params, Some(&kps));

    let dn = angle_between(&baseline.patch.normal(), &anchored.patch.normal());
    let dphi = (baseline.photoconsistency - anchored.photoconsistency).abs();
    assert!(
        dn > 1e-4 || dphi > 1e-4,
        "offset keypoints should change the refined result: Δnormal {} rad, ΔΦ {}",
        dn,
        dphi
    );
}

#[test]
fn view_render_patch_anchors_center_at_keypoint() {
    // Provable anchoring: recentering a view's patch onto a keypoint makes the
    // recentered center reproject to *exactly* that keypoint. Guards the
    // seed_offset axis order (x before y) and the round-trip math — a transposition
    // would fail since the offset is deliberately asymmetric in x and y.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(true_normal());
    for v in &views {
        let c = project_center(&patch, v);
        let kp = [c[0] + 9.0, c[1] - 6.0]; // asymmetric in x and y
        let rp = view_render_patch(&patch, v, Some(kp));
        let pc = v
            .cam_from_world
            .transform_point_homogeneous(rp.center.coords, rp.w);
        let (px, py) = v.camera.ray_to_pixel([pc.x, pc.y, pc.z]).expect("in frame");
        assert_relative_eq!(px, kp[0], epsilon = 1e-6);
        assert_relative_eq!(py, kp[1], epsilon = 1e-6);
    }
}

#[test]
fn mixed_some_none_keypoints_apply_per_view() {
    // A keypoint on only some views (None on the rest) must still apply the offset
    // on the Some views — exercising the mixed Some/None indexing. With one view
    // offset and the rest centered (None), the refined result differs from the
    // all-None baseline.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [10.0f64.to_radians(), 0.0]));
    let params = NormalRefineParams {
        cache: CacheMode::Off,
        ..test_params(Objective::MeanPairwise)
    };
    let baseline = refine_patch_normal(&patch, &views, 15, &params, None);

    let mut kps: Vec<Option<[f64; 2]>> = vec![None; views.len()];
    let c0 = project_center(&patch, &views[0]);
    kps[0] = Some([c0[0] + 16.0, c0[1] - 13.0]);
    let mixed = refine_patch_normal(&patch, &views, 15, &params, Some(&kps));

    let dn = angle_between(&baseline.patch.normal(), &mixed.patch.normal());
    let dphi = (baseline.photoconsistency - mixed.photoconsistency).abs();
    assert!(
        dn > 1e-4 || dphi > 1e-4,
        "a single offset view (rest None) must perturb the result: Δn {} ΔΦ {}",
        dn,
        dphi
    );
}

#[test]
fn keypoints_leave_points_at_infinity_untouched() {
    // An infinity patch (w==0) is skipped before any keypoint is indexed; passing
    // Some(keypoints) must not panic and must leave the frame byte-for-byte.
    let scene = Scene::new(&[[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]);
    let views = scene.views();
    let patch = OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.02, 0.02],
    );
    let params = test_params(Objective::MeanPairwise);
    let kps: Vec<Option<[f64; 2]>> = views.iter().map(|_| Some([10.0, 12.0])).collect();

    let result = refine_patch_normal(&patch, &views, 16, &params, Some(&kps));

    assert_eq!(result.patch.w, 0.0);
    assert_eq!(result.patch.center, patch.center);
    assert_eq!(result.patch.u_axis, patch.u_axis);
    assert_eq!(result.patch.v_axis, patch.v_axis);
    assert_eq!(result.valid_view_count, 0);
}

#[test]
fn cache_matches_exact_with_keypoints() {
    // Approximation-budget check: the fronto cache holds the keypoint offset at the
    // seed normal (vs the exact path recomputing it per candidate), so the cached
    // keypoint-anchored refine must land within the cache's resampling tolerance of
    // the exact one. (This bounds the approximation; the *honoring* of the keypoint
    // is guarded by `cache_honors_keypoints`, since on a flat plane the normal is
    // nearly invariant to an in-plane shift.)
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [12.0f64.to_radians(), 0.0]));
    // Real (a few px), view-dependent keypoint offsets from the projection.
    let kps: Vec<Option<[f64; 2]>> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let c = project_center(&patch, v);
            let s = 1.0 + i as f64;
            Some([c[0] + 0.6 * s, c[1] - 0.4 * s])
        })
        .collect();
    let exact = NormalRefineParams {
        cache: CacheMode::Off,
        ..test_params(Objective::RobustWeighted { iters: 3 })
    };
    let cached = NormalRefineParams {
        cache: CacheMode::FrontoParallel,
        ..test_params(Objective::RobustWeighted { iters: 3 })
    };
    let r_exact = refine_patch_normal(&patch, &views, 24, &exact, Some(&kps));
    let r_cached = refine_patch_normal(&patch, &views, 24, &cached, Some(&kps));

    let dn = angle_between(&r_exact.patch.normal(), &r_cached.patch.normal());
    assert!(
        dn < 2.0f64.to_radians(),
        "cache vs exact (keypoints) normal Δ {} rad exceeds the resampling budget",
        dn
    );
}

#[test]
fn cache_honors_keypoints() {
    // Guard that the cache path actually *applies* the keypoint offset: with a
    // large, view-dependent offset the cached refine WITH keypoints must differ
    // from the cached refine WITHOUT keypoints. If `prerender` ignored the
    // keypoint (center_offset == 0) the two would be byte-identical and this fails
    // — the regression an "anchored ≈ exact" tolerance test cannot catch on a flat
    // plane.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [10.0f64.to_radians(), 0.0]));
    let params = NormalRefineParams {
        cache: CacheMode::FrontoParallel,
        ..test_params(Objective::MeanPairwise)
    };
    let kps: Vec<Option<[f64; 2]>> = views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let c = project_center(&patch, v);
            let s = 1.0 + i as f64;
            Some([c[0] + 14.0 + 3.0 * s, c[1] - 11.0 - 2.0 * s])
        })
        .collect();

    let no_kp = refine_patch_normal(&patch, &views, 15, &params, None);
    let with_kp = refine_patch_normal(&patch, &views, 15, &params, Some(&kps));

    let dn = angle_between(&no_kp.patch.normal(), &with_kp.patch.normal());
    let dphi = (no_kp.photoconsistency - with_kp.photoconsistency).abs();
    assert!(
        dn > 1e-4 || dphi > 1e-4,
        "cached refine must apply the keypoint offset: Δn {} rad ΔΦ {}",
        dn,
        dphi
    );
}

#[test]
fn all_none_keypoints_match_no_keypoints() {
    // An all-`None` keypoint slice must reproduce the no-keypoint refine exactly —
    // the documented `Cow::Borrowed` no-op path, locked byte-for-byte.
    let scene = Scene::new(&[
        [0.8, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
    ]);
    let views = scene.views();
    let patch = plane_patch(exp_map_normal(&true_normal(), [12.0f64.to_radians(), 0.0]));
    let params = test_params(Objective::RobustWeighted { iters: 3 });

    let baseline = refine_patch_normal(&patch, &views, 15, &params, None);
    let all_none: Vec<Option<[f64; 2]>> = vec![None; views.len()];
    let anchored = refine_patch_normal(&patch, &views, 15, &params, Some(&all_none));

    assert_eq!(baseline.patch.normal(), anchored.patch.normal());
    assert_eq!(baseline.photoconsistency, anchored.photoconsistency);
    assert_eq!(baseline.valid_view_count, anchored.valid_view_count);
}
