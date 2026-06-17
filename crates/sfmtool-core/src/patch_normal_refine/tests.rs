use approx::assert_relative_eq;
use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera_intrinsics::CameraModel;
use crate::remap::ImageU8;

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

/// Synthesize the image a pinhole camera at `center` (identity rotation,
/// looking down +z) sees of the textured plane `z = PLANE_Z`.
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
                // Identity rotation; cam_from_world translation = -center.
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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
        let phi = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, &mut sc).unwrap();
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
    let unweighted = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, &mut sc).unwrap();
    let robust = consensus_phi(
        &d,
        vw,
        ch,
        nn,
        Objective::RobustWeighted { iters: 3 },
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
    let unweighted = consensus_phi(&d, vw, ch, nn, Objective::MeanPairwise, &mut sc).unwrap();
    let robust = consensus_phi(
        &d,
        vw,
        ch,
        nn,
        Objective::RobustWeighted { iters: 5 },
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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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
    // onto the new plane, v = n × u.
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
    let result = refine_patch_normal(&patch, &views, 15, &params);

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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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
            let result = refine_patch_normal(&patch, &views, 13, &test_params(objective));
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

    let result = refine_patch_normal(&patch, &views, 15, &params);

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
    let result = refine_patch_normal(&patch, &views, 15, &test_params(Objective::MeanPairwise));
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

    let result = refine_patch_normal(&patch, &views, 15, &test_params(Objective::MeanPairwise));
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
    let wide_result = refine_patch_normal(&patch, &wide_views, 15, &params);
    let narrow_result = refine_patch_normal(&patch, &narrow_views, 15, &params);

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
fn refine_patch_cloud_refines_in_place() {
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
        point_ids: vec![0, 1, 2],
    };
    let patch_views: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3]; 3];

    let results = refine_patch_cloud(
        &mut cloud,
        &views,
        &patch_views,
        15,
        &test_params(Objective::MeanPairwise),
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
