// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::{Rotation3, Unit};

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Ground-truth metric scene: per-image rotation + scale + translation, 3D
/// points, and the exact affine projections `u = s_i·R_i[0..2]·X + t_i`.
struct Scene {
    rotations: Vec<Matrix3<f64>>,
    scales: Vec<f64>,
    points: Vec<Vector3<f64>>,
    translations: Vec<[f64; 2]>,
    num_images: usize,
    num_clusters: usize,
}

impl Scene {
    fn new(num_images: usize, num_clusters: usize, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let rotations: Vec<Matrix3<f64>> = (0..num_images)
            .map(|_| {
                let axis = Unit::new_normalize(Vector3::new(
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(-1.0, 1.0),
                ));
                let angle = rng.uniform(0.05, 0.8);
                *Rotation3::from_axis_angle(&axis, angle).matrix()
            })
            .collect();
        let scales: Vec<f64> = (0..num_images).map(|_| rng.uniform(0.8, 1.3)).collect();
        // Pixel-like scale: the scene signal must dominate outlier offsets
        // (tens of px), or the rank-3 fit can legitimately spend an axis
        // interpolating them (Eckart-Young).
        let translations: Vec<[f64; 2]> = (0..num_images)
            .map(|_| [rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0)])
            .collect();
        let points: Vec<Vector3<f64>> = (0..num_clusters)
            .map(|_| {
                Vector3::new(
                    rng.uniform(-80.0, 80.0),
                    rng.uniform(-80.0, 80.0),
                    rng.uniform(-80.0, 80.0),
                )
            })
            .collect();
        Self {
            rotations,
            scales,
            points,
            translations,
            num_images,
            num_clusters,
        }
    }

    /// Clean affine projection of cluster `c` into image `i`.
    fn clean_uv(&self, i: usize, c: usize) -> [f64; 2] {
        let p = self.rotations[i] * self.points[c] * self.scales[i];
        [p.x + self.translations[i][0], p.y + self.translations[i][1]]
    }

    /// Every (image, cluster) combination observed.
    fn full_observations(&self) -> (Vec<u32>, Vec<u32>, Vec<[f64; 2]>) {
        let mut clusters = Vec::new();
        let mut images = Vec::new();
        let mut xy = Vec::new();
        for i in 0..self.num_images {
            for c in 0..self.num_clusters {
                images.push(i as u32);
                clusters.push(c as u32);
                xy.push(self.clean_uv(i, c));
            }
        }
        (clusters, images, xy)
    }
}

/// Reprojection of the factorization for observation (i, c).
fn predict(fac: &AffineFactorization, i: usize, c: usize) -> [f64; 2] {
    let m = &fac.cameras[i];
    let t = &fac.translations[i];
    let x = &fac.points[c];
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2] + t[0],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2] + t[1],
    ]
}

/// True iff the hypothesis' rotations match the ground truth after global
/// rotation alignment (estimates are defined up to `R_i → R_i·D`).
fn hypothesis_matches_gt(hyp: &MetricHypothesis, scene: &Scene, used: &[bool]) -> bool {
    let first = match used.iter().position(|&u| u) {
        Some(i) => i,
        None => return false,
    };
    let est_first = Matrix3::from_fn(|r, c| hyp.rotations[first][r][c]);
    let d = scene.rotations[first].transpose() * est_first;
    (0..scene.num_images).filter(|&i| used[i]).all(|i| {
        let est = Matrix3::from_fn(|r, c| hyp.rotations[i][r][c]);
        (scene.rotations[i] * d - est).norm() < 1e-6
    })
}

#[test]
fn full_scene_recovered_exactly_up_to_gauge() {
    let scene = Scene::new(6, 40, 17);
    let (clusters, images, mut xy) = scene.full_observations();
    // A whiff of i.i.d. noise (1e-6 px on a ~100 px scene): on bitwise-exact
    // data the trimming quantile chews on *structured* numerical noise and
    // can drop a whole image; real inputs always carry measurement noise.
    let mut rng = Lcg(5);
    for u in xy.iter_mut() {
        u[0] += rng.uniform(-1e-6, 1e-6);
        u[1] += rng.uniform(-1e-6, 1e-6);
    }
    let fac = factorize_affine(
        &clusters,
        &images,
        &xy,
        scene.num_images,
        scene.num_clusters,
        &AffineFactorizationParams::default(),
    )
    .unwrap();

    assert!(fac.used_images.iter().all(|&u| u));
    // M·X + t reproduces every clean measurement (gauge cancels in the
    // product) down to the injected noise floor, and the final kept
    // residuals are at that floor.
    for (o, (&i, &c)) in images.iter().zip(&clusters).enumerate() {
        let pred = predict(&fac, i as usize, c as usize);
        let clean = scene.clean_uv(i as usize, c as usize);
        assert!(
            (pred[0] - clean[0]).hypot(pred[1] - clean[1]) < 1e-4,
            "obs {o}: pred {pred:?} vs clean {clean:?}"
        );
    }
    for (o, r) in fac.residuals.iter().enumerate() {
        if fac.keep[o] {
            assert!(r[0].hypot(r[1]) < 1e-5, "kept obs {o} residual {r:?}");
        }
    }

    // Metric upgrade: one reflection hypothesis matches the ground-truth
    // rotations after global alignment, and its scales match up to a global
    // factor.
    let hyps = metric_upgrade(&fac).expect("well-posed scene upgrades");
    assert!(
        hypothesis_matches_gt(&hyps[0], &scene, &fac.used_images)
            || hypothesis_matches_gt(&hyps[1], &scene, &fac.used_images),
        "neither hypothesis matches ground truth"
    );
    for hyp in &hyps {
        let ratios: Vec<f64> = (0..scene.num_images)
            .map(|i| hyp.scales[i] / scene.scales[i])
            .collect();
        let (lo, hi) = ratios.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &r| {
            (lo.min(r), hi.max(r))
        });
        assert!(
            hi / lo - 1.0 < 1e-7,
            "scale ratios not constant: {ratios:?}"
        );
        assert!(lo > 0.0);
    }
}

#[test]
fn missing_data_pattern_completes_unobserved_entries() {
    let scene = Scene::new(6, 40, 23);
    let (clusters, images, xy) = scene.full_observations();
    // Drop ~25% deterministically; the fixture must keep every image and
    // cluster above the sub-minimum thresholds.
    let mut rng = Lcg(99);
    let sel: Vec<bool> = (0..xy.len()).map(|_| rng.next_f64() < 0.75).collect();
    let clusters: Vec<u32> = clusters
        .iter()
        .zip(&sel)
        .filter_map(|(&c, &s)| s.then_some(c))
        .collect();
    let images: Vec<u32> = images
        .iter()
        .zip(&sel)
        .filter_map(|(&i, &s)| s.then_some(i))
        .collect();
    let xy: Vec<[f64; 2]> = xy
        .iter()
        .zip(&sel)
        .filter_map(|(&u, &s)| s.then_some(u))
        .collect();
    for i in 0..scene.num_images {
        let n = images.iter().filter(|&&v| v as usize == i).count();
        assert!(n >= 8, "fixture: image {i} has only {n} observations");
    }
    for c in 0..scene.num_clusters {
        let n = clusters.iter().filter(|&&v| v as usize == c).count();
        assert!(n >= 2, "fixture: cluster {c} has only {n} observations");
    }

    let fac = factorize_affine(
        &clusters,
        &images,
        &xy,
        scene.num_images,
        scene.num_clusters,
        &AffineFactorizationParams::default(),
    )
    .unwrap();
    assert!(fac.used_images.iter().all(|&u| u));
    // Rank-3 completion: every (image, cluster) combination — including the
    // unobserved ones — reprojects to the clean measurement. The mean-filled
    // init is inexact under missing data, so the alternation converges
    // rather than lands exactly; ~1e-4 relative on the ~100 px scene.
    for i in 0..scene.num_images {
        for c in 0..scene.num_clusters {
            let pred = predict(&fac, i, c);
            let clean = scene.clean_uv(i, c);
            assert!(
                (pred[0] - clean[0]).hypot(pred[1] - clean[1]) < 1e-2,
                "({i}, {c}): pred {pred:?} vs clean {clean:?}"
            );
        }
    }
}

/// One planted outlier per image (2.5% of 240 observations), each with its
/// own gross offset of 25-60 px. Correlated outliers (shared image + shared
/// offset) can be absorbed as a coherent camera perturbation — avoided here
/// on purpose.
fn outlier_fixture(scene: &Scene) -> (Vec<u32>, Vec<u32>, Vec<[f64; 2]>, Vec<usize>) {
    let (clusters, images, mut xy) = scene.full_observations();
    let outliers: Vec<usize> = (0..scene.num_images)
        .map(|i| i * scene.num_clusters + 7 * i + 3)
        .collect();
    let mut rng = Lcg(1234);
    for &o in &outliers {
        let ang = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
        let mag = rng.uniform(25.0, 60.0);
        xy[o][0] += mag * ang.cos();
        xy[o][1] += mag * ang.sin();
    }
    (clusters, images, xy, outliers)
}

#[test]
fn planted_outliers_rejected_and_inlier_fit_unaffected() {
    let scene = Scene::new(6, 40, 41);
    let (clusters, images, xy, outliers) = outlier_fixture(&scene);

    let fac = factorize_affine(
        &clusters,
        &images,
        &xy,
        scene.num_images,
        scene.num_clusters,
        &AffineFactorizationParams::default(),
    )
    .unwrap();
    for &o in &outliers {
        assert!(!fac.keep[o], "outlier obs {o} was kept");
        let norm = fac.residuals[o][0].hypot(fac.residuals[o][1]);
        assert!(norm > 10.0, "outlier obs {o} residual too small: {norm}");
    }
    assert!(fac.used_images.iter().all(|&u| u));
    // The inlier fit is unaffected by the 25-60 px corruptions: every clean
    // measurement reprojects to well under a pixel (the alternation restarts
    // its convergence when the outliers drop out at the first trim, so the
    // fit is tight rather than exact), and the kept set fits tightly.
    for (o, (&i, &c)) in images.iter().zip(&clusters).enumerate() {
        let pred = predict(&fac, i as usize, c as usize);
        let clean = scene.clean_uv(i as usize, c as usize);
        assert!(
            (pred[0] - clean[0]).hypot(pred[1] - clean[1]) < 0.5,
            "obs {o}: pred {pred:?} vs clean {clean:?}"
        );
        if fac.keep[o] {
            let r = fac.residuals[o];
            assert!(r[0].hypot(r[1]) < 1e-2, "kept obs {o} residual {r:?}");
            assert!(!outliers.contains(&o));
        }
    }
}

#[test]
fn bitwise_determinism() {
    let scene = Scene::new(6, 40, 41);
    let (clusters, images, xy, _) = outlier_fixture(&scene);
    let params = AffineFactorizationParams::default();
    let run = || {
        factorize_affine(
            &clusters,
            &images,
            &xy,
            scene.num_images,
            scene.num_clusters,
            &params,
        )
        .unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a, b); // PartialEq on f64 fields: bitwise-identical, no NaN
    assert_eq!(metric_upgrade(&a), metric_upgrade(&b));
}

#[test]
fn sub_minimum_images_and_clusters_keep_previous_values() {
    // 6 images and 32 clusters, but image 5 has only 3 observations,
    // cluster 30 has exactly 1, and cluster 31 has none.
    let scene = Scene::new(6, 32, 7);
    let mut clusters = Vec::new();
    let mut images = Vec::new();
    let mut xy = Vec::new();
    for i in 0..5 {
        for c in 0..30 {
            images.push(i as u32);
            clusters.push(c as u32);
            xy.push(scene.clean_uv(i, c));
        }
    }
    for c in 0..3 {
        images.push(5);
        clusters.push(c as u32);
        xy.push(scene.clean_uv(5, c));
    }
    images.push(0);
    clusters.push(30);
    xy.push(scene.clean_uv(0, 30));

    let params = AffineFactorizationParams::default();
    let fac = factorize_affine(&clusters, &images, &xy, 6, 32, &params).unwrap();

    // Image 5 never reaches 4 kept observations: zero-initialized camera
    // kept, not used.
    assert!(!fac.used_images[5]);
    assert_eq!(fac.cameras[5], [[0.0; 3]; 2]);
    assert_eq!(fac.translations[5], [0.0; 2]);
    for i in 0..5 {
        assert!(fac.used_images[i]);
    }

    // Cluster 30 (1 observation) is never refit: it stays bitwise at the
    // SVD initialization, which a rounds=0 run returns directly.
    let init = factorize_affine(
        &clusters,
        &images,
        &xy,
        6,
        32,
        &AffineFactorizationParams {
            rounds: 0,
            ..params
        },
    )
    .unwrap();
    assert_eq!(fac.points[30], init.points[30]);

    // Cluster 31 (no observations): its measurement column is zero, so the
    // top singular vectors are (numerically) orthogonal to it.
    for v in fac.points[31] {
        assert!(
            v.abs() < 1e-10,
            "unobserved cluster moved: {:?}",
            fac.points[31]
        );
    }
}

#[test]
fn error_cases() {
    let params = AffineFactorizationParams::default();
    // Non-parallel arrays.
    let err = factorize_affine(&[0, 1], &[0], &[[0.0, 0.0]; 2], 2, 2, &params).unwrap_err();
    assert_eq!(
        err,
        FactorizationError::NotParallel {
            clusters: 2,
            images: 1,
            xy: 2
        }
    );
    assert!(err.to_string().contains("parallel"));
    // Cluster index out of range.
    let err = factorize_affine(&[5], &[0], &[[0.0, 0.0]], 2, 2, &params).unwrap_err();
    assert_eq!(
        err,
        FactorizationError::ClusterIndexOutOfRange {
            index: 5,
            num_clusters: 2
        }
    );
    // Image index out of range.
    let err = factorize_affine(&[0], &[7], &[[0.0, 0.0]], 2, 2, &params).unwrap_err();
    assert_eq!(
        err,
        FactorizationError::ImageIndexOutOfRange {
            index: 7,
            num_images: 2
        }
    );
    // Dense-init size bound.
    let err = factorize_affine(&[], &[], &[], 3000, 2000, &params).unwrap_err();
    assert_eq!(
        err,
        FactorizationError::TooLarge {
            num_images: 3000,
            num_clusters: 2000
        }
    );
    assert!(err.to_string().contains("dense factorization bound"));
}

#[test]
fn quantile_matches_numpy_linear_interpolation() {
    // np.quantile reference values (default linear method), bitwise: numpy
    // returns 3.8499999999999996 here, not 3.85.
    assert_eq!(
        quantile_linear(&[1.0, 2.0, 3.0, 4.0], 0.95),
        3.849_999_999_999_999_6
    );
    assert_eq!(quantile_linear(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0);
    // t >= 0.5 lerp branch: 10 - 8 * (1 - 0.9).
    assert_eq!(quantile_linear(&[2.0, 10.0], 0.9), 10.0 - 8.0 * (1.0 - 0.9));
    // t < 0.5 branch.
    assert_eq!(quantile_linear(&[2.0, 10.0], 0.25), 4.0);
    assert_eq!(quantile_linear(&[7.0], 0.3), 7.0);
    assert_eq!(quantile_linear(&[1.0, 2.0], 1.0), 2.0);
    assert_eq!(quantile_linear(&[1.0, 2.0], 0.0), 1.0);
}

#[test]
fn metric_upgrade_none_when_unused_or_degenerate() {
    // No used images.
    let empty = AffineFactorization {
        cameras: vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; 3],
        translations: vec![[0.0; 2]; 3],
        points: vec![],
        residuals: vec![],
        keep: vec![],
        used_images: vec![false; 3],
    };
    assert_eq!(metric_upgrade(&empty), None);

    // Two used images give 5 constraint rows for 6 unknowns: degenerate.
    let scene = Scene::new(2, 0, 3);
    let two = AffineFactorization {
        cameras: (0..2)
            .map(|i| {
                let r = &scene.rotations[i];
                let s = scene.scales[i];
                [
                    [r[(0, 0)] * s, r[(0, 1)] * s, r[(0, 2)] * s],
                    [r[(1, 0)] * s, r[(1, 1)] * s, r[(1, 2)] * s],
                ]
            })
            .collect(),
        translations: vec![[0.0; 2]; 2],
        points: vec![],
        residuals: vec![],
        keep: vec![],
        used_images: vec![true; 2],
    };
    assert_eq!(metric_upgrade(&two), None);
}

#[test]
fn metric_upgrade_marks_unused_images_identity_and_zero_scale() {
    let scene = Scene::new(6, 40, 17);
    let (clusters, images, xy) = scene.full_observations();
    // Drop image 5 down to 3 observations so it ends up unused.
    let sel: Vec<bool> = (0..xy.len())
        .map(|o| images[o] != 5 || clusters[o] < 3)
        .collect();
    let clusters: Vec<u32> = clusters
        .iter()
        .zip(&sel)
        .filter_map(|(&c, &s)| s.then_some(c))
        .collect();
    let images: Vec<u32> = images
        .iter()
        .zip(&sel)
        .filter_map(|(&i, &s)| s.then_some(i))
        .collect();
    let xy: Vec<[f64; 2]> = xy
        .iter()
        .zip(&sel)
        .filter_map(|(&u, &s)| s.then_some(u))
        .collect();
    let fac = factorize_affine(
        &clusters,
        &images,
        &xy,
        6,
        40,
        &AffineFactorizationParams::default(),
    )
    .unwrap();
    assert!(!fac.used_images[5]);
    let hyps = metric_upgrade(&fac).unwrap();
    for hyp in &hyps {
        assert_eq!(
            hyp.rotations[5],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(hyp.scales[5], 0.0);
        assert!(
            hypothesis_matches_gt(&hyps[0], &scene, &fac.used_images)
                || hypothesis_matches_gt(&hyps[1], &scene, &fac.used_images)
        );
    }
}
