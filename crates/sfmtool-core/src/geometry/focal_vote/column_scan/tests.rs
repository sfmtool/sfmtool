// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

const W: u32 = 480;
const H: u32 = 480;
/// Planted equidistant focal: a 480 px sensor at ~195° full field of view.
const F_FISH: f64 = 140.0;
/// Planted pinhole focal on the same sensor (~50° full field of view).
const F_PIN: f64 = 520.0;

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable.
struct Lcg(u64);

impl Lcg {
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
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn ry(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
}
fn rx(a: f64) -> Matrix3<f64> {
    let (s, c) = a.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

/// Project a camera-frame point through the equidistant map `θ = r/f`, centred
/// on the principal point. `None` outside the imaged circle.
fn project_equidistant(xc: Vector3<f64>, f: f64) -> Option<[f64; 2]> {
    let rho = xc.x.hypot(xc.y);
    let th = rho.atan2(xc.z);
    if !(0.0..0.98 * std::f64::consts::PI).contains(&th) || rho < 1e-12 {
        return None;
    }
    let r = f * th;
    (r <= 0.5 * W as f64).then(|| [r * xc.x / rho, r * xc.y / rho])
}

/// Project through the pinhole map, centred on the principal point.
fn project_pinhole(xc: Vector3<f64>, f: f64) -> Option<[f64; 2]> {
    if xc.z <= 1e-3 {
        return None;
    }
    let p = [f * xc.x / xc.z, f * xc.y / xc.z];
    (p[0].abs() < 0.5 * W as f64 && p[1].abs() < 0.5 * H as f64).then_some(p)
}

/// A synthetic two-view scene: `n` correspondences between views related by
/// `r_rel`/`t_rel`, over world directions in the half-angle cone `theta`,
/// imaged through `project` at focal `focal`.
struct SceneSpec {
    r_rel: Matrix3<f64>,
    t_rel: Vector3<f64>,
    /// `None` places the points at infinity (a parallax-free pair).
    depth: Option<(f64, f64)>,
    n: usize,
    focal: f64,
    project: fn(Vector3<f64>, f64) -> Option<[f64; 2]>,
    /// Half-angle range the world directions are drawn from — correspondences
    /// must reach the periphery, where the two maps disagree, or nothing can
    /// discriminate the columns.
    theta: (f64, f64),
    noise_px: f64,
    /// Fraction of correspondences replaced by mismatches, as every real pair
    /// carries.
    outlier_frac: f64,
}

fn fisheye_rotation_scene() -> SceneSpec {
    SceneSpec {
        r_rel: rx(0.06) * ry(0.35),
        t_rel: Vector3::zeros(),
        depth: None,
        n: 300,
        focal: F_FISH,
        project: project_equidistant,
        theta: (0.35, 1.68),
        noise_px: 0.2,
        outlier_frac: 0.0,
    }
}

fn fisheye_parallax_scene() -> SceneSpec {
    SceneSpec {
        r_rel: rx(0.02) * ry(0.05),
        t_rel: Vector3::new(-1.4, 0.05, 0.1),
        depth: Some((4.0, 12.0)),
        n: 400,
        focal: F_FISH,
        project: project_equidistant,
        theta: (0.35, 1.68),
        noise_px: 0.2,
        outlier_frac: 0.0,
    }
}

fn synthetic_pair(spec: &SceneSpec, rng: &mut Lcg) -> ScanCandidate {
    let mut uv1 = Vec::new();
    let mut uv2 = Vec::new();
    let mut guard = 0;
    let r_max = 0.5 * W as f64;
    while uv1.len() < spec.n && guard < spec.n * 2000 {
        guard += 1;
        let th = rng.uniform(spec.theta.0, spec.theta.1);
        let ph = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
        let dir = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
        let d = spec.depth.map_or(1e6, |(lo, hi)| rng.uniform(lo, hi));
        let x1 = dir * d;
        let x2 = spec.r_rel * x1 + spec.t_rel;
        let (Some(mut a), Some(mut b)) = (
            (spec.project)(x1, spec.focal),
            (spec.project)(x2, spec.focal),
        ) else {
            continue;
        };
        a[0] += spec.noise_px * rng.gaussian();
        a[1] += spec.noise_px * rng.gaussian();
        if rng.next_f64() < spec.outlier_frac {
            // A mismatch: image 2 lands somewhere else entirely.
            let rr = r_max * rng.next_f64().sqrt();
            let pp = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
            b = [rr * pp.cos(), rr * pp.sin()];
        } else {
            b[0] += spec.noise_px * rng.gaussian();
            b[1] += spec.noise_px * rng.gaussian();
        }
        uv1.push(a);
        uv2.push(b);
    }
    assert_eq!(uv1.len(), spec.n, "scene generator starved");
    ScanCandidate::from_centred(0, 1, uv1, uv2, 0xC0FFEE)
}

fn max_wh() -> f64 {
    W.max(H) as f64
}
fn half_diag() -> f64 {
    0.5 * (W as f64).hypot(H as f64)
}

/// Scan one candidate pair through one column's rotation cell.
fn rotation_scan_of(model: CameraModel, cand: &ScanCandidate) -> ColumnScan {
    scan_column(
        model,
        &[],
        std::slice::from_ref(cand),
        max_wh(),
        half_diag(),
    )
}

/// Scan one candidate pair through one column's epipolar cell.
fn epipolar_scan_of(model: CameraModel, cand: &ScanCandidate) -> ColumnScan {
    scan_column(
        model,
        std::slice::from_ref(cand),
        &[],
        max_wh(),
        half_diag(),
    )
}

// ── The two fisheye cells recover a planted equidistant focal ────────────────

#[test]
fn equidistant_rotation_cell_recovers_planted_focal() {
    // A pure rotation of a 195°-FOV fisheye: under the correct focal the rays
    // are related by a rotation exactly, and no other focal explains them.
    let cand = synthetic_pair(&fisheye_rotation_scene(), &mut Lcg(11));
    let scan = rotation_scan_of(CameraModel::EquidistantFisheye, &cand);
    let v = scan.rotation.first().expect("rotation scan");
    assert!(v.certified, "{v:?}");
    assert!(
        (v.focal_px - F_FISH).abs() / F_FISH < 0.02,
        "recovered {} vs planted {F_FISH}",
        v.focal_px
    );
    assert!(!v.at_grid_edge);
    assert!(v.in_fov_band, "a 195° FOV focal must sit in the FOV window");
    assert!(v.cost <= ROTATION_FIT_FLOOR_RAD, "cost {}", v.cost);
    assert!(v.model_informative, "wide-FOV support must clear coverage");
}

#[test]
fn equidistant_epipolar_cell_recovers_planted_focal() {
    // Genuine parallax over a finite cloud: the ray-space epipolar matrix is
    // essential only at the correct focal.
    let cand = synthetic_pair(&fisheye_parallax_scene(), &mut Lcg(29));
    let scan = epipolar_scan_of(CameraModel::EquidistantFisheye, &cand);
    let v = scan.epipolar.first().expect("epipolar scan");
    assert!(v.certified, "{v:?}");
    // A single pair resolves the focal to about the grid's own step (the 64
    // log-spaced points of a 40x band sit 6% apart); the consensus over many
    // pairs is what sharpens it. Measured here: 5.5% low.
    assert!(
        (v.focal_px - F_FISH).abs() / F_FISH < 0.10,
        "recovered {} vs planted {F_FISH}",
        v.focal_px
    );
    assert!(v.cost <= ESSENTIALNESS_FLOOR, "essentialness {}", v.cost);
    assert!(v.model_informative, "wide-FOV inliers must clear coverage");
}

#[test]
fn wrong_column_contributes_no_model_informative_mass() {
    // The fisheye rotation pair read through the pinhole map. No focal makes
    // the whole ray field a rotation, so the only fit that survives lives on a
    // centre-hugging subset — exactly where the two maps agree to first order
    // and nothing can discriminate them. The radial-coverage floor keeps that
    // vote out of the model verdict, which is the asymmetry the arbitration
    // reads.
    let cand = synthetic_pair(&fisheye_rotation_scene(), &mut Lcg(11));
    let wrong = rotation_scan_of(CameraModel::Pinhole, &cand);
    let right = rotation_scan_of(CameraModel::EquidistantFisheye, &cand);
    assert_eq!(wrong.n_informative(), 0, "{:?}", wrong.rotation);
    assert_eq!(right.n_informative(), 1, "{:?}", right.rotation);
    for v in &wrong.rotation {
        assert!(
            v.coverage_p90 < COVERAGE_FLOOR,
            "the pinhole fit survives only near the centre, got coverage {}",
            v.coverage_p90
        );
        assert!(
            !v.in_fov_band,
            "and its minimum falls outside the credible half-FOV window"
        );
    }
}

#[test]
fn pinhole_input_is_the_pinhole_column_s_own_ground() {
    // The converse: a narrow-FOV pinhole capture. Both cells certify under the
    // pinhole map and recover the planted focal; the equidistant column musters
    // strictly less model-informative mass on the same pairs.
    let mut rng = Lcg(613);
    let rot = synthetic_pair(
        &SceneSpec {
            r_rel: rx(0.03) * ry(0.10),
            t_rel: Vector3::zeros(),
            depth: None,
            n: 300,
            focal: F_PIN,
            project: project_pinhole,
            theta: (0.05, 0.42),
            noise_px: 0.2,
            outlier_frac: 0.0,
        },
        &mut rng,
    );
    let epi = synthetic_pair(
        &SceneSpec {
            r_rel: rx(0.01) * ry(0.03),
            t_rel: Vector3::new(-1.4, 0.05, 0.1),
            depth: Some((4.0, 12.0)),
            n: 400,
            focal: F_PIN,
            project: project_pinhole,
            theta: (0.05, 0.42),
            noise_px: 0.2,
            outlier_frac: 0.0,
        },
        &mut rng,
    );
    let pin = scan_column(
        CameraModel::Pinhole,
        std::slice::from_ref(&epi),
        std::slice::from_ref(&rot),
        max_wh(),
        half_diag(),
    );
    let fish = scan_column(
        CameraModel::EquidistantFisheye,
        std::slice::from_ref(&epi),
        std::slice::from_ref(&rot),
        max_wh(),
        half_diag(),
    );
    for v in pin.epipolar.iter().chain(pin.rotation.iter()) {
        assert!(v.certified, "{v:?}");
        assert!(
            (v.focal_px - F_PIN).abs() / F_PIN < 0.10,
            "recovered {} vs planted {F_PIN}",
            v.focal_px
        );
    }
    assert!(
        pin.n_informative() > fish.n_informative(),
        "pinhole {} vs equidistant {} informative votes",
        pin.n_informative(),
        fish.n_informative()
    );
}

// ── Rotation domination ──────────────────────────────────────────────────────

#[test]
fn rotation_dominated_pairs_cast_no_epipolar_vote() {
    // A parallax-free pair has no baseline, so `E = [t]×R` is degenerate and
    // its essentialness minima are broad — the epipolar cell must abstain, the
    // way the pinhole family abstains on a homography-dominated pair. The
    // rotation consensus is what detects it.
    let rot_only = synthetic_pair(&fisheye_rotation_scene(), &mut Lcg(11));
    let dominated = epipolar_scan_of(CameraModel::EquidistantFisheye, &rot_only);
    let v = dominated.epipolar.first().expect("epipolar scan");
    assert!(v.rotation_dominated, "{v:?}");
    assert!(!v.certified, "a rotation-dominated pair casts no vote");
    assert!(
        v.rotation_ratio.expect("ratio") >= ROTATION_DOMINATION_FRAC,
        "ratio {:?}",
        v.rotation_ratio
    );
    assert_eq!(dominated.n_rotation_dominated(), 1);
    assert!(
        dominated.parallax_poverty() >= ROTATION_DOMINATION_FRAC,
        "poverty {}",
        dominated.parallax_poverty()
    );

    // The parallax pair of the same lens is NOT dominated and still votes, so
    // the gate is discriminating rather than muting the cell.
    let parallax = synthetic_pair(&fisheye_parallax_scene(), &mut Lcg(29));
    let voting = epipolar_scan_of(CameraModel::EquidistantFisheye, &parallax);
    let v = voting.epipolar.first().expect("epipolar scan");
    assert!(!v.rotation_dominated, "{v:?}");
    assert!(v.certified);
    assert_eq!(voting.n_rotation_dominated(), 0);
    assert!(
        voting.parallax_poverty() < ROTATION_DOMINATION_FRAC,
        "poverty {}",
        voting.parallax_poverty()
    );
}

// ── The direction certificate is not vacuous ─────────────────────────────────

#[test]
fn one_sided_residuals_differ_where_a_symmetric_one_could_not() {
    // The spec forbids a symmetric residual: the epipolar matrix of the SWAPPED
    // correspondences is exactly the transpose, with identical singular values,
    // so a symmetric consensus scores the two directions as one measurement and
    // the direction certificate is vacuous. Pinned three ways.
    let cand = synthetic_pair(
        &SceneSpec {
            r_rel: rx(0.02) * ry(0.06),
            t_rel: Vector3::new(-1.2, 0.0, 0.05),
            depth: Some((5.0, 11.0)),
            n: 200,
            focal: F_FISH,
            project: project_equidistant,
            theta: (0.35, 1.68),
            noise_px: 0.3,
            outlier_frac: 0.15,
        },
        &mut Lcg(4242),
    );
    let n = cand.uv1.len();
    let model = CameraModel::EquidistantFisheye;
    let r1: Vec<Vector3<f64>> = (0..n)
        .map(|i| model.ray(cand.uv1[i], cand.rad1[i], F_FISH))
        .collect();
    let r2: Vec<Vector3<f64>> = (0..n)
        .map(|i| model.ray(cand.uv2[i], cand.rad2[i], F_FISH))
        .collect();
    // Some epipolar matrix of the pair (its exact value is irrelevant to the
    // symmetry being pinned).
    let rows: Vec<SVector<f64, 9>> = (0..n)
        .map(|i| {
            let (a, b) = (&r2[i], &r1[i]);
            SVector::<f64, 9>::from_column_slice(&[
                a[0] * b[0],
                a[0] * b[1],
                a[0] * b[2],
                a[1] * b[0],
                a[1] * b[1],
                a[1] * b[2],
                a[2] * b[0],
                a[2] * b[1],
                a[2] * b[2],
            ])
        })
        .collect();
    let e = null_from_rows(&rows, 0..n).expect("epipolar matrix");

    let mut side2 = vec![0.0; n];
    let mut side1 = vec![0.0; n];
    epipolar_residuals(&e, &r1, &r2, true, &mut side2);
    epipolar_residuals(&e, &r1, &r2, false, &mut side1);

    // (a) The two directions genuinely measure different things: the shared
    // numerator is normalized by ‖E x₁‖ in one direction and ‖Eᵀ x₂‖ in the
    // other, so the per-point residuals — and with them the consensus set a
    // scan keeps — diverge over a wide field. Measured: 44 of 200 points
    // scored more than 10% apart, worst 1.4 in log ratio.
    let ratio: Vec<f64> = (0..n)
        .map(|i| (side2[i].max(1e-18) / side1[i].max(1e-18)).ln().abs())
        .collect();
    let spread = ratio.iter().cloned().fold(0.0, f64::max);
    assert!(
        spread > 0.5,
        "one-sided residuals must separate: max |ln(side2/side1)| = {spread}"
    );
    let differing = ratio.iter().filter(|&&r| r > 0.1).count();
    assert!(
        differing * 5 >= n,
        "at least a fifth of the points must be scored materially \
         differently, got {differing} of {n}"
    );
    // ...so the two directions certify different consensus sets: at a shared
    // bound they keep different points.
    let worst = (1..10)
        .map(|q| {
            let tol = quantile(&side2, 0.1 * q as f64);
            (0..n)
                .filter(|&i| (side2[i] < tol) != (side1[i] < tol))
                .count()
        })
        .max()
        .unwrap_or(0);
    assert!(
        worst >= 5,
        "the two directions must certify different consensus sets, \
         worst disagreement over the tolerance sweep {worst} of {n}"
    );

    // (b) ...and the swap identity that makes a SYMMETRIC residual vacuous:
    // scoring the swapped correspondences against Eᵀ reproduces the other side
    // exactly, bit for bit.
    let et = e.transpose();
    let mut swapped2 = vec![0.0; n];
    let mut swapped1 = vec![0.0; n];
    epipolar_residuals(&et, &r2, &r1, true, &mut swapped2);
    epipolar_residuals(&et, &r2, &r1, false, &mut swapped1);
    for i in 0..n {
        assert_eq!(swapped2[i].to_bits(), side1[i].to_bits(), "index {i}");
        assert_eq!(swapped1[i].to_bits(), side2[i].to_bits(), "index {i}");
        // Any residual symmetric in the two sides is therefore invariant under
        // the swap — it cannot tell the two directions apart at all.
        let sym = 0.5 * (side2[i] + side1[i]);
        let sym_swapped = 0.5 * (swapped2[i] + swapped1[i]);
        assert_eq!(sym.to_bits(), sym_swapped.to_bits(), "index {i}");
    }
    // Nor can the essentialness cost: the transpose has the same singular
    // values.
    let s = singular_values_desc(&e);
    let st = singular_values_desc(&et);
    for k in 0..3 {
        assert!((s[k] - st[k]).abs() < 1e-12, "singular value {k}");
    }
}

// ── Freezing the rotation support is load-bearing ────────────────────────────

#[test]
fn unfrozen_rotation_support_pins_at_the_grid_top() {
    // Both maps shrink every ray angle as 1/f, so a per-candidate support lets
    // a bad focal buy a low cost by keeping fewer points: the angular scan then
    // has no interior minimum and pins at the top of the grid. The frozen
    // support is what makes the same data an observation of the focal.
    let mut spec = fisheye_rotation_scene();
    // A contaminated pair, as every real one is: at a bad focal the estimator
    // can always retreat to some small clean-looking subset.
    spec.outlier_frac = 0.35;
    let cand = synthetic_pair(&spec, &mut Lcg(777));

    let model = CameraModel::EquidistantFisheye;
    let grid = scan_grid(max_wh());
    let r_hi = pair_edge_radius(&cand);
    let mut state = cand.seed;
    let samples = draw_samples(&mut state, cand.uv1.len(), ROT_SAMPLE, SCAN_SAMPLES);

    let frozen = scan_rotation(model, &cand, &grid, r_hi, &samples, true).expect("frozen scan");
    let unfrozen =
        scan_rotation(model, &cand, &grid, r_hi, &samples, false).expect("unfrozen scan");
    let f_rad = frozen.rad_minimum.expect("frozen angular curve");
    let u_rad = unfrozen.rad_minimum.expect("unfrozen angular curve");

    // Frozen: an interior minimum, on the planted focal.
    assert!(!f_rad.edge, "frozen angular minimum must be interior");
    assert!(
        (f_rad.focal_px - F_FISH).abs() / F_FISH < 0.05,
        "frozen angular minimum {} vs planted {F_FISH}",
        f_rad.focal_px
    );
    // Unfrozen: the support shrinks as the focal grows and the cost follows it
    // down, so the minimum slides to the top of the grid and observes nothing.
    assert!(
        u_rad.edge,
        "unfrozen angular minimum must pin at a grid end, got {}",
        u_rad.focal_px
    );
    assert!(
        u_rad.focal_px > 0.9 * SCAN_BAND_HI * max_wh(),
        "and specifically at the TOP, got {} (band top {})",
        u_rad.focal_px,
        SCAN_BAND_HI * max_wh()
    );

    // The pixel-scaled curve the vote reads removes the shared 1/f drift, so
    // the frozen scan still locates the focal and certifies.
    let v = rotation_scan_of(model, &cand);
    let v = v.rotation.first().copied().expect("frozen vote");
    assert!(v.certified && !v.at_grid_edge, "{v:?}");
    assert!(
        (v.focal_px - F_FISH).abs() / F_FISH < 0.05,
        "{}",
        v.focal_px
    );
}

// ── Determinism ──────────────────────────────────────────────────────────────

#[test]
fn scans_are_deterministic() {
    let mut rng = Lcg(5150);
    let rot = synthetic_pair(&fisheye_rotation_scene(), &mut rng);
    let epi = synthetic_pair(&fisheye_parallax_scene(), &mut rng);
    let run = || {
        scan_column(
            CameraModel::EquidistantFisheye,
            std::slice::from_ref(&epi),
            std::slice::from_ref(&rot),
            max_wh(),
            half_diag(),
        )
    };
    let a = run();
    let b = run();
    let pairs = a
        .epipolar
        .iter()
        .chain(a.rotation.iter())
        .zip(b.epipolar.iter().chain(b.rotation.iter()));
    let mut seen = 0;
    for (x, y) in pairs {
        assert_eq!(x.focal_px.to_bits(), y.focal_px.to_bits());
        assert_eq!(x.cost.to_bits(), y.cost.to_bits());
        assert_eq!(x.sharpness.to_bits(), y.sharpness.to_bits());
        assert_eq!(x.coverage_p90.to_bits(), y.coverage_p90.to_bits());
        assert_eq!(x.n_inliers, y.n_inliers);
        assert_eq!(x.certified, y.certified);
        assert_eq!(x.model_informative, y.model_informative);
        seen += 1;
    }
    assert_eq!(seen, 2, "both cells must have produced a scan");
}

// ── Map and helper unit checks ───────────────────────────────────────────────

#[test]
fn ray_maps_round_trip_and_scale() {
    let f = 200.0;
    for &r in &[1.0, 50.0, 180.0, 300.0] {
        let uv = [r * 0.6, r * 0.8];
        let e = CameraModel::EquidistantFisheye.ray(uv, r, f);
        assert!((e.norm() - 1.0).abs() < 1e-12);
        // θ = r/f exactly.
        assert!((e.z.acos() - r / f).abs() < 1e-9, "r {r}");
        assert!((CameraModel::EquidistantFisheye.scale(r, f) - f).abs() < 1e-12);

        let p = CameraModel::Pinhole.ray(uv, r, f);
        assert!((p.norm() - 1.0).abs() < 1e-12);
        // Pinhole dr/dθ = f (1 + (r/f)²) = f sec²θ.
        let th = (r / f).atan();
        assert!(
            (CameraModel::Pinhole.scale(r, f) - f / (th.cos() * th.cos())).abs() < 1e-9,
            "r {r}"
        );
    }
    // The equidistant map folds once θ passes π; the pinhole map never does.
    assert!(!CameraModel::EquidistantFisheye.admits(50.0, 200.0));
    assert!(CameraModel::EquidistantFisheye.admits(200.0, 200.0));
    assert!(CameraModel::Pinhole.admits(1.0, 5000.0));
}

#[test]
fn kabsch_maps_the_first_rays_onto_the_second() {
    let mut rng = Lcg(3);
    let rot = rx(0.4) * ry(-0.9);
    let r1: Vec<Vector3<f64>> = (0..40)
        .map(|_| Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalize())
        .collect();
    let r2: Vec<Vector3<f64>> = r1.iter().map(|v| rot * v).collect();
    let idx: Vec<usize> = (0..r1.len()).collect();
    let fitted = kabsch(&r1, &r2, &idx).expect("rotation");
    assert!(
        (fitted - rot).norm() < 1e-12,
        "kabsch must recover the r1 -> r2 rotation, not its inverse"
    );
}

#[test]
fn column_names_round_trip() {
    for m in [CameraModel::Pinhole, CameraModel::EquidistantFisheye] {
        assert_eq!(CameraModel::from_str_name(m.as_str()), Some(m));
    }
    assert_eq!(
        CameraModel::from_str_name("equidistant"),
        Some(CameraModel::EquidistantFisheye)
    );
    assert_eq!(
        CameraModel::from_str_name("Equidistant-Fisheye"),
        Some(CameraModel::EquidistantFisheye)
    );
    assert_eq!(CameraModel::from_str_name("brown"), None);
}
