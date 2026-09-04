// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::Vector3;

pub(crate) const W: u32 = 1000;
pub(crate) const H: u32 = 1000;
pub(crate) const F_TRUE: f64 = 800.0;
const CX: f64 = 500.0;
const CY: f64 = 500.0;

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable.
pub(crate) struct Lcg(pub(crate) u64);

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

pub(crate) struct Cam {
    r: Matrix3<f64>,
    t: Vector3<f64>,
}

impl Cam {
    /// Project a world point to pixels through focal `f` (the rest of `K` is
    /// the image centre); `None` when behind the camera or out of the image.
    fn project_f(&self, x: Vector3<f64>, f: f64) -> Option<[f64; 2]> {
        let xc = self.r * x + self.t;
        if xc.z <= 1e-3 {
            return None;
        }
        let k = Matrix3::new(f, 0.0, CX, 0.0, f, CY, 0.0, 0.0, 1.0);
        let p = k * xc;
        let u = p.x / p.z;
        let v = p.y / p.z;
        if !(0.0..W as f64).contains(&u) || !(0.0..H as f64).contains(&v) {
            return None;
        }
        Some([u, v])
    }
}

impl Cam {
    /// Project a world point through the equidistant fisheye map `θ = r/f`;
    /// `None` outside the imaged circle or the image rectangle.
    fn project_fisheye(&self, x: Vector3<f64>, f: f64) -> Option<[f64; 2]> {
        let xc = self.r * x + self.t;
        let rho = xc.x.hypot(xc.y);
        if rho < 1e-12 {
            return None;
        }
        let th = rho.atan2(xc.z);
        if !(0.0..0.98 * std::f64::consts::PI).contains(&th) {
            return None;
        }
        let r = f * th;
        let u = CX + r * xc.x / rho;
        let v = CY + r * xc.y / rho;
        if !(0.0..W as f64).contains(&u) || !(0.0..H as f64).contains(&v) {
            return None;
        }
        Some([u, v])
    }
}

/// Accumulating builder for flat observation arrays (one span-2 cluster per
/// emitted correspondence).
#[derive(Default)]
pub(crate) struct Obs {
    pub(crate) cluster: Vec<u32>,
    pub(crate) image: Vec<u32>,
    pub(crate) pos: Vec<[f64; 2]>,
    next: u32,
}

impl Obs {
    fn push_pair(&mut self, ia: u32, pa: [f64; 2], ib: u32, pb: [f64; 2]) {
        let c = self.next;
        self.next += 1;
        self.cluster.push(c);
        self.image.push(ia);
        self.pos.push(pa);
        self.cluster.push(c);
        self.image.push(ib);
        self.pos.push(pb);
    }
    fn run(&self, seed: u64) -> FocalVoteResult {
        focal_vote(&self.cluster, &self.image, &self.pos, W, H, seed)
    }
    fn run_columns(&self, seed: u64, columns: &[CameraModel]) -> FocalVoteResult {
        focal_vote_with_options(
            &self.cluster,
            &self.image,
            &self.pos,
            W,
            H,
            &FocalVoteOptions {
                seed,
                columns: columns.to_vec(),
                ..Default::default()
            },
        )
    }
}

/// Pure-rotation rig (all camera centres at the world origin): `n_img` views
/// panned across `±span` radians with a small per-view tilt.
fn rotation_cameras(n_img: usize, span: f64, rng: &mut Lcg) -> Vec<Cam> {
    (0..n_img)
        .map(|i| {
            let pan = -span + 2.0 * span * (i as f64) / ((n_img - 1) as f64);
            let tilt = rng.uniform(-0.02, 0.02);
            Cam {
                r: rx(tilt) * ry(pan),
                t: Vector3::zeros(),
            }
        })
        .collect()
}

/// Emit `m` span-2 clusters between cameras `ia`,`ib`, sampling world
/// directions (rotation rig, points at infinity) visible in both.
fn emit_rotation_pair(obs: &mut Obs, cams: &[Cam], ia: usize, ib: usize, m: usize, rng: &mut Lcg) {
    emit_rotation_pair_f(obs, cams, ia, ib, m, rng, F_TRUE);
}

/// [`emit_rotation_pair`] through an explicit shared focal.
fn emit_rotation_pair_f(
    obs: &mut Obs,
    cams: &[Cam],
    ia: usize,
    ib: usize,
    m: usize,
    rng: &mut Lcg,
    f: f64,
) {
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 200 {
        guard += 1;
        let yaw = rng.uniform(-0.9, 0.9);
        let pitch = rng.uniform(-0.6, 0.6);
        let dir = Vector3::new(yaw.sin(), pitch.sin(), 1.0).normalize() * 30.0;
        if let (Some(mut pa), Some(mut pb)) =
            (cams[ia].project_f(dir, f), cams[ib].project_f(dir, f))
        {
            pa[0] += 0.3 * rng.gaussian();
            pa[1] += 0.3 * rng.gaussian();
            pb[0] += 0.3 * rng.gaussian();
            pb[1] += 0.3 * rng.gaussian();
            obs.push_pair(ia as u32, pa, ib as u32, pb);
            done += 1;
        }
    }
}

/// Baseline cameras along `+X`, all looking roughly `+Z`, for a parallax scene.
pub(crate) fn baseline_cameras(n_img: usize, baseline: f64, rng: &mut Lcg) -> Vec<Cam> {
    (0..n_img)
        .map(|i| {
            let r = rx(rng.uniform(-0.03, 0.03)) * ry(rng.uniform(-0.03, 0.03));
            let center = Vector3::new(i as f64 * baseline, 0.0, 0.0);
            Cam { r, t: -r * center }
        })
        .collect()
}

/// Emit `m` span-2 clusters between baseline cameras `ia`,`ib`, sampling finite
/// 3D points visible in both.
fn emit_parallax_pair(obs: &mut Obs, cams: &[Cam], ia: usize, ib: usize, m: usize, rng: &mut Lcg) {
    emit_parallax_pair_f(obs, cams, ia, ib, m, rng, F_TRUE);
}

/// [`emit_parallax_pair`] through an explicit shared focal.
fn emit_parallax_pair_f(
    obs: &mut Obs,
    cams: &[Cam],
    ia: usize,
    ib: usize,
    m: usize,
    rng: &mut Lcg,
    f: f64,
) {
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 200 {
        guard += 1;
        let x = Vector3::new(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(4.0, 9.0),
        );
        if let (Some(mut pa), Some(mut pb)) = (cams[ia].project_f(x, f), cams[ib].project_f(x, f)) {
            pa[0] += 0.3 * rng.gaussian();
            pa[1] += 0.3 * rng.gaussian();
            pb[0] += 0.3 * rng.gaussian();
            pb[1] += 0.3 * rng.gaussian();
            obs.push_pair(ia as u32, pa, ib as u32, pb);
            done += 1;
        }
    }
}

/// Two disjoint sub-captures in one observation set: images `0..rot_n` are a
/// pure-rotation rig imaged at `f_rot` (far field, no parallax), images
/// `rot_n..rot_n + bl_n` a baseline track over finite structure imaged at
/// `f_bl`. No cluster spans the two sub-captures, so each family votes from
/// its own — the scene is how both families come to vote at once.
pub(crate) fn two_subcapture_scene(
    rot_n: usize,
    bl_n: usize,
    f_rot: f64,
    f_bl: f64,
    seed: u64,
) -> Obs {
    let mut rng = Lcg(seed);
    let mut cams = rotation_cameras(rot_n, 0.24, &mut rng);
    cams.extend(baseline_cameras(bl_n, 0.35, &mut rng));
    let mut obs = Obs::default();
    for i in 0..rot_n - 1 {
        emit_rotation_pair_f(&mut obs, &cams, i, i + 1, 45, &mut rng, f_rot);
    }
    for i in 0..rot_n.saturating_sub(3) {
        emit_rotation_pair_f(&mut obs, &cams, i, i + 3, 45, &mut rng, f_rot);
    }
    let b = rot_n;
    for i in 0..bl_n - 1 {
        emit_parallax_pair_f(&mut obs, &cams, b + i, b + i + 1, 45, &mut rng, f_bl);
    }
    for i in 0..bl_n.saturating_sub(2) {
        emit_parallax_pair_f(&mut obs, &cams, b + i, b + i + 2, 45, &mut rng, f_bl);
    }
    obs
}

/// Distinct unordered image pairs in the rotation detail list.
fn distinct_rotation_pairs(res: &FocalVoteResult) -> usize {
    let mut pairs: Vec<(u32, u32)> = res
        .rotation_votes
        .iter()
        .map(|v| (v.image.min(v.partner), v.image.max(v.partner)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs.len()
}

/// Rebuild the pooled epipolar pair votes from the directional detail list —
/// the geometric mean of each pair's two in-band directions. Valid only when
/// every candidate pair contributed both directions
/// (`epipolar_votes.len() == 2 * n_epipolar`).
fn epipolar_pair_votes(res: &FocalVoteResult) -> Vec<f64> {
    assert_eq!(res.epipolar_votes.len(), 2 * res.n_epipolar);
    let mut by_pair: Vec<((u32, u32), f64)> = Vec::new();
    for v in &res.epipolar_votes {
        let key = (v.image_a, v.image_b);
        match by_pair.iter_mut().find(|(k, _)| *k == key) {
            Some((_, prod)) => *prod *= v.focal_px,
            None => by_pair.push((key, v.focal_px)),
        }
    }
    by_pair.into_iter().map(|(_, prod)| prod.sqrt()).collect()
}

/// The log-space median the whole pool WOULD produce if the two families were
/// blended — what the family-disagreement rule exists to avoid.
fn naive_blended_median(res: &FocalVoteResult) -> f64 {
    let mut pool = epipolar_pair_votes(res);
    pool.extend(res.rotation_votes.iter().map(|v| v.focal_px));
    log_median(&pool).expect("non-empty pool")
}

// ── Rotation self-calibration unit tests ─────────────────────────────────────

#[test]
fn pure_rotation_recovers_focal() {
    let k = Matrix3::new(F_TRUE, 0.0, 0.0, 0.0, F_TRUE, 0.0, 0.0, 0.0, 1.0);
    let kinv = k.try_inverse().unwrap();
    let max_wh = W.max(H) as f64;
    for &(pan, tilt) in &[(0.20, 0.10), (0.30, -0.05), (-0.25, 0.15)] {
        let h = k * (ry(pan) * rx(tilt)) * kinv;
        let f = rotation_self_calib_focal(&h, max_wh).expect("observable rotation");
        assert!(
            (f - F_TRUE).abs() / F_TRUE < 0.02,
            "pan {pan}: recovered {f}, true {F_TRUE}"
        );
    }
}

#[test]
fn finite_plane_homography_rejected() {
    // H = K (R - t nᵀ/d) K⁻¹ with a real baseline over a finite plane carries a
    // translation term and never gets orthogonal.
    let k = Matrix3::new(F_TRUE, 0.0, 0.0, 0.0, F_TRUE, 0.0, 0.0, 0.0, 1.0);
    let kinv = k.try_inverse().unwrap();
    let max_wh = W.max(H) as f64;
    let r = ry(0.15) * rx(0.05);
    let t = Vector3::new(1.0, 0.2, 0.1);
    let n = Vector3::new(0.0, 0.0, 1.0);
    let d = 2.0;
    let h = k * (r - t * n.transpose() / d) * kinv;
    assert!(
        rotation_self_calib_focal(&h, max_wh).is_none(),
        "finite-plane homography should be rejected by the residual floor"
    );
}

#[test]
fn roll_only_is_flat_in_focal() {
    // Roll about the optical axis is conjugate to a rotation for *every* f
    // (K = diag(f, f, 1) commutes with a 2D rotation), so H = Rz and the
    // orthogonality residual is flat in f: the scan cannot observe the focal.
    let (s, c) = 0.35f64.sin_cos(); // roll Rz(~20°)
    let h_roll = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    let max_wh = W.max(H) as f64;
    let lo = ortho_cost(&h_roll, 0.4 * max_wh);
    let hi = ortho_cost(&h_roll, 2.5 * max_wh);
    assert!(
        (lo - hi).abs() < 1e-9,
        "roll cost should be flat in f: {lo} vs {hi}"
    );
    // Because f is unobservable, the scan can never recover the true focal from
    // a roll — whatever it returns is an arbitrary boundary value, far from the
    // truth, so a roll never contributes a valid focal to the consensus.
    if let Some(f) = rotation_self_calib_focal(&h_roll, max_wh) {
        assert!(
            (f - F_TRUE).abs() / F_TRUE > 0.3,
            "roll must not recover the true focal, got {f}"
        );
    }
}

// ── End-to-end consensus ─────────────────────────────────────────────────────

#[test]
fn rotation_scene_pools_a_rotation_majority() {
    let mut rng = Lcg(2024);
    let n = 8;
    let cams = rotation_cameras(n, 0.24, &mut rng); // ±13.7°
    let mut obs = Obs::default();
    for i in 0..n - 1 {
        emit_rotation_pair(&mut obs, &cams, i, i + 1, 45, &mut rng); // near (epipolar-candidate)
    }
    for i in 0..n - 3 {
        emit_rotation_pair(&mut obs, &cams, i, i + 3, 45, &mut rng); // far (rotation partner)
    }
    let res = obs.run(0);
    // Parallax-free scene: every epipolar candidate is homography-dominated, so
    // the pool is all rotation and the majority family is Rotation. The scan
    // visits 8 images but four of them are their widest partner's widest
    // partner, so only 5 distinct pairs vote.
    assert_eq!(res.n_epipolar, 0);
    assert_eq!(res.n_rotation, 5);
    assert_eq!(distinct_rotation_pairs(&res), res.n_rotation);
    assert_eq!(res.n_pool, res.n_epipolar + res.n_rotation);
    assert_eq!(res.n_h_dominated, 8);
    assert_eq!(res.family_disagreement, None, "no epipolar votes");
    assert_eq!(
        res.family,
        Some(VoteFamily::Rotation),
        "n_epi {}, n_rot {}, poverty {:.2}, epi {:?}, rot {:?}",
        res.n_epipolar,
        res.n_rotation,
        res.parallax_poverty,
        res.epipolar_focal_px,
        res.rotation_focal_px
    );
    // The pool is exactly the rotation votes here, so the consensus is their
    // log-space median: 804.66 px, 0.6% above the true 800.
    assert_eq!(res.focal_px, res.rotation_focal_px);
    assert_eq!(res.pool_spread, res.rotation_spread);
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_TRUE).abs() / F_TRUE < 0.01,
        "rotation focal {f}, true {F_TRUE}"
    );
    // Every candidate pair's correspondences are explained by a homography.
    assert!(
        res.parallax_poverty >= 0.9,
        "poverty {} — expected a parallax-free regime",
        res.parallax_poverty
    );
}

#[test]
fn parallax_scene_pools_an_epipolar_majority() {
    let mut rng = Lcg(4048);
    let n = 8;
    let cams = baseline_cameras(n, 0.35, &mut rng);
    let mut obs = Obs::default();
    for i in 0..n - 1 {
        emit_parallax_pair(&mut obs, &cams, i, i + 1, 45, &mut rng);
    }
    for i in 0..n - 2 {
        emit_parallax_pair(&mut obs, &cams, i, i + 2, 45, &mut rng);
    }
    let res = obs.run(0);
    // 7 candidate pairs survive the homography gate and all 7 are
    // direction-consistent: 14 in-band directional focals in the detail list
    // pool as 7 pair votes. No pair displacement reaches the rotation gate.
    assert_eq!(res.epipolar_votes.len(), 14);
    assert_eq!(res.n_epipolar, 7);
    assert_eq!(res.n_inconsistent_pairs, 0);
    assert_eq!(res.n_rotation, 0);
    assert_eq!(res.n_pool, 7);
    assert_eq!(
        res.family,
        Some(VoteFamily::Epipolar),
        "n_epi {}, n_rot {}, poverty {:.2}, epi {:?}, rot {:?}",
        res.n_epipolar,
        res.n_rotation,
        res.parallax_poverty,
        res.epipolar_focal_px,
        res.rotation_focal_px
    );
    // The pool is exactly the epipolar pair votes, so the consensus is their
    // log-space median (odd count: the middle vote).
    assert_eq!(res.focal_px, res.epipolar_focal_px);
    assert_eq!(res.pool_spread, res.epipolar_spread);
    assert_eq!(res.family_disagreement, None, "no rotation votes");
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_TRUE).abs() / F_TRUE < 0.03,
        "epipolar focal {f}, true {F_TRUE}"
    );
    assert!(
        res.parallax_poverty <= 0.4,
        "poverty {} — expected a parallax-rich regime",
        res.parallax_poverty
    );
}

/// One baseline pair over a finite point cloud, imaged through two genuinely
/// different focals — each direction of `F` then reports its own camera's
/// focal, so the pair's directional disagreement is tunable by `f_b`.
fn two_focal_pair_scene(f_a: f64, f_b: f64) -> Obs {
    let mut rng = Lcg(31337);
    let cams = baseline_cameras(2, 1.2, &mut rng);
    let mut obs = Obs::default();
    let (mut done, mut guard) = (0, 0);
    while done < 60 && guard < 20000 {
        guard += 1;
        let x = Vector3::new(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(4.0, 9.0),
        );
        if let (Some(pa), Some(pb)) = (cams[0].project_f(x, f_a), cams[1].project_f(x, f_b)) {
            obs.push_pair(0, pa, 1, pb);
            done += 1;
        }
    }
    obs
}

/// The pair's directional disagreement `|ln(f_F / f_Fᵀ)|`, from the detail list.
fn direction_disagreement(res: &FocalVoteResult) -> f64 {
    assert_eq!(res.epipolar_votes.len(), 2, "{:?}", res.epipolar_votes);
    (res.epipolar_votes[0].focal_px.ln() - res.epipolar_votes[1].focal_px.ln()).abs()
}

#[test]
fn direction_disagreement_casts_no_vote() {
    // Two cameras with genuinely DIFFERENT focals over a finite point cloud.
    // Each direction of F reports its own camera's focal, so the pair's two
    // directional Bougnoux focals are far apart (ln ratio ~0.49) even though
    // both are inside the plausibility band. The pair is therefore not a
    // consistent measurement of one shared focal and must cast no vote.
    let res = two_focal_pair_scene(800.0, 1300.0).run(0);
    // Both directions are in band and land near their own camera's focal.
    assert_eq!(res.n_band_rejected, 0);
    assert_eq!(res.n_degenerate, 0);
    let d = direction_disagreement(&res);
    assert!(d > 0.4, "directional disagreement {d}"); // measured 0.4903
                                                      // ... so the pair casts no pooled vote and is counted as inconsistent.
    assert_eq!(res.n_epipolar, 0);
    assert_eq!(res.n_inconsistent_pairs, 1);
    assert_eq!(res.n_pool, 0);
    assert_eq!(res.focal_px, None);
    assert_eq!(res.family, None);
}

#[test]
fn direction_agreement_band_is_pinned() {
    // Pins the band by a literal, not by the constant: a pair whose measured
    // directional disagreement lies strictly between the band (0.05) and twice
    // the band (0.10) must cast no vote. A band loosened by any factor >= 1.45
    // would admit this pair and fail the test.
    //
    // 800 px against 860 px: measured disagreement 0.0723.
    let res = two_focal_pair_scene(800.0, 860.0).run(0);
    let d = direction_disagreement(&res);
    assert!(
        d > 0.05 && d <= 0.10,
        "disagreement {d} must sit between the band and twice the band"
    );
    assert_eq!(res.n_band_rejected, 0);
    assert_eq!(res.n_epipolar, 0, "pair should cast no vote at {d}");
    assert_eq!(res.n_inconsistent_pairs, 1);
    assert_eq!(res.focal_px, None);

    // Just inside the band the same construction DOES vote, so the test above
    // is pinning the threshold and not a permanently mute fixture.
    let res = two_focal_pair_scene(800.0, 840.0).run(0);
    let d = direction_disagreement(&res);
    assert!(d < 0.05, "disagreement {d} should be inside the band");
    assert_eq!(res.n_epipolar, 1);
    assert_eq!(res.n_inconsistent_pairs, 0);
}

#[test]
fn mixed_scene_pools_both_families() {
    // A pure-rotation sub-capture (images 0..4) and a baseline sub-capture over
    // finite structure (images 4..12), both imaged at the SAME focal. Each
    // family votes from the sub-capture its estimator can observe, and the two
    // populations pool into one median.
    let res = two_subcapture_scene(4, 8, F_TRUE, F_TRUE, 1234).run(0);
    assert!(res.n_epipolar > 0 && res.n_rotation > 0, "{res:?}");
    assert_eq!((res.n_epipolar, res.n_rotation, res.n_pool), (7, 2, 9));
    assert_eq!(res.family, Some(VoteFamily::Epipolar));
    // Both families land on the true focal, so they do not trip the
    // family-disagreement rule (measured gap 0.0055 in log-focal).
    let d = res.family_disagreement.expect("both families voted");
    assert!(d < 0.05, "family disagreement {d} — expected agreement");
    // The consensus is the median of the POOL: 803.69 px, 0.5% above the true
    // 800, and strictly between the two family medians (802.66 and 807.11) —
    // it is neither family's median.
    let f = res.focal_px.expect("consensus focal");
    let (ep, rt) = (
        res.epipolar_focal_px.unwrap(),
        res.rotation_focal_px.unwrap(),
    );
    assert!(rt < f && f < ep, "pooled {f} not between {rt} and {ep}");
    assert!((f - F_TRUE).abs() / F_TRUE < 0.006, "pooled focal {f}");
    // Tight pool: measured log-IQR 0.017.
    assert!(res.pool_spread < 0.05, "pool spread {}", res.pool_spread);
}

#[test]
fn bimodal_families_take_the_majority_median() {
    // Rotation sub-capture at 800 px, baseline sub-capture at 1300 px: the two
    // families genuinely measure different focals, so the pool is bimodal and
    // its blend would report a focal no pair voted for.
    //
    // (a) A strict rotation majority (5 rotation votes vs 3 epipolar).
    let res = two_subcapture_scene(8, 4, 800.0, 1300.0, 1234).run(0);
    assert_eq!((res.n_epipolar, res.n_rotation), (3, 5));
    let d = res.family_disagreement.expect("both families voted");
    assert!(d > 0.25, "family disagreement {d} should exceed the band");
    assert!((d - 0.4785).abs() < 0.01, "family disagreement {d}"); // ln(1294/802)
    assert_eq!(res.family, Some(VoteFamily::Rotation));
    // The consensus is EXACTLY the majority family's median, not a blend.
    assert_eq!(res.focal_px, res.rotation_focal_px);
    let f = res.focal_px.expect("consensus focal");
    assert!((f - 800.0).abs() / 800.0 < 0.005, "majority focal {f}");
    // pool_spread describes the majority family alone (measured 0.0014), not
    // the bimodal pool (whose log-IQR would be ~0.48).
    assert!(res.pool_spread < 0.01, "pool spread {}", res.pool_spread);

    // (b) Five votes per family: here the blended median would be the geometric
    // mean of the two modes' facing votes — a focal no pair voted for. The rule
    // returns the majority family's median instead (ties go to Rotation).
    let res = two_subcapture_scene(8, 6, 800.0, 1300.0, 1234).run(0);
    assert_eq!((res.n_epipolar, res.n_rotation), (5, 5));
    let d = res.family_disagreement.expect("both families voted");
    assert!(d > 0.25, "family disagreement {d}");
    // Measured 981.08 px — squarely between the modes.
    let blend = naive_blended_median(&res);
    assert!(
        blend > 950.0 && blend < 1020.0,
        "blended median {blend} should sit between the 800 and 1300 modes"
    );
    assert_eq!(res.family, Some(VoteFamily::Rotation));
    assert_eq!(res.focal_px, res.rotation_focal_px);
    let f = res.focal_px.expect("consensus focal");
    assert!((f - 800.0).abs() / 800.0 < 0.006, "majority focal {f}");
}

#[test]
fn family_tie_goes_to_rotation() {
    // Equal vote counts (3 and 3) from two sub-captures at the SAME focal: the
    // tie resolves to Rotation, and because the families agree the consensus is
    // still the pooled median — 805.42 px, which is neither family's median.
    let res = two_subcapture_scene(5, 4, F_TRUE, F_TRUE, 1234).run(0);
    assert_eq!((res.n_epipolar, res.n_rotation), (3, 3));
    assert_eq!(res.family, Some(VoteFamily::Rotation));
    let d = res.family_disagreement.expect("both families voted");
    assert!(d < 0.25, "family disagreement {d} — expected agreement");
    assert_ne!(res.focal_px, res.rotation_focal_px);
    let f = res.focal_px.expect("consensus focal");
    assert!((f - F_TRUE).abs() / F_TRUE < 0.01, "pooled focal {f}");

    // The same tie under disagreement instead takes the Rotation median.
    let res = two_subcapture_scene(5, 4, 800.0, 1300.0, 1234).run(0);
    assert_eq!((res.n_epipolar, res.n_rotation), (3, 3));
    assert!(res.family_disagreement.expect("both voted") > 0.25);
    assert_eq!(res.family, Some(VoteFamily::Rotation));
    assert_eq!(res.focal_px, res.rotation_focal_px);
}

#[test]
fn mutual_widest_partners_vote_once() {
    // A two-image rotation capture: image 0's widest partner is 1 and image 1's
    // is 0, so the scan reaches the pair twice. It votes once — and a pool of
    // one is below the 2-vote floor, so there is no consensus.
    let mut rng = Lcg(555);
    let cams = rotation_cameras(2, 0.10, &mut rng);
    let mut obs = Obs::default();
    emit_rotation_pair(&mut obs, &cams, 0, 1, 45, &mut rng);
    let res = obs.run(0);
    assert_eq!(res.n_rotation, 1, "{:?}", res.rotation_votes);
    assert_eq!(res.rotation_votes.len(), 1);
    assert_eq!(res.n_epipolar, 0);
    assert_eq!(res.n_pool, 1);
    assert_eq!(res.focal_px, None);
    assert_eq!(res.family, None);
    assert_eq!(res.family_disagreement, None);
    assert_eq!(res.pool_spread, 0.0);
    // The one vote is still visible as a diagnostic: 803.77 px.
    let f = res.rotation_focal_px.expect("one rotation vote");
    assert!((f - F_TRUE).abs() / F_TRUE < 0.01, "rotation focal {f}");
}

#[test]
fn determinism_same_seed() {
    let mut rng = Lcg(7);
    let n = 8;
    let cams = rotation_cameras(n, 0.24, &mut rng);
    let mut obs = Obs::default();
    for i in 0..n - 1 {
        emit_rotation_pair(&mut obs, &cams, i, i + 1, 40, &mut rng);
    }
    for i in 0..n - 3 {
        emit_rotation_pair(&mut obs, &cams, i, i + 3, 40, &mut rng);
    }
    let a = obs.run(42);
    let b = obs.run(42);
    assert_eq!(a.focal_px.map(f64::to_bits), b.focal_px.map(f64::to_bits));
    assert_eq!(a.family, b.family);
    assert_eq!(a.n_epipolar, b.n_epipolar);
    assert_eq!(a.n_rotation, b.n_rotation);
    assert_eq!(a.n_pool, b.n_pool);
    assert_eq!(a.n_inconsistent_pairs, b.n_inconsistent_pairs);
    assert_eq!(a.n_degenerate, b.n_degenerate);
    assert_eq!(a.parallax_poverty.to_bits(), b.parallax_poverty.to_bits());
    assert_eq!(a.pool_spread.to_bits(), b.pool_spread.to_bits());
    assert_eq!(
        a.family_disagreement.map(f64::to_bits),
        b.family_disagreement.map(f64::to_bits)
    );
}

// ── Camera-model columns ─────────────────────────────────────────────────────

/// Planted equidistant focal on the 1000 px test sensor: `θ = r/f` puts the
/// image corner at 89°, a ~179° field of view.
pub(crate) const F_FISH: f64 = 320.0;

/// Emit `m` span-2 clusters between cameras `ia`,`ib` of a pure-rotation rig,
/// imaged through the equidistant fisheye map. Directions are drawn over a wide
/// cone so the correspondences reach the periphery, where the two camera models
/// disagree.
fn emit_fisheye_rotation_pair(
    obs: &mut Obs,
    cams: &[Cam],
    ia: usize,
    ib: usize,
    m: usize,
    rng: &mut Lcg,
) {
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 2000 {
        guard += 1;
        let th = rng.uniform(0.25, 1.45);
        let ph = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
        let dir = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()) * 30.0;
        if let (Some(mut pa), Some(mut pb)) = (
            cams[ia].project_fisheye(dir, F_FISH),
            cams[ib].project_fisheye(dir, F_FISH),
        ) {
            pa[0] += 0.2 * rng.gaussian();
            pa[1] += 0.2 * rng.gaussian();
            pb[0] += 0.2 * rng.gaussian();
            pb[1] += 0.2 * rng.gaussian();
            obs.push_pair(ia as u32, pa, ib as u32, pb);
            done += 1;
        }
    }
}

/// [`emit_fisheye_rotation_pair`] over finite structure instead: a baseline
/// pair with genuine parallax, the epipolar cell's own ground.
pub(crate) fn emit_fisheye_parallax_pair(
    obs: &mut Obs,
    cams: &[Cam],
    ia: usize,
    ib: usize,
    m: usize,
    rng: &mut Lcg,
) {
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 2000 {
        guard += 1;
        let th = rng.uniform(0.25, 1.45);
        let ph = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
        let d = rng.uniform(5.0, 12.0);
        let x = Vector3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()) * d;
        if let (Some(mut pa), Some(mut pb)) = (
            cams[ia].project_fisheye(x, F_FISH),
            cams[ib].project_fisheye(x, F_FISH),
        ) {
            pa[0] += 0.2 * rng.gaussian();
            pa[1] += 0.2 * rng.gaussian();
            pb[0] += 0.2 * rng.gaussian();
            pb[1] += 0.2 * rng.gaussian();
            obs.push_pair(ia as u32, pa, ib as u32, pb);
            done += 1;
        }
    }
}

/// A fisheye capture imaged at [`F_FISH`], in two sub-captures so that each
/// cell has its own ground, exactly as [`two_subcapture_scene`] does for the
/// pinhole kernel: images `0..8` are a pure-rotation rig panning across
/// ±1.4 rad (far field), images `8..16` a baseline track over finite structure.
pub(crate) fn fisheye_scene(seed: u64) -> Obs {
    let mut rng = Lcg(seed);
    let (rot_n, bl_n) = (8usize, 8usize);
    let mut cams = rotation_cameras(rot_n, 1.4, &mut rng);
    cams.extend(baseline_cameras(bl_n, 1.0, &mut rng));
    let mut obs = Obs::default();
    for i in 0..rot_n - 1 {
        emit_fisheye_rotation_pair(&mut obs, &cams, i, i + 1, 60, &mut rng);
    }
    for i in 0..rot_n - 2 {
        emit_fisheye_rotation_pair(&mut obs, &cams, i, i + 2, 60, &mut rng);
    }
    for i in 0..bl_n - 1 {
        emit_fisheye_parallax_pair(&mut obs, &cams, rot_n + i, rot_n + i + 1, 60, &mut rng);
    }
    for i in 0..bl_n - 2 {
        emit_fisheye_parallax_pair(&mut obs, &cams, rot_n + i, rot_n + i + 2, 60, &mut rng);
    }
    obs
}

#[test]
fn default_column_set_is_pinhole_only_and_bit_identical() {
    // The multi-column code path is present, but the default column set is
    // pinhole-only and reproduces the closed-form kernel bit for bit — no scan
    // runs and no column diagnostics are produced.
    let obs = two_subcapture_scene(5, 6, F_TRUE, 1300.0, 1234);
    let legacy = obs.run(7);
    let explicit = obs.run_columns(7, &[CameraModel::Pinhole]);
    assert_eq!(
        legacy.focal_px.map(f64::to_bits),
        explicit.focal_px.map(f64::to_bits)
    );
    assert_eq!(legacy.family, explicit.family);
    assert_eq!(
        legacy.epipolar_focal_px.map(f64::to_bits),
        explicit.epipolar_focal_px.map(f64::to_bits)
    );
    assert_eq!(
        legacy.rotation_focal_px.map(f64::to_bits),
        explicit.rotation_focal_px.map(f64::to_bits)
    );
    assert_eq!(
        (legacy.n_epipolar, legacy.n_rotation, legacy.n_pool),
        (explicit.n_epipolar, explicit.n_rotation, explicit.n_pool)
    );
    assert_eq!(legacy.pool_spread.to_bits(), explicit.pool_spread.to_bits());
    assert_eq!(
        legacy.parallax_poverty.to_bits(),
        explicit.parallax_poverty.to_bits()
    );
    assert_eq!(
        legacy.family_disagreement.map(f64::to_bits),
        explicit.family_disagreement.map(f64::to_bits)
    );
    assert_eq!(legacy.epipolar_votes.len(), explicit.epipolar_votes.len());
    assert_eq!(legacy.rotation_votes.len(), explicit.rotation_votes.len());
    // A single requested column has nothing to arbitrate: the verdict is that
    // column, and there is no scan to report.
    assert_eq!(legacy.camera_model, Some(CameraModel::Pinhole));
    assert!(legacy.columns.is_empty());

    // ...and when BOTH columns run and pinhole wins, the top-level focal is
    // still exactly the closed-form answer — column focals are never blended.
    let both = obs.run_columns(7, &[CameraModel::Pinhole, CameraModel::EquidistantFisheye]);
    assert_eq!(both.camera_model, Some(CameraModel::Pinhole));
    assert_eq!(
        both.focal_px.map(f64::to_bits),
        legacy.focal_px.map(f64::to_bits)
    );
    assert_eq!(both.family, legacy.family);
    assert_eq!(both.n_pool, legacy.n_pool);
    assert_eq!(both.columns.len(), 2);
}

#[test]
fn pinhole_capture_is_arbitrated_pinhole() {
    let res = two_subcapture_scene(6, 8, F_TRUE, F_TRUE, 99)
        .run_columns(0, &[CameraModel::Pinhole, CameraModel::EquidistantFisheye]);
    assert_eq!(
        res.camera_model,
        Some(CameraModel::Pinhole),
        "{:?}",
        res.columns
    );
    let pin = &res.columns[0];
    let fish = &res.columns[1];
    assert_eq!(pin.model, CameraModel::Pinhole);
    assert_eq!(fish.model, CameraModel::EquidistantFisheye);
    assert!(
        pin.n_informative > fish.n_informative,
        "pinhole {} vs equidistant {}",
        pin.n_informative,
        fish.n_informative
    );
    // The reported focal is the winning column's closed-form consensus.
    let f = res.focal_px.expect("consensus focal");
    assert!((f - F_TRUE).abs() / F_TRUE < 0.02, "focal {f}");
    // The losing column's own median survives as a diagnostic, and it is NOT
    // the reported focal — the two parameterize different maps.
    assert_ne!(
        fish.focal_px.map(f64::to_bits),
        res.focal_px.map(f64::to_bits)
    );
}

#[test]
fn fisheye_capture_is_arbitrated_equidistant() {
    let res = fisheye_scene(2718)
        .run_columns(0, &[CameraModel::Pinhole, CameraModel::EquidistantFisheye]);
    assert_eq!(
        res.camera_model,
        Some(CameraModel::EquidistantFisheye),
        "{:?}",
        res.columns
    );
    let pin = &res.columns[0];
    let fish = &res.columns[1];
    // The pinhole column's epipolar cell still fits *something* to every pair,
    // so the margin is structurally thin on a synthetic rig; the rotation cell
    // separates cleanly, which is where the model evidence lives.
    assert!(
        fish.n_informative > pin.n_informative,
        "equidistant {} (epi {}, rot {}) vs pinhole {} (epi {}, rot {})",
        fish.n_informative,
        fish.n_informative_epipolar,
        fish.n_informative_rotation,
        pin.n_informative,
        pin.n_informative_epipolar,
        pin.n_informative_rotation
    );
    assert!(
        fish.n_informative_rotation > pin.n_informative_rotation,
        "the rotation cell is where the models separate: equidistant {} vs \
         pinhole {}",
        fish.n_informative_rotation,
        pin.n_informative_rotation
    );
    // The top level now reports the EQUIDISTANT column's consensus, which is
    // the planted equidistant focal — not the pinhole column's answer.
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_FISH).abs() / F_FISH < 0.05,
        "recovered {f} vs planted {F_FISH}"
    );
    assert_eq!(
        res.focal_px.map(f64::to_bits),
        fish.focal_px.map(f64::to_bits)
    );
    assert_eq!(
        (res.n_epipolar, res.n_rotation),
        (fish.n_epipolar, fish.n_rotation)
    );
    // The losing column's diagnostics survive: it was offered the same
    // candidate pairs, and its certificate counts are still there to read —
    // including the fact that its rotation cell certifies nothing at all on
    // fisheye input, which is where its mass goes missing.
    assert_eq!(pin.n_scanned_epipolar, fish.n_scanned_epipolar);
    assert_eq!(pin.n_certified_rotation, 0, "{:?}", pin.scan_votes);
    assert!(pin.n_certified > 0);
}

/// A fisheye capture with NO parallax anywhere: 8 views of a pure-rotation rig
/// panning across ±1.3 rad, imaged at [`F_FISH`]. Every epipolar candidate pair
/// is parallax-free, so the epipolar cell has nothing to observe and must
/// abstain — the scene the rotation-domination gate exists for. The pan is wide
/// enough that neighbours clear the rotation family's displacement floor, so
/// the rotation cell has real candidates to carry the verdict with.
fn pure_rotation_fisheye_scene(seed: u64) -> Obs {
    let mut rng = Lcg(seed);
    let n = 8;
    let cams = rotation_cameras(n, 1.3, &mut rng);
    let mut obs = Obs::default();
    for i in 0..n - 1 {
        emit_fisheye_rotation_pair(&mut obs, &cams, i, i + 1, 60, &mut rng);
    }
    for i in 0..n - 2 {
        emit_fisheye_rotation_pair(&mut obs, &cams, i, i + 2, 60, &mut rng);
    }
    obs
}

#[test]
fn parallax_free_fisheye_capture_abstains_in_the_epipolar_cell() {
    // Without the rotation-domination gate the fisheye epipolar cell fits
    // *something* to every parallax-free pair — the essentialness minima of a
    // degenerate `E = [t]×R` are broad — and those junk votes dragged this
    // capture's pooled median to 624 px against a planted 320. The gate is the
    // pinhole family's homography domination in its fisheye form: the rotation
    // cell carries the verdict alone.
    let res = pure_rotation_fisheye_scene(2718)
        .run_columns(0, &[CameraModel::Pinhole, CameraModel::EquidistantFisheye]);
    assert_eq!(
        res.camera_model,
        Some(CameraModel::EquidistantFisheye),
        "{:?}",
        res.columns
    );
    let fish = &res.columns[1];
    // Every epipolar candidate is gated out, so the pool is rotation votes.
    assert_eq!(fish.n_certified_epipolar, 0, "{:?}", fish.scan_votes);
    assert_eq!(fish.n_rotation_dominated, fish.n_scanned_epipolar);
    assert!(fish.n_rotation_dominated > 0);
    assert_eq!(res.n_epipolar, 0);
    assert!(res.n_rotation >= 2);
    assert_eq!(res.family, Some(VoteFamily::Rotation));
    // ...and the consensus is the planted focal, not the 624 px blend.
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_FISH).abs() / F_FISH < 0.05,
        "recovered {f} vs planted {F_FISH}"
    );
    // The capture reads as parallax-poor in the column's own diagnostic.
    assert!(
        fish.parallax_poverty >= 0.8,
        "parallax poverty {}",
        fish.parallax_poverty
    );
}

#[test]
fn column_scans_are_deterministic() {
    let obs = fisheye_scene(2718);
    let cols = [CameraModel::Pinhole, CameraModel::EquidistantFisheye];
    let a = obs.run_columns(42, &cols);
    let b = obs.run_columns(42, &cols);
    assert_eq!(a.camera_model, b.camera_model);
    assert_eq!(a.focal_px.map(f64::to_bits), b.focal_px.map(f64::to_bits));
    assert_eq!(a.columns.len(), b.columns.len());
    for (x, y) in a.columns.iter().zip(b.columns.iter()) {
        assert_eq!(x.model, y.model);
        assert_eq!(x.n_certified, y.n_certified);
        assert_eq!(x.n_informative, y.n_informative);
        assert_eq!(x.focal_px.map(f64::to_bits), y.focal_px.map(f64::to_bits));
        assert_eq!(x.scan_votes.len(), y.scan_votes.len());
        for (u, v) in x.scan_votes.iter().zip(y.scan_votes.iter()) {
            assert_eq!(u.focal_px.to_bits(), v.focal_px.to_bits());
            assert_eq!(u.cost.to_bits(), v.cost.to_bits());
            assert_eq!(u.certified, v.certified);
            assert_eq!(u.model_informative, v.model_informative);
        }
    }
}

#[test]
fn empty_input_no_consensus() {
    let res = focal_vote(&[], &[], &[], W, H, 0);
    assert!(res.focal_px.is_none());
    assert_eq!(res.family, None);
    assert_eq!(res.n_epipolar, 0);
    assert_eq!(res.n_rotation, 0);
    assert_eq!(res.n_pool, 0);
    assert_eq!(res.n_inconsistent_pairs, 0);
    assert_eq!(res.n_degenerate, 0);
    assert_eq!(res.pool_spread, 0.0);
    assert_eq!(res.family_disagreement, None);
}

#[test]
fn single_vote_is_not_a_consensus() {
    // One direction-consistent pair and no rotation partner: a pool of one is
    // below the 2-vote floor, so there is no consensus focal — but the pair
    // vote is still visible through the epipolar diagnostics.
    let mut rng = Lcg(909);
    let cams = baseline_cameras(2, 0.35, &mut rng);
    let mut obs = Obs::default();
    emit_parallax_pair(&mut obs, &cams, 0, 1, 45, &mut rng);
    let res = obs.run(0);
    assert_eq!(res.n_epipolar, 1);
    assert_eq!(res.n_rotation, 0);
    assert_eq!(res.n_pool, 1);
    assert_eq!(res.focal_px, None);
    assert_eq!(res.family, None);
    assert!(res.epipolar_focal_px.is_some());
    // One family only: no inter-family gap, and no pool to spread.
    assert_eq!(res.family_disagreement, None);
    assert_eq!(res.pool_spread, 0.0);
}
