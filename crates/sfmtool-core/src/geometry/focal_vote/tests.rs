// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::Vector3;

const W: u32 = 1000;
const H: u32 = 1000;
const F_TRUE: f64 = 800.0;
const CX: f64 = 500.0;
const CY: f64 = 500.0;

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

fn k_mat() -> Matrix3<f64> {
    Matrix3::new(F_TRUE, 0.0, CX, 0.0, F_TRUE, CY, 0.0, 0.0, 1.0)
}

struct Cam {
    r: Matrix3<f64>,
    t: Vector3<f64>,
}

impl Cam {
    /// Project a world point to pixels, `None` when behind the camera or out of
    /// the image.
    fn project(&self, x: Vector3<f64>) -> Option<[f64; 2]> {
        let xc = self.r * x + self.t;
        if xc.z <= 1e-3 {
            return None;
        }
        let p = k_mat() * xc;
        let u = p.x / p.z;
        let v = p.y / p.z;
        if !(0.0..W as f64).contains(&u) || !(0.0..H as f64).contains(&v) {
            return None;
        }
        Some([u, v])
    }
}

/// Accumulating builder for flat observation arrays (one span-2 cluster per
/// emitted correspondence).
#[derive(Default)]
struct Obs {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
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
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 200 {
        guard += 1;
        let yaw = rng.uniform(-0.9, 0.9);
        let pitch = rng.uniform(-0.6, 0.6);
        let dir = Vector3::new(yaw.sin(), pitch.sin(), 1.0).normalize() * 30.0;
        if let (Some(mut pa), Some(mut pb)) = (cams[ia].project(dir), cams[ib].project(dir)) {
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
fn baseline_cameras(n_img: usize, baseline: f64, rng: &mut Lcg) -> Vec<Cam> {
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
    let mut done = 0;
    let mut guard = 0;
    while done < m && guard < m * 200 {
        guard += 1;
        let x = Vector3::new(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(4.0, 9.0),
        );
        if let (Some(mut pa), Some(mut pb)) = (cams[ia].project(x), cams[ib].project(x)) {
            pa[0] += 0.3 * rng.gaussian();
            pa[1] += 0.3 * rng.gaussian();
            pb[0] += 0.3 * rng.gaussian();
            pb[1] += 0.3 * rng.gaussian();
            obs.push_pair(ia as u32, pa, ib as u32, pb);
            done += 1;
        }
    }
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

// ── End-to-end arbitration ───────────────────────────────────────────────────

#[test]
fn rotation_scene_arbitrates_to_rotation() {
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
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_TRUE).abs() / F_TRUE < 0.1,
        "rotation focal {f}, true {F_TRUE}"
    );
    assert!(res.parallax_poverty >= POVERTY_THRESHOLD, "poverty too low");
}

#[test]
fn parallax_scene_arbitrates_to_epipolar() {
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
    assert!(res.n_epipolar >= EPIPOLAR_QUORUM);
    let f = res.focal_px.expect("consensus focal");
    assert!(
        (f - F_TRUE).abs() / F_TRUE < 0.15,
        "epipolar focal {f}, true {F_TRUE}"
    );
    assert!(res.parallax_poverty < POVERTY_THRESHOLD, "poverty too high");
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
    assert_eq!(a.parallax_poverty.to_bits(), b.parallax_poverty.to_bits());
}

#[test]
fn empty_input_no_consensus() {
    let res = focal_vote(&[], &[], &[], W, H, 0);
    assert!(res.focal_px.is_none());
    assert_eq!(res.family, None);
    assert_eq!(res.n_epipolar, 0);
    assert_eq!(res.n_rotation, 0);
}
