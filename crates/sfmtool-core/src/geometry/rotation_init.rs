// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Far-field rotation initialization ([`rotation_init`]). See
//! `specs/core/rotation-init.md`.
//!
//! Builds an initial multi-camera reconstruction from cluster tracks by using
//! the two point populations for what each observes: parallax-free
//! (far-field) correspondences fix rotations between arbitrary image pairs
//! through conjugate homographies `H = K R K⁻¹`, independent of baseline;
//! parallax-bearing (near-field) correspondences then supply the metric side —
//! a seed baseline, structure, and translation growth — with rotations held.
//!
//! The stages:
//!
//! 1. **Rotation edge graph** — per image, the largest-mean-displacement
//!    covisible partners; per candidate pair a robust homography over the
//!    centred shared-cluster correspondences, validated as a conjugate
//!    rotation at `f0` by the orthogonality residual. A validated edge stores
//!    the polar-orthogonalized `K⁻¹ H K` and its inlier partition (H-inliers
//!    are the edge's far field, H-outliers its near field).
//! 2. **Global rotations** — over the largest connected component:
//!    spanning-tree propagation from the highest-degree image, then iterative
//!    chordal-mean rotation averaging to absorb tree drift.
//! 3. **Seed baseline and structure** — the component edge with the most
//!    near-field correspondences; with both rotations known the translation
//!    direction is linear (`x₂ · (t × R_rel x₁) = 0`), sign fixed by
//!    triangulation cheirality; the near clusters triangulate into the
//!    initial structure at unit baseline scale.
//! 4. **Translation growth** — rotation-locked resection of unposed images
//!    against the triangulated structure, retriangulating over the posed set
//!    each round, finishing with one staged bundle adjustment at fixed `f0`.
//!    The far-field cluster ids feed the adjustment's points-at-infinity
//!    mask: left finite, a dominant far cloud rewards baseline collapse (the
//!    LM walks the scale gauge until the near field crosses the trim depth
//!    floor and the core degenerates to a panorama). After the adjustment
//!    the gauge is renormalized so the seed baseline is unit again.
//!
//! Everything runs in the canonical camera frame (the camera looks along
//! `−Z`); the conjugate-homography rotation is extracted in the optical pixel
//! frame and conjugated by `S = diag(1, −1, −1)` at the boundary. The pair
//! table pass and all selections are deterministic, and the RANSAC estimator
//! derives its sampling from the input seed, so identical inputs and seed
//! reproduce identical output.

use std::collections::{HashMap, HashSet, VecDeque};

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};

use crate::camera::CameraModel;
use crate::geometry::bundle_adjust::{bundle_adjust, DEFAULT_SCHEDULE};
use crate::geometry::focal_vote::ortho_cost;
use crate::geometry::homography_estimation::{estimate_homography, HomographyOptions};
use crate::geometry::resect_translation::resect_translation;
use crate::reconstruction::triangulation::{triangulate_batch, Triangulation};
use crate::CameraIntrinsics;

// ── Tuning (see the spec) ────────────────────────────────────────────────────

/// Candidate pairs need at least this many shared clusters.
const MIN_SHARED_CLUSTERS: usize = 25;
/// … and a mean feature displacement of at least this fraction of the image
/// diagonal.
const MIN_DISP_FRAC: f64 = 0.05;
/// Rotation-edge candidates per image (largest mean displacement first).
const MAX_EDGES_PER_IMAGE: usize = 3;
/// Inlier gate (px, symmetric transfer) for the per-edge homography.
const H_MAX_ERROR_PX: f64 = 3.0;
/// A homography supported by fewer inliers than this is rejected.
const MIN_H_INLIERS: usize = 12;
/// Conjugate-rotation orthogonality residual ceiling at `f0` (a finite-plane
/// homography never passes).
const ORTHO_MAX_RESIDUAL: f64 = 0.12;
/// Rotation-averaging convergence: stop when the largest single-image update
/// falls below this (radians; 0.1°).
const AVG_TOL_RAD: f64 = 0.1 * std::f64::consts::PI / 180.0;
/// Rotation-averaging sweep budget.
const MAX_AVG_SWEEPS: usize = 20;
/// The seed sign needs at least this many in-front triangulations.
const SEED_MIN_CHEIRALITY: usize = 10;
/// An unposed image must observe at least this many triangulated points to
/// attempt resection.
const GROW_MIN_POINTS: usize = 12;
/// Trim gate (px) for the rotation-locked resection.
const RESECT_MAX_ERROR_PX: f64 = 8.0;
/// Survivor floor for the rotation-locked resection.
const RESECT_MIN_INLIERS: usize = 10;
/// LM iteration budget per round of the finishing bundle adjustment.
const BA_MAX_ITERS: usize = 60;
/// Trim survivors a point needs to stay in a bundle-adjustment solve.
const BA_MIN_TRACK: usize = 2;
/// Degenerate floor for the bundle adjustment.
const BA_MIN_OBS: usize = 12;

/// Default component-size floor (`min_images`).
pub const DEFAULT_MIN_IMAGES: usize = 8;
/// Default core size budget (`max_images`).
pub const DEFAULT_MAX_IMAGES: usize = 14;

/// Result of [`rotation_init`]: a posed core for a caller's refinement
/// machinery.
#[derive(Clone, Debug)]
pub struct RotationInit {
    /// Posed image indices, ascending.
    pub image_indexes: Vec<u32>,
    /// World-to-camera rotations (WXYZ), aligned with `image_indexes`.
    pub quaternions_wxyz: Vec<[f64; 4]>,
    /// World-to-camera translations, aligned with `image_indexes`. The seed
    /// pair's baseline defines unit scale.
    pub translations: Vec<[f64; 3]>,
    /// World points indexed by cluster id (`NaN` where absent). Rows listed
    /// in `far_cluster_indexes` are unit world-frame directions (the final
    /// adjustment models the far field at infinity); every other finite row
    /// is a triangulated position.
    pub points: Vec<[f64; 3]>,
    /// Per-posed-image fraction of its observations surviving the final
    /// adjustment's last trim gate, aligned with `image_indexes`.
    pub inlier_fractions: Vec<f64>,
    /// Far-field cluster ids: the union of the component edges' H-inlier
    /// clusters, sorted ascending (callers may feed these to the bundle
    /// adjustment's points-at-infinity mask).
    pub far_cluster_indexes: Vec<u32>,
}

/// One validated rotation edge: the canonical relative rotation
/// (`x_b = r_ab · x_a`) and the pair's shared-cluster correspondences with
/// their H-inlier partition (`far[k]` ⇒ far field).
struct RotationEdge {
    a: usize,
    b: usize,
    r_ab: Matrix3<f64>,
    clusters: Vec<u32>,
    x1: Vec<[f64; 2]>,
    x2: Vec<[f64; 2]>,
    far: Vec<bool>,
}

impl RotationEdge {
    fn near_count(&self) -> usize {
        self.far.iter().filter(|&&f| !f).count()
    }
}

// ── Small numeric helpers ────────────────────────────────────────────────────

/// Nearest rotation to `m` (polar factor `U Vᵀ` of the SVD). A negative
/// determinant flips the sign — a homography is defined up to scale
/// *including sign*, so `M ≈ −R` must come back as `R`, not as the (distant)
/// proper projection of `−R`. `None` for a non-finite or degenerate input.
fn polar_rotation(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let p = u * v_t;
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(if p.determinant() < 0.0 { -p } else { p })
}

/// Rotation angle of `r` in radians.
fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
}

// ── Pair tables (the focal-vote pattern) ─────────────────────────────────────

/// Per-image-pair accumulator: shared-cluster count and summed feature
/// displacement.
#[derive(Clone, Copy, Default)]
struct PairAccum {
    count: f64,
    disp_sum: f64,
}

impl PairAccum {
    fn mean_disp(&self) -> f64 {
        if self.count > 0.0 {
            self.disp_sum / self.count
        } else {
            0.0
        }
    }
}

/// Per-image observation list (cluster id, pixel position), sorted by cluster
/// id.
type ImageClusters = Vec<Vec<(u32, [f64; 2])>>;

/// One pass over the cluster runs: per-image (cluster, position) lists (per
/// image deduped, last observation wins) and the covisible-pair displacement
/// table.
fn build_pair_tables(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    n_img: usize,
) -> (ImageClusters, HashMap<(u32, u32), PairAccum>) {
    let n_obs = cluster_indexes.len();
    let mut image_clusters: ImageClusters = vec![Vec::new(); n_img];
    let mut pair_accum: HashMap<(u32, u32), PairAccum> = HashMap::new();

    let mut run_start = 0usize;
    while run_start < n_obs {
        let cid = cluster_indexes[run_start];
        let mut run_end = run_start + 1;
        while run_end < n_obs && cluster_indexes[run_end] == cid {
            run_end += 1;
        }

        let mut last_seen: HashMap<u32, [f64; 2]> = HashMap::new();
        for r in run_start..run_end {
            last_seen.insert(image_indexes[r], positions_xy[r]);
        }
        let mut members: Vec<(u32, [f64; 2])> = last_seen.into_iter().collect();
        members.sort_by_key(|m| m.0);
        for &(img, pos) in &members {
            image_clusters[img as usize].push((cid, pos));
        }

        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let (ia, pa) = members[a];
                let (ib, pb) = members[b];
                let d = (pa[0] - pb[0]).hypot(pa[1] - pb[1]);
                let e = pair_accum.entry((ia, ib)).or_default();
                e.count += 1.0;
                e.disp_sum += d;
            }
        }

        run_start = run_end;
    }
    (image_clusters, pair_accum)
}

/// Full-correspondence merge-join of two images over their shared clusters.
/// Returns `(cluster ids, positions in a, positions in b)`.
fn pair_correspondences(
    image_clusters: &ImageClusters,
    a: usize,
    b: usize,
) -> (Vec<u32>, Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (la, lb) = (&image_clusters[a], &image_clusters[b]);
    let mut cids = Vec::new();
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < la.len() && j < lb.len() {
        match la[i].0.cmp(&lb[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                cids.push(la[i].0);
                x1.push(la[i].1);
                x2.push(lb[j].1);
                i += 1;
                j += 1;
            }
        }
    }
    (cids, x1, x2)
}

// ── Stage 1: rotation edge graph ─────────────────────────────────────────────

/// Build the validated rotation edges: per image the top
/// [`MAX_EDGES_PER_IMAGE`] covisible partners by mean displacement (gated on
/// [`MIN_SHARED_CLUSTERS`] / [`MIN_DISP_FRAC`]), each candidate validated as a
/// conjugate rotation at `f0`.
fn build_edges(
    image_clusters: &ImageClusters,
    pair_accum: &HashMap<(u32, u32), PairAccum>,
    diag: f64,
    pp: [f64; 2],
    f0: f64,
    seed: u64,
) -> Vec<RotationEdge> {
    let n_img = image_clusters.len();
    let mut partners: Vec<Vec<(f64, u32)>> = vec![Vec::new(); n_img];
    for (&(a, b), acc) in pair_accum {
        if (acc.count as usize) < MIN_SHARED_CLUSTERS {
            continue;
        }
        let d = acc.mean_disp();
        if d < MIN_DISP_FRAC * diag {
            continue;
        }
        partners[a as usize].push((d, b));
        partners[b as usize].push((d, a));
    }

    let h_opts = HomographyOptions {
        max_error_px: H_MAX_ERROR_PX,
        seed,
        min_inliers: MIN_H_INLIERS,
        ..Default::default()
    };
    // Optical ↔ canonical conjugation and the calibration at f0.
    let s = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let k = Matrix3::new(f0, 0.0, 0.0, 0.0, f0, 0.0, 0.0, 0.0, 1.0);
    let kinv = Matrix3::new(1.0 / f0, 0.0, 0.0, 0.0, 1.0 / f0, 0.0, 0.0, 0.0, 1.0);

    let mut edges: Vec<RotationEdge> = Vec::new();
    let mut tried: HashSet<(usize, usize)> = HashSet::new();
    for (i, plist) in partners.iter_mut().enumerate() {
        // Deterministic despite the hash-map source: displacement descending,
        // then partner index ascending.
        plist.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        for &(_d, j) in plist.iter().take(MAX_EDGES_PER_IMAGE) {
            let (a, b) = (i.min(j as usize), i.max(j as usize));
            if !tried.insert((a, b)) {
                continue;
            }
            let (cids, x1, x2) = pair_correspondences(image_clusters, a, b);
            // Centre on the principal point: H = K R K⁻¹ has K at the origin.
            let x1c: Vec<[f64; 2]> = x1.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
            let x2c: Vec<[f64; 2]> = x2.iter().map(|p| [p[0] - pp[0], p[1] - pp[1]]).collect();
            let Some(hest) = estimate_homography(&x1c, &x2c, &h_opts) else {
                continue;
            };
            if ortho_cost(&hest.h_matrix, f0) >= ORTHO_MAX_RESIDUAL {
                continue;
            }
            let m = kinv * hest.h_matrix * k;
            let Some(r_opt) = polar_rotation(&m) else {
                continue;
            };
            edges.push(RotationEdge {
                a,
                b,
                r_ab: s * r_opt * s,
                clusters: cids,
                x1,
                x2,
                far: hest.inliers,
            });
        }
    }
    edges
}

// ── Stage 2: global rotations ────────────────────────────────────────────────

/// Per-image neighbour lists `(neighbor, edge index)`, sorted for
/// deterministic traversal.
fn neighbor_lists(edges: &[RotationEdge], n_img: usize) -> Vec<Vec<(usize, usize)>> {
    let mut nbrs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_img];
    for (idx, e) in edges.iter().enumerate() {
        nbrs[e.a].push((e.b, idx));
        nbrs[e.b].push((e.a, idx));
    }
    for l in nbrs.iter_mut() {
        l.sort_unstable();
    }
    nbrs
}

/// Largest connected component of the edge graph (ties: the one containing
/// the smallest image index), sorted ascending.
fn largest_component(nbrs: &[Vec<(usize, usize)>]) -> Vec<usize> {
    let n_img = nbrs.len();
    let mut seen = vec![false; n_img];
    let mut best: Vec<usize> = Vec::new();
    for start in 0..n_img {
        if seen[start] || nbrs[start].is_empty() {
            continue;
        }
        let mut comp = vec![start];
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(x) = queue.pop_front() {
            for &(y, _) in &nbrs[x] {
                if !seen[y] {
                    seen[y] = true;
                    comp.push(y);
                    queue.push_back(y);
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    best.sort_unstable();
    best
}

/// Spanning-tree propagation of the edge rotations from the highest-degree
/// image of the component (BFS over sorted neighbour lists). Returns
/// world-to-camera rotations in the root's gauge (`rot[root] = I`).
fn propagate_tree(
    edges: &[RotationEdge],
    nbrs: &[Vec<(usize, usize)>],
    comp: &[usize],
) -> Vec<Option<Matrix3<f64>>> {
    let mut root = comp[0];
    for &i in comp {
        if nbrs[i].len() > nbrs[root].len() {
            root = i;
        }
    }
    let mut rot: Vec<Option<Matrix3<f64>>> = vec![None; nbrs.len()];
    rot[root] = Some(Matrix3::identity());
    let mut queue = VecDeque::from([root]);
    while let Some(cur) = queue.pop_front() {
        let r_cur = rot[cur].unwrap();
        for &(nbr, eidx) in &nbrs[cur] {
            if rot[nbr].is_some() {
                continue;
            }
            let e = &edges[eidx];
            rot[nbr] = Some(if e.a == cur {
                e.r_ab * r_cur
            } else {
                e.r_ab.transpose() * r_cur
            });
            queue.push_back(nbr);
        }
    }
    rot
}

/// Iterative chordal-mean rotation averaging over the component: each image's
/// rotation is re-estimated as the chordal mean of its neighbours' propagated
/// estimates (`R_j ← polar(Σ over edges (i,j) of R_ij · R_i)`), sweeping
/// (Gauss–Seidel, ascending image order) until the largest single-image
/// update falls below [`AVG_TOL_RAD`] or [`MAX_AVG_SWEEPS`]. Averaging
/// absorbs the drift a chain of edge rotations passes to a tree's leaves.
/// Returns the number of sweeps run.
fn average_rotations(
    edges: &[RotationEdge],
    nbrs: &[Vec<(usize, usize)>],
    comp: &[usize],
    rot: &mut [Option<Matrix3<f64>>],
) -> usize {
    for sweep in 0..MAX_AVG_SWEEPS {
        let mut max_update = 0.0f64;
        for &j in comp {
            let mut sum = Matrix3::<f64>::zeros();
            for &(_nbr, eidx) in &nbrs[j] {
                let e = &edges[eidx];
                sum += if e.b == j {
                    e.r_ab * rot[e.a].unwrap()
                } else {
                    e.r_ab.transpose() * rot[e.b].unwrap()
                };
            }
            if let Some(new_r) = polar_rotation(&sum) {
                let old = rot[j].unwrap();
                max_update = max_update.max(rotation_angle(&(new_r * old.transpose())));
                rot[j] = Some(new_r);
            }
        }
        if max_update < AVG_TOL_RAD {
            return sweep + 1;
        }
    }
    MAX_AVG_SWEEPS
}

// ── Stage 3: seed baseline and structure ─────────────────────────────────────

/// Two-view triangulation of per-correspondence ray pairs for a candidate
/// relative pose (camera `a` at the origin). Returns the per-correspondence
/// triangulations and the in-front count.
fn triangulate_two_view(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    r_rel: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> (Vec<Triangulation>, usize) {
    let n = r1.len();
    let center_b = Point3::from(-(r_rel.transpose() * t));
    let mut dirs = Vec::with_capacity(2 * n);
    let mut centers = Vec::with_capacity(2 * n);
    let mut offsets = Vec::with_capacity(n + 1);
    for k in 0..n {
        offsets.push(dirs.len());
        dirs.push(r1[k]);
        centers.push(Point3::origin());
        dirs.push(r_rel.transpose() * r2[k]);
        centers.push(center_b);
    }
    offsets.push(dirs.len());
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    let in_front = tris.iter().filter(|t| t.in_front_of_all_cameras).count();
    (tris, in_front)
}

/// Seed the metric frame from the component edge with the most near-field
/// correspondences: linear translation direction over the near rows, sign by
/// triangulation cheirality, near clusters triangulated into `points` (world
/// = the rotations' gauge, camera `a` at the origin). Returns the seed images
/// and the second camera's translation (unit scale).
fn seed_baseline(
    cam: &CameraIntrinsics,
    edges: &[RotationEdge],
    in_comp: &[bool],
    rot: &[Option<Matrix3<f64>>],
    points: &mut [[f64; 3]],
) -> Option<(usize, usize, Vector3<f64>)> {
    let mut best: Option<(usize, usize)> = None; // (near count, edge index)
    for (idx, e) in edges.iter().enumerate() {
        if !in_comp[e.a] || !in_comp[e.b] {
            continue;
        }
        let nc = e.near_count();
        if best.is_none_or(|(bc, _)| nc > bc) {
            best = Some((nc, idx));
        }
    }
    let (_, eidx) = best?;
    let e = &edges[eidx];
    let (rw_a, rw_b) = (rot[e.a]?, rot[e.b]?);
    let r_rel = rw_b * rw_a.transpose();

    // Near-field unit rays and cluster ids.
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    let mut cids = Vec::new();
    for k in 0..e.clusters.len() {
        if e.far[k] {
            continue;
        }
        let d1 = cam.pixel_to_ray(e.x1[k][0], e.x1[k][1]);
        let d2 = cam.pixel_to_ray(e.x2[k][0], e.x2[k][1]);
        r1.push(Vector3::new(d1[0], d1[1], d1[2]).normalize());
        r2.push(Vector3::new(d2[0], d2[1], d2[2]).normalize());
        cids.push(e.clusters[k]);
    }
    if r1.len() < SEED_MIN_CHEIRALITY {
        return None;
    }

    // The epipolar constraint is linear in the translation direction:
    // x₂ · (t × R_rel x₁) = 0 ⇒ each near row contributes
    // w = (R_rel x₁) × x₂ with w · t = 0; t is the null direction of Σ w wᵀ.
    let mut ata = Matrix3::<f64>::zeros();
    for k in 0..r1.len() {
        let w = (r_rel * r1[k]).cross(&r2[k]);
        ata += w * w.transpose();
    }
    let eig = ata.symmetric_eigen();
    let mut kmin = 0usize;
    let mut kmax = 0usize;
    for j in 1..3 {
        if eig.eigenvalues[j] < eig.eigenvalues[kmin] {
            kmin = j;
        }
        if eig.eigenvalues[j] > eig.eigenvalues[kmax] {
            kmax = j;
        }
    }
    if eig.eigenvalues[kmax] <= 0.0 {
        return None;
    }
    let t_dir = eig.eigenvectors.column(kmin).into_owned();
    let norm = t_dir.norm();
    if !norm.is_finite() || norm < 1e-12 {
        return None;
    }
    let t_dir = t_dir / norm;

    // Sign by triangulation cheirality: majority in front wins, minimum
    // SEED_MIN_CHEIRALITY.
    let (tris_p, front_p) = triangulate_two_view(&r1, &r2, &r_rel, &t_dir);
    let (tris_n, front_n) = triangulate_two_view(&r1, &r2, &r_rel, &(-t_dir));
    let (t, tris, front) = if front_n > front_p {
        (-t_dir, tris_n, front_n)
    } else {
        (t_dir, tris_p, front_p)
    };
    if front < SEED_MIN_CHEIRALITY {
        return None;
    }

    // Seed structure: in-front near clusters, camera-a frame → world
    // (t_a = 0, so X = Rw[a]ᵀ · x_a).
    for (k, tri) in tris.iter().enumerate() {
        if tri.in_front_of_all_cameras {
            let xw = rw_a.transpose() * tri.point.coords;
            points[cids[k] as usize] = [xw.x, xw.y, xw.z];
        }
    }
    Some((e.a, e.b, t))
}

// ── Stage 4: translation growth ──────────────────────────────────────────────

/// Rebuild every cluster's point from the posed images' observations
/// (ray-midpoint batch triangulation; fewer than two observations ⇒ `NaN`).
fn retriangulate_posed(
    image_clusters: &ImageClusters,
    posed: &[bool],
    cam: &CameraIntrinsics,
    quats: &[UnitQuaternion<f64>],
    trans: &[Vector3<f64>],
    points: &mut [[f64; 3]],
) {
    let mut rows: Vec<(u32, Vector3<f64>, Point3<f64>)> = Vec::new();
    for (i, obs) in image_clusters.iter().enumerate() {
        if !posed[i] {
            continue;
        }
        let r_inv = quats[i].inverse();
        let center = Point3::from(-(r_inv * trans[i]));
        for &(cid, pos) in obs {
            let d = cam.pixel_to_ray(pos[0], pos[1]);
            rows.push((cid, r_inv * Vector3::new(d[0], d[1], d[2]), center));
        }
    }
    rows.sort_by_key(|r| r.0);

    let mut dirs = Vec::with_capacity(rows.len());
    let mut centers = Vec::with_capacity(rows.len());
    let mut offsets = Vec::new();
    let mut track_cid = Vec::new();
    let mut prev: Option<u32> = None;
    for &(cid, dir, center) in &rows {
        if prev != Some(cid) {
            offsets.push(dirs.len());
            track_cid.push(cid);
            prev = Some(cid);
        }
        dirs.push(dir);
        centers.push(center);
    }
    offsets.push(dirs.len());

    for p in points.iter_mut() {
        *p = [f64::NAN; 3];
    }
    let tris = triangulate_batch(&dirs, &centers, &offsets);
    for (t, tri) in tris.iter().enumerate() {
        if offsets[t + 1] - offsets[t] >= 2 {
            points[track_cid[t] as usize] = [tri.point.x, tri.point.y, tri.point.z];
        }
    }
}

// ── The kernel ───────────────────────────────────────────────────────────────

/// Build an initial multi-camera reconstruction from cluster tracks by
/// far-field rotation initialization. See `specs/core/rotation-init.md` and
/// the module docs for the mechanism.
///
/// `cluster_indexes` must be nondecreasing (each distinct cluster is a
/// contiguous run); `image_indexes` and `positions_xy` are the image id and
/// full-pixel keypoint position per observation; all images share
/// `width × height` and the focal `f0` (typically a focal-vote consensus),
/// with the principal point at the image centre. `seed` drives the homography
/// RANSAC; same inputs and seed give bit-identical output.
///
/// Returns `None` when no rotation edge validates, the largest connected
/// component has fewer than `min_images` images, or the seed baseline fails
/// its cheirality floor.
#[allow(clippy::too_many_arguments)]
pub fn rotation_init(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    f0: f64,
    seed: u64,
    min_images: usize,
    max_images: usize,
) -> Option<RotationInit> {
    let n_obs = cluster_indexes.len();
    if n_obs == 0 || image_indexes.len() != n_obs || positions_xy.len() != n_obs {
        return None;
    }
    if !(f0.is_finite() && f0 > 0.0) || max_images < 2 {
        return None;
    }
    let n_img = *image_indexes.iter().max().unwrap() as usize + 1;
    let n_pts = *cluster_indexes.iter().max().unwrap() as usize + 1;
    let pp = [width as f64 / 2.0, height as f64 / 2.0];
    let diag = (width as f64).hypot(height as f64);

    // Stage 1: rotation edge graph.
    let (image_clusters, pair_accum) =
        build_pair_tables(cluster_indexes, image_indexes, positions_xy, n_img);
    let edges = build_edges(&image_clusters, &pair_accum, diag, pp, f0, seed);
    if edges.is_empty() {
        return None;
    }

    // Stage 2: global rotations over the largest connected component.
    let nbrs = neighbor_lists(&edges, n_img);
    let comp = largest_component(&nbrs);
    if comp.len() < min_images.max(2) {
        return None;
    }
    let mut rot = propagate_tree(&edges, &nbrs, &comp);
    average_rotations(&edges, &nbrs, &comp, &mut rot);

    // Stage 3: seed baseline and structure.
    let cam = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f0,
            principal_point_x: pp[0],
            principal_point_y: pp[1],
        },
        width,
        height,
    };
    let mut in_comp = vec![false; n_img];
    for &i in &comp {
        in_comp[i] = true;
    }
    let mut points = vec![[f64::NAN; 3]; n_pts];
    let (seed_a, seed_b, t_seed) = seed_baseline(&cam, &edges, &in_comp, &rot, &mut points)?;

    let mut quats: Vec<UnitQuaternion<f64>> = (0..n_img)
        .map(|i| match rot[i] {
            Some(m) => UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(m)),
            None => UnitQuaternion::identity(),
        })
        .collect();
    let mut trans: Vec<Vector3<f64>> = vec![Vector3::zeros(); n_img];
    trans[seed_b] = t_seed;
    let mut posed = vec![false; n_img];
    posed[seed_a] = true;
    posed[seed_b] = true;
    let mut n_posed = 2usize;

    // Stage 4: translation growth by rotation-locked resection, most
    // triangulated observations first, retriangulating over the posed set
    // after each round.
    loop {
        let mut cands: Vec<(usize, usize)> = Vec::new(); // (finite obs count, image)
        for &i in &comp {
            if posed[i] {
                continue;
            }
            let cnt = image_clusters[i]
                .iter()
                .filter(|(cid, _)| points[*cid as usize][0].is_finite())
                .count();
            if cnt >= GROW_MIN_POINTS {
                cands.push((cnt, i));
            }
        }
        cands.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));

        let mut added = false;
        for &(_cnt, i) in &cands {
            if n_posed >= max_images {
                break;
            }
            let mut world = Vec::new();
            let mut uv = Vec::new();
            for &(cid, pos) in &image_clusters[i] {
                let p = points[cid as usize];
                if p[0].is_finite() {
                    world.push(p);
                    uv.push(pos);
                }
            }
            if let Some(res) = resect_translation(
                &cam,
                &quats[i],
                &world,
                &uv,
                RESECT_MAX_ERROR_PX,
                RESECT_MIN_INLIERS,
            ) {
                trans[i] = res.translation;
                posed[i] = true;
                n_posed += 1;
                added = true;
            }
        }
        if added {
            retriangulate_posed(&image_clusters, &posed, &cam, &quats, &trans, &mut points);
        }
        if !added || n_posed >= max_images {
            break;
        }
    }

    // Far-field cluster ids: union over the component's validated edges.
    // They double as the finishing adjustment's points-at-infinity mask —
    // left finite, a dominant far cloud rewards baseline collapse (the LM
    // walks the scale gauge until the near field crosses the trim depth
    // floor and the core degenerates to a panorama).
    let mut far_mask = vec![false; n_pts];
    for e in &edges {
        if !in_comp[e.a] || !in_comp[e.b] {
            continue;
        }
        for k in 0..e.clusters.len() {
            if e.far[k] {
                far_mask[e.clusters[k] as usize] = true;
            }
        }
    }

    // Finishing staged bundle adjustment (full default schedule) at fixed f0
    // over the posed set's observations.
    let mut obs_img: Vec<u32> = Vec::new();
    let mut obs_pt: Vec<u32> = Vec::new();
    let mut uv: Vec<[f64; 2]> = Vec::new();
    for (i, obs) in image_clusters.iter().enumerate() {
        if !posed[i] {
            continue;
        }
        for &(cid, pos) in obs {
            obs_img.push(i as u32);
            obs_pt.push(cid);
            uv.push(pos);
        }
    }
    let ba = bundle_adjust(
        &cam,
        &mut quats,
        &mut trans,
        &mut points,
        &uv,
        &obs_img,
        &obs_pt,
        Some(&far_mask),
        false,
        &DEFAULT_SCHEDULE,
        BA_MAX_ITERS,
        BA_MIN_TRACK,
        BA_MIN_OBS,
    );

    // The scale gauge is flat under the adjustment (it can wander); pin it
    // back to the contract: the seed pair's baseline is unit. Directions
    // (far rows) stay unit and are not rescaled.
    let center = |i: usize| -(quats[i].inverse() * trans[i]);
    let baseline = (center(seed_b) - center(seed_a)).norm();
    if baseline.is_finite() && baseline > 1e-12 {
        let scale = 1.0 / baseline;
        for (i, t) in trans.iter_mut().enumerate() {
            if posed[i] {
                *t *= scale;
            }
        }
        for (cid, p) in points.iter_mut().enumerate() {
            if !far_mask[cid] && p[0].is_finite() {
                for c in p.iter_mut() {
                    *c *= scale;
                }
            }
        }
    }

    // Per-posed-image surviving inlier fraction at the final trim gate.
    let final_trim = DEFAULT_SCHEDULE.last().unwrap().trim_px;
    let mut obs_total = vec![0usize; n_img];
    let mut obs_kept = vec![0usize; n_img];
    for (k, &i) in obs_img.iter().enumerate() {
        obs_total[i as usize] += 1;
        if ba.residual_norms[k] < final_trim {
            obs_kept[i as usize] += 1;
        }
    }

    let image_indexes_out: Vec<u32> = (0..n_img).filter(|&i| posed[i]).map(|i| i as u32).collect();
    let quaternions_wxyz = image_indexes_out
        .iter()
        .map(|&i| {
            let q = quats[i as usize].into_inner();
            [q.w, q.i, q.j, q.k]
        })
        .collect();
    let translations = image_indexes_out
        .iter()
        .map(|&i| {
            let t = trans[i as usize];
            [t.x, t.y, t.z]
        })
        .collect();
    let inlier_fractions = image_indexes_out
        .iter()
        .map(|&i| {
            let i = i as usize;
            if obs_total[i] > 0 {
                obs_kept[i] as f64 / obs_total[i] as f64
            } else {
                0.0
            }
        })
        .collect();

    let far_cluster_indexes: Vec<u32> = (0..n_pts as u32)
        .filter(|&c| far_mask[c as usize])
        .collect();

    Some(RotationInit {
        image_indexes: image_indexes_out,
        quaternions_wxyz,
        translations,
        points,
        inlier_fractions,
        far_cluster_indexes,
    })
}

#[cfg(test)]
mod tests;
