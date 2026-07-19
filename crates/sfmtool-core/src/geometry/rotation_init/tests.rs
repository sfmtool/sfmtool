// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

const W: u32 = 1000;
const H: u32 = 1000;
const F0: f64 = 800.0;

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

fn test_cam() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: F0,
            principal_point_x: W as f64 / 2.0,
            principal_point_y: H as f64 / 2.0,
        },
        width: W,
        height: H,
    }
}

/// A ground-truth camera in the canonical convention (looks along -Z):
/// world-to-camera rotation and camera center (`x_cam = R (X - C)`).
struct Pose {
    r: Matrix3<f64>,
    c: Vector3<f64>,
}

impl Pose {
    fn project(&self, cam: &CameraIntrinsics, x: Vector3<f64>) -> Option<[f64; 2]> {
        let pc = self.r * (x - self.c);
        if pc.z >= -1e-6 {
            return None;
        }
        let (u, v) = cam.ray_to_pixel([pc.x, pc.y, pc.z])?;
        if !(0.0..W as f64).contains(&u) || !(0.0..H as f64).contains(&v) {
            return None;
        }
        Some([u, v])
    }
}

/// Flat cluster-observation arrays plus scene bookkeeping.
struct Scene {
    cluster: Vec<u32>,
    image: Vec<u32>,
    pos: Vec<[f64; 2]>,
    poses: Vec<Pose>,
    /// World point per assigned cluster id.
    world: Vec<Vector3<f64>>,
    /// First cluster id belonging to the far cloud (near cids are below it).
    far_cid_start: u32,
}

/// Far-field-rich capture: cameras strung along +X panning across the scene,
/// a near cloud (strong parallax) and a distant cloud (parallax below the
/// homography gate). Every world point visible in >= 2 images becomes one
/// cluster observed in all of them.
fn far_field_scene(n_img: usize, n_near: usize, n_far: usize, noise: f64, rng: &mut Lcg) -> Scene {
    let cam = test_cam();
    let mid = (n_img - 1) as f64 / 2.0;
    let poses: Vec<Pose> = (0..n_img)
        .map(|i| {
            let pan = (i as f64 - mid) * 0.04;
            let tilt = rng.uniform(-0.01, 0.01);
            Pose {
                r: rx(tilt) * ry(pan),
                c: Vector3::new((i as f64 - mid) * 0.4, rng.uniform(-0.05, 0.05), 0.0),
            }
        })
        .collect();

    // Near cloud in front of the rig (canonical: in front is z < 0), then the
    // far cloud at ~5000 units — parallax over the full camera spread stays
    // under a pixel.
    let mut world_all: Vec<Vector3<f64>> = Vec::new();
    for _ in 0..n_near {
        world_all.push(Vector3::new(
            rng.uniform(-4.0, 4.0),
            rng.uniform(-4.0, 4.0),
            rng.uniform(-9.0, -4.0),
        ));
    }
    let far_world_start = world_all.len();
    for _ in 0..n_far {
        let dir = Vector3::new(rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), -1.0).normalize();
        world_all.push(dir * rng.uniform(4000.0, 6000.0));
    }

    let mut scene = Scene {
        cluster: Vec::new(),
        image: Vec::new(),
        pos: Vec::new(),
        poses,
        world: Vec::new(),
        far_cid_start: 0,
    };
    let mut next_cid = 0u32;
    for (w_idx, &x) in world_all.iter().enumerate() {
        let members: Vec<(u32, [f64; 2])> = (0..n_img)
            .filter_map(|i| scene.poses[i].project(&cam, x).map(|p| (i as u32, p)))
            .collect();
        if members.len() < 2 {
            continue;
        }
        if w_idx == far_world_start {
            scene.far_cid_start = next_cid;
        }
        for (img, p) in members {
            scene.cluster.push(next_cid);
            scene.image.push(img);
            scene
                .pos
                .push([p[0] + noise * rng.gaussian(), p[1] + noise * rng.gaussian()]);
        }
        scene.world.push(x);
        next_cid += 1;
    }
    // The sentinel above sets far_cid_start unless the first far point was
    // dropped for visibility; fall back to counting near clusters.
    if scene.far_cid_start == 0 {
        scene.far_cid_start = scene.world.iter().take_while(|w| w.norm() < 100.0).count() as u32;
    }
    scene
}

/// Gauge-align estimated rotations to ground truth (`est ≈ gt · G`) and
/// return per-image angular errors in degrees.
fn aligned_rotation_errors(gt: &[Matrix3<f64>], est: &[Matrix3<f64>]) -> Vec<f64> {
    let mut sum = Matrix3::<f64>::zeros();
    for (g, e) in gt.iter().zip(est) {
        sum += g.transpose() * e;
    }
    let gauge = polar_rotation(&sum).unwrap();
    gt.iter()
        .zip(est)
        .map(|(g, e)| rotation_angle(&(e * (g * gauge).transpose())).to_degrees())
        .collect()
}

/// Similarity (s, R, t) mapping `xs` onto `ys` by least squares (Umeyama
/// without the scale-correction refinement — adequate for near-exact fits),
/// returning per-point residual norms and the spread of `ys`.
fn similarity_residuals(xs: &[Vector3<f64>], ys: &[Vector3<f64>]) -> (Vec<f64>, f64) {
    let n = xs.len() as f64;
    let cx = xs.iter().sum::<Vector3<f64>>() / n;
    let cy = ys.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::<f64>::zeros();
    let mut var_x = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        cov += (y - cy) * (x - cx).transpose();
        var_x += (x - cx).norm_squared();
    }
    let r = polar_rotation(&cov).unwrap();
    let s = (r.transpose() * cov).trace() / var_x.max(1e-300);
    let res = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| ((y - cy) - s * (r * (x - cx))).norm())
        .collect();
    let spread = (ys.iter().map(|y| (y - cy).norm_squared()).sum::<f64>() / n).sqrt();
    (res, spread)
}

fn run(scene: &Scene, seed: u64, min_images: usize, max_images: usize) -> Option<RotationInit> {
    rotation_init(
        &scene.cluster,
        &scene.image,
        &scene.pos,
        W,
        H,
        F0,
        seed,
        min_images,
        max_images,
    )
}

// ── Rotations, seed, and structure on a far-field-rich scene ─────────────────

#[test]
fn far_field_scene_recovers_rotations_sub_degree() {
    let mut rng = Lcg(11);
    let scene = far_field_scene(10, 130, 160, 0.2, &mut rng);
    let out = run(&scene, 0, 8, 14).expect("far-field scene must initialize");
    assert!(
        out.image_indexes.len() >= 8,
        "posed {}",
        out.image_indexes.len()
    );

    let gt: Vec<Matrix3<f64>> = out
        .image_indexes
        .iter()
        .map(|&i| scene.poses[i as usize].r)
        .collect();
    let est: Vec<Matrix3<f64>> = out
        .quaternions_wxyz
        .iter()
        .map(|q| {
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]))
                .to_rotation_matrix()
                .into_inner()
        })
        .collect();
    let errs = aligned_rotation_errors(&gt, &est);
    let max_err = errs.iter().cloned().fold(0.0, f64::max);
    assert!(max_err < 1.0, "rotation errors (deg): {errs:?}");

    // Far-field clusters flagged for the caller's points-at-infinity mask are
    // overwhelmingly from the far cloud.
    assert!(!out.far_cluster_indexes.is_empty());
    let far_hits = out
        .far_cluster_indexes
        .iter()
        .filter(|&&c| c >= scene.far_cid_start)
        .count();
    assert!(
        far_hits * 10 >= out.far_cluster_indexes.len() * 8,
        "far ids should be dominated by the far cloud: {far_hits}/{}",
        out.far_cluster_indexes.len()
    );
}

#[test]
fn seed_and_growth_recover_translations_and_structure_up_to_similarity() {
    let mut rng = Lcg(11);
    let scene = far_field_scene(10, 130, 160, 0.2, &mut rng);
    let out = run(&scene, 0, 8, 14).expect("far-field scene must initialize");

    // Camera centers match ground truth up to similarity.
    let est_centers: Vec<Vector3<f64>> = out
        .image_indexes
        .iter()
        .zip(out.quaternions_wxyz.iter().zip(&out.translations))
        .map(|(_, (q, t))| {
            let rq =
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]));
            -(rq.inverse() * Vector3::new(t[0], t[1], t[2]))
        })
        .collect();
    let gt_centers: Vec<Vector3<f64>> = out
        .image_indexes
        .iter()
        .map(|&i| scene.poses[i as usize].c)
        .collect();
    let (res, spread) = similarity_residuals(&est_centers, &gt_centers);
    let max_res = res.iter().cloned().fold(0.0, f64::max);
    assert!(
        max_res < 0.05 * spread,
        "center residuals {res:?} vs spread {spread}"
    );

    // Near structure matches ground truth under the same kind of fit: pair
    // recovered near-cluster points with their world positions.
    let mut est_pts = Vec::new();
    let mut gt_pts = Vec::new();
    for cid in 0..scene.far_cid_start as usize {
        let p = out.points[cid];
        if p[0].is_finite() {
            est_pts.push(Vector3::new(p[0], p[1], p[2]));
            gt_pts.push(scene.world[cid]);
        }
    }
    assert!(
        est_pts.len() >= 50,
        "too few near points: {}",
        est_pts.len()
    );
    let (res, spread) = similarity_residuals(&est_pts, &gt_pts);
    let mut sorted = res.clone();
    sorted.sort_by(f64::total_cmp);
    let med = sorted[sorted.len() / 2];
    assert!(
        med < 0.02 * spread,
        "median structure residual {med} vs spread {spread}"
    );

    // Inlier fractions are meaningful and high on a clean synthetic capture.
    assert!(out
        .inlier_fractions
        .iter()
        .all(|&f| (0.0..=1.0).contains(&f)));
    assert!(
        out.inlier_fractions.iter().sum::<f64>() / out.inlier_fractions.len() as f64 > 0.7,
        "inlier fractions {:?}",
        out.inlier_fractions
    );

    // The seed baseline gauge: the two closest-spaced posed cameras cannot be
    // farther apart than the unit seed baseline, and the structure sits at a
    // plausible near-field depth in that gauge (the depth-floor collapse of a
    // finite-far-point adjustment would shrink both by an order of magnitude).
    let rq0 = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        out.quaternions_wxyz[0][0],
        out.quaternions_wxyz[0][1],
        out.quaternions_wxyz[0][2],
        out.quaternions_wxyz[0][3],
    ));
    let t0 = Vector3::new(
        out.translations[0][0],
        out.translations[0][1],
        out.translations[0][2],
    );
    let mean_depth = est_pts.iter().map(|p| -(rq0 * p + t0).z).sum::<f64>() / est_pts.len() as f64;
    assert!(
        mean_depth > 1.0,
        "near structure collapsed toward the cameras: mean depth {mean_depth}"
    );
}

// ── Averaging vs tree-only propagation ───────────────────────────────────────

/// A long chain with redundant skip edges and per-edge rotation noise: tree
/// propagation accumulates drift toward the leaves, which the chordal-mean
/// sweeps absorb through the redundant edges.
#[test]
fn averaging_beats_tree_only_on_noisy_chain() {
    let n = 30usize;
    let mut rng = Lcg(77);
    let gt: Vec<Matrix3<f64>> = (0..n)
        .map(|i| rx(0.03 * (i as f64 * 1.3).sin()) * ry(0.06 * i as f64))
        .collect();
    let noise_rad = 0.8f64.to_radians();
    let mut edges: Vec<RotationEdge> = Vec::new();
    for step in [1usize, 2, 3] {
        for a in 0..n - step {
            let b = a + step;
            let axis = Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalize();
            let perturb =
                Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), noise_rad)
                    .into_inner();
            edges.push(RotationEdge {
                a,
                b,
                r_ab: perturb * gt[b] * gt[a].transpose(),
                clusters: Vec::new(),
                x1: Vec::new(),
                x2: Vec::new(),
                far: Vec::new(),
            });
        }
    }
    let nbrs = neighbor_lists(&edges, n);
    let comp = largest_component(&nbrs);
    assert_eq!(comp.len(), n);

    let tree = propagate_tree(&edges, &nbrs, &comp);
    let tree_rots: Vec<Matrix3<f64>> = (0..n).map(|i| tree[i].unwrap()).collect();
    let tree_errs = aligned_rotation_errors(&gt, &tree_rots);
    let tree_max = tree_errs.iter().cloned().fold(0.0, f64::max);

    let mut avg = tree.clone();
    let sweeps = average_rotations(&edges, &nbrs, &comp, &mut avg);
    let avg_rots: Vec<Matrix3<f64>> = (0..n).map(|i| avg[i].unwrap()).collect();
    let avg_errs = aligned_rotation_errors(&gt, &avg_rots);
    let avg_max = avg_errs.iter().cloned().fold(0.0, f64::max);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    println!(
        "tree-only err max {tree_max:.4} / mean {:.4} deg; averaged err max {avg_max:.4} / \
         mean {:.4} deg ({sweeps} sweeps)",
        mean(&tree_errs),
        mean(&avg_errs)
    );
    assert!(
        avg_max < 0.6 * tree_max,
        "averaging (max {avg_max:.4} deg) should clearly beat tree-only \
         (max {tree_max:.4} deg)"
    );
    assert!(mean(&avg_errs) < 0.6 * mean(&tree_errs));
}

// ── Budgets and failure modes ────────────────────────────────────────────────

#[test]
fn growth_stops_at_max_images() {
    let mut rng = Lcg(11);
    let scene = far_field_scene(10, 130, 160, 0.2, &mut rng);
    let out = run(&scene, 0, 6, 6).expect("capped run must still initialize");
    assert_eq!(out.image_indexes.len(), 6);
}

#[test]
fn component_below_min_images_returns_none() {
    let mut rng = Lcg(11);
    let scene = far_field_scene(10, 130, 160, 0.2, &mut rng);
    // The component can never reach 20 images in a 10-image capture.
    assert!(run(&scene, 0, 20, 24).is_none());
}

#[test]
fn all_parallax_scene_returns_none() {
    // No far cloud and strong baselines: every candidate homography is either
    // under-supported or fails the conjugate-rotation orthogonality floor.
    let cam = test_cam();
    let mut rng = Lcg(5);
    let n_img = 10usize;
    let mid = (n_img - 1) as f64 / 2.0;
    let poses: Vec<Pose> = (0..n_img)
        .map(|i| Pose {
            r: rx(rng.uniform(-0.02, 0.02)) * ry((i as f64 - mid) * 0.05),
            c: Vector3::new((i as f64 - mid) * 0.8, rng.uniform(-0.1, 0.1), 0.0),
        })
        .collect();
    let mut cluster = Vec::new();
    let mut image = Vec::new();
    let mut pos = Vec::new();
    let mut next_cid = 0u32;
    for _ in 0..250 {
        let x = Vector3::new(
            rng.uniform(-4.0, 4.0),
            rng.uniform(-4.0, 4.0),
            rng.uniform(-8.0, -3.0),
        );
        let members: Vec<(u32, [f64; 2])> = (0..n_img)
            .filter_map(|i| poses[i].project(&cam, x).map(|p| (i as u32, p)))
            .collect();
        if members.len() < 2 {
            continue;
        }
        for (img, p) in members {
            cluster.push(next_cid);
            image.push(img);
            pos.push([p[0] + 0.2 * rng.gaussian(), p[1] + 0.2 * rng.gaussian()]);
        }
        next_cid += 1;
    }
    assert!(rotation_init(&cluster, &image, &pos, W, H, F0, 0, 8, 14).is_none());
}

#[test]
fn determinism_same_seed() {
    let mut rng = Lcg(11);
    let scene = far_field_scene(10, 130, 160, 0.2, &mut rng);
    let a = run(&scene, 42, 8, 14).expect("run a");
    let b = run(&scene, 42, 8, 14).expect("run b");
    assert_eq!(a.image_indexes, b.image_indexes);
    assert_eq!(a.far_cluster_indexes, b.far_cluster_indexes);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (qa, qb) in a.quaternions_wxyz.iter().zip(&b.quaternions_wxyz) {
        assert_eq!(bits(qa), bits(qb));
    }
    for (ta, tb) in a.translations.iter().zip(&b.translations) {
        assert_eq!(bits(ta), bits(tb));
    }
    for (pa, pb) in a.points.iter().zip(&b.points) {
        assert_eq!(bits(pa), bits(pb));
    }
    assert_eq!(bits(&a.inlier_fractions), bits(&b.inlier_fractions));
}

#[test]
fn empty_and_invalid_inputs_return_none() {
    assert!(rotation_init(&[], &[], &[], W, H, F0, 0, 8, 14).is_none());
    // Mismatched lengths.
    assert!(rotation_init(&[0, 0], &[0], &[[0.0, 0.0]; 2], W, H, F0, 0, 8, 14).is_none());
    // Non-positive focal.
    assert!(rotation_init(&[0, 0], &[0, 1], &[[0.0, 0.0]; 2], W, H, 0.0, 0, 8, 14).is_none());
}
