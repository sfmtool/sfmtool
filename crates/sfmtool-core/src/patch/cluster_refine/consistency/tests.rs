// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::params::MemberStatus;
use super::*;
use ndarray::Array3;

/// `(cluster_starts, member_images, member_status, reference_members,
/// member_affines)` — the [`warp_consistency_residuals`] inputs.
type SyntheticScene = (Vec<u32>, Vec<u32>, Vec<MemberStatus>, Vec<u32>, Array3<f64>);

/// Build a synthetic scene: `n_images` scaled-orthographic cameras and
/// `n_clusters` planar frames, emitted as clusters whose reference warp
/// is re-gauged to the identity (exactly the stored representation).
fn synthetic(n_images: usize, n_clusters: usize, members_per_cluster: usize) -> SyntheticScene {
    let mut state = 12345u64;
    let mut rnd = move || {
        let mut s = state;
        let v = noise(&mut s);
        state = s;
        v
    };
    // Cameras: random small rotations around identity, orthographic.
    let cams: Vec<Mat2x3> = (0..n_images)
        .map(|_| {
            let (a, b, c) = (0.4 * rnd(), 0.4 * rnd(), 0.4 * rnd());
            let (ca, sa) = (a.cos(), a.sin());
            let (cb, sb) = (b.cos(), b.sin());
            let (cc, sc) = (c.cos(), c.sin());
            // Rz(c)·Ry(b)·Rx(a), top two rows.
            [
                [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
            ]
        })
        .collect();
    let m = n_clusters * members_per_cluster;
    let mut cluster_starts = vec![0u32];
    let mut member_images = Vec::with_capacity(m);
    let mut status = Vec::with_capacity(m);
    let mut refs = Vec::with_capacity(n_clusters);
    let mut affines = Array3::<f64>::zeros((m, 2, 3));
    for c in 0..n_clusters {
        // Random tangent frame with a definite out-of-plane component.
        let t: Mat3x2 = [
            [1.0 + 0.2 * rnd(), 0.2 * rnd()],
            [0.2 * rnd(), 1.0 + 0.2 * rnd()],
            [0.6 * rnd(), 0.6 * rnd()],
        ];
        let base = c * members_per_cluster;
        let ref_img = c % n_images;
        let jr = predict(&cams[ref_img], &t);
        // Re-gauge so the reference sees the identity.
        let det = jr[0][0] * jr[1][1] - jr[0][1] * jr[1][0];
        let jr_inv = [
            [jr[1][1] / det, -jr[0][1] / det],
            [-jr[1][0] / det, jr[0][0] / det],
        ];
        for i in 0..members_per_cluster {
            let k = base + i;
            // Distinct pseudo-random member images per cluster (dense
            // cross-view coupling, the shape of real cluster graphs;
            // n_images - 1 must be coprime with the stride multiplier).
            let img = if i == 0 {
                ref_img
            } else {
                (ref_img + 1 + (c * 7 + i * 5) % (n_images - 1)) % n_images
            };
            member_images.push(img as u32);
            if i == 0 {
                status.push(MemberStatus::Reference);
                refs.push(k as u32);
                affines[[k, 0, 0]] = 1.0;
                affines[[k, 1, 1]] = 1.0;
            } else {
                status.push(MemberStatus::Kept);
                let jm = predict(&cams[img], &t);
                for r in 0..2 {
                    for cc2 in 0..2 {
                        affines[[k, r, cc2]] =
                            jm[r][0] * jr_inv[0][cc2] + jm[r][1] * jr_inv[1][cc2];
                    }
                }
            }
        }
        cluster_starts.push((base + members_per_cluster) as u32);
    }
    (cluster_starts, member_images, status, refs, affines)
}

#[test]
fn oracle_cameras_fit_exactly() {
    // With the TRUE cameras, per-cluster tangent solves must reproduce
    // every warp exactly (validates generation + solve_tangent).
    let n_images = 12;
    let mut state = 12345u64;
    let mut rnd = move || {
        let mut s = state;
        let v = noise(&mut s);
        state = s;
        v
    };
    let cams: Vec<Mat2x3> = (0..n_images)
        .map(|_| {
            let (a, b, c) = (0.4 * rnd(), 0.4 * rnd(), 0.4 * rnd());
            let (ca, sa) = (a.cos(), a.sin());
            let (cb, sb) = (b.cos(), b.sin());
            let (cc, sc) = (c.cos(), c.sin());
            [
                [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
            ]
        })
        .collect();
    // One cluster: ref image 0, members 1..3.
    let t: Mat3x2 = [[1.1, 0.1], [-0.05, 0.95], [0.4, -0.3]];
    let jr = predict(&cams[0], &t);
    let det = jr[0][0] * jr[1][1] - jr[0][1] * jr[1][0];
    let jr_inv = [
        [jr[1][1] / det, -jr[0][1] / det],
        [-jr[1][0] / det, jr[0][0] / det],
    ];
    let mut members = vec![FitMember {
        member_index: 0,
        image: 0,
        j: [[1.0, 0.0], [0.0, 1.0]],
    }];
    for i in 1..4u32 {
        let jm = predict(&cams[i as usize], &t);
        let mut j = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                j[r][c] = jm[r][0] * jr_inv[0][c] + jm[r][1] * jr_inv[1][c];
            }
        }
        members.push(FitMember {
            member_index: i,
            image: i,
            j,
        });
    }
    let tsol = solve_tangent(&cams, &members);
    let mut worst = 0.0f64;
    for fm in &members {
        let p = predict(&cams[fm.image as usize], &tsol);
        let d = [
            [p[0][0] - fm.j[0][0], p[0][1] - fm.j[0][1]],
            [p[1][0] - fm.j[1][0], p[1][1] - fm.j[1][1]],
        ];
        worst = worst.max(frob(&d));
    }
    // The ridge (1e-9) biases the exact solve by a comparable amount.
    assert!(worst < 1e-6, "oracle-camera misfit {worst} should be ~0");
}

#[test]
fn consistent_synthetic_scene_has_near_zero_residuals() {
    let (starts, images, status, refs, affines) = synthetic(12, 400, 4);
    let res = warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
    let finite: Vec<f32> = res.iter().copied().filter(|v| v.is_finite()).collect();
    assert_eq!(finite.len(), 400 * 4);
    let median = {
        let mut s = finite.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    assert!(median < 1e-3, "median residual {median} should be ~0");
}

#[test]
fn contaminated_member_scores_highest() {
    let (starts, images, status, refs, mut affines) = synthetic(12, 400, 4);
    // Corrupt one kept member's warp (wrong-match simulation).
    let bad = 4 * 7 + 2;
    affines[[bad, 0, 0]] = -0.3;
    affines[[bad, 0, 1]] = 1.1;
    affines[[bad, 1, 0]] = 0.9;
    affines[[bad, 1, 1]] = 0.4;
    let res = warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
    assert!(
        res[bad] > 0.2,
        "contaminated member residual {} should be large",
        res[bad]
    );
    // Members of untouched clusters stay near zero.
    let clean_max = res
        .iter()
        .enumerate()
        .filter(|(k, v)| *k / 4 != bad / 4 && v.is_finite())
        .map(|(_, v)| *v)
        .fold(0.0f32, f32::max);
    assert!(
        clean_max < 0.05,
        "clean clusters should stay consistent (max {clean_max})"
    );
}

#[test]
fn non_participants_are_nan_and_runs_are_deterministic() {
    let (starts, images, mut status, refs, affines) = synthetic(12, 50, 4);
    // Demote one member: it must come back NaN.
    status[4 * 3 + 1] = MemberStatus::RejectedLowZncc;
    // Demote ALL of cluster 5's non-reference members: with fewer than
    // 2 fitted members the whole cluster leaves the fit, so even its
    // reference is NaN.
    for i in 1..4 {
        status[4 * 5 + i] = MemberStatus::RejectedLowZncc;
    }
    let run = || warp_consistency_residuals(&starts, &images, &status, &refs, affines.view(), 12);
    let a = run();
    let b = run();
    assert!(a[4 * 3 + 1].is_nan());
    for i in 0..4 {
        assert!(a[4 * 5 + i].is_nan(), "member {i} of the dropped cluster");
    }
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b), "runs must be bit-identical");
}
