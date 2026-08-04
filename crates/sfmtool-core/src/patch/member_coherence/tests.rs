// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Decision-rule tests over synthetic pairwise matrices with known block
//! structure, plus an end-to-end matrix build over rendered synthetic imagery.

use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use nalgebra::{Point3, Vector3};

/// Build a matrix from a k×k table of pairwise ZNCC (diagonal ignored).
fn matrix(rows: &[&[f64]]) -> MemberMatrix {
    let k = rows.len();
    let mut flat = Vec::with_capacity(k * k);
    for r in rows {
        assert_eq!(r.len(), k, "matrix must be square");
        flat.extend_from_slice(r);
    }
    MemberMatrix::from_zncc((0..k as u32).collect(), flat)
}

fn defaults() -> MemberCoherenceParams {
    MemberCoherenceParams::default()
}

fn kept_indexes(d: &MemberDecision) -> Vec<usize> {
    d.kept
        .iter()
        .enumerate()
        .filter_map(|(i, &k)| k.then_some(i))
        .collect()
}

fn block_indexes(d: &MemberDecision) -> Vec<usize> {
    d.block
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect()
}

#[test]
fn all_agree_keeps_every_member() {
    let m = matrix(&[&[1.0, 0.90, 0.88], &[0.90, 1.0, 0.92], &[0.88, 0.92, 1.0]]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
    assert_eq!(d.support, 3);
    // Block spans the track, so there is no cross side and no margin.
    assert!(d.margin.is_nan());
    assert!(d.max_cross.is_nan());
}

#[test]
fn clean_three_plus_two_split_evicts_the_minority() {
    // {0,1,2} agree tightly, {3,4} agree tightly, the two sides do not.
    let m = matrix(&[
        &[1.0, 0.95, 0.93, 0.20, 0.18],
        &[0.95, 1.0, 0.94, 0.19, 0.21],
        &[0.93, 0.94, 1.0, 0.22, 0.17],
        &[0.20, 0.19, 0.22, 1.0, 0.91],
        &[0.18, 0.21, 0.17, 0.91, 1.0],
    ]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
    assert_eq!(d.support, 3);
    assert!((d.min_intra - 0.93).abs() < 1e-12);
    assert!((d.max_cross - 0.22).abs() < 1e-12);
    assert!((d.margin - 0.71).abs() < 1e-12);
}

#[test]
fn balanced_two_plus_two_retires_the_point() {
    let m = matrix(&[
        &[1.0, 0.94, 0.15, 0.17],
        &[0.94, 1.0, 0.16, 0.14],
        &[0.15, 0.16, 1.0, 0.96],
        &[0.17, 0.14, 0.96, 1.0],
    ]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Retire);
    // Nothing ships on a retirement; the block is informational only.
    assert!(kept_indexes(&d).is_empty());
    assert_eq!(d.support, 2);
    assert_eq!(block_indexes(&d).len(), 2);
}

#[test]
fn balanced_split_block_choice_is_deterministic_and_tie_broken_on_coherence() {
    // Both blocks have support 2; {2,3} is the tighter one, so it wins the tie
    // regardless of member order, and the choice never depends on iteration.
    let m = matrix(&[
        &[1.0, 0.80, 0.15, 0.17],
        &[0.80, 1.0, 0.16, 0.14],
        &[0.15, 0.16, 1.0, 0.98],
        &[0.17, 0.14, 0.98, 1.0],
    ]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Retire);
    assert_eq!(block_indexes(&d), vec![2, 3]);
    for _ in 0..8 {
        let again = decide_member_coherence(&m, &defaults());
        assert_eq!(block_indexes(&again), vec![2, 3]);
    }
}

#[test]
fn perfectly_symmetric_tie_falls_back_to_the_lowest_member_index() {
    // Two blocks with identical support *and* identical mean coherence: only the
    // index tie-break can separate them.
    let m = matrix(&[
        &[1.0, 0.90, 0.10, 0.10],
        &[0.90, 1.0, 0.10, 0.10],
        &[0.10, 0.10, 1.0, 0.90],
        &[0.10, 0.10, 0.90, 1.0],
    ]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Retire);
    assert_eq!(block_indexes(&d), vec![0, 1]);
}

#[test]
fn drift_chain_is_kept_whole_by_the_margin_gate() {
    // Monotone appearance decay along the member order: every consecutive pair
    // agrees, the far ends do not. The max-support block's weakest internal link
    // sits right next to its strongest external one, so there is no gap to cut.
    let k = 6usize;
    let mut flat = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let d = (i as f64 - j as f64).abs();
            flat[i * k + j] = 0.99 - 0.16 * d;
        }
    }
    let m = MemberMatrix::from_zncc((0..k as u32).collect(), flat);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(kept_indexes(&d), (0..k).collect::<Vec<_>>());
    // The gate, not block size, is what saved it: the block was a strict subset.
    assert!(d.support < k as u32);
    assert!(d.margin <= defaults().margin_gate);
}

#[test]
fn margin_gate_is_the_only_thing_between_keep_and_split() {
    // Same matrix, two gates: a hair of separation splits when the gate allows it.
    let m = matrix(&[
        &[1.0, 0.90, 0.88, 0.86],
        &[0.90, 1.0, 0.89, 0.85],
        &[0.88, 0.89, 1.0, 0.84],
        &[0.86, 0.85, 0.84, 0.30],
    ]);
    let strict = MemberCoherenceParams {
        bar: 0.87,
        margin_gate: 0.05,
        ..defaults()
    };
    assert_eq!(
        decide_member_coherence(&m, &strict).verdict,
        MemberVerdict::KeepAll
    );
    let lax = MemberCoherenceParams {
        margin_gate: 0.0,
        ..strict
    };
    let d = decide_member_coherence(&m, &lax);
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
}

#[test]
fn three_member_minority_block_retires_and_majority_block_splits() {
    // k = 3: a 2-block is a strict majority (splits); a lone member is not.
    let split = matrix(&[&[1.0, 0.95, 0.10], &[0.95, 1.0, 0.12], &[0.10, 0.12, 1.0]]);
    let d = decide_member_coherence(&split, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1]);

    // Every member isolated: the winning block is a singleton, its margin is
    // undefined, and the track is kept whole rather than cut arbitrarily.
    let isolated = matrix(&[&[1.0, 0.10, 0.12], &[0.10, 1.0, 0.11], &[0.12, 0.11, 1.0]]);
    let d = decide_member_coherence(&isolated, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(d.support, 1);
    assert!(d.margin.is_nan());
}

#[test]
fn unscoreable_pairs_do_not_join_a_block() {
    // Member 3 could not be rendered: NaN row/column. It supports only itself,
    // so it falls outside the winning block and is evicted by the split.
    let m = matrix(&[
        &[1.0, 0.95, 0.93, f64::NAN],
        &[0.95, 1.0, 0.94, f64::NAN],
        &[0.93, 0.94, 1.0, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN, 1.0],
    ]);
    assert_eq!(m.scored, vec![true, true, true, false]);
    let d = decide_member_coherence(&m, &defaults());
    // No finite cross link, so the margin is undefined and the track is kept
    // whole: an unrenderable member is missing evidence, not contrary evidence.
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert!(d.max_cross.is_nan());
}

#[test]
fn two_member_and_empty_tracks_are_always_kept() {
    for z in [0.99, 0.10] {
        let m = matrix(&[&[1.0, z], &[z, 1.0]]);
        assert_eq!(
            decide_member_coherence(&m, &defaults()).verdict,
            MemberVerdict::KeepAll
        );
    }
    let empty = MemberMatrix::from_zncc(Vec::new(), Vec::new());
    let d = decide_member_coherence(&empty, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert!(d.kept.is_empty());
}

#[test]
fn bar_selects_which_members_form_the_block() {
    let m = matrix(&[
        &[1.0, 0.75, 0.72, 0.20],
        &[0.75, 1.0, 0.74, 0.18],
        &[0.72, 0.74, 1.0, 0.19],
        &[0.20, 0.18, 0.19, 1.0],
    ]);
    // At the default bar every 0.7x link is an edge -> {0,1,2} splits off member 3.
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
    // Raise the bar past those links and no member has a partner: all blocks are
    // singletons, the margin is undefined, and nothing is cut.
    let high = MemberCoherenceParams {
        bar: 0.80,
        ..defaults()
    };
    let d = decide_member_coherence(&m, &high);
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(d.support, 1);
}

// ---------------------------------------------------------------------------
// End-to-end: render a synthetic scene and build the matrix through the shared
// view-selection machinery. Same scene shape as the view-selection tests —
// pinhole cameras (rotated 180° about X so the canonical −Z-forward camera looks
// down world +z) viewing a textured plane at z = PLANE_Z.
// ---------------------------------------------------------------------------

const PLANE_Z: f64 = 4.0;
const IMG_W: u32 = 320;
const IMG_H: u32 = 240;
const FOCAL: f64 = 260.0;

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

fn surface_a(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// A different surface — a member showing this disagrees photometrically.
fn surface_b(x: f64, y: f64) -> f64 {
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
            let lambda = PLANE_Z - center[2];
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
    fn new(centers: &[[f64; 3]], texs: &[fn(f64, f64) -> f64]) -> Self {
        let cams = centers.iter().map(|_| pinhole()).collect();
        let poses = centers
            .iter()
            .map(|c| {
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

/// Patch on the plane, normal toward the cameras (-z).
fn plane_patch() -> OrientedPatch {
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.4, 0.4],
    )
}

fn render_params() -> MemberCoherenceParams {
    MemberCoherenceParams {
        resolution: 15,
        min_valid_fraction: 0.5,
        ..MemberCoherenceParams::default()
    }
}

#[test]
fn rendered_matrix_is_symmetric_and_separates_the_odd_member_out() {
    // Members 0/1/2 image one surface; member 3 images another.
    let scene = Scene::new(
        &[
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.4],
            [-0.6, 0.2, 0.3],
            [0.3, -0.5, 0.2],
        ],
        &[surface_a, surface_a, surface_a, surface_b],
    );
    let views = scene.views();
    let m = member_zncc_matrix(&plane_patch(), &views, &[0, 1, 2, 3], &render_params());

    assert_eq!(m.len(), 4);
    assert!(m.scored.iter().all(|&s| s), "every member should render");
    for i in 0..4 {
        assert_eq!(m.get(i, i), 1.0);
        for j in 0..4 {
            assert_eq!(m.get(i, j), m.get(j, i), "matrix must be symmetric");
        }
    }
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert!(
            m.get(i, j) > 0.9,
            "same-surface pair {i},{j}: {}",
            m.get(i, j)
        );
    }
    for i in 0..3 {
        assert!(
            m.get(i, 3) < 0.6,
            "cross-surface pair {i},3 should be weak: {}",
            m.get(i, 3)
        );
    }

    // And the decision the matrix drives: a 3-of-4 majority evicts member 3.
    let d = decide_member_coherence(&m, &render_params());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
}

#[test]
fn one_surface_seen_by_every_member_is_kept_whole() {
    let scene = Scene::new(
        &[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4], [-0.6, 0.2, 0.3]],
        &[surface_a, surface_a, surface_a],
    );
    let views = scene.views();
    let out = validate_member_coherence(&plane_patch(), &views, &[0, 1, 2], &render_params());
    assert_eq!(out.decision.verdict, MemberVerdict::KeepAll);
    assert_eq!(out.decision.support, 3);
}

#[test]
fn balanced_two_surface_track_retires_end_to_end() {
    let scene = Scene::new(
        &[
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.4],
            [-0.6, 0.2, 0.3],
            [0.3, -0.5, 0.2],
        ],
        &[surface_a, surface_a, surface_b, surface_b],
    );
    let views = scene.views();
    let out = validate_member_coherence(&plane_patch(), &views, &[0, 1, 2, 3], &render_params());
    assert_eq!(out.decision.verdict, MemberVerdict::Retire);
    assert_eq!(out.decision.support, 2);
    assert!(out.decision.kept.iter().all(|&k| !k));
}

#[test]
fn duplicate_members_are_deduplicated_before_the_matrix() {
    let scene = Scene::new(&[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4]], &[surface_a, surface_a]);
    let views = scene.views();
    let m = member_zncc_matrix(&plane_patch(), &views, &[1, 0, 1, 0], &render_params());
    assert_eq!(m.members, vec![1, 0]);
    assert_eq!(m.len(), 2);
}
