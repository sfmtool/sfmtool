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
fn an_unscored_member_sits_outside_the_rule_entirely() {
    // Member 3 could not be rendered: NaN row/column, so it carries no pairwise
    // evidence. The three that did render agree, so they are the whole block —
    // the rule runs over them alone, the track is kept whole, and member 3 is
    // neither in the block nor counted against it.
    let m = matrix(&[
        &[1.0, 0.95, 0.93, f64::NAN],
        &[0.95, 1.0, 0.94, f64::NAN],
        &[0.93, 0.94, 1.0, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN, 1.0],
    ]);
    assert_eq!(m.scored, vec![true, true, true, false]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2, 3]);
    // Support and block are over the scored members: 3 of them, all in.
    assert_eq!(d.support, 3);
    assert_eq!(block_indexes(&d), vec![0, 1, 2]);
    assert!(d.margin.is_nan());
    assert!(d.max_cross.is_nan());
}

#[test]
fn a_split_evicts_the_outlier_and_keeps_the_unscored_member() {
    // {0,1,2} agree, member 3 is a scored outlier, member 4 never rendered. The
    // cut takes 3 (there is evidence against it) and passes 4 through: an
    // unscored member is missing evidence, not contrary evidence, so a rule it
    // took no part in cannot evict it.
    let m = matrix(&[
        &[1.0, 0.95, 0.93, 0.20, f64::NAN],
        &[0.95, 1.0, 0.94, 0.19, f64::NAN],
        &[0.93, 0.94, 1.0, 0.22, f64::NAN],
        &[0.20, 0.19, 0.22, 1.0, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0],
    ]);
    assert_eq!(m.scored, vec![true, true, true, true, false]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2, 4]);
    assert_eq!(block_indexes(&d), vec![0, 1, 2]);
    assert_eq!(d.support, 3);
}

#[test]
fn unscored_members_do_not_dilute_the_majority() {
    // A clean 2-of-3 split among the members that scored, plus two that did not.
    // The majority is taken over the scored members (2 of 3 -> Split); counting
    // the unscored pair in the denominator would sink it to 2 of 5 and retire a
    // point on evidence that does not exist.
    let m = matrix(&[
        &[1.0, 0.95, 0.20, f64::NAN, f64::NAN],
        &[0.95, 1.0, 0.18, f64::NAN, f64::NAN],
        &[0.20, 0.18, 1.0, f64::NAN, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN, 1.0, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0],
    ]);
    assert_eq!(m.scored, vec![true, true, true, false, false]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 3, 4]);
    assert_eq!(block_indexes(&d), vec![0, 1]);
    assert_eq!(d.support, 2);
}

#[test]
fn a_track_with_no_pairwise_evidence_is_kept_whole() {
    // Nothing rendered: no hypothesis, no block, no margin — and no cut.
    let nan = f64::NAN;
    let m = matrix(&[&[1.0, nan, nan], &[nan, 1.0, nan], &[nan, nan, 1.0]]);
    assert_eq!(m.scored, vec![false, false, false]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
    assert_eq!(d.support, 0);
    assert!(block_indexes(&d).is_empty());
    assert!(d.margin.is_nan() && d.min_intra.is_nan() && d.max_cross.is_nan());
}

#[test]
fn one_unscoreable_pair_does_not_unscore_its_members() {
    // Members 1 and 2 could not be correlated with *each other* (their supports
    // met nowhere) but both correlate with 0, so both carry evidence and both are
    // in play. The missing entry is skipped by the margin, not treated as a
    // disagreement. Values are exact binary fractions so the margin is exact.
    let nan = f64::NAN;
    let m = matrix(&[
        &[1.0, 0.75, 0.75, 0.25],
        &[0.75, 1.0, nan, 0.25],
        &[0.75, nan, 1.0, 0.25],
        &[0.25, 0.25, 0.25, 1.0],
    ]);
    assert_eq!(m.scored, vec![true, true, true, true]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2]);
    assert_eq!(d.min_intra, 0.75);
    assert_eq!(d.max_cross, 0.25);
    assert_eq!(d.margin, 0.5);
}

#[test]
fn a_two_of_four_block_retires_at_the_majority_boundary() {
    // 2 + 1 + 1: {0,1} agree, members 2 and 3 agree with nobody — including each
    // other. The block is still exactly half the track (2s == k), so it is not a
    // majority and the point ships nothing, however isolated the other two are.
    let m = matrix(&[
        &[1.0, 0.75, 0.25, 0.25],
        &[0.75, 1.0, 0.25, 0.25],
        &[0.25, 0.25, 1.0, 0.25],
        &[0.25, 0.25, 0.25, 1.0],
    ]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Retire);
    assert_eq!(d.support, 2);
    assert_eq!(block_indexes(&d), vec![0, 1]);
    assert!(kept_indexes(&d).is_empty());
    assert_eq!(d.margin, 0.5);
}

#[test]
fn the_bar_and_the_margin_gate_are_inclusive_at_their_boundaries() {
    // Exact binary fractions: no float residue anywhere near the comparisons.
    // 0-1 sits *exactly* on the bar, everything else exactly at 0.25.
    let m = matrix(&[&[1.0, 0.5, 0.25], &[0.5, 1.0, 0.25], &[0.25, 0.25, 1.0]]);
    // `>= bar` puts the 0.5 link in the graph, so {0,1} is a block of two and the
    // margin is exactly 0.5 - 0.25.
    let at_gate = MemberCoherenceParams {
        bar: 0.5,
        margin_gate: 0.25,
        ..defaults()
    };
    let d = decide_member_coherence(&m, &at_gate);
    assert_eq!(d.margin, 0.25);
    // `margin <= margin_gate` refuses: a margin exactly at the gate is not past it.
    assert_eq!(d.verdict, MemberVerdict::KeepAll);

    let below_gate = MemberCoherenceParams {
        margin_gate: 0.125,
        ..at_gate
    };
    let d = decide_member_coherence(&m, &below_gate);
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1]);
    assert_eq!(d.margin, 0.25);

    // And a hair above the bar's own boundary there is no block at all: the 0.5
    // link is the only candidate edge in the matrix.
    let above_bar = MemberCoherenceParams {
        bar: 0.5000000000000001,
        ..below_gate
    };
    let d = decide_member_coherence(&m, &above_bar);
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(d.support, 1);
}

#[test]
fn a_hand_built_matrix_reports_no_rendered_support() {
    let m = matrix(&[&[1.0, 0.90], &[0.90, 1.0]]);
    assert_eq!(m.n_support, 0);
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

/// No texture at all — a blown highlight, a patch of sky, a saturated sensor.
/// Nothing can be correlated with it.
fn surface_flat(_x: f64, _y: f64) -> f64 {
    200.0
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
    let m = member_zncc_matrix(
        &plane_patch(),
        &views,
        &[0, 1, 2, 3],
        None,
        &render_params(),
    );

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
    let out = validate_member_coherence(&plane_patch(), &views, &[0, 1, 2], None, &render_params());
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
    let out = validate_member_coherence(
        &plane_patch(),
        &views,
        &[0, 1, 2, 3],
        None,
        &render_params(),
    );
    assert_eq!(out.decision.verdict, MemberVerdict::Retire);
    assert_eq!(out.decision.support, 2);
    assert!(out.decision.kept.iter().all(|&k| !k));
}

#[test]
fn a_textureless_member_is_unscored_and_the_rest_still_score() {
    // Member 3 sees a blown-out, textureless surface. The shared z-normalization
    // drops a channel that is flat in ANY member, for EVERY member, which here
    // would flatten the only channel and leave the whole track unscored. Member 3
    // is dropped from the stack first instead, so the three members that do carry
    // texture still score against each other.
    let scene = Scene::new(
        &[
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.4],
            [-0.6, 0.2, 0.3],
            [0.3, -0.5, 0.2],
        ],
        &[surface_a, surface_a, surface_a, surface_flat],
    );
    let views = scene.views();
    let out = validate_member_coherence(
        &plane_patch(),
        &views,
        &[0, 1, 2, 3],
        None,
        &render_params(),
    );

    assert_eq!(out.matrix.scored, vec![true, true, true, false]);
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert!(
            out.matrix.get(i, j) > 0.9,
            "textured pair {i},{j}: {}",
            out.matrix.get(i, j)
        );
    }
    for i in 0..3 {
        assert!(out.matrix.get(i, 3).is_nan(), "pair {i},3 must be unscored");
    }
    // The verdict is the three scored members', and the flat one rides along.
    assert_eq!(out.decision.verdict, MemberVerdict::KeepAll);
    assert_eq!(out.decision.support, 3);
    assert!(out.decision.kept.iter().all(|&k| k));
}

#[test]
fn the_common_support_is_reported_and_min_support_pixels_gates_on_it() {
    let scene = Scene::new(
        &[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4], [-0.6, 0.2, 0.3]],
        &[surface_a, surface_a, surface_a],
    );
    let views = scene.views();
    let params = render_params();
    let m = member_zncc_matrix(&plane_patch(), &views, &[0, 1, 2], None, &params);
    let n = m.n_support;
    // A real support: past the shared floor, and short of the full grid (the
    // gaussian-disk window clips the corners).
    assert!(n >= MIN_MASK_PIXELS as u32, "n_support {n}");
    assert!(n < params.resolution * params.resolution, "n_support {n}");
    assert!(m.scored.iter().all(|&s| s));

    // One pixel more than the track can offer: nothing is correlated, the count
    // is still reported, and the fail-open verdict keeps every member.
    let strict = MemberCoherenceParams {
        min_support_pixels: n + 1,
        ..params
    };
    let gated = member_zncc_matrix(&plane_patch(), &views, &[0, 1, 2], None, &strict);
    assert_eq!(gated.n_support, n);
    assert!(gated.scored.iter().all(|&s| !s));
    assert!(gated.get(0, 1).is_nan());
    let d = decide_member_coherence(&gated, &strict);
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    assert_eq!(d.support, 0);
    assert!(d.kept.iter().all(|&k| k));
}

#[test]
fn duplicate_members_are_deduplicated_before_the_matrix() {
    let scene = Scene::new(&[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4]], &[surface_a, surface_a]);
    let views = scene.views();
    let m = member_zncc_matrix(
        &plane_patch(),
        &views,
        &[1, 0, 1, 0],
        None,
        &render_params(),
    );
    assert_eq!(m.members, vec![1, 0]);
    assert_eq!(m.len(), 2);
}

/// The pixel `patch`'s centre reprojects to in `view` — where projection
/// anchoring samples the member.
fn project_center(patch: &OrientedPatch, view: &ProjectedImage<'_>) -> [f64; 2] {
    let p = view
        .cam_from_world
        .transform_point_homogeneous(patch.center.coords, patch.w);
    let (x, y) = view
        .camera
        .ray_to_pixel([p.x, p.y, p.z])
        .expect("patch centre projects");
    [x, y]
}

#[test]
fn keypoints_at_the_projections_reproduce_the_unanchored_matrix() {
    // Anchoring is a recentring: hand it the reprojections themselves and the
    // recentred patch IS the patch, so the whole matrix must come back unchanged.
    // This is what makes the parameter a strict generalization rather than a
    // second render path.
    let scene = Scene::new(
        &[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4], [-0.6, 0.2, 0.3]],
        &[surface_a, surface_a, surface_a],
    );
    let views = scene.views();
    let patch = plane_patch();
    let members = [0u32, 1, 2];
    let kps: Vec<Option<[f64; 2]>> = members
        .iter()
        .map(|&i| Some(project_center(&patch, &views[i as usize])))
        .collect();

    let plain = member_zncc_matrix(&patch, &views, &members, None, &render_params());
    let anchored = member_zncc_matrix(&patch, &views, &members, Some(&kps), &render_params());
    assert_eq!(anchored.n_support, plain.n_support);
    assert_eq!(anchored.scored, plain.scored);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (anchored.get(i, j) - plain.get(i, j)).abs() < 1e-9,
                "pair {i},{j}: anchored {} vs plain {}",
                anchored.get(i, j),
                plain.get(i, j)
            );
        }
    }
}

#[test]
fn a_member_with_a_reprojection_residual_recovers_when_anchored_at_its_keypoint() {
    // Member 2's POSE carries a lateral error while its image does not: the point
    // reprojects a few pixels off the content the matcher actually matched. That
    // is a geometric defect, but projection anchoring samples the member at the
    // wrong place and charges it to the photometry. Anchoring at the stored
    // keypoint samples the content instead, and the score comes back.
    let scene = Scene::new(
        &[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4], [-0.6, 0.2, 0.3]],
        &[surface_a, surface_a, surface_a],
    );
    let truth = scene.views();
    let patch = plane_patch();
    // Where member 2's feature really is (its stored keypoint).
    let kp2 = project_center(&patch, &truth[2]);

    // Same camera, same image, centre displaced by 0.046 world units ~ 3 px at
    // this depth — ~11% of the patch's 26 px half-width.
    let bad_pose =
        RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.6 - 0.046, 0.2, 0.3]);
    let mut views = scene.views();
    views[2].cam_from_world = &bad_pose;
    let moved = project_center(&patch, &views[2]);
    let residual = (moved[0] - kp2[0]).hypot(moved[1] - kp2[1]);
    assert!(
        (2.0..5.0).contains(&residual),
        "test setup: residual {residual} px"
    );

    let members = [0u32, 1, 2];
    let plain = member_zncc_matrix(&patch, &views, &members, None, &render_params());
    let kps = [None, None, Some(kp2)];
    let anchored = member_zncc_matrix(&patch, &views, &members, Some(&kps), &render_params());

    // Members 0 and 1 are untouched by the anchoring (no keypoint given).
    assert!(plain.get(0, 1) > 0.9, "control pair: {}", plain.get(0, 1));
    // The misaligned member is deflated at its projection...
    assert!(
        plain.get(0, 2) < 0.8,
        "projection-anchored 0,2 should be depressed: {}",
        plain.get(0, 2)
    );
    // ...and recovers once it is sampled where its feature is.
    assert!(
        anchored.get(0, 2) > plain.get(0, 2) + 0.1,
        "keypoint anchoring should raise 0,2: {} -> {}",
        plain.get(0, 2),
        anchored.get(0, 2)
    );
    assert!(
        anchored.get(1, 2) > plain.get(1, 2) + 0.1,
        "keypoint anchoring should raise 1,2: {} -> {}",
        plain.get(1, 2),
        anchored.get(1, 2)
    );

    // Which is the calibration statement: the track's weakest link — the quantity
    // `bar` is compared against — moves UP under anchoring, so a bar picked on
    // projection-anchored scores sits lower against these than it did against
    // those.
    let weakest = |m: &MemberMatrix| {
        (0..m.len())
            .flat_map(|i| (0..m.len()).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| m.get(i, j))
            .fold(f64::INFINITY, f64::min)
    };
    assert!(
        weakest(&anchored) > weakest(&plain) + 0.1,
        "weakest link {} -> {}",
        weakest(&plain),
        weakest(&anchored)
    );
}

#[test]
fn member_keypoints_are_deduplicated_alongside_their_members() {
    // The keypoint slice is parallel to the INPUT member list, so the dedup has to
    // carry it: a duplicated member must keep the keypoint of its first
    // occurrence, not slide onto its neighbour's.
    let scene = Scene::new(&[[0.0, 0.0, 0.0], [0.6, 0.0, 0.4]], &[surface_a, surface_a]);
    let views = scene.views();
    let patch = plane_patch();
    let k0 = Some(project_center(&patch, &views[0]));
    let k1 = Some(project_center(&patch, &views[1]));

    let dup = member_zncc_matrix(
        &patch,
        &views,
        &[1, 0, 1, 0],
        Some(&[k1, k0, k1, k0]),
        &render_params(),
    );
    let plain = member_zncc_matrix(&patch, &views, &[1, 0], Some(&[k1, k0]), &render_params());
    assert_eq!(dup.members, vec![1, 0]);
    assert!((dup.get(0, 1) - plain.get(0, 1)).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// The self-normalized admission bar.
// ---------------------------------------------------------------------------

/// `defaults()` with the relative term disabled â€” the absolute rule.
fn absolute() -> MemberCoherenceParams {
    MemberCoherenceParams {
        self_bar_k: 0.0,
        ..MemberCoherenceParams::default()
    }
}

/// A tight core of `n` members agreeing at ~0.99, plus one outsider (member 0)
/// that correlates `outside` against every one of them â€” the occluding-member
/// shape, where the whole block structure sits above the absolute bar.
fn tight_core_plus_outsider(n: usize, outside: f64) -> MemberMatrix {
    let k = n + 1;
    let mut z = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            z[a * k + b] = if a == b {
                1.0
            } else if a == 0 || b == 0 {
                outside
            } else {
                // A little structure, so the core is not exactly uniform.
                0.99 - 0.002 * ((a + b) % 3) as f64
            };
        }
    }
    MemberMatrix::from_zncc((0..k as u32).collect(), z)
}

#[test]
fn self_bar_k_zero_reproduces_the_absolute_rule_exactly() {
    // Parity across every branch of the rule: the occluding shape the relative
    // term exists for, a clean split, a balanced retirement, and a drift chain.
    let cases = [
        (tight_core_plus_outsider(8, 0.90), MemberVerdict::KeepAll),
        (tight_core_plus_outsider(3, 0.90), MemberVerdict::KeepAll),
        (
            matrix(&[
                &[1.0, 0.95, 0.93, 0.20, 0.18],
                &[0.95, 1.0, 0.94, 0.19, 0.21],
                &[0.93, 0.94, 1.0, 0.22, 0.17],
                &[0.20, 0.19, 0.22, 1.0, 0.91],
                &[0.18, 0.21, 0.17, 0.91, 1.0],
            ]),
            MemberVerdict::Split,
        ),
        (
            matrix(&[
                &[1.0, 0.94, 0.15, 0.17],
                &[0.94, 1.0, 0.16, 0.14],
                &[0.15, 0.16, 1.0, 0.96],
                &[0.17, 0.14, 0.96, 1.0],
            ]),
            MemberVerdict::Retire,
        ),
    ];
    for (m, want) in &cases {
        let d = decide_member_coherence(m, &absolute());
        assert_eq!(d.verdict, *want);
        // A disabled relative term reports the absolute thresholds it really
        // swept at, and no statistics at all.
        assert_eq!(d.effective_bar, absolute().bar);
        assert_eq!(d.effective_margin_gate, absolute().margin_gate);
        assert!(d.core_center.is_nan() && d.core_scatter.is_nan());
    }
}

#[test]
fn a_monotone_drift_chain_is_untouched_by_the_relative_bar() {
    // The banded shape the margin gate exists for, long enough (7 members, 21
    // pairs) that the relative term does estimate: its margin is negative, so
    // no threshold on the admission side can turn it into a cut.
    let k = 7;
    let mut z = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            z[a * k + b] = 1.0 - 0.14 * (a as f64 - b as f64).abs();
        }
    }
    let m = MemberMatrix::from_zncc((0..k as u32).collect(), z);
    for p in [absolute(), defaults()] {
        let d = decide_member_coherence(&m, &p);
        assert_eq!(d.verdict, MemberVerdict::KeepAll);
        assert!(kept_indexes(&d).len() == k);
    }
}

#[test]
fn the_relative_bar_engages_on_a_tight_core_and_evicts_the_outsider() {
    // 8 core members at ~0.99 and one outsider at 0.90 against all of them:
    // every link is far above the absolute 0.65, so the absolute rule keeps the
    // track whole and only the self-normalized one cuts it.
    let m = tight_core_plus_outsider(8, 0.90);
    assert_eq!(
        decide_member_coherence(&m, &absolute()).verdict,
        MemberVerdict::KeepAll
    );

    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), (1..9).collect::<Vec<_>>());
    assert!(
        d.effective_bar > defaults().bar && d.effective_bar <= SELF_BAR_CEILING,
        "effective bar {}",
        d.effective_bar
    );
    // The bar really is centre minus k units of scatter.
    let expect = d.core_center - defaults().self_bar_k * d.core_scatter;
    assert!((d.effective_bar - expect).abs() < 1e-12);
    assert!(d.core_center > 0.98 && d.core_scatter >= SELF_BAR_MIN_SCATTER);
    // The margin floor moved with the bar: 0.99 against 0.90 is a real gap in
    // the core's own units and nowhere near the absolute 0.05.
    assert!(d.effective_margin_gate < defaults().margin_gate);
    assert!(d.margin > d.effective_margin_gate);
}

#[test]
fn a_wide_scatter_track_collapses_back_to_the_absolute_thresholds() {
    // Two populations 0.23 apart inside one block: the intra-pair scatter is
    // large, centre minus k units of it falls under the absolute bar, and the
    // rule is the absolute one â€” verdict, membership and thresholds alike.
    let k = 8;
    let mut z = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            z[a * k + b] = if a == b {
                1.0
            } else if (a < 4) == (b < 4) {
                0.95
            } else {
                0.72
            };
        }
    }
    let m = MemberMatrix::from_zncc((0..k as u32).collect(), z);
    let d = decide_member_coherence(&m, &defaults());
    let base = decide_member_coherence(&m, &absolute());
    assert_eq!(d.verdict, base.verdict);
    assert_eq!(d.kept, base.kept);
    assert_eq!(d.effective_bar, defaults().bar);
    assert_eq!(d.effective_margin_gate, defaults().margin_gate);
    // The statistics are still reported: the term ran and collapsed, which is a
    // different thing from never having run.
    assert!(
        d.core_scatter > defaults().margin_gate,
        "{}",
        d.core_scatter
    );
}

#[test]
fn small_blocks_leave_the_relative_term_inactive() {
    // Below SELF_BAR_MIN_PAIRS the centre and its quartile distance are read off
    // two or three numbers, so nothing is estimated. A 3-member block is 3 pairs
    // and stays inactive; a 4-member block is exactly 6 and estimates.
    {
        let m = tight_core_plus_outsider(2, 0.90);
        let d = decide_member_coherence(&m, &defaults());
        assert!(
            d.core_center.is_nan() && d.core_scatter.is_nan(),
            "k = {} must leave the relative term inactive",
            m.len()
        );
        assert_eq!(d.effective_bar, defaults().bar);
        assert_eq!(d.effective_margin_gate, defaults().margin_gate);
        assert_eq!(d.verdict, decide_member_coherence(&m, &absolute()).verdict);
    }
    let m = tight_core_plus_outsider(3, 0.90);
    assert_eq!(m.len(), 4);
    let d = decide_member_coherence(&m, &defaults());
    assert!(d.core_center.is_finite() && d.core_scatter.is_finite());
    assert!(d.effective_bar > defaults().bar);
}

#[test]
fn the_ceiling_stops_a_perfect_core_demanding_perfection() {
    // Every core pair exactly 1.0: the measured spread is zero, the scatter
    // floor is the smallest dispersion the rule will believe, and the ceiling
    // caps whatever the two would otherwise ask of a newcomer.
    let k = 7;
    let mut z = vec![1.0; k * k];
    for a in 1..k {
        z[a] = 0.80;
        z[a * k] = 0.80;
    }
    let m = MemberMatrix::from_zncc((0..k as u32).collect(), z);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.core_center, 1.0);
    assert_eq!(d.core_scatter, SELF_BAR_MIN_SCATTER);
    assert!(d.effective_bar <= SELF_BAR_CEILING);
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), (1..k).collect::<Vec<_>>());

    // With a vanishing k the floor alone would put the bar on the centre; the
    // ceiling is what stops it.
    let greedy = MemberCoherenceParams {
        self_bar_k: 1e-9,
        ..MemberCoherenceParams::default()
    };
    assert_eq!(
        decide_member_coherence(&m, &greedy).effective_bar,
        SELF_BAR_CEILING
    );
}

#[test]
fn the_scatter_is_one_sided_so_contamination_cannot_set_its_own_bar() {
    // One core, two outsiders below it. A two-sided dispersion widens with the
    // second outsider â€” letting the members under suspicion loosen the bar that
    // is meant to exclude them. The upper-half one does not move, because the
    // contamination lives entirely below the centre.
    // One nine-member core, three contaminated members in the same block, all
    // three at `depth` — the only thing that varies.
    let contaminated = |depth: f64| {
        let k = 12;
        let mut z = vec![0.0; k * k];
        for a in 0..k {
            for b in 0..k {
                z[a * k + b] = if a == b {
                    1.0
                } else if a < 3 || b < 3 {
                    depth
                } else {
                    0.99 - 0.004 * ((a + b) % 4) as f64
                };
            }
        }
        MemberMatrix::from_zncc((0..k as u32).collect(), z)
    };
    let all = vec![true; 12];
    let want = core_coherence(&contaminated(0.88).zncc, 12, &all).unwrap();
    assert!(
        want.0 > 0.97,
        "the centre sits in the core, not between: {}",
        want.0
    );
    for depth in [0.80, 0.50, 0.20, -0.30] {
        let m = contaminated(depth);
        let (c, s) = core_coherence(&m.zncc, 12, &all).unwrap();
        // However far below the centre the contamination falls, neither the
        // centre nor the scatter moves: the estimator never reads that tail.
        assert!((c - want.0).abs() < 1e-12, "centre {c} at depth {depth}");
        assert!((s - want.1).abs() < 1e-12, "scatter {s} at depth {depth}");
    }
    // A plain standard deviation over the same sample is the thing this avoids:
    // it widens with the contamination and would hand the members under
    // suspicion the power to loosen the bar meant to exclude them.
    let sd = |m: &MemberMatrix| {
        let v: Vec<f64> = (0..12)
            .flat_map(|a| ((a + 1)..12).map(move |b| (a, b)))
            .map(|(a, b)| m.get(a, b))
            .collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    assert!(sd(&contaminated(0.20)) > 4.0 * sd(&contaminated(0.88)));
    // A block of one member has no pairs at all, so there is nothing to read.
    let mut lone = vec![false; 12];
    lone[0] = true;
    assert!(core_coherence(&contaminated(0.88).zncc, 12, &lone).is_none());
}

#[test]
fn the_tighten_pass_runs_once_and_is_deterministic() {
    // Repeated evaluation is bit-identical, and the block landed on is the one a
    // SINGLE re-sweep at the tightened bar gives â€” not the fixed point of
    // tightening off successive survivors.
    let m = tight_core_plus_outsider(8, 0.90);
    let first = decide_member_coherence(&m, &defaults());
    for _ in 0..8 {
        let again = decide_member_coherence(&m, &defaults());
        assert_eq!(again.effective_bar, first.effective_bar);
        assert_eq!(again.core_center, first.core_center);
        assert_eq!(again.core_scatter, first.core_scatter);
        assert_eq!(again.kept, first.kept);
    }
    let k = m.len();
    let pass1 = max_support_block(&m.zncc, k, defaults().bar);
    assert!(pass1.iter().all(|&b| b), "pass 1 admits the whole track");
    let (c, sigma) = core_coherence(&m.zncc, k, &pass1).unwrap();
    let bar = defaults().bar.max(c - defaults().self_bar_k * sigma);
    assert_eq!(bar, first.effective_bar);
    assert_eq!(max_support_block(&m.zncc, k, bar), first.block);
}

// ---------------------------------------------------------------------------
// Multi-scale exoneration.
// ---------------------------------------------------------------------------

/// `tight_core_plus_outsider` again, with a second table standing in for the
/// half-scale measurement: the outsider correlates `outside_coarse` there
/// instead, so the retained-deficit ratio can be dialled directly. The core is
/// identical at both scales, which is what a real core does — coarsening two
/// renders of the same surface does not change how well they agree.
fn two_scale_outsider(n: usize, outside: f64, outside_coarse: f64) -> MemberMatrix {
    let fine = tight_core_plus_outsider(n, outside);
    let coarse = tight_core_plus_outsider(n, outside_coarse);
    MemberMatrix::from_zncc_scales(fine.members.clone(), fine.zncc, vec![coarse.zncc], vec![2])
}

/// The retained-deficit ratio `two_scale_outsider` produces, straight off the
/// definition, so the tests can name the quantity they are dialling.
fn ratio_of(m: &MemberMatrix) -> f64 {
    let k = m.len();
    let block: Vec<bool> = (0..k).map(|i| i != 0).collect();
    core_deficit(&m.zncc_coarse[0], k, &block, 0) / core_deficit(&m.zncc, k, &block, 0)
}

#[test]
fn a_structural_outsider_keeps_its_deficit_across_the_halving_and_is_evicted() {
    // The disagreement survives coarsening — the member's low frequencies are
    // already the wrong content — so exoneration must not reach it.
    let m = two_scale_outsider(8, 0.90, 0.90);
    let r = ratio_of(&m);
    assert!(r > 0.99, "an unchanged deficit is fully retained: {r}");
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), (1..9).collect::<Vec<_>>());
    assert!(
        d.relative_flagged[0],
        "the relative term is what flagged it"
    );
    assert!(!d.exonerated[0]);
    assert!((d.retained_deficit[0] - r).abs() < 1e-12);
}

#[test]
fn a_spectral_outsider_loses_its_deficit_across_the_halving_and_is_spared() {
    // Same full-scale score, but the disagreement is made of the fine detail:
    // at half scale the member is back inside its core. It ships.
    let m = two_scale_outsider(8, 0.90, 0.988);
    let r = ratio_of(&m);
    assert!(r < 0.5, "most of the deficit evaporated: {r}");
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(
        d.verdict,
        MemberVerdict::KeepAll,
        "the whole rejected side was spared, so there is no cut left"
    );
    assert_eq!(kept_indexes(&d), (0..9).collect::<Vec<_>>());
    assert!(d.relative_flagged[0]);
    assert!(d.exonerated[0]);
    // The block still reports the cut the sweep proposed: exoneration spares a
    // member, it does not re-run the sweep.
    assert_eq!(block_indexes(&d), (1..9).collect::<Vec<_>>());
}

#[test]
fn the_exoneration_ratio_is_the_only_thing_between_evicting_and_sparing() {
    // One matrix, one knob: the verdict flips exactly at the member's own ratio,
    // and the comparison is inclusive at the threshold.
    let m = two_scale_outsider(8, 0.90, 0.95);
    let r = ratio_of(&m);
    assert!(
        (0.1..0.9).contains(&r),
        "a ratio the sweep can straddle: {r}"
    );
    let at = |tau: f64| {
        decide_member_coherence(
            &m,
            &MemberCoherenceParams {
                exoneration_ratio: tau,
                ..MemberCoherenceParams::default()
            },
        )
    };
    assert_eq!(at(r - 1e-9).verdict, MemberVerdict::Split);
    assert_eq!(at(r).verdict, MemberVerdict::KeepAll, "inclusive at tau");
    assert_eq!(at(r + 1e-9).verdict, MemberVerdict::KeepAll);
}

#[test]
fn exoneration_off_and_no_coarse_scale_both_leave_the_self_bar_alone() {
    // Three ways for the machinery to be inert, all agreeing with each other:
    // the knob at zero, a matrix carrying no coarse scale, and the relative term
    // itself disabled (nothing is ever flagged, so nothing can be spared).
    let m = two_scale_outsider(8, 0.90, 0.988);
    let raw = decide_member_coherence(
        &m,
        &MemberCoherenceParams {
            exoneration_ratio: 0.0,
            ..MemberCoherenceParams::default()
        },
    );
    assert_eq!(raw.verdict, MemberVerdict::Split);
    assert!(raw.relative_flagged[0], "still flagged, just not spared");
    assert!(!raw.exonerated[0]);
    // The ratio is still reported: the measurement is made either way, and only
    // acting on it is switched off.
    assert!(raw.retained_deficit[0].is_finite());

    let no_scale = MemberMatrix::from_zncc(m.members.clone(), m.zncc.clone());
    let d = decide_member_coherence(&no_scale, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert!(d.relative_flagged[0]);
    assert!(!d.exonerated[0]);
    assert!(d.retained_deficit[0].is_nan(), "nothing to measure against");

    let abs = decide_member_coherence(&m, &absolute());
    assert!(
        abs.relative_flagged.iter().all(|&f| !f),
        "with no relative term there is nothing exoneration may reach"
    );
    assert!(abs.exonerated.iter().all(|&e| !e));
}

#[test]
fn only_the_relative_terms_evictions_are_exonerable() {
    // A cross-surface member the ABSOLUTE bar rejects, whose deficit evaporates
    // at half scale exactly as a soft frame's would. It is still evicted: how a
    // disagreement is spread across scales says nothing about whether a member
    // images the track's surface at all.
    let k = 5;
    let mut fine = vec![0.0; k * k];
    let mut coarse = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let (f, c) = if a == b {
                (1.0, 1.0)
            } else if a == 0 || b == 0 {
                (0.30, 0.985) // below the absolute bar; agrees coarsely
            } else {
                (0.99 - 0.002 * ((a + b) % 3) as f64, 0.99)
            };
            fine[a * k + b] = f;
            coarse[a * k + b] = c;
        }
    }
    let m = MemberMatrix::from_zncc_scales((0..k as u32).collect(), fine, vec![coarse], vec![2]);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![1, 2, 3, 4]);
    assert!(
        !d.relative_flagged[0],
        "the absolute bar already rejected it, so it was never a candidate"
    );
    assert!(!d.exonerated[0]);
    assert!(d.retained_deficit[0].is_nan(), "no ratio is even computed");
}

#[test]
fn sparing_enough_members_turns_a_retirement_into_a_split() {
    // Ten members: a tight core of five, one soft member (5) the tightened bar
    // rejects and whose deficit evaporates coarsely, and a second surface (6..10)
    // the ABSOLUTE bar rejects. The tightened block is five of ten scored, so the
    // raw rule retires the point; sparing the soft member restores the majority
    // and the cut lands on the second surface alone.
    let k = 10;
    let mut fine = vec![0.0; k * k];
    let mut coarse = vec![0.0; k * k];
    {
        let mut put = |a: usize, b: usize, f: f64, c: f64| {
            fine[a * k + b] = f;
            fine[b * k + a] = f;
            coarse[a * k + b] = c;
            coarse[b * k + a] = c;
        };
        for a in 0..5 {
            for b in (a + 1)..5 {
                // A tight core with a little structure, so its scatter is real
                // but small — which is what lets the bar tighten onto it.
                let z = 0.99 - 0.002 * ((a + b) % 3) as f64;
                put(a, b, z, z);
            }
            // The soft member: below the tightened bar, back inside the core
            // once one octave of detail is gone.
            put(a, 5, 0.96, 0.986);
        }
        // A second surface: the absolute bar's business, and wrong at every scale.
        for a in 0..6 {
            for b in 6..k {
                put(a, b, 0.20, 0.20);
            }
        }
        for a in 6..k {
            for b in (a + 1)..k {
                put(a, b, 0.97, 0.97);
            }
        }
    }
    let m = MemberMatrix::from_zncc_scales((0..k as u32).collect(), fine, vec![coarse], vec![2]);

    let raw = decide_member_coherence(
        &m,
        &MemberCoherenceParams {
            exoneration_ratio: 0.0,
            ..MemberCoherenceParams::default()
        },
    );
    assert_eq!(raw.verdict, MemberVerdict::Retire);
    assert_eq!(raw.support, 5, "five of ten scored is no majority");
    assert!(
        raw.relative_flagged[5],
        "the soft member is the relative term's"
    );
    assert!(
        (6..k).all(|i| !raw.relative_flagged[i]),
        "the second surface is the absolute bar's, and not exonerable"
    );

    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::Split);
    assert_eq!(kept_indexes(&d), vec![0, 1, 2, 3, 4, 5]);
    assert!(d.exonerated[5]);
    assert!((6..k).all(|i| !d.exonerated[i]));
    // `support` and `block` keep describing the sweep's own block, so the
    // majority the verdict turned on is not the one they report.
    assert_eq!(d.support, 5);
    assert_eq!(block_indexes(&d), vec![0, 1, 2, 3, 4]);
}

#[test]
fn a_member_with_no_measurable_deficit_is_not_spared() {
    // Exoneration wants positive evidence that a real deficit decayed. A member
    // whose full-scale deficit is below the floor has no ratio worth taking, and
    // the rule's own verdict stands.
    let m = two_scale_outsider(8, 0.987, 0.9875);
    let k = m.len();
    let block: Vec<bool> = (0..k).map(|i| i != 0).collect();
    let df = core_deficit(&m.zncc, k, &block, 0);
    assert!(
        df > 0.0 && df <= EXONERATION_MIN_DEFICIT,
        "a deficit under the floor: {df}"
    );
    let d = decide_member_coherence(&m, &defaults());
    if d.verdict != MemberVerdict::KeepAll {
        assert!(d.relative_flagged[0]);
        assert!(!d.exonerated[0]);
        assert!(d.retained_deficit[0].is_nan());
    }
}

#[test]
fn sharpness_is_reported_for_every_scored_member_whatever_the_verdict() {
    // Unlike the ratio, it describes the observations the point SHIPS, so it is
    // measured on members no one is thinking about evicting — and it separates
    // the soft member from the core by construction.
    let m = two_scale_outsider(8, 0.90, 0.988);
    let d = decide_member_coherence(&m, &defaults());
    assert_eq!(d.verdict, MemberVerdict::KeepAll);
    for i in 0..m.len() {
        assert!(
            d.sharpness_deficit[i].is_finite(),
            "member {i} carries no sharpness"
        );
    }
    let soft = d.sharpness_deficit[0];
    let core_max = (1..m.len())
        .map(|i| d.sharpness_deficit[i])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        soft > 10.0 * core_max.abs().max(1e-6),
        "the soft member stands out: {soft} against a core max of {core_max}"
    );
    // A matrix with no coarse scale reports nothing rather than zero.
    let plain = MemberMatrix::from_zncc(m.members.clone(), m.zncc.clone());
    let dp = decide_member_coherence(&plain, &defaults());
    assert!(dp.sharpness_deficit.iter().all(|s| s.is_nan()));
}

#[test]
fn the_coarse_scales_are_rendered_from_the_same_stack_and_agree_with_the_full_one() {
    // End to end: the coarse tables come out of the same render, symmetric, with
    // a unit diagonal, over the factors the resolution admits — and a member
    // unscored at full scale is unscored at every scale.
    let scene = Scene::new(
        &[
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.4],
            [-0.6, 0.2, 0.3],
            [0.3, -0.5, 0.2],
        ],
        &[surface_a, surface_a, surface_a, surface_b],
    );
    let params = MemberCoherenceParams {
        resolution: 24,
        min_valid_fraction: 0.5,
        ..MemberCoherenceParams::default()
    };
    let m = member_zncc_matrix(&plane_patch(), &scene.views(), &[0, 1, 2, 3], None, &params);
    assert_eq!(m.coarse_factors, coarse_factors_for(params.resolution));
    assert_eq!(m.zncc_coarse.len(), m.coarse_factors.len());
    assert_eq!(m.coarse_factors, vec![2, 4], "24 admits both halvings");
    let k = m.len();
    for table in &m.zncc_coarse {
        assert_eq!(table.len(), k * k);
        for a in 0..k {
            assert_eq!(table[a * k + a], 1.0);
            for b in 0..k {
                let (x, y) = (table[a * k + b], table[b * k + a]);
                assert!(
                    (x - y).abs() < 1e-12 || (x.is_nan() && y.is_nan()),
                    "asymmetric at ({a},{b})"
                );
                assert_eq!(
                    x.is_finite(),
                    m.get(a, b).is_finite(),
                    "scoredness differs by scale at ({a},{b})"
                );
            }
        }
    }
    // The odd member out is odd at every scale: this is a real surface
    // difference, not a spectral one.
    for i in 0..3 {
        assert!(
            m.zncc_coarse[0][i * k + 3] < 0.8,
            "cross-surface pair {i},3 stays weak at half scale: {}",
            m.zncc_coarse[0][i * k + 3]
        );
    }
}

#[test]
fn coarse_factors_follow_the_resolution() {
    assert_eq!(coarse_factors_for(24), vec![2, 4]);
    assert_eq!(coarse_factors_for(16), vec![2, 4]);
    // 12/2 = 6 clears the floor, 12/4 = 3 does not.
    assert_eq!(coarse_factors_for(12), vec![2]);
    // 10 does not divide by 4 at all, and 10/2 = 5 clears the floor.
    assert_eq!(coarse_factors_for(10), vec![2]);
    // Nothing usable: every candidate grid is too small or does not divide.
    assert!(coarse_factors_for(6).is_empty());
    assert!(coarse_factors_for(2).is_empty());
}
