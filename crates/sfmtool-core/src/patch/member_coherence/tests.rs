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
