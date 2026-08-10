// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{sample_bilinear_u8_all, ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// A small synthetic scene mirroring the normal_refine tests: pinhole cameras
// (rotated 180° about X so the canonical −Z-forward camera looks down world +z) viewing a textured world plane at
// z = PLANE_Z. The patch sits on that plane with a normal pointing back toward
// the cameras (-z), so a camera in front (z < PLANE_Z) is front-facing.

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

fn texture(x: f64, y: f64) -> f64 {
    127.5 + 55.0 * (x * 17.0).sin() + 45.0 * (y * 23.0).cos() + 25.0 * ((x + y) * 31.0).sin()
}

/// A different surface — a view showing this disagrees photometrically.
fn occluder_texture(x: f64, y: f64) -> f64 {
    127.5 + 60.0 * (y * 13.0 + 1.7).sin() + 40.0 * (x * 29.0 - 0.4).cos()
}

/// Synthesize the image a pinhole camera at `center` (looking down world +z) sees of
/// the textured plane z = PLANE_Z.
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

/// Direction-only texture for a point at infinity (function of the ray direction
/// `(dx, dy)`); the `30·` factor gives spatial frequency over the angular patch.
fn dir_texture(dx: f64, dy: f64) -> f64 {
    texture(dx * 30.0, dy * 30.0)
}

/// A different directional surface (an infinity view showing this disagrees).
fn dir_occluder(dx: f64, dy: f64) -> f64 {
    occluder_texture(dx * 30.0, dy * 30.0)
}

/// Synthesize what an plus-z-looking pinhole sees of a point at infinity in
/// the `+z` direction: each pixel's value is `tex` of its ray direction,
/// independent of camera position (no parallax).
fn render_infinity_view(tex: fn(f64, f64) -> f64) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            data.push(tex(dx, dy).clamp(0.0, 255.0).round() as u8);
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

    /// Identity-rotation cameras at `centers`, each viewing a direction-only
    /// texture for a point at infinity (`+z`); camera translation is irrelevant
    /// (no parallax), so views differ only by their `tex` function.
    fn infinity(centers: &[[f64; 3]], texs: &[fn(f64, f64) -> f64]) -> Self {
        let cams = centers.iter().map(|_| pinhole()).collect();
        let poses = centers
            .iter()
            .map(|c| {
                RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
            })
            .collect();
        let pyrs = texs
            .iter()
            .map(|&tex| ImageU8Pyramid::build(&render_infinity_view(tex), 5))
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

/// Tangent-sphere patch for a point at infinity in the `+z` direction.
fn infinity_patch() -> OrientedPatch {
    OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, -1.0, 0.0),
        [0.05, 0.05],
    )
}

fn params() -> ViewSelectParams {
    ViewSelectParams {
        resolution: 15,
        min_valid_fraction: 0.5,
        min_track_views: 2,
        ..ViewSelectParams::default()
    }
}

// --- Multi-channel (RGB) scene, for the A1 channel-alignment regression. ---

/// Per-channel texture function for the RGB test scene (`None` = a flat channel).
type ChannelTex = Option<fn(f64, f64) -> f64>;

/// Synthesize a 3-channel image a pinhole camera at `center` sees of the plane
/// `z = PLANE_Z`, with an independent texture function per channel. A `None`
/// channel renders a flat mid-grey (constant -> windowed variance ≈ 0, so
/// `znormalize` drops it as flat).
fn render_plane_view_rgb(center: [f64; 3], texs: [ChannelTex; 3]) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H * 3) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            let lambda = PLANE_Z - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            for tex in &texs {
                let v = match tex {
                    Some(t) => t(x, y),
                    None => 127.0, // flat channel
                };
                data.push(v.clamp(0.0, 255.0).round() as u8);
            }
        }
    }
    ImageU8::new(IMG_W, IMG_H, 3, data)
}

struct RgbScene {
    cams: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyrs: Vec<ImageU8Pyramid>,
}

impl RgbScene {
    fn new(centers: &[[f64; 3]], texs: &[[ChannelTex; 3]]) -> Self {
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
            .map(|(c, tex)| ImageU8Pyramid::build(&render_plane_view_rgb(*c, *tex), 5))
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

/// A second textured surface, distinct from `texture`.
fn texture2(x: f64, y: f64) -> f64 {
    127.5 + 50.0 * (x * 11.0 + 0.3).cos() + 48.0 * (y * 19.0 - 0.6).sin()
}

#[test]
fn admits_agreeing_views_keeps_track_rejects_disagreeing() {
    // Views 0,1 are the track (agreeing). View 2 agrees but is NOT in the track
    // -> should be admitted as a photometric candidate. View 3 shows a different
    // surface -> rejected. View 4 is far off-axis so the patch falls out of frame
    // -> rejected (unscoreable). View 5 is behind the plane (z > PLANE_Z) so the
    // patch is back-facing to it -> rejected geometrically.
    let centers = [
        [0.6, 0.0, 0.0],  // 0 track
        [-0.6, 0.0, 0.0], // 1 track
        [0.0, 0.6, 0.0],  // 2 agreeing candidate
        [0.0, -0.6, 0.0], // 3 disagreeing candidate
        [40.0, 0.0, 0.0], // 4 out of frame
        [0.0, 0.0, 8.0],  // 5 behind the plane (back-facing)
    ];
    let texs: Vec<fn(f64, f64) -> f64> = vec![
        texture,
        texture,
        texture,
        occluder_texture,
        texture,
        texture,
    ];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    // Track views are always present and come first.
    assert_eq!(&sel.admitted[..2], &[0, 1]);
    // The agreeing non-track view is admitted; the disagreeing / out-of-frame /
    // back-facing ones are not.
    assert!(
        sel.admitted.contains(&2),
        "agreeing view 2 should be admitted"
    );
    assert!(
        !sel.admitted.contains(&3),
        "disagreeing view 3 should be rejected: {:?}",
        sel.admitted
    );
    assert!(
        !sel.admitted.contains(&4),
        "out-of-frame view 4 should be rejected"
    );
    assert!(
        !sel.admitted.contains(&5),
        "back-facing view 5 should be rejected"
    );

    // Self-agreement is high (track sees the same surface).
    assert!(
        sel.self_agreement > 0.8,
        "self-agreement should be high, got {}",
        sel.self_agreement
    );
    // The agreeing candidate's score should be high; scores are parallel.
    let pos = sel.admitted.iter().position(|&i| i == 2).unwrap();
    assert!(
        sel.scores[pos] > 0.8,
        "agreeing candidate ZNCC should be high, got {}",
        sel.scores[pos]
    );
    assert_eq!(sel.admitted.len(), sel.scores.len());
}

#[test]
fn infinity_point_admits_agreeing_views() {
    // A point at infinity (+z) seen by plus-z-looking cameras at different
    // positions: appearance is direction-only (no parallax), so the track and an
    // agreeing candidate are admitted while a candidate showing a different
    // directional surface is rejected. Exercises the w == 0 cheirality gate
    // (is_in_front) and rendering an infinity patch through the vetting path.
    let centers = [
        [0.0, 0.0, 0.0], // 0 track
        [6.0, 0.0, 0.0], // 1 track (far translation, same content)
        [0.0, 5.0, 0.0], // 2 agreeing candidate
        [2.0, 2.0, 3.0], // 3 disagreeing candidate
    ];
    let texs: Vec<fn(f64, f64) -> f64> = vec![dir_texture, dir_texture, dir_texture, dir_occluder];
    let scene = Scene::infinity(&centers, &texs);
    let views = scene.views();
    let patch = infinity_patch();
    let track = vec![0u32, 1];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    assert_eq!(&sel.admitted[..2], &[0, 1], "track views come first");
    assert!(
        sel.admitted.contains(&2),
        "agreeing infinity candidate should be admitted: {:?}",
        sel.admitted
    );
    assert!(
        !sel.admitted.contains(&3),
        "disagreeing infinity candidate should be rejected: {:?}",
        sel.admitted
    );
    assert!(
        sel.self_agreement > 0.8,
        "infinity track self-agreement should be high, got {}",
        sel.self_agreement
    );
}

#[test]
fn single_track_view_admits_verbatim_no_candidates() {
    // A single-view track: self-agreement is undefined, so no reference can be
    // built; the track view is admitted verbatim with no candidate vetting, even
    // though other views agree.
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    assert_eq!(sel.admitted, vec![0]);
    assert!(sel.self_agreement.is_nan());
}

#[test]
fn track_views_always_admitted_even_when_one_disagrees() {
    // Track view 2 shows a different surface (a wrong match). It is still admitted
    // (track views are unconditional), but its score is low and the robust
    // reference is not dragged down by it (views 0,1 agree).
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, occluder_texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1, 2];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    // All three track views are admitted.
    for t in [0u32, 1, 2] {
        assert!(sel.admitted.contains(&t), "track view {t} must be admitted");
    }
    // The robust reference favors the agreeing majority, so the odd-one-out
    // scores below the agreeing pair.
    let s0 = sel.scores[sel.admitted.iter().position(|&i| i == 0).unwrap()];
    let s2 = sel.scores[sel.admitted.iter().position(|&i| i == 2).unwrap()];
    assert!(
        s2 < s0,
        "disagreeing track view should score below agreeing ones: {s2} vs {s0}"
    );
}

#[test]
fn batch_matches_per_patch() {
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let track_views = vec![vec![0u32, 1], vec![0u32, 1]];

    let batch = select_patch_cloud_views(&cloud, &views, &track_views, None, &params(), None);
    assert_eq!(batch.len(), 2);
    for (i, sel) in batch.iter().enumerate() {
        let single =
            select_patch_views(&cloud.patches[i], &views, &track_views[i], None, &params());
        assert_eq!(sel.admitted, single.admitted);
    }
    // View 2 agrees and is geometrically visible, so the expanded set is a strict
    // superset of the 2-view track.
    assert!(batch[0].admitted.contains(&2));
    assert!(batch[0].admitted.len() >= track_views[0].len());
}

/// A2 regression: a duplicated track image index must not be admitted twice nor
/// double-weight the reference. Before the dedup fix the repeated index appeared
/// twice in `admitted` (and was counted twice in the consensus).
#[test]
fn duplicate_track_index_is_deduped() {
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    // View 0 listed twice (e.g. two observations in the same rig image).
    let track = vec![0u32, 0, 1];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    // No duplicates in the admitted set.
    let mut uniq = sel.admitted.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        sel.admitted.len(),
        "admitted has duplicates: {:?}",
        sel.admitted
    );
    // Each track view appears exactly once.
    assert_eq!(sel.admitted.iter().filter(|&&i| i == 0).count(), 1);
    assert_eq!(sel.admitted.len(), sel.scores.len());

    // The reference is not double-weighted: the dedup'd 2-view track agrees with
    // itself, so self-agreement matches the plain (non-duplicated) 2-view track.
    let sel_plain = select_patch_views(&patch, &views, &[0u32, 1], None, &params());
    assert!(
        (sel.self_agreement - sel_plain.self_agreement).abs() < 1e-9,
        "dedup self-agreement {} != plain {}",
        sel.self_agreement,
        sel_plain.self_agreement
    );
}

/// A1 regression: the reference and a candidate keep *different* original
/// channels. The score must reflect the reference's surviving channels (not a
/// misaligned cross-channel dot). Track sees red textured / green-blue flat;
/// candidate B sees red flat / green textured (with the *same* spatial texture
/// the track has in red). The buggy code compacted both to one channel and dotted
/// ref-red against cand-green -> a spurious high correlation; the fix scores the
/// reference's red channel against the candidate's (flat) red -> ≈ 0.
#[test]
fn a1_channel_alignment_no_cross_channel_artifact() {
    // Track views: red = `texture`, green/blue flat.
    let track_tex: [ChannelTex; 3] = [Some(texture), None, None];
    // Candidate "agree": red = same `texture`, green/blue flat -> should score high.
    let agree_tex: [ChannelTex; 3] = [Some(texture), None, None];
    // Candidate "cross": red flat, green = the *same* `texture` -> the buggy
    // cross-channel dot would have correlated it with the reference's red.
    let cross_tex: [ChannelTex; 3] = [None, Some(texture), None];

    let centers = [
        [0.6, 0.0, 0.0],  // 0 track
        [-0.6, 0.0, 0.0], // 1 track
        [0.0, 0.6, 0.0],  // 2 agree (red textured)
        [0.0, -0.6, 0.0], // 3 cross (only green textured, same pattern)
    ];
    let texs = [track_tex, track_tex, agree_tex, cross_tex];
    let scene = RgbScene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    // Reference's red channel survives; self-agreement is high.
    assert!(
        sel.self_agreement > 0.8,
        "self-agreement should be high on the red channel, got {}",
        sel.self_agreement
    );
    // The red-textured candidate is admitted.
    assert!(
        sel.admitted.contains(&2),
        "agreeing red candidate should be admitted"
    );
    // The cross-channel candidate (red flat) must NOT be admitted: the reference's
    // red channel correlates against its flat red -> ≈ 0, well below the bar. Under
    // the old c_use truncation this would have spuriously correlated and admitted.
    assert!(
        !sel.admitted.contains(&3),
        "cross-channel candidate must be rejected (no misaligned-channel dot): {:?}",
        sel.admitted
    );
}

/// Edge case: a track with no other geometrically-visible views expands by
/// nothing — `admitted` is exactly the track and stays parallel to `scores`.
#[test]
fn no_candidates_admits_only_track() {
    // Two close track cameras; every other potential view is the track itself.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1];

    let sel = select_patch_views(&patch, &views, &track, None, &params());
    assert_eq!(sel.admitted, vec![0, 1]);
    assert_eq!(sel.admitted.len(), sel.scores.len());
}

/// Edge case: a track whose self-agreement is below `min_self_agreement` is
/// admitted verbatim with no expansion, even though an agreeing candidate exists.
#[test]
fn below_min_self_agreement_admits_verbatim_no_expansion() {
    // Track views see two *different* surfaces, so they disagree with each other
    // and the reference's self-agreement is low.
    let centers = [
        [0.6, 0.0, 0.0],  // 0 track (surface A)
        [-0.6, 0.0, 0.0], // 1 track (surface B)
        [0.0, 0.6, 0.0],  // 2 would-agree-with-A candidate
    ];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture2, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1];

    // Force the trust gate high so the (genuinely low) self-agreement is below it.
    let p = ViewSelectParams {
        min_self_agreement: 0.95,
        ..params()
    };
    let sel = select_patch_views(&patch, &views, &track, None, &p);

    // Track admitted verbatim, no candidate added.
    assert_eq!(sel.admitted, vec![0, 1]);
    assert!(
        !sel.admitted.contains(&2),
        "no expansion below the trust gate: {:?}",
        sel.admitted
    );
    // The measured self-agreement is still reported (finite, below the gate).
    assert!(sel.self_agreement.is_finite());
    assert!(sel.self_agreement < 0.95);
}

/// Edge case: a force-admitted track view that the per-view validity gate drops
/// (out of frame) gets a NaN score, and `admitted` / `scores` stay parallel.
#[test]
fn track_view_dropped_by_validity_gate_scores_nan() {
    let centers = [
        [0.6, 0.0, 0.0],  // 0 track (valid)
        [-0.6, 0.0, 0.0], // 1 track (valid)
        [40.0, 0.0, 0.0], // 2 track but far off-axis: patch out of frame
    ];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1, 2];

    let sel = select_patch_views(&patch, &views, &track, None, &params());

    // All track views are admitted (unconditional), parallel to scores.
    assert_eq!(&sel.admitted[..3], &[0, 1, 2]);
    assert_eq!(sel.admitted.len(), sel.scores.len());
    // View 2's render misses the reference support -> NaN score.
    let pos = sel.admitted.iter().position(|&i| i == 2).unwrap();
    assert!(
        sel.scores[pos].is_nan(),
        "out-of-frame track view should score NaN, got {}",
        sel.scores[pos]
    );
}

/// B1 regression: a candidate whose camera is in front (front-facing normal,
/// in-frame projection) but for which the point is *behind* the camera in its
/// own frame must be rejected by the cheirality gate. We synthesize a pose whose
/// normal test passes but whose camera-frame z is positive (depth −z negative).
#[test]
fn behind_camera_candidate_rejected_by_cheirality() {
    // Two front track cameras on the textured plane.
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture];
    let scene = Scene::new(&centers, &texs);
    let mut views = scene.views();

    // A candidate camera sitting in front of the plane (z = 1 < PLANE_Z, so the
    // patch is front-facing to it) but looking away from the plane: under the
    // canonical −Z-forward convention an identity rotation looks down world −z,
    // so the patch centre lands behind the camera (positive camera-frame z).
    // `is_front_facing` (normal vs. centre) still passes; only the cheirality
    // gate rejects it.
    let cam = pinhole();
    // Identity rotation (canonical camera looking down world −z); centre at
    // world z = 1, so cam_from_world translation = -centre.
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
    let extra_pyr = ImageU8Pyramid::build(&render_plane_view([0.0, 0.6, 0.0], texture), 5);

    // Sanity: patch is front-facing to this pose, but the point is behind it.
    let patch = plane_patch();
    assert!(
        patch.is_front_facing(&pose),
        "test setup: pose must be front-facing so only cheirality can reject"
    );
    // The patch centre must be behind the camera: canonical camera-frame z
    // positive (depth −z negative).
    let z_cam = pose.transform_point(&patch.center).z;
    assert!(
        z_cam > 0.0,
        "test setup: patch must be behind the camera, camera-frame z = {z_cam}"
    );

    views.push(ProjectedImage {
        camera: &cam,
        cam_from_world: &pose,
        pyramid: &extra_pyr,
    });

    let track = vec![0u32, 1];
    let sel = select_patch_views(&patch, &views, &track, None, &params());

    assert!(
        !sel.admitted.contains(&2),
        "behind-camera candidate must be rejected by cheirality: {:?}",
        sel.admitted
    );
}

/// The same geometry a `z < 0` cheirality gate calls "behind the camera" is an
/// ordinary peripheral observation for a ray-path model, and `is_in_front` must
/// split the two by model rather than by the sign of `z`.
#[test]
fn ray_path_camera_sees_past_ninety_degrees() {
    let equi = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    };
    let pin = pinhole();
    // Identity rotation: the canonical camera looks down world −z, from world
    // z = 1. A patch just past the camera's own plane (world z = 1.53) and well
    // off to the side sits ~10° behind the 90° horizon, i.e. camera-frame
    // z > 0 — which an equidistant map images and a pinhole cannot.
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
    let patch = OrientedPatch::from_center_normal(
        Point3::new(3.0, 0.0, 1.53),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::y(),
        [0.05, 0.05],
    );
    let pc = pose.transform_point(&patch.center);
    let theta = (-pc.z / pc.coords.norm()).acos().to_degrees();
    assert!(
        pc.z > 0.0 && (95.0..=110.0).contains(&theta),
        "test setup: patch must sit past 90 deg off axis, theta = {theta}"
    );
    assert!(
        is_in_front(&patch, &equi, &pose),
        "an equidistant fisheye images theta = {theta} deg; z < 0 is not its cheirality"
    );
    assert!(
        !is_in_front(&patch, &pin, &pose),
        "the perspective family keeps the half-space test"
    );
    // And the ray-path arm is the model's own domain, not "always true": the
    // antipode of a real capture is outside the image, which the caller's frame
    // test rejects, but the projection itself must still be defined here.
    assert!(equi.ray_to_pixel([pc.x, pc.y, pc.z]).is_some());
}

// ── Affine candidate-scoring fast path ───────────────────────────────────────

/// A `SimpleRadial` camera with radial distortion `k1` (same frame/focal as
/// the pinhole helper).
fn radial_cam(k1: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
            radial_distortion_k1: k1,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// A one-view context over every positive-weight pixel of the window (the
/// affine tests need a full support to compare against the exact render).
fn full_support_ctx(resolution: u32) -> LevelContext {
    let w_full = window_weights(PatchWindow::GaussianDisk { sigma: 0.6 }, resolution);
    let pixels: Vec<usize> = (0..(resolution * resolution) as usize)
        .filter(|&p| w_full[p] > 0.0)
        .collect();
    let weights: Vec<f64> = pixels.iter().map(|&p| w_full[p]).collect();
    LevelContext {
        kept: vec![0],
        pixels,
        weights,
    }
}

#[test]
fn affine_sampling_matches_exact_render_on_mild_distortion() {
    // On a mildly distorted camera the 4th-corner residual is inside the
    // bound, the affine map is accepted, and the sampled support values match
    // the exact per-pixel warp to within (gradient × residual) — the accepted
    // approximation. The image content itself is arbitrary; both paths sample
    // the SAME image through the SAME camera, so this is a pure fidelity test
    // of the affine position approximation.
    let cam = radial_cam(0.03);
    let img = render_plane_view([0.3, 0.1, 0.0], texture);
    let pyr = ImageU8Pyramid::build(&img, 5);
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-0.3, 0.1, 0.0]);
    let view = ProjectedImage {
        camera: &cam,
        cam_from_world: &pose,
        pyramid: &pyr,
    };
    let patch = plane_patch();
    let resolution = 16u32;
    let ctx = full_support_ctx(resolution);
    let n = ctx.pixels.len();

    let (map, level) = affine_core_map(&patch, &view, resolution, Sampler::Bilinear)
        .expect("mild distortion must fit the affine map");
    assert_eq!(level, 0, "the bilinear sampler always samples level 0");
    let mut aff = vec![0f32; n];
    sample_support_affine(
        pyr.level(0),
        &map,
        &ctx.pixels,
        resolution as usize,
        &mut aff,
    );

    let (exact, channels) =
        normalized_stack(&patch, &ctx, &[view], resolution, Sampler::Bilinear, None)
            .expect("patch renders in frame");
    assert_eq!(channels, 1);

    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f64;
    for k in 0..n {
        let d = (aff[k] - exact[k]).abs();
        max_d = max_d.max(d);
        sum_d += d as f64;
    }
    let mean_d = sum_d / n as f64;
    assert!(
        max_d <= 4.0 && mean_d <= 0.8,
        "affine samples must track the exact warp: max {max_d}, mean {mean_d:.3}"
    );
}

#[test]
fn affine_score_matches_exact_score() {
    // The bilinear pair: fast path vs exact warp, both at level 0.
    let level = affine_vs_exact_score(Sampler::Bilinear);
    assert_eq!(level, 0, "the bilinear sampler always samples level 0");
}

#[test]
fn affine_mip_score_matches_exact_mip_score() {
    // The mip pair, held to the same parity budget as the bilinear one: the
    // fast path must compose the affine map with the level the per-pixel
    // `remap_bilinear_mip` would pick and sample *that* level, so the two legs
    // agree. The fixture's patch minifies (≈3.5 source px per grid px at
    // `params().resolution`), so this genuinely exercises a level above 0 — if
    // it ever stops doing so, the assertion below turns the test's silent
    // degradation into a failure.
    let level = affine_vs_exact_score(Sampler::BilinearMip);
    assert!(
        level > 0,
        "the fixture must exercise a mip level above 0, got {level}"
    );
}

/// Score the same candidate through the affine fast path and through the
/// forced exact-warp leg under `sampler`, assert they agree inside the parity
/// budget, and return the pyramid level the fast path selected.
///
/// The end quantity — the candidate's ZNCC against a real reference — must
/// agree between the affine fast path (what `candidate_zncc` picks on this
/// pinhole scene) and the exact-warp leg, well inside any plausible admission
/// bar's sensitivity.
fn affine_vs_exact_score(sampler: Sampler) -> usize {
    let scene = Scene::new(
        &[
            [0.4, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.35, 0.2, 0.1],
        ],
        &[texture as fn(f64, f64) -> f64; 4],
    );
    let views = scene.views();
    let patch = plane_patch();
    let p = ViewSelectParams {
        sampler,
        ..params()
    };
    let w_full = window_weights(p.window, p.resolution);
    let (reference, _agree) = build_reference(&patch, &views, &[0, 1, 2], None, &w_full, &p)
        .expect("track builds a reference");
    let single_ctx = LevelContext {
        kept: vec![0],
        pixels: reference.ctx.pixels.clone(),
        weights: reference.ctx.weights.clone(),
    };
    let sqrt_weights: Vec<f32> = single_ctx
        .weights
        .iter()
        .map(|&w| w.sqrt() as f32)
        .collect();

    // Fast path (the pinhole map is near-affine, so `candidate_zncc` takes it).
    let (_, level) = affine_core_map(&patch, &views[3], p.resolution, sampler)
        .expect("the pinhole map must fit the affine bound on this fixture");
    let mut scratch = Vec::new();
    let affine_score = candidate_zncc(
        &patch,
        &views[3],
        &reference,
        &single_ctx,
        &sqrt_weights,
        &p,
        &mut scratch,
    )
    .expect("candidate scores");

    // Exact leg, forced.
    let (raw, channels) = normalized_stack(
        &patch,
        &single_ctx,
        &[views[3]],
        p.resolution,
        p.sampler,
        None,
    )
    .expect("candidate renders in frame");
    let exact_score =
        score_raw_against_reference(&raw, channels, &reference, &single_ctx, &sqrt_weights)
            .expect("exact leg scores");

    assert!(
        (affine_score - exact_score).abs() < 0.02,
        "{sampler:?} affine vs exact score: {affine_score:.5} vs {exact_score:.5}"
    );
    level
}

#[test]
fn affine_mip_sampling_matches_exact_mip_render() {
    // Sample-level parity for the mip fast path: composing the affine map with
    // the selected level must land on the same source content the per-pixel
    // `remap_bilinear_mip` reads, so the support values track the exact warp
    // within the same (gradient × residual) band the level-0 pair holds to.
    let scene = Scene::new(&[[0.3, 0.1, 0.0]], &[texture as fn(f64, f64) -> f64]);
    let views = scene.views();
    let patch = plane_patch();
    let resolution = 15u32;
    let ctx = full_support_ctx(resolution);
    let n = ctx.pixels.len();

    let (map, level) = affine_core_map(&patch, &views[0], resolution, Sampler::BilinearMip)
        .expect("the pinhole map must fit the affine bound");
    assert!(
        level > 0,
        "fixture must minify into a mip level, got {level}"
    );
    let mut aff = vec![0f32; n];
    sample_support_affine(
        views[0].pyramid.level(level),
        &map,
        &ctx.pixels,
        resolution as usize,
        &mut aff,
    );

    let (exact, channels) = normalized_stack(
        &patch,
        &ctx,
        &[views[0]],
        resolution,
        Sampler::BilinearMip,
        None,
    )
    .expect("patch renders in frame");
    assert_eq!(channels, 1);

    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f64;
    for k in 0..n {
        let d = (aff[k] - exact[k]).abs();
        max_d = max_d.max(d);
        sum_d += d as f64;
    }
    let mean_d = sum_d / n as f64;
    assert!(
        max_d <= 4.0 && mean_d <= 0.8,
        "mip affine samples must track the exact mip warp: max {max_d}, mean {mean_d:.3}"
    );
}

#[test]
fn affine_mip_selection_matches_exact_selection() {
    // End-to-end under the default mip sampler: every score `select_patch_views`
    // reports (all of which come from the affine fast path on this near-affine
    // pinhole scene) must match the exact-warp leg's score for the same view,
    // inside the same 0.02 parity band the per-candidate test pins — so the
    // admission decisions the scores drive are the exact path's decisions.
    let scene = Scene::new(
        &[
            [0.4, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.35, 0.2, 0.1],
            [-0.2, -0.35, 0.05],
        ],
        &[texture as fn(f64, f64) -> f64; 5],
    );
    let views = scene.views();
    let patch = plane_patch();
    let track = [0u32, 1];

    let fast = select_patch_views(&patch, &views, &track, None, &params());
    assert!(
        fast.admitted.len() > track.len(),
        "the fixture must admit at least one vetted candidate, got {:?}",
        fast.admitted
    );
    // The exact leg, rebuilt against the same reference and re-scored per
    // admitted view (the residual bound is not reachable through the public
    // params, so the exact path is driven directly here).
    let p = params();
    let w_full = window_weights(p.window, p.resolution);
    let (reference, _agree) = build_reference(&patch, &views, &track, None, &w_full, &p)
        .expect("track builds a reference");
    let single_ctx = LevelContext {
        kept: vec![0],
        pixels: reference.ctx.pixels.clone(),
        weights: reference.ctx.weights.clone(),
    };
    let sqrt_weights: Vec<f32> = single_ctx
        .weights
        .iter()
        .map(|&w| w.sqrt() as f32)
        .collect();

    for (slot, &v) in fast.admitted.iter().enumerate() {
        let (raw, channels) = normalized_stack(
            &patch,
            &single_ctx,
            &[views[v as usize]],
            p.resolution,
            p.sampler,
            None,
        )
        .expect("admitted view renders in frame");
        let exact =
            score_raw_against_reference(&raw, channels, &reference, &single_ctx, &sqrt_weights)
                .expect("exact leg scores");
        assert!(
            (fast.scores[slot] - exact).abs() < 0.02,
            "view {v}: fast {:.5} vs exact {:.5}",
            fast.scores[slot],
            exact
        );
    }
}

#[test]
fn affine_declines_strong_distortion_and_border() {
    let patch = plane_patch();
    let resolution = 16u32;
    // Strong radial distortion, patch pushed off-centre: the 4th-corner
    // residual exceeds the bound -> the fast path declines (exact warp
    // fallback).
    let cam = radial_cam(0.8);
    let img = render_plane_view([0.9, 0.6, 0.0], texture);
    let pyr = ImageU8Pyramid::build(&img, 5);
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-0.9, 0.6, 0.0]);
    let view = ProjectedImage {
        camera: &cam,
        cam_from_world: &pose,
        pyramid: &pyr,
    };
    assert!(
        affine_core_map(&patch, &view, resolution, Sampler::Bilinear).is_none(),
        "strong distortion must fall back to the exact warp"
    );
    assert!(
        affine_core_map(&patch, &view, resolution, Sampler::BilinearMip).is_none(),
        "the residual gate is sampler-independent (it bounds the warp curvature)"
    );

    // Patch projecting hard against the frame border: the border margin
    // declines it even though the map itself fits (the exact path owns the
    // out-of-frame-support semantics).
    let cam = pinhole();
    let img = render_plane_view([0.0, 0.0, 0.0], texture);
    let pyr = ImageU8Pyramid::build(&img, 5);
    // Camera far off to the side: the patch lands at the frame edge.
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [2.32, 0.0, 0.0]);
    let view = ProjectedImage {
        camera: &cam,
        cam_from_world: &pose,
        pyramid: &pyr,
    };
    assert!(
        affine_core_map(&patch, &view, resolution, Sampler::Bilinear).is_none(),
        "an edge-straddling patch must fall back to the exact warp"
    );
    assert!(
        affine_core_map(&patch, &view, resolution, Sampler::BilinearMip).is_none(),
        "the mip border gate (in level px) is at least as strict as level 0's"
    );
}

#[test]
fn affine_sampler_matches_reference() {
    // The hand-rolled interior sampler (no per-pixel clamping, licensed by the
    // border gate) must reproduce the generic clamped reference sampler on an
    // interior map, up to the shared u8 rounding.
    let img = render_plane_view([0.2, -0.1, 0.0], texture);
    let map = AffineCoreMap {
        a: [3.1, 0.4, 40.0, -0.2, 2.9, 60.0],
    };
    let resolution = 16usize;
    let pixels: Vec<usize> = (0..resolution * resolution).collect();
    let mut lean = vec![0f32; pixels.len()];
    sample_support_affine(&img, &map, &pixels, resolution, &mut lean);
    let mut buf = [0u8; 4];
    for (k, &p) in pixels.iter().enumerate() {
        let col = (p % resolution) as f64;
        let row = (p / resolution) as f64;
        let x = map.a[0] * col + map.a[1] * row + map.a[2];
        let y = map.a[3] * col + map.a[4] * row + map.a[5];
        sample_bilinear_u8_all(&img, x as f32, y as f32, &mut buf[..1]);
        assert!(
            (lean[k] - buf[0] as f32).abs() <= 1.0,
            "pixel {k}: lean {} vs reference {}",
            lean[k],
            buf[0]
        );
    }
}

/// The pixel `patch`'s centre reprojects to in `view` — where projection
/// anchoring samples a track view.
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
fn track_keypoints_at_the_projections_reproduce_the_unanchored_selection() {
    // Anchoring recentres each track view's render at its keypoint; hand it the
    // reprojections themselves and the recentred patch IS the patch, so the whole
    // selection — reference, self-agreement, scores, admitted set — must come back
    // unchanged. That is what makes `track_keypoints` a strict generalization
    // rather than a second render path, and it is what the opt-in default rests on.
    let centers = [
        [0.6, 0.0, 0.0],
        [-0.6, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, -0.6, 0.0],
    ];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture, occluder_texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let track = vec![0u32, 1];
    let kps: Vec<Option<[f64; 2]>> = track
        .iter()
        .map(|&i| Some(project_center(&patch, &views[i as usize])))
        .collect();

    let plain = select_patch_views(&patch, &views, &track, None, &params());
    let anchored = select_patch_views(&patch, &views, &track, Some(&kps), &params());

    assert_eq!(anchored.admitted, plain.admitted);
    assert_eq!(anchored.track_view_count, plain.track_view_count);
    assert!(
        (anchored.self_agreement - plain.self_agreement).abs() < 1e-9,
        "self-agreement {} != {}",
        anchored.self_agreement,
        plain.self_agreement
    );
    for (a, p) in anchored.scores.iter().zip(&plain.scores) {
        assert!((a - p).abs() < 1e-9, "score {a} != {p}");
    }
}

#[test]
fn a_track_view_with_a_reprojection_residual_recovers_when_anchored() {
    // Track view 1's POSE carries a lateral error while its image does not, so the
    // point reprojects a few px off the content the matcher matched. Projection
    // anchoring smears that into the reference and into view 1's own score —
    // which is the quantity an eviction bar reads. Anchoring at the stored
    // keypoint samples the content instead.
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let patch = plane_patch();
    let truth = scene.views();
    let kp1 = project_center(&patch, &truth[1]);

    // Same camera, same image, centre displaced laterally.
    let bad_pose =
        RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.6 - 0.05, 0.0, 0.0]);
    let mut views = scene.views();
    views[1].cam_from_world = &bad_pose;
    let moved = project_center(&patch, &views[1]);
    let residual = (moved[0] - kp1[0]).hypot(moved[1] - kp1[1]);
    assert!(residual > 1.0, "test setup: residual {residual} px");

    let track = vec![0u32, 1];
    let plain = select_patch_views(&patch, &views, &track, None, &params());
    let kps = [None, Some(kp1)];
    let anchored = select_patch_views(&patch, &views, &track, Some(&kps), &params());

    assert!(
        anchored.self_agreement > plain.self_agreement + 0.02,
        "self-agreement should rise: {} -> {}",
        plain.self_agreement,
        anchored.self_agreement
    );
    assert!(
        anchored.scores[1] > plain.scores[1],
        "the misaligned track view's own score should rise: {} -> {}",
        plain.scores[1],
        anchored.scores[1]
    );
}

/// The batch entry's keypoint table threads per patch: an all-`None` table
/// anchors every member at its projection, so the result must equal the
/// no-keypoints call bit for bit — while exercising the parallel-table
/// validation and the per-patch threading.
#[test]
fn batch_with_all_none_keypoints_matches_unanchored() {
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.6, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture];
    let scene = Scene::new(&centers, &texs);
    let views = scene.views();
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let track_views = vec![vec![0u32, 1], vec![0u32, 1]];
    let kps: Vec<Vec<Option<[f64; 2]>>> = vec![vec![None, None], vec![None, None]];

    let anchored =
        select_patch_cloud_views(&cloud, &views, &track_views, Some(&kps), &params(), None);
    let plain = select_patch_cloud_views(&cloud, &views, &track_views, None, &params(), None);
    assert_eq!(anchored.len(), plain.len());
    for (a, b) in anchored.iter().zip(&plain) {
        assert_eq!(a.admitted, b.admitted);
        assert_eq!(a.scores, b.scores);
    }
}

/// A `sift_files` reconstruction stores no inline keypoints, so the helper's
/// documented behavior is an all-`None` table parallel to the track views —
/// those members anchor at their projections.
#[test]
fn track_keypoints_from_reconstruction_is_all_none_without_inline_keypoints() {
    use crate::patch::cloud::{PatchExtent, PatchNormal};

    let recon = crate::reconstruction::SfmrReconstruction::demo(6);
    let cloud = PatchCloud::from_reconstruction(
        &recon,
        PatchNormal::MeanViewing,
        PatchExtent::Fixed(0.05),
        true,
    )
    .unwrap();
    let tv = track_views_from_reconstruction(&recon, &cloud);
    let kps = track_keypoints_from_reconstruction(&recon, &cloud);
    assert_eq!(kps.len(), tv.len());
    for (k, v) in kps.iter().zip(&tv) {
        assert_eq!(k.len(), v.len(), "keypoints parallel to track views");
        assert!(
            k.iter().all(|e| e.is_none()),
            "sift_files reconstruction has no inline keypoints"
        );
    }
}
