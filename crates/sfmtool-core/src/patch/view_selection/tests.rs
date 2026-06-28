// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// A small synthetic scene mirroring the normal_refine tests: pinhole cameras
// (identity rotation, looking down +z) viewing a textured world plane at
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

/// Synthesize the image a pinhole camera at `center` (looking down +z) sees of
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

/// Synthesize what an identity-rotation pinhole sees of a point at infinity in
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
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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

    let sel = select_patch_views(&patch, &views, &track, &params());

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
    // A point at infinity (+z) seen by identity-rotation cameras at different
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

    let sel = select_patch_views(&patch, &views, &track, &params());

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

    let sel = select_patch_views(&patch, &views, &track, &params());

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

    let sel = select_patch_views(&patch, &views, &track, &params());

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

    let batch = select_patch_cloud_views(&cloud, &views, &track_views, &params());
    assert_eq!(batch.len(), 2);
    for (i, sel) in batch.iter().enumerate() {
        let single = select_patch_views(&cloud.patches[i], &views, &track_views[i], &params());
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

    let sel = select_patch_views(&patch, &views, &track, &params());

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
    let sel_plain = select_patch_views(&patch, &views, &[0u32, 1], &params());
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

    let sel = select_patch_views(&patch, &views, &track, &params());

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

    let sel = select_patch_views(&patch, &views, &track, &params());
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
    let sel = select_patch_views(&patch, &views, &track, &p);

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

    let sel = select_patch_views(&patch, &views, &track, &params());

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
/// normal test passes but whose camera-frame depth is negative.
#[test]
fn behind_camera_candidate_rejected_by_cheirality() {
    // Two front track cameras on the textured plane.
    let centers = [[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture];
    let scene = Scene::new(&centers, &texs);
    let mut views = scene.views();

    // A candidate camera sitting in front of the plane (z = 1 < PLANE_Z, so the
    // patch is front-facing to it) but rotated 180° about y, so its forward axis
    // points away from the plane: the patch centre lands behind the camera
    // (negative camera-frame z). `is_front_facing` (normal vs. centre) still
    // passes; only the cheirality gate rejects it.
    let cam = pinhole();
    // Rotation 180° about y: quaternion (w=0, x=0, y=1, z=0). cam_from_world with
    // R = diag(-1, 1, -1); place the centre at world z = 1.
    let pose = RigidTransform::from_wxyz_translation([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    let extra_pyr = ImageU8Pyramid::build(&render_plane_view([0.0, 0.6, 0.0], texture), 5);

    // Sanity: patch is front-facing to this pose, but the point is behind it.
    let patch = plane_patch();
    assert!(
        patch.is_front_facing(&pose),
        "test setup: pose must be front-facing so only cheirality can reject"
    );
    // Camera-frame depth of the patch centre must be negative.
    let depth = pose.transform_point(&patch.center).z;
    assert!(
        depth < 0.0,
        "test setup: patch must be behind the camera, depth = {depth}"
    );

    views.push(ProjectedImage {
        camera: &cam,
        cam_from_world: &pose,
        pyramid: &extra_pyr,
    });

    let track = vec![0u32, 1];
    let sel = select_patch_views(&patch, &views, &track, &params());

    assert!(
        !sel.admitted.contains(&2),
        "behind-camera candidate must be rejected by cheirality: {:?}",
        sel.admitted
    );
}
