// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// A synthetic scene mirroring the view_selection tests: pinhole cameras (identity
// rotation, looking down +z) viewing a textured world plane at z = PLANE_Z. The
// patch sits on that plane with a normal pointing back toward the cameras (-z).
//
// To exercise *registration*, each view can render the plane texture translated
// in-plane by a per-view world offset `o_k`: the same patch then renders, in view
// k, content shifted by `-o_k`, so the views disagree until the localizer shifts
// each to the reference's render. With view 0 as the reference at `o_0 = 0`, the
// shift the kernel must recover for view k is `o_k / wpp` patch-grid px
// (`wpp = 2·half_extent / R`).

const PLANE_Z: f64 = 4.0;
const IMG_W: u32 = 320;
const IMG_H: u32 = 240;
const FOCAL: f64 = 260.0;
const HALF_EXTENT: f64 = 0.4;
const RES: u32 = 20;

/// World-units per patch-grid pixel at the test resolution.
fn wpp() -> f64 {
    2.0 * HALF_EXTENT / RES as f64
}

/// Source-image pixels per patch-grid pixel for a fronto camera at z = 0.
fn src_per_grid() -> f64 {
    wpp() * FOCAL / PLANE_Z
}

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
/// the textured plane z = PLANE_Z, with the texture pattern translated in-plane
/// by the world offset `off`.
fn render_plane_view(center: [f64; 3], off: [f64; 2], tex: fn(f64, f64) -> f64) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            let lambda = PLANE_Z - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(tex(x - off[0], y - off[1]).clamp(0.0, 255.0).round() as u8);
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
    /// Cameras at `centers`, each rendering `tex` translated by the matching
    /// `offsets` entry.
    fn new(centers: &[[f64; 3]], offsets: &[[f64; 2]], texs: &[fn(f64, f64) -> f64]) -> Self {
        let cams = centers.iter().map(|_| pinhole()).collect();
        let poses = centers
            .iter()
            .map(|c| {
                RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
            })
            .collect();
        let pyrs = centers
            .iter()
            .zip(offsets)
            .zip(texs)
            .map(|((c, o), tex)| ImageU8Pyramid::build(&render_plane_view(*c, *o, *tex), 5))
            .collect();
        Self { cams, poses, pyrs }
    }

    /// Cameras at `centers` (looking down world +z), each viewing the **same**
    /// direction-only texture for a point at infinity — appearance depends only on
    /// the ray direction, so camera translation is irrelevant (no parallax). Per
    /// view the directional texture is shifted by the matching angular `offset`.
    fn infinity(centers: &[[f64; 3]], offsets: &[[f64; 2]]) -> Self {
        let cams = centers.iter().map(|_| pinhole()).collect();
        let poses = centers
            .iter()
            .map(|c| {
                RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]])
            })
            .collect();
        let pyrs = offsets
            .iter()
            .map(|o| ImageU8Pyramid::build(&render_infinity_view(*o), 5))
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

/// Texture as a function of ray direction `(dx, dy)` (small-angle pinhole
/// coords); the `30·` factor gives spatial frequency over the angular patch.
fn dir_texture(dx: f64, dy: f64) -> f64 {
    texture(dx * 30.0, dy * 30.0)
}

/// Synthesize what an plus-z-looking pinhole sees of a point at infinity in
/// the `+z` direction: each pixel's value is `dir_texture` of its ray direction,
/// shifted by the angular offset `off`. Independent of camera position.
fn render_infinity_view(off: [f64; 2]) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            data.push(
                dir_texture(dx - off[0], dy - off[1])
                    .clamp(0.0, 255.0)
                    .round() as u8,
            );
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

/// Patch on the plane, normal toward the cameras (-z).
fn plane_patch() -> OrientedPatch {
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF_EXTENT, HALF_EXTENT],
    )
}

/// Tangent-sphere patch for a point at infinity in the `+z` direction. Angular
/// half-extent `0.05` rad.
fn infinity_patch() -> OrientedPatch {
    OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, -1.0, 0.0),
        [0.05, 0.05],
    )
}

fn params() -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        resolution: RES,
        ..KeypointLocalizeParams::default()
    }
}

/// [`params`] with both **absolute** per-view gates disabled — the behaviour
/// before they existed, which `0.0` must reproduce exactly.
fn gates_off() -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        min_absolute_zncc: 0.0,
        max_member_zncc_self_similarity_radius: 0.0,
        ..params()
    }
}

/// The index into `res.views` where image `i` was kept, if any.
fn pos(res: &KeypointLocalization, i: u32) -> Option<usize> {
    res.views.iter().position(|&v| v == i)
}

#[test]
fn aligned_views_keep_all_and_barely_shift() {
    // Every view sees the same texture, perfectly aligned -> the search should
    // find no residual shift, keep all views, and land each keypoint on its
    // projection.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    assert_eq!(res.views, vec![0, 1, 2, 3], "all aligned views kept");
    for &o in &res.offsets_px {
        assert!(o < 0.6, "aligned view should barely move, got {o} px");
    }
    assert_eq!(res.zncc.len(), res.views.len());
    for &z in &res.zncc {
        assert!(
            z > 0.8,
            "aligned views should match the reference, ZNCC {z}"
        );
    }
    assert_eq!(res.reference, Some(0));
}

#[test]
fn infinity_point_views_co_register_independent_of_translation() {
    // A point at infinity (+z) seen by plus-z-looking cameras at very different
    // positions: appearance depends only on ray direction, so all views see the
    // same content and co-register. Each keypoint lands on the projection of the
    // direction (the principal point), independent of camera translation — the
    // defining homogeneous behavior.
    let scene = Scene::infinity(
        &[
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, -5.0, 3.0],
            [2.0, 2.0, 9.0],
        ],
        &[[0.0; 2]; 4],
    );
    let views = scene.views();
    let patch = infinity_patch();

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    assert_eq!(
        res.views,
        vec![0, 1, 2, 3],
        "all aligned infinity views kept"
    );
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    for (k, &i) in res.views.iter().enumerate() {
        // +z projects to the principal point in these +z-looking views, the same in
        // every camera regardless of translation.
        let pj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        assert!((pj.0 - cx).abs() < 1e-6 && (pj.1 - cy).abs() < 1e-6);
        assert!((res.keypoints[k][0] - cx).abs() < 0.6 && (res.keypoints[k][1] - cy).abs() < 0.6);
        assert!(
            res.offsets_px[k] < 0.6,
            "aligned infinity view barely moves"
        );
    }
    for &z in &res.zncc {
        assert!(z > 0.8, "infinity views match the reference, ZNCC {z}");
    }
}

#[test]
fn infinity_point_seed_offsets_align_back() {
    // Identical content across views; the reference and two other views are
    // seeded at the projection, the fourth a few source px off. The search must
    // pull the off view back onto the reference's render — exercises the w == 0
    // branch of seed_offset (angular ray→offset inversion) and the w == 0
    // render/project path through the search. Pinned to
    // `SearchStrategy::Exhaustive` because the 4-source-px seed shift puts view
    // 3 ~3 grid steps from the reference through a multi-modal angular
    // `dir_texture`; the default `PlusDescent` can walk into a local maximum on
    // the way home, which is the documented trade-off.
    let scene = Scene::infinity(
        &[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [3.0, 0.0, 2.0],
        ],
        &[[0.0; 2]; 4],
    );
    let views = scene.views();
    let patch = infinity_patch();
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let seeds = [
        Some([cx, cy]),
        Some([cx, cy]),
        Some([cx, cy]),
        Some([cx + 4.0, cy]),
    ];
    let exhaustive = KeypointLocalizeParams {
        search_strategy: SearchStrategy::Exhaustive,
        ..params()
    };

    let res = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1, 2, 3],
        Some(&seeds),
        Some(0),
        &exhaustive,
    );

    let p3 = pos(&res, 3).expect("the seeded-off view aligns back and is kept");
    assert!(
        (res.keypoints[p3][0] - cx).abs() < 1.0 && (res.keypoints[p3][1] - cy).abs() < 1.0,
        "seeded-off infinity view should align back to the projection, got {:?}",
        res.keypoints[p3]
    );
    for &z in &res.zncc {
        assert!(z > 0.8, "aligned infinity ZNCC should be high, got {z}");
    }
}

#[test]
fn aligns_misregistered_view_to_the_reference() {
    // Views 0,1,2 are aligned and view 0 is the reference; view 3 sees the
    // texture shifted by +1 patch-grid px in x. The search should recover +1,
    // putting its keypoint ~1 grid px (in source px) off its projection while
    // the aligned views stay put. View 3 is kept (its shift < max_shift_px).
    let shift_grid = 1.0;
    let ox = shift_grid * wpp();
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ox, 0.0]];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    let p3 = pos(&res, 3).expect("the misregistered view co-registers and is kept");
    // The texture is shifted in world-x; for this patch the in-plane v-axis is
    // world-x, and a fronto camera at z=0 maps +world-x to +image-x, so the
    // recovered keypoint must move by +shift_grid·src_per_grid in image-x (SIGNED,
    // so a wrong-direction recovery — which `offsets_px` magnitude would hide —
    // fails here). The y-component must stay put.
    let expected_px = shift_grid * src_per_grid();
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res.keypoints[p3][0] - proj3.0;
    let dy = res.keypoints[p3][1] - proj3.1;
    assert!(
        (dx - expected_px).abs() < 0.4 * src_per_grid(),
        "view 3 should recover signed +{expected_px:.2}px in x, got {dx:.2}px"
    );
    assert!(
        dy.abs() < 0.4 * src_per_grid(),
        "view 3 should not move in y, got {dy:.2}px"
    );
    // The aligned views barely move (in either axis).
    for i in [0u32, 1, 2] {
        let pi = pos(&res, i).unwrap();
        let pj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        assert!(
            (res.keypoints[pi][0] - pj.0).abs() < 0.4 * src_per_grid()
                && (res.keypoints[pi][1] - pj.1).abs() < 0.4 * src_per_grid(),
            "aligned view {i} should barely move"
        );
    }
    // After alignment every view matches the reference well.
    for &z in &res.zncc {
        assert!(z > 0.9, "aligned ZNCC should be high, got {z}");
    }
}

#[test]
fn drops_disagreeing_surface_view() {
    // Three views see the same surface; view 3 shows a different surface. It cannot
    // match the reference's render, so its ZNCC falls below the bars and it is
    // dropped, leaving the agreeing three.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture, occluder_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    assert!(
        pos(&res, 3).is_none(),
        "disagreeing view 3 should be dropped: {:?}",
        res.views
    );
    for i in [0u32, 1, 2] {
        assert!(pos(&res, i).is_some(), "agreeing view {i} should be kept");
    }
}

#[test]
fn drops_view_shifted_beyond_max_shift_px() {
    // View 3's texture is shifted by 2 grid px (~2·src_per_grid source px); with
    // max_shift_px = 3 and src_per_grid ≈ 2.6, that is ~5.2px > 3, so even though it
    // matches the reference (high ZNCC), it is dropped for sitting too far from its
    // projection.
    let ox = 2.0 * wpp();
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ox, 0.0]];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    // Sanity: 2 grid px maps above the 3px gate.
    assert!(2.0 * src_per_grid() > params().max_shift_px);

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    assert!(
        pos(&res, 3).is_none(),
        "the far-shifted view should be dropped by max_shift_px: {:?} offsets {:?}",
        res.views,
        res.offsets_px
    );
    assert_eq!(res.views, vec![0, 1, 2]);
}

#[test]
fn grazing_views_are_prefiltered() {
    // View 3 is oblique to the plane (|d̂·n̂| ≈ 0.94). With a high grazing cutoff it
    // is pre-filtered; with the permissive default it is kept and co-registers.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [1.5, 0.0, 0.0], // oblique
    ];
    let offs = [[0.0; 2]; 4];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    // Sanity: the oblique view is in front, front-facing, and projects in-frame
    // (so only the grazing gate, not projection, can exclude it).
    assert!(patch.is_front_facing(views[3].cam_from_world));
    assert!(project(&views[3], &patch.center, patch.w).is_some());

    let strict = KeypointLocalizeParams {
        min_grazing_cos: 0.95,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &strict);
    assert!(
        pos(&res, 3).is_none(),
        "oblique view should be grazing-filtered: {:?}",
        res.views
    );

    let permissive = KeypointLocalizeParams {
        min_grazing_cos: 0.1,
        ..params()
    };
    let res2 = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &permissive);
    assert!(
        pos(&res2, 3).is_some(),
        "with a permissive cutoff the oblique view is kept: {:?}",
        res2.views
    );
}

#[test]
fn fewer_than_two_views_returns_seed_projection() {
    // A single-view set has nothing to align: its one view is the reference,
    // with or without the caller naming it, and keeps its projection.
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let proj = project(&views[0], &patch.center, patch.w).unwrap();

    for reference in [Some(0), None] {
        let res = localize_patch_keypoints(&patch, &views, &[0], None, reference, &params());
        assert_eq!(res.views, vec![0]);
        assert_eq!(res.keypoints[0], [proj.0, proj.1]);
        assert_eq!(res.zncc, vec![1.0], "the reference scores 1 against itself");
        assert_eq!(res.reference, Some(0));
    }
}

#[test]
fn duplicate_view_index_is_deduped() {
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    // View 0 listed twice.
    let res = localize_patch_keypoints(&patch, &views, &[0, 0, 1, 2], None, Some(0), &params());
    assert_eq!(res.views.iter().filter(|&&v| v == 0).count(), 1);
    let mut uniq = res.views.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), res.views.len(), "no duplicate kept views");
}

#[test]
fn batch_matches_per_patch() {
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2, 3], vec![0u32, 1, 2, 3]];

    let batch = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        None,
        None,
        &params(),
        None,
        &crate::progress::Progress::none(),
    )
    .expect("Progress::none never cancels");
    assert_eq!(batch.len(), 2);
    for (i, res) in batch.iter().enumerate() {
        let single = localize_patch_keypoints(
            &cloud.patches[i],
            &views,
            &view_sets[i],
            None,
            None,
            &params(),
        );
        assert_eq!(res.views, single.views);
        assert_eq!(res.reference, single.reference);
        for (a, b) in res.keypoints.iter().zip(&single.keypoints) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
    }

    // Per-patch references are threaded through: patch 1 names view 2.
    let references = [Some(0), Some(2)];
    let batch = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        None,
        Some(&references),
        &params(),
        None,
        &crate::progress::Progress::none(),
    )
    .expect("Progress::none never cancels");
    for (i, res) in batch.iter().enumerate() {
        let single = localize_patch_keypoints(
            &cloud.patches[i],
            &views,
            &view_sets[i],
            None,
            references[i],
            &params(),
        );
        assert_eq!(res.views, single.views);
        assert_eq!(res.keypoints, single.keypoints);
        assert_eq!(res.reference, Some(view_sets[i][references[i].unwrap()]));
    }
}

#[test]
fn seed_keypoint_offset_round_trips() {
    // Seeding a view at a keypoint that is already on the aligned content (its
    // projection) reproduces the no-seed result on the aligned scene.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let seeds: Vec<Option<[f64; 2]>> = (0..3)
        .map(|i| {
            let (x, y) = project(&views[i], &patch.center, patch.w).unwrap();
            Some([x, y])
        })
        .collect();
    let res =
        localize_patch_keypoints(&patch, &views, &[0, 1, 2], Some(&seeds), Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2]);
    for &o in &res.offsets_px {
        assert!(
            o < 0.6,
            "projection-seeded aligned view should barely move: {o}"
        );
    }
}

#[test]
fn seed_offset_unprojection_round_trips() {
    // A non-projection seed exercises `seed_offset`'s unprojection (rotation
    // transpose, ray∩plane, /wpp): re-anchoring the patch at the offset it
    // returns and projecting gives the seed back, so `seed_offset` is the exact
    // inverse of the keypoint the localizer reports for an offset.
    // `keypoint_grid_offset` asks the same question at a params' resolution.
    let centers = [[0.4, 0.2, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let proj = project(&views[0], &patch.center, patch.w).unwrap();
    let seed = [proj.0 + 5.0, proj.1 - 3.0]; // a few px off the projection
    let off = seed_offset(&patch, &views[0], seed, wpp(), wpp()).expect("the seed hits the plane");
    // A few source px is a couple of grid px.
    assert!(off[0].hypot(off[1]) > 1.0, "{off:?}");
    let center = shifted_center(&patch, off[0], off[1], wpp(), wpp());
    let back = project(&views[0], &center, patch.w).unwrap();
    assert!(
        (back.0 - seed[0]).abs() < 1e-6 && (back.1 - seed[1]).abs() < 1e-6,
        "seed {seed:?} should round-trip through seed_offset, got {back:?}"
    );
    let grid = keypoint_grid_offset(&patch, &views[0], seed, &params()).unwrap();
    assert_eq!(grid, off, "the same offset at the params' resolution");
    // The projection itself is the zero offset.
    let zero = keypoint_grid_offset(&patch, &views[0], [proj.0, proj.1], &params()).unwrap();
    assert!(zero[0].abs() < 1e-9 && zero[1].abs() < 1e-9, "{zero:?}");
}

#[test]
fn all_none_seeds_match_the_unseeded_run() {
    // A seed table of all `None` says "every view seeds at its projection", which
    // is what passing no table at all means — so the two runs must agree to the
    // bit, not merely closely. This is the contract a caller relies on when it
    // builds one table for a view set where no view happens to have a keypoint.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0, 0.0], [0.6 * wpp(), 0.0], [0.0, -wpp()], [0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let unseeded =
        localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    let all_none = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1, 2, 3],
        Some(&[None; 4]),
        Some(0),
        &params(),
    );

    assert_eq!(all_none.views, unseeded.views);
    assert_eq!(all_none.keypoints, unseeded.keypoints);
    assert_eq!(all_none.offsets_px, unseeded.offsets_px);
    assert_eq!(all_none.zncc, unseeded.zncc);
    assert_eq!(all_none.reference, unseeded.reference);
}

#[test]
fn mixed_seeds_apply_per_view() {
    // The mixed table: views 0-2 carry no seed (`None` — they anchor at their
    // projections, as an expansion candidate with no observation does), view 3
    // carries an explicit seed a few source px off the aligned content. The
    // unseeded reference (view 0) fixes the template and the search pulls the
    // seeded one home, so the run matches the equivalent all-`Some` table where
    // 0-2 are seeded at their own projections. Pinned to `Exhaustive` for the
    // same reason `infinity_point_seed_offsets_align_back` is: the long walk
    // home from a multi-px seed is an `Exhaustive` guarantee.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let exhaustive = KeypointLocalizeParams {
        search_strategy: SearchStrategy::Exhaustive,
        ..params()
    };

    let projs: Vec<[f64; 2]> = (0..4)
        .map(|i| {
            let (x, y) = project(&views[i], &patch.center, patch.w).unwrap();
            [x, y]
        })
        .collect();
    // Two source px off view 3's projection — well inside `search`, and enough
    // that seeding it at the projection instead would be visible below.
    let off_seed = [projs[3][0] + 2.0 * src_per_grid(), projs[3][1]];
    let mixed = [None, None, None, Some(off_seed)];
    let all_some = [
        Some(projs[0]),
        Some(projs[1]),
        Some(projs[2]),
        Some(off_seed),
    ];

    let res = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1, 2, 3],
        Some(&mixed),
        Some(0),
        &exhaustive,
    );
    let ref_res = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1, 2, 3],
        Some(&all_some),
        Some(0),
        &exhaustive,
    );

    assert_eq!(res.views, vec![0, 1, 2, 3], "every view is kept");
    assert_eq!(res.views, ref_res.views);
    for (k, (a, b)) in res.keypoints.iter().zip(&ref_res.keypoints).enumerate() {
        assert!(
            (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6,
            "view {k}: an unseeded entry must match a projection-seeded one, {a:?} vs {b:?}"
        );
    }
    // The unseeded views started (and stay) at their projections; the seeded one
    // aligns back onto the same content.
    for i in 0..3 {
        let k = pos(&res, i).unwrap();
        assert!(
            res.offsets_px[k] < 0.6,
            "unseeded view {i} should sit on its projection, got {}",
            res.offsets_px[k]
        );
    }
    let k3 = pos(&res, 3).unwrap();
    assert!(
        res.offsets_px[k3] < 0.6,
        "the seeded view should align back onto the reference's content, got {}",
        res.offsets_px[k3]
    );
}

#[test]
fn batch_mixed_seeds_match_the_single_patch_call() {
    // The cloud entry point's per-patch seed lists carry the same per-view
    // `Option`s: an empty list is an unseeded patch, and a `None` inside a
    // seeded patch's list is an unseeded view.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2], vec![0u32, 1, 2]];
    let proj0 = {
        let (x, y) = project(&views[0], &cloud.patches[0].center, cloud.patches[0].w).unwrap();
        [x + src_per_grid(), y]
    };
    // Patch 0 seeds only its first view; patch 1 is unseeded (empty list).
    let seeds = vec![vec![Some(proj0), None, None], Vec::new()];

    let batch = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        Some(&seeds),
        None,
        &params(),
        None,
        &crate::progress::Progress::none(),
    )
    .expect("Progress::none never cancels");

    let single0 = localize_patch_keypoints(
        &cloud.patches[0],
        &views,
        &view_sets[0],
        Some(&seeds[0]),
        None,
        &params(),
    );
    let single1 = localize_patch_keypoints(
        &cloud.patches[1],
        &views,
        &view_sets[1],
        None,
        None,
        &params(),
    );
    assert_eq!(batch[0].views, single0.views);
    assert_eq!(batch[0].keypoints, single0.keypoints);
    assert_eq!(batch[1].views, single1.views);
    assert_eq!(batch[1].keypoints, single1.keypoints);
}

#[test]
fn drops_low_relative_zncc_view_in_isolation() {
    // Isolate the relative drop gate: three aligned views plus an occluder, with
    // `max_shift_px` set so high and the absolute floor off, so only the
    // relative bar can drop a view. The occluder cannot match the reference, so
    // it (and only it) is dropped.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture, occluder_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let p = KeypointLocalizeParams {
        max_shift_px: 1e6, // disable the shift gate so only the relative bar can drop
        min_absolute_zncc: 0.0,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &p);
    assert!(
        pos(&res, 3).is_none(),
        "occluder must be dropped by the relative bar: {:?}",
        res.views
    );
    assert_eq!(res.views, vec![0, 1, 2]);
}

/// [`params`] with the member self-similarity gate at `bar`.
fn self_similarity_bar(bar: f64) -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        max_member_zncc_self_similarity_radius: bar,
        ..params()
    }
}

/// A straight edge running along the plane's `y` axis: a tile of it matches
/// itself at every shift along the edge.
fn edge_texture(x: f64, _y: f64) -> f64 {
    127.5 + 70.0 * (x * 12.0).tanh()
}

#[test]
fn two_view_flat_member_is_dropped_as_unlocalizable() {
    // A two-view point whose second member is a textureless tile (flat sky /
    // water). It matches itself at every shift, so its ZNCC self-similarity
    // radius reads the largest shift searched and the member gate refuses it
    // before it is searched — leaving the point with its reference alone, which
    // the caller's `min_views` cull removes. With the gates off it survives:
    // on a two-view point the relative bar is `min_relative_zncc ×` the very
    // correlation it is testing.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]; 2];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, flat_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1],
        None,
        Some(0),
        &self_similarity_bar(2.0),
    );
    assert!(
        pos(&res, 1).is_none(),
        "the flat member must be dropped as unlocalizable: {:?}",
        res.views
    );
    assert!(
        res.views.len() < 2,
        "the point is left below min_views: {:?}",
        res.views
    );

    let off_gates = localize_patch_keypoints(&patch, &views, &[0, 1], None, Some(0), &gates_off());
    assert_eq!(
        off_gates.views,
        vec![0, 1],
        "with the gates disabled the flat member survives (the old behaviour)"
    );
}

#[test]
fn the_member_gate_judges_the_self_similarity_radius() {
    // Four views: two of the textured plane, one of a flat surface and one of
    // a straight edge. The flat and edge tiles match themselves the whole way
    // across the shifts searched, so a bar under the largest radius drops
    // them; the textured views read well under 1 px and stay.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, flat_texture, edge_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let loose = |bar: f64| KeypointLocalizeParams {
        // Only the member gate decides here: an edge view searched against a
        // textured reference can slide along the edge, so the shift gate is
        // opened too.
        max_shift_px: 1e6,
        min_absolute_zncc: 0.0,
        min_relative_zncc: 0.0,
        ..self_similarity_bar(bar)
    };

    let gated = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &loose(2.0));
    assert_eq!(
        gated.views,
        vec![0, 1],
        "the flat and edge members are dropped, the textured ones kept"
    );

    // `0` turns the gate off, and a bar at the largest radius read turns
    // nothing out.
    for bar in [0.0, 3.0] {
        let open =
            localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &loose(bar));
        assert_eq!(open.views, vec![0, 1, 2, 3], "bar {bar} keeps every member");
    }
}

#[test]
fn the_member_gate_reads_the_same_radius_at_any_search() {
    // The gate reads the view's `R×R` core alone, so a search of 1 grid px,
    // whose tiles have one grid px around the core, gives the verdicts a wide
    // search gives.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, edge_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    for search in [1.0, 6.0] {
        let p = KeypointLocalizeParams {
            search,
            min_absolute_zncc: 0.0,
            min_relative_zncc: 0.0,
            ..self_similarity_bar(2.0)
        };
        let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &p);
        assert_eq!(res.views, vec![0, 1], "search {search}");
    }
}

/// The member gate reads the `R×R` core and no pixel of the tile around it: a
/// straight edge whose ring is filled with noise reads the same radius as the
/// edge with a clean ring, the largest the reading searches, at every core
/// offset in the tile. A reading that took its shifted windows from the ring
/// would see the noise there and read the edge shorter.
#[test]
fn the_member_gate_reads_no_pixel_outside_the_core() {
    let (resolution, margin, channels) = (24usize, 6usize, 3usize);
    let cr = resolution + 2 * margin;
    let edge = |col: usize| if col < cr / 2 { 60.0f32 } else { 190.0 };
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut noise = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 40) as f32) / (1u64 << 24) as f32 * 255.0
    };
    for (oy, ox) in [(margin, margin), (margin - 2, margin + 3)] {
        let mut clean = vec![0f32; cr * cr * channels];
        let mut noisy = clean.clone();
        for row in 0..cr {
            for col in 0..cr {
                let inside =
                    (oy..oy + resolution).contains(&row) && (ox..ox + resolution).contains(&col);
                for c in 0..channels {
                    let i = (row * cr + col) * channels + c;
                    clean[i] = edge(col) + 10.0 * c as f32;
                    noisy[i] = if inside { clean[i] } else { noise() };
                }
            }
        }
        let valid = vec![true; cr * cr];
        let read = |raw: &[f32]| {
            let tile = build_tile_from_interleaved(raw, cr, channels, &valid);
            member_self_similarity_radius(&tile, resolution, oy, ox, &mut Vec::new())
        };
        let (clean, noisy) = (read(&clean), read(&noisy));
        assert_eq!(clean, 3.0, "core at ({ox}, {oy})");
        assert_eq!(clean.to_bits(), noisy.to_bits(), "core at ({ox}, {oy})");
    }
}

#[test]
fn the_member_gate_admits_at_or_under_its_bar_and_fails_nan() {
    let on = self_similarity_bar(2.0);
    assert!(on.member_self_similarity_gate_is_on());
    assert!(on.admits_member_zncc_self_similarity_radius(0.4));
    assert!(on.admits_member_zncc_self_similarity_radius(2.0));
    assert!(!on.admits_member_zncc_self_similarity_radius(2.1));
    assert!(!on.admits_member_zncc_self_similarity_radius(f64::NAN));
    for bar in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let off = self_similarity_bar(bar);
        assert!(!off.member_self_similarity_gate_is_on(), "bar {bar}");
        assert!(off.admits_member_zncc_self_similarity_radius(f64::NAN));
        assert!(off.admits_member_zncc_self_similarity_radius(3.0));
    }
    // The default is on, at `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`.
    let default = KeypointLocalizeParams::default();
    assert!(default.member_self_similarity_gate_is_on());
    assert_eq!(default.max_member_zncc_self_similarity_radius, 2.5);
}

#[test]
fn a_reference_search_reports_the_view_s_own_self_similarity_radius() {
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, texture, texture, edge_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let p = params();
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let references = TrackReferences::build(
        &patch,
        &views,
        &[0, 1],
        &[[cx, cy], [cx, cy]],
        None,
        None,
        &p,
    )
    .expect("two textured references");
    let textured = references
        .search(&patch, &views[2], None, false, &p)
        .expect("a search");
    let edge = references
        .search(&patch, &views[3], None, false, &p)
        .expect("a search");
    assert!(
        textured.zncc_self_similarity_radius < 1.0,
        "{}",
        textured.zncc_self_similarity_radius
    );
    assert_eq!(edge.zncc_self_similarity_radius, 3.0);
}

#[test]
fn two_view_disagreeing_pair_is_dropped_by_the_absolute_floor() {
    // Two views of *different* surfaces. Both tiles are perfectly localizable on
    // their own, so only the correlation between them can refuse the pair — and
    // the relative bar cannot: the only view it reads besides the reference is
    // the one it tests, which clears `min_relative_zncc ×` itself. The absolute
    // floor is what sees the pair for what it is, and it drops the view that is
    // not the reference.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]; 2];
    let texs: Vec<fn(f64, f64) -> f64> = vec![texture, occluder_texture];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    // `max_shift_px` off in both arms: a view chasing a match on the wrong
    // surface can wander past it, and the point of this test is the ZNCC gates.
    let loose = KeypointLocalizeParams {
        max_shift_px: 1e6,
        ..params()
    };
    let off_gates = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1],
        None,
        Some(0),
        &KeypointLocalizeParams {
            max_shift_px: 1e6,
            ..gates_off()
        },
    );
    assert_eq!(
        off_gates.views,
        vec![0, 1],
        "without the floor the mismatched pair survives"
    );
    assert!(
        off_gates.zncc[1] < 0.5,
        "the view's ZNCC against the reference is well under the 0.5 floor: {:?}",
        off_gates.zncc
    );

    let res = localize_patch_keypoints(&patch, &views, &[0, 1], None, Some(0), &loose);
    assert_eq!(
        res.views,
        vec![0],
        "the absolute floor must break the mismatched pair, keeping the reference: {:?}",
        res.zncc
    );
}

#[test]
fn absolute_gates_at_zero_reproduce_the_ungated_run() {
    // `0.0` disables each absolute gate *exactly*: on a scene none of them bite
    // on, the gated defaults and the fully-disabled params agree bit for bit,
    // arrays included — so the gates add a refusal and change nothing else.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0; 2]; 4];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let gated = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    let ungated =
        localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &gates_off());

    assert_eq!(gated.views, ungated.views);
    assert_eq!(gated.keypoints, ungated.keypoints);
    assert_eq!(gated.offsets_px, ungated.offsets_px);
    assert_eq!(gated.zncc, ungated.zncc);
    assert_eq!(gated.reference, ungated.reference);
}

#[test]
fn empty_view_set_returns_empty() {
    // A patch with no views to refine yields an empty (but well-formed) result.
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    for reference in [None, Some(0)] {
        let res = localize_patch_keypoints(&patch, &views, &[], None, reference, &params());
        assert!(res.views.is_empty());
        assert!(res.keypoints.is_empty());
        assert!(res.offsets_px.is_empty());
        assert!(res.zncc.is_empty());
        assert_eq!(res.reference, None);
    }
}

// ── Accumulation search_shift vs. the per-candidate reference ────────────────

/// Build a window support over the `R×R` core (mirrors `localize_patch_keypoints`).
fn disk_support(resolution: usize) -> Support {
    build_support(PatchWindow::GaussianDisk { sigma: 0.6 }, resolution as u32)
}

/// A deterministic, textured `cr×cr×channels` context tile. `flat_last` forces the
/// final channel constant (to exercise the flat-channel path). Builds the
/// centered-planar / `istride`-padded `ContextTile` the production cache uses.
fn synthetic_tile(cr: usize, channels: usize, flat_last: bool) -> ContextTile {
    let mut px = vec![0f32; cr * cr * channels];
    for row in 0..cr {
        for col in 0..cr {
            let i = (row * cr + col) * channels;
            let x = col as f64 * 0.11;
            let y = row as f64 * 0.13;
            for c in 0..channels {
                let v = if flat_last && c == channels - 1 {
                    100.0
                } else {
                    127.5
                        + 55.0 * (x * (5.0 + c as f64) + 0.3 * c as f64).sin()
                        + 45.0 * (y * (7.0 + c as f64)).cos()
                };
                px[i + c] = v as f32;
            }
        }
    }
    build_tile_from_interleaved(&px, cr, channels, &vec![true; cr * cr])
}

/// Round-trip a [`ContextTile`] back to interleaved raw f32 values
/// `[(row·cr+col)·channels+c]`, undoing the centering. Used by tests that need to
/// rebuild the tile with a different validity mask after construction.
fn unpack_tile_to_interleaved(tile: &ContextTile) -> Vec<f32> {
    let cr = tile.res;
    let ch = tile.channels;
    let mut raw = vec![0f32; cr * cr * ch];
    for row in 0..cr {
        for col in 0..cr {
            let row_off = row * tile.istride + col;
            let base = (row * cr + col) * ch;
            for c in 0..ch {
                raw[base + c] = tile.planes[c][row_off] + tile.means[c];
            }
        }
    }
    raw
}

/// Build a [`ContextTile`] from interleaved raw values `[(row·cr+col)·channels+c]`
/// plus a per-pixel validity mask — mirroring `render_context` (centered planes +
/// `istride` padding + invalid plane). Test-only helper.
fn build_tile_from_interleaved(
    raw: &[f32],
    cr: usize,
    channels: usize,
    valid: &[bool],
) -> ContextTile {
    let istride = cache_istride(cr);
    let mut sums = vec![0.0f64; channels];
    for row in 0..cr {
        for col in 0..cr {
            let base = (row * cr + col) * channels;
            for c in 0..channels {
                sums[c] += raw[base + c] as f64;
            }
        }
    }
    let total = (cr * cr) as f64;
    let means: Vec<f32> = sums.iter().map(|&s| (s / total) as f32).collect();
    let mut planes: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; istride * cr]).collect();
    let mut invalid_plane = vec![0.0f32; istride * cr];
    for row in 0..cr {
        for col in 0..cr {
            let row_off = row * istride + col;
            let v = valid[row * cr + col];
            invalid_plane[row_off] = if v { 0.0 } else { 1.0 };
            let base = (row * cr + col) * channels;
            for c in 0..channels {
                planes[c][row_off] = raw[base + c] - means[c];
            }
        }
    }
    ContextTile {
        res: cr,
        istride,
        channels,
        means,
        planes,
        invalid_plane,
        valid: valid.to_vec(),
    }
}

/// Build the unit template (`[c·n+k]`) from the tile's own core at a known shift,
/// so the ZNCC peak sits unambiguously at that shift (≈1) — a clean oracle target.
#[allow(clippy::too_many_arguments)]
fn template_at(
    tile: &ContextTile,
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    resolution: usize,
    margin: i64,
    dy0: i64,
    dx0: i64,
) -> Vec<f32> {
    let n = support.pixels.len();
    let mut raw = vec![0f32; tile.channels * n];
    let oy = (margin + dy0) as usize;
    let ox = (margin + dx0) as usize;
    assert!(extract_core(tile, support, resolution, oy, ox, &mut raw));
    let mut tmpl = vec![0f32; channels * n];
    znorm_core(&raw, support, keep_mask, &mut tmpl);
    tmpl
}

fn assert_search_eq(got: Option<ShiftResult>, want: Option<ShiftResult>) {
    match (got, want) {
        (Some(g), Some(w)) => {
            assert!((g.dx - w.dx).abs() < 1e-4, "dx {} vs {}", g.dx, w.dx);
            assert!((g.dy - w.dy).abs() < 1e-4, "dy {} vs {}", g.dy, w.dy);
            assert!(
                (g.peak - w.peak).abs() < 1e-5,
                "peak {} vs {}",
                g.peak,
                w.peak
            );
            // The integer argmax must agree exactly (it drives the read accumulator).
            assert_eq!(g.ix, w.ix, "ix {} vs {}", g.ix, w.ix);
            assert_eq!(g.iy, w.iy, "iy {} vs {}", g.iy, w.iy);
        }
        (None, None) => {}
        _ => panic!("one returned None: got={got:?} want={want:?}"),
    }
}

#[test]
fn search_shift_matches_reference() {
    let resolution = 20usize;
    let margin = 4i64;
    let cr = resolution + 2 * margin as usize;
    let channels = 3usize;
    let support = disk_support(resolution);
    let keep_mask = vec![true; channels];
    let tile = synthetic_tile(cr, channels, false);

    // Template from a known shift → unambiguous peak there.
    let (dy0, dx0) = (1i64, -2i64);
    let base = margin as usize; // search centred on the tile (base offset = margin)
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, resolution, margin, dy0, dx0,
    );

    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    assert_search_eq(got, want);
    // And it actually recovered the planted shift.
    let g = got.unwrap();
    assert!((g.dx - dx0 as f64).abs() < 0.25 && (g.dy - dy0 as f64).abs() < 0.25);
    assert!(
        g.peak > 0.99,
        "self-template peak should be ≈1, got {}",
        g.peak
    );
}

#[test]
fn search_shift_matches_reference_flat_channel_and_invalid() {
    let resolution = 20usize;
    let margin = 4i64;
    let cr = resolution + 2 * margin as usize;
    let channels = 3usize;
    let support = disk_support(resolution);
    let keep_mask = vec![true; channels];
    // Channel 2 flat (exercises FLAT_NORM_SQ_EPS), and a border band invalid
    // (exercises the validity grid → some shifts unscorable). Rebuild from
    // interleaved raw + a custom validity mask (the production cache layout
    // co-locates the validity plane with the centered planes; flipping a `bool`
    // post-hoc would leave the SIMD validity plane stale).
    let tile0 = synthetic_tile(cr, channels, true);
    let raw = unpack_tile_to_interleaved(&tile0);
    let mut valid = vec![true; cr * cr];
    for row in 0..cr {
        for col in 0..cr {
            if row < 2 || col < 2 {
                valid[row * cr + col] = false;
            }
        }
    }
    let tile = build_tile_from_interleaved(&raw, cr, channels, &valid);

    let (dy0, dx0) = (2i64, 1i64);
    let base = margin as usize;
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, resolution, margin, dy0, dx0,
    );
    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    assert_search_eq(got, want);
}

/// The dispatched (AVX2-where-available) `compute_channel_grids` agrees with
/// the scalar reference within tight `f32` tolerance, across:
///
///   * the typical default (`span = 13`),
///   * the tightest AVX2-path boundary reachable via integer margin
///     (`span = 15`; `span = 16` is unreachable since `span = 2·margin + 1` is
///     always odd — the 16-lane store still exercises by spilling 15 lanes),
///   * a span that overflows the AVX2 kernel's 16-lane cap (`span = 17`),
///     which forces the dispatcher to the scalar fallback (verifies the gate).
///
/// Mirrors `super::normal_refine::fronto_cache::tests::resample_avx2_matches_scalar`.
#[test]
fn compute_channel_grids_avx2_matches_scalar() {
    for &(resolution, margin) in &[
        (20usize, 4i64), // span = 9
        (24, 6),         // span = 13 (production default)
        (16, 7),         // span = 15 (AVX2 path, tightest reachable boundary)
        (24, 7),         // span = 15, larger core
        (20, 8),         // span = 17 (overflows AVX2 cap → scalar fallback)
    ] {
        run_compute_channel_grids_equivalence(resolution, margin);
    }
}

fn run_compute_channel_grids_equivalence(resolution: usize, margin: i64) {
    let span = (2 * margin + 1) as usize;
    let cr = resolution + 2 * margin as usize;
    let channels = 3usize;
    let support = disk_support(resolution);
    let tile = synthetic_tile(cr, channels, false);
    let n = support.pixels.len();
    let w_f32: Vec<f32> = support.weights.iter().map(|&w| w as f32).collect();
    let kern: Vec<f32> = (0..n)
        .map(|k| support.sqrt_weights[k] * (0.7 + 0.3 * ((k as f32) * 0.13).sin()))
        .collect();
    let base = margin as usize;
    let win_oy = base - margin as usize;
    let win_ox = base - margin as usize;
    let gsz = span * span;
    // Exercise every channel — the AVX2 kernel is channel-agnostic but a real
    // bug could hide in plane indexing under a non-zero channel.
    for c in 0..channels {
        let mut g_n_s = vec![0f32; gsz];
        let mut g_s1_s = vec![0f32; gsz];
        let mut g_s2_s = vec![0f32; gsz];
        compute_channel_grids_scalar(
            &tile.planes[c],
            &support,
            &kern,
            &w_f32,
            resolution,
            tile.istride,
            span,
            win_oy,
            win_ox,
            &mut g_n_s,
            &mut g_s1_s,
            &mut g_s2_s,
        );
        let mut g_n_d = vec![0f32; gsz];
        let mut g_s1_d = vec![0f32; gsz];
        let mut g_s2_d = vec![0f32; gsz];
        compute_channel_grids(
            &tile.planes[c],
            &support,
            &kern,
            &w_f32,
            resolution,
            tile.istride,
            span,
            win_oy,
            win_ox,
            &mut g_n_d,
            &mut g_s1_d,
            &mut g_s2_d,
        );
        // Tight tolerance: scalar and AVX2 compute the same algebra in `f32`,
        // but scalar's mul-then-add takes two roundings per term while AVX2's
        // FMA takes one, so the per-cell drift is bounded by `~n · ε_f32` over
        // ~400 support pixels — empirically a few ×1e-5 relative on the worst
        // cells. 5e-5 covers this with margin while still being 20× tighter
        // than the spec's "relative tolerance ~1e-3" guideline, so a real
        // precision regression of 0.01%+ still fails the test.
        let tol = |a: f32, b: f32| -> bool { (a - b).abs() <= 5e-5 * (1.0 + a.abs().max(b.abs())) };
        for s in 0..gsz {
            assert!(
                tol(g_n_s[s], g_n_d[s]),
                "ch {c} n[{s}] {} vs {}",
                g_n_s[s],
                g_n_d[s]
            );
            assert!(
                tol(g_s1_s[s], g_s1_d[s]),
                "ch {c} s1[{s}] {} vs {}",
                g_s1_s[s],
                g_s1_d[s]
            );
            assert!(
                tol(g_s2_s[s], g_s2_d[s]),
                "ch {c} s2[{s}] {} vs {}",
                g_s2_s[s],
                g_s2_d[s]
            );
        }
    }
}

/// The whole dispatched `search_shift` agrees with the per-candidate `f64`
/// oracle (`search_shift_ref`) — exercising the AVX2 inner loop end-to-end on
/// the same clear-peak fixtures `search_shift_matches_reference*` use. The
/// integer argmax must match exactly (it drives the read accumulator).
#[test]
fn search_shift_avx2_matches_scalar() {
    for &(resolution, margin, dy0, dx0) in
        &[(20usize, 4i64, 1i64, -2i64), (24, 6, -3, 2), (16, 3, -1, 2)]
    {
        let cr = resolution + 2 * margin as usize;
        let channels = 3usize;
        let support = disk_support(resolution);
        let keep_mask = vec![true; channels];
        let tile = synthetic_tile(cr, channels, false);
        let base = margin as usize;
        let tmpl = template_at(
            &tile, &support, &keep_mask, channels, resolution, margin, dy0, dx0,
        );
        let mut sc = SearchScratch {
            tmpl: tmpl.clone(),
            ..Default::default()
        };
        let got = search_shift(
            &tile, &mut sc, &support, &keep_mask, channels, resolution, margin, base, base,
        );
        let want = search_shift_ref(
            &tile, &tmpl, &support, &keep_mask, channels, resolution, margin, base, base,
        );
        assert_search_eq(got, want);
    }
}

/// The single-cell scoring kernel (`score_cell_one_channel`) must agree with
/// the per-shift slice of the existing whole-grid SAXPY (`compute_channel_grids`)
/// at every cell of the search grid — the algebra both compute is identical, so
/// any cell on the dispatched grid must equal the same cell scored via the
/// per-cell path within `f32` rounding. Locks the AVX2 gather kernel against
/// the SAXPY's accumulator and the scalar fallback against both.
#[test]
fn score_cell_matches_compute_channel_grids() {
    use super::{compute_channel_grids, score_cell_one_channel, score_cell_one_channel_scalar};
    for &(resolution, margin) in &[
        (20usize, 4i64), // span = 9
        (24, 6),         // span = 13 (production default)
        (16, 7),         // span = 15
    ] {
        let span = (2 * margin + 1) as usize;
        let cr = resolution + 2 * margin as usize;
        let channels = 3usize;
        let support = disk_support(resolution);
        let tile = synthetic_tile(cr, channels, false);
        let n = support.pixels.len();
        let w_f32: Vec<f32> = support.weights.iter().map(|&w| w as f32).collect();
        let kern: Vec<f32> = (0..n)
            .map(|k| support.sqrt_weights[k] * (0.7 + 0.3 * ((k as f32) * 0.13).sin()))
            .collect();
        let base = margin as usize;
        let win_oy = base - margin as usize;
        let win_ox = base - margin as usize;
        let gsz = span * span;
        for c in 0..channels {
            // Reference whole-grid sums (the dispatched SAXPY which the existing
            // `compute_channel_grids_avx2_matches_scalar` test locks AVX2 vs
            // scalar).
            let mut g_n = vec![0f32; gsz];
            let mut g_s1 = vec![0f32; gsz];
            let mut g_s2 = vec![0f32; gsz];
            compute_channel_grids(
                &tile.planes[c],
                &support,
                &kern,
                &w_f32,
                resolution,
                tile.istride,
                span,
                win_oy,
                win_ox,
                &mut g_n,
                &mut g_s1,
                &mut g_s2,
            );
            // The per-cell kernels (dispatched + scalar) must agree with the
            // SAXPY's slice at every cell. Accumulation orders differ (SAXPY
            // accumulates across support pixels at fixed gx, broadcasting
            // kern/w across 16 lanes; the per-cell path accumulates across
            // support pixels into 8-lane gathers and horizontal-reduces at
            // the end), so the per-cell ordering compounds the FMA-vs-scalar
            // drift over the ~400 support pixels. Empirically ~1e-4 relative
            // on the worst cells; 3e-4 covers it with margin, still 3× tighter
            // than the spec's "relative tolerance ~1e-3" guideline.
            let tol =
                |a: f32, b: f32| -> bool { (a - b).abs() <= 3e-4 * (1.0 + a.abs().max(b.abs())) };
            for gy in 0..span {
                for gx in 0..span {
                    let win_y = win_oy + gy;
                    let win_x = win_ox + gx;
                    let (n_d, s1_d, s2_d) = score_cell_one_channel(
                        &tile.planes[c],
                        &support,
                        &kern,
                        &w_f32,
                        resolution,
                        tile.istride,
                        win_y,
                        win_x,
                    );
                    let (n_s, s1_s, s2_s) = score_cell_one_channel_scalar(
                        &tile.planes[c],
                        &support,
                        &kern,
                        &w_f32,
                        resolution,
                        tile.istride,
                        win_y,
                        win_x,
                    );
                    let s = gy * span + gx;
                    assert!(
                        tol(n_d, g_n[s]) && tol(s1_d, g_s1[s]) && tol(s2_d, g_s2[s]),
                        "dispatched ch {c} @ ({gy},{gx}): \
                         got n/s1/s2 {n_d}/{s1_d}/{s2_d} vs grid {}/{}/{}",
                        g_n[s],
                        g_s1[s],
                        g_s2[s]
                    );
                    assert!(
                        tol(n_s, g_n[s]) && tol(s1_s, g_s1[s]) && tol(s2_s, g_s2[s]),
                        "scalar ch {c} @ ({gy},{gx}): \
                         got n/s1/s2 {n_s}/{s1_s}/{s2_s} vs grid {}/{}/{}",
                        g_n[s],
                        g_s1[s],
                        g_s2[s]
                    );
                }
            }
        }
    }
}

/// Multi-step descent equivalence with the exhaustive SAXPY: same synthetic
/// tile, same template, same search params — PlusDescent and `search_shift`
/// must agree on the integer argmax for templates planted multiple cells away
/// from the search origin, and agree to within sampling noise on the
/// sub-pixel residual. Closes the round-1 test-coverage gap on non-trivial
/// descent walks; the existing
/// `plus_descent_agrees_with_exhaustive_on_well_posed_scene` end-to-end test
/// converges in a single step.
///
/// Walk length is implicit: with the seed at `(0, 0)` and `Exhaustive` finding
/// `(iy, ix) = (dy0, dx0)`, PlusDescent must walk `|dy0| + |dx0|` cells minimum
/// (more if it weaves around the peak's neighborhood). The fixtures span 3-,
/// 4-, and pure-axis walks. Mirrors `search_shift_matches_reference`'s tile +
/// margin recipe so the same texture-recoverable shifts are used.
#[test]
fn search_shift_plus_descent_walks_multi_step() {
    for &(resolution, margin, dy0, dx0) in &[
        (20usize, 4i64, 1, -2), // 3-step walk (same fixture as search_shift_matches_reference)
        (20, 4, 1, 1),          // 2-step walk
        (20, 4, -1, 1),         // 2-step walk, opposite quadrant
        (24, 4, 0, 1),          // pure-x walk, single step
        (24, 4, 1, 0),          // pure-y walk, single step
    ] {
        let cr = resolution + 2 * margin as usize;
        let channels = 3usize;
        let support = disk_support(resolution);
        let keep_mask = vec![true; channels];
        let tile = synthetic_tile(cr, channels, false);
        let base = margin as usize;
        let tmpl = template_at(
            &tile, &support, &keep_mask, channels, resolution, margin, dy0, dx0,
        );
        let mut sc_plus = SearchScratch {
            tmpl: tmpl.clone(),
            ..Default::default()
        };
        let plus = search_shift_plus_descent(
            &tile,
            &mut sc_plus,
            &support,
            &keep_mask,
            channels,
            resolution,
            margin,
            base,
            base,
        )
        .expect("descent scores the seed cell");
        let mut sc_exh = SearchScratch {
            tmpl: tmpl.clone(),
            ..Default::default()
        };
        let exh = search_shift(
            &tile,
            &mut sc_exh,
            &support,
            &keep_mask,
            channels,
            resolution,
            margin,
            base,
            base,
        )
        .expect("exhaustive scores the search grid");

        // Both strategies must agree on the integer argmax (this is what
        // drives the read accumulator). For these fixtures Exhaustive
        // recovers `(dy0, dx0)` — `search_shift_matches_reference` already
        // pins this — so PlusDescent must too, and the descent walked at
        // least `|dy0| + |dx0|` cells getting there.
        assert_eq!(
            (plus.iy, plus.ix),
            (exh.iy, exh.ix),
            "argmax disagreement: plus ({}, {}) vs exhaustive ({}, {}) \
             (case res={resolution}, margin={margin}, planted dy0={dy0}, dx0={dx0}; \
             walked {} cells minimum)",
            plus.iy,
            plus.ix,
            exh.iy,
            exh.ix,
            dy0.unsigned_abs() + dx0.unsigned_abs(),
        );
        // Sub-pixel residuals agree within sampling noise. The exhaustive
        // SAXPY and per-cell scoring accumulate in slightly different orders,
        // so the parabolic-input cells can drift by a few `ε_f32` — bounded
        // empirically at < 5e-3 of a grid step.
        assert!(
            (plus.dx - exh.dx).abs() < 5e-3 && (plus.dy - exh.dy).abs() < 5e-3,
            "sub-pixel disagreement: plus ({}, {}) vs exhaustive ({}, {}) \
             (case res={resolution}, margin={margin}, planted dy0={dy0}, dx0={dx0})",
            plus.dx,
            plus.dy,
            exh.dx,
            exh.dy,
        );
        // Combined ZNCC at the integer peak agrees too.
        assert!(
            (plus.peak - exh.peak).abs() < 1e-4 * (1.0 + plus.peak.abs()),
            "peak disagreement: plus {} vs exhaustive {} \
             (case res={resolution}, margin={margin}, planted dy0={dy0}, dx0={dx0})",
            plus.peak,
            exh.peak,
        );
    }
}

#[test]
fn search_shift_matches_reference_dropped_channel() {
    // A kept-mask that drops the middle channel (kept channels need not be
    // contiguous tile channels).
    let resolution = 16usize;
    let margin = 3i64;
    let cr = resolution + 2 * margin as usize;
    let tile_channels = 3usize;
    let keep_mask = vec![true, false, true];
    let channels = 2usize; // kept
    let support = disk_support(resolution);
    let tile = synthetic_tile(cr, tile_channels, false);

    let (dy0, dx0) = (-1i64, 2i64);
    let base = margin as usize;
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, resolution, margin, dy0, dx0,
    );
    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, resolution, margin, base, base,
    );
    assert_search_eq(got, want);
}

/// End-to-end equivalence: `PlusDescent` and `Exhaustive` must agree on the
/// kept view set and converge to nearly-the-same per-view keypoints on a clean
/// well-posed scene (aligned plus one misregistered view). Locks
/// the new default's behaviour against the original whole-grid path on a case
/// where the ZNCC landscape is unimodal — the descent's local-optima failure
/// mode (~9 % of observations on dino-full) is expected on multi-modal real
/// data and not tested here.
#[test]
fn plus_descent_agrees_with_exhaustive_on_well_posed_scene() {
    let shift_grid = 1.0;
    let ox = shift_grid * wpp();
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ox, 0.0]];
    let texs = vec![texture as fn(f64, f64) -> f64; 4];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let plus = KeypointLocalizeParams {
        search_strategy: SearchStrategy::PlusDescent,
        ..params()
    };
    let exhaustive = KeypointLocalizeParams {
        search_strategy: SearchStrategy::Exhaustive,
        ..params()
    };

    let a = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &plus);
    let b = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &exhaustive);

    assert_eq!(
        a.views, b.views,
        "the two strategies must agree on the kept view set on a well-posed scene"
    );
    for (i, (&ka, &kb)) in a.keypoints.iter().zip(&b.keypoints).enumerate() {
        let dx = ka[0] - kb[0];
        let dy = ka[1] - kb[1];
        let d = (dx * dx + dy * dy).sqrt();
        // Sub-pixel agreement: both walks land on the same integer cell on this
        // unimodal scene; the parabolic residual differs only via the FMA vs
        // SAXPY rounding gap on the cardinal cells, well under 0.1 src px.
        assert!(
            d < 0.1,
            "view {} ({:?} in input): keypoints diverged by {:.4} src px \
             (plus {:?} vs exhaustive {:?})",
            i,
            a.views[i],
            d,
            ka,
            kb,
        );
    }
}

/// Widen a 1-channel image to `channels` by giving each channel its own gain —
/// so every channel is textured (no flat-channel drop) but they are not
/// bit-identical copies.
fn widen(img: &ImageU8, channels: u32) -> ImageU8 {
    let (w, h) = (img.width(), img.height());
    let gains = [1.0f64, 0.85, 0.7, 0.55];
    let mut data = Vec::with_capacity((w * h * channels) as usize);
    for row in 0..h {
        for col in 0..w {
            let v = img.get_pixel(col, row, 0) as f64;
            for c in 0..channels {
                data.push((v * gains[c as usize % 4]).clamp(0.0, 255.0).round() as u8);
            }
        }
    }
    ImageU8::new(w, h, channels, data)
}

/// A ring scene whose views render `channels[k]`-channel imagery — the mixed
/// grayscale/colour capture the search has to survive.
fn ring_scene_mixed_channels(channels: &[u32]) -> (Scene, Vec<u32>) {
    let n = channels.len();
    let mut centers = Vec::with_capacity(n);
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        centers.push([0.45 * a.cos(), 0.45 * a.sin(), 0.0]);
    }
    let cams: Vec<CameraIntrinsics> = centers.iter().map(|_| pinhole()).collect();
    let poses: Vec<RigidTransform> = centers
        .iter()
        .map(|c| RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-c[0], c[1], c[2]]))
        .collect();
    let pyrs = centers
        .iter()
        .zip(channels)
        .map(|(c, &ch)| {
            let gray = render_plane_view(*c, [0.0, 0.0], texture);
            let img = if ch == 1 { gray } else { widen(&gray, ch) };
            ImageU8Pyramid::build(&img, 5)
        })
        .collect();
    let scene = Scene { cams, poses, pyrs };
    (scene, (0..n as u32).collect())
}

#[test]
fn views_narrower_or_wider_than_the_reference_are_scored_not_panicked() {
    // Every third view is grayscale among 3-channel ones. A 1-channel tile has
    // no plane for a colour template's channels 1 and 2, and a 1-channel
    // template scores only the leading channel of a colour tile; either way the
    // view is searched on the channels both carry.
    let channels: Vec<u32> = (0..9).map(|k| if k % 3 == 0 { 1 } else { 3 }).collect();
    let (scene, set) = ring_scene_mixed_channels(&channels);
    let views = scene.views();
    let patch = plane_patch();
    // A colour reference (view 1) and a grayscale one (view 0).
    for reference in [1usize, 0] {
        let res = localize_patch_keypoints(&patch, &views, &set, None, Some(reference), &params());
        assert_eq!(res.reference, Some(reference as u32));
        assert_eq!(res.keypoints.len(), res.views.len());
        assert_eq!(res.zncc.len(), res.views.len());
        assert_eq!(res.views, set, "every view matches the reference's surface");
        for (k, &v) in res.views.iter().enumerate() {
            assert!(
                res.zncc[k].is_finite() && res.zncc[k] > 0.5,
                "reference {reference}: view {v} ({} channels) scored {}",
                channels[v as usize],
                res.zncc[k]
            );
        }
    }
}

/// A textureless surface — flat sky or water. Every channel is flat, so the
/// z-normalization finds no channel to score on and no template can be built;
/// a member rendering this reads the largest ZNCC self-similarity radius (it
/// fixes no 2D position, and its ZNCC to anything is noise).
fn flat_texture(_x: f64, _y: f64) -> f64 {
    128.0
}

#[test]
fn views_with_no_template_stay_at_their_starts_and_face_the_shift_gate() {
    // A flat reference gives the z-normalization no textured channel, so there
    // is no template to align to: every view keeps its starting keypoint,
    // unscored, and no view is anchored. A start further than `max_shift_px`
    // from the projection is still dropped.
    let centers: Vec<[f64; 3]> = (0..8)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / 8.0;
            [0.45 * a.cos(), 0.45 * a.sin(), 0.0]
        })
        .collect();
    let offs = [[0.0; 2]; 8];
    let texs = vec![flat_texture as fn(f64, f64) -> f64; 8];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let set: Vec<u32> = (0..8).collect();
    let patch = plane_patch();
    // Seed every view well off its projection.
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let seeds: Vec<Option<[f64; 2]>> = (0..8).map(|_| Some([cx + 12.0, cy])).collect();
    let open = |max_shift_px: f64| KeypointLocalizeParams {
        max_shift_px,
        // Hold the member self-similarity gate off: it would refuse these flat
        // tiles outright, and the path under test is the one the
        // z-normalization reaches.
        max_member_zncc_self_similarity_radius: 0.0,
        ..params()
    };

    let loose = localize_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(0), &open(1e6));
    assert_eq!(loose.views, set, "a loose shift gate keeps every view");
    assert_eq!(loose.reference, None, "there is nothing to anchor to");
    assert!(
        loose.zncc.iter().all(|z| z.is_nan()),
        "no template was built, so every ZNCC is unknown: {:?}",
        loose.zncc
    );
    for (k, kp) in loose.keypoints.iter().enumerate() {
        assert!(
            (kp[0] - cx - 12.0).abs() < 1e-6 && (kp[1] - cy).abs() < 1e-6,
            "view {k} keeps its start, got {kp:?}"
        );
    }

    let tight = localize_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(0), &open(0.5));
    assert!(
        tight.views.is_empty(),
        "every start sits 12 px from its projection: {:?} {:?}",
        tight.views,
        tight.offsets_px
    );
}

/// `project_unclipped` is the only world→pixel entry in this module and the
/// sub-pixel refiner, so its cheirality test decides which views either can
/// reach. Under a ray-path model that test must be the camera's own domain: a
/// `z >= 0` short-circuit ahead of `ray_to_pixel` makes the whole θ > 90°
/// annulus of a >180° capture invisible to the localizer.
#[test]
fn project_unclipped_reaches_past_ninety_degrees_on_a_ray_path_model() {
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
    // Identity rotation → the canonical camera looks down world −z from z = 1;
    // a point off to the side and just past that plane sits ~100° off axis.
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
    let p = Point3::new(3.0, 0.0, 1.53);
    let pc = pose.transform_point(&p);
    let theta = (-pc.z / pc.coords.norm()).acos().to_degrees();
    assert!(
        pc.z > 0.0 && (95.0..=110.0).contains(&theta),
        "test setup: theta = {theta}"
    );

    let pyr = ImageU8Pyramid::build(&render_plane_view([0.0, 0.0, 0.0], [0.0, 0.0], texture), 3);
    let equi_view = ProjectedImage {
        camera: &equi,
        cam_from_world: &pose,
        pyramid: &pyr,
    };
    let pin_view = ProjectedImage {
        camera: &pin,
        cam_from_world: &pose,
        pyramid: &pyr,
    };

    let got = project_unclipped(&equi_view, &p, 1.0).expect("equidistant sees past 90 deg");
    let want = equi
        .ray_to_pixel([pc.x, pc.y, pc.z])
        .expect("the model's own projection");
    assert!(
        (got.0 - want.0).abs() < 1e-12 && (got.1 - want.1).abs() < 1e-12,
        "projection must be the model's: got {got:?} want {want:?}"
    );
    assert!(
        project_unclipped(&pin_view, &p, 1.0).is_none(),
        "the perspective family keeps its half-space cheirality"
    );
}

/// `seed_offset` intersects the observed bearing with the patch plane. Past 90°
/// a bearing can meet the plane *behind* the camera centre — a mirrored hit
/// that is not an observation of the patch — so the ray-path arm rejects a
/// backward intersection instead of returning a plausible-looking offset.
#[test]
fn seed_offset_rejects_a_backward_plane_hit_on_a_ray_path_model() {
    let equi = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    };
    // Camera at the origin looking down world −z; the patch plane is at
    // z = +PLANE_Z, i.e. *behind* the camera, with its normal along −z.
    let pose = RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::y(),
        [HALF_EXTENT, HALF_EXTENT],
    );
    let pyr = ImageU8Pyramid::build(&render_plane_view([0.0, 0.0, 0.0], [0.0, 0.0], texture), 3);
    let view = ProjectedImage {
        camera: &equi,
        cam_from_world: &pose,
        pyramid: &pyr,
    };
    // A pixel near the principal point is a bearing along world −z, which meets
    // the plane at s < 0.
    let forward_px = [IMG_W as f64 / 2.0 + 1.0, IMG_H as f64 / 2.0 + 1.0];
    assert!(
        seed_offset(&patch, &view, forward_px, wpp(), wpp()).is_none(),
        "a bearing that meets the plane behind the camera is not an observation"
    );
    // A bearing on the other side of the horizon does hit the plane forward and
    // must still be accepted.
    let back_px = [
        IMG_W as f64 / 2.0 + FOCAL * std::f64::consts::PI * 0.9,
        IMG_H as f64 / 2.0,
    ];
    assert!(
        seed_offset(&patch, &view, back_px, wpp(), wpp()).is_some(),
        "a bearing toward the plane must still register"
    );
}

/// An allocation no machine can make is a refusal the caller can report, not an
/// abort.
///
/// The buffers here are sized by the caller's search radius and grow as its
/// square, so a wide enough window asks the global allocator for something it
/// cannot give -- and a failure inside the global allocator ends the process
/// where it stands, taking a window and everything unsaved in it. Asking
/// through `try_reserve_exact` is what turns that into a value.
#[test]
fn a_buffer_the_allocator_cannot_give_is_an_error_rather_than_an_abort() {
    // 2^46 lanes is 256 TB of `f32`: no allocator says yes, and none of it is
    // touched.
    let refused = try_zeroed_f32(1 << 46).expect_err("256 TB is not available");
    let LocalizeError::OutOfMemory { bytes } = refused else {
        panic!("the refusal names what it asked for: {refused}");
    };
    assert_eq!(bytes, (1usize << 46) * std::mem::size_of::<f32>());
    assert!(
        refused.to_string().contains("could not allocate"),
        "{refused}"
    );
    // And an ordinary size is still an ordinary buffer.
    assert_eq!(try_zeroed_f32(8).expect("eight lanes"), vec![0.0f32; 8]);
}

/// The shift grids are reserved before the rounds, so a window past what the
/// machine has is refused where the caller can see it rather than inside a
/// `resize` in the search.
#[test]
fn the_shift_grids_are_reserved_before_the_search_uses_them() {
    let mut scratch = SearchScratch::default();
    assert!(scratch.try_reserve_grids(13).is_ok());
    assert!(
        scratch.try_reserve_grids(1 << 24).is_err(),
        "a 16-million-cell span is 2 TB of grids"
    );
}

// ── Aligning every view to the reference render ─────────────────────────────

/// The four fronto cameras most tests use, each rendering `texs[k]` translated
/// in-plane by `offs_grid[k]` patch-grid px.
fn four_view_scene(offs_grid: [[f64; 2]; 4], texs: [fn(f64, f64) -> f64; 4]) -> Scene {
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
    ];
    let offs = offs_grid.map(|[u, v]| [u * wpp(), v * wpp()]);
    Scene::new(&centers, &offs, &texs)
}

/// Where view `view` sees the content the reference (offset 0) sees at the
/// patch centre, when its texture is translated by `off_grid` patch-grid px:
/// the projection of the centre moved by the same world offset.
fn true_keypoint(views: &[ProjectedImage<'_>], view: usize, off_grid: [f64; 2]) -> [f64; 2] {
    let patch = plane_patch();
    let moved = patch.center + Vector3::new(off_grid[0] * wpp(), off_grid[1] * wpp(), 0.0);
    let (x, y) = project(&views[view], &moved, patch.w).unwrap();
    [x, y]
}

fn projection(views: &[ProjectedImage<'_>], view: usize) -> [f64; 2] {
    let patch = plane_patch();
    let (x, y) = project(&views[view], &patch.center, patch.w).unwrap();
    [x, y]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

const STRATEGIES: [SearchStrategy; 2] = [SearchStrategy::PlusDescent, SearchStrategy::Exhaustive];

fn with_strategy(search_strategy: SearchStrategy) -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        search_strategy,
        ..params()
    }
}

#[test]
fn the_reference_is_not_moved_and_the_others_follow_it() {
    // An aligned planar scene, with the reference's starting keypoint
    // deliberately displaced from the truth by `d`. The reference keeps exactly
    // the keypoint it was given, scores 1, and the other views follow it: its
    // render at the displaced keypoint shows the plane moved by `d`, and every
    // camera here sees the plane at the same depth, so each other view moves
    // by about `d` too.
    //
    // A whole grid px of displacement is recovered to a few hundredths of a
    // source px. A fractional one is recovered to about 0.2 source px (under a
    // tenth of a grid px, which is 2.6 source px here): the quadratic through
    // the integer peak's 3×3 neighbourhood is not the exact shape of the
    // correlation peak, and removing that last error is the sub-pixel
    // refiner's job.
    let scene = four_view_scene([[0.0; 2]; 4], [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    for (d, tolerance) in [([src_per_grid(), 0.0], 0.05), ([1.5, -1.0], 0.3)] {
        let p0 = projection(&views, 0);
        let start = [p0[0] + d[0], p0[1] + d[1]];
        let seeds = [Some(start), None, None, None];
        for strategy in STRATEGIES {
            let res = localize_patch_keypoints(
                &patch,
                &views,
                &[0, 1, 2, 3],
                Some(&seeds),
                Some(0),
                &with_strategy(strategy),
            );
            assert_eq!(res.views, vec![0, 1, 2, 3], "{strategy:?}");
            assert_eq!(res.reference, Some(0));
            assert_eq!(
                res.keypoints[0].map(f64::to_bits),
                start.map(f64::to_bits),
                "{strategy:?}: the reference's keypoint is returned bit for bit"
            );
            assert_eq!(res.zncc[0], 1.0);
            for k in 1..4 {
                let want = projection(&views, k);
                let want = [want[0] + d[0], want[1] + d[1]];
                let err = dist(res.keypoints[k], want);
                assert!(
                    err < tolerance,
                    "{strategy:?}: view {k} should follow the reference by {d:?}, \
                 off by {err:.3} px ({:?} vs {want:?})",
                    res.keypoints[k]
                );
                assert!(res.zncc[k] > 0.9, "{strategy:?}: view {k} {}", res.zncc[k]);
            }
        }
    }
}

#[test]
fn planted_offsets_are_recovered_relative_to_the_reference() {
    // Views 1-3 see the texture translated by known sub-grid offsets, and each
    // starts 1-2 px away from where it truly sees the reference's content. The
    // search recovers each view's offset relative to the reference to
    // sub-pixel accuracy (within about a tenth of a grid px, the error of the
    // 3×3 quadratic fit; see `the_reference_is_not_moved_and_the_others_follow_it`).
    let offs = [[0.0, 0.0], [0.6, 0.0], [0.0, -0.7], [0.4, 0.5]];
    let scene = four_view_scene(offs, [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let displacement = [[0.0, 0.0], [1.5, 0.5], [-1.0, 1.5], [1.2, -1.2]];
    let truth: Vec<[f64; 2]> = (0..4).map(|k| true_keypoint(&views, k, offs[k])).collect();
    let seeds: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| {
            Some([
                truth[k][0] + displacement[k][0],
                truth[k][1] + displacement[k][1],
            ])
        })
        .collect();
    for strategy in STRATEGIES {
        let res = localize_patch_keypoints(
            &patch,
            &views,
            &[0, 1, 2, 3],
            Some(&seeds),
            Some(0),
            &with_strategy(strategy),
        );
        assert_eq!(res.views, vec![0, 1, 2, 3], "{strategy:?}");
        for k in 1..4 {
            let err = dist(res.keypoints[k], truth[k]);
            assert!(
                err < 0.35,
                "{strategy:?}: view {k} off its planted offset by {err:.3} px \
                 ({:?} vs {:?}, started {:?} px away)",
                res.keypoints[k],
                truth[k],
                displacement[k]
            );
        }
    }
}

/// A texture whose structure runs along the diagonals, with a different
/// frequency along each, so its correlation surface has a strong `xy` term.
fn diagonal_texture(x: f64, y: f64) -> f64 {
    127.5 + 60.0 * ((x + y) * 9.0).sin() + 35.0 * ((x - y) * 5.0 + 0.5).cos()
}

#[test]
fn the_sub_pixel_fit_finds_the_vertex_of_a_tilted_quadratic() {
    // f(y, x) = −a·(y − y0)² − b·(x − x0)² − c·(y − y0)(x − x0): a peak whose
    // axes are tilted, so the separable parabola through the integer cell is
    // off on both axes. The 3×3 fit is exact on a quadratic.
    let (a, b, c) = (0.3, 0.2, 0.25);
    let (y0, x0) = (0.3, -0.4);
    let f = |y: f64, x: f64| {
        1.0 - a * (y - y0).powi(2) - b * (x - x0).powi(2) - c * (y - y0) * (x - x0)
    };
    let nb = |y: i64, x: i64| Some(f(y as f64, x as f64));
    let (sy, sx) = subpixel_peak(f(0.0, 0.0), 0, 0, nb);
    assert!(
        (sy - y0).abs() < 1e-12 && (sx - x0).abs() < 1e-12,
        "({sy}, {sx})"
    );
    // The separable parabola's answer, for contrast, is off on both axes.
    let sep_y = parabolic(f(0.0, 0.0), f(-1.0, 0.0), f(1.0, 0.0));
    let sep_x = parabolic(f(0.0, 0.0), f(0.0, -1.0), f(0.0, 1.0));
    assert!(
        (sep_y - y0).abs() > 0.1 && (sep_x - x0).abs() > 0.1,
        "({sep_y}, {sep_x})"
    );

    // Without a diagonal neighbour there is no cross term, so the fit falls
    // back to the separable parabola.
    let no_diag = |y: i64, x: i64| (y == 0 || x == 0).then(|| f(y as f64, x as f64));
    assert_eq!(subpixel_peak(f(0.0, 0.0), 0, 0, no_diag), (sep_y, sep_x));

    // A saddle (Hessian not negative definite) falls back the same way.
    let saddle = |y: f64, x: f64| 1.0 - 0.3 * y * y - 0.2 * x * x - 0.6 * y * x;
    let (sy, sx) = subpixel_peak(saddle(0.0, 0.0), 0, 0, |y, x| {
        Some(saddle(y as f64, x as f64))
    });
    let want_y = parabolic(saddle(0.0, 0.0), saddle(-1.0, 0.0), saddle(1.0, 0.0));
    let want_x = parabolic(saddle(0.0, 0.0), saddle(0.0, -1.0), saddle(0.0, 1.0));
    assert_eq!((sy, sx), (want_y, want_x));

    // A vertex past the neighbourhood is clamped to one cell.
    let far = |y: f64, x: f64| 1.0 - 0.05 * (y - 3.0).powi(2) - 0.05 * x * x - 0.01 * y * x;
    let (sy, _) = subpixel_peak(far(0.0, 0.0), 0, 0, |y, x| Some(far(y as f64, x as f64)));
    assert_eq!(sy, 1.0);

    // A missing cardinal neighbour leaves its axis at the integer cell.
    let no_up = |y: i64, x: i64| (y != -1).then(|| f(y as f64, x as f64));
    let (sy, sx) = subpixel_peak(f(0.0, 0.0), 0, 0, no_up);
    assert_eq!((sy, sx), (0.0, sep_x));
}

/// Per-axis error, in patch-grid px, of a keypoint against `truth`.
fn grid_error(got: [f64; 2], truth: [f64; 2]) -> [f64; 2] {
    [
        (got[0] - truth[0]) / src_per_grid(),
        (got[1] - truth[1]) / src_per_grid(),
    ]
}

#[test]
fn a_fractional_shift_on_one_axis_does_not_show_up_on_the_other() {
    // Views 1 and 2 see the texture translated by a fractional shift along x
    // alone, view 3 along y alone. On a texture with diagonal structure, a
    // separate parabola per axis through the integer peak puts part of that
    // shift on the other axis and misses part of it on its own: measured with
    // that fit, the error reached 0.40 grid px on `diagonal_texture` and 0.12
    // on `texture`. The 3×3 quadratic fit's cross term removes the part that
    // comes from the other axis; what remains (under 0.08 grid px) is the
    // quadratic not being the exact shape of the peak. Both strategies share
    // the fit, so both are checked, and they agree with each other.
    for (tex, bound) in [
        (diagonal_texture as fn(f64, f64) -> f64, 0.1),
        (texture as fn(f64, f64) -> f64, 0.06),
    ] {
        for shift in [0.25, 0.5, 0.6, 0.75] {
            let offs = [[0.0, 0.0], [shift, 0.0], [-shift, 0.0], [0.0, shift]];
            let scene = four_view_scene(offs, [tex; 4]);
            let views = scene.views();
            let patch = plane_patch();
            let truth: Vec<[f64; 2]> = (0..4).map(|k| true_keypoint(&views, k, offs[k])).collect();
            // Every view starts at the projection, so the fractional shift is
            // what the sub-pixel fit has to recover.
            let seeds = [None; 4];
            let runs: Vec<KeypointLocalization> = STRATEGIES
                .iter()
                .map(|&strategy| {
                    localize_patch_keypoints(
                        &patch,
                        &views,
                        &[0, 1, 2, 3],
                        Some(&seeds),
                        Some(0),
                        &with_strategy(strategy),
                    )
                })
                .collect();
            for (res, strategy) in runs.iter().zip(STRATEGIES) {
                assert_eq!(res.views, vec![0, 1, 2, 3], "{strategy:?}");
                for (k, (&kp, &want)) in res.keypoints.iter().zip(&truth).enumerate().skip(1) {
                    let e = grid_error(kp, want);
                    assert!(
                        e[0].hypot(e[1]) < bound,
                        "{strategy:?}, shift {shift}: view {k} off by {e:?} grid px"
                    );
                }
            }
            for k in 1..4 {
                let gap = dist(runs[0].keypoints[k], runs[1].keypoints[k]);
                assert!(
                    gap < 1e-3,
                    "shift {shift}: the strategies differ by {gap} px at view {k}"
                );
            }
        }
    }
}

#[test]
fn with_no_reference_given_the_rule_picks_one_and_it_is_not_moved() {
    // No reference: the reference-view rule reads the views' renders at their
    // starting keypoints and its pick is the reference. The result names it,
    // and its keypoint is the one it started at.
    let offs = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.4], [-0.3, 0.3]];
    let scene = four_view_scene(offs, [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let seeds: Vec<Option<[f64; 2]>> = (0..4).map(|k| Some(projection(&views, k))).collect();
    let p = params();
    let rule = crate::patch::stored_bitmap::render_reference(
        &patch,
        &views,
        &set,
        &seeds,
        p.resolution,
        p.sampler,
        &Progress::none(),
    );
    let pick = rule
        .stored_reference()
        .expect("the rule picks a view it would store on this scene");

    for strategy in STRATEGIES {
        let res = localize_patch_keypoints(
            &patch,
            &views,
            &set,
            Some(&seeds),
            None,
            &with_strategy(strategy),
        );
        assert_eq!(res.reference, Some(set[pick]), "{strategy:?}");
        let k = pos(&res, set[pick]).unwrap();
        assert_eq!(res.keypoints[k], seeds[pick].unwrap(), "{strategy:?}");
        assert_eq!(res.zncc[k], 1.0);
        // And the result is the one the same reference given explicitly gives.
        let given = localize_patch_keypoints(
            &patch,
            &views,
            &set,
            Some(&seeds),
            Some(pick),
            &with_strategy(strategy),
        );
        assert_eq!(res.views, given.views);
        assert_eq!(res.keypoints, given.keypoints);
        assert_eq!(res.zncc, given.zncc);
    }
}

#[test]
fn a_changed_reference_realigns_the_views_to_its_render() {
    // Localize with reference A (view 0), then hand the result to a second
    // localization whose reference is B (view 1), with B's start displaced by
    // `e`. The views are aligned again, to B's render: the others move by
    // about `e`, while with A kept as the reference the same starts are pulled
    // back to A's alignment. Which reference is given decides the result.
    //
    // The first localization starts every view where it truly sees A's
    // content, and `e` is a whole grid px, so every shift the search makes is
    // a whole number of grid px and is recovered without the sub-pixel fit's error
    // on fractional shifts. The shift gate is opened: it is not the subject.
    let offs = [[0.0, 0.0], [0.8, 0.0], [0.0, 0.0], [0.0, 0.6]];
    let scene = four_view_scene(offs, [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let e = [0.0, -src_per_grid()];
    let truth: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| Some(true_keypoint(&views, k, offs[k])))
        .collect();
    for strategy in STRATEGIES {
        let p = KeypointLocalizeParams {
            max_shift_px: 1e6,
            ..with_strategy(strategy)
        };
        let a = localize_patch_keypoints(&patch, &views, &set, Some(&truth), Some(0), &p);
        assert_eq!(a.views, set, "{strategy:?}");
        let mut seeds: Vec<Option<[f64; 2]>> = a.keypoints.iter().map(|&kp| Some(kp)).collect();
        let b_start = [a.keypoints[1][0] + e[0], a.keypoints[1][1] + e[1]];
        seeds[1] = Some(b_start);

        let b = localize_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(1), &p);
        assert_eq!(b.views, set, "{strategy:?}");
        assert_eq!(b.reference, Some(1));
        assert_eq!(b.keypoints[1], b_start, "{strategy:?}: B is not moved");
        for k in [0, 2, 3] {
            let want = [a.keypoints[k][0] + e[0], a.keypoints[k][1] + e[1]];
            let err = dist(b.keypoints[k], want);
            assert!(
                err < 0.1,
                "{strategy:?}: view {k} should agree with B's render, off by {err:.3} px"
            );
        }

        let kept = localize_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(0), &p);
        assert_eq!(kept.reference, Some(0));
        for k in 0..4 {
            let err = dist(kept.keypoints[k], a.keypoints[k]);
            assert!(
                err < 0.1,
                "{strategy:?}: with A as the reference view {k} stays aligned to A, \
                 off by {err:.3} px"
            );
        }
        assert!(
            dist(kept.keypoints[0], b.keypoints[0]) > 1.0,
            "{strategy:?}: the two references place view 0 apart"
        );
    }
}

/// Half `texture`, half `occluder_texture`: a view that only partly matches.
fn half_occluded_texture(x: f64, y: f64) -> f64 {
    0.5 * texture(x, y) + 0.5 * occluder_texture(x, y)
}

/// [`params`] with no gate but the ZNCC bars: the shift gate opened, and the
/// absolute floor and relative bar at the values given.
fn only_zncc_gates(min_absolute_zncc: f64, min_relative_zncc: f64) -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        max_shift_px: 1e6,
        min_absolute_zncc,
        min_relative_zncc,
        ..params()
    }
}

#[test]
fn the_absolute_floor_drops_a_view_of_an_unrelated_texture() {
    let scene = four_view_scene([[0.0; 2]; 4], [texture, texture, texture, occluder_texture]);
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let run = |abs: f64| {
        localize_patch_keypoints(
            &patch,
            &views,
            &set,
            None,
            Some(0),
            &only_zncc_gates(abs, 0.0),
        )
    };

    let open = run(0.0);
    assert_eq!(open.views, set, "0.0 turns both bars off");
    assert!(open.zncc[3] < 0.5, "{:?}", open.zncc);
    assert_eq!(run(0.5).views, vec![0, 1, 2], "{:?}", open.zncc);
}

#[test]
fn the_gates_read_the_plain_score_against_the_reference() {
    // View 3 sees a half-occluded surface, so it matches the reference only
    // partly. Each bar, set just above its score, drops it and nothing else;
    // set just below, keeps it.
    let scene = four_view_scene(
        [[0.0; 2]; 4],
        [texture, texture, texture, half_occluded_texture],
    );
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let run = |abs: f64, rel: f64| {
        localize_patch_keypoints(
            &patch,
            &views,
            &set,
            None,
            Some(0),
            &only_zncc_gates(abs, rel),
        )
    };
    let open = run(0.0, 0.0);
    assert_eq!(open.views, set);
    let z3 = open.zncc[3];
    // The relative bar reads the median over the views other than the
    // reference: views 1, 2 and 3.
    let mut others = open.zncc[1..].to_vec();
    others.sort_by(f64::total_cmp);
    let median = others[1];
    assert!(
        z3 > 0.1 && z3 < median - 0.1,
        "the fixture scores view 3 partly: {:?}",
        open.zncc
    );

    assert_eq!(
        run(z3 + 0.01, 0.0).views,
        vec![0, 1, 2],
        "floor above view 3"
    );
    assert_eq!(run(z3 - 0.01, 0.0).views, set, "floor below view 3");
    let ratio = z3 / median;
    assert_eq!(
        run(0.0, ratio + 0.02).views,
        vec![0, 1, 2],
        "bar above view 3"
    );
    assert_eq!(run(0.0, ratio - 0.02).views, set, "bar below view 3");
}

#[test]
fn the_reference_faces_no_gate() {
    // The reference is view 3, of a surface no other view shows, and its start
    // sits further from its projection than `max_shift_px`. Every other view
    // fails both bars against it; the reference stays, as given.
    let scene = four_view_scene([[0.0; 2]; 4], [texture, texture, texture, occluder_texture]);
    let views = scene.views();
    let patch = plane_patch();
    let p3 = projection(&views, 3);
    let start = [p3[0] + 5.0, p3[1]];
    let seeds = [None, None, None, Some(start)];
    let strict = KeypointLocalizeParams {
        min_relative_zncc: 2.0,
        ..params()
    };
    assert!(5.0 > strict.max_shift_px);
    let res = localize_patch_keypoints(
        &patch,
        &views,
        &[0, 1, 2, 3],
        Some(&seeds),
        Some(3),
        &strict,
    );
    assert_eq!(res.views, vec![3]);
    assert_eq!(res.reference, Some(3));
    assert_eq!(res.keypoints[0], start);
    assert_eq!(res.zncc[0], 1.0);
    assert!((res.offsets_px[0] - 5.0).abs() < 1e-9);
}

#[test]
fn view_cache_bytes_counts_a_tile_of_side_r_plus_twice_the_margin() {
    // At the defaults (R = 24, search 6) the tile is 36 px on a side, its rows
    // padded to 56 lanes: four planes of `f32` (three channels and the
    // invalidity plane) and a `bool` validity map, plus the render scratch of
    // ten `f32` lanes and three `u8` channels per pixel.
    let p = KeypointLocalizeParams::default();
    let tile = 56 * 36 * 4 * 4 + 36 * 36;
    let render = 36 * 36 * 10 * 4 + 36 * 36 * 3;
    assert_eq!(view_cache_bytes(&p, 3), tile + render);
    // It grows as the square of the search radius.
    let wide = KeypointLocalizeParams {
        search: 60.0,
        ..p.clone()
    };
    let ratio = view_cache_bytes(&wide, 3) as f64 / view_cache_bytes(&p, 3) as f64;
    assert!(ratio > 10.0, "{ratio}");
}

#[test]
fn a_search_past_the_machine_s_memory_is_refused() {
    let scene = four_view_scene([[0.0; 2]; 4], [texture; 4]);
    let views = scene.views();
    let p = KeypointLocalizeParams {
        search: 1e7,
        ..params()
    };
    let refused = try_localize_patch_keypoints(
        &plane_patch(),
        &views,
        &[0, 1, 2, 3],
        None,
        Some(0),
        &p,
        &Progress::none(),
    )
    .expect_err("petabytes of search buffers are not available");
    assert!(
        matches!(refused, LocalizeError::OutOfMemory { .. }),
        "{refused}"
    );
}

#[test]
fn a_cancelled_progress_stops_the_localization() {
    let scene = four_view_scene([[0.0; 2]; 4], [texture; 4]);
    let views = scene.views();
    let flag = std::sync::atomic::AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    for reference in [Some(0), None] {
        let stopped = try_localize_patch_keypoints(
            &plane_patch(),
            &views,
            &[0, 1, 2, 3],
            None,
            reference,
            &params(),
            &progress,
        );
        assert_eq!(stopped.err(), Some(LocalizeError::Cancelled));
    }
    let cloud = PatchCloud {
        patches: vec![plane_patch(); 3],
        point_indexes: vec![0, 1, 2],
    };
    let stopped = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        &vec![vec![0u32, 1, 2, 3]; 3],
        None,
        None,
        &params(),
        None,
        &progress,
    );
    assert!(stopped.is_err());
}

#[test]
fn a_reference_out_of_frame_leaves_every_view_at_its_start() {
    // The reference's keypoint is far outside its frame, so its tile cannot be
    // rendered there and there is no template. There is then no reference:
    // every view is kept at its start with an unscored ZNCC, and still faces
    // the shift gate, which drops the reference's own 5000 px start.
    let scene = four_view_scene([[0.0; 2]; 4], [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let mut seeds: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| {
            let p = projection(&views, k);
            Some([p[0] + 0.5, p[1] - 0.5])
        })
        .collect();
    let p0 = projection(&views, 0);
    seeds[0] = Some([p0[0] + 5000.0, p0[1]]);
    for strategy in STRATEGIES {
        let res = localize_patch_keypoints(
            &patch,
            &views,
            &[0, 1, 2, 3],
            Some(&seeds),
            Some(0),
            &with_strategy(strategy),
        );
        assert_eq!(res.reference, None, "{strategy:?}");
        assert_eq!(res.views, vec![1, 2, 3], "{strategy:?}");
        assert!(res.zncc.iter().all(|z| z.is_nan()), "{:?}", res.zncc);
        for (k, kp) in res.keypoints.iter().enumerate() {
            let err = dist(*kp, seeds[k + 1].unwrap());
            assert!(err < 1e-6, "{strategy:?}: view {} moved {err} px", k + 1);
        }
    }
}

#[test]
fn a_grazing_reference_is_turned_away_and_the_rule_picks() {
    // View 3 is oblique; given as the reference under a strict grazing cutoff,
    // it is pre-filtered, and the rule picks the reference among the others.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [1.5, 0.0, 0.0],
    ];
    let scene = Scene::new(
        &centers,
        &[[0.0; 2]; 4],
        &[texture as fn(f64, f64) -> f64; 4],
    );
    let views = scene.views();
    let patch = plane_patch();
    let strict = KeypointLocalizeParams {
        min_grazing_cos: 0.95,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(3), &strict);
    assert!(pos(&res, 3).is_none(), "{:?}", res.views);
    let reference = res
        .reference
        .expect("the rule picks among the facing views");
    assert_ne!(reference, 3);
    assert_eq!(res.zncc[pos(&res, reference).unwrap()], 1.0);
}

#[test]
fn a_reference_given_at_a_repeated_image_is_kept() {
    // Image 1 is listed twice and the reference is given at its second slot,
    // which deduplication drops. The reference is matched by image.
    let scene = four_view_scene([[0.0; 2]; 4], [texture; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 1, 2], None, Some(2), &params());
    assert_eq!(res.views, vec![0, 1, 2]);
    assert_eq!(res.reference, Some(1));
    assert_eq!(res.zncc[1], 1.0);
}
