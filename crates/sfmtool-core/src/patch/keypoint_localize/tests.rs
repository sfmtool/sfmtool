// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// A synthetic scene mirroring the view_selection tests: pinhole cameras (identity
// rotation, looking down +z) viewing a textured world plane at z = PLANE_Z. The
// patch sits on that plane with a normal pointing back toward the cameras (-z).
//
// To exercise *registration*, each view can render the plane texture translated
// in-plane by a per-view world offset `o_k`: the same patch then renders, in view
// k, content shifted by `-o_k`, so the views disagree until congealing shifts each
// by `o_k`. The shift the kernel must recover for view k is `acc_k = o_k / wpp`
// patch-grid px (`wpp = 2·half_extent / R`).

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

/// Synthesize the image a pinhole camera at `center` (looking down +z) sees of
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
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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

    /// Cameras at `centers` (identity rotation), each viewing the **same**
    /// direction-only texture for a point at infinity — appearance depends only on
    /// the ray direction, so camera translation is irrelevant (no parallax). Per
    /// view the directional texture is shifted by the matching angular `offset`.
    fn infinity(centers: &[[f64; 3]], offsets: &[[f64; 2]]) -> Self {
        let cams = centers.iter().map(|_| pinhole()).collect();
        let poses = centers
            .iter()
            .map(|c| {
                RigidTransform::from_wxyz_translation([1.0, 0.0, 0.0, 0.0], [-c[0], -c[1], -c[2]])
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

/// Synthesize what an identity-rotation pinhole sees of a point at infinity in
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

/// The index into `res.views` where image `i` was kept, if any.
fn pos(res: &KeypointLocalization, i: u32) -> Option<usize> {
    res.views.iter().position(|&v| v == i)
}

#[test]
fn aligned_views_keep_all_and_barely_shift() {
    // Every view sees the same texture, perfectly aligned -> congealing should find
    // no residual shift, keep all views, and land each keypoint on its projection.
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

    assert_eq!(res.views, vec![0, 1, 2, 3], "all aligned views kept");
    for &o in &res.offsets_px {
        assert!(o < 0.6, "aligned view should barely move, got {o} px");
    }
    for &z in &res.loo_zncc {
        assert!(z > 0.8, "aligned views should co-register, LOO {z}");
    }
}

#[test]
fn infinity_point_views_co_register_independent_of_translation() {
    // A point at infinity (+z) seen by identity-rotation cameras at very different
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

    assert_eq!(
        res.views,
        vec![0, 1, 2, 3],
        "all aligned infinity views kept"
    );
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    for (k, &i) in res.views.iter().enumerate() {
        // +z projects to the principal point under identity rotation, the same in
        // every camera regardless of translation.
        let pj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        assert!((pj.0 - cx).abs() < 1e-6 && (pj.1 - cy).abs() < 1e-6);
        assert!((res.keypoints[k][0] - cx).abs() < 0.6 && (res.keypoints[k][1] - cy).abs() < 0.6);
        assert!(
            res.offsets_px[k] < 0.6,
            "aligned infinity view barely moves"
        );
    }
    for &z in &res.loo_zncc {
        assert!(z > 0.8, "infinity views co-register, LOO {z}");
    }
}

#[test]
fn infinity_point_seed_offsets_congeal_back() {
    // Identical content across views; three views seeded at the projection pin the
    // gauge, the fourth is seeded a few source px off. Congealing must pull the
    // off view back to alignment — exercises the w == 0 branch of seed_offset
    // (angular ray→offset inversion) and the w == 0 render/project path through
    // the congealing loop.
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
    let seeds = [[cx, cy], [cx, cy], [cx, cy], [cx + 4.0, cy]];

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), &params());

    let p3 = pos(&res, 3).expect("the seeded-off view congeals back and is kept");
    assert!(
        (res.keypoints[p3][0] - cx).abs() < 1.0 && (res.keypoints[p3][1] - cy).abs() < 1.0,
        "seeded-off infinity view should congeal back to the projection, got {:?}",
        res.keypoints[p3]
    );
    for &z in &res.loo_zncc {
        assert!(z > 0.8, "post-congeal infinity LOO should be high, got {z}");
    }
}

#[test]
fn congeals_misregistered_view_into_alignment() {
    // Views 0,1,2 are aligned (they pin the gauge); view 3 sees the texture shifted
    // by +1 patch-grid px in x. Congealing should recover acc_3 ≈ +1, putting its
    // keypoint ~1 grid px (in source px) off its projection while the aligned views
    // stay put. All four co-register, so view 3 is kept (its shift < max_shift_px).
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

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
    // After registration every view agrees well.
    for &z in &res.loo_zncc {
        assert!(z > 0.9, "post-congeal LOO should be high, got {z}");
    }
}

#[test]
fn drops_disagreeing_surface_view() {
    // Three views see the same surface; view 3 shows a different surface. It cannot
    // register, so its leave-one-out ZNCC falls below the relative bar and it is
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

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
    // re-registers (high LOO), it is dropped for sitting too far from its projection.
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

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
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &strict);
    assert!(
        pos(&res, 3).is_none(),
        "oblique view should be grazing-filtered: {:?}",
        res.views
    );

    let permissive = KeypointLocalizeParams {
        min_grazing_cos: 0.1,
        ..params()
    };
    let res2 = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &permissive);
    assert!(
        pos(&res2, 3).is_some(),
        "with a permissive cutoff the oblique view is kept: {:?}",
        res2.views
    );
}

#[test]
fn fewer_than_two_views_returns_seed_projection() {
    // A single-view set can't congeal; the view's keypoint is its projection.
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(&patch, &views, &[0], None, &params());
    assert_eq!(res.views, vec![0]);
    let proj = project(&views[0], &patch.center, patch.w).unwrap();
    assert!((res.keypoints[0][0] - proj.0).abs() < 1e-9);
    assert!((res.keypoints[0][1] - proj.1).abs() < 1e-9);
    assert!(res.loo_zncc[0].is_nan(), "no LOO consensus for a lone view");
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
    let res = localize_patch_keypoints(&patch, &views, &[0, 0, 1, 2], None, &params());
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
        point_ids: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2, 3], vec![0u32, 1, 2, 3]];

    let batch = localize_patch_cloud_keypoints(&cloud, &views, &view_sets, None, &params());
    assert_eq!(batch.len(), 2);
    for (i, res) in batch.iter().enumerate() {
        let single =
            localize_patch_keypoints(&cloud.patches[i], &views, &view_sets[i], None, &params());
        assert_eq!(res.views, single.views);
        for (a, b) in res.keypoints.iter().zip(&single.keypoints) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
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

    let seeds: Vec<[f64; 2]> = (0..3)
        .map(|i| {
            let (x, y) = project(&views[i], &patch.center, patch.w).unwrap();
            [x, y]
        })
        .collect();
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2], Some(&seeds), &params());
    assert_eq!(res.views, vec![0, 1, 2]);
    for &o in &res.offsets_px {
        assert!(
            o < 0.6,
            "projection-seeded aligned view should barely move: {o}"
        );
    }
}

#[test]
fn seed_offset_unprojection_round_trips_on_lone_view() {
    // A non-projection seed exercises `seed_offset`'s unprojection (rotation
    // transpose, ray∩plane, /wpp). On a lone-view set the kernel returns the seed
    // straight through (no congealing), so the emitted keypoint must equal the seed
    // — i.e. seed_offset is the exact inverse of finalize's projection. (The
    // projection-seeded round_trips test above only seeds at acc≈0, so this is the
    // only check of a *non-zero* unprojected seed.)
    let centers = [[0.4, 0.2, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let proj = project(&views[0], &patch.center, patch.w).unwrap();
    let seed = [proj.0 + 5.0, proj.1 - 3.0]; // a few px off the projection
    let res = localize_patch_keypoints(&patch, &views, &[0], Some(&[seed]), &params());
    assert_eq!(res.views, vec![0]);
    assert!(
        (res.keypoints[0][0] - seed[0]).abs() < 1e-6
            && (res.keypoints[0][1] - seed[1]).abs() < 1e-6,
        "seed {seed:?} should round-trip through seed_offset, got {:?}",
        res.keypoints[0]
    );
    // And the reported offset is the seed's distance from the projection.
    let want = ((seed[0] - proj.0).powi(2) + (seed[1] - proj.1).powi(2)).sqrt();
    assert!((res.offsets_px[0] - want).abs() < 1e-6);
}

#[test]
fn drops_low_relative_zncc_view_in_isolation() {
    // Isolate the relative-LOO drop gate: three aligned views plus an occluder, but
    // with `max_shift_px` set so high that only a low leave-one-out ZNCC can drop
    // a view. The occluder cannot register, so it (and only it) is dropped.
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
        max_shift_px: 1e6, // disable the shift gate so only the LOO bar can drop
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &p);
    assert!(
        pos(&res, 3).is_none(),
        "occluder must be dropped by the relative-LOO bar: {:?}",
        res.views
    );
    assert_eq!(res.views, vec![0, 1, 2]);
}

#[test]
fn two_view_floor_keeps_exactly_two_when_all_fail() {
    // With `min_relative_zncc > 1` the bar `min_relative_zncc × median` is
    // unsatisfiable (every LOO ZNCC is below it), so the gates would drop every
    // view. The two-view leave-one-out floor must instead retain exactly two (the
    // best-agreeing), never zero and never all three.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let p = KeypointLocalizeParams {
        min_relative_zncc: 1.5,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2], None, &p);
    assert_eq!(
        res.views.len(),
        2,
        "floor keeps exactly two: {:?}",
        res.views
    );
}

#[test]
fn converges_early_and_extra_rounds_are_idempotent() {
    // One round already recovers a 1-grid misregistration (the integer search range
    // spans it), and once converged extra rounds change nothing.
    let ox = 1.0 * wpp();
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

    let one = KeypointLocalizeParams {
        max_iters: 1,
        ..params()
    };
    let res1 = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &one);
    let p3 = pos(&res1, 3).expect("one round keeps the misregistered view");
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res1.keypoints[p3][0] - proj3.0;
    // ox = 1 grid px, so the expected source-px recovery is src_per_grid().
    assert!(
        (dx - src_per_grid()).abs() < 0.5 * src_per_grid(),
        "a single round should already recover most of the shift, got {dx:.2}px"
    );

    // Converged result is stable: 5 vs 50 rounds give identical keypoints.
    let many = KeypointLocalizeParams {
        max_iters: 50,
        ..params()
    };
    let res5 = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
    let res50 = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &many);
    assert_eq!(res5.views, res50.views);
    for (a, b) in res5.keypoints.iter().zip(&res50.keypoints) {
        assert!(
            (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9,
            "extra rounds past convergence changed the result"
        );
    }

    // Directly observe the `convergence_px` early-exit: a huge threshold forces the
    // loop to stop after round 1, so 50 rounds must equal 1 round exactly. A kernel
    // that ignored `convergence_px` (always ran `max_iters`) would diverge here.
    let early = KeypointLocalizeParams {
        max_iters: 50,
        convergence_px: 1e9,
        ..params()
    };
    let res_early = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &early);
    assert_eq!(
        res_early.views, res1.views,
        "early-exit must match a single round"
    );
    for (a, b) in res_early.keypoints.iter().zip(&res1.keypoints) {
        assert!(
            (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9,
            "convergence_px early-exit did not stop after round 1"
        );
    }
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

    let res = localize_patch_keypoints(&patch, &views, &[], None, &params());
    assert!(res.views.is_empty());
    assert!(res.keypoints.is_empty());
    assert!(res.offsets_px.is_empty());
    assert!(res.loo_zncc.is_empty());
}

// ── Accumulation search_shift vs. the per-candidate reference ────────────────

/// Build a window support over the `R×R` core (mirrors `localize_patch_keypoints`).
fn disk_support(resolution: usize) -> Support {
    let w_full = window_weights(PatchWindow::GaussianDisk { sigma: 0.6 }, resolution as u32);
    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for (p, &w) in w_full.iter().enumerate() {
        if w > 0.0 {
            pixels.push(p);
            weights.push(w);
        }
    }
    let total_w: f64 = weights.iter().sum();
    let sqrt_w: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    Support {
        pixels,
        weights,
        sqrt_w,
        total_w,
    }
}

/// A deterministic, textured `cr×cr×channels` context tile. `flat_last` forces the
/// final channel constant (to exercise the flat-channel path).
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
    ContextTile {
        px,
        valid: vec![true; cr * cr],
        channels,
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
    cr: usize,
    resolution: usize,
    margin: i64,
    dy0: i64,
    dx0: i64,
) -> Vec<f32> {
    let n = support.pixels.len();
    let mut raw = vec![0f32; tile.channels * n];
    let oy = (margin + dy0) as usize;
    let ox = (margin + dx0) as usize;
    assert!(extract_core(
        tile, support, cr, resolution, oy, ox, &mut raw
    ));
    let mut tmpl = vec![0f32; channels * n];
    znorm_core(&raw, support, keep_mask, &mut tmpl);
    tmpl
}

fn assert_search_eq(got: Option<(f64, f64, f64)>, want: Option<(f64, f64, f64)>) {
    match (got, want) {
        (Some(g), Some(w)) => {
            assert!((g.0 - w.0).abs() < 1e-4, "dx {} vs {}", g.0, w.0);
            assert!((g.1 - w.1).abs() < 1e-4, "dy {} vs {}", g.1, w.1);
            assert!((g.2 - w.2).abs() < 1e-5, "peak {} vs {}", g.2, w.2);
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
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, cr, resolution, margin, dy0, dx0,
    );

    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, cr, resolution, margin,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, cr, resolution, margin,
    );
    assert_search_eq(got, want);
    // And it actually recovered the planted shift.
    let g = got.unwrap();
    assert!((g.0 - dx0 as f64).abs() < 0.25 && (g.1 - dy0 as f64).abs() < 0.25);
    assert!(g.2 > 0.99, "self-template peak should be ≈1, got {}", g.2);
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
    // (exercises the validity grid → some shifts unscorable).
    let mut tile = synthetic_tile(cr, channels, true);
    for row in 0..cr {
        for col in 0..cr {
            if row < 2 || col < 2 {
                tile.valid[row * cr + col] = false;
            }
        }
    }

    let (dy0, dx0) = (2i64, 1i64);
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, cr, resolution, margin, dy0, dx0,
    );
    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, cr, resolution, margin,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, cr, resolution, margin,
    );
    assert_search_eq(got, want);
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
    let tmpl = template_at(
        &tile, &support, &keep_mask, channels, cr, resolution, margin, dy0, dx0,
    );
    let mut sc = SearchScratch {
        tmpl: tmpl.clone(),
        ..Default::default()
    };
    let got = search_shift(
        &tile, &mut sc, &support, &keep_mask, channels, cr, resolution, margin,
    );
    let want = search_shift_ref(
        &tile, &tmpl, &support, &keep_mask, channels, cr, resolution, margin,
    );
    assert_search_eq(got, want);
}
