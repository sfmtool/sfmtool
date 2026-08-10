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
        // The congealing-behavior tests are the uncapped reference; the cap
        // (production default 8) is exercised by the basis tests' `capped(k)`.
        basis_max_views: 0,
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

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
    // the congealing loop. Pinned to `SearchStrategy::Exhaustive` because the
    // 4-source-px seed shift puts view 3 ~3 grid steps from consensus through
    // a multi-modal angular `dir_texture`; the default `PlusDescent` walks
    // into a local maximum on the way home, which is the documented trade-off
    // (`PlusDescent` keeps the ~1.9× wall win on real data at the cost of a
    // long-walk accuracy tail). The capability tested here — congealing back
    // from a non-trivial seed offset — is an `Exhaustive` guarantee.
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), &exhaustive);

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
fn search_resolution_multiplier_one_is_a_noop() {
    // `search_resolution_multiplier = 1.0` (the explicit default) must produce
    // byte-identical results to leaving the knob at its `Default`, on a scene that
    // exercises both congealing (view 3 shifted) and the kept-view set.
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

    let baseline = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
    let explicit_one = KeypointLocalizeParams {
        search_resolution_multiplier: 1.0,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &explicit_one);

    assert_eq!(res.views, baseline.views);
    for (a, b) in res.keypoints.iter().zip(&baseline.keypoints) {
        assert!(
            (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12,
            "m = 1.0 must be a no-op: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn supersampled_search_resolution_still_congeals() {
    // With `m = 2.0` the search runs at R_s = 2·R (a finer grid; one integer step
    // is 1/2 patch-grid px). A 1-grid-px misregistration must still be recovered and
    // the recovered keypoint scaled back to patch-grid px (the `1/m` factor folded
    // into the R_s `wpp`), so the same +src_per_grid x-recovery as at m = 1.
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

    let p = KeypointLocalizeParams {
        search_resolution_multiplier: 2.0,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &p);

    let p3 = pos(&res, 3).expect("the misregistered view co-registers and is kept");
    let expected_px = shift_grid * src_per_grid();
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res.keypoints[p3][0] - proj3.0;
    let dy = res.keypoints[p3][1] - proj3.1;
    assert!(
        (dx - expected_px).abs() < 0.4 * src_per_grid(),
        "m = 2 should still recover +{expected_px:.2}px in x, got {dx:.2}px"
    );
    assert!(
        dy.abs() < 0.4 * src_per_grid(),
        "m = 2: view 3 should not move in y, got {dy:.2}px"
    );
    for &z in &res.loo_zncc {
        assert!(z > 0.9, "m = 2 post-congeal LOO should be high, got {z}");
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
        point_indexes: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2, 3], vec![0u32, 1, 2, 3]];

    let batch =
        localize_patch_cloud_keypoints(&cloud, &views, &view_sets, None, None, &params(), None);
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

    let seeds: Vec<Option<[f64; 2]>> = (0..3)
        .map(|i| {
            let (x, y) = project(&views[i], &patch.center, patch.w).unwrap();
            Some([x, y])
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
    let res = localize_patch_keypoints(&patch, &views, &[0], Some(&[Some(seed)]), &params());
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

    let unseeded = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
    let all_none =
        localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&[None; 4]), &params());

    assert_eq!(all_none.views, unseeded.views);
    assert_eq!(all_none.keypoints, unseeded.keypoints);
    assert_eq!(all_none.offsets_px, unseeded.offsets_px);
    assert_eq!(all_none.loo_zncc, unseeded.loo_zncc);
    assert_eq!(all_none.is_basis, unseeded.is_basis);
}

#[test]
fn mixed_seeds_apply_per_view() {
    // The mixed table: views 0-2 carry no seed (`None` — they anchor at their
    // projections, as an expansion candidate with no observation does), view 3
    // carries an explicit seed a few source px off the aligned content. The
    // unseeded views pin the gauge and congealing pulls the seeded one home, so
    // the run matches the equivalent all-`Some` table where 0-2 are seeded at
    // their own projections. Pinned to `Exhaustive` for the same reason
    // `infinity_point_seed_offsets_congeal_back` is: the long walk home from a
    // multi-px seed is an `Exhaustive` guarantee.
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

    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&mixed), &exhaustive);
    let ref_res =
        localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&all_some), &exhaustive);

    assert_eq!(res.views, vec![0, 1, 2, 3], "every view is kept");
    assert_eq!(res.views, ref_res.views);
    for (k, (a, b)) in res.keypoints.iter().zip(&ref_res.keypoints).enumerate() {
        assert!(
            (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6,
            "view {k}: an unseeded entry must match a projection-seeded one, {a:?} vs {b:?}"
        );
    }
    // The unseeded views started (and stay) at their projections; the seeded one
    // congeals back onto the same aligned content.
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
        "the seeded view should congeal back onto the aligned content, got {}",
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
    );

    let single0 = localize_patch_keypoints(
        &cloud.patches[0],
        &views,
        &view_sets[0],
        Some(&seeds[0]),
        &params(),
    );
    let single1 =
        localize_patch_keypoints(&cloud.patches[1], &views, &view_sets[1], None, &params());
    assert_eq!(batch[0].views, single0.views);
    assert_eq!(batch[0].keypoints, single0.keypoints);
    assert_eq!(batch[1].views, single1.views);
    assert_eq!(batch[1].keypoints, single1.keypoints);
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
fn stable_scene_converges_before_max_iters() {
    // A perfectly aligned scene reaches positional stationarity almost
    // immediately, so at the DEFAULT `convergence_px` the loop must exit well
    // before a generous `max_iters` — pinning the round-over-round-change
    // convergence metric. (Under the old absolute-offset metric the freshly
    // recomputed parabolic residual kept `mean_shift` above the threshold and
    // every point ran all `max_iters` rounds; this test would fail with
    // `rounds == 40`.)
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

    let p = KeypointLocalizeParams {
        max_iters: 40,
        ..params()
    };
    let res = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &p);
    assert_eq!(res.views.len(), 4, "aligned views must all be kept");
    assert!(
        (1..40).contains(&res.rounds),
        "a stable scene must converge before max_iters at the default \
         convergence_px, ran {} rounds",
        res.rounds
    );
}

#[test]
fn no_convergence_in_a_round_that_dropped_views() {
    // A round that drops a view changes the consensus the survivors registered
    // against, so convergence must not fire in that same round — the survivors
    // get at least one more round against the survivor-only template. With an
    // unsatisfiable ZNCC bar the first round culls 3 views -> the two-view
    // floor; positional stationarity is immediate (aligned scene), but the
    // membership guard must still force a second round.
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
    assert_eq!(res.views.len(), 2, "floor keeps exactly two");
    assert!(
        res.rounds >= 2,
        "convergence fired in the same round that dropped views (rounds = {})",
        res.rounds
    );
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
/// well-posed congealing scene (aligned plus one misregistered view). Locks
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

    let a = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &plus);
    let b = localize_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &exhaustive);

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

#[test]
fn incremental_loo_template_matches_reference() {
    // The Gram-space incremental LOO consensus (`loo_consensus_template`) must
    // reproduce the compacted-holdout-stack reference — copy the other views'
    // rows, `irls_view_weights`, `weighted_unit_template_into` — for every
    // holdout, within float-accumulation tolerance (f64 Gram algebra vs the
    // f32 SAXPY/residual path). Includes an outlier view so the Tukey cutoff
    // (weight → 0) actually fires on some holdout sets.
    let (nv, kept_ch, n) = (6usize, 3usize, 40usize);
    let cn = kept_ch * n;

    // Deterministic pseudo-random stack (LCG), values in [-1, 1]; view 5 is a
    // gross outlier (a different generator stream, amplified).
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let base: Vec<f32> = (0..cn).map(|_| next() as f32).collect();
    let mut xs = vec![0f32; nv * cn];
    for v in 0..nv {
        for k in 0..cn {
            let noise = next() as f32 * 0.05;
            xs[v * cn + k] = if v == 5 {
                // Outlier: unrelated content at 3x amplitude.
                3.0 * next() as f32
            } else {
                base[k] + noise
            };
        }
    }

    let robust_iters = 3;
    let mut loo = LooScratch::default();
    build_loo_gram(&xs, nv, cn, &mut loo);

    let mut sc = ConsensusScratch::default();
    let mut loo_xs = vec![0f32; (nv - 1) * cn];
    let mut got = Vec::new();
    let mut want = Vec::new();
    for v in 0..nv {
        // Reference: compact the holdout stack, IRLS, unit template.
        let mut w = 0;
        for u in 0..nv {
            if u == v {
                continue;
            }
            loo_xs[w * cn..][..cn].copy_from_slice(&xs[u * cn..][..cn]);
            w += 1;
        }
        irls_view_weights(&loo_xs, w, kept_ch, n, robust_iters, None, &mut sc);
        weighted_unit_template_into(&loo_xs, &sc.w, w, kept_ch, n, &mut want);

        // Production: Gram-space IRLS + skip-materialization.
        loo_consensus_template(&xs, nv, v, kept_ch, n, robust_iters, &mut loo, &mut got);

        assert_eq!(got.len(), want.len(), "holdout {v}: template length");
        for (k, (&g, &r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() < 1e-4,
                "holdout {v}, element {k}: incremental {g} vs reference {r}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Consensus-basis cap (specs/core/keypoint-localization-consensus-basis.md)
// ---------------------------------------------------------------------------

/// A ring of `n` cameras around the patch, all seeing the same aligned texture
/// except the entries named in `shifted` (which see it translated by that many
/// patch-grid px in world-x) and those in `odd_tex` (a different surface).
fn ring_scene(n: usize, shifted: &[(usize, f64)], odd_tex: &[usize]) -> (Scene, Vec<u32>) {
    let mut centers = Vec::with_capacity(n);
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        centers.push([0.45 * a.cos(), 0.45 * a.sin(), 0.0]);
    }
    let mut offs = vec![[0.0, 0.0]; n];
    for &(i, s) in shifted {
        offs[i] = [s * wpp(), 0.0];
    }
    let mut texs = vec![texture as fn(f64, f64) -> f64; n];
    for &i in odd_tex {
        texs[i] = occluder_texture as fn(f64, f64) -> f64;
    }
    let scene = Scene::new(&centers, &offs, &texs);
    let set: Vec<u32> = (0..n as u32).collect();
    (scene, set)
}

fn capped(k: u32) -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        basis_max_views: k,
        ..params()
    }
}

/// The kept views that congealed as consensus-basis members.
fn basis_views(res: &KeypointLocalization) -> Vec<u32> {
    res.views
        .iter()
        .zip(&res.is_basis)
        .filter_map(|(&v, &b)| b.then_some(v))
        .collect()
}

#[test]
fn basis_cap_off_is_bit_identical_to_the_uncapped_path() {
    let (scene, set) = ring_scene(10, &[(3, 1.0), (7, -1.0)], &[]);
    let views = scene.views();
    let patch = plane_patch();
    let scores: Vec<f64> = (0..set.len()).map(|i| 0.9 - 0.01 * i as f64).collect();
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 3,
    };

    let base = localize_patch_keypoints(&patch, &views, &set, None, &params());
    // K = 0 with evidence supplied must not consult it at all.
    let off = localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(0));
    // A cap at or above the view count is likewise the uncapped path.
    let wide =
        localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(10));
    let wider =
        localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(64));

    for other in [&off, &wide, &wider] {
        assert_eq!(other.views, base.views);
        assert_eq!(other.keypoints, base.keypoints, "keypoints bit-identical");
        assert_eq!(other.offsets_px, base.offsets_px);
        assert_eq!(other.loo_zncc, base.loo_zncc);
        assert_eq!(other.rounds, base.rounds);
        assert!(
            other.is_basis.iter().all(|&b| b),
            "every kept view is a basis member when the cap does not bite"
        );
    }
}

#[test]
fn tail_views_recover_their_planted_shifts_against_the_basis_template() {
    // 12 views; two of them (8 and 11) see the texture shifted by +1 / -1
    // patch-grid px. With K = 4 both land in the tail (the basis takes the
    // top-scoring views 0..3), so they must recover their planted shift from a
    // single search against the finished basis template.
    let (scene, set) = ring_scene(12, &[(8, 1.0), (11, -1.0)], &[]);
    let views = scene.views();
    let patch = plane_patch();
    // Rank the aligned leading views highest so the basis is exactly 0..3.
    let scores: Vec<f64> = (0..12).map(|i| 0.9 - 0.01 * i as f64).collect();
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 0,
    };
    let p = KeypointLocalizeParams {
        search_strategy: SearchStrategy::Exhaustive,
        ..capped(4)
    };

    let res = localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &p);

    assert_eq!(basis_views(&res), vec![0, 1, 2, 3], "exactly K congealed");
    for (want_sign, i) in [(1.0, 8u32), (-1.0, 11u32)] {
        let pi = pos(&res, i).expect("planted-shift tail view is kept");
        assert!(!res.is_basis[pi], "view {i} should be a tail view");
        let proj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        let dx = res.keypoints[pi][0] - proj.0;
        let dy = res.keypoints[pi][1] - proj.1;
        let expected = want_sign * src_per_grid();
        assert!(
            (dx - expected).abs() < 0.35 * src_per_grid(),
            "tail view {i} should recover dx {expected:.3}, got {dx:.3}"
        );
        assert!(dy.abs() < 0.35 * src_per_grid(), "tail view {i} dy {dy:.3}");
    }
}

#[test]
fn tail_gate_drops_a_mismatched_tail_view() {
    // View 9 renders a different surface entirely; ranked last, it lands in the
    // tail and must fail the relative-ZNCC gate against the basis template.
    let (scene, set) = ring_scene(10, &[], &[9]);
    let views = scene.views();
    let patch = plane_patch();
    let mut scores: Vec<f64> = vec![0.9; 10];
    scores[9] = 0.1;
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 0,
    };

    let res = localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(4));

    assert!(
        pos(&res, 9).is_none(),
        "the wrong-surface tail view must be dropped, kept {:?}",
        res.views
    );
    assert_eq!(res.views, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(res.is_basis.len(), res.views.len());
}

#[test]
fn capped_result_preserves_input_order_and_parallel_arrays() {
    let (scene, set) = ring_scene(9, &[(5, 1.0)], &[]);
    let views = scene.views();
    let patch = plane_patch();
    // Score the LAST views highest so the basis is not a prefix of the input —
    // the merge must still report kept views in input order.
    let scores: Vec<f64> = (0..9).map(|i| 0.1 + 0.05 * i as f64).collect();
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 0,
    };

    let res = localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(3));

    assert!(
        res.views.windows(2).all(|w| w[0] < w[1]),
        "kept views must stay in input order, got {:?}",
        res.views
    );
    assert_eq!(res.keypoints.len(), res.views.len());
    assert_eq!(res.offsets_px.len(), res.views.len());
    assert_eq!(res.loo_zncc.len(), res.views.len());
    assert_eq!(res.is_basis.len(), res.views.len());
    assert_eq!(basis_views(&res), vec![6, 7, 8]);
}

#[test]
fn force_track_puts_the_track_views_in_the_basis() {
    // The track views (the leading 3 entries) score worst; with the default
    // `basis_force_track_views` they still claim the basis seats.
    let (scene, set) = ring_scene(10, &[], &[]);
    let views = scene.views();
    let patch = plane_patch();
    let mut scores: Vec<f64> = vec![0.9; 10];
    scores[0] = 0.2;
    scores[1] = 0.2;
    scores[2] = 0.2;
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 3,
    };

    let forced =
        localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(3));
    assert_eq!(
        basis_views(&forced),
        vec![0, 1, 2],
        "track views claim the basis seats"
    );

    let free = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        None,
        evidence,
        &KeypointLocalizeParams {
            basis_force_track_views: false,
            ..capped(3)
        },
    );
    assert_eq!(
        basis_views(&free),
        vec![3, 4, 5],
        "without the reservation, by score"
    );
}

#[test]
fn unscored_views_fall_back_to_the_grazing_rank() {
    // No caller scores at all: the pick ranks by |dÂ·n| (most frontal first).
    // The ring cameras are symmetric, so this only has to be deterministic and
    // produce a well-formed capped result.
    let (scene, set) = ring_scene(8, &[], &[]);
    let views = scene.views();
    let patch = plane_patch();

    let a = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        None,
        BasisEvidence::default(),
        &capped(3),
    );
    let b = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        None,
        BasisEvidence::default(),
        &capped(3),
    );
    assert_eq!(a.views, b.views);
    assert_eq!(a.keypoints, b.keypoints);
    assert_eq!(a.is_basis, b.is_basis);
    assert_eq!(a.is_basis.iter().filter(|&&x| x).count(), 3);
}

#[test]
fn batch_threads_the_basis_inputs_per_patch() {
    let (scene, set) = ring_scene(10, &[], &[]);
    let views = scene.views();
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let view_sets = vec![set.clone(), set.clone()];
    let scores = vec![
        (0..10).map(|i| 0.9 - 0.01 * i as f64).collect::<Vec<f64>>(),
        (0..10).map(|i| 0.1 + 0.05 * i as f64).collect::<Vec<f64>>(),
    ];
    let counts = vec![0u32, 0];
    let inputs = BasisInputs {
        view_scores: Some(&scores),
        track_view_counts: Some(&counts),
    };

    let batch = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        None,
        Some(&inputs),
        &capped(3),
        None,
    );

    // Patch 0 ranks the leading views highest, patch 1 the trailing ones — so
    // the per-patch scores really did reach the pick.
    assert_eq!(basis_views(&batch[0]), vec![0, 1, 2]);
    assert_eq!(basis_views(&batch[1]), vec![7, 8, 9]);
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
/// grayscale/colour capture the phase-B tail search has to survive.
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
fn tail_view_narrower_than_the_basis_template_is_scored_not_panicked() {
    // Every third view is grayscale among 3-channel ones. The basis template is
    // built over the colour views' 3 channels; a 1-channel tail tile has no
    // plane for channels 1 and 2, which used to index `tile.planes` out of
    // bounds inside the shift search.
    let channels: Vec<u32> = (0..9).map(|k| if k % 3 == 0 { 1 } else { 3 }).collect();
    let (scene, set) = ring_scene_mixed_channels(&channels);
    let views = scene.views();
    let patch = plane_patch();
    // Rank the colour views highest so the grayscale ones land in the tail.
    let scores: Vec<f64> = channels
        .iter()
        .map(|&c| if c == 1 { 0.1 } else { 0.9 })
        .collect();
    let evidence = BasisEvidence {
        view_scores: Some(&scores),
        track_view_count: 0,
    };

    let res = localize_patch_keypoints_with_basis(&patch, &views, &set, None, evidence, &capped(3));

    // The basis is colour-only; the grayscale views are tail members.
    for v in basis_views(&res) {
        assert_eq!(
            channels[v as usize], 3,
            "view {v} should not be in the basis"
        );
    }
    // Sensible output: parallel arrays, and the grayscale tail views that
    // survive carry a finite ZNCC from the truncated (1-channel) score.
    assert_eq!(res.keypoints.len(), res.views.len());
    assert_eq!(res.loo_zncc.len(), res.views.len());
    assert_eq!(res.is_basis.len(), res.views.len());
    let mut scored_gray = 0;
    for (k, &v) in res.views.iter().enumerate() {
        if channels[v as usize] == 1 {
            assert!(!res.is_basis[k]);
            assert!(
                res.loo_zncc[k].is_finite() && res.loo_zncc[k] > 0.5,
                "grayscale tail view {v} scored {}",
                res.loo_zncc[k]
            );
            scored_gray += 1;
        }
    }
    assert!(
        scored_gray > 0,
        "no grayscale tail view survived to be checked"
    );
}

#[test]
fn mixed_channel_scene_is_unaffected_when_the_cap_is_off() {
    // The same scene through the uncapped path: no tail, so no truncation —
    // this pins that the blocker fix did not change the K = 0 behaviour.
    let channels: Vec<u32> = (0..9).map(|k| if k % 3 == 0 { 1 } else { 3 }).collect();
    let (scene, set) = ring_scene_mixed_channels(&channels);
    let views = scene.views();
    let patch = plane_patch();

    let res = localize_patch_keypoints(&patch, &views, &set, None, &params());
    assert!(res.is_basis.iter().all(|&b| b));
    assert_eq!(res.keypoints.len(), res.views.len());
}

#[test]
fn empty_view_scores_fall_back_to_the_grazing_rank() {
    // The batch entry point says "this point is unscored" with an empty score
    // list (it has no per-point Option). That must behave exactly like passing
    // no scores at all — not like an all-NaN score vector, which would rank the
    // candidates in input order instead.
    let (scene, set) = ring_scene(8, &[], &[]);
    let views = scene.views();
    let patch = plane_patch();
    let empty: Vec<f64> = Vec::new();

    let none = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        None,
        BasisEvidence::default(),
        &capped(3),
    );
    let empty_slice = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        None,
        BasisEvidence {
            view_scores: Some(&empty),
            track_view_count: 0,
        },
        &capped(3),
    );

    assert_eq!(none.views, empty_slice.views);
    assert_eq!(none.keypoints, empty_slice.keypoints);
    assert_eq!(none.is_basis, empty_slice.is_basis);
    // And the batch form threads an empty per-patch entry the same way.
    let cloud = PatchCloud {
        patches: vec![plane_patch()],
        point_indexes: vec![0],
    };
    let scores = vec![Vec::<f64>::new()];
    let inputs = BasisInputs {
        view_scores: Some(&scores),
        track_view_counts: None,
    };
    let batch = localize_patch_cloud_keypoints(
        &cloud,
        &views,
        std::slice::from_ref(&set),
        None,
        Some(&inputs),
        &capped(3),
        None,
    );
    assert_eq!(batch[0].views, none.views);
    assert_eq!(batch[0].is_basis, none.is_basis);
}

/// A textureless surface: every channel is flat, so the z-normalization finds
/// no channel to score on and no consensus template can be built.
fn flat_texture(_x: f64, _y: f64) -> f64 {
    128.0
}

#[test]
fn tail_without_a_basis_template_still_faces_the_shift_gate() {
    // A flat scene gives the z-normalization no textured channel, so the round
    // loop bails before building any consensus and `basis_template` returns
    // `None` — the "no usable basis" path. The tail then keeps its seed
    // offsets, and a seed further than `max_shift_px` from the projection must
    // still be dropped: nothing downstream re-checks it, and the early return
    // used to skip the gate entirely.
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

    // Loose gate: the un-registered tail views survive, which is what proves
    // this scene really exercises the no-basis path.
    let loose = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        Some(&seeds),
        BasisEvidence::default(),
        &KeypointLocalizeParams {
            max_shift_px: 1e6,
            ..capped(3)
        },
    );
    assert!(
        loose.is_basis.iter().any(|&b| !b),
        "expected un-registered tail views to be reported under a loose gate"
    );
    assert!(
        loose.loo_zncc.iter().all(|z| z.is_nan()),
        "no template was built, so every ZNCC should be unknown"
    );

    // Tight gate: those same tail views sit far past `max_shift_px` and must go.
    let tight = localize_patch_keypoints_with_basis(
        &patch,
        &views,
        &set,
        Some(&seeds),
        BasisEvidence::default(),
        &KeypointLocalizeParams {
            max_shift_px: 0.5,
            ..capped(3)
        },
    );
    for (k, &v) in tight.views.iter().enumerate() {
        assert!(
            tight.is_basis[k],
            "tail view {v} was emitted {} px from its projection despite the \
             0.5 px gate",
            tight.offsets_px[k]
        );
    }
}

/// `project_unclipped` is the only world→pixel entry in this module and the
/// sub-pixel refiner, so its cheirality test decides which views either can
/// reach. Under a ray-path model that test must be the camera's own domain: a
/// `z >= 0` short-circuit ahead of `ray_to_pixel` makes the whole θ > 90°
/// annulus of a >180° capture invisible to congealing.
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
