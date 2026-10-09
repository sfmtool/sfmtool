// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::normal_refine::Sampler;

// A synthetic scene mirroring the keypoint_localize tests: pinhole cameras
// (rotated 180° about X so the canonical −Z-forward camera looks down world +z) viewing a textured world plane at
// z = PLANE_Z. The patch sits on that plane with a normal pointing back toward
// the cameras (-z). Each view can render the plane texture translated in-plane by
// a per-view world offset `o_k`: the patch then renders, in view k, content
// shifted by `-o_k`, so the views disagree until refinement shifts each by `o_k`.
// The offset the refiner must recover for view k is `o_k / wpp` patch-grid px
// (`wpp = 2·half_extent / R`), and the recovered keypoint moves by
// `(o_k/wpp)·src_per_grid` source px from the projection.
//
// To plant a *sub-pixel* offset we render the texture continuously (no integer
// snapping), so a world offset of, e.g., 0.37·wpp shifts the content by 0.37
// patch-grid px — the kind of fractional offset the continuous refiner exists to
// resolve and a discrete grid cannot reach.

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

/// A flat (textureless) surface — the aperture / low-texture case.
fn flat_texture(_x: f64, _y: f64) -> f64 {
    127.0
}

/// Synthesize the image a pinhole camera at `center` (looking down world +z) sees of
/// the textured plane z = PLANE_Z, with the texture pattern translated in-plane
/// by the (possibly fractional) world offset `off`. The texture is sampled
/// continuously, so a fractional `off` plants a genuine sub-pixel shift.
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
    /// ray direction (no parallax). Per view the directional texture is shifted by
    /// the matching angular `offset`.
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

/// Texture as a function of ray direction `(dx, dy)` (small-angle pinhole coords).
fn dir_texture(dx: f64, dy: f64) -> f64 {
    texture(dx * 30.0, dy * 30.0)
}

/// Synthesize what an plus-z-looking pinhole sees of a point at infinity in the
/// `+z` direction: each pixel's value is `dir_texture` of its ray direction,
/// shifted by the (fractional) angular offset `off`. Independent of camera position.
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

/// Tangent-sphere patch for a point at infinity in the `+z` direction.
fn infinity_patch() -> OrientedPatch {
    OrientedPatch::from_infinity_direction(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, -1.0, 0.0),
        [0.05, 0.05],
    )
}

fn params() -> KeypointSubpixelParams {
    KeypointSubpixelParams {
        resolution: RES,
        ..KeypointSubpixelParams::default()
    }
}

/// Index into `res.views` for image `i`.
fn pos(res: &KeypointRefinement, i: u32) -> usize {
    res.views
        .iter()
        .position(|&v| v == i)
        .expect("view present")
}

// ── Consensus sharpness helper (validation §3) ───────────────────────────────

/// Build the robust consensus image (channel-averaged, R×R) from the views at the
/// given per-view offsets, and return its gradient energy (a sharpness metric:
/// well-registered views average without blurring detail away, so it rises as the
/// views co-register). Offsets are patch-grid px, parallel to `view_set`.
fn consensus_sharpness(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    offsets: &[[f64; 2]],
    p: &KeypointSubpixelParams,
) -> f64 {
    let r = p.resolution as usize;
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    // Full-grid support (uniform), so the consensus image covers the whole R×R.
    let n = r * r;
    let channels = views[view_set[0] as usize].pyramid.level(0).channels() as usize;
    // Robust weighted mean over views, per pixel/channel (raw intensities).
    let mut sum = vec![0f64; n * channels];
    let mut count = 0;
    for (k, &i) in view_set.iter().enumerate() {
        let center = shifted_center(patch, offsets[k][0], offsets[k][1], wpp_u, wpp_v);
        let mut cp = OrientedPatch::from_center_normal(
            center,
            patch.normal(),
            patch.v_axis,
            patch.half_extent,
        );
        cp.w = patch.w;
        let map = WarpMap::from_patch(
            &cp,
            views[i as usize].camera,
            views[i as usize].cam_from_world,
            p.resolution,
        );
        let img = remap_bilinear(views[i as usize].pyramid.level(0), &map);
        let mut all_valid = true;
        for row in 0..p.resolution {
            for col in 0..p.resolution {
                if !map.is_valid(col, row) {
                    all_valid = false;
                }
            }
        }
        if !all_valid {
            continue;
        }
        count += 1;
        for row in 0..r {
            for col in 0..r {
                let pix = row * r + col;
                for c in 0..channels {
                    sum[pix * channels + c] +=
                        img.get_pixel(col as u32, row as u32, c as u32) as f64;
                }
            }
        }
    }
    assert!(count > 0, "no in-frame views for sharpness");
    let inv = 1.0 / count as f64;
    // Channel-averaged consensus image.
    let mut gray = vec![0f64; n];
    for pix in 0..n {
        let mut s = 0.0;
        for c in 0..channels {
            s += sum[pix * channels + c] * inv;
        }
        gray[pix] = s / channels as f64;
    }
    // Gradient energy over the interior.
    let mut energy = 0.0;
    for row in 1..r - 1 {
        for col in 1..r - 1 {
            let gx = gray[row * r + col + 1] - gray[row * r + col - 1];
            let gy = gray[(row + 1) * r + col] - gray[(row - 1) * r + col];
            energy += gx * gx + gy * gy;
        }
    }
    energy / ((r - 2) * (r - 2)) as f64
}

// ── Validation §1: synthetic recovery to < 0.02 px ───────────────────────────

#[test]
fn recovers_planted_subpixel_offset_fine_grid() {
    // Same planted-offset recovery on a *fine-grid* patch (patch grid finer
    // than the source, `grid_to_source_scale < 1.2`) — the production-typical
    // case where the refiner builds and reads the render-once context tile
    // (the coarse default fixture below goes through the exact direct path
    // instead). Recovery through the tile must stay inside the same < 0.02 px
    // spec target.
    let he = 0.15; // grid ≈ 0.98 source px per grid px at RES = 20
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [he, he],
    );
    let wpp_fine = 2.0 * he / RES as f64;
    let planted_grid = 0.37;
    let ox = planted_grid * wpp_fine;
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
    // The gate must actually pick the tile path for this fixture.
    let s = grid_to_source_scale(&patch, &views[3], RES).expect("corners project");
    assert!(
        s <= TILE_MAX_GRID_TO_SOURCE,
        "fixture must exercise the tile path (grid→source scale {s:.2})"
    );

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2, 3], "view set unchanged");
    let p3 = pos(&res, 3);
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res.keypoints[p3][0] - proj3.0;
    let dy = res.keypoints[p3][1] - proj3.1;
    let expected_px = planted_grid * wpp_fine * FOCAL / PLANE_Z;
    assert!(
        (dx - expected_px).abs() < 0.02,
        "tile-path recovery to < 0.02 px: got dx={dx:.4}, expected {expected_px:.4}"
    );
    assert!(dy.abs() < 0.02, "no y motion expected, got {dy:.4}");
    for i in [0u32, 1, 2] {
        let pi = pos(&res, i);
        assert!(res.offsets_px[pi] < 0.02, "aligned view {i} barely moves");
    }
}

#[test]
fn recovers_planted_subpixel_offset_finite() {
    // Three aligned views, view 0 the reference; view 3's texture is shifted by a fractional
    // 0.37 patch-grid px. Seeding every view at its projection, the refiner must pull
    // view 3 back into alignment, recovering the planted offset to < 0.02 px.
    let planted_grid = 0.37;
    let ox = planted_grid * wpp();
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2, 3], "view set unchanged");

    // The planted world-x shift maps to +image-x for this fronto camera; the
    // recovered keypoint must move by +planted_grid·src_per_grid in image-x, with y
    // unchanged. Tolerance: 0.02 source px (the spec's < 0.02 px target).
    let p3 = pos(&res, 3);
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res.keypoints[p3][0] - proj3.0;
    let dy = res.keypoints[p3][1] - proj3.1;
    let expected_px = planted_grid * src_per_grid();
    assert!(
        (dx - expected_px).abs() < 0.02,
        "recover planted offset to < 0.02 px: got dx={dx:.4}, expected {expected_px:.4}"
    );
    assert!(dy.abs() < 0.02, "no y motion expected, got {dy:.4}");

    // The aligned views barely move.
    for i in [0u32, 1, 2] {
        let pi = pos(&res, i);
        assert!(res.offsets_px[pi] < 0.02, "aligned view {i} barely moves");
    }
}

#[test]
fn recovers_planted_subpixel_offset_two_views() {
    // The minimal cross-view case: two views, one planted off by a fractional shift.
    // View 0 is the reference and is not moved; the *relative* offset (the
    // recovered keypoint separation) must close to the planted shift, recovered to
    // < 0.02 px.
    let planted_grid = 0.30;
    let ox = planted_grid * wpp();
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let offs = [[0.0, 0.0], [ox, 0.0]];
    let texs = vec![texture as fn(f64, f64) -> f64; 2];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0, 1], None, Some(0), &params());
    let proj0 = project(&views[0], &patch.center, patch.w).unwrap();
    let proj1 = project(&views[1], &patch.center, patch.w).unwrap();
    let p0 = pos(&res, 0);
    let p1 = pos(&res, 1);
    let dx0 = res.keypoints[p0][0] - proj0.0;
    let dx1 = res.keypoints[p1][0] - proj1.0;
    // View 1's content is shifted +ox in world-x; to align with view 0 it must move
    // +planted_grid·src_per_grid more than view 0 does. The relative recovery is the
    // robust signal (view 0, the reference, does not move at all).
    let relative = dx1 - dx0;
    let expected_px = planted_grid * src_per_grid();
    // This fixture's patch grid minifies the source (`src_per_grid()` = 2.6
    // source px per grid px — a scale effect, not obliquity; the views are only
    // ~5.7° off fronto and `sigma_major == sigma_minor` here), so the default
    // `BilinearMip` sampler refines on pyramid level 1, whose samples are 2
    // source px apart. A single bilinear tap on that level locates the ECC
    // optimum a little short of the truth — the phase-dependent interpolation
    // bias described in the spec's Validation section, which scales with the
    // level's sample spacing and is not a convergence or scaling artifact
    // (tightening `convergence_px` to 0 over 200 GN steps moves the answer by
    // < 1e-4 px). Two views is the weakest case: at N = 4 the same planted
    // offset comes back within 0.005 px. The default must still close the
    // offset; the spec's < 0.02 px target is pinned on the level-0 reference
    // sampler below.
    assert!(
        (relative - expected_px).abs() < 0.03,
        "two-view relative offset to < 0.03 px: got {relative:.4}, expected {expected_px:.4}"
    );

    let bl = KeypointSubpixelParams {
        sampler: Sampler::Bilinear.into(),
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1], None, Some(0), &bl);
    let p0 = pos(&res, 0);
    let p1 = pos(&res, 1);
    let relative = (res.keypoints[p1][0] - proj1.0) - (res.keypoints[p0][0] - proj0.0);
    assert!(
        (relative - expected_px).abs() < 0.02,
        "bilinear two-view relative offset to < 0.02 px: got {relative:.4}, expected {expected_px:.4}"
    );
}

// ── Validation §4: infinity (w = 0) recovery + guard ─────────────────────────

#[test]
fn recovers_planted_subpixel_offset_infinity() {
    // Same planted-offset recovery for a w = 0 point at infinity: the refiner must
    // run the w = 0 render/project/Jacobian path and recover the fractional angular
    // shift to < 0.02 px. View 3's directional texture is shifted angularly.
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let patch = infinity_patch();
    let wpp_u = 2.0 * patch.half_extent[0] / RES as f64;
    // angular world-per-grid: half-extent is angular (0.05 rad) so wpp is rad/grid.
    let planted_grid = 0.40;
    let ang = planted_grid * wpp_u;
    let scene = Scene::infinity(
        &[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [3.0, 0.0, 2.0],
        ],
        &[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ang, 0.0]],
    );
    let views = scene.views();
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2, 3]);

    // The angular shift dx maps to image-x via the focal length; +ang rad -> +ang·F px.
    let p3 = pos(&res, 3);
    let dx = res.keypoints[p3][0] - cx;
    let dy = res.keypoints[p3][1] - cy;
    let expected_px = planted_grid * wpp_u * FOCAL;
    assert!(
        (dx - expected_px).abs() < 0.02,
        "infinity recovery to < 0.02 px: got dx={dx:.4}, expected {expected_px:.4}"
    );
    assert!(dy.abs() < 0.02, "no y motion expected, got {dy:.4}");
}

#[test]
fn infinity_never_worse_than_seed() {
    // The never-worse guard must hold for w = 0 (infinity) patches as for finite
    // ones (spec §Validation "Points at infinity"): every refined view's ECC score
    // must be ≥ its seed score against the same reference render, AND the keypoint must
    // not be pushed off when the views are already aligned. We exercise both with
    // a directionally-aligned set seeded at the projections.
    let scene = Scene::infinity(
        &[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, -5.0, 3.0]],
        &[[0.0; 2]; 3],
    );
    let views = scene.views();
    let patch = infinity_patch();

    let seed_only = KeypointSubpixelParams {
        max_gn_steps: 0,
        ..params()
    };
    let seed = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &seed_only);
    let refined = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &params());

    assert_eq!(refined.views, vec![0, 1, 2]);
    for &o in &refined.offsets_px {
        assert!(o < 0.05, "aligned infinity view barely moves, got {o}");
    }
    // Score floor against the same reference render.
    for i in 0..3u32 {
        let ps = pos(&seed, i);
        let pr = pos(&refined, i);
        assert!(
            refined.scores[pr] >= seed.scores[ps] - 1e-9,
            "infinity view {i} refined score {} must be >= seed score {}",
            refined.scores[pr],
            seed.scores[ps]
        );
    }
}

// ── Validation §2: quality — refined ≥ seed by ECC score ─────────────────────

#[test]
fn refined_score_never_below_seed() {
    // The guard's core promise: every view's final ECC score is ≥ its seed score.
    // We measure the seed score by running with zero GN steps, then compare to a
    // full refine on the same (misregistered) scene.
    let ox = 0.45 * wpp();
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

    let seed_only = KeypointSubpixelParams {
        max_gn_steps: 0,
        ..params()
    };
    let seed = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &seed_only);
    let refined = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());

    for i in 0..4u32 {
        let ps = pos(&seed, i);
        let pr = pos(&refined, i);
        assert!(
            refined.scores[pr] >= seed.scores[ps] - 1e-9,
            "view {i} refined score {} must be >= seed score {}",
            refined.scores[pr],
            seed.scores[ps]
        );
    }
    // The misregistered view should have *improved* (strictly), proving the refiner
    // actually does something, not just preserves the seed.
    let pr = pos(&refined, 3);
    let ps = pos(&seed, 3);
    assert!(
        refined.scores[pr] > seed.scores[ps] + 1e-4,
        "the misregistered view should improve: {} vs {}",
        refined.scores[pr],
        seed.scores[ps]
    );
}

// ── Validation §3: consensus sharpness rises after refinement ────────────────

#[test]
fn consensus_sharpens_after_refinement() {
    // Several views misregistered by distinct fractional offsets: their seed-aligned
    // consensus is blurred. After refinement the views co-register, so the consensus
    // image's gradient energy (sharpness) must rise (non-decrease) — the prototype's
    // observed effect.
    let centers = [
        [0.4, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
        [0.3, 0.3, 0.0],
    ];
    let offs = [
        [0.0, 0.0],
        [0.35 * wpp(), 0.0],
        [0.0, -0.4 * wpp()],
        [-0.3 * wpp(), 0.25 * wpp()],
        [0.2 * wpp(), 0.3 * wpp()],
    ];
    let texs = vec![texture as fn(f64, f64) -> f64; 5];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();
    let view_set = vec![0u32, 1, 2, 3, 4];

    let p = params();
    let seed_offsets = vec![[0.0, 0.0]; 5];
    let before = consensus_sharpness(&patch, &views, &view_set, &seed_offsets, &p);

    // Recover the per-view offsets by refining, then convert keypoint offsets back to
    // patch-grid px for the after-consensus (the refiner reports source-px offsets;
    // here we re-derive grid offsets from the refined δ via a second refine that
    // exposes them — instead we re-run and read the keypoints, converting through the
    // known src_per_grid mapping along x/y).
    let res = refine_patch_keypoints(&patch, &views, &view_set, None, Some(0), &p);
    let mut after_offsets = vec![[0.0, 0.0]; 5];
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    for (k, &i) in view_set.iter().enumerate() {
        // Convert each refined source-px keypoint back to a patch-grid offset with
        // the same inverse mapping production uses (`seed_offset`), so this stays
        // correct for any patch frame rather than baking in a fixed u/v→x/y layout.
        after_offsets[k] = seed_offset(&patch, &views[i as usize], res.keypoints[k], wpp_u, wpp_v)
            .unwrap_or([0.0, 0.0]);
    }
    let after = consensus_sharpness(&patch, &views, &view_set, &after_offsets, &p);

    assert!(
        after >= before * 0.999,
        "consensus sharpness must not decrease after refinement: before={before:.3}, after={after:.3}"
    );
    // On this strongly-misregistered case it should visibly rise.
    assert!(
        after > before,
        "consensus should sharpen: before={before:.3}, after={after:.3}"
    );
}

// ── Validation §5: guard correctness ─────────────────────────────────────────

#[test]
fn flat_texture_keeps_seed() {
    // Low-texture (flat) views: the Jacobian is singular (aperture problem), so the
    // GN solve must abandon and keep the seed — no NaN, no spurious motion.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![flat_texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2], "view set preserved");
    for (k, &o) in res.offsets_px.iter().enumerate() {
        assert!(
            o.is_finite() && o < 1e-6,
            "flat view {k} must keep the seed (no motion), got {o}"
        );
    }
}

#[test]
fn aligned_views_do_not_move() {
    // Perfectly aligned, textured views seeded at the projection: there is no
    // improving step, so the guard keeps every seed (offset ≈ 0) and the view set is
    // unchanged. Never worse than the seed.
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &params());
    assert_eq!(res.views, vec![0, 1, 2, 3]);
    for &o in &res.offsets_px {
        assert!(o < 0.02, "aligned view should not move, got {o}");
    }
    for &s in &res.scores {
        assert!(s > 0.95, "aligned views should agree strongly, score {s}");
    }
}

#[test]
fn out_of_frame_seed_keeps_seed() {
    // A view whose patch core leaves the frame at the seed cannot be scored; it must
    // keep its seed (NaN score, projection keypoint) and not crash. We place a camera
    // so far off-axis that the patch projects outside the image.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]; 2];
    let texs = vec![texture as fn(f64, f64) -> f64; 2];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    // Seed view 1 at a keypoint far outside the frame: seed_offset maps it to a huge
    // in-plane offset whose core is out of frame. It must keep that seed (no panic).
    let proj1 = project(&views[1], &patch.center, patch.w).unwrap();
    let seeds = [Some([proj1.0, proj1.1]), Some([proj1.0 + 5000.0, proj1.1])];
    let res = refine_patch_keypoints(&patch, &views, &[0, 1], Some(&seeds), Some(0), &params());
    assert_eq!(
        res.views,
        vec![0, 1],
        "view set preserved even with an OOF seed"
    );
    // View 1's score is NaN (never scored) and its keypoint falls back to the
    // projection (the shifted center failed to project, or its core was OOF).
    let p1 = pos(&res, 1);
    assert!(res.scores[p1].is_nan(), "OOF-seed view keeps a NaN score");
}

#[test]
fn never_overshoots_beyond_max_offset() {
    // A seed already near alignment must not be driven past `max_offset_px` from the
    // seed by an aggressive step — the guard clamps the line search to the bound.
    let ox = 0.3 * wpp();
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

    let tight = KeypointSubpixelParams {
        max_offset_px: 0.1, // far below the planted 0.3 grid px
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(0), &tight);
    // Every recovered grid offset must respect the bound (converted to grid px).
    for (k, &i) in res.views.iter().enumerate() {
        let proj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        let dx = res.keypoints[k][0] - proj.0;
        let dy = res.keypoints[k][1] - proj.1;
        let grid = (dx.hypot(dy)) / src_per_grid();
        assert!(
            grid <= 0.1 + 1e-6,
            "view {i} offset {grid:.4} grid px must stay within max_offset_px=0.1"
        );
    }
}

// ── Membership / shape invariants ────────────────────────────────────────────

#[test]
fn duplicate_view_index_is_deduped() {
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0, 0, 1, 2], None, Some(0), &params());
    assert_eq!(
        res.views,
        vec![0, 1, 2],
        "duplicate deduped, order preserved"
    );
}

#[test]
fn fewer_than_two_views_returns_seed_projection() {
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0], None, Some(0), &params());
    assert_eq!(res.views, vec![0]);
    let proj = project(&views[0], &patch.center, patch.w).unwrap();
    assert!((res.keypoints[0][0] - proj.0).abs() < 1e-9);
    assert!((res.keypoints[0][1] - proj.1).abs() < 1e-9);
    assert!(res.scores[0].is_nan(), "no consensus for a lone view");
}

#[test]
fn empty_view_set_returns_empty() {
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[], None, Some(0), &params());
    assert!(res.views.is_empty());
    assert!(res.keypoints.is_empty());
    assert!(res.offsets_px.is_empty());
    assert!(res.scores.is_empty());
}

#[test]
fn batch_matches_per_patch() {
    let ox = 0.4 * wpp();
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
    let cloud = PatchCloud {
        patches: vec![plane_patch(), plane_patch()],
        point_indexes: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2, 3], vec![0u32, 1, 2, 3]];

    let batch = refine_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        None,
        None,
        &params(),
        &crate::progress::Progress::none(),
    )
    .expect("Progress::none never cancels");
    assert_eq!(batch.len(), 2);
    for (i, res) in batch.iter().enumerate() {
        let single = refine_patch_keypoints(
            &cloud.patches[i],
            &views,
            &view_sets[i],
            None,
            None,
            &params(),
        );
        assert_eq!(res.views, single.views);
        for (a, b) in res.keypoints.iter().zip(&single.keypoints) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
    }

    // Per-patch references are threaded through: patch 1 names view 3.
    let references = [Some(0), Some(3)];
    let batch = refine_patch_cloud_keypoints(
        &cloud,
        &views,
        &view_sets,
        None,
        Some(&references),
        &params(),
        &crate::progress::Progress::none(),
    )
    .expect("Progress::none never cancels");
    for (i, res) in batch.iter().enumerate() {
        let single = refine_patch_keypoints(
            &cloud.patches[i],
            &views,
            &view_sets[i],
            None,
            references[i],
            &params(),
        );
        assert_eq!(res.keypoints, single.keypoints);
        assert_eq!(res.scores[references[i].unwrap()], 1.0);
    }
    // View 3 sees the texture moved: as the reference it stays, and the
    // others move toward it instead.
    let moved = pos(&batch[1], 0);
    assert!(
        batch[1].offsets_px[moved] > 0.5,
        "{:?}",
        batch[1].offsets_px
    );
}

// ── render_bitmaps: the stored bitmap at the final keypoints ─────────────────
//
// With `render_bitmaps = true` the refiner also renders each point's stored
// RGBA bitmap: the reference observation's tile at its keypoint, which the
// refinement did not move, or the fused mean of the views at their final
// keypoints where there is no reference. A well-observed point — finite OR at
// infinity — gets a real (nonzero) texture, and a point with fewer than two
// usable views gets `None`, which `sfm embed-patches` uses to drop it.

/// True when some pixel of the flat R·R·4 texture has a nonzero alpha.
fn has_nonzero_alpha(rep: &[u8]) -> bool {
    rep.as_chunks::<4>().0.iter().any(|px| px[3] > 0)
}

#[test]
fn render_bitmaps_fuses_representative_for_finite_point() {
    // Four aligned, textured views and no reference given: the rule picks one,
    // and the bitmap must exist, have the R·R·4 shape, and carry nonzero RGB +
    // alpha.
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

    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, None, &p);
    let rep = res
        .representative
        .expect("a well-observed finite point fuses a representative");
    assert!(res.reference.is_some(), "the rule picks a reference here");
    assert_eq!(rep.len(), (RES * RES * 4) as usize);
    assert!(
        has_nonzero_alpha(&rep),
        "aligned views must agree somewhere (alpha > 0)"
    );
    assert!(
        rep.as_chunks::<4>().0.iter().any(|px| px[0] > 0),
        "the fused texture must carry real image content"
    );
}

#[test]
fn render_bitmaps_fuses_representative_for_infinity_point() {
    // The same contract for a w = 0 point at infinity: it is NOT skipped (unlike
    // normal refinement) — the direction-patch render path fuses a real
    // representative, fixing the all-black infinity bitmaps.
    let scene = Scene::infinity(
        &[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, -5.0, 3.0]],
        &[[0.0; 2]; 3],
    );
    let views = scene.views();
    let patch = infinity_patch();

    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &p);
    let rep = res
        .representative
        .expect("a well-observed infinity point fuses a representative");
    assert_eq!(rep.len(), (RES * RES * 4) as usize);
    assert!(
        has_nonzero_alpha(&rep),
        "aligned infinity views must agree somewhere (alpha > 0)"
    );
}

#[test]
fn render_bitmaps_lone_view_has_no_representative() {
    // One view = no cross-view consensus: the representative must be `None` (the
    // culled-point signal), not a single-view or zero texture.
    let centers = [[0.4, 0.0, 0.0]];
    let offs = [[0.0; 2]];
    let texs = vec![texture as fn(f64, f64) -> f64];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0], None, Some(0), &p);
    assert!(res.representative.is_none(), "no consensus, no bitmap");
}

#[test]
fn render_bitmaps_off_by_default() {
    // Without the opt-in the refiner must not pay for (or return) the texture.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, Some(0), &params());
    assert!(res.representative.is_none());
    // The reference is reported all the same: it costs no render.
    assert_eq!(res.reference, Some(0));
}

#[test]
fn render_bitmaps_stores_the_reference_tile() {
    // With a reference given, the stored bitmap is its tile rendered at its
    // keypoint, which the refinement leaves where it was given, and the result
    // names its position.
    let ox = 0.4 * wpp();
    let scene = Scene::new(
        &[
            [0.4, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, -0.4, 0.0],
        ],
        &[[0.0, 0.0], [0.0, 0.0], [ox, 0.0], [0.0, 0.0]],
        &[texture as fn(f64, f64) -> f64; 4],
    );
    let views = scene.views();
    let patch = plane_patch();
    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    let proj2 = project(&views[2], &patch.center, patch.w).unwrap();
    let start = [proj2.0 + 0.6, proj2.1 - 0.3];
    let seeds = [None, None, Some(start), None];
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), Some(2), &p);
    assert_eq!(res.reference, Some(2));
    assert_eq!(res.keypoints[2], start);
    let tile = render_view_tile(
        &patch,
        &views[2],
        Some(start),
        RES as usize,
        p.sampler,
        &Progress::none(),
    );
    assert_eq!(
        res.representative.expect("the reference's tile renders"),
        bitmap_from_tile(&tile)
    );
}

#[test]
fn fuse_patch_bitmap_moves_nothing_and_needs_two_views() {
    let scene = Scene::new(
        &[[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
        &[[0.0; 2]; 3],
        &[texture as fn(f64, f64) -> f64; 3],
    );
    let views = scene.views();
    let patch = plane_patch();
    let keypoints: Vec<[f64; 2]> = (0..3)
        .map(|i| {
            let (x, y) = project(&views[i], &patch.center, patch.w).unwrap();
            [x, y]
        })
        .collect();
    let fused = fuse_patch_bitmap(&patch, &views, &[0, 1, 2], &keypoints, &params())
        .expect("three aligned views fuse");
    assert_eq!(fused.len(), (RES * RES * 4) as usize);
    assert!(has_nonzero_alpha(&fused));
    // The fuse does not pick a reference: it is the mean, not one view's tile.
    let tile = render_view_tile(
        &patch,
        &views[0],
        Some(keypoints[0]),
        RES as usize,
        params().sampler,
        &Progress::none(),
    );
    assert_ne!(fused, bitmap_from_tile(&tile));
    assert!(fuse_patch_bitmap(&patch, &views, &[0], &keypoints[..1], &params()).is_none());
}

// ── Aligning every view to the reference render ─────────────────────────────

/// The four fronto cameras of the planted-offset tests, view `k` rendering the
/// texture translated by `offs_grid[k]` patch-grid px.
fn four_view_scene(offs_grid: [[f64; 2]; 4]) -> Scene {
    Scene::new(
        &[
            [0.4, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, -0.4, 0.0],
        ],
        &offs_grid.map(|[u, v]| [u * wpp(), v * wpp()]),
        &[texture as fn(f64, f64) -> f64; 4],
    )
}

/// Where view `view` sees the content the reference (offset 0) sees at the
/// patch centre, when its texture is translated by `off_grid` patch-grid px.
fn true_keypoint(views: &[ProjectedImage<'_>], view: usize, off_grid: [f64; 2]) -> [f64; 2] {
    let patch = plane_patch();
    let moved = patch.center + Vector3::new(off_grid[0] * wpp(), off_grid[1] * wpp(), 0.0);
    let (x, y) = project(&views[view], &moved, patch.w).unwrap();
    [x, y]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[test]
fn the_reference_is_not_moved_and_the_others_follow_it() {
    // Views planted at known sub-grid offsets, and the reference (view 1, not
    // the first) started `d` away from where it truly sees the content. The
    // reference keeps exactly the keypoint it was given, with score 1, and
    // every other view is refined onto its render: to its own truth moved by
    // the same `d`, since every camera sees the plane at the same depth.
    let offs = [[0.0, 0.0], [0.3, 0.0], [0.0, -0.35], [0.25, 0.2]];
    let scene = four_view_scene(offs);
    let views = scene.views();
    let patch = plane_patch();
    let d = [0.6, -0.4];
    let truth: Vec<[f64; 2]> = (0..4).map(|k| true_keypoint(&views, k, offs[k])).collect();
    let start = [truth[1][0] + d[0], truth[1][1] + d[1]];
    let seeds = [None, Some(start), None, None];
    let bl = KeypointSubpixelParams {
        sampler: Sampler::Bilinear.into(),
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), Some(1), &bl);
    assert_eq!(res.views, vec![0, 1, 2, 3]);
    assert_eq!(
        res.keypoints[1].map(f64::to_bits),
        start.map(f64::to_bits),
        "the reference's keypoint is returned bit for bit"
    );
    assert_eq!(res.scores[1], 1.0);
    for k in [0, 2, 3] {
        let want = [truth[k][0] + d[0], truth[k][1] + d[1]];
        let err = dist(res.keypoints[k], want);
        assert!(
            err < 0.03,
            "view {k} should follow the reference, off by {err:.4} px ({:?} vs {want:?})",
            res.keypoints[k]
        );
        assert!(res.scores[k] > 0.95, "view {k} {}", res.scores[k]);
    }

    // The same starts with view 0 as the reference: the views are aligned to
    // its render instead, and view 1 is pulled off its start onto its truth.
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), Some(0), &bl);
    for (k, (&kp, &want)) in res.keypoints.iter().zip(&truth).enumerate().skip(1) {
        let err = dist(kp, want);
        assert!(
            err < 0.03,
            "view {k} should agree with view 0's render, off by {err:.4} px"
        );
    }
}

#[test]
fn with_no_reference_given_the_rule_picks_one_and_it_is_not_moved() {
    let offs = [[0.0, 0.0], [0.3, 0.0], [0.0, 0.25], [-0.2, 0.2]];
    let scene = four_view_scene(offs);
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let seeds: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| {
            let (x, y) = project(&views[k], &patch.center, patch.w).unwrap();
            Some([x, y])
        })
        .collect();
    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
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

    let res = refine_patch_keypoints(&patch, &views, &set, Some(&seeds), None, &p);
    assert_eq!(res.reference, Some(pick));
    assert_eq!(res.keypoints[pick], seeds[pick].unwrap());
    assert_eq!(res.scores[pick], 1.0);
    let given = refine_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(pick), &p);
    assert_eq!(res.keypoints, given.keypoints);
    assert_eq!(res.representative, given.representative);
}

#[test]
fn refining_one_view_against_the_reference_recovers_a_planted_offset() {
    // View 3 is planted 0.37 grid px off; refined alone against view 0's
    // render, given as the observation or as its stored bitmap, it moves onto
    // the planted offset and reports its ECC score.
    let offs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.37, 0.0]];
    let scene = four_view_scene(offs);
    let views = scene.views();
    let patch = plane_patch();
    let bl = KeypointSubpixelParams {
        sampler: Sampler::Bilinear.into(),
        ..params()
    };
    let p0 = project(&views[0], &patch.center, patch.w).unwrap();
    let p3 = project(&views[3], &patch.center, patch.w).unwrap();
    let truth = true_keypoint(&views, 3, offs[3]);

    let tile = render_view_tile(
        &patch,
        &views[0],
        Some([p0.0, p0.1]),
        RES as usize,
        bl.sampler,
        &Progress::none(),
    );
    let bitmap = bitmap_from_tile(&tile);
    let templates = [
        (
            "observation",
            ReferenceTemplate::Observation {
                image: 0,
                keypoint: [p0.0, p0.1],
            },
            0.02,
        ),
        // The stored bitmap is the same render quantized to `u8`.
        ("bitmap", ReferenceTemplate::Bitmap(&bitmap), 0.05),
    ];
    for (name, template, tolerance) in templates {
        let (kp, score) =
            refine_view_against_reference(&patch, &views, template, 3, [p3.0, p3.1], &bl)
                .unwrap_or_else(|| panic!("{name}: the target refines"));
        let err = dist(kp, truth);
        assert!(
            err < tolerance,
            "{name}: off the planted offset by {err:.4} px ({kp:?} vs {truth:?})"
        );
        assert!(score > 0.95 && score <= 1.0, "{name}: score {score}");
    }

    // A bitmap that is not on the refinement's grid is refused.
    assert!(refine_view_against_reference(
        &patch,
        &views,
        ReferenceTemplate::Bitmap(&bitmap[..bitmap.len() - 4]),
        3,
        [p3.0, p3.1],
        &bl,
    )
    .is_none());
}

// ── Render-once context tile (RefineTile) ────────────────────────────────────

/// Shared fixture for the tile tests: one slightly off-axis view of the
/// textured plane, the default support, and a fractional seed offset.
fn tile_fixture() -> (Scene, OrientedPatch, Support, [f64; 2]) {
    let scene = Scene::new(
        &[[0.3, 0.1, 0.0]],
        &[[0.0, 0.0]],
        &[texture as fn(f64, f64) -> f64],
    );
    let patch = plane_patch();
    let p = params();
    let support = build_support(p.window, p.resolution);
    (scene, patch, support, [0.37, -0.21])
}

#[test]
fn coarse_grid_gate_routes_around_the_tile() {
    // The coarse-grid gate must route by `grid_to_source_scale`: a fine-grid
    // view (scale <= TILE_MAX_GRID_TO_SOURCE) gets a tile, a coarse-grid view
    // (a patch spanning many source px per grid px) gets `None` and keeps the
    // exact direct path. An inverted comparison would flip both arms.
    // The tile fixture itself is coarse-grid (scale ~2.6 > 1.2 — the tile
    // mechanics tests bypass the gate on purpose), so it is the natural
    // coarse case; a 4x-shrunk patch (scale ~0.65) is the fine case.
    let (scene, coarse, _support, seed) = tile_fixture();
    let views = scene.views();
    let p = params();
    let mut img = ImageF32WithGrad::empty();

    let coarse_scale =
        grid_to_source_scale(&coarse, &views[0], p.resolution).expect("fixture corners project");
    assert!(
        coarse_scale > TILE_MAX_GRID_TO_SOURCE,
        "fixture patch must be coarse-grid (scale {coarse_scale})"
    );
    let cw_u = 2.0 * coarse.half_extent[0] / p.resolution as f64;
    let cw_v = 2.0 * coarse.half_extent[1] / p.resolution as f64;
    assert!(
        try_render_refine_tile(
            &coarse,
            &views[0],
            seed,
            cw_u,
            cw_v,
            p.resolution,
            4,
            Sampler::BilinearMip,
            &mut img,
        )
        .is_none(),
        "coarse-grid view must skip the tile and keep the direct path"
    );

    let fine = OrientedPatch::from_center_normal(
        coarse.center,
        coarse.normal(),
        coarse.v_axis,
        [coarse.half_extent[0] * 0.25, coarse.half_extent[1] * 0.25],
    );
    let fine_scale =
        grid_to_source_scale(&fine, &views[0], p.resolution).expect("shrunk patch corners project");
    assert!(
        fine_scale <= TILE_MAX_GRID_TO_SOURCE,
        "shrunk patch must be fine-grid (scale {fine_scale})"
    );
    let fw_u = 2.0 * fine.half_extent[0] / p.resolution as f64;
    let fw_v = 2.0 * fine.half_extent[1] / p.resolution as f64;
    assert!(
        try_render_refine_tile(
            &fine,
            &views[0],
            seed,
            fw_u,
            fw_v,
            p.resolution,
            4,
            Sampler::BilinearMip,
            &mut img,
        )
        .is_some(),
        "fine-grid view must get a tile"
    );
}

#[test]
fn tile_read_matches_direct_render_at_integer_shifts() {
    // A tile read at an integer shift from the tile's seed hits tile texels
    // exactly, so it must reproduce the direct render up to the direct path's
    // u8 output quantization (the tile keeps the sampler's unquantized f32).
    let (scene, patch, support, seed) = tile_fixture();
    let views = scene.views();
    let p = params();
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    let n = support.pixels.len();
    let mut img = ImageF32WithGrad::empty();
    let tile = render_refine_tile(
        &patch,
        &views[0],
        seed,
        wpp_u,
        wpp_v,
        p.resolution,
        4,
        Sampler::BilinearMip,
        &mut img,
    );
    let mut got = vec![0f32; n];
    let mut want = vec![0f32; n];
    for (di, dj) in [(0i64, 0i64), (1, 0), (-2, 1), (2, -2), (0, 2)] {
        let au = seed[0] + di as f64;
        let av = seed[1] + dj as f64;
        assert_eq!(
            tile.read_core(au, av, p.resolution as usize, &support, &mut got),
            Some(true),
            "in-coverage integer shift ({di},{dj}) must read"
        );
        assert!(render_core(
            &patch,
            &views[0],
            au,
            av,
            wpp_u,
            wpp_v,
            p.resolution,
            Sampler::BilinearMip,
            &support,
            1,
            &mut want,
        ));
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 0.5 + 1e-2,
                "shift ({di},{dj}) pixel {k}: tile {a} vs direct {b}"
            );
        }
    }
}

#[test]
fn tile_gradient_matches_direct_render_at_integer_shifts() {
    // The tile's pre-composed gradient planes, read at an integer shift, must
    // match the direct render's analytic ∂I/∂δ. Interior texels use the same
    // sampler gradient and (central-difference) warp Jacobian; the direct
    // path's R×R map falls back to one-sided differences on its boundary ring,
    // so the tolerance is loose relative to the ~40 intensity/grid-px gradients
    // of the test texture.
    let (scene, patch, support, seed) = tile_fixture();
    let views = scene.views();
    let p = params();
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    let n = support.pixels.len();
    let mut img = ImageF32WithGrad::empty();
    let tile = render_refine_tile(
        &patch,
        &views[0],
        seed,
        wpp_u,
        wpp_v,
        p.resolution,
        4,
        Sampler::BilinearMip,
        &mut img,
    );
    let (mut g_t, mut ju_t, mut jv_t) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut g_d, mut ju_d, mut jv_d) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let mut scratch = ImageF32WithGrad::empty();
    for (di, dj) in [(0i64, 0i64), (1, -1), (-2, 2)] {
        let au = seed[0] + di as f64;
        let av = seed[1] + dj as f64;
        assert_eq!(
            tile.read_core(au, av, p.resolution as usize, &support, &mut g_t),
            Some(true)
        );
        assert_eq!(
            tile.read_jg(
                au,
                av,
                p.resolution as usize,
                &support,
                &mut ju_t,
                &mut jv_t
            ),
            Some(true)
        );
        assert!(render_core_with_jg(
            &patch,
            &views[0],
            au,
            av,
            wpp_u,
            wpp_v,
            p.resolution,
            Sampler::BilinearMip,
            &support,
            1,
            &mut g_d,
            &mut ju_d,
            &mut jv_d,
            &mut scratch,
        ));
        for k in 0..n {
            assert!(
                (g_t[k] - g_d[k]).abs() <= 1e-2,
                "value {k}: {} vs {}",
                g_t[k],
                g_d[k]
            );
            assert!(
                (ju_t[k] - ju_d[k]).abs() <= 0.15,
                "jg_u {k}: {} vs {}",
                ju_t[k],
                ju_d[k]
            );
            assert!(
                (jv_t[k] - jv_d[k]).abs() <= 0.15,
                "jg_v {k}: {} vs {}",
                jv_t[k],
                jv_d[k]
            );
        }
    }
}

/// A low-frequency texture whose bilinear source interpolant is smooth at the
/// FD probe scale: the analytic (per-source-cell) sampler gradient and a small
/// central difference then measure the same slope. The default [`texture`] has
/// content at the source-cell scale, where the bilinear interpolant's gradient
/// is genuinely discontinuous across cell boundaries and a pointwise
/// FD-vs-analytic comparison is ill-posed (for the direct render path just as
/// much as for the tile).
fn smooth_texture(x: f64, y: f64) -> f64 {
    127.5 + 60.0 * (x * 1.5).sin() + 50.0 * (y * 1.2).cos()
}

#[test]
fn tile_gradient_matches_finite_differences() {
    // At a fractional offset, central differences of the tile's own value
    // reads must approximate the interpolated analytic gradient planes.
    let scene = Scene::new(
        &[[0.3, 0.1, 0.0]],
        &[[0.0, 0.0]],
        &[smooth_texture as fn(f64, f64) -> f64],
    );
    // Fine-grid patch (the tile's production regime, ≈ 1 source px per grid
    // px): the quantization slope noise (see the tolerance note below) scales
    // with the grid→source factor, so the coarse default fixture would triple
    // it.
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.15, 0.15],
    );
    let support = build_support(params().window, params().resolution);
    let seed = [0.37, -0.21];
    let views = scene.views();
    let p = params();
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    let n = support.pixels.len();
    let mut img = ImageF32WithGrad::empty();
    let tile = render_refine_tile(
        &patch,
        &views[0],
        seed,
        wpp_u,
        wpp_v,
        p.resolution,
        4,
        Sampler::BilinearMip,
        &mut img,
    );
    let (au, av) = (seed[0] + 0.43, seed[1] - 0.68);
    let h = 0.1;
    let read = |a: f64, b: f64, out: &mut [f32]| {
        assert_eq!(
            tile.read_core(a, b, p.resolution as usize, &support, out),
            Some(true)
        );
    };
    let (mut ju, mut jv) = (vec![0f32; n], vec![0f32; n]);
    assert_eq!(
        tile.read_jg(au, av, p.resolution as usize, &support, &mut ju, &mut jv),
        Some(true)
    );
    let (mut up, mut um, mut vp, mut vm) =
        (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    read(au + h, av, &mut up);
    read(au - h, av, &mut um);
    read(au, av + h, &mut vp);
    read(au, av - h, &mut vm);
    // The absolute tolerance term is the u8 quantization floor: the analytic
    // sampler gradient is built from adjacent-source-pixel differences of
    // *rounded* values, so per-cell slopes carry ±1-intensity-level jumps
    // (≈ ±0.5 per grid px at this fixture's ~1 source px per grid px) that a
    // small FD of the (slightly smoothing) spline value field doesn't
    // reproduce pointwise.
    for k in 0..n {
        let fd_u = (up[k] - um[k]) as f64 / (2.0 * h);
        let fd_v = (vp[k] - vm[k]) as f64 / (2.0 * h);
        assert!(
            (ju[k] as f64 - fd_u).abs() <= 0.1 * fd_u.abs() + 0.8,
            "jg_u {k}: analytic {} vs FD {fd_u}",
            ju[k]
        );
        assert!(
            (jv[k] as f64 - fd_v).abs() <= 0.1 * fd_v.abs() + 0.8,
            "jg_v {k}: analytic {} vs FD {fd_v}",
            jv[k]
        );
    }
}

/// On-demand micro-benchmark: `cargo test -p sfmtool-core --release
/// tile_read_bench -- --ignored --nocapture`. Times value and Jacobian tile
/// reads at a fractional offset on an RGB-like 3-channel tile.
#[test]
#[ignore]
fn tile_read_bench() {
    use std::time::Instant;
    // Synthetic 3-channel tile, production-sized (R = 24, pad = 4 -> 32²).
    let resolution = 24usize;
    let pad = 4usize;
    let t = resolution + 2 * pad;
    let ch = 3usize;
    let mk = |seed: u32| -> Vec<f32> {
        (0..t * t * ch)
            .map(|k| ((k as u32).wrapping_mul(2654435761).wrapping_add(seed) % 256) as f32)
            .collect()
    };
    let tile = RefineTile {
        res: t,
        pad,
        channels: ch,
        seed: [0.0, 0.0],
        value: mk(1),
        jg_u: mk(2),
        jg_v: mk(3),
        valid: vec![true; t * t],
        valid4: vec![true; t * t],
    };
    let support = build_support(PatchWindow::GaussianDisk { sigma: 0.6 }, resolution as u32);
    let n = support.pixels.len();
    let mut out = vec![0f32; ch * n];
    let mut ju = vec![0f32; ch * n];
    let mut jv = vec![0f32; ch * n];
    let iters = 100_000;
    let t0 = Instant::now();
    for i in 0..iters {
        let au = 0.3 + (i % 7) as f64 * 0.01;
        std::hint::black_box(tile.read_core(au, -0.4, resolution, &support, &mut out));
    }
    let val_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let t0 = Instant::now();
    for i in 0..iters {
        let au = 0.3 + (i % 7) as f64 * 0.01;
        std::hint::black_box(tile.read_jg(au, -0.4, resolution, &support, &mut ju, &mut jv));
    }
    let jg_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("read_core {val_us:.2} us/call ({n} px, {ch} ch)  read_jg {jg_us:.2} us/call");
}

#[test]
fn tile_read_out_of_coverage_falls_back_to_direct_render() {
    // An offset past the tile's coverage returns `None` from the read; the
    // `core_value` wrapper must then produce exactly the direct render.
    let (scene, patch, support, seed) = tile_fixture();
    let views = scene.views();
    let p = params();
    let wpp_u = 2.0 * patch.half_extent[0] / p.resolution as f64;
    let wpp_v = 2.0 * patch.half_extent[1] / p.resolution as f64;
    let n = support.pixels.len();
    let mut img = ImageF32WithGrad::empty();
    let tile = render_refine_tile(
        &patch,
        &views[0],
        seed,
        wpp_u,
        wpp_v,
        p.resolution,
        4,
        Sampler::BilinearMip,
        &mut img,
    );
    let (au, av) = (seed[0] + 10.0, seed[1]);
    let mut buf = vec![0f32; n];
    assert_eq!(
        tile.read_core(au, av, p.resolution as usize, &support, &mut buf),
        None,
        "offset outside the tile coverage must not read"
    );
    let mut got = vec![0f32; n];
    let mut want = vec![0f32; n];
    let ok_got = core_value(
        &patch,
        &views[0],
        Some(&tile),
        au,
        av,
        wpp_u,
        wpp_v,
        p.resolution,
        Sampler::BilinearMip,
        &support,
        1,
        &mut got,
    );
    let ok_want = render_core(
        &patch,
        &views[0],
        au,
        av,
        wpp_u,
        wpp_v,
        p.resolution,
        Sampler::BilinearMip,
        &support,
        1,
        &mut want,
    );
    assert_eq!(ok_got, ok_want, "fallback must mirror the direct render");
    if ok_got {
        assert_eq!(got, want, "fallback is the direct render bit-for-bit");
    }
}

// ── Which reference the refiner reports ─────────────────────────────────────

#[test]
fn the_reference_is_reported_without_rendering_bitmaps() {
    // Callers that refine without bitmaps (`embed-patches` rounds before the
    // last, `xform --refine-keypoints bitmaps=false`) still store which
    // observation the keypoints were aligned to.
    let offs = [[0.0, 0.0], [0.3, 0.0], [0.0, 0.25], [-0.2, 0.2]];
    let scene = four_view_scene(offs);
    let views = scene.views();
    let patch = plane_patch();
    let set = [0u32, 1, 2, 3];
    let seeds: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| {
            let (x, y) = project(&views[k], &patch.center, patch.w).unwrap();
            Some([x, y])
        })
        .collect();
    let bare = params();
    assert!(!bare.render_bitmaps);
    let with_bitmaps = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    for reference in [None, Some(2)] {
        let a = refine_patch_keypoints(&patch, &views, &set, Some(&seeds), reference, &bare);
        let b =
            refine_patch_keypoints(&patch, &views, &set, Some(&seeds), reference, &with_bitmaps);
        assert!(a.reference.is_some(), "{reference:?}");
        assert_eq!(a.reference, b.reference, "{reference:?}");
        assert_eq!(a.keypoints, b.keypoints, "{reference:?}");
        assert!(a.representative.is_none());
    }
    let given = refine_patch_keypoints(&patch, &views, &set, Some(&seeds), Some(2), &bare);
    assert_eq!(given.reference, Some(2));
}

#[test]
fn a_reference_out_of_frame_is_no_reference() {
    // The reference's keypoint is far outside its frame, so its core cannot
    // be rendered there. As in the localizer, that is no reference: nothing
    // is aligned, every view keeps its seed unscored, and the stored bitmap
    // is the fused mean of the other views.
    let scene = four_view_scene([[0.0; 2]; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let mut seeds: Vec<Option<[f64; 2]>> = (0..4)
        .map(|k| {
            let (x, y) = project(&views[k], &patch.center, patch.w).unwrap();
            Some([x + 0.4, y - 0.3])
        })
        .collect();
    let far = [seeds[0].unwrap()[0] + 5000.0, seeds[0].unwrap()[1]];
    seeds[0] = Some(far);
    let p = KeypointSubpixelParams {
        render_bitmaps: true,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], Some(&seeds), Some(0), &p);
    assert_eq!(res.views, vec![0, 1, 2, 3]);
    assert_eq!(res.reference, None);
    assert!(res.scores.iter().all(|s| s.is_nan()), "{:?}", res.scores);
    for (k, (&kp, seed)) in res.keypoints.iter().zip(&seeds).enumerate().skip(1) {
        let err = dist(kp, seed.unwrap());
        assert!(
            err < 1e-6,
            "view {k} moved {err} px with nothing to align to"
        );
    }
    assert!(res.representative.is_some(), "the fused mean of views 1-3");
}

#[test]
fn the_rule_never_picks_a_grazing_view_but_a_given_one_is_kept() {
    // View 3 is oblique to the plane. With a grazing cutoff above its cosine
    // the rule does not pick it, as the localizer, which drops it, cannot; it
    // is still refined. Given as the reference, it is used as given.
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
    let strict = KeypointSubpixelParams {
        min_grazing_cos: 0.95,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, None, &strict);
    assert_eq!(
        res.views,
        vec![0, 1, 2, 3],
        "the oblique view is still refined"
    );
    let reference = res
        .reference
        .expect("the rule picks among the facing views");
    assert_ne!(res.views[reference], 3);
    assert!(res.scores[3].is_finite());
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, Some(3), &strict);
    assert_eq!(res.reference, Some(3));
    assert_eq!(res.scores[3], 1.0);
}

#[test]
fn a_reference_given_at_a_repeated_image_is_kept() {
    // Image 1 is listed twice and the reference is given at its second slot,
    // which deduplication drops. The reference is matched by image, so image
    // 1 is still the reference rather than the rule's pick.
    let scene = four_view_scene([[0.0; 2]; 4]);
    let views = scene.views();
    let patch = plane_patch();
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 1, 2], None, Some(2), &params());
    assert_eq!(res.views, vec![0, 1, 2]);
    assert_eq!(res.reference.map(|r| res.views[r]), Some(1));
    assert_eq!(res.scores[1], 1.0);
}
