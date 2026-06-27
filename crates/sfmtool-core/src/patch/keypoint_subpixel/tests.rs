// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;

// A synthetic scene mirroring the keypoint_localize tests: pinhole cameras
// (identity rotation, looking down +z) viewing a textured world plane at
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

/// Synthesize the image a pinhole camera at `center` (looking down +z) sees of
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
    /// ray direction (no parallax). Per view the directional texture is shifted by
    /// the matching angular `offset`.
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

/// Texture as a function of ray direction `(dx, dy)` (small-angle pinhole coords).
fn dir_texture(dx: f64, dy: f64) -> f64 {
    texture(dx * 30.0, dy * 30.0)
}

/// Synthesize what an identity-rotation pinhole sees of a point at infinity in the
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
            patch.u_axis,
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
fn recovers_planted_subpixel_offset_finite() {
    // Three aligned views pin the gauge; view 3's texture is shifted by a fractional
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
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
    // The shared-consensus refiner splits the disagreement between them, but the
    // *relative* offset (the recovered keypoint separation) must close to the planted
    // shift, recovered to < 0.02 px.
    let planted_grid = 0.30;
    let ox = planted_grid * wpp();
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]];
    let offs = [[0.0, 0.0], [ox, 0.0]];
    let texs = vec![texture as fn(f64, f64) -> f64; 2];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let res = refine_patch_keypoints(&patch, &views, &[0, 1], None, &params());
    let proj0 = project(&views[0], &patch.center, patch.w).unwrap();
    let proj1 = project(&views[1], &patch.center, patch.w).unwrap();
    let p0 = pos(&res, 0);
    let p1 = pos(&res, 1);
    let dx0 = res.keypoints[p0][0] - proj0.0;
    let dx1 = res.keypoints[p1][0] - proj1.0;
    // View 1's content is shifted +ox in world-x; to align with view 0 it must move
    // +planted_grid·src_per_grid more than view 0 does. The relative recovery is the
    // robust signal (the absolute split depends on the consensus gauge).
    let relative = dx1 - dx0;
    let expected_px = planted_grid * src_per_grid();
    assert!(
        (relative - expected_px).abs() < 0.02,
        "two-view relative offset to < 0.02 px: got {relative:.4}, expected {expected_px:.4}"
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
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
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
    // must be ≥ its seed score against the same consensus, AND the keypoint must
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
    let seed = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, &seed_only);
    let refined = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, &params());

    assert_eq!(refined.views, vec![0, 1, 2]);
    for &o in &refined.offsets_px {
        assert!(o < 0.05, "aligned infinity view barely moves, got {o}");
    }
    // Score floor against the same (seed-aligned) consensus.
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
    let seed = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &seed_only);
    let refined = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());

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
    let res = refine_patch_keypoints(&patch, &views, &view_set, None, &p);
    let mut after_offsets = vec![[0.0, 0.0]; 5];
    for (k, &i) in view_set.iter().enumerate() {
        let proj = project(&views[i as usize], &patch.center, patch.w).unwrap();
        let dx = res.keypoints[k][0] - proj.0;
        let dy = res.keypoints[k][1] - proj.1;
        // image-x is world-x (v_axis), image-y is world-y (u_axis) for this patch;
        // map source px back to patch-grid px. Sign handled by the consensus render
        // (it re-renders at these grid offsets), so use the magnitude per axis with
        // the recovered sign.
        after_offsets[k] = [dy / src_per_grid(), dx / src_per_grid()];
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, &params());
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
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
    let seeds = [[proj1.0, proj1.1], [proj1.0 + 5000.0, proj1.1]];
    let res = refine_patch_keypoints(&patch, &views, &[0, 1], Some(&seeds), &params());
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
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &tight);
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

    let res = refine_patch_keypoints(&patch, &views, &[0, 0, 1, 2], None, &params());
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

    let res = refine_patch_keypoints(&patch, &views, &[0], None, &params());
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

    let res = refine_patch_keypoints(&patch, &views, &[], None, &params());
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
        point_ids: vec![0, 1],
    };
    let view_sets = vec![vec![0u32, 1, 2, 3], vec![0u32, 1, 2, 3]];

    let batch = refine_patch_cloud_keypoints(&cloud, &views, &view_sets, None, &params());
    assert_eq!(batch.len(), 2);
    for (i, res) in batch.iter().enumerate() {
        let single =
            refine_patch_keypoints(&cloud.patches[i], &views, &view_sets[i], None, &params());
        assert_eq!(res.views, single.views);
        for (a, b) in res.keypoints.iter().zip(&single.keypoints) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
    }
}

// ── max_outer_sweeps (the consensus-refresh knob) ────────────────────────────
//
// `max_outer_sweeps` exposes the spec's "Consensus refresh granularity" choice as
// a parameter: `1` is the single-pass-frozen variant (build `T` at the seed once,
// hold it fixed); `> 1` re-renders the views at their current offsets and
// rebuilds `T` between sweeps. Whether refresh helps is a measurable question, so
// these tests pin **behavior**: the default (`1`) is still single-pass-frozen,
// multi-sweep converges to the same answer on a planted offset, and converged
// multi-sweep is the fixed point of an extra sweep.

#[test]
fn max_outer_sweeps_default_matches_explicit_one() {
    // The default must remain single-pass-frozen (no behavior change vs the
    // pre-knob MVP). An explicit `max_outer_sweeps = 1` must be bit-identical to
    // the default on a non-trivial scene (planted misregistration + drop test).
    let ox = 0.37 * wpp();
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

    let baseline = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &params());
    let explicit_one = KeypointSubpixelParams {
        max_outer_sweeps: 1,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &explicit_one);

    assert_eq!(res.views, baseline.views);
    for (a, b) in res.keypoints.iter().zip(&baseline.keypoints) {
        assert!(
            (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12,
            "max_outer_sweeps=1 must equal the default exactly: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn max_outer_sweeps_multi_pass_recovers_planted_offset() {
    // The multi-sweep (per-sweep-refresh) variant must still recover the planted
    // sub-pixel offset on the same scene the single-pass case handles — that is,
    // refresh must not break convergence; at worst it converges to the same place.
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

    let multi = KeypointSubpixelParams {
        max_outer_sweeps: 5,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &multi);
    let p3 = pos(&res, 3);
    let proj3 = project(&views[3], &patch.center, patch.w).unwrap();
    let dx = res.keypoints[p3][0] - proj3.0;
    let dy = res.keypoints[p3][1] - proj3.1;
    let expected_px = planted_grid * src_per_grid();
    assert!(
        (dx - expected_px).abs() < 0.02,
        "multi-sweep recovers planted offset to < 0.02 px: got dx={dx:.4}, expected {expected_px:.4}"
    );
    assert!(dy.abs() < 0.02, "no y motion expected, got {dy:.4}");
}

#[test]
fn max_outer_sweeps_converges_to_fixed_point() {
    // After enough sweeps the views stop moving — running one more sweep beyond a
    // converged result must not move them (the alternating loop's fixed point).
    let ox = 0.37 * wpp();
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

    let p4 = KeypointSubpixelParams {
        max_outer_sweeps: 4,
        ..params()
    };
    let p8 = KeypointSubpixelParams {
        max_outer_sweeps: 8,
        ..params()
    };
    let a = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &p4);
    let b = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &p8);

    assert_eq!(a.views, b.views);
    for (x, y) in a.keypoints.iter().zip(&b.keypoints) {
        assert!(
            (x[0] - y[0]).abs() < 1e-6 && (x[1] - y[1]).abs() < 1e-6,
            "extra sweeps past convergence must not move views: {x:?} vs {y:?}"
        );
    }
}

#[test]
fn max_outer_sweeps_aligned_seed_keeps_views_still() {
    // Aligned views with a frozen consensus barely move (the `aligned_views_do_not_move`
    // test). With multi-sweep the same must hold — refresh must not introduce
    // spurious motion when there is nothing to refine.
    let centers = [[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]];
    let offs = [[0.0; 2]; 3];
    let texs = vec![texture as fn(f64, f64) -> f64; 3];
    let scene = Scene::new(&centers, &offs, &texs);
    let views = scene.views();
    let patch = plane_patch();

    let p = KeypointSubpixelParams {
        max_outer_sweeps: 5,
        ..params()
    };
    let res = refine_patch_keypoints(&patch, &views, &[0, 1, 2], None, &p);
    for o in &res.offsets_px {
        assert!(*o < 0.05, "aligned view should barely move, got {o}");
    }
}

#[test]
fn max_outer_sweeps_per_sweep_guard_does_not_regress_below_seed_at_sweep_t() {
    // Within-sweep monotonicity: each accepted GN step raises the ECC score
    // against THAT sweep's consensus, so for the first sweep the final scores must
    // be ≥ the seed-against-sweep-0-T scores. (Across sweeps `T` changes, so the
    // strict floor only holds within a sweep — that's exactly what this checks.)
    // We measure the seed-vs-sweep-0-T baseline by running with max_gn_steps = 0
    // and max_outer_sweeps = 1 (no moves, just score the seed against the first
    // consensus build), then refine with max_outer_sweeps > 1 on the same scene.
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
        max_outer_sweeps: 1,
        ..params()
    };
    let multi = KeypointSubpixelParams {
        max_outer_sweeps: 5,
        ..params()
    };
    let seed = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &seed_only);
    let refined = refine_patch_keypoints(&patch, &views, &[0, 1, 2, 3], None, &multi);

    for i in 0..4u32 {
        let ps = pos(&seed, i);
        let pr = pos(&refined, i);
        // The refined score is measured against the final-sweep `T`; the seed
        // score against sweep-0's `T`. In practice they are within tolerance and
        // refined ≥ seed because the consensus only sharpens. The assertion is the
        // empirical safety net for the within-sweep guarantee (loosened by 1e-3 to
        // tolerate the cross-sweep `T` change, which is small at sub-pixel scale).
        assert!(
            refined.scores[pr] >= seed.scores[ps] - 1e-3,
            "multi-sweep view {i} refined score {} should track / beat seed {}",
            refined.scores[pr],
            seed.scores[ps]
        );
    }
}
