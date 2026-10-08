// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};

use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::blur_matched::test_tiles::{grained, textured};
use crate::patch::pair_sharpness::MAX_MATCHED_LENGTH;
use crate::patch::reference_view::choose_reference_view;

fn window() -> PatchWindow {
    PatchWindow::GaussianDisk { sigma: 0.6 }
}

// ---- The scores --------------------------------------------------------------

/// A bitmap sharper than an observation along every direction is blurred to
/// the observation's sharpness, and only the bitmap: the blur-matched score
/// rises over the plain one, and the width is reported.
#[test]
fn a_sharp_bitmap_is_blurred_to_a_blurrier_observation() {
    let bitmap = textured(11, 0.8);
    let observation = bitmap.blurred(1.5, &mut BlurScratch::default());
    let untouched = observation.clone();
    let mut scorer = BitmapScorer::new(&bitmap, window());
    assert!(
        scorer.assessment().is_none(),
        "not read before it is needed"
    );
    let score = scorer.score(&observation, None);
    assert!(score.blur_sigma > 0.0, "{score:?}");
    assert!(!score.sharper_than_bitmap);
    assert!(
        score.blur_matched_zncc > score.zncc + 0.02,
        "blur-matched {} against plain {}",
        score.blur_matched_zncc,
        score.zncc
    );
    assert!(scorer.assessment().is_some());
    // The observation's tile is read as it is.
    assert_eq!(observation, untouched);
    // The width aims at the observation's semi-minor axis, at most 2 grid px.
    let target = semi_axes(
        &read_tile_ellipse(
            &observation.values,
            observation.channels,
            observation.side,
            &observation.data,
        )
        .unwrap(),
    )[1]
    .min(MAX_MATCHED_LENGTH);
    let reached = scorer.assessment().unwrap().sigma_to_reach(target).unwrap();
    assert_eq!(score.blur_sigma, reached);
}

/// An observation sharper than the bitmap is read plain, neither tile
/// blurred, and is flagged as a candidate to replace the reference.
#[test]
fn an_observation_sharper_than_the_bitmap_is_read_plain() {
    let sharp = textured(12, 0.8);
    let bitmap = sharp.blurred(1.5, &mut BlurScratch::default());
    let mut scorer = BitmapScorer::new(&bitmap, window());
    let score = scorer.score(&sharp, None);
    assert!(score.sharper_than_bitmap);
    assert_eq!(score.blur_sigma, 0.0);
    assert_eq!(
        score.blur_matched_zncc.to_bits(),
        score.zncc.to_bits(),
        "read plain"
    );
    assert!(scorer.assessment().is_none(), "no blur was needed");
}

/// Grain at another angle is not read as blur: neither tile is shorter than
/// the other along every direction, so the pair is read plain.
#[test]
fn grain_at_another_angle_is_read_plain() {
    let bitmap = grained(5, [0.0, 1.0]);
    let observation = grained(6, [1.0, 0.0]);
    let score = BitmapScorer::new(&bitmap, window()).score(&observation, None);
    assert_eq!(score.blur_sigma, 0.0);
    assert!(!score.sharper_than_bitmap);
    assert_eq!(score.blur_matched_zncc.to_bits(), score.zncc.to_bits());
}

/// A pair whose sharpness differs by less than the ratio of 1.25 is read
/// plain.
#[test]
fn a_difference_under_the_ratio_is_read_plain() {
    let bitmap = textured(13, 1.0);
    let observation = bitmap.blurred(0.15, &mut BlurScratch::default());
    let e = |t: &TilePlanes| {
        semi_axes(&read_tile_ellipse(&t.values, t.channels, t.side, &t.data).unwrap())
    };
    let ([major, _], [_, minor]) = (e(&bitmap), e(&observation));
    assert!(minor < DEFAULT_MIN_ELLIPSE_RATIO * major, "{major} {minor}");
    let score = BitmapScorer::new(&bitmap, window()).score(&observation, None);
    assert_eq!(score.blur_sigma, 0.0);
}

/// The reference observation is not scored, and every other observation is
/// scored as the scorer scores it alone; the assessment is read once however
/// many observations need it.
#[test]
fn the_reference_is_skipped_and_the_rest_scored_against_one_assessment() {
    let bitmap = textured(14, 0.8);
    let views: Vec<TilePlanes> = [0.0, 1.2, 1.4, 1.6]
        .iter()
        .map(|&s| bitmap.blurred(s, &mut BlurScratch::default()))
        .collect();
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let scores = score_against_bitmap(&bitmap, &refs, &[None; 4], Some(0), window());
    assert!(scores[0].is_none());
    for (v, score) in scores.iter().enumerate().skip(1) {
        let alone = BitmapScorer::new(&bitmap, window()).score(&views[v], None);
        assert_eq!(score.unwrap(), alone, "view {v}");
        assert!(alone.blur_sigma > 0.0, "view {v}");
    }
    // Blurrier observations get the bitmap blurred further.
    let widths: Vec<f64> = scores[1..].iter().map(|s| s.unwrap().blur_sigma).collect();
    assert!(widths.windows(2).all(|w| w[0] <= w[1]), "{widths:?}");
}

/// A sample without data in either tile is left out of both scores.
#[test]
fn samples_without_data_are_left_out() {
    let bitmap = textured(15, 0.8);
    let mut holed = bitmap.clone();
    for k in 0..holed.side * 3 {
        holed.data[k] = false;
        for c in 0..3 {
            holed.values[c * holed.side * holed.side + k] = 0.0;
        }
    }
    let score = BitmapScorer::new(&bitmap, window()).score(&holed, None);
    assert!((score.zncc - 1.0).abs() < 1e-9, "{}", score.zncc);
}

// ---- The bitmap --------------------------------------------------------------

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 200.0,
            focal_length_y: 200.0,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
    }
}

/// A camera at `(x, 0, 0)` looking down world `+z`.
fn pose_at(x: f64) -> RigidTransform {
    RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-x, 0.0, 0.0])
}

/// The photograph a camera at `(x, 0, 0)` takes of the plane `z = 4` painted
/// with `paint(X, Y)`.
fn photograph(x: f64, paint: impl Fn(f64, f64) -> f64) -> ImageU8Pyramid {
    let data: Vec<u8> = (0..128u32)
        .flat_map(|v| (0..128u32).map(move |u| (u, v)))
        .map(|(u, v)| {
            let wx = x + (f64::from(u) + 0.5 - 64.0) * 4.0 / 200.0;
            let wy = -(f64::from(v) + 0.5 - 64.0) * 4.0 / 200.0;
            paint(wx, wy).clamp(0.0, 255.0) as u8
        })
        .collect();
    ImageU8Pyramid::build(&ImageU8::new(128, 128, 1, data), 4)
}

/// Fine and coarse sinusoids on the plane; `fine` scales the fine ones, so a
/// photograph painted with less of them reads blurrier.
fn paint(fine: f64) -> impl Fn(f64, f64) -> f64 {
    move |x, y| {
        128.0
            + 40.0 * (7.0 * x + 3.0 * y).sin()
            + 30.0 * (-4.0 * x + 9.0 * y + 1.0).sin()
            + fine * 25.0 * (31.0 * x - 17.0 * y).sin()
            + fine * 20.0 * (23.0 * x + 29.0 * y + 2.0).sin()
    }
}

/// Three views of a patch on the plane, the first sharp and the others with
/// less fine detail, each keypoint where the patch centre projects.
struct Scene {
    camera: CameraIntrinsics,
    poses: Vec<RigidTransform>,
    pyramids: Vec<ImageU8Pyramid>,
}

impl Scene {
    fn new() -> Self {
        let xs = [0.0, 0.15, -0.15];
        let fine = [1.0, 0.3, 0.2];
        Self {
            camera: pinhole(),
            poses: xs.iter().map(|&x| pose_at(x)).collect(),
            pyramids: xs
                .iter()
                .zip(fine)
                .map(|(&x, f)| photograph(x, paint(f)))
                .collect(),
        }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.poses
            .iter()
            .zip(&self.pyramids)
            .map(|(pose, pyramid)| ProjectedImage {
                camera: &self.camera,
                cam_from_world: pose,
                pyramid,
            })
            .collect()
    }

    /// Where the camera at index `i` sees the world point `(x, y, 4)`.
    fn keypoint(&self, i: usize, x: f64, y: f64) -> [f64; 2] {
        let cx = [0.0, 0.15, -0.15][i];
        [64.0 + (x - cx) * 200.0 / 4.0, 64.0 - y * 200.0 / 4.0]
    }
}

fn patch(normal: Vector3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 4.0),
        normal,
        Vector3::y(),
        [0.25, 0.25],
    )
}

/// The stored bitmap is the tile of the view the rule picks, rendered at that
/// view's keypoint, with alpha 255 on the samples that carry data.
#[test]
fn the_bitmap_is_the_picked_view_s_tile() {
    let scene = Scene::new();
    let views = scene.views();
    let patch = patch(-Vector3::z());
    let keypoints: Vec<[f64; 2]> = (0..3).map(|i| scene.keypoint(i, 0.0, 0.0)).collect();
    let params = KeypointSubpixelParams {
        resolution: 24,
        ..Default::default()
    };
    let render = render_reference(
        &patch,
        &views,
        &[0, 1, 2],
        &keypoints.iter().map(|&k| Some(k)).collect::<Vec<_>>(),
        24,
        params.sampler,
        &Progress::none(),
    );
    let picked = render.reading.choice.reference.expect("a view is picked");
    // The pick is the rule's on the readings taken.
    assert_eq!(
        choose_reference_view(&render.reading.readings).reference,
        Some(picked)
    );
    // The sharp view has the shortest radius and is the pick.
    assert_eq!(picked, 0, "{:?}", render.semi_axes);
    let bitmap = render_patch_bitmap(
        &patch,
        &views,
        &[0, 1, 2],
        &keypoints,
        &params,
        &Progress::none(),
    )
    .expect("a bitmap");
    assert_eq!(bitmap.reference, Some(picked));
    assert_eq!(bitmap.rgba, bitmap_from_tile(&render.tiles[picked]));
    let tile = &render.tiles[picked];
    for (k, pixel) in bitmap.rgba.as_chunks::<4>().0.iter().enumerate() {
        let (row, col) = (k / 24, k % 24);
        let grey = tile.samples[[row, col, 0]];
        assert_eq!(pixel[..3], [grey, grey, grey]);
        assert_eq!(pixel[3], if tile.valid[k] { 255 } else { 0 });
    }
}

/// Where the rule picks no view, here because every view sees the patch from
/// behind, the bitmap is the fused mean and names no reference.
#[test]
fn with_no_view_picked_the_bitmap_is_the_fused_mean() {
    let scene = Scene::new();
    let views = scene.views();
    let backwards = patch(Vector3::z());
    let keypoints: Vec<[f64; 2]> = (0..3).map(|i| scene.keypoint(i, 0.0, 0.0)).collect();
    let params = KeypointSubpixelParams {
        resolution: 24,
        ..Default::default()
    };
    let bitmap = render_patch_bitmap(
        &backwards,
        &views,
        &[0, 1, 2],
        &keypoints,
        &params,
        &Progress::none(),
    )
    .expect("a mean");
    assert_eq!(bitmap.reference, None);
    let fused = crate::patch::keypoint_subpixel::fuse_patch_bitmap(
        &backwards,
        &views,
        &[0, 1, 2],
        &keypoints,
        &params,
    );
    assert_eq!(Some(bitmap.rgba), fused);
    // Fewer than two views render no bitmap at all.
    assert!(render_patch_bitmap(
        &patch(-Vector3::z()),
        &views,
        &[0],
        &keypoints[..1],
        &params,
        &Progress::none()
    )
    .is_none());
}
