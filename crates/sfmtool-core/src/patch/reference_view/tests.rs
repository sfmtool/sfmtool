// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Point3, Vector3};
use ndarray::Array3;

use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::sampler::{Sampler, SamplerChoice};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::{viewing_angle, ProjectedImage};
use crate::progress::Progress;

// ---- The viewing angle -----------------------------------------------------

/// A pose whose camera centre is `center`, with the identity rotation.
fn at(center: [f64; 3]) -> RigidTransform {
    RigidTransform::from_wxyz_translation(
        [1.0, 0.0, 0.0, 0.0],
        [-center[0], -center[1], -center[2]],
    )
}

#[test]
fn the_viewing_angle_and_tilt_direction_follow_the_camera_round_the_normal() {
    // A patch facing +z with u along +x and v along +y. A camera at height 1
    // and distance tan(θ) out along the direction at angle φ in the plane sees
    // it at θ, and its ray leans the opposite way, at φ + 180°.
    let patch = OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [0.1, 0.1]);
    for theta in [5.0f64, 30.0, 60.0, 80.0] {
        for phi in [0.0f64, 45.0, 90.0, 135.0, -60.0] {
            let (t, p) = (theta.to_radians(), phi.to_radians());
            let out = t.tan();
            let camera = at([out * p.cos(), out * p.sin(), 1.0]);
            let seen = viewing_angle(&patch, &camera).expect("the camera is off the patch");
            assert!((seen.angle_deg - theta).abs() < 1e-9, "{theta} {phi}");
            let tilt = seen.tilt_direction_deg.expect("an oblique view has a tilt");
            let expected = (phi + 180.0 + 180.0).rem_euclid(360.0) - 180.0;
            let wrapped = (tilt - expected + 180.0).rem_euclid(360.0) - 180.0;
            assert!(wrapped.abs() < 1e-9, "tilt {tilt} for phi {phi}");
        }
    }
}

#[test]
fn a_view_facing_the_patch_has_no_tilt_direction_and_a_view_of_its_back_reads_over_90() {
    let patch = OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [0.1, 0.1]);
    let facing = viewing_angle(&patch, &at([0.0, 0.0, 3.0])).unwrap();
    assert_eq!(facing.angle_deg, 0.0);
    assert_eq!(facing.tilt_direction_deg, None);
    let back = viewing_angle(&patch, &at([0.0, 0.5, -1.0])).unwrap();
    assert!(back.angle_deg > 90.0);
    // A camera on the patch's centre has no direction to it.
    assert_eq!(viewing_angle(&patch, &at([0.0, 0.0, 0.0])), None);
}

#[test]
fn a_patch_at_infinity_is_seen_along_its_bearing() {
    // The patch's centre is the bearing (0, 0, 1) from every camera, with the
    // normal facing back along it, tilted by 20° about v.
    let tilt = 20f64.to_radians();
    let normal = Vector3::new(-tilt.sin(), 0.0, -tilt.cos());
    let mut patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 1.0),
        normal,
        Vector3::y(),
        [0.01, 0.01],
    );
    patch.w = 0.0;
    for camera in [at([0.0, 0.0, 0.0]), at([5.0, -3.0, 2.0])] {
        let seen = viewing_angle(&patch, &camera).unwrap();
        assert!((seen.angle_deg - 20.0).abs() < 1e-9);
    }
}

// ---- The clipped share ------------------------------------------------------

fn grey(width: u32, height: u32, value: impl Fn(u32, u32) -> u8) -> ImageU8Pyramid {
    let data = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| value(x, y))
        .collect();
    ImageU8Pyramid::build(&ImageU8::new(width, height, 1, data), 1)
}

#[test]
fn the_clipped_share_counts_pixels_at_either_limit_inside_the_outline() {
    // Columns 0..4 are black, 4..8 mid grey, 8..12 white, 12.. mid grey.
    let pyramid = grey(20, 10, |x, _| match x {
        0..4 => 0,
        8..12 => 255,
        _ => 128,
    });
    let rect = |x0: f64, x1: f64| [[x0, 0.0], [x1, 0.0], [x1, 10.0], [x0, 10.0]];
    assert_eq!(clipped_share(&pyramid, &rect(4.0, 8.0)), Some(0.0));
    assert_eq!(clipped_share(&pyramid, &rect(0.0, 8.0)), Some(0.5));
    assert_eq!(clipped_share(&pyramid, &rect(4.0, 12.0)), Some(0.5));
    assert_eq!(clipped_share(&pyramid, &rect(0.0, 12.0)), Some(8.0 / 12.0));
    // A triangle holds the pixels whose centres it holds.
    let triangle = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let share = clipped_share(&pyramid, &triangle).unwrap();
    assert!(share > 0.5 && share < 1.0, "{share}");
    // Fewer than three points is no outline.
    assert_eq!(clipped_share(&pyramid, &[[1.0, 1.0], [2.0, 2.0]]), None);
}

#[test]
fn an_outline_smaller_than_a_pixel_reads_the_pixels_under_its_points() {
    let pyramid = grey(10, 10, |x, _| if x < 5 { 255 } else { 100 });
    let tiny = [[2.1, 2.1], [2.3, 2.1], [2.2, 2.3]];
    assert_eq!(clipped_share(&pyramid, &tiny), Some(1.0));
    let tiny = [[6.1, 2.1], [6.3, 2.1], [6.2, 2.3]];
    assert_eq!(clipped_share(&pyramid, &tiny), Some(0.0));
}

#[test]
fn a_colour_pixel_is_clipped_when_any_of_its_three_channels_is() {
    // One pixel per kind: none at a limit, red at 255, blue at 0.
    let data = vec![10, 20, 30, 255, 20, 30, 10, 20, 0];
    let pyramid = ImageU8Pyramid::build(&ImageU8::new(3, 1, 3, data), 1);
    let all = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]];
    let share = clipped_share(&pyramid, &all).unwrap();
    assert!((share - 2.0 / 3.0).abs() < 1e-12);
}

/// The share by testing every pixel centre in the outline's bounding box
/// against every edge: the definition the row-by-row spans compute.
fn clipped_share_by_pixel(pyramid: &ImageU8Pyramid, outline: &[[f64; 2]]) -> f64 {
    let image = pyramid.level(0);
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let inside = |x: f64, y: f64| {
        let mut odd = false;
        let mut j = outline.len() - 1;
        for i in 0..outline.len() {
            let ([xi, yi], [xj, yj]) = (outline[i], outline[j]);
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                odd = !odd;
            }
            j = i;
        }
        odd
    };
    let (mut n, mut clipped) = (0, 0);
    for y in 0..h {
        for x in 0..w {
            if inside(x as f64 + 0.5, y as f64 + 0.5) {
                n += 1;
                let v = image.data()[(y * w + x) as usize];
                clipped += usize::from(v == 0 || v == 255);
            }
        }
    }
    clipped as f64 / n as f64
}

#[test]
fn the_row_spans_count_the_pixels_a_test_of_every_centre_counts() {
    // A photograph with clipped pixels scattered through it, and outlines that
    // are rotated, concave, self-crossing, run off the photograph, and put
    // vertices exactly on pixel centres.
    let pyramid = grey(40, 30, |x, y| match (x * 7 + y * 13) % 11 {
        0 => 0,
        1 => 255,
        _ => 100,
    });
    let outlines: Vec<Vec<[f64; 2]>> = vec![
        vec![[5.3, 2.1], [30.7, 8.4], [24.2, 27.9], [1.1, 20.6]],
        vec![
            [2.0, 2.0],
            [20.0, 2.0],
            [11.0, 12.0],
            [20.0, 25.0],
            [2.0, 25.0],
        ],
        vec![[3.5, 3.5], [35.5, 25.5], [35.5, 3.5], [3.5, 25.5]],
        vec![[-10.0, -5.0], [55.0, 4.0], [33.0, 45.0]],
        vec![[10.5, 10.5], [20.5, 10.5], [20.5, 20.5], [10.5, 20.5]],
    ];
    for outline in &outlines {
        let spans = clipped_share(&pyramid, outline).unwrap();
        let by_pixel = clipped_share_by_pixel(&pyramid, outline);
        assert_eq!(spans, by_pixel, "{outline:?}");
    }
}

// ---- One view's tile --------------------------------------------------------

fn pinhole(width: u32, height: u32) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 100.0,
            focal_length_y: 100.0,
            principal_point_x: f64::from(width) / 2.0,
            principal_point_y: f64::from(height) / 2.0,
        },
        width,
        height,
    }
}

#[test]
fn a_tile_inside_the_photograph_is_fully_covered_and_one_over_its_edge_is_not() {
    // A camera at the origin looking down +z (a half turn about x puts the
    // canonical -z axis along +z), and a patch at z = 4 facing it.
    let camera = pinhole(64, 64);
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let pyramid = grey(64, 64, |x, y| ((x * 7 + y * 3) % 200 + 20) as u8);
    let view = ProjectedImage {
        camera: &camera,
        cam_from_world: &pose,
        pyramid: &pyramid,
    };
    let patch = OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, 4.0),
        -Vector3::z(),
        Vector3::y(),
        [0.2, 0.2],
    );
    let tile = render_view_tile(
        &patch,
        &view,
        None,
        16,
        SamplerChoice::per_view(),
        &Progress::none(),
    );
    assert_eq!(tile.samples.shape(), &[16, 16, 1]);
    assert_eq!(tile.coverage, 1.0);
    assert!(tile.valid.iter().all(|&v| v));
    assert_eq!(tile.clipped_share, Some(0.0));
    assert!(tile.viewing_angle.unwrap().angle_deg < 1e-6);
    assert_eq!(tile.sampler, Sampler::BilinearMip);
    // Moved so its centre sits near the photograph's right edge, about half of
    // it is off the photograph.
    let mut edge = patch.clone();
    edge.center = Point3::new(1.28, 0.0, 4.0);
    let tile = render_view_tile(
        &edge,
        &view,
        None,
        16,
        SamplerChoice::per_view(),
        &Progress::none(),
    );
    assert!(
        tile.coverage > 0.3 && tile.coverage < 0.7,
        "{}",
        tile.coverage
    );
    // Off the photograph a sample is black.
    let off = tile.valid.iter().position(|&v| !v).unwrap();
    assert_eq!(tile.samples[[off / 16, off % 16, 0]], 0);
}

/// The polygon through the photograph positions of the centres of `patch`'s
/// border samples at resolution `r`, in the order the tile's outline takes
/// them, each projected on its own through the camera rather than read off a
/// warp map. A sample that lands off the photograph is left out, as the tile
/// leaves it out.
fn projected_outline(
    patch: &OrientedPatch,
    camera: &CameraIntrinsics,
    pose: &RigidTransform,
    r: usize,
) -> Vec<[f64; 2]> {
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let at = |col: usize, row: usize| {
        let s = (col as f64 + 0.5) * 2.0 / r as f64 - 1.0;
        let t = (row as f64 + 0.5) * 2.0 / r as f64 - 1.0;
        // Rows run down the tile, along -v.
        let world = patch.center + patch.u_axis * (patch.half_extent[0] * s)
            - patch.v_axis * (patch.half_extent[1] * t);
        let ray = pose.transform_point(&world);
        camera
            .ray_to_pixel([ray.x, ray.y, ray.z])
            .filter(|&(x, y)| x >= 0.0 && y >= 0.0 && x < w && y < h)
            .map(|(x, y)| [x, y])
    };
    let border = (0..r)
        .map(|col| (col, 0))
        .chain((1..r).map(|row| (r - 1, row)))
        .chain((0..r - 1).rev().map(|col| (col, r - 1)))
        .chain((1..r - 1).rev().map(|row| (0, row)));
    border.filter_map(|(col, row)| at(col, row)).collect()
}

/// A camera at the origin looking down world `+z`, and a patch at `z = 4`
/// facing it with half-extent `half`, centred at `x`.
fn facing_patch(x: f64, half: f64) -> (RigidTransform, OrientedPatch) {
    let pose = RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let patch = OrientedPatch::from_center_normal(
        Point3::new(x, 0.0, 4.0),
        -Vector3::z(),
        Vector3::y(),
        [half, half],
    );
    (pose, patch)
}

#[test]
fn the_clipped_share_of_a_fisheye_tile_is_read_inside_its_distorted_outline() {
    // A fisheye, and a photograph blown out left of a line thirty pixels
    // right of the principal point, so the share depends on the outline's
    // shape and not only on its symmetry.
    let camera = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 60.0,
            focal_length_y: 60.0,
            principal_point_x: 80.0,
            principal_point_y: 80.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 160,
        height: 160,
    };
    let pyramid = grey(160, 160, |x, _| if x < 110 { 255 } else { 100 });
    let (pose, patch) = facing_patch(0.0, 3.0);
    let view = ProjectedImage {
        camera: &camera,
        cam_from_world: &pose,
        pyramid: &pyramid,
    };
    let r = 24;
    let tile = render_view_tile(
        &patch,
        &view,
        None,
        r,
        SamplerChoice::per_view(),
        &Progress::none(),
    );
    assert_eq!(tile.coverage, 1.0);
    let share = tile.clipped_share.expect("the tile is on the photograph");
    let outline = projected_outline(&patch, &camera, &pose, r);
    let expected = clipped_share_by_pixel(&pyramid, &outline);
    // The warp map projects a fisheye through a coarse grid with a bounded
    // error, so a pixel on the outline's edge may fall either way.
    assert!(
        (share - expected).abs() < 0.01,
        "{share} against {expected}"
    );

    // The distortion matters: the same patch through a pinhole of the same
    // focal length has a larger outline, with a different share.
    let pinhole = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 60.0,
            focal_length_y: 60.0,
            principal_point_x: 80.0,
            principal_point_y: 80.0,
        },
        width: 160,
        height: 160,
    };
    let undistorted =
        clipped_share_by_pixel(&pyramid, &projected_outline(&patch, &pinhole, &pose, r));
    assert!(
        (undistorted - share).abs() > 0.02,
        "{undistorted} against {share}"
    );
}

#[test]
fn the_clipped_share_of_a_tile_cut_by_the_photograph_s_edge_is_read_inside_the_part_on_it() {
    // The rightmost four columns are blown out, and the patch sits over the
    // right edge, so part of its outline is the chord between the last border
    // samples on the photograph.
    let camera = pinhole(64, 64);
    let pyramid = grey(64, 64, |x, _| if x >= 60 { 255 } else { 100 });
    let (pose, patch) = facing_patch(1.2, 0.2);
    let view = ProjectedImage {
        camera: &camera,
        cam_from_world: &pose,
        pyramid: &pyramid,
    };
    let r = 16;
    let tile = render_view_tile(
        &patch,
        &view,
        None,
        r,
        SamplerChoice::per_view(),
        &Progress::none(),
    );
    assert!(
        tile.coverage > 0.2 && tile.coverage < 0.8,
        "{}",
        tile.coverage
    );
    let share = tile
        .clipped_share
        .expect("part of the tile is on the photograph");
    let outline = projected_outline(&patch, &camera, &pose, r);
    assert!(outline.len() < 4 * r - 4, "some border samples are off it");
    let expected = clipped_share_by_pixel(&pyramid, &outline);
    assert!(
        (share - expected).abs() < 1e-3,
        "{share} against {expected}"
    );
    assert!(share > 0.1 && share < 0.9, "{share}");
}

// ---- The cell check -----------------------------------------------------------

/// A one-channel tile of side 24 whose sample at `(row, col)` is `value`,
/// every sample carrying data.
fn tile_of(value: impl Fn(usize, usize) -> u8) -> ViewTile {
    let r = 24;
    let samples = Array3::from_shape_fn((r, r, 1), |(row, col, _)| value(row, col));
    ViewTile {
        samples,
        valid: vec![true; r * r],
        placement: OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [1.0, 1.0]),
        jacobian: None,
        sampler: Sampler::BilinearMip,
        coverage: 1.0,
        clipped_share: Some(0.0),
        viewing_angle: None,
    }
}

fn texture(row: usize, col: usize, seed: usize) -> u8 {
    // A fixed pseudo-random texture plus a little view-dependent noise.
    let base = (row * 37 + col * 91 + (row * col) % 17) % 200;
    let noise = (row * 13 + col * 7 + seed * 29) % 9;
    (20 + base + noise) as u8
}

#[test]
fn identical_tiles_agree_fully_in_every_cell() {
    let a = tile_of(|r, c| texture(r, c, 0));
    let grid = pair_zncc_grid(&a, &a);
    for row in grid {
        for z in row {
            assert!((z - 1.0).abs() < 1e-12, "{z}");
        }
    }
}

#[test]
fn a_cell_flat_in_either_tile_or_short_of_samples_has_no_reading() {
    let a = tile_of(|r, c| texture(r, c, 0));
    // Flat over the top-left cell.
    let b = tile_of(|r, c| if r < 8 && c < 8 { 90 } else { texture(r, c, 1) });
    let grid = pair_zncc_grid(&a, &b);
    assert!(grid[0][0].is_nan());
    assert!(grid[1][1].is_finite());
    // Only 15 samples of the bottom-right cell carry data in both.
    let mut c = tile_of(|r, c| texture(r, c, 2));
    for row in 16..24 {
        for col in 16..24 {
            if (row - 16) * 8 + (col - 16) >= 15 {
                c.valid[row * 24 + col] = false;
            }
        }
    }
    let grid = pair_zncc_grid(&a, &c);
    assert!(grid[2][2].is_nan());
    assert!(grid[2][1].is_finite());
}

#[test]
fn a_view_with_an_occluder_in_one_cell_carries_the_deficit() {
    // Five views of one texture; the fourth has something else in front of the
    // middle-right cell.
    let tiles: Vec<ViewTile> = (0..5)
        .map(|v| {
            tile_of(move |r, c| {
                if v == 3 && (8..16).contains(&r) && c >= 16 {
                    ((r * 11 + c * 5) % 50 + 100) as u8
                } else {
                    texture(r, c, v)
                }
            })
        })
        .collect();
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let agreement = cell_agreement(&refs);
    assert!(agreement.typical[1][2] > 0.9);
    for (v, deficit) in agreement.deficit.iter().enumerate() {
        if v == 3 {
            assert!(*deficit > REFERENCE_MAX_CELL_DEFICIT, "{deficit}");
        } else {
            assert!(*deficit < 0.1, "view {v}: {deficit}");
        }
    }
    // The other views' agreement in that cell is the median over four
    // partners, one of them the occluded view, so it barely moves.
    assert!(agreement.pair_zncc_grid[0][1][2] > 0.9);
}

#[test]
fn cells_the_track_does_not_agree_on_are_not_judged() {
    // A typical agreement under the floor in every cell leaves no cell judged,
    // and every deficit is 0.
    let low = [[0.2; 3]; 3];
    let k = 3;
    let mut pairs = vec![low; k * k];
    pairs[1] = [[-0.5; 3]; 3];
    pairs[k] = [[-0.5; 3]; 3];
    let agreement = cell_agreement_from_pairs(&pairs, k);
    assert_eq!(agreement.deficit, vec![0.0; 3]);
    // With one judged cell, the deficit is read there alone.
    let mut pairs = vec![low; k * k];
    for (a, b, z) in [(0, 1, 0.9), (0, 2, 0.9), (1, 2, 0.3)] {
        pairs[a * k + b][0][0] = z;
        pairs[b * k + a][0][0] = z;
    }
    let agreement = cell_agreement_from_pairs(&pairs, k);
    // Views 0, 1, 2 read 0.9, 0.6, 0.6 in the top-left cell; typical 0.6.
    assert!((agreement.typical[0][0] - 0.6).abs() < 1e-12);
    assert!((agreement.deficit[0] - -0.3).abs() < 1e-12);
    assert!(agreement.deficit[1].abs() < 1e-12);
}

// ---- The rule -------------------------------------------------------------------

/// A view that passes every candidate test, with median pairwise ZNCC `pair`
/// and self-similarity semi-major axis `radius`.
fn good(pair: f64, radius: f64) -> ReferenceReadings {
    ReferenceReadings {
        coverage: Some(1.0),
        clipped_share: Some(0.0),
        viewing_angle_deg: Some(20.0),
        cell_deficit: Some(0.0),
        pair_zncc: Some(pair),
        semi_major: Some(radius),
        semi_minor: Some(radius * 0.5),
    }
}

#[test]
fn each_candidate_test_turns_away_the_view_that_fails_it() {
    let mut views = vec![good(0.9, 0.3); 6];
    views[0].coverage = Some(0.98);
    views[1].clipped_share = Some(0.06);
    views[2].viewing_angle_deg = Some(65.5);
    views[3].cell_deficit = Some(0.31);
    views[4].semi_major = Some(0.8);
    let choice = choose_reference_view(&views);
    assert_eq!(choice.fallback, ReferenceFallback::None);
    assert_eq!(choice.reference, Some(5));
    assert_eq!(
        choice.rejected_by,
        vec![
            Some(ReferenceTest::Coverage),
            Some(ReferenceTest::Clipped),
            Some(ReferenceTest::Angle),
            Some(ReferenceTest::Cells),
            Some(ReferenceTest::Sharpness),
            None,
        ]
    );
}

#[test]
fn the_thresholds_are_inclusive() {
    let mut view = good(0.9, 0.3);
    view.coverage = Some(REFERENCE_MIN_COVERAGE);
    view.clipped_share = Some(REFERENCE_MAX_CLIPPED_SHARE);
    view.viewing_angle_deg = Some(REFERENCE_MAX_VIEWING_ANGLE_DEG);
    view.cell_deficit = Some(REFERENCE_MAX_CELL_DEFICIT);
    // Exactly the margin below the best, so the sharpest view stays in.
    let margin = good(0.9 - REFERENCE_AGREEMENT_MARGIN, 0.2);
    let choice = choose_reference_view(&[view, good(0.9, 0.4), margin]);
    assert_eq!(choice.fallback, ReferenceFallback::None);
    assert_eq!(choice.reference, Some(2));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Sharpness));
}

#[test]
fn the_agreement_margin_keeps_sharp_views_close_to_the_best_and_drops_the_rest() {
    let views = [good(0.95, 0.9), good(0.82, 0.3), good(0.79, 0.1)];
    let choice = choose_reference_view(&views);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(choice.rejected_by[2], Some(ReferenceTest::Agreement));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Sharpness));
}

#[test]
fn the_margin_is_measured_from_the_best_candidate_only() {
    // A view that fails a candidate test does not set the best agreement.
    let mut grazing = good(0.99, 0.5);
    grazing.viewing_angle_deg = Some(80.0);
    let choice = choose_reference_view(&[grazing, good(0.8, 0.4), good(0.7, 0.2)]);
    assert_eq!(choice.reference, Some(2));
}

#[test]
fn a_tie_on_the_semi_major_axis_goes_to_the_smaller_semi_minor_then_the_earlier_view() {
    let mut a = good(0.9, 0.4);
    a.semi_minor = Some(0.3);
    let mut b = good(0.9, 0.4);
    b.semi_minor = Some(0.2);
    assert_eq!(choose_reference_view(&[a, b]).reference, Some(1));
    assert_eq!(choose_reference_view(&[b, b]).reference, Some(0));
}

#[test]
fn when_no_view_passes_the_rule_drops_the_angle_then_the_cells_then_usability() {
    // Every view grazes: the angle test is dropped and nothing else.
    let mut views = vec![good(0.9, 0.5); 3];
    for v in &mut views {
        v.viewing_angle_deg = Some(75.0);
    }
    views[1].coverage = Some(0.5);
    let choice = choose_reference_view(&views);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAngle);
    assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Coverage));
    assert_eq!(choice.reference, Some(0));

    // Every view also fails the cell check.
    for v in &mut views {
        v.cell_deficit = Some(0.5);
    }
    let choice = choose_reference_view(&views);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAngleOrCells);
    assert_eq!(choice.reference, Some(0));
    assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Coverage));

    // And every view is partial: every view is a candidate.
    views[0].coverage = Some(0.9);
    views[2].clipped_share = Some(0.5);
    views[2].coverage = Some(1.0);
    let choice = choose_reference_view(&views);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAny);
    assert!(choice.reference.is_some());
}

#[test]
fn a_view_that_passes_a_dropped_test_does_not_bring_it_back() {
    // One view passes everything but the cell check, another everything but
    // the angle: no view passes all four, so the angle is dropped and the
    // first view, which fails the cells, stays out.
    let mut grazing = good(0.9, 0.5);
    grazing.viewing_angle_deg = Some(70.0);
    let mut occluded = good(0.9, 0.1);
    occluded.cell_deficit = Some(0.6);
    let choice = choose_reference_view(&[occluded, grazing]);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAngle);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Cells));
}

#[test]
fn missing_readings_fail_coverage_and_angle_and_pass_clipped_and_cells() {
    let mut no_coverage = good(0.9, 0.1);
    no_coverage.coverage = None;
    let mut no_angle = good(0.9, 0.1);
    no_angle.viewing_angle_deg = None;
    let mut no_clip_or_cells = good(0.9, 0.3);
    no_clip_or_cells.clipped_share = None;
    no_clip_or_cells.cell_deficit = None;
    let choice = choose_reference_view(&[no_coverage, no_angle, no_clip_or_cells]);
    assert_eq!(choice.reference, Some(2));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Coverage));
    assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Angle));
}

#[test]
fn a_view_without_a_pairwise_reading_is_held_out_unless_none_has_one() {
    let mut unscored = good(0.9, 0.1);
    unscored.pair_zncc = None;
    let choice = choose_reference_view(&[unscored, good(0.5, 0.4)]);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Agreement));
    let mut other = good(0.9, 0.4);
    other.pair_zncc = None;
    let choice = choose_reference_view(&[unscored, other]);
    assert_eq!(choice.reference, Some(0));
}

#[test]
fn no_views_or_no_radius_picks_nothing() {
    let choice = choose_reference_view(&[]);
    assert_eq!(choice.reference, None);
    assert!(choice.rejected_by.is_empty());
    let mut flat = good(0.9, 0.1);
    flat.semi_major = None;
    let choice = choose_reference_view(&[flat]);
    assert_eq!(choice.reference, None);
    assert_eq!(choice.rejected_by, vec![Some(ReferenceTest::Sharpness)]);
}

#[test]
fn a_view_of_the_patch_edge_on_or_from_behind_is_never_a_candidate() {
    // The sharpest view sees the patch's back, the next is edge on, and the
    // third grazes: no view passes the 65° limit, the fallback drops it, and
    // the facing limit still turns the first two away.
    let mut back = good(0.9, 0.1);
    back.viewing_angle_deg = Some(111.0);
    let mut edge_on = good(0.9, 0.2);
    edge_on.viewing_angle_deg = Some(REFERENCE_FACING_LIMIT_DEG);
    let mut grazing = good(0.9, 0.5);
    grazing.viewing_angle_deg = Some(89.0);
    let choice = choose_reference_view(&[back, edge_on, grazing]);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAngle);
    assert_eq!(choice.reference, Some(2));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Angle));
    assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Angle));

    // With every other test dropped as well, the facing limit still holds.
    let mut views = [back, edge_on, grazing];
    for v in &mut views {
        v.coverage = Some(0.5);
        v.cell_deficit = Some(0.9);
    }
    let choice = choose_reference_view(&views);
    assert_eq!(choice.fallback, ReferenceFallback::WithoutAny);
    assert_eq!(choice.reference, Some(2));
    assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Angle));

    // Where every view faces away, the rule picks nothing.
    let choice = choose_reference_view(&[back, edge_on]);
    assert_eq!(choice.reference, None);
    assert_eq!(
        choice.rejected_by,
        vec![Some(ReferenceTest::Angle), Some(ReferenceTest::Angle)]
    );
}

// ---- Blur-matched readings -------------------------------------------------------

/// The rule on blur-matched readings applies the thresholds that go with them,
/// and says so in its standings.
#[test]
fn the_blur_matched_inputs_apply_their_own_margin_and_cell_bar() {
    let inputs = ReferenceRuleInputs {
        agreement: PairZnccReading::BlurMatched,
        cells: PairZnccReading::BlurMatched,
    };
    assert_eq!(
        inputs.agreement_margin(),
        REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN
    );
    assert_eq!(
        inputs.max_cell_deficit(),
        REFERENCE_MAX_BLUR_MATCHED_CELL_DEFICIT
    );
    assert_eq!(
        ReferenceRuleInputs::PLAIN.agreement_margin(),
        REFERENCE_AGREEMENT_MARGIN
    );
    // The blur-matched margin is inclusive.
    let views = [
        good(0.95, 0.9),
        good(0.95 - REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN, 0.2),
        good(0.95 - REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN - 0.01, 0.1),
    ];
    let choice = choose_reference_view_with(&views, inputs);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(choice.rejected_by[2], Some(ReferenceTest::Agreement));
    assert_eq!(choice.standing(0).unwrap().inputs, inputs);
    // A cell deficit between the two bars.
    let mut cells = good(0.95, 0.2);
    cells.cell_deficit =
        Some(0.5 * (REFERENCE_MAX_CELL_DEFICIT + REFERENCE_MAX_BLUR_MATCHED_CELL_DEFICIT));
    let views = [good(0.95, 0.9), cells];
    assert_eq!(choose_reference_view(&views).reference, Some(1));
    let choice = choose_reference_view_with(&views, inputs);
    assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Cells));
}

/// Blur-matched agreement with [`PairMatching::Plain`] reads the cells exactly
/// as the plain cell agreement does, and on views blurred by differing amounts
/// it lifts the blurriest view's agreement.
#[test]
fn blur_matched_agreement_reads_plain_cells_without_blur_and_lifts_a_blurred_view() {
    use crate::patch::blur_matched::BlurScratch;
    use crate::patch::normal_refine::PatchWindow;
    use crate::patch::pair_sharpness::PairMatching;
    use crate::patch::self_similarity::{
        zncc_self_similarity_parts, PatchTile, SelfSimilarityParams,
    };

    // Four views of one smooth texture; the last blurred by 1.5 grid px.
    let smooth = |r: usize, c: usize| {
        let (x, y) = (c as f64, r as f64);
        (128.0
            + 50.0 * (0.9 * x + 0.3 * y).sin()
            + 40.0 * (-0.5 * x + 1.1 * y + 1.3).sin()
            + 25.0 * (1.2 * x - 0.8 * y + 2.0).sin()) as u8
    };
    let mut tiles: Vec<ViewTile> = (0..4).map(|_| tile_of(smooth)).collect();
    let planes = tiles[3].planes();
    let blurred = planes.blurred(1.5, &mut BlurScratch::default());
    for (k, v) in blurred.values.iter().enumerate() {
        tiles[3].samples[[k / 24, k % 24, 0]] = v.round() as u8;
    }
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = tiles
        .iter()
        .map(|t| {
            let p = t.planes();
            let parts = zncc_self_similarity_parts(
                &PatchTile {
                    values: &p.values,
                    channels: 1,
                    width: 24,
                    height: 24,
                },
                None,
                &SelfSimilarityParams::default(),
            );
            Some(parts.whole.ellipse.matrix)
        })
        .collect();
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let window = PatchWindow::GaussianDisk { sigma: 0.6 };
    let plain = blur_matched_agreement(
        &refs,
        &ellipses,
        PairMatching::Plain,
        window,
        &Progress::none(),
    );
    assert_eq!(plain.pairs.pairs_blurred, 0);
    assert_eq!(
        format!("{:?}", plain.cells),
        format!("{:?}", cell_agreement(&refs))
    );
    let matched = blur_matched_agreement(
        &refs,
        &ellipses,
        PairMatching::BlurMatched,
        window,
        &Progress::none(),
    );
    assert!(matched.pairs.pairs_blurred >= 3);
    assert!(
        matched.pair_zncc[3] > plain.pair_zncc[3] + 0.01,
        "blur-matched {} against plain {}",
        matched.pair_zncc[3],
        plain.pair_zncc[3]
    );
    assert!(matched.pair_zncc[3] > 0.92, "{}", matched.pair_zncc[3]);
}
