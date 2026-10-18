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
        patch_frame: OrientedPatch::new(Point3::origin(), Vector3::x(), Vector3::y(), [1.0, 1.0]),
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

/// Five views of one texture, view 3 showing `odd(row, col)` over the
/// middle-right cell instead.
fn five_views(odd: impl Fn(usize, usize) -> u8 + Copy) -> Vec<ViewTile> {
    (0..5)
        .map(|v| {
            tile_of(move |r, c| {
                if v == 3 && (8..16).contains(&r) && c >= 16 {
                    odd(r, c)
                } else {
                    texture(r, c, v)
                }
            })
        })
        .collect()
}

/// The cell agreement of `tiles`, each with its own self-similarity reading.
fn agreement_of(tiles: &[ViewTile]) -> CellAgreement {
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let semi_axes: Vec<Option<TileSemiAxes>> = tiles.iter().map(tile_semi_axes).collect();
    cell_agreement(&refs, &semi_axes)
}

#[test]
fn a_view_that_disagrees_in_one_textured_cell_carries_the_deficit() {
    // View 3 shows other texture over the middle-right cell, every sample of
    // it on the photograph: something in front of the surface there, or the
    // surface seen with parallax.
    let tiles = five_views(|r, c| ((r * 11 + c * 5) % 50 + 100) as u8);
    let cell = tile_semi_axes(&tiles[3]).unwrap().grid[1][2];
    assert!(
        cell[1] < tile_max_radius(),
        "the cell holds texture: {cell:?}"
    );
    let agreement = agreement_of(&tiles);
    assert!(agreement.typical[1][2] > 0.9);
    assert!(agreement.judged.iter().all(|v| v[1][2]));
    for (v, deficit) in agreement.deficit.iter().enumerate() {
        if v == 3 {
            assert!(
                *deficit > ReferenceViewOptions::default().cell_deficit_band,
                "{deficit}"
            );
        } else {
            assert!(*deficit < 0.1, "view {v}: {deficit}");
        }
    }
    // The other views' agreement in that cell is the median over four
    // partners, one of them view 3, so it barely moves.
    assert!(agreement.pair_zncc_grid[0][1][2] > 0.9);
}

#[test]
fn a_cell_with_no_texture_in_the_view_is_not_judged() {
    // View 3 holds almost nothing over the middle-right cell: a pattern a grey
    // level deep, which the noise of a view hides, so that the cell matches
    // itself shifted anywhere in the search.
    let tiles = five_views(|r, c| (100 + (r * 7 + c * 3) % 3) as u8);
    let cell = tile_semi_axes(&tiles[3]).unwrap().grid[1][2];
    assert_eq!(cell, [tile_max_radius(); 2], "both axes at the cap");
    let agreement = agreement_of(&tiles);
    assert!(
        agreement.pair_zncc_grid[3][1][2] < 0.3,
        "it disagrees there"
    );
    assert!(!agreement.judged[3][1][2]);
    assert!(agreement.judged[3][1][1] && agreement.judged[0][1][2]);
    assert!(agreement.deficit[3] < 0.1, "{}", agreement.deficit[3]);
    // Read with no self-similarity reading, the same cell is judged.
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let unread = cell_agreement(&refs, &[None; 5]);
    assert!(unread.judged[3][1][2]);
    assert!(
        unread.deficit[3] > ReferenceViewOptions::default().cell_deficit_band,
        "{}",
        unread.deficit[3]
    );
}

#[test]
fn a_cell_with_samples_off_the_photograph_is_not_judged() {
    // View 3 shows other texture over the middle-right cell, as in the test of
    // a textured cell, but its tile runs off the photograph over the cell's
    // last column, 8 of its 64 samples.
    let mut tiles = five_views(|r, c| ((r * 11 + c * 5) % 50 + 100) as u8);
    for row in 8..16 {
        tiles[3].valid[row * 24 + 23] = false;
        tiles[3].samples[[row, 23, 0]] = 0;
    }
    tiles[3].coverage = (24.0 * 24.0 - 8.0) / (24.0 * 24.0);
    let agreement = agreement_of(&tiles);
    assert!(
        agreement.pair_zncc_grid[3][1][2].is_finite(),
        "the cell still has a reading"
    );
    assert!(!agreement.judged[3][1][2]);
    assert!(agreement.judged[3][1][1] && agreement.judged[0][1][2]);
    assert!(agreement.deficit[3] < 0.1, "{}", agreement.deficit[3]);
}

#[test]
fn a_cell_off_the_photograph_leaves_the_typical_agreement_and_one_with_no_texture_stays() {
    // Three views agree at 0.9 in the top-left cell and a fourth reads 0.1
    // with each of them there. Views 0 to 2 read the median of 0.9, 0.9 and
    // 0.1; view 3 reads 0.1.
    let k = 4;
    let mut pairs = vec![[[0.9; 3]; 3]; k * k];
    for w in 0..3 {
        pairs[3 * k + w][0][0] = 0.1;
        pairs[w * k + 3][0][0] = 0.1;
    }
    let every = [[[true; 3]; 3]; 4];
    let read = cell_agreement_from_pairs(&pairs, k, &every, &every);
    assert!((read.typical[0][0] - 0.9).abs() < 1e-12);
    assert!((read.deficit[3] - 0.8).abs() < 1e-12);
    // View 3 with no texture there: not judged, but still in the typical.
    let mut flat = every;
    flat[3][0][0] = false;
    let no_texture = cell_agreement_from_pairs(&pairs, k, &every, &flat);
    assert_eq!(no_texture.typical, read.typical);
    assert!(!no_texture.judged[3][0][0] && no_texture.judged[0][0][0]);
    assert!(
        no_texture.deficit[3].abs() < 1e-12,
        "{}",
        no_texture.deficit[3]
    );
    // Views 0 to 2 off the photograph there: neither judged nor in the
    // typical, which view 3 alone then sets, under the floor.
    let mut off = every;
    for own in off.iter_mut().take(3) {
        own[0][0] = false;
    }
    let off_photo = cell_agreement_from_pairs(&pairs, k, &off, &every);
    assert!((off_photo.typical[0][0] - 0.1).abs() < 1e-12);
    assert!(off_photo.judged.iter().all(|v| !v[0][0]));
    assert!(off_photo.deficit.iter().all(|d| d.abs() < 1e-12));
}

#[test]
fn cells_the_track_does_not_agree_on_are_not_judged() {
    // A typical agreement under the floor in every cell leaves no cell judged,
    // and every deficit is 0.
    let low = [[0.2; 3]; 3];
    let k = 3;
    let every = [[[true; 3]; 3]; 3];
    let mut pairs = vec![low; k * k];
    pairs[1] = [[-0.5; 3]; 3];
    pairs[k] = [[-0.5; 3]; 3];
    let agreement = cell_agreement_from_pairs(&pairs, k, &every, &every);
    assert_eq!(agreement.deficit, vec![0.0; 3]);
    assert!(agreement.judged.iter().flatten().flatten().all(|j| !j));
    // With one judged cell, the deficit is read there alone.
    let mut pairs = vec![low; k * k];
    for (a, b, z) in [(0, 1, 0.9), (0, 2, 0.9), (1, 2, 0.3)] {
        pairs[a * k + b][0][0] = z;
        pairs[b * k + a][0][0] = z;
    }
    let agreement = cell_agreement_from_pairs(&pairs, k, &every, &every);
    // Views 0, 1, 2 read 0.9, 0.6, 0.6 in the top-left cell; typical 0.6.
    assert!((agreement.typical[0][0] - 0.6).abs() < 1e-12);
    assert!((agreement.deficit[0] - -0.3).abs() < 1e-12);
    assert!(agreement.deficit[1].abs() < 1e-12);
}

#[test]
fn a_cell_holds_texture_unless_it_reads_the_cap_along_both_axes() {
    let mut axes = TileSemiAxes {
        whole: [0.5, 0.4],
        grid: [[[0.5, 0.4]; 3]; 3],
    };
    // An edge: at the cap along it, short across it.
    axes.grid[0][1] = [tile_max_radius(), 0.3];
    // No texture: at the cap along both axes.
    axes.grid[0][2] = [tile_max_radius(); 2];
    // No reading.
    axes.grid[1][0] = [f64::NAN; 2];
    assert_eq!(
        cells_with_texture(Some(&axes)),
        [[true, true, false], [true, true, true], [true, true, true]]
    );
    assert_eq!(cells_with_texture(None), [[true; 3]; 3]);
}

#[test]
fn a_cell_is_on_the_photograph_only_with_every_sample_on_it() {
    let mut tile = tile_of(|r, c| texture(r, c, 0));
    assert_eq!(cells_on_photograph(&tile), [[true; 3]; 3]);
    // One sample of the bottom-right cell off the photograph.
    tile.valid[23 * 24 + 23] = false;
    assert_eq!(
        cells_on_photograph(&tile),
        [[true, true, true], [true, true, true], [true, true, false]]
    );
}

// ---- The rule -------------------------------------------------------------------

/// A view inside every band, at 20°, with median pairwise ZNCC `pair` and
/// self-similarity semi-major axis `radius`, read with a cap of 3 grid px.
fn good(pair: f64, radius: f64) -> ReferenceReadings {
    ReferenceReadings {
        coverage: Some(1.0),
        clipped_share: Some(0.0),
        viewing_angle_deg: Some(20.0),
        cell_deficit: Some(0.0),
        pair_zncc: Some(pair),
        semi_major: Some(radius),
        semi_minor: Some(radius * 0.5),
        max_radius: Some(CAP),
    }
}

/// The cap of the readings `good` makes.
const CAP: f64 = 3.0;

/// The rule at the default options.
fn choose(views: &[ReferenceReadings]) -> ReferenceChoice {
    choose_reference_view(views, &ReferenceViewOptions::default())
}

fn cost_of(choice: &ReferenceChoice, i: usize) -> ReferenceCost {
    choice.costs[i].expect("the view has a cost")
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn a_view_facing_the_patch_beats_a_slightly_sharper_oblique_one() {
    let mut oblique = good(0.9, 0.60);
    oblique.viewing_angle_deg = Some(55.0);
    let facing = good(0.9, 0.62);
    let choice = choose(&[oblique, facing]);
    assert_eq!(choice.reference, Some(1));
    let k = ReferenceViewOptions::default().angle_weight;
    assert_eq!(k, 0.25);
    let oblique_cost = cost_of(&choice, 0);
    assert!(close(
        oblique_cost.angle,
        1.0 + k * 55f64.to_radians().tan()
    ));
    assert!(
        (oblique_cost.cost - 0.81).abs() < 0.01,
        "{}",
        oblique_cost.cost
    );
    assert!((cost_of(&choice, 1).cost - 0.68).abs() < 0.01);
    let standing = choice.standing(0).unwrap();
    assert_eq!(standing.costliest_term, Some(ReferenceTerm::Angle));
    assert!(!standing.is_reference());
    assert!(choice.standing(1).unwrap().is_reference());
    // With no angle weight the angle has no say, and the sharper view wins.
    let options = ReferenceViewOptions {
        angle_weight: 0.0,
        ..ReferenceViewOptions::default()
    };
    let choice = choose_reference_view(&[oblique, facing], &options);
    assert_eq!(choice.reference, Some(0));
    assert_eq!(cost_of(&choice, 0).angle, 1.0);
}

#[test]
fn the_angle_factor_is_one_facing_the_patch_and_grows_with_tan_theta() {
    let options = ReferenceViewOptions::default();
    let k = options.angle_weight;
    assert_eq!(options.angle_factor(0.0), 1.0);
    // A negative angle, which only a rounding gives, counts as facing.
    assert_eq!(options.angle_factor(-1.0), 1.0);
    assert!(close(options.angle_factor(45.0), 1.0 + k));
    for angle in [10.0, 30.0, 60.0, 80.0, 89.9] {
        let factor = options.angle_factor(angle);
        assert!(
            close(factor - 1.0, k * f64::to_radians(angle).tan()),
            "{angle}"
        );
        assert!(factor > options.angle_factor(angle - 5.0), "{angle}");
    }
    // The factor multiplies the radius: the same pair of angles trades the
    // same share of radius on a sharp track as on a blurry one.
    for scale in [0.5, 1.0, 2.5] {
        let mut oblique = good(0.9, 0.97 * scale);
        oblique.viewing_angle_deg = Some(40.0);
        let mut facing = good(0.9, scale);
        facing.viewing_angle_deg = Some(10.0);
        let choice = choose(&[oblique, facing]);
        assert_eq!(choice.reference, Some(1), "scale {scale}");
        let ratio = choice.standing(0).unwrap().cost_ratio().unwrap();
        let expected = 0.97 * options.angle_factor(40.0) / options.angle_factor(10.0);
        assert!(close(ratio, expected), "scale {scale}");
    }
}

#[test]
fn a_view_at_19_degrees_beats_a_slightly_sharper_one_at_29() {
    // seoul_bull point 10: the views of the hand pick (19°, radius 1.584
    // grid px) and of the next sharpest (29°, 1.572), the rest of their
    // readings within every band. The angle added in quadrature to the
    // radius gave them 1.589 and 1.585 and picked the second.
    let mut hand_pick = good(0.911, 1.584);
    hand_pick.viewing_angle_deg = Some(18.9);
    let mut oblique = good(0.906, 1.572);
    oblique.viewing_angle_deg = Some(29.3);
    let choice = choose(&[oblique, hand_pick]);
    assert_eq!(choice.reference, Some(1));
    let standing = choice.standing(0).unwrap();
    assert_eq!(standing.costliest_term, Some(ReferenceTerm::Angle));
    assert!(standing.cost_ratio().unwrap() > 1.03);
}

#[test]
fn each_penalty_is_one_inside_its_band_and_grows_at_its_scale_past_it() {
    let options = ReferenceViewOptions::default();
    // The agreement shortfall is read against the best view, the first.
    let mut views = vec![good(0.9, 0.5); 9];
    views[1].pair_zncc = Some(0.9 - options.agreement_band);
    views[2].pair_zncc = Some(0.9 - options.agreement_band - options.agreement_scale);
    views[3].cell_deficit = Some(options.cell_deficit_band);
    views[4].cell_deficit = Some(options.cell_deficit_band + options.cell_deficit_scale);
    views[5].coverage = Some(options.coverage_band);
    views[6].coverage = Some(options.coverage_band - 2.0 * options.coverage_scale);
    views[7].clipped_share = Some(options.clipped_band);
    views[8].clipped_share = Some(options.clipped_band + 0.5 * options.clipped_scale);
    let choice = choose(&views);
    let c = |i| cost_of(&choice, i);
    assert!(close(c(1).agreement, 1.0));
    assert!(close(c(2).agreement, 2.0));
    assert!(close(c(3).cells, 1.0));
    assert!(close(c(4).cells, 2.0));
    assert!(close(c(5).coverage, 1.0));
    assert!(close(c(6).coverage, 3.0));
    assert!(close(c(7).clipped, 1.0));
    assert!(close(c(8).clipped, 1.5));
    // Each factor multiplies the radius and the angle factor, and a view
    // inside every band costs those two alone.
    let base = c(0).radius * c(0).angle;
    assert!(close(c(0).cost, base));
    for (i, factor) in [(2, 2.0), (4, 2.0), (6, 3.0), (8, 1.5)] {
        assert!(close(c(i).cost, base * factor), "view {i}");
    }
    // The terms each view's cost turns on.
    let term = |i: usize| choice.standing(i).unwrap().costliest_term;
    assert_eq!(term(2), Some(ReferenceTerm::Agreement));
    assert_eq!(term(4), Some(ReferenceTerm::Cells));
    assert_eq!(term(6), Some(ReferenceTerm::Coverage));
    assert_eq!(term(8), Some(ReferenceTerm::Clipped));
    assert!(close(
        choice.standing(2).unwrap().cost_ratio().unwrap(),
        2.0
    ));
}

#[test]
fn the_coverage_floor_keeps_a_view_at_it_and_excludes_one_under_it() {
    let floor = ReferenceViewOptions::default().min_coverage;
    let mut at = good(0.9, 0.3);
    at.coverage = Some(floor);
    let mut under = good(0.9, 0.1);
    under.coverage = Some(floor - 1e-9);
    let mut no_tile = good(0.9, 0.1);
    no_tile.coverage = None;
    let choice = choose(&[under, at, no_tile]);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(choice.costs[0], Err(ReferenceExclusion::Coverage));
    assert_eq!(choice.costs[2], Err(ReferenceExclusion::NoTile));
    // At the floor the coverage penalty is 4.
    assert!(close(cost_of(&choice, 1).coverage, 4.0));
    let standing = choice.standing(0).unwrap();
    assert_eq!(standing.costliest_term, None);
    assert_eq!(standing.cost_ratio(), None);
    // With no eligible view the rule picks nothing.
    let choice = choose(&[under, no_tile]);
    assert_eq!(choice.reference, None);
    assert_eq!(choice.standing(0).unwrap().pick_cost, None);
    let choice = choose(&[]);
    assert_eq!(choice.reference, None);
    assert!(choice.costs.is_empty());
}

#[test]
fn a_track_seen_only_from_behind_has_no_eligible_view() {
    let mut grazing = good(0.9, 0.5);
    grazing.viewing_angle_deg = Some(100.0);
    let mut facing_back = good(0.9, 0.5);
    facing_back.viewing_angle_deg = Some(150.0);
    let choice = choose(&[grazing, facing_back]);
    assert_eq!(choice.reference, None);
    assert_eq!(
        choice.costs,
        vec![
            Err(ReferenceExclusion::BackFacing),
            Err(ReferenceExclusion::BackFacing)
        ]
    );
    let standing = choice.standing(1).unwrap();
    assert_eq!(standing.pick_cost, None);
    assert_eq!(standing.costliest_term, None);
}

#[test]
fn with_views_on_both_sides_the_pick_is_made_among_the_front_facing_ones() {
    // The view behind the patch is the sharpest and agrees best, and is still
    // left out; a view with no angle stays in.
    let mut behind = good(0.99, 0.1);
    behind.viewing_angle_deg = Some(120.0);
    let facing = good(0.9, 0.5);
    let mut unread = good(0.9, 0.6);
    unread.viewing_angle_deg = None;
    let choice = choose(&[behind, facing, unread]);
    assert_eq!(choice.costs[0], Err(ReferenceExclusion::BackFacing));
    assert!(choice.costs[2].is_ok());
    assert_eq!(choice.reference, Some(1));
    // The best pair ZNCC is read among the views the pick is made from.
    assert!(close(cost_of(&choice, 1).agreement, 1.0));
}

#[test]
fn a_view_at_exactly_ninety_degrees_is_excluded() {
    let mut edge_on = good(0.9, 0.1);
    edge_on.viewing_angle_deg = Some(90.0);
    let mut grazing = good(0.9, 0.9);
    grazing.viewing_angle_deg = Some(85.0);
    let choice = choose(&[edge_on, grazing]);
    assert_eq!(choice.costs[0], Err(ReferenceExclusion::BackFacing));
    assert_eq!(choice.reference, Some(1));
    assert!(cost_of(&choice, 1).cost.is_finite());
    // Where every view is edge on, none is eligible.
    let choice = choose(&[edge_on, edge_on]);
    assert_eq!(choice.reference, None);
    // A view just short of 90° is eligible, with a large but finite cost.
    let mut nearly = edge_on;
    nearly.viewing_angle_deg = Some(89.9);
    let choice = choose(&[nearly]);
    assert_eq!(choice.reference, Some(0));
    assert!(cost_of(&choice, 0).cost.is_finite());
}

#[test]
fn a_missing_radius_takes_the_largest_one_read_or_the_unread_radius() {
    let mut unread = good(0.9, 0.1);
    unread.semi_major = None;
    let choice = choose(&[unread, good(0.9, 0.4), good(0.9, 0.7)]);
    let cost = cost_of(&choice, 0);
    assert!(!cost.radius_read);
    assert!(close(cost.radius, 0.7));
    assert!(cost_of(&choice, 1).radius_read);
    assert_eq!(choice.reference, Some(1));
    // With no radius anywhere every view takes the unread radius, and the
    // angle and the penalties pick.
    let mut a = unread;
    a.viewing_angle_deg = Some(40.0);
    let b = unread;
    let choice = choose(&[a, b]);
    let unread_radius = ReferenceViewOptions::default().unread_radius;
    assert!(close(cost_of(&choice, 0).radius, unread_radius));
    assert!(close(cost_of(&choice, 1).radius, unread_radius));
    assert_eq!(choice.reference, Some(1));
}

#[test]
fn a_missing_pair_angle_clipped_share_or_cell_deficit_is_filled_in() {
    let options = ReferenceViewOptions::default();
    // No pair ZNCC: the largest shortfall among the views that have one.
    let mut no_pair = good(0.9, 0.3);
    no_pair.pair_zncc = None;
    let low = good(0.9 - options.agreement_band - options.agreement_scale, 0.5);
    let choice = choose(&[no_pair, good(0.9, 0.5), low]);
    assert!(close(cost_of(&choice, 0).agreement, 2.0));
    // Where no view has one, every agreement penalty is 1.
    let choice = choose(&[no_pair, no_pair]);
    assert!(close(cost_of(&choice, 0).agreement, 1.0));

    // No angle: the largest angle factor among the views that have one.
    let mut no_angle = good(0.9, 0.3);
    no_angle.viewing_angle_deg = None;
    let mut oblique = good(0.9, 0.5);
    oblique.viewing_angle_deg = Some(60.0);
    let choice = choose(&[no_angle, good(0.9, 0.5), oblique]);
    assert!(close(cost_of(&choice, 0).angle, cost_of(&choice, 2).angle));
    // Where no view has one, the angle factor is 1.
    let choice = choose(&[no_angle, no_angle]);
    assert_eq!(cost_of(&choice, 0).angle, 1.0);
    // Where every view with an angle is back-facing, the views with none are
    // eligible and take an angle factor of 1.
    let mut behind = good(0.9, 0.1);
    behind.viewing_angle_deg = Some(120.0);
    let choice = choose(&[behind, no_angle]);
    assert_eq!(choice.costs[0], Err(ReferenceExclusion::BackFacing));
    assert_eq!(choice.reference, Some(1));
    assert_eq!(cost_of(&choice, 1).angle, 1.0);

    // No clipped share or cell deficit: a factor of 1.
    let mut neither = good(0.9, 0.3);
    neither.clipped_share = None;
    neither.cell_deficit = None;
    let choice = choose(&[neither]);
    assert_eq!(choice.reference, Some(0));
    assert!(close(cost_of(&choice, 0).clipped, 1.0));
    assert!(close(cost_of(&choice, 0).cells, 1.0));
}

#[test]
fn a_radius_at_the_cap_loses_to_one_just_under_it() {
    let options = ReferenceViewOptions::default();
    assert_eq!(options.capped_radius_factor, 1.5);
    // Every other factor equal: the reading at the cap is a lower bound, and
    // costs 1.5 times its radius.
    let capped = good(0.9, CAP);
    let under = good(0.9, 2.95);
    let choice = choose(&[capped, under]);
    assert_eq!(choice.reference, Some(1));
    assert!(close(cost_of(&choice, 0).capped_radius, 1.5));
    assert!(close(cost_of(&choice, 1).capped_radius, 1.0));
    let standing = choice.standing(0).unwrap();
    assert_eq!(standing.costliest_term, Some(ReferenceTerm::CappedRadius));
    assert!(close(standing.cost_ratio().unwrap(), 1.5 * CAP / 2.95));
    // A capped view facing the patch more squarely beat the uncapped one
    // with no step; the step decides it, and is the costliest term.
    let mut oblique = good(0.9, 2.95);
    oblique.viewing_angle_deg = Some(25.0);
    let no_step = ReferenceViewOptions {
        capped_radius_factor: 1.0,
        ..options
    };
    assert_eq!(
        choose_reference_view(&[capped, oblique], &no_step).reference,
        Some(0)
    );
    let choice = choose(&[capped, oblique]);
    assert_eq!(choice.reference, Some(1));
    assert_eq!(
        choice.standing(0).unwrap().costliest_term,
        Some(ReferenceTerm::CappedRadius)
    );
    // A radius under the cap, and one read with no cap, take no step.
    let mut uncapped = good(0.9, CAP);
    uncapped.max_radius = None;
    let choice = choose(&[uncapped, good(0.9, 2.999)]);
    assert!(close(cost_of(&choice, 0).capped_radius, 1.0));
    assert!(close(cost_of(&choice, 1).capped_radius, 1.0));
}

#[test]
fn a_capped_view_still_beats_an_outlier_whose_penalties_exceed_the_step() {
    // The uncapped view is sharper but differs from the others in a ninth:
    // its cell penalty, 3, outweighs the capped view's step, 1.5.
    let capped = good(0.9, CAP);
    let mut outlier = good(0.9, 2.0);
    outlier.cell_deficit = Some(0.9);
    let choice = choose(&[capped, outlier]);
    assert_eq!(choice.reference, Some(0));
    assert!(close(cost_of(&choice, 1).cells, 3.0));
    assert!(close(
        cost_of(&choice, 0).cost,
        1.5 * CAP * cost_of(&choice, 0).angle
    ));
    assert_eq!(
        choice.standing(1).unwrap().costliest_term,
        Some(ReferenceTerm::Cells)
    );
}

#[test]
fn a_track_whose_every_view_is_capped_picks_as_with_no_step() {
    let no_step = ReferenceViewOptions {
        capped_radius_factor: 1.0,
        ..ReferenceViewOptions::default()
    };
    let mut views = vec![good(0.9, CAP); 4];
    views[0].viewing_angle_deg = Some(50.0);
    views[1].viewing_angle_deg = Some(10.0);
    views[2].pair_zncc = Some(0.6);
    views[3].clipped_share = Some(0.2);
    let stepped = choose(&views);
    let plain = choose_reference_view(&views, &no_step);
    assert_eq!(stepped.reference, Some(1));
    assert_eq!(stepped.reference, plain.reference);
    for i in 0..views.len() {
        let (a, b) = (stepped.standing(i).unwrap(), plain.standing(i).unwrap());
        assert!(
            close(a.cost_ratio().unwrap(), b.cost_ratio().unwrap()),
            "{i}"
        );
        assert_eq!(a.costliest_term, b.costliest_term, "{i}");
        assert!(close(cost_of(&stepped, i).capped_radius, 1.5));
    }
}

#[test]
fn a_missing_radius_takes_the_step_with_the_largest_radius_where_it_is_capped() {
    let mut unread = good(0.9, 0.1);
    unread.semi_major = None;
    // The largest radius read is at the cap, a lower bound: the view with no
    // radius takes it with the step, so it ranks no better than that view.
    let choice = choose(&[unread, good(0.9, 1.0), good(0.9, CAP)]);
    let cost = cost_of(&choice, 0);
    assert!(!cost.radius_read);
    assert!(close(cost.radius, CAP));
    assert!(close(cost.capped_radius, 1.5));
    assert!(close(cost.cost, cost_of(&choice, 2).cost));
    // Under the cap it takes none.
    let choice = choose(&[unread, good(0.9, 1.0), good(0.9, 2.5)]);
    assert!(close(cost_of(&choice, 0).capped_radius, 1.0));
    // Where no view has a radius, the unread radius takes no step.
    let mut other = unread;
    other.viewing_angle_deg = Some(40.0);
    let choice = choose(&[unread, other]);
    assert!(close(cost_of(&choice, 0).capped_radius, 1.0));
    assert!(close(cost_of(&choice, 1).capped_radius, 1.0));
}

#[test]
fn a_tie_on_the_cost_goes_to_the_smaller_semi_minor_then_the_earlier_view() {
    let mut a = good(0.9, 0.4);
    a.semi_minor = Some(0.3);
    let mut b = good(0.9, 0.4);
    b.semi_minor = Some(0.2);
    let mut none = good(0.9, 0.4);
    none.semi_minor = None;
    assert_eq!(choose(&[a, b]).reference, Some(1));
    assert_eq!(choose(&[b, b]).reference, Some(0));
    assert_eq!(choose(&[none, a]).reference, Some(1));
}

#[test]
fn the_costliest_terms_parts_sum_to_the_cost_over_the_picks() {
    let pick = good(0.9, 0.4);
    let mut other = good(0.6, CAP);
    other.viewing_angle_deg = Some(50.0);
    other.cell_deficit = Some(0.45);
    other.coverage = Some(0.97);
    other.clipped_share = Some(0.07);
    let choice = choose(&[pick, other]);
    assert_eq!(choice.reference, Some(0));
    let (own, picked) = (cost_of(&choice, 1), cost_of(&choice, 0));
    let parts: Vec<(ReferenceTerm, f64)> = own
        .ln_parts()
        .iter()
        .zip(picked.ln_parts())
        .map(|(&(term, a), (_, b))| (term, a - b))
        .collect();
    let sum: f64 = parts.iter().map(|(_, p)| p).sum();
    let standing = choice.standing(1).unwrap();
    assert!(close(sum, standing.cost_ratio().unwrap().ln()));
    let largest = parts
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|&(term, _)| term);
    assert_eq!(standing.costliest_term, largest);
    // Each part is the log of the view's factor over the pick's: the pick's
    // penalties and step are 1, and its angle factor is that of 20°.
    assert!(close(own.capped_radius, 1.5));
    assert!(close(picked.capped_radius, 1.0));
    for (term, part) in &parts {
        let ratio = match term {
            ReferenceTerm::Sharpness => own.radius / picked.radius,
            ReferenceTerm::CappedRadius => own.capped_radius / picked.capped_radius,
            ReferenceTerm::Angle => own.angle / picked.angle,
            ReferenceTerm::Agreement => own.agreement,
            ReferenceTerm::Cells => own.cells,
            ReferenceTerm::Coverage => own.coverage,
            ReferenceTerm::Clipped => own.clipped,
        };
        assert!(close(*part, ratio.ln()), "{term}");
    }
    assert!(picked.angle > 1.0);
    assert!(close(
        picked.angle,
        ReferenceViewOptions::default().angle_factor(20.0)
    ));
}
