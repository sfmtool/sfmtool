// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use nalgebra::Vector3;
use ndarray::{Array2, Array4};
use serde_json::Value;

use super::atlas::AtlasLayout;
use super::*;
use crate::camera::CameraModel;
use crate::CameraIntrinsics;

/// A tile whose texel `(x, y)` is `[k, x, y]`, so every texel says where it
/// came from.
fn labelled_tile(k: usize, size: usize) -> Vec<u8> {
    (0..size)
        .flat_map(|y| (0..size).flat_map(move |x| [k as u8, x as u8, y as u8]))
        .collect()
}

fn texel(page: &atlas::RgbPage, x: usize, y: usize) -> [u8; 3] {
    let i = (y * page.width + x) * 3;
    [page.pixels[i], page.pixels[i + 1], page.pixels[i + 2]]
}

#[test]
fn tiles_land_in_their_cells_ringed_by_a_copy_of_their_edge() {
    let size = 4;
    let layout = AtlasLayout::new(5, size, 4096).unwrap();
    // Five tiles fit one page: three columns, two rows.
    assert_eq!((layout.cols, layout.rows_per_page, layout.tile), (3, 2, 6));
    assert_eq!(layout.page_count(), 1);
    assert_eq!(layout.page_dims(0), (18, 12));
    let pages = layout.pack(|k| labelled_tile(k, size));
    let page = &pages[0];
    for k in 0..5 {
        let cell = layout.place(k);
        assert_eq!(cell.page, 0);
        let (x0, y0) = layout.origin(cell.cell);
        assert_eq!((x0, y0), ((k % 3) * 6, (k / 3) * 6));
        // The tile itself, one texel in.
        for y in 0..size {
            for x in 0..size {
                assert_eq!(
                    texel(page, x0 + 1 + x, y0 + 1 + y),
                    [k as u8, x as u8, y as u8]
                );
            }
        }
        // The border repeats the nearest edge texel, corners included.
        assert_eq!(texel(page, x0, y0), [k as u8, 0, 0]);
        assert_eq!(texel(page, x0 + 5, y0), [k as u8, 3, 0]);
        assert_eq!(texel(page, x0, y0 + 5), [k as u8, 0, 3]);
        assert_eq!(texel(page, x0 + 5, y0 + 5), [k as u8, 3, 3]);
        assert_eq!(texel(page, x0 + 2, y0), [k as u8, 1, 0]);
        assert_eq!(texel(page, x0, y0 + 3), [k as u8, 0, 2]);
    }
    // The cell no tile uses stays black.
    assert_eq!(texel(page, 13, 7), [0, 0, 0]);
}

#[test]
fn a_set_larger_than_a_page_overflows_onto_more_pages() {
    // A 32-texel page holds five 6-texel tiles a side, 25 a page.
    let layout = AtlasLayout::new(60, 4, 32).unwrap();
    assert_eq!((layout.cols, layout.rows_per_page), (5, 5));
    assert_eq!(layout.per_page(), 25);
    assert_eq!(layout.page_count(), 3);
    assert_eq!(layout.page_dims(0), (30, 30));
    assert_eq!(layout.page_dims(1), (30, 30));
    // The last page holds ten, two rows.
    assert_eq!(layout.page_dims(2), (30, 12));
    assert_eq!(layout.place(24), atlas::AtlasCell { page: 0, cell: 24 });
    assert_eq!(layout.place(25), atlas::AtlasCell { page: 1, cell: 0 });
    assert_eq!(layout.place(59), atlas::AtlasCell { page: 2, cell: 9 });
    let pages = layout.pack(|k| labelled_tile(k, 4));
    assert_eq!(pages.len(), 3);
    let (x0, y0) = layout.origin(9);
    assert_eq!(texel(&pages[2], x0 + 1, y0 + 1), [59, 0, 0]);
    // A tile larger than the page does not lay out.
    assert!(AtlasLayout::new(1, 40, 32).is_none());
    assert!(AtlasLayout::new(0, 4, 32).is_none());
}

#[test]
fn a_page_encodes_as_a_jpeg_of_its_size() {
    let layout = AtlasLayout::new(3, 8, 4096).unwrap();
    let pages = layout.pack(|k| labelled_tile(k, 8));
    let bytes = pages[0].to_jpeg(85).unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap();
    assert_eq!(
        (decoded.width() as usize, decoded.height() as usize),
        layout.page_dims(0)
    );
}

fn image_looking_down_minus_z() -> crate::SfmrImage {
    // Identity pose: the camera sits at the origin looking down world -Z.
    crate::SfmrImage {
        name: "a.jpg".into(),
        camera_index: 0,
        quaternion_wxyz: nalgebra::UnitQuaternion::identity(),
        translation_xyz: Vector3::zeros(),
    }
}

fn grid_point(grid: &LensGrid, i: usize) -> Vector3<f64> {
    Vector3::new(
        grid.positions[i * 3],
        grid.positions[i * 3 + 1],
        grid.positions[i * 3 + 2],
    )
}

#[test]
fn a_pinhole_lens_grid_is_the_flat_far_plane() {
    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 100.0,
            focal_length_y: 100.0,
            principal_point_x: 50.0,
            principal_point_y: 40.0,
        },
        width: 100,
        height: 80,
    };
    let grid = lens_grid(
        &image_looking_down_minus_z(),
        &camera,
        &Point3::origin(),
        2.0,
    );
    assert_eq!(grid.size, 2);
    for i in 0..4 {
        // Every corner on the plane two units out.
        assert!((grid_point(&grid, i).z + 2.0).abs() < 1e-12);
    }
    // Top-left is up and to the left; x and y reach half the image over f.
    let top_left = grid_point(&grid, 0);
    assert!((top_left.x + 1.0).abs() < 1e-12 && (top_left.y - 0.8).abs() < 1e-12);
}

#[test]
fn a_fisheye_lens_grid_bends_onto_the_frustum_sphere() {
    let camera = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 30.0,
            principal_point_x: 50.0,
            principal_point_y: 50.0,
        },
        width: 100,
        height: 100,
    };
    let centre = Point3::new(1.0, 2.0, 3.0);
    let grid = lens_grid(&image_looking_down_minus_z(), &camera, &centre, 2.0);
    assert_eq!(grid.size, LENS_GRID_SIZE);
    let apex = Vector3::new(-1.0, -2.0, -3.0);
    let n = LENS_GRID_SIZE * LENS_GRID_SIZE;
    for i in 0..n {
        // Every vertex is one frustum length from the camera, relative to the
        // scene centre.
        assert!(((grid_point(&grid, i) - apex).norm() - 2.0).abs() < 1e-9);
    }
    // Not a plane: the middle of the grid is farther out along the axis than
    // a corner, which is off to the side.
    let depth = |i: usize| -(grid_point(&grid, i) - apex).z;
    assert!(depth(n / 2) > depth(0) + 0.5);
}

/// The demo reconstruction moved far from the origin, with patch frames and
/// bitmaps on every point but the last, and thumbnails.
fn far_demo(points: usize) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(points);
    let shift = Vector3::new(4.0e6, -3.0e6, 250.0);
    for point in &mut recon.point_set.points {
        point.position += shift;
    }
    for image in &mut recon.image_table.images {
        let centre = image.camera_center().coords + shift;
        image.translation_xyz = -(image.quaternion_wxyz * centre);
    }
    let mut u = Array2::<f32>::zeros((points, 3));
    let mut v = Array2::<f32>::zeros((points, 3));
    let mut bitmaps = Array4::<u8>::zeros((points, 8, 8, 4));
    for p in 0..points - 1 {
        u[[p, 0]] = 0.01;
        v[[p, 1]] = 0.01;
        bitmaps
            .index_axis_mut(ndarray::Axis(0), p)
            .fill(10 + (p % 200) as u8);
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    let n = recon.image_table.images.len();
    recon.image_table.thumbnails_y_x_rgb = Some(Arc::new(Array4::from_elem(
        (n, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        90,
    )));
    recon
}

fn scene_of(export: &WebExport) -> Value {
    assert_eq!(export.files[0].name, "scene.json");
    serde_json::from_slice(&export.files[0].bytes).unwrap()
}

/// The `name` array of a byte block, as f32 or u32 values.
fn array_bytes<'a>(scene: &Value, block: &'a [u8], arrays: &str, name: &str) -> (&'a [u8], usize) {
    let entry = &scene_path(scene, arrays)[name];
    let offset = entry["offset"].as_u64().unwrap() as usize;
    let count = entry["count"].as_u64().unwrap() as usize;
    let components = entry["components"].as_u64().unwrap() as usize;
    let width = match entry["type"].as_str().unwrap() {
        "f32" | "u32" => 4,
        "u16" => 2,
        _ => 1,
    };
    assert_eq!(offset % 4, 0, "{name} starts on a 4-byte boundary");
    (&block[offset..offset + count * components * width], count)
}

fn scene_path<'a>(scene: &'a Value, path: &str) -> &'a Value {
    path.split('.').fold(scene, |v, key| &v[key])
}

fn decode(scene: &Value, key: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(scene[key].as_str().unwrap())
        .unwrap()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn point_arrays_sit_at_their_offsets_and_positions_shift_back_to_the_file() {
    let recon = far_demo(50);
    let export = build_web_export(&recon, &WebExportOptions::default(), &Progress::none()).unwrap();
    let scene = scene_of(&export);
    let block = decode(&scene, "points_b64");
    let centre: Vec<f64> = scene["center"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(centre[0] > 3.0e6, "the centre carries the offset, in f64");

    let (bytes, count) = array_bytes(&scene, &block, "points.arrays", "position");
    assert_eq!(count, 50);
    let positions = f32s(bytes);
    for (p, point) in recon.point_set.points.iter().enumerate() {
        for axis in 0..3 {
            let back = positions[p * 3 + axis] as f64 + centre[axis];
            // Relative to the centre, a few units fit f32 to about 1e-6.
            assert!(
                (back - point.position[axis]).abs() < 1e-5,
                "point {p} axis {axis}"
            );
        }
    }

    // Every point but the last has a patch; the last reads "none".
    let (bytes, _) = array_bytes(&scene, &block, "points.arrays", "patch_cell");
    let cells: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    assert_eq!(cells[0], 0);
    assert_eq!(cells[48], 48);
    assert_eq!(cells[49], u32::MAX);
    let (bytes, _) = array_bytes(&scene, &block, "points.arrays", "color_w");
    assert!(
        bytes.as_chunks::<4>().0.iter().all(|c| c[3] == 1),
        "every point is finite"
    );
    let (bytes, _) = array_bytes(&scene, &block, "points.arrays", "observations");
    let first = u16::from_le_bytes([bytes[0], bytes[1]]);
    assert_eq!(first as u32, recon.point_set.observation_counts[0]);
    assert_eq!(scene["points"]["with_patch"], 49);
    assert!(scene["points"]["arrays"].get("source_index").is_none());

    // One file per atlas page, named as scene.json names them.
    let names: Vec<&str> = export.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["scene.json", "patches-0.jpg", "thumbs-0.jpg"]);
    assert_eq!(
        scene["atlases"]["patches"]["pages"][0]["file"],
        "patches-0.jpg"
    );
    assert_eq!(scene["atlases"]["patches"]["size"], 8);
    assert_eq!(export.report.thumbnails.file, 8);
    assert_eq!(scene["cameras"][3]["thumb"], serde_json::json!([0, 3]));

    // The frustum grids decode to vertices around the camera centres.
    let grids = decode(&scene, "frustums_b64");
    let (bytes, count) = array_bytes(&scene, &grids, "frustum_arrays", "vertex");
    assert_eq!(count, 8 * 4, "eight pinhole cameras, four corners each");
    let vertices = f32s(bytes);
    let camera = &scene["cameras"][0];
    let apex: Vec<f64> = camera["center"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let first = camera["grid"][0].as_u64().unwrap() as usize;
    let d = (0..3)
        .map(|a| (vertices[first * 3 + a] as f64 - apex[a]).powi(2))
        .sum::<f64>()
        .sqrt();
    let length = scene["frustum_length"].as_f64().unwrap();
    assert!(
        d >= length * 0.999,
        "a corner is at least the frustum length out"
    );
}

#[test]
fn max_points_keeps_the_most_observed_and_records_where_they_came_from() {
    let mut recon = far_demo(40);
    let counts = &recon.point_set.observation_counts;
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&p| (std::cmp::Reverse(counts[p]), p));
    let mut expected: Vec<u32> = order[..10].iter().map(|&p| p as u32).collect();
    expected.sort_unstable();
    recon.point_set.patch_bitmaps_y_x_rgba = None;
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    let options = WebExportOptions {
        max_points: Some(10),
        thumbnails: false,
        ..WebExportOptions::default()
    };
    let export = build_web_export(&recon, &options, &Progress::none()).unwrap();
    let scene = scene_of(&export);
    assert_eq!(scene["points"]["count"], 10);
    assert_eq!(scene["points"]["left_out"], 30);
    let block = decode(&scene, "points_b64");
    let (bytes, _) = array_bytes(&scene, &block, "points.arrays", "source_index");
    let indexes: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    assert_eq!(indexes, expected);
    assert_eq!(
        export.files.len(),
        1,
        "no patches, no thumbnails: scene.json alone"
    );
    assert!(scene["atlases"]["patches"].is_null());
    assert!(scene["cameras"][0].get("thumb").is_none());
}

#[test]
fn patch_size_resamples_the_tiles() {
    let recon = far_demo(10);
    let options = WebExportOptions {
        patch_size: Some(4),
        ..WebExportOptions::default()
    };
    let export = build_web_export(&recon, &options, &Progress::none()).unwrap();
    let scene = scene_of(&export);
    assert_eq!(scene["atlases"]["patches"]["size"], 4);
    assert_eq!(scene["atlases"]["patches"]["tile"], 6);
    assert_eq!(export.report.patch_size, 4);
}

#[test]
fn start_image_opens_through_that_camera_and_an_unknown_one_is_refused() {
    let recon = far_demo(20);
    let options = WebExportOptions {
        start_image: Some("image_002.jpg".into()),
        ..WebExportOptions::default()
    };
    let scene = scene_of(&build_web_export(&recon, &options, &Progress::none()).unwrap());
    assert_eq!(scene["view"]["start_image"], "image_002.jpg");
    assert_eq!(scene["view"]["eye"], scene["cameras"][2]["center"]);

    let options = WebExportOptions {
        start_image: Some("nope.jpg".into()),
        ..WebExportOptions::default()
    };
    assert!(matches!(
        build_web_export(&recon, &options, &Progress::none()),
        Err(WebExportError::UnknownStartImage(_))
    ));
}

#[test]
fn many_patches_spill_onto_several_pages_and_the_report_counts_their_memory() {
    let recon = far_demo(120);
    let options = WebExportOptions {
        max_page_size: 50,
        thumbnails: false,
        ..WebExportOptions::default()
    };
    let export = build_web_export(&recon, &options, &Progress::none()).unwrap();
    let scene = scene_of(&export);
    // A 50-texel page holds five 10-texel tiles a side, 25 a page: 119 patches
    // take five pages.
    let pages = scene["atlases"]["patches"]["pages"].as_array().unwrap();
    assert_eq!(pages.len(), 5);
    let block = decode(&scene, "points_b64");
    let (bytes, _) = array_bytes(&scene, &block, "points.arrays", "patch_cell");
    let cell = u32::from_le_bytes(bytes[30 * 4..31 * 4].try_into().unwrap());
    assert_eq!((cell >> 24, cell & 0x00ff_ffff), (1, 5));
    let decoded: u64 = pages
        .iter()
        .map(|p| p["width"].as_u64().unwrap() * p["height"].as_u64().unwrap() * 4)
        .sum();
    assert_eq!(export.report.decoded_atlas_bytes, decoded);
    assert!(export.report.warnings.is_empty());
}
