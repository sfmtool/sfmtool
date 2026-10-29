// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-image radial B-spline correction and rolling-shutter rates of
//! version 14: the two optional `images/` columns, the names
//! `"radial_bspline"` and `"rolling_shutter"` in the cameras'
//! `per_image_adjustments` key, the cameras' `rolling_shutter_scanlines` key,
//! the rules a reader, a writer and a verifier hold them to, and the refusal
//! of the two corrections at the file boundary.

use super::{
    archive_names, images_digest, make_embedded_test_data, make_test_data, read_json_entry_of,
    restamp_hashes, rewrite_entries,
};
use crate::read::read_sfmr_file;
use crate::write::write_sfmr_file;
use crate::*;
use ndarray::Array2;

const RADIAL_ENTRY: &str = "images/radial_bspline_adjustments.3.2.float64.zst";
const ROLLING_ENTRY: &str = "images/rolling_shutter_rates_xyz.3.3.float64.zst";

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Give `data` a second camera, holding image 2, that varies nothing, and
/// make camera 0 an `SFMTOOL_PINHOLE` camera that reads rows and varies the
/// radial B-spline correction and the rolling shutter, with values in images
/// 0 and 1 whose bits a round trip has to keep.
fn add_corrections(data: &mut SfmrData) {
    let mut second = data.cameras[0].clone();
    second.width = 640;
    second.height = 480;
    data.cameras.push(second);
    data.metadata.camera_count = 2;
    data.camera_indexes[2] = 1;
    data.cameras[0].model = "SFMTOOL_PINHOLE".into();
    data.cameras[0].per_image_adjustments = vec!["radial_bspline".into(), "rolling_shutter".into()];
    data.cameras[0].rolling_shutter_scanlines = Some(RollingShutterScanlines::Rows);
    data.radial_bspline_adjustments = Some(
        Array2::from_shape_vec(
            (3, 2),
            vec![1.25e-3, -7.0e-4, -1.0e-17, 1.0 / 3.0, 0.0, 0.0],
        )
        .unwrap(),
    );
    data.rolling_shutter_rates_xyz = Some(
        Array2::from_shape_vec(
            (3, 3),
            vec![
                2.0e-3,
                -3.1e-4,
                1.0 / 7.0,
                -0.0,
                5.0e-5,
                -1.0e-3,
                0.0,
                0.0,
                0.0,
            ],
        )
        .unwrap(),
    );
}

fn data_with_corrections() -> SfmrData {
    let mut data = make_test_data();
    add_corrections(&mut data);
    data
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn both_columns_and_both_camera_keys_round_trip_bit_for_bit() {
    let dir = temp_dir("sfmr_test_lens_corrections_round_trip");
    for (name, mut data) in [
        ("sift.sfmr", data_with_corrections()),
        ("embedded.sfmr", {
            let mut embedded = make_embedded_test_data();
            add_corrections(&mut embedded);
            embedded
        }),
    ] {
        let path = dir.join(name);
        let radial = data.radial_bspline_adjustments.clone().unwrap();
        let rolling = data.rolling_shutter_rates_xyz.clone().unwrap();
        write_sfmr_file(&path, &mut data, &WriteOptions::default()).unwrap();
        assert_eq!(data.metadata.version, 14);

        let names = archive_names(&path);
        assert!(names.contains(&RADIAL_ENTRY.to_string()), "{names:?}");
        assert!(names.contains(&ROLLING_ENTRY.to_string()), "{names:?}");

        let loaded = read_sfmr_file(&path).unwrap();
        assert_eq!(
            bits(loaded.radial_bspline_adjustments.as_ref().unwrap()),
            bits(&radial)
        );
        assert_eq!(
            bits(loaded.rolling_shutter_rates_xyz.as_ref().unwrap()),
            bits(&rolling)
        );
        assert_eq!(
            loaded.cameras[0].per_image_adjustments,
            vec!["radial_bspline".to_string(), "rolling_shutter".to_string()]
        );
        assert_eq!(
            loaded.cameras[0].rolling_shutter_scanlines,
            Some(RollingShutterScanlines::Rows)
        );
        assert!(loaded.cameras[1].per_image_adjustments.is_empty());
        assert_eq!(loaded.cameras[1].rolling_shutter_scanlines, None);
        let cameras = read_json_entry_of(&path, "cameras/metadata.json.zst");
        assert_eq!(cameras[0]["rolling_shutter_scanlines"], "ROWS");
        assert!(cameras[1].get("rolling_shutter_scanlines").is_none());

        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "{name}: {errors:?}");
        // Each column in its lexicographic slot: the format's rule, spelled
        // out independently of the writer and the verifier.
        assert_eq!(
            format!("{:032x}", images_digest(&path)),
            loaded.content_hash.images_xxh128,
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every one of the four names on one camera, with all four columns, sorts and
/// hashes in the format's order.
#[test]
fn all_four_per_image_columns_round_trip_together() {
    let dir = temp_dir("sfmr_test_lens_corrections_all_four");
    let path = dir.join("all.sfmr");
    let mut data = data_with_corrections();
    data.cameras[0].per_image_adjustments = PER_IMAGE_ADJUSTMENT_NAMES
        .iter()
        .map(|n| n.to_string())
        .collect();
    data.focal_adjustments = Some(ndarray::Array1::from_vec(vec![0.01, -0.02, 0.0]));
    data.principal_point_adjustments_xy = Some(Array2::from_elem((3, 2), 0.0));
    data.principal_point_adjustments_xy.as_mut().unwrap()[[1, 0]] = 0.004;
    write_sfmr_file(&path, &mut data, &WriteOptions::default()).unwrap();
    let loaded = read_sfmr_file(&path).unwrap();
    assert_eq!(loaded.focal_adjustments, data.focal_adjustments);
    assert_eq!(
        loaded.principal_point_adjustments_xy,
        data.principal_point_adjustments_xy
    );
    assert_eq!(
        loaded.radial_bspline_adjustments,
        data.radial_bspline_adjustments
    );
    assert_eq!(
        loaded.rolling_shutter_rates_xyz,
        data.rolling_shutter_rates_xyz
    );
    assert!(verify_sfmr(&path).unwrap().0);
    assert_eq!(
        format!("{:032x}", images_digest(&path)),
        loaded.content_hash.images_xxh128
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_scanlines_setting_without_rates_round_trips_through_the_guarded_paths() {
    let dir = temp_dir("sfmr_test_lens_corrections_scanlines_only");
    let path = dir.join("scanlines.sfmr");
    let mut data = make_test_data();
    data.cameras[0].rolling_shutter_scanlines = Some(RollingShutterScanlines::Columns);
    write_sfmr(&path, &mut data).unwrap();
    let names = archive_names(&path);
    assert!(
        !names
            .iter()
            .any(|n| n.contains("radial_bspline") || n.contains("rolling_shutter")),
        "{names:?}"
    );
    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(
        loaded.cameras[0].rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Columns)
    );
    assert!(loaded.radial_bspline_adjustments.is_none());
    assert!(loaded.rolling_shutter_rates_xyz.is_none());
    assert!(verify_sfmr(&path).unwrap().0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_without_corrections_writes_no_column_and_no_key() {
    let dir = temp_dir("sfmr_test_lens_corrections_absent");
    let path = dir.join("plain.sfmr");
    write_sfmr(&path, &mut make_test_data()).unwrap();
    let names = archive_names(&path);
    assert!(
        !names
            .iter()
            .any(|n| n.contains("radial_bspline") || n.contains("rolling_shutter")),
        "{names:?}"
    );
    let cameras = read_json_entry_of(&path, "cameras/metadata.json.zst");
    assert!(
        cameras[0].get("rolling_shutter_scanlines").is_none(),
        "{cameras}"
    );
    let loaded = read_sfmr(&path).unwrap();
    assert!(loaded.radial_bspline_adjustments.is_none());
    assert!(loaded.rolling_shutter_rates_xyz.is_none());
    assert_eq!(loaded.cameras[0].rolling_shutter_scanlines, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_images_and_cameras_hashes_change_with_the_columns_and_the_keys() {
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    let mut plain = make_test_data();
    let without = content_hash_of(&mut plain, &options).unwrap();

    // The scanlines setting alone changes the cameras JSON and nothing in the
    // images section.
    let mut scanlines = make_test_data();
    scanlines.cameras[0].rolling_shutter_scanlines = Some(RollingShutterScanlines::Rows);
    let with_scanlines = content_hash_of(&mut scanlines, &options).unwrap();
    assert_ne!(with_scanlines.cameras_xxh128, without.cameras_xxh128);
    assert_eq!(with_scanlines.images_xxh128, without.images_xxh128);
    assert_ne!(with_scanlines.content_xxh128, without.content_xxh128);
    let mut columns = make_test_data();
    columns.cameras[0].rolling_shutter_scanlines = Some(RollingShutterScanlines::Columns);
    let with_columns = content_hash_of(&mut columns, &options).unwrap();
    assert_ne!(with_columns.cameras_xxh128, with_scanlines.cameras_xxh128);

    // The two columns change the images section; a value in one changes the
    // images section and nothing else.
    let mut corrected = data_with_corrections();
    let with_values = content_hash_of(&mut corrected, &options).unwrap();
    let mut zero = data_with_corrections();
    zero.radial_bspline_adjustments = Some(Array2::zeros((3, 2)));
    zero.rolling_shutter_rates_xyz = Some(Array2::zeros((3, 3)));
    let with_zero = content_hash_of(&mut zero, &options).unwrap();
    assert_eq!(with_values.cameras_xxh128, with_zero.cameras_xxh128);
    assert_ne!(with_values.images_xxh128, with_zero.images_xxh128);
    assert_eq!(with_values.points3d_xxh128, with_zero.points3d_xxh128);
    assert_eq!(with_values.tracks_xxh128, with_zero.tracks_xxh128);

    let mut rates_changed = data_with_corrections();
    rates_changed.rolling_shutter_rates_xyz.as_mut().unwrap()[[1, 2]] = 2.0e-3;
    let with_other_rates = content_hash_of(&mut rates_changed, &options).unwrap();
    assert_ne!(with_other_rates.images_xxh128, with_values.images_xxh128);
}

#[test]
fn a_version_13_file_reads_with_no_corrections_and_saves_as_version_14_unchanged() {
    let dir = temp_dir("sfmr_test_lens_corrections_v13");
    let current = dir.join("current.sfmr");
    write_sfmr(&current, &mut make_test_data()).unwrap();
    let old = dir.join("v13.sfmr");
    rewrite_entries(&current, &old, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(13);
            serde_json::to_vec(&json).unwrap()
        })
    });
    restamp_hashes(&old, &dir);

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 13);
    assert!(loaded.radial_bspline_adjustments.is_none());
    assert!(loaded.rolling_shutter_rates_xyz.is_none());
    assert_eq!(loaded.cameras[0].rolling_shutter_scanlines, None);

    let saved = dir.join("saved.sfmr");
    write_sfmr(&saved, &mut loaded).unwrap();
    assert_eq!(read_sfmr_metadata(&saved).unwrap().version, 14);
    // Apart from the version, the cameras and images sections are what the
    // version 13 file stored.
    let old_hash = read_sfmr_content_hash(&old).unwrap();
    let new_hash = read_sfmr_content_hash(&saved).unwrap();
    assert_eq!(new_hash.cameras_xxh128, old_hash.cameras_xxh128);
    assert_eq!(new_hash.images_xxh128, old_hash.images_xxh128);
    assert_ne!(new_hash.metadata_xxh128, old_hash.metadata_xxh128);
    assert!(verify_sfmr(&saved).unwrap().0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_below_version_14_with_a_new_column_or_key_is_refused() {
    let to_version_13 = |name: &str, raw: &[u8]| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(13);
            serde_json::to_vec(&json).unwrap()
        })
    };
    let dir = temp_dir("sfmr_test_lens_corrections_old_version");
    let (err, errors) = read_and_verify_edited(&dir, to_version_13);
    assert_named(&err, &errors, "which version 14 introduced");

    // The scanlines setting alone is refused below version 14 as well.
    let good = dir.join("scanlines.sfmr");
    let mut data = make_test_data();
    data.cameras[0].rolling_shutter_scanlines = Some(RollingShutterScanlines::Rows);
    write_sfmr(&good, &mut data).unwrap();
    let bad = dir.join("scanlines_v13.sfmr");
    rewrite_entries(&good, &bad, to_version_13);
    let Err(err) = read_sfmr_file(&bad) else {
        panic!("the reader accepted a version 13 file with a scanlines setting");
    };
    let err = err.to_string();
    assert!(err.contains("which version 14 introduced"), "{err}");
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("which version 14 introduced")),
        "{errors:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Write `data_with_corrections` without the guard, rewrite it with `edit`,
/// and return the reader's error and the verifier's errors.
fn read_and_verify_edited(
    dir: &std::path::Path,
    edit: impl Fn(&str, &[u8]) -> Option<Vec<u8>>,
) -> (String, Vec<String>) {
    let good = dir.join("good.sfmr");
    write_sfmr_file(
        &good,
        &mut data_with_corrections(),
        &WriteOptions::default(),
    )
    .unwrap();
    let bad = dir.join("bad.sfmr");
    rewrite_entries(&good, &bad, edit);
    let err = match read_sfmr_file(&bad) {
        Ok(_) => panic!("the reader accepted the edited file"),
        Err(e) => e.to_string(),
    };
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    (err, errors)
}

/// The edit that replaces the cameras JSON with `edit` applied to it.
fn edit_cameras(edit: impl Fn(&mut serde_json::Value)) -> impl Fn(&str, &[u8]) -> Option<Vec<u8>> {
    move |name, raw| {
        (name == "cameras/metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            edit(&mut json);
            serde_json::to_vec(&json).unwrap()
        })
    }
}

/// The edit that replaces the rolling-shutter column's values.
fn edit_rates(values: [f64; 9]) -> impl Fn(&str, &[u8]) -> Option<Vec<u8>> {
    move |name, _| (name == ROLLING_ENTRY).then(|| bytemuck::cast_slice(&values).to_vec())
}

/// The edit that replaces the radial B-spline column's values.
fn edit_radial(values: [f64; 6]) -> impl Fn(&str, &[u8]) -> Option<Vec<u8>> {
    move |name, _| (name == RADIAL_ENTRY).then(|| bytemuck::cast_slice(&values).to_vec())
}

fn assert_named(err: &str, errors: &[String], needle: &str) {
    assert!(err.contains(needle), "reader: {err}");
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "verifier: {errors:?}"
    );
}

/// The writer's error for `data`, which it must refuse.
fn write_error(dir: &std::path::Path, mut data: SfmrData) -> String {
    write_sfmr_file(&dir.join("w.sfmr"), &mut data, &WriteOptions::default())
        .unwrap_err()
        .to_string()
}

#[test]
fn a_column_without_a_listing_camera_is_refused() {
    let dir = temp_dir("sfmr_test_lens_corrections_unlisted_column");
    for (names, needle) in [
        (
            serde_json::json!(["rolling_shutter"]),
            "no camera lists \"radial_bspline\"",
        ),
        (
            serde_json::json!(["radial_bspline"]),
            "no camera lists \"rolling_shutter\"",
        ),
    ] {
        let (err, errors) = read_and_verify_edited(
            &dir,
            edit_cameras(move |json| json[0]["per_image_adjustments"] = names.clone()),
        );
        assert_named(&err, &errors, needle);
    }
    let mut data = data_with_corrections();
    data.cameras[0].per_image_adjustments = vec!["radial_bspline".into()];
    let err = write_error(&dir, data);
    assert!(err.contains("no camera lists \"rolling_shutter\""), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_listing_camera_without_its_column_is_refused_by_the_writer() {
    let dir = temp_dir("sfmr_test_lens_corrections_missing_column");
    let mut data = data_with_corrections();
    data.radial_bspline_adjustments = None;
    let err = write_error(&dir, data);
    assert!(
        err.contains("images/radial_bspline_adjustments is absent"),
        "{err}"
    );
    let mut data = data_with_corrections();
    data.rolling_shutter_rates_xyz = None;
    let err = write_error(&dir, data);
    assert!(
        err.contains("images/rolling_shutter_rates_xyz is absent"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_non_zero_row_under_a_camera_that_does_not_list_it_is_refused() {
    let dir = temp_dir("sfmr_test_lens_corrections_unlisted_row");
    // Image 2 is camera 1's, which varies nothing.
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_rates([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4, 0.0]),
    );
    assert_named(&err, &errors, "row 2");
    assert!(err.contains("does not list"), "{err}");
    let (err, errors) =
        read_and_verify_edited(&dir, edit_radial([0.0, 0.0, 0.0, 0.0, 3.0e-3, 0.0]));
    assert_named(&err, &errors, "row 2");

    let mut data = data_with_corrections();
    data.radial_bspline_adjustments.as_mut().unwrap()[[2, 1]] = 0.5;
    let err = write_error(&dir, data);
    assert!(
        err.contains("radial_bspline_adjustments row 2") && err.contains("does not list"),
        "{err}"
    );
    // A negative zero is a zero.
    let mut data = data_with_corrections();
    data.rolling_shutter_rates_xyz.as_mut().unwrap()[[2, 0]] = -0.0;
    write_sfmr_file(&dir.join("z.sfmr"), &mut data, &WriteOptions::default()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_non_finite_value_is_refused() {
    let dir = temp_dir("sfmr_test_lens_corrections_non_finite");
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_rates([0.0, f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );
    assert_named(
        &err,
        &errors,
        "rolling_shutter_rates_xyz row 0 is not finite",
    );
    let (err, errors) =
        read_and_verify_edited(&dir, edit_radial([0.0, 0.0, f64::INFINITY, 0.0, 0.0, 0.0]));
    assert_named(
        &err,
        &errors,
        "radial_bspline_adjustments row 1 is not finite",
    );

    let mut data = data_with_corrections();
    data.rolling_shutter_rates_xyz.as_mut().unwrap()[[1, 1]] = f64::NEG_INFINITY;
    let err = write_error(&dir, data);
    assert!(err.contains("not finite"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_radial_bspline_correction_on_a_camera_that_is_not_sfmtool_pinhole_is_refused() {
    let dir = temp_dir("sfmr_test_lens_corrections_radial_model");
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_cameras(|json| json[0]["model"] = serde_json::json!("SFMTOOL_FISHEYE")),
    );
    assert_named(&err, &errors, "only an SFMTOOL_PINHOLE camera may list it");

    let mut data = data_with_corrections();
    data.cameras[0].model = "PINHOLE".into();
    let err = write_error(&dir, data);
    assert!(err.contains("on a PINHOLE camera"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rolling_shutter_rates_without_a_scanlines_setting_are_refused() {
    let dir = temp_dir("sfmr_test_lens_corrections_no_scanlines");
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_cameras(|json| {
            json[0]
                .as_object_mut()
                .unwrap()
                .remove("rolling_shutter_scanlines");
        }),
    );
    assert_named(&err, &errors, "has no rolling_shutter_scanlines");

    let mut data = data_with_corrections();
    data.cameras[0].rolling_shutter_scanlines = None;
    let err = write_error(&dir, data);
    assert!(err.contains("has no rolling_shutter_scanlines"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_scanlines_value_other_than_rows_or_columns_is_refused() {
    for value in ["Rows", "X", "UNKNOWN"] {
        let dir = temp_dir(&format!("sfmr_test_lens_corrections_scanlines_{value}"));
        let (err, errors) = read_and_verify_edited(
            &dir,
            edit_cameras(move |json| {
                json[1]["rolling_shutter_scanlines"] = serde_json::json!(value);
            }),
        );
        assert_named(&err, &errors, value);
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert_eq!(
        RollingShutterScanlines::parse("COLUMNS"),
        Ok(RollingShutterScanlines::Columns)
    );
    assert_eq!(RollingShutterScanlines::Rows.as_str(), "ROWS");
    assert!(RollingShutterScanlines::parse("UNKNOWN")
        .unwrap_err()
        .contains("\"UNKNOWN\""));
}

#[test]
fn the_new_names_parse_in_order() {
    assert_eq!(
        parse_per_image_adjustment_names(&["radial_bspline"]).unwrap(),
        [false, false, true, false]
    );
    assert_eq!(
        parse_per_image_adjustment_names(&["focal", "rolling_shutter"]).unwrap(),
        [true, false, false, true]
    );
    assert_eq!(
        parse_per_image_adjustment_names(&PER_IMAGE_ADJUSTMENT_NAMES).unwrap(),
        [true, true, true, true]
    );
    for out_of_order in [
        &["rolling_shutter", "radial_bspline"][..],
        &["radial_bspline", "principal_point"][..],
    ] {
        let err = parse_per_image_adjustment_names(out_of_order).unwrap_err();
        assert!(err.contains("out of the order"), "{err}");
    }

    let dir = temp_dir("sfmr_test_lens_corrections_key_order");
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_cameras(|json| {
            json[0]["per_image_adjustments"] =
                serde_json::json!(["rolling_shutter", "radial_bspline"]);
        }),
    );
    assert_named(&err, &errors, "out of the order");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Until this build applies the two corrections where an image is read
/// through its camera, the public reader and writer refuse a reconstruction
/// that varies either, even with every value zero, naming the camera and the
/// column. The format's own rules are checked first.
#[test]
fn the_reader_and_the_writer_refuse_the_corrections() {
    let dir = temp_dir("sfmr_test_lens_corrections_guard");
    for (tag, names, needle) in [
        (
            "radial",
            vec!["radial_bspline"],
            "images/radial_bspline_adjustments",
        ),
        (
            "rolling",
            vec!["rolling_shutter"],
            "images/rolling_shutter_rates_xyz",
        ),
    ] {
        let mut data = data_with_corrections();
        data.cameras[0].per_image_adjustments = names.iter().map(|n| n.to_string()).collect();
        if tag == "radial" {
            data.rolling_shutter_rates_xyz = None;
        } else {
            data.radial_bspline_adjustments = None;
            data.rolling_shutter_rates_xyz = Some(Array2::zeros((3, 3)));
        }
        let path = dir.join(format!("{tag}.sfmr"));
        write_sfmr_file(&path, &mut data, &WriteOptions::default()).unwrap();

        let Err(err) = read_sfmr(&path) else {
            panic!("{tag}: the reader accepted the corrections");
        };
        assert!(matches!(err, SfmrError::Unsupported(_)), "{tag}: {err}");
        let err = err.to_string();
        assert!(
            err.contains("camera 0") && err.contains(needle) && err.contains(names[0]),
            "{tag}: {err}"
        );

        let target = dir.join(format!("{tag}_written.sfmr"));
        let err = write_sfmr(&target, &mut data).unwrap_err();
        assert!(matches!(err, SfmrError::Unsupported(_)), "{tag}: {err}");
        assert!(err.to_string().contains(needle), "{tag}: {err}");
        assert!(!target.exists(), "{tag}: a refused write left a file");
    }
    // The reader refuses only a file that is otherwise valid: a broken rule
    // is reported as the format error it is.
    let path = dir.join("broken.sfmr");
    write_sfmr_file(
        &dir.join("good.sfmr"),
        &mut data_with_corrections(),
        &WriteOptions::default(),
    )
    .unwrap();
    rewrite_entries(
        &dir.join("good.sfmr"),
        &path,
        edit_cameras(|json| {
            json[0]
                .as_object_mut()
                .unwrap()
                .remove("rolling_shutter_scanlines");
        }),
    );
    let Err(err) = read_sfmr(&path) else {
        panic!("the reader accepted a broken file");
    };
    assert!(matches!(err, SfmrError::InvalidFormat(_)), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
