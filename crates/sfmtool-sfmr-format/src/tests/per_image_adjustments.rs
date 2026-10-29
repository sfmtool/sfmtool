// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-image intrinsic adjustments of version 13: the two optional
//! `images/` columns, the cameras' `per_image_adjustments` key, the rules a
//! reader, a writer and a verifier hold them to.

use super::{
    archive_names, images_digest, make_embedded_test_data, make_test_data, read_json_entry_of,
    restamp_hashes, rewrite_entries,
};
use crate::*;
use ndarray::{Array1, Array2};

const FOCAL_ENTRY: &str = "images/focal_adjustments.3.float64.zst";
const PRINCIPAL_POINT_ENTRY: &str = "images/principal_point_adjustments_xy.3.2.float64.zst";

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `make_test_data` with a second camera, holding image 2, that varies
/// nothing; camera 0 varies both parameters, and images 0 and 1 carry
/// adjustments whose bits a round trip has to keep.
fn data_with_adjustments() -> SfmrData {
    let mut data = make_test_data();
    let mut second = data.cameras[0].clone();
    second.width = 640;
    second.height = 480;
    data.cameras.push(second);
    data.metadata.camera_count = 2;
    data.camera_indexes[2] = 1;
    data.cameras[0].per_image_adjustments = vec!["focal".into(), "principal_point".into()];
    data.focal_adjustments = Some(Array1::from_vec(vec![0.013_1, -1.0e-17, 0.0]));
    data.principal_point_adjustments_xy = Some(
        Array2::from_shape_vec((3, 2), vec![0.012_3, -0.008_7, 1.0 / 3.0, -0.0, 0.0, 0.0]).unwrap(),
    );
    data
}

/// `data_with_adjustments` with every adjustment zero: the flags are kept,
/// the values are not adjustments.
fn data_with_zero_adjustments() -> SfmrData {
    let mut data = data_with_adjustments();
    data.focal_adjustments = Some(Array1::zeros(3));
    data.principal_point_adjustments_xy = Some(Array2::zeros((3, 2)));
    data
}

fn bits1(a: &Array1<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

fn bits2(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn both_columns_and_the_camera_key_round_trip_bit_for_bit() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_round_trip");
    for (name, mut data) in [
        ("sift.sfmr", data_with_adjustments()),
        ("embedded.sfmr", {
            let mut embedded = make_embedded_test_data();
            let adjusted = data_with_adjustments();
            embedded.cameras = adjusted.cameras;
            embedded.metadata.camera_count = 2;
            embedded.camera_indexes = adjusted.camera_indexes;
            embedded.focal_adjustments = adjusted.focal_adjustments;
            embedded.principal_point_adjustments_xy = adjusted.principal_point_adjustments_xy;
            embedded
        }),
    ] {
        let path = dir.join(name);
        let focal = data.focal_adjustments.clone().unwrap();
        let principal_point = data.principal_point_adjustments_xy.clone().unwrap();
        write_sfmr_with_options(&path, &mut data, &WriteOptions::default()).unwrap();
        assert_eq!(data.metadata.version, 14);

        let names = archive_names(&path);
        assert!(names.contains(&FOCAL_ENTRY.to_string()), "{names:?}");
        assert!(
            names.contains(&PRINCIPAL_POINT_ENTRY.to_string()),
            "{names:?}"
        );

        let loaded = read_sfmr(&path).unwrap();
        assert_eq!(
            bits1(loaded.focal_adjustments.as_ref().unwrap()),
            bits1(&focal)
        );
        assert_eq!(
            bits2(loaded.principal_point_adjustments_xy.as_ref().unwrap()),
            bits2(&principal_point)
        );
        assert_eq!(
            loaded.cameras[0].per_image_adjustments,
            vec!["focal".to_string(), "principal_point".to_string()]
        );
        assert!(loaded.cameras[1].per_image_adjustments.is_empty());

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

#[test]
fn a_file_without_adjustments_writes_no_column_and_no_key() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_absent");
    let path = dir.join("plain.sfmr");
    write_sfmr(&path, &mut make_test_data()).unwrap();

    let names = archive_names(&path);
    assert!(
        !names.iter().any(|n| n.contains("adjustments")),
        "{names:?}"
    );
    let cameras = read_json_entry_of(&path, "cameras/metadata.json.zst");
    assert!(
        cameras[0].get("per_image_adjustments").is_none(),
        "{cameras}"
    );

    let loaded = read_sfmr(&path).unwrap();
    assert!(loaded.focal_adjustments.is_none());
    assert!(loaded.principal_point_adjustments_xy.is_none());
    assert!(loaded.cameras[0].per_image_adjustments.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_images_and_cameras_hashes_change_with_the_columns_and_the_key() {
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    let mut plain = make_test_data();
    let mut second = plain.cameras[0].clone();
    second.width = 640;
    second.height = 480;
    plain.cameras.push(second);
    plain.metadata.camera_count = 2;
    plain.camera_indexes[2] = 1;
    let without = content_hash_of(&mut plain, &options).unwrap();

    // The key alone, on a camera, with all-zero columns: the cameras JSON and
    // the images section both change.
    let mut zero = data_with_zero_adjustments();
    let with_zero = content_hash_of(&mut zero, &options).unwrap();
    assert_ne!(with_zero.cameras_xxh128, without.cameras_xxh128);
    assert_ne!(with_zero.images_xxh128, without.images_xxh128);
    assert_ne!(with_zero.content_xxh128, without.content_xxh128);

    // A value in a column changes the images section and nothing else.
    let mut adjusted = data_with_adjustments();
    let with_values = content_hash_of(&mut adjusted, &options).unwrap();
    assert_eq!(with_values.cameras_xxh128, with_zero.cameras_xxh128);
    assert_ne!(with_values.images_xxh128, with_zero.images_xxh128);
    assert_eq!(with_values.points3d_xxh128, with_zero.points3d_xxh128);
    assert_eq!(with_values.tracks_xxh128, with_zero.tracks_xxh128);

    // Each column on its own.
    let mut focal_only = data_with_zero_adjustments();
    focal_only.cameras[0].per_image_adjustments = vec!["focal".into()];
    focal_only.principal_point_adjustments_xy = None;
    let focal_hash = content_hash_of(&mut focal_only, &options).unwrap();
    assert_ne!(focal_hash.images_xxh128, without.images_xxh128);
    assert_ne!(focal_hash.images_xxh128, with_zero.images_xxh128);
}

#[test]
fn a_version_12_file_reads_with_zero_adjustments_and_saves_as_the_current_version_without_them() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_v12");
    let current = dir.join("current.sfmr");
    write_sfmr(&current, &mut make_test_data()).unwrap();
    let old = dir.join("v12.sfmr");
    rewrite_entries(&current, &old, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(12);
            serde_json::to_vec(&json).unwrap()
        })
    });
    restamp_hashes(&old, &dir);

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 12);
    assert!(loaded.focal_adjustments.is_none());
    assert!(loaded.principal_point_adjustments_xy.is_none());
    assert!(loaded.cameras[0].per_image_adjustments.is_empty());

    let saved = dir.join("saved.sfmr");
    write_sfmr(&saved, &mut loaded).unwrap();
    assert_eq!(read_sfmr_metadata(&saved).unwrap().version, 14);
    assert!(!archive_names(&saved)
        .iter()
        .any(|n| n.contains("adjustments")));
    // Apart from the version, the cameras and images sections are what the
    // version 12 file stored.
    let old_hash = read_sfmr_content_hash(&old).unwrap();
    let new_hash = read_sfmr_content_hash(&saved).unwrap();
    assert_eq!(new_hash.cameras_xxh128, old_hash.cameras_xxh128);
    assert_eq!(new_hash.images_xxh128, old_hash.images_xxh128);
    assert!(verify_sfmr(&saved).unwrap().0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Write `data_with_adjustments`, rewrite it with `edit`,
/// and return the reader's error and the verifier's errors.
fn read_and_verify_edited(
    dir: &std::path::Path,
    edit: impl Fn(&str, &[u8]) -> Option<Vec<u8>>,
) -> (String, Vec<String>) {
    let good = dir.join("good.sfmr");
    write_sfmr_with_options(
        &good,
        &mut data_with_adjustments(),
        &WriteOptions::default(),
    )
    .unwrap();
    let bad = dir.join("bad.sfmr");
    rewrite_entries(&good, &bad, edit);
    let err = match read_sfmr(&bad) {
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

/// The edit that replaces the focal column's values.
fn edit_focal(values: [f64; 3]) -> impl Fn(&str, &[u8]) -> Option<Vec<u8>> {
    move |name, _| (name == FOCAL_ENTRY).then(|| bytemuck::cast_slice(&values).to_vec())
}

fn assert_named(err: &str, errors: &[String], needle: &str) {
    assert!(err.contains(needle), "reader: {err}");
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "verifier: {errors:?}"
    );
}

#[test]
fn a_column_without_a_listing_camera_is_refused() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_unlisted_column");
    let (err, errors) = read_and_verify_edited(
        &dir,
        edit_cameras(|json| {
            json[0]["per_image_adjustments"] = serde_json::json!(["principal_point"]);
        }),
    );
    assert_named(&err, &errors, "no camera lists \"focal\"");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_listing_camera_without_its_column_is_refused() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_missing_column");
    let good = dir.join("good.sfmr");
    let mut data = data_with_adjustments();
    write_sfmr_with_options(&good, &mut data, &WriteOptions::default()).unwrap();
    // Drop the focal entry from the archive.
    let bad = dir.join("bad.sfmr");
    {
        use std::io::{Read, Write};
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&good).unwrap()).unwrap();
        let names: Vec<String> = archive.file_names().map(|s| s.unwrap().into()).collect();
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&bad).unwrap());
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for name in names.iter().filter(|n| *n != FOCAL_ENTRY) {
            let mut bytes = Vec::new();
            archive
                .by_name(name)
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            zip.start_file(name, stored).unwrap();
            zip.write_all(&bytes).unwrap();
        }
        zip.finish().unwrap();
    }
    assert!(read_sfmr(&bad).is_err());
    // The verifier cannot hash an entry that is not there, and says so as an
    // error rather than a list of failures.
    assert!(verify_sfmr(&bad).is_err());

    // The writer refuses the same value.
    let mut data = data_with_adjustments();
    data.focal_adjustments = None;
    let err = write_sfmr_with_options(&dir.join("w.sfmr"), &mut data, &WriteOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("images/focal_adjustments is absent"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_non_zero_row_under_a_camera_that_does_not_list_it_is_refused() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_unlisted_row");
    // Image 2 is camera 1's, which varies nothing.
    let (err, errors) = read_and_verify_edited(&dir, edit_focal([0.01, 0.0, 0.002]));
    assert_named(&err, &errors, "row 2");
    assert!(err.contains("does not list"), "{err}");

    let mut data = data_with_adjustments();
    data.principal_point_adjustments_xy.as_mut().unwrap()[[2, 1]] = 0.5;
    let err = write_sfmr_with_options(&dir.join("w.sfmr"), &mut data, &WriteOptions::default())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("row 2") && err.contains("does not list"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_non_finite_value_or_a_focal_at_or_below_minus_one_is_refused() {
    for (tag, values, needle) in [
        ("nan", [f64::NAN, 0.0, 0.0], "not finite"),
        ("inf", [0.0, f64::INFINITY, 0.0], "not finite"),
        ("minus_one", [-1.0, 0.0, 0.0], "at or below -1"),
    ] {
        let dir = temp_dir(&format!("sfmr_test_per_image_adjustments_{tag}"));
        let (err, errors) = read_and_verify_edited(&dir, edit_focal(values));
        assert_named(&err, &errors, needle);

        let mut data = data_with_adjustments();
        data.focal_adjustments = Some(Array1::from_vec(values.to_vec()));
        let err = write_sfmr_with_options(&dir.join("w.sfmr"), &mut data, &WriteOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains(needle), "{tag}: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
    // Just above -1 is a focal, however short.
    let mut data = data_with_adjustments();
    data.focal_adjustments = Some(Array1::from_vec(vec![-0.999, 0.0, 0.0]));
    let dir = temp_dir("sfmr_test_per_image_adjustments_above_minus_one");
    write_sfmr_with_options(&dir.join("ok.sfmr"), &mut data, &WriteOptions::default()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_malformed_camera_key_is_refused() {
    for (tag, names, needle) in [
        ("unknown", serde_json::json!(["focal", "skew"]), "\"skew\""),
        ("twice", serde_json::json!(["focal", "focal"]), "twice"),
        (
            "order",
            serde_json::json!(["principal_point", "focal"]),
            "out of the order",
        ),
    ] {
        let dir = temp_dir(&format!("sfmr_test_per_image_adjustments_key_{tag}"));
        let (err, errors) = read_and_verify_edited(
            &dir,
            edit_cameras(move |json| json[0]["per_image_adjustments"] = names.clone()),
        );
        assert_named(&err, &errors, needle);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_file_below_version_13_with_the_columns_or_the_key_is_refused() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_old_version");
    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(12);
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert_named(&err, &errors, "which version 13 introduced");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn all_zero_columns_and_the_camera_key_round_trip() {
    let dir = temp_dir("sfmr_test_per_image_adjustments_zero");
    let zero = dir.join("zero.sfmr");
    write_sfmr(&zero, &mut data_with_zero_adjustments()).unwrap();
    let loaded = read_sfmr(&zero).unwrap();
    assert_eq!(loaded.focal_adjustments, Some(Array1::zeros(3)));
    assert_eq!(
        loaded.principal_point_adjustments_xy,
        Some(Array2::zeros((3, 2)))
    );
    assert_eq!(
        loaded.cameras[0].per_image_adjustments,
        vec!["focal".to_string(), "principal_point".to_string()]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_negative_zero_counts_as_zero() {
    let mut data = data_with_zero_adjustments();
    data.focal_adjustments.as_mut().unwrap()[2] = -0.0;
    data.principal_point_adjustments_xy.as_mut().unwrap()[[0, 1]] = -0.0;
    let dir = temp_dir("sfmr_test_per_image_adjustments_negative_zero");
    write_sfmr(&dir.join("z.sfmr"), &mut data).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_key_names_parse_in_order() {
    assert_eq!(
        parse_per_image_adjustment_names::<&str>(&[]).unwrap(),
        [false, false, false, false]
    );
    assert_eq!(
        parse_per_image_adjustment_names(&["focal"]).unwrap(),
        [true, false, false, false]
    );
    assert_eq!(
        parse_per_image_adjustment_names(&["principal_point"]).unwrap(),
        [false, true, false, false]
    );
    assert_eq!(
        parse_per_image_adjustment_names(&["focal", "principal_point"]).unwrap(),
        [true, true, false, false]
    );
    assert!(parse_per_image_adjustment_names(&["principal_point", "focal"]).is_err());
    assert!(parse_per_image_adjustment_names(&["Focal"]).is_err());
}
