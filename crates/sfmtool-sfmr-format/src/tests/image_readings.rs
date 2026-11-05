// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image readings of version 15: the required columns
//! `images/sharpness` and `images/gradient_structure_tensors_xx_yy_xy`, the
//! required `image_to_gray` key of `images/metadata.json`, and the rules a
//! reader, a writer and a verifier hold them to; and the sharpness's change
//! from `float32` to `float16` in version 16.

use super::{
    archive_names, float16_payload, float32_payload, images_digest, make_embedded_test_data,
    make_test_data, read_json_entry_of, restamp_hashes, rewrite_as_version_15, rewrite_entries,
    rewrite_renaming, test_image_readings,
};
use crate::*;

const SHARPNESS_ENTRY: &str = "images/sharpness.3.float16.zst";
const SHARPNESS_ENTRY_V15: &str = "images/sharpness.3.float32.zst";
const TENSORS_ENTRY: &str = "images/gradient_structure_tensors_xx_yy_xy.3.3.float32.zst";

/// The little-endian `float16` bytes of `values`, each rounded to the nearest.
fn float16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Readings whose bits a round trip has to keep: a subnormal, a value with
/// every mantissa bit set, a negative zero cross term and a negative `Exy`.
/// The sharpness values are those of a `float16` (`2^-20` is a subnormal
/// there, and `1 - 2^-11` has every mantissa bit set), the tensor's those of
/// a `float32`.
fn awkward_readings() -> ImageReadingColumns {
    ImageReadingColumns {
        image_to_gray: serde_json::json!({"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}),
        sharpness: ndarray::Array1::from_vec(vec![
            half::f16::from_bits(0x0010).to_f32(),
            half::f16::from_bits(0x3bff).to_f32(),
            0.0,
        ]),
        gradient_structure_tensors_xx_yy_xy: ndarray::Array2::from_shape_vec(
            (3, 3),
            vec![0.25, 1.0e-7, -0.0, 3.0e-3, 2.5e-3, -1.25e-3, 0.0, 0.0, 0.0],
        )
        .unwrap(),
    }
}

fn bits(values: impl IntoIterator<Item = f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
fn the_readings_round_trip_bit_for_bit_in_both_feature_sources() {
    let dir = temp_dir("sfmr_test_image_readings_round_trip");
    for (name, mut data) in [
        ("sift.sfmr", make_test_data()),
        ("embedded.sfmr", make_embedded_test_data()),
    ] {
        data.image_readings = Some(awkward_readings());
        let path = dir.join(name);
        write_sfmr(&path, &mut data).unwrap();
        assert_eq!(data.metadata.version, SFMR_FORMAT_VERSION);

        let names = archive_names(&path);
        assert!(names.contains(&SHARPNESS_ENTRY.to_string()), "{names:?}");
        assert!(names.contains(&TENSORS_ENTRY.to_string()), "{names:?}");
        let images_meta = read_json_entry_of(&path, "images/metadata.json.zst");
        assert_eq!(
            images_meta["image_to_gray"],
            awkward_readings().image_to_gray
        );

        let loaded = read_sfmr(&path).unwrap().image_readings.unwrap();
        let expected = awkward_readings();
        assert_eq!(loaded.image_to_gray, expected.image_to_gray);
        assert_eq!(
            bits(loaded.sharpness.iter().copied()),
            bits(expected.sharpness.iter().copied())
        );
        assert_eq!(
            bits(loaded.gradient_structure_tensors_xx_yy_xy.iter().copied()),
            bits(expected.gradient_structure_tensors_xx_yy_xy.iter().copied())
        );

        // Both columns sit in their lexicographic slots of the images digest.
        let stored = read_sfmr_content_hash(&path).unwrap();
        assert_eq!(
            stored.images_xxh128,
            format!("{:032x}", images_digest(&path)),
            "{name}"
        );
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "{name}: {errors:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_reading_and_the_conversion_reach_the_images_hash() {
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    type Edit = fn(&mut ImageReadingColumns);
    let hash = |edit: Edit| {
        let mut data = make_test_data();
        edit(data.image_readings.as_mut().unwrap());
        content_hash_of(&mut data, &options).unwrap()
    };
    let base = hash(|_| {});
    let edits: [(&str, Edit); 3] = [
        ("sharpness", |r| r.sharpness[1] = 0.5),
        ("tensor", |r| {
            r.gradient_structure_tensors_xx_yy_xy[[2, 2]] = 0.1
        }),
        ("conversion", |r| {
            r.image_to_gray = serde_json::json!({"formula": "G"})
        }),
    ];
    for (what, edit) in edits {
        let changed = hash(edit);
        assert_ne!(changed.images_xxh128, base.images_xxh128, "{what}");
        assert_ne!(changed.content_xxh128, base.content_xxh128, "{what}");
        assert_eq!(changed.points3d_xxh128, base.points3d_xxh128, "{what}");
    }
}

#[test]
fn a_value_without_readings_is_refused_by_the_writer_and_hashed_as_it_stands() {
    let dir = temp_dir("sfmr_test_image_readings_required");
    let path = dir.join("none.sfmr");
    let mut data = make_test_data();
    data.image_readings = None;
    let err = write_sfmr(&path, &mut data).unwrap_err().to_string();
    assert!(err.contains("has none"), "{err}");
    assert!(!path.exists(), "a refused write leaves no file");

    // The hash of the value as it stands, with no reading column.
    let without = content_hash_of(&mut data, &WriteOptions::default()).unwrap();
    let mut with = make_test_data();
    let with = content_hash_of(&mut with, &WriteOptions::default()).unwrap();
    assert_ne!(without.images_xxh128, with.images_xxh128);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_writer_refuses_readings_the_format_does_not_allow() {
    let dir = temp_dir("sfmr_test_image_readings_invalid");
    type Edit = fn(&mut ImageReadingColumns);
    let cases: [(&str, Edit, &str); 11] = [
        (
            "nan sharpness",
            |r| r.sharpness[0] = f32::NAN,
            "images/sharpness row 0",
        ),
        (
            "negative sharpness",
            |r| r.sharpness[2] = -1.0e-9,
            "images/sharpness row 2",
        ),
        (
            "infinite cross term",
            |r| r.gradient_structure_tensors_xx_yy_xy[[1, 2]] = f32::INFINITY,
            "gradient_structure_tensors_xx_yy_xy row 1",
        ),
        (
            "negative exx",
            |r| r.gradient_structure_tensors_xx_yy_xy[[0, 0]] = -0.5,
            "gradient_structure_tensors_xx_yy_xy row 0",
        ),
        (
            "negative eyy",
            |r| r.gradient_structure_tensors_xx_yy_xy[[2, 1]] = -0.5,
            "gradient_structure_tensors_xx_yy_xy row 2",
        ),
        (
            "short sharpness",
            |r| r.sharpness = ndarray::Array1::zeros(2),
            "images/sharpness has 2 rows",
        ),
        (
            "narrow tensors",
            |r| r.gradient_structure_tensors_xx_yy_xy = ndarray::Array2::zeros((3, 2)),
            "has shape [3, 2]",
        ),
        (
            "not an object",
            |r| r.image_to_gray = serde_json::json!("0.2126*R"),
            "must be a JSON object",
        ),
        (
            "no formula",
            |r| r.image_to_gray = serde_json::json!({"quantize_bits": 8}),
            "has no 'formula'",
        ),
        (
            "bits out of range",
            |r| r.image_to_gray = serde_json::json!({"formula": "G", "quantize_bits": 17}),
            "'quantize_bits' must be an integer from 1 to 16",
        ),
        (
            "unknown key",
            |r| r.image_to_gray = serde_json::json!({"formula": "G", "gamma": 2.2}),
            "unknown key",
        ),
    ];
    for (what, edit, needle) in cases {
        let mut data = make_test_data();
        edit(data.image_readings.as_mut().unwrap());
        let path = dir.join("bad.sfmr");
        let err = write_sfmr(&path, &mut data).unwrap_err().to_string();
        assert!(err.contains(needle), "{what}: {err}");
        assert!(!path.exists(), "{what}: a refused write leaves no file");
    }
    // A zero row is a valid reading: an image with no interior reads 0.
    let mut data = make_test_data();
    data.image_readings.as_mut().unwrap().sharpness[0] = 0.0;
    write_sfmr(&dir.join("zero.sfmr"), &mut data).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Write `make_test_data`, rewrite it with `edit`, restamp nothing, and return
/// the reader's error and the verifier's errors.
fn read_and_verify_edited(
    dir: &std::path::Path,
    edit: impl Fn(&str, &[u8]) -> Option<Vec<u8>>,
) -> (String, Vec<String>) {
    let good = dir.join("good.sfmr");
    write_sfmr(&good, &mut make_test_data()).unwrap();
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

/// Copy `src` to `dst` without the entry `skip`.
fn copy_without(src: &std::path::Path, dst: &std::path::Path, skip: &str) {
    use std::io::{Read, Write};
    let mut archive = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let mut zip = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let name = entry.name().unwrap().into_owned();
        if name == skip {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        zip.start_file(&name, stored).unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.finish().unwrap();
}

fn assert_named(err: &str, errors: &[String], needle: &str) {
    assert!(err.contains(needle), "reader: {err}");
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "verifier: {errors:?}"
    );
}

#[test]
fn a_current_file_without_a_column_or_the_key_is_refused() {
    let dir = temp_dir("sfmr_test_image_readings_missing");
    let good = dir.join("good.sfmr");
    write_sfmr(&good, &mut make_test_data()).unwrap();
    for (skip, needle) in [
        (SHARPNESS_ENTRY, "lacks images/sharpness"),
        (
            TENSORS_ENTRY,
            "lacks images/gradient_structure_tensors_xx_yy_xy",
        ),
    ] {
        let bad = dir.join("bad.sfmr");
        copy_without(&good, &bad, skip);
        let err = read_sfmr(&bad)
            .err()
            .expect("the reader refuses it")
            .to_string();
        let (valid, errors) = verify_sfmr(&bad).unwrap();
        assert!(!valid);
        assert_named(&err, &errors, needle);
    }

    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == "images/metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json.as_object_mut().unwrap().remove("image_to_gray");
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert_named(&err, &errors, "lacks the image_to_gray key");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_reader_and_the_verifier_refuse_a_reading_out_of_range() {
    let dir = temp_dir("sfmr_test_image_readings_out_of_range");
    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == SHARPNESS_ENTRY).then(|| {
            let mut values = float16_payload(raw);
            values[1] = -2.0;
            float16_bytes(&values)
        })
    });
    assert_named(&err, &errors, "images/sharpness row 1 is -2");

    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == TENSORS_ENTRY).then(|| {
            let mut values: Vec<f32> = bytemuck::cast_slice(raw).to_vec();
            values[4] = f32::NAN;
            bytemuck::cast_slice(&values).to_vec()
        })
    });
    assert_named(&err, &errors, "gradient_structure_tensors_xx_yy_xy row 1");

    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == "images/metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["image_to_gray"]["quantize_bits"] = serde_json::json!(0);
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert_named(&err, &errors, "'quantize_bits' must be an integer");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupted_reading_fails_the_images_hash() {
    let dir = temp_dir("sfmr_test_image_readings_corrupt");
    let good = dir.join("good.sfmr");
    write_sfmr(&good, &mut make_test_data()).unwrap();
    let bad = dir.join("bad.sfmr");
    // A different valid value: only the hash can tell.
    rewrite_entries(&good, &bad, |name, raw| {
        (name == SHARPNESS_ENTRY).then(|| {
            let mut values = float16_payload(raw);
            values[0] *= 2.0;
            float16_bytes(&values)
        })
    });
    assert!(
        read_sfmr(&bad).is_ok(),
        "the reader checks values, not hashes"
    );
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    assert!(
        errors.iter().any(|e| e.contains("Images hash mismatch")),
        "{errors:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `make_test_data` written as a version 14 file: no reading column and no
/// `image_to_gray` key, hashes restamped so it verifies.
fn write_version_14(path: &std::path::Path, dir: &std::path::Path) {
    let current = dir.join("current.sfmr");
    let mut data = make_test_data();
    data.image_readings = None;
    crate::write::write_sfmr_without_readings_check(&current, &mut data).unwrap();
    rewrite_entries(&current, path, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(14);
            serde_json::to_vec(&json).unwrap()
        })
    });
    restamp_hashes(path, dir);
}

#[test]
fn a_version_14_file_reads_with_no_readings_and_saves_with_them_as_the_current_version() {
    let dir = temp_dir("sfmr_test_image_readings_v14");
    let old = dir.join("v14.sfmr");
    write_version_14(&old, &dir);
    let (valid, errors) = verify_sfmr(&old).unwrap();
    assert!(valid, "{errors:?}");

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 14);
    assert!(loaded.image_readings.is_none());

    // The format crate computes nothing: the value cannot be saved until the
    // readings are supplied.
    let saved = dir.join("saved.sfmr");
    assert!(write_sfmr(&saved, &mut loaded).is_err());
    loaded.image_readings = Some(test_image_readings(3));
    write_sfmr(&saved, &mut loaded).unwrap();
    let reloaded = read_sfmr(&saved).unwrap();
    assert_eq!(reloaded.metadata.version, SFMR_FORMAT_VERSION);
    assert_eq!(reloaded.image_readings, Some(test_image_readings(3)));
    assert!(verify_sfmr(&saved).unwrap().0);

    // Dropping both columns and the key maps the saved file back to 14: the
    // two files then differ only in the version.
    let old_hash = read_sfmr_content_hash(&old).unwrap();
    let new_hash = read_sfmr_content_hash(&saved).unwrap();
    assert_eq!(new_hash.cameras_xxh128, old_hash.cameras_xxh128);
    assert_eq!(new_hash.points3d_xxh128, old_hash.points3d_xxh128);
    assert_eq!(new_hash.tracks_xxh128, old_hash.tracks_xxh128);
    assert_ne!(new_hash.images_xxh128, old_hash.images_xxh128);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_below_version_15_with_a_column_or_the_key_is_refused() {
    let dir = temp_dir("sfmr_test_image_readings_old_version");
    let (err, errors) = read_and_verify_edited(&dir, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["version"] = serde_json::json!(14);
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert_named(&err, &errors, "which version 15 introduced");
    assert!(err.contains("images/sharpness"), "{err}");

    // The key alone is refused as well.
    let old = dir.join("v14.sfmr");
    write_version_14(&old, &dir);
    let bad = dir.join("v14_key.sfmr");
    rewrite_entries(&old, &bad, |name, raw| {
        (name == "images/metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["image_to_gray"] = serde_json::json!({"formula": "G"});
            serde_json::to_vec(&json).unwrap()
        })
    });
    let err = read_sfmr(&bad)
        .err()
        .expect("the reader refuses it")
        .to_string();
    let (_, errors) = verify_sfmr(&bad).unwrap();
    assert_named(
        &err,
        &errors,
        "the image_to_gray key of images/metadata.json",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_writer_rounds_the_sharpness_to_float16_in_the_value_it_writes() {
    let dir = temp_dir("sfmr_test_image_readings_float16_rounding");
    let path = dir.join("rounded.sfmr");
    // None of these is a float16: 0.01 rounds to a neighbour, 3.5e-8 (just
    // above 2^-25) to the smallest subnormal 2^-24, and 0.0123 to its
    // nearest. The tensor values are not float16 either and stay as they are.
    let given = [0.01_f32, 3.5e-8, 0.0123];
    let tensors = ndarray::Array2::from_shape_vec(
        (3, 3),
        vec![
            0.0123, 1.0e-7, -3.0e-7, 1.0e-3, 2.0e-3, 0.0, 0.25, 0.125, 1.0e-6,
        ],
    )
    .unwrap();
    let unrounded = || {
        let mut data = make_test_data();
        let readings = data.image_readings.as_mut().unwrap();
        readings.sharpness = ndarray::Array1::from_vec(given.to_vec());
        readings.gradient_structure_tensors_xx_yy_xy = tensors.clone();
        data
    };
    let mut data = unrounded();
    write_sfmr(&path, &mut data).unwrap();

    let stored = data.image_readings.as_ref().unwrap().sharpness.clone();
    for (k, (&s, &g)) in stored.iter().zip(&given).enumerate() {
        assert_ne!(s, g, "row {k} was not rounded");
        assert_eq!(s, half::f16::from_f32(g).to_f32(), "row {k}");
    }
    assert!(
        (stored[0] - 0.01).abs() <= 0.01 * 2f32.powi(-11),
        "{}",
        stored[0]
    );
    assert_eq!(stored[1], 2f32.powi(-24));

    // What the value holds after the write is what the file gives back, bit
    // for bit, and the tensor is the one handed in.
    let mut loaded = read_sfmr(&path).unwrap();
    let readings = loaded.image_readings.as_ref().unwrap();
    assert_eq!(
        bits(readings.sharpness.iter().copied()),
        bits(stored.iter().copied())
    );
    assert_eq!(
        bits(readings.gradient_structure_tensors_xx_yy_xy.iter().copied()),
        bits(tensors.iter().copied())
    );

    // The hash of the unrounded value is the stored one, since hashing runs
    // the same rounding, and saving what was read writes the same hash.
    let stored_hash = read_sfmr_content_hash(&path).unwrap();
    assert_eq!(
        content_hash_of(&mut unrounded(), &WriteOptions::default())
            .unwrap()
            .content_xxh128,
        stored_hash.content_xxh128
    );
    let resaved = dir.join("resaved.sfmr");
    write_sfmr(&resaved, &mut loaded).unwrap();
    assert_eq!(
        read_sfmr_content_hash(&resaved).unwrap().content_xxh128,
        stored_hash.content_xxh128
    );
    assert!(verify_sfmr(&resaved).unwrap().0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_sharpness_above_the_largest_float16_is_refused() {
    let dir = temp_dir("sfmr_test_image_readings_float16_overflow");
    let path = dir.join("bad.sfmr");
    let mut data = make_test_data();
    data.image_readings.as_mut().unwrap().sharpness[2] = 70_000.0;
    let err = write_sfmr(&path, &mut data).unwrap_err().to_string();
    assert!(
        err.contains("images/sharpness row 2 is 70000, above 65504"),
        "{err}"
    );
    assert!(!path.exists(), "a refused write leaves no file");
    // 65504 itself is stored.
    data.image_readings.as_mut().unwrap().sharpness[2] = 65_504.0;
    write_sfmr(&path, &mut data).unwrap();
    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(loaded.image_readings.unwrap().sharpness[2], 65_504.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_version_15_file_reads_its_float32_sharpness_and_saves_it_as_float16() {
    let dir = temp_dir("sfmr_test_image_readings_v15");
    let current = dir.join("current.sfmr");
    write_sfmr(&current, &mut make_test_data()).unwrap();
    // A version 15 writer stored the sharpness as float32, at values no
    // float16 holds.
    let given = [0.01_f32, 0.02, 0.03];
    let old = dir.join("v15.sfmr");
    rewrite_as_version_15(&current, &old, &dir, |name, values| {
        if name == SHARPNESS_ENTRY_V15 {
            values.copy_from_slice(&given);
        }
    });
    let names = archive_names(&old);
    assert!(
        names.contains(&SHARPNESS_ENTRY_V15.to_string()),
        "{names:?}"
    );
    assert!(!names.contains(&SHARPNESS_ENTRY.to_string()), "{names:?}");
    let (valid, errors) = verify_sfmr(&old).unwrap();
    assert!(valid, "{errors:?}");

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 15);
    let readings = loaded.image_readings.as_ref().unwrap();
    assert_eq!(bits(readings.sharpness.iter().copied()), bits(given));
    assert_eq!(
        readings.gradient_structure_tensors_xx_yy_xy,
        test_image_readings(3).gradient_structure_tensors_xx_yy_xy
    );

    // Saved, it is the current version with the sharpness rounded, which
    // changes the images and content hashes and no other section's.
    let saved = dir.join("saved.sfmr");
    write_sfmr(&saved, &mut loaded).unwrap();
    let names = archive_names(&saved);
    assert!(names.contains(&SHARPNESS_ENTRY.to_string()), "{names:?}");
    assert!(
        !names.contains(&SHARPNESS_ENTRY_V15.to_string()),
        "{names:?}"
    );
    let reloaded = read_sfmr(&saved).unwrap();
    assert_eq!(reloaded.metadata.version, SFMR_FORMAT_VERSION);
    assert_eq!(
        bits(reloaded.image_readings.unwrap().sharpness.iter().copied()),
        bits(given.map(|g| half::f16::from_f32(g).to_f32()))
    );
    let old_hash = read_sfmr_content_hash(&old).unwrap();
    let new_hash = read_sfmr_content_hash(&saved).unwrap();
    assert_ne!(new_hash.images_xxh128, old_hash.images_xxh128);
    assert_ne!(new_hash.content_xxh128, old_hash.content_xxh128);
    assert_eq!(new_hash.cameras_xxh128, old_hash.cameras_xxh128);
    assert_eq!(new_hash.points3d_xxh128, old_hash.points3d_xxh128);
    assert_eq!(new_hash.tracks_xxh128, old_hash.tracks_xxh128);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_sharpness_in_the_dtype_of_another_version_is_refused() {
    let dir = temp_dir("sfmr_test_image_readings_wrong_dtype");
    let current = dir.join("current.sfmr");
    write_sfmr(&current, &mut make_test_data()).unwrap();

    // A current file holding the sharpness as float32.
    let bad = dir.join("float32_in_current.sfmr");
    rewrite_renaming(&current, &bad, |name, raw| {
        (name == SHARPNESS_ENTRY).then(|| {
            (
                SHARPNESS_ENTRY_V15.to_string(),
                float32_payload(&float16_payload(raw)),
            )
        })
    });
    let err = read_sfmr(&bad)
        .err()
        .expect("the reader refuses it")
        .to_string();
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    assert_named(
        &err,
        &errors,
        &format!(
            "a version {SFMR_FORMAT_VERSION} file stores {SHARPNESS_ENTRY}, and this one holds \
             {SHARPNESS_ENTRY_V15} instead"
        ),
    );

    // A version 15 file holding it as float16.
    let old = dir.join("v15.sfmr");
    rewrite_as_version_15(&current, &old, &dir, |_, _| {});
    let bad = dir.join("float16_in_v15.sfmr");
    rewrite_renaming(&old, &bad, |name, raw| {
        (name == SHARPNESS_ENTRY_V15).then(|| {
            let values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            (SHARPNESS_ENTRY.to_string(), float16_bytes(&values))
        })
    });
    let err = read_sfmr(&bad)
        .err()
        .expect("the reader refuses it")
        .to_string();
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    assert_named(
        &err,
        &errors,
        &format!(
            "a version 15 file stores {SHARPNESS_ENTRY_V15}, and this one holds \
             {SHARPNESS_ENTRY} instead"
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}
