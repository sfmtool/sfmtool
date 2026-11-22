// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image readings: the required columns `images/sharpness` (`float16`)
//! and `images/gradient_structure_tensors_xx_yy_xy`, the required
//! `image_to_gray` key of `images/metadata.json`, and the rules a reader, a
//! writer and a verifier hold them to.

use super::{
    archive_names, float16_payload, images_digest, make_embedded_test_data, make_test_data,
    read_json_entry_of, rewrite_entries,
};
use crate::*;

const SHARPNESS_ENTRY: &str = "images/sharpness.3.float16.zst";
const TENSORS_ENTRY: &str = "images/gradient_structure_tensors_xx_yy_xy.3.3.float32.zst";

/// The little-endian `float16` bytes of `values`, each rounded to the nearest.
fn float16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn the_writer_refuses_readings_the_format_does_not_allow() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn the_reader_and_the_verifier_refuse_a_reading_out_of_range() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn a_corrupted_reading_fails_the_images_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn the_writer_rounds_the_sharpness_to_float16_in_the_value_it_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn a_sharpness_above_the_largest_float16_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}
