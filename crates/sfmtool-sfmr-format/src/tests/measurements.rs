// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Physical distance measurements (`measurements/`), their `unit_names` legend,
//! version 16/17 compatibility, and read/write/verify validation.

use std::io::{Read, Write};
use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use sfmtool_archive_io::{format_hash, read_zst_entry, SectionDigests};

use super::{
    archive_names, make_embedded_test_data, make_test_data, read_json_entry_of, relabel_version,
    rewrite_entries,
};
use crate::*;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build `n` valid physical distance measurements for `make_test_data`'s three
/// `1920x1080` images, cycling through all five units, both same-image and
/// cross-image endpoint pairs, and empty + non-empty UTF-8 labels.
pub(super) fn sample_measurements(n: usize) -> MeasurementsData {
    let distances = Array1::from_iter((0..n).map(|i| 0.25 + (i as f32) * 12.5));
    let units =
        Array1::from_iter((0..n).map(|i| LengthUnit::ALL[i % LengthUnit::ALL.len()].code()));
    let image_indexes = Array2::from_shape_fn((n, 2), |(i, k)| {
        if i == 0 {
            // Same-image pair (`img_0 == img_1`) is valid.
            1
        } else {
            ((i + k) % 3) as u32
        }
    });
    let endpoints_xy = Array3::from_shape_fn((n, 2, 2), |(i, k, c)| {
        if c == 0 {
            100.5 + (i * 17 + k * 53) as f32
        } else {
            200.25 + (i * 11 + k * 37) as f32
        }
    });
    let labels = (0..n)
        .map(|i| match i % 3 {
            0 => "ruler edge — front".to_string(),
            1 => String::new(),
            _ => format!("span #{i}"),
        })
        .collect();
    MeasurementsData {
        distances,
        units,
        image_indexes,
        endpoints_xy,
        labels,
    }
}

/// XXH128 over the decompressed `measurements/` entries of a written file, in
/// lexicographic path order: the format's rule, spelled out independently of
/// the writer.
fn measurements_digest(path: &Path) -> u128 {
    let file = std::fs::File::open(path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    let mut names: Vec<String> = archive
        .file_names()
        .map(Result::unwrap)
        .filter(|n| n.starts_with("measurements/"))
        .map(String::from)
        .collect();
    names.sort();
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    for name in &names {
        let mut compressed = Vec::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();
        hasher.update(&zstd::stream::decode_all(&compressed[..]).unwrap());
    }
    hasher.digest128()
}

/// Recompute `metadata_xxh128`, `measurements_xxh128` (when present) and
/// `content_xxh128` in `path` after a surgical entry edit, so `verify_sfmr`
/// checks the structural rules with intact digests.
fn restamp_metadata_and_measurements_hashes(src: &Path, dst: &Path) {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let metadata_raw = read_zst_entry(&mut archive, "metadata.json.zst").unwrap();
    let stored: ContentHash =
        serde_json::from_slice(&read_zst_entry(&mut archive, "content_hash.json.zst").unwrap())
            .unwrap();
    let has_measurements = archive
        .file_names()
        .any(|n| n.unwrap().starts_with("measurements/"));
    drop(archive);

    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_raw);
    let meas_hash = has_measurements.then(|| measurements_digest(src));

    let mut digests = SectionDigests::new();
    digests.push(metadata_hash);
    digests.push(u128::from_str_radix(&stored.cameras_xxh128, 16).unwrap());
    if let Some(h) = &stored.rigs_xxh128 {
        digests.push(u128::from_str_radix(h, 16).unwrap());
    }
    if let Some(h) = &stored.frames_xxh128 {
        digests.push(u128::from_str_radix(h, 16).unwrap());
    }
    digests.push(u128::from_str_radix(&stored.images_xxh128, 16).unwrap());
    digests.push(u128::from_str_radix(&stored.points3d_xxh128, 16).unwrap());
    digests.push(u128::from_str_radix(&stored.tracks_xxh128, 16).unwrap());
    if let Some(h) = meas_hash {
        digests.push(h);
    }

    let updated = ContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        measurements_xxh128: meas_hash.map(format_hash),
        content_xxh128: format_hash(digests.finish()),
        ..stored
    };
    let hash_bytes = serde_json::to_vec(&updated).unwrap();
    rewrite_entries(src, dst, |name, _| {
        (name == "content_hash.json.zst").then(|| hash_bytes.clone())
    });
}

#[test]
fn length_unit_table_agrees_with_world_space_units() {
    assert_eq!(LengthUnit::ALL.len(), WORLD_SPACE_UNITS.len());
    for (i, unit) in LengthUnit::ALL.into_iter().enumerate() {
        assert_eq!(unit.code() as usize, i);
        assert_eq!(LengthUnit::from_code(i as u8), Some(unit));
        assert_eq!(unit.name(), WORLD_SPACE_UNITS[i].0);
        assert_eq!(unit.name(), LengthUnit::NAMES[i]);
        assert_eq!(unit.in_metres(), WORLD_SPACE_UNITS[i].1);
        assert_eq!(LengthUnit::from_name(unit.name()), Some(unit));
        assert_eq!(
            world_space_unit_in_metres(unit.name()),
            Some(unit.in_metres())
        );
    }
    assert_eq!(LengthUnit::from_code(5), None);
    assert_eq!(LengthUnit::from_name("km"), None);
    assert_eq!(LengthUnit::from_name("M"), None);
}

#[test]
fn round_trip_with_measurements_in_both_feature_sources() {
    let dir = temp_dir("sfmr_test_measurements_round_trip");
    let expected = sample_measurements(5);

    for (name, mut data) in [
        ("sift.sfmr", make_test_data()),
        ("embedded.sfmr", make_embedded_test_data()),
    ] {
        data.measurements = Some(expected.clone());
        let path = dir.join(name);
        write_sfmr(&path, &mut data).unwrap();
        assert_eq!(data.metadata.version, SFMR_FORMAT_VERSION);
        assert_eq!(data.metadata.measurement_count, Some(5));

        let loaded = read_sfmr(&path).unwrap();
        assert_eq!(loaded.metadata.version, SFMR_FORMAT_VERSION);
        assert_eq!(loaded.metadata.measurement_count, Some(5));
        assert_eq!(loaded.measurements.as_ref(), Some(&expected));

        // Canonical legend and count in measurements/metadata.json.
        let meta = read_json_entry_of(&path, "measurements/metadata.json.zst");
        assert_eq!(meta["measurement_count"], serde_json::json!(5));
        assert_eq!(
            meta["unit_names"],
            serde_json::json!(["mm", "cm", "m", "in", "ft"])
        );

        // Section digest matches lexicographic XXH128 over measurements/ entries.
        let stored = read_sfmr_content_hash(&path).unwrap();
        assert_eq!(
            stored.measurements_xxh128.as_deref(),
            Some(format!("{:032x}", measurements_digest(&path)).as_str()),
            "{name}"
        );

        // content_hash_of agrees with what write_sfmr stored.
        let mut for_hash = loaded.clone_for_hash_test();
        let computed = content_hash_of(&mut for_hash, &WriteOptions::default()).unwrap();
        assert_eq!(computed.measurements_xxh128, stored.measurements_xxh128);
        assert_eq!(computed.content_xxh128, stored.content_xxh128);

        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "{name}: {errors:?}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

impl SfmrData {
    fn clone_for_hash_test(&self) -> Self {
        Self {
            workspace_dir: self.workspace_dir.clone(),
            metadata: self.metadata.clone(),
            content_hash: self.content_hash.clone(),
            cameras: self.cameras.clone(),
            rig_frame_data: self.rig_frame_data.clone(),
            image_names: self.image_names.clone(),
            camera_indexes: self.camera_indexes.clone(),
            quaternions_wxyz: self.quaternions_wxyz.clone(),
            translations_xyz: self.translations_xyz.clone(),
            feature_tool_hashes: self.feature_tool_hashes.clone(),
            sift_content_hashes: self.sift_content_hashes.clone(),
            image_file_hashes: self.image_file_hashes.clone(),
            thumbnails_y_x_rgb: self.thumbnails_y_x_rgb.clone(),
            focal_adjustments: self.focal_adjustments.clone(),
            principal_point_adjustments_xy: self.principal_point_adjustments_xy.clone(),
            radial_bspline_adjustments: self.radial_bspline_adjustments.clone(),
            rolling_shutter_rates_xyz: self.rolling_shutter_rates_xyz.clone(),
            image_readings: self.image_readings.clone(),
            positions_xyzw: self.positions_xyzw.clone(),
            colors_rgb: self.colors_rgb.clone(),
            reprojection_errors: self.reprojection_errors.clone(),
            normals_xyz: self.normals_xyz.clone(),
            normal_confidence: self.normal_confidence.clone(),
            point_constraints: self.point_constraints.clone(),
            constraint_distances: self.constraint_distances.clone(),
            constraint_reference_images: self.constraint_reference_images.clone(),
            patch_u_halfvec_xyz: self.patch_u_halfvec_xyz.clone(),
            patch_v_halfvec_xyz: self.patch_v_halfvec_xyz.clone(),
            patch_bitmaps_y_x_rgba: self.patch_bitmaps_y_x_rgba.clone(),
            image_indexes: self.image_indexes.clone(),
            feature_indexes: self.feature_indexes.clone(),
            keypoints_xy: self.keypoints_xy.clone(),
            observation_confidence: self.observation_confidence.clone(),
            observation_readings: self.observation_readings.clone(),
            point_indexes: self.point_indexes.clone(),
            observation_counts: self.observation_counts.clone(),
            reference_observations: self.reference_observations.clone(),
            depth_statistics: self.depth_statistics.clone(),
            observed_depth_histogram_counts: self.observed_depth_histogram_counts.clone(),
            measurements: self.measurements.clone(),
        }
    }
}

#[test]
fn round_trip_without_measurements_omits_section_and_keys() {
    let dir = temp_dir("sfmr_test_without_measurements");
    let path = dir.join("plain.sfmr");
    let mut data = make_test_data();
    assert!(data.measurements.is_none());
    write_sfmr(&path, &mut data).unwrap();

    assert!(
        !archive_names(&path)
            .iter()
            .any(|n| n.starts_with("measurements/")),
        "no measurements/ entry should be written"
    );
    let top_meta = read_json_entry_of(&path, "metadata.json.zst");
    assert!(top_meta.get("measurement_count").is_none());
    let hash_json = read_json_entry_of(&path, "content_hash.json.zst");
    assert!(hash_json.get("measurements_xxh128").is_none());

    let loaded = read_sfmr(&path).unwrap();
    assert!(loaded.metadata.measurement_count.is_none());
    assert!(loaded.content_hash.measurements_xxh128.is_none());
    assert!(loaded.measurements.is_none());
    assert_eq!(verify_sfmr(&path).unwrap(), (true, vec![]));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_measurement_field_reaches_the_section_and_content_hash() {
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    let hash_of = |edit: fn(&mut MeasurementsData)| {
        let mut data = make_test_data();
        let mut m = sample_measurements(3);
        edit(&mut m);
        data.measurements = Some(m);
        content_hash_of(&mut data, &options).unwrap()
    };
    let base = hash_of(|_| {});
    let mut bare_data = make_test_data();
    let bare = content_hash_of(&mut bare_data, &options).unwrap();
    assert_ne!(base.content_xxh128, bare.content_xxh128);
    assert!(base.measurements_xxh128.is_some());
    assert!(bare.measurements_xxh128.is_none());

    type Edit = fn(&mut MeasurementsData);
    let edits: [(&str, Edit); 5] = [
        ("distances", |m| m.distances[0] += 0.5),
        ("units", |m| m.units[0] = LENGTH_UNIT_FT),
        ("image_indexes", |m| m.image_indexes[[0, 1]] = 2),
        ("endpoints_xy", |m| m.endpoints_xy[[1, 0, 1]] += 1.0),
        ("labels", |m| m.labels[1] = "labelled".into()),
    ];
    for (what, edit) in edits {
        let changed = hash_of(edit);
        assert_ne!(
            changed.measurements_xxh128, base.measurements_xxh128,
            "{what} must change measurements_xxh128"
        );
        assert_ne!(
            changed.content_xxh128, base.content_xxh128,
            "{what} must change content_xxh128"
        );
        assert_eq!(changed.tracks_xxh128, base.tracks_xxh128);
    }
}

#[test]
fn measurements_read_and_verify_through_permuted_and_subset_unit_legends() {
    let dir = temp_dir("sfmr_test_measurements_legends");
    let good = dir.join("good.sfmr");
    let mut data = make_test_data();
    data.measurements = Some(sample_measurements(5));
    write_sfmr(&good, &mut data).unwrap();

    // 1. Permuted full legend: reversed canonical order.
    let permuted = [
        LengthUnit::Foot,
        LengthUnit::Inch,
        LengthUnit::Metre,
        LengthUnit::Centimetre,
        LengthUnit::Millimetre,
    ];
    let permuted_raw = dir.join("permuted_raw.sfmr");
    let permuted_path = dir.join("permuted.sfmr");
    rewrite_entries(&good, &permuted_raw, |name, raw| {
        if name == "measurements/metadata.json.zst" {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["unit_names"] = serde_json::json!(permuted.map(|u| u.name()));
            Some(serde_json::to_vec(&json).unwrap())
        } else if name == "measurements/units.5.uint8.zst" {
            Some(
                raw.iter()
                    .map(|&code| {
                        let unit = LengthUnit::ALL[code as usize];
                        permuted.iter().position(|&u| u == unit).unwrap() as u8
                    })
                    .collect(),
            )
        } else {
            None
        }
    });
    restamp_metadata_and_measurements_hashes(&permuted_raw, &permuted_path);

    let loaded = read_sfmr(&permuted_path).unwrap();
    assert_eq!(
        loaded.measurements.as_ref().unwrap().units.to_vec(),
        vec![
            LENGTH_UNIT_MM,
            LENGTH_UNIT_CM,
            LENGTH_UNIT_M,
            LENGTH_UNIT_IN,
            LENGTH_UNIT_FT,
        ]
    );
    assert_eq!(verify_sfmr(&permuted_path).unwrap(), (true, vec![]));

    // Resaving normalises back to the canonical legend and matches `good`.
    let resaved = dir.join("resaved.sfmr");
    let mut again = loaded;
    write_sfmr(&resaved, &mut again).unwrap();
    assert_eq!(
        read_sfmr_content_hash(&resaved)
            .unwrap()
            .measurements_xxh128,
        read_sfmr_content_hash(&good).unwrap().measurements_xxh128
    );

    // 2. Subset legend naming only the two units used by the rows.
    let mut sub_data = make_test_data();
    let mut m = sample_measurements(2);
    m.units = Array1::from_vec(vec![LENGTH_UNIT_IN, LENGTH_UNIT_CM]);
    sub_data.measurements = Some(m);
    let sub_good = dir.join("sub_good.sfmr");
    write_sfmr(&sub_good, &mut sub_data).unwrap();

    let subset = [LengthUnit::Centimetre, LengthUnit::Inch];
    let sub_raw = dir.join("sub_raw.sfmr");
    let sub_path = dir.join("sub.sfmr");
    rewrite_entries(&sub_good, &sub_raw, |name, raw| {
        if name == "measurements/metadata.json.zst" {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["unit_names"] = serde_json::json!(subset.map(|u| u.name()));
            Some(serde_json::to_vec(&json).unwrap())
        } else if name == "measurements/units.2.uint8.zst" {
            Some(vec![1u8, 0u8]) // 1 -> "in", 0 -> "cm"
        } else {
            None
        }
    });
    restamp_metadata_and_measurements_hashes(&sub_raw, &sub_path);

    let loaded_sub = read_sfmr(&sub_path).unwrap();
    assert_eq!(
        loaded_sub.measurements.unwrap().units.to_vec(),
        vec![LENGTH_UNIT_IN, LENGTH_UNIT_CM]
    );
    assert_eq!(verify_sfmr(&sub_path).unwrap(), (true, vec![]));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn malformed_unit_legend_is_rejected_on_read_and_verify() {
    let dir = temp_dir("sfmr_test_bad_measurement_legend");
    let good = dir.join("good.sfmr");
    let mut data = make_test_data();
    data.measurements = Some(sample_measurements(5));
    write_sfmr(&good, &mut data).unwrap();

    type Case = (&'static str, Option<serde_json::Value>, &'static str);
    let cases: Vec<Case> = vec![
        (
            "a code past the end of the legend",
            Some(serde_json::json!(["mm", "cm"])),
            "past the 2 names its legend gives",
        ),
        (
            "a unit name the format does not define",
            Some(serde_json::json!(["mm", "cm", "km", "in", "ft"])),
            "not one of",
        ),
        (
            "a duplicate unit name",
            Some(serde_json::json!(["mm", "mm", "m", "in", "ft"])),
            "repeats",
        ),
        ("missing unit_names", None, "carries no unit_names"),
        (
            "unit_names is not a list of names",
            Some(serde_json::json!("mm")),
            "not a list of names",
        ),
        (
            "empty unit_names",
            Some(serde_json::json!([])),
            "names no unit at all",
        ),
    ];

    for (what, legend, expected) in cases {
        let path = dir.join("bad.sfmr");
        let _ = std::fs::remove_file(&path);
        rewrite_entries(&good, &path, |name, raw| {
            (name == "measurements/metadata.json.zst").then(|| {
                let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
                let obj = json.as_object_mut().unwrap();
                match &legend {
                    Some(v) => {
                        obj.insert("unit_names".into(), v.clone());
                    }
                    None => {
                        obj.remove("unit_names");
                    }
                }
                serde_json::to_vec(&json).unwrap()
            })
        });

        let err = read_sfmr(&path)
            .err()
            .unwrap_or_else(|| panic!("{what}: reader accepted it"))
            .to_string();
        assert!(
            err.contains(expected),
            "{what}: expected {expected:?} in {err:?}"
        );
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(!valid, "{what}: verify accepted it");
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "{what}: expected {expected:?} in {errors:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_measurements_are_rejected_on_write() {
    let dir = temp_dir("sfmr_test_bad_measurements_write");
    let path = dir.join("bad.sfmr");

    type Case = (&'static str, Box<dyn Fn(&mut SfmrData)>, &'static str);
    let cases: Vec<Case> = vec![
        (
            "empty measurements section",
            Box::new(|d| {
                d.measurements = Some(MeasurementsData {
                    distances: Array1::from_vec(vec![]),
                    units: Array1::from_vec(vec![]),
                    image_indexes: Array2::zeros((0, 2)),
                    endpoints_xy: Array3::zeros((0, 2, 2)),
                    labels: vec![],
                });
            }),
            "at least one measurement",
        ),
        (
            "zero distance",
            Box::new(|d| d.measurements.as_mut().unwrap().distances[0] = 0.0),
            "finite and strictly positive",
        ),
        (
            "negative distance",
            Box::new(|d| d.measurements.as_mut().unwrap().distances[1] = -2.5),
            "finite and strictly positive",
        ),
        (
            "NaN distance",
            Box::new(|d| d.measurements.as_mut().unwrap().distances[0] = f32::NAN),
            "finite and strictly positive",
        ),
        (
            "infinite distance",
            Box::new(|d| d.measurements.as_mut().unwrap().distances[0] = f32::INFINITY),
            "finite and strictly positive",
        ),
        (
            "unit code out of range",
            Box::new(|d| d.measurements.as_mut().unwrap().units[0] = 5),
            "past the 5 names",
        ),
        (
            "units length mismatch",
            Box::new(|d| d.measurements.as_mut().unwrap().units = Array1::from_vec(vec![0])),
            "measurements/units len 1 != measurement_count 3",
        ),
        (
            "image index out of range",
            Box::new(|d| d.measurements.as_mut().unwrap().image_indexes[[1, 1]] = 3),
            "out of range for 3 images",
        ),
        (
            "image_indexes shape mismatch",
            Box::new(|d| d.measurements.as_mut().unwrap().image_indexes = Array2::zeros((2, 2))),
            "measurements/image_indexes shape",
        ),
        (
            "NaN endpoint coordinate",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy[[0, 1, 0]] = f32::NAN),
            "not finite",
        ),
        (
            "infinite endpoint coordinate",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy[[0, 0, 1]] = f32::INFINITY),
            "not finite",
        ),
        (
            "negative endpoint x",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy[[0, 0, 0]] = -0.1),
            "outside image bounds",
        ),
        (
            "endpoint x at width",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy[[0, 0, 0]] = 1920.0),
            "outside image bounds",
        ),
        (
            "endpoint y at height",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy[[0, 1, 1]] = 1080.0),
            "outside image bounds",
        ),
        (
            "endpoints_xy shape mismatch",
            Box::new(|d| d.measurements.as_mut().unwrap().endpoints_xy = Array3::zeros((2, 2, 2))),
            "measurements/endpoints_xy shape",
        ),
        (
            "labels length mismatch",
            Box::new(|d| {
                d.measurements.as_mut().unwrap().labels.pop();
            }),
            "measurements/labels len 2 != measurement_count 3",
        ),
        (
            "metadata.measurement_count disagrees with measurements len",
            Box::new(|d| d.metadata.measurement_count = Some(9)),
            "metadata.measurement_count (9) != measurements len (3)",
        ),
        (
            "metadata.measurement_count set while measurements is None",
            Box::new(|d| {
                d.measurements = None;
                d.metadata.measurement_count = Some(1);
            }),
            "metadata.measurement_count is set, but measurements is None",
        ),
    ];

    for (what, break_it, expected) in cases {
        let _ = std::fs::remove_file(&path);
        let mut data = make_test_data();
        data.measurements = Some(sample_measurements(3));
        break_it(&mut data);
        let err = write_sfmr(&path, &mut data)
            .err()
            .unwrap_or_else(|| panic!("{what}: writer accepted it"))
            .to_string();
        assert!(
            err.contains(expected),
            "{what}: expected {expected:?} in {err:?}"
        );
        assert!(!path.exists(), "{what}: failed write left a file behind");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_measurements_in_archive_are_rejected_on_read_and_verify() {
    let dir = temp_dir("sfmr_test_bad_measurements_archive");
    let good = dir.join("good.sfmr");
    let mut data = make_test_data();
    data.measurements = Some(sample_measurements(2));
    write_sfmr(&good, &mut data).unwrap();

    // 1. Bad distance (<= 0), bad image_index, bad endpoint, bad labels length,
    //    mismatched top-level measurement_count, missing top-level measurement_count,
    //    missing measurements_xxh128.
    type EditFn = Box<dyn Fn(&str, &[u8]) -> Option<Vec<u8>>>;
    let cases: Vec<(&str, EditFn, &str)> = vec![
        (
            "non-positive distance",
            Box::new(|name, raw| {
                (name == "measurements/distances.2.float32.zst").then(|| {
                    let mut out = raw.to_vec();
                    out[0..4].copy_from_slice(&0.0f32.to_le_bytes());
                    out
                })
            }),
            "finite and strictly positive",
        ),
        (
            "NaN distance",
            Box::new(|name, raw| {
                (name == "measurements/distances.2.float32.zst").then(|| {
                    let mut out = raw.to_vec();
                    out[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
                    out
                })
            }),
            "finite and strictly positive",
        ),
        (
            "image index out of range",
            Box::new(|name, raw| {
                (name == "measurements/image_indexes.2.2.uint32.zst").then(|| {
                    let mut out = raw.to_vec();
                    out[0..4].copy_from_slice(&99u32.to_le_bytes());
                    out
                })
            }),
            "out of range for 3 images",
        ),
        (
            "endpoint out of bounds",
            Box::new(|name, raw| {
                (name == "measurements/endpoints_xy.2.2.2.float32.zst").then(|| {
                    let mut out = raw.to_vec();
                    out[0..4].copy_from_slice(&2000.0f32.to_le_bytes());
                    out
                })
            }),
            "outside image bounds",
        ),
        (
            "NaN endpoint",
            Box::new(|name, raw| {
                (name == "measurements/endpoints_xy.2.2.2.float32.zst").then(|| {
                    let mut out = raw.to_vec();
                    out[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
                    out
                })
            }),
            "not finite",
        ),
        (
            "labels length mismatch",
            Box::new(|name, _| {
                (name == "measurements/labels.json.zst")
                    .then(|| serde_json::to_vec(&vec!["only one"]).unwrap())
            }),
            "measurements/labels len 1 != measurement_count 2",
        ),
        (
            "top-level measurement_count disagrees with section",
            Box::new(|name, raw| {
                (name == "metadata.json.zst").then(|| {
                    let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
                    json["measurement_count"] = serde_json::json!(7);
                    serde_json::to_vec(&json).unwrap()
                })
            }),
            "doesn't match top-level metadata measurement_count",
        ),
        (
            "missing top-level measurement_count when section is present",
            Box::new(|name, raw| {
                (name == "metadata.json.zst").then(|| {
                    let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
                    json.as_object_mut().unwrap().remove("measurement_count");
                    serde_json::to_vec(&json).unwrap()
                })
            }),
            "has no measurement_count",
        ),
        (
            "missing measurements_xxh128 when section is present",
            Box::new(|name, raw| {
                (name == "content_hash.json.zst").then(|| {
                    let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
                    json.as_object_mut().unwrap().remove("measurements_xxh128");
                    serde_json::to_vec(&json).unwrap()
                })
            }),
            "has no measurements_xxh128",
        ),
    ];

    for (what, edit, expected) in cases {
        let bad = dir.join("bad.sfmr");
        rewrite_entries(&good, &bad, |n, r| edit(n, r));
        let err = read_sfmr(&bad)
            .err()
            .unwrap_or_else(|| panic!("{what}: reader accepted it"))
            .to_string();
        assert!(
            err.contains(expected),
            "{what}: expected {expected:?} in {err:?}"
        );
        let (valid, errors) = verify_sfmr(&bad).unwrap();
        assert!(!valid, "{what}: verify accepted it");
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "{what}: expected {expected:?} in {errors:?}"
        );
    }

    // 2. Plain file (no measurements/) with stray measurement_count or measurements_xxh128.
    let plain = dir.join("plain.sfmr");
    write_sfmr(&plain, &mut make_test_data()).unwrap();

    let extra_count = dir.join("extra_count.sfmr");
    rewrite_entries(&plain, &extra_count, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["measurement_count"] = serde_json::json!(2);
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert!(read_sfmr(&extra_count)
        .err()
        .unwrap()
        .to_string()
        .contains("no measurements/ section"));
    assert!(read_sfmr_content_hash(&extra_count)
        .err()
        .unwrap()
        .to_string()
        .contains(
            "metadata.json.zst measurement_count and content_hash.json.zst measurements_xxh128 \
             must be present together or absent together"
        ));
    let (valid, errors) = verify_sfmr(&extra_count).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("no measurements/ section"))
    );

    let extra_hash = dir.join("extra_hash.sfmr");
    rewrite_entries(&plain, &extra_hash, |name, raw| {
        (name == "content_hash.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["measurements_xxh128"] = serde_json::json!("0".repeat(32));
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert!(read_sfmr(&extra_hash)
        .err()
        .unwrap()
        .to_string()
        .contains("no measurements/ section"));
    assert!(read_sfmr_content_hash(&extra_hash)
        .err()
        .unwrap()
        .to_string()
        .contains(
            "metadata.json.zst measurement_count and content_hash.json.zst measurements_xxh128 \
             must be present together or absent together"
        ));
    let (valid, errors) = verify_sfmr(&extra_hash).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("no measurements/ section"))
    );

    // 3. Stray measurements/ entry without measurements/metadata.json.zst.
    let stray = dir.join("stray.sfmr");
    {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&good).unwrap()).unwrap();
        let mut out = zip::ZipWriter::new(std::fs::File::create(&stray).unwrap());
        for i in 0..archive.len() {
            let entry = archive.by_index_raw(i).unwrap();
            if entry.name().unwrap() != "measurements/metadata.json.zst" {
                out.raw_copy_file(entry).unwrap();
            }
        }
        out.finish().unwrap();
    }
    assert!(read_sfmr(&stray)
        .err()
        .unwrap()
        .to_string()
        .contains("without measurements/metadata.json.zst"));
    let (valid, errors) = verify_sfmr(&stray).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("without measurements/metadata.json.zst"))
    );

    // 4. measurements/metadata.json with measurement_count == 0, while
    //    top-level measurement_count and measurements_xxh128 are present so
    //    read_sfmr reaches the section_count == 0 guard.
    let zero_count = dir.join("zero_count.sfmr");
    {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&good).unwrap()).unwrap();
        let mut out = zip::ZipWriter::new(std::fs::File::create(&zero_count).unwrap());
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for i in 0..archive.len() {
            let entry = archive.by_index_raw(i).unwrap();
            if !entry.name().unwrap().starts_with("measurements/") {
                out.raw_copy_file(entry).unwrap();
            }
        }
        let add_entry = |zip: &mut zip::ZipWriter<std::fs::File>, name: &str, bytes: &[u8]| {
            zip.start_file(name, stored).unwrap();
            zip.write_all(&zstd::bulk::compress(bytes, 3).unwrap())
                .unwrap();
        };
        add_entry(
            &mut out,
            "measurements/metadata.json.zst",
            br#"{"measurement_count":0,"unit_names":["mm"]}"#,
        );
        add_entry(&mut out, "measurements/distances.0.float32.zst", &[]);
        add_entry(&mut out, "measurements/endpoints_xy.0.2.2.float32.zst", &[]);
        add_entry(&mut out, "measurements/image_indexes.0.2.uint32.zst", &[]);
        add_entry(&mut out, "measurements/labels.json.zst", b"[]");
        add_entry(&mut out, "measurements/units.0.uint8.zst", &[]);
        out.finish().unwrap();
    }
    let err = read_sfmr(&zero_count).err().unwrap().to_string();
    assert!(
        err.contains("measurements/metadata.json.zst measurement_count must be >= 1"),
        "{err}"
    );
    let (valid, errors) = verify_sfmr(&zero_count).unwrap();
    assert!(
        !valid
            && errors.iter().any(
                |e| e.contains("measurements/metadata.json.zst measurement_count must be >= 1")
            ),
        "{errors:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn top_level_zero_measurement_count_is_refused_by_every_reader_and_verifier() {
    let dir = temp_dir("sfmr_test_top_level_zero_measurement_count");
    let plain = dir.join("plain.sfmr");
    write_sfmr(&plain, &mut make_test_data()).unwrap();

    let zero = dir.join("zero.sfmr");
    rewrite_entries(&plain, &zero, |name, raw| {
        (name == "metadata.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["measurement_count"] = serde_json::json!(0);
            serde_json::to_vec(&json).unwrap()
        })
    });

    let expected = "metadata.json.zst measurement_count must be >= 1 when present";
    for (reader, error) in [
        ("read_sfmr", read_sfmr(&zero).err()),
        ("read_sfmr_metadata", read_sfmr_metadata(&zero).err()),
        ("read_sfmr_image_names", read_sfmr_image_names(&zero).err()),
        (
            "read_sfmr_content_hash",
            read_sfmr_content_hash(&zero).err(),
        ),
    ] {
        let err = error
            .unwrap_or_else(|| panic!("{reader}: accepted top-level measurement_count == 0"))
            .to_string();
        assert!(
            err.contains(expected),
            "{reader}: expected {expected:?} in {err:?}"
        );
    }
    let (valid, errors) = verify_sfmr(&zero).unwrap();
    assert!(
        !valid,
        "verify_sfmr accepted top-level measurement_count == 0"
    );
    assert!(
        errors.iter().any(|e| e.contains(expected)),
        "expected {expected:?} in {errors:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn version_16_reads_and_verifies_without_measurements_and_refuses_measurements() {
    let dir = temp_dir("sfmr_test_v16_measurements");

    // A valid version 16 file (no measurements) reads, verifies, and rewrites as version 17.
    let v17_plain = dir.join("v17_plain.sfmr");
    let v16_raw = dir.join("v16_raw.sfmr");
    let v16_plain = dir.join("v16_plain.sfmr");
    write_sfmr(&v17_plain, &mut make_test_data()).unwrap();
    relabel_version(&v17_plain, &v16_raw, 16);
    restamp_metadata_and_measurements_hashes(&v16_raw, &v16_plain);

    assert_eq!(read_sfmr_metadata(&v16_plain).unwrap().version, 16);
    assert_eq!(read_sfmr_image_names(&v16_plain).unwrap().0.version, 16);
    assert!(read_sfmr_content_hash(&v16_plain)
        .unwrap()
        .measurements_xxh128
        .is_none());
    let mut loaded = read_sfmr(&v16_plain).unwrap();
    assert_eq!(loaded.metadata.version, 16);
    assert!(loaded.metadata.measurement_count.is_none());
    assert!(loaded.measurements.is_none());
    assert_eq!(verify_sfmr(&v16_plain).unwrap(), (true, vec![]));

    // Writing what was read produces version 17.
    let upgraded = dir.join("upgraded.sfmr");
    write_sfmr(&upgraded, &mut loaded).unwrap();
    assert_eq!(read_sfmr_metadata(&upgraded).unwrap().version, 17);
    assert_eq!(verify_sfmr(&upgraded).unwrap(), (true, vec![]));

    // 1. A version 16 file carrying measurements/ entries (relabelled from a v17 with measurements)
    //    is refused by every reader and by verify_sfmr.
    let v17_meas = dir.join("v17_meas.sfmr");
    let mut with_meas = make_test_data();
    with_meas.measurements = Some(sample_measurements(2));
    write_sfmr(&v17_meas, &mut with_meas).unwrap();

    let v16_with_all = dir.join("v16_with_all.sfmr");
    relabel_version(&v17_meas, &v16_with_all, 16);
    assert!(read_sfmr(&v16_with_all)
        .err()
        .unwrap()
        .to_string()
        .contains("must not carry measurement_count"));
    assert!(read_sfmr_metadata(&v16_with_all).is_err());
    assert!(read_sfmr_image_names(&v16_with_all).is_err());
    assert!(read_sfmr_content_hash(&v16_with_all).is_err());
    let (valid, errors) = verify_sfmr(&v16_with_all).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("must not carry measurement_count")),
        "{errors:?}"
    );

    // 2. A version 16 file carrying measurements_xxh128 alone is refused.
    let v16_with_hash = dir.join("v16_with_hash.sfmr");
    rewrite_entries(&v16_plain, &v16_with_hash, |name, raw| {
        (name == "content_hash.json.zst").then(|| {
            let mut json: serde_json::Value = serde_json::from_slice(raw).unwrap();
            json["measurements_xxh128"] = serde_json::json!("0".repeat(32));
            serde_json::to_vec(&json).unwrap()
        })
    });
    assert!(read_sfmr(&v16_with_hash)
        .err()
        .unwrap()
        .to_string()
        .contains("must not carry measurements_xxh128"));
    assert!(read_sfmr_content_hash(&v16_with_hash).is_err());
    let (valid, errors) = verify_sfmr(&v16_with_hash).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("must not carry measurements_xxh128")),
        "{errors:?}"
    );

    // 3. A version 16 file carrying a measurements/ entry without measurement_count or
    //    measurements_xxh128 is refused by every reader and by verify_sfmr.
    let v16_with_entry = dir.join("v16_with_entry.sfmr");
    {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&v16_plain).unwrap()).unwrap();
        let mut out = zip::ZipWriter::new(std::fs::File::create(&v16_with_entry).unwrap());
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for i in 0..archive.len() {
            let entry = archive.by_index_raw(i).unwrap();
            out.raw_copy_file(entry).unwrap();
        }
        out.start_file("measurements/metadata.json.zst", stored)
            .unwrap();
        out.write_all(
            &zstd::bulk::compress(br#"{"measurement_count":1,"unit_names":["m"]}"#, 3).unwrap(),
        )
        .unwrap();
        out.finish().unwrap();
    }
    assert!(read_sfmr(&v16_with_entry)
        .err()
        .unwrap()
        .to_string()
        .contains("must not carry measurements/ entries"));
    assert!(read_sfmr_metadata(&v16_with_entry).is_err());
    assert!(read_sfmr_image_names(&v16_with_entry).is_err());
    assert!(read_sfmr_content_hash(&v16_with_entry).is_err());
    let (valid, errors) = verify_sfmr(&v16_with_entry).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("must not carry measurements/ entries")),
        "{errors:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
