// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The eight optional observation-reading columns in `tracks/`.

use super::{make_test_data, rewrite_entries, rewrite_without_meta_key};
use crate::*;

const OPTIONS: ObservationReadingOptions = ObservationReadingOptions {
    max_radius: 3,
    flat_floor: 0.5,
    noise: 2.0,
    relative_tolerance: 0.05,
    anisotropic_threshold: Some(1.5),
};

/// A row whose every value says which observation it is: `tag` is unique per
/// observation.
fn tagged_row(tag: f32) -> ObservationReading {
    ObservationReading {
        ellipse_axes: [tag, tag / 2.0],
        ellipse_axes_is_at_least: [(tag as u32).is_multiple_of(2), false],
        ellipse_major_angle: 0.1 * tag,
        cos_view_angle: 0.9,
        tilt_angle: f32::NAN,
        zoom: [0.5, 0.75],
        plain_bitmap_zncc: 0.01 * tag,
        blur_matched_bitmap_zncc: 0.02 * tag,
    }
}

fn readings_for(rows: &[ObservationReading]) -> ObservationReadingColumns {
    ObservationReadingColumns::from_rows(rows, OPTIONS)
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tracks_metadata(path: &std::path::Path) -> serde_json::Value {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut compressed = Vec::new();
    archive
        .by_name("tracks/metadata.json.zst")
        .unwrap()
        .read_to_end(&mut compressed)
        .unwrap();
    serde_json::from_slice(&zstd::stream::decode_all(&compressed[..]).unwrap()).unwrap()
}

const ENTRIES: [&str; 8] = [
    "tracks/blur_matched_bitmap_zncc.8.float32.zst",
    "tracks/plain_bitmap_zncc.8.float32.zst",
    "tracks/zncc_self_similarity_cos_view_angle.8.float32.zst",
    "tracks/zncc_self_similarity_ellipse_axes.8.2.float32.zst",
    "tracks/zncc_self_similarity_ellipse_axes_is_at_least.8.2.uint8.zst",
    "tracks/zncc_self_similarity_ellipse_major_angle.8.float32.zst",
    "tracks/zncc_self_similarity_tilt_angle.8.float32.zst",
    "tracks/zncc_self_similarity_zoom.8.2.float32.zst",
];

#[test]
fn readings_round_trip_and_verify() {
    let mut data = make_test_data();
    let mut rows: Vec<ObservationReading> = (0..8).map(|j| tagged_row(j as f32 + 1.0)).collect();
    rows[3] = ObservationReading::NOT_MEASURED;
    data.observation_readings = Some(readings_for(&rows));

    let dir = temp_dir("sfmr_test_observation_readings_round_trip");
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let loaded = read_sfmr(&path).unwrap();
    let readings = loaded.observation_readings.unwrap();
    assert_eq!(readings.rows(), rows);
    assert_eq!(readings.options, OPTIONS);

    let meta = tracks_metadata(&path);
    assert_eq!(meta[HAS_OBSERVATION_READINGS], true);
    assert_eq!(meta[OBSERVATION_READING_OPTIONS]["max_radius"], 3);
    assert_eq!(
        meta[OBSERVATION_READING_OPTIONS]["anisotropic_threshold"],
        1.5
    );

    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
    for name in ENTRIES {
        assert!(archive.by_name(name).is_ok(), "{name} missing");
    }
    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "verification failed: {errors:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn absent_readings_write_no_entry_and_no_flag() {
    let mut data = make_test_data();
    let dir = temp_dir("sfmr_test_observation_readings_absent");
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    assert!(read_sfmr(&path).unwrap().observation_readings.is_none());
    let meta = tracks_metadata(&path);
    assert!(meta.get(HAS_OBSERVATION_READINGS).is_none());
    assert!(meta.get(OBSERVATION_READING_OPTIONS).is_none());
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
    for name in ENTRIES {
        assert!(archive.by_name(name).is_err(), "{name} present");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_file_from_before_the_columns_loads_without_them() {
    // A file with readings, its flag removed: the shape of every older file,
    // whose tracks metadata names no such key.
    let mut data = make_test_data();
    data.observation_readings = Some(readings_for(&[tagged_row(1.0); 8]));
    let dir = temp_dir("sfmr_test_observation_readings_legacy");
    let path = dir.join("test.sfmr");
    let legacy = dir.join("legacy.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    rewrite_without_meta_key(
        &path,
        &legacy,
        "tracks/metadata.json.zst",
        HAS_OBSERVATION_READINGS,
    );
    assert!(read_sfmr(&legacy).unwrap().observation_readings.is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_flag_without_options_is_refused() {
    let mut data = make_test_data();
    data.observation_readings = Some(readings_for(&[tagged_row(1.0); 8]));
    let dir = temp_dir("sfmr_test_observation_readings_no_options");
    let path = dir.join("test.sfmr");
    let edited = dir.join("edited.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    rewrite_without_meta_key(
        &path,
        &edited,
        "tracks/metadata.json.zst",
        OBSERVATION_READING_OPTIONS,
    );
    let Err(err) = read_sfmr(&edited) else {
        panic!("a flag without options was read");
    };
    assert!(
        format!("{err}").contains(OBSERVATION_READING_OPTIONS),
        "{err}"
    );
    let (valid, _) = verify_sfmr(&edited).unwrap();
    assert!(!valid);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_shape_mismatch_is_refused_on_write() {
    let mut data = make_test_data();
    let mut readings = readings_for(&[tagged_row(1.0); 8]);
    readings.plain_bitmap_zncc = ndarray::Array1::from_vec(vec![0.5; 7]);
    data.observation_readings = Some(readings);
    let dir = temp_dir("sfmr_test_observation_readings_shape_write");
    let err = write_sfmr(&dir.join("test.sfmr"), &mut data).unwrap_err();
    assert!(
        format!("{err}").contains("plain_bitmap_zncc len 7 != observation_count 8"),
        "{err}"
    );

    let mut data = make_test_data();
    let mut readings = readings_for(&[tagged_row(1.0); 8]);
    readings.zncc_self_similarity_ellipse_axes_is_at_least[[0, 0]] = 2;
    data.observation_readings = Some(readings);
    let err = write_sfmr(&dir.join("test.sfmr"), &mut data).unwrap_err();
    assert!(format!("{err}").contains("not 0 or 1"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_shape_mismatch_is_refused_on_read_and_verify() {
    let mut data = make_test_data();
    data.observation_readings = Some(readings_for(&[tagged_row(1.0); 8]));
    let dir = temp_dir("sfmr_test_observation_readings_shape_read");
    let path = dir.join("test.sfmr");
    let short = dir.join("short.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    // One row fewer than the entry name and the metadata say.
    rewrite_entries(&path, &short, |name, raw| {
        (name == "tracks/zncc_self_similarity_zoom.8.2.float32.zst").then(|| raw[..56].to_vec())
    });
    assert!(read_sfmr(&short).is_err());
    let (valid, errors) = verify_sfmr(&short).unwrap();
    assert!(!valid);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("zncc_self_similarity_zoom holds 56 bytes")),
        "{errors:?}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_column_is_part_of_the_tracks_hash() {
    let dir = temp_dir("sfmr_test_observation_readings_hash");
    let rows: Vec<ObservationReading> = (0..8).map(|j| tagged_row(j as f32 + 1.0)).collect();
    let hash_of = |rows: &[ObservationReading], name: &str| {
        let mut data = make_test_data();
        data.observation_readings = Some(readings_for(rows));
        let path = dir.join(name);
        write_sfmr(&path, &mut data).unwrap();
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "{errors:?}");
        read_sfmr(&path).unwrap().content_hash.tracks_xxh128
    };
    let base = hash_of(&rows, "base.sfmr");
    let edits: [fn(&mut ObservationReading); 8] = [
        |r| r.ellipse_axes[1] += 0.5,
        |r| r.ellipse_axes_is_at_least[1] = true,
        |r| r.ellipse_major_angle += 0.5,
        |r| r.cos_view_angle -= 0.5,
        |r| r.tilt_angle = 1.0,
        |r| r.zoom[0] += 0.5,
        |r| r.plain_bitmap_zncc -= 0.5,
        |r| r.blur_matched_bitmap_zncc -= 0.5,
    ];
    for (k, edit) in edits.iter().enumerate() {
        let mut changed = rows.clone();
        edit(&mut changed[2]);
        assert_ne!(
            base,
            hash_of(&changed, &format!("edit{k}.sfmr")),
            "edit {k}"
        );
    }
    let mut data = make_test_data();
    let path = dir.join("none.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    assert_ne!(base, read_sfmr(&path).unwrap().content_hash.tracks_xxh128);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rows_follow_their_observation_through_the_sort() {
    let mut d = make_test_data();
    let pts = [2u32, 0, 4, 1, 0, 3, 1, 1];
    let imgs = [0u32, 0, 2, 0, 1, 1, 1, 2];
    d.point_indexes = ndarray::Array1::from_vec(pts.to_vec());
    d.image_indexes = ndarray::Array1::from_vec(imgs.to_vec());
    d.observation_counts = ndarray::Array1::from_vec(vec![2, 3, 1, 1, 1]);
    let tag = |p: u32, i: u32| (1 + p * 8 + i) as f32;
    let rows: Vec<ObservationReading> = (0..8).map(|j| tagged_row(tag(pts[j], imgs[j]))).collect();
    d.observation_readings = Some(readings_for(&rows));

    let dir = temp_dir("sfmr_test_observation_readings_sort");
    let path = dir.join("u.sfmr");
    write_sfmr(&path, &mut d).unwrap();
    let loaded = read_sfmr(&path).unwrap();
    let readings = loaded.observation_readings.unwrap();
    for j in 0..8 {
        assert_eq!(
            readings.row(j),
            tagged_row(tag(loaded.point_indexes[j], loaded.image_indexes[j])),
            "row {j} misaligned"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
