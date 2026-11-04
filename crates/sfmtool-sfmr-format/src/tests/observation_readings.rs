// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The eight optional observation-reading columns in `tracks/`, and the
//! change of five of them from `float32` to `float16` in version 16.

use super::{
    float16_payload, float32_payload, make_test_data, rewrite_as_version_15, rewrite_entries,
    rewrite_renaming, rewrite_without_meta_key,
};
use crate::*;

const OPTIONS: ObservationReadingOptions = ObservationReadingOptions {
    resolution: 24,
    sampler: ReadingSampler::PerView,
    score_window: ReadingWindow::GaussianDisk,
    score_window_sigma: Some(0.6),
    max_radius: 3,
    flat_floor: 0.5,
    noise: 2.0,
    relative_tolerance: 0.05,
    anisotropic_threshold: Some(1.5),
};

/// A row whose every value says which observation it is: `tag` is unique per
/// observation. The five values stored as `float16` are `float16` values, so
/// a round trip gives the row back exactly; the scores are not.
fn tagged_row(tag: f32) -> ObservationReading {
    ObservationReading {
        ellipse_axes: [tag, tag / 2.0],
        ellipse_axes_is_at_least: [(tag as u32).is_multiple_of(2), false],
        ellipse_major_angle: tag / 16.0,
        cos_view_angle: 0.875,
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
    "tracks/zncc_self_similarity_cos_view_angle.8.float16.zst",
    "tracks/zncc_self_similarity_ellipse_axes.8.2.float16.zst",
    "tracks/zncc_self_similarity_ellipse_axes_is_at_least.8.2.uint8.zst",
    "tracks/zncc_self_similarity_ellipse_major_angle.8.float16.zst",
    "tracks/zncc_self_similarity_tilt_angle.8.float16.zst",
    "tracks/zncc_self_similarity_zoom.8.2.float16.zst",
];

/// The entries a version 15 file stores the same columns in.
const ENTRIES_V15: [&str; 8] = [
    "tracks/blur_matched_bitmap_zncc.8.float32.zst",
    "tracks/plain_bitmap_zncc.8.float32.zst",
    "tracks/zncc_self_similarity_cos_view_angle.8.float32.zst",
    "tracks/zncc_self_similarity_ellipse_axes.8.2.float32.zst",
    "tracks/zncc_self_similarity_ellipse_axes_is_at_least.8.2.uint8.zst",
    "tracks/zncc_self_similarity_ellipse_major_angle.8.float32.zst",
    "tracks/zncc_self_similarity_tilt_angle.8.float32.zst",
    "tracks/zncc_self_similarity_zoom.8.2.float32.zst",
];

fn archive_names(path: &std::path::Path) -> Vec<String> {
    super::archive_names(path)
}

/// Every float of `row`, compared bit for bit with `NaN` as `NaN`.
fn row_bits(row: &ObservationReading) -> Vec<u32> {
    [
        row.ellipse_axes[0],
        row.ellipse_axes[1],
        row.ellipse_major_angle,
        row.cos_view_angle,
        row.tilt_angle,
        row.zoom[0],
        row.zoom[1],
        row.plain_bitmap_zncc,
        row.blur_matched_bitmap_zncc,
    ]
    .iter()
    .map(|v| {
        if v.is_nan() {
            f32::NAN.to_bits()
        } else {
            v.to_bits()
        }
    })
    .collect()
}

/// `row` with its five `float16` values rounded the way a version 16 writer
/// rounds them, computed here from `half` rather than by the crate.
fn rounded(row: ObservationReading) -> ObservationReading {
    let r = |v: f32| half::f16::from_f32(v).to_f32();
    ObservationReading {
        ellipse_axes: row.ellipse_axes.map(r),
        ellipse_major_angle: r(row.ellipse_major_angle),
        cos_view_angle: r(row.cos_view_angle),
        tilt_angle: r(row.tilt_angle),
        zoom: row.zoom.map(r),
        ..row
    }
}

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
        (name == "tracks/zncc_self_similarity_zoom.8.2.float16.zst").then(|| raw[..28].to_vec())
    });
    assert!(read_sfmr(&short).is_err());
    let (valid, errors) = verify_sfmr(&short).unwrap();
    assert!(!valid);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("zncc_self_similarity_zoom holds 28 bytes")),
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

#[test]
fn the_tracks_hash_is_over_the_entries_in_sorted_name_order() {
    // Taken from the archive alone: every `tracks/` entry, sorted by name,
    // decompressed and hashed in that order. It must be what the writer
    // stored, whatever order the writer's own table lists the columns in.
    use std::io::Read;
    let mut data = make_test_data();
    let rows: Vec<ObservationReading> = (0..8).map(|j| tagged_row(j as f32 + 1.0)).collect();
    data.observation_readings = Some(readings_for(&rows));
    let dir = temp_dir("sfmr_test_observation_readings_hash_order");
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
    let mut names: Vec<String> = archive
        .file_names()
        .map(Result::unwrap)
        .filter(|n| n.starts_with("tracks/"))
        .map(String::from)
        .collect();
    names.sort();
    assert_eq!(names.len(), 5 + 8, "{names:?}");
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
    let stored = read_sfmr(&path).unwrap().content_hash.tracks_xxh128;
    assert_eq!(sfmtool_archive_io::format_hash(hasher.digest128()), stored);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_column_in_another_layout_is_written_in_row_major_order() {
    // A reversed view, as numpy hands over `a[::-1]`, writes its values in
    // their logical order rather than panicking.
    let mut data = make_test_data();
    let rows: Vec<ObservationReading> = (0..8).map(|j| tagged_row(j as f32 + 1.0)).collect();
    let mut readings = readings_for(&rows);
    let reversed: Vec<f32> = rows.iter().rev().map(|r| r.plain_bitmap_zncc).collect();
    let mut column = ndarray::Array1::from_vec(reversed);
    column.invert_axis(ndarray::Axis(0));
    assert!(column.as_slice().is_none());
    readings.plain_bitmap_zncc = column;
    data.observation_readings = Some(readings);
    let dir = temp_dir("sfmr_test_observation_readings_layout");
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    assert_eq!(
        read_sfmr(&path)
            .unwrap()
            .observation_readings
            .unwrap()
            .rows(),
        rows
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Rows none of whose five `float16` values is a `float16`, with the edge
/// cases a writer has to round: an angle just below `π`, a `cos θ` just below
/// 1, a zoom past the largest `float16`, and the `NaN` of nothing measured.
fn unrounded_rows() -> Vec<ObservationReading> {
    let mut rows: Vec<ObservationReading> = (0..8)
        .map(|j| {
            let t = j as f32 + 1.0;
            ObservationReading {
                ellipse_axes: [0.3 * t, 0.1 * t],
                ellipse_axes_is_at_least: [false, j == 2],
                ellipse_major_angle: 0.37 * t,
                cos_view_angle: 0.9 - 0.01 * t,
                tilt_angle: 0.29 * t,
                zoom: [0.7 * t, 1.3 * t],
                plain_bitmap_zncc: 0.981 - 0.001 * t,
                blur_matched_bitmap_zncc: 0.987 - 0.001 * t,
            }
        })
        .collect();
    rows[1].ellipse_major_angle = std::f32::consts::PI - 1.0e-6;
    rows[1].cos_view_angle = 0.999_9;
    rows[1].tilt_angle = f32::NAN;
    rows[4].zoom = [65_519.0, 70_000.0];
    rows[6] = ObservationReading::NOT_MEASURED;
    rows
}

#[test]
fn the_writer_rounds_five_columns_to_float16_in_the_value_it_writes() {
    let dir = temp_dir("sfmr_test_observation_readings_float16");
    let path = dir.join("rounded.sfmr");
    let rows = unrounded_rows();
    let unrounded = || {
        let mut data = make_test_data();
        data.observation_readings = Some(readings_for(&rows));
        data
    };
    let mut data = unrounded();
    write_sfmr(&path, &mut data).unwrap();

    // The value now holds the rounded rows: each float16 column at the
    // nearest float16, the scores and the flags as they were.
    let written = data.observation_readings.as_ref().unwrap().rows();
    for (j, (got, given)) in written.iter().zip(&rows).enumerate() {
        assert_eq!(row_bits(got), row_bits(&rounded(*given)), "row {j}");
        assert_eq!(got.ellipse_axes_is_at_least, given.ellipse_axes_is_at_least);
        assert_eq!(
            got.plain_bitmap_zncc.to_bits(),
            given.plain_bitmap_zncc.to_bits(),
            "row {j}"
        );
    }
    assert_ne!(written[0].ellipse_axes[0], rows[0].ellipse_axes[0]);
    // An angle just below π stays below it, in [0, π).
    assert!(written[1].ellipse_major_angle < std::f32::consts::PI);
    assert_eq!(written[1].ellipse_major_angle, 3.140625);
    // A cos θ just below 1 rounds to 1, the nearest float16.
    assert_eq!(written[1].cos_view_angle, 1.0);
    // The nothing-measured row and the direction-less tilt keep their NaN.
    assert!(!written[6].is_measured());
    assert!(written[1].tilt_angle.is_nan());
    // Under 65,520 rounds to the largest float16, at or past it to infinity.
    assert_eq!(written[4].zoom, [65_504.0, f32::INFINITY]);

    // The file gives back what the value holds, and the entries are float16.
    let mut loaded = read_sfmr(&path).unwrap();
    let read_back = loaded.observation_readings.as_ref().unwrap().rows();
    for (j, (got, held)) in read_back.iter().zip(&written).enumerate() {
        assert_eq!(row_bits(got), row_bits(held), "row {j}");
    }
    let names = archive_names(&path);
    for name in ENTRIES {
        assert!(names.contains(&name.to_string()), "{name} missing");
    }

    // Hashing the unrounded value gives the stored hash, and saving what was
    // read writes the same hash again.
    let stored = read_sfmr_content_hash(&path).unwrap();
    assert_eq!(
        content_hash_of(&mut unrounded(), &WriteOptions::default())
            .unwrap()
            .content_xxh128,
        stored.content_xxh128
    );
    let resaved = dir.join("resaved.sfmr");
    write_sfmr(&resaved, &mut loaded).unwrap();
    assert_eq!(
        read_sfmr_content_hash(&resaved).unwrap().content_xxh128,
        stored.content_xxh128
    );
    let (valid, errors) = verify_sfmr(&resaved).unwrap();
    assert!(valid, "{errors:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_version_15_file_reads_its_float32_readings_and_saves_them_as_float16() {
    let dir = temp_dir("sfmr_test_observation_readings_v15");
    let current = dir.join("current.sfmr");
    let rows = unrounded_rows();
    let mut data = make_test_data();
    data.observation_readings = Some(readings_for(&rows));
    write_sfmr(&current, &mut data).unwrap();

    // The version 15 file of the unrounded rows: its writer stored each
    // float column as float32, at the values it was given.
    let given = readings_for(&rows);
    let old = dir.join("v15.sfmr");
    rewrite_as_version_15(&current, &old, &dir, |name, values| {
        let column = |a: &[f32]| a.to_vec();
        let full = match name {
            "tracks/zncc_self_similarity_cos_view_angle.8.float32.zst" => column(
                given
                    .zncc_self_similarity_cos_view_angle
                    .as_slice()
                    .unwrap(),
            ),
            "tracks/zncc_self_similarity_ellipse_axes.8.2.float32.zst" => {
                column(given.zncc_self_similarity_ellipse_axes.as_slice().unwrap())
            }
            "tracks/zncc_self_similarity_ellipse_major_angle.8.float32.zst" => column(
                given
                    .zncc_self_similarity_ellipse_major_angle
                    .as_slice()
                    .unwrap(),
            ),
            "tracks/zncc_self_similarity_tilt_angle.8.float32.zst" => {
                column(given.zncc_self_similarity_tilt_angle.as_slice().unwrap())
            }
            "tracks/zncc_self_similarity_zoom.8.2.float32.zst" => {
                column(given.zncc_self_similarity_zoom.as_slice().unwrap())
            }
            _ => return,
        };
        *values = full;
    });
    let names = archive_names(&old);
    for name in ENTRIES_V15 {
        assert!(names.contains(&name.to_string()), "{name} missing");
    }
    let (valid, errors) = verify_sfmr(&old).unwrap();
    assert!(valid, "{errors:?}");

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 15);
    let read_back = loaded.observation_readings.as_ref().unwrap().rows();
    for (j, (got, given)) in read_back.iter().zip(&rows).enumerate() {
        assert_eq!(row_bits(got), row_bits(given), "row {j}");
    }

    let saved = dir.join("saved.sfmr");
    write_sfmr(&saved, &mut loaded).unwrap();
    let names = archive_names(&saved);
    for name in ENTRIES {
        assert!(names.contains(&name.to_string()), "{name} missing");
    }
    let reloaded = read_sfmr(&saved).unwrap();
    assert_eq!(reloaded.metadata.version, SFMR_FORMAT_VERSION);
    for (j, (got, given)) in reloaded
        .observation_readings
        .unwrap()
        .rows()
        .iter()
        .zip(&rows)
        .enumerate()
    {
        assert_eq!(row_bits(got), row_bits(&rounded(*given)), "row {j}");
    }
    let old_hash = read_sfmr_content_hash(&old).unwrap();
    let new_hash = read_sfmr_content_hash(&saved).unwrap();
    assert_ne!(new_hash.tracks_xxh128, old_hash.tracks_xxh128);
    assert_eq!(new_hash.points3d_xxh128, old_hash.points3d_xxh128);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_reading_column_in_the_dtype_of_another_version_is_refused() {
    let dir = temp_dir("sfmr_test_observation_readings_wrong_dtype");
    let current = dir.join("current.sfmr");
    let mut data = make_test_data();
    data.observation_readings = Some(readings_for(&[tagged_row(1.0); 8]));
    write_sfmr(&current, &mut data).unwrap();
    let assert_refused = |bad: &std::path::Path, needle: &str| {
        let err = read_sfmr(bad)
            .err()
            .expect("the reader refuses it")
            .to_string();
        assert!(err.contains(needle), "reader: {err}");
        let (valid, errors) = verify_sfmr(bad).unwrap();
        assert!(!valid);
        assert!(
            errors.iter().any(|e| e.contains(needle)),
            "verifier: {errors:?}"
        );
    };

    // A current file holding the tilt angle as float32.
    let float16_name = "tracks/zncc_self_similarity_tilt_angle.8.float16.zst";
    let float32_name = "tracks/zncc_self_similarity_tilt_angle.8.float32.zst";
    let bad = dir.join("float32_in_current.sfmr");
    rewrite_renaming(&current, &bad, |name, raw| {
        (name == float16_name).then(|| {
            (
                float32_name.to_string(),
                float32_payload(&float16_payload(raw)),
            )
        })
    });
    assert_refused(
        &bad,
        &format!(
            "a version {SFMR_FORMAT_VERSION} file stores {float16_name}, and this one holds \
             {float32_name} instead"
        ),
    );

    // A version 15 file holding the zoom as float16.
    let old = dir.join("v15.sfmr");
    rewrite_as_version_15(&current, &old, &dir, |_, _| {});
    let float16_name = "tracks/zncc_self_similarity_zoom.8.2.float16.zst";
    let float32_name = "tracks/zncc_self_similarity_zoom.8.2.float32.zst";
    let bad = dir.join("float16_in_v15.sfmr");
    rewrite_renaming(&old, &bad, |name, raw| {
        (name == float32_name).then(|| {
            let bits: Vec<u8> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|b| half::f16::from_f32(f32::from_le_bytes(*b)).to_le_bytes())
                .collect();
            (float16_name.to_string(), bits)
        })
    });
    assert_refused(
        &bad,
        &format!(
            "a version 15 file stores {float32_name}, and this one holds {float16_name} instead"
        ),
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
