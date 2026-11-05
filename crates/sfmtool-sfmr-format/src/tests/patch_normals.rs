// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The normals of version 16: a finite point with a patch has the normal of
//! its frame, `normalize(u × v)`, which the file derives rather than stores.

use super::{archive_names, make_test_data, restamp_hashes, rewrite_renaming};
use crate::*;
use ndarray::{Array1, Array2};

/// `normalize(u × v)` in `f64`, the format's derived normal computed another
/// way, for comparing with what the crate derives in `f32`.
pub(super) fn cross_normal_f64(u: Vec<f32>, v: Vec<f32>) -> [f64; 3] {
    let (u, v) = (u.map_f64(), v.map_f64());
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / len, n[1] / len, n[2] / len]
}

trait MapF64 {
    fn map_f64(self) -> [f64; 3];
}

impl MapF64 for Vec<f32> {
    fn map_f64(self) -> [f64; 3] {
        [self[0].into(), self[1].into(), self[2].into()]
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The test value with a patch frame on five points of each kind: points 0
/// and 1 finite with a patch, point 2 finite with a degenerate frame (`v`
/// along `u`), point 3 finite without a patch, and point 4 at infinity with a
/// patch. Every stored normal is one no frame gives, so a test sees which
/// rows were derived.
fn data_with_every_kind_of_point() -> SfmrData {
    let mut data = make_test_data();
    data.positions_xyzw[[4, 3]] = 0.0;
    data.patch_u_halfvec_xyz = Some(
        Array2::from_shape_vec(
            (5, 3),
            vec![
                0.2, 0.0, 0.0, //
                0.0, 0.1, 0.05, //
                0.3, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.1,
            ],
        )
        .unwrap(),
    );
    data.patch_v_halfvec_xyz = Some(
        Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.3, 0.0, //
                0.1, 0.0, 0.02, //
                0.6, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.1, 0.0, 0.0,
            ],
        )
        .unwrap(),
    );
    data.reference_observations = Some(Array1::from_vec(vec![0, 0, 0, -1, 0]));
    data.normals_xyz = Some(
        Array2::from_shape_vec(
            (5, 3),
            vec![
                0.6, 0.8, 0.0, //
                0.0, 0.6, 0.8, //
                0.8, 0.0, 0.6, //
                0.0, 0.8, 0.6, //
                0.0, 0.0, 0.0,
            ],
        )
        .unwrap(),
    );
    data
}

const FULL_ENTRY: &str = "points3d/normals_xyz.5.3.float32.zst";
const UNDERIVED_ENTRY: &str = "points3d/underived_normals_xyz.2.3.float32.zst";

/// The values of the entry `name` of the archive at `path`.
fn entry_floats(path: &std::path::Path, name: &str) -> Vec<f32> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut compressed = Vec::new();
    archive
        .by_name(name)
        .unwrap()
        .read_to_end(&mut compressed)
        .unwrap();
    zstd::stream::decode_all(&compressed[..])
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn floats_payload(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn row(a: &Array2<f32>, i: usize) -> Vec<f32> {
    a.row(i).to_vec()
}

#[test]
fn only_the_normals_the_frame_does_not_give_are_stored() {
    let dir = temp_dir("sfmr_test_patch_normals_round_trip");
    let path = dir.join("normals.sfmr");
    let given = data_with_every_kind_of_point();
    let given_normals = given.normals_xyz.clone().unwrap();
    let (u, v) = (
        given.patch_u_halfvec_xyz.clone().unwrap(),
        given.patch_v_halfvec_xyz.clone().unwrap(),
    );
    let mut data = given;
    write_sfmr(&path, &mut data).unwrap();

    // The file holds no full column, and the subset holds the normals of
    // points 2 (degenerate frame) and 3 (no patch), in point order; points 0
    // and 1 are derived and point 4 is at infinity.
    let names = archive_names(&path);
    assert!(!names.contains(&FULL_ENTRY.to_string()), "{names:?}");
    assert_eq!(
        entry_floats(&path, UNDERIVED_ENTRY),
        given_normals.as_slice().unwrap()[6..12].to_vec()
    );
    assert_eq!(super::read_points3d_metadata(&path)["has_normals"], true);

    // A reader fills points 0 and 1 with normalize(u × v), places the stored
    // rows, and gives point 4 its zero row.
    let loaded = read_sfmr(&path).unwrap();
    let normals = loaded.normals_xyz.as_ref().unwrap();
    assert_eq!(normals.shape(), [5, 3]);
    for i in 0..2 {
        let expected = cross_normal_f64(row(&u, i), row(&v, i));
        for k in 0..3 {
            assert!(
                (f64::from(normals[[i, k]]) - expected[k]).abs() < 1e-6,
                "point {i}: {:?} vs {expected:?}",
                row(normals, i)
            );
        }
        assert_ne!(row(normals, i), row(&given_normals, i), "point {i}");
    }
    assert_eq!(row(normals, 0), vec![0.0, 0.0, 1.0]);
    for i in 2..5 {
        assert_eq!(row(normals, i), row(&given_normals, i), "point {i}");
    }

    // The write set the value it was handed to what the file reads back, and
    // a save of what was read writes the same hash and the same entry.
    assert_eq!(data.normals_xyz.as_ref(), Some(normals));
    let stored_hash = read_sfmr_content_hash(&path).unwrap();
    let mut loaded = loaded;
    let resaved = dir.join("resaved.sfmr");
    write_sfmr(&resaved, &mut loaded).unwrap();
    assert_eq!(
        read_sfmr_content_hash(&resaved).unwrap().content_xxh128,
        stored_hash.content_xxh128
    );
    assert_eq!(
        entry_floats(&resaved, UNDERIVED_ENTRY),
        entry_floats(&path, UNDERIVED_ENTRY)
    );
    let (valid, errors) = verify_sfmr(&resaved).unwrap();
    assert!(valid, "{errors:?}");

    // An edit of a derived normal does not reach the file: the hash is the
    // frame's. An edit of a stored one does, and so does turning a frame.
    let options = WriteOptions::default();
    let hash = |edit: fn(&mut SfmrData)| {
        let mut d = data_with_every_kind_of_point();
        edit(&mut d);
        content_hash_of(&mut d, &options).unwrap().content_xxh128
    };
    assert_eq!(
        hash(|d| d.normals_xyz.as_mut().unwrap()[[1, 0]] = 1.0),
        stored_hash.content_xxh128
    );
    assert_ne!(
        hash(|d| d.normals_xyz.as_mut().unwrap()[[3, 0]] = 0.6),
        stored_hash.content_xxh128
    );
    assert_ne!(
        hash(|d| d.patch_v_halfvec_xyz.as_mut().unwrap()[[1, 2]] = 0.07),
        stored_hash.content_xxh128
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_whose_every_normal_is_derived_stores_no_normals_entry() {
    let dir = temp_dir("sfmr_test_patch_normals_all_derived");
    let path = dir.join("all.sfmr");
    let mut data = data_with_every_kind_of_point();
    // Every point finite with a usable frame.
    data.positions_xyzw[[4, 3]] = 1.0;
    {
        let u = data.patch_u_halfvec_xyz.as_mut().unwrap();
        let v = data.patch_v_halfvec_xyz.as_mut().unwrap();
        for i in 2..5 {
            u.row_mut(i).assign(&ndarray::arr1(&[0.1, 0.0, 0.0]));
            v.row_mut(i).assign(&ndarray::arr1(&[0.0, 0.1, 0.0]));
        }
    }
    write_sfmr(&path, &mut data).unwrap();
    let names = archive_names(&path);
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("points3d/underived_normals_xyz.")
                || n.starts_with("points3d/normals_xyz.")),
        "{names:?}"
    );
    // The flag, not an entry, says the file has normals.
    assert_eq!(super::read_points3d_metadata(&path)["has_normals"], true);
    let normals = read_sfmr(&path).unwrap().normals_xyz.unwrap();
    for i in 2..5 {
        assert_eq!(row(&normals, i), vec![0.0, 0.0, 1.0], "point {i}");
    }
    assert_eq!(&normals, data.normals_xyz.as_ref().unwrap());
    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "{errors:?}");

    // Without the normals flag the same frames give no normals at all.
    let mut none = data_with_every_kind_of_point();
    none.normals_xyz = None;
    let path = dir.join("none.sfmr");
    write_sfmr(&path, &mut none).unwrap();
    assert_eq!(super::read_points3d_metadata(&path)["has_normals"], false);
    assert!(read_sfmr(&path).unwrap().normals_xyz.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_point_at_infinity_stores_no_normal_and_reads_back_zero() {
    let dir = temp_dir("sfmr_test_patch_normals_infinity");
    let path = dir.join("infinity.sfmr");
    let mut data = data_with_every_kind_of_point();
    // A normal left on a point at infinity is not a normal the file holds.
    data.normals_xyz.as_mut().unwrap()[[4, 2]] = 1.0;
    write_sfmr(&path, &mut data).unwrap();
    assert_eq!(entry_floats(&path, UNDERIVED_ENTRY).len(), 6);
    assert_eq!(row(data.normals_xyz.as_ref().unwrap(), 4), vec![0.0; 3]);
    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(row(loaded.normals_xyz.as_ref().unwrap(), 4), vec![0.0; 3]);
    let _ = std::fs::remove_dir_all(&dir);
}

fn assert_refused(path: &std::path::Path, needle: &str) {
    let err = read_sfmr(path).err().expect("refused").to_string();
    assert!(err.contains(needle), "reader: {err}");
    let (valid, errors) = verify_sfmr(path).unwrap();
    assert!(!valid);
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "verifier: {errors:?}"
    );
}

#[test]
fn an_entry_of_the_wrong_count_is_refused() {
    let dir = temp_dir("sfmr_test_patch_normals_wrong_count");
    let good = dir.join("good.sfmr");
    write_sfmr(&good, &mut data_with_every_kind_of_point()).unwrap();
    // Three rows where two points' normals are not derived.
    let bad = dir.join("three.sfmr");
    rewrite_renaming(&good, &bad, |name, raw| {
        (name == UNDERIVED_ENTRY).then(|| {
            let mut values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            values.extend([0.0, 1.0, 0.0]);
            (
                "points3d/underived_normals_xyz.3.3.float32.zst".to_string(),
                floats_payload(&values),
            )
        })
    });
    assert_refused(
        &bad,
        "points3d/underived_normals_xyz.3.3.float32.zst does not match the 2 finite points",
    );
    // The right name over the wrong number of bytes.
    let short = dir.join("short.sfmr");
    rewrite_renaming(&good, &short, |name, raw| {
        (name == UNDERIVED_ENTRY).then(|| (name.to_string(), raw[..12].to_vec()))
    });
    let err = read_sfmr(&short).err().expect("refused").to_string();
    assert!(err.contains("underived_normals_xyz"), "{err}");
    let (valid, errors) = verify_sfmr(&short).unwrap();
    assert!(!valid);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("underived_normals_xyz holds 12 bytes")),
        "{errors:?}"
    );
    // No entry where two normals are not derived.
    let missing = dir.join("missing.sfmr");
    rewrite_renaming(&good, &missing, |name, raw| {
        (name == UNDERIVED_ENTRY).then(|| ("points3d/unrelated.bin".to_string(), raw.to_vec()))
    });
    assert_refused(
        &missing,
        "and no points3d/underived_normals_xyz.2.3.float32.zst",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_full_normals_column_in_a_version_16_file_with_the_frame_is_refused() {
    let dir = temp_dir("sfmr_test_patch_normals_full_column");
    let good = dir.join("good.sfmr");
    let mut data = data_with_every_kind_of_point();
    write_sfmr(&good, &mut data).unwrap();
    let full = floats_payload(data.normals_xyz.as_ref().unwrap().as_slice().unwrap());
    let bad = dir.join("full.sfmr");
    rewrite_renaming(&good, &bad, |name, _| {
        (name == UNDERIVED_ENTRY).then(|| (FULL_ENTRY.to_string(), full.clone()))
    });
    assert_refused(
        &bad,
        "and this one holds the full column points3d/normals_xyz.5.3",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn normal_confidence_is_kept_as_stored() {
    let dir = temp_dir("sfmr_test_patch_normals_confidence");
    let path = dir.join("confidence.sfmr");
    let mut data = data_with_every_kind_of_point();
    let confidence = Array1::from_vec(vec![0u8, 7, 0, 9, 0]);
    data.normal_confidence = Some(confidence.clone());
    write_sfmr(&path, &mut data).unwrap();
    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(loaded.normal_confidence, Some(confidence));
    assert_eq!(
        row(loaded.normals_xyz.as_ref().unwrap(), 0),
        vec![0.0, 0.0, 1.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_patch_frame_without_normals_stays_without_them() {
    let dir = temp_dir("sfmr_test_patch_normals_absent");
    let path = dir.join("no_normals.sfmr");
    let mut data = data_with_every_kind_of_point();
    data.normals_xyz = None;
    write_sfmr(&path, &mut data).unwrap();
    assert!(read_sfmr(&path).unwrap().normals_xyz.is_none());
    assert!(data.normals_xyz.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_version_15_file_reads_its_normals_as_stored_and_saves_them_from_the_frame() {
    let dir = temp_dir("sfmr_test_patch_normals_v15");
    let current = dir.join("current.sfmr");
    let given = data_with_every_kind_of_point();
    let given_normals = given.normals_xyz.clone().unwrap();
    let mut data = given;
    write_sfmr(&current, &mut data).unwrap();

    // The version 15 file of the given value: its normals stored as given,
    // disagreeing with the frames of points 0 and 1.
    let old = dir.join("v15.sfmr");
    super::rewrite_as_version_15(&current, &old, &dir, |_, _| {});
    let staged = dir.join("staged.sfmr");
    rewrite_renaming(&old, &staged, |name, _| {
        (name == FULL_ENTRY).then(|| {
            (
                FULL_ENTRY.to_string(),
                floats_payload(given_normals.as_slice().unwrap()),
            )
        })
    });
    std::fs::rename(&staged, &old).unwrap();
    restamp_hashes(&old, &dir);

    let mut loaded = read_sfmr(&old).unwrap();
    assert_eq!(loaded.metadata.version, 15);
    assert_eq!(loaded.normals_xyz.as_ref(), Some(&given_normals));

    // Saved as version 16, points 0 and 1 take their frames' normals.
    let saved = dir.join("saved.sfmr");
    write_sfmr(&saved, &mut loaded).unwrap();
    let reloaded = read_sfmr(&saved).unwrap();
    let normals = reloaded.normals_xyz.unwrap();
    assert_eq!(row(&normals, 0), vec![0.0, 0.0, 1.0]);
    assert_ne!(row(&normals, 1), row(&given_normals, 1));
    for i in 2..5 {
        assert_eq!(row(&normals, i), row(&given_normals, i), "point {i}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn normals_of_the_wrong_shape_are_written_with_the_frames_normals() {
    // A (0, 3) normals array makes the writer fall back to the recomputed
    // mean-viewing normals wholesale; the finite patched points still take
    // their frames' normals, so the file is one its reader accepts.
    let dir = temp_dir("sfmr_test_patch_normals_wrong_shape");
    let path = dir.join("fallback.sfmr");
    let mut data = data_with_every_kind_of_point();
    let (u, v) = (
        data.patch_u_halfvec_xyz.clone().unwrap(),
        data.patch_v_halfvec_xyz.clone().unwrap(),
    );
    data.normals_xyz = Some(Array2::zeros((0, 3)));
    write_sfmr(&path, &mut data).unwrap();

    assert!(!archive_names(&path).contains(&FULL_ENTRY.to_string()));
    let loaded = read_sfmr(&path).expect("the reader accepts what the writer wrote");
    let normals = loaded.normals_xyz.unwrap();
    for i in 0..2 {
        let expected = cross_normal_f64(row(&u, i), row(&v, i));
        for k in 0..3 {
            assert!(
                (f64::from(normals[[i, k]]) - expected[k]).abs() < 1e-6,
                "point {i}"
            );
        }
    }
    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "{errors:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Copy `src` to `dst` with the entry `name` holding `raw` added, its bytes
/// compressed as the writer compresses them. Hashes are not restamped.
fn with_extra_entry(src: &std::path::Path, dst: &std::path::Path, name: &str, raw: &[u8]) {
    use std::io::{Read, Write};
    let mut archive = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let mut zip = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let entry_name = entry.name().unwrap().into_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        zip.start_file(&entry_name, stored).unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.start_file(name, stored).unwrap();
    zip.write_all(&zstd::bulk::compress(raw, 3).unwrap())
        .unwrap();
    zip.finish().unwrap();
}

const STRAY_ENTRY: &str = "points3d/underived_normals_xyz.2.3.float32.zst";

#[test]
fn an_underived_entry_in_a_version_16_file_without_the_frame_is_refused() {
    let dir = temp_dir("sfmr_test_patch_normals_stray_no_frame");
    let good = dir.join("good.sfmr");
    // Normals and no patch frame: the full column, and nothing derived.
    write_sfmr(&good, &mut make_test_data()).unwrap();
    assert!(archive_names(&good).contains(&FULL_ENTRY.to_string()));
    let bad = dir.join("stray.sfmr");
    with_extra_entry(
        &good,
        &bad,
        STRAY_ENTRY,
        &floats_payload(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    );
    assert_refused(
        &bad,
        &format!(
            "{STRAY_ENTRY} is stored only by a file of version 16 or later with both normals \
             and the patch frame, and this version 16 file is not one"
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_underived_entry_in_a_version_15_file_is_refused() {
    let dir = temp_dir("sfmr_test_patch_normals_stray_v15");
    let current = dir.join("current.sfmr");
    write_sfmr(&current, &mut data_with_every_kind_of_point()).unwrap();
    // The version 15 file of the same value, with the full column it
    // stores, and the version 16 entry beside it.
    let old = dir.join("v15.sfmr");
    super::rewrite_as_version_15(&current, &old, &dir, |_, _| {});
    assert!(archive_names(&old).contains(&FULL_ENTRY.to_string()));
    assert!(read_sfmr(&old).is_ok());
    let bad = dir.join("stray.sfmr");
    with_extra_entry(&old, &bad, STRAY_ENTRY, &floats_payload(&[0.0; 6]));
    assert_refused(
        &bad,
        &format!(
            "{STRAY_ENTRY} is stored only by a file of version 16 or later with both normals \
             and the patch frame, and this version 15 file is not one"
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_that_sets_has_normals_without_its_column_fails_verification() {
    let dir = temp_dir("sfmr_test_patch_normals_missing_column");
    let good = dir.join("good.sfmr");
    // Without the patch frame no normal is derived, so has_normals needs the
    // full column.
    write_sfmr(&good, &mut make_test_data()).unwrap();
    let bad = dir.join("missing.sfmr");
    rewrite_renaming(&good, &bad, |name, raw| {
        (name == FULL_ENTRY).then(|| ("points3d/unrelated.bin".to_string(), raw.to_vec()))
    });
    assert!(read_sfmr(&bad).is_err(), "the reader refuses it");
    let (valid, errors) = verify_sfmr(&bad).expect("a list of failures, not an error");
    assert!(!valid);
    assert!(
        errors.iter().any(|e| e.contains(&format!(
            "points3d/metadata.json sets has_normals, and the file holds no {FULL_ENTRY}"
        ))),
        "{errors:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
