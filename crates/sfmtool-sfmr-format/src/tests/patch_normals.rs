// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The normals of version 16: a finite point with a patch has the normal of
//! its frame, `normalize(u × v)`, which the file derives rather than stores.

use super::{make_test_data, restamp_hashes, rewrite_entries};
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

fn stored_normals(path: &std::path::Path) -> Vec<f32> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut compressed = Vec::new();
    archive
        .by_name("points3d/normals_xyz.5.3.float32.zst")
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

fn row(a: &Array2<f32>, i: usize) -> Vec<f32> {
    a.row(i).to_vec()
}

#[test]
fn a_finite_patched_point_stores_no_normal_and_reads_its_frames() {
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

    // The file holds zero rows for points 0 and 1, and every other row as
    // given: the degenerate frame's, the unpatched point's and the point at
    // infinity's.
    let stored = stored_normals(&path);
    assert_eq!(&stored[0..6], &[0.0; 6]);
    assert_eq!(
        &stored[6..15],
        given_normals.as_slice().unwrap()[6..15].as_ref()
    );

    // A reader fills points 0 and 1 with normalize(u × v) and reads the rest.
    let loaded = read_sfmr(&path).unwrap();
    let normals = loaded.normals_xyz.as_ref().unwrap();
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
    // a save of what was read writes the same hash.
    assert_eq!(data.normals_xyz.as_ref(), Some(normals));
    let stored_hash = read_sfmr_content_hash(&path).unwrap();
    let mut loaded = loaded;
    let resaved = dir.join("resaved.sfmr");
    write_sfmr(&resaved, &mut loaded).unwrap();
    assert_eq!(
        read_sfmr_content_hash(&resaved).unwrap().content_xxh128,
        stored_hash.content_xxh128
    );
    let (valid, errors) = verify_sfmr(&resaved).unwrap();
    assert!(valid, "{errors:?}");

    // An edit of a derived normal does not reach the file: the hash is the
    // frame's.
    let mut edited = data_with_every_kind_of_point();
    edited.normals_xyz.as_mut().unwrap()[[1, 0]] = 1.0;
    let options = WriteOptions::default();
    assert_eq!(
        content_hash_of(&mut edited, &options)
            .unwrap()
            .content_xxh128,
        stored_hash.content_xxh128
    );
    // Turning the frame does.
    let mut turned = data_with_every_kind_of_point();
    turned.patch_v_halfvec_xyz.as_mut().unwrap()[[1, 2]] = 0.07;
    assert_ne!(
        content_hash_of(&mut turned, &options)
            .unwrap()
            .content_xxh128,
        stored_hash.content_xxh128
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
fn a_stored_normal_for_a_derived_row_is_refused() {
    let dir = temp_dir("sfmr_test_patch_normals_stored_row");
    let good = dir.join("good.sfmr");
    write_sfmr(&good, &mut data_with_every_kind_of_point()).unwrap();
    let bad = dir.join("bad.sfmr");
    rewrite_entries(&good, &bad, |name, raw| {
        (name == "points3d/normals_xyz.5.3.float32.zst").then(|| {
            let mut values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            values[3..6].copy_from_slice(&[0.0, 0.6, 0.8]);
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        })
    });
    restamp_hashes_allowing(&bad, &dir);
    let needle = "points3d/normals_xyz row 1 is [0.0, 0.6, 0.8], but point 1 is finite";
    let err = read_sfmr(&bad).err().expect("refused").to_string();
    assert!(err.contains(needle), "{err}");
    let (valid, errors) = verify_sfmr(&bad).unwrap();
    assert!(!valid);
    assert!(errors.iter().any(|e| e.contains(needle)), "{errors:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Restamp the hashes of `path` so the only errors left are the ones its
/// content carries.
fn restamp_hashes_allowing(path: &std::path::Path, dir: &std::path::Path) {
    let (_, errors) = verify_sfmr(path).unwrap();
    let pairs: Vec<(String, String)> = errors
        .iter()
        .filter_map(|e| {
            let (_, rest) = e.split_once("mismatch: computed ")?;
            let (computed, stored) = rest.split_once(", stored ")?;
            Some((stored.to_string(), computed.to_string()))
        })
        .collect();
    let staged = dir.join("restamp.sfmr");
    rewrite_entries(path, &staged, |name, raw| {
        (name == "content_hash.json.zst").then(|| {
            let mut text = String::from_utf8(raw.to_vec()).unwrap();
            for (stored, computed) in &pairs {
                text = text.replace(stored.as_str(), computed);
            }
            text.into_bytes()
        })
    });
    std::fs::rename(&staged, path).unwrap();
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
    rewrite_entries(&old, &staged, |name, _| {
        (name == "points3d/normals_xyz.5.3.float32.zst")
            .then(|| given_normals.iter().flat_map(|v| v.to_le_bytes()).collect())
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

    let stored = stored_normals(&path);
    assert_eq!(&stored[0..6], &[0.0; 6]);
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
