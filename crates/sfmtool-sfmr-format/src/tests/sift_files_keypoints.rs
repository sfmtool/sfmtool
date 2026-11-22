// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `tracks/keypoints_xy` in a `sift_files` file: required from version 17,
//! and absent only from a version 16 file written without it.

use std::io::Read;
use std::path::Path;

use sfmtool_archive_io::{format_hash, SectionDigests};

use super::{make_test_data, read_json_entry_of};
use crate::*;

/// Every entry of the archive at `path`, decompressed, in archive order.
fn decompressed_entries(path: &Path) -> Vec<(String, Vec<u8>)> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let names: Vec<String> = archive
        .file_names()
        .map(|s| s.unwrap().into_owned())
        .collect();
    names
        .into_iter()
        .map(|name| {
            let mut compressed = Vec::new();
            archive
                .by_name(&name)
                .unwrap()
                .read_to_end(&mut compressed)
                .unwrap();
            let raw = zstd::stream::decode_all(&compressed[..]).unwrap();
            (name, raw)
        })
        .collect()
}

/// XXH128 over the `tracks/` entries of `entries` in lexicographic path order:
/// the format's rule, spelled out rather than borrowed from the writer.
fn tracks_digest(entries: &[(String, Vec<u8>)]) -> u128 {
    let mut tracks: Vec<&(String, Vec<u8>)> = entries
        .iter()
        .filter(|(name, _)| name.starts_with("tracks/"))
        .collect();
    tracks.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    for (_, raw) in tracks {
        hasher.update(raw);
    }
    hasher.digest128()
}

/// Copy the `sift_files` file `src` to `dst` as a file of `version` written
/// without `tracks/keypoints_xy`: the entry left out, `has_keypoints_xy` set
/// to `has_keypoints_xy`, and the metadata, tracks and whole-file hashes
/// recomputed, so that the reader and the verifier judge the file by its
/// structure alone.
fn without_keypoints(src: &Path, dst: &Path, version: u32, has_keypoints_xy: bool) {
    let entries = decompressed_entries(src);
    let stored: ContentHash = serde_json::from_slice(
        &entries
            .iter()
            .find(|(n, _)| n == "content_hash.json.zst")
            .unwrap()
            .1,
    )
    .unwrap();
    // The digest helper reproduces the stored hash of the untouched file.
    assert_eq!(
        format_hash(tracks_digest(&entries)),
        stored.tracks_xxh128,
        "tracks_digest does not follow the format's rule"
    );

    let mut edited: Vec<(String, Vec<u8>)> = Vec::new();
    for (name, raw) in entries {
        if name.starts_with("tracks/keypoints_xy.") {
            continue;
        }
        let raw = match name.as_str() {
            "metadata.json.zst" => {
                let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                json["version"] = serde_json::json!(version);
                serde_json::to_vec(&json).unwrap()
            }
            "tracks/metadata.json.zst" => {
                let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                json["has_keypoints_xy"] = serde_json::json!(has_keypoints_xy);
                serde_json::to_vec(&json).unwrap()
            }
            _ => raw,
        };
        edited.push((name, raw));
    }

    let raw_of = |name: &str| &edited.iter().find(|(n, _)| n == name).unwrap().1;
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(raw_of("metadata.json.zst"));
    let tracks_hash = tracks_digest(&edited);
    let mut digests = SectionDigests::new();
    digests.push(metadata_hash);
    digests.push(u128::from_str_radix(&stored.cameras_xxh128, 16).unwrap());
    assert!(stored.rigs_xxh128.is_none() && stored.frames_xxh128.is_none());
    digests.push(u128::from_str_radix(&stored.images_xxh128, 16).unwrap());
    digests.push(u128::from_str_radix(&stored.points3d_xxh128, 16).unwrap());
    digests.push(tracks_hash);
    assert!(stored.measurements_xxh128.is_none());
    let hashes = ContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        tracks_xxh128: format_hash(tracks_hash),
        content_xxh128: format_hash(digests.finish()),
        ..stored
    };
    let hash_bytes = serde_json::to_vec(&hashes).unwrap();

    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    let stored_method =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, raw) in &edited {
        let raw = if name == "content_hash.json.zst" {
            &hash_bytes
        } else {
            raw
        };
        zip.start_file(name, stored_method).unwrap();
        zip.write_all(&zstd::bulk::compress(raw, 3).unwrap())
            .unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn the_writer_refuses_a_sift_files_value_without_keypoints() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("refused.sfmr");
    let mut data = make_test_data();
    data.keypoints_xy = None;

    let err = write_sfmr(&path, &mut data).unwrap_err();
    match &err {
        SfmrError::InvalidFormat(m) => assert!(
            m.contains("tracks/keypoints_xy") && m.contains("sift_files"),
            "{m}"
        ),
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
    assert!(!path.exists(), "a refused write left a file behind");

    // The value still has a content hash, without the column, and it differs
    // from the hash of the same value with the column.
    let options = WriteOptions::default();
    let bare = content_hash_of(&mut data, &options).unwrap();
    let with = content_hash_of(&mut make_test_data(), &options).unwrap();
    assert_ne!(bare.tracks_xxh128, with.tracks_xxh128);
}

#[test]
fn a_version_17_sift_files_file_without_keypoints_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let src = dir.join("src.sfmr");
    write_sfmr(&src, &mut make_test_data()).unwrap();

    // The flag says the column is absent, and it is.
    let unflagged = dir.join("unflagged.sfmr");
    without_keypoints(&src, &unflagged, 17, false);
    assert_eq!(
        read_json_entry_of(&unflagged, "tracks/metadata.json.zst")["has_keypoints_xy"],
        serde_json::json!(false)
    );
    let expected = "a version 17 sift_files file must carry tracks/keypoints_xy";
    match read_sfmr(&unflagged).err() {
        Some(SfmrError::InvalidFormat(m)) => assert!(m.contains(expected), "{m}"),
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
    let (valid, errors) = verify_sfmr(&unflagged).unwrap();
    assert!(!valid);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains(expected), "{errors:?}");

    // The flag says the column is there, and the entry is missing.
    let missing = dir.join("missing.sfmr");
    without_keypoints(&src, &missing, 17, true);
    assert!(read_sfmr(&missing).is_err());
    assert!(!matches!(verify_sfmr(&missing), Ok((true, _))));
}

#[test]
fn a_version_16_sift_files_file_without_keypoints_reads_and_verifies() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let src = dir.join("src.sfmr");
    write_sfmr(&src, &mut make_test_data()).unwrap();
    let v16 = dir.join("v16.sfmr");
    without_keypoints(&src, &v16, 16, false);

    let mut loaded = read_sfmr(&v16).unwrap();
    assert_eq!(loaded.metadata.version, 16);
    assert_eq!(loaded.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert!(loaded.feature_indexes.is_some());
    assert!(loaded.keypoints_xy.is_none());
    assert_eq!(verify_sfmr(&v16).unwrap(), (true, vec![]));

    // It is not written back as version 17 until the column is supplied.
    let out = dir.join("v17.sfmr");
    assert!(matches!(
        write_sfmr(&out, &mut loaded),
        Err(SfmrError::InvalidFormat(_))
    ));
    assert!(!out.exists());
    loaded.keypoints_xy = make_test_data().keypoints_xy;
    write_sfmr(&out, &mut loaded).unwrap();
    let rewritten = read_sfmr(&out).unwrap();
    assert_eq!(rewritten.metadata.version, SFMR_FORMAT_VERSION);
    assert_eq!(rewritten.keypoints_xy, make_test_data().keypoints_xy);
}
