// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Negative-format tests use a hash-aware archive mutator. Without refreshing
//! the section directory, every structural mutation stops at the preceding
//! integrity check and never exercises the validator named by the test.

use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use sfmtool_archive_io::format_hash;
use sfmtool_progress::Progress;
use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::write::SimpleFileOptions;

use crate::*;

type StoredEntries = Vec<(String, Vec<u8>)>;

fn tiny_u8<'a>(vectors: &'a [u8], trees: usize) -> KdfForestData<'a, u8> {
    KdfForestData {
        vectors,
        feature_count: 3,
        dimension: 2,
        trees: (0..trees)
            .map(|_| KdfTree {
                nodes: vec![
                    KdfNode::Internal {
                        split_dimension: 0,
                        split: 1,
                        left: 1,
                        right: 2,
                    },
                    KdfNode::Leaf { start: 0, len: 2 },
                    KdfNode::Leaf { start: 2, len: 1 },
                ],
                feature_ids: vec![0, 1, 2],
            })
            .collect(),
        provenance: None,
        descriptor_order: None,
    }
}

fn tiny_f32<'a>(vectors: &'a [f32]) -> KdfForestData<'a, f32> {
    KdfForestData {
        vectors,
        feature_count: 3,
        dimension: 2,
        trees: vec![KdfTree {
            nodes: vec![
                KdfNode::Internal {
                    split_dimension: 0,
                    split: 1.0,
                    left: 1,
                    right: 2,
                },
                KdfNode::Leaf { start: 0, len: 2 },
                KdfNode::Leaf { start: 2, len: 1 },
            ],
            feature_ids: vec![0, 1, 2],
        }],
        provenance: None,
        descriptor_order: None,
    }
}

fn roomy() -> LazyKdForestOptions {
    LazyKdForestOptions {
        cache_bytes: 4096,
        max_in_flight_bytes: 4096,
        max_chunk_bytes: 4096,
        max_metadata_bytes: 1 << 20,
        ..Default::default()
    }
}

fn write_tiny_u8(dir: &Path, name: &str, trees: usize) -> PathBuf {
    let path = dir.join(name);
    let vectors = [0, 0, 1, 1, 9, 9];
    let options = KdfWriteOptions {
        target_descriptor_block_bytes: 2,
        ..Default::default()
    };
    write_kdf(
        &path,
        &tiny_u8(&vectors, trees),
        None,
        &options,
        &Progress::none(),
    )
    .unwrap();
    path
}

#[test]
fn tiny_u8_content_hash_is_stable() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_tiny_u8(dir.path(), "stable.kdf", 1);
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let raw = sfmtool_archive_io::read_zst_entry(&mut archive, "content_hash.json.zst").unwrap();
    let hash: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    // Frozen from the version 3 writer. The whole-file digest depends on
    // packing, node layout and JSON serialization, so a change to any of those
    // lands here first.
    assert_eq!(hash["content_xxh128"], "cf4fe8eb28d7891f29fe619fe2e60f52");
    // One digest a section, whatever the corpus: nothing in this object is a
    // list, so its size does not grow with the number of blocks or chunks.
    let object = hash.as_object().unwrap();
    assert!(
        object.values().all(serde_json::Value::is_string),
        "the integrity directory holds a digest per section, not a list: {object:?}"
    );
}

fn read_stored(path: &Path) -> StoredEntries {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    (0..archive.len())
        .map(|i| {
            let mut entry = archive.by_index(i).unwrap();
            let mut stored = Vec::new();
            entry.read_to_end(&mut stored).unwrap();
            (entry.name().to_string(), stored)
        })
        .collect()
}

fn write_stored(path: &Path, entries: &StoredEntries) {
    let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, stored) in entries {
        archive
            .start_file(
                name,
                SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored)
                    .large_file(true),
            )
            .unwrap();
        archive.write_all(stored).unwrap();
    }
    archive.finish().unwrap();
}

fn stored<'a>(entries: &'a StoredEntries, name: &str) -> &'a [u8] {
    &entries
        .iter()
        .find(|(candidate, _)| candidate == name)
        .unwrap_or_else(|| panic!("missing test entry {name}"))
        .1
}

fn stored_mut<'a>(entries: &'a mut StoredEntries, name: &str) -> &'a mut Vec<u8> {
    &mut entries
        .iter_mut()
        .find(|(candidate, _)| candidate == name)
        .unwrap_or_else(|| panic!("missing test entry {name}"))
        .1
}

fn entry_starting_with(entries: &StoredEntries, prefix: &str) -> String {
    entries
        .iter()
        .find(|(name, _)| name.starts_with(prefix))
        .unwrap_or_else(|| panic!("missing test entry starting with {prefix}"))
        .0
        .clone()
}

fn decode_entry(entries: &StoredEntries, name: &str) -> Vec<u8> {
    zstd::decode_all(Cursor::new(stored(entries, name))).unwrap()
}

fn replace_decoded(entries: &mut StoredEntries, name: &str, decoded: &[u8]) {
    *stored_mut(entries, name) = zstd::encode_all(Cursor::new(decoded), 3).unwrap();
}

fn hash_pair(first: &[u8], second: &[u8]) -> u128 {
    let mut hash = Xxh3::new();
    hash.update(first);
    hash.update(second);
    hash.digest128()
}

fn parse_hash(value: &str) -> u128 {
    u128::from_str_radix(value, 16).unwrap()
}

/// The format's one folding rule: sixteen big-endian bytes per item, in order.
fn fold(items: &[u128]) -> u128 {
    let mut composition = Vec::with_capacity(items.len() * 16);
    for value in items {
        composition.extend_from_slice(&value.to_be_bytes());
    }
    xxh3_128(&composition)
}

/// The name of every entry under `prefix`, in ascending block or chunk index.
fn indexed_entries(entries: &StoredEntries, prefix: &str, suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    while let Some((name, _)) = entries
        .iter()
        .find(|(name, _)| name.starts_with(&format!("{prefix}{}/{suffix}", out.len())))
    {
        out.push(name.clone());
    }
    out
}

/// Every block frame of one container entry, sliced by its offsets array.
fn container_frames(entries: &StoredEntries, container: &str, offsets: &str) -> Vec<Vec<u8>> {
    let container = entry_starting_with(entries, container);
    let offsets_name = entry_starting_with(entries, offsets);
    let offsets_raw = decode_entry(entries, &offsets_name);
    let offsets: &[u64] = bytemuck::cast_slice(&offsets_raw);
    let corpus = stored(entries, &container);
    offsets
        .windows(2)
        .map(|pair| corpus[pair[0] as usize..pair[1] as usize].to_vec())
        .collect()
}

/// Fold a blocked container's raw block bytes into its one section digest.
fn container_digest(entries: &StoredEntries, container: &str, offsets: &str) -> u128 {
    let digests: Vec<u128> = container_frames(entries, container, offsets)
        .iter()
        .map(|frame| xxh3_128(&zstd::decode_all(Cursor::new(frame)).unwrap()))
        .collect();
    fold(&digests)
}

/// Recompute every section digest from the current decoded payloads.
///
/// The helper intentionally mirrors the format's composition order rather than
/// calling writer internals: a regression in either side then cannot make a
/// malformed fixture self-consistent by sharing the same implementation.
fn refresh_hashes(entries: &mut StoredEntries) {
    let metadata_raw = decode_entry(entries, "metadata.json.zst");
    let hash_raw = decode_entry(entries, "content_hash.json.zst");
    let mut hashes: ContentHash = serde_json::from_slice(&hash_raw).unwrap();

    hashes.metadata_xxh128 = format_hash(xxh3_128(&metadata_raw));

    if hashes.images_xxh128.is_some() {
        let feature_hashes = entry_starting_with(entries, "images/feature_tool_hashes.");
        let sift_hashes = entry_starting_with(entries, "images/sift_content_hashes.");
        let mut hash = Xxh3::new();
        hash.update(&decode_entry(entries, &feature_hashes));
        hash.update(&decode_entry(entries, "images/metadata.json.zst"));
        hash.update(&decode_entry(entries, "images/names.json.zst"));
        hash.update(&decode_entry(entries, &sift_hashes));
        hashes.images_xxh128 = Some(format_hash(hash.digest128()));
    }

    if hashes.origins_xxh128.is_some() {
        let images = indexed_entries(entries, "origins/", "image_indexes.");
        let features = indexed_entries(entries, "origins/", "image_feature_indexes.");
        let digests: Vec<u128> = images
            .iter()
            .zip(&features)
            .map(|(a, f)| hash_pair(&decode_entry(entries, a), &decode_entry(entries, f)))
            .collect();
        hashes.origins_xxh128 = Some(format_hash(fold(&digests)));
    }

    let name = entry_starting_with(entries, "features/storage_rows.");
    hashes.storage_rows_xxh128 = format_hash(xxh3_128(&decode_entry(entries, &name)));

    hashes.descriptors_xxh128 = format_hash(container_digest(
        entries,
        "features/corpus.",
        "features/block_offsets.",
    ));

    if hashes.geometry_xxh128.is_some() {
        hashes.geometry_xxh128 = Some(format_hash(container_digest(
            entries,
            "features/geometry.",
            "features/geometry_block_offsets.",
        )));
    }

    let mut chunk_digests = Vec::new();
    for tree in 0.. {
        let chunks = indexed_entries(entries, &format!("trees/{tree}/chunks/"), "chunk.");
        if chunks.is_empty()
            && !entries
                .iter()
                .any(|(n, _)| n.starts_with(&format!("trees/{tree}/")))
        {
            break;
        }
        for name in chunks {
            chunk_digests.push(xxh3_128(&decode_entry(entries, &name)));
        }
    }
    hashes.trees_xxh128 = format_hash(fold(&chunk_digests));

    let sections: Vec<u128> = [Some(&hashes.metadata_xxh128), hashes.images_xxh128.as_ref()]
        .into_iter()
        .chain([
            hashes.origins_xxh128.as_ref(),
            Some(&hashes.storage_rows_xxh128),
            Some(&hashes.descriptors_xxh128),
            hashes.geometry_xxh128.as_ref(),
            Some(&hashes.trees_xxh128),
        ])
        .flatten()
        .map(|value| parse_hash(value))
        .collect();
    hashes.content_xxh128 = format_hash(fold(&sections));
    replace_decoded(
        entries,
        "content_hash.json.zst",
        &serde_json::to_vec(&hashes).unwrap(),
    );
}

fn mutate_rehashed(good: &Path, bad: &Path, entry_prefix: &str, mutate: impl FnOnce(&mut Vec<u8>)) {
    let mut entries = read_stored(good);
    let name = entry_starting_with(&entries, entry_prefix);
    let mut raw = decode_entry(&entries, &name);
    mutate(&mut raw);
    replace_decoded(&mut entries, &name, &raw);
    refresh_hashes(&mut entries);
    write_stored(bad, &entries);
}

fn mutate_frame_rehashed(
    good: &Path,
    bad: &Path,
    container_prefix: &str,
    offsets_prefix: &str,
    block: usize,
    mutate: impl FnOnce(&mut Vec<u8>),
) {
    let mut entries = read_stored(good);
    let container_name = entry_starting_with(&entries, container_prefix);
    let offsets_name = entry_starting_with(&entries, offsets_prefix);
    let offsets_raw = decode_entry(&entries, &offsets_name);
    let mut offsets: Vec<u64> = bytemuck::cast_slice(&offsets_raw).to_vec();
    let corpus = stored(&entries, &container_name);
    let mut frames: Vec<Vec<u8>> = offsets
        .windows(2)
        .map(|pair| corpus[pair[0] as usize..pair[1] as usize].to_vec())
        .collect();
    let mut raw = zstd::decode_all(Cursor::new(&frames[block])).unwrap();
    mutate(&mut raw);
    frames[block] = zstd::encode_all(Cursor::new(raw), 3).unwrap();
    let mut rebuilt = Vec::new();
    offsets.clear();
    offsets.push(0);
    for frame in frames {
        rebuilt.extend_from_slice(&frame);
        offsets.push(rebuilt.len() as u64);
    }
    *stored_mut(&mut entries, &container_name) = rebuilt;
    replace_decoded(
        &mut entries,
        &offsets_name,
        bytemuck::cast_slice(offsets.as_slice()),
    );
    refresh_hashes(&mut entries);
    write_stored(bad, &entries);
}

fn set_u32(raw: &mut [u8], index: usize, value: u32) {
    raw[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn assert_invalid(error: KdfError, message: &str) {
    match error {
        KdfError::InvalidFormat(actual) => assert!(
            actual.contains(message),
            "expected {message:?}, got {actual:?}"
        ),
        other => panic!("expected InvalidFormat containing {message:?}, got {other:?}"),
    }
}

fn open_error(path: &Path) -> KdfError {
    match KdfFile::<u8>::open(path, roomy()) {
        Ok(_) => panic!("malformed KDF unexpectedly opened: {}", path.display()),
        Err(error) => error,
    }
}

#[test]
fn malformed_node_references_and_leaf_shapes_reach_their_validators() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);
    // Three nodes. Columns are stored column-major as ten rows of length three.
    let cases = [
        ("child-chunk", 3 * 3, 7, "child chunk out of range"),
        ("child-node", 4 * 3, 7, "child node/logical ID out of range"),
        (
            "logical-mismatch",
            5 * 3,
            2,
            "child logical ID does not match addressed node",
        ),
        (
            "noncontiguous-leaf",
            9 * 3 + 1,
            1,
            "leaf starts are not contiguous",
        ),
        ("leaf-past-end", 9 * 3 + 2, 4, "invalid leaf range"),
        (
            "reserved-leaf-field",
            2 * 3 + 1,
            1,
            "unused leaf fields are nonzero",
        ),
        ("unknown-kind", 1, 9, "unknown node kind"),
    ];
    for (name, index, value, expected) in cases {
        let bad = dir.path().join(format!("{name}.kdf"));
        mutate_rehashed(&good, &bad, "trees/0/chunks/0/chunk.", |raw| {
            set_u32(raw, index, value);
        });
        assert_invalid(verify_kdf::<u8>(&bad, roomy()).unwrap_err(), expected);
    }
}

#[test]
fn cycles_and_shared_children_are_rejected_by_full_verification() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);

    let cycle = dir.path().join("cycle.kdf");
    mutate_rehashed(&good, &cycle, "trees/0/chunks/0/chunk.", |raw| {
        // Root's left child points back to the root with a consistent logical ID.
        set_u32(raw, 3 * 3, 0);
        set_u32(raw, 4 * 3, 0);
        set_u32(raw, 5 * 3, 0);
    });
    assert_invalid(
        verify_kdf::<u8>(&cycle, roomy()).unwrap_err(),
        "revisits logical node 0",
    );

    let shared = dir.path().join("shared-child.kdf");
    mutate_rehashed(&good, &shared, "trees/0/chunks/0/chunk.", |raw| {
        // Both root branches point at leaf 1.
        set_u32(raw, 6 * 3, 0);
        set_u32(raw, 7 * 3, 1);
        set_u32(raw, 8 * 3, 1);
    });
    assert_invalid(
        verify_kdf::<u8>(&shared, roomy()).unwrap_err(),
        "revisits logical node 1",
    );
}

#[test]
fn bad_feature_permutations_and_split_constraints_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);

    let duplicate = dir.path().join("duplicate-id.kdf");
    mutate_rehashed(&good, &duplicate, "trees/0/chunks/0/chunk.", |raw| {
        // Topology is 10 * 3 u32 plus three u8 splits; IDs follow it.
        set_u32(raw, 31, 0);
    });
    assert_invalid(
        verify_kdf::<u8>(&duplicate, roomy()).unwrap_err(),
        "repeats feature ID 0",
    );

    let out_of_range = dir.path().join("out-of-range-id.kdf");
    mutate_rehashed(&good, &out_of_range, "trees/0/chunks/0/chunk.", |raw| {
        set_u32(raw, 31, 9)
    });
    assert_invalid(
        verify_kdf::<u8>(&out_of_range, roomy()).unwrap_err(),
        "feature ID out of range",
    );

    let wrong_side = dir.path().join("wrong-side.kdf");
    mutate_frame_rehashed(
        &good,
        &wrong_side,
        "features/corpus.",
        "features/block_offsets.",
        0,
        |raw| raw[0] = 9,
    );
    assert_invalid(
        verify_kdf::<u8>(&wrong_side, roomy()).unwrap_err(),
        "violates a split constraint",
    );
}

#[test]
fn the_storage_row_map_must_be_a_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);
    let bad = dir.path().join("duplicate-storage-row.kdf");
    mutate_rehashed(&good, &bad, "features/storage_rows.", |raw| {
        set_u32(raw, 1, 0);
    });
    assert_invalid(open_error(&bad), "storage row map contains a duplicate row");
}

#[test]
fn descriptors_are_stored_once_regardless_of_tree_count() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 2);
    let entries = read_stored(&good);
    assert_eq!(
        entries
            .iter()
            .filter(|(name, _)| name.starts_with("features/corpus."))
            .count(),
        1
    );
    assert!(!entries.iter().any(|(name, _)| name.contains("/vectors.")));
}

#[test]
fn nonfinite_f32_splits_and_vectors_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.kdf");
    let vectors = [0.0, 0.0, 1.0, 1.0, 9.0, 9.0];
    write_kdf(
        &good,
        &tiny_f32(&vectors),
        None,
        &KdfWriteOptions {
            target_descriptor_block_bytes: 8,
            ..Default::default()
        },
        &Progress::none(),
    )
    .unwrap();

    let split = dir.path().join("nan-split.kdf");
    mutate_rehashed(&good, &split, "trees/0/chunks/0/chunk.", |raw| {
        let offset = NODE_COLUMNS * 3 * 4;
        raw[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    assert_invalid(
        verify_kdf::<f32>(&split, roomy()).unwrap_err(),
        "invalid internal split",
    );

    let vector = dir.path().join("nan-vector.kdf");
    mutate_frame_rehashed(
        &good,
        &vector,
        "features/corpus.",
        "features/block_offsets.",
        0,
        |raw| raw[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
    );
    assert_invalid(
        verify_kdf::<f32>(&vector, roomy()).unwrap_err(),
        "descriptor block contains non-finite vector",
    );
}

#[test]
fn metadata_version_scalar_and_declared_shape_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);

    for old_or_future in [1, 99] {
        let version = dir.path().join(format!("version-{old_or_future}.kdf"));
        mutate_rehashed(&good, &version, "metadata.json.zst", |raw| {
            let mut value: serde_json::Value = serde_json::from_slice(raw).unwrap();
            value["version"] = old_or_future.into();
            *raw = serde_json::to_vec(&value).unwrap();
        });
        // A version this build does not write is refused with the remedy in the
        // message, because rebuilding is the only way past it: there is no
        // translation from an older shape and none to a newer one.
        assert_invalid(
            open_error(&version),
            &format!(
                "this is a version {old_or_future} index and this build reads version 3; rebuild the index"
            ),
        );
    }

    let scalar = dir.path().join("scalar.kdf");
    mutate_rehashed(&good, &scalar, "metadata.json.zst", |raw| {
        let mut value: serde_json::Value = serde_json::from_slice(raw).unwrap();
        value["scalar_type"] = "float32".into();
        *raw = serde_json::to_vec(&value).unwrap();
    });
    assert!(matches!(
        KdfFile::<u8>::open(&scalar, roomy()),
        Err(KdfError::ScalarType { .. })
    ));

    let shape = dir.path().join("declared-shape.kdf");
    mutate_rehashed(&good, &shape, "metadata.json.zst", |raw| {
        let mut value: serde_json::Value = serde_json::from_slice(raw).unwrap();
        value["trees"][0]["chunks"][0]["decoded_bytes"] = 999.into();
        *raw = serde_json::to_vec(&value).unwrap();
    });
    assert_invalid(
        open_error(&shape),
        "declared chunk bytes disagree with its shape",
    );
}

#[test]
fn missing_unexpected_and_duplicate_entries_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);

    let mut missing_entries = read_stored(&good);
    missing_entries.retain(|(name, _)| !name.starts_with("features/corpus."));
    let missing = dir.path().join("missing.kdf");
    write_stored(&missing, &missing_entries);
    assert_invalid(open_error(&missing), "archive entry set mismatch");

    let mut unexpected_entries = read_stored(&good);
    unexpected_entries.push((
        "trees/0/chunks/0/vectors.3.2.uint8.zst".into(),
        zstd::encode_all(Cursor::new([0u8; 12]), 3).unwrap(),
    ));
    let unexpected = dir.path().join("unexpected-layout-entry.kdf");
    write_stored(&unexpected, &unexpected_entries);
    assert_invalid(open_error(&unexpected), "archive entry set mismatch");

    let mut duplicate_entries = read_stored(&good);
    let metadata = duplicate_entries
        .iter()
        .find(|(name, _)| name == "metadata.json.zst")
        .unwrap()
        .1
        .clone();
    // zip refuses to write duplicate names itself. Write an equal-length
    // placeholder, then rename both its local and central-directory records in
    // the finished bytes, as a malformed producer could.
    const PLACEHOLDER: &[u8] = b"duplicat.json.zst";
    const DUPLICATE: &[u8] = b"metadata.json.zst";
    assert_eq!(PLACEHOLDER.len(), DUPLICATE.len());
    duplicate_entries.push((String::from_utf8(PLACEHOLDER.to_vec()).unwrap(), metadata));
    let duplicate = dir.path().join("duplicate.kdf");
    write_stored(&duplicate, &duplicate_entries);
    let mut bytes = std::fs::read(&duplicate).unwrap();
    let mut replacements = 0;
    for offset in 0..=bytes.len() - PLACEHOLDER.len() {
        if &bytes[offset..offset + PLACEHOLDER.len()] == PLACEHOLDER {
            bytes[offset..offset + DUPLICATE.len()].copy_from_slice(DUPLICATE);
            replacements += 1;
        }
    }
    assert_eq!(
        replacements, 2,
        "local and central ZIP names must be replaced"
    );
    std::fs::write(&duplicate, bytes).unwrap();
    assert_invalid(open_error(&duplicate), "duplicate ZIP entry");
}

#[test]
fn a_truncated_frame_reports_the_entry_decode_failure() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_tiny_u8(dir.path(), "good.kdf", 1);
    let mut entries = read_stored(&good);
    let name = entry_starting_with(&entries, "trees/0/chunks/0/chunk.");
    stored_mut(&mut entries, &name).truncate(5);
    let bad = dir.path().join("truncated.kdf");
    write_stored(&bad, &entries);
    assert_invalid(
        verify_kdf::<u8>(&bad, roomy()).unwrap_err(),
        "zstd decode failed",
    );
}

fn sources() -> KdfSiftSources {
    KdfSiftSources {
        workspace: KdfWorkspaceMetadata {
            absolute_path: "unused".into(),
            relative_path: ".".into(),
            contents: KdfWorkspaceContents {
                feature_tool: "sfmtool".into(),
                feature_type: "sift-sfmtool".into(),
                feature_options: serde_json::json!({}),
                feature_prefix_dir: "features/sift".into(),
            },
        },
        image_names: vec!["bull.jpg".into(), "other.jpg".into()],
        feature_tool_hashes: vec![[1; 16], [2; 16]],
        sift_content_hashes: vec![[3; 16], [4; 16]],
        origins: vec![
            FeatureOrigin {
                image_index: 0,
                image_feature_index: 0,
            },
            FeatureOrigin {
                image_index: 0,
                image_feature_index: 1,
            },
            FeatureOrigin {
                image_index: 1,
                image_feature_index: 0,
            },
        ],
        geometry: vec![
            [[1.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
            [[3.0, 4.0], [2.0, 0.0], [0.0, 2.0]],
            [[5.0, 6.0], [3.0, 0.0], [0.0, 3.0]],
        ],
    }
}

fn write_sourced(dir: &Path) -> PathBuf {
    let path = dir.join("sourced.kdf");
    let mut vectors = vec![0u8; 3 * 128];
    vectors[128] = 1;
    vectors[256] = 9;
    let mut data = tiny_u8(&[0, 0, 1, 1, 9, 9], 1);
    data.vectors = &vectors;
    data.dimension = 128;
    write_kdf(
        &path,
        &data,
        Some(&sources()),
        &KdfWriteOptions {
            origin_block_rows: 3,
            target_descriptor_block_bytes: 128,
            ..Default::default()
        },
        &Progress::none(),
    )
    .unwrap();
    path
}

#[test]
fn invalid_and_duplicate_origins_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_sourced(dir.path());

    let invalid = dir.path().join("invalid-image.kdf");
    mutate_rehashed(&good, &invalid, "origins/0/image_indexes.", |raw| {
        set_u32(raw, 0, 7);
    });
    assert_invalid(
        verify_kdf::<u8>(&invalid, roomy()).unwrap_err(),
        "origin image index out of range",
    );

    let duplicate = dir.path().join("duplicate-origin.kdf");
    mutate_rehashed(
        &good,
        &duplicate,
        "origins/0/image_feature_indexes.",
        |raw| set_u32(raw, 1, 0),
    );
    assert_invalid(
        verify_kdf::<u8>(&duplicate, roomy()).unwrap_err(),
        "invalid or duplicate feature origin",
    );
}

#[test]
fn nonfinite_and_missing_sift_geometry_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_sourced(dir.path());

    let nonfinite = dir.path().join("nonfinite-geometry.kdf");
    mutate_frame_rehashed(
        &good,
        &nonfinite,
        "features/geometry.",
        "features/geometry_block_offsets.",
        0,
        |raw| raw[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
    );
    let file = KdfFile::<u8>::open(&nonfinite, roomy()).unwrap();
    assert_eq!(file.io_stats().read_calls, 0);
    assert_invalid(
        file.feature_geometry(0).unwrap_err(),
        "geometry block contains non-finite value",
    );
    assert_invalid(
        verify_kdf::<u8>(&nonfinite, roomy()).unwrap_err(),
        "geometry block contains non-finite value",
    );

    let mut entries = read_stored(&good);
    entries.retain(|(name, _)| !name.starts_with("features/geometry."));
    let missing = dir.path().join("missing-geometry.kdf");
    write_stored(&missing, &entries);
    assert_invalid(open_error(&missing), "archive entry set mismatch");

    let wrong_width = dir.path().join("wrong-sift-width.kdf");
    mutate_rehashed(&good, &wrong_width, "metadata.json.zst", |raw| {
        let mut value: serde_json::Value = serde_json::from_slice(raw).unwrap();
        value["dimension"] = 127.into();
        *raw = serde_json::to_vec(&value).unwrap();
    });
    assert_invalid(
        open_error(&wrong_width),
        "SIFT references require unchanged 128-D uint8 descriptors",
    );
}

#[test]
fn unvisited_chunks_stay_unread_and_warm_chunks_cost_no_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunked.kdf");
    let vectors = [0, 0, 1, 1, 9, 9];
    write_kdf(
        &path,
        &tiny_u8(&vectors, 1),
        None,
        &KdfWriteOptions {
            target_chunk_bytes: 90,
            target_descriptor_block_bytes: 2,
            ..Default::default()
        },
        &Progress::none(),
    )
    .unwrap();
    let file = KdfFile::<u8>::open(&path, roomy()).unwrap();
    assert!(file.metadata().trees[0].chunks.len() > 1);
    assert_eq!(file.io_stats().read_calls, 0);

    let root = file.root(0).unwrap();
    file.node(0, root).unwrap();
    let cold = file.io_stats();
    assert!(cold.read_calls > 0);
    let total: u64 = file.metadata().trees[0]
        .chunks
        .iter()
        .map(|chunk| chunk.decoded_bytes)
        .sum();
    assert!(cold.decoded_bytes < total, "an unvisited chunk was decoded");

    file.node(0, root).unwrap();
    let warm = file.io_stats();
    assert_eq!(warm.read_calls, cold.read_calls);
    assert_eq!(warm.decoded_bytes, cold.decoded_bytes);
    assert!(warm.cache_hits > cold.cache_hits);
}
