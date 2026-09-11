// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::*;

fn tiny_u8<'a>(vectors: &'a [u8]) -> KdfForestData<'a, u8> {
    KdfForestData {
        vectors,
        feature_count: 3,
        dimension: 2,
        trees: vec![KdfTree {
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

#[test]
fn round_trip_tree_local_and_shared() {
    let vectors = [0, 0, 1, 1, 9, 9];
    for options in [
        KdfWriteOptions {
            target_chunk_bytes: 90,
            ..KdfWriteOptions::tree_local()
        },
        KdfWriteOptions {
            target_chunk_bytes: 90,
            ..KdfWriteOptions::shared(2)
        },
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.kdf");
        write_kdf(&path, &tiny_u8(&vectors), None, &options).unwrap();
        let file = KdfFile::<u8>::open(&path, roomy()).unwrap();
        assert_eq!(file.len(), 3);
        let root = file.root(0).unwrap();
        let DecodedNode::Internal { left, right, .. } = file.node(0, root).unwrap() else {
            panic!()
        };
        assert_eq!(left.logical, 1);
        assert_eq!(right.logical, 2);
        let leaf = file.leaf(0, left).unwrap();
        assert_eq!(leaf.feature_ids, [0, 1]);
        if file.is_shared() {
            assert_eq!(file.shared_vector(2).unwrap(), [9, 9]);
        } else {
            assert_eq!(leaf.vectors.unwrap(), [0, 0, 1, 1]);
        }
        let verified = verify_kdf::<u8>(&path, roomy()).unwrap();
        assert_eq!(verified.features, 3);
    }
}

#[test]
fn source_origins_are_lazy_and_keep_requested_order() {
    let vectors = [0, 0, 1, 1, 9, 9];
    let sources = KdfSiftSources {
        workspace: KdfWorkspaceMetadata {
            absolute_path: "x".into(),
            relative_path: ".".into(),
            contents: KdfWorkspaceContents {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
                feature_prefix_dir: "features/sift".into(),
            },
        },
        image_names: vec!["a.jpg".into(), "b.jpg".into()],
        feature_tool_hashes: vec![[1; 16], [2; 16]],
        sift_content_hashes: vec![[3; 16], [4; 16]],
        origins: vec![
            FeatureOrigin {
                image_index: 0,
                image_feature_index: 4,
            },
            FeatureOrigin {
                image_index: 1,
                image_feature_index: 7,
            },
            FeatureOrigin {
                image_index: 0,
                image_feature_index: 8,
            },
        ],
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("origins.kdf");
    let options = KdfWriteOptions {
        origin_block_rows: 2,
        ..KdfWriteOptions::tree_local()
    };
    // Source mode deliberately requires real SIFT-shaped descriptors.
    let mut sift_vectors = vec![0u8; 3 * 128];
    sift_vectors[128] = 1;
    sift_vectors[256] = 9;
    let mut data = tiny_u8(&vectors);
    data.vectors = &sift_vectors;
    data.dimension = 128;
    write_kdf(&path, &data, Some(&sources), &options).unwrap();
    let file = KdfFile::<u8>::open(&path, roomy()).unwrap();
    assert_eq!(file.io_stats().read_calls, 0);
    let got = file.resolve_origins(&[2, 0, 2]).unwrap().unwrap();
    assert_eq!(
        got,
        [sources.origins[2], sources.origins[0], sources.origins[2]]
    );
    assert_eq!(
        file.image_table().unwrap().unwrap().names,
        sources.image_names
    );
}

#[test]
fn rejects_existing_destination_and_invalid_shared_map_budget() {
    let vectors = [0, 0, 1, 1, 9, 9];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.kdf");
    write_kdf(&path, &tiny_u8(&vectors), None, &KdfWriteOptions::shared(2)).unwrap();
    assert!(write_kdf(
        &path,
        &tiny_u8(&vectors),
        None,
        &KdfWriteOptions::tree_local()
    )
    .is_err());
    let options = LazyKdForestOptions {
        max_address_map_bytes: 4,
        ..roomy()
    };
    assert!(matches!(
        KdfFile::<u8>::open(Path::new(&path), options),
        Err(KdfError::ResourceLimit(_))
    ));
}

#[test]
fn corruption_in_lazy_descriptor_is_deferred_until_access() {
    use std::io::{Read, Write};
    use zip::write::SimpleFileOptions;

    let vectors = [0, 0, 1, 1, 9, 9];
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.kdf");
    let corrupt = dir.path().join("corrupt.kdf");
    write_kdf(&good, &tiny_u8(&vectors), None, &KdfWriteOptions::shared(2)).unwrap();
    let mut input = zip::ZipArchive::new(std::fs::File::open(&good).unwrap()).unwrap();
    let mut output = zip::ZipWriter::new(std::fs::File::create(&corrupt).unwrap());
    for i in 0..input.len() {
        let mut entry = input.by_index(i).unwrap();
        let name = entry.name().to_string();
        let mut raw = Vec::new();
        entry.read_to_end(&mut raw).unwrap();
        // The descriptor corpus is one entry of per-block frames, so damaging
        // block 0 means damaging the first frame in it. Byte 8 is inside that
        // frame and past its magic, so the failure surfaces as either a decode
        // error or a digest mismatch — both of which must be errors, and neither
        // of which may appear before the block is asked for.
        if name.starts_with("features/corpus.") {
            assert!(raw.len() > 8, "corpus container is unexpectedly small");
            raw[8] ^= 0x55;
        }
        output
            .start_file(
                name,
                SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
        output.write_all(&raw).unwrap();
    }
    output.finish().unwrap();
    let file = KdfFile::<u8>::open(&corrupt, roomy()).unwrap();
    assert_eq!(file.io_stats().read_calls, 0);
    assert!(file.shared_vector(0).is_err());
    assert!(verify_kdf::<u8>(&corrupt, roomy()).is_err());
}

/// The per-section decoded sizes must be the real uncompressed lengths, not the
/// stored frame lengths the ZIP directory reports.
///
/// Entries are STORE-wrapped zstd frames, so `uncompressed_size` from the
/// directory equals the compressed size for every entry — a summary built on it
/// reports a 100% compression ratio everywhere, which looks like a plausible
/// answer rather than a broken one. This asserts against sizes computed from the
/// data that was written.
#[test]
fn summary_decoded_sizes_are_uncompressed_lengths() {
    // Highly compressible: all-zero vectors, so a correct decoded size must
    // come out far larger than the stored frame.
    let vectors = vec![0u8; 3 * 2];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sizes.kdf");
    write_kdf(
        &path,
        &tiny_u8(&vectors),
        None,
        &KdfWriteOptions::tree_local(),
    )
    .unwrap();

    let summary = kdf_summary(&path, 1 << 20).unwrap();
    let section = |name: &str| {
        summary
            .sections
            .iter()
            .find(|s| s.section == name)
            .unwrap_or_else(|| panic!("no {name} section"))
    };

    // The chunk's three integer arrays share one entry: ten uint32 node columns
    // per node, one uint8 split per node, one uint32 feature ID per feature.
    // Three nodes and three features here.
    let chunks = section("tree_chunks");
    assert_eq!(chunks.entries, 1, "the integer arrays share one entry");
    assert_eq!(
        chunks.decoded_bytes,
        3 * 10 * 4 + 3 + 3 * 4,
        "decoded size must come from the shape in the name"
    );
    // Vectors stay their own entry, so their bytes are still attributable.
    let vectors = section("tree_vectors");
    assert_eq!(vectors.entries, 1);
    assert_eq!(vectors.decoded_bytes, 3 * 2);
    // The regression this guards: reading the size off the ZIP directory would
    // report the stored frame instead, making the two equal.
    assert_ne!(
        chunks.decoded_bytes, chunks.compressed_bytes,
        "decoded size looks like the stored frame length"
    );

    // The JSON entries are decoded rather than guessed, so they are nonzero and
    // differ from their stored frames.
    let metadata = section("metadata");
    assert!(metadata.decoded_bytes > 0);
    assert!(
        metadata.decoded_bytes > metadata.compressed_bytes,
        "JSON should compress: {metadata:?}"
    );

    // Nothing may be left unaccounted for.
    let summed: u64 = summary.sections.iter().map(|s| s.decoded_bytes).sum();
    assert_eq!(summed, summary.payload_decoded_bytes);
    assert!(summary.file_bytes > summary.payload_compressed_bytes);
}

/// A shared-layout file accounts for its descriptor corpus and row map.
#[test]
fn summary_accounts_for_the_shared_corpus_and_row_map() {
    let vectors = [0u8, 0, 1, 1, 9, 9];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.kdf");
    write_kdf(&path, &tiny_u8(&vectors), None, &KdfWriteOptions::shared(2)).unwrap();

    let summary = kdf_summary(&path, 1 << 20).unwrap();
    let names: Vec<&str> = summary
        .sections
        .iter()
        .map(|s| s.section.as_str())
        .collect();
    assert!(names.contains(&"shared_vectors"));
    assert!(names.contains(&"shared_row_map"));
    assert!(!names.contains(&"tree_vectors"));
    assert_eq!(summary.descriptor_storage, "shared");

    let row_map = summary
        .sections
        .iter()
        .find(|s| s.section == "shared_row_map")
        .unwrap();
    // One uint32 storage row per feature.
    assert_eq!(row_map.decoded_bytes, 3 * 4);
}

/// An explicit storage order is honoured, and answers do not depend on it.
///
/// The row map is what a reader follows, so reordering the corpus must be
/// invisible above the storage layer — that invisibility is what makes an
/// ordering policy safe to change.
#[test]
fn an_explicit_descriptor_order_is_stored_and_changes_no_answer() {
    let vectors = [0u8, 0, 1, 1, 9, 9];
    let dir = tempfile::tempdir().unwrap();
    let options = KdfWriteOptions {
        target_chunk_bytes: 90,
        ..KdfWriteOptions::shared(2)
    };

    let mut reference = None;
    for order in [None, Some(&[2u32, 0, 1][..]), Some(&[1u32, 2, 0][..])] {
        let path = dir
            .path()
            .join(format!("{}.kdf", order.map_or(0, |o| o[0] + 1)));
        let mut data = tiny_u8(&vectors);
        data.descriptor_order = order;
        write_kdf(&path, &data, None, &options).unwrap();
        let file = KdfFile::<u8>::open(&path, roomy()).unwrap();
        // Feature IDs, not rows: the same ID must give the same vector whatever
        // row it was stored in.
        let got: Vec<Vec<u8>> = (0..3).map(|id| file.shared_vector(id).unwrap()).collect();
        assert_eq!(
            got,
            vec![vec![0, 0], vec![1, 1], vec![9, 9]],
            "order={order:?}"
        );
        verify_kdf::<u8>(&path, roomy()).unwrap();
        match &reference {
            None => reference = Some(got),
            Some(first) => assert_eq!(first, &got),
        }
    }
}

/// A storage order that is not a permutation is refused, with the reason named.
#[test]
fn a_malformed_descriptor_order_is_refused() {
    let vectors = [0u8, 0, 1, 1, 9, 9];
    let dir = tempfile::tempdir().unwrap();
    for (order, want) in [
        (&[0u32, 1][..], "expected 3"),
        (&[0u32, 1, 1][..], "repeats"),
        (&[0u32, 1, 7][..], "out-of-range"),
    ] {
        let path = dir.path().join(format!(
            "bad{}.kdf",
            order.len() * 10 + order[2 % order.len()] as usize
        ));
        let mut data = tiny_u8(&vectors);
        data.descriptor_order = Some(order);
        let err = write_kdf(&path, &data, None, &KdfWriteOptions::shared(2))
            .expect_err("must reject")
            .to_string();
        assert!(err.contains(want), "order={order:?} gave {err:?}");
    }
}
