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
        if name.starts_with("features/blocks/0/") {
            raw[0] ^= 0x55;
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
