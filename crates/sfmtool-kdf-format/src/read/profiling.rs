// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use sfmtool_progress::Progress;

use super::*;

#[cfg(windows)]
#[test]
fn independent_handles_keep_the_open_snapshot_after_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.bin");
    std::fs::write(&path, b"old").unwrap();
    let mut original = std::fs::File::open(&path).unwrap();
    std::fs::rename(&path, dir.path().join("previous.bin")).unwrap();
    std::fs::write(&path, b"new").unwrap();
    let reopened = independent_read_handle(&original).unwrap();
    let mut out = [0; 3];
    read_at_exact(&reopened, &mut out, 0).unwrap();
    assert_eq!(&out, b"old");
    assert_eq!(
        original.stream_position().unwrap(),
        0,
        "reopening must not share the cursor"
    );
}

#[test]
fn chunk_shapes_cannot_understate_admission_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape.kdf");
    crate::write_kdf(
        &path,
        &KdfForestData {
            vectors: &[0u8],
            feature_count: 1,
            dimension: 1,
            trees: vec![KdfTree {
                nodes: vec![KdfNode::Leaf { start: 0, len: 1 }],
                feature_ids: vec![0],
            }],
            provenance: None,
            descriptor_order: None,
        },
        None,
        &KdfWriteOptions {
            target_descriptor_block_bytes: 4,
            ..Default::default()
        },
        &Progress::none(),
    )
    .unwrap();
    let file = KdfFile::<u8>::open(&path, LazyKdForestOptions::default()).unwrap();
    let mut metadata: Metadata =
        serde_json::from_value(serde_json::to_value(&file.metadata).unwrap()).unwrap();
    metadata.trees[0].chunks[0].decoded_bytes -= 1;
    let error = validate_metadata::<u8>(&metadata, &file.hashes, &LazyKdForestOptions::default())
        .unwrap_err();
    assert!(
        matches!(error, KdfError::InvalidFormat(ref message) if message.contains("disagree with its shape"))
    );
}

#[test]
fn concurrent_positional_reads_and_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ranges.bin");
    let data: Vec<u8> = (0..65536)
        .map(|i| ((i * 17 + i / 257) % 251) as u8)
        .collect();
    std::fs::write(&path, &data).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let file = &file;
            let data = &data;
            scope.spawn(move || {
                for i in 0..1000 {
                    let offset = (worker * 7919 + i * 2311) % (data.len() - 37);
                    let mut out = [0u8; 37];
                    read_at_exact(file, &mut out, offset as u64).unwrap();
                    assert_eq!(&out, &data[offset..offset + 37]);
                }
            });
        }
    });
    assert_eq!(
        read_at_exact(&file, &mut [0; 2], 65535).unwrap_err().kind(),
        std::io::ErrorKind::UnexpectedEof
    );
    read_at_exact(&file, &mut [], 65536).unwrap();
}

/// Diagnostic decomposition, not a throughput test: time calls separately
/// on one thread, with an OS-warm file and no decoded-cache lookup.
#[test]
#[ignore = "set KDF_PROFILE_PATH to a u8 file; run in release mode"]
fn profile_corpus_misses() {
    let path = std::env::var("KDF_PROFILE_PATH").expect("KDF_PROFILE_PATH");
    let file = KdfFile::<u8>::open(Path::new(&path), LazyKdForestOptions::default()).unwrap();
    let corpus = &file.corpus;
    let (rows, blocks) = file.descriptor_block_shape();
    let mut totals = [std::time::Duration::ZERO; 5];
    let samples = 20_000;
    for i in 0..samples {
        let b = (i * 7919) % blocks;
        let from = corpus.offsets[b];
        let length = (corpus.offsets[b + 1] - from) as usize;
        let declared = rows.min(file.len() - b * rows) * file.dim();
        let start = std::time::Instant::now();
        let mut frame = vec![0u8; length];
        {
            let mut handle = &corpus.file;
            handle
                .seek(std::io::SeekFrom::Start(corpus.data_start + from))
                .unwrap();
            handle.read_exact(&mut frame).unwrap();
        }
        totals[0] += start.elapsed();
        let start = std::time::Instant::now();
        let mut positioned = vec![0u8; length];
        read_at_exact(&corpus.file, &mut positioned, corpus.data_start + from).unwrap();
        totals[4] += start.elapsed();
        assert_eq!(positioned, frame);
        let start = std::time::Instant::now();
        let raw = zstd::bulk::decompress(&frame, declared).unwrap();
        totals[1] += start.elapsed();
        let start = std::time::Instant::now();
        let reused = decode_frame(&frame, declared).unwrap();
        totals[3] += start.elapsed();
        assert_eq!(raw, reused);
        let start = std::time::Instant::now();
        // Hashed here only to keep the cost in the "hash+copy" column: a block
        // read does not hash, and the digest it would produce is folded into a
        // section digest the whole-file verify recomputes.
        std::hint::black_box(hash_string(xxh3_128(&raw)));
        std::hint::black_box(bytes_to_pod::<u8>("profile", &raw, declared).unwrap());
        totals[2] += start.elapsed();
    }
    for mode in [0, 1, 2] {
        let reader = KdfFile::<u8>::open(
            Path::new(&path),
            LazyKdForestOptions {
                cache_bytes: 16 << 20,
                query_workers: if mode == 0 { 1 } else { 4 },
                ..Default::default()
            },
        )
        .unwrap();
        let start = std::time::Instant::now();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let reader = &reader;
                let path = &path;
                scope.spawn(move || {
                    let private_handle = std::fs::File::open(path).unwrap();
                    for i in (worker..samples).step_by(4) {
                        let b = (i * 7919) % blocks;
                        if mode == 1 {
                            std::hint::black_box(reader.descriptor_block(b as u32).unwrap());
                        } else {
                            let declared = rows.min(reader.len() - b * rows) * reader.dim();
                            let raw = if mode == 2 {
                                let corpus = &reader.corpus;
                                let from = corpus.offsets[b];
                                let mut frame = vec![0u8; (corpus.offsets[b + 1] - from) as usize];
                                read_at_exact(
                                    &private_handle,
                                    &mut frame,
                                    corpus.data_start + from,
                                )
                                .unwrap();
                                decode_frame(&frame, declared).unwrap()
                            } else {
                                reader
                                    .read_corpus_frame(
                                        &reader.corpus,
                                        "descriptor",
                                        b as u32,
                                        declared,
                                    )
                                    .unwrap()
                                    .0
                            };
                            std::hint::black_box(hash_string(xxh3_128(&raw)));
                            std::hint::black_box(
                                bytes_to_pod::<u8>("profile", &raw, declared).unwrap(),
                            );
                        }
                    }
                });
            }
        });
        eprintln!(
            "4 workers mode={mode} ns/completed-block={:.0} stats={:?}",
            start.elapsed().as_nanos() as f64 / samples as f64,
            reader.io_stats()
        );
    }
    eprintln!("samples={samples} block_rows={rows} ns/block read+allocation={:.0} zstd={:.0} hash+copy={:.0} reused-zstd={:.0} positional-read={:.0}",
        totals[0].as_nanos() as f64 / samples as f64,
        totals[1].as_nanos() as f64 / samples as f64,
        totals[2].as_nanos() as f64 / samples as f64,
        totals[3].as_nanos() as f64 / samples as f64,
        totals[4].as_nanos() as f64 / samples as f64);
}
