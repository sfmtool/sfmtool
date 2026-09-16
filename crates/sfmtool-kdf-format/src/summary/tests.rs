// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn offset_shapes_are_sized_without_decoding() {
    assert_eq!(
        decoded_bytes_from_name("features/block_offsets.303001.uint64.zst"),
        Some(303001 * 8)
    );
    assert_eq!(
        decoded_bytes_from_name("trees/0/chunks/0/chunk.18446744073709551615.1.uint8.zst"),
        None
    );
}

#[test]
fn metadata_expansion_is_bounded() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.kdf");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    zip.start_file(
        "metadata.json.zst",
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )
    .unwrap();
    zip.write_all(&zstd::bulk::compress(&vec![b' '; 1 << 20], 1).unwrap())
        .unwrap();
    zip.finish().unwrap();
    assert!(kdf_summary(&path, 1024).is_err());
}
