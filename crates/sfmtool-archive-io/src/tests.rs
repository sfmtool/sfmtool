// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the shared archive container primitives.

use std::borrow::Cow;
use std::io::Cursor;

use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::{ZipArchive, ZipWriter};

use super::*;

const LEVEL: i32 = 3;

/// Build an in-memory archive with `f`, then hand back a reader over it.
fn round_trip(
    f: impl FnOnce(&mut ZipWriter<Cursor<Vec<u8>>>) -> Result<(), ArchiveIoError>,
) -> ZipArchive<Cursor<Vec<u8>>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    f(&mut zip).expect("write failed");
    let buf = zip.finish().expect("finish failed").into_inner();
    ZipArchive::new(Cursor::new(buf)).expect("archive open failed")
}

#[test]
fn json_entry_round_trips() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Meta {
        version: u32,
        name: String,
    }
    let meta = Meta {
        version: 3,
        name: "seoul_bull".to_string(),
    };

    let mut written = Vec::new();
    let mut archive = round_trip(|zip| {
        written = write_json_entry(zip, "metadata.json.zst", &meta, LEVEL)?;
        Ok(())
    });

    // The returned bytes are the uncompressed JSON, which is what the section
    // hash is computed over.
    assert_eq!(written, serde_json::to_vec(&meta).unwrap());
    let read: Meta = read_json_entry(&mut archive, "metadata.json.zst").unwrap();
    assert_eq!(read, meta);
}

#[test]
fn binary_array_round_trips() {
    let values: Vec<u32> = (0..64).map(|i| i * 7 + 1).collect();
    let mut archive = round_trip(|zip| {
        write_binary_entry(
            zip,
            "features/values.64.uint32.zst",
            bytemuck::cast_slice(&values),
            LEVEL,
        )
    });

    let read: Vec<u32> =
        read_binary_array(&mut archive, "features/values.64.uint32.zst", 64).expect("read failed");
    assert_eq!(read, values);
}

#[test]
fn binary_array_rejects_a_wrong_element_count() {
    let values: Vec<u32> = (0..8).collect();
    let mut archive = round_trip(|zip| {
        write_binary_entry(
            zip,
            "values.8.uint32.zst",
            bytemuck::cast_slice(&values),
            LEVEL,
        )
    });

    let err = read_binary_array::<_, u32>(&mut archive, "values.8.uint32.zst", 9).unwrap_err();
    assert!(
        matches!(err, ArchiveIoError::ShapeMismatch(ref m) if m.contains("expected 36 bytes")),
        "unexpected error: {err}"
    );
}

#[test]
fn empty_binary_array_round_trips() {
    let mut archive = round_trip(|zip| write_binary_entry(zip, "empty.0.uint32.zst", &[], LEVEL));
    let read: Vec<u32> = read_binary_array(&mut archive, "empty.0.uint32.zst", 0).unwrap();
    assert!(read.is_empty());
}

#[test]
fn uint128_array_round_trips() {
    let hashes: Vec<[u8; 16]> = (0u8..4).map(|i| [i; 16]).collect();
    let flat: Vec<u8> = hashes.iter().flat_map(|h| h.iter().copied()).collect();
    let mut archive =
        round_trip(|zip| write_binary_entry(zip, "hashes.4.uint128.zst", &flat, LEVEL));

    let read = read_uint128_array(&mut archive, "hashes.4.uint128.zst", 4).unwrap();
    assert_eq!(read, hashes);
}

#[test]
fn write_binary_entry_hashed_matches_hashing_by_hand() {
    // The rolling section hash must see exactly the uncompressed bytes, in
    // write order — this is what makes the two spellings interchangeable, and
    // what keeps previously written files verifiable.
    let a: Vec<u32> = (0..16).collect();
    let b: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();

    let mut rolling = Xxh3::new();
    round_trip(|zip| {
        write_binary_entry_hashed(
            zip,
            "a.16.uint32.zst",
            bytemuck::cast_slice(&a),
            LEVEL,
            &mut rolling,
        )?;
        write_binary_entry_hashed(
            zip,
            "b.8.float64.zst",
            bytemuck::cast_slice(&b),
            LEVEL,
            &mut rolling,
        )
    });

    let mut by_hand = Xxh3::new();
    by_hand.update(bytemuck::cast_slice(&a));
    by_hand.update(bytemuck::cast_slice(&b));

    assert_eq!(rolling.digest128(), by_hand.digest128());
}

#[test]
fn format_hash_is_zero_padded_lowercase_hex() {
    assert_eq!(format_hash(0), "0".repeat(32));
    assert_eq!(format_hash(0xabcdef), "00000000000000000000000000abcdef");
    assert_eq!(format_hash(u128::MAX), "f".repeat(32));
    // Round-trips against the digest the format crates store.
    assert_eq!(format_hash(xxh3_128(b"")).len(), 32);
}

#[test]
fn raw_to_u32_handles_unaligned_buffer() {
    // Build a u32 byte payload starting at an odd offset so the slice is
    // not 4-aligned — exactly the layout a freshly decompressed buffer can
    // land on, and the case where `bytemuck::cast_slice::<u8, u32>` panics.
    // The aligned-copy path must read it correctly instead.
    let mut backing = vec![0u8; 1];
    backing.extend_from_slice(&7u32.to_ne_bytes());
    backing.extend_from_slice(&4_000_000_000u32.to_ne_bytes());
    let unaligned = &backing[1..];
    assert_eq!(unaligned.as_ptr() as usize % 4, 1);
    // Unaligned input must fall back to the owned (copied) path, not panic.
    let got = raw_to_u32(unaligned);
    assert!(matches!(got, Cow::Owned(_)));
    assert_eq!(got.as_ref(), &[7u32, 4_000_000_000][..]);

    // A trailing partial u32 (truncated entry) is dropped, not panicked on.
    assert_eq!(
        raw_to_u32(&backing[1..backing.len() - 1]).as_ref(),
        &[7u32][..]
    );
}

#[test]
fn raw_to_f64_handles_unaligned_buffer() {
    let mut backing = vec![0u8; 1];
    backing.extend_from_slice(&(-1.5f64).to_ne_bytes());
    backing.extend_from_slice(&f64::MAX.to_ne_bytes());
    let unaligned = &backing[1..];
    assert_ne!(unaligned.as_ptr() as usize % 8, 0);

    let got = raw_to_f64(unaligned);
    assert!(matches!(got, Cow::Owned(_)));
    assert_eq!(got.as_ref(), &[-1.5f64, f64::MAX][..]);

    // A trailing partial f64 (truncated entry) is dropped, not panicked on.
    assert_eq!(
        raw_to_f64(&backing[1..backing.len() - 1]).as_ref(),
        &[-1.5f64][..]
    );
}

#[test]
fn raw_to_u32_borrows_an_aligned_buffer() {
    let values: Vec<u32> = vec![1, 2, 3, 4];
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    assert!(matches!(raw_to_u32(bytes), Cow::Borrowed(_)));
}

#[test]
fn missing_entry_is_a_zip_error() {
    let mut archive = round_trip(|zip| write_binary_entry(zip, "present.zst", &[1, 2, 3], LEVEL));
    let err = read_zst_entry(&mut archive, "absent.zst").unwrap_err();
    assert!(matches!(err, ArchiveIoError::Zip(_)), "unexpected: {err}");
}

#[test]
fn corrupt_zst_payload_is_an_invalid_format_error() {
    // An entry whose bytes are not a zstd frame at all.
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file(
        "bad.zst",
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )
    .unwrap();
    std::io::Write::write_all(&mut zip, b"not a zstd frame").unwrap();
    let buf = zip.finish().unwrap().into_inner();
    let mut archive = ZipArchive::new(Cursor::new(buf)).unwrap();

    let err = read_zst_entry(&mut archive, "bad.zst").unwrap_err();
    assert!(
        matches!(err, ArchiveIoError::InvalidFormat(ref m) if m.contains("bad.zst")),
        "unexpected: {err}"
    );
}
