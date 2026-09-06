// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the shared archive container primitives.

use std::borrow::Cow;
use std::io::Cursor;

use xxhash_rust::xxh3::Xxh3;
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

/// Place `payload` at an address that is *guaranteed* congruent to 1 mod
/// `align`, and return the backing buffer plus that offset.
///
/// `Vec<u8>` only promises 1-byte alignment, so "prepend one byte and hope"
/// happens to work on every mainstream allocator but is not guaranteed by
/// anything. Deriving the offset from the actual base address makes the
/// misaligned branch unconditional rather than allocator-dependent.
fn misaligned(payload: &[u8], align: usize) -> (Vec<u8>, usize) {
    let mut backing = vec![0u8; align + payload.len()];
    let base = backing.as_ptr() as usize;
    let offset = (1 + align - (base % align)) % align;
    backing[offset..offset + payload.len()].copy_from_slice(payload);
    debug_assert_eq!((base + offset) % align, 1 % align);
    (backing, offset)
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
    let mut archive = round_trip(|zip| {
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

    // Hashing is only half the contract — the entries must actually be in the
    // archive. Without this, gutting the write and keeping the hash passes.
    let read_a: Vec<u32> = read_binary_array(&mut archive, "a.16.uint32.zst", 16).unwrap();
    let read_b: Vec<f64> = read_binary_array(&mut archive, "b.8.float64.zst", 8).unwrap();
    assert_eq!(read_a, a);
    assert_eq!(read_b, b);
}

#[test]
fn format_hash_is_zero_padded_lowercase_hex() {
    assert_eq!(format_hash(0), "0".repeat(32));
    assert_eq!(format_hash(0xabcdef), "00000000000000000000000000abcdef");
    assert_eq!(format_hash(u128::MAX), "f".repeat(32));
    assert_eq!(format_hash(0xABCDEF), format_hash(0xabcdef));
}

#[test]
fn uint128_array_rejects_a_wrong_hash_count() {
    let flat: Vec<u8> = (0u8..3).flat_map(|i| [i; 16]).collect();
    let mut archive =
        round_trip(|zip| write_binary_entry(zip, "hashes.3.uint128.zst", &flat, LEVEL));

    let err = read_uint128_array(&mut archive, "hashes.3.uint128.zst", 4).unwrap_err();
    assert!(
        matches!(err, ArchiveIoError::ShapeMismatch(ref m) if m.contains("expected 64 bytes")),
        "unexpected error: {err}"
    );
}

#[test]
fn malformed_json_entry_is_a_json_error() {
    // A well-formed zstd entry whose payload is not JSON.
    let mut archive = round_trip(|zip| write_binary_entry(zip, "meta.json.zst", b"{ nope", LEVEL));
    let err = read_json_entry::<_, serde_json::Value>(&mut archive, "meta.json.zst").unwrap_err();
    assert!(matches!(err, ArchiveIoError::Json(_)), "unexpected: {err}");
}

#[test]
fn zstd_compress_honours_the_level() {
    // The payload has to be chosen with care: a trivially repetitive buffer
    // bottoms out at the same size for every level (both 25 bytes), and
    // incompressible noise produces byte-identical output. Pseudo-random data
    // drawn from a 4-symbol alphabet sits in between, where the search effort
    // actually pays: measured 20,030 bytes at level 1 vs 16,135 at level 19.
    let mut state: u32 = 12_345;
    let data: Vec<u8> = (0..64_000)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) % 4) as u8
        })
        .collect();

    let fast = zstd_compress(&data, 1).unwrap();
    let slow = zstd_compress(&data, 19).unwrap();
    assert!(
        slow.len() < fast.len(),
        "level 19 ({}) should beat level 1 ({})",
        slow.len(),
        fast.len()
    );
    // Both must still decode back to the original.
    for c in [&fast, &slow] {
        let mut out = Vec::new();
        zstd::stream::copy_decode(&c[..], &mut out).unwrap();
        assert_eq!(out, data);
    }
}

#[test]
fn cast_or_copy_falls_back_when_the_buffer_is_misaligned() {
    // `read_binary_array`'s misaligned branch cannot be driven through the
    // archive path — the alignment of a decompressed `Vec<u8>` is the
    // allocator's choice — so exercise the extracted helper directly.
    let values: Vec<u32> = vec![9, 8, 7, 6];
    let aligned: &[u8] = bytemuck::cast_slice(&values);
    assert_eq!(cast_or_copy::<u32>(aligned, 4), values);

    let (backing, off) = misaligned(aligned, 4);
    let unaligned = &backing[off..off + aligned.len()];
    assert_ne!(unaligned.as_ptr() as usize % 4, 0);
    assert_eq!(cast_or_copy::<u32>(unaligned, 4), values);

    assert!(cast_or_copy::<u32>(&[], 0).is_empty());
}

#[test]
fn raw_to_u32_handles_unaligned_buffer() {
    // A u32 payload at an address that is not 4-aligned — exactly the layout a
    // freshly decompressed buffer can land on, and the case where
    // `bytemuck::cast_slice::<u8, u32>` panics. The copy path must read it
    // correctly instead.
    let mut payload = Vec::new();
    payload.extend_from_slice(&7u32.to_ne_bytes());
    payload.extend_from_slice(&4_000_000_000u32.to_ne_bytes());
    let (backing, off) = misaligned(&payload, 4);
    let unaligned = &backing[off..off + payload.len()];
    assert_ne!(unaligned.as_ptr() as usize % 4, 0);

    // Unaligned input must fall back to the owned (copied) path, not panic.
    let got = raw_to_u32(unaligned);
    assert!(matches!(got, Cow::Owned(_)));
    assert_eq!(got.as_ref(), &[7u32, 4_000_000_000][..]);

    // A trailing partial u32 (truncated entry) is dropped, not panicked on.
    assert_eq!(
        raw_to_u32(&unaligned[..unaligned.len() - 1]).as_ref(),
        &[7u32][..]
    );
}

#[test]
fn raw_to_f64_handles_unaligned_buffer() {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(-1.5f64).to_ne_bytes());
    payload.extend_from_slice(&f64::MAX.to_ne_bytes());
    let (backing, off) = misaligned(&payload, 8);
    let unaligned = &backing[off..off + payload.len()];
    assert_ne!(unaligned.as_ptr() as usize % 8, 0);

    let got = raw_to_f64(unaligned);
    assert!(matches!(got, Cow::Owned(_)));
    assert_eq!(got.as_ref(), &[-1.5f64, f64::MAX][..]);

    // A trailing partial f64 (truncated entry) is dropped, not panicked on.
    assert_eq!(
        raw_to_f64(&unaligned[..unaligned.len() - 1]).as_ref(),
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

/// Build an archive holding one of every kind of entry a format crate reads:
/// JSON, a `u32` column, an `f32` column, a digest column, and an empty one.
fn mixed_archive() -> ZipArchive<Cursor<Vec<u8>>> {
    let counts: Vec<u32> = (0..4096).map(|i| i * 3 + 1).collect();
    let residuals: Vec<f32> = (0..4096).map(|i| i as f32 * -0.25).collect();
    let digests: Vec<u8> = (0u8..64).flat_map(|i| [i; 16]).collect();
    round_trip(|zip| {
        write_json_entry(
            zip,
            "metadata.json.zst",
            &serde_json::json!({"version": 6}),
            LEVEL,
        )?;
        write_json_entry(zip, "images/names.json.zst", &vec!["a.jpg", "b.jpg"], LEVEL)?;
        write_binary_entry(
            zip,
            "counts.4096.uint32.zst",
            bytemuck::cast_slice(&counts),
            LEVEL,
        )?;
        write_binary_entry(
            zip,
            "residuals.4096.float32.zst",
            bytemuck::cast_slice(&residuals),
            LEVEL,
        )?;
        write_binary_entry(zip, "hashes.64.uint128.zst", &digests, LEVEL)?;
        write_binary_entry(zip, "empty.0.uint32.zst", &[], LEVEL)
    })
}

#[test]
fn decoded_entries_are_byte_identical_to_the_entry_at_a_time_reads() {
    // The bit-identity claim the batch path rests on: parallel decoding is a
    // scheduling change, so every buffer it hands out must equal the one
    // `read_zst_entry` produces for the same entry, and the typed accessors
    // must agree with the free functions built on it.
    let mut archive = mixed_archive();
    let names: Vec<String> = (0..archive.len())
        .map(|i| archive.by_index(i).unwrap().name().to_string())
        .collect();

    let decoded = DecodedEntries::read_all(&mut archive).unwrap();
    for name in &names {
        assert_eq!(
            decoded.zst_entry(name).unwrap(),
            read_zst_entry(&mut archive, name).unwrap(),
            "{name} decoded differently"
        );
    }

    let meta: serde_json::Value = decoded.json_entry("metadata.json.zst").unwrap();
    assert_eq!(
        meta,
        read_json_entry::<_, serde_json::Value>(&mut archive, "metadata.json.zst").unwrap()
    );
    assert_eq!(
        decoded
            .binary_array::<u32>("counts.4096.uint32.zst", 4096)
            .unwrap(),
        read_binary_array::<_, u32>(&mut archive, "counts.4096.uint32.zst", 4096).unwrap()
    );
    let batch_f32 = decoded
        .binary_array::<f32>("residuals.4096.float32.zst", 4096)
        .unwrap();
    let seq_f32 =
        read_binary_array::<_, f32>(&mut archive, "residuals.4096.float32.zst", 4096).unwrap();
    assert!(
        batch_f32
            .iter()
            .map(|v| v.to_bits())
            .eq(seq_f32.iter().map(|v| v.to_bits())),
        "float column decoded differently"
    );
    assert_eq!(
        decoded.uint128_array("hashes.64.uint128.zst", 64).unwrap(),
        read_uint128_array(&mut archive, "hashes.64.uint128.zst", 64).unwrap()
    );
    assert!(decoded
        .binary_array::<u32>("empty.0.uint32.zst", 0)
        .unwrap()
        .is_empty());
}

#[test]
fn decoded_entries_reject_a_wrong_element_count_like_the_sequential_read() {
    let mut archive = mixed_archive();
    let decoded = DecodedEntries::read_all(&mut archive).unwrap();

    let batch = decoded
        .binary_array::<u32>("counts.4096.uint32.zst", 4095)
        .unwrap_err();
    let sequential =
        read_binary_array::<_, u32>(&mut archive, "counts.4096.uint32.zst", 4095).unwrap_err();
    assert_eq!(batch.to_string(), sequential.to_string());

    let batch = decoded
        .uint128_array("hashes.64.uint128.zst", 63)
        .unwrap_err();
    let sequential = read_uint128_array(&mut archive, "hashes.64.uint128.zst", 63).unwrap_err();
    assert_eq!(batch.to_string(), sequential.to_string());
}

#[test]
fn a_missing_entry_is_a_zip_error_on_the_batch_path_too() {
    let mut archive = mixed_archive();
    let decoded = DecodedEntries::read_all(&mut archive).unwrap();
    let err = decoded.zst_entry("absent.zst").unwrap_err();
    assert!(matches!(err, ArchiveIoError::Zip(_)), "unexpected: {err}");
    assert_eq!(
        err.to_string(),
        read_zst_entry(&mut archive, "absent.zst")
            .unwrap_err()
            .to_string()
    );
}

#[test]
fn several_unreadable_entries_report_the_first_in_archive_order() {
    // Decoding runs concurrently, so the failure a caller sees must come from
    // the archive's own order rather than from whichever worker finished
    // first — otherwise the same broken file reports differently run to run.
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    for name in ["a_bad.zst", "b_good.zst", "c_bad.zst"] {
        zip.start_file(name, stored).unwrap();
        if name.ends_with("bad.zst") {
            std::io::Write::write_all(&mut zip, b"not a zstd frame").unwrap();
        } else {
            std::io::Write::write_all(&mut zip, &zstd_compress(b"payload", LEVEL).unwrap())
                .unwrap();
        }
    }
    let buf = zip.finish().unwrap().into_inner();

    for _ in 0..8 {
        let mut archive = ZipArchive::new(Cursor::new(buf.clone())).unwrap();
        let err = DecodedEntries::read_all(&mut archive).unwrap_err();
        assert!(
            matches!(err, ArchiveIoError::InvalidFormat(ref m) if m.contains("a_bad.zst")),
            "unexpected: {err}"
        );
    }
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
