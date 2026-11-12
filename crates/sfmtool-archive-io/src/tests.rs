// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the shared archive container primitives.

use std::borrow::Cow;
use std::io::Cursor;

use xxhash_rust::xxh3::Xxh3;
use zip::{ZipArchive, ZipWriter};

use super::*;

const LEVEL: i32 = 3;

#[test]
fn workspace_metadata_preserves_its_json_shape() {
    let workspace = WorkspaceMetadata {
        absolute_path: "/projects/example".to_string(),
        relative_path: "../example".to_string(),
        contents: WorkspaceContents {
            feature_tool: "colmap".to_string(),
            feature_type: "sift".to_string(),
            feature_options: serde_json::json!({"max_num_features": 8192}),
            feature_prefix_dir: "features/sift-colmap".to_string(),
        },
    };

    let json = serde_json::to_string(&workspace).unwrap();
    assert_eq!(
        json,
        r#"{"absolute_path":"/projects/example","relative_path":"../example","contents":{"feature_tool":"colmap","feature_type":"sift","feature_options":{"max_num_features":8192},"feature_prefix_dir":"features/sift-colmap"}}"#
    );

    let decoded: WorkspaceMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.absolute_path, workspace.absolute_path);
    assert_eq!(decoded.relative_path, workspace.relative_path);
    assert_eq!(
        decoded.contents.feature_tool,
        workspace.contents.feature_tool
    );
    assert_eq!(
        decoded.contents.feature_type,
        workspace.contents.feature_type
    );
    assert_eq!(
        decoded.contents.feature_options,
        workspace.contents.feature_options
    );
    assert_eq!(
        decoded.contents.feature_prefix_dir,
        workspace.contents.feature_prefix_dir
    );
}

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
fn parse_hash_inverts_format_hash() {
    for digest in [
        0,
        1,
        0xabcdef,
        0x0011_2233_4455_6677_8899_aabb_ccdd_eeff,
        u128::MAX,
    ] {
        assert_eq!(parse_hash(&format_hash(digest)), Ok(digest));
    }
}

#[test]
fn parse_hash_bytes_puts_the_first_hex_pair_first() {
    let bytes = parse_hash_bytes("00112233445566778899aabbccddeeff").unwrap();
    let expected: [u8; 16] = std::array::from_fn(|i| (i as u8) * 0x11);
    assert_eq!(bytes, expected);
    assert_eq!(
        parse_hash_bytes(&format_hash(0xabcdef)).unwrap(),
        0xabcdef_u128.to_be_bytes()
    );
}

#[test]
fn parse_hash_rejects_anything_but_32_lowercase_hex_digits() {
    let rejected = [
        String::new(),
        "ab".to_string(),
        "0".repeat(31),
        "0".repeat(33),
        // Uppercase: no writer produces it, and every verifier compares the
        // lowercase spelling.
        "ABCDEF0123456789ABCDEF0123456789".to_string(),
        format!("{}A", "0".repeat(31)),
        // `u128::from_str_radix` alone would accept a leading `+`.
        format!("+{}", "0".repeat(31)),
        format!("{}g", "0".repeat(31)),
        format!("{} ", "0".repeat(31)),
        // 32 bytes, but 31 characters.
        format!("{}\u{e9}", "0".repeat(30)),
    ];
    for hex in &rejected {
        assert_eq!(parse_hash(hex), Err(HashParseError), "{hex:?}");
        assert_eq!(parse_hash_bytes(hex), Err(HashParseError), "{hex:?}");
    }
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
        .map(|i| archive.by_index(i).unwrap().name().unwrap().into_owned())
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

// ── One-call decoding ───────────────────────────────────────────────────

/// An archive of one stored entry whose bytes are `stored`, as they are.
fn archive_of(name: &str, stored: &[u8]) -> ZipArchive<Cursor<Vec<u8>>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file(
        name,
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )
    .unwrap();
    std::io::Write::write_all(&mut zip, stored).unwrap();
    let buf = zip.finish().unwrap().into_inner();
    ZipArchive::new(Cursor::new(buf)).unwrap()
}

/// `len` bytes that zstd compresses, but not to nothing: runs of one symbol
/// broken by pseudo-random symbols from a four-symbol alphabet.
fn varied_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if i % 64 < 40 {
                b'a'
            } else {
                b'a' + (state % 4) as u8
            }
        })
        .collect()
}

/// `frame`, one zstd frame, with the content size its header records
/// rewritten to `size`, in a field of the width the header already has.
fn with_recorded_size(frame: &[u8], size: u64) -> Vec<u8> {
    let descriptor = frame[4];
    let single_segment = descriptor & 0x20 != 0;
    let dictionary_bytes = [0, 1, 2, 4][usize::from(descriptor & 3)];
    let at = 5 + usize::from(!single_segment) + dictionary_bytes;
    let mut out = frame.to_vec();
    match descriptor >> 6 {
        0 => {
            assert!(single_segment, "the frame records no size");
            out[at] = u8::try_from(size).unwrap();
        }
        1 => out[at..at + 2].copy_from_slice(&u16::try_from(size - 256).unwrap().to_le_bytes()),
        2 => out[at..at + 4].copy_from_slice(&u32::try_from(size).unwrap().to_le_bytes()),
        _ => out[at..at + 8].copy_from_slice(&size.to_le_bytes()),
    }
    assert_eq!(
        zstd::bulk::Decompressor::upper_bound(&out),
        Some(usize::try_from(size).unwrap()),
        "the rewritten header must record {size}"
    );
    out
}

#[test]
fn one_call_and_stream_decoding_return_the_same_bytes() {
    // Empty, one short block, one block, and several blocks of 128 KiB.
    for len in [0, 7, 1000, 300_000] {
        let payload = varied_bytes(len);
        let stored = zstd_compress(&payload, LEVEL).unwrap();
        // The one-call path applies to what `zstd_compress` writes, with or
        // without a size from the caller, and gives the stream's bytes.
        assert_eq!(
            decode_zst_one_call(&stored, Some(len)).as_deref(),
            Some(payload.as_slice()),
            "{len} bytes"
        );
        assert_eq!(
            decode_zst_one_call(&stored, None).as_deref(),
            Some(payload.as_slice()),
            "{len} bytes"
        );
        assert_eq!(decode_zst_stream("e.zst", &stored).unwrap(), payload);

        let mut archive = archive_of("e.zst", &stored);
        assert_eq!(read_zst_entry(&mut archive, "e.zst").unwrap(), payload);
        assert_eq!(
            read_binary_array::<_, u8>(&mut archive, "e.zst", len).unwrap(),
            payload
        );
    }
}

#[test]
fn a_frame_that_records_no_size_is_decoded_as_a_stream() {
    let payload = varied_bytes(5000);
    // The streaming encoder is not told the length, so the frame header
    // records none.
    let stored = zstd::stream::encode_all(payload.as_slice(), LEVEL).unwrap();
    assert_eq!(one_call_size(&stored, Some(payload.len())), None);

    let mut archive = archive_of("e.zst", &stored);
    assert_eq!(
        read_binary_array::<_, u8>(&mut archive, "e.zst", payload.len()).unwrap(),
        payload
    );
}

#[test]
fn a_recorded_size_that_lies_gives_the_stream_paths_error() {
    let payload = varied_bytes(1000);
    let stored = zstd_compress(&payload, LEVEL).unwrap();
    for claimed in [900, 1100] {
        let lying = with_recorded_size(&stored, claimed);
        let expected = decode_zst_stream("e.zst", &lying).unwrap_err().to_string();
        assert!(expected.contains("e.zst"), "{expected}");

        let mut archive = archive_of("e.zst", &lying);
        // Told the size the header claims, and the true size.
        for len in [claimed as usize, payload.len()] {
            let err = read_binary_array::<_, u8>(&mut archive, "e.zst", len).unwrap_err();
            assert!(matches!(err, ArchiveIoError::InvalidFormat(_)), "{err}");
            assert_eq!(err.to_string(), expected, "claimed {claimed}, told {len}");
        }
        let err = read_zst_entry(&mut archive, "e.zst").unwrap_err();
        assert_eq!(err.to_string(), expected);
    }
}

#[test]
fn the_one_call_size_is_bounded_by_the_stored_size_and_the_callers() {
    let payload = vec![0u8; 100_000];
    let stored = zstd_compress(&payload, LEVEL).unwrap();
    assert_eq!(one_call_size(&stored, None), Some(payload.len()));
    assert_eq!(
        one_call_size(&stored, Some(payload.len())),
        Some(payload.len())
    );
    // A caller whose metadata gives another size gets the stream path, and
    // with it the length check's error.
    assert_eq!(one_call_size(&stored, Some(payload.len() + 1)), None);

    // A few bytes of zstd data cannot expand to 4 GiB, so no buffer that
    // size is reserved, whatever the caller expects.
    let lying = with_recorded_size(&stored, u64::from(u32::MAX));
    assert!(lying.len() * MAX_ZSTD_EXPANSION < u32::MAX as usize);
    assert_eq!(one_call_size(&lying, None), None);
    assert_eq!(one_call_size(&lying, Some(u32::MAX as usize)), None);

    let mut archive = archive_of("e.zst", &lying);
    let err = read_binary_array::<_, u8>(&mut archive, "e.zst", u32::MAX as usize).unwrap_err();
    assert!(matches!(err, ArchiveIoError::InvalidFormat(_)), "{err}");
}

#[test]
fn a_truncated_entry_gives_the_stream_paths_error() {
    let payload = varied_bytes(300_000);
    let stored = zstd_compress(&payload, LEVEL).unwrap();
    let truncated = &stored[..stored.len() - 3];
    let expected = decode_zst_stream("e.zst", truncated)
        .unwrap_err()
        .to_string();

    let mut archive = archive_of("e.zst", truncated);
    let err = read_binary_array::<_, u8>(&mut archive, "e.zst", payload.len()).unwrap_err();
    assert!(matches!(err, ArchiveIoError::InvalidFormat(_)), "{err}");
    assert_eq!(err.to_string(), expected);
}

#[test]
fn an_empty_entry_gives_the_stream_paths_error() {
    // No stored bytes record no size of zero: they are no frame at all.
    assert_eq!(one_call_size(&[], None), None);
    let expected = decode_zst_stream("e.zst", &[]).unwrap_err().to_string();
    let mut archive = archive_of("e.zst", &[]);
    let err = read_binary_array::<_, u8>(&mut archive, "e.zst", 0).unwrap_err();
    assert_eq!(err.to_string(), expected);
    let err = read_zst_entry(&mut archive, "e.zst").unwrap_err();
    assert_eq!(err.to_string(), expected);
    let err = DecodedEntries::read_all(&mut archive).unwrap_err();
    assert_eq!(err.to_string(), expected);
}

#[test]
fn the_stored_bytes_reservation_is_capped_at_the_central_directory() {
    assert!(stored_buffer(1 << 20, 1000).capacity() < 2000);
    assert!(stored_buffer(u64::MAX, 1000).capacity() < 2000);
    assert!(stored_buffer(500, 1000).capacity() >= 500);
}

#[test]
fn an_element_count_too_large_to_hold_is_a_shape_mismatch() {
    let mut archive = archive_of("e.zst", &zstd_compress(&[0; 16], LEVEL).unwrap());
    let err = read_binary_array::<_, u64>(&mut archive, "e.zst", usize::MAX / 4).unwrap_err();
    assert!(matches!(err, ArchiveIoError::ShapeMismatch(_)), "{err}");
    let err = read_uint128_array(&mut archive, "e.zst", usize::MAX / 8).unwrap_err();
    assert!(matches!(err, ArchiveIoError::ShapeMismatch(_)), "{err}");
}

// ── Atomic writes ───────────────────────────────────────────────────────

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's files.
fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("archive_io_atomic_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// The names of the files in `dir`, sorted.
fn entries(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("a readable dir")
        .map(|e| {
            e.expect("a readable entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn a_successful_write_leaves_the_target_and_no_temporary_file() {
    let dir = temp_dir("success");
    let path = dir.join("out.bin");

    let bytes: usize = write_atomically(&path, |file| {
        std::io::Write::write_all(file, b"hello")?;
        Ok::<usize, std::io::Error>(5)
    })
    .expect("a writable path");

    assert_eq!(bytes, 5, "the closure's value comes back");
    assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    assert_eq!(entries(&dir), vec!["out.bin".to_string()]);
}

#[test]
fn overwriting_an_existing_target_replaces_it() {
    // The rename has to replace, not fail, on every platform this builds for.
    let dir = temp_dir("overwrite");
    let path = dir.join("out.bin");
    std::fs::write(&path, b"old content, longer than the new").unwrap();

    write_atomically(&path, |file| {
        std::io::Write::write_all(file, b"new")?;
        Ok::<(), std::io::Error>(())
    })
    .expect("a writable path");

    assert_eq!(std::fs::read(&path).unwrap(), b"new");
    assert_eq!(entries(&dir), vec!["out.bin".to_string()]);
}

#[test]
fn a_failed_write_leaves_the_previous_target_untouched() {
    // The whole point: the target is either what it was or what was written,
    // never a prefix of what was being written.
    let dir = temp_dir("failure");
    let path = dir.join("out.bin");
    std::fs::write(&path, b"the only copy").unwrap();

    let error = write_atomically(&path, |file| {
        // Some of the new content reaches the temporary file before the failure,
        // which is exactly the case a direct write would have destroyed.
        std::io::Write::write_all(file, b"half of something else")?;
        Err::<(), std::io::Error>(std::io::Error::other("ran out of things to say"))
    })
    .expect_err("the closure failed");

    assert_eq!(error.to_string(), "ran out of things to say");
    assert_eq!(std::fs::read(&path).unwrap(), b"the only copy");
    assert_eq!(entries(&dir), vec!["out.bin".to_string()]);
}

#[test]
fn a_panic_in_the_write_leaves_no_temporary_file() {
    let dir = temp_dir("panic");
    let path = dir.join("out.bin");
    std::fs::write(&path, b"the only copy").unwrap();

    let panicked = std::panic::catch_unwind(|| {
        write_atomically(&path, |file| -> Result<(), std::io::Error> {
            std::io::Write::write_all(file, b"half of").unwrap();
            panic!("something went wrong mid-write");
        })
    });

    assert!(panicked.is_err(), "the panic propagates");
    assert_eq!(std::fs::read(&path).unwrap(), b"the only copy");
    assert_eq!(entries(&dir), vec!["out.bin".to_string()]);
}

#[test]
fn a_missing_parent_directory_is_an_error_and_creates_nothing() {
    // The same answer a direct write gives. Whether a missing parent should be
    // created is a decision about what a path means, and it belongs to the
    // caller: some writers create it first and some deliberately do not.
    let dir = temp_dir("parent");
    let path = dir.join("nested").join("out.bin");

    let error = write_atomically(&path, |file| {
        std::io::Write::write_all(file, b"x")?;
        Ok::<(), std::io::Error>(())
    })
    .expect_err("no such directory");

    assert_eq!(error.kind(), std::io::ErrorKind::NotFound, "{error}");
    assert!(!dir.join("nested").exists());
    assert!(entries(&dir).is_empty(), "{:?}", entries(&dir));
}
#[test]
fn section_digests_preserve_big_endian_content_identity() {
    use crate::SectionDigests;
    use xxhash_rust::xxh3::xxh3_128;

    let first = 0x00112233445566778899aabbccddeeff_u128;
    let second_bytes = b"uncompressed entry";
    let second = xxh3_128(second_bytes);
    let mut reference = Vec::new();
    reference.extend_from_slice(&first.to_be_bytes());
    reference.extend_from_slice(&second.to_be_bytes());

    let mut digests = SectionDigests::new();
    digests.push(first);
    digests.push_bytes(second_bytes);
    assert_eq!(digests.finish(), xxh3_128(&reference));

    for count in [0, 1, 2, 15, 16, 32] {
        let mut old_fold = Vec::new();
        let mut streamed = SectionDigests::new();
        for n in 0..count {
            let digest = xxh3_128(&(n as u128).to_le_bytes());
            old_fold.extend_from_slice(&digest.to_be_bytes());
            streamed.push(digest);
        }
        assert_eq!(streamed.finish(), xxh3_128(&old_fold), "{count} sections");
    }
}

/// A zstd frame made by hand of `blocks` RLE blocks of `block_size` bytes of
/// `7`, with frame header descriptor `descriptor`, then `window` (the window
/// descriptor byte, absent from a single-segment frame) and the content size
/// field `content_size` as written.
fn rle_frame(
    descriptor: u8,
    window: &[u8],
    content_size: &[u8],
    blocks: u32,
    block_size: u32,
) -> Vec<u8> {
    let mut frame = 0xFD2F_B528_u32.to_le_bytes().to_vec();
    frame.push(descriptor);
    frame.extend_from_slice(window);
    frame.extend_from_slice(content_size);
    for i in 0..blocks {
        let last = u32::from(i + 1 == blocks);
        let header = last | (1 << 1) | (block_size << 3);
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.push(7);
    }
    frame
}

#[test]
fn a_frame_needing_a_larger_window_than_the_stream_accepts_gives_the_stream_paths_error() {
    // Single segment, so the window is the content size: 1032 blocks of
    // 128 KiB, past the stream decoder's 2^27 + 1.
    let size = 1032u64 * 128 * 1024;
    let single = rle_frame(0xE0, &[], &size.to_le_bytes(), 1032, 128 * 1024);
    // Not single segment, with a window log of 28 and two blocks of
    // 128 KiB: more than the stream decoder's output buffer, which a frame
    // must be for the stream decoder to check its window rather than decode
    // it whole in one step.
    let windowed_size = 2u32 * 128 * 1024;
    let windowed = rle_frame(
        0x80,
        &[18 << 3],
        &windowed_size.to_le_bytes(),
        2,
        128 * 1024,
    );
    for (frame, len) in [(single, size as usize), (windowed, windowed_size as usize)] {
        assert_eq!(
            zstd::bulk::Decompressor::upper_bound(&frame),
            Some(len),
            "a valid frame header"
        );
        assert_eq!(one_call_size(&frame, None), None, "{len} bytes");
        let expected = decode_zst_stream("e.zst", &frame).unwrap_err().to_string();
        assert!(expected.contains("too much memory"), "{expected}");

        let mut archive = archive_of("e.zst", &frame);
        let err = read_zst_entry(&mut archive, "e.zst").unwrap_err();
        assert_eq!(err.to_string(), expected);
        let err = read_binary_array::<_, u8>(&mut archive, "e.zst", len).unwrap_err();
        assert_eq!(err.to_string(), expected);
        let err = DecodedEntries::read_all(&mut archive).unwrap_err();
        assert_eq!(err.to_string(), expected);
    }
}
