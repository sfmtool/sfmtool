// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// One block's worth of feature IDs, for as many blocks as asked for.
fn blocks(count: usize, per_block: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|b| (0..per_block).map(|i| (b * per_block + i) as u32).collect())
        .collect()
}

/// Encoding is per block, so how blocks are grouped into batches is invisible.
///
/// This is what makes the write's parallelism safe to change: the batch is only
/// the unit of hand-off, progress and cancellation, and a frame is decided by
/// its own bytes and the compression level. A regression that let a batch carry
/// state from one block into the next would change the frames here, and with
/// them the section digest and every file this crate writes.
#[test]
fn a_batch_boundary_changes_neither_a_frame_nor_the_section_digest() {
    let owned = blocks(700, 5);
    let ids: Vec<&[u32]> = owned.iter().map(Vec::as_slice).collect();
    // Something a compressor has to work at: constant blocks would compress to
    // the same handful of bytes whatever went wrong.
    let gather = |ids: &[u32], raw: &mut Vec<u8>| {
        for &id in ids {
            raw.extend_from_slice(&(id.wrapping_mul(2_654_435_761)).to_le_bytes());
        }
    };

    let mut reference = None;
    for batch in [1usize, 3, 256, 4096] {
        let mut frames = Vec::new();
        let mut digests = SectionDigests::new();
        for group in ids.chunks(batch) {
            for (frame, digest) in encode_batch(group, 3, gather).unwrap() {
                frames.extend_from_slice(&frame);
                digests.push(digest);
            }
        }
        let produced = (frames, digests.finish());
        match &reference {
            None => reference = Some(produced),
            Some(expected) => assert_eq!(
                *expected, produced,
                "batches of {batch} produced different bytes"
            ),
        }
    }
}
