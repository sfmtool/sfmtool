// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The conversion of the `float16` reading columns (version 16+) at the I/O
//! boundary.
//!
//! The values are `f32` in memory. A writer rounds each one to the nearest
//! `float16`, ties to even, and stores that; a reader widens it to `f32`,
//! which is exact. A writer also rounds the value it was handed in place, so
//! after a write the value equals what a reader gets back from the file, and
//! a load and a save write the same bytes and the same content hash. See
//! `specs/formats/sfmr-file-format.md` § "Version 15 → Version 16".

use half::f16;

use crate::SFMR_FLOAT16_READINGS_VERSION;

/// Whether a file of `version` stores the reading columns that have a
/// `float16` form as `float16`.
pub(crate) const fn stores_float16(version: u32) -> bool {
    version >= SFMR_FLOAT16_READINGS_VERSION
}

/// `value` rounded to the nearest `float16`, ties to even, and widened back
/// to `f32`: what a version 16 file stores for it and what a reader gets
/// back.
///
/// `NaN` stays `NaN` and `±∞` and `±0` stay as they are. A magnitude of
/// `2^-25` (about `3 · 10^-8`) or less rounds to zero, one below `2^-14`
/// (about `6.1 · 10^-5`) to a subnormal with less than 11 significant bits,
/// and a finite magnitude of 65,520 or more to infinity: the rounding IEEE
/// 754 defines.
pub(crate) fn round_to_float16(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

/// Round every value of `values` in place by [`round_to_float16`].
pub(crate) fn round_all_to_float16<'a>(values: impl IntoIterator<Item = &'a mut f32>) {
    for v in values {
        *v = round_to_float16(*v);
    }
}

/// The little-endian `float16` bytes of `values`, each rounded by
/// [`round_to_float16`].
pub(crate) fn float16_bytes<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

/// `score` rounded to what a version 16 file stores for it: its ZNCC deficit
/// `w = 1 − score` in `f32`, rounded by [`round_to_float16`], and `1 − w` in
/// `f32` again, which a reader computes from the stored `w`.
///
/// The second subtraction, and the `1 − z` of a later save, are exact once
/// `w` is a `float16` in `[0, 2]`: every bit of `w`, of `1 − w` and of `1`
/// lies at or above `2^-24`, and below `1` in `1 − w` when `w` is below `1`,
/// so each result fits `f32`'s 24 significant bits. So a value this returns maps to itself, and a
/// second save stores the same bytes. A score of `1` stores `0` and reads
/// back `1`; `NaN` stays `NaN`.
pub(crate) fn round_score_by_deficit(score: f32) -> f32 {
    1.0 - round_to_float16(1.0 - score)
}

/// Round every score of `scores` in place by [`round_score_by_deficit`].
pub(crate) fn round_all_scores_by_deficit<'a>(scores: impl IntoIterator<Item = &'a mut f32>) {
    for s in scores {
        *s = round_score_by_deficit(*s);
    }
}

/// The little-endian `float16` bytes of the ZNCC deficits `1 − z` of
/// `scores`, each computed in `f32` and rounded by [`round_to_float16`].
pub(crate) fn deficit_float16_bytes<'a>(scores: impl IntoIterator<Item = &'a f32>) -> Vec<u8> {
    scores
        .into_iter()
        .flat_map(|&z| f16::from_f32(1.0 - z).to_bits().to_le_bytes())
        .collect()
}

/// The scores `1 − w`, in `f32`, of `bits`, the stored `float16` ZNCC
/// deficits `w`.
pub(crate) fn scores_from_deficits(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| 1.0 - f16::from_bits(b).to_f32())
        .collect()
}

/// `bits`, the stored `float16` values, widened to `f32`.
pub(crate) fn widen_float16(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// The `f32` values of a little-endian `float16` entry, or a message when its
/// length is not `expected` values.
pub(crate) fn float16_values(raw: &[u8], name: &str, expected: usize) -> Result<Vec<f32>, String> {
    if raw.len() != expected * 2 {
        return Err(format!(
            "{name} holds {} bytes, not {}",
            raw.len(),
            expected * 2
        ));
    }
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
        .collect())
}

/// Why a file of `version` is refused for holding `found`, the entry of a
/// reading column in the dtype another version stores it in, where `expected`
/// is the entry its own version names.
pub(crate) fn wrong_dtype_error(version: u32, found: &str, expected: &str) -> String {
    format!(
        "a version {version} file stores {expected}, and this one holds {found} instead; \
         version {SFMR_FLOAT16_READINGS_VERSION} changed the column's entry"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rounded_score_maps_to_itself_and_stores_its_own_deficit() {
        // Every float16 deficit in [0, 2]: the score a reader computes from it
        // rounds to itself and stores the same bits again.
        for bits in 0..=f16::from_f32(2.0).to_bits() {
            let w = f16::from_bits(bits).to_f32();
            let z = 1.0 - w;
            assert_eq!(round_score_by_deficit(z).to_bits(), z.to_bits(), "w = {w}");
            assert_eq!(1.0 - z, w, "w = {w}");
            assert_eq!(
                deficit_float16_bytes([z].iter()),
                bits.to_le_bytes(),
                "w = {w}"
            );
        }
        assert_eq!(round_score_by_deficit(1.0), 1.0);
        assert!(round_score_by_deficit(f32::NAN).is_nan());
    }
}
