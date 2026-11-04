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
         version {SFMR_FLOAT16_READINGS_VERSION} changed the column from float32 to float16"
    )
}
