// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Row gathering for the flat row-major arrays the matchers pass around.
//!
//! Every array the matchers in [`super`] carry is a flat `N × stride` buffer
//! whose rows are indexed by feature: positions (stride 2), affine shapes (4),
//! descriptors (`desc_len`). Both sweeps reorder features — by Y in
//! [`super::sweep`], by polar angle in [`super::polar`] — and must carry every
//! parallel array through the *same* permutation, or a window index will name
//! one feature's descriptor and another's shape.

/// Gather `stride`-wide rows out of a flat row-major array, in the order given.
///
/// # Parameters
///
/// * `rows` – The source array, flat row-major with `stride` elements per row.
/// * `stride` – Elements per row.
/// * `row_indices` – Source row indices, in the output order. Callers that sort
///   a filtered subset compose the two levels here — `sort_idx.iter().map(|&si|
///   valid[si])` — rather than passing both and re-deriving the composition
///   inside.
///
/// # Returns
///
/// A new flat array holding the named rows: `stride` elements for each index
/// `row_indices` yields, in the order it yields them.
///
/// # Panics
///
/// If any index names a row that is not wholly inside `rows`.
pub(super) fn gather_rows<T: Copy>(
    rows: &[T],
    stride: usize,
    row_indices: impl IntoIterator<Item = usize>,
) -> Vec<T> {
    let indices = row_indices.into_iter();
    let mut out = Vec::with_capacity(indices.size_hint().0 * stride);
    for idx in indices {
        let start = idx * stride;
        out.extend_from_slice(&rows[start..start + stride]);
    }
    out
}

#[cfg(test)]
mod tests;
