// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Robust scalar statistics shared across every topic group.
//!
//! Crate-level rather than filed under one group because the callers are not
//! related: the growth and pose-verification kernels ([`crate::geometry`]),
//! the adjacency and census passes ([`crate::analysis`]), and the two patch
//! IRLS loops ([`crate::patch`]) all want the same number.
//!
//! ## Why this is centralized
//!
//! There were six `median` implementations here with **three incompatible NaN
//! policies**, and they agreed only on clean data:
//!
//! - three sorted with [`f64::total_cmp`] (a total order, NaN last),
//! - two with `partial_cmp().unwrap_or(Ordering::Equal)`, which is not a valid
//!   total order when a NaN is present — the sort is then free to produce any
//!   permutation, so the result was unspecified rather than merely surprising,
//! - one with `partial_cmp().unwrap()`, which **panicked** on a NaN input.
//!
//! Their empty-slice behaviour disagreed too: `0.0`, `None`, `NaN`, and one
//! that only `debug_assert!`ed and so indexed out of bounds in release.
//!
//! Three of the six were reachable from the same photometric pipeline, which
//! is what made the divergence worth removing: the same residual population
//! could be summarized three different ways depending on which module got to
//! it, and none of the differences would fail to compile.
//!
//! ## The rule
//!
//! One rule, applied everywhere, chosen to match what the callers already do
//! with the answer:
//!
//! - **Ordering is [`f64::total_cmp`]** — a genuine total order, so the sort is
//!   deterministic whatever the input contains and no comparison can panic.
//!   NaN sorts above every finite value (and `−NaN` below every finite value),
//!   so a NaN median means NaN reached the *middle* of the population, not
//!   merely that one was present. A minority of NaNs therefore leaves the
//!   median finite, which is the robustness a median is chosen for.
//! - **Empty input yields NaN.** Every caller in the crate already tests the
//!   result with `is_finite()` / `is_nan()` and has a defined path for "no
//!   answer" — the surfel-normals IRLS substitutes its sigma floor
//!   (`specs/core/adjacency-surfel-normals.md` documents this: "the floor also
//!   replaces a non-finite median"), the keypoint localizer drops its
//!   relative-ZNCC bar to `−∞`, and the pair vetter rejects the pair. NaN
//!   routes an empty population into those same paths instead of inventing a
//!   number for it.
//! - **Even counts average the two middle values** (numpy convention), which
//!   all six already did.

/// Median of `values`, sorting it in place; `NaN` when empty.
///
/// Even counts average the two middle values. See the [module
/// docs](self) for the NaN rule — briefly: ordering is [`f64::total_cmp`], so
/// NaN sorts above every finite value and a NaN result means NaN reached the
/// middle of the population.
///
/// Prefer this over [`median`] wherever the caller already owns a scratch
/// buffer; the IRLS loops reuse one across iterations.
pub(crate) fn median_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    // `(n - 1) / 2` and `n / 2` are the same index for odd `n`, so this is the
    // even-count average and the odd-count middle in one expression.
    0.5 * (values[(n - 1) / 2] + values[n / 2])
}

/// Median of `values`, copying to sort; `NaN` when empty.
///
/// The borrowing counterpart to [`median_in_place`], for callers holding a
/// shared slice they cannot disturb. Identical semantics otherwise.
pub(crate) fn median(values: &[f64]) -> f64 {
    let mut scratch = values.to_vec();
    median_in_place(&mut scratch)
}

#[cfg(test)]
mod tests;
