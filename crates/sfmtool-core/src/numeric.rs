// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scalar primitives that must compute the same number everywhere: the
//! crate's one median, and the one RNG step behind its deterministic sampling.
//!
//! Crate-level rather than filed under one group because the callers are not
//! related: the growth and pose-verification kernels ([`crate::geometry`]),
//! the adjacency and census passes ([`crate::analysis`]), the two patch IRLS
//! loops ([`crate::patch`]) and the covisibility sampler
//! ([`crate::features`]) all want the same number. The two median entry
//! points are `pub` so that the bindings in `sfmtool-py` reach this
//! implementation rather than spelling their own beside it.
//!
//! ## Why the median is centralized
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
//!   (`specs/core/analysis/adjacency-surfel-normals.md` documents this: "the floor also
//!   replaces a non-finite median"), the keypoint localizer drops its
//!   relative-ZNCC bar to `−∞`, and the pair vetter rejects the pair. NaN
//!   routes an empty population into those same paths instead of inventing a
//!   number for it.
//! - **Even counts average the two middle values** (numpy convention), which
//!   all six already did. This is `numpy.median`'s rule, so a binding that
//!   used to compute its own "numpy median" gets the same answers from
//!   [`median_in_place`] — with the NaN and empty cases defined instead of
//!   panicking.
//!
//! There is one median in this workspace and it is here. `numeric/tests.rs`
//! enforces that mechanically: it scans the crates for `fn …median…`
//! definitions and fails on any that is not in its allowlist of the few that
//! are genuinely a different operation. Add a caller, not a copy.

/// Length at or above which quickselect beats a full sort.
///
/// Both are correct at every length; this is purely where the constant factors
/// cross. Measured on this workload (random `f64`, warm cache), full sort wins
/// below ~16 and quickselect wins from ~32 — 2.1× at 32, 2.6× at 128, 4.3× at
/// 512, 10× at 65536 — so the threshold sits at the top of the ambiguous band.
/// It matters because the callers straddle it: the photometric IRLS loops take
/// a median per patch per iteration over a handful of views (the hot, small
/// case), while the census, adjacency and growth medians scale with the
/// reconstruction (the large, asymptotic case).
const SELECT_MIN_LEN: usize = 32;

/// Median of `values`, reordering it in place; `NaN` when empty.
///
/// Even counts average the two middle values. See the [module docs](self) for
/// the NaN rule — briefly: ordering is [`f64::total_cmp`], so NaN sorts above
/// every finite value and a NaN result means NaN reached the middle of the
/// population.
///
/// Prefer this over [`median`] wherever the caller already owns a scratch
/// buffer; the IRLS loops reuse one across iterations.
///
/// `values` is left permuted — **sorted** on the short path and merely
/// **partitioned around the median** on the long one. No caller relies on the
/// order afterwards; they either overwrite the buffer or drop it. Do not add
/// one that does.
pub fn median_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let n = values.len();
    // The same index for odd `n`, which is how one expression covers both the
    // odd-count middle and the even-count average.
    let (lo_idx, hi_idx) = ((n - 1) / 2, n / 2);

    if n < SELECT_MIN_LEN {
        values.sort_unstable_by(f64::total_cmp);
        return 0.5 * (values[lo_idx] + values[hi_idx]);
    }

    // Quickselect places the `hi_idx`-th order statistic and partitions around
    // it; `total_cmp` is a total order, so that statistic — and therefore the
    // median — is the same value a full sort would have selected.
    let (below, at_hi, _) = values.select_nth_unstable_by(hi_idx, f64::total_cmp);
    let hi = *at_hi;
    if lo_idx == hi_idx {
        return hi;
    }
    // Even count: `below` holds every value at or under `hi`, so its maximum is
    // the lower of the two middles. `max_by(total_cmp)` rather than `f64::max`,
    // which skips NaN and would silently pick the wrong element.
    let lo = below
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("even length ≥ SELECT_MIN_LEN leaves a non-empty low partition");
    0.5 * (lo + hi)
}

/// Median of `values`, copying to sort; `NaN` when empty.
///
/// The borrowing counterpart to [`median_in_place`], for callers holding a
/// shared slice they cannot disturb. Identical semantics otherwise.
pub fn median(values: &[f64]) -> f64 {
    let mut scratch = values.to_vec();
    median_in_place(&mut scratch)
}

/// SplitMix64 step: advance `state` and return the mixed output.
///
/// The deterministic RANSAC samplers seed one of these per kernel, so the exact
/// bit mixing is part of every sampling-dependent result. Crate-level for the
/// same reason the median is: the geometry kernels' minimal samples, the
/// covisibility sampler and the patch cluster-consistency draws are unrelated
/// callers that must nonetheless step the same generator.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests;
