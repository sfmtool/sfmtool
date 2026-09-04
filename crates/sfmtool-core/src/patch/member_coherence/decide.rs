// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The verdict read off a point's pairwise member-agreement matrix.
//!
//! Pure policy over an already-computed [`MemberMatrix`]: the max-support block
//! of the agreement graph, the self-normalized admission bar, and the
//! multi-scale exoneration that separates a structural disagreement from a
//! spectral one. Nothing here renders — the matrix is built in
//! [`super::matrix`].
//!
//! See `specs/core/patch/member-coherence-validation.md` for the design.

use super::{
    scored_mask, MemberCoherenceParams, MemberDecision, MemberMatrix, MemberVerdict,
    EXONERATION_MIN_DEFICIT, SELF_BAR_CEILING, SELF_BAR_MIN_PAIRS, SELF_BAR_MIN_SCATTER,
};

/// The winning max-support block of the agreement graph, as a member mask.
///
/// `zncc` is the `k×k` table of the members in play — [`decide_member_coherence`]
/// passes the *scored* sub-matrix, so an unscored member is never a hypothesis and
/// never a tie-break candidate.
///
/// Every member is a hypothesis; its support is the set of members whose pairwise
/// ZNCC to it reaches `bar` (itself always included, so a member with no partner
/// supports a block of one). The largest support wins.
///
/// Ties are broken **deterministically**: first on the block's own mean coherence
/// (the mean of its finite intra-block links, `-1` when it has none), then on the
/// lowest member index. Nothing in the rule consults iteration or thread order.
pub(super) fn max_support_block(zncc: &[f64], k: usize, bar: f64) -> Vec<bool> {
    // Agreement graph, self-loops forced on.
    let mut adj = vec![false; k * k];
    for i in 0..k {
        for j in 0..k {
            let z = zncc[i * k + j];
            adj[i * k + j] = i == j || (z.is_finite() && z >= bar);
        }
    }
    let support = |i: usize| (0..k).filter(|&j| adj[i * k + j]).count();
    let best_support = (0..k).map(support).max().unwrap_or(0);
    let ties: Vec<usize> = (0..k).filter(|&i| support(i) == best_support).collect();

    let best = if ties.len() > 1 {
        // Mean coherence of the block each tied hypothesis induces.
        let mean_intra = |t: usize| {
            let block: Vec<usize> = (0..k).filter(|&j| adj[t * k + j]).collect();
            let mut sum = 0.0;
            let mut count = 0usize;
            for (bi, &a) in block.iter().enumerate() {
                for &b in block.iter().skip(bi + 1) {
                    for (x, y) in [(a, b), (b, a)] {
                        let z = zncc[x * k + y];
                        if z.is_finite() {
                            sum += z;
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 {
                -1.0
            } else {
                sum / count as f64
            }
        };
        let mut best = ties[0];
        let mut best_score = mean_intra(ties[0]);
        for &t in &ties[1..] {
            let s = mean_intra(t);
            if s > best_score {
                best_score = s;
                best = t;
            }
        }
        best
    } else {
        ties.first().copied().unwrap_or(0)
    };

    (0..k).map(|j| adj[best * k + j]).collect()
}

/// The `q`-quantile of an already-sorted non-empty slice, by linear
/// interpolation between the bracketing order statistics — a pure function of
/// the multiset, with no dependence on how it arrived.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    }
}

/// The **centre and scatter of one block's own agreement**: the statistics the
/// self-normalized admission bar is measured in.
///
/// `zncc` is a `k×k` table and `block` a mask over it (in practice the *scored*
/// sub-matrix and its pass-1 max-support block). The sample is every finite
/// pairwise link **inside** the block, each counted once. Returns `None` — the
/// relative term stays inactive — when that sample holds fewer than
/// [`SELF_BAR_MIN_PAIRS`] links.
///
/// - **Centre** `c` is the median of those links.
/// - **Scatter** `σ` is the **upper** semi-interquartile distance, made
///   normal-consistent the same way a MAD is: `1.4826 · (Q₇₅ − median)`, floored
///   at [`SELF_BAR_MIN_SCATTER`].
///
/// The one-sidedness is the point. The block admitted at the absolute bar is the
/// very thing under suspicion: on a track with an occluding member the block
/// still contains it, and its links sit in the **lower** tail. A two-sided MAD
/// reads that tail as spread and inflates σ — letting the contamination loosen
/// the bar that is supposed to exclude it. The half above the median is the part
/// of the sample the contamination cannot reach (it is a minority, or the block
/// would not be the core), so it measures the core's own tightness. For a
/// symmetric sample the two coincide, which is what the `1.4826` is for.
///
/// It is read as an order statistic (`Q₇₅ − Q₅₀`) rather than as the median of
/// the members above the centre, because those two differ exactly when the
/// sample has a **mass at the median** — a two-population matrix whose median
/// lands on the lower mode. Counting the ties would report that matrix as
/// tightly coherent; the quartile distance reports the spread that is really
/// there and the relative term collapses, which is the intended behaviour for a
/// track with no single core.
pub fn core_coherence(zncc: &[f64], k: usize, block: &[bool]) -> Option<(f64, f64)> {
    let mut v = Vec::new();
    for a in 0..k {
        if !block[a] {
            continue;
        }
        for b in (a + 1)..k {
            if !block[b] {
                continue;
            }
            let z = zncc[a * k + b];
            if z.is_finite() {
                v.push(z);
            }
        }
    }
    if v.len() < SELF_BAR_MIN_PAIRS {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let center = quantile_sorted(&v, 0.5);
    let scatter = (1.4826 * (quantile_sorted(&v, 0.75) - center)).max(SELF_BAR_MIN_SCATTER);
    Some((center, scatter))
}

/// The absolute thresholds tightened by one block's own coherence: the pair
/// `(effective_bar, effective_margin_gate)` plus the `(centre, scatter)` they
/// were derived from.
///
/// One tighten pass, never iterated to a fixed point — see
/// [`decide_member_coherence`].
fn self_normalized_thresholds(
    sub: &[f64],
    s: usize,
    block: &[bool],
    params: &MemberCoherenceParams,
) -> (f64, f64, Option<(f64, f64)>) {
    let k_self = params.self_bar_k;
    if k_self.is_nan() || k_self <= 0.0 {
        return (params.bar, params.margin_gate, None);
    }
    match core_coherence(sub, s, block) {
        None => (params.bar, params.margin_gate, None),
        Some((c, sigma)) => {
            let relative = (c - k_self * sigma).min(SELF_BAR_CEILING);
            (
                params.bar.max(relative),
                params.margin_gate.min(sigma),
                Some((c, sigma)),
            )
        }
    }
}

/// Read a verdict off a pairwise member matrix.
///
/// **The whole rule runs over the [scored](scored_mask) members only** — the
/// block sweep, both margin sides, and the majority denominator. An unscored
/// member carries no pairwise evidence, so it can neither be evicted by a cut it
/// took no part in nor dilute a majority among members that did: it passes
/// through `kept` and stays out of `block`. Fewer than two scored members means
/// no evidence at all: `KeepAll`, empty block, `support = 0`, undefined margin.
///
/// The pass, in order:
///
/// 1. Sweep the max-support block (`max_support_block`) at the absolute `bar`.
/// 2. Measure that block's own [`core_coherence`] — centre `c`, scatter `σ` —
///    and re-derive both thresholds in the track's own units:
///    `effective_bar = max(bar, min(c − self_bar_k · σ, `[`SELF_BAR_CEILING`]`))`
///    and `effective_margin_gate = min(margin_gate, σ)`. Both relax back to the
///    absolute pair exactly when `σ` is large. The tightening runs **once**, and
///    is inactive at `self_bar_k = 0` or below [`SELF_BAR_MIN_PAIRS`]
///    intra-block pairs.
/// 3. Re-sweep the block at `effective_bar`, but only when it actually rose.
/// 4. Gate the cut on its **separation margin** — the weakest link inside the
///    block minus the strongest link leaving it — against
///    `effective_margin_gate`. A margin at or below the gate, or an undefined
///    one, is `KeepAll`.
/// 5. **Exonerate**: of the members the *relative* term alone would evict, spare
///    those whose disagreement does not survive coarsening. The retained deficit
///    is [`core_deficit`] at the **first** coarse table of
///    [`MemberMatrix::zncc_coarse`] — one halving — over the deficit at full
///    scale, compared against
///    [`MemberCoherenceParams::exoneration_ratio`]. A spared member stays in
///    `kept`, out of `block`, and is marked in [`MemberDecision::exonerated`],
///    while `margin`, `min_intra`, `max_cross`, `support` and `block` keep
///    describing the cut that was proposed. Inert at `exoneration_ratio = 0`,
///    whenever the relative term is inactive, and on a matrix with no coarse
///    scale.
/// 6. A block holding a strict majority of the scored members splits the track;
///    a block that does not retires the point. Sparing every rejected member
///    falls back to `KeepAll`; sparing enough to restore a majority turns a
///    `Retire` into a `Split`.
///
/// See `specs/core/patch/member-coherence-validation.md` for why each step is
/// shaped this way: what the absolute bar is calibrated against and why a
/// per-track bar is needed on top of it, why the admission/statistics
/// circularity is cut rather than iterated, why the margin gate refuses to cut a
/// drift continuum, why only the relative term's evictions are exonerable, and
/// why one halving is the comparison scale.
pub fn decide_member_coherence(
    matrix: &MemberMatrix,
    params: &MemberCoherenceParams,
) -> MemberDecision {
    let k = matrix.len();
    if k == 0 {
        return MemberDecision::default();
    }
    let zncc = &matrix.zncc;
    let scored = scored_mask(zncc, k);
    let idx: Vec<usize> = (0..k).filter(|&i| scored[i]).collect();
    let s = idx.len();
    if s < 2 {
        return MemberDecision {
            verdict: MemberVerdict::KeepAll,
            kept: vec![true; k],
            block: vec![false; k],
            support: 0,
            margin: f64::NAN,
            min_intra: f64::NAN,
            max_cross: f64::NAN,
            effective_bar: f64::NAN,
            effective_margin_gate: f64::NAN,
            core_center: f64::NAN,
            core_scatter: f64::NAN,
            relative_flagged: vec![false; k],
            exonerated: vec![false; k],
            retained_deficit: vec![f64::NAN; k],
            sharpness_deficit: vec![f64::NAN; k],
        };
    }

    // The scored sub-matrix, in member order. Every quantity below is computed on
    // it, so unscored members are structurally outside the rule rather than
    // half-counted by it. The coarse tables are sliced to the same members, so a
    // scale index means the same thing at every scale.
    let mut sub = vec![f64::NAN; s * s];
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            sub[a * s + b] = zncc[ia * k + ib];
        }
    }
    let slice_scale = |table: &Vec<f64>| {
        let mut out = vec![f64::NAN; s * s];
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                out[a * s + b] = table[ia * k + ib];
            }
        }
        out
    };
    // Two coarse scales, two different questions — see the module docs on
    // `zncc_coarse`. Exoneration asks whether the disagreement SURVIVES the loss
    // of one octave, so it reads the finest coarse table; sharpness measures how
    // much of the deficit is detail, so it reads the coarsest for the widest span.
    let exon_scale: Option<Vec<f64>> = matrix.zncc_coarse.first().map(&slice_scale);
    let sharp_scale: Option<Vec<f64>> = matrix.zncc_coarse.last().map(&slice_scale);
    // Pass 1 at the absolute bar, then the one tighten pass off that block's own
    // coherence. `effective_bar == params.bar` short-circuits the re-sweep, which
    // is what makes `self_bar_k = 0` bit-for-bit the absolute rule.
    let pass1 = max_support_block(&sub, s, params.bar);
    let (effective_bar, effective_margin_gate, core) =
        self_normalized_thresholds(&sub, s, &pass1, params);
    let relative_engaged = effective_bar > params.bar;
    let sub_block = if relative_engaged {
        max_support_block(&sub, s, effective_bar)
    } else {
        pass1.clone()
    };
    let (core_center, core_scatter) = core.unwrap_or((f64::NAN, f64::NAN));
    let support = sub_block.iter().filter(|&&b| b).count();

    // Margin components. Undefined (NaN) for a block of one, a block spanning
    // every scored member, or a side with no finite link.
    let mut min_intra = f64::INFINITY;
    let mut max_cross = f64::NEG_INFINITY;
    for a in 0..s {
        if !sub_block[a] {
            continue;
        }
        for b in 0..s {
            if a == b {
                continue;
            }
            let z = sub[a * s + b];
            if !z.is_finite() {
                continue;
            }
            if sub_block[b] {
                min_intra = min_intra.min(z);
            } else {
                max_cross = max_cross.max(z);
            }
        }
    }
    let min_intra = if min_intra.is_finite() {
        min_intra
    } else {
        f64::NAN
    };
    let max_cross = if max_cross.is_finite() {
        max_cross
    } else {
        f64::NAN
    };
    let whole = support == s;
    let margin = if whole || support < 2 {
        f64::NAN
    } else {
        min_intra - max_cross
    };

    // Scatter the block back over the full member list; unscored members are not
    // in it.
    let mut block = vec![false; k];
    for (a, &ia) in idx.iter().enumerate() {
        block[ia] = sub_block[a];
    }

    // Per-member sharpness, for every scored member and independently of any
    // verdict: the part of the member's deficit that only exists at fine scale.
    // Reported so a consumer can read the observations the point ships, which is
    // why it is not confined to the members under suspicion.
    let mut sharpness_deficit = vec![f64::NAN; k];
    if let Some(coarse) = sharp_scale.as_ref() {
        for (a, &ia) in idx.iter().enumerate() {
            let df = core_deficit(&sub, s, &sub_block, a);
            let dc = core_deficit(coarse, s, &sub_block, a);
            if df.is_finite() && dc.is_finite() {
                sharpness_deficit[ia] = df - dc;
            }
        }
    }

    let base = |verdict, kept: Vec<bool>, block: Vec<bool>, relative_flagged, exonerated, rd| {
        MemberDecision {
            verdict,
            kept,
            block,
            support: support as u32,
            margin,
            min_intra,
            max_cross,
            effective_bar,
            effective_margin_gate,
            core_center,
            core_scatter,
            relative_flagged,
            exonerated,
            retained_deficit: rd,
            sharpness_deficit: sharpness_deficit.clone(),
        }
    };
    // Nothing was being evicted, so nothing was flagged and nothing was spared.
    let keep_all = || {
        base(
            MemberVerdict::KeepAll,
            vec![true; k],
            block.clone(),
            vec![false; k],
            vec![false; k],
            vec![f64::NAN; k],
        )
    };

    if whole {
        return keep_all();
    }
    // Refuse to cut a continuum: no gap between the block and its outside, in the
    // track's own units.
    if margin.is_nan() || margin <= effective_margin_gate {
        return keep_all();
    }

    // A cut is on the table. Which of its evictions does the RELATIVE term own?
    // Only those: a member the absolute bar already rejected is not exonerable,
    // whatever its deficit does across scales.
    let mut relative_flagged = vec![false; k];
    let mut exonerated = vec![false; k];
    let mut retained_deficit = vec![f64::NAN; k];
    let mut spared = 0usize;
    if relative_engaged {
        for (a, &ia) in idx.iter().enumerate() {
            if sub_block[a] || !pass1[a] {
                continue;
            }
            relative_flagged[ia] = true;
            let Some(coarse) = exon_scale.as_ref() else {
                continue;
            };
            let df = core_deficit(&sub, s, &sub_block, a);
            if !(df.is_finite() && df > EXONERATION_MIN_DEFICIT) {
                continue;
            }
            let dc = core_deficit(coarse, s, &sub_block, a);
            if !dc.is_finite() {
                continue;
            }
            let ratio = dc / df;
            retained_deficit[ia] = ratio;
            if params.exoneration_ratio > 0.0 && ratio <= params.exoneration_ratio {
                exonerated[ia] = true;
                spared += 1;
            }
        }
    }

    // Everything the cut would have taken was spared: there is no cut left.
    let kept_scored = support + spared;
    if kept_scored == s {
        return base(
            MemberVerdict::KeepAll,
            vec![true; k],
            block,
            relative_flagged,
            exonerated,
            retained_deficit,
        );
    }
    // No strict majority among the members that carry evidence: the two sides are
    // equally supported, so neither can be called the point's surface. The spared
    // members count here — they ship, so they are part of the side that would.
    if 2 * kept_scored <= s {
        return base(
            MemberVerdict::Retire,
            vec![false; k],
            block,
            relative_flagged,
            exonerated,
            retained_deficit,
        );
    }
    // The cut evicts the scored members outside the block that were not spared.
    let kept = (0..k)
        .map(|i| block[i] || exonerated[i] || !scored[i])
        .collect();
    base(
        MemberVerdict::Split,
        kept,
        block,
        relative_flagged,
        exonerated,
        retained_deficit,
    )
}

/// One member's **agreement deficit** against a block, on one scale's `s×s`
/// table: how much worse the member agrees with the block than the block agrees
/// with itself.
///
/// `member` is excluded from the core on both sides, so the quantity means the
/// same thing for a member inside the block and one outside it. Both means are
/// over finite links only. `NaN` when the core holds fewer than two members, or
/// when either side has no finite link to average.
pub fn core_deficit(zncc: &[f64], s: usize, block: &[bool], member: usize) -> f64 {
    let core: Vec<usize> = (0..s).filter(|&i| block[i] && i != member).collect();
    if core.len() < 2 {
        return f64::NAN;
    }
    let (mut intra_sum, mut intra_n) = (0.0, 0usize);
    for (a, &ia) in core.iter().enumerate() {
        for &ib in core.iter().skip(a + 1) {
            let z = zncc[ia * s + ib];
            if z.is_finite() {
                intra_sum += z;
                intra_n += 1;
            }
        }
    }
    let (mut cross_sum, mut cross_n) = (0.0, 0usize);
    for &ic in &core {
        let z = zncc[member * s + ic];
        if z.is_finite() {
            cross_sum += z;
            cross_n += 1;
        }
    }
    if intra_n == 0 || cross_n == 0 {
        return f64::NAN;
    }
    intra_sum / intra_n as f64 - cross_sum / cross_n as f64
}
