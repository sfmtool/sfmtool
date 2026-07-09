// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-view windowed-ZNCC translation search and the incremental
//! leave-one-out robust consensus it registers against, split out of the
//! congealing orchestration ([`super`]).
//!
//! [`search_shift`] scores the whole `±margin` shift grid by accumulation
//! ([`SearchStrategy::Exhaustive`](super::SearchStrategy::Exhaustive));
//! [`search_shift_plus_descent`] walks a steepest-descent path scoring one cell
//! at a time ([`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)).
//! Both correlate a view's per-view render-once cache ([`ContextTile`]) against a
//! template built by [`loo_consensus_template`] from the shared per-round Gram
//! matrix ([`build_loo_gram`]). The data-touching correlation kernels live in
//! [`super::kernels`].

use crate::patch::normal_refine::{
    tukey_reweight_from_residuals, weighted_unit_template_skip_into, Support, FLAT_NORM_SQ_EPS,
};

use super::kernels::{
    accumulate_count, compute_channel_grids, count_invalid_at_cell, score_cell_one_channel,
};
use super::{parabolic, prof, ContextTile};

/// Reused per-call scratch for [`search_shift`], created once per
/// [`localize_patch_keypoints`](super::localize_patch_keypoints) and shared
/// across every view and round (the search then allocates nothing after
/// warm-up), mirroring
/// [`ConsensusScratch`](crate::patch::normal_refine::ConsensusScratch).
///
/// The cache itself (centered planar `f32` planes + invalidity plane) lives on
/// the [`ContextTile`] now, so the per-call scratch holds only the per-channel
/// kernel, the running correlation maps, and the combined grid.
// Fields are `pub(super)` so the sibling test module can build a scratch with
// `SearchScratch { tmpl, ..Default::default() }` (functional-update needs every
// field visible); production only writes `tmpl`.
#[derive(Default)]
pub(super) struct SearchScratch {
    /// The leave-one-out template the candidates score against (`kept_ch · n`),
    /// laid out `[c · n + k]`; the caller writes it each round.
    pub(super) tmpl: Vec<f32>,
    /// Per-support-pixel kernel `√w · tmpl` for the channel being accumulated
    /// (`f32` — the AVX2 kernel broadcasts these as `f32` lanes).
    pub(super) kern: Vec<f32>,
    /// Per-support-pixel window weight `w` as `f32` (one-time conversion from the
    /// `f64` `support.weights`; broadcast each k-step).
    pub(super) w_f32: Vec<f32>,
    /// Per-channel correlation maps over the shift grid (`(2·margin+1)²`): the
    /// numerator `Σ kern·I_c` and the centered window moments `Σ w·I_c`,
    /// `Σ w·I_c²` (`I_c = I − cache_mean`, so the centering algebra absorbs the
    /// mean: the windowed ZNCC formula on centered S1/S2 is identical to the
    /// raw-value form — see the [`ContextTile`] doc and `search_shift_scalar`).
    pub(super) g_n: Vec<f32>,
    pub(super) g_s1: Vec<f32>,
    pub(super) g_s2: Vec<f32>,
    /// Per-shift count of out-of-frame support pixels (a shift is scorable iff 0).
    /// `f32` — values are small integers (≤ `n`).
    pub(super) ginv: Vec<f32>,
    /// Combined ZNCC grid over the `±margin` window (`(2·margin+1)²`).
    pub(super) grid: Vec<f64>,
    /// [`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)-only:
    /// flat per-(kept channel, support pixel) kernel buffer, `[c · n + k]`, built
    /// once per `search_shift_plus_descent` call and reused across every cell
    /// scored. The exhaustive path's `kern` rebuilds per channel inside its loop;
    /// the descent visits ~10 cells per call, so amortising the kern build over
    /// all of them takes the per-cell rebuild off the hot path. Reused here so the
    /// descent allocates nothing per cell in the steady state.
    pub(super) pd_kerns: Vec<f32>,
    /// PlusDescent-only: per-kept-channel `tsum_c = Σ kern[c · n + k]`, parallel
    /// to the channel dimension of [`Self::pd_kerns`].
    pub(super) pd_tsums: Vec<f64>,
    /// PlusDescent-only: per-kept-channel `(n, s1, s2)` scratch, overwritten by
    /// every cell the descent scores. The combine pass reads it back to fold into
    /// the cell's ZNCC. Sized once per `search_shift_plus_descent` call
    /// (`resize(channels, ..)`); the per-cell scoring writes by index, so no Vec
    /// bookkeeping happens inside the timed `SEARCH_ACC` block.
    pub(super) pd_per_channel: Vec<(f32, f32, f32)>,
    /// PlusDescent-only: kept-channel index → tile-channel index lookup, parallel
    /// to the channel dimension of [`Self::pd_kerns`]. Walked once at the top of
    /// every `search_shift_plus_descent` call so per-cell scoring can index this
    /// rather than re-walking `keep_mask`.
    pub(super) pd_kept_channels: Vec<usize>,
    /// PlusDescent-only: dense visited cache sized `(2·margin+1)²`, indexed
    /// `((dy + margin) · span + (dx + margin))`. Sentinel-encoded to avoid
    /// carrying a separate "visited" bitmap: `f64::NAN` = unvisited (skip the
    /// score, evaluate it), `f64::NEG_INFINITY` = visited and unscorable
    /// (oob / oof / disk — don't re-evaluate, don't admit as a neighbor), any
    /// other finite value = visited and that cell's combined ZNCC. Reset to
    /// all-NaN at the start of every `search_shift_plus_descent` call. Replaces a
    /// per-call `HashMap<(i64,i64), Option<f64>>`.
    pub(super) pd_visited: Vec<f64>,
}

/// Reused scratch for the **incremental leave-one-out consensus**: the
/// per-round Gram matrix over the live views' z-normalized cores plus the
/// per-holdout Gram-space IRLS buffers. Created once per
/// [`localize_patch_keypoints`](super::localize_patch_keypoints) call, mirroring
/// [`SearchScratch`].
#[derive(Default)]
pub(super) struct LooScratch {
    /// Live-view Gram matrix `G[a·nv + b] = ⟨x_a, x_b⟩` (f64, stacked over the
    /// kept channels), rebuilt once per round.
    gram: Vec<f64>,
    /// Per-holdout IRLS weights over the **full** live index range; the
    /// held-out view's slot is pinned at `0` so `G·w` sums holdout members only.
    w: Vec<f64>,
    /// `y = G·w` per live view (only holdout members are read).
    y: Vec<f64>,
    /// Per-holdout-member residuals `‖x_u − x̄‖`, compacted (skip the holdout).
    resid: Vec<f64>,
    /// Compacted holdout weights, parallel to [`resid`](Self::resid) — the
    /// in/out slice for the shared Tukey reweight.
    wh: Vec<f64>,
    /// Median/MAD sort scratch for the reweight.
    sorted: Vec<f64>,
    /// Raw (pre-normalization) Tukey weights scratch for the reweight.
    wt: Vec<f64>,
}

/// `Σ_k a[k]·b[k]` accumulated in `f64` over `f32` inputs, with 8 independent
/// accumulators so the compiler can vectorize the (otherwise
/// associativity-serialized) f64 adds. Feeds the per-round Gram matrix.
fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    const LANES: usize = 8;
    let n = a.len().min(b.len());
    let mut acc = [0f64; LANES];
    let body = n / LANES * LANES;
    let mut i = 0;
    while i < body {
        for (l, s) in acc.iter_mut().enumerate() {
            *s += a[i + l] as f64 * b[i + l] as f64;
        }
        i += LANES;
    }
    let mut s: f64 = acc.iter().sum();
    for k in body..n {
        s += a[k] as f64 * b[k] as f64;
    }
    s
}

/// Build the symmetric live-view Gram matrix `G[a][b] = ⟨x_a, x_b⟩` into
/// `loo.gram` (`nv × nv`, row-major), where `xs` holds `nv` contiguous rows of
/// `cn` f32s. Computed **once per round** and shared by every holdout's IRLS.
pub(super) fn build_loo_gram(xs: &[f32], nv: usize, cn: usize, loo: &mut LooScratch) {
    loo.gram.resize(nv * nv, 0.0);
    for a in 0..nv {
        let xa = &xs[a * cn..][..cn];
        loo.gram[a * nv + a] = dot_f64(xa, xa);
        for b in (a + 1)..nv {
            let d = dot_f64(xa, &xs[b * cn..][..cn]);
            loo.gram[a * nv + b] = d;
            loo.gram[b * nv + a] = d;
        }
    }
}

/// Build view `v`'s **leave-one-out robust consensus template** into `out`
/// from the live stack `xs` (`nv` rows × `kept_ch · n`), using the per-round
/// Gram matrix in `loo.gram` — the incremental replacement for the
/// copy-the-holdout-stack + `irls_view_weights` + `weighted_unit_template_into`
/// rebuild that ran per (view, round).
///
/// The IRLS recursion touches the pixel data only through inner products, so
/// the whole per-holdout iteration runs in Gram space: with weights `w` (the
/// holdout's slot pinned at 0), `x̄ = Σ w_u x_u` gives residuals
/// `r_u² = G[u][u] − 2·(G·w)_u + wᵀ·G·w`, which feed the **exact** shared
/// Tukey/MAD reweight ([`tukey_reweight_from_residuals`]). Only the final unit
/// template is materialized in pixel space
/// ([`weighted_unit_template_skip_into`], no holdout-stack copy).
///
/// Real-arithmetic semantics are identical to the compacted-stack path
/// (uniform `1/(nv−1)` init, `robust_iters` reweights, degenerate-reweight
/// early-out keeping the previous weights); float results differ only at
/// accumulation-order level (f64 Gram algebra vs the f32 SAXPY/residual path),
/// which the `incremental_loo_template_matches_reference` test bounds.
#[allow(clippy::too_many_arguments)]
pub(super) fn loo_consensus_template(
    xs: &[f32],
    nv: usize,
    v: usize,
    kept_ch: usize,
    n: usize,
    robust_iters: u32,
    loo: &mut LooScratch,
    out: &mut Vec<f32>,
) {
    debug_assert!(nv >= 2 && v < nv);
    debug_assert_eq!(loo.gram.len(), nv * nv);
    loo.w.clear();
    loo.w.resize(nv, 1.0 / (nv - 1) as f64);
    loo.w[v] = 0.0;
    for _ in 0..robust_iters {
        // y = G·w and ‖x̄‖² = wᵀ·y over the holdout members (w[v] = 0 keeps the
        // held-out view out of both).
        loo.y.clear();
        loo.y.resize(nv, 0.0);
        let mut xbar_sq = 0.0;
        for u in 0..nv {
            if u == v {
                continue;
            }
            let row = &loo.gram[u * nv..][..nv];
            let mut acc = 0.0;
            for (uu, &wu) in loo.w.iter().enumerate() {
                acc += wu * row[uu];
            }
            loo.y[u] = acc;
            xbar_sq += loo.w[u] * acc;
        }
        // Residuals ‖x_u − x̄‖ per holdout member (compacted). The algebraic r²
        // can dip epsilon-negative for a view equal to the consensus; clamp.
        loo.resid.clear();
        for u in 0..nv {
            if u == v {
                continue;
            }
            let r2 = loo.gram[u * nv + u] - 2.0 * loo.y[u] + xbar_sq;
            loo.resid.push(r2.max(0.0).sqrt());
        }
        // Shared Tukey/MAD reweight on the compacted weights, scattered back.
        loo.wh.clear();
        loo.wh.extend((0..nv).filter(|&u| u != v).map(|u| loo.w[u]));
        let degenerate = {
            let LooScratch {
                resid,
                sorted,
                wt,
                wh,
                ..
            } = loo;
            tukey_reweight_from_residuals(resid, None, sorted, wt, wh)
        };
        if degenerate {
            break; // keep the previous weights, as irls_view_weights does
        }
        let mut j = 0;
        for u in 0..nv {
            if u == v {
                continue;
            }
            loo.w[u] = loo.wh[j];
            j += 1;
        }
    }
    weighted_unit_template_skip_into(xs, &loo.w, v, nv, kept_ch, n, out);
}

/// The result of a [`search_shift`] — the residual shift of one view relative to
/// its current integer base offset, in `R_s`-grid steps.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShiftResult {
    /// Sub-pixel-refined shift in the grid's x (column) axis: integer argmax plus
    /// the separable parabolic residual.
    pub(super) dx: f64,
    /// Sub-pixel-refined shift in the grid's y (row) axis.
    pub(super) dy: f64,
    /// The integer argmax shift in x — the part that moves the integer read
    /// accumulator `iacc` (every cache read stays at an integer index).
    pub(super) ix: i64,
    /// The integer argmax shift in y.
    pub(super) iy: i64,
    /// ZNCC at the integer peak.
    pub(super) peak: f64,
}

/// Integer windowed-ZNCC translation search of view `v`'s cache against
/// `sc.tmpl`, refined to sub-pixel by a separable parabolic fit. Returns a
/// [`ShiftResult`] — the integer argmax shift `(ix, iy)`, its sub-pixel-refined
/// counterpart `(dx, dy)`, and the ZNCC `peak` at the integer peak — or `None` if
/// no in-frame window position could be scored. The search centres on
/// the view's current integer offset `iacc`: the `(dy, dx) = (0, 0)` shift scores
/// the `R×R` core window whose top-left support pixel sits at cache index
/// `(base_y, base_x) = ((cache_res − R) / 2 + iacc_y, … + iacc_x)`, and the search
/// slides the window over `±margin` around it. Both `base ± margin` are guaranteed
/// in-bounds by the cache sizing (`cache_res = R + 4·margin`, `|iacc| ≤ search`).
///
/// Rather than re-extract + z-normalize + dot each of the `(2·margin+1)²`
/// candidates (a strided gather and a horizontal reduction per candidate), the
/// whole grid is scored by accumulation. Because the template carries a fixed
/// weighted mean, the windowed ZNCC of every shift factors into three
/// correlation maps per channel — `Σ kern·I`, `Σ w·I`, `Σ w·I²` — whose inner
/// loop is a contiguous fused SAXPY across the shift row (the vectorizable core).
/// `zncc = (Σkern·I − mean·Σ√w·tmpl) / √(Σw·I² − mean·Σw·I)`, averaged over
/// channels — algebraically identical to the per-candidate z-normalize-then-dot
/// path (see `search_shift_ref`).
#[allow(clippy::too_many_arguments)]
pub(super) fn search_shift(
    tile: &ContextTile,
    sc: &mut SearchScratch,
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
    let n = support.pixels.len();
    let istride = tile.istride;
    // The search grid's origin (`gy = gx = 0`) reads the window at `base − margin`.
    let win_oy = base_y - margin as usize;
    let win_ox = base_x - margin as usize;
    let span = (2 * margin + 1) as usize;
    let gsz = span * span;

    debug_assert_eq!(keep_mask.iter().filter(|&&k| k).count(), channels);

    // Per-support `w` as f32 (one-time conversion the AVX2 kernel can broadcast).
    sc.w_f32.clear();
    sc.w_f32.extend(support.weights.iter().map(|&w| w as f32));

    // Validity: count out-of-frame support pixels per shift (channel-independent);
    // a shift with any is unscorable, matching `extract_core`'s all-valid gate.
    sc.ginv.clear();
    sc.ginv.resize(gsz, 0.0);
    accumulate_count(
        &tile.invalid_plane,
        support,
        resolution,
        istride,
        span,
        win_oy,
        win_ox,
        &mut sc.ginv,
    );

    // Per kept channel: accumulate the three correlation maps over the centered
    // plane, then fold the channel's ZNCC into the combined grid. With centered
    // values, `S2 − S1²/W` is the same algebra as raw and the mean offset in the
    // numerator cancels (`Σkern = tsum` exactly by construction), so the combine
    // step is identical to the raw-value formula — see the [`ContextTile`] doc.
    // The combined `grid` accumulates across channels and must start at zero;
    // `clear()` + `resize()` so a reused scratch is reliably zeroed up front,
    // not only on grow. `g_n / g_s1 / g_s2` are sized here (overwritten per
    // channel by `compute_channel_grids`, no pre-zero needed).
    sc.grid.clear();
    sc.grid.resize(gsz, 0.0);
    sc.g_n.resize(gsz, 0.0);
    sc.g_s1.resize(gsz, 0.0);
    sc.g_s2.resize(gsz, 0.0);
    let inv_total_weight = 1.0 / support.total_weight;
    let mut kc_out = 0usize;
    for (c, &keep) in keep_mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let tmpl_c = &sc.tmpl[kc_out * n..][..n];
        sc.kern.clear();
        let mut tsum = 0.0f64;
        for (&sw, &t) in support.sqrt_weights.iter().zip(tmpl_c) {
            let kk = sw * t;
            sc.kern.push(kk);
            tsum += kk as f64;
        }
        // `compute_channel_grids` overwrites; no pre-zero needed.
        prof::SEARCH_ACC.time(|| {
            compute_channel_grids(
                &tile.planes[c],
                support,
                &sc.kern,
                &sc.w_f32,
                resolution,
                istride,
                span,
                win_oy,
                win_ox,
                &mut sc.g_n,
                &mut sc.g_s1,
                &mut sc.g_s2,
            );
        });
        prof::SEARCH_COMBINE.time(|| {
            for s in 0..gsz {
                let s1 = sc.g_s1[s] as f64;
                let s2 = sc.g_s2[s] as f64;
                let nval = sc.g_n[s] as f64;
                let mean = s1 * inv_total_weight;
                let norm_sq = s2 - s1 * mean;
                // A channel flat in this window contributes 0 (matches `znorm_core`).
                if norm_sq >= FLAT_NORM_SQ_EPS {
                    sc.grid[s] += (nval - mean * tsum) / norm_sq.sqrt();
                }
            }
        });
        kc_out += 1;
    }

    // Average over channels, find the integer argmax, and refine sub-pixel by a
    // separable parabolic fit. Out-of-frame shifts score −∞ (never chosen).
    prof::SEARCH_ARGMAX.time(|| {
        let chf = channels as f64;
        let at = |dy: i64, dx: i64| -> usize {
            ((dy + margin) as usize) * span + (dx + margin) as usize
        };
        let mut best = (f64::NEG_INFINITY, 0i64, 0i64);
        for dy in -margin..=margin {
            for dx in -margin..=margin {
                let s = at(dy, dx);
                let z = if sc.ginv[s] > 0.5 {
                    f64::NEG_INFINITY
                } else {
                    sc.grid[s] / chf
                };
                sc.grid[s] = z;
                if z > best.0 {
                    best = (z, dy, dx);
                }
            }
        }
        if !best.0.is_finite() {
            return None;
        }
        let (peak, py, px) = best;
        let grid = &sc.grid;
        let nb = |dy: i64, dx: i64| -> Option<f64> {
            if dy.abs() <= margin && dx.abs() <= margin {
                let g = grid[at(dy, dx)];
                g.is_finite().then_some(g)
            } else {
                None
            }
        };
        let sy = match (nb(py - 1, px), nb(py + 1, px)) {
            (Some(l), Some(r)) => parabolic(peak, l, r),
            _ => 0.0,
        };
        let sx = match (nb(py, px - 1), nb(py, px + 1)) {
            (Some(l), Some(r)) => parabolic(peak, l, r),
            _ => 0.0,
        };
        Some(ShiftResult {
            dx: px as f64 + sx,
            dy: py as f64 + sy,
            ix: px,
            iy: py,
            peak,
        })
    })
}

/// [`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)
/// counterpart to [`search_shift`]: starts at `(dy, dx) = (0, 0)` (the view's
/// current integer base offset), evaluates the 4 axis neighbors per step, moves
/// to the best improver, and stops when no neighbor beats the current cell. Each
/// cell is scored at most once via [`score_cell_one_channel`]; the visited cache
/// stores the combined ZNCC per cell. The final separable parabolic sub-pixel fit
/// reuses the 4 cardinal neighbors already in the cache (each was evaluated to
/// discover the STOP condition).
///
/// Same `ShiftResult` contract as `search_shift` — the integer argmax `(ix, iy)`
/// drives the read accumulator, `(dx, dy)` carry the sub-pixel residual, and
/// `peak` is the combined ZNCC at the integer cell. Bounded to `|dy|, |dx| ≤
/// margin`; neighbors past the bound or with any out-of-frame support pixel
/// score `None` (skipped, never chosen).
///
/// Cells visited per call: `5 + 3 · walk_steps` (1 seed + 4 neighbors per step,
/// with 1 cache hit per move). On dino the average is ~6 cells per call — vs
/// the 169 cells of the default ±6 grid `search_shift` processes — at ~32 µs
/// per call (the per-cell `vgatherdps` kernel) vs ~145 µs (the SAXPY). The
/// crossover with `search_shift`'s whole-grid SIMD is around 50–80 cells
/// visited, so the descent loses on pathological long walks; `Exhaustive`
/// remains the right pick for the global-argmax fallback. See
/// `specs/core/keypoint-localization-search-cache.md` for the strategy
/// trade-off discussion.
///
/// **Profile attribution** (`SFMTOOL_PROFILE=1`): the descent reports per-cell
/// `SEARCH_ACC` (invalidity-count + per-channel scoring), per-cell
/// `SEARCH_COMBINE` (mean / ZNCC fold + cross-channel sum), and per-call
/// `SEARCH_ARGMAX` (the final parabolic). The `N_CELLS` event counter bumps
/// once per cell scored (visited-cache hits and oob/oof/disk skips do not
/// count), so `N_CELLS / N_SEARCH` is the average cells-per-call directly out
/// of the profile output. `N_CELLS` is `0` under `Exhaustive` — its whole-
/// grid SAXPY has no per-cell event.
#[allow(clippy::too_many_arguments)]
pub(super) fn search_shift_plus_descent(
    tile: &ContextTile,
    sc: &mut SearchScratch,
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
    let n = support.pixels.len();
    let istride = tile.istride;
    debug_assert_eq!(keep_mask.iter().filter(|&&k| k).count(), channels);

    // Per-support `w` as f32 — mirrors `search_shift`'s one-time conversion.
    sc.w_f32.clear();
    sc.w_f32.extend(support.weights.iter().map(|&w| w as f32));

    // Build the flat per-(kept channel, support pixel) kern + per-channel
    // tsum into the reused `SearchScratch` slots. Layout: `pd_kerns[c · n + k]
    // = √w[k] · tmpl[c · n + k]` for `c in 0..channels` (kept-channel index).
    // The SAXPY path rebuilds these per channel inside its loop; the descent
    // visits ~10 cells per call, so amortising the rebuild across all of them
    // takes the kern build off the per-cell hot path. Reusing the
    // `SearchScratch` buffers means no allocation per `search_shift` call
    // after the first.
    sc.pd_kerns.clear();
    sc.pd_kerns.resize(channels * n, 0.0);
    sc.pd_tsums.clear();
    sc.pd_tsums.resize(channels, 0.0);
    // Precompute the kept-channel-index → tile-channel-index lookup once per
    // call; the per-cell scoring loop indexes into this rather than walking
    // `keep_mask` linearly each time. Drops the per-cell scoring's keep_mask
    // dispatch from O(channels) (one walk per kept channel) to O(1).
    sc.pd_kept_channels.clear();
    sc.pd_kept_channels.extend(
        keep_mask
            .iter()
            .enumerate()
            .filter_map(|(c, &k)| k.then_some(c)),
    );
    debug_assert_eq!(sc.pd_kept_channels.len(), channels);
    for kc_out in 0..channels {
        let tmpl_c = &sc.tmpl[kc_out * n..][..n];
        let kern_c = &mut sc.pd_kerns[kc_out * n..][..n];
        let mut tsum = 0.0f64;
        for ((&sw, &t), kk) in support
            .sqrt_weights
            .iter()
            .zip(tmpl_c)
            .zip(kern_c.iter_mut())
        {
            let v = sw * t;
            *kk = v;
            tsum += v as f64;
        }
        sc.pd_tsums[kc_out] = tsum;
    }

    // Dense visited cache, sentinel-encoded in `pd_visited[idx(dy, dx)]`:
    // `f64::NAN` = unvisited, `f64::NEG_INFINITY` = visited+unscorable,
    // any other finite value = visited+scored. Replaces the previous
    // `HashMap<(i64, i64), Option<f64>>` — the dense Vec is sized
    // `(2·margin+1)²` (169 for the production default), fits in one cache
    // line of pointers, and the index→slot lookup is a couple of arithmetic
    // ops vs a hash + bucket walk.
    let span_axis = (2 * margin + 1) as usize;
    let gsz = span_axis * span_axis;
    sc.pd_visited.clear();
    sc.pd_visited.resize(gsz, f64::NAN);
    let idx_of = |dy: i64, dx: i64| -> usize {
        ((dy + margin) as usize) * span_axis + ((dx + margin) as usize)
    };

    // Pre-size the per-cell `(n, s1, s2)` scratch to exactly `channels` slots
    // so the per-cell SEARCH_ACC block writes by index — no Vec bookkeeping
    // (clear/push) inside the timed kernel block. Initial value is overwritten
    // by every scoreable cell before COMBINE reads it.
    sc.pd_per_channel.clear();
    sc.pd_per_channel.resize(channels, (0.0, 0.0, 0.0));

    let inv_total_weight = 1.0 / support.total_weight;
    let chf = channels as f64;

    // Score one (dy, dx) cell across all kept channels and combine into the
    // mean-over-channels ZNCC. Returns `None` for out-of-bounds, out-of-disk,
    // or any-invalid-support-pixel cells; cached either way so neighbors
    // revisited by the descent walk hit the slot. Profile attribution:
    // `SEARCH_ACC` wraps the data-touching kernel (invalidity-count +
    // per-channel scoring), `SEARCH_COMBINE` wraps the per-channel ZNCC
    // algebra + cross-channel sum, and `N_CELLS` is bumped once per cell that
    // gets actually scored (visited-cache hits and oob/oof/disk skips do not
    // count).
    macro_rules! score_cell {
        ($dy:expr, $dx:expr) => {{
            let dy_: i64 = $dy;
            let dx_: i64 = $dx;
            if dy_.abs() > margin || dx_.abs() > margin {
                None
            } else {
                let slot = sc.pd_visited[idx_of(dy_, dx_)];
                if slot.is_nan() {
                    // First visit: compute the score (or detect unscorable).
                    let win_y = (base_y as i64 + dy_) as usize;
                    let win_x = (base_x as i64 + dx_) as usize;
                    let ginv = prof::SEARCH_ACC.time(|| {
                        let ginv = count_invalid_at_cell(
                            &tile.invalid_plane,
                            support,
                            resolution,
                            istride,
                            win_y,
                            win_x,
                        );
                        if ginv <= 0.5 {
                            for k in 0..channels {
                                let tile_c = sc.pd_kept_channels[k];
                                let kern_k = &sc.pd_kerns[k * n..][..n];
                                sc.pd_per_channel[k] = score_cell_one_channel(
                                    &tile.planes[tile_c],
                                    support,
                                    kern_k,
                                    &sc.w_f32,
                                    resolution,
                                    istride,
                                    win_y,
                                    win_x,
                                );
                            }
                        }
                        ginv
                    });
                    let result = if ginv > 0.5 {
                        sc.pd_visited[idx_of(dy_, dx_)] = f64::NEG_INFINITY;
                        None
                    } else {
                        let zncc = prof::SEARCH_COMBINE.time(|| {
                            let mut combined = 0.0_f64;
                            for (k, &(n_acc, s1_acc, s2_acc)) in
                                sc.pd_per_channel.iter().enumerate()
                            {
                                let s1 = s1_acc as f64;
                                let s2 = s2_acc as f64;
                                let nval = n_acc as f64;
                                let mean = s1 * inv_total_weight;
                                let norm_sq = s2 - s1 * mean;
                                if norm_sq >= FLAT_NORM_SQ_EPS {
                                    combined += (nval - mean * sc.pd_tsums[k]) / norm_sq.sqrt();
                                }
                            }
                            combined / chf
                        });
                        prof::count(&prof::N_CELLS, 1);
                        sc.pd_visited[idx_of(dy_, dx_)] = zncc;
                        Some(zncc)
                    };
                    result
                } else if slot == f64::NEG_INFINITY {
                    None
                } else {
                    Some(slot)
                }
            }
        }};
    }

    // Seed at the cache's centre (current integer base offset).
    let mut current_phi = score_cell!(0_i64, 0_i64)?;
    let mut current = (0_i64, 0_i64);

    // Steepest-descent walk: pick the best improving neighbor, stop when none.
    // The per-neighbor best-of-4 pick is a handful of `f64` comparisons —
    // small enough relative to the score_cell calls driving it that it is not
    // separately timed.
    loop {
        let mut best_move: Option<((i64, i64), f64)> = None;
        for (dy_step, dx_step) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = (current.0 + dy_step, current.1 + dx_step);
            if let Some(phi) = score_cell!(next.0, next.1) {
                if phi > current_phi && best_move.is_none_or(|(_, bs)| phi > bs) {
                    best_move = Some((next, phi));
                }
            }
        }
        match best_move {
            None => break,
            Some((next, phi)) => {
                current = next;
                current_phi = phi;
            }
        }
    }

    // SEARCH_ARGMAX: separable parabolic sub-pixel refinement from the 4
    // cardinal-neighbor cells — all already in the visited cache (each was
    // evaluated by the STOP-check loop above). A neighbor that scored `None`
    // (out-of-grid / out-of-disk / out-of-frame) drops its axis to the
    // integer offset. Matches `search_shift`'s SEARCH_ARGMAX wrap, which
    // similarly times the argmax + parabolic.
    prof::SEARCH_ARGMAX.time(|| {
        let py = current.0;
        let px = current.1;
        let nb = |dy: i64, dx: i64| -> Option<f64> {
            if dy.abs() > margin || dx.abs() > margin {
                return None;
            }
            let v = sc.pd_visited[idx_of(dy, dx)];
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        };
        let sy = match (nb(py - 1, px), nb(py + 1, px)) {
            (Some(l), Some(r)) => parabolic(current_phi, l, r),
            _ => 0.0,
        };
        let sx = match (nb(py, px - 1), nb(py, px + 1)) {
            (Some(l), Some(r)) => parabolic(current_phi, l, r),
            _ => 0.0,
        };
        Some(ShiftResult {
            dx: px as f64 + sx,
            dy: py as f64 + sy,
            ix: px,
            iy: py,
            peak: current_phi,
        })
    })
}
