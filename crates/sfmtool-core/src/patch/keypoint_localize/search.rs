// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-view windowed-ZNCC translation search, split out of the
//! localization orchestration ([`super`]).
//!
//! [`search_shift`] scores the whole `±margin` shift grid by accumulation
//! ([`SearchStrategy::Exhaustive`](super::SearchStrategy::Exhaustive));
//! [`search_shift_plus_descent`] walks a steepest-ascent path scoring one cell
//! at a time ([`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)).
//! Both correlate a view's context tile ([`ContextTile`]) against a fixed
//! template, the point's reference render. The data-touching correlation
//! kernels live in [`super::kernels`].

use crate::patch::normal_refine::{Support, FLAT_NORM_SQ_EPS};

use super::kernels::{
    accumulate_count, compute_channel_grids, count_invalid_at_cell, score_cell_one_channel,
    support_offsets,
};
use super::{prof, subpixel_peak, ContextTile, LocalizeError};

/// Reused per-call scratch for [`search_shift`], created once per
/// [`localize_patch_keypoints`](super::localize_patch_keypoints) and shared
/// across every view (the search then allocates nothing after
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
    /// The template the candidates score against (`kept_ch · n`), laid out
    /// `[c · n + k]`; the caller writes it before each search.
    pub(super) tmpl: Vec<f32>,
    /// Per-support-pixel kernel `√w · tmpl` for the channel being accumulated
    /// (`f32` — the AVX2 kernel broadcasts these as `f32` lanes).
    pub(super) kern: Vec<f32>,
    /// Per-support-pixel window weight `w` as `f32` (one-time conversion from the
    /// `f64` `support.weights`; broadcast each k-step).
    pub(super) w_f32: Vec<f32>,
    /// Per-support-pixel offset `r_k · istride + c_k` in the cache plane
    /// ([`support_offsets`]), computed once per [`search_shift`] call and read
    /// by both whole-grid kernels for every channel and grid row.
    pub(super) offsets: Vec<usize>,
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

impl SearchScratch {
    /// Reserve the shift grids for a `span × span` search window, fallibly.
    ///
    /// Every grid here is `span²` and the search resizes into it, so reserving
    /// the capacity once means each later `resize` sits inside the capacity and
    /// allocates nothing. That is what makes the size the **caller's** search
    /// radius implies a refusal a step can report rather than an abort inside
    /// the global allocator: a window wide enough to matter is `(2 · margin +
    /// 1)²` cells, which grows as the square of the radius.
    pub(super) fn try_reserve_grids(&mut self, span: usize) -> Result<(), LocalizeError> {
        let cells = span.saturating_mul(span);
        let reserve_f32 = |buffer: &mut Vec<f32>| -> Result<(), LocalizeError> {
            buffer
                .try_reserve_exact(cells.saturating_sub(buffer.len()))
                .map_err(|_| LocalizeError::OutOfMemory {
                    bytes: cells.saturating_mul(std::mem::size_of::<f32>()),
                })
        };
        let reserve_f64 = |buffer: &mut Vec<f64>| -> Result<(), LocalizeError> {
            buffer
                .try_reserve_exact(cells.saturating_sub(buffer.len()))
                .map_err(|_| LocalizeError::OutOfMemory {
                    bytes: cells.saturating_mul(std::mem::size_of::<f64>()),
                })
        };
        reserve_f32(&mut self.g_n)?;
        reserve_f32(&mut self.g_s1)?;
        reserve_f32(&mut self.g_s2)?;
        reserve_f32(&mut self.ginv)?;
        reserve_f64(&mut self.grid)?;
        reserve_f64(&mut self.pd_visited)?;
        Ok(())
    }
}

/// The result of a [`search_shift`] — the shift of one view relative to the
/// core at its starting keypoint, in grid px.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShiftResult {
    /// Sub-pixel-refined shift in the grid's x (column) axis: integer argmax plus
    /// the residual of the 3×3 quadratic fit (`subpixel_peak`).
    pub(super) dx: f64,
    /// Sub-pixel-refined shift in the grid's y (row) axis.
    pub(super) dy: f64,
    /// The integer argmax shift in x (every tile read stays at an integer
    /// index).
    pub(super) ix: i64,
    /// The integer argmax shift in y.
    pub(super) iy: i64,
    /// ZNCC at the integer peak.
    pub(super) peak: f64,
}

/// Integer windowed-ZNCC translation search of one view's context tile against
/// `sc.tmpl`, refined to sub-pixel by a quadratic fit over the 3×3 neighbourhood
/// of the integer peak (`subpixel_peak`). Returns a
/// [`ShiftResult`] — the integer argmax shift `(ix, iy)`, its sub-pixel-refined
/// counterpart `(dx, dy)`, and the ZNCC `peak` at the integer peak — or `None` if
/// no in-frame window position could be scored. The `(dy, dx) = (0, 0)` shift
/// scores the `R×R` core window whose top-left support pixel sits at tile index
/// `(base_y, base_x)`, and the search slides the window over `±margin` around it.
/// Both `base ± margin` must be in bounds: the localizer renders the tile
/// `R + 2·margin` wide and passes `base = margin`.
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

    support_offsets(support, resolution, istride, &mut sc.offsets);

    // Validity: count out-of-frame support pixels per shift (channel-independent);
    // a shift with any is unscorable, matching `extract_core`'s all-valid gate.
    // A tile with every pixel in frame has an all-zero invalidity plane (its
    // pad columns are zero too), so the count would add only zeros to the
    // zeroed grid and is skipped.
    sc.ginv.clear();
    sc.ginv.resize(gsz, 0.0);
    if !tile.all_valid {
        accumulate_count(
            &tile.invalid_plane,
            &sc.offsets,
            istride,
            span,
            win_oy,
            win_ox,
            &mut sc.ginv,
        );
    }

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
                &sc.offsets,
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
    // quadratic fit over its 3×3 neighbourhood. Out-of-frame shifts score −∞
    // (never chosen).
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
                // An exact tie goes to the cell nearer the start, so a window
                // that scores the same everywhere (a flat view scores 0 at
                // every shift) leaves the view where it started, as the
                // descent does, rather than at the window's first corner.
                let nearer = || dy.abs() + dx.abs() < best.1.abs() + best.2.abs();
                if z > best.0 || (z == best.0 && nearer()) {
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
        let (sy, sx) = subpixel_peak(peak, py, px, nb);
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
/// to the best improver, and stops when no neighbor beats the current cell. It
/// then scores the 4 diagonal neighbors; where one beats the current cell, it
/// moves to the best of them and walks on. Each
/// cell is scored at most once via [`score_cell_one_channel`]; the visited cache
/// stores the combined ZNCC per cell. The final sub-pixel fit is a quadratic
/// over the 3×3 neighbourhood of the final cell, which no scored neighbour
/// beats: the 4 cardinal neighbors are in the cache from the STOP check, and
/// the 4 diagonal ones from the diagonal check.
///
/// Same `ShiftResult` contract as `search_shift` — the integer argmax `(ix, iy)`
/// drives the read accumulator, `(dx, dy)` carry the sub-pixel residual, and
/// `peak` is the combined ZNCC at the integer cell. Bounded to `|dy|, |dx| ≤
/// margin`; neighbors past the bound or with any out-of-frame support pixel
/// score `None` (skipped, never chosen).
///
/// Cells visited per call: about `9 + 3 · walk_steps` (1 seed + 4 neighbors per
/// step, with 1 cache hit per move, + 4 diagonals for the fit, and a few more
/// each time a diagonal move restarts the walk). Before the diagonals
/// were added, the average on dino was ~6 cells per call — vs
/// the 169 cells of the default ±6 grid `search_shift` processes — at ~32 µs
/// per call (the per-cell `vgatherdps` kernel) vs ~145 µs (the SAXPY). The
/// crossover with `search_shift`'s whole-grid SIMD is around 50–80 cells
/// visited, so the descent loses on pathological long walks; `Exhaustive`
/// remains the right pick for the global-argmax fallback. See
/// `specs/core/patch/keypoint-localization-search-cache.md` for the strategy
/// trade-off discussion.
///
/// **Profile attribution** (`SFMTOOL_PROFILE=1`): the descent reports per-cell
/// `SEARCH_ACC` (invalidity-count + per-channel scoring), per-cell
/// `SEARCH_COMBINE` (mean / ZNCC fold + cross-channel sum), and per-call
/// `SEARCH_ARGMAX` (the final sub-pixel fit). The `N_CELLS` event counter bumps
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
                        // A tile with every pixel in frame has no invalid
                        // pixel to count.
                        let ginv = if tile.all_valid {
                            0.0
                        } else {
                            count_invalid_at_cell(
                                &tile.invalid_plane,
                                support,
                                resolution,
                                istride,
                                win_y,
                                win_x,
                            )
                        };
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
    //
    // The 2-D sub-pixel fit needs the four diagonal cells around the final
    // cell too; the 4 cardinal ones are already in the visited cache (each was
    // evaluated by the STOP-check loop). Scoring them costs 4 cells per stop,
    // and without them the fit has no cross term. A diagonal that beats the
    // cell the walk stopped at means the peak is not there: the walk moves to
    // the best such diagonal and goes on, so the fit is always centred on a
    // cell no scored neighbour beats.
    loop {
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
        let mut best_diagonal: Option<((i64, i64), f64)> = None;
        for (dy_step, dx_step) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let next = (current.0 + dy_step, current.1 + dx_step);
            if let Some(phi) = score_cell!(next.0, next.1) {
                if phi > current_phi && best_diagonal.is_none_or(|(_, bs)| phi > bs) {
                    best_diagonal = Some((next, phi));
                }
            }
        }
        match best_diagonal {
            None => break,
            Some((next, phi)) => {
                current = next;
                current_phi = phi;
            }
        }
    }

    // SEARCH_ARGMAX: 2-D quadratic sub-pixel refinement over the 3×3
    // neighbourhood (see `subpixel_peak`). A neighbor that scored `None`
    // (out-of-grid / out-of-disk / out-of-frame) drops to the separable fit,
    // or to the integer offset on its axis. Matches `search_shift`'s
    // SEARCH_ARGMAX wrap, which similarly times the argmax + sub-pixel fit.
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
        let (sy, sx) = subpixel_peak(current_phi, py, px, nb);
        Some(ShiftResult {
            dx: px as f64 + sx,
            dy: py as f64 + sy,
            ix: px,
            iy: py,
            peak: current_phi,
        })
    })
}
