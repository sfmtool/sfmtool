# Photometric refinement via per-tile RANSAC subset partition

**Status:** Draft. Standalone specification — implementable from
scratch without reference to any other photometric refinement
approach. The production target is Rust on the existing
`PerSphericalTileSourceStack` data structures, with PyO3 bindings;
Python code throughout this spec is illustrative pseudocode (it
also matches the validation prototype that produced the empirical
metrics in the table below). Folds into `specs/core/` once a Rust
v1 lands and the production pipeline consumes its outputs.

[`PerSphericalTileSourceStack`]: ../core/per-spherical-tile-source-stack.md
[`SphericalTileRig`]: ../core/spherical-tiles-rig.md

## Problem statement

The input is a [`PerSphericalTileSourceStack`] — a set of source
images that have already been projected into a common spherical
projection space and split per direction-tile, so each tile carries
one patch per source that contributed to it.

Each `(tile, source)` contribution — one patch from one source
aimed at one direction tile — is the algorithm's atomic unit, and
is called a *row* throughout this spec. The full stack flattens
into `R` such rows; per-contribution data (patch pixels, source
ID, tile ID, cluster membership) lives in arrays indexed by
`r ∈ [0, R)`.

Those source images do not necessarily agree photometrically:
cameras, exposures, and capture moments differ, and we do not have
full confidence in how the per-source colour values relate. The
colour-correction model this spec solves for is **fixed gamma +
per-source gain**: gamma is supplied by the caller and held fixed
(see the `gamma` knob and rationale below), and per-source gain is
what the algorithm estimates. Richer corrections (per-channel gain,
vignetting, per-source gamma) are explicitly out of scope for v1.

The end goal is a per-tile estimate of the "correct" pixel value in
each direction. This is an estimation problem solved by *finding a
large subset of contributing sources whose patches agree* — after
gain correction — on what the scene looks like in that direction,
and treating that subset's consensus as the truth. The disagreeing
minority is itself informative: for a tile that should show distant
radiance from infinity, some sources may see that radiance while
others have a near-object obscuring the line of sight; the consensus
picks out the agreeing group, and the disagreeing rows get tagged
for the downstream consensus rule to handle as occluders / parallax
/ sky-leak / mask edges.

The outputs are recovered together:

- **Per-source gain** colour-corrects each source's contribution
  before agreement is assessed.
- **Per-tile primary cluster** — the largest agreeing subset. Its
  rows drive the gain fit and its consensus is the algorithm's best
  estimate of the per-tile "correct" value.
- **Per-tile secondary cluster** — the largest agreeing subset
  *within the primary's complement*. When a tile genuinely splits
  into two coherent groups (e.g. some sources see distant radiance
  and others see a near-object obscuring it), this is the other
  group, named explicitly. When there is no coherent runner-up the
  cluster is empty.

The two cluster sizes together give downstream consumers a
**validity signal** for each tile: a large primary with a small
secondary is high-confidence; comparably sized clusters mean the
tile is ambiguous and should either be flagged or resolved by
appeal to neighbouring tiles. Exposing both clusters as raw
row-membership masks — rather than collapsing them into a single
confidence score — lets a neighbour-consistency post-pass promote
the secondary to "chosen" when it agrees with surrounding tiles'
choices, or downgrade the tile entirely when both clusters are
small.

Per-tile populations are small — typically 4–25 contributors on the
bundled datasets — so per-tile work is cheap; the dominant cost is
the global gain solve over all tiles together.

## Design rationale

The spec separates the two coupled subproblems — partition and
gain — and alternates between them:

1. **Identify the consensus subset per tile by RANSAC.** Sample
   small subsets, score each by how many contributors agree with
   the subset's hypothesised consensus luminance under a fixed
   inlier threshold, keep the highest-scoring inlier set. The
   "largest agreeing subset" is a well-defined object (defined by
   score, not by an arbitrary cluster index), so the binary inlier
   mask has no permutation ambiguity.

2. **Recover per-image gains by least squares on inlier rows
   only.** With the partition fixed, per-image log-gain is a small,
   well-conditioned linear problem in log-luminance space — no
   autodiff, no line search.

The two steps couple through alternation: the partition depends on
the gains used to colour-correct the rows before scoring, and the
gains are fit against the partition's inliers. Empirically (see
[Empirical metrics](#empirical-metrics) below) the alternation
converges in 3–8 outer iterations on the bundled datasets.

### Threshold loose enough to admit gain-only disagreements

The first outer iteration runs RANSAC with `log_gain = 0`, i.e. on
patches that have not yet been colour-corrected. If `inlier_threshold`
is tight relative to the worst-case raw gain disagreement between
sources, a brighter (or darker) source's rows will fail the L1
inlier test on iteration 1, get excluded from the LSQ, and the
alternation can settle into a partition that has silently decided
which sources are "primary" before any gain has been recovered. The
behaviour is fundamental to alternation, not a bug, and the
mitigation is to keep the threshold loose enough that gain-only
disagreements (rows whose patches differ only by a multiplicative
luminance factor within plausible inter-camera gains, ≤ ~1.5×) are
inliers on iteration 1. The default `inlier_threshold = 8.0`
admits gain-only disagreements up to ~3 % of u8 luminance per pixel,
which suffices for the bundled rigs; on captures with wider
inter-camera variation the threshold should be relaxed first
before reaching for damping or other knobs.

### Why per-tile RANSAC fits the cost shape

RANSAC's headline cost is combinatorial blowup with set size —
(K choose m) candidate subsets at minimal subset size `m`. Per-tile
populations of 4–25 keep (K choose 2) ≤ 300 and (K choose 3) ≤ 2300,
so near-exhaustive subset enumeration is cheap. The `max_subsets` cap
keeps the per-tile budget bounded even on the densest rigs, and
because the work is per-tile-independent the entire RANSAC step
parallelises trivially.

## High-level algorithm

One outer iteration carries three pieces of state forward and back:

```
state at iteration k:
    log_gain[i]     : f32  per-image log-gain          (init zero)
    primary_mask[r] : bool per-row primary-cluster flag (init all-false;
                                                         overwritten on
                                                         iteration 1 by
                                                         the first RANSAC)

per outer iteration:
    1. Apply log_gain to every row's scoring sub-patch and
       per-row mean luminance.
    2. For each tile t with at least min_inliers contributors,
       run RANSAC against the corrected sub-patches to get the
       tile's new primary cluster. Update primary_mask.
    3. Solve a per-image log-gain LSQ on primary rows only,
       producing a fresh log_gain (mean-zero).
    4. Stop if primary_mask did not change (or changed by
       fewer than mask_change_tolerance · R rows), or if outer_iter
       hit max_outer.

after the alternation converges:
    5. For each tile t, run RANSAC against the corrected
       sub-patches of t's primary-rejected rows to get t's
       secondary cluster (empty when too few coherent rows
       remain). Write to secondary_mask.
```

The output is `(log_gain, primary_mask, secondary_mask)` plus
diagnostics. Downstream consumers use `primary_mask[r]` as a hard
"is row `r` part of its tile's consensus?" bit and may consult
`secondary_mask[r]` for the runner-up partition (e.g. for
neighbour-consistency post-passes). Gains are applied
multiplicatively in linear-luminance space (i.e. multiply pixel
values by `exp(log_gain[i])`) before any consensus or compositing
pass.

The secondary cluster is computed only once, after the alternation
converges, because it does not influence the gain solve and the
extra per-tile RANSAC pass would otherwise multiply the
per-iteration cost without changing the result.

## Inputs

The algorithm consumes a flat row-major representation of the
[`PerSphericalTileSourceStack`] read at the pyramid level whose
patch side equals `target_patch_size` (a knob, see below): one
row per `(tile, source)` contribution as defined in the Problem
statement, indexed by `r ∈ [0, R)`. Rows are sorted by `tile_index`
so each tile occupies a contiguous range, given by `tile_offsets`
(CSR-style). Each `(tile, source)` pair appears at most once —
guaranteed by the upstream stack and assumed throughout the LSQ
formulation, which would otherwise double-count a source's
contribution to a tile's mean.

Choosing the level by `target_patch_size` rather than reading
level 0 directly avoids redoing the spatial averaging the pyramid
has already done — at level `L`, each pixel is already the mean
of a `2^L × 2^L` block of the level-0 patch, so picking the right
level is the cheapest way to get a spatially smoothed value. The
match is exact: the algorithm errors if no pyramid level has patch
side equal to `target_patch_size`, putting the responsibility for
building a pyramid that contains the requested level on the caller
and avoiding any rounding or resampling at this layer.

In the field shapes below:
- `s` is the patch side at the chosen level, equal to
  `target_patch_size`.
- `ss` is the side of the central scoring sub-patch, equal to
  `scoring_patch_size`.
- `C` is the channel count of the source images (typically 3 for
  RGB).

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `patches` | `[R, s, s, C]` | f32 | per-row patch pixels in u8 units (i.e. f32 cast of u8 values, 0–255) |
| `valid` | `[R, s, s]` | u8 | per-pixel valid mask, strictly `{0, 1}`: `1` where the source projection lands inside its image bounds, `0` outside. Matches the upstream [`PerSphericalTileSourceStack`]'s `PatchLevel::valid` semantics |
| `src_index` | `[R]` | i32 | which source image contributed this row, in `[0, N)` |
| `tile_index` | `[R]` | i32 | which tile this row contributes to, in `[0, n_tiles)` |
| `tile_offsets` | `[n_tiles + 1]` | i32 | CSR-style row offsets: rows for tile `t` are `[tile_offsets[t], tile_offsets[t+1])`. Implies rows are sorted by `tile_index`. |
| `n_tiles` | scalar | i32 | tile count `n` |
| `n_sources` | scalar | i32 | source count `N` |

Plus tuning knobs:

| Knob | Default | Meaning |
|------|---------|---------|
| `inlier_threshold` | `8.0` u8 lum units | residual threshold separating inliers from outliers, applied to the validity-weighted L1-mean of `(row_patch − consensus_patch)` over the central `ss × ss` sub-patch in linearised u8 luminance units. See Step 3 |
| `gamma` | `1.0` | decoding gamma applied to source patches (`linear = 255·(patch/255)^gamma`) before per-row mean luminance; not refit by the algorithm. Default `1.0` keeps the algorithm in sRGB-encoded u8 space — empirically the only value that meets the inter-seed stability criterion across the bundled rigs. Override with a known camera gamma when downstream demands true linear-light behaviour (see rationale below) |
| `target_patch_size` | `4` | side length of patches read from the pyramid; the algorithm picks the level whose patch side equals this **exactly** and errors if no such level exists. Must be a power of two between 2 and the stack's `base_patch_size` (the upstream pyramid only produces sizes `base_patch_size >> level_idx`) |
| `scoring_patch_size` | `2` | side length of the centred sub-patch used by the scorer; an exact integer crop of the read patch (no resampling). Must satisfy `1 ≤ scoring_patch_size ≤ target_patch_size` and `(target_patch_size − scoring_patch_size)` even, so the central crop is symmetric. Since `target_patch_size` is a power of two (and so even), `scoring_patch_size` must be even too — `1` is rejected |
| `subset_size` | `2` | RANSAC minimal subset size `m`. `m=2` is the recommended default: more tiles fall under the `max_subsets = 64` exhaustive-enumeration cap (the largest K still enumerated exhaustively is K=11 at m=2 vs K=8 at m=3), and inter-seed gain stability is measurably better than at `m=3` on dense rigs |
| `max_subsets` | `64` | cap on candidate subsets per tile when `C(K, m) > max_subsets` |
| `min_inliers` | `2` | tiles with fewer contributors than this are skipped |
| `max_outer` | `8` | cap on outer alternation iterations |
| `mask_change_tolerance` | `0.05` | fraction of `R`; stop when fewer than `mask_change_tolerance · R` rows in the inlier mask flip between iterations. Empirically the typical bundled-rig run hits `max_outer` rather than this tolerance — final-iteration mask flutter sits at 3–5 % of `R` even after the median primary MAD has stopped moving; `0.05` matches the operationally-observed convergence floor while leaving headroom |
| `seed` | `0` | RNG seed for reproducible subset sampling. Per-tile RNG state is derived from `(seed, tile_index)` so per-tile RANSAC can be parallelised (e.g. via `rayon`) without making the result depend on execution order |
| `saturation_threshold` | `254` | per-channel u8 cutoff above which a pixel is treated as saturated; pixels at or above this value have their entry in the working validity mask zeroed at the start of Step 1. The upstream `PatchLevel::valid` only encodes source-bounds visibility, so saturation masking is the algorithm's own responsibility. Set to `255` to disable; `254` is the default because sensor clipping at u8 is ragged and a 1-unit margin catches most clipped highlights |

Sources are assumed to share a single fixed decoding gamma that the
algorithm does not refit. The value is a caller-supplied knob
(`gamma`, see the table above) defaulting to `1.0`: the algorithm
operates directly on sRGB-encoded u8 values, and a fixed
`inlier_threshold = 8.0` u8 units is roughly perceptually-uniform
in sRGB — uniform discrimination across the luminance range is what
RANSAC's hard threshold actually wants. Patches are linearised by
`linear = 255 · (patch/255)^gamma` before Step 1's per-row mean
luminance is computed; at `gamma = 1.0` this is a no-op. Subsequent
steps and `inlier_threshold` are interpreted on the linearised
representation.

The standard sRGB decoding gamma `2.2` was tested but produced
runaway gains on dense outdoor rigs: linearising the u8
representation expands dynamic range non-uniformly, so a fixed
threshold becomes asymmetrically loose at low luminance and tight
at high luminance, and the iteration never settles on bright
content like sky or sunlit foliage. Within-seed `log_gain` std hit
0.99 at γ=2.2 (vs 0.07 at γ=1.0), and inter-seed std exceeded the
spec's 0.02 stability criterion by an order of magnitude. If true
linear-light correctness becomes important downstream, the right
move is to scale `inlier_threshold` with per-tile luminance — a
multiplicative threshold of e.g. `8 · L_t / 100` — rather than
operate in linear space with a constant additive threshold.

Gamma is held fixed rather than jointly refit because
`(log_gamma, log_gain)` is unidentifiable on inputs without
saturation cues: an optimiser can trade exponent mass for gain
without changing the loss, so `log_gamma` runs off without
constraint. Fixing gamma at a caller-supplied value sidesteps the
ambiguity entirely.

## Outputs

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `log_gain` | `[N]` | f32 | per-image log-gain, mean-zero across sources by construction |
| `primary_mask` | `[R]` | bool | per-row primary-cluster flag — `true` ⇔ row is in its tile's largest agreeing subset |
| `secondary_mask` | `[R]` | bool | per-row secondary-cluster flag — `true` ⇔ row is in the largest agreeing subset within the primary's complement; `false` for primary rows and for tiles with no coherent runner-up |
| `tile_primary_count` | `[n_tiles]` | i32 | size of each tile's primary cluster (0 for skipped tiles) |
| `tile_secondary_count` | `[n_tiles]` | i32 | size of each tile's secondary cluster (0 where no runner-up) |
| `tile_primary_lum_mad` | `[n_tiles]` | f32 | Median Absolute Deviation `median_i(|x_i − median(x))|` of `x_i = row_lum_corrected[i]` over each tile's primary cluster, in u8 luminance units (NaN where `tile_primary_count < min_inliers`). Textbook robust scale estimator; `1.4826 · MAD ≈ σ` for normal data. Note this is computed on the per-row scalar mean luminance, **not** on the patch-L1 residual the RANSAC scorer uses — it is a downstream-friendly summary statistic, not the algorithm's loss. A tile in which patches with the same mean luminance but different spatial structure (e.g. checker vs constant) get split between primary and secondary will have a small `lum_mad` on both clusters even though the patch-L1 score that drove the split was discriminating |
| `tile_secondary_lum_mad` | `[n_tiles]` | f32 | Same MAD definition over each tile's secondary cluster (NaN where `tile_secondary_count < min_inliers`); same caveat as `tile_primary_lum_mad` |
| `outer_iters` | scalar | i32 | number of outer iterations executed |
| `mask_change_history` | `[outer_iters]` | i32 | per-iteration `primary_mask` Hamming distance from previous iteration |

`primary_mask` names which rows participate in each tile's
consensus and can be consumed directly by the downstream consensus
rule without a disambiguation step. `secondary_mask` together with
the `tile_*_count` fields is the per-tile validity signal: high
confidence when the primary dominates, ambiguous when the two
clusters are comparable, low confidence when both are small. Raw
row memberships and counts are exposed deliberately rather than a
single derived confidence score, so downstream is free to apply its
own formula (margin, fraction-dominant, neighbour-consistency
fusion) without renegotiating the wire format. `log_gain` is
applied multiplicatively in linear space (`exp(log_gain[i])` per
source) before any consensus or compositing pass.

## Detailed steps

### Step 1: Per-row scoring patch and mean luminance

For each row `r` with patch `P[r] ∈ R^{s × s × C}` and valid mask
`V[r] ∈ R^{s × s}`, extract the central `ss × ss` sub-patch
(`ss = scoring_patch_size`), zero the validity entries for pixels
where any channel is at or above `saturation_threshold`, and
linearise the patch values by the fixed `gamma` knob
(`linear = 255 · (P/255)^gamma`). The result is a per-row linearised
scoring patch used by Step 3's RANSAC, a per-row working validity
patch used by both Step 3 and Step 4, and a per-row scalar mean
luminance — the validity-weighted mean of the linearised sub-patch
across pixels and channels — used by Step 4's LSQ:

```python
def per_row_scoring_patch(patches, valid, gamma, ss, saturation_threshold):
    # patches: [R, s, s, C] in u8 units, read at the
    #          target_patch_size pyramid level (s = target_patch_size,
    #          a power of two, 2 ≤ s ≤ base_patch_size)
    # valid:   [R, s, s] strictly {0, 1}
    # gamma:   scalar
    # ss:      scoring_patch_size, even, 2 ≤ ss ≤ s (and (s − ss) even)
    # saturation_threshold: u8 cutoff (default 254); set to 255 to disable
    s = patches.shape[1]
    half_diff = (s - ss) // 2
    lo, hi    = half_diff, half_diff + ss
    centre    = patches[:, lo:hi, lo:hi, :]              # [R, ss, ss, C]
    centre_v  = valid[:,   lo:hi, lo:hi].astype(np.float32)  # [R, ss, ss]
    sat_mask  = (centre.max(axis=-1) < saturation_threshold).astype(np.float32)
    centre_v  = centre_v * sat_mask                       # [R, ss, ss]
    linear    = 255.0 * (centre / 255.0) ** gamma         # [R, ss, ss, C]
    return linear, centre_v                               # uncorrected

def per_row_mean_lum(linear_patch, valid_patch):
    # linear_patch: [R, ss, ss, C] linearised central sub-patch
    # valid_patch:  [R, ss, ss] per-pixel validity (in [0, 1])
    valid_b = valid_patch[..., None]                 # [R, ss, ss, 1]
    num     = (linear_patch * valid_b).sum(axis=(1,2,3))
    denom   = valid_b.sum(axis=(1,2,3)) * linear_patch.shape[-1]
    return num / np.clip(denom, 1.0, None)           # [R]
```

The pyramid has already done the spatial averaging that took the
original level-0 patch down to side `s`; the central `ss × ss`
sub-patch picks out the tile-direction-pointed-at sub-region —
covering the central `(ss/s)²` of the original patch's area —
without redoing that work. At `target_patch_size = 4` and
`scoring_patch_size = 2`, the central 2×2 covers `(s_0/2)²` of the
original level-0 pixels, so a 16×16 level-0 patch contributes its
central 8×8 worth of source pixels (already averaged by the pyramid
into 4 values). With `scoring_patch_size = target_patch_size`, the
full pyramid-level patch is used.

Both the linearised sub-patch tensor `[R, ss, ss, C]` and the
per-row scalar `row_lum_uncorrected[r]` are computed once at the
start of the algorithm. Because gain is multiplicative in linear
space, the corrected versions are
`row_patch_uncorrected * exp(log_gain[src_index[r]])` and
`row_lum_uncorrected * exp(log_gain[src_index[r]])` — no need to
re-linearise or re-reduce the patch tensor inside the outer loop.

Validity at the chosen level is binary as produced by the upstream
stack, but after saturation masking the working `centre_v` is in
`{0, 1}` per pixel — entries are zeroed wherever any channel is at
or above `saturation_threshold`. The central `ss × ss` of an
interior, well-exposed tile is virtually always fully valid, but
edge tiles, tiles with masked corners, and tiles aimed at sky or
sunlit foliage can produce a row where `denom` reflects fewer than
`ss² · C` effective samples. The `np.clip(denom, 1.0, None)` floor
avoids division by zero on fully-masked rows; such rows will fall
out as outliers because their luminance value is unreliable, and
the algorithm never reads NaN. The algorithm is forward-compatible
with a future fractional-validity upstream (`centre_v ∈ [0, 1]`):
the validity-weighted reductions all interpret `centre_v` as a
soft weight, so admitting fractional values requires no code
change in this layer.

### Step 2: Apply current gains

```python
gain_per_row        = np.exp(log_gain[src_index])                            # [R]
row_lum_corrected   = row_lum_uncorrected * gain_per_row                  # [R]
row_patch_corrected = (row_patch_uncorrected
                       * gain_per_row[:, None, None, None])               # [R, ss, ss, C]
```

`gain_per_row` is computed in f64 (or f32 with a clamp; gains are
small) and the multiply happens once per outer iteration. The same
gain factor scales both the per-row scalar mean luminance (used by
Step 4) and the per-row sub-patch tensor (used by Step 3).

### Step 3: Per-tile RANSAC (primary cluster)

For each tile `t`, slice its rows out of `row_patch_corrected` and
`valid_patch` using `tile_offsets`, call `ransac_cluster_for_tile`,
and write the boolean result back into the global `primary_mask`.
Each tile gets its own RNG instance seeded from `(seed, t)` (e.g.
`ChaCha8::seed_from_u64(seed.rotate_left(32) ^ t as u64)`) so the
per-tile RANSAC results are independent of execution order — a
prerequisite for the trivially-parallel `rayon` over tiles
promised by the [Complexity](#complexity) section, and for the
seed-stability tests in the [Validation plan](#validation-plan).

```python
new_primary_mask = np.zeros(R, dtype=bool)
for t in range(n_tiles):
    a, b = int(tile_offsets[t]), int(tile_offsets[t+1])
    if b - a < min_inliers:
        continue
    new_primary_mask[a:b] = ransac_cluster_for_tile(
        row_patch_corrected[a:b],
        valid_patch[a:b],
        threshold=inlier_threshold,
        subset_size=subset_size,
        rng=rng,
        max_subsets=max_subsets,
    )
```

#### `ransac_cluster_for_tile`

The per-tile RANSAC inner kernel. This is the algorithmic core of
the spec. The non-obvious details — subset enumeration policy, score
tie-break, degenerate cases — are encoded in the prototype as
follows:

```python
def ransac_cluster_for_tile(row_patch, valid_patch, threshold,
                            subset_size, rng, max_subsets=64):
    """
    row_patch:   [K, ss, ss, C] linearised gain-corrected sub-patches.
    valid_patch: [K, ss, ss] per-pixel validity (in [0, 1]).
    threshold:   inlier threshold in luminance units.
    subset_size: minimal subset size m.
    Returns boolean [K] cluster-membership mask (the largest
    agreeing subset within row_patch).
    """
    K = row_patch.shape[0]
    if K == 0:
        return np.zeros(0, dtype=bool)

    def patch_l1_score(rp, hyp, vp):
        # rp:  [K, ss, ss, C] candidate patches
        # hyp: [ss, ss, C] hypothesis
        # vp:  [K, ss, ss] validity
        residual = np.abs(rp - hyp)                          # [K, ss, ss, C]
        valid_b  = vp[..., None]                             # [K, ss, ss, 1]
        num      = (residual * valid_b).sum(axis=(1, 2, 3))  # [K]
        denom    = valid_b.sum(axis=(1, 2, 3)) * rp.shape[-1]
        return num / np.clip(denom, 1.0, None)               # [K]

    def consensus_patch(idx):
        # validity-weighted per-pixel-per-channel mean over the subset
        sub_p = row_patch[list(idx)]                         # [m, ss, ss, C]
        sub_v = valid_patch[list(idx), ..., None]            # [m, ss, ss, 1]
        num   = (sub_p * sub_v).sum(axis=0)                  # [ss, ss, C]
        denom = sub_v.sum(axis=0)                            # [ss, ss, 1]
        return num / np.clip(denom, 1e-3, None)              # [ss, ss, C]

    if K <= subset_size:
        # Below sampling threshold — accept rows whose patch-L1
        # residual to the per-pixel median is within threshold.
        # Avoids the all-accept degenerate which would break the
        # LSQ step.
        med_patch = np.median(row_patch, axis=0)             # [ss, ss, C]
        return patch_l1_score(row_patch, med_patch, valid_patch) <= threshold

    # Enumerate or sample candidate subsets.
    n_combos = comb(K, subset_size)
    if n_combos <= max_subsets:
        candidates = list(itertools.combinations(range(K), subset_size))
    else:
        seen, candidates = set(), []
        while len(candidates) < max_subsets:
            idx = tuple(sorted(rng.choice(K, subset_size,
                                          replace=False).tolist()))
            if idx in seen:
                continue
            seen.add(idx)
            candidates.append(idx)

    best_inliers = None
    best_count   = -1
    best_score   = float("inf")
    for idx in candidates:
        hyp     = consensus_patch(idx)                            # [ss, ss, C]
        scores  = patch_l1_score(row_patch, hyp, valid_patch)     # [K]
        inliers = scores <= threshold
        n_in    = int(inliers.sum())
        if n_in > best_count:
            best_count, best_inliers = n_in, inliers
            best_score = (float(scores[inliers].mean())
                          if n_in > 0 else float("inf"))
        elif n_in == best_count and n_in > 0:
            mean_s = float(scores[inliers].mean())
            if mean_s < best_score:
                best_inliers, best_score = inliers, mean_s
    return best_inliers if best_inliers is not None \
                        else np.zeros(K, dtype=bool)
```

Key design points:

1. **Hypothesis = validity-weighted subset mean patch.** The "model"
   being fit by each subset is the consensus patch — a `[ss, ss, C]`
   tensor computed as the validity-weighted per-pixel-per-channel
   mean of the subset's patches.

2. **Score is L1-mean of the residual patch.** The inlier test
   `mean(|row_patch - consensus|) ≤ threshold` aggregates the
   `ss² · C` per-pixel residuals into a single scalar in u8
   luminance units. L1 was picked over L2 RMS for outlier
   tolerance — a single hot pixel from a glint shouldn't dominate
   the row's score — and keeps `inlier_threshold` in units
   downstream can reason about directly. See "Why patch-L1 and not
   per-pixel RANSAC / NCC" below for the full rationale.

3. **Tie-break by mean inlier score.** Two subsets that produce
   the same inlier *count* are distinguished by the average score
   among their inliers (lower wins). This avoids RANSAC's classic
   pathology where the first hypothesis with the maximal count is
   kept arbitrarily; tying on the score instead yields a more
   reproducible answer when the threshold is loose.

4. **Degenerate K ≤ m.** When the tile has at most `m` contributors,
   any subset is trivially the full set and always scores K inliers.
   Falling back to "per-pixel median patch + threshold" still
   produces a sensible partition: rows whose patch-L1 distance to
   the per-pixel median is above threshold are flagged, and the
   tile retains a small but coherent primary cluster. **Do not**
   return all-true here — that would feed the LSQ step rows that
   disagree with each other and biases the gain solve.

5. **Exhaustive when feasible, sampled otherwise.** With
   `max_subsets = 64`, exhaustive enumeration covers tiles up to
   K=11 for m=2 ((11 choose 2) = 55 ≤ 64; K=12 already samples
   because (12 choose 2) = 66 > 64) and K=8 for m=3 ((8 choose 3)
   = 56 ≤ 64; K=9 samples because (9 choose 3) = 84). Below the
   boundary the algorithm is fully exhaustive and seed-independent
   on that tile; above it, the subset sampler kicks in. The sampler
   dedups `idx` tuples (a `frozenset` would also work) so that the
   candidate budget is spent on distinct hypotheses.

#### Why patch-L1 and not per-pixel RANSAC / NCC

The per-row score must be (a) robust to per-pixel noise so an
isolated bright/dark pixel doesn't dominate, (b) smooth in
`log_gain` so the LSQ step's Jacobian is well-conditioned, and (c)
cheap (`O(ss² · C)` per row per subset). Validity-weighted L1-mean
over the central `ss × ss` sub-patch ticks all three: it's robust
because individual pixel disagreements are averaged, smooth because
L1 is piecewise-linear in gain, and cheap at default `ss = 2`.

Why not NCC: NCC is invariant to per-source affine shifts in pixel
values, which is exactly the disagreement this primitive is *trying*
to characterise — a row whose patch matches consensus only after a
luminance offset is precisely what the gain solve should be
correcting. NCC is the wrong score here.

Why not per-pixel-independent RANSAC: the standard literature
pattern for multi-dimensional residuals (image registration,
sensor fusion, point-cloud alignment) is "vector residual + scalar
agreement metric," not per-coordinate independent RANSAC followed
by reconciliation. Per-pixel RANSAC discards the joint-evidence
signal — a row that agrees on all `ss² · C` pixels is much stronger
evidence of inlier-ness than a row that agrees on a subset — and
adds an ad hoc reconciliation knob (intersect? union? majority?)
with no principled default.

Why L1 over L2 RMS: both work and have the same O-cost; L1 is less
sensitive to a single outlier pixel and is the simpler default.
A `score_norm` knob is allowed as opt-in surface area but defaults
to L1.

### Step 4: Per-image log-gain LSQ

Once the primary mask is fixed, the per-image gain solve is a small
linear problem in log-luminance space. The non-obvious detail is
the exact row formulation that makes the all-ones direction land
exactly in `null(A)`, which lets `lstsq`'s minimum-norm solution
return a mean-zero answer with no manual recentering.

For a primary row `r` in tile `t`, source `s = src_index[r]`, write
`g[i] = log_gain[i]` and `ℓ[r] = log(row_lum_uncorrected[r])`. After
gain correction, the row's log-luminance is `g[s] + ℓ[r]`. The tile
log-luminance "target" is the per-tile primary mean of the corrected
log-luminance, `(1/|P_t|) Σ_{r' ∈ P_t} (g[src(r')] + ℓ[r'])`. The
residual to drive to zero is

```
res[r] = (g[s] + ℓ[r])
       - (1/|P_t|) Σ_{r' ∈ P_t} (g[src(r')] + ℓ[r'])
```

which is linear in `g`. Stack one such row per primary row into
`A · g = b` where:

```
A[k, src(r)]      += 1
A[k, src(r')]     -= 1 / |P_t|     for each r' ∈ P_t (incl. r itself)
b[k]               =  (1/|P_t|) Σ_{r' ∈ P_t} ℓ[r']  -  ℓ[r]
```

Each row's coefficients on `g` sum to `1 - |P_t| · (1/|P_t|) = 0`,
so `A · 1 = 0` exactly: the all-ones direction is in `null(A)`,
which is the linear-algebra restatement of the shift ambiguity —
adding the same constant to every entry of `g` lands in the
nullspace and produces an identical residual. `np.linalg.lstsq`
(or any minimum-norm solver) returns the projection onto the
row-space of `A`, which is orthogonal to `null(A)` — the answer is
mean-zero by construction, with no need for a soft constraint row
or post-projection.

```python
def solve_log_gain_lsq(row_lum_uncorrected, src_index, tile_offsets,
                       primary_mask, n_sources, n_tiles, log_gain):
    R_total = len(row_lum_uncorrected)
    log_lum = np.log(np.clip(row_lum_uncorrected.astype(np.float64),
                             1e-3, None))

    primary_rows = np.where(primary_mask)[0]
    if len(primary_rows) == 0:
        # Nothing to fit — keep current gains.
        return log_gain.astype(np.float32)

    row_to_tile = np.zeros(R_total, dtype=np.int32)
    for t in range(n_tiles):
        a, b = int(tile_offsets[t]), int(tile_offsets[t+1])
        row_to_tile[a:b] = t
    primary_tiles = row_to_tile[primary_rows]
    src_index64 = src_index.astype(np.int64)

    tile_to_primary_srcs    = {}
    tile_to_primary_log_lum = {}
    for t in np.unique(primary_tiles):
        m_t = primary_tiles == t
        tile_to_primary_srcs[int(t)]    = src_index64[primary_rows[m_t]]
        tile_to_primary_log_lum[int(t)] = log_lum[primary_rows[m_t]]

    K_rows = len(primary_rows)
    A = np.zeros((K_rows, n_sources), dtype=np.float64)
    b = np.zeros(K_rows, dtype=np.float64)
    for k, r in enumerate(primary_rows):
        t      = int(primary_tiles[k])
        srcs_t = tile_to_primary_srcs[t]
        n_in_t = len(srcs_t)
        if n_in_t < 2:
            continue
        s = int(src_index64[r])
        A[k, s] += 1.0
        for s2 in srcs_t:
            A[k, int(s2)] -= 1.0 / n_in_t
        b[k] = float(tile_to_primary_log_lum[t].mean()) \
             - float(log_lum[r])

    delta_log_gain, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Belt-and-braces recenter; the answer is already mean-zero in
    # exact arithmetic but float-roundoff can leave a tiny offset.
    return (delta_log_gain - delta_log_gain.mean()).astype(np.float32)
```

Implementation notes:

- The equations are written in absolute log-luminance terms, so
  `lstsq` returns absolute mean-zero log-gains directly — not a
  delta to add to the current `log_gain`. This is by design: the
  system is well-conditioned even when the current iterate is far
  from optimal, so there's no value in linearising around it.
- A sparse build (`scipy.sparse.coo_matrix` or Rust `nalgebra`'s
  sparse-COO) is the right shape for production: each primary row
  has at most `|P_t| + 1` nonzero columns, and `|P_t|` is bounded by
  the per-tile contributor count. The dense build above is fine for
  Python prototype; a Rust port should use a sparse normal-equations
  solve (`A^T A · g = A^T b`) — `n_sources` is the small dimension,
  so `A^T A` is `[N × N]` and Cholesky on it is `O(N³)`.
- Underdetermined inputs are handled by `lstsq` itself: when `A`
  has fewer rows than columns, or when the partition leaves some
  source's column entirely zero, `np.linalg.lstsq` returns the
  minimum-norm solution and the unidentifiable components stay at
  zero (i.e. those sources keep gain 1.0). This is the right
  behaviour — a source with no inlier rows in any tile has no
  evidence to fit against. The only explicit bail is "no primary
  rows at all" (above), which short-circuits before building an
  empty `A`.
- Floor on `log` argument: `np.log(np.clip(..., 1e-3, None))`. Rows
  whose mean luminance is exactly zero (fully-masked patches) would
  otherwise blow up; the floor parks them at `log(1e-3) ≈ -6.9` and
  the LSQ down-weights them to whatever the rest of the tile says.

### Step 5: Convergence and termination

```python
mask_change     = int((new_primary_mask != primary_mask).sum())
mask_change_cap = int(mask_change_tolerance * R)
if mask_change == 0:
    break
if mask_change < mask_change_cap:
    break
primary_mask = new_primary_mask
```

Three termination conditions, OR'd:

1. **Stable mask**: `mask_change == 0`. Strict convergence.
2. **Negligible mask change**: `mask_change < mask_change_tolerance · R`
   (default `0.05 · R`). Catches the "flutter" case where rows
   right at the threshold flip back and forth between iterations
   without changing the recovered gains in any meaningful way.
3. **Iteration cap**: `outer_iter == max_outer`. Empirically the
   typical bundled-rig run *hits* this cap rather than converging
   via tolerance — final-iteration mask flutter sits at 3–5 % of
   `R` (e.g. 7/197 on `seoul_bull`, 25/636 on `seattle_backyard`)
   even after the median primary MAD has stopped moving (< 1 %
   between iterations 6 and 8). `max_outer = 8` is the operational
   stop; it adds < 0.5 s to the largest dataset and prevents
   pathological inputs from running unbounded.

A damped LSQ update — scale each `log_gain` delta by 0.7 between
iterations — is the natural follow-up if the threshold-edge flutter
becomes a problem downstream. It is not a v1 default; the
prototype demonstrated convergence is monotone-ish (median primary
MAD changes < 1 % between iterations 6 and 8) without it, so the
extra knob isn't worth introducing until a downstream consumer
needs the strict mask-stable property.

### Step 6: Per-tile secondary cluster (post-convergence)

After the alternation terminates, run one more per-tile RANSAC
pass — this time over each tile's *primary-rejected* rows — to
identify the largest coherent disagreement subset. The kernel is
the same `ransac_cluster_for_tile` used in Step 3; only the input
rows change.

```python
secondary_mask = np.zeros(R, dtype=bool)
for t in range(n_tiles):
    a, b = int(tile_offsets[t]), int(tile_offsets[t+1])
    rejected_local = np.where(~primary_mask[a:b])[0]   # [K_rej]
    if len(rejected_local) < min_inliers:
        continue
    sub = ransac_cluster_for_tile(
        row_patch_corrected[a:b][rejected_local],
        valid_patch[a:b][rejected_local],
        threshold=inlier_threshold,
        subset_size=subset_size,
        rng=rng,
        max_subsets=max_subsets,
    )                                                  # [K_rej] bool
    secondary_mask[a + rejected_local[sub]] = True
```

Notes on the post-convergence pass:

- **Same kernel, same threshold.** The runner-up cluster is defined
  identically to the primary — largest agreeing subset under the
  same `inlier_threshold` — just over a smaller candidate row set.
  Using the same kernel keeps the meaning of "agreement" symmetric
  between the two clusters, which is what makes downstream
  validity comparisons (e.g. `n_secondary / n_primary`) coherent.
- **Skip rule.** Tiles whose primary-rejected set has fewer than
  `min_inliers` rows produce an empty secondary cluster (mask stays
  `false` and `tile_secondary_count = 0`); there is nothing to
  cluster.
- **Cost.** One extra per-tile RANSAC pass over a strict subset of
  the rows — bounded above by Step 3's per-iteration cost and
  almost always cheaper because `K_rej < K`. On the bundled
  datasets this is well under 100 ms even on the dense rig.
- **No effect on `log_gain`.** The secondary cluster is not fed
  back into the LSQ; the gain solve has already converged on the
  primary. The output is purely informational for downstream.

## Identifiability and the shift ambiguity

Per-image log-gain has a global one-parameter ambiguity: shifting
every `log_gain[i]` by `+c` shifts every per-tile mean by `+c` too,
and the per-row residual is unchanged. So the algorithm has to
pick one specific answer out of a one-parameter family of equally
good ones. The RANSAC step is invariant under this shift away from
the threshold edge — every row's luminance scales by `exp(c)`,
every hypothesis scales the same way, and inlier flags are
preserved as long as no row's score sits within
`|exp(c) − 1| · score` of the threshold. In practice rows that
close to the boundary are exactly the cases the algorithm has no
strong opinion on, so the LSQ step is the only thing that picks a
specific representative from the family (see Step 4), and the
near-boundary RANSAC flutter is absorbed by the
`mask_change_tolerance` termination criterion.

The picked representative is the **mean-zero** one, achieved
automatically by `lstsq`'s minimum-norm property because the
all-ones direction is exactly in `null(A)`. Mean-zero is preferred
over pinning a "reference source": the reference source could be
one whose contributions are mostly outliers under the partition,
in which case its gain is determined by very few residual rows and
becomes a noisy anchor for everyone else. Mean-zero is symmetric
across sources and well-defined regardless of the partition.

## Numeric stability

- **f32 vs f64.** Per-row luminances stay in f32 throughout RANSAC
  (the threshold is in u8 units, well above f32's resolution). The
  LSQ step uses f64 for `log_lum`, `A`, `b`, and the solve, then
  casts back to f32. This is enough to keep the mean-zero residual
  below 1e-7 on all tested configurations.
- **Floor on log-luminance.** `np.clip(row_lum, 1e-3, None)` before
  `np.log`. Fully-masked rows become `log(1e-3) ≈ -6.9`; the LSQ
  treats them as a noisy outlier and the per-image gain is
  determined by the rest of the primary cluster.
- **Skip rule for tiny tiles.** Tiles with `b - a < min_inliers`
  contribute no LSQ rows and no cluster flags. Their `primary_mask`
  and `secondary_mask` entries stay false; downstream consumers
  know to ignore them.
- **Subset enumeration roundoff.** Subset means are computed in f64;
  the residual comparison is in f32 against the `f32` threshold.
  This prevents one-bit f32 roundoff from flipping cluster verdicts
  at the threshold edge across runs on different SIMD widths.

## Complexity

Per outer iteration, on a stack of `R` rows, `n` tiles, `N` sources,
patch side `s`, `C` channels, average per-tile contributor count
`K = R / n_active_tiles`:

| Step | Cost |
|------|------|
| Step 1 (linearise sub-patches + per-row mean luminance, once at start) | `R · ss² · C` |
| Step 2 (apply gains, per outer iteration) | `R · ss² · C` |
| Step 3 (per-tile RANSAC, primary) | `n · min((K choose m), max_subsets) · K · ss² · C` |
| Step 4 (LSQ build) | `n · K² + N²` |
| Step 4 (LSQ solve, dense) | `O(N³)` |
| Step 5 (mask compare) | `R` |
| Step 6 (per-tile RANSAC, secondary, once post-convergence) | `n · min((K_rej choose m), max_subsets) · K_rej · ss² · C` |

The RANSAC term dominates on dense rigs; the LSQ tail dominates only
once `N` is in the hundreds. With `m = 2`, `max_subsets = 64`,
`ss = 2`, `C = 3`, the per-iteration RANSAC budget is
`n · 64 · K · 12`. On the largest bundled rig (`R ≈ 3 k`, `n ≈ 160`,
`K ≈ 20`) one outer iteration is tens of millions of arithmetic
ops, single-CPU seconds in numpy and tens of milliseconds in Rust.
Doubling `ss` to 4 quadruples the RANSAC inner cost; on the bundled
rigs that still fits inside the wall-clock budget below. End-to-end
wall-clock targets, from the prototype: ≤ 0.2 s for sparse rigs
(R ≈ 200–800), ≤ 2 s for dense rigs (R ≈ 3000) in Python; an order
of magnitude below that in a Rust port (sparse `A^T A` solve in
`nalgebra`, per-tile RANSAC parallelised via `rayon`).

## Empirical metrics

Headline numbers from a Python prototype that implements the
algorithm exactly as specified above (patch-L1 score, γ=1.0,
defaults from the inputs table). Numbers are median per-tile MAD
on per-row mean luminance in u8 units, on multi-contributor tiles,
post colour correction.

**Tested in CI** (bundled image fixtures, drives the PyO3 binding
end-to-end through `tests/`):

| Dataset            | R   | MAD pre | MAD post (all rows) | MAD post (primary) | wall, Python prototype | inter-seed log_gain σ |
|--------------------|-----|---------|----------------------|---------------------|------------------------|-----------------------|
| seoul_bull (17)    | 197 | 5.92    | 4.62 (m=2)           | 1.59                | 0.05 s                 | 0.0000 (exhaustive)   |
| seattle_back. (26) | 636 | 3.90    | 3.22 ± 0.08          | 1.65 ± 0.11         | 0.34 s                 | 0.0076                |

**Informational** (not exercised in CI; demonstrates dense-rig
behaviour from the prototype's `kerrypark` runs):

| Dataset         | R    | MAD pre | MAD post (all rows) | MAD post (primary) | wall, Python prototype | inter-seed log_gain σ |
|-----------------|------|---------|----------------------|---------------------|------------------------|-----------------------|
| kerrypark (32)  | 787  | 6.62    | 5.06 ± 0.01          | 2.19 ± 0.03         | 0.40 s                 | 0.0015                |
| kerrypark (124) | 3067 | 6.89    | 4.46 ± 0.15          | 2.12 ± 0.06         | 2.10 s                 | 0.0164                |

Order-of-magnitude expectations a v1 Rust implementation should
match without surprise (asserted in CI for the two bundled rigs;
for `kerrypark` these are demonstrated by the prototype but not
gated):

- **Primary-cluster MAD reduction ≥ 40 %** vs uncorrected baseline.
  This is the metric a downstream consensus consumer that respects
  the partition cares about (it ignores rows outside the primary).
  Prototype delivered -73 % / -58 % / -67 % / -69 % on the four
  configurations.
- **All-rows MAD reduction ≥ 15 %** vs baseline. Smaller than the
  primary reduction by design: the patch-L1 score correctly rejects
  rows the older scalar-mean score would have admitted, and those
  rejected rows still appear in the all-rows MAD calculation. The
  intended "tight partition / loose all-rows" trade.
- **Inter-seed `log_gain` mean stddev < 0.02** at `m=2` on all
  bundled datasets. Prototype delivered 0.000 / 0.008 / 0.002 /
  0.016 on the four configurations; `m=3` slips on dense rigs
  (0.023 on `kerrypark` 32) — `m=2` is the recommended default.
- **Recovered `log_gain` standard deviation in 0.04–0.14**, with
  `exp(max - min)` ratios in 1.10–1.45 — physically plausible for
  handheld phone capture.
- **Outer-iteration count typically hits `max_outer = 8`** with a
  3–5 % mask flutter rather than strict convergence; median primary
  MAD stops moving by iteration 6.
- **Wall time well under 1 s** for the bundled rigs in pure Python;
  expect a Rust port to be ~10× faster (the integration-test
  budgets in §[Validation plan](#validation-plan) account for this).

## Validation plan

The v1 should ship with the following unit and regression tests.
The intent of each test is described; concrete assertion thresholds
should be set in code to match the empirical metrics above (with
small headroom for flake).

### Unit tests on synthetic stacks

1. **Single-cluster recovery.** Build a stack of `N = 4` synthetic
   sources with known per-source gains `[g_0, g_1, g_2, g_3]` (e.g.
   `[0, 0.1, -0.05, 0.02]`), all observing the same ground-truth
   luminance per tile, with no outliers. **Use an `inlier_threshold`
   loose enough to admit every ground-truth source on iteration 1**
   (e.g. `16.0` u8 units for a ~150-luminance synthetic stack);
   too-tight a threshold rejects some sources before LSQ runs and
   leaves their gains stuck at the initial value, breaking the strict
   0.01 gain-recovery bound below. Run the algorithm. Assert recovered
   `log_gain` matches ground truth (after mean-zero recentering)
   within 0.01 per source, all rows are in the primary cluster, the
   secondary cluster is empty everywhere, and one outer iteration
   suffices.

2. **Two-cluster partition.** Build a stack where each tile has 6
   contributors, 4 of which observe ground-truth luminance `L_t`
   and 2 of which observe `L_t + 50` (a clean second group). Run
   the algorithm. Assert `primary_mask` has exactly the 4
   ground-truth rows in each tile, `secondary_mask` has exactly the
   2 displaced rows, and the recovered gains match the ground-truth
   gains of the primary sources.

3. **Below-threshold tile (`K ≤ m`).** Build a tile with exactly 2
   contributors at `m = 2`. Assert the algorithm uses the
   median-fallback path defined in `ransac_cluster_for_tile`: **rows
   whose patch-L1 distance to the per-pixel median is strictly
   greater than `inlier_threshold` are rejected**, rows within the
   threshold are kept. Note this rule is symmetric on a disagreeing
   pair: when the two rows differ by more than `2·threshold`, both
   sit equidistant from the per-pixel median and *both* are rejected
   (not just one); the test asserts this symmetric-rejection
   behaviour explicitly. Secondary cluster is empty (post-rejection
   set is empty or below `min_inliers`).

4. **Empty / single-contributor tiles.** Build a tile with 0 or 1
   contributor and `min_inliers = 2`. Assert the tile is skipped
   (its rows have `primary_mask = secondary_mask = false`) and the
   LSQ does not crash.

5. **Mean-zero output.** Across all of the above, assert `|mean(
   log_gain)| < 1e-6`.

6. **Shift invariance.** Apply a known constant shift to every
   ground-truth gain (`g_i ← g_i + 0.5`) before generating row
   luminances. Synthesise rows so every per-row residual is well
   inside the threshold (e.g. < 0.5 · `inlier_threshold`) — the
   RANSAC step's invariance to a constant shift in `log_gain` only
   holds *away* from the threshold edge, so a near-edge synthetic
   could legitimately flip cluster verdicts under the shift. Use
   the same loose `inlier_threshold` as test 1 so every source is
   admitted on iteration 1. Assert the recovered `log_gain` is
   bit-identical to the unshifted run (because per-tile residuals
   are unchanged by a constant shift, the LSQ step always returns
   the mean-zero representative of the family, and so the answer
   doesn't depend on which constant the input got shifted by).

7. **Subset-enumeration boundary.** With `max_subsets = 64` and
   `m = 2`, run on tiles with `K = 11, 12, 13`. Assert K=11
   produces seed-independent results (exhaustive: all
   (11 choose 2) = 55 subsets enumerated); K=12 ((12 choose 2) =
   66, samples 64) and K=13 ((13 choose 2) = 78, samples 64)
   produce results that vary by `< 1 %` of `primary_mask` Hamming
   distance across 8 different seeds.

8. **Secondary cluster recovery.** Build a tile with 8
   contributors that split cleanly into a 5-row group at `L_t` and a
   3-row group at `L_t + 60` (well above threshold). Assert the
   primary cluster recovers exactly the 5-row group, the secondary
   cluster recovers exactly the 3-row group, and
   `tile_secondary_lum_mad` is within noise tolerance of the
   displaced group's MAD. With the
   second group reduced to 1 row, assert the secondary cluster is
   empty (`min_inliers = 2` skip).

9. **Patch-aware scoring distinguishes spatial pattern.** Build a
   tile with K=4 contributors at gain 1.0 (no gain correction
   needed): two rows with the checker patch
   `[[100, 200], [200, 100]]` (identical) and two rows with constant
   patches `[[150, 150], [150, 150]]` and `[[154, 154], [154, 154]]`,
   defined here in scoring-sub-patch coordinates (`ss × ss` =
   `2 × 2`). All four rows have mean luminance ≈ 150, so under a
   scalar mean-luminance scorer they would be mutual inliers under
   `inlier_threshold = 8.0` and the algorithm would land all four
   in the primary cluster.

   **Synthesizer note.** The level-0 patch fed to the algorithm is
   `target_patch_size · 2^L` per side (e.g. `16 × 16` when reading
   level 2 to land at `target_patch_size = 4`); the pyramid build
   block-mean-pools level 0 down to `target_patch_size`. Naively
   tiling the `2 × 2` checker pattern across the `16 × 16` level-0
   patch will collapse the spatial structure under the block-mean
   and *every* row will look constant at the scoring level — the
   test silently false-passes. Construct the level-0 input *so
   block-mean-pool to `target_patch_size` recovers the intended
   `ss × ss` pattern*: the central `ss × ss` super-block region of
   the level-0 input should partition into `(target_patch_size / ss)
   × (target_patch_size / ss)` cells whose means equal the desired
   sub-patch entries, and the surrounding annulus can be any value
   (it falls outside the central crop).

   With the patch-L1 score and `scoring_patch_size = 2`, assert the
   primary cluster is exactly the two checker rows (mean inlier
   residual 0, beating the constant pair's residual 2 on tie-break)
   and the secondary cluster is exactly the two constant-patch
   rows. This is the regression test for the patch-vs-scalar
   upgrade.

### Integration tests on the bundled datasets

These run via the pytest harness in `tests/`, exercising the PyO3
binding end-to-end against the `seoul_bull_sculpture` (17 imgs) and
`seattle_backyard` (26 imgs) image fixtures. `kerrypark` is not a
bundled dataset and is not exercised in CI; the §[Empirical
metrics](#empirical-metrics) table cites its prototype numbers
as informational evidence of dense-rig behaviour.

10. **All-rows MAD reduction.** Assert post-correction median
    per-tile MAD on multi-contributor tiles is at least **15 %**
    lower than the pre-correction baseline on `seoul_bull` and on
    `seattle_backyard` at `subset_size = 2`. Prototype delivered
    -22 % / -17 %.

11. **Primary-cluster MAD reduction.** Assert post-correction
    primary-cluster median MAD is at least **50 %** lower on
    `seoul_bull` and at least **40 %** lower on `seattle_backyard`.
    prototype delivered -73 % / -58 %.

12. **Inter-seed gain stability.** Run with 8 seeds at `m=2`.
    Assert per-source `log_gain` standard deviation across seeds
    is < 0.02 on both datasets (the v1 pass criterion). Prototype
    delivered 0.000 / 0.008 — `seoul_bull` exhaustively enumerates
    so the result is byte-identical across seeds.

13. **Wall-clock smoke test.** Run on each dataset via the PyO3
    binding; assert end-to-end algorithm time (excluding stack
    construction) < 1 s for `seoul_bull` and < 2 s for
    `seattle_backyard`. The Python prototype delivered 0.05 s and
    0.34 s respectively; the Rust port should be substantially
    faster, so these bounds are deliberately generous to avoid CI
    flake.

### Property tests

14. **Permutation invariance of source order.** Permute the
    `src_index` indexing. Assert the output `log_gain` is the same
    permutation of the unpermuted output (modulo float roundoff).

15. **Permutation invariance of row order within a tile.** For a
    single tile, shuffle the rows within `tile_offsets[t]:tile_
    offsets[t+1]` and assert that both `primary_mask` and
    `secondary_mask` after the algorithm name the same set of rows
    (with the permutation applied).

16. **Idempotence at the fixed point.** Run the algorithm to
    convergence, then run *one more* outer iteration starting from
    the converged state. Assert `primary_mask` does not change and
    `log_gain` changes by < 1e-6 per source.

### Failure-mode tests

17. **All rows fully masked.** Construct a stack where every row's
    `valid` mask is zero. Assert the algorithm returns
    `log_gain = 0`, both `primary_mask` and `secondary_mask` all
    false (or the underdetermined fallback path), and does not
    raise.

18. **Single-tile underdetermined.** Construct a stack with
    `n_sources = 10` and only one multi-contributor tile (so the
    LSQ has fewer rows than sources). Assert the algorithm returns
    without crashing, that the involved sources end up with
    sensible mean-zero gains, and that sources
    with no primary rows are returned with `log_gain[i] = 0` (the
    minimum-norm answer for an unidentifiable component).

## Open issues for the v1

1. **Neighbour-consistency fusion using the secondary cluster.**
   The prototype evaluation resolved (a) and (b) of the original
   secondary-cluster question: 67 % of active tiles on the dense
   `kerrypark` 124-img rig have `secondary_count ≥ 2`, the median
   secondary/primary size ratio is 0.29 on dense rigs, and the
   per-tile galleries show visually coherent "alternate scene
   interpretation" rather than noise. What remains open is (c):
   whether a downstream neighbour-consistency rule that promotes
   the secondary to chosen — when it agrees with surrounding tiles'
   choices — measurably improves the consensus output. This depends
   on the downstream consensus consumer landing first and so cannot
   be resolved at this layer; the secondary-mask wire format is
   designed to support it without renegotiation.

## Deliverables

The v1 of this spec ships as a tightly-coupled trio. Python is the
binding surface and the test driver, not a parallel implementation
— there is one algorithm, written once, in Rust.

1. **Rust core** in `crates/sfmtool-core/`. Implements the algorithm
   on `PerSphericalTileSourceStack` and exposes the single entry
   point sketched below. All algorithmic decisions — score function,
   subset enumeration, LSQ formulation, secondary pass — live here
   in deterministic, allocation-light Rust.

2. **PyO3 bindings** in `crates/sfmtool-py/`, exposed through the
   existing `sfmtool._sfmtool` extension module. The binding accepts
   the same `PerSphericalTileSourceStack` Python class as other
   rust-exposed pipeline stages and returns NumPy-friendly outputs.
   It does not expose any algorithmic surface that isn't in the
   Rust API; it is a thin numpy-marshalling layer.

3. **Test suite**, split intentionally between two tiers (covering
   the validation plan above):
   - **Rust unit tests** in `crates/sfmtool-core/tests/` exercise
     synthetic stacks (validation-plan tests 1–9, 14–18). These are
     fast, deterministic, and have no external dependencies. They
     are the primary verification surface — they own the algorithmic
     correctness contract and cover code paths the integration tests
     can't reach (e.g. the `K ≤ m` median fallback, the secondary
     cluster on a synthesised disagreeing pair).
   - **Python integration tests** in `tests/` drive the PyO3 binding
     end-to-end against the bundled `seoul_bull_sculpture` and
     `seattle_backyard` image fixtures (validation-plan tests
     10–13). These catch what synthetic stacks can't: floating-point
     accumulation order on real RGB content with non-uniform
     validity, and any drift between the spec's empirical-metrics
     promises and the implemented behaviour.

After Rust changes that touch the binding surface, rebuild via
`pixi run maturin develop --release` before running Python tests
(per repo convention; the editable Python install does not auto-
rebuild the native extension).

## Rust API sketch

```rust
pub struct RansacPhotometricParams {
    pub inlier_threshold:       f32,    // default 8.0
    pub gamma:                  f32,    // default 1.0
    pub target_patch_size:      u32,    // default 4 (power of two; must match a stack pyramid level)
    pub scoring_patch_size:     u32,    // default 2 (even, 2 ≤ ss ≤ target_patch_size)
    pub subset_size:            u32,    // default 2
    pub max_subsets_per_tile:   u32,    // default 64
    pub min_inliers:            u32,    // default 2
    pub max_outer_iters:        u32,    // default 8
    pub mask_change_tolerance:  f32,    // default 0.05 (fraction of R)
    pub saturation_threshold:   u8,     // default 254 (255 to disable)
    pub seed:                   u64,    // default 0
}

pub struct RansacPhotometricOutput {
    pub log_gain:             Vec<f32>,    // [n_sources]
    pub primary_mask:         Vec<bool>,   // [R]
    pub secondary_mask:       Vec<bool>,   // [R]
    pub tile_primary_count:   Vec<i32>,    // [n_tiles]
    pub tile_secondary_count: Vec<i32>,    // [n_tiles]
    pub tile_primary_lum_mad:   Vec<f32>,    // [n_tiles], NaN where skipped
    pub tile_secondary_lum_mad: Vec<f32>,    // [n_tiles], NaN where skipped
    pub outer_iters:          u32,
    pub mask_change_history:  Vec<u32>,    // length outer_iters
}

pub fn refine_photometric_ransac(
    stack: &PerSphericalTileSourceStack<f32>,
    params: &RansacPhotometricParams,
) -> Result<RansacPhotometricOutput, RansacPhotometricError>;
```

The Rust API takes an `f32`-pixel stack directly (the `gamma`
exponentiation in Step 1 needs a floating-point representation;
callers with a `<u8>` stack should rebuild as `<f32>` first or use
a thin adapter that `from_u8`-converts level by level). The function
picks the pyramid level whose patch side equals `target_patch_size`
exactly, returning `RansacPhotometricError::NoMatchingLevel` if no
such level exists; reads patches and valid masks via
`stack.level_patches(L)` / `stack.level_valid(L)`; constructs the
flat row-major layout internally; and runs the algorithm.

## Python binding sketch

The PyO3 binding lives in `crates/sfmtool-py/` and exposes the
algorithm as a single function on the existing
`PerSphericalTileSourceStack` Python class. Parameter defaults
mirror the Rust struct above; outputs are returned as a NumPy-
friendly object whose fields are owned NumPy arrays.

```python
from sfmtool import _sfmtool

# stack is a PerSphericalTileSourceStack constructed by the bundled
# pipeline (e.g. from a workspace via the existing rust-exposed
# stack-construction stage).
out = _sfmtool.refine_photometric_ransac(
    stack,
    inlier_threshold=8.0,
    gamma=1.0,
    target_patch_size=4,
    scoring_patch_size=2,
    subset_size=2,
    max_subsets_per_tile=64,
    min_inliers=2,
    max_outer_iters=8,
    mask_change_tolerance=0.05,
    saturation_threshold=254,
    seed=0,
)
# out.log_gain             : np.ndarray[float32], shape [n_sources]
# out.primary_mask         : np.ndarray[bool],    shape [R]
# out.secondary_mask       : np.ndarray[bool],    shape [R]
# out.tile_primary_count   : np.ndarray[int32],   shape [n_tiles]
# out.tile_secondary_count : np.ndarray[int32],   shape [n_tiles]
# out.tile_primary_lum_mad   : np.ndarray[float32], shape [n_tiles]
# out.tile_secondary_lum_mad : np.ndarray[float32], shape [n_tiles]
# out.outer_iters          : int
# out.mask_change_history  : np.ndarray[uint32],  shape [outer_iters]
```

Integration tests (validation plan 10–13) call this function
directly from pytest fixtures; the bindings are part of the v1
deliverable, not an after-the-fact addition.
