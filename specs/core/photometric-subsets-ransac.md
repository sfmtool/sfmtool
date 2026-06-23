# Photometric refinement via per-tile RANSAC subset partition

**Status:** Implemented — `crates/sfmtool-core/src/spherical/photometric_ransac.rs`
(`refine_photometric_ransac`, `RansacPhotometricParams`), with PyO3 bindings
(`py_photometric_ransac.rs`). Consumed in production by the tile-batched
consensus-atlas pipeline ([tile-batched-consensus-atlas.md](tile-batched-consensus-atlas.md)
→ `spherical/consensus_atlas.rs`), which `sfm panorama` drives via `rig/panorama.py`.
Originally a standalone draft; promoted to `specs/core/` once the production
pipeline consumed its outputs, per its own graduation trigger.

[`PerSphericalTileSourceStack`]: per-spherical-tile-source-stack.md
[`SphericalTileRig`]: spherical-tiles-rig.md

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

The end goal is a per-tile estimate of the "correct" pixel value in
each direction. This is an estimation problem solved by *finding a
large subset of contributing sources whose patches agree* on what
the scene looks like in that direction, and treating that subset's
consensus as the truth. The disagreeing minority is itself
informative: for a tile that should show distant radiance from
infinity, some sources may see that radiance while others have a
near-object obscuring the line of sight; the consensus picks out
the agreeing group, and the disagreeing rows get tagged for the
downstream consensus rule to handle as occluders / parallax /
sky-leak / mask edges.

Per-source colour correction (gain, gamma refit, vignetting,
white-balance) is **out of scope** for v1: the algorithm assesses
agreement on the raw linearised pixel values, and downstream
compositing is responsible for any colour harmonisation. A future
revision may reintroduce a gain estimate driven by validated need.

The outputs are recovered together:

- **Per-tile primary cluster** — the largest agreeing subset. Its
  consensus is the algorithm's best estimate of the per-tile
  "correct" value.
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
bundled datasets — so per-tile work is cheap.

## Design rationale

The algorithm is a per-tile RANSAC: sample small subsets, score
each by how many contributors agree with the subset's hypothesised
consensus patch under a fixed inlier threshold, keep the
highest-scoring inlier set. The "largest agreeing subset" is a
well-defined object (defined by score, not by an arbitrary cluster
index), so the binary inlier mask has no permutation ambiguity.

The secondary cluster is computed by re-running the same RANSAC
kernel over each tile's primary-rejected rows. Same threshold, same
scoring, no shared state with the primary pass.

Per-tile RANSAC is embarrassingly parallel — each tile depends only
on its own rows.

## Inputs

The algorithm consumes a flat row-major view over the stack at the
chosen pyramid level. The bindings extract this from a
[`PerSphericalTileSourceStack`] under the hood.

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
| `tile_offsets` | `[n_tiles + 1]` | i32 | CSR-style row offsets: rows for tile `t` are `[tile_offsets[t], tile_offsets[t+1])`. Implies rows are sorted by `tile_index`. |
| `n_tiles` | scalar | i32 | tile count `n` |

Plus tuning knobs:

| Knob | Default | Meaning |
|------|---------|---------|
| `inlier_threshold` | `8.0` u8 lum units | residual threshold separating inliers from outliers, applied to the validity-weighted L1-mean of `(row_patch − consensus_patch)` over the central `ss × ss` sub-patch in linearised u8 luminance units. See Step 2 |
| `gamma` | `1.0` | decoding gamma applied to source patches (`linear = 255·(patch/255)^gamma`) before per-row mean luminance. Default `1.0` keeps the algorithm in sRGB-encoded u8 space, where a fixed `inlier_threshold = 8.0` u8 units is roughly perceptually-uniform — uniform discrimination across the luminance range is what RANSAC's hard threshold actually wants. Override with a known camera gamma when downstream demands true linear-light behaviour |
| `target_patch_size` | `4` | side length of patches read from the pyramid; the algorithm picks the level whose patch side equals this **exactly** and errors if no such level exists. Must be a power of two between 2 and the stack's `base_patch_size` |
| `scoring_patch_size` | `2` | side length of the centred sub-patch used by the scorer; an exact integer crop of the read patch (no resampling). Must satisfy `2 ≤ scoring_patch_size ≤ target_patch_size` and `(target_patch_size − scoring_patch_size)` even, so the central crop is symmetric. Must be even |
| `subset_size` | `2` | RANSAC minimal subset size `m`. `m=2` is the recommended default: more tiles fall under the `max_subsets_per_tile = 64` exhaustive-enumeration cap (the largest K still enumerated exhaustively is K=11 at m=2 vs K=8 at m=3) |
| `max_subsets_per_tile` | `64` | cap on candidate subsets per tile when `C(K, m) > max_subsets_per_tile` |
| `min_inliers` | `2` | tiles with fewer contributors than this are skipped |
| `seed` | `0` | RNG seed for reproducible subset sampling. Per-tile RNG state is derived from `(seed, tile_index)` so per-tile RANSAC can be parallelised (e.g. via `rayon`) without making the result depend on execution order |
| `saturation_threshold` | `254` | per-channel u8 cutoff above which a pixel is treated as saturated; pixels at or above this value have their entry in the working validity mask zeroed at the start of Step 1. The upstream `PatchLevel::valid` only encodes source-bounds visibility, so saturation masking is the algorithm's own responsibility. Set to `255` to disable; `254` is the default because sensor clipping at u8 is ragged and a 1-unit margin catches most clipped highlights |

Sources are assumed to share a single fixed decoding gamma that the
algorithm does not refit. Patches are linearised by
`linear = 255 · (patch/255)^gamma` before Step 1's per-row mean
luminance is computed; at `gamma = 1.0` this is a no-op. Subsequent
steps and `inlier_threshold` are interpreted on the linearised
representation.

## Outputs

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `primary_mask` | `[R]` | bool | per-row primary-cluster flag — `true` ⇔ row is in its tile's largest agreeing subset |
| `secondary_mask` | `[R]` | bool | per-row secondary-cluster flag — `true` ⇔ row is in the largest agreeing subset within the primary's complement; `false` for primary rows and for tiles with no coherent runner-up |
| `tile_primary_count` | `[n_tiles]` | i32 | size of each tile's primary cluster (0 for skipped tiles) |
| `tile_secondary_count` | `[n_tiles]` | i32 | size of each tile's secondary cluster (0 where no runner-up) |
| `tile_primary_lum_mad` | `[n_tiles]` | f32 | Median Absolute Deviation `median_i(|x_i − median(x))|` of `x_i = row_lum[i]` over each tile's primary cluster, in u8 luminance units (NaN where `tile_primary_count < min_inliers`). Textbook robust scale estimator; `1.4826 · MAD ≈ σ` for normal data. Note this is computed on the per-row scalar mean luminance, **not** on the patch-L1 residual the RANSAC scorer uses — it is a downstream-friendly summary statistic, not the algorithm's loss. A tile in which patches with the same mean luminance but different spatial structure (e.g. checker vs constant) get split between primary and secondary will have a small `lum_mad` on both clusters even though the patch-L1 score that drove the split was discriminating |
| `tile_secondary_lum_mad` | `[n_tiles]` | f32 | Same MAD definition over each tile's secondary cluster (NaN where `tile_secondary_count < min_inliers`); same caveat as `tile_primary_lum_mad` |

`primary_mask` names which rows participate in each tile's
consensus and can be consumed directly by the downstream consensus
rule without a disambiguation step. `secondary_mask` together with
the `tile_*_count` fields is the per-tile validity signal: high
confidence when the primary dominates, ambiguous when the two
clusters are comparable, low confidence when both are small.

## Detailed steps

### Step 1: Per-row scoring patch and mean luminance

For each row `r` with patch `P[r] ∈ R^{s × s × C}` and valid mask
`V[r] ∈ R^{s × s}`, extract the central `ss × ss` sub-patch
(`ss = scoring_patch_size`), zero the validity entries for pixels
saturated at any channel, linearise by gamma, and compute a
validity-weighted mean luminance.

```
for r in 0..R:
    centre_p[r]   = central ss×ss crop of P[r]            # [ss, ss, C]
    centre_v[r]   = central ss×ss crop of V[r]            # [ss, ss]
    centre_v[r] &= 1 - (centre_p[r].max(axis=-1) >= sat)  # saturation mask
    row_patch[r]  = 255 * (centre_p[r] / 255) ** gamma    # linearised
    row_lum[r]    = sum(row_patch[r] * centre_v[r, ..., None])
                   / max(1, sum(centre_v[r]) * C)         # weighted mean lum
```

`row_lum` is a downstream diagnostic (used by the per-tile
luminance MAD reporter); the RANSAC scorer in Step 2 operates on
`row_patch` directly.

### Step 2: Per-tile primary RANSAC

Each tile is processed independently (in parallel via rayon). For
tile `t` with rows `R_t = [tile_offsets[t], tile_offsets[t + 1])`
and `K = |R_t|`:

```
def ransac_cluster_for_tile(row_patch, valid_patch, K, ss, C,
                            threshold, m, max_subsets, rng):
    """
    Returns boolean [K] cluster-membership mask (the largest
    agreeing subset, or empty if K < min_inliers).
    """
    if K < min_inliers:
        return [false] * K

    # Below-sampling fallback: median + threshold.
    if K <= m:
        med = per_pixel_median(row_patch)
        return [patch_l1_score(row, valid, med) <= threshold
                for row, valid in zip(row_patch, valid_patch)]

    # Build candidate list: exhaustive when feasible, sampled otherwise.
    n_combos = comb(K, m)
    if n_combos <= max_subsets:
        candidates = enumerate_combinations(K, m)
    else:
        candidates = sample_combinations(rng, K, m, max_subsets)

    best_count, best_score, best_inliers = -1, +inf, None
    for idx_set in candidates:
        hyp = consensus_patch(row_patch, valid_patch, idx_set)
        scores = [patch_l1_score(row, valid, hyp)
                  for row, valid in zip(row_patch, valid_patch)]
        inliers = [s <= threshold for s in scores]
        n_in = sum(inliers)
        mean_in_score = mean(s for s, in_ in zip(scores, inliers) if in_)
        # Tie-break on lower mean residual, then on best inlier count.
        if n_in > best_count or (n_in == best_count and mean_in_score < best_score):
            best_count, best_score, best_inliers = n_in, mean_in_score, inliers
    return best_inliers
```

Key decisions:

1. **Validity-weighted L1 on patches.** The thing being fit by each
   subset is the consensus patch — a `[ss, ss, C]` average over
   the subset's rows weighted by per-pixel validity. The scorer is
   the validity-weighted mean of per-pixel L1 residuals between a
   row's patch and the consensus patch, divided by channel count
   so the threshold reads in u8 luminance units.

2. **Below-sampling fallback (`K ≤ m`).** With `m = 2` and `K = 2`,
   exhaustive enumeration would only sample one subset (the only
   pair), and that pair's consensus would always include both rows
   as inliers — even when they disagree by an arbitrary amount.
   Falling back to the median patch and thresholding instead
   produces the symmetric outcome: both rows in if they agree, both
   out if they don't.

3. **Score-defined cluster identity.** RANSAC returns "the largest
   agreeing subset", a function of the inputs alone. There is no
   per-tile cluster ID and therefore no per-tile permutation
   ambiguity to disambiguate.

4. **Skipped tiles.** A tile with fewer contributors than
   `min_inliers` returns an all-false mask and reports
   `tile_primary_count = 0`. No fallback heuristics — downstream
   sees the tile as "no consensus available" and decides whether to
   inpaint, defer, or warn.

### Step 3: Per-tile secondary RANSAC

After the primary pass, run the same `ransac_cluster_for_tile`
kernel over each tile's primary-rejected rows. Tiles whose
primary-rejected set has fewer than `min_inliers` rows produce an
empty secondary cluster (mask stays `false` and
`tile_secondary_count = 0`).

```
secondary_mask = [false] * R
for t in 0..n_tiles:
    rejected_local = [i for i, r in enumerate(R_t) if not primary_mask[r]]
    if len(rejected_local) < min_inliers: continue
    sub_mask = ransac_cluster_for_tile(
        row_patch[R_t][rejected_local],
        centre_v[R_t][rejected_local],
        len(rejected_local), ss, C,
        inlier_threshold, m, max_subsets, rng)
    for i, m in zip(rejected_local, sub_mask):
        if m: secondary_mask[R_t[i]] = true
```

The runner-up cluster is defined identically to the primary —
largest agreeing subset under the same threshold and the same
patch-L1 scorer — so `n_primary` and `n_secondary` are directly
comparable.

### Step 4: Per-tile counts and luminance MADs

Aggregate per-tile metrics in one pass:

```
for t in 0..n_tiles:
    R_t = rows in tile t
    tile_primary_count[t]    = count(primary_mask[r] for r in R_t)
    tile_secondary_count[t]  = count(secondary_mask[r] for r in R_t)
    tile_primary_lum_mad[t]  = MAD(row_lum[r] for r in R_t if primary_mask[r])
                               if tile_primary_count[t] >= min_inliers else NaN
    tile_secondary_lum_mad[t] = ... (analogous)
```

Where `MAD(x) = median(|x - median(x)|)`.

## Implementation notes

### Numerics

- `patch_l1_score` accumulates in f64 and casts to f32 only at the
  end. The L1-mean is bounded by 255 in the worst case, so f32
  precision is sufficient for the residual comparison itself.
- A row whose working validity is identically zero (every pixel
  masked) returns `+inf` from `patch_l1_score`. Without this guard
  the formula `num / max(denom, 1.0)` would return 0 (because
  `num = 0` too), making fully-masked rows automatically inliers.

### Parallelism

- Per-tile RANSAC parallelises trivially with rayon. Per-tile RNG
  state is derived from `(seed, tile_index)` so the result is
  independent of thread scheduling.
- The secondary pass is similarly per-tile-parallel.
- Step 1 is per-row independent and amortises across `R`; level-0
  patches and validity buffers are read sequentially.

### Complexity

For per-tile contributor count `K`, the per-tile RANSAC cost is
roughly `min(C(K, m), max_subsets) · K · ss² · C`. With `m = 2` and
`max_subsets = 64`, the per-tile cost is bounded.

Step 1 is `R · ss² · C` with a small linearisation constant.

## Validation plan

1. **Two-cluster partition recovery.** Build a single-tile stack
   with 4 rows at `L = 100` and 2 rows at `L = 160` (well above
   `inlier_threshold = 8`). Assert `primary_mask` is the 4-row
   group, `secondary_mask` is the 2-row group, and the two are
   disjoint.

2. **Below-threshold tile (K ≤ m).** With `K = m = 2` and a
   between-row diff `> 2 · threshold`, both rows should be rejected
   from the primary cluster. With diff `< 2 · threshold`, both
   should be in.

3. **Empty / single-contributor tiles.** Tiles with `K < min_inliers`
   produce all-false masks and `tile_*_count = 0` without crashing.

4. **Subset-enumeration boundary determinism.** With `K` chosen so
   `C(K, m) <= max_subsets`, the primary mask must be byte-identical
   across all seeds (exhaustive enumeration). With `K` just above the
   boundary, the inter-seed Hamming distance must be small (<1 % of K).

5. **Secondary cluster recovery.** Like (1) but also assert that
   when the rejected set has `< min_inliers` members, the secondary
   cluster is empty.

6. **Patch-aware scoring.** A tile with two checker patches and two
   constant patches at the same scalar mean must split into the
   checker pair (primary) and the constant pair (secondary), even
   though their per-row mean luminances are identical.

7. **Permutation invariance of row order within a tile.** Reorder
   a tile's contributors and assert the cluster memberships rename
   consistently.

8. **Fully-masked input.** Setting `valid` to all zeros produces
   no panic and well-defined (NaN) per-tile MADs.

9. **End-to-end on `seoul_bull_sculpture` (17 imgs).** Build the
   stack from a real reconstruction, run the algorithm, and assert
   the per-tile primary luminance MAD is strictly lower than the
   all-rows per-tile luminance MAD (the cluster split must
   compress the within-tile spread).

10. **Wall-clock budget on the 17-image stack.** Refinement should
    complete in well under 1 s so it can run interactively.
