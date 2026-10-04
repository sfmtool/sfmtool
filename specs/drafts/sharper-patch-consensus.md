# Sharper Patch Consensus

**Status:** Draft. Decided:
- the consensus and the stored patch bitmap are weighted by each view's resolution in the patch grid (its zoom) and by its sharpness (its ZNCC self-similarity radius), not by agreement with the mean alone;
- the self-similarity radius the weights read is the reading of exactly the `R×R` tile, with no ring of pixels from outside it (Part 1). That change is a prerequisite and ships first.

Not decided: the functional forms of the weights, the blur used for matched-bandwidth scoring, and the order the consumers adopt it in. See [Open questions](#open-questions).

Amends:
- [core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md): the bench reading
- [core/patch/patch-keypoint-localization.md](../core/patch/patch-keypoint-localization.md): the member gate and the congealing consensus
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md): the member gate
- [core/patch/keypoint-subpixel-refinement.md](../core/patch/keypoint-subpixel-refinement.md): the representative fuse
- [core/patch/patch-normal-refinement.md](../core/patch/patch-normal-refinement.md): the weighted consensus
- [core/bench/editable-track.md](../core/bench/editable-track.md): the bench readings

## Purpose

A point in a patch-based reconstruction carries a small square bitmap of the surface around it. Every photograph that sees the point is resampled into that square and compared against a template built from the other views. The pipeline builds the template, and the stored bitmap, as a robust weighted mean of the views. That mean is blurrier than the best views:
- the far views carry less detail;
- some photographs are out of focus;
- the views do not line up to the last fraction of a pixel.

The sharpest views then correlate worst with it, because the detail they carry has nothing in the template to match. This draft proposes building the template from the views that hold the most detail, judged by two per-view numbers the code already computes:
- how many photograph pixels each patch-grid pixel covers (the zoom);
- how far the view's own tile can slide over itself and still match (the ZNCC self-similarity radius).

It also proposes scoring each view against the template at a resolution the view can actually show. A sharp template then does not penalise a far or blurry view for detail it could never have had.

## The problem, measured

Both cases are tracks on the bench of a local ground-truth candidate for the checked-in `dino_dog_toy` images (`dino_dog_toy_ground_truth_candidate_v001.sfmr`, 85 images at 2040×1536, patch resolution `R = 24`). They were read with the Track View's *Zoom* column and the reach of the self-similarity contour ([core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md) § "The contour and its reach").

**A wood-grain patch sized to about 1× at its closest view.** The patch has half-extent 0.053, with its v axis turned along the grain.
- Its five original views read zoom 0.59/0.92× to 0.92/1.3×.
- Each locks to 0.1–0.2 grid px across the grain and 0.85–1.4 grid px along it.
- A geometry search added nine views spanning zoom 0.47/0.78× to 1.6/2.5×.
- Fit then moved some observations along the grain and lowered their ZNCC.

That is the congealing drift the template's blur along the grain allows.

**A larger patch over the same texture.** The patch has half-extent 0.120, with 25 views, all in. Its largest zoom is 0.76/1.6×. Twenty-two of the views are below 0.6×, so the mip sampler reads them from pyramid levels 1–2. Among the views read at level 2 (zoom 0.24–0.32), the ones whose photographs are sharp correlate worst with the consensus:

| Image | Zoom | Self-similarity radius (grid px) | Across / along grain | ZNCC |
|---|---|---|---|---|
| 57 | 0.19/0.31× | 0.39 | 0.18 / 0.39 | 0.726 |
| 30 | 0.23/0.35× | 0.40 | 0.12 / 0.40 | 0.785 |
| 19 | 0.26/0.35× | 0.36 | 0.16 / 0.36 | 0.795 |
| 37 | 0.26/0.33× | 0.82 | 0.27 / 0.82 | 0.903 |
| 38 | 0.28/0.31× | 1.38 | 0.51 / 1.38 | 0.860 |
| 39 | 0.28/0.31× | 1.31 | 0.42 / 1.31 | 0.810 |

- **Why the comparison is fair.** Zoom and mip level are the same across these rows, so their difference in radius comes from the photographs.
- **Images 38 and 39 are out of focus.** Their radius is three times the sharp views'. The difference is largest across the grain, where the detail is.
- **The blurry views still score best.** They correlate as well as or better than the sharp ones. The level-1 views show the same pattern: images 21, 73 and 75 have radius 0.32–0.43 and ZNCC 0.67–0.73.
- **The whole track shows only a weak trend.** Over all 25 views, ZNCC has a Spearman rank correlation of −0.19 with mean zoom and +0.09 with the radius. Other factors (patch planarity, lighting, misregistration) are of similar size. The effect is clear only once zoom is held fixed.

### Why the mean is blurry

The template is the Tukey/MAD IRLS weighted mean of z-normalized cores (`irls_view_weights`, `weighted_unit_template_into` and `PatchViewStack::fuse` in [normal_refine/](../../crates/sfmtool-core/src/patch/normal_refine/)). Five effects combine:

1. **Misregistration.** A mean of views offset by sub-pixel amounts is the surface convolved with the spread of those offsets. On a directional texture the spread is largest along the direction the views can slide.
2. **Sampling.** `BilinearMip` reads each texel from pyramid level `round(log2 σ_major)`.
   - Each level is a 2×2 box average.
   - Rounding to the nearest level can pick one up to √2 too coarse.
   - The level follows the most compressed axis, so the other axis of an oblique view is blurred further.
   - A view at zoom 0.3 contributes a tile that is low-pass at about 3 photograph pixels before any averaging.
3. **Photographs out of focus or with motion blur**, which enter the mean at full weight.
4. **The reweighting favours blur.** IRLS weights each view by its residual against the mean. A blurry view sits close to a blurry mean and gets a small residual. A sharp view carries detail the mean lacks, so it gets a larger residual and a smaller weight. Each reweight moves the template further toward the blurry views.
5. **Normalized correlation charges for unmatched detail.** Detail in a view that the template lacks adds to that view's variance but not to its covariance with the template. So a sharp, well-aligned view reads a lower ZNCC than a blurry one.

## Part 1 (prerequisite): self-similarity of exactly the R×R tile

**Today.** The bench and both member gates read the ZNCC self-similarity radius on a tile grown by `(R + 2r)/R` and rendered at `R + 2r`. The ring of `r` pixels around the `R×R` core supplies the windows the shifted template reads:
- the bench: `patch_tile_readings` in [bench/evaluate.rs](../../crates/sfmtool-core/src/bench/evaluate.rs);
- the localizer's member gate: `member_self_similarity_radius` in [keypoint_localize.rs](../../crates/sfmtool-core/src/patch/keypoint_localize.rs);
- cluster refinement's member gate: `sample_member_self_similarity_tile` in [cluster_refine/mod.rs](../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs).

The stored-bitmap culls already read an exact `R×R` bitmap with the overlap reading (`embed-patches --max-zncc-self-similarity-radius`, `xform --filter-by-zncc-self-similarity-radius`).

**Change.** All three read the overlap reading of exactly the `R×R` tile:
- The bench reads the tile it renders for its ZNCC readings. That is the same grid as the stored bitmap.
- The localizer's gate reads the core of the tile congealing already rendered for that view.
- Cluster refinement's gate reads the member's own `R×R` grid.

At each shift the overlap reading correlates only the samples that both windows hold. The radius is then the self-similarity *of the bitmap*: it depends on no pixel outside it, and it is the same number whichever of the bench, the gates and the culls computes it.

**Consequences:**
- **One render fewer.** Each reading no longer needs a second render of the wider tile, so self-similarity costs only its shift search: `(2r + 1)²` windowed correlations over the template, about 85k multiply-adds at `R = 24`, `r = 3` with three channels, for the whole tile.
- **The reach gets simpler.** Its conversions no longer have to use the half-extent from before the ring growth, because there is no growth. The anchored placement at `R` is the tile.
- **The parts.** The middle square and the centre cell read exactly what they read today, since their shifted windows stay inside the tile. The edge and corner cells lose the rows and columns past the tile, as the overlap reading already specifies.
- **The default bar.** The overlap reading measured slightly shorter than the ringed one on embed-patches runs over seoul_bull and kerry_park:
  - correlation 0.98–0.99, mean −0.05 grid px;
  - about 3% of verdicts at 2.5 differed, nearly all ringed-fails/overlap-passes.

  Those figures predate the radius reading the outer ring of shifts at their own distance, which removed the main source of disagreement. The 2.5 bar (bench, localizer and cluster refinement defaults) is re-measured on both datasets, and its new value is stated with the measurement.

Part 1 is a behaviour change on its own. It ships and is re-measured before Parts 2–4 depend on it.

## Part 2: what each view can contribute

Each view `v` gets two numbers, read on the tile the consumer already renders.

### The view's footprint, from its zoom

The Jacobian `J_v` of the patch grid's map into the photograph at the patch centre, at the patch resolution `R` (`patch_grid_jacobian` in [camera/warp_map.rs](../../crates/sfmtool-core/src/camera/warp_map.rs)), has singular values `σ_major ≥ σ_minor`. These are photograph pixels per grid pixel. Under `BilinearMip` the sampler reads level `l = round(log2 max(σ_major, 1))`, whose pixels are `2^l` photograph pixels wide. One sample therefore spans, in grid pixels along each singular direction:

```
φ_a = max(2^l, 1) / σ_a,   a ∈ {major, minor}
```

With the floor at 1, a view whose tile magnifies the photograph (σ < 1) spans `1/σ` grid px per photograph pixel. A view read at the right level spans about 1 along its major axis, and more along its minor axis when the level was set by the other axis. The view's **footprint** `φ_v = max(φ_major, φ_minor, 1)` is the scale, in grid px, below which its tile holds no detail.
- **Cost.** It needs four projections and a 2×2 singular value decomposition, so it is free next to a render.
- **What it misses.** It knows nothing about the photograph itself.

The `Anisotropic` sampler sets the level from `σ_minor` and walks along the major axis. Under it, `φ` is close to `max(1, 1/σ_a)` per axis. The definition follows the sampler the consumer uses.

### The view's sharpness, from its self-similarity

The overlap reading of the view's own `R×R` tile (Part 1) gives the radius `ρ_v`, in grid px. It measures the detail the tile actually holds after sampling, focus and motion, and it is the only per-view number that sees a photograph out of focus. It also depends on the texture: a straight edge or grain reads long along itself. So it is compared only within one track, as the ratio to the track's sharpest view:

```
s_v = ρ_min / ρ_v ∈ (0, 1],   ρ_min = min over the track's views of ρ
```

Over a track, `ρ_v / φ_v` separates the two causes: a view whose radius is large for its footprint is out of focus. The reach's per-axis values `grid_axes` give the same split along and across a directional texture.

**Cost.**
- **Localizer.** Its gate already reads `ρ_v` once per view, at the seed, before the first round. The weight keeps that reading instead of using it only for pass and fail.
- **Bench.** The bench reads it per evaluation already.
- **Other consumers.** For a consumer with no gate (the subpixel refiner's fuse, normal refinement), the reading is one shift search per view on the tile it renders.
- **Budget, if needed.** Rank views by `φ_v` first and read `ρ_v` only for the views whose footprint is within a factor (say 2) of the track's smallest. Every other view gets the footprint term alone.

## Part 3: weighting the consensus and the fuse

Each view's weight in the consensus and in the fused bitmap becomes

```
w_v ∝ w_v^IRLS · f(φ_v) · g(s_v)
```

- **`w_v^IRLS`** is the Tukey/MAD weight from the view's residual, computed against the template at matched bandwidth (Part 4). That way a blurry view no longer earns weight by sitting close to a blurry mean.
- **`f`** falls with the footprint relative to the track's smallest: `f = (φ_min / φ_v)^p`. A view at twice the smallest footprint carries half the detail per axis. Above 1× zoom, `φ = 1` and `f` saturates: an `R`-pixel grid cannot hold detail finer than its own pixels.
- **`g`** falls with the sharpness ratio: `g = s_v^q`.

The exponents `p` and `q` are measured (see [Evaluation](#evaluation)).

**Floors.**
- The weights keep the existing guard against weight concentrating on one view (the effective view count `1/Σw²` normal refinement checks), so a track whose sharpest view is an outlier does not reduce to that one view.
- The obliquity prior `|cos θ|^p` of normal refinement stays where it is and multiplies in.

**Which consumers adopt it:**
- **The representative fuse** (`fuse_patch_bitmap`, `PatchViewStack::fuse`). This is what is stored in `patch_bitmaps_y_x_rgba` and what the bench and the culls read. It adopts the weights first, because a sharper stored bitmap is useful on its own and changes no geometry.
- **The congealing consensus** (keypoint localization), each round's leave-one-out template.
- **The subpixel refiner's IRLS weights.**
- **The add-image-to-tracks reference consensus.**
- **Normal refinement's weighted consensus Φ.** Weighting changes the objective the normal is chosen by, so it adopts the weights last and only after its own measurement.

## Part 4: scoring at matched bandwidth

A sharper template makes the fifth effect worse unless scoring changes with it. A view with footprint `φ_v` is scored against the template low-passed to that footprint:

```
T_v = G(σ_v) * T,   σ_v = c · sqrt(max(φ_v² − φ_T², 0))
```

- **`G`** is a separable Gaussian.
- **`φ_T`** is the template's own footprint, the weighted footprint of the views that built it.
- **`c`** converts a box footprint to a Gaussian width. `1/sqrt(12)` matches the variance of a box one footprint wide; the value is measured.

The view is scored against `T_v`: its ZNCC, its leave-one-out score and its IRLS residual. Its own tile is not touched.

- **Cost.** One separable blur of the `R×R` template per view, about `2·R²·(kernel width)` per channel. Small next to the render.
- **What it does not correct.** The footprint does not include blur in the photograph. A view out of focus still scores lower against a sharp template. That is intended: the low score reflects the view, and its sharpness weight `g` already lowers its influence.
- **Variant.** Fold the view's own self-similarity into `φ_v`, so an out-of-focus view is also scored at its own bandwidth. This is an [open question](#open-questions), since it would hide focus misses from the ZNCC bars.

**Effect on the bars.** ZNCC values change: far views score higher against a template blurred to their footprint. The bench's `min_zncc` bars (whole and middle) and the localizer's `min_absolute_zncc` / `min_relative_zncc` are re-measured once Part 4 lands.

## Evaluation

**Cases:**
- **The two dino_dog_toy tracks above**, as fixed cases for directional texture and mixed focus. The candidate file is local, not checked in. Committing it, or a small `.sfmr` holding just these two points, is part of this work.
- **seoul_bull_sculpture and kerry_park ground truths** (checked in, metric). These give keypoint accuracy against known poses.
- **The leave-one-track-out harness** in [scripts/track_at_pixel/](../../scripts/track_at_pixel/README.md), for track building end to end.

**Measures, for each configuration** (current, Part 1, Parts 1+3 fuse only, Parts 1+3+4, then each further consumer):

1. **Sharpness of the stored bitmap**: its own overlap-reading radius and its gradient energy, per point.
2. **ZNCC of each view against the consensus, by footprint and by sharpness.** The current pattern, sharp views below blurry ones at equal zoom, should go away.
3. **Keypoint error against ground truth**: reprojection of the ground-truth point, per view, split by footprint.
4. **Drift along a directional texture.** On the wood-grain track, how far Fit moves observations along the grain, and their ZNCC after.
5. **Verdict changes at the current bars**, to size the re-tuning in Parts 1 and 4.
6. **Time per point** for embed-patches on kerry_park.

## Open questions

- **The forms of `f` and `g`.** Whether power laws in `φ_min / φ_v` and `ρ_min / ρ_v` are enough, or whether a view should drop out entirely below some ratio.
- **Per-axis weighting.** On a directional texture a view may be sharp across the grain and blurry along it. Weighting each axis of the template separately, per pixel in the Fourier sense or by a directional blur, is possible but much more machinery. Is the isotropic weight enough?
- **Folding the radius into `φ_v`** for matched-bandwidth scoring (Part 4, variant). Fairer to out-of-focus views, but it hides their blur from the ZNCC bars.
- **The sampler.** `Anisotropic` keeps the minor axis of oblique views sharper than `BilinearMip`, at 1.6–3× the cost. With views weighted by footprint, does the cheaper sampler lose enough to matter?
- **The patch resolution.** On the 25-view track most views are far below 1× zoom, so the 24-px grid discards detail the near views hold and the far views cannot. Choosing `R` per track from its footprints is a separate change. It interacts with this one, because a larger `R` widens the range of `φ`.
- **Normal refinement.** Whether its objective should take these weights at all, or keep the agreement weights and only the matched-bandwidth scoring.

## Non-goals

- **Super-resolution.** The fuse does not reconstruct detail finer than the `R×R` grid from several views. A view above 1× zoom contributes at most the grid's own resolution.
- **Deconvolution.** A blurry view is down-weighted, not sharpened.
