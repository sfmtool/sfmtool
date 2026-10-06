# Sharper Patch Bitmap

**Status:** Draft. Decided:
- the patch bitmap, stored and used as the template views are scored against, is computed from the views that hold the most detail, judged by each view's sharpness (its ZNCC self-similarity radius) and its resolution in the patch grid (its zoom), not by agreement with the mean alone. The leading candidate is to take the single best view's bitmap rather than averaging, because a mean over differently exposed photographs is unreliable without a model of their brightness and colour shifts. A weighted mean is kept for comparison (Part 5);
- the angle between the view's ray and the patch normal is a third input, with its own two axes: the view foreshortens the patch by `cos θ` along the tilt direction and not across it. It is kept separate from the anisotropy of the Jacobian:
  - the Jacobian measures resolution, whether it is compressed by obliquity or by lens distortion;
  - the angle measures how sensitive the view is to errors in the patch model, and along which direction;
- the sampler is chosen per view from the Jacobian's anisotropy, so an oblique or distorted view keeps the detail along its less compressed axis (Part 3). That step is built, with the threshold `a = 1.5`;
- the `.sfmr` file stores each observation's self-similarity radius, measured on its own `R×R` render, as a measurement in grid px. It does not store the derived sharpness or the final weight. Every operation that re-renders a point's bitmap reads the stored radii, and recomputes the geometric factors from the file's current geometry (Part 7);
- the self-similarity reading summarises its region by an ellipse, whose semi-major axis is the radius, in place of the contour's furthest point, the slide and the reach (Part 2). That step is built;
- the self-similarity radius the weights read is the reading of exactly the `R×R` tile, with no pixels from outside it (Part 1). That prerequisite is built.

Not decided: the functional forms of the weights, the blur used for matched-bandwidth scoring, and the order the consumers adopt it in. See [Open questions](#open-questions).

Amends:
- [core/patch/patch-keypoint-localization.md](../core/patch/patch-keypoint-localization.md): the congealing consensus
- [core/patch/keypoint-subpixel-refinement.md](../core/patch/keypoint-subpixel-refinement.md): the representative fuse
- [core/patch/patch-normal-refinement.md](../core/patch/patch-normal-refinement.md): the weighted consensus
- [core/bench/editable-track.md](../core/bench/editable-track.md): the ZNCC bars (`min_zncc`, whole and middle), re-measured against the matched-bandwidth score
- [formats/sfmr-file-format.md](../formats/sfmr-file-format.md): per-observation self-similarity columns in `tracks/`

## Purpose

A point in a patch-based reconstruction carries a small square bitmap of the surface around it. Every photograph that sees the point is resampled into that square and compared against a template built from the other views. The pipeline builds the template, and the stored bitmap, as a robust weighted mean of the views. That mean is blurrier than the best views:
- the far views carry less detail;
- some photographs are out of focus;
- the views do not line up to the last fraction of a pixel.

The sharpest views then correlate worst with it, because the detail they carry has nothing in the template to match. This draft proposes computing the bitmap from the views that hold the most detail and fit the patch best, whether as a weighted mean of them or by choosing a single reference view. Three per-view numbers judge them, each computed by the code already or in a few operations:
- how many photograph pixels each patch-grid pixel covers along each axis (the zoom);
- how far the view's own tile can slide over itself and still match (the ZNCC self-similarity radius);
- the angle between the view's ray and the patch normal.

It renders each view with the single-sample mip sampler or the anisotropic sampler, chosen per view from how much coarser the mip sampler would read its less compressed axis than that axis needs (Part 3), so a view seen at an angle or through a distorting lens keeps the detail it holds along its better axis.

It also proposes scoring each view against the template at a resolution the view can actually show. A sharp template then does not penalise a far or blurry view for detail it could never have had.

## The problem, measured

Both cases are tracks on the bench of a local ground-truth candidate for the checked-in `dino_dog_toy` images (`dino_dog_toy_ground_truth_candidate_v001.sfmr`, 85 images at 2040×1536, patch resolution `R = 24`). They were read with the Track View's *Zoom* column and the extent of the self-similarity region along the patch's axes, which the Track View's hover then showed; it now shows the region's ellipse ([core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md) § "The ellipse in other units"). The radii in the table are the contour's furthest point, which reads about a tenth of a pixel longer than the ellipse's semi-major axis.

**A wood-grain patch sized to about 1× at its closest view.** The patch has half-extent 0.053, with its v axis turned along the grain.
- Its five original views read zoom 0.59/0.92× to 0.92/1.3×.
- Each locks to 0.1–0.2 grid px across the grain and 0.85–1.4 grid px along it.
- A geometry search added nine views spanning zoom 0.47/0.78× to 1.6/2.5×.
- Fit then moved some observations along the grain and lowered their ZNCC.

That is the congealing drift the template's blur along the grain allows.

**A larger patch over the same texture.** The patch has half-extent 0.120, with 25 views, all in. Its largest zoom is 0.76/1.6×. Twenty-two of the views are below 0.6×, so the mip sampler reads them from pyramid levels 1–2. Among the views read at level 2 (zoom 0.24–0.32), the ones whose photographs are sharp correlate worst with the patch bitmap:

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

The measurements above were made before Part 3, with `BilinearMip` rendering every view.

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

## Part 1 (prerequisite, built): self-similarity of exactly the R×R tile

The bench, both member gates and the culls read the ZNCC self-similarity radius of exactly the `R×R` tile or bitmap they judge, with no pixels from outside it, as [core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md) § "The overlap reading" describes; the 2.5 bar was re-measured there and kept.

## Part 2 (built): the self-similarity region as an ellipse

The self-similarity reading summarises the region of shifts that match the tile by the ellipse with the same second moments about the true position, with its semi-axes, the angle of its major axis, a flag on each axis where the true length may be larger, and its 2×2 matrix, for the whole tile, its middle and each ninth. The radius is the ellipse's semi-major axis. The ellipse maps into image px through the tile's Jacobian and onto the patch through its half-extents, and the bench, the Track View's hover, the MCP fields and the Python bindings carry it in place of the furthest-point radius, the slide and the reach, as [core/patch/zncc-self-similarity-radius.md](../core/patch/zncc-self-similarity-radius.md) § "The region, its ellipse and the radius" describes; the 2.5 bar was re-measured there and kept. What else it could serve is left to later work: a radius longer than the zoom predicts as a sign of blur in the photograph, the ellipse through the Jacobian as a per-observation, direction-dependent weight for bundle adjustment, and limiting the localizer's moves along the major axis, the direction congealing drifts along on a grain.

## Part 3 (built): choosing the sampler per view

Every kernel that renders a view's tile, the bench's evaluation, the member gates, view selection, normal refinement, congealing, the sub-pixel refiner, the fuse and Track View, renders it with the sampler the sampler rule picks for that observation: `Anisotropic` when `σ_major ≥ √2` and `L = 2^l / max(σ_minor, 1) ≥ a`, with `l = round(log2 max(σ_major, 1))`, read from the Jacobian of the patch re-anchored on the keypoint at the patch resolution, and `BilinearMip` otherwise, with `a = 1.5`. A view the rule leaves on `BilinearMip` renders the same tile as under `Fixed(BilinearMip)`, bit for bit. The renders are timed in detail phases, the batches that render views take a `Progress` and can be cancelled, and the anisotropic sampler has an AVX2 kernel that matches its scalar path bit for bit. The anisotropic sampler takes its samples along the direction in the photograph that the view compresses most, the left singular vector of the Jacobian, and `a` was measured with that walk. The rule, the measurements `a` was set from and the kernel are in [core/camera/image-warping.md](../core/camera/image-warping.md) § "Choosing the sampler per view", and so is the view's **footprint** `φ_v`, the scale in grid px below which its tile holds no detail, which Parts 4 to 6 weight and blur by: `φ_a = max(2^l, 1) / σ_a` along each singular direction under `BilinearMip`, close to `max(1, 1/σ_a)` under `Anisotropic`, and `φ_v = max(φ_major, φ_minor, 1)`. Storing `a` beside the self-similarity readings, so a reader can tell each stored reading's sampler from its zoom, is part of Part 7; `sfm embed-patches` records it in the file's `tool_options` as `anisotropic_threshold`.

## Part 4: what each view can contribute

A sharper template needs to know, for each view, how much detail its tile carries and how well it lines up with the others. This part takes stock of what we can measure on the views' bitmaps that serves that goal, and notes what else each measurement serves. A measurement is then made once, stored (Part 7), and read by every consumer. The weights (Part 5) are one of those consumers.

### What we can measure

- **The ZNCC self-similarity grid.** The tile correlated with itself at each small shift. A sharp tile stops matching itself within a fraction of a pixel, and a blurry one keeps matching further. Its values are computed only from the patch bitmap's pixels, so they measure the patch itself.
  - We judge the bitmap with a radius that is the semi-major axis of an ellipse fitted to the region where the ZNCC is at or above `1 − τ`, where `τ` adapts to the bitmap's contrast (`1 − τ` is about 0.95 on a high-contrast bitmap and lower on a faint one). When the semi-minor axis is much shorter, the patch can slide one way but is held in place perpendicular to it (Part 2).
  - Reading it costs one shift search per view. The localizer's gate and the bench already read it, and a stored value (Part 7) saves reading it again.
- **The zoom.** The Jacobian of the patch grid's map into the photograph, at the patch centre. Like the self-similarity, it has a major and a minor axis: its singular values give the least and the most zoom, in grid px per photograph px, along two perpendicular directions. Its values are computed only from the camera and the patch geometry, not from any pixels.
  - Where the zoom along an axis is above 1, one photograph pixel covers more than one grid pixel. The zoom is then an upper bound on how sharp the bitmap can be along that axis. Below 1, the sampler reads the mip level that matches, so the bitmap can be as sharp as its grid (Part 3).
- **The viewing angle.** The angle `θ` between the view's ray through the keypoint and the patch normal. It also has a major and a minor axis. Along the tilt direction (the ray projected into the patch plane) the view foreshortens the patch by `cos θ`, and across it not at all. Its values are computed only from the camera pose and the patch geometry, not from any pixels.
  - Along the tilt direction the tile depends most on the patch model being right. An error in the normal shears the tile by an amount that grows with `tan θ`, and an error in depth or relief on the surface shifts it. Across the tilt direction the view is insensitive to both.
  - The foreshortening is the part of the zoom's anisotropy that comes from obliquity. The angle is the same whatever lens took the view, so it separates obliquity from lens distortion.
- **Coverage.** The fraction of the tile that carries image data. A tile whose patch crosses the photograph's border has samples with nothing behind them. Its values are computed from the tile's data flags. [Member-coherence validation](../core/patch/member-coherence-validation.md) already gates on it (`min_valid_fraction`).
  - A partial tile holds less of the patch, and cannot be a single reference on its own.
  - Masks on the photographs, when they are added, affect it the same way: a masked sample carries no data, just as a sample past the border does. The self-similarity reading already leaves out samples without data, and member coherence correlates pairs only over the samples both views cover.
- **Clipped pixels.** The share of the tile at the limits of the photograph's range, 0 or the maximum in any channel. Specular highlights and blown-out sky clip, and a clipped region has neither texture nor its true colour. It is read from the photograph's own pixels, since the sampler's blending moves a clipped value off the limit.
- **Brightness and colour.** Each tile's mean and spread per channel, which the ZNCC's normalization computes and then discards. A view in shadow, overexposed or under a different white balance correlates as well as the others, but its colours differ. Compared across a track's views, they say which views are typical in exposure and colour.
- **The ZNCC between observation bitmaps.** Two views' tiles correlated directly, with no bitmap built from them in between. Its values are computed from the pixels of the two views.
  - It says which views agree with which, not only with a mean, so it can judge a candidate reference without the mean's bias toward blur. [Member-coherence validation](../core/patch/member-coherence-validation.md) builds the full matrix over a point's members, to find tracks that mix two surfaces.
  - It is also read on parts: the 3×3 grid of ZNCC says where in the tile two views disagree (an occluder, a highlight, a corner off the plane), and the coarser grids member coherence reads say whether they disagree only in fine detail, which is what blur does.
  - The full matrix grows with the square of the number of views, so a long track needs a choice of which pairs to correlate (Part 5).

### What else the zoom serves

The zoom and the footprint it gives (Part 3) also serve:
- **Matched-bandwidth scoring** (Part 6).
- **The baseline for focus:** the radius a view's sampling alone would give, against which a longer radius points to the photograph.
- **Choosing `R`** per track: views above 1× zoom mean the grid discards detail they hold.
- **Converting** grid px to image px, as the Track View's hover does.
- **Ranking views** before spending a shift search on them.

**Cost.** It needs four projections and a 2×2 singular value decomposition, so it is free next to a render.

### How much the tile depends on the patch model: the viewing angle

`θ_v` is the angle between the view's ray through the observation's keypoint and the patch's outward normal: `cos θ_v = −n · d̂`, with `d̂` the unit ray from the camera centre through the keypoint. Because the render re-anchors the patch so its centre projects onto the keypoint, this is also the ray to the rendered patch's centre. Computing `d̂` unprojects the keypoint through the camera model.

**Its two axes.** The **tilt direction** `t̂_v` is `d̂` projected into the patch plane and normalized, with its angle `α_v` in the patch's u, v frame. The view's foreshortening of the patch is an ellipse:
- along `t̂_v`, its minor axis, the view compresses the patch by `cos θ_v`;
- across `t̂_v`, its major axis, not at all.

A view facing the patch (`θ = 0`) has no tilt direction, and its ellipse is a circle.

**What it measures.** It measures what the Jacobian does not: how much the view's tile changes when the patch model is slightly wrong, and along which direction. Most of that change is along the tilt direction:
- **An error in the normal** shears the rendered tile along `t̂_v` by an amount that grows with `tan θ`. A view facing the patch barely changes.
- **Depth.** An error in the point's depth moves the tile across the patch along `t̂_v`, faster the more oblique the view.
- **Relief and non-planarity.** At grazing angles, bumps on the surface hide or stretch parts of the patch along `t̂_v`, and a planar warp fits worse.
- **Appearance.** Shading, specular highlights and how much of the texture is visible change with the angle. An oblique view correlates worse even when it is sharp and well aligned. This part has no direction.

**What else the angle serves:**
- **The weights**, in the patch bitmap and in normal refinement (whose obliquity prior it is).
- **Grazing views**, which a gate can reject before rendering.
- **The reference view**, chosen among the views facing the patch.
- **Positional uncertainty**, beside the self-similarity ellipse: a view's expected error from an error in the normal or depth, along `t̂_v`.

### Obliquity and distortion are separate inputs

Obliquity and lens distortion both make the Jacobian anisotropic. For a small patch the Jacobian is, roughly, a scale (distance and focal length), times the foreshortening ellipse of the viewing angle, times what the lens does at that point in the image. The angle's ellipse is the obliquity part, so taking it out of the Jacobian leaves the scale and the lens.
- **For resolution they are the same thing.** The footprint and the sampler rule read the whole Jacobian and treat them alike.
- **They differ in geometry.** A patch near the edge of a fisheye image is compressed radially even when the camera faces it (`θ ≈ 0`). Its view is penalised for resolution through `φ_v`, rendered with the sampler that suits it, and not penalised for geometry. A patch at 70° through a pinhole lens can have the same Jacobian, and is penalised through both `φ_v` and `θ_v`.

So the weights read obliquity only from the angle, never from the Jacobian's anisotropy, and read resolution only from the footprint and the radius, never from the angle. That keeps an oblique view from being penalised twice.

## Part 5: computing the patch bitmap

The pipeline computes the patch bitmap today as a weighted mean of the views, a consensus. The measurements in Part 4 rank the views by what they can contribute, which serves other ways to compute it as well.

**The leading candidate is a single reference view.** The bitmap is the tile of the one view the measurements rank best:
- a short self-similarity radius, with a zoom near 1 and a view facing the patch;
- full coverage, few clipped pixels, and brightness and colour typical of the track;
- agreement with the other views, by the ZNCC between their bitmaps, so a view that disagrees with them is not chosen.

**Why not a mean.** The photographs differ in exposure and white balance. The ZNCC is blind to those differences, but a mean of the tiles is not. Without a model of each photograph's brightness and colour shift, a mean over differently exposed photographs mixes colours that never appeared together on the surface, and its detail is blurred by every view that does not line up exactly. A single view has neither problem.

**What a single view costs.** It keeps that view's noise, and any highlight or occluder the measurements missed. The bitmap also changes all at once when another view comes to rank higher.

**The alternatives** are kept for comparison in the evaluation (see [Evaluation](#evaluation)):
- **a weighted mean** of the views, weighted by the measurements (below);
- **a mean of the few best views**, between the two.

### Which pairs of views to correlate

Choosing a reference, and checking a mean's members against each other, read the ZNCC between observation bitmaps. The full matrix of `k` views costs `k(k−1)/2` correlations: 300 for the 25-view track, about 5000 for a track of 100. Each is cheap at `R×R`, but the cost grows with the square of the track. Strategies that read fewer pairs:
- **Candidates against all.** Rank the views by the Part 4 measurements, which cost one reading per view. Correlate only the top few candidates with every view, `m·k` pairs. The reference is the candidate that agrees best with the rest.
- **Neighbours in viewing direction.** Views with similar rays and zoom should agree most. Correlating each view with its nearest few gives a sparse graph that still shows a group of views that disagrees with the rest.
- **Coarse first.** Read all pairs on the coarse grid, which costs a fraction of the full one, and the full grid only for the pairs the decision turns on.
- **Incremental.** A view added to a track (Add Image to Tracks, a bench geometry search) is correlated with the reference and the top candidates only.

Which strategy, and how many candidates, is measured against the full matrix on tracks small enough to compute it.

### The weighted mean

Each view's weight in the mean becomes

```
w_v ∝ w_v^IRLS · f(φ_v) · g(ρ_v) · h(θ_v)
```

- **`w_v^IRLS`** is the Tukey/MAD weight from the view's residual, computed against the template at matched bandwidth (Part 6). That way a blurry view no longer earns weight by sitting close to a blurry mean.
- **`f`** falls with the footprint relative to the track's smallest: `f = (φ_min / φ_v)^p`. A view at twice the smallest footprint carries half the detail per axis. For views that shrink the photograph `φ` stays between 1 and √2 (Part 3), so `f` mostly falls for views that magnify it. The radius already reads long on those tiles, so whether `f` adds anything beside `g` is measured.
- **`g`** falls with the radius relative to the track's shortest: `g = (ρ_min / ρ_v)^q`, computed when the view is weighted. The radius is compared only within one track, since it also depends on the texture.
- **`h`** falls with the viewing angle: `h = |cos θ_v|^k`. This is the same term as normal refinement's obliquity prior (`obliquity_weight_power`, off by default there), used here in every consumer. It is the isotropic form. The angle's sensitivity lies along the tilt direction, so the directional form weights the view by `|cos θ_v|^k` along `t̂_v` and fully across it. That needs per-axis weighting of the template (see [Open questions](#open-questions)).

The exponents `p`, `q` and `k` are measured (see [Evaluation](#evaluation)). `f` uses the footprint under the sampler the view is rendered with. A view that the sampler rule moves to `Anisotropic` therefore gets the smaller footprint its tile actually has.

**Floors.**
- The weights keep the existing guard against weight concentrating on one view (the effective view count `1/Σw²` normal refinement checks), so a track whose sharpest view is an outlier does not reduce to that one view.
- Normal refinement's obliquity prior is `h`. Where normal refinement adopts these weights, `h` replaces its prior rather than being applied a second time.

**Which consumers adopt it.** Each computes a patch bitmap, stored or used as a template, from the views:
- **The representative fuse** (`fuse_patch_bitmap`, `PatchViewStack::fuse`). This is what is stored in `patch_bitmaps_y_x_rgba` and what the bench and the culls read. It adopts the new method first, because a sharper stored bitmap is useful on its own and changes no geometry.
- **The congealing consensus** (keypoint localization), each round's leave-one-out template. Under a single reference, the reference view itself is scored against the next best.
- **The subpixel refiner's IRLS weights.**
- **The add-image-to-tracks reference consensus.**
- **Normal refinement's weighted consensus Φ.** Weighting changes the objective the normal is chosen by, so it adopts the weights last and only after its own measurement.

## Part 6: scoring at matched bandwidth

A sharper template makes the fifth effect worse unless scoring changes with it. A view with footprint `φ_v` is scored against the template low-passed to that footprint:

```
T_v = G(σ_v) * T,   σ_v = c · sqrt(max(φ_v² − φ_T², 0))
```

- **`G`** is a separable Gaussian.
- **`φ_T`** is the template's own footprint, the weighted footprint of the views that built it.
- **`c`** converts a box footprint to a Gaussian width. `1/sqrt(12)` matches the variance of a box one footprint wide; the value is measured.

The view is scored against `T_v`: its ZNCC, its leave-one-out score and its IRLS residual. Its own tile is not touched.

- **Anisotropic footprints.** A view that the sampler rule leaves on `BilinearMip` can still have unequal per-axis footprints, below the threshold `a`. It can be matched with a Gaussian elongated along the Jacobian's singular directions, at the same cost as a separable blur along those directions. Whether the isotropic blur is enough is measured.
- **Cost.** One separable blur of the `R×R` template per view, about `2·R²·(kernel width)` per channel. Small next to the render.
- **What it does not correct.** The footprint does not include blur in the photograph. A view out of focus still scores lower against a sharp template. That is intended: the low score reflects the view, and its sharpness weight `g` already lowers its influence.
- **Variant.** Fold the view's own self-similarity into `φ_v`, so an out-of-focus view is also scored at its own bandwidth. This is an [open question](#open-questions), since it would hide focus misses from the ZNCC bars.

**Effect on the bars.** ZNCC values change: far views score higher against a template blurred to their footprint. The bench's `min_zncc` bars (whole and middle) and the localizer's `min_absolute_zncc` / `min_relative_zncc` are re-measured once Part 6 lands.

## Part 7: storing the self-similarity radii in the `.sfmr` file

Each observation's self-similarity radius is the one input to the weights that needs the photograph and a shift search. The other inputs need no photograph: the footprint and the viewing angle come from geometry, and the IRLS agreement is cheap once the views are rendered. So the file stores the radii, and every operation that re-renders a point's bitmap reads them instead of measuring again. These operations include:
- `embed-patches`;
- the bench commit's fuse;
- `fuse_patch_bitmap` / `fuse_patch_cloud_bitmaps`;
- conversion to embedded patches;
- any later re-render.

The file stores the radius itself, in grid px, not a ratio to the track's shortest radius or the weight `w_v`. A ratio and a weight depend on the rest of the track and on functional forms that are still being tuned. A radius is a measurement with units, and it serves more than this draft:
- the weights here (`g` is computed from it at render time);
- the cull bars (`max_zncc_self_similarity_radius`), applied without the photographs;
- the ellipse as a confidence bound on the point on the patch plane, converted to scene units through the patch's half-extents;
- the Track View, which can show a committed track's readings before its evaluation runs.

### What is stored

For observation `j` of point `i`, measured on the observation's own `R×R` render:
- **The render.** It goes through point `i`'s patch, re-anchored on observation `j`'s keypoint, at the point's patch resolution `R`. It uses the sampler the rule in Part 3 picks for that view. This is the tile the bench and the member gates read (Part 1).
- **The reading.** The overlap reading, with the default `max_radius` `r`.

Optional columns, parallel to the other `tracks/*` arrays. Their names follow the glossary: the measurement is the ZNCC self-similarity radius, its region is summarised by the contour's ellipse, whose semi-major axis is the radius, and a value whose true length may be larger is "at least" that value. Each 0/1 column is named after the value it qualifies and has the same shape, so no column needs a legend.

| Column | Shape, type | Meaning |
|---|---|---|
| `tracks/zncc_self_similarity_ellipse_axes` | `(M, 2)` `float32` | Semi-major and semi-minor axis lengths of the whole bitmap's contour ellipse, grid px. The semi-major axis is the radius |
| `tracks/zncc_self_similarity_ellipse_axes_is_at_least` | `(M, 2)` `uint8` | 1 per axis where the true length may be larger |
| `tracks/zncc_self_similarity_ellipse_major_angle` | `(M,)` `float32` | Angle of the major axis from the patch's u axis towards its v axis, radians in `[0, π)` |
| `tracks/zncc_self_similarity_cos_view_angle` | `(M,)` `float32` | `cos θ = −n · d̂` of the render the radius was read on: `n` the re-anchored patch's outward normal, `d̂` the unit ray from the camera centre through the observation's keypoint. Positive where the patch faces the camera |
| `tracks/zncc_self_similarity_tilt_angle` | `(M,)` `float32` | `α`, the angle of that render's tilt direction (`d̂` projected into the patch plane) from the patch's u axis towards its v axis, radians in `[0, π)`. `NaN` where the view faces the patch head on |
| `tracks/zncc_self_similarity_zoom` | `(M, 2)` `float32` | `[least, most]` zoom of that render: `[1/σ_major, 1/σ_minor]` of the Jacobian of the patch grid into the photograph at the patch centre, grid px per photograph px |

- **What the last three columns describe.** They record the geometry of the render the radius was read on, not the file's current geometry. The format text says so in those words.
  - **Why the prefix.** They carry the `zncc_self_similarity_` prefix so a reader sees they belong to that measurement.
  - **Why store them.** All need the camera model to recompute. The angle and the tilt direction need the keypoint unprojected, and the zoom needs the projection's derivative. Storing them gives a reader the exact measured values without implementing the camera models.
- **The fallback render.** Where the keypoint's ray cannot meet the patch plane (parallel to it, or the plane behind the camera), the render uses the stored patch without re-anchoring, and `d̂` is the ray to the stored patch's centre. Such a view is at or past grazing, so its value is near zero or negative either way.
- **Not measured.** `NaN` in the ellipse axes means the observation was not measured. Its angles and zoom are then `NaN` and its flags 0.
- **Metadata.** `tracks/metadata.json` records `r` and the flat floor, the noise and the relative tolerance the reading used, so a reader can tell whether stored radii are comparable with its own. It also records the sampler threshold `a` the renders were made under (Part 3), from which a reader works out each render's sampler from its stored zoom. `sfm embed-patches` already records `anisotropic_threshold` in the file's `tool_options`, but `tool_options` describes one tool run, and a later re-render of the radii, such as `sfm xform --add-patch-bitmaps sampler=per_view`, records no threshold there, so the radii need their own record.
- **Grid px.** The grid px are those of the point's `R` (`points3d/metadata.json`'s `patch_bitmap_resolution`), so the radius converts to scene units through the point's patch half-extents, as the ellipse does.

The middle-square and per-cell readings are left out. They are cheap to recompute once a render exists, and nothing proposed here reads them without one.

### When a stored radius stops describing the view

A radius describes one render: the point's patch (centre, normal, axes, half-extents), the observation's keypoint, the image's pose and intrinsics, and the photograph. These change often, and by small amounts. Bundle adjustment moves every pose, and Fit and normal refinement move keypoints and normals.

**The reader judges staleness, not the writer.** A writer that changes the geometry does not have to clear or re-measure the radii. It carries the stored values through, and a writer that copies, filters or reorders observations does the same. A reader that uses a radius compares the stored angle and zoom with the ones it computes from the current geometry:
- small differences mean the radius still describes this view;
- a large change means it no longer does. Examples are a normal refit that turns the patch 20°, a resize that halves the zoom, or a keypoint moved onto other texture.

Each consumer applies the tolerance that suits it. A weight can tolerate more drift than a cull bar.

**Two cases the stored conditions cannot detect, so a writer clears the radius (sets it to `NaN`):**
- a keypoint moved so far that the render covers different texture while the angle and zoom stay about the same, as when a Fit walks an observation to another place;
- the observation's photograph replaced.

The threshold for "so far" is stated in grid px, e.g. more than half the radius's own `r`.

### A hand-set weight

The bench can down-weight a view by hand, e.g. a photograph the user sees is out of focus. That is a judgement, not a measurement, so it lives in its own optional column, `tracks/appearance_weight_override` (`(M,)` `float32`, `NaN` where not set). A render multiplies it into `w_v`. It never overwrites the measured radius, so a later re-measurement does not erase the user's decision, and a reader can always tell which is which.

### `observation_confidence`

The format already has an optional per-observation column, `tracks/observation_confidence` (`uint8`). Its spec defines it as the observation's photometric sharpness relative to its track's consensus. The bench commit and Add Image to Tracks fill it with the observation's leave-one-out ZNCC against the consensus, which this draft shows is biased against sharp views (§ "The problem, measured"). Two changes keep the column's meaning and its contents in agreement:
- writers fill it from the stored radius, quantizing `ρ_min / ρ_v`;
- or its spec is changed to say it is the leave-one-out ZNCC.

Which one is part of this work. The radius columns do not depend on the choice.

## Evaluation

**Cases:**
- **The two dino_dog_toy tracks above**, as fixed cases for directional texture and mixed focus. The candidate file is local, not checked in. Committing it, or a small `.sfmr` holding just these two points, is part of this work.
- **seoul_bull_sculpture and kerry_park ground truths** (checked in, metric). These give keypoint accuracy against known poses.
- **The leave-one-track-out harness** in [scripts/track_at_pixel/](../../scripts/track_at_pixel/README.md), for track building end to end.
- **Distortion against obliquity.** Points near the edges of the kerry_park fisheye images, seen roughly face on, compared with points on dino_dog_toy seen at large angles with a similar Jacobian anisotropy:
  - the fisheye points should gain from the sampler rule and lose nothing to `h`;
  - the oblique points should gain from the sampler rule and be down-weighted by `h`.

**Measures, for each configuration** (current; Part 5's weighted mean in the fuse only; Part 5's single reference in the fuse only; then with Part 6; then each further consumer):

1. **Sharpness of the stored bitmap**: its own overlap-reading radius and its gradient energy, per point.
2. **ZNCC of each view against the patch bitmap, by footprint and by sharpness.** The current pattern, sharp views below blurry ones at equal zoom, should go away.
3. **Keypoint error against ground truth**: reprojection of the ground-truth point, per view, split by footprint.
4. **Drift along a directional texture.** On the wood-grain track, how far Fit moves observations along the grain, and their ZNCC after.
5. **Verdict changes at the current bars**, to size the re-tuning in Part 6.
6. **The sampler rule's time and changes to the patches.** Measured, and filed in [image-warping.md](../core/camera/image-warping.md) § "How `a = 1.5` was set" and § "Cost, and the AVX2 kernel".

## Open questions

- **The staleness tolerances** a consumer applies to the stored angle and zoom (Part 7), and the keypoint distance past which a writer clears a stored radius.
- **`observation_confidence`**: refill it from the radius as the quantized ratio `ρ_min / ρ_v`, or redefine it as the leave-one-out ZNCC its writers already put in it (Part 7).

- **Which pairs to correlate** (Part 5), and whether the pairwise ZNCC's coarse-grid sharpness (member coherence's `sharpness_deficit`) adds anything beside the self-similarity radius.
- **The forms of `f` and `g`.** Whether power laws in `φ_min / φ_v` and `ρ_min / ρ_v` are enough, or whether a view should drop out entirely below some ratio.
- **Per-axis weighting.** On a directional texture a view may be sharp across the grain and blurry along it. Weighting each axis of the template separately, per pixel in the Fourier sense or by a directional blur, is possible but much more machinery. Is the isotropic weight enough?
- **Folding the radius into `φ_v`** for matched-bandwidth scoring (Part 6, variant). Fairer to out-of-focus views, but it hides their blur from the ZNCC bars.
- **Whether the sampler rule should also consider the view's weight.** A view with a small weight contributes little to the template, so rendering it with the anisotropic sampler may not pay. With the AVX2 kernel an anisotropic render costs 0.65 to 1.55 times what a `BilinearMip` one does, the most on views compressed 10 times or more along one axis, which take the most samples; on a CPU without AVX2 it costs 1.8 to 4 times as much. The question matters most on such views and on such CPUs.
- **Directional angle terms.** The angle's sensitivity lies along the tilt direction (Part 4). Weighting the template along `t̂_v` by `|cos θ_v|^k` and fully across it is the directional form of `h`, and belongs with per-axis weighting.
- **The patch resolution.** On the 25-view track most views are far below 1× zoom, so the 24-px grid discards detail the near views hold and the far views cannot. Choosing `R` per track from its footprints is a separate change. It interacts with this one, because a larger `R` widens the range of `φ`.
- **Normal refinement.** Whether its objective should take these weights at all, or keep the agreement weights and only the matched-bandwidth scoring.

## Non-goals

- **Super-resolution.** The fuse does not reconstruct detail finer than the `R×R` grid from several views. A view above 1× zoom contributes at most the grid's own resolution.
- **Deconvolution.** A blurry view is down-weighted, not sharpened.
