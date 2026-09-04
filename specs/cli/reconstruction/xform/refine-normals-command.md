# `sfm xform --refine-normals` Design

The `sfm xform --refine-normals` operation surfaces the photometric
patch-normal refinement described in
[patch-normal-refinement.md](../../../core/patch/patch-normal-refinement.md) as
a reconstruction transform: it rewrites each point's surface normal in place by
rendering the point's patch into its observing views and maximizing their
photometric agreement, then re-persists the refined patch cloud alongside the
normals.

## Why this fits `xform` (and how)

`xform` is the reconstruction-in / reconstruction-out pipeline. Normal
refinement reads a reconstruction and produces a reconstruction with the same
points, poses, and cameras but **better surface normals**, so it slots in
naturally — with three characteristics that shape the design:

1. **It is a modifier, not a point filter.** Unlike `--remove-short-tracks`
   et al. it removes no points and changes no positions; it only rewrites the
   per-point `normal` field (stored in the `.sfmr` version 3+ as `normals_xyz`,
   surfaced as `recon.normals` and writable via
   `recon.clone_with_changes(normals=...)`). It also re-persists the refined patch
   geometry (the stored frame stays consistent with the normals). It therefore belongs
   in the **Optimization** group of `xform-command.md` next to `--bundle-adjust`,
   not under "Filtering Operations." (The user's "filter" is loose usage; the
   operation is a refinement transform.) The point count is unchanged.

2. **It needs the source images.** Every other geometric transform works on
   reconstruction arrays alone. Refinement is photometric: it renders each
   point's patch into its observing views and maximizes cross-view consensus,
   so it must read the original images. There is direct precedent for an
   `xform` op reaching back into the workspace for the artifacts it needs —
   `--remove-large-features` and `--find-points-at-infinity` both read the
   `.sift` files via `recon.workspace_dir` / `recon.image_names` /
   `recon.metadata()`. This op reads the **images** the same way, resolving
   `workspace_dir / image_name` (the same path `scripts/patch_crossval.py`
   already loads with `cv2.imread`). A missing image is a hard error
   (`FileNotFoundError`), mirroring the SIFT-reading filters.

3. **It takes an `embedded_patches` reconstruction only.** Refinement anchors
   each view's patch at that observation's **stored keypoint** rather than at
   the reprojected point centre — the real detected feature gives a cleaner
   cross-view consensus — and it reads the point's stored patch frame back
   rather than building one. Neither the inline per-observation keypoints nor
   the frame exists on a `sift_files` reconstruction, so `--refine-normals`
   accepts a `feature_source == "embedded_patches"` input and rejects
   `sift_files` with a `click.UsageError` naming
   `sfm xform --to-embedded-patches` (or `sfm embed-patches`) — the shared
   `require_embedded_patches` gate in `_feature_source.py`. The check is
   per-step: `xform/_apply.py` reads the
   `RefineNormalsTransform.required_feature_source` attribute and validates the
   reconstruction *as it stands at that point in the pipeline*, so a
   `--to-embedded-patches --refine-normals` chain converts first and passes.
   The pass re-refines the normal (and, with `bitmaps`, re-renders the texture)
   over the stored keypoints and the stored view set; it runs no view selection
   and no keypoint localization of its own — those are `--localize-keypoints`
   and `--refine-keypoints`.

The result: a transform that reads the stored `PatchCloud` off the
reconstruction (`recon.patches`), calls
`PatchCloud.refine_normals(recon, images, …)`, and writes the refined normals —
and the refined cloud — back onto the points.

## Command syntax

The operation is `RefineNormalsTransform` in
[_refine_normals.py](../../../../src/sfmtool/xform/_refine_normals.py), wired
through `parse_refine_normals_params` in
[_arg_parser.py](../../../../src/sfmtool/xform/_arg_parser.py) and the
`--refine-normals` Click option in
[xform.py](../../../../src/sfmtool/_commands/xform.py); the refinement kernel
is [normal_refine/](../../../../crates/sfmtool-core/src/patch/normal_refine/),
reached through the `PatchCloud.refine_normals` PyO3 binding.

```
sfm xform <input.sfmr> [<output.sfmr>] --refine-normals [<params>] [...]
```

`--refine-normals` takes an **optional** comma-separated parameter string of
`key=value` modifiers. With no value it runs the recommended v1 defaults. Every
knob is `key=value` — there are no positional slots — so the argument is
order-free and self-documenting however many knobs are set:

```
--refine-normals
--refine-normals angular_range_deg=25,init_steps=7
--refine-normals angular_range_deg=25,init_steps=7,sampler=anisotropic,objective=mean
--refine-normals resolution=32,bitmaps=false
```

The option is declared `is_flag=False, flag_value=""` in `_commands/xform.py`,
which is what makes all three spellings work — bare
`--refine-normals`, space-separated `--refine-normals init_steps=7`, and joined
`--refine-normals=init_steps=7` — while leaving a following option untouched:
`--refine-normals --bundle-adjust` runs the refinement on its defaults and then
bundle-adjusts. Because `xform` re-walks the raw argument list to recover
operation order, `xform/_arg_parser.py` reproduces that tokenization itself — a
value joined with `=` is taken verbatim, otherwise the next token becomes the
value only if it does not start with `-`.

(This differs from the older `xform` mini-DSLs — `--remove-isolated
factor,value_spec`, `--include-by-distribution COUNT[,verbose]` — which lead
with positionals; with a dozen knobs here, all-`key=value` reads better than a
long positional tuple.)

### `key=value` modifiers

These pass straight through to `PatchCloud.refine_normals`, reusing the binding's
own defaults — the "Default" column matches each binding default exactly, so the
CLI re-specifies nothing and the two layers cannot drift.

| Key                 | Default         | Forwards to                       |
|---------------------|-----------------|-----------------------------------|
| `angular_range_deg` | `25`            | `refine_normals` (search-cone half-extent, deg) |
| `init_steps`        | `7`             | `refine_normals` (coarse grid resolution per axis) |
| `refine_levels`     | `3`             | `refine_normals`                  |
| `resolution`        | `24`            | `refine_normals` (R×R patch grid) |
| `objective`         | `robust`        | `refine_normals` (`robust`/`mean`)|
| `robust_iters`      | `3`             | `refine_normals`                  |
| `search_robust_iters`| `none`         | `refine_normals` (cheaper search-only objective; see below) |
| `window`            | `gaussian_disk` | `refine_normals`                  |
| `window_sigma`      | `0.6`           | `refine_normals`                  |
| `sampler`           | `bilinear_mip`  | `refine_normals` (`bilinear`/`bilinear_mip`/`anisotropic`; `bilinear_mip` takes one bilinear tap from the mip level nearest the warp's compression, bounding the aliasing `bilinear` suffers on cross-scale views at the same cost; `anisotropic` resolves oblique footprints at 1.6–3× the cost) |
| `min_valid_fraction`| `0.6`           | `refine_normals`                  |
| `min_views`         | `3`             | `refine_normals`                  |
| `cache`             | `fronto`        | `refine_normals` candidate scoring (`off`/`fronto`; see below) |
| `cache_supersample` | `2.0`           | `refine_normals` (fronto base density, ≥ 1) |
| `quality`           | `none`          | preset for `cache`/`cache_supersample` (`none`/`coarse`/`fine`) |
| `confidence`        | `false`         | `refine_normals` (compute + report the Φ-peakedness; see below) |
| `bitmaps`           | `true`          | render + persist the per-point RGBA patch bitmaps (below); `bitmaps=false` skips the render |

Unknown keys, malformed `key=value` tokens (no `=`, empty key), or out-of-range
values raise `click.UsageError`, consistent with the other parsers in
`xform/_arg_parser.py`.

That table is the whole key set (`_REFINE_NORMALS_KEYS` in
`xform/_arg_parser.py`). Frame sizing and seeding are deliberately not in it:
`extent`, `extent_value` and `normal` are `--to-embedded-patches` keys, because
that is the step which builds the frame this one reuses (see "Patch frame"
below). Persisting the frame is not a knob either — the refined cloud is always
written back, so the stored frame cannot fall out of step with the rewritten
normals. `bitmaps` is the only persistence choice here, and it governs the RGBA
textures alone.

### Candidate-scoring cache (`cache` / `cache_supersample` / `quality`)

`cache=fronto` (**the default**, with `cache_supersample=2`) selects the
fronto-parallel patch cache (`specs/core/patch/fronto-parallel-patch-cache.md`):
instead of re-rendering every candidate normal from the source images, it renders
one supersampled fronto-parallel base per view up front and affine-resamples each
candidate from it — ~2× faster at Φ-equivalent median accuracy, trading a little
tail accuracy on flat-`Φ` data. The reported photoconsistency is always
source-scored in the final pass regardless of the cache. `cache_supersample`
(≥ 1) renders the base denser than the candidate grid to sharpen the resample.
`cache=off` is the exact source-rendering path (opt in for the tightest tail).

`quality` is a convenience preset: `coarse` → `cache=fronto,
cache_supersample=2` (the default operating point); `fine` → `cache=off` (exact).
A non-`none` preset **overrides** any explicit `cache`/`cache_supersample` so the
two never disagree; `quality=none` (default) defers to the explicit knobs.

### Search-only objective (`search_robust_iters`)

The coarse-to-fine search evaluates the consensus `Φ` for every candidate normal
(the bulk of the work), but only *ranks* candidates with it — the final pass
re-scores the surviving winners with the full `objective` and is what the
reported `photoconsistency` reflects. `search_robust_iters` lets the search use a
cheaper objective than that final pass: `none` (the default) searches with the
same `objective`; `0` searches with the unweighted mean-pairwise consensus
(`objective=mean`, the cheapest); `k ≥ 1` searches with `robust` IRLS at `k`
iterations. Because the final pass is unchanged, the reported `Φ` stays honest —
this only trades a little tail accuracy in the *found normal* for a faster search.
It only helps when `objective=robust`: under `objective=mean` the final pass is
already the cheapest objective, so the knob has no benefit (and `k ≥ 1` would make
the search *dearer* than the final pass).

### Confidence (`confidence`)

`confidence=true` computes the per-patch Φ-peakedness (a curvature stencil around
the optimum) and includes a `… N low-confidence` count in the summary. It is
**off by default**: the stencil is an extra un-cached source-render pass per patch
(~1/6 of the cached runtime) and is purely informational — refinement does not
persist it. When off, the summary omits the low-confidence count and the
`refine_normals` `confidence` array is `NaN`. A patch counts toward the reported
count when its normalized confidence is below a fixed reporting threshold of
`0.1` (`_LOW_CONFIDENCE_THRESHOLD` in `xform/_refine_normals.py`); the count is
diagnostic only, with no per-point gating (see "Confidence is report-only"
below).

**Not surfaced in v1.** The `refine_normals` binding also accepts tuning knobs
(e.g. for the non-default window/sampler combinations) that are left at their
binding defaults and **not** exposed as CLI keys for v1 — they add noise to the
common case. Add them later if a use case appears; the parser rejects unknown
keys until then.

## Patch frame: built upstream, reused here

Refinement needs a per-point patch frame — a starting normal and a world-space
patch size. On an `embedded_patches` reconstruction that frame is **already
stored**, so `--refine-normals` reads it back (`recon.patches`) and never builds
one. The frame's seeding (`normal`) and sizing (`extent` / `extent_value`) are
therefore set by the upstream `--to-embedded-patches` step — see
[xform-to-embedded-patches](xform-command.md) and the core pipeline spec
[sift-to-patch-reconstruction.md](../../../core/patch/sift-to-patch-reconstruction.md). The
refine pass only rotates each stored normal toward the photometric optimum (it
re-seeds from the stored normal plus the routine's internal mean-viewing seed;
refinement is intentionally **not idempotent** — a second `--refine-normals` can
improve further). It does **not** resize the patch.

Because the rotation changes the frame, the cloud that comes back is always
re-persisted: the stored `u`/`v` pair spans the patch plane, so leaving the old
frame in place would put it at odds with the normal just written beside it. That
is why the frame is the upstream step's business and its consistency is this
step's — sizing and seeding are knobs on `--to-embedded-patches`, while keeping
the frame true to the refined normal is unconditional here. An
`embedded_patches` input always carries the frame; `apply` still raises a
`ValueError` when `recon.patches` is `None`, naming `--to-embedded-patches` as
the fix.

## Which points are refined

- **Finite points only.** `--refine-normals` masks the stored cloud to finite
  points (`finite = ~recon.point_is_at_infinity[cloud.point_indexes]`) and scatters
  only those refined normals back; points at infinity (`w = 0`) keep their stored
  normal. The mask is required: the copy-and-scatter write-back would otherwise
  overwrite an infinity point's `(0, 0, 0)` normal with the skipped patch's
  `normalize(-d)`. Leaving them untouched is also the right behavior — a point at
  infinity has a fixed outward normal (`normalize(-d)`, set by its direction), so
  there is nothing to refine. `refine_patch_normal` itself also skips `w = 0`
  patches and returns their frame unchanged (it never rotates them).
- **Degenerate / low-view points keep their seed.** The core routine skips a
  patch that has fewer than `min_views` valid views (and honors
  never-worse-than-its-own-init for the rest), so those points retain their
  seed normal rather than getting a garbage one. v1 does **not** gate on the
  confidence value, so a low-confidence-but-improved normal is still written.

**Confidence is report-only (decided).** The routine returns a per-point
confidence, but the `.sfmr` has nowhere to store it, so for now it is summarized
in the CLI output (e.g. a low-confidence count) and **not** otherwise acted on —
no per-point gating and no format change. A later `keep_below_confidence=`
threshold (keep the initial normal where confidence is low) or a per-point
confidence field in the format can be added without disturbing this design if a
need arises; neither is in scope here.

## Write-back semantics

```python
class RefineNormalsTransform:
    def apply(self, recon):  # recon is embedded_patches (gate-enforced)
        images = [load_full_res(workspace_dir / name) for name in recon.image_names]
        cloud = recon.patches            # the stored per-point frame; not rebuilt
        point_indexes = cloud.point_indexes
        finite = ~recon.point_is_at_infinity[point_indexes]   # refiner skips infinity
        result = cloud.refine_normals(
            recon, images, use_stored_keypoints=True, angular_range_deg=..., ...)
        normals = np.asarray(recon.normals, np.float32).copy()  # (P, 3)
        normals[point_indexes[finite]] = result["normal"][finite]
        # The frame is always re-persisted (it must match the rewritten normals).
        return recon.clone_with_changes(normals=normals, patches=cloud)
```

`recon.normals` is always point-count-sized (each point carries a `normal`), so
the copy-and-scatter keeps the normals of infinity points intact while
overwriting every refined finite point. `clone_with_changes` validates the
`(point_count, 3)` shape. The scatter relies on `cloud.point_indexes` being
**point-array indices** (the 3D-point index, not a track id) — which is what
`recon.patches` emits and what the binding range-checks — so `normals[pid] = …`
indexes the right row directly. The `finite` mask is keyed off the cloud's own
`point_indexes`, so it stays aligned with both `point_indexes` and the per-patch
`result["normal"]` rows.

**Persisting the patch cloud (always).** The refined `PatchCloud` is always
attached to the reconstruction via `clone_with_changes(patches=cloud)` and
written as a per-point patch frame beside the normals in the `.sfmr` `points3d/`
section (format version 3+; see `specs/formats/sfmr-file-format.md`): two
in-plane half-extent vectors `u` and `v` per point (the patch centre is the
point's position, and `normalize(u × v)` is the per-point normal scattered
above). There is no opt-out — the stored frame must stay consistent with the
rewritten normals.

**Persisting the patch bitmaps (`bitmaps`).** With `bitmaps` (the default) the
command additionally renders each refined patch's canonical RGBA texture at the
found normal and writes the per-point `patch_bitmaps_y_x_rgba` array
(`(point_count, R, R, 4)` uint8, `R = resolution`) beside the frame. The binding
(`PatchCloud.refine_normals(render_bitmaps=True)`) returns the textures already
scattered to per-3D-point rows (zero rows for points with no refined patch), and
the command attaches them via `clone_with_changes(patch_bitmaps=…)`. Each patch
texture is the cross-view **fusion** of the kept views at the optimum: RGB is the
robust IRLS-weighted mean (the same per-view weights the consensus uses; an
unweighted mean under `objective=mean`), and the **alpha channel is a per-pixel
cross-view agreement confidence** — high where the views agree, `0` where no kept
view covers the pixel (and `0` for a pixel seen by a single view, which carries no
cross-view evidence). Rendering costs one extra full-grid source render per kept
view per patch. It is on by default so the refined reconstruction carries its
per-point patch textures and can display them without re-rendering; a multi-stage
pipeline can pass `bitmaps=false` on intermediate stages to skip the redundant
render and render once on the finalizing stage.

**Persisting the refined normals (decided).** `.sfmr` *write* used to recompute
the per-point normals from geometry (the mean-viewing normals; see
`sfmr-format/src/depth_stats.rs`) on every save, which would silently discard the
refinement. The write path now **preserves** every stored normal and recomputes
only the *missing* ones — the zero vector left for points whose normal was never
set and for degenerate / infinity points (`merge_preserving_normals` in
`sfmr-format/src/write.rs`, gated on `MISSING_NORMAL_NORM_SQ`). Depth statistics
and histograms are still recomputed so they track the current geometry (e.g.
after a prior `--bundle-adjust`). This is a global save-pipeline change, not
special-cased to this command: a freshly imported reconstruction (normals start
all-zero) still gets a full set computed on its first write, while any normals a
consumer set — refined here, or otherwise — survive subsequent saves. So
`RefineNormalsTransform.apply` just writes the normals back via
`clone_with_changes` and the ordinary `recon.save` keeps them; no save flag is
needed.

Image loading resolves `workspace_dir / image_name` exactly as
`RemoveLargeFeaturesFilter` resolves its `.sift` paths (via `recon.workspace_dir`
/ `recon.image_names`). A reconstruction whose workspace is not resolvable, or a
missing image, is a hard error (`FileNotFoundError`) — the same "artifacts must
still be present where the reconstruction was created" contract the SIFT-reading
ops carry.

The transform should print a one-line summary in the established `xform` style,
e.g.:

```
  Refined N normals (mean Φ 0.71 → 0.78, +0.07; M improved, K low-confidence)
```

reporting refined-count, mean `init_photoconsistency` → `photoconsistency`, how
many strictly improved, and how many fell below a (reported-only) confidence
threshold — all available in the returned dict.

## Ordering and interactions

As with every `xform` operation, `--refine-normals` is applied at exactly its
position in the command line — the pipeline never reorders. The points below are
**recommendations** about where to place it for the best result, not behavior
the op enforces.

- **Order relative to `--bundle-adjust`.** BA moves points and poses; whichever
  geometry is current when `--refine-normals` runs is the geometry it refines
  against (BA does not re-refine normals afterward). Both orders are allowed.
- **Invariant to global similarity.** `--rotate` / `--translate` / `--scale`
  move points and poses together, so the projected patches — and thus the
  photometric consensus — are unchanged; ordering relative to those is
  immaterial to the result (only to cost). Refinement does **not** need an
  `--align-to-input` follow-up: it introduces no frame change.
- **Cheaper after filters.** Filters that shrink the cloud
  (`--remove-short-tracks`, range/glob filters) reduce the patch count and
  therefore the work, so placing `--refine-normals` after them is faster — a
  cost consideration only; the refined normals of any surviving point are the
  same either way.
- **Repeatable.** Because refinement is not idempotent, `--refine-normals
  --refine-normals` runs two passes (the second starting from the first's stored
  normals), each driving Φ further toward the continuous optimum — the
  "thorough" setting. Documented, not special-cased.
- **Images must be resolvable.** The op fails fast if `workspace_dir` or any
  image is missing, exactly like the `.sift`-reading ops.

## Performance and memory

**Compute.** Work scales as roughly `points × seeds · refine_levels ·
init_steps² · V` patch renders (`V` = observing views per point); on the
17-image seoul_bull set it is seconds, but it grows with the point count and the
per-point view count, so dense scenes are correspondingly more expensive. The
GIL is released during refinement (`py.detach`), so it parallelizes across
points internally. The summary line should note the point and image counts up
front.

**Memory.** The binding loads **all** full-resolution images into memory at once
(`images` must be parallel to every reconstruction image) and builds a pyramid
per image; the core routine borrows these read-only for the whole run, so the
decoded images plus their pyramids are the resident footprint (≈ the total
decoded pixel volume × ~1.33 for the pyramid). This is a known limitation —
there is no streaming/tiling of the image set in v1.

**Disk.** Read-only: the op reads each source image once from the workspace
(`workspace_dir / image_name`); it needs no `.sift` files (the patch frame is
already stored on the `embedded_patches` input). It writes only the output `.sfmr`.

Unlike `scripts/patch_crossval.py`, the CLI op refines the **whole** cloud (no
`point_indexes` subset) — the subset argument exists for the strip renderer's
"only the displayed tracks" case, which has no CLI analogue.

## Integration points

The transform is `RefineNormalsTransform` in
[`_refine_normals.py`](../../../../src/sfmtool/xform/_refine_normals.py) — image
loading and workspace resolution mirroring `RemoveLargeFeaturesFilter`, then a
normal scatter-write. It is exported from `xform/__init__.py`, parsed out of the
`key=value` string by `xform/_arg_parser.py`, and declared on the CLI in
[`_commands/xform.py`](../../../../src/sfmtool/_commands/xform.py) as
`@click.option("--refine-normals", is_flag=False, flag_value="", multiple=True,
…)`, which is what gives it a `--help` entry and unknown-option rejection.
[xform-command.md](xform-command.md)'s Optimization subsection links here, the
pattern `--find-points-at-infinity` and `--include-by-distribution` use.

[`tests/xform/test_refine_normals.py`](../../../../tests/xform/test_refine_normals.py)
runs it over the real 17-image seoul_bull reconstruction and asserts the point
count is unchanged, normals change for finite points, infinity points pass
through, and mean Φ does not decrease; the arg-parser grammar and its error cases
are unit-tested alongside.

## Open questions

- **Subset / progress.** Long runs on dense clouds may want a progress readout
  or a sampling option; deferred until a user hits it.
