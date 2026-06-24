# `sfm xform --refine-normals` Design

**Status:** Implemented (2026-06-13) in `src/sfmtool/xform/_refine_normals.py`
(`RefineNormalsTransform`), wired through `xform/_arg_parser.py`
(`parse_refine_normals_params`) and the `--refine-normals` Click option in
`_commands/xform.py`. Surfaces the photometric patch-normal refinement
(`specs/core/patch-normal-refinement.md`, v1 implemented in
`sfmtool-core/src/patch/normal_refine.rs` + the `PatchCloud.refine_normals`
PyO3 binding) as an `sfm xform` operation that rewrites a reconstruction's
per-point `normals` in place, and can optionally persist the full refined patch
cloud (`save_patches`).

> _Note (2026-06-13): the Click option is declared `is_flag=False,
> flag_value=""` so the optional value accepts all three forms — bare
> `--refine-normals`, space-separated `--refine-normals init_steps=7`, and
> joined `--refine-normals=init_steps=7` — while still leaving a following
> option (e.g. `--refine-normals --bundle-adjust`) untouched. The ordered
> `_arg_parser` walk mirrors that tokenization. The reported low-confidence
> count uses a fixed reporting threshold (0.1) on the normalized confidence; it
> is diagnostic only (no per-point gating), as specified below._

## Why this fits `xform` (and how)

`xform` is the reconstruction-in / reconstruction-out pipeline. Normal
refinement reads a reconstruction and produces a reconstruction with the same
points, poses, and cameras but **better surface normals**, so it slots in
naturally — with two characteristics that shape the design:

1. **It is a modifier, not a point filter.** Unlike `--remove-short-tracks`
   et al. it removes no points and changes no positions; it only rewrites the
   per-point `normal` field (stored in the `.sfmr` version 3+ as `normals_xyz`,
   surfaced as `recon.normals` and writable via
   `recon.clone_with_changes(normals=...)`). With `save_patches` it additionally
   attaches the refined patch geometry to the reconstruction. It therefore belongs
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

The result: a transform that builds a `PatchCloud` from the reconstruction,
calls `PatchCloud.refine_normals(recon, images, …)`, and writes the refined
normals back onto the points.

## Command syntax

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
--refine-normals resolution=32,initial_normals=geometric
```

(This differs from the older `xform` mini-DSLs — `--remove-isolated
factor,value_spec`, `--include-by-distribution COUNT[,verbose]` — which lead
with positionals; with a dozen knobs here, all-`key=value` reads better than a
long positional tuple.)

### `key=value` modifiers

These pass straight through to `PatchCloud.refine_normals` /
`PatchCloud.from_reconstruction`, reusing the binding's own defaults — the
"Default" column matches each binding default exactly (including
`initial_normals=stored`, which is the `from_reconstruction` `normal` default),
so the CLI re-specifies nothing and the two layers cannot drift.

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
| `sampler`           | `bilinear`      | `refine_normals` (`bilinear`/`anisotropic`) |
| `min_valid_fraction`| `0.6`           | `refine_normals`                  |
| `min_views`         | `3`             | `refine_normals`                  |
| `cache`             | `fronto`        | `refine_normals` candidate scoring (`off`/`fronto`; see below) |
| `cache_supersample` | `2.0`           | `refine_normals` (fronto base density, ≥ 1) |
| `quality`           | `none`          | preset for `cache`/`cache_supersample` (`none`/`coarse`/`fine`) |
| `confidence`        | `false`         | `refine_normals` (compute + report the Φ-peakedness; see below) |
| `initial_normals`   | `stored`        | `from_reconstruction` normal policy (below) |
| `extent`            | `feature_size`  | `from_reconstruction` extent policy (`feature_size`/`fixed`/`relative_spacing`/`pixel_size`) |
| `extent_value`      | `10.0`          | `from_reconstruction` (full patch size; halved to the library half-extent) |
| `save_patches`      | `false`         | persist the full refined patch cloud, not just per-point normals (below) |
| `bitmaps`           | `false`         | also render + persist the per-point RGBA patch bitmaps (implies `save_patches`; below) |

Unknown keys, malformed `key=value` tokens (no `=`, empty key), or out-of-range
values raise `click.UsageError`, consistent with the other parsers in
`xform/_arg_parser.py`.

### Candidate-scoring cache (`cache` / `cache_supersample` / `quality`)

`cache=fronto` (**the default**, with `cache_supersample=2`) selects the
fronto-parallel patch cache (`specs/core/fronto-parallel-patch-cache.md`):
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
`refine_normals` `confidence` array is `NaN`.

**Not surfaced in v1.** The bindings also accept `k_neighbors` (for
`initial_normals=geometric`), `pixel_reduce` (for `extent=pixel_size`), and
`feature_reduce` (for `extent=feature_size`). These are intentionally left at
their binding defaults (`12`, `min`, `median`) and **not** exposed as CLI keys
for v1 — they only matter for the non-default policies and add noise to the
common case. Add them later if a use case appears; the parser should reject them
as unknown keys until then.

## Initial normals (`initial_normals=`)

`from_reconstruction` needs a starting normal per patch; that starting normal is
also one of the search seeds. (The parameter is named `initial_normals`, not
`seed`, to avoid any confusion with a random-number seed — there is no RNG in
the search.) Choices map to `PatchNormal`:

- `stored` *(default, decided)* — refine the reconstruction's existing
  `normals` (falls back to mean-viewing where a stored normal is
  zero/degenerate). This makes `--refine-normals` read as "improve the normals
  already in this model," and lets a second `--refine-normals` in the chain
  continue from the first (refinement is intentionally **not idempotent** — each
  pass re-seeds and can improve further; see the core spec). `stored` is the
  right default precisely because the op's job is to *improve what's there*; the
  core routine still adds its own mean-viewing seed internally, so starting from
  `stored` never does worse than a fresh mean-viewing start. This is now also the
  `PatchCloud.from_reconstruction` `normal` default, so the CLI and the binding
  agree.
- `mean_viewing` — seed every patch from the mean viewing direction, ignoring
  whatever is stored. Useful when the stored normals are absent or untrusted.
- `geometric` — local PCA plane fit over k nearest points (`k_neighbors`
  defaults to the binding's `12`; exposed as `k_neighbors=` if needed).

The core routine internally also tries the mean-viewing seed regardless of this
choice, so `initial_normals=` selects the *cloud's* initial normal (the one
written back for points that are skipped or not improved), not the only basin
searched.

## Patch sizing (`extent=`)

Refinement needs a world-space patch size per point. `extent_value` is the
**full** patch size — the whole edge length of the surfel — in the chosen
policy's units. The CLI deliberately speaks in full size rather than the
half-extent the library and on-disk format store: a CLI user should not have to
reason in radii. The transform halves `extent_value` to the library half-extent
before calling `PatchCloud.from_reconstruction`, so `extent_value=10`
(`feature_size`, the default) is the same patch the library produces for
`extent_value=5.0`.

The default is `extent=feature_size` (full factor `10.0`, median over views) —
**decided**: it sizes each patch from the observing keypoints' SIFT scales,
tying the patch to the real feature support, which is the right size for a
photometric match. This **requires the workspace `.sift` files**, the same
dependency as `--remove-large-features`; a point with no readable scale in any
view is an error (`PatchCloudError::MissingFeatureScale`, surfaced as
`ValueError`). The `.sift`-requiring default is intentional — `--help` should
call this out prominently, the same "artifacts must still be present" caveat the
SIFT filters carry. Operators who do not have `.sift` present (or want size
decoupled from features) can opt into `extent=relative_spacing`,
`extent=pixel_size` (full diameter in pixels, the CLI spelling of the library's
`pixel_radius` policy), or `extent=fixed`, none of which read `.sift`.

## Which points are refined

- **Finite points only.** This command **opts out** of the default by building
  its cloud with `PatchCloud.from_reconstruction(..., exclude_points_at_infinity=True)`,
  so only finite points are refined; points at infinity (`w = 0`) are excluded and
  their stored normals pass through unchanged. The opt-out is required: the
  copy-and-scatter write-back below would otherwise overwrite an infinity point's
  `(0, 0, 0)` normal with the skipped patch's `normalize(-d)`. Leaving them
  untouched is also the right behavior — a point at infinity has a fixed outward
  normal (`normalize(-d)`, set by its direction), so there is nothing to refine.
  Even if an infinity-bearing cloud were passed, `refine_patch_normal` skips
  `w = 0` patches and returns their frame unchanged (it never rotates them).
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
    def apply(self, recon):
        images = [load_full_res(workspace_dir / name) for name in recon.image_names]
        # extent_value is the full CLI size; the library takes a half-extent.
        cloud = PatchCloud.from_reconstruction(
            recon, normal=initial_normals,
            extent=_to_library_policy(extent),  # pixel_size -> pixel_radius
            extent_value=extent_value / 2.0, ...)
        result = cloud.refine_normals(recon, images, angular_range_deg=..., ...)
        normals = np.asarray(recon.normals, np.float32).copy()  # (P, 3)
        for i, pid in enumerate(cloud.point_ids):
            normals[pid] = result["normal"][i]
        if save_patches:
            return recon.clone_with_changes(normals=normals, patches=cloud)
        return recon.clone_with_changes(normals=normals)
```

`recon.normals` is always point-count-sized (each point carries a `normal`), so
the copy-and-scatter keeps the normals of excluded (infinity) points intact while
overwriting every refined finite point. `clone_with_changes` validates the
`(point_count, 3)` shape. The scatter relies on `cloud.point_ids` being
**point-array indices** (the 3D-point index, not a track id) — which is what
`from_reconstruction` emits and what the binding range-checks — so
`normals[pid] = …` indexes the right row directly.

**Persisting the full patch cloud (`save_patches`).** By default only the
per-point normal is written. With `save_patches=true`, the refined `PatchCloud`
is attached to the reconstruction via `clone_with_changes(patches=cloud)` and
written as a per-point patch frame beside the normals in the `.sfmr` `points3d/`
section (format version 3+; see `specs/formats/sfmr-file-format.md`): two
in-plane half-extent vectors `u` and `v` per point (the patch centre is the
point's position, and `normalize(u × v)` is the per-point normal scattered
above).

**Persisting the patch bitmaps (`bitmaps`).** With `bitmaps=true` the command
additionally renders each refined patch's canonical RGBA texture at the found
normal and writes the per-point `patch_bitmaps_y_x_rgba` array
(`(point_count, R, R, 4)` uint8, `R = resolution`) beside the frame. The binding
(`PatchCloud.refine_normals(render_bitmaps=True)`) returns the textures already
scattered to per-3D-point rows (zero rows for points with no refined patch), and
the command attaches them via `clone_with_changes(patch_bitmaps=…)`. Because the
bitmaps require the frame, `bitmaps=true` forces `save_patches` on. Each patch
texture is the cross-view **fusion** of the kept views at the optimum: RGB is the
robust IRLS-weighted mean (the same per-view weights the consensus uses; an
unweighted mean under `objective=mean`), and the **alpha channel is a per-pixel
cross-view agreement confidence** — high where the views agree, `0` where no kept
view covers the pixel (and `0` for a pixel seen by a single view, which carries no
cross-view evidence). Rendering costs one extra full-grid source render per kept
view per patch, so it is off by default.

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

- **Recommended after `--bundle-adjust`.** BA moves points and poses; normals
  are most accurate when refined against the final geometry, so
  `--bundle-adjust … --refine-normals` is the suggested sequence. Placing it
  *before* BA is allowed and simply refines against the pre-BA geometry (and BA
  does not re-refine normals afterward).
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
  --refine-normals` runs two passes (the second starting from the first via
  `initial_normals=stored`), each driving Φ further toward the continuous optimum — the
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
(`workspace_dir / image_name`) and, under the default `extent=feature_size`, the
`.sift` files. It writes only the output `.sfmr`.

Unlike `scripts/patch_crossval.py`, the CLI op refines the **whole** cloud (no
`point_ids` subset) — the subset argument exists for the strip renderer's
"only the displayed tracks" case, which has no CLI analogue.

## Integration points

- New module `src/sfmtool/xform/_refine_normals.py` with
  `RefineNormalsTransform` (image loading + workspace resolution mirroring
  `RemoveLargeFeaturesFilter`, normal scatter-write).
- Export it from `xform/__init__.py`; add the `--refine-normals` branch to
  `xform/_arg_parser.py` (`key=value` parsing) and a Click
  `@click.option("--refine-normals", multiple=True, …)` to
  `_commands/xform.py` for `--help` / unknown-option rejection.
- Add a short **Optimization** subsection to `specs/cli/xform-command.md`
  linking here (the pattern used by `--find-points-at-infinity` and
  `--include-by-distribution`).
- Tests: a pytest over the real 17-image seoul_bull reconstruction asserting
  the point count is unchanged, normals change for finite points, infinity
  points pass through, and mean Φ does not decrease (mirrors the existing
  `refine_normals` integration test); plus arg-parser unit tests for the
  `key=value` grammar and error cases.

## Open questions

- **Subset / progress.** Long runs on dense clouds may want a progress readout
  or a sampling option; deferred until a user hits it.
