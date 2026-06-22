# Exploratory Testing Audit — 2026-06-13

Read-only snapshot from a hands-on exploratory-testing pass over the `sfm` CLI
and its Rust core, exercised against **all four checked-in datasets**. The goal
was to find reproducible Rust panics and rough/unintuitive interface edges, with
copy-pasteable reproducers for each finding. This complements the static
`audit-hygiene` / `audit-specs` reports: everything below was observed at
runtime.

## Methodology

Environment: `pixi run …` from the repo root, native extension built with
`maturin develop --release` (already current in this container).

All four datasets were reconstructed from scratch using the documented
pipelines (sfmtool SIFT → track-cluster matching → solve):

| Dataset | Images | Solver | Result |
|---|---|---|---|
| `seoul_bull_sculpture` | 17 @ 270×480 | incremental | 1 cam, 1108 pts |
| `seattle_backyard` | 26 @ 360×640 | global (GLOMAP) | 1 cam, 3346 pts |
| `kerry_park` (rig) | 24 frames × 2 fisheye | global (GLOMAP) | 2 cams, 775 pts |
| `dino_dog_toy` | 85 @ 2040×1536 | incremental | 1 cam, 19024 pts |

Reproducers below assume these workspaces exist under `/tmp/x/…` with the
reconstructions symlinked as `/tmp/x/{seoul,seattle,kerry,dino}.sfmr`, and that
`sfm` runs via `pixi run sfm` from the repo root. Setup used:

```bash
# per dataset, e.g. seoul:
mkdir -p /tmp/x/seoul_bull_ws/images
cp test-data/images/seoul_bull_sculpture/*.jpg /tmp/x/seoul_bull_ws/images/
cd /tmp/x/seoul_bull_ws
sfm ws init --feature-tool sfmtool .
sfm sift --extract -t 3 images/*.jpg
sfm match --cluster --cluster-d 28 images/ -o tvg-matches/seoul_bull.matches
sfm solve --incremental --seed 42 tvg-matches/seoul_bull.matches
```

**Overall impression:** the Python validation layer is genuinely robust. Scores
of malformed-input probes (zero-length axis vectors, scale 0 / negative,
out-of-range filters, descending/garbage range expressions, wrong file
extensions, truncated/empty/cross-typed ZIP containers, coincident measurement
points, out-of-range point indices) all produced clean, well-worded errors with
correct exit codes. The panics that *do* exist live below that layer: they are
reached when bad data slips past extension/type checks and lands in the Rust
core or PyO3 boundary without a defensive check. The common theme is **missing
validation of numeric/geometric invariants** (counts, focal length /
invertibility) before the data reaches Rust.

---

## A. Reproducible panics (Rust / PyO3)

All three surface as `pyo3_runtime.PanicException` with a Rust backtrace dumped
to the user — i.e. an uncaught crash rather than a handled error. None should be
a panic; each should be a typed Python error or an up-front validation failure.

### A1. `from-colmap-bin`: unvalidated element counts → `capacity overflow`

> _Status (2026-06-13): Done — `read_colmap_binary` now clamps every
> count-driven `Vec::with_capacity` to `file_len / min_record_bytes` via a new
> `capped_capacity` helper, so a corrupt count fails cleanly with an I/O error
> instead of panicking. Regression test `capped_capacity_clamps_to_available_bytes`._

**Severity: medium (robustness / DoS on untrusted COLMAP models).**

The COLMAP binary reader trusts the `u64` element counts embedded in the file
and passes them straight to `Vec::with_capacity`. A corrupt/hostile
`points3D.bin` (or `cameras.bin` / `images.bin`) with an absurd count panics the
allocator before any data is read.

Repro:
```bash
# Make a valid COLMAP model, then corrupt the point count in points3D.bin
sfm to-colmap-bin /tmp/x/seoul.sfmr /tmp/x/colmap_out
cp -r /tmp/x/colmap_out /tmp/x/colmap_huge
printf '\xff\xff\xff\xff\xff\xff\xff\x7f' \
  | dd of=/tmp/x/colmap_huge/points3D.bin bs=1 count=8 conv=notrunc
sfm from-colmap-bin /tmp/x/colmap_huge \
  --image-dir /tmp/x/seoul_bull_ws/images -o /tmp/x/ch.sfmr
```
Output:
```
thread '<unnamed>' panicked at .../alloc/src/raw_vec/mod.rs:28:5:
capacity overflow
  4: sfmr_colmap::colmap_io::read::read_colmap_binary
pyo3_runtime.PanicException: capacity overflow
```

Source: `crates/sfmr-colmap/src/colmap_io/read.rs` — every
`Vec::with_capacity(<count> as usize)` driven by `read_u64`:
lines **254** (`num_cameras`), **266** (`num_params`), **290** (`num_images`),
**305** (`num_points2d`), **342–346** (`num_points`), **359** (`track_length`).
A smaller-but-still-too-big count instead reads past EOF and *does* fail cleanly
(`Error: I/O error: failed to fill whole buffer`), so the panic is specifically
the unbounded `with_capacity`. Fix: sanity-cap counts against remaining bytes
(each record has a known minimum size) or use fallible/lazy allocation.

### A2. `undistort` on a camera with `focal_length == 0` → `cast_slice` alignment panic

> _Status (2026-06-13): Done — root cause was `sift-format` reading a freshly
> decompressed (1-aligned) `Vec<u8>` and reinterpreting it as `&[f32]`/`&[T]`
> via `bytemuck::cast_slice`, which panics on an unaligned buffer (a latent,
> address-dependent bug that focal=0 happened to trigger reliably).
> `read_binary_array` and `read_partial_f32_array` now copy into a properly
> aligned `Vec<T>` via `cast_slice_mut`. `undistort` of a focal=0 model now
> fails cleanly on the resulting empty data instead of crashing._

**Severity: medium.**

A reconstruction whose camera has a zero (or otherwise degenerate) focal length
makes `undistort` build a reconstruction dict that, when handed back to
`SfmrReconstruction.from_data`, fails an **unchecked `bytemuck::cast_slice`**
alignment assertion. (A focal of 0 is reachable from a malformed COLMAP import —
see repro — but the underlying gap is that neither `undistort` nor `from_data`
validates the data.)

Repro:
```bash
# Build a reconstruction with focal_length = 0 by zeroing the focal in cameras.bin
sfm to-colmap-bin /tmp/x/seoul.sfmr /tmp/x/colmap_out
cp -r /tmp/x/colmap_out /tmp/x/colmap_zf
printf '\x00\x00\x00\x00\x00\x00\x00\x00' \
  | dd of=/tmp/x/colmap_zf/cameras.bin bs=1 seek=32 count=8 conv=notrunc
sfm from-colmap-bin /tmp/x/colmap_zf \
  --image-dir /tmp/x/seoul_bull_ws/images -o /tmp/x/zf.sfmr   # imports OK, focal=0
sfm undistort /tmp/x/zf.sfmr -o /tmp/x/zf_undist
```
Output:
```
File ".../_undistort_images.py", line 571, in undistort_reconstruction_images
    new_recon = SfmrReconstruction.from_data(output_dir.resolve(), sfmr_dict)
thread '<unnamed>' panicked at bytemuck-1.25.0/src/internal.rs:33:3:
cast_slice>TargetAlignmentGreaterAndInputNotAligned
  7: _sfmtool::py_sfmr_reconstruction::...__pymethod_from_data__
pyo3_runtime.PanicException: cast_slice>TargetAlignmentGreaterAndInputNotAligned
```

Entry point: `SfmrReconstruction::from_data`
(`crates/sfmtool-py/src/py_sfmr_reconstruction.rs:51`) →
`parse_sfmr_data_from_dict`. A normal `undistort` of the same dataset (nonzero
focal) succeeds, so the degenerate camera is what tips an array into a layout
the unchecked cast can't handle. Two independent hardening points: (a)
`undistort` should reject non-finite / non-positive focal length up front; (b)
`from_data` should use a checked/allocating cast (`try_cast_slice` /
`pod_collect_to_vec`) so an unaligned or non-contiguous buffer is a typed error,
not a panic.

### A3. `densify` / feature matching with `focal_length == 0` → `Intrinsic matrix K2 must be invertible`

> _Status (2026-06-13): Done — `compute_fundamental_matrix` now returns
> `Option<Matrix3<f64>>` (`None` when an intrinsic matrix is singular) instead
> of `.expect()`-ing the inverse. `match_image_pair` returns no matches and
> `check_rectification_safe` returns `false` for such degenerate pairs.
> Regression test `fundamental_matrix_none_for_singular_intrinsics`._

**Severity: medium.**

Any path that computes a fundamental matrix inverts the camera intrinsics with
`.expect(...)`. A singular `K` (focal 0) panics. Reached here via `densify`
(which sweep-matches covisible pairs), and the same `compute_fundamental_matrix`
is used by `epipolar` and flow/match-against-reconstruction paths.

Repro (reusing `/tmp/x/zf.sfmr` from A2):
```bash
sfm densify /tmp/x/zf.sfmr /tmp/x/zfd.sfmr --max-features 512
```
Output:
```
File ".../feature_match/_core.py", line 91, in match_image_pair
thread '<unnamed>' panicked at crates/sfmtool-core/src/epipolar.rs:45:10:
Intrinsic matrix K2 must be invertible
  3: sfmtool_core::epipolar::compute_fundamental_matrix
pyo3_runtime.PanicException: Intrinsic matrix K2 must be invertible
```

Source: `crates/sfmtool-core/src/epipolar.rs:45` and `:48`
(`k2.try_inverse().expect(...)`, `k1.try_inverse().expect(...)`), plus a sibling
`svd.v_t.expect("SVD failed to compute V^T")` at `:66` in `compute_epipole`.
Fix: return `Result`/`Option` from `compute_fundamental_matrix` and surface a
typed error; validate intrinsics (finite, positive focal) at the PyO3 boundary.

> A2 and A3 share a root cause — **invalid camera intrinsics are never
> validated** before reaching Rust geometry/IO. A single up-front
> "intrinsics must be finite and have positive focal length" check (on import
> and at the PyO3 boundary) would close both, plus the COLMAP-import gap in A1
> rounds out a "harden the COLMAP/intrinsics ingress" theme.

---

## B. Interface inconsistencies (confirmed at runtime)

### B1. The three export commands use three different output conventions

> _Status (2026-06-20): Done — unified on a positional `OUTPUT`. `to-colmap-db`
> now takes `OUTPUT_DB_PATH` as a required positional (dropped `--out-db`);
> `to-nerfstudio` takes an optional positional `OUTPUT_DIR` (dropped `-o/--output`),
> keeping its `<input>_nerfstudio/` default. `to-colmap-bin` was already
> positional. Specs and tests updated. Clean break (no deprecated aliases)._

**Severity: medium (papercut, but they're a single cohesive family).**

```
sfm to-colmap-bin  INPUT_SFMR OUTPUT_DIR        # output = required positional
sfm to-colmap-db   INPUT_PATH --out-db PATH     # output = required named option
sfm to-nerfstudio  INPUT_SFMR -o/--output DIR   # output = optional named, has default
```
Observed: `sfm to-nerfstudio in.sfmr out_dir` → `Error: Got unexpected extra
argument (out_dir)`; `sfm to-colmap-db in.sfmr db.db` → `Error: Missing option
'--out-db'`. A user who learns one of these three cannot guess the other two.
Pick one convention (a positional `OUTPUT` reads best, matching `to-colmap-bin`
and `xform`).

### B2. `camrig create` reverses the input/output positional order

> _Status (2026-06-20): Done — reordered to `camrig create IMAGE_PATTERN OUTPUT_FILE`,
> matching the `INPUT … OUTPUT` convention of `xform`, `to-colmap-bin`, and
> `camrig cp`. Spec and tests updated. Clean break._

**Severity: low.**

`camrig create OUTPUT_FILE IMAGE_PATTERN` puts the output **first**, opposite to
`xform INPUT [OUTPUT]`, `densify INPUT OUTPUT`, `to-colmap-bin INPUT OUTPUT`.
Easy to invoke backwards.

### B3. `align` flattens inputs to basenames in the output dir — silent overwrite / data loss

> _Status (2026-06-20): Done — `align_command` now rejects any input set in which
> two paths share a basename (different directories, or the same file passed
> twice) up front with a clear `ClickException`, alongside the existing
> "input inside output directory" guard, before anything is written
> (`src/sfmtool/align/multi.py`). Regression tests
> `test_align_basename_collision_rejected` / `test_align_same_file_twice_rejected`;
> documented under a new "Output" section in `specs/cli/align-command.md`._

**Severity: medium (correctness, not just UX).**

`align` writes each input into `--output-dir` under its **basename only**, with
no collision check. Two inputs that share a basename (different directories,
generic names like `reference.sfmr`, or the same file) collapse into one output
file — the reference copy is overwritten by an aligned result, so the user
silently gets fewer files than inputs.

Repro:
```bash
sfm xform /tmp/x/seoul.sfmr /tmp/x/half_a.sfmr --include-range 1-10
mkdir -p /tmp/x/dirB && cp /tmp/x/half_a.sfmr /tmp/x/dirB/half_a.sfmr
sfm align /tmp/x/half_a.sfmr /tmp/x/dirB/half_a.sfmr -o /tmp/x/collide
ls /tmp/x/collide/        # -> a single half_a.sfmr; the reference copy was clobbered
```
Fix: detect basename collisions and disambiguate (e.g. suffix `-1`, `-2`) or
error out.

### B4. Options silently ignored when their companion mode isn't selected

> _Status (2026-06-20): Done — both runtime-confirmed cases now raise a
> `UsageError` (matching the existing `align`/`to-nerfstudio` option-dependency
> precedent) instead of being silently dropped. `match` rejects any
> method-specific option (`--sequential-overlap`, `--flow-preset`/`--flow-skip`,
> `--cluster-*`) passed without its companion method, detected via click's
> parameter-source so it is robust to default values; `xform` rejects
> `--max-features` without `--find-points-at-infinity`. Specs and regression
> tests updated. (RANSAC-under-`--method cameras` was already handled by `align`.)_

**Severity: low–medium (no feedback → silent wrong results).**

Confirmed cases (command accepts the flag, exits 0, ignores it):
- `sfm match --exhaustive --sequential-overlap 5 …` — `--sequential-overlap`
  only applies to `--sequential`; here it's silently dropped.
- `sfm xform in.sfmr out.sfmr --max-features 500 --scale 2` — `--max-features`
  only applies to `--find-points-at-infinity`; silently dropped.

These should either warn ("`--sequential-overlap` ignored without
`--sequential`") or error. (Several more option-dependency cases exist by static
review — `--flow-preset`/`--flow-skip` without `--flow-match`, RANSAC options
under `--method cameras`, etc.; the two above are the runtime-confirmed ones.)

### B5. `--camera-model` accepts a different model set across commands

> _Status (2026-06-20): Done — `solve`, `match`, `camrig create`, and
> `to-colmap-db` now share a single `CAMERA_MODEL_NAMES` tuple (derived from the
> canonical parameter-name table in `camera/cameras.py`), so all four accept the
> same 11 COLMAP models including `FULL_OPENCV`. `to-colmap-db`'s `--camera-model`
> was switched from free-form `str` to the shared `click.Choice` in the same
> pass (post-review follow-up), so a typo'd model name no longer forwards
> silently to COLMAP. Specs and a regression test updated._

**Severity: low.**

`camrig create --camera-model` includes `full_opencv`; `solve`/`match`
`--camera-model` do **not** (their enum is `simple_pinhole … radial …
opencv/opencv_fisheye … rad_tan_thin_prism_fisheye`). Same flag name, different
accepted vocabulary depending on the subcommand.

### B6. One `--draw`/`-o` path can emit several files

> _Status (2026-06-20): Done — the `--draw` help for both `epipolar` and `flow`
> now spells out the sibling files written (`<stem>_other<ext>` for epipolar;
> `<stem>_flow<ext>` plus `<stem>_A/_B<ext>` for flow). Help-text only; behavior
> unchanged._

**Severity: low (surprising, mostly undocumented).**

- `sfm epipolar recon.sfmr 1 2 --draw out.png` writes **`out.png` and
  `out_other.png`** (two files from one path).
- `sfm flow a.jpg b.jpg --draw out.png` writes `out_flow.png`, `out_A.png`,
  `out_B.png`.

Reasonable behavior, but a single `--draw PATH` reading as "exactly this file"
makes the extra siblings a surprise; worth a one-line note in `--help`.

---

## C. Minor observations / wording

- **`--find-points-at-infinity` summary wording is confusing.** On seoul it
  prints `[find-infinity] kept 0 finite + 7 at-infinity …` while the overall
  point count *rises* 1108 → 1115. The "kept N" line refers only to the newly
  discovered candidates (which are added on top of the existing cloud), not the
  whole reconstruction, so "kept 0 finite" next to a growing total reads as a
  contradiction. Consider "discovered 7 new points at infinity (+7)".
  > _Status (2026-06-22): Done — `infinity/discover.rs` now prints
  > `discovered N new finite + M new at-infinity points, dropped K indeterminate
  > (finite_horizon=…)`._
- **`densify` produces *fewer* points than its input** (seoul 1108 → 325; it
  re-triangulates covisible pairs rather than augmenting). Counterintuitive for
  a command named "densify". The spec already flags densify as experimental and
  poorly tuned (`specs/cli/densify-command.md`), so this is expected-but-rough;
  noting it here for the record.
- **`analyze --depth-reliability` is insensitive to broken intrinsics.** On the
  focal-0 reconstruction it reports all-zero depths (100% near-infinity) yet the
  condition-number histogram is byte-for-byte identical to the healthy
  reconstruction (median 41.1, max 4778.9). Not a crash, but the
  condition-number path appears not to reflect the degenerate camera, which
  could mask a real problem.
  > _Status (2026-06-22): Done — kept the condition number's purely-geometric
  > definition (it depends only on camera centers and ray directions, not on
  > intrinsics), but `print_depth_reliability` now scans `recon.cameras` up
  > front and prints a yellow `WARNING: N of M camera(s) have non-positive or
  > non-finite focal length …` explaining that the inverse-depth z column
  > collapses for those points and the condition number is geometry-only._
- **Same latent `cast_slice` pattern lived across the other format crates.**
  > _Status (2026-06-13): Done — the A2 alignment fix was generalized. Every
  > read path that reinterpreted a freshly decompressed `Vec<u8>` as a wider
  > type now copies into a properly aligned `Vec<T>`: `read_binary_array` in
  > `sfmr-format`, `matches-format`, and `camrig-format`, plus the structural
  > `&[u32]` casts in `sfmr-format/src/verify.rs` and
  > `matches-format/src/verify.rs`. Write-side casts (aligned ndarray → `&[u8]`)
  > are unaffected. Regression test `raw_to_u32_handles_unaligned_buffer`._

  `crates/matches-format/src/verify.rs` (≈lines 139, 170, 184–187, 314) and the
  `read_binary_array` helpers reinterpreted freshly decompressed `Vec<u8>`
  buffers as `&[u32]`/`&[T]` via `bytemuck::cast_slice`, the identical
  unaligned-buffer panic risk as A2, reachable through `sfm inspect`/verify on a
  `.matches`/`.sfmr`/`.camrig`. Not reproduced as a crash here (alignment is
  address-dependent), but given the same treatment.

---

## What was tried and held up well (no defects)

For completeness — these were probed specifically to break them and did the
right thing: `xform` with zero-axis rotate / scale 0 / scale negative / garbage
scale / 3-component rotate; `--include-range` with `999-1000` / `abc` / `5-1` /
`1,,3`; filters that empty the reconstruction (`--remove-short-tracks 9999`,
`--remove-narrow-tracks 179deg`); truncated/empty/wrong-extension and
cross-typed ZIP containers (`.sift`↔`.matches`↔`.sfmr`, image-as-`.sfmr`);
`scale-by-measurements` with coincident points (clean "points are coincident")
and out-of-range indices; `align`/`compare` of non-overlapping and identical
reconstructions; `align`/`merge` of split-then-realigned halves; `epipolar` of
an image against itself; `motion`, `panorama`, `heatmap`, `to-colmap-*`,
`camrig create`/`inspect`, and all six `analyze` modes across the regular and
rig datasets. Type/extension validation and "exactly one mode" checks
(`solve -i/-g`, `match`, `analyze`) are consistent and well-worded.

---

## Suggested follow-ups (priority order)

> _Status (2026-06-22): All numbered follow-ups (1–4) closed. A1–A3 closed
> earlier; B1/B3 (export conventions + align collision) and B4 (ignored-option
> errors) done; B2/B5/B6 papercuts done; and the two remaining Section C wording
> items — the `--find-points-at-infinity` summary reword and the
> `analyze --depth-reliability` broken-intrinsics warning — landed this run.
> The `densify`-shrinks-the-cloud note from Section C is documented as
> expected-but-rough in `specs/cli/densify-command.md` (it stays a feature note,
> not a fix item)._

1. **Harden the intrinsics/COLMAP ingress (A1–A3).** Validate finite, positive
   focal length and invertible `K` on import and at the PyO3 boundary; bound
   `with_capacity` counts in the COLMAP reader; make `compute_fundamental_matrix`
   and `from_data` fallible instead of `expect`/unchecked-cast. One focused PR
   closes all three panics.
2. **Unify export output conventions (B1)** and fix the `align` basename
   collision (B3) — the latter is a quiet data-loss footgun.
3. **Warn on silently-ignored mode-dependent options (B4).**
4. Wording/papercuts (B2, B5, B6, Section C) as time permits.
