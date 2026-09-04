# `sfm estimate-intrinsics` Command

## Overview

Estimates a shared focal length and camera-model family for a set of images
from their cluster matches alone -- no reconstruction is run. The command is
the CLI face of the structure-free focal vote
([`focal-vote.md`](../../core/geometry/focal-vote.md)): image pairs vote
through whichever estimator their geometry can observe, the pooled log-median
is the focal consensus, and the camera-model columns arbitrate pinhole
against equidistant fisheye on the certified mass of model-informative scan
votes. Because no structure is estimated, the answer cannot be biased by the
depth/focal (bas-relief) compensation of structure-based estimation.

The output is a report; with `--write-camrig` the command also commits the
estimate to the workspace as a one-sensor `.camrig`
([camrig-file-format.md](../../formats/camrig-file-format.md)), the same
shape `sfm camrig create` writes -- which `sfm solve` auto-discovers as the
intrinsics prior for the images its stored pattern matches, taking
precedence over `camera_config.json`.

## Command Syntax

```bash
sfm estimate-intrinsics -i MATCHES [OPTIONS...]
```

| Option | Default | Description |
|---|---|---|
| `-i, --input MATCHES` | required | Cluster-bearing `.matches` file (from `sfm match --cluster`) |
| `--model [auto\|pinhole\|fisheye]` | `auto` | Which camera-model columns to run: `auto` runs both and arbitrates; the named forms run one column and skip arbitration |
| `--write-camrig PATH` | off | Write the estimate as a one-sensor `.camrig` at PATH (see below) |
| `--pattern PATTERN` | derived | Image pattern stored in the `.camrig`; defaults to the matches' common image directory plus one `*` segment per directory level below it, ending `*<ext>` for the extension the image names share (`*` when they mix) |
| `--force` | off | Allow `--write-camrig` to overwrite an existing file |
| `--json` | off | Emit the full vote result as JSON on stdout instead of the report |
| `--seed N` | `0` | RANSAC / pair-sampling seed; same inputs + seed give bit-identical output |

## Behavior

The command reads the cluster tracks and image dimensions from the
`.matches` file and hands them to the
`sfmtool._sfmtool.geometry.estimate_intrinsics` binding
([estimate-intrinsics.md](../../core/geometry/estimate-intrinsics.md)) with
the column set implied by `--model`; the verdict, its confirmation and the
focal are read off that result, and the raw vote it nests under `vote`
supplies the report's diagnostics. The whole file's admission votes -- the vote is a referee over
the capture's full pair graph, and restricting it is the caller's job (pass
a smaller `.matches`), not this command's.

The vote assumes one shared camera with a centred principal point. A
`.matches` file whose images carry more than one distinct `(width, height)`
is rejected up front with a message naming the differing dimensions; split
the images into per-camera match files to estimate each.

### The model verdict and its confirmation

Under `--model auto` the binding's verdict (the column with the greater
certified mass of model-informative scan votes) is reported together with a
**confirmation** reading: a Fisheye verdict is CONFIRMED only when the
equidistant column carries nonzero certified rotation-cell mass. The rule is
structural rather than a threshold -- a wrong ray map cannot fake a pure
rotation of rays, so a fisheye verdict without rotation-cell mass is an
arbitration artifact, not a lens. Both the verdict and its confirmation come
from the estimate; the command holds no rule of its own and the kernel runs
once.

An unconfirmed Fisheye verdict still prints, marked UNCONFIRMED, and the
report recommends treating the capture as pinhole.

### Report

The human-readable report leads with the answer and keeps the diagnostics
below it:

- model verdict (with CONFIRMED / UNCONFIRMED under `auto`), the winning
  column's consensus focal in pixels, and the implied diagonal field of view
  under that model's map;
- vote counts (`n_epipolar`, `n_rotation`, `n_pool`), the pool spread, and
  the family disagreement where both families voted;
- under `auto`, one line per column with its own consensus focal, spread,
  and certified epipolar / rotation masses.

`focal_px = None` (fewer than 2 pooled votes) reports as "no consensus" with
the rejection counters (`n_h_dominated`, `n_estimator_failed`,
`n_band_rejected`), so the user can see whether the capture is match-starved
or parallax-poor. No consensus is still exit code 0 -- the report is the
product; only I/O failures and rejected inputs are errors.

`--json` emits the vote dict verbatim at the top level, for scripting, plus
four keys the estimate supplies: `fisheye_confirmed`,
`certified_rotation_mass`, `diagonal_fov_deg`, and `verdict_votes` (the
winning column's certified scan votes -- the evidence behind this verdict,
which the top-level `epipolar_votes` / `rotation_votes` are not, those always
describing the pinhole closed-form kernel). The vote's keys stay at the top
level rather than nesting under `vote` as the binding does: they are already
the payload, and nesting them as well would duplicate every one.

### `--write-camrig`

Writes a one-sensor `.camrig` whose camera is the verdict model
(`SIMPLE_PINHOLE` or `EQUIDISTANT_FISHEYE`) at the consensus focal, the
matches' `width` / `height`, and the centred principal point, with identity
sensor extrinsics -- the same shape `sfm camrig create` produces. The stored
image pattern is `--pattern` when given, else derived from the matches'
image names: their common directory, one `*` segment per directory level
between it and the names (a glob's `*` does not cross `/`, so a rig's
per-sensor subdirectories need their own segment), ending in `*<ext>` for
the extension the names share -- so a flat layout's derived pattern does
not sweep up the workspace's own files. Names at mixed depths cannot be one
glob; the derivation then keeps a single level and validation hands the
caller a message naming `--pattern`. The pattern is validated the way
`camrig create` validates one.

The write refuses -- with the report still printed -- when:

- there is no focal consensus;
- the verdict is an UNCONFIRMED Fisheye (the message says to re-run with
  `--model pinhole` to commit the pinhole estimate instead);
- the output file already exists and `--force` was not given.

## Implementation Notes

- Registered under the Reconstruction category in `cli.py`; source in
  `src/sfmtool/_commands/estimate_intrinsics.py`.
- The `.matches` read goes through `sfmtool._sfmtool.io.MatchesFile`; the
  binding wants cluster-contiguous observation arrays
  (`cluster_indexes` nondecreasing), which is the order the selection
  handle's `cluster_starts` / `member_images` arrays already deliver.
- Keypoint positions come from the file's `member_positions()` — the backbone's
  own array, whose content is whatever stage the file is at: the detections of
  a matcher output, the refinement's answer in a cluster-patches one. There is
  no other source and no fallback; a `.matches` file predating format version 6
  is refused by the reader, naming its regeneration. The values are stored
  float32 and widened here because the vote solves in float64. Member exclusion
  is the default selection's job, by status — the same rule the refined path
  has always used.
- An explicit `--model pinhole` runs no scan, so its result carries no
  per-column block (the closed-form kernel is the whole answer); the
  report's Columns section appears only where a column actually scanned.
- The fisheye diagonal FOV is `2 * theta(r_corner)` under `theta = r / f`;
  the pinhole one is `2 * atan(r_corner / f)`. Both are reported from the
  same corner radius `hypot(width, height) / 2`.
- A real fisheye lens is generally not exactly equidistant, so a calibrated
  reference under a richer distortion model can sit a few percent from this
  estimate. That gap is a property of the estimate's model, not an estimator
  error; the `.camrig` this command writes is the equidistant best-fit,
  which is what the solver-side model can represent.
