# `sfm cluster-patches` Command

## Overview

Refines a cluster-bearing `.matches` file into **patch clusters**: per
cluster, a reference member plus, for every other member, a photometrically
refined and vetted affine warp that maps the reference's local patch onto
that member's image. The result is written as the `cluster_patches/` section
of a **new** `.matches` file that copies the input's images and clusters
sections verbatim (write-once workflow, like adding two-view geometries).

Design: [`specs/core/patch/cluster-patches.md`](../../core/patch/cluster-patches.md).
Implementation (Rust kernel, algorithm, bindings):
[`specs/core/patch/cluster-patch-refinement.md`](../../core/patch/cluster-patch-refinement.md).
Format: the `clusters/` and `cluster_patches/` sections of
[`matches-file-format.md`](../../formats/matches-file-format.md).

## Command Syntax

```bash
sfm cluster-patches -i clusters.matches [-o out.matches] [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i, --input` | path | required | Cluster-bearing `.matches` file (from `sfm match --cluster`) |
| `-o, --output` | path | input with a `-patches` suffix | Output `.matches` path; must not already exist |
| `--patch-size` | float > 0 | 12.0 | Template size — the full patch edge length, keypoint-frame units; halved to the kernel's template half-width. The default sits at SIFT's ~12× descriptor window |
| `--resolution` | int ≥ 3 | 25 | Template samples per axis |
| `--min-zncc` | float in [−1, 1] | 0.85 | Member acceptance threshold on the achieved windowed ZNCC |
| `--max-shift` | float ≥ 0 | 3.0 | Max translation drift from the SIFT seed, px |
| `--max-keypoint-uncertainty` | float ≥ 0 | 0.35 | Localizability gate: exclude members whose own patch scores a predicted keypoint position uncertainty (`σ_pos`, template-grid px) above this, before reference selection and refinement; `0` disables |

The `patch_size` default sits at SIFT's ~12× descriptor window — the template
vets a member against roughly the texture context the detector deemed
characteristic of the feature. A `patch_size` of 4 (half-width 2) is too small
for the affine DOF; larger templates vet members more selectively — the
members a 12-unit template rejects that a smaller one accepts are
disproportionately epipolar outliers against reference poses — while sizes
past ~12 grow the fraction of members dropped unjudged because the wider
template's support leaves the frame (see `specs/core/patch/cluster-patches.md`,
"The operation"). `min_zncc` is permissive by design —
over-culling, not contamination, is the observed failure mode, and downstream
stages re-gate on the stored signals. `--max-keypoint-uncertainty` shares its
default value with `embed-patches` (the conservative tail cut of
[`patch-localizability.md`](../../core/patch/patch-localizability.md)); it is scored
on each member's own template-grid patch with the refinement window (not on
a consensus), which catches the flat/edge aperture cases that agree
photometrically yet cannot pin a 2D position.

## Process

1. **Read and gate.** `read_matches(input)`; reject unless the file carries a
   `clusters/` section (the fix is named: run `sfm match --cluster`). Reject
   when `cluster_patches/` is already present (write-once: enrich the
   original clusters file instead). Reject when the output path exists.
2. **Locate inputs.** Resolve the workspace directory from the file's
   workspace reference (relative path first, then absolute, then an ancestor
   search — each candidate must hold `.sfm-workspace.json`), and read the
   images from there. The seed geometry is the input's own: an accepted input
   is a matcher output, so its `clusters/member_positions` and
   `clusters/member_affine_shapes` are the members' detections
   ([`matches-file-format.md`](../../formats/matches-file-format.md), Member
   geometry). No `.sift` file is opened. The kernel wants per-image feature
   arrays and reads only the rows its members name, so the member values are
   scattered back to those rows, sized by `feature_counts[i]` — which presents
   them exactly as a `.sift` read would.
3. **Refine.** Load the images with cv2 (color) — decoded through a thread
   pool (cv2 releases the GIL; the embed-patches pattern), results collected
   in submission order — present the seed geometry from step 2 in
   images-section order,
   and call `_sfmtool.matching.refine_cluster_patches` (the
   `patch::cluster_refine` kernel — per-member localizability gate,
   reference selection by largest SIFT scale, Gaussian-windowed-ZNCC shift →
   similarity → affine Nelder-Mead cascade seeded from the SIFT affine
   shapes, vetting, one kept member per image), with a `ProgressCounter`
   poller reporting per-cluster progress.
4. **Write.** A new `.matches` file at the current format version: the images
   and clusters sections carried over, with the backbone's geometry advanced
   to this file's stage. For every member the cascade **measured** — status
   `reference`, `kept`, `rejected_low_zncc` or `rejected_shift` —
   `member_positions` and `member_affine_shapes` take the kernel's absolute
   position and absolute shape, downcast to float32; every other member keeps
   the detection the input carried, unchanged. Nothing is NaN, and
   `member_status` is what tells the two readings apart (the stage semantics of
   [`matches-file-format.md`](../../formats/matches-file-format.md)'s Member
   geometry section). `cluster_patches/` comes from the kernel output — including the
   per-member warp-consistency residual
   ([`cluster-warp-consistency.md`](../../core/patch/cluster-warp-consistency.md), a
   stored signal computed in the same kernel call, no CLI knobs) —
   `refine_options` = the CLI parameters, metadata updated
   (`has_cluster_patches: true`, fresh timestamp, workspace `relative_path`
   recomputed from the output location; the content hash is recomputed by
   the writer). Summary lines report the consistency distribution (median /
   p90) and the status breakdown (references / kept / rejected /
   unlocalizable / duplicate-image / not evaluated).

## Output statuses

`member_status` values in the written file (see
[`matches-file-format.md`](../../formats/matches-file-format.md), Cluster
Patches): `0 reference`, `1 kept`, `2 rejected_low_zncc`,
`3 rejected_shift`, `4 duplicate_image`, `5 not_evaluated`,
`6 rejected_unlocalizable`. A patch cluster = the reference plus its `kept`
members; rejected members keep their measured ZNCC / shift signals so
consumers can re-gate without re-running (`rejected_unlocalizable` members
are excluded before refinement, so their ZNCC / shift are NaN).

## Usage Examples

```bash
# Enrich the cluster matcher's output in place (writes clusters-patches.matches)
sfm cluster-patches -i matches/clusters.matches

# Stricter vetting, explicit output
sfm cluster-patches -i matches/clusters.matches -o matches/strict.matches \
    --min-zncc 0.9 --max-shift 2.0
```

## Notes

- `sfm match --cluster` writes the cluster-bearing input file (default
  `matches/<verified stem>-clusters.matches`, `--clusters-output` to override).
  Consumers that need pairwise matches from such a file obtain them through
  `sfmtool.feature_match.pairs_from_matches`
  ([`cluster-patches.md`](../../core/patch/cluster-patches.md)).
