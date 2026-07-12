# `sfm cluster-patches` Command

## Overview

Refines a cluster-bearing `.matches` file into **patch clusters**: per
cluster, a reference member plus, for every other member, a photometrically
refined and vetted affine warp that maps the reference's local patch onto
that member's image. The result is written as the `cluster_patches/` section
of a **new** `.matches` file that copies the input's images and clusters
sections verbatim (write-once workflow, like adding two-view geometries).

Design: [`specs/core/cluster-patches.md`](../core/cluster-patches.md).
Implementation (Rust kernel, algorithm, bindings):
[`specs/core/cluster-patch-refinement.md`](../core/cluster-patch-refinement.md).
Format: the `clusters/` and `cluster_patches/` sections of
[`matches-file-format.md`](../formats/matches-file-format.md).

## Command Syntax

```bash
sfm cluster-patches -i clusters.matches [-o out.matches] [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i, --input` | path | required | Cluster-bearing `.matches` file (from `sfm match --cluster`) |
| `-o, --output` | path | input with a `-patches` suffix | Output `.matches` path; must not already exist |
| `--radius` | float > 0 | 4.0 | Template half-width, keypoint-frame units |
| `--resolution` | int ≥ 3 | 15 | Template samples per axis |
| `--min-zncc` | float in [−1, 1] | 0.85 | Member acceptance threshold on the achieved windowed ZNCC |
| `--max-shift` | float ≥ 0 | 3.0 | Max translation drift from the SIFT seed, px |
| `--max-keypoint-uncertainty` | float ≥ 0 | 0.35 | Localizability gate: exclude members whose own patch scores a predicted keypoint position uncertainty (`σ_pos`, template-grid px) above this, before reference selection and refinement; `0` disables |

The defaults come from the experiment calibration
(`specs/core/cluster-patches.md`, "The operation"): `radius` 2 is too small
for the affine DOF and 6–8 buys nothing; `min_zncc` is permissive by design —
over-culling, not contamination, is the observed failure mode, and downstream
stages re-gate on the stored signals. `--max-keypoint-uncertainty` shares its
default value with `embed-patches` (the conservative tail cut of
[`patch-localizability.md`](../core/patch-localizability.md)); it is scored
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
   search — each candidate must hold `.sfm-workspace.json`). Locate each
   image's `.sift` as
   `{workspace}/{image_parent}/{feature_prefix_dir}/{basename}.sift`, verify
   the `sift_content_hashes`, and read `positions_xy` / `affine_shapes`
   capped at `feature_counts[i]` (the count used during matching, so member
   feature indices line up).
3. **Refine.** Load the images with cv2 (color) and the `.sift` geometry in
   images-section order — decoded through a thread pool (cv2 releases the
   GIL; the embed-patches pattern), results collected in submission order —
   and call `_sfmtool.matching.refine_cluster_patches` (the
   `patch::cluster_refine` kernel — per-member localizability gate,
   reference selection by largest SIFT scale, Gaussian-windowed-ZNCC shift →
   similarity → affine Nelder-Mead cascade seeded from the SIFT affine
   shapes, vetting, one kept member per image), with a `ProgressCounter`
   poller reporting per-cluster progress.
4. **Write.** A new `.matches` file: images + clusters sections copied
   verbatim (with `image_dims` populated from the hash-verified `.sift`
   metadata rather than the input file, so a version-3 cluster backbone —
   which stored no dims — still enriches into a self-contained version-4
   file), `cluster_patches/` from the kernel output — including the
   per-member warp-consistency residual
   ([`cluster-warp-consistency.md`](../core/cluster-warp-consistency.md), a
   stored signal computed in the same kernel call, no CLI knobs) —
   `refine_options` = the CLI parameters, metadata updated
   (`has_cluster_patches: true`, fresh timestamp, workspace `relative_path`
   recomputed from the output location; the content hash is recomputed by
   the writer). Summary lines report the consistency distribution (median /
   p90) and the status breakdown (references / kept / rejected /
   unlocalizable / duplicate-image / not evaluated).

## Output statuses

`member_status` values in the written file (see
[`matches-file-format.md`](../formats/matches-file-format.md), Cluster
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
  `matches/<verified stem>-clusters.matches`; the derived-pairs migration of
  [`cluster-patch-refinement.md`](../core/cluster-patch-refinement.md) §1,
  shipped 2026-07-10).
