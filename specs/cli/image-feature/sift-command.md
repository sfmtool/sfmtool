# `sfm sift` Command

## Overview

`sfm sift` detects SIFT keypoints and descriptors in a set of images and writes
one `.sift` file per image for `sfm match` and `sfm solve`; with `--draw` it
draws those keypoints onto copies of the images. Exactly one action mode must
be specified per invocation.

The command is implemented in
[`src/sfmtool/_commands/sift.py`](../../../src/sfmtool/_commands/sift.py), with
extraction in [`src/sfmtool/sift/extract.py`](../../../src/sfmtool/sift/extract.py).

To inspect an existing `.sift` file, use `sfm inspect <FILE.sift>`.

## Command Syntax

```bash
sfm sift [PATHS...] --extract | --draw <DIR> [OPTIONS...]
```

`PATHS` are image files or directories. When omitted, the current directory is
used. Directories are expanded recursively, keeping only files ending in
`.png`, `.jpg` or `.jpeg` (in any letter case); a file named directly is used
whatever its extension. The command errors if no image remains after expansion
and `--range` filtering.

## Action Modes

Exactly one of these is required:

| Mode | Description |
|------|-------------|
| `--extract / -e` | Extract SIFT features from images, writing `.sift` files |
| `--draw / -d <DIR>` | Draw SIFT features as ellipses on images, saving to `<DIR>` |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter-sfm` | path | | Only draw features used in a `.sfmr` reconstruction (with `--draw`) |
| `--range / -r` | string | | Range expression for file numbers (e.g., `1-100`, `1-100:2`, `5,10,15`) |
| `--num-threads / -t` | int | -1 (all) | Thread count for extraction (`colmap` and `opencv` only) |
| `--tool` | `colmap` \| `opencv` \| `sfmtool` | workspace | Use this feature tool instead of the workspace |
| `--dsp / --no-dsp` | bool | off | Domain size pooling; requires `--tool colmap` |

`--dsp / --no-dsp` is rejected without `--tool`, and with any `--tool` other
than `colmap`. In workspace mode domain size pooling comes from the workspace
configuration; change it by reinitializing the workspace with
`sfm ws init --dsp --force`.

The `sfmtool` tool is the toolkit's own SIFT (the Rust `sfmtool-core`
implementation), calibrated to match COLMAP's keypoint density and descriptor
convention. It needs no external library and parallelizes internally, so it
ignores `--num-threads`. It chooses how many images to decode and extract at
once from the core count and available memory, and the
`SFMTOOL_SIFT_EXTRACT_WORKERS` environment variable overrides that number (see
[Extraction-orchestration pipelining](#extraction-orchestration-pipelining)).

## Workspace and Tool Modes

The feature tool and its options come from one of two places:

- **Workspace mode** (no `--tool`): the command finds the workspace containing
  the common parent directory of the images and reads the feature tool, its
  options and `feature_prefix_dir` from its `.sfm-workspace.json` (see
  [workspace.md](../../workspace/workspace.md)). If no workspace is found, the
  command errors and asks for either `sfm ws init` or `--tool`.
- **Tool mode** (`--tool` given): the workspace is not consulted at all. The
  tool runs with its default options (plus `--dsp` for `colmap`). Both
  `--extract` and `--draw` use this tool's feature directory, even for images
  inside a workspace.

## Output Location

Each image `<dir>/<name>` gets one `.sift` file:

- In workspace mode: `<dir>/<feature_prefix_dir>/<name>.sift`.
- In tool mode: `<dir>/features/<type>-<xxh128>/<name>.sift`, where `<type>` is
  `sift-colmap` (with `-dsp` and `-max<N>` suffixes when those options differ
  from the defaults), `sift-opencv` or `sift-sfmtool`, and `<xxh128>` is a
  hash of the tool name, type and options.

`--extract` skips an image whose `.sift` file already exists and is at least as
new as the image (by modification time), and reports how many images were
skipped and how many were extracted.

`--draw <DIR>` writes `<DIR>/<name>` for each image. An image whose `.sift` file
is missing is reported as an error and the command continues with the next
image.

## Extraction-orchestration pipelining

The Python extraction backend (`extract_sift_with_sfmtool`) processes images in
three stages: load+decode (`cv2.imread`), extract (the Rust core, GIL released,
internally rayon-parallel), then save (`write_sift`, zstd+ZIP). Measured
per-image stage split (ms): extract dominates at **85–91%** of the work;
load+thumbnail+save together are only **~7% on large images (2040×1536), ~15% on
small (270×480)**, and the source-file re-read used for content hashing is
effectively free (OS page cache serves it after the decode).

A load ∥ extract ∥ save pipeline is therefore a **single-digit-percent** win on
local/SSD storage, and is limited less by the GIL than by **CPU saturation**:
the rayon extract already uses every core, so overlapping the save — or decoding
the next image — *contends* for cores rather than hiding idle time. The one
stage with genuine idle time to hide is the **disk read** (`cv2.imread` already
releases the GIL) — *and*, on small images, the cores the per-image rayon extract
cannot itself saturate (measured: 1→4 threads scales only **3.1×** on a 270×480
image, ~23% idle), plus each image's serial floor (octave-0 build, setup). So
rather than a full three-stage pipeline, the backend implements several targeted,
low-risk overlaps:

- **Decode and extract several images concurrently**, yielding results in input
  order. A `ThreadPoolExecutor(max_workers=_extract_workers())` runs up to *K*
  decode+extract pipelines at once (FIFO over the in-flight futures preserves
  order and re-raises decode/extract errors in order); the bounded look-ahead
  caps memory to ~*K* decoded frames + pyramids. This hides disk-read latency
  **and** overlaps one image's serial floor with another's parallel work, so the
  cores the per-image rayon leaves idle on small images get filled. Because both
  `cv2.imread` and the Rust extract release the GIL and rayon's *single global
  pool* caps total CPU threads at the core count, more in-flight images never
  oversubscribe — they just keep that pool fed. *K* must therefore scale *with*
  the core count to fill a many-core host, so it defaults to `os.cpu_count()`,
  bounded only by memory: each in-flight image holds a decoded frame plus its
  f32 gaussian pyramid (`_EXTRACT_BYTES_PER_SOURCE_PIXEL` ≈ 192 B per source
  pixel), so *K* is capped to keep the in-flight set within
  `_extract_mem_budget_bytes()` (~half of physical RAM). The per-image footprint
  is estimated from the first image's dimensions — decoded once on the calling
  thread to size the pool, then reused as image 0's input — assuming a batch of
  like-sized images (one workspace). When the size is unknown (e.g. a test fake)
  *K* falls back to the historical conservative `min(os.cpu_count(), 4)`.
  `SFMTOOL_SIFT_EXTRACT_WORKERS` overrides everything (`1` restores
  one-at-a-time; a value below 1 counts as 1, and a non-integer value is
  ignored with a warning). Measured win (point-in-time, 4-core box, release
  build, seoul_bull 270×480): **~1.27× batch throughput** (100 imgs 37.6→29.6 ms/img at
  1→4+ workers), no single-image regression; because a 270×480 image already
  reaches ~4.5 effective cores, that box is near-saturated at *K*=4 and the win
  there is modest — the memory-bounded core-scaled default matters on many-core
  hosts, where the old constant `4` left most cores idle. (Illustrative
  snapshots, not invariants — rerun `bench-sift` to refresh.) See
  `_stream_sift_with_sfmtool`, `_extract_workers` and `_extract_mem_budget_bytes`
  in `sift/extract_sfmtool.py`.
- **Stream `.sift` writes per image** instead of buffering a whole chunk
  (`chunk_size = 500` images) in memory — `extract_sift_with_sfmtool` is a
  generator that yields one result at a time, and `image_files_to_sift_files`
  in [`sift/extract.py`](../../../src/sfmtool/sift/extract.py) writes each as it
  arrives. A peak-memory win (hundreds of MB for dense, high-resolution inputs).
  The pipeline only runs on images that the up-to-date check in
  [Output Location](#output-location) did not skip.
- **The `write_sift` binding releases the GIL** (`py.detach`) around its
  zstd/ZIP compression and file write, so a save can run concurrently with other
  Rust work.
- **Overlap the save with the next extract — on the rayon pool, not a thread.**
  `image_files_to_sift_files` drains results into a `SiftWriteQueue` (the
  `_sfmtool.fileio` PyO3 class): `submit` copies the data (GIL held) and `rayon::spawn`s
  the compression+write onto the **same global pool the extract uses**, then
  returns; `join_oldest` (bounded look-ahead for backpressure) and `join` await
  saves and surface their errors in order. The save of image *i* thus overlaps
  the extract of image *i+1*. This is **unconditional** — no spare-core gate.
  Backpressure bounds the queue at a **two-save look-ahead** (`write_lookahead`):
  before each `submit`, if two saves are already in flight the oldest is joined
  first, so at most ~2 compressed images are buffered. (Distinct from the
  *extract* concurrency's *K*-image look-ahead above.)
  The drain is guaranteed on every exit: the write loop's `finally` calls
  `drain` (a non-raising await of in-flight saves, so it can't mask an
  exception already unwinding), and `SiftWriteQueue::Drop` awaits any stragglers
  as a structural backstop — a spawned save never outlives the queue, so an
  error mid-stream can't leave a half-written `.sift` racing the next step.

Why both the decode and the save are worth hiding: they are **fixed per-image
costs that do not shrink with core count**, while the rayon extract does.
Measured on dino (2040×1536): decode ≈ 13 ms, save (zstd/ZIP) ≈ 26 ms — both
constant — versus extract ≈ 1244 / 730 / 546 / 407 ms at 1 / 2 / 3 / 4 threads
(Amdahl fit `≈ 165 + 1088/p` ms). So the decode+save "gap between images" is a
small slice when extract dominates (few cores, large images) but a growing
fraction as cores scale and extract collapses toward its ~165 ms serial floor.

**Why the save goes on the rayon pool, not a worker thread (the contention
trap).** Decode is I/O-bound (`cv2.imread` waits on disk and releases the GIL),
so prefetching it on a Python thread is free — it never burns a core. The save is
*CPU-bound* (single-stream zstd). An earlier attempt offloaded it to a dedicated
`ThreadPoolExecutor` writer thread; that **regressed ~25–30%** (wall *and*
CPU-seconds) on a fully-subscribed box. Root cause: a separate OS thread pushes
the runnable-thread count to *N+1* on *N* cores, so the kernel deschedules a
rayon worker mid-chunk; the descheduled worker stalls at one of the extract's
many sync barriers (sequential octaves, separable-blur passes) and the others
**busy-spin**, burning CPU for no work (hence the CPU-seconds inflation ≫ the
~2 s of actual save work). Verified: the same concurrent save caused **0%
slowdown with spare cores**, and external tenant load reproduced the identical
inflation — i.e. the cause is core oversubscription, not the write path.

Submitting the save as a **rayon task** instead avoids this: the pool stays at
*N* threads, one worker runs the ~26 ms save while the extract's `par_iter`
proceeds on the other *N−1* (rayon routes chunks only to available workers, so no
barrier waits on the saving worker → no spin). The cost is only the genuine
`save/N` of lost worker-time. Measured back-to-back on a 4-core box, the in-pool
overlap is **CPU-neutral vs inline** (the +25 % inflation of the thread approach
is gone), so it is safe to enable unconditionally; the wall-time win itself
materialises on many-core boxes, where the extract's serial floor leaves workers
genuinely idle for the save to fill. Ordering is preserved (the generator yields
in input order; `join`/`join_oldest` re-raise decode/write errors), and writes
target distinct files so a single in-flight save keeps up (save ≪ extract). The
COLMAP/OpenCV backends share the same queue unchanged (they return eager lists).

## Usage Examples

```bash
# Extract features for all images in workspace
sfm sift --extract

# Extract with specific tool override
sfm sift --extract --tool opencv

# Extract with the sfmtool (Rust) backend
sfm sift --extract --tool sfmtool

# Visualize features on images
sfm sift --draw ./sift_viz

# Visualize only features used in a reconstruction
sfm sift --draw ./sift_viz --filter-sfm sfmr/solve_001.sfmr
```
