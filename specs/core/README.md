# Core Algorithm Specifications

Design documents for the algorithms in `crates/sfmtool-core/`, one subdirectory
per module of `crates/sfmtool-core/src/`. Each spec's status line names the file
that implements it.

A few specs here describe Python pipelines in `src/sfmtool/` that orchestrate
these kernels rather than an `sfmtool-core` module of their own; they are filed
under the module they drive. One,
[cluster selection](features/cluster-selection.md), describes an operation of
the `sfmtool-matches-format` crate and is filed under the module whose output
it narrows.

The Python bindings to these modules are indexed by submodule in
[../python-bindings.md](../python-bindings.md), which names the spec for each.

| Directory | Module | Contents |
|-----------|--------|----------|
| [analysis/](analysis/) | `analysis/` | Graphs and measurements over a reconstruction: covisibility, adjacency, coverage, alignment between reconstructions. |
| [bench/](bench/) | `bench/` | The bench beside a reconstruction and the editable track on it: the item list and its labels, the observations with their measurements and verdicts, and the commit that writes one back as a point. |
| [camera/](camera/) | `camera/` | Camera models, distortion kernels, projection derivatives, image warping, and the photograph cache. |
| [features/](features/) | `features/` | Feature extraction and matching: SIFT, descriptor search, optical flow, cluster matching. |
| [geometry/](geometry/) | `geometry/` | Pose estimation, epipolar geometry, and bundle adjustment. |
| [patch/](patch/) | `patch/` | Everything about oriented patches: normals, keypoint localization, and cluster patches. |
| [reconstruction/](reconstruction/) | `reconstruction/` | Operations on reconstruction data itself: triangulation, point correspondence. |
| [spatial/](spatial/) | `spatial/` | The shared KD-tree index behind every proximity query: nearest, k-nearest and radius, in world units. |
| [spherical/](spherical/) | `spherical/` | Spherical tiling and panorama compositing. |

## Phase timing with `SFMTOOL_PROFILE`

Several kernels carry opt-in phase timers and event counters, switched on by
the `SFMTOOL_PROFILE` environment variable. Any value other than empty or `0`
turns them on (the specs use `SFMTOOL_PROFILE=1`); unset, empty or
`0` leaves them off. The variable is read once, the first time a kernel checks
it, and the answer is cached for the rest of the process, so set it before the
process starts (or, from Python, before the first profiled call). With
profiling off, every timer and counter is one branch on the cached flag, so the
hot paths are unaffected. Profiling changes no output, only adds the summaries.

With profiling on, each profiled batch resets its counters when it starts and
prints a summary to stderr when it ends, every line prefixed
`[sfmtool-profile]`. The summary is a header line naming the batch (with its
wall or total time and, in all but `focal_vote`, its size), one row per phase
(its seconds, its share of the
batch total and, in most modules, its call count and mean microseconds per
call), an `other/overhead` row for the time no leaf phase covers, and, in the
four `patch` modules, rows of event counts. In the parallel kernels the phase times are summed
over threads, so they can add up to more than the wall time.

Six `prof` modules print a batch summary:

| Summary header | Printed at the end of | Profiler |
|----------------|-----------------------|----------|
| `cluster_covisibility` | `ClusterCovisibility::from_clusters_with_positions` (and `from_clusters`, which calls it) | [covisibility/prof.rs](../../crates/sfmtool-core/src/features/cluster_match/covisibility/prof.rs) |
| `focal_vote` | `focal_vote_with_options` | [focal_vote/prof.rs](../../crates/sfmtool-core/src/geometry/focal_vote/prof.rs) |
| `select_patch_cloud_views` | `select_patch_cloud_views` | [view_selection/prof.rs](../../crates/sfmtool-core/src/patch/view_selection/prof.rs) |
| `localize_patch_cloud_keypoints` | `localize_patch_cloud_keypoints` | [keypoint_localize/prof.rs](../../crates/sfmtool-core/src/patch/keypoint_localize/prof.rs) |
| `refine_patch_keypoints` | `refine_patch_cloud_keypoints`, and the Python `PatchCloud.refine_keypoints` | [keypoint_subpixel/prof.rs](../../crates/sfmtool-core/src/patch/keypoint_subpixel/prof.rs) |
| `refine_cluster_patches` | `refine_cluster_patches_borrowed` (and `refine_cluster_patches`) | [cluster_refine/prof.rs](../../crates/sfmtool-core/src/patch/cluster_refine/prof.rs) |

A seventh, [remap/prof.rs](../../crates/sfmtool-core/src/camera/remap/prof.rs),
counts the resample samplers' work and prints no summary of its own: the
`select_patch_cloud_views`, `localize_patch_cloud_keypoints` and
`refine_patch_keypoints` summaries reset its counters with theirs and end with
its `remap-sampler` row (calls, pixels sampled, taps) and, when the anisotropic
sampler ran, its `remap-aniso` row (pixels that took the single-tap fast path
or the multi-tap footprint walk, and the mean footprint length). Neither row
prints when no remap ran.

The shared pieces (the cached gate, the accumulating phase timer, the counter
increment and the row printers) are in
[profiling.rs](../../crates/sfmtool-core/src/profiling.rs); each `prof` module
declares its own phases, counters and report text from them. These summaries
are a separate mechanism from the per-call detail phases a `Progress` can carry
([operation-progress.md](../gui/operation-progress.md)), which are not read
from the environment.
