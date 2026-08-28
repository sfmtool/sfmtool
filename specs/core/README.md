# Core Algorithm Specifications

Design documents for the algorithms in `crates/sfmtool-core/`, one subdirectory
per module of `crates/sfmtool-core/src/`. Each spec's status line names the file
that implements it.

A few specs here describe Python pipelines in `src/sfmtool/` that orchestrate
these kernels rather than an `sfmtool-core` module of their own; they are filed
under the module they drive.

| Directory | Module | Contents |
|-----------|--------|----------|
| [analysis/](analysis/) | `analysis/` | Graphs and measurements over a reconstruction: covisibility, adjacency, coverage, alignment between reconstructions. |
| [camera/](camera/) | `camera/` | Camera models, distortion kernels, projection derivatives, and image warping. |
| [features/](features/) | `features/` | Feature extraction and matching: SIFT, descriptor search, optical flow, cluster matching. |
| [geometry/](geometry/) | `geometry/` | Pose estimation, epipolar geometry, and bundle adjustment. |
| [patch/](patch/) | `patch/` | Everything about oriented patches: normals, keypoint localization, and cluster patches. |
| [reconstruction/](reconstruction/) | `reconstruction/` | Operations on reconstruction data itself: triangulation, point correspondence. |
| [spherical/](spherical/) | `spherical/` | Spherical tiling and panorama compositing. |
