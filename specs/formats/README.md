# File Format Specifications

The on-disk formats, one spec per format crate in `crates/`. All four
containers share the ZIP + zstd primitives in `sfmtool-archive-io`.

| Document | Crate | Description |
|----------|-------|-------------|
| [sfmr-file-format.md](sfmr-file-format.md) | `sfmr-format` | The `.sfmr` reconstruction container: sections, schemas, point IDs, and the coordinate-system conventions everything else inherits. |
| [matches-file-format.md](matches-file-format.md) | `matches-format` | The `.matches` container: the cluster backbone, its members, and the derived cluster-patch sections. |
| [sift-file-format.md](sift-file-format.md) | `sift-format` | The `.sift` feature file: the zip entries holding keypoints, descriptors and thumbnail, their descending-size ordering, and the hashes that identify the extraction. |
| [camrig-file-format.md](camrig-file-format.md) | `camrig-format` | The `.camrig` camera-rig description and its pattern matching. |
| [sfmtool-camera-models.md](sfmtool-camera-models.md) | — | The `SFMTOOL_PINHOLE` and `SFMTOOL_FISHEYE` camera models as they appear on disk. Kernels in [../core/camera/](../core/camera/README.md). |
| [cluster-selection.md](cluster-selection.md) | `matches-format` | `MatchesData::select_clusters`: deriving a smaller, self-contained `.matches` working set. |
