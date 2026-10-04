# File Format Specifications

The on-disk formats, one spec per format crate in `crates/`, plus the container
those five formats are built on, specified once in
[archive-container.md](archive-container.md). They share the ZIP + zstd
primitives in `sfmtool-archive-io`, whose interface is in
[archive-io-crate.md](archive-io-crate.md).

Every format spec here is written to be read without this repository: it defines what
each stored value means and what a conforming file is, independently of the
operations this library performs on the data and of the code that implements
it, which each spec names once under **Implementations**. The standard is
stated in [../TEMPLATE.md](../TEMPLATE.md) ("File format specs stand alone")
and audited by the `audit-specs` skill. The one spec here that is not a format
spec is [archive-io-crate.md](archive-io-crate.md), which describes the Rust
crate behind the container and is kept beside the container spec it
implements.

| Document | Crate | Description |
|----------|-------|-------------|
| [archive-container.md](archive-container.md) | — | The container the formats are built on: ZIP with STORE, per-entry zstd, columnar binary entries, and the XXH128 section and whole-file hashes. |
| [archive-io-crate.md](archive-io-crate.md) | `sfmtool-archive-io` | The Rust crate that implements the container: entry-at-a-time read and write primitives, whole-archive parallel reads, section-digest folding and atomic file writes. |
| [kdf-file-format.md](kdf-file-format.md) | `sfmtool-kdf-format` | Immutable chunked randomized kd-forests over one shared descriptor corpus, with optional SIFT origins and geometry. |
| [sfmr-file-format.md](sfmr-file-format.md) | `sfmtool-sfmr-format` | The `.sfmr` reconstruction container: sections, schemas, point IDs, and the coordinate-system conventions everything else inherits. |
| [matches-file-format.md](matches-file-format.md) | `sfmtool-matches-format` | The `.matches` container: the cluster backbone, its members, and the derived cluster-patch sections. |
| [sift-file-format.md](sift-file-format.md) | `sfmtool-sift-format` | The `.sift` feature file: the zip entries holding keypoints, descriptors and thumbnail, their descending-size ordering, and the hashes that identify the extraction. |
| [camrig-file-format.md](camrig-file-format.md) | `sfmtool-camrig-format` | The `.camrig` camera-rig description and its pattern matching. |
| [sfmtool-camera-models.md](sfmtool-camera-models.md) | — | The `SFMTOOL_PINHOLE` and `SFMTOOL_FISHEYE` camera models as they appear on disk. Kernels in [../core/camera/](../core/camera/README.md). |
| [cluster-selection.md](cluster-selection.md) | `sfmtool-matches-format` | `MatchesData::select_clusters`: deriving a smaller, self-contained `.matches` working set. |
