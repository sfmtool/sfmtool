// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file format reading and writing.
//!
//! The `.sfmr` format is sfmtool's native reconstruction file format.
//! It stores SfM reconstructions as ZIP archives with zstandard-compressed
//! columnar binary data and JSON metadata.
//!
//! See `docs/sfmr-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "sfmr-format requires a little-endian target (binary arrays are stored as little-endian)"
);

mod depth_stats;
mod entries;
mod lineage;
mod read;
mod types;
mod verify;
mod write;

pub use depth_stats::{compute_depth_statistics, DepthStatsResult};
pub use lineage::{LineageEntry, LineageMap, LINEAGE_KIND_BASE, LINEAGE_KIND_POINT_EDIT};
pub use read::{read_sfmr, read_sfmr_content_hash, read_sfmr_metadata, resolve_workspace_dir};
pub use types::*;
pub use verify::verify_sfmr;
pub use write::{content_hash_of, write_sfmr, write_sfmr_with_options, WriteOptions};

#[cfg(test)]
mod tests;
