// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file format reading, writing, and verification.
//!
//! The `.matches` format stores feature match correspondences between images.
//! It is a ZIP archive with zstandard-compressed JSON metadata and binary
//! arrays for match data, with an optional two-view geometries section.
//!
//! See `docs/matches-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "matches-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod read;
mod types;
mod verify;
mod write;

pub use read::{read_matches, read_matches_metadata};
pub use types::*;
pub use verify::verify_matches;
pub use write::write_matches;

#[cfg(test)]
mod tests;
