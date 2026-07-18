// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file format reading, writing, and verification.
//!
//! The `.sift` format stores SIFT feature descriptors extracted from images.
//! It is a ZIP archive with zstandard-compressed JSON metadata and binary
//! feature arrays (positions_xy, affine_shapes, descriptors, thumbnail_y_x_rgb).
//!
//! See `docs/sift-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "sift-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod read;
mod types;
mod verify;
mod write;

pub use read::{read_sift, read_sift_metadata, read_sift_partial, read_sift_positions};
pub use types::*;
pub use verify::verify_sift;
pub use write::write_sift;

#[cfg(test)]
mod tests;
