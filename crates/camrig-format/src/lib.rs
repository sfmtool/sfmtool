// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` file format reading, writing, and verification.
//!
//! The `.camrig` format stores a camera rig — a set of cameras (*sensors*)
//! held in fixed relative poses. One format covers the whole range: a single
//! camera, a back-to-back fisheye pair, a six-face cubemap, and a spherical
//! tile rig of up to ~100 000 sensors.
//!
//! It is a ZIP archive with zstandard-compressed JSON metadata and binary
//! columnar arrays (a shared camera pool plus per-sensor camera indexes,
//! `sensor_from_rig` quaternions, and translations).
//!
//! See `specs/formats/camrig-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "camrig-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod pattern;
mod read;
mod types;
mod verify;
mod write;

pub use pattern::*;
pub use read::{read_camrig, read_camrig_metadata};
pub use types::*;
pub use verify::verify_camrig;
pub use write::write_camrig;

#[cfg(test)]
mod tests;
