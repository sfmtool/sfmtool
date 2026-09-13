// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Persistent, independently compressed randomized kd-forest storage.
//!
//! A `.kdf` is a ZIP directory of zstd frames. Opening reads only its directory,
//! metadata, integrity directory, and the bounded corpus row map. Tree chunks,
//! descriptor blocks, optional SIFT geometry blocks, and origins remain lazy.

mod cache;
mod read;
mod summary;
mod types;
mod verify;
mod write;

pub use read::KdfFile;
pub use summary::{kdf_summary, KdfSection, KdfSummary};
pub use types::*;
pub use verify::{verify_kdf, verify_sift_sources};
pub use write::write_kdf;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod validation_tests;
