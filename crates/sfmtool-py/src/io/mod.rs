// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! File-format I/O bindings: `.sfmr`, `.sift`, `.matches`, `.camrig`, and the
//! COLMAP binary + SQLite database formats.
//!
//! Functions are still registered flat on `_sfmtool` for now (Python callers
//! see `from sfmtool._sfmtool import read_sfmr` unchanged); this module just
//! groups the wrappers by the format they bridge. The eventual
//! `_sfmtool.io` Python submodule is a follow-up — see hygiene audit #4.

pub mod camrig;
pub mod colmap_binary;
pub mod colmap_db;
pub mod matches;
pub mod sfmr;
pub mod sift;
