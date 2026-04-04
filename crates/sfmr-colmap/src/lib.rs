// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! COLMAP format interop for sfmr reconstructions.
//!
//! This crate handles reading and writing COLMAP formats (binary reconstruction
//! files, SQLite databases) with the conventions and transformations needed for
//! sfmr interop (0-based indexing, sorted images, named camera parameters, etc.).

pub mod colmap_db;
pub mod colmap_io;