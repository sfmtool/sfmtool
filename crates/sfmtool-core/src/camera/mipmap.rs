// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A photograph's mipmap: its reduced levels, and when needed its
//! full-resolution copy, stored as JPEG mip level files under the workspace's
//! mipmap directory.

/// The directory at a workspace root that holds the workspace's mip level
/// files. Every search for a workspace's images skips a directory of this
/// name, since the mip level files are JPEGs named after the photographs.
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";
