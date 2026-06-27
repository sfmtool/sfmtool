// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch clouds: point-patch storage and normal refinement.

pub mod cloud;
pub mod keypoint_localize;
pub mod keypoint_subpixel;
pub mod normal_refine;
pub mod view_selection;

pub use cloud::{PatchCloud, PatchCloudError};
