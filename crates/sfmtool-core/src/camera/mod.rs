// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera model: SfM camera intrinsics, distortion, frustum / epipolar geometry,
//! rectification, image warping, plus a 3D-viewport [`Camera`] for orbit-style
//! navigation in `sfm-explorer`.

pub mod distortion;
pub mod epipolar;
pub mod frustum;
pub mod intrinsics;
pub mod rectification;
pub mod remap;
pub mod viewport;
pub mod warp_map;

pub use intrinsics::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};
pub use viewport::Camera;
pub use warp_map::WarpMap;
