// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera model: SfM camera intrinsics, distortion, frustum / epipolar geometry,
//! rectification, image warping, the per-view choice of sampler ([`sampler`]),
//! the image containers it reads and writes
//! ([`image`]), the derived quantities a set of intrinsics
//! implies ([`report`]), a byte-bounded cache of decoded photograph pyramids
//! ([`PhotographCache`]), plus a 3D-viewport [`Camera`] for orbit-style
//! navigation in `sfm-explorer`.

pub mod distortion;
pub mod epipolar;
pub mod frustum;
pub mod image;
pub mod intrinsics;
pub mod photograph_cache;
pub mod rectification;
pub mod refit_intrinsics;
pub mod remap;
pub mod report;
pub mod sampler;
pub mod viewport;
pub mod warp_map;

pub use distortion::PixelJacobian;
pub use intrinsics::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};
pub use photograph_cache::{GetManyTally, PeekedPhotograph, PhotographCache, PhotographCacheStats};
pub use viewport::Camera;
pub use warp_map::WarpMap;
