// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-model distortion math kernels for [`super::CameraModel`].
//!
//! These are the private, model-specific forward/inverse distortion
//! implementations and ray-direction helpers used by the public
//! `distort` / `undistort` / `project` / `unproject` API in the parent
//! module. Kept separate so `distortion.rs` holds only the two public
//! `impl` blocks and the model dispatch.
//!
//! One module per camera-model family, in the order `distortion.rs`'s
//! dispatch reaches them:
//!
//! - [`brown`] — `OPENCV` / `FULL_OPENCV`, the radial + tangential family.
//! - [`equidistant`] — `OPENCV_FISHEYE`, `SIMPLE_RADIAL_FISHEYE` and
//!   `RADIAL_FISHEYE`, their ray-direction helpers, and the distortion-free
//!   `θ = r/f` map the spline models build on.
//! - [`thin_prism`] — `THIN_PRISM_FISHEYE`.
//! - [`rad_tan`] — `RAD_TAN_THIN_PRISM_FISHEYE`.
//! - [`sfmtool_fisheye`] — `SFMTOOL_FISHEYE`, the equidistant base plus a
//!   monotone radial spline.
//! - [`sfmtool_pinhole`] — `SFMTOOL_PINHOLE`, the pinhole base plus the same
//!   spline.
//! - [`blend`] — the tail the fisheye inverses share.
//!
//! Every kernel is re-exported here, so the parent keeps reaching them
//! through a single `use kernels::*` and no name carries its family in the
//! path.

mod blend;
mod brown;
mod equidistant;
mod rad_tan;
mod sfmtool_fisheye;
mod sfmtool_pinhole;
mod thin_prism;

pub(super) use blend::*;
pub(super) use brown::*;
pub(super) use equidistant::*;
pub(super) use rad_tan::*;
pub(super) use sfmtool_fisheye::*;
pub(super) use sfmtool_pinhole::*;
pub(super) use thin_prism::*;
