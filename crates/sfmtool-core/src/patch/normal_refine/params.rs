// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-view [`ProjectedImage`], the [`PatchWindow`] scoring weight, the
//! [`Sampler`] / [`SamplerChoice`] re-exports, and the shared numeric
//! thresholds.

use crate::camera::{CameraIntrinsics, TiledPhotograph};
use crate::geometry::RigidTransform;

/// A source image together with the projection that maps world points into it:
/// the camera intrinsics, its world-to-camera pose, and the tiled photograph a
/// kernel reads its colour from. Everything a patch needs to be rendered from
/// one view. The photograph is made once and borrowed for every render, so a
/// kernel that renders a view many times allocates no photograph per render,
/// and one that reads through the render entry points of
/// [`sampler`](crate::camera::sampler) reads only the regions each render
/// needs.
#[derive(Clone, Copy)]
pub struct ProjectedImage<'a> {
    pub camera: &'a CameraIntrinsics,
    pub cam_from_world: &'a RigidTransform,
    pub source: &'a TiledPhotograph,
}

/// Per-pixel weight applied to the `R×R` patch when scoring (the NCC window).
///
/// `sigma` is expressed in normalized patch coordinates (the patch spans
/// `(s, t) ∈ [-1, 1]²`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchWindow {
    /// Uniform weight over the whole square grid (rotation-leaky; mainly a
    /// baseline).
    Uniform,
    /// Gaussian center weight over the square grid.
    Gaussian { sigma: f64 },
    /// Gaussian weight confined to the inscribed disk — radial, so in-plane
    /// rotation is exactly free and grazing corners don't leak in.
    /// Recommended default.
    GaussianDisk { sigma: f64 },
}

pub use crate::camera::sampler::{Sampler, SamplerChoice};

/// Windowed norm² below which a colour channel counts as flat (no texture
/// signal); flat channels are dropped from a z-normalized stack so every inner
/// product stays well-defined.
pub(in crate::patch) const FLAT_NORM_SQ_EPS: f64 = 1e-6;

/// Minimum number of commonly-valid pixels for a support to be scoreable.
pub(in crate::patch) const MIN_MASK_PIXELS: usize = 8;
