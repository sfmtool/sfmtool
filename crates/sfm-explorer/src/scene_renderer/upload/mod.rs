// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data upload logic — transfers point cloud, frustum, thumbnail, patch,
//! background image, and track ray data to the GPU.
//!
//! One submodule per GPU resource, each contributing its own
//! `impl SceneRenderer` block. This mirrors the per-resource layout of the
//! sibling [`super::pipelines`] and [`super::render`] modules: the uploads
//! share no state with each other, only the renderer they write into.

mod additions;
mod bg_image;
mod frustums;
mod overlay;
mod patches;
mod points;
mod thumbnails;
mod track_rays;

#[cfg(test)]
mod tests;

/// What an upload did, for the note beside its phase's time in an Action Log
/// entry.
///
/// An upload that kept what the GPU already held and one that rewrote it in
/// under a millisecond both read `<1 ms`, and nothing in the timing tells them
/// apart. Each upload below already decides whether what it holds is still the
/// right thing; this is that decision handed back to the caller, which is the
/// only place the frame's phases are opened and so the only place that can say
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Uploaded {
    /// What the GPU held was still right, and nothing was written.
    Reused,
    /// Written, over this many items, in whichever unit the upload counts:
    /// points, atlas tiles, images.
    Built(usize),
}
