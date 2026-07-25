// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data upload logic — transfers point cloud, frustum, thumbnail, patch,
//! background image, and track ray data to the GPU.
//!
//! One submodule per GPU resource, each contributing its own
//! `impl SceneRenderer` block. This mirrors the per-resource layout of the
//! sibling [`super::pipelines`] and [`super::render`] modules: the uploads
//! share no state with each other, only the renderer they write into.

mod bg_image;
mod frustums;
mod patches;
mod points;
mod thumbnails;
mod track_rays;

#[cfg(test)]
mod tests;
