// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rendering and numeric kernels for subpixel keypoint refinement, split out of
//! the Gauss–Newton orchestration ([`super`]).
//!
//! Two halves, one module each:
//!
//! - [`render`] — **rendering / render-once tile**: [`render_core`](render::render_core) /
//!   [`render_core_with_jg`](render::render_core_with_jg) (direct projective renders, the out-of-tile
//!   fallback), the [`RefineTile`] prerender + cubic-B-spline reads
//!   ([`render_refine_tile`](render::render_refine_tile) /
//!   [`try_render_refine_tile`] / [`core_value`] / [`core_value_with_jg`]),
//!   and the coarse-grid gate
//!   ([`grid_to_source_scale`](render::grid_to_source_scale) / [`TILE_MAX_GRID_TO_SOURCE`](render::TILE_MAX_GRID_TO_SOURCE)).
//! - [`score`] — **scoring kernels**: [`znorm_core`] (z-normalize),
//!   [`ecc_score`] (the ECC criterion), [`view_jacobian`] (the analytic ECC
//!   Gauss–Newton normal equations), and [`solve_2x2`] (the damped
//!   normal-equation solve).
//!
//! The names production uses are re-exported here, so callers keep reaching
//! them through `kernels::…` as before. The render entry points and the
//! coarse-grid gate are re-exported under `cfg(test)` only — production
//! reaches those through [`core_value`] and the tile,
//! and the sibling test module names them directly (the same arrangement
//! `keypoint_subpixel` already uses).

mod render;
mod score;

pub(super) use render::{core_value, core_value_with_jg, try_render_refine_tile, RefineTile};
pub(super) use score::{ecc_score, solve_2x2, view_jacobian, znorm_core};

// Reached only by the sibling test module (production goes through the names
// above), so these are test-gated to stay warning-clean in release.
#[cfg(test)]
pub(super) use render::{
    grid_to_source_scale, render_core, render_core_with_jg, render_refine_tile,
    TILE_MAX_GRID_TO_SOURCE,
};
