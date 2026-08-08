// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Coarse ray-grid projection: the non-perspective acceleration path of
//! [`CameraIntrinsics::ray_to_pixel_grid`].
//!
//! The exact projection is evaluated on a sub-grid and cell interiors are
//! bilinearly interpolated, with every cell probe-checked against the exact
//! projection and demoted when it would exceed the tolerance — so accuracy
//! never depends on the stride, only the speedup does. This is a caching /
//! interpolation strategy rather than a lens model; it lives here rather
//! than in `distortion.rs` for that reason. See
//! `specs/core/ray-grid-projection.md`.

use crate::camera::CameraIntrinsics;

// --- Coarse ray-grid projection (non-perspective path of `ray_to_pixel_grid`) ---

/// Sub-grid spacing (in destination grid pixels) for the non-perspective path of
/// [`CameraIntrinsics::ray_to_pixel_grid`]: the exact projection is evaluated
/// every `COARSE_GRID_STRIDE` pixels and the interior is bilinearly interpolated.
/// A larger stride speeds smooth (low-curvature) tiles but is bounded for free:
/// every cell is probe-checked against the exact projection and demoted to exact
/// when it would exceed [`COARSE_GRID_TOL_PX`], so accuracy never depends on this
/// value — only the speedup does. See `specs/core/ray-grid-projection.md` and the
/// `coarse_grid_error_*` tests.
pub(super) const COARSE_GRID_STRIDE: u32 = 8;

/// Per-cell source-pixel error tolerance for the coarse-grid path of
/// [`CameraIntrinsics::ray_to_pixel_grid`]. A cell is interpolated only if its
/// center and edge-midpoints match the exact projection to within this many
/// source pixels; otherwise it is projected exactly. Set an order of magnitude
/// below the localizer's sub-pixel needs.
pub(super) const COARSE_GRID_TOL_PX: f32 = 0.02;

/// Linear interpolation `a + (b − a)·f`.
#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

/// Per-grid projection constants for [`CameraIntrinsics::project_ray_node`],
/// hoisted once per grid so the per-node projection touches no enum-match for
/// the (loop-invariant) intrinsics or image bounds.
#[derive(Clone, Copy)]
struct GridProj {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
}

/// Bilinear interpolation of the four cell corners (`p00` at `(0,0)`, `p10` at
/// `(1,0)`, `p01` at `(0,1)`, `p11` at `(1,1)`) at fractional `(sf, tf)`. Shared
/// by the coarse-grid probe (acceptance) and fill so the probe predicts the fill
/// exactly.
#[inline]
fn bilerp(
    p00: [f32; 2],
    p10: [f32; 2],
    p01: [f32; 2],
    p11: [f32; 2],
    sf: f32,
    tf: f32,
) -> [f32; 2] {
    [
        lerp(lerp(p00[0], p10[0], sf), lerp(p01[0], p11[0], sf), tf),
        lerp(lerp(p00[1], p10[1], sf), lerp(p01[1], p11[1], sf), tf),
    ]
}

impl CameraIntrinsics {
    /// Hoist the loop-invariant projection constants (intrinsics + image bounds)
    /// for a grid projection — see [`GridProj`] and [`project_ray_node`](Self::project_ray_node).
    #[inline]
    fn grid_proj(&self) -> GridProj {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        GridProj {
            fx,
            fy,
            cx,
            cy,
            w: self.width as f64,
            h: self.height as f64,
        }
    }

    /// Project a single camera-frame ray to a source pixel using pre-hoisted
    /// [`GridProj`] constants, returning `(NaN, NaN)` when the model rejects the
    /// ray (behind camera / outside the invertible domain) or the result falls
    /// outside the image rectangle `[0, w) × [0, h)`. This is
    /// [`ray_to_pixel`](Self::ray_to_pixel) inlined so the grid paths fetch the
    /// (loop-invariant) intrinsics once per grid instead of once per node.
    #[inline]
    fn project_ray_node(&self, ray: [f64; 3], p: GridProj) -> [f32; 2] {
        match self.model.distort_ray(ray) {
            Some((x_d, y_d)) => {
                let px = p.fx * x_d + p.cx;
                let py = p.fy * y_d + p.cy;
                if px >= 0.0 && py >= 0.0 && px < p.w && py < p.h {
                    [px as f32, py as f32]
                } else {
                    [f32::NAN, f32::NAN]
                }
            }
            None => [f32::NAN, f32::NAN],
        }
    }

    /// Exact, per-node version of [`ray_to_pixel_grid`](Self::ray_to_pixel_grid):
    /// projects every grid node through the full camera model. Used directly for
    /// perspective models (where it is the fast path — the affine ray basis
    /// removes the per-node pose multiply, and the divide + distortion are
    /// cheap) and as the coarse-grid fallback and test reference for the
    /// non-perspective path.
    ///
    /// Sequential: this renders one patch-sized tile and every caller runs it
    /// inside a per-patch/`par_iter` loop, so the caller owns parallelism (an
    /// inner `par_chunks` would just nest rayon over ~`rows` rows). Full-image
    /// warps that want row parallelism use [`crate::camera::WarpMap::from_cameras`]
    /// instead.
    pub(crate) fn ray_to_pixel_grid_exact(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), 2 * cols as usize * rows as usize);
        let gp = self.grid_proj();
        let cols_u = cols as usize;
        for (row, dst) in out.chunks_exact_mut(2 * cols_u).enumerate() {
            let r = row as f64;
            let ox = origin[0] + r * row_step[0];
            let oy = origin[1] + r * row_step[1];
            let oz = origin[2] + r * row_step[2];
            for col in 0..cols_u {
                let c = col as f64;
                let ray = [
                    ox + c * col_step[0],
                    oy + c * col_step[1],
                    oz + c * col_step[2],
                ];
                let p = self.project_ray_node(ray, gp);
                dst[2 * col] = p[0];
                dst[2 * col + 1] = p[1];
            }
        }
    }

    /// Project an **affine grid of camera-frame rays** to source pixel
    /// coordinates — the grid sibling of [`ray_to_pixel`](Self::ray_to_pixel).
    ///
    /// The ray at integer grid node `(col, row)` (with `col ∈ 0..cols`,
    /// `row ∈ 0..rows`) is `origin + col·col_step + row·row_step`. Results are
    /// written as interleaved `(sx, sy)` f32 pairs, row-major, into `out` (which
    /// must have length `2·cols·rows`); a node that is behind the camera, outside
    /// the distortion model's invertible domain, or outside the image rectangle
    /// is written as `(NaN, NaN)` — identical to [`Self::ray_to_pixel`] followed
    /// by the in-frame test.
    ///
    /// Affineness of the input grid is the contract that licenses the
    /// model-specific fast paths:
    /// * **Perspective** models project every node exactly (the divide +
    ///   distortion are cheap; the win over the scalar caller is that the pose
    ///   multiply has already been folded into the affine basis).
    /// * **Fisheye / equirectangular** models, whose per-node projection is
    ///   expensive (`atan2`/`asin`) but spatially smooth, evaluate the exact
    ///   projection only on a coarse sub-grid (stride `COARSE_GRID_STRIDE`)
    ///   and bilinearly interpolate the interior, falling back to exact
    ///   projection wherever a bracketing sub-grid node is invalid. The
    ///   interpolation error is bounded; see
    ///   `specs/core/ray-grid-projection.md`.
    pub fn ray_to_pixel_grid(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), 2 * cols as usize * rows as usize);
        // Perspective models, or grids too small to amortize the sub-grid setup,
        // go straight through the exact per-node path.
        if !self.model.needs_ray_path()
            || cols < 2 * COARSE_GRID_STRIDE
            || rows < 2 * COARSE_GRID_STRIDE
        {
            self.ray_to_pixel_grid_exact(origin, col_step, row_step, cols, rows, out);
            return;
        }
        let _ = self.ray_to_pixel_grid_coarse(origin, col_step, row_step, cols, rows, out);
    }

    /// Coarse-grid interpolation path for [`Self::ray_to_pixel_grid`]
    /// (non-perspective models). See that method's docs and
    /// `specs/core/ray-grid-projection.md`.
    /// Returns `(interpolated_cells, total_cells)` for diagnostics/tests — the
    /// hit rate of the fast (interpolated) path on this tile.
    ///
    /// The error is bounded **by construction**: each sub-grid cell is accepted
    /// for bilinear interpolation only after its center and edge-midpoints — the
    /// points where bilinear is least accurate — are projected exactly and agree
    /// with the interpolant to within [`COARSE_GRID_TOL_PX`]. Cells that fail the
    /// probe (high curvature, or an invalid corner) are projected exactly per
    /// pixel, so the worst-case deviation from the exact map stays at the
    /// tolerance regardless of geometry.
    pub(super) fn ray_to_pixel_grid_coarse(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) -> (usize, usize) {
        let gp = self.grid_proj();
        let proj = |c: f64, r: f64| {
            let ray = [
                origin[0] + c * col_step[0] + r * row_step[0],
                origin[1] + c * col_step[1] + r * row_step[1],
                origin[2] + c * col_step[2] + r * row_step[2],
            ];
            self.project_ray_node(ray, gp)
        };

        // Sub-grid nodes at 0, stride, 2·stride, …, with the final node forced to
        // (axis − 1) so the endpoints are always nodes. Positions are computed
        // arithmetically (no node-index Vec): node `i` sits at `min(i·stride, n−1)`,
        // and there are `ceil((n−1)/stride)` cells.
        let stride = COARSE_GRID_STRIDE;
        let n_cells = |n: u32| (n - 1).div_ceil(stride) as usize;
        let node_pos = |n: u32, i: usize| (i as u32 * stride).min(n - 1);
        let (seg_c, seg_r) = (n_cells(cols), n_cells(rows));
        let nc = seg_c + 1;

        // Project each sub-grid node exactly, once (shared by adjacent cells).
        let mut node_px = vec![[f32::NAN; 2]; nc * (seg_r + 1)];
        for j in 0..=seg_r {
            let r = node_pos(rows, j) as f64;
            for i in 0..=seg_c {
                node_px[j * nc + i] = proj(node_pos(cols, i) as f64, r);
            }
        }

        // `[(0.5,0.5),(0.5,0),(0.5,1),(0,0.5),(1,0.5)]` are the bilinear-error
        // extrema for a separable-quadratic warp.
        const PROBES: [(f32, f32); 5] =
            [(0.5, 0.5), (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)];
        let cols_u = cols as usize;
        let mut interp_count = 0usize;

        // Walk cells. Each cell owns the half-open pixel block `[r0,r1) × [c0,c1)`;
        // the last cell on each axis additionally owns the final node row/column,
        // so every pixel is written exactly once.
        for j in 0..seg_r {
            let (r0, r1) = (node_pos(rows, j), node_pos(rows, j + 1));
            let r_end = if j == seg_r - 1 { r1 + 1 } else { r1 };
            let inv_ch = 1.0 / (r1 - r0) as f32;
            for i in 0..seg_c {
                let (c0, c1) = (node_pos(cols, i), node_pos(cols, i + 1));
                let c_end = if i == seg_c - 1 { c1 + 1 } else { c1 };
                let inv_cw = 1.0 / (c1 - c0) as f32;
                let p00 = node_px[j * nc + i];
                let p10 = node_px[j * nc + i + 1];
                let p01 = node_px[(j + 1) * nc + i];
                let p11 = node_px[(j + 1) * nc + i + 1];

                // Interpolate the cell only if every corner is valid and the probe
                // points agree with the interpolant to within the tolerance.
                let interp = p00[0].is_finite()
                    && p10[0].is_finite()
                    && p01[0].is_finite()
                    && p11[0].is_finite()
                    && PROBES.iter().all(|&(sf, tf)| {
                        let est = bilerp(p00, p10, p01, p11, sf, tf);
                        let c = c0 as f64 + sf as f64 * (c1 - c0) as f64;
                        let r = r0 as f64 + tf as f64 * (r1 - r0) as f64;
                        let ex = proj(c, r);
                        ex[0].is_finite()
                            && (est[0] - ex[0]).hypot(est[1] - ex[1]) <= COARSE_GRID_TOL_PX
                    });
                if interp {
                    interp_count += 1;
                }

                for row in r0..r_end {
                    let tf = (row - r0) as f32 * inv_ch;
                    let base = row as usize * 2 * cols_u;
                    for col in c0..c_end {
                        let o = base + 2 * col as usize;
                        let p = if interp {
                            bilerp(p00, p10, p01, p11, (col - c0) as f32 * inv_cw, tf)
                        } else {
                            proj(col as f64, row as f64)
                        };
                        out[o] = p[0];
                        out[o + 1] = p[1];
                    }
                }
            }
        }
        (interp_count, seg_c * seg_r)
    }
}
