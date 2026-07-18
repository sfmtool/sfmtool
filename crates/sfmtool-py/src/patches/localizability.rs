// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.score_localizability` + `window_weights`: keypoint
//! positional-uncertainty scoring.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::localizability::{
    score_localizability_stack, window_weights as localizability_window_weights,
};

use super::args::{np_median, parse_patch_window};
use super::cloud::PyPatchCloud;
use crate::PySfmrReconstruction;

#[pymethods]
impl PyPatchCloud {
    /// Score each point's **keypoint positional uncertainty** ``σ_pos`` (source-
    /// image px) from its cross-view consensus ``patch_bitmaps`` and the recon
    /// geometry — the patch-localizability score (see
    /// ``specs/core/patch-localizability.md``). No source images are read.
    ///
    /// The noise-normalized structure tensor of each consensus patch gives the
    /// weak-axis uncertainty in patch-grid px (``σ_pos_grid = σ_noise /
    /// √λ₂_sum``); that is mapped to source px per observing view by the patch's
    /// projected scale (``half_extent / (R/2) · f / depth``, canonical ``depth =
    /// −(R·X + t)_z``) and reduced by the **median** over the point's views.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides poses,
    ///         intrinsics, point positions, and the per-point observing-view lists
    ///         via its tracks). The cloud's ``point_indexes`` must index it.
    ///     patch_bitmaps: An ``(N, R, R, C)`` uint8 consensus stack scattered per
    ///         **source 3D point** (``N`` = ``recon`` point count; zero rows for
    ///         points with no consensus) — e.g. ``recon.patch_bitmaps`` or the
    ///         ``bitmaps`` from a ``render_bitmaps`` refine pass. Luminance
    ///         (``0.299R + 0.587G + 0.114B``) is scored; alpha is used only to tell
    ///         a covered flat patch from an empty (culled) one.
    ///     sigma_noise: Global consensus photometric-noise constant (intensity
    ///         units, ~3 gray levels for a u8 consensus). Sets the absolute px
    ///         scale of ``σ_pos``; the per-recon ranking is scale-free.
    ///     window: Scoring window — ``"gaussian_disk"`` (default), ``"gaussian"``,
    ///         or ``"uniform"`` — matched to the consensus render window.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///
    /// Returns a dict of per-**source-point** numpy arrays (length ``N``, parallel
    /// to ``recon``'s points): ``sigma_pos_grid`` (grid px — **the cull quantity**;
    /// intrinsic and resolution-independent, so a fixed threshold transfers across
    /// datasets; NaN where the patch is empty), ``sigma_pos_px`` (source px, via the
    /// grid→px map, median over views — a diagnostic that does *not* transfer;
    /// additionally NaN where no view gives a finite depth), ``lam1``/``lam2``
    /// (structure-tensor eigenvalues, summed form), and ``theta`` (radians; the
    /// weak-axis / slide direction).
    #[pyo3(signature = (
        recon, patch_bitmaps, *, sigma_noise=3.0, window="gaussian_disk", window_sigma=0.6
    ))]
    fn score_localizability<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        patch_bitmaps: PyReadonlyArray4<'py, u8>,
        sigma_noise: f64,
        window: &str,
        window_sigma: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let recon = &recon.inner;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }
        let bm = patch_bitmaps.as_array();
        let (n, r, r2, c) = (bm.shape()[0], bm.shape()[1], bm.shape()[2], bm.shape()[3]);
        if r != r2 {
            return Err(PyValueError::new_err(format!(
                "patch_bitmaps must be square R×R per point, got {r}×{r2}"
            )));
        }
        if n != recon.points.len() {
            return Err(PyValueError::new_err(format!(
                "patch_bitmaps has {n} rows but the reconstruction has {} points; \
                 pass a stack scattered per source 3D point",
                recon.points.len()
            )));
        }
        let window = parse_patch_window(window, window_sigma)?;

        // Logical (row-major) copy of the consensus stack as f32. `iter()` walks
        // the array in C order regardless of its physical layout, so the flat
        // buffer is the `(N, R, R, C)` the batch scorer expects.
        let flat: Vec<f32> = bm.iter().map(|&v| v as f32).collect();

        // Per-point half-extent (world) from the cloud frames, scattered to source
        // points; NaN for points the cloud has no patch for.
        let mut half = vec![f64::NAN; n];
        for (i, &pid) in self.inner.point_indexes.iter().enumerate() {
            half[pid as usize] = self.inner.patch(i).half_extent[0];
        }
        // One pose + mean focal per image.
        let poses: Vec<RigidTransform> = recon
            .images
            .iter()
            .map(|im| {
                let q = im.quaternion_wxyz;
                RigidTransform::from_wxyz_translation(
                    [q.w, q.i, q.j, q.k],
                    [
                        im.translation_xyz.x,
                        im.translation_xyz.y,
                        im.translation_xyz.z,
                    ],
                )
            })
            .collect();
        let focal: Vec<f64> = recon
            .images
            .iter()
            .map(|im| {
                let (fx, fy) = recon.cameras[im.camera_index as usize].focal_lengths();
                0.5 * (fx + fy)
            })
            .collect();

        // The structure-tensor scoring is the expensive, rayon-parallel part; run
        // it with the GIL released. The grid→source-px map below is a cheap
        // per-observation pass, so it stays on the GIL-holding thread.
        let scores = py.detach(|| score_localizability_stack(&flat, n, r, c, window, sigma_noise));

        // Median-over-views grid→source-px scale per point (canonical −Z depth:
        // `depth = -(R·X + t)_z`).
        let half_r = r as f64 / 2.0;
        let mut scale = vec![f64::NAN; n];
        for (p_idx, s) in scale.iter_mut().enumerate() {
            let h = half[p_idx];
            if !h.is_finite() {
                continue;
            }
            let point = &recon.points[p_idx];
            let mut vals: Vec<f64> = Vec::new();
            for obs in &recon.tracks
                [recon.observation_offsets[p_idx]..recon.observation_offsets[p_idx + 1]]
            {
                let im = obs.image_index as usize;
                // Homogeneous transform so a point at infinity (`w == 0`, whose
                // `position` is a direction) is rotated without translation —
                // `R·d` rather than `R·d + t` — giving a meaningful bearing depth.
                let depth = -poses[im]
                    .transform_point_homogeneous(point.position.coords, point.w)
                    .z;
                if depth > 1e-6 {
                    vals.push((h / half_r) * focal[im] / depth);
                }
            }
            if !vals.is_empty() {
                *s = np_median(&mut vals);
            }
        }

        let mut sigma_px = Vec::with_capacity(n);
        let mut sigma_grid = Vec::with_capacity(n);
        let mut lam1 = Vec::with_capacity(n);
        let mut lam2 = Vec::with_capacity(n);
        let mut theta = Vec::with_capacity(n);
        for (s, &sc) in scores.iter().zip(&scale) {
            sigma_px.push(s.sigma_pos_grid * sc);
            sigma_grid.push(s.sigma_pos_grid);
            lam1.push(s.lam1);
            lam2.push(s.lam2);
            theta.push(s.theta);
        }

        let out = PyDict::new(py);
        out.set_item("sigma_pos_px", sigma_px.into_pyarray(py))?;
        out.set_item("sigma_pos_grid", sigma_grid.into_pyarray(py))?;
        out.set_item("lam1", lam1.into_pyarray(py))?;
        out.set_item("lam2", lam2.into_pyarray(py))?;
        out.set_item("theta", theta.into_pyarray(py))?;
        Ok(out)
    }

    /// The scorer's `R×R` window weights (row-major), for `window` (default
    /// ``"gaussian_disk"``) at `window_sigma`. The same kernel
    /// [`score_localizability`](Self::score_localizability) accumulates the
    /// structure tensor over — exposed so callers score against it rather than
    /// reimplementing the window formula.
    #[staticmethod]
    #[pyo3(signature = (resolution, *, window="gaussian_disk", window_sigma=0.6))]
    fn window_weights<'py>(
        py: Python<'py>,
        resolution: usize,
        window: &str,
        window_sigma: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = parse_patch_window(window, window_sigma)?;
        let weights = localizability_window_weights(w, resolution as u32);
        let arr = Array2::from_shape_vec((resolution, resolution), weights)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }
}
