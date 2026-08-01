// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the per-image observation coverage grids.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::analysis::observation_coverage::{
    ObservationCoverage, DEFAULT_CELL_PX, MAX_SECTORS,
};

/// Per-image occupancy grids over the image-space footprints of a
/// reconstruction's observations.
///
/// Each observation claims a disk of the given radius around its keypoint; a
/// grid cell of ``cell_px`` pixels counts how many footprints contain its
/// center, saturating at 255. Build once, then ask batch questions of it:
/// which regions no track claims yet (aim new candidates there), which
/// directions around a point have no coverage (reach expansion that way), and
/// which regions are contested by many claims (a duplicate / alias signal).
///
/// Args:
///     image_sizes: Per image, shape ``(I, 2)`` uint32 ``[width, height]`` px.
///     track_image_indexes: Per observation, shape ``(N,)`` uint32.
///     keypoints_xy: Per observation, shape ``(N, 2)`` float64.
///     radii_px: Per observation footprint radius, shape ``(N,)`` float32.
///         Non-positive or non-finite radii contribute nothing.
///     cell_px: Grid cell size in pixels (default 4).
///
/// See specs/core/observation-coverage.md.
#[pyclass(name = "ObservationCoverage", module = "sfmtool.analysis")]
pub struct PyObservationCoverage {
    inner: ObservationCoverage,
}

#[pymethods]
impl PyObservationCoverage {
    #[new]
    #[pyo3(signature = (image_sizes, track_image_indexes, keypoints_xy, radii_px, cell_px = DEFAULT_CELL_PX))]
    fn new(
        py: Python<'_>,
        image_sizes: PyReadonlyArray2<u32>,
        track_image_indexes: PyReadonlyArray1<u32>,
        keypoints_xy: PyReadonlyArray2<f64>,
        radii_px: PyReadonlyArray1<f32>,
        cell_px: u32,
    ) -> PyResult<Self> {
        if cell_px == 0 {
            return Err(PyValueError::new_err("cell_px must be positive"));
        }
        if image_sizes.shape()[1] != 2 {
            return Err(PyValueError::new_err("image_sizes must have shape (I, 2)"));
        }
        if keypoints_xy.shape()[1] != 2 {
            return Err(PyValueError::new_err("keypoints_xy must have shape (N, 2)"));
        }
        let n_obs = keypoints_xy.shape()[0];
        for (name, len) in [
            ("track_image_indexes", track_image_indexes.shape()[0]),
            ("radii_px", radii_px.shape()[0]),
        ] {
            if len != n_obs {
                return Err(PyValueError::new_err(format!(
                    "{name} has {len} entries but keypoints_xy has {n_obs}"
                )));
            }
        }

        let sizes_data = to_contiguous!(image_sizes);
        let image_idx_data = to_contiguous!(track_image_indexes);
        let keypoints_data = to_contiguous!(keypoints_xy);
        let radii_data = to_contiguous!(radii_px);

        let n_images = image_sizes.shape()[0];
        if let Some(&bad) = image_idx_data.iter().find(|&&i| i as usize >= n_images) {
            return Err(PyValueError::new_err(format!(
                "track_image_indexes contains {bad}, out of range for {n_images} images"
            )));
        }

        let sizes: Vec<[u32; 2]> = sizes_data.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let keypoints: Vec<[f64; 2]> = keypoints_data
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();

        let inner = py.detach(|| {
            ObservationCoverage::build(&sizes, &image_idx_data, &keypoints, &radii_data, cell_px)
        });
        Ok(Self { inner })
    }

    /// Number of images the coverage spans.
    #[getter]
    fn image_count(&self) -> usize {
        self.inner.image_count()
    }

    /// Grid cell size in pixels.
    #[getter]
    fn cell_px(&self) -> u32 {
        self.inner.cell_px()
    }

    /// One image's counts as a ``(height_cells, width_cells)`` uint8 array.
    fn grid<'py>(&self, py: Python<'py>, image_index: usize) -> PyResult<Py<PyAny>> {
        let (cells, w, h) = self.inner.grid(image_index).ok_or_else(|| {
            PyValueError::new_err(format!(
                "image_index {image_index} is out of range for {} images",
                self.inner.image_count()
            ))
        })?;
        let out = PyArray1::from_slice(py, cells).reshape([h as usize, w as usize])?;
        Ok(out.into_any().unbind())
    }

    /// The count of the cell containing each pixel coordinate.
    ///
    /// Args:
    ///     image_indexes: Shape ``(M,)`` uint32.
    ///     xy: Shape ``(M, 2)`` float64 pixel coordinates.
    ///
    /// Returns:
    ///     ``(M,)`` uint8 array; 0 for a coordinate outside the grid.
    fn counts_at<'py>(
        &self,
        py: Python<'py>,
        image_indexes: PyReadonlyArray1<u32>,
        xy: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let (images, points) = self.check_batch(&image_indexes, &xy)?;
        let counts = py.detach(|| self.inner.counts_at(&images, &points));
        Ok(PyArray1::from_vec(py, counts).into_any().unbind())
    }

    /// Of the cells whose centers lie within ``radius_px`` of each query point,
    /// the fraction with a non-zero count.
    ///
    /// Args:
    ///     image_indexes: Shape ``(M,)`` uint32.
    ///     xy: Shape ``(M, 2)`` float64 pixel coordinates.
    ///     radius_px: Shape ``(M,)`` float32 query radii.
    ///
    /// Returns:
    ///     ``(M,)`` float32 array; 0 when no cell center falls in the disk.
    fn covered_fraction<'py>(
        &self,
        py: Python<'py>,
        image_indexes: PyReadonlyArray1<u32>,
        xy: PyReadonlyArray2<f64>,
        radius_px: PyReadonlyArray1<f32>,
    ) -> PyResult<Py<PyAny>> {
        let (images, points) = self.check_batch(&image_indexes, &xy)?;
        let radii = check_radii(&radius_px, points.len())?;
        let fractions = py.detach(|| self.inner.covered_fraction(&images, &points, &radii));
        Ok(PyArray1::from_vec(py, fractions).into_any().unbind())
    }

    /// Bitmask of the angular sectors around each query point that still hold
    /// an uncovered cell — the directions worth reaching into.
    ///
    /// Sector ``k`` spans ``[k, k + 1) * 2*pi / n_sectors`` of ``atan2(dy, dx)``
    /// of the cell center relative to the query point, and bit ``k`` is set when
    /// that sector holds at least one in-grid cell with count 0. Sectors with no
    /// in-grid cell contribute no bit, and the cell exactly at the query point
    /// has no direction and is skipped.
    ///
    /// Args:
    ///     image_indexes: Shape ``(M,)`` uint32.
    ///     xy: Shape ``(M, 2)`` float64 pixel coordinates.
    ///     radius_px: Shape ``(M,)`` float32 query radii.
    ///     n_sectors: Number of sectors, 1 to 32.
    ///
    /// Returns:
    ///     ``(M,)`` uint32 array of bitmasks.
    fn uncovered_sectors<'py>(
        &self,
        py: Python<'py>,
        image_indexes: PyReadonlyArray1<u32>,
        xy: PyReadonlyArray2<f64>,
        radius_px: PyReadonlyArray1<f32>,
        n_sectors: u32,
    ) -> PyResult<Py<PyAny>> {
        if n_sectors == 0 || n_sectors > MAX_SECTORS {
            return Err(PyValueError::new_err(format!(
                "n_sectors must be between 1 and {MAX_SECTORS}, got {n_sectors}"
            )));
        }
        let (images, points) = self.check_batch(&image_indexes, &xy)?;
        let radii = check_radii(&radius_px, points.len())?;
        let masks = py.detach(|| {
            self.inner
                .uncovered_sectors(&images, &points, &radii, n_sectors)
        });
        Ok(PyArray1::from_vec(py, masks).into_any().unbind())
    }

    /// Fraction of one image's cells with a non-zero count.
    fn image_covered_fraction(&self, image_index: usize) -> PyResult<f32> {
        if image_index >= self.inner.image_count() {
            return Err(PyValueError::new_err(format!(
                "image_index {image_index} is out of range for {} images",
                self.inner.image_count()
            )));
        }
        Ok(self.inner.image_covered_fraction(image_index))
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservationCoverage(images={}, cell_px={})",
            self.inner.image_count(),
            self.inner.cell_px()
        )
    }
}

impl PyObservationCoverage {
    /// Validate the two arrays every batch query shares and unpack them.
    fn check_batch(
        &self,
        image_indexes: &PyReadonlyArray1<u32>,
        xy: &PyReadonlyArray2<f64>,
    ) -> PyResult<(Vec<u32>, Vec<[f64; 2]>)> {
        if xy.shape()[1] != 2 {
            return Err(PyValueError::new_err("xy must have shape (M, 2)"));
        }
        let n = xy.shape()[0];
        if image_indexes.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "image_indexes has {} entries but xy has {n}",
                image_indexes.shape()[0]
            )));
        }
        let images: Vec<u32> = to_contiguous!(image_indexes).into_owned();
        let n_images = self.inner.image_count();
        if let Some(&bad) = images.iter().find(|&&i| i as usize >= n_images) {
            return Err(PyValueError::new_err(format!(
                "image_indexes contains {bad}, out of range for {n_images} images"
            )));
        }
        let points: Vec<[f64; 2]> = to_contiguous!(xy)
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        Ok((images, points))
    }
}

/// Validate a per-query radius array against the query count.
fn check_radii(radius_px: &PyReadonlyArray1<f32>, n: usize) -> PyResult<Vec<f32>> {
    if radius_px.shape()[0] != n {
        return Err(PyValueError::new_err(format!(
            "radius_px has {} entries but xy has {n}",
            radius_px.shape()[0]
        )));
    }
    Ok(to_contiguous!(radius_px).into_owned())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObservationCoverage>()?;
    Ok(())
}
