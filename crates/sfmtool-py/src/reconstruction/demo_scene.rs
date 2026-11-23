// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the demo scene (`sfmtool_core::reconstruction::DemoScene`):
//! a textured ground plane with a cube on it, photographed by five pinhole
//! cameras, with its reconstruction written down from the scene.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use numpy::ndarray::ArrayView3;
use numpy::{PyArray, PyArray3};
use pyo3::exceptions::{PyInterruptedError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::progress::{Event, Progress};
use sfmtool_core::reconstruction::{DemoScene, DemoSceneError, DemoSceneOptions};

use crate::helpers::read_only;
use crate::signals::SignalWatch;
use crate::{ProgressCounter, PySfmrReconstruction};

/// The Python exception for `e`: `ValueError` for an option out of its range
/// or a directory `write_workspace` refuses, `InterruptedError` for a
/// cancelled call, and the `OSError` for a photograph that could not be
/// encoded, decoded or written, a workspace directory or file that could not
/// be written, and a `.sfmr` that could not be saved.
fn to_py_err(e: DemoSceneError) -> PyErr {
    let message = e.to_string();
    match e {
        DemoSceneError::Options(_) | DemoSceneError::Directory(_) => PyValueError::new_err(message),
        DemoSceneError::Cancelled => PyInterruptedError::new_err(message),
        DemoSceneError::Io(source) => PyErr::from(std::io::Error::new(source.kind(), message)),
        DemoSceneError::Photograph(_) | DemoSceneError::Save(_) => PyOSError::new_err(message),
    }
}

/// Run `call` with the GIL released, its `Progress` setting `progress` to the
/// count of the phase that is running and cancelled when a Python signal
/// handler raises, as Ctrl+C does; that handler's exception is raised in
/// place of the call's result.
fn run_with_progress<T: Send>(
    py: Python<'_>,
    progress: Option<ProgressCounter>,
    call: impl FnOnce(&Progress<'_>) -> Result<T, DemoSceneError> + Send,
) -> PyResult<T> {
    let counter = progress.as_ref().map(ProgressCounter::handle);
    let watch = SignalWatch::new();
    let sink = |event: Event<'_>| {
        if let (Event::Count { done, .. }, Some(counter)) = (event, &counter) {
            counter.store(done as usize, Ordering::Relaxed);
        }
        watch.check();
    };
    let result = py.detach(|| call(&Progress::to(&sink).cancelled_by(watch.cancel_flag())));
    watch.finish(result)?.map_err(to_py_err)
}

/// The demo scene: a textured ground plane with a 1 m cube standing on it,
/// photographed by five pinhole cameras, with the reconstruction written down
/// from the scene, so that every pose, point, keypoint, normal and patch is
/// known to be right. See ``specs/drafts/demo-scene.md``.
///
/// Made by :meth:`render`. The photographs are read-only ``(H, W, 3)``
/// ``uint8`` arrays, RGB, in image order, each a view of the scene's own
/// pixels that copies nothing; a caller that wants to edit one takes
/// ``.copy()``.
#[pyclass(name = "DemoScene", module = "sfmtool.reconstruction", frozen)]
pub struct PyDemoScene {
    inner: DemoScene,
}

/// A read-only numpy view of `image`, which `owner` holds, with `owner` as
/// its base.
fn photograph_view<'py>(
    owner: &Bound<'py, PyDemoScene>,
    image: &ImageU8,
) -> Bound<'py, PyArray3<u8>> {
    let shape = (
        image.height() as usize,
        image.width() as usize,
        image.channels() as usize,
    );
    let block = ArrayView3::from_shape(shape, image.data())
        .expect("an ImageU8 holds height * width * channels bytes");
    // SAFETY: the base of the view is `owner`, which owns `image`. The class
    // is frozen and has no method that changes its scene, so the pixels are
    // not moved, reallocated or written while `owner` lives.
    let array = unsafe { PyArray::borrow_from_array(&block, owner.clone().into_any()) };
    read_only(array)
}

#[pymethods]
impl PyDemoScene {
    /// Render the scene's five photographs, encode each as a JPEG and decode
    /// it again.
    ///
    /// The keywords are the scene's options, each with the default the spec
    /// states: the photographs' size, the pinhole focal length in pixels, the
    /// supersampling (rays per pixel along each axis), the spacings of the
    /// point grids on the ground and on each cube face in metres, each
    /// point's patch half-extent in metres, the patch bitmaps' side, the
    /// largest angle in degrees between a patch's normal and a camera that
    /// observes it, and the seed that picks the textures and the points'
    /// offsets. A spacing left as ``None`` is its default, 0.25 on the ground
    /// and 0.125 on the faces. ``point_count`` gives both spacings instead,
    /// scaled from the defaults so that the scene keeps about that many
    /// points, and cannot be given with either spacing.
    ///
    /// ``progress``, a ``ProgressCounter``, is set to the count of the phase
    /// that is running: the rows rendered, then the images encoded. The GIL
    /// is released while the scene renders.
    ///
    /// Raises:
    ///     ValueError: for an option out of its range, or ``point_count``
    ///         given with a spacing or as 0.
    ///     OverflowError: for a negative integer, or one too large for its
    ///         option's type.
    ///     OSError: when a photograph cannot be encoded or decoded.
    ///     The exception a signal handler raised while the call ran
    ///     (``KeyboardInterrupt`` for Ctrl+C), which stops it.
    ///     InterruptedError: when the call was cancelled otherwise.
    #[staticmethod]
    #[pyo3(signature = (
        *,
        image_width=640,
        image_height=480,
        focal_length_px=560.0,
        supersampling=2,
        ground_point_spacing=None,
        face_point_spacing=None,
        patch_half_extent=0.09,
        patch_bitmap_resolution=24,
        max_obliquity_deg=80.0,
        seed=0,
        point_count=None,
        progress=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn render(
        py: Python<'_>,
        image_width: u32,
        image_height: u32,
        focal_length_px: f64,
        supersampling: u32,
        ground_point_spacing: Option<f64>,
        face_point_spacing: Option<f64>,
        patch_half_extent: f64,
        patch_bitmap_resolution: u32,
        max_obliquity_deg: f64,
        seed: u64,
        point_count: Option<u32>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Self> {
        let base = match point_count {
            Some(_) if ground_point_spacing.is_some() || face_point_spacing.is_some() => {
                return Err(PyValueError::new_err(
                    "point_count gives both point spacings; give it or the spacings, not both",
                ));
            }
            Some(count) => DemoSceneOptions::for_point_count(count).map_err(to_py_err)?,
            None => DemoSceneOptions::default(),
        };
        let options = DemoSceneOptions {
            image_width,
            image_height,
            focal_length_px,
            supersampling,
            ground_point_spacing: ground_point_spacing.unwrap_or(base.ground_point_spacing),
            face_point_spacing: face_point_spacing.unwrap_or(base.face_point_spacing),
            patch_half_extent,
            patch_bitmap_resolution,
            max_obliquity_deg,
            seed,
        };
        let inner = run_with_progress(py, progress, |p| DemoScene::render(&options, p))?;
        Ok(Self { inner })
    }

    /// Each photograph's image name, ``images/demo_0.jpg`` to
    /// ``images/demo_4.jpg``.
    #[getter]
    fn image_names(&self) -> Vec<String> {
        self.inner.image_names().to_vec()
    }

    /// The renderer's exact output, before JPEG: a list of read-only
    /// ``(H, W, 3)`` ``uint8`` arrays.
    #[getter]
    fn rendered_photographs<'py>(slf: &Bound<'py, Self>) -> Vec<Bound<'py, PyArray3<u8>>> {
        let scene = &slf.get().inner;
        scene
            .rendered_photographs()
            .iter()
            .map(|image| photograph_view(slf, image))
            .collect()
    }

    /// The photographs as a reader of the written files decodes them, each
    /// JPEG of :attr:`encoded_photographs` decoded: a list of read-only
    /// ``(H, W, 3)`` ``uint8`` arrays. The reconstruction's thumbnails,
    /// readings and patch bitmaps are taken from these.
    #[getter]
    fn photographs<'py>(slf: &Bound<'py, Self>) -> Vec<Bound<'py, PyArray3<u8>>> {
        let scene = &slf.get().inner;
        scene
            .photographs()
            .iter()
            .map(|image| photograph_view(slf, image))
            .collect()
    }

    /// Each photograph's JPEG file, the bytes :meth:`write_workspace` writes,
    /// as a list of ``bytes``.
    #[getter]
    fn encoded_photographs<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .encoded_photographs()
            .iter()
            .map(|bytes| PyBytes::new(py, bytes))
            .collect()
    }

    /// The scene's ``embedded_patches`` reconstruction, with no files behind
    /// it: the points sampled on the surfaces, the cameras that see each one,
    /// the patch bitmaps rendered from :attr:`photographs`, and a 1 m distance
    /// measurement along a cube edge.
    ///
    /// ``progress``, a ``ProgressCounter``, is set to the count of the phase
    /// that is running: the points sampled, then the bitmaps rendered. The GIL
    /// is released while the value is built.
    ///
    /// Raises:
    ///     The exception a signal handler raised while the call ran
    ///     (``KeyboardInterrupt`` for Ctrl+C), which stops it.
    ///     InterruptedError: when the call was cancelled otherwise.
    #[pyo3(signature = (progress=None))]
    fn reconstruction(
        &self,
        py: Python<'_>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<PySfmrReconstruction> {
        let recon = run_with_progress(py, progress, |p| self.inner.reconstruction(p))?;
        Ok(PySfmrReconstruction::new(recon))
    }

    /// Write the scene as a workspace in ``dir``, which must be an empty
    /// directory or not exist, and must not lie inside another workspace: the
    /// JPEGs under ``images/``, a ``.sfm-workspace.json``, and
    /// :meth:`reconstruction`'s value as ``demo.sfmr`` at the workspace root.
    /// Returns the ``demo.sfmr`` path as a ``pathlib.Path``.
    ///
    /// ``progress``, a ``ProgressCounter``, is set to the count of the phase
    /// that is running: the points sampled, the bitmaps rendered, then the
    /// files written. The value is built before any file is written, so a
    /// call stopped while building writes nothing. The GIL is released while
    /// the call runs.
    ///
    /// Raises:
    ///     ValueError: when ``dir`` is not a directory, holds files or lies
    ///         inside another workspace.
    ///     OSError: when a photograph, a directory, the workspace file or the
    ///         ``.sfmr`` cannot be written.
    ///     The exception a signal handler raised while the call ran
    ///     (``KeyboardInterrupt`` for Ctrl+C), which stops it.
    ///     InterruptedError: when the call was cancelled otherwise.
    #[pyo3(signature = (dir, progress=None))]
    fn write_workspace(
        &self,
        py: Python<'_>,
        dir: PathBuf,
        progress: Option<ProgressCounter>,
    ) -> PyResult<PathBuf> {
        run_with_progress(py, progress, |p| self.inner.write_workspace(&dir, p))
    }
}
