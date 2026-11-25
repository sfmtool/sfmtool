// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scene/pyramid plumbing shared by every patch kernel: `CameraViews`,
//! `ImagePyramidSet`, and the reconstruction-or-views resolution helpers.

use std::path::PathBuf;
use std::sync::Arc;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::{PhotographCache, PhotographCacheOptions, TiledPhotograph};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::PatchCloud;
use sfmtool_core::CameraIntrinsics;

use crate::flow::warp::extract_image_u8;
use crate::photographs::{get_photographs, PyPhotographCache};
use crate::{ProgressCounter, PySfmrReconstruction};

/// Validate that image `i` (dimensions `w × h`) matches its camera's resolution.
pub(super) fn check_image_matches_camera(
    cam: &CameraIntrinsics,
    i: usize,
    w: u32,
    h: u32,
) -> PyResult<()> {
    if w != cam.width || h != cam.height {
        return Err(PyValueError::new_err(format!(
            "image {i} is {w}x{h} but its camera is {}x{}; \
             pass full-resolution images",
            cam.width, cam.height
        )));
    }
    Ok(())
}

/// Extract each numpy image and make the tiled photograph a kernel reads of
/// it, [`TiledPhotograph::from_image`]: every tile of its full-depth pyramid
/// (`ImageU8Pyramid::full_levels`), so a prebuilt [`PyImagePyramidSet`] is
/// tile for tile what a per-call build makes. `check` runs on every decoded
/// image (index, decoded source). The extraction runs under the GIL; the
/// pyramid build and the cutting of its tiles (the expensive part) run
/// GIL-free and rayon-parallel. Shared by the recon-coupled
/// [`build_photographs_from_cameras`] (camera-dimension check) and the
/// recon-free cluster-refinement binding (`matching::cluster`, no check).
pub(crate) fn build_photographs_from_image_list(
    py: Python<'_>,
    images: &[Bound<'_, PyAny>],
    mut check: impl FnMut(usize, &ImageU8) -> PyResult<()>,
) -> PyResult<Vec<TiledPhotograph>> {
    let srcs: Vec<ImageU8> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let src = extract_image_u8(im)?;
            check(i, &src)?;
            Ok(src)
        })
        .collect::<PyResult<_>>()?;
    Ok(py.detach(|| {
        srcs.into_par_iter()
            .map(TiledPhotograph::from_image)
            .collect()
    }))
}

/// Extract each numpy image and make its tiled photograph, validating
/// dimensions against the per-view cameras (one camera per image, parallel
/// to `images`), as [`build_photographs_from_image_list`] makes them.
pub(super) fn build_photographs_from_cameras(
    py: Python<'_>,
    cameras: &[CameraIntrinsics],
    images: &[Bound<'_, PyAny>],
) -> PyResult<Vec<TiledPhotograph>> {
    if images.len() != cameras.len() {
        return Err(PyValueError::new_err(format!(
            "images must be parallel to the scene's {} views, got {}",
            cameras.len(),
            images.len()
        )));
    }
    build_photographs_from_image_list(py, images, |i, src| {
        check_image_matches_camera(&cameras[i], i, src.width(), src.height())
    })
}

/// The per-image photographs a kernel call reads: either owned (built from a numpy
/// image list for this one call) or shared (an [`PyImagePyramidSet`] handle,
/// built once and reused across calls).
pub(crate) enum PyramidSet {
    Owned(Vec<TiledPhotograph>),
    Shared(Arc<Vec<TiledPhotograph>>),
}

impl PyramidSet {
    pub(crate) fn as_slice(&self) -> &[TiledPhotograph] {
        match self {
            PyramidSet::Owned(v) => v,
            PyramidSet::Shared(a) => a,
        }
    }
}

/// The camera + pose inputs a patch kernel reads, materialized once per call from
/// either a [`PySfmrReconstruction`] or a [`PyCameraViews`]: one owned camera and
/// one `cam_from_world` pose per image (index = image index). Both scene sources
/// reduce to this, so [`resolve_pyramids`] and the kernels'
/// [`sfmtool_core::patch::normal_refine::ProjectedImage`] assembly are identical
/// for a reconstruction and a bare set of views.
pub(crate) struct PosedViews {
    /// Per-image intrinsics: each view's camera, with the image's intrinsic
    /// adjustment applied for a reconstruction.
    pub(crate) cameras: Vec<CameraIntrinsics>,
    /// Per-image `cam_from_world` pose.
    pub(crate) poses: Vec<RigidTransform>,
}

impl PosedViews {
    pub(super) fn len(&self) -> usize {
        self.cameras.len()
    }

    pub(crate) fn from_reconstruction(recon: &sfmtool_core::SfmrReconstruction) -> Self {
        // Each image's intrinsics: its camera's with its adjustment applied.
        let cameras = (0..recon.image_set.images.len())
            .map(|i| recon.image_intrinsics(i).into_owned())
            .collect();
        let poses = recon
            .image_set
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
        Self { cameras, poses }
    }
}

/// The scene passed as a patch kernel's first argument, resolved to owned
/// [`PosedViews`] plus (when it is a reconstruction) a borrow of it for the
/// track-derived defaults, per-observation keypoints, and point-range check that
/// only a reconstruction carries. A [`PyCameraViews`] has none of those, so its
/// `recon` is `None` and the caller must supply the per-patch view lists.
pub(super) fn resolve_scene<'py>(
    scene: &Bound<'py, PyAny>,
) -> PyResult<(PosedViews, Option<PyRef<'py, PySfmrReconstruction>>)> {
    if let Ok(views) = scene.cast::<PyCameraViews>() {
        Ok((views.get().to_posed_views(), None))
    } else if let Ok(recon) = scene.extract::<PyRef<'py, PySfmrReconstruction>>() {
        let posed = PosedViews::from_reconstruction(recon.inner());
        Ok((posed, Some(recon)))
    } else {
        Err(PyTypeError::new_err(
            "expected an SfmrReconstruction or CameraViews as the first argument",
        ))
    }
}

/// Resolve and validate the scene shared by patch-cloud kernel bindings.
/// The point-range check applies only to reconstructions because
/// [`PyCameraViews`] carries no points. The explicit-view guard runs here,
/// before callers decode any imagery.
pub(super) fn resolve_patch_scene<'py>(
    scene: &Bound<'py, PyAny>,
    cloud: &PatchCloud,
    explicit_views_present: bool,
    view_arg_name: &str,
    derived_views_name: &str,
) -> PyResult<(PosedViews, Option<PyRef<'py, PySfmrReconstruction>>, u32)> {
    let (posed, recon_guard) = resolve_scene(scene)?;
    check_cloud_points(cloud, recon_guard.as_deref())?;
    if recon_guard.is_none() && !explicit_views_present {
        return Err(PyValueError::new_err(format!(
            "{view_arg_name} is required when the first argument is a CameraViews \
             (there are no tracks to derive {derived_views_name} from)"
        )));
    }
    let n_images = posed.len() as u32;
    Ok((posed, recon_guard, n_images))
}

/// Resolve and validate the scene of a patch-cloud kernel binding that reads
/// a reconstruction's stored per-observation data and so takes no
/// [`PyCameraViews`]: a `CameraViews` is a `ValueError` saying that `method`
/// needs a reconstruction. The checks otherwise match [`resolve_patch_scene`]'s,
/// and run before callers decode any imagery.
pub(super) fn resolve_patch_reconstruction<'py>(
    scene: &Bound<'py, PyAny>,
    cloud: &PatchCloud,
    method: &str,
) -> PyResult<(PosedViews, PyRef<'py, PySfmrReconstruction>)> {
    let (posed, recon_guard) = resolve_scene(scene)?;
    let Some(recon) = recon_guard else {
        return Err(PyValueError::new_err(format!(
            "{method} needs a reconstruction, not a CameraViews"
        )));
    };
    check_cloud_points(cloud, Some(&recon))?;
    Ok((posed, recon))
}

/// Check that `cloud` carries its per-patch point indexes and, given a
/// reconstruction, that each one is a point of it.
fn check_cloud_points(cloud: &PatchCloud, recon: Option<&PySfmrReconstruction>) -> PyResult<()> {
    if cloud.point_indexes.len() != cloud.len() {
        return Err(PyValueError::new_err(
            "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
        ));
    }
    if let Some(recon) = recon {
        if cloud
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.inner().point_set.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }
    }
    Ok(())
}

/// Resolve the `images` argument of a PatchCloud kernel method — either a
/// prebuilt [`PyImagePyramidSet`] (validated against the scene, shared) or a list
/// of numpy images (pyramids built for this call). Shared by
/// `select_views`, `localize_keypoints`, and `refine_keypoints` so they handle
/// imagery identically for a reconstruction and a `CameraViews`.
pub(crate) fn resolve_pyramids(
    posed: &PosedViews,
    images: &Bound<'_, PyAny>,
) -> PyResult<PyramidSet> {
    if let Ok(set) = images.cast::<PyImagePyramidSet>() {
        let set = set.get();
        if set.pyramids.len() != posed.len() {
            return Err(PyValueError::new_err(format!(
                "ImagePyramidSet holds {} images but the scene has {}; \
                 build the set from this scene's image list",
                set.pyramids.len(),
                posed.len()
            )));
        }
        for (i, pyr) in set.pyramids.iter().enumerate() {
            let (width, height) = pyr.layout().level_size(0);
            check_image_matches_camera(&posed.cameras[i], i, width, height)?;
        }
        Ok(PyramidSet::Shared(Arc::clone(&set.pyramids)))
    } else if let Ok(list) = images.extract::<Vec<Bound<'_, PyAny>>>() {
        Ok(PyramidSet::Owned(build_photographs_from_cameras(
            images.py(),
            &posed.cameras,
            &list,
        )?))
    } else {
        Err(PyTypeError::new_err(
            "images must be a list of HxW[xC] uint8 numpy arrays or an ImagePyramidSet",
        ))
    }
}

/// Extract a `list[CameraIntrinsics]` as owned core cameras.
pub(super) fn extract_camera_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<CameraIntrinsics>> {
    let list = obj
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("cameras must be a list of CameraIntrinsics"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let cam: crate::PyCameraIntrinsics = item.extract().map_err(|_| {
            PyTypeError::new_err("cameras must be a list of CameraIntrinsics objects")
        })?;
        out.push(cam.inner);
    }
    Ok(out)
}

/// The posed views of an in-memory scene — cameras and `cam_from_world` poses —
/// accepted anywhere a patch kernel takes a reconstruction (see
/// ``specs/core/patch/patch-cloud.md``, "Patch operations without a reconstruction").
///
/// A ``CameraViews`` carries no tracks, so the per-point view lists the kernels
/// otherwise derive from the reconstruction's tracks must be supplied explicitly:
/// ``patch_views`` (:meth:`PatchCloud.localize_keypoints`,
/// :meth:`PatchCloud.refine_keypoints`,
/// :meth:`PatchCloud.validate_member_coherence`) and ``track_views``
/// (:meth:`PatchCloud.select_views`) become required.
///
/// Args:
///     cameras: ``list[CameraIntrinsics]``; each view uses one of these (see
///         ``camera_indexes``).
///     quaternions_wxyz: ``(N, 4)`` float64 unit ``cam_from_world`` rotations.
///     translations_xyz: ``(N, 3)`` float64 ``cam_from_world`` translations.
///     camera_indexes: Optional ``(N,)`` uint32 index into ``cameras`` per view;
///         ``None`` means every view uses ``cameras[0]``.
#[pyclass(name = "CameraViews", module = "sfmtool.patches", frozen)]
pub struct PyCameraViews {
    /// Per-view resolved camera intrinsics (length ``N``).
    pub(super) cameras: Vec<CameraIntrinsics>,
    /// Per-view ``cam_from_world`` rotation (length ``N``).
    pub(super) quaternions: Vec<UnitQuaternion<f64>>,
    /// Per-view ``cam_from_world`` translation (length ``N``).
    pub(super) translations: Vec<Vector3<f64>>,
}

impl PyCameraViews {
    /// Materialize the owned cameras + poses a kernel reads.
    pub(super) fn to_posed_views(&self) -> PosedViews {
        let poses = (0..self.cameras.len())
            .map(|i| {
                let q = self.quaternions[i];
                RigidTransform::from_wxyz_translation(
                    [q.w, q.i, q.j, q.k],
                    [
                        self.translations[i].x,
                        self.translations[i].y,
                        self.translations[i].z,
                    ],
                )
            })
            .collect();
        PosedViews {
            cameras: self.cameras.clone(),
            poses,
        }
    }
}

#[pymethods]
impl PyCameraViews {
    #[new]
    #[pyo3(signature = (cameras, quaternions_wxyz, translations_xyz, camera_indexes=None))]
    fn new(
        cameras: &Bound<'_, PyAny>,
        quaternions_wxyz: PyReadonlyArray2<'_, f64>,
        translations_xyz: PyReadonlyArray2<'_, f64>,
        camera_indexes: Option<PyReadonlyArray1<'_, u32>>,
    ) -> PyResult<Self> {
        let cams = extract_camera_list(cameras)?;
        if cams.is_empty() {
            return Err(PyValueError::new_err(
                "cameras must be a non-empty list of CameraIntrinsics",
            ));
        }
        let q = quaternions_wxyz.as_array();
        let t = translations_xyz.as_array();
        let n = q.shape()[0];
        if q.shape()[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "quaternions_wxyz must have shape (N, 4), got (_, {})",
                q.shape()[1]
            )));
        }
        if t.shape()[0] != n || t.shape()[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "translations_xyz must have shape ({n}, 3), got ({}, {})",
                t.shape()[0],
                t.shape()[1]
            )));
        }
        let cam_idx: Vec<u32> = match &camera_indexes {
            Some(arr) => {
                let a = arr.as_array();
                if a.shape()[0] != n {
                    return Err(PyValueError::new_err(format!(
                        "camera_indexes must have length {n} (parallel to the views), got {}",
                        a.shape()[0]
                    )));
                }
                a.to_vec()
            }
            None => vec![0u32; n],
        };

        let mut resolved = Vec::with_capacity(n);
        let mut quats = Vec::with_capacity(n);
        let mut trans = Vec::with_capacity(n);
        for i in 0..n {
            let ci = cam_idx[i] as usize;
            if ci >= cams.len() {
                return Err(PyValueError::new_err(format!(
                    "camera_indexes[{i}] = {ci} is out of range for {} cameras",
                    cams.len()
                )));
            }
            resolved.push(cams[ci].clone());
            let (qw, qx, qy, qz) = (q[[i, 0]], q[[i, 1]], q[[i, 2]], q[[i, 3]]);
            let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
            if (norm - 1.0).abs() > 1e-6 {
                return Err(PyValueError::new_err(format!(
                    "quaternions_wxyz[{i}] has norm {norm:.6}, expected a unit quaternion"
                )));
            }
            quats.push(UnitQuaternion::from_quaternion(Quaternion::new(
                qw, qx, qy, qz,
            )));
            trans.push(Vector3::new(t[[i, 0]], t[[i, 1]], t[[i, 2]]));
        }
        Ok(Self {
            cameras: resolved,
            quaternions: quats,
            translations: trans,
        })
    }

    /// The number of views ``N``.
    fn __len__(&self) -> usize {
        self.cameras.len()
    }
}

/// Per-image source pyramids prebuilt **once** and shared across PatchCloud
/// kernel calls.
///
/// Every :class:`PatchCloud` kernel method
/// (:meth:`PatchCloud.select_views`, :meth:`PatchCloud.localize_keypoints`,
/// :meth:`PatchCloud.refine_keypoints`), and
/// ``sfmtool.matching.refine_cluster_patches``, accepts one of these anywhere
/// it accepts the list of numpy images: rather than rebuilding a full image
/// pyramid per image on **every** call (the per-call ``images`` list path,
/// kept for back-compat), build the set once and pass it to each call. The
/// pyramids are identical to the per-call build (same levels, same box-filter
/// downsample), so results are unchanged; the build itself is rayon-parallel.
///
/// The constructor holds every tile of each image's pyramid. :meth:`read`
/// instead reads the photographs from their files through a
/// ``PhotographCache`` and holds none of their pixels: the cache holds the
/// tiles under its budget, as it does for the viewer.
///
/// Args:
///     views_or_recon: The scene the images belong to — an
///         :class:`SfmrReconstruction` or a :class:`CameraViews`. Each image is
///         validated against its camera's resolution at build time; kernel calls
///         re-validate the set against *their* scene (image count and per-image
///         camera dimensions), so a set built here works for any scene with the
///         same images (e.g. across ``embed-patches`` rounds).
///     images: One source image (HxWxC uint8 numpy array) per view, parallel to
///         ``views_or_recon`` (index = image index).
#[pyclass(name = "ImagePyramidSet", module = "sfmtool.patches", frozen)]
pub struct PyImagePyramidSet {
    pub(crate) pyramids: Arc<Vec<TiledPhotograph>>,
}

#[pymethods]
impl PyImagePyramidSet {
    #[new]
    #[pyo3(signature = (views_or_recon, images))]
    fn new(
        py: Python<'_>,
        views_or_recon: &Bound<'_, PyAny>,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (posed, _recon) = resolve_scene(views_or_recon)?;
        let pyramids = build_photographs_from_cameras(py, &posed.cameras, &images)?;
        Ok(Self {
            pyramids: Arc::new(pyramids),
        })
    }

    /// Read a set of photographs from their files through a photograph
    /// cache, as the viewer reads them.
    ///
    /// The set holds no pixel: each photograph reads its tiles through the
    /// cache, which keeps them under its budget and decodes an in-core
    /// photograph again for a tile it has evicted. An out-of-core photograph
    /// (its workspace has ``mipmap_options`` and a side over their
    /// threshold) whose mipmap is missing or stale has it made first, in its
    /// workspace's mipmap directory; its levels are then the stored ones,
    /// which the constructor's exact levels are not. The GIL is released
    /// while it reads.
    ///
    /// Args:
    ///     recon_or_paths: An :class:`SfmrReconstruction`, whose
    ///         ``image_names`` are read under ``workspace_dir``, each checked
    ///         against its camera's size; or a list of photograph paths, read
    ///         in that order with no check of their sizes (a kernel checks
    ///         the set against its scene). A :class:`CameraViews` names no
    ///         photograph and is refused.
    ///     workspace_dir: For a reconstruction, the directory its image names
    ///         are under; ``None`` is the reconstruction's own workspace.
    ///         Refused with a list of paths.
    ///     cache: The ``sfmtool.photographs.PhotographCache`` to read through,
    ///         so that a caller holding one shares its tiles, mipmaps and
    ///         build budget across calls. ``None`` makes a cache of the
    ///         default budgets for this set alone, which lives as long as the
    ///         set.
    ///     progress: A ``ProgressCounter``, set to the number of mipmaps made.
    ///
    /// Returns:
    ///     An :class:`ImagePyramidSet`, one photograph per path, in order.
    ///
    /// Raises:
    ///     TypeError: for a :class:`CameraViews`, or anything else that is
    ///         neither a reconstruction nor a list of paths.
    ///     ValueError: for ``workspace_dir`` with a list of paths, or a
    ///         reconstruction's photograph that is not its camera's size.
    ///     FileNotFoundError: when a photograph does not exist.
    ///     OSError: when one cannot be read, or has a side of 65536 px or more.
    ///     RuntimeError: when a workspace file holds an invalid
    ///         ``mipmap_options``, naming the key.
    ///     KeyboardInterrupt: for Ctrl+C, which stops the read within one
    ///         stage.
    ///
    /// A kernel that reads a tile the cache has evicted, of a photograph
    /// whose file changed since this read, raises ``RuntimeError`` naming
    /// the file.
    #[staticmethod]
    #[pyo3(signature = (recon_or_paths, workspace_dir=None, cache=None, progress=None))]
    fn read(
        py: Python<'_>,
        recon_or_paths: &Bound<'_, PyAny>,
        workspace_dir: Option<PathBuf>,
        cache: Option<PyRef<'_, PyPhotographCache>>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Self> {
        let (paths, cameras) = if recon_or_paths.cast::<PyCameraViews>().is_ok() {
            return Err(PyTypeError::new_err(
                "ImagePyramidSet.read needs an SfmrReconstruction or a list of photograph \
                 paths; a CameraViews names no photographs",
            ));
        } else if let Ok(recon) = recon_or_paths.extract::<PyRef<'_, PySfmrReconstruction>>() {
            let recon = recon.inner();
            let dir = workspace_dir.unwrap_or_else(|| recon.workspace_dir.clone());
            let paths: Vec<PathBuf> = recon
                .image_set
                .images
                .iter()
                .map(|image| dir.join(&image.name))
                .collect();
            (paths, Some(PosedViews::from_reconstruction(recon).cameras))
        } else if let Ok(paths) = recon_or_paths.extract::<Vec<PathBuf>>() {
            if workspace_dir.is_some() {
                return Err(PyValueError::new_err(
                    "workspace_dir is for a reconstruction's image names; \
                     pass a list of whole photograph paths without it",
                ));
            }
            (paths, None)
        } else {
            return Err(PyTypeError::new_err(
                "ImagePyramidSet.read needs an SfmrReconstruction or a list of photograph paths",
            ));
        };
        let own;
        let cache = match &cache {
            Some(cache) => &cache.inner,
            None => {
                own = PhotographCache::new(PhotographCacheOptions::default());
                &own
            }
        };
        let pyramids = get_photographs(py, cache, &paths, progress.as_ref())?;
        if let Some(cameras) = cameras {
            for (i, photograph) in pyramids.iter().enumerate() {
                let (width, height) = photograph.layout().level_size(0);
                check_image_matches_camera(&cameras[i], i, width, height)?;
            }
        }
        Ok(Self {
            pyramids: Arc::new(pyramids),
        })
    }

    /// Number of per-image pyramids in the set.
    fn __len__(&self) -> usize {
        self.pyramids.len()
    }
}
