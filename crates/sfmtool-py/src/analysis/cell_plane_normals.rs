// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the cell plane normals: a cluster's patch normal from
//! the piecewise refinement's stored cell displacements, once poses exist.

use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::patch::cell_plane_normals::{
    cell_plane_normals as core_cell_plane_normals, CellPlaneCameras, CellPlaneClusters,
    CellPlaneParams, CellPlaneStatus, NormalDeterminacy,
};
use sfmtool_core::RigidTransform;
use sfmtool_matches_format::{ClusterCellStatus, ClusterMemberStatus};

use crate::PyCameraIntrinsics;

/// The names of the `determinacy` codes, in code order.
const DETERMINACY_NAMES: [&str; 3] = ["none", "one_axis", "both_axes"];

/// Estimate each cluster's patch normal from its stored cell displacements.
///
/// Reads the piecewise refinement's per-cell entries of a cluster-patches
/// ``.matches`` file (``MatchesFile.member_cell_shift_px`` and
/// ``member_cell_status``, with the backbone's member arrays) and the images'
/// cameras and poses. For each of a cluster's nine cells, the reference
/// member's ray through the cell's centre and every ``kept`` member's ray
/// through ``position + (patch_size / resolution) · S · (c + d)`` are
/// intersected in the least-squares sense, over the members whose image is
/// posed and whose cell status is ``fitted`` (and ``refused_outlier`` with
/// ``include_refused_outlier``). The cell positions are fitted with a plane by
/// Tukey IRLS, each cell weighted by the inverse of its position's variance
/// along the normal, from its rays and their intersection residual. See
/// ``specs/drafts/cell-plane-normals.md``.
///
/// Args:
///     cluster_starts: ``(C + 1,)`` uint32 member-range boundaries.
///     reference_members: ``(C,)`` uint32, ``0xFFFFFFFF`` for none.
///     member_images: ``(K,)`` uint32 image index of each member, into
///         ``image_camera`` / the pose arrays.
///     member_status: ``(K,)`` uint8 canonical member status codes.
///     member_positions: ``(K, 2)`` float32 positions, pixels.
///     member_affine_shapes: ``(K, 2, 2)`` float32 shapes.
///     member_cell_shift_px: ``(K, 3, 3, 2)`` float32 displacements, grid px.
///     member_cell_status: ``(K, 3, 3)`` uint8 canonical cell status codes.
///     patch_size: ``refine_options["patch_size"]``.
///     resolution: ``refine_options["resolution"]``.
///     cameras: list of ``CameraIntrinsics``.
///     image_camera: ``(N,)`` uint32 camera of each image.
///     quaternions_wxyz: ``(N, 4)`` float64 ``cam_from_world`` rotations; a row
///         with a non-finite entry marks the image unposed.
///     translations: ``(N, 3)`` float64 ``cam_from_world`` translations.
///     include_refused_outlier: also use ``refused_outlier`` cells.
///     min_rays: fewest rays a cell is triangulated from (the reference's
///         counts).
///     min_triangulation_angle_deg: widest ray pair a cell needs.
///     ray_noise_floor_px: floor on a cell's ray-intersection residual.
///     irls_iters: IRLS passes before the final solve.
///     tukey_c: Tukey cut-off, in robust scales.
///     min_cells: fewest live cells a normal is fitted from.
///     det_aniso: in-plane anisotropy at which both axes count as fixed.
///
/// Returns:
///     dict of numpy arrays over the ``C`` clusters: ``normal`` ``(C, 3)``
///     float64 (``NaN`` where no axis is fixed), ``determinacy`` ``(C,)``
///     uint8 (codes into ``determinacy_names``: none, one_axis, both_axes),
///     ``free_axis`` ``(C, 3)`` float64 (``NaN`` unless one axis),
///     ``view_dir`` ``(C, 3)``, ``cell_positions`` ``(C, 3, 3, 3)`` float64
///     (``NaN`` where not triangulated), ``cell_rays`` ``(C, 3, 3)`` uint32,
///     ``cell_status`` ``(C, 3, 3)`` uint8 (codes into ``cell_status_names``),
///     ``cell_weight`` and ``cell_residual_px`` ``(C, 3, 3)`` float64, and
///     ``plane_rms``, ``anisotropy``, ``n_eff`` ``(C,)`` float64; plus the two
///     name lists.
#[pyfunction]
#[pyo3(signature = (
    cluster_starts,
    reference_members,
    member_images,
    member_status,
    member_positions,
    member_affine_shapes,
    member_cell_shift_px,
    member_cell_status,
    patch_size,
    resolution,
    cameras,
    image_camera,
    quaternions_wxyz,
    translations,
    *,
    include_refused_outlier=false,
    min_rays=2,
    min_triangulation_angle_deg=2.0,
    ray_noise_floor_px=0.05,
    irls_iters=3,
    tukey_c=4.685,
    min_cells=3,
    det_aniso=0.10,
))]
#[allow(clippy::too_many_arguments)]
pub fn cell_plane_normals<'py>(
    py: Python<'py>,
    cluster_starts: PyReadonlyArray1<'py, u32>,
    reference_members: PyReadonlyArray1<'py, u32>,
    member_images: PyReadonlyArray1<'py, u32>,
    member_status: PyReadonlyArray1<'py, u8>,
    member_positions: PyReadonlyArray2<'py, f32>,
    member_affine_shapes: PyReadonlyArray3<'py, f32>,
    member_cell_shift_px: PyReadonlyArray4<'py, f32>,
    member_cell_status: PyReadonlyArray3<'py, u8>,
    patch_size: f64,
    resolution: u32,
    cameras: Vec<PyRef<'_, PyCameraIntrinsics>>,
    image_camera: PyReadonlyArray1<'py, u32>,
    quaternions_wxyz: PyReadonlyArray2<'py, f64>,
    translations: PyReadonlyArray2<'py, f64>,
    include_refused_outlier: bool,
    min_rays: usize,
    min_triangulation_angle_deg: f64,
    ray_noise_floor_px: f64,
    irls_iters: u32,
    tukey_c: f64,
    min_cells: usize,
    det_aniso: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let k = member_images.shape()[0];
    let check = |ok: bool, msg: &str| -> PyResult<()> {
        if ok {
            Ok(())
        } else {
            Err(PyValueError::new_err(msg.to_string()))
        }
    };
    check(
        member_status.shape()[0] == k,
        "member_status must have shape (K,)",
    )?;
    check(
        member_positions.shape() == [k, 2],
        "member_positions must have shape (K, 2)",
    )?;
    check(
        member_affine_shapes.shape() == [k, 2, 2],
        "member_affine_shapes must have shape (K, 2, 2)",
    )?;
    check(
        member_cell_shift_px.shape() == [k, 3, 3, 2],
        "member_cell_shift_px must have shape (K, 3, 3, 2)",
    )?;
    check(
        member_cell_status.shape() == [k, 3, 3],
        "member_cell_status must have shape (K, 3, 3)",
    )?;
    let c = reference_members.shape()[0];
    check(
        cluster_starts.shape()[0] == c + 1,
        "cluster_starts must have shape (C + 1,)",
    )?;
    check(resolution >= 3, "resolution must be at least 3")?;
    check(
        patch_size.is_finite() && patch_size > 0.0,
        "patch_size must be positive and finite",
    )?;
    let n_img = image_camera.shape()[0];
    check(
        quaternions_wxyz.shape() == [n_img, 4],
        "quaternions_wxyz must have shape (N, 4)",
    )?;
    check(
        translations.shape() == [n_img, 3],
        "translations must have shape (N, 3)",
    )?;
    let cams: Vec<CameraIntrinsics> = cameras.iter().map(|c| c.inner.clone()).collect();

    let starts = to_contiguous!(cluster_starts).into_owned();
    check(
        starts.windows(2).all(|w| w[0] <= w[1]) && starts.last().is_none_or(|&s| s as usize <= k),
        "cluster_starts must be non-decreasing and end within the member count",
    )?;
    let references = to_contiguous!(reference_members).into_owned();
    for (ci, &r) in references.iter().enumerate() {
        if r != u32::MAX && !(starts[ci] <= r && r < starts[ci + 1]) {
            return Err(PyValueError::new_err(format!(
                "reference_members[{ci}] = {r} is not in cluster {ci}'s member range"
            )));
        }
    }
    let images = to_contiguous!(member_images).into_owned();
    if let Some(&bad) = images.iter().find(|&&i| i as usize >= n_img) {
        return Err(PyValueError::new_err(format!(
            "member_images contains {bad}, out of range for {n_img} images"
        )));
    }
    let image_camera = to_contiguous!(image_camera).into_owned();
    if let Some(&bad) = image_camera.iter().find(|&&j| j as usize >= cams.len()) {
        return Err(PyValueError::new_err(format!(
            "image_camera contains {bad}, out of range for {} cameras",
            cams.len()
        )));
    }
    let status: Vec<ClusterMemberStatus> = to_contiguous!(member_status)
        .iter()
        .map(|&s| {
            ClusterMemberStatus::from_u8(s).ok_or_else(|| {
                PyValueError::new_err(format!("member_status code {s} is not a member status"))
            })
        })
        .collect::<PyResult<_>>()?;
    let positions: Vec<[f32; 2]> = to_contiguous!(member_positions).as_chunks::<2>().0.to_vec();
    let shapes: Vec<[[f32; 2]; 2]> = to_contiguous!(member_affine_shapes)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|s| [[s[0], s[1]], [s[2], s[3]]])
        .collect();
    let shifts: Vec<[[f32; 2]; 9]> = to_contiguous!(member_cell_shift_px)
        .as_chunks::<18>()
        .0
        .iter()
        .map(|s| std::array::from_fn(|j| [s[2 * j], s[2 * j + 1]]))
        .collect();
    let cells: Vec<[ClusterCellStatus; 9]> = to_contiguous!(member_cell_status)
        .as_chunks::<9>()
        .0
        .iter()
        .map(|row| {
            let mut out = [ClusterCellStatus::NotAttempted; 9];
            for (o, &s) in out.iter_mut().zip(row) {
                *o = ClusterCellStatus::from_u8(s).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "member_cell_status code {s} is not a cell status"
                    ))
                })?;
            }
            Ok(out)
        })
        .collect::<PyResult<_>>()?;
    let q = to_contiguous!(quaternions_wxyz);
    let t = to_contiguous!(translations);
    let poses: Vec<Option<RigidTransform>> = (0..n_img)
        .map(|i| {
            let qi = [q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]];
            let ti = [t[3 * i], t[3 * i + 1], t[3 * i + 2]];
            (qi.iter().chain(&ti).all(|v| v.is_finite()))
                .then(|| RigidTransform::from_wxyz_translation(qi, ti))
        })
        .collect();

    let params = CellPlaneParams {
        include_refused_outlier,
        min_rays,
        min_triangulation_angle_deg,
        ray_noise_floor_px,
        irls_iters,
        tukey_c,
        min_cells,
        det_aniso,
    };
    let out = py.detach(|| {
        core_cell_plane_normals(
            &CellPlaneClusters {
                cluster_starts: &starts,
                reference_members: &references,
                member_images: &images,
                member_status: &status,
                member_positions: &positions,
                member_shapes: &shapes,
                cell_shift_px: &shifts,
                cell_status: &cells,
                patch_size,
                resolution,
            },
            &CellPlaneCameras {
                cameras: &cams,
                image_camera: &image_camera,
                cam_from_world: &poses,
            },
            &params,
        )
    });

    let flat3 = |f: &dyn Fn(usize) -> [f64; 3]| -> Vec<f64> { (0..c).flat_map(f).collect() };
    let normal = flat3(&|i| out[i].normal);
    let view_dir = flat3(&|i| out[i].view_dir);
    let free_axis = flat3(&|i| match out[i].determinacy {
        NormalDeterminacy::OneAxis { free_axis } => free_axis,
        _ => [f64::NAN; 3],
    });
    let determinacy: Vec<u8> = out.iter().map(|r| r.determinacy.code()).collect();
    let cell_positions: Vec<f64> = out
        .iter()
        .flat_map(|r| r.cell_positions.into_iter().flatten())
        .collect();
    let cell_rays: Vec<u32> = out.iter().flat_map(|r| r.cell_rays).collect();
    let cell_status: Vec<u8> = out
        .iter()
        .flat_map(|r| r.cell_status.map(|s| s as u8))
        .collect();
    let cell_weight: Vec<f64> = out.iter().flat_map(|r| r.cell_weight).collect();
    let cell_residual: Vec<f64> = out.iter().flat_map(|r| r.cell_residual_px).collect();

    let d = PyDict::new(py);
    d.set_item("normal", PyArray1::from_vec(py, normal).reshape([c, 3])?)?;
    d.set_item("determinacy", PyArray1::from_vec(py, determinacy))?;
    d.set_item("determinacy_names", DETERMINACY_NAMES.to_vec())?;
    d.set_item(
        "free_axis",
        PyArray1::from_vec(py, free_axis).reshape([c, 3])?,
    )?;
    d.set_item(
        "view_dir",
        PyArray1::from_vec(py, view_dir).reshape([c, 3])?,
    )?;
    d.set_item(
        "cell_positions",
        PyArray1::from_vec(py, cell_positions).reshape([c, 3, 3, 3])?,
    )?;
    d.set_item(
        "cell_rays",
        PyArray1::from_vec(py, cell_rays).reshape([c, 3, 3])?,
    )?;
    d.set_item(
        "cell_status",
        PyArray1::from_vec(py, cell_status).reshape([c, 3, 3])?,
    )?;
    d.set_item("cell_status_names", CellPlaneStatus::NAMES.to_vec())?;
    d.set_item(
        "cell_weight",
        PyArray1::from_vec(py, cell_weight).reshape([c, 3, 3])?,
    )?;
    d.set_item(
        "cell_residual_px",
        PyArray1::from_vec(py, cell_residual).reshape([c, 3, 3])?,
    )?;
    d.set_item(
        "plane_rms",
        PyArray1::from_vec(py, out.iter().map(|r| r.plane_rms).collect()),
    )?;
    d.set_item(
        "anisotropy",
        PyArray1::from_vec(py, out.iter().map(|r| r.anisotropy).collect()),
    )?;
    d.set_item(
        "n_eff",
        PyArray1::from_vec(py, out.iter().map(|r| r.n_eff).collect()),
    )?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cell_plane_normals, m)?)?;
    Ok(())
}
