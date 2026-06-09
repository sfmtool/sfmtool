// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sfmtool core functionality.
//!
//! Exposes file I/O (`.sfmr`, `.sift`, and COLMAP formats), geometric types,
//! feature matching, alignment, optical flow, and GUI viewer to Python via PyO3.
//!
//! # Example
//!
//! ```python
//! from sfmtool._sfmtool import read_sfmr, write_sfmr, verify_sfmr
//! from sfmtool._sfmtool import read_sift, write_sift, verify_sift
//!
//! data = read_sfmr("reconstruction.sfmr")
//! valid, errors = verify_sfmr("reconstruction.sfmr")
//! ```

use pyo3::prelude::*;

/// Try zero-copy `as_slice()` for contiguous arrays, fall back to copying for non-contiguous.
macro_rules! to_contiguous {
    ($arr:expr) => {
        match $arr.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned($arr.as_array().iter().copied().collect::<Vec<_>>()),
        }
    };
}

// ── Shared helpers ────────────────────────────────────────────────────────

pub(crate) mod helpers;

// ── Geometric types ───────────────────────────────────────────────────────

mod py_rot_quaternion;
pub use py_rot_quaternion::PyRotQuaternion;

mod py_camera_intrinsics;
pub use py_camera_intrinsics::PyCameraIntrinsics;

mod py_rigid_transform;
pub use py_rigid_transform::PyRigidTransform;

mod py_se3_transform;
pub use py_se3_transform::PySe3Transform;

mod py_sfmr_reconstruction;
pub use py_sfmr_reconstruction::PySfmrReconstruction;
mod recon_clone;

mod py_range_expr;
pub use py_range_expr::PyRangeExpr;

// ── File I/O ──────────────────────────────────────────────────────────────

mod py_camrig_io;
mod py_colmap_binary;
mod py_colmap_db;
mod py_matches_io;
mod py_sfmr_io;
mod py_sift_io;

// ── Image inspection ──────────────────────────────────────────────────────

mod py_image;

// ── Feature matching ──────────────────────────────────────────────────────

mod py_descriptor_match;
mod py_image_match;
mod py_sift;
mod py_sweep_match;

// ── Image warping ────────────────────────────────────────────────────────

mod py_warp_map;
pub use py_warp_map::PyWarpMap;

// ── Analysis & algorithms ─────────────────────────────────────────────────

mod py_analysis;
mod py_consensus_atlas;
mod py_epipolar;
mod py_image_pair_graph;
mod py_kdforest;
mod py_kdtree;
mod py_optical_flow;
mod py_per_spherical_tile_source_stack;
mod py_photometric_ransac;
mod py_sphere_points;
mod py_spherical_tile_rig;
mod py_triangulation;
pub use py_per_spherical_tile_source_stack::PyPerSphericalTileSourceStack;
pub use py_photometric_ransac::PyRansacPhotometricOutput;
pub use py_spherical_tile_rig::PySphericalTileRig;

// ── Module registration ───────────────────────────────────────────────────

/// Python module for sfmtool core functionality.
#[pymodule]
fn _sfmtool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // .sfmr file I/O
    m.add_function(wrap_pyfunction!(py_sfmr_io::read_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(py_sfmr_io::read_sfmr_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_sfmr_io::read_sfmr_content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_sfmr_io::write_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(py_sfmr_io::verify_sfmr, m)?)?;

    // COLMAP binary I/O
    m.add_function(wrap_pyfunction!(py_colmap_binary::read_colmap_binary, m)?)?;
    m.add_function(wrap_pyfunction!(py_colmap_binary::write_colmap_binary, m)?)?;

    // COLMAP SQLite database I/O
    m.add_function(wrap_pyfunction!(py_colmap_db::write_colmap_db, m)?)?;
    m.add_function(wrap_pyfunction!(py_colmap_db::read_colmap_db_matches, m)?)?;

    // .matches file I/O
    m.add_function(wrap_pyfunction!(py_matches_io::read_matches, m)?)?;
    m.add_function(wrap_pyfunction!(py_matches_io::read_matches_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_matches_io::write_matches, m)?)?;
    m.add_function(wrap_pyfunction!(py_matches_io::verify_matches, m)?)?;

    // .camrig file I/O
    m.add_function(wrap_pyfunction!(py_camrig_io::read_camrig_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_camrig_io::read_camrig, m)?)?;
    m.add_function(wrap_pyfunction!(py_camrig_io::verify_camrig, m)?)?;
    m.add_function(wrap_pyfunction!(py_camrig_io::write_camrig, m)?)?;

    // .camrig image patterns
    m.add_function(wrap_pyfunction!(py_camrig_io::validate_camrig_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(py_camrig_io::camrig_pattern_to_glob, m)?)?;
    m.add_function(wrap_pyfunction!(py_camrig_io::camrig_pattern_matches, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_camrig_io::camrig_pattern_frame_index,
        m
    )?)?;

    // .sift file I/O
    m.add_function(wrap_pyfunction!(py_sift_io::read_sift, m)?)?;
    m.add_function(wrap_pyfunction!(py_sift_io::read_sift_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_sift_io::read_sift_partial, m)?)?;
    m.add_function(wrap_pyfunction!(py_sift_io::write_sift, m)?)?;
    m.add_class::<py_sift_io::SiftWriteQueue>()?;
    m.add_function(wrap_pyfunction!(py_sift_io::verify_sift, m)?)?;

    // sfmtool SIFT detection / extraction
    m.add_function(wrap_pyfunction!(py_sift::detect_sift_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(py_sift::extract_sift, m)?)?;

    // Image inspection
    m.add_function(wrap_pyfunction!(py_image::image_dimensions, m)?)?;

    // Feature matching
    m.add_function(wrap_pyfunction!(
        py_descriptor_match::descriptor_distance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_descriptor_match::find_best_descriptor_match,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_sweep_match::match_one_way_sweep_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_sweep_match::match_one_way_sweep_geometric_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_sweep_match::mutual_best_match_sweep_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_sweep_match::polar_mutual_best_match_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_sweep_match::mutual_best_match_sweep_geometric_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_sweep_match::polar_mutual_best_match_geometric_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_image_match::match_image_pair_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_image_match::match_image_pairs_batch_py,
        m
    )?)?;

    // SE3 transform acceleration
    m.add_function(wrap_pyfunction!(
        py_analysis::apply_se3_to_camera_poses_py,
        m
    )?)?;

    // Viewing angle analysis
    m.add_function(wrap_pyfunction!(py_analysis::compute_narrow_track_mask, m)?)?;

    // Batch triangulation
    m.add_function(wrap_pyfunction!(py_triangulation::triangulate_batch, m)?)?;

    // Alignment (Kabsch + RANSAC)
    m.add_function(wrap_pyfunction!(py_analysis::kabsch_algorithm_rs, m)?)?;
    m.add_function(wrap_pyfunction!(py_analysis::ransac_alignment_rs, m)?)?;

    // Point correspondence
    m.add_function(wrap_pyfunction!(
        py_analysis::find_point_correspondences_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_analysis::merge_points_and_tracks_py,
        m
    )?)?;

    // Track filtering
    m.add_function(wrap_pyfunction!(
        py_analysis::filter_tracks_by_point_mask_py,
        m
    )?)?;

    // Image pair graph
    m.add_function(wrap_pyfunction!(
        py_image_pair_graph::build_covisibility_pairs_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_image_pair_graph::build_frustum_intersection_pairs_py,
        m
    )?)?;

    // Optical flow
    m.add_function(wrap_pyfunction!(py_optical_flow::gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(py_optical_flow::compute_optical_flow, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_optical_flow::compute_optical_flow_timed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_optical_flow::compute_optical_flow_with_init,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_optical_flow::compose_flow, m)?)?;
    m.add_function(wrap_pyfunction!(py_optical_flow::advect_points, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_optical_flow::match_candidates_by_descriptor,
        m
    )?)?;

    // Sphere point generation
    m.add_function(wrap_pyfunction!(
        py_sphere_points::evenly_distributed_sphere_points,
        m
    )?)?;

    // Epipolar curves (distortion-aware epipolar lines)
    m.add_function(wrap_pyfunction!(py_epipolar::epipolar_curves_py, m)?)?;

    // Types
    m.add_class::<PyCameraIntrinsics>()?;
    m.add_class::<PyRigidTransform>()?;
    m.add_class::<PyRotQuaternion>()?;
    m.add_class::<PySe3Transform>()?;
    m.add_class::<PySfmrReconstruction>()?;
    m.add_class::<PyRangeExpr>()?;
    m.add_class::<py_kdtree::PyKdTree2d>()?;
    m.add_class::<py_kdtree::PyKdTree3d>()?;
    m.add_class::<py_kdforest::PyKdForest>()?;
    m.add_class::<PyPerSphericalTileSourceStack>()?;
    m.add_class::<PyRansacPhotometricOutput>()?;
    m.add_class::<PySphericalTileRig>()?;
    m.add_class::<PyWarpMap>()?;

    // Photometric refinement.
    m.add_function(wrap_pyfunction!(
        py_photometric_ransac::refine_photometric_ransac_py,
        m
    )?)?;

    // Tile-batched consensus atlas compositing.
    m.add_function(wrap_pyfunction!(
        py_consensus_atlas::render_consensus_atlas_py,
        m
    )?)?;

    Ok(())
}
