// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for high-level image pair matching.

use nalgebra::{Matrix3, Vector3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::feature_match;

/// Unified image pair matching: automatically selects rectified sweep or polar matching.
///
/// Computes fundamental matrix, checks rectification safety, and dispatches
/// to the appropriate matching algorithm. Optionally applies geometric filtering.
/// Returns list of (idx1, idx2, distance) tuples.
#[pyfunction]
#[pyo3(signature = (k1, k2, r1, r2, t1, t2, width1, height1, width2, height2,
                     positions1, descriptors1, positions2, descriptors2,
                     window_size, threshold=None, rectification_margin=50,
                     affines1=None, affines2=None,
                     max_angle_difference=None, min_triangulation_angle=None,
                     geometric_size_ratio_min=None, geometric_size_ratio_max=None))]
#[allow(clippy::too_many_arguments)]
pub fn match_image_pair_py(
    k1: PyReadonlyArray2<f64>,
    k2: PyReadonlyArray2<f64>,
    r1: PyReadonlyArray2<f64>,
    r2: PyReadonlyArray2<f64>,
    t1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    positions1: PyReadonlyArray2<f64>,
    descriptors1: PyReadonlyArray2<u8>,
    positions2: PyReadonlyArray2<f64>,
    descriptors2: PyReadonlyArray2<u8>,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    affines1: Option<PyReadonlyArray2<f64>>,
    affines2: Option<PyReadonlyArray2<f64>>,
    max_angle_difference: Option<f64>,
    min_triangulation_angle: Option<f64>,
    geometric_size_ratio_min: Option<f64>,
    geometric_size_ratio_max: Option<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let k1_data = to_contiguous!(k1);
    let k2_data = to_contiguous!(k2);
    let r1_data = to_contiguous!(r1);
    let r2_data = to_contiguous!(r2);
    let t1_data = to_contiguous!(t1);
    let t2_data = to_contiguous!(t2);

    let k1_mat = Matrix3::from_row_slice(&k1_data);
    let k2_mat = Matrix3::from_row_slice(&k2_data);
    let r1_mat = Matrix3::from_row_slice(&r1_data);
    let r2_mat = Matrix3::from_row_slice(&r2_data);
    let t1_vec = Vector3::from_row_slice(&t1_data);
    let t2_vec = Vector3::from_row_slice(&t2_data);

    let p1_data = to_contiguous!(positions1);
    let d1_data = to_contiguous!(descriptors1);
    let p2_data = to_contiguous!(positions2);
    let d2_data = to_contiguous!(descriptors2);

    let n1 = positions1.shape()[0];
    let n2 = positions2.shape()[0];
    let desc_len = descriptors1.shape()[1];

    // Extract optional affine data
    let a1_cow = affines1.as_ref().map(|a| to_contiguous!(a));
    let a1_ref = a1_cow.as_deref();
    let a2_cow = affines2.as_ref().map(|a| to_contiguous!(a));
    let a2_ref = a2_cow.as_deref();

    // Build geometric config if all params are provided
    let geo_config = match (
        max_angle_difference,
        min_triangulation_angle,
        geometric_size_ratio_min,
        geometric_size_ratio_max,
    ) {
        (Some(mad), Some(mta), Some(srmin), Some(srmax)) => {
            Some(feature_match::GeometricFilterConfig {
                max_angle_difference: mad,
                min_triangulation_angle: mta,
                geometric_size_ratio_min: srmin,
                geometric_size_ratio_max: srmax,
            })
        }
        _ => None,
    };

    Ok(feature_match::match_image_pair(
        &k1_mat,
        &k2_mat,
        &r1_mat,
        &t1_vec,
        &r2_mat,
        &t2_vec,
        width1,
        height1,
        width2,
        height2,
        &p1_data,
        &d1_data,
        n1,
        &p2_data,
        &d2_data,
        n2,
        desc_len,
        window_size,
        threshold,
        rectification_margin,
        a1_ref,
        a2_ref,
        geo_config.as_ref(),
    ))
}

/// Match features across multiple image pairs in parallel using Rayon.
///
/// This is the batch version of `match_image_pair_py` that processes all pairs
/// concurrently, releasing the GIL so Rayon threads can run in parallel.
#[pyfunction]
#[pyo3(signature = (pairs, intrinsics, rotations, translations, camera_indices,
                     positions_list, descriptors_list,
                     widths, heights, window_size, threshold, rectification_margin,
                     affines_list=None,
                     max_angle_difference=None, min_triangulation_angle=None,
                     geometric_size_ratio_min=None, geometric_size_ratio_max=None))]
#[allow(clippy::too_many_arguments)]
pub fn match_image_pairs_batch_py(
    py: Python<'_>,
    pairs: Vec<(usize, usize)>,
    intrinsics: Vec<PyReadonlyArray2<f64>>,
    rotations: Vec<PyReadonlyArray2<f64>>,
    translations: Vec<PyReadonlyArray1<f64>>,
    camera_indices: PyReadonlyArray1<i64>,
    positions_list: Vec<PyReadonlyArray2<f64>>,
    descriptors_list: Vec<PyReadonlyArray2<u8>>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    window_size: usize,
    threshold: Option<f64>,
    rectification_margin: u32,
    affines_list: Option<Vec<PyReadonlyArray2<f64>>>,
    max_angle_difference: Option<f64>,
    min_triangulation_angle: Option<f64>,
    geometric_size_ratio_min: Option<f64>,
    geometric_size_ratio_max: Option<f64>,
) -> PyResult<Vec<Vec<(usize, usize, f64)>>> {
    // Convert numpy arrays to nalgebra types
    let k_mats: Vec<Matrix3<f64>> = intrinsics
        .iter()
        .map(|k| {
            let s = to_contiguous!(k);
            Matrix3::from_row_slice(&s)
        })
        .collect();

    let r_mats: Vec<Matrix3<f64>> = rotations
        .iter()
        .map(|r| {
            let s = to_contiguous!(r);
            Matrix3::from_row_slice(&s)
        })
        .collect();

    let t_vecs: Vec<Vector3<f64>> = translations
        .iter()
        .map(|t| {
            let s = to_contiguous!(t);
            Vector3::from_row_slice(&s)
        })
        .collect();

    let cam_idx_data = to_contiguous!(camera_indices);
    let cam_idx: Vec<usize> = cam_idx_data.iter().map(|&x| x as usize).collect();

    // Get contiguous data for positions and descriptors
    let pos_cows: Vec<Cow<[f64]>> = positions_list.iter().map(|p| to_contiguous!(p)).collect();
    let desc_cows: Vec<Cow<[u8]>> = descriptors_list.iter().map(|d| to_contiguous!(d)).collect();
    let pos_slices: Vec<&[f64]> = pos_cows.iter().map(|c| c.as_ref()).collect();
    let desc_slices: Vec<&[u8]> = desc_cows.iter().map(|c| c.as_ref()).collect();
    let num_feats: Vec<usize> = positions_list.iter().map(|p| p.shape()[0]).collect();
    let desc_len = if descriptors_list.is_empty() {
        128
    } else {
        descriptors_list[0].shape()[1]
    };

    // Handle optional affines
    let aff_cows: Option<Vec<Cow<[f64]>>> = affines_list
        .as_ref()
        .map(|list| list.iter().map(|a| to_contiguous!(a)).collect());
    let aff_slices: Option<Vec<&[f64]>> = aff_cows
        .as_ref()
        .map(|cows| cows.iter().map(|c| c.as_ref()).collect());
    let aff_ref: Option<&[&[f64]]> = aff_slices.as_deref();

    // Build geometric config if all params are provided
    let geo_config = match (
        max_angle_difference,
        min_triangulation_angle,
        geometric_size_ratio_min,
        geometric_size_ratio_max,
    ) {
        (Some(mad), Some(mta), Some(srmin), Some(srmax)) => {
            Some(feature_match::GeometricFilterConfig {
                max_angle_difference: mad,
                min_triangulation_angle: mta,
                geometric_size_ratio_min: srmin,
                geometric_size_ratio_max: srmax,
            })
        }
        _ => None,
    };

    // Release the GIL so Rayon threads can run in parallel
    py.detach(|| {
        Ok(feature_match::match_image_pairs_batch(
            &pairs,
            &k_mats,
            &r_mats,
            &t_vecs,
            &cam_idx,
            &pos_slices,
            &desc_slices,
            &num_feats,
            &widths,
            &heights,
            desc_len,
            window_size,
            threshold,
            rectification_margin,
            aff_ref,
            geo_config.as_ref(),
        ))
    })
}
