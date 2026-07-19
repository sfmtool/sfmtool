// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the core geometric value types: camera intrinsics, rotation
//! quaternions, rigid transforms, and SE3 transforms — plus the
//! coordinate-convention conversion functions and the affine factorization.

use pyo3::prelude::*;

pub mod absolute_pose;
pub mod affine_factorization;
pub mod bundle_adjust;
pub mod camera_intrinsics;
pub mod convention;
pub mod epipolar_estimation;
pub mod focal_vote;
pub mod homography_estimation;
pub mod pose_refine;
pub mod reprojection;
pub mod resect_translation;
pub mod rigid_transform;
pub mod rot_quaternion;
pub mod rotation_init;
pub mod se3_transform;

pub use camera_intrinsics::PyCameraIntrinsics;
pub use rigid_transform::PyRigidTransform;
pub use rot_quaternion::PyRotQuaternion;
pub use se3_transform::PySe3Transform;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCameraIntrinsics>()?;
    m.add_class::<PyRigidTransform>()?;
    m.add_class::<PyRotQuaternion>()?;
    m.add_class::<PySe3Transform>()?;
    convention::register(m)?;
    affine_factorization::register(m)?;
    absolute_pose::register(m)?;
    epipolar_estimation::register(m)?;
    homography_estimation::register(m)?;
    focal_vote::register(m)?;
    reprojection::register(m)?;
    pose_refine::register(m)?;
    resect_translation::register(m)?;
    rotation_init::register(m)?;
    bundle_adjust::register(m)?;
    Ok(())
}
