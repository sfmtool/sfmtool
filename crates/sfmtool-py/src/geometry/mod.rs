// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the core geometric value types: camera intrinsics, rotation
//! quaternions, rigid transforms, and SE3 transforms.

use pyo3::prelude::*;

pub mod camera_intrinsics;
pub mod rigid_transform;
pub mod rot_quaternion;
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
    Ok(())
}
