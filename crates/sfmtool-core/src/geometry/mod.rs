// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Geometric primitives: rotations, rigid transforms, and ray geometry.

pub mod affine_factorization;
pub mod convention;
pub mod rigid_transform;
pub mod rot_quaternion;
pub mod rotation;
pub mod se3_transform;
pub mod transform;
pub mod viewing_angle;

pub use rigid_transform::RigidTransform;
pub use rot_quaternion::RotQuaternion;
pub use se3_transform::Se3Transform;
