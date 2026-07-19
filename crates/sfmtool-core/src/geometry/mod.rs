// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Geometric primitives: rotations, rigid transforms, and ray geometry.

pub mod absolute_pose;
pub mod affine_factorization;
pub mod batch_resection;
pub mod bundle_adjust;
pub mod convention;
pub mod epipolar_estimation;
pub mod focal_vote;
pub mod homography_estimation;
pub(crate) mod polynomial;
pub mod pose_refine;
pub mod reconstruction_growth;
pub mod reprojection;
pub mod resect_translation;
pub mod rigid_transform;
pub mod rot_quaternion;
pub mod rotation;
pub mod rotation_init;
pub mod se3_transform;
pub mod transform;
pub mod viewing_angle;

pub use batch_resection::{resect_images_batch, BatchResection, ResectOptions};
pub use bundle_adjust::{bundle_adjust, BaSchedule, BundleAdjustment, DEFAULT_SCHEDULE};
pub use pose_refine::{refine_absolute_pose, PoseRefinement};
pub use reconstruction_growth::{grow_reconstruction, GrowOptions, ReconstructionGrowth};
pub use reprojection::{inlier_fraction, reprojection_residuals};
pub use resect_translation::{resect_translation, TranslationResection};
pub use rigid_transform::RigidTransform;
pub use rot_quaternion::RotQuaternion;
pub use rotation_init::{rotation_init, RotationInit};
pub use se3_transform::Se3Transform;
