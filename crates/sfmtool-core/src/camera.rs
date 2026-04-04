// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera representation for 3D viewing.
//!
//! Provides a unified camera model with quaternion-based orientation that can be
//! used for both viewport navigation and SfM camera representation.

use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, UnitQuaternion, Vector3};

/// A camera in 3D space with position, orientation, and target distance.
///
/// The orientation is stored as a quaternion representing the rotation from
/// world coordinates to camera coordinates. The camera looks down its local -Z
/// axis (OpenGL convention).
///
/// The `target_distance` defines a point in front of the camera that serves as
/// the focus point for orbiting and other navigation operations.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera position in world coordinates.
    pub position: Point3<f64>,
    /// Rotation from world to camera coordinates.
    pub orientation: UnitQuaternion<f64>,
    /// Distance to the target point along the forward (-Z) axis.
    pub target_distance: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    /// Creates a new camera at the origin, looking down -Y with Z up.
    pub fn new() -> Self {
        // Default orientation: looking down -Y axis, Z is up
        // This means camera's local -Z maps to world -Y
        let orientation =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        Self {
            position: Point3::origin(),
            orientation,
            target_distance: 1.0,
        }
    }

    /// Creates a camera at the given position, looking at a target point.
    pub fn look_at(position: Point3<f64>, target: Point3<f64>, up: Vector3<f64>) -> Self {
        let target_distance = (target - position).norm().max(0.1);
        let mut camera = Self {
            position,
            orientation: UnitQuaternion::identity(),
            target_distance,
        };
        camera.set_look_at(target, up);
        camera
    }

    /// Returns the target point (the point the camera is looking at).
    ///
    /// This is the point `target_distance` units in front of the camera.
    pub fn target(&self) -> Point3<f64> {
        self.position + self.forward() * self.target_distance
    }

    /// Computes the camera orientation quaternion for a given forward direction.
    ///
    /// Returns the rotation from world to camera coordinates such that the
    /// camera's local -Z axis aligns with `forward` and local Y is as close
    /// to `up` as possible.
    pub fn orientation_from_forward(
        forward: Vector3<f64>,
        up: Vector3<f64>,
    ) -> UnitQuaternion<f64> {
        // Handle degenerate case where forward is parallel to up
        let right = if forward.cross(&up).norm() < 1e-10 {
            // Pick an arbitrary perpendicular vector
            if forward.x.abs() < 0.9 {
                forward.cross(&Vector3::x()).normalize()
            } else {
                forward.cross(&Vector3::y()).normalize()
            }
        } else {
            forward.cross(&up).normalize()
        };

        let corrected_up = right.cross(&forward).normalize();

        // Build rotation matrix from world to camera
        // Camera convention: -Z is forward, Y is up, X is right
        // The ROWS of the rotation matrix should be the camera axes in world coordinates
        // so that R * v gives (v · cam_X, v · cam_Y, v · cam_Z) = camera coords of v
        let rotation_matrix = Matrix3::new(
            right.x,
            right.y,
            right.z,
            corrected_up.x,
            corrected_up.y,
            corrected_up.z,
            -forward.x,
            -forward.y,
            -forward.z,
        );
        let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

        UnitQuaternion::from_rotation_matrix(&rotation)
    }

    /// Sets the camera orientation to look at a target point.
    ///
    /// This also updates the `target_distance` to match the distance to the target.
    ///
    /// The camera will be oriented so that:
    /// - The local -Z axis points toward the target
    /// - The local Y axis is as close to the `up` vector as possible
    pub fn set_look_at(&mut self, target: Point3<f64>, up: Vector3<f64>) {
        self.target_distance = (target - self.position).norm().max(0.1);
        let forward = (target - self.position).normalize();
        self.orientation = Self::orientation_from_forward(forward, up);
    }

    /// Returns the camera's forward direction (the direction it's looking) in world coordinates.
    ///
    /// This is the local -Z axis transformed to world space.
    pub fn forward(&self) -> Vector3<f64> {
        self.orientation.inverse() * Vector3::new(0.0, 0.0, -1.0)
    }

    /// Returns the camera's right direction in world coordinates.
    ///
    /// This is the local +X axis transformed to world space.
    pub fn right(&self) -> Vector3<f64> {
        self.orientation.inverse() * Vector3::new(1.0, 0.0, 0.0)
    }

    /// Returns the camera's up direction in world coordinates.
    ///
    /// This is the local +Y axis transformed to world space.
    pub fn up(&self) -> Vector3<f64> {
        self.orientation.inverse() * Vector3::new(0.0, 1.0, 0.0)
    }

    /// Returns the rotation matrix from world to camera coordinates.
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        *self.orientation.to_rotation_matrix().matrix()
    }

    /// Returns the view matrix (world to camera transform).
    ///
    /// This transforms points from world space to camera space, where the camera
    /// is at the origin looking down -Z.
    pub fn view_matrix(&self) -> Matrix4<f64> {
        let r = self.rotation_matrix();
        let t = -(r * self.position.coords);

        Matrix4::new(
            r[(0, 0)],
            r[(0, 1)],
            r[(0, 2)],
            t.x,
            r[(1, 0)],
            r[(1, 1)],
            r[(1, 2)],
            t.y,
            r[(2, 0)],
            r[(2, 1)],
            r[(2, 2)],
            t.z,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Transforms a point from world space to camera (view) space.
    pub fn world_to_camera(&self, point: &Point3<f64>) -> Point3<f64> {
        let rotated = self.orientation * (point - self.position);
        Point3::from(rotated)
    }

    /// Transforms a point from camera space to world space.
    pub fn camera_to_world(&self, point: &Point3<f64>) -> Point3<f64> {
        self.position + self.orientation.inverse() * point.coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_look_at_basic() {
        let camera = Camera::look_at(Point3::new(0.0, -5.0, 0.0), Point3::origin(), Vector3::z());

        // Camera should be looking toward +Y (forward = +Y)
        let forward = camera.forward();
        assert_relative_eq!(forward.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(forward.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(forward.z, 0.0, epsilon = 1e-10);

        // Up should be +Z
        let up = camera.up();
        assert_relative_eq!(up.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(up.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(up.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_view_matrix_identity_at_origin() {
        // Camera at origin looking down -Z with Y up
        let camera = Camera {
            position: Point3::origin(),
            orientation: UnitQuaternion::identity(),
            target_distance: 5.0,
        };

        let _view = camera.view_matrix();

        // Point in front of camera (negative Z in world = negative Z in camera)
        let world_point = Point3::new(0.0, 0.0, -5.0);
        let camera_point = camera.world_to_camera(&world_point);

        assert_relative_eq!(camera_point.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(camera_point.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(camera_point.z, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_round_trip() {
        let camera = Camera::look_at(
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(4.0, 5.0, 6.0),
            Vector3::z(),
        );

        let world_point = Point3::new(10.0, 20.0, 30.0);
        let camera_point = camera.world_to_camera(&world_point);
        let back_to_world = camera.camera_to_world(&camera_point);

        assert_relative_eq!(back_to_world.x, world_point.x, epsilon = 1e-10);
        assert_relative_eq!(back_to_world.y, world_point.y, epsilon = 1e-10);
        assert_relative_eq!(back_to_world.z, world_point.z, epsilon = 1e-10);
    }
}