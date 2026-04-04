// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Viewport camera for 3D navigation.
//!
//! [`ViewportCamera`] wraps a [`Camera`] with viewport-specific settings
//! (FOV, clip planes) and provides navigation methods (orbit, pan, zoom, fly).

use eframe::egui::{Pos2, Rect};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3, Vector4};
use sfmtool_core::Camera;

/// Camera for 3D viewport navigation.
///
/// Wraps a [`Camera`] with viewport-specific settings (FOV, clip planes) and
/// provides navigation methods (orbit, pan, zoom).
pub struct ViewportCamera {
    /// The underlying camera with position and orientation.
    pub camera: Camera,
    /// World up direction for navigation (typically Z-up).
    pub world_up: Vector3<f64>,
    /// Field of view of the shorter viewport dimension, in radians.
    ///
    /// In landscape windows, this is the vertical FOV. In portrait windows,
    /// this is the horizontal FOV and the vertical FOV is derived as
    /// `fov / aspect`. This keeps the perceived "how much you can see"
    /// consistent regardless of window shape.
    pub fov: f64,
    /// Near clip plane distance.
    ///
    /// With reversed-Z infinite far projection, this is the only clip plane.
    /// Objects closer than `near` are clipped; there is no far-plane clipping.
    pub near: f64,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        let position = Point3::new(0.0, -5.0, 2.0);
        let target = Point3::origin();
        let world_up = Vector3::z();

        Self {
            camera: Camera::look_at(position, target, world_up),
            world_up,
            fov: std::f64::consts::FRAC_PI_4,
            near: 0.1,
        }
    }
}

impl ViewportCamera {
    /// Returns the vertical FOV for the given aspect ratio.
    ///
    /// `self.fov` is the FOV of the shorter viewport dimension. In landscape
    /// (aspect >= 1) that's vertical, so this returns `self.fov` directly.
    /// In portrait (aspect < 1) `self.fov` is horizontal, and the vertical
    /// FOV is wider: `atan(tan(fov/2) / aspect) * 2`.
    pub fn vertical_fov(&self, aspect: f64) -> f64 {
        if aspect >= 1.0 {
            self.fov
        } else {
            ((self.fov / 2.0).tan() / aspect).atan() * 2.0
        }
    }

    /// Returns the camera position in world space.
    pub fn position(&self) -> Point3<f64> {
        self.camera.position
    }

    /// Returns the target point the camera is looking at.
    pub fn target(&self) -> Point3<f64> {
        self.camera.target()
    }

    /// Orbit the camera around the target point.
    ///
    /// The orbit pivot is the point `target_distance` in front of the camera.
    /// Uses spherical coordinates relative to `world_up`: horizontal drag
    /// rotates around the up axis, vertical drag tilts toward/away from it.
    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        let pivot = self.camera.target();
        let radius = self.camera.target_distance;
        let dir = (self.camera.position - pivot).normalize();

        // Decompose dir into spherical coordinates relative to world_up.
        // theta = angle from world_up axis, phi = angle in the perpendicular plane.
        let up = self.world_up;
        let cos_theta = dir.dot(&up);
        let theta = cos_theta.acos();

        // Project dir onto the plane perpendicular to world_up
        let dir_flat = dir - up * cos_theta;
        let flat_norm = dir_flat.norm();

        // Build an orthonormal basis in the perpendicular plane
        let (basis_x, basis_y) = if flat_norm > 1e-10 {
            let bx = dir_flat / flat_norm;
            let by = up.cross(&bx);
            (bx, by)
        } else {
            // Camera is at a pole — pick arbitrary perpendicular axes
            let arbitrary = if up.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let bx = up.cross(&arbitrary).normalize();
            let by = up.cross(&bx);
            (bx, by)
        };

        let phi = dir_flat.dot(&basis_y).atan2(dir_flat.dot(&basis_x));

        let new_phi = phi - delta_x * 0.01;
        let new_theta = (theta - delta_y * 0.01).clamp(0.01, std::f64::consts::PI - 0.01);

        let new_dir = up * new_theta.cos()
            + (basis_x * new_phi.cos() + basis_y * new_phi.sin()) * new_theta.sin();

        self.camera.position = pivot + new_dir.normalize() * radius;
        let forward = (pivot - self.camera.position).normalize();
        self.set_orientation_from_forward(forward);
    }

    /// Pan the camera parallel to the view plane.
    pub fn pan(&mut self, delta_x: f64, delta_y: f64, viewport_width: f64, viewport_height: f64) {
        let right = self.camera.right();
        let up = self.camera.up();

        // Scale so objects at target_distance track 1:1 with input pixels
        let aspect = viewport_width / viewport_height;
        let vfov = self.vertical_fov(aspect);
        let speed = 2.0 * self.camera.target_distance * (vfov / 2.0).tan() / viewport_height;
        let pan = right * delta_x * speed + up * delta_y * speed;
        self.camera.position += pan;
    }

    /// Zoom the camera (move closer/further from target).
    pub fn zoom(&mut self, delta: f64) {
        let new_distance = (self.camera.target_distance * (1.0 - delta * 0.1)).max(0.1);
        // Move camera along forward direction to maintain the same target point
        let target = self.camera.target();
        self.camera.target_distance = new_distance;
        self.camera.position = target - self.camera.forward() * new_distance;
    }

    /// Zoom the FOV (narrow/widen field of view without moving the camera).
    ///
    /// Positive delta zooms in (narrows FOV), negative zooms out (widens FOV).
    /// Clamps to 5°–160° range.
    pub fn zoom_fov(&mut self, delta: f64) {
        let min_fov: f64 = 5.0_f64.to_radians();
        let max_fov: f64 = 160.0_f64.to_radians();
        self.fov = (self.fov * (1.0 - delta * 0.1)).clamp(min_fov, max_fov);
    }

    /// Nodal pan: camera stays in place, view direction changes.
    ///
    /// The target slides to the new look-at point at the current distance.
    /// This is the dual of [`orbit`]: orbit moves the camera around a fixed target,
    /// while nodal pan moves the target around a fixed camera.
    pub fn nodal_pan(&mut self, delta_x: f64, delta_y: f64) {
        let forward = self.camera.forward();
        let up = self.world_up;

        // Decompose forward into spherical coordinates relative to world_up
        let cos_theta = forward.dot(&up);
        let theta = cos_theta.acos();
        let fwd_flat = forward - up * cos_theta;
        let flat_norm = fwd_flat.norm();

        let (basis_x, basis_y) = if flat_norm > 1e-10 {
            let bx = fwd_flat / flat_norm;
            let by = up.cross(&bx);
            (bx, by)
        } else {
            let arbitrary = if up.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let bx = up.cross(&arbitrary).normalize();
            let by = up.cross(&bx);
            (bx, by)
        };

        let phi = fwd_flat.dot(&basis_y).atan2(fwd_flat.dot(&basis_x));

        let new_phi = phi - delta_x * 0.01;
        let new_theta = (theta + delta_y * 0.01).clamp(0.01, std::f64::consts::PI - 0.01);

        let new_forward = up * new_theta.cos()
            + (basis_x * new_phi.cos() + basis_y * new_phi.sin()) * new_theta.sin();

        self.set_orientation_from_forward(new_forward.normalize());
    }

    /// Push/pull the target point closer or further from the camera.
    ///
    /// Camera position and orientation remain unchanged; only target_distance is adjusted.
    pub fn target_push_pull(&mut self, delta: f64) {
        self.camera.target_distance = (self.camera.target_distance * (1.0 + delta * 0.1)).max(0.1);
    }

    /// Move the camera in first-person fly mode.
    ///
    /// `forward` moves along the camera's view direction, `right` strafes along
    /// `world_up × forward`, and `up` moves along the camera's visual up (the
    /// component of `world_up` perpendicular to forward). The target point moves
    /// with the camera (target_distance preserved).
    pub fn fly_move(&mut self, forward: f64, right: f64, up: f64) {
        // Use camera's actual forward and derive right/up perpendicular to it.
        // This makes R/F move up/down relative to the camera view rather than
        // along the world up axis.
        let cam_forward = self.camera.forward();
        let cam_right = cam_forward.cross(&self.world_up);
        let cam_right = if cam_right.norm() > 1e-10 {
            cam_right.normalize()
        } else {
            // Looking straight up/down — use camera's right vector
            self.camera.right()
        };
        let cam_up = cam_right.cross(&cam_forward).normalize();

        let offset = cam_forward * forward + cam_right * right + cam_up * up;
        self.camera.position += offset;
    }

    /// Tilt (roll) the camera by rotating `world_up` around the view axis.
    ///
    /// This tilts the horizon — subsequent orbit and pan operations will use
    /// the new up direction. Use Home key to level the horizon.
    pub fn tilt(&mut self, angle: f64) {
        let forward = self.camera.forward();
        let rotation = UnitQuaternion::from_scaled_axis(forward * angle);
        self.world_up = (rotation * self.world_up).normalize();
        self.set_orientation_from_forward(forward);
    }

    /// Sets the orbit target to a specific world-space point.
    ///
    /// The camera position stays fixed. Target distance is updated to the
    /// actual distance from the camera to the new target.
    pub fn set_target_to_point(&mut self, target: Point3<f64>) {
        let new_forward = (target - self.camera.position).normalize();
        self.set_orientation_from_forward(new_forward);
        self.camera.target_distance = (target - self.camera.position).norm().max(0.1);
    }

    /// Computes the end state for zoom-to-fit without applying it.
    ///
    /// Returns `(position, target_distance)` or `None` if points is empty.
    pub fn compute_zoom_to_fit(
        &self,
        points: &[Point3<f64>],
        aspect: f64,
    ) -> Option<(Point3<f64>, f64)> {
        if points.is_empty() {
            return None;
        }

        // Transform points into camera space (right, up, -forward)
        let right = self.camera.right();
        let up = self.camera.up();
        let forward = self.camera.forward();
        let cam_pos = self.camera.position;

        let mut view_xs: Vec<f64> = Vec::with_capacity(points.len());
        let mut view_ys: Vec<f64> = Vec::with_capacity(points.len());
        let mut view_zs: Vec<f64> = Vec::with_capacity(points.len());
        for p in points {
            let d = p - cam_pos;
            view_xs.push(d.dot(&right));
            view_ys.push(d.dot(&up));
            view_zs.push(d.dot(&forward)); // positive = in front
        }

        view_xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        view_ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        view_zs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = view_xs.len();
        let lo = n / 5; // 20th percentile
        let hi = n * 4 / 5; // 80th percentile

        // Percentile bounding box center in camera space
        let cx = (view_xs[lo] + view_xs[hi]) * 0.5;
        let cy = (view_ys[lo] + view_ys[hi]) * 0.5;
        let cz = (view_zs[lo] + view_zs[hi]) * 0.5;

        // Size of the bounding box in the view plane (right/up extents)
        let sx = view_xs[hi] - view_xs[lo];
        let sy = view_ys[hi] - view_ys[lo];

        // Required distance to frame the bounding box with margin.
        // Check both axes and take the larger required distance.
        let margin = 1.2;
        let vfov = self.vertical_fov(aspect);
        let tan_half_vfov = (vfov / 2.0).tan();
        let tan_half_hfov = tan_half_vfov * aspect;
        let dist_for_height = (sy * margin) / tan_half_vfov;
        let dist_for_width = (sx * margin) / tan_half_hfov;
        let camera_distance = dist_for_height.max(dist_for_width);

        // World-space center of the percentile bounding box
        let world_center = cam_pos + right * cx + up * cy + forward * cz;

        Some((world_center - forward * camera_distance, camera_distance))
    }

    /// Zoom to fit the bulk of the points in view, keeping the current camera orientation.
    ///
    /// Transforms points into camera space, computes the 20th–80th percentile
    /// bounding box aligned to the view axes, then repositions the camera
    /// to center and frame that box.
    pub fn zoom_to_fit(&mut self, points: &[Point3<f64>], aspect: f64) {
        if let Some((position, distance)) = self.compute_zoom_to_fit(points, aspect) {
            self.camera.position = position;
            self.camera.target_distance = distance;
        }
    }

    /// Update the near clip plane based on scene bounds and camera position.
    ///
    /// With reversed-Z infinite far projection there is no far plane to adjust.
    /// The near plane is set to a fraction of the camera-to-scene distance so
    /// that close-up inspection still works. Uses time-based exponential decay
    /// so the transition speed is independent of frame rate.
    pub fn update_clip_planes(&mut self, scene_center: Point3<f64>, scene_radius: f64, dt: f64) {
        let d = (self.position() - scene_center).norm();
        // Near = 1/1000th of distance to the far edge of the scene, floor at 0.0001
        let target_near = ((d + scene_radius) / 1000.0).max(0.0001);

        // Time-based exponential decay (~120ms settling at any frame rate)
        let alpha = (1.0 - (-dt * 8.0).exp()).clamp(0.0, 1.0);
        self.near += (target_near - self.near) * alpha;
    }

    /// Sets the camera orientation from a forward direction, preserving target_distance.
    pub(crate) fn set_orientation_from_forward(&mut self, forward: Vector3<f64>) {
        self.camera.orientation = Camera::orientation_from_forward(forward, self.world_up);
    }

    /// Returns the view matrix (world to camera transform).
    pub fn view_matrix(&self) -> Matrix4<f64> {
        self.camera.view_matrix()
    }

    /// Returns the projection matrix for the given aspect ratio.
    ///
    /// Uses **reversed-Z with an infinite far plane**: near maps to depth 1.0,
    /// infinity maps to depth 0.0. This eliminates far-plane clipping entirely
    /// and gives optimal depth precision across all distances when used with a
    /// `Depth32Float` buffer and `CompareFunction::Greater`.
    pub fn projection_matrix(&self, aspect: f64) -> Matrix4<f64> {
        let vfov = self.vertical_fov(aspect);
        let f = 1.0 / (vfov / 2.0).tan();

        // Reversed-Z infinite far plane projection matrix.
        // Maps z_view = -near to ndc_z = 1, z_view = -∞ to ndc_z = 0.
        Matrix4::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.near,
            0.0,
            0.0,
            -1.0,
            0.0,
        )
    }

    /// Transforms a point from world space to view (camera) space.
    pub fn world_to_view(&self, point: &Point3<f64>) -> Point3<f64> {
        self.camera.world_to_camera(point)
    }

    /// Projects a 3D point to normalized device coordinates.
    /// Returns None if the point is behind the camera.
    pub fn project(&self, point: &Point3<f64>, aspect: f64) -> Option<(f64, f64, f64)> {
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let mvp = proj * view;

        let p = Vector4::new(point.x, point.y, point.z, 1.0);
        let clip = mvp * p;

        // Behind camera check
        if clip.w <= 0.0 {
            return None;
        }

        // Perspective divide to get NDC
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let depth = clip.z / clip.w;

        Some((ndc_x, ndc_y, depth))
    }

    /// Projects a line segment to screen coordinates, clipping against the near plane.
    ///
    /// Returns None if the entire line is behind the camera.
    /// If one endpoint is behind the near plane, the line is clipped to the near plane.
    pub fn project_line_clipped(
        &self,
        p1: &Point3<f64>,
        p2: &Point3<f64>,
        rect: Rect,
    ) -> Option<(Pos2, Pos2)> {
        // Transform to view space (camera looks down -Z, so near plane is at z = -near)
        let v1 = self.camera.world_to_camera(p1);
        let v2 = self.camera.world_to_camera(p2);

        // In view space, camera looks down -Z, so points in front have z < 0
        // Near plane is at z = -near
        let near_z = -self.near;

        let in_front_1 = v1.z < near_z;
        let in_front_2 = v2.z < near_z;

        // Both behind - no line to draw
        if !in_front_1 && !in_front_2 {
            return None;
        }

        // Clip against near plane if needed
        let (clipped_p1, clipped_p2) = if in_front_1 && in_front_2 {
            // Both in front - no clipping needed
            (*p1, *p2)
        } else {
            // One point behind - find intersection with near plane
            let t = (near_z - v1.z) / (v2.z - v1.z);
            let intersection = Point3::new(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y),
                p1.z + t * (p2.z - p1.z),
            );

            if in_front_1 {
                (*p1, intersection)
            } else {
                (intersection, *p2)
            }
        };

        // Project the clipped endpoints
        let aspect = rect.width() as f64 / rect.height() as f64;
        let s1 = self.project_to_screen_unchecked(&clipped_p1, rect, aspect)?;
        let s2 = self.project_to_screen_unchecked(&clipped_p2, rect, aspect)?;

        Some((s1, s2))
    }

    /// Projects a point to screen coordinates without the behind-camera check.
    /// Used after manual near-plane clipping.
    fn project_to_screen_unchecked(
        &self,
        point: &Point3<f64>,
        rect: Rect,
        aspect: f64,
    ) -> Option<Pos2> {
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let mvp = proj * view;

        let p = Vector4::new(point.x, point.y, point.z, 1.0);
        let clip = mvp * p;

        if clip.w <= 0.0 {
            return None;
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        let screen_x = rect.center().x + (ndc_x as f32) * rect.width() * 0.5;
        let screen_y = rect.center().y - (ndc_y as f32) * rect.height() * 0.5;

        Some(Pos2::new(screen_x, screen_y))
    }

    /// Unprojects a screen pixel position and linear depth to world space.
    ///
    /// Given a screen pixel `(sx, sy)` within `rect` and a positive linear
    /// view-space depth `d`, returns the corresponding world-space point.
    pub fn unproject(&self, sx: f32, sy: f32, depth: f64, rect: Rect) -> Point3<f64> {
        let aspect = rect.width() as f64 / rect.height() as f64;
        let vfov = self.vertical_fov(aspect);
        let half_vfov_tan = (vfov / 2.0).tan();

        // Screen to NDC (Y negated: screen Y down, NDC Y up)
        let ndc_x = (sx - rect.center().x) as f64 / (rect.width() as f64 * 0.5);
        let ndc_y = -((sy - rect.center().y) as f64 / (rect.height() as f64 * 0.5));

        // NDC to view space
        let view_x = ndc_x * depth * aspect * half_vfov_tan;
        let view_y = ndc_y * depth * half_vfov_tan;
        let view_z = -depth; // camera looks down -Z in view space

        // View space to world space
        self.camera
            .camera_to_world(&Point3::new(view_x, view_y, view_z))
    }

    /// Projects a 3D point to screen coordinates.
    /// Returns None if the point is behind the camera or outside the view.
    pub fn project_to_screen(&self, point: &Point3<f64>, rect: Rect) -> Option<(Pos2, f64)> {
        let aspect = rect.width() as f64 / rect.height() as f64;
        let (ndc_x, ndc_y, depth) = self.project(point, aspect)?;

        // Check if in view frustum (with some margin)
        let frustum_range = -1.5..=1.5;
        if !frustum_range.contains(&ndc_x) || !frustum_range.contains(&ndc_y) {
            return None;
        }

        // Convert NDC to screen coordinates
        let screen_x = rect.center().x + (ndc_x as f32) * rect.width() * 0.5;
        let screen_y = rect.center().y - (ndc_y as f32) * rect.height() * 0.5;

        Some((Pos2::new(screen_x, screen_y), depth))
    }
}

/// Compute the best-fit `fov` value for a camera's intrinsic FOVs in the given viewport.
///
/// Fits the camera's field of view into the viewport so the image is as
/// large as possible without cropping. Returns the `fov` value in the
/// min-dimension convention used by [`ViewportCamera`].
///
/// `vfov_cam` and `hfov_cam` are the camera's intrinsic vertical and horizontal
/// fields of view in radians.
pub fn best_fit_fov(vfov_cam: f64, hfov_cam: f64, viewport_aspect: f64) -> f64 {
    // Try fitting by vertical FOV: check if horizontal FOV fits
    let hfov_at_vfov = ((vfov_cam / 2.0).tan() * viewport_aspect).atan() * 2.0;
    let vfov = if hfov_at_vfov >= hfov_cam {
        // Image fits horizontally at vfov_cam — use it
        vfov_cam
    } else {
        // Image is wider than viewport can show — fit by horizontal FOV
        ((hfov_cam / 2.0).tan() / viewport_aspect).atan() * 2.0
    };
    // Convert vertical FOV to min-dimension convention
    if viewport_aspect >= 1.0 {
        // Landscape: min dimension is vertical, so fov = vfov
        vfov
    } else {
        // Portrait: min dimension is horizontal, derive from vfov
        ((vfov / 2.0).tan() * viewport_aspect).atan() * 2.0
    }
}