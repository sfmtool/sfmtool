// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Uniform buffer update logic for the scene renderer.

use super::gpu_types::*;
use super::SceneRenderer;
use crate::viewer_3d::ViewportCamera;

impl SceneRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &ViewportCamera,
        size_log2: f32,
        edl_line_thickness: f32,
        target_view_pos: [f32; 3],
        target_active: f32,
        target_radius: f32,
        time: f32,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        hovered_image: Option<usize>,
    ) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }

        let aspect = w as f64 / h as f64;

        // ── Point uniforms ──
        if let Some(buf) = &self.point_uniform_buffer {
            let view = camera.view_matrix();
            let view_proj = camera.projection_matrix(aspect) * view;

            let point_size = self.auto_point_size * 2.0f32.powf(size_log2);

            let uniforms = PointUniforms {
                view_proj: mat4_to_cols(&view_proj),
                view: mat4_to_cols(&view),
                camera_right: vec3_to_f32(&camera.camera.right()),
                point_size,
                camera_up: vec3_to_f32(&camera.camera.up()),
                selected_point_index: selected_point.map(|i| i as u32).unwrap_or(0xFFFFFFFF),
                hovered_point_index: hovered_point.map(|i| i as u32).unwrap_or(0xFFFFFFFF),
                _pad: [0; 3],
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // ── Frustum uniforms ──
        if let Some(buf) = &self.frustum_uniform_buffer {
            let view = camera.view_matrix();
            let view_proj = camera.projection_matrix(aspect) * view;

            let uniforms = FrustumUniforms {
                view_proj: mat4_to_cols(&view_proj),
                view: mat4_to_cols(&view),
                screen_size: [w as f32, h as f32],
                line_half_width: FRUSTUM_LINE_HALF_WIDTH,
                hovered_image_index: hovered_image.map(|i| i as u32).unwrap_or(0xFFFFFFFF),
                near: camera.near as f32,
                _pad: [0.0; 3],
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // ── Image quad uniforms (thumbnail atlas) ──
        if let Some(buf) = &self.image_quad_uniform_buffer {
            let view = camera.view_matrix();
            let view_proj = camera.projection_matrix(aspect) * view;

            let uniforms = ImageQuadUniforms {
                view_proj: mat4_to_cols(&view_proj),
                atlas_cols: self.atlas_cols,
                atlas_rows: self.atlas_rows,
                images_per_page: self.images_per_page,
                _pad: 0,
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // ── EDL uniforms ──
        if let Some(buf) = &self.edl_uniform_buffer {
            let point_size = self.auto_point_size * 2.0f32.powf(size_log2);
            let tan_half_fov = (camera.fov / 2.0).tan() as f32;
            let uniforms = EdlUniforms {
                screen_size: [w as f32, h as f32],
                radius: edl_line_thickness,
                strength: 0.7,
                opacity: 1.0,
                point_size,
                target_view_pos: [target_view_pos[0], target_view_pos[1]],
                target_view_z: target_view_pos[2],
                target_active,
                tan_half_fov,
                aspect: aspect as f32,
                target_radius,
                time,
                _pad: [0.0; 2],
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    /// Update track ray uniforms.
    pub fn update_track_ray_uniforms(&self, queue: &wgpu::Queue, camera: &ViewportCamera) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }
        let Some(buf) = &self.track_ray_uniform_buffer else {
            return;
        };
        if self.track_ray_count == 0 {
            return;
        }

        let aspect = w as f64 / h as f64;
        let view = camera.view_matrix();
        let view_proj = camera.projection_matrix(aspect) * view;

        let uniforms = FrustumUniforms {
            view_proj: mat4_to_cols(&view_proj),
            view: mat4_to_cols(&view),
            screen_size: [w as f32, h as f32],
            line_half_width: 1.5,
            hovered_image_index: 0xFFFFFFFF, // no hover for track rays
            near: camera.near as f32,
            _pad: [0.0; 3],
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Update background image uniforms for camera view mode.
    ///
    /// The BG mesh vertices are world-space ray directions (transformed from
    /// camera-local rays by the camera-to-world rotation during mesh generation).
    /// This is the same coordinate convention as frustum wireframes and image
    /// quads, so we use the same `projection * view` transform pipeline.
    ///
    /// The shader uses `w=0` to treat vertices as directions (ignoring the
    /// translation component of the view matrix), so only the rotation part
    /// of the view matrix has any effect.
    pub fn update_bg_image_uniforms(&self, queue: &wgpu::Queue, camera: &ViewportCamera) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }
        let Some(buf) = &self.bg_image_uniform_buffer else {
            return;
        };

        let aspect = w as f64 / h as f64;
        let view = camera.view_matrix();
        let view_proj = camera.projection_matrix(aspect) * view;

        let uniforms = BgImageUniforms {
            view_proj: mat4_to_cols(&view_proj),
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Update target indicator uniforms.
    #[allow(clippy::too_many_arguments)]
    pub fn update_target_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &ViewportCamera,
        target_pos: [f32; 3],
        rotation_angle: f32,
        world_up: [f32; 3],
        alpha_scale: f32,
        size_multiplier: f32,
        fog_multiplier: f32,
        length_scale: f32,
    ) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }

        let Some(buf) = &self.target_uniform_buffer else {
            return;
        };

        let aspect = w as f64 / h as f64;
        let view = camera.view_matrix();
        let view_proj = camera.projection_matrix(aspect) * view;

        let indicator_radius = size_multiplier * length_scale;
        // fog_distance is in NDC depth space (0-1 range). A small value gives
        // a quick fade; 0.1 means full fade at 10% of the depth range.
        let fog_distance = 0.1 * fog_multiplier / DEFAULT_TARGET_FOG_MULTIPLIER;

        // Build rotation matrix: rotate Z-up compass to align with world_up,
        // then spin around world_up by rotation_angle.
        let up = nalgebra::Vector3::new(world_up[0] as f64, world_up[1] as f64, world_up[2] as f64)
            .normalize();
        let z = nalgebra::Vector3::z();

        // Rotation from Z to world_up
        let align = if (up - z).norm() < 1e-10 {
            nalgebra::UnitQuaternion::identity()
        } else if (up + z).norm() < 1e-10 {
            // 180° flip — rotate around X
            nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                std::f64::consts::PI,
            )
        } else {
            nalgebra::UnitQuaternion::rotation_between(&z, &up).unwrap()
        };

        // Spin around world_up
        let spin = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(up),
            rotation_angle as f64,
        );

        let rot = (spin * align).to_rotation_matrix();
        let m = rot.matrix();

        let uniforms = TargetIndicatorUniforms {
            view_proj: mat4_to_cols(&view_proj),
            view: mat4_to_cols(&view),
            target_pos_radius: [
                target_pos[0],
                target_pos[1],
                target_pos[2],
                indicator_radius,
            ],
            // Columns of rotation matrix (WGSL mat3x3 constructor takes columns)
            indicator_rot_0: [
                m[(0, 0)] as f32,
                m[(1, 0)] as f32,
                m[(2, 0)] as f32,
                alpha_scale,
            ],
            indicator_rot_1: [
                m[(0, 1)] as f32,
                m[(1, 1)] as f32,
                m[(2, 1)] as f32,
                fog_distance,
            ],
            indicator_rot_2: [m[(0, 2)] as f32, m[(1, 2)] as f32, m[(2, 2)] as f32, 0.0],
            screen_size_ps: [w as f32, h as f32, 0.0, INDICATOR_LINE_HALF_WIDTH],
        };

        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }
}
