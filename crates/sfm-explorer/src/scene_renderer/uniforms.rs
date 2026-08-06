// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Uniform buffer update logic for the scene renderer.

use super::gpu_types::*;
use super::picking::PICK_INDEX_NONE;
use super::SceneRenderer;
use crate::scene::{ImageRef, PointRef};
use crate::viewer_3d::ViewportCamera;

/// The identity model matrix every node currently draws with. Node transforms
/// arrive in phase 4; the plumbing that carries them is already here.
const IDENTITY_MODEL: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl SceneRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &ViewportCamera,
        size_log2: f32,
        infinity_point_px: f32,
        show_infinity: bool,
        edl_line_thickness: f32,
        target_view_pos: [f32; 3],
        target_active: f32,
        target_radius: f32,
        time: f32,
        selected_point: Option<PointRef>,
        hovered_point: Option<PointRef>,
        hovered_image: Option<ImageRef>,
        patch_size_log2: f32,
        patch_opacity: f32,
        patch_alpha_cutoff: f32,
    ) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }

        let aspect = w as f64 / h as f64;
        let view = camera.view_matrix();
        let view_proj = camera.projection_matrix(aspect) * view;
        let size_multiplier = 2.0f32.powf(size_log2);

        // Selection and hover cross the GPU boundary as *global* pick indices,
        // so the shader compare stays one u32 test however many nodes are
        // loaded. A ref into an unloaded node resolves to the none sentinel.
        let global_point = |p: Option<PointRef>| {
            p.and_then(|p| self.global_point_index(p))
                .unwrap_or(PICK_INDEX_NONE)
        };
        let global_image = |i: Option<ImageRef>| {
            i.and_then(|i| self.global_image_index(i))
                .unwrap_or(PICK_INDEX_NONE)
        };

        // ── Point uniforms ──
        if let Some(buf) = &self.point_uniform_buffer {
            let uniforms = PointUniforms {
                view_proj: mat4_to_cols(&view_proj),
                view: mat4_to_cols(&view),
                camera_right: vec3_to_f32(&camera.camera.right()),
                _pad0: 0.0,
                camera_up: vec3_to_f32(&camera.camera.up()),
                selected_point_index: global_point(selected_point),
                hovered_point_index: global_point(hovered_point),
                screen_width: w as f32,
                screen_height: h as f32,
                infinity_point_px,
                show_infinity: if show_infinity { 1.0 } else { 0.0 },
                _pad: [0.0; 3],
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // ── Frustum uniforms ──
        if let Some(buf) = &self.frustum_uniform_buffer {
            let uniforms = FrustumUniforms {
                view_proj: mat4_to_cols(&view_proj),
                view: mat4_to_cols(&view),
                screen_size: [w as f32, h as f32],
                line_half_width: FRUSTUM_LINE_HALF_WIDTH,
                hovered_image_index: global_image(hovered_image),
                near: camera.near as f32,
                _pad: [0.0; 3],
            };

            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // ── Per-reconstruction uniforms ──
        //
        // Model matrix, splat size, pick bases, pickability and tint, plus the
        // per-node atlas blocks. Rewritten every frame, which is also what
        // makes a pick-base reassignment free: the new bases simply travel with
        // the next frame's write, and no instance buffer is touched.
        let cam_pos = camera.position();
        for bundle in self.recons.values() {
            queue.write_buffer(
                &bundle.uniform_buffer,
                0,
                bytemuck::bytes_of(&ReconUniforms {
                    model: IDENTITY_MODEL,
                    point_size: bundle.auto_point_size * size_multiplier,
                    point_pick_base: bundle.point_pick_base,
                    image_pick_base: bundle.image_pick_base,
                    // Every node is pickable until the Scene panel's
                    // interaction toggle arrives (phase 3).
                    pickable: 1,
                    // Alpha 0 = draw the node's original colors (phase 5).
                    tint_color: [0.0; 4],
                }),
            );

            // Image quad uniforms (this node's thumbnail atlas)
            if let Some(buf) = &bundle.image_quad_uniform_buffer {
                queue.write_buffer(
                    buf,
                    0,
                    bytemuck::bytes_of(&ImageQuadUniforms {
                        view_proj: mat4_to_cols(&view_proj),
                        atlas_cols: bundle.atlas_cols,
                        atlas_rows: bundle.atlas_rows,
                        images_per_page: bundle.images_per_page,
                        _pad: 0,
                    }),
                );
            }

            // Patch uniforms (this node's surfel atlas + the global controls)
            if let Some(patch) = &bundle.patch {
                queue.write_buffer(
                    &patch.uniform_buffer,
                    0,
                    bytemuck::bytes_of(&PatchUniforms {
                        view_proj: mat4_to_cols(&view_proj),
                        atlas_cols: patch.atlas_cols,
                        atlas_rows: patch.atlas_rows,
                        patches_per_page: patch.patches_per_page,
                        patch_scale: 2.0f32.powf(patch_size_log2),
                        patch_opacity,
                        alpha_cutoff: patch_alpha_cutoff,
                        _pad0: [0.0; 2],
                        camera_pos: [cam_pos.x as f32, cam_pos.y as f32, cam_pos.z as f32],
                        _pad1: 0.0,
                    }),
                );
            }
        }

        // ── EDL uniforms ──
        if let Some(buf) = &self.edl_uniform_buffer {
            // One fullscreen pass shades every node, so this is the one splat
            // size that cannot be per-recon.
            let point_size = self.max_auto_point_size() * size_multiplier;
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
            hovered_image_index: PICK_INDEX_NONE, // no hover for track rays
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
