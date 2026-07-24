// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU data types, constants, and small helpers for the scene renderer.

use nalgebra::{Matrix4, Vector3};

// ── GPU data types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct QuadVertex {
    pub position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PointInstance {
    pub position: [f32; 3],
    pub color: u32, // packed R8G8B8A8
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PointUniforms {
    pub view_proj: [[f32; 4]; 4], // column-major
    pub view: [[f32; 4]; 4],      // view matrix for linear depth
    pub camera_right: [f32; 3],
    pub point_size: f32,
    pub camera_up: [f32; 3],
    /// Index of the selected point (0xFFFFFFFF = no selection).
    pub selected_point_index: u32,
    /// Index of the hovered point (0xFFFFFFFF = no hover).
    pub hovered_point_index: u32,
    /// Viewport size in pixels — converts the infinity splat pixel radius to NDC.
    pub screen_width: f32,
    pub screen_height: f32,
    /// On-screen splat radius (pixels) for points at infinity.
    pub infinity_point_px: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EdgeInstance {
    pub endpoint_a: [f32; 3],
    pub endpoint_b: [f32; 3],
}

/// Edge instance with per-endpoint width for tapered compass spikes.
/// The w component of each endpoint is a width multiplier (0.0 = point, 1.0 = full width).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CompassEdgeInstance {
    pub endpoint_a: [f32; 4],
    pub endpoint_b: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FrustumEdge {
    pub endpoint_a: [f32; 3],
    pub _pad0: u32,
    pub endpoint_b: [f32; 3],
    pub frustum_index: u32, // image index for pick buffer
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageQuadInstance {
    pub corner_tl: [f32; 3],
    pub frustum_index: u32,
    pub corner_tr: [f32; 3],
    pub _pad0: u32,
    pub corner_bl: [f32; 3],
    pub _pad1: u32,
    pub corner_br: [f32; 3],
    pub _pad2: u32,
}

/// Per-instance data for one patch surfel (an oriented, textured quad).
///
/// The four corners are expanded in the vertex shader from the static unit
/// quad: `corner = center + s·u_halfvec + t·v_halfvec` for `(s, t) ∈ {±1}²`.
/// `atlas_layer` is the compacted cell index into the patch texture atlas
/// (patch-less points are skipped, so it is *not* the point index);
/// `point_index` is carried separately so picking resolves to the point.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PatchInstance {
    /// World position (unit direction when `w == 0`).
    pub center: [f32; 3],
    /// Homogeneous flag: 1.0 finite, 0.0 at infinity.
    pub w: f32,
    /// World-space u axis × half-extent.
    pub u_halfvec: [f32; 3],
    pub _pad0: f32,
    /// World-space v axis × half-extent.
    pub v_halfvec: [f32; 3],
    /// Compacted cell index into the patch atlas (page decoded in the shader).
    pub atlas_layer: u32,
    /// Global `recon.points` index, for the pick buffer.
    pub point_index: u32,
}

/// Uniforms for patch surfel rendering: view-projection plus the atlas grid
/// dimensions (mirroring [`ImageQuadUniforms`]) and the user patch controls.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PatchUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    pub patches_per_page: u32,
    /// User scale multiplier on the stored half-vecs (`2^patch_size_log2`).
    pub patch_scale: f32,
    /// Global opacity multiplier for patch color.
    pub patch_opacity: f32,
    /// Coverage discard threshold on the bitmap alpha (per-pixel confidence).
    pub alpha_cutoff: f32,
    pub _pad0: [f32; 2],
    /// Camera world position, for front-face culling of patch surfels.
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
}

/// Vertex for a tessellated (distorted) image quad.
///
/// All cameras' tessellated meshes are concatenated into a single vertex buffer
/// and index buffer. The position and UV are pre-computed on CPU.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DistortedQuadVertex {
    pub position: [f32; 3],
    pub frustum_index: u32,
    pub uv: [f32; 2],
    pub _pad: [f32; 2],
}

/// Vertex for the tessellated (distorted) background image mesh.
///
/// Positions are unit ray directions in the SfM camera's local space,
/// placed on the unit sphere via [`CameraIntrinsics::pixel_to_ray`].
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BgDistortedVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FrustumUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub line_half_width: f32,
    /// Index of the hovered image (0xFFFFFFFF = no hover).
    pub hovered_image_index: u32,
    /// Near clip plane distance. Used by the shader to clip line segments
    /// in view space before the manual perspective divide, so endpoints
    /// behind the camera don't flip sign and launch ribbons across the screen.
    pub near: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TargetIndicatorUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub target_pos_radius: [f32; 4], // xyz = target pos, w = radius
    pub indicator_rot_0: [f32; 4],   // rotation matrix column 0 (xyz), w = alpha_scale
    pub indicator_rot_1: [f32; 4],   // rotation matrix column 1 (xyz), w = fog_distance
    pub indicator_rot_2: [f32; 4],   // rotation matrix column 2 (xyz), w = unused
    pub screen_size_ps: [f32; 4],    // xy = screen_size, z = point_size, w = line_half_width
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EdlUniforms {
    pub screen_size: [f32; 2],
    pub radius: f32,
    pub strength: f32,
    pub opacity: f32,
    pub point_size: f32,
    pub target_view_pos: [f32; 2], // xy of target in view space
    pub target_view_z: f32,        // z of target in view space (positive = in front)
    pub target_active: f32,
    pub tan_half_fov: f32,
    pub aspect: f32,
    pub target_radius: f32,
    pub time: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BgImageUniforms {
    /// Projection matrix for the BG mesh (camera-space → clip-space).
    pub view_proj: [[f32; 4]; 4],
}

// ── Pick buffer constants ────────────────────────────────────────────────

/// Pick ID for "nothing" (background / no entity).
pub const PICK_TAG_NONE: u32 = 0x00_000000;
/// Pick ID tag for frustum / camera image entities.
pub const PICK_TAG_FRUSTUM: u32 = 0x01_000000;
/// Pick ID tag for 3D point entities.
pub const PICK_TAG_POINT: u32 = 0x02_000000;
/// Mask to extract the entity type tag (top 8 bits).
pub const PICK_TAG_MASK: u32 = 0xFF_000000;
/// Mask to extract the entity index (bottom 24 bits).
pub const PICK_INDEX_MASK: u32 = 0x00_FFFFFF;

// ── Related constants ────────────────────────────────────────────────────

/// Fallback point size when fewer than 2 points are loaded.
pub(super) const FALLBACK_POINT_SIZE: f32 = 0.03;

/// Number of points to subsample for nearest-neighbor distance queries.
pub(super) const NN_SUBSAMPLE_COUNT: usize = 10_000;

/// Default length scale multiplier (length_scale = multiplier * point_size).
pub const DEFAULT_LENGTH_SCALE_MULTIPLIER: f32 = 10.0;

/// Default target indicator size multiplier (radius = multiplier * length_scale).
pub const DEFAULT_TARGET_SIZE_MULTIPLIER: f32 = 0.3;

/// Default target indicator fog multiplier (fog_distance = multiplier * length_scale).
pub const DEFAULT_TARGET_FOG_MULTIPLIER: f32 = 10.0;

/// Default frustum stub depth as a fraction of `length_scale`.
pub const DEFAULT_FRUSTUM_SIZE_MULTIPLIER: f32 = 0.5;

/// Half-width of indicator lines in pixels.
pub(super) const INDICATOR_LINE_HALF_WIDTH: f32 = 2.0;

/// Half-width of frustum lines in pixels.
pub(super) const FRUSTUM_LINE_HALF_WIDTH: f32 = 1.0;

/// Uniforms for image quad / distorted quad thumbnail rendering.
///
/// Contains the view-projection matrix plus atlas grid dimensions so the
/// shader can compute UV coordinates into the 2D texture array atlas.
/// Each layer of the array holds one page of the grid.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageQuadUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    pub images_per_page: u32,
    pub _pad: u32,
}

/// Thumbnail dimensions for the texture atlas (square, non-square pixels).
///
/// Source images are stretched to this fixed square size regardless of their
/// native aspect ratio. The distortion is compensated on the GPU: each
/// frustum quad has the correct aspect ratio from camera intrinsics, and the
/// UV 0→1 mapping stretches the square texture back to the original proportions.
pub(super) const THUMBNAIL_SIZE: u32 = 128;

/// Maximum number of thumbnail columns per atlas page.
///
/// Also constrained at runtime by the GPU's `max_texture_dimension_2d` limit.
/// The atlas uses a `texture_2d_array` with multiple pages when one page is
/// not enough to hold all thumbnails.
pub(super) const MAX_ATLAS_COLS: u32 = 128;

/// Number of subdivisions per edge for distorted frustum grids.
/// Grid has (DISTORTION_SUBDIVISIONS + 1)^2 vertices per camera.
pub(super) const DISTORTION_SUBDIVISIONS: usize = 4;

/// Number of subdivisions per edge for fisheye frustum grids.
/// Higher than perspective grids because the spherical far-surface has more curvature.
pub(super) const FISHEYE_SUBDIVISIONS: usize = 16;

/// Number of subdivisions for pinhole background image mesh.
/// Pinhole cameras have no distortion, so minimal tessellation suffices.
pub(super) const BG_PINHOLE_SUBDIVISIONS: usize = 1;

/// Number of subdivisions per edge for the distorted background image mesh.
/// Higher than frustum grids because the background fills the viewport.
pub(super) const BG_DISTORTION_SUBDIVISIONS: usize = 32;

/// Number of subdivisions for fisheye background image mesh.
/// Higher than perspective cameras due to the highly non-linear re-projection.
pub(super) const BG_FISHEYE_SUBDIVISIONS: usize = 64;

// ── Helpers ─────────────────────────────────────────────────────────────

pub(super) fn mat4_to_cols(m: &Matrix4<f64>) -> [[f32; 4]; 4] {
    // Each inner array is one column: [row0, row1, row2, row3].
    [
        [
            m[(0, 0)] as f32,
            m[(1, 0)] as f32,
            m[(2, 0)] as f32,
            m[(3, 0)] as f32,
        ],
        [
            m[(0, 1)] as f32,
            m[(1, 1)] as f32,
            m[(2, 1)] as f32,
            m[(3, 1)] as f32,
        ],
        [
            m[(0, 2)] as f32,
            m[(1, 2)] as f32,
            m[(2, 2)] as f32,
            m[(3, 2)] as f32,
        ],
        [
            m[(0, 3)] as f32,
            m[(1, 3)] as f32,
            m[(2, 3)] as f32,
            m[(3, 3)] as f32,
        ],
    ]
}

pub(super) fn vec3_to_f32(v: &Vector3<f64>) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}
