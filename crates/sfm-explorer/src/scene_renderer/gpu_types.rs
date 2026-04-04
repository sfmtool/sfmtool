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
    pub _pad: [u32; 3], // Pad to 16-byte alignment
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
    pub color_packed: u32, // R8G8B8A8
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

// ── Compass geometry ────────────────────────────────────────────────────

/// Number of segments for the horizontal ring circle.
const RING_SEGMENTS: usize = 32;

/// Radius of the horizontal ring circle and star outer tips.
const RING_RADIUS: f32 = 0.6;

/// Number of star points (cardinal + intercardinal directions).
const STAR_POINTS: usize = 8;

/// Inner circle radius for the star indentations (1/5 of outer radius).
const STAR_INNER_RADIUS: f32 = RING_RADIUS / 5.0;

/// Outer tip radii: cardinal (N/S/E/W) tips extend beyond the ring,
/// intercardinal tips are shorter.
const STAR_CARDINAL_RADIUS: f32 = RING_RADIUS * 1.25;
const STAR_INTERCARDINAL_RADIUS: f32 = RING_RADIUS * 0.8;

/// Wireframe edges: vertical axis + circular ring.
///
/// The horizontal compass rose is rendered as a filled star polygon, so only
/// the vertical spikes and ring remain as wireframe edges.
pub(super) fn create_compass_edge_instances() -> Vec<CompassEdgeInstance> {
    let mut edges: Vec<CompassEdgeInstance> = Vec::new();

    // Vertical axis: center to top, center to bottom
    let center = [0.0, 0.0, 0.0, 1.0];
    edges.push(CompassEdgeInstance {
        endpoint_a: center,
        endpoint_b: [0.0, 0.0, 1.5, 1.0], // top spike
    });
    edges.push(CompassEdgeInstance {
        endpoint_a: center,
        endpoint_b: [0.0, 0.0, -0.7, 1.0], // bottom spike
    });

    // Circular ring in the horizontal (z=0) plane
    for i in 0..RING_SEGMENTS {
        let angle_a = (i as f32) * std::f32::consts::TAU / RING_SEGMENTS as f32;
        let angle_b =
            ((i + 1) % RING_SEGMENTS) as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
        edges.push(CompassEdgeInstance {
            endpoint_a: [
                RING_RADIUS * angle_a.cos(),
                RING_RADIUS * angle_a.sin(),
                0.0,
                1.0,
            ],
            endpoint_b: [
                RING_RADIUS * angle_b.cos(),
                RING_RADIUS * angle_b.sin(),
                0.0,
                1.0,
            ],
        });
    }

    edges
}

/// Generate a filled 8-point star polygon as a triangle list in the z=0 plane.
///
/// Cardinal tips (N/E/S/W, indices 0/2/4/6) extend to `STAR_CARDINAL_RADIUS`.
/// Intercardinal tips (NE/SE/SW/NW, indices 1/3/5/7) extend to
/// `STAR_INTERCARDINAL_RADIUS`. Inner indentations are offset 2/3 towards
/// the intercardinal tip, giving cardinals wider bases and intercardinals
/// narrower bases. Triangulated as a center fan (16 triangles).
pub(super) fn create_compass_star_mesh() -> Vec<[f32; 3]> {
    use std::f32::consts::TAU;

    let tip_radius = |i: usize| -> f32 {
        if i.is_multiple_of(2) {
            STAR_CARDINAL_RADIUS
        } else {
            STAR_INTERCARDINAL_RADIUS
        }
    };

    let mut verts = Vec::with_capacity(STAR_POINTS * 2 * 3);
    let center = [0.0f32, 0.0, 0.0];

    for i in 0..STAR_POINTS {
        let outer_angle = i as f32 * TAU / STAR_POINTS as f32;
        // Offset inner vertex 2/3 towards the intercardinal side:
        // even i (cardinal→intercardinal): 2/3 towards next
        // odd i (intercardinal→cardinal): 1/3 towards next (= 2/3 back towards current)
        let inner_offset = if i.is_multiple_of(2) {
            2.0 / 3.0
        } else {
            1.0 / 3.0
        };
        let inner_angle = (i as f32 + inner_offset) * TAU / STAR_POINTS as f32;
        let next_i = (i + 1) % STAR_POINTS;
        let next_outer_angle = next_i as f32 * TAU / STAR_POINTS as f32;

        let r = tip_radius(i);
        let outer = [r * outer_angle.cos(), r * outer_angle.sin(), 0.0];
        let inner = [
            STAR_INNER_RADIUS * inner_angle.cos(),
            STAR_INNER_RADIUS * inner_angle.sin(),
            0.0,
        ];
        let r_next = tip_radius(next_i);
        let next_outer = [
            r_next * next_outer_angle.cos(),
            r_next * next_outer_angle.sin(),
            0.0,
        ];

        // Triangle: center → outer tip → inner indentation
        verts.extend_from_slice(&[center, outer, inner]);
        // Triangle: center → inner indentation → next outer tip
        verts.extend_from_slice(&[center, inner, next_outer]);
    }

    verts
}

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