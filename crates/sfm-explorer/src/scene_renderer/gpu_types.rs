// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU data types, constants, and small helpers for the scene renderer.

use nalgebra::{Matrix4, Vector3};

// ── GPU data types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct QuadVertex {
    pub(crate) position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PointInstance {
    pub(crate) position: [f32; 3],
    pub(crate) color: u32, // packed R8G8B8A8
}

/// Per-reconstruction uniform block — one buffer per loaded node.
///
/// Everything a shader needs to know about *which* reconstruction it is
/// drawing. Its WGSL twin is `struct ReconUniforms` in `shaders/common.wgsl`.
/// See `specs/gui/scene-graph.md` ("Per-recon uniforms").
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReconUniforms {
    /// The node's similarity transform (`SceneNode::transform`), applied before
    /// `view_proj` in every scene vertex shader. Identity until the Scene
    /// panel's `Align to…` fits one.
    pub(crate) model: [[f32; 4]; 4],
    /// This node's auto point size × the global `2^point_size_log2`.
    pub(crate) point_size: f32,
    /// Start of this node's slice of the global point index space.
    pub(crate) point_pick_base: u32,
    /// Start of this node's slice of the global image index space.
    pub(crate) image_pick_base: u32,
    /// 0 → emit `PICK_TAG_NONE` instead of a pick id (a display-only node,
    /// i.e. the Scene panel's interaction cursor switched off).
    pub(crate) pickable: u32,
    /// The node's comparison tint: the palette color in `rgb`, the strength it
    /// is mixed in at in `a`. Every scene shader composites the same way —
    /// `mix(color, tint_color.rgb, tint_color.a)` — so `a == 0` draws the
    /// node's original colors. Highlight colors are exempt; see the shaders.
    pub(crate) tint_color: [f32; 4],
    /// 1.0 to draw this node's points at infinity, 0.0 to cull them in the
    /// vertex shader — the global HUD toggle AND the node's ∞ mini-toggle.
    /// A uniform rather than a filtered upload because `instance_index` has to
    /// stay equal to the local `recon.points` index for picking to work.
    pub(crate) show_infinity: f32,
    /// The camera model proposal's preview blend on this node: 0 draws the
    /// stored positions, 1 the targets. Read only by the preview variants of
    /// the point and patch pipelines, so 0 on every other node.
    pub(crate) preview_s: f32,
    /// How a point whose mask word carries `MASK_REJECTED` is drawn:
    /// `FILTER_DIM` or `FILTER_HIDE` (`shaders/common.wgsl`). Only read where a
    /// word carries the bit, which is only on the node the Track Table filters.
    pub(crate) filter_mode: u32,
    /// 1 when the patch pass draws this node's surfels
    /// (`NodeDisplay::points_as_patches`): the point shader then leaves out the
    /// splat of every point whose mask word carries `MASK_HAS_SURFEL`, so each
    /// point is drawn once, as its surfel or as its dot. 0 draws every point as
    /// a splat.
    pub(crate) points_as_patches: u32,
}

/// [`ReconUniforms::filter_mode`] for a node whose rejected points are dimmed:
/// drawn at the filter's dim alpha by a second draw after the kept ones.
/// Matches `FILTER_DIM` in `shaders/common.wgsl`.
pub(super) const FILTER_DIM: u32 = 0;
/// [`ReconUniforms::filter_mode`] for a node whose rejected points are hidden:
/// a clipped vertex, so they draw nothing and answer no pick. Matches
/// `FILTER_HIDE` in `shaders/common.wgsl`.
pub(super) const FILTER_HIDE: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PointUniforms {
    pub(crate) view_proj: [[f32; 4]; 4], // column-major
    pub(crate) view: [[f32; 4]; 4],      // view matrix for linear depth
    pub(crate) camera_right: [f32; 3],
    /// Padding where the point size used to live: splat size is per-recon now
    /// and travels in [`ReconUniforms`].
    pub(crate) _pad0: f32,
    pub(crate) camera_up: [f32; 3],
    /// Global pick index of the selected point (0xFFFFFFFF = no selection).
    pub(crate) selected_point_index: u32,
    /// Global pick index of the hovered point (0xFFFFFFFF = no hover).
    pub(crate) hovered_point_index: u32,
    /// Viewport size in pixels — converts the infinity splat pixel radius to NDC.
    pub(crate) screen_width: f32,
    pub(crate) screen_height: f32,
    /// On-screen splat radius (pixels) for points at infinity.
    pub(crate) infinity_point_px: f32,
    pub(crate) _pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EdgeInstance {
    pub(crate) endpoint_a: [f32; 3],
    pub(crate) endpoint_b: [f32; 3],
}

/// Edge instance with per-endpoint width for tapered compass spikes.
/// The w component of each endpoint is a width multiplier (0.0 = point, 1.0 = full width).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CompassEdgeInstance {
    pub(crate) endpoint_a: [f32; 4],
    pub(crate) endpoint_b: [f32; 4],
}

/// One piece of an overlay figure, as the
/// [overlay ribbon store](super::overlay_ribbons::OverlayRibbons) holds it: a
/// segment, a ring or a disc of the bench figure or of a drawn measurement.
///
/// Nothing in it depends on the camera. Its places are homogeneous world
/// points, `w = 1.0` for a place and `w = 0.0` for a direction, which is drawn
/// at infinity, rotation-only, as a point at infinity is. Its sizes are in
/// panel points, which the shader turns into pixels with the frame's
/// `pixels_per_point`, and the pull toward the eye a flagged piece is drawn
/// with is the frame's too ([`BenchTrackUniforms`]). So a producer builds its
/// instances once from what it draws, and a camera move, a zoom or a resize
/// writes nothing.
///
/// Wider than [`CompassEdgeInstance`] because the compass is one object in one
/// colour, while this pass's pieces each carry their own: an observation's
/// verdict colour, or a measurement's colour and its halo.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayInstance {
    /// A segment's one end; a ring's or a disc's centre.
    pub(crate) a: [f32; 4],
    /// A segment's other end; unused by a ring or a disc.
    pub(crate) b: [f32; 4],
    /// Not premultiplied, in the colour space of the target it is drawn into.
    pub(crate) color: [f32; 4],
    /// Half the stroke's width, in panel points; 0 for a disc.
    pub(crate) half_width: f32,
    /// A ring's or a disc's radius, in panel points; 0 for a segment.
    pub(crate) radius: f32,
    /// [`Self::SEGMENT`], [`Self::RING`] or [`Self::DISC`].
    pub(crate) kind: u32,
    /// How its part behind the scene is drawn, [`Self::HOLD`] or
    /// [`Self::FADE`], with [`Self::PULL_TOWARD_EYE`] or not.
    pub(crate) flags: u32,
    /// How many panel points in from `a` and from `b` a segment starts and
    /// stops on screen ([`Self::inset`]); `[0, 0]` for a ring or a disc.
    pub(crate) inset: [f32; 2],
}

impl OverlayInstance {
    /// A straight stroke from `a` to `b`, a ribbon facing the eye.
    pub(crate) const SEGMENT: u32 = 0;
    /// A hollow circle facing the eye, `radius` points from its centre to the
    /// middle of its stroke.
    pub(crate) const RING: u32 = 1;
    /// A filled circle facing the eye, `radius` points from its centre to its
    /// edge.
    pub(crate) const DISC: u32 = 2;

    /// Behind the scene it is held at `BEHIND_OPACITY`, however far behind,
    /// as a measurement is and as a highlighted surfel's hidden outline is.
    pub(crate) const HOLD: u32 = 0;
    /// Behind the scene it fades with how far behind it is, down to a floor,
    /// as the bench figure does.
    pub(crate) const FADE: u32 = 1;
    /// It is drawn toward the eye by the frame's pull fraction, along its own
    /// line of sight, so it projects where it would without the pull.
    pub(crate) const PULL_TOWARD_EYE: u32 = 2;

    /// A segment from `a` to `b`, `half_width` points to either side.
    pub(crate) fn segment(
        a: [f32; 4],
        b: [f32; 4],
        color: [f32; 4],
        half_width: f32,
        flags: u32,
    ) -> Self {
        Self {
            a,
            b,
            color,
            half_width,
            radius: 0.0,
            kind: Self::SEGMENT,
            flags,
            inset: [0.0; 2],
        }
    }

    /// This segment, started `at_a` points in from `a` and stopped `at_b`
    /// points short of `b` along its projection on screen, stroke and halo
    /// alike, with each cut end anti-aliased as its sides are.
    ///
    /// A measurement's line stops at the outer edge of the ring at each of
    /// its ends this way, so it draws neither over that ring's stroke nor
    /// inside it. An end the pass clips at the near plane is not inset, since
    /// the place it was inset from is not drawn, and a segment shorter on
    /// screen than its two insets together draws nothing.
    pub(crate) fn inset(self, at_a: f32, at_b: f32) -> Self {
        Self {
            inset: [at_a, at_b],
            ..self
        }
    }

    /// A ring about `centre`, `radius` points to the middle of a stroke
    /// `half_width` points to either side.
    pub(crate) fn ring(
        centre: [f32; 4],
        color: [f32; 4],
        radius: f32,
        half_width: f32,
        flags: u32,
    ) -> Self {
        Self {
            a: centre,
            b: [0.0; 4],
            color,
            half_width,
            radius,
            kind: Self::RING,
            flags,
            inset: [0.0; 2],
        }
    }

    /// A disc about `centre`, of radius `radius` points.
    pub(crate) fn disc(centre: [f32; 4], color: [f32; 4], radius: f32, flags: u32) -> Self {
        Self {
            a: centre,
            b: [0.0; 4],
            color,
            half_width: 0.0,
            radius,
            kind: Self::DISC,
            flags,
            inset: [0.0; 2],
        }
    }
}

/// One vertex of the bench figure's filled centre disc.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BenchDiscVertex {
    pub(crate) position: [f32; 4],
    pub(crate) color: [f32; 4],
}

/// Uniforms for the bench track pass.
///
/// [`FrustumUniforms`]'s camera block, with the hover index it has no use for
/// replaced by what this pass needs that no other does: the projection on its
/// own, which a piece drawn toward the eye is projected through after the pull;
/// how far behind the scene a fragment fades to the floor opacity; how many
/// pixels a panel point is; and the pull toward the eye. The widths and radii
/// are each instance's own, in points ([`OverlayInstance`]).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BenchTrackUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) proj: [[f32; 4]; 4],
    pub(crate) screen_size: [f32; 2],
    /// Near clip distance, which the shader clips segments against and reads
    /// the scene's own view-space depth back out of the reversed-Z buffer with.
    pub(crate) near: f32,
    /// World units, the figure's own `4h`.
    pub(crate) fog_distance: f32,
    /// Physical pixels to a panel point, which turns the instances' widths
    /// and radii into pixels.
    pub(crate) pixels_per_point: f32,
    /// What fraction of its distance to the eye a piece flagged
    /// [`OverlayInstance::PULL_TOWARD_EYE`] is drawn toward it
    /// (`viewer_3d::measurements::toward_eye_fraction`).
    pub(crate) pull_fraction: f32,
    /// Rounds the block up to the 16 bytes a uniform buffer's struct is
    /// aligned to.
    pub(crate) _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FrustumEdge {
    pub(crate) endpoint_a: [f32; 3],
    pub(crate) _pad0: u32,
    pub(crate) endpoint_b: [f32; 3],
    pub(crate) frustum_index: u32, // image index for pick buffer
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageQuadInstance {
    pub(crate) corner_tl: [f32; 3],
    pub(crate) frustum_index: u32,
    pub(crate) corner_tr: [f32; 3],
    pub(crate) _pad0: u32,
    pub(crate) corner_bl: [f32; 3],
    pub(crate) _pad1: u32,
    pub(crate) corner_br: [f32; 3],
    pub(crate) _pad2: u32,
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
    pub(crate) center: [f32; 3],
    /// Homogeneous flag: 1.0 finite, 0.0 at infinity.
    pub(crate) w: f32,
    /// World-space u axis × half-extent.
    pub(crate) u_halfvec: [f32; 3],
    pub(crate) _pad0: f32,
    /// World-space v axis × half-extent.
    pub(crate) v_halfvec: [f32; 3],
    /// Compacted cell index into the patch atlas (page decoded in the shader).
    pub(crate) atlas_layer: u32,
    /// Global `recon.points` index, for the pick buffer.
    pub(crate) point_index: u32,
}

/// One surfel slot's row of the camera model proposal's preview: the target
/// centre in `xyz` (a unit direction for a point at infinity) with the preview
/// flag in `w`, and the factor the surfel's half-vectors reach at the target.
/// See `upload::preview`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SurfelPreview {
    pub(crate) target: [f32; 4],
    pub(crate) size_factor: f32,
}

/// Uniforms for patch surfel rendering: view-projection plus the atlas grid
/// dimensions (mirroring [`ImageQuadUniforms`]) and the user patch controls.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PatchUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) atlas_cols: u32,
    pub(crate) atlas_rows: u32,
    pub(crate) patches_per_page: u32,
    /// User scale multiplier on the stored half-vecs (`2^patch_size_log2`).
    pub(crate) patch_scale: f32,
    /// Global opacity multiplier for patch color.
    pub(crate) patch_opacity: f32,
    /// Coverage discard threshold on the bitmap alpha (per-pixel confidence).
    pub(crate) alpha_cutoff: f32,
    /// The selected and hovered points' global pick indexes, as in
    /// `PointUniforms`: a surfel of either is outlined in the highlight colour,
    /// is not culled as a back face, and is drawn whatever the Track Table's
    /// filter says of its point.
    pub(crate) selected_point_index: u32,
    pub(crate) hovered_point_index: u32,
    /// Camera world position, for front-face culling of patch surfels.
    pub(crate) camera_pos: [f32; 3],
    /// Physical pixels to a panel point, which turns the highlighted surfel's
    /// outline width into pixels. It fills the vec3's padding.
    pub(crate) pixels_per_point: f32,
}

/// Vertex for a tessellated (distorted) image quad.
///
/// All cameras' tessellated meshes are concatenated into a single vertex buffer
/// and index buffer. The position and UV are pre-computed on CPU.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DistortedQuadVertex {
    pub(crate) position: [f32; 3],
    pub(crate) frustum_index: u32,
    pub(crate) uv: [f32; 2],
    pub(crate) _pad: [f32; 2],
}

/// Vertex for the tessellated (distorted) background image mesh.
///
/// Positions are unit ray directions in the SfM camera's local space,
/// placed on the unit sphere via [`sfmtool_core::CameraIntrinsics::pixel_to_ray`].
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BgDistortedVertex {
    pub(crate) position: [f32; 3],
    pub(crate) uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FrustumUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) screen_size: [f32; 2],
    pub(crate) line_half_width: f32,
    /// Global pick index of the hovered image (0xFFFFFFFF = no hover).
    pub(crate) hovered_image_index: u32,
    /// Near clip plane distance. Used by the shader to clip line segments
    /// in view space before the manual perspective divide, so endpoints
    /// behind the camera don't flip sign and launch ribbons across the screen.
    pub(crate) near: f32,
    pub(crate) _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TargetIndicatorUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) target_pos_radius: [f32; 4], // xyz = target pos, w = radius
    pub(crate) indicator_rot_0: [f32; 4],   // rotation matrix column 0 (xyz), w = alpha_scale
    pub(crate) indicator_rot_1: [f32; 4],   // rotation matrix column 1 (xyz), w = fog_distance
    pub(crate) indicator_rot_2: [f32; 4],   // rotation matrix column 2 (xyz), w = unused
    pub(crate) screen_size_ps: [f32; 4],    // xy = screen_size, z = point_size, w = line_half_width
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EdlUniforms {
    pub(crate) screen_size: [f32; 2],
    pub(crate) radius: f32,
    pub(crate) strength: f32,
    pub(crate) opacity: f32,
    pub(crate) point_size: f32,
    pub(crate) target_view_pos: [f32; 2], // xy of target in view space
    pub(crate) target_view_z: f32,        // z of target in view space (positive = in front)
    pub(crate) target_active: f32,
    pub(crate) tan_half_fov: f32,
    pub(crate) aspect: f32,
    pub(crate) target_radius: f32,
    pub(crate) time: f32,
    pub(crate) _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BgImageUniforms {
    /// Projection matrix for the BG mesh (camera-space → clip-space).
    pub(crate) view_proj: [[f32; 4]; 4],
    /// The owning node's transform, as the same `model` matrix every other
    /// pass gets. The mesh's ray directions are in the node's own coordinates,
    /// so an aligned node's background image only faces the way its cameras do
    /// once this has been applied.
    pub(crate) model: [[f32; 4]; 4],
}

// ── Related constants ────────────────────────────────────────────────────

/// Fallback point size when fewer than 2 points are loaded.
pub(super) use sfmtool_core::analysis::scene_scale::FALLBACK_POINT_SIZE;

/// Default length scale multiplier (length_scale = multiplier * point_size),
/// the one `sfmtool-core` measures a node's length scale with.
pub(crate) const DEFAULT_LENGTH_SCALE_MULTIPLIER: f32 =
    sfmtool_core::analysis::scene_scale::LENGTH_SCALE_MULTIPLIER;

/// Default target indicator size multiplier (radius = multiplier * length_scale).
pub(crate) const DEFAULT_TARGET_SIZE_MULTIPLIER: f32 = 0.3;

/// Default target indicator fog multiplier. The fog distance is the reversed-Z
/// NDC depth difference `0.1 * multiplier / DEFAULT_TARGET_FOG_MULTIPLIER`.
pub(crate) const DEFAULT_TARGET_FOG_MULTIPLIER: f32 = 10.0;

/// Default frustum stub depth as a fraction of `length_scale`.
pub(crate) const DEFAULT_FRUSTUM_SIZE_MULTIPLIER: f32 = 0.5;

/// Half-width of indicator lines in pixels.
pub(super) const INDICATOR_LINE_HALF_WIDTH: f32 = 2.0;

/// Half-width of frustum lines in pixels.
pub(super) const FRUSTUM_LINE_HALF_WIDTH: f32 = 1.0;

/// Half-width of the bench figure's lines, in panel points.
///
/// Half of the 2.5-point stroke the Image Detail panel's bench layer draws the
/// same track with, so the figure reads as the same layer in both panels.
pub(super) const BENCH_LINE_HALF_WIDTH: f32 = 1.25;

/// Uniforms for image quad / distorted quad thumbnail rendering.
///
/// Contains the view-projection matrix plus atlas grid dimensions so the
/// shader can compute UV coordinates into the 2D texture array atlas.
/// Each layer of the array holds one page of the grid.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageQuadUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) atlas_cols: u32,
    pub(crate) atlas_rows: u32,
    pub(crate) images_per_page: u32,
    pub(crate) _pad: u32,
}

/// Thumbnail dimensions for the texture atlas (square, non-square pixels).
///
/// Source images are stretched to this fixed square size regardless of their
/// native aspect ratio. The distortion is compensated on the GPU: each
/// frustum quad has the correct aspect ratio from camera intrinsics, and the
/// UV 0→1 mapping stretches the square texture back to the original proportions.
/// The value is the `.sfmr` format's own [`sfmtool_core::THUMBNAIL_SIZE`]: the
/// atlas cell holds a thumbnail read straight out of the reconstruction, and
/// [`super::upload`] uses this same constant for the cell origin, the row
/// stride and the copy extent. A cell that disagreed with the stored extent
/// would upload skewed rows, so it is derived rather than declared again.
pub(super) const THUMBNAIL_SIZE: u32 = sfmtool_core::THUMBNAIL_SIZE as u32;

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

// ── The scene render-target contract ─────────────────────────────────────
//
// Pass 1 draws every piece of scene geometry into a three-attachment G-buffer
// under a reversed-Z hardware depth buffer; pass 2 resolves that into the EDL
// output texture egui displays. The formats below are the single declaration of
// that contract: `super::sizing` allocates the textures from them and every
// pipeline in `super::pipelines` declares its targets from them, so a change
// here moves the producers and the consumer together. Spelling a format out at
// a `create_render_pipeline` call site instead is how one pipeline ends up
// disagreeing with the pass it is drawn in — which wgpu reports as a validation
// error at `set_pipeline`, i.e. at the first frame, on whatever machine happens
// to run it. `the_gbuffer_pipelines_match_the_textures_sizing_allocates` in
// `super::pipelines::tests` is the mechanical version of that statement.

/// `@location(0)` of the scene G-buffer — the visible colour, sRGB-encoded by
/// the hardware on write, sampled by the pass-2 EDL resolve.
pub(super) const GBUFFER_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The pass-2 output texture: what EDL resolves into, what the pass-2 overlays
/// (target indicator, track rays, background mesh) draw on top of, and what
/// egui displays. Deliberately the same encoding as [`GBUFFER_COLOR_FORMAT`] —
/// EDL samples that texture and writes this one, and a pair that disagreed
/// would shift the whole viewport's gamma rather than fail validation. The
/// second, `Rgba8Unorm` view `super::sizing` hands egui is a re-interpretation
/// of these same bytes, not a different texture; the comment there says why.
pub(super) const EDL_OUTPUT_FORMAT: wgpu::TextureFormat = GBUFFER_COLOR_FORMAT;

/// The format of the measurements' layer: sRGB-encoded values with no
/// conversion, as egui's own textures hold them. The measurements' pipeline
/// draws into the layer and egui paints it over the scene texture, so a
/// translucent mark or line composites on encoded values as egui's own
/// painting does and shows as egui would have painted it.
pub(super) const MEASUREMENT_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// `@location(1)` — positive view-space depth, read by the EDL pass for its
/// edge term and copied back to the CPU for Alt+click depth queries. Geometry
/// that should not take EDL shading (frustums, image quads, patches) writes
/// `0.0` here rather than its real depth; see `specs/gui/camera-views.md`.
pub(super) const GBUFFER_LINEAR_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// `@location(2)` — the pick id, an entity tag packed with an index. Integer,
/// so it is never blended or filtered; see `specs/gui/scene-graph.md`.
pub(super) const GBUFFER_PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// The hardware depth buffer every pass-1 pipeline shares, so that points,
/// frustums, image quads and patches occlude each other correctly.
pub(super) const HW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The reversed-Z depth state shared by every pipeline that draws into the
/// G-buffer: `Greater` against a buffer cleared to `0.0`, which is the far
/// plane under the infinite reversed-Z projection
/// (`specs/gui/adaptive-clip-and-grid.md`). Both sides have to agree — a
/// pipeline left on the `Less` default draws nothing and reports nothing.
///
/// `stencil` and `bias` are the `Default::default()` the pipelines used before
/// they shared this, written out because a `const` cannot call `default()`.
pub(super) const GBUFFER_DEPTH_STATE: wgpu::DepthStencilState = wgpu::DepthStencilState {
    format: HW_DEPTH_FORMAT,
    depth_write_enabled: Some(true),
    depth_compare: Some(wgpu::CompareFunction::Greater),
    stencil: wgpu::StencilState {
        front: wgpu::StencilFaceState::IGNORE,
        back: wgpu::StencilFaceState::IGNORE,
        read_mask: 0,
        write_mask: 0,
    },
    bias: wgpu::DepthBiasState {
        constant: 0,
        slope_scale: 0.0,
        clamp: 0.0,
    },
};

/// The depth state of the hidden-outline pass, which draws the part of a
/// highlighted surfel's outline the scene hides: the scene's depth, read and
/// never written, compared `Less`, the strict reverse of
/// [`GBUFFER_DEPTH_STATE`]'s `Greater`. Under reversed-Z a fragment passes only
/// where something nearer than it was drawn, and fails where it lands on the
/// depth its own surfel wrote.
pub(super) const HIDDEN_OUTLINE_DEPTH_STATE: wgpu::DepthStencilState = wgpu::DepthStencilState {
    depth_write_enabled: Some(false),
    depth_compare: Some(wgpu::CompareFunction::Less),
    ..GBUFFER_DEPTH_STATE
};

/// The three colour targets of the G-buffer, in attachment order.
///
/// How the colour attachment blends is the one thing the pass-1 pipelines
/// legitimately disagree about — splats and frustum lines composite with
/// premultiplied alpha, opaque image quads overwrite — so it is the parameter.
/// The depth-readback and pick attachments never blend: both carry values that
/// are meaningless when mixed, a view-space distance and a packed integer id.
pub(super) const fn gbuffer_targets(
    color_blend: Option<wgpu::BlendState>,
) -> [Option<wgpu::ColorTargetState>; 3] {
    [
        Some(wgpu::ColorTargetState {
            format: GBUFFER_COLOR_FORMAT,
            blend: color_blend,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: GBUFFER_LINEAR_DEPTH_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: GBUFFER_PICK_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ]
}

/// One corner of the quad the point and patch pipelines expand each instance
/// into, with the draw it belongs to: `0` for the first draw, `1` for the
/// Track Table filter's dimmed draw (`render::DIMMED_DRAW`).
///
/// The draw is vertex data rather than read off `vertex_index`, because a
/// backend need not add the draw's first vertex to `vertex_index` (DX12's
/// `SV_VertexID` does not), while every backend fetches vertex data from it.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SplatCorner {
    pub(crate) position: [f32; 2],
    pub(crate) dimmed: u32,
}

/// The splat corner buffer's contents: the unit quad's four corners as a
/// triangle strip for the first draw, then the same four for the dimmed draw.
pub(super) const SPLAT_CORNERS: [SplatCorner; 8] = {
    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
    let mut corners = [SplatCorner {
        position: [0.0; 2],
        dimmed: 0,
    }; 8];
    let mut i = 0;
    while i < 8 {
        corners[i] = SplatCorner {
            position: CORNERS[i % 4],
            dimmed: (i / 4) as u32,
        };
        i += 1;
    }
    corners
};

/// The shader location of [`SplatCorner::dimmed`] in `points.wgsl` and
/// `patch.wgsl`, the last one a pipeline may use, clear of both shaders'
/// instance attributes.
pub(super) const DIMMED_DRAW_LOCATION: u32 = 15;

/// The vertex layout of [`SplatCorner`], slot 0 of the point and patch
/// pipelines.
pub(super) const SPLAT_CORNER_LAYOUT: wgpu::VertexBufferLayout<'static> =
    wgpu::VertexBufferLayout {
        array_stride: size_of::<SplatCorner>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 8,
                shader_location: DIMMED_DRAW_LOCATION,
            },
        ],
    };

/// Slot 0 of the instanced scene pipelines other than the point and patch
/// ones: the static unit quad in `PointPipelineResources::quad_vertex_buffer`,
/// which frustum edges, image quads, the target indicator, track rays and the
/// bench track expand their instances from. One buffer, so one layout — the
/// stride is [`QuadVertex`]'s and the pipelines do not get a say in it. The
/// point and patch pipelines read [`SPLAT_CORNER_LAYOUT`] instead.
pub(super) const QUAD_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: size_of::<QuadVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    }],
};

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
