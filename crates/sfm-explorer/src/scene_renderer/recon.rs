// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-reconstruction GPU resources.
//!
//! Everything that belongs to *one* loaded reconstruction lives in a
//! [`ReconResources`] bundle keyed by [`ReconId`](crate::scene::ReconId) on the
//! renderer; everything shared — pipelines, render targets, samplers, the unit
//! quad, the EDL / target-indicator / track-ray / background-image resources —
//! stays a singleton on [`SceneRenderer`](super::SceneRenderer).
//!
//! Loading a node builds one bundle; closing a node drops one bundle. No other
//! node's GPU data is touched, which is why these are per-recon buffers rather
//! than one concatenated buffer that a membership change would have to rebuild.
//! See `specs/gui/scene-graph.md` ("Rendering: Per-Reconstruction GPU
//! Resources").

use nalgebra::Point3;
use sfmtool_core::Se3Transform;

use super::gpu_types::FALLBACK_POINT_SIZE;
use crate::scene::NodeTint;

/// The patch (surfel) half of a bundle: present only when the reconstruction
/// carries patch frames *and* bitmaps, so its fields need no individual
/// `Option`s.
pub(super) struct PatchResources {
    pub instance_buffer: wgpu::Buffer,
    /// The atlas itself. Nothing reads it after the bind group is built — it is
    /// held so the node *owns* its atlas: dropping the bundle is what returns
    /// that GPU memory, which is the whole point of per-node resources.
    #[allow(dead_code)]
    pub atlas_texture: wgpu::Texture,
    /// Per-recon `PatchUniforms` — the atlas grid it carries is per-recon, so
    /// the buffer is too, even though the camera half of it is global.
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub count: u32,
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    pub patches_per_page: u32,
}

/// One node's display state as the renderer needs it: which layers to draw,
/// and whether the node captures picks.
///
/// Mirrored from [`SceneNode`](crate::scene::SceneNode) once per frame rather
/// than looked up per draw, so the draw loop and the uniform write — which see
/// only bundles — need no access to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDisplay {
    /// **Effective** whole-node visibility: the node's master eye composed with
    /// the scene's solo override (`crate::scene::is_visible`). Off = the node
    /// contributes nothing to any pass, and nothing to the scene bounds.
    ///
    /// Composed once, by `app.rs`'s per-frame mirror, so the draw loop, the
    /// bounds union and the per-recon uniform write cannot disagree about what
    /// "visible" means.
    pub visible: bool,
    pub show_points: bool,
    /// Group eye: the node's camera frustums and image quads.
    pub show_camera_images: bool,
    pub show_patches: bool,
    /// Sub-toggle of `show_points`, applied in the point shader so instance
    /// indices (and therefore pick ids) stay unfiltered.
    pub show_points_at_infinity: bool,
    /// Off → the node's shaders emit `PICK_TAG_NONE`. It still renders, still
    /// occludes, and still answers the depth readback.
    pub interactive: bool,
    /// The node's comparison tint, mixed into its colors by every scene shader.
    pub tint: NodeTint,
}

impl Default for NodeDisplay {
    /// Everything on: what a freshly loaded node shows before the Scene panel
    /// has had a frame to say otherwise.
    fn default() -> Self {
        Self {
            visible: true,
            show_points: true,
            show_camera_images: true,
            show_patches: true,
            show_points_at_infinity: true,
            interactive: true,
            tint: NodeTint::Original,
        }
    }
}

/// One loaded reconstruction's GPU resources and derived scalars.
pub(super) struct ReconResources {
    /// This node's display state, refreshed every frame from its scene node.
    pub display: NodeDisplay,

    /// This node's similarity transform into the shared world space, mirrored
    /// from its scene node alongside `display`.
    ///
    /// Kept as the `Se3Transform` rather than only as the `model` matrix
    /// because two CPU-side consumers need the pieces: the bounding sphere
    /// (centre through the transform, radius times the scale) and the splat
    /// size (scaled with the node).
    pub transform: Se3Transform,

    /// This node's `ReconUniforms` slice: model matrix, point size, pick bases,
    /// pickable flag, tint. Written every frame by `update_uniforms`.
    pub uniform_buffer: wgpu::Buffer,

    // ── points ──
    pub point_instance_buffer: Option<wgpu::Buffer>,
    pub point_count: u32,
    /// Global point uniforms + this node's `ReconUniforms`.
    pub point_bind_group: wgpu::BindGroup,

    // ── frustums + image quads ──
    pub frustum_edge_buffer: Option<wgpu::Buffer>,
    pub frustum_edge_count: u32,
    pub frustum_image_count: u32,
    /// Per-image ABGR, cheap write path for selection/hover recolouring.
    pub frustum_color_buffer: Option<wgpu::Buffer>,
    pub frustum_bind_group: Option<wgpu::BindGroup>,
    pub image_quad_instance_buffer: Option<wgpu::Buffer>,
    pub image_quad_count: u32,
    pub distorted_quad_vertex_buffer: Option<wgpu::Buffer>,
    pub distorted_quad_index_buffer: Option<wgpu::Buffer>,
    pub distorted_quad_index_count: u32,

    // ── thumbnails: per-recon atlas + bind group ──
    pub thumbnail_texture: Option<wgpu::Texture>,
    pub thumbnail_view: Option<wgpu::TextureView>,
    /// Per-recon `ImageQuadUniforms` (view-projection + this atlas's grid).
    pub image_quad_uniform_buffer: Option<wgpu::Buffer>,
    /// Shared by the pinhole and distorted image-quad pipelines.
    pub image_quad_bind_group: Option<wgpu::BindGroup>,
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    pub images_per_page: u32,

    // ── patches (optional) ──
    pub patch: Option<PatchResources>,

    // ── per-recon derived scalars (formerly singletons on SceneRenderer) ──
    /// Auto-computed splat size (world space, before the global user scaling).
    pub auto_point_size: f32,
    /// Characteristic inter-camera distance (p90 of camera-centre NN
    /// distances), or `None` with fewer than 2 cameras.
    pub camera_nn_scale: Option<f32>,
    /// Bounding sphere (centre, radius) of this node's finite points, in the
    /// node's own coordinates. `None` until points have been uploaded, so an
    /// empty bundle cannot drag the union bounds toward the origin.
    pub bounds: Option<(Point3<f64>, f64)>,

    // ── pick bases (see `super::picking`) ──
    pub point_pick_base: u32,
    pub image_pick_base: u32,
}

impl ReconResources {
    /// An empty bundle: its uniform buffer and point bind group exist from the
    /// start (both are needed before any data is uploaded), everything else
    /// arrives with the corresponding upload.
    pub(super) fn new(
        device: &wgpu::Device,
        point_bind_group_layout: &wgpu::BindGroupLayout,
        point_uniform_buffer: &wgpu::Buffer,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("recon uniforms"),
            size: std::mem::size_of::<super::gpu_types::ReconUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let point_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point bind group"),
            layout: point_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: point_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            display: NodeDisplay::default(),
            transform: Se3Transform::identity(),
            uniform_buffer,
            point_instance_buffer: None,
            point_count: 0,
            point_bind_group,
            frustum_edge_buffer: None,
            frustum_edge_count: 0,
            frustum_image_count: 0,
            frustum_color_buffer: None,
            frustum_bind_group: None,
            image_quad_instance_buffer: None,
            image_quad_count: 0,
            distorted_quad_vertex_buffer: None,
            distorted_quad_index_buffer: None,
            distorted_quad_index_count: 0,
            thumbnail_texture: None,
            thumbnail_view: None,
            image_quad_uniform_buffer: None,
            image_quad_bind_group: None,
            atlas_cols: 0,
            atlas_rows: 0,
            images_per_page: 0,
            patch: None,
            auto_point_size: FALLBACK_POINT_SIZE,
            camera_nn_scale: None,
            bounds: None,
            point_pick_base: 0,
            image_pick_base: 0,
        }
    }

    /// The seed this node contributes to the global `length_scale`: the point
    /// splat scale, capped by the inter-camera distance when there is one.
    ///
    /// Both inputs are measured in the node's own coordinates, so the node
    /// transform's scale converts them to the world-space quantity
    /// `length_scale` actually is. That is what lets a scaled node stop
    /// dominating the global frustum size once it has been aligned.
    pub(super) fn length_scale_seed(&self) -> f32 {
        let point_scale = super::DEFAULT_LENGTH_SCALE_MULTIPLIER * self.auto_point_size;
        let seed = match self.camera_nn_scale {
            Some(camera_scale) => point_scale.min(camera_scale),
            None => point_scale,
        };
        seed * self.transform.scale as f32
    }

    /// This node's bounding sphere in the **shared world space**: its own
    /// bounds put through the node transform. `None` until its points are
    /// uploaded.
    pub(super) fn world_bounds(&self) -> Option<(Point3<f64>, f64)> {
        let (centre, radius) = self.bounds?;
        Some((
            self.transform.apply_to_point(&centre),
            radius * self.transform.scale,
        ))
    }
}
