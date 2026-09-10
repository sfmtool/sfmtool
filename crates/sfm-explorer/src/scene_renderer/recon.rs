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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nalgebra::Point3;
use sfmtool_core::{Se3Transform, SfmrReconstruction};

use super::gpu_types::FALLBACK_POINT_SIZE;
use crate::scene::NodeTint;

/// The patch (surfel) half of a bundle: present only when the reconstruction
/// carries patch frames *and* bitmaps, so its fields need no individual
/// `Option`s.
pub(super) struct PatchResources {
    pub instance_buffer: wgpu::Buffer,
    /// Per-patch liveness, `1` alive and `0` deleted, as a second instance
    /// buffer stepping with `instance_buffer`. Rewritten entry by entry when
    /// the overlay's deleted set changes; never rebuilt for a point edit.
    pub alive_buffer: wgpu::Buffer,
    /// Which instance slot each point that carries a patch was packed into.
    /// The atlas and the instance buffer are compacted, so a patch's slot is
    /// not its point index, and this is what turns a deleted point index into
    /// the entry of `alive_buffer` to clear.
    pub slot_of_point: HashMap<u32, u32>,
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

/// The overlay's **additions** as GPU state: a second set of instance buffers
/// drawn after the base's, in the same passes and with the same global
/// uniforms.
///
/// The base's buffers are what a run of point edits shares, so they must not be
/// rebuilt when the overlay gains a point. The additions get their own buffers
/// instead, sized to the addition set alone. It carries its own copy of the
/// node's `ReconUniforms` for one field: the additions' pick base is the node's
/// plus its base point count, which is what makes an addition's global pick id
/// still `point_pick_base + edited index`.
///
/// Patches likewise get a **second, small atlas** rather than an appended
/// slot range in the base's. The base's atlas texture and bind group then keep
/// their identity across every point edit, and the packing arithmetic is the
/// same code over a shorter list. The two atlases are drawn as two draws in one
/// pass; nothing about the shader distinguishes them.
pub(super) struct AdditionResources {
    /// This node's `ReconUniforms` with the additions' pick base.
    pub uniform_buffer: wgpu::Buffer,
    /// The point bind group over `uniform_buffer`.
    pub bind_group: wgpu::BindGroup,
    pub point_instance_buffer: wgpu::Buffer,
    /// Per-addition liveness, as for the base's.
    pub point_alive_buffer: wgpu::Buffer,
    pub point_count: u32,
    /// The additions' surfels and their own atlas, when any addition carries a
    /// patch bitmap.
    pub patch: Option<PatchResources>,
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

    /// The base this bundle's buffers were built from, or `None` before the
    /// first upload.
    ///
    /// The whole of the renderer's change detection: an upload runs when the
    /// node's base is a different `Arc` from this one, and a run of point edits
    /// -- which never writes through the base -- leaves it alone and re-uploads
    /// nothing. See `specs/gui/document-model.md`.
    pub uploaded_base: Option<Arc<SfmrReconstruction>>,

    /// Which base indexes the deleted mask below currently marks as gone. What
    /// a mask update diffs against, so the write is the size of the change.
    pub masked_deleted: HashSet<u32>,

    /// Which indexes the mask currently marks as highlighted -- the viewport
    /// calling a set of points out, which today is the points a camera being
    /// moved observes. Diffed the same way, and for the same reason.
    pub masked_highlighted: HashSet<u32>,

    // ── points ──
    pub point_instance_buffer: Option<wgpu::Buffer>,
    /// Per-point liveness, `1` alive and `0` deleted, as a second instance
    /// buffer stepping with `point_instance_buffer`. The overlay's deleted set
    /// reaches the point shader through this and nothing else.
    pub point_alive_buffer: Option<wgpu::Buffer>,
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

    // ── the overlay's additions (optional) ──
    /// The additions' instance buffers and their own patch atlas, when this
    /// version holds any.
    pub additions: Option<AdditionResources>,
    /// The addition set those buffers were built from, as `(points, tracks)`.
    ///
    /// A complete change signal, and O(1): additions are append-only along a
    /// node's history, and the history is linear because a new edit discards
    /// the redo tail. So two versions of one base whose addition sets have the
    /// same point and observation counts hold the same additions, and a count
    /// that moved is the only way the set can differ.
    pub uploaded_additions: (usize, usize),

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
            uploaded_base: None,
            masked_deleted: HashSet::new(),
            masked_highlighted: HashSet::new(),
            point_instance_buffer: None,
            point_alive_buffer: None,
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
            additions: None,
            uploaded_additions: (0, 0),
            auto_point_size: FALLBACK_POINT_SIZE,
            camera_nn_scale: None,
            bounds: None,
            point_pick_base: 0,
            image_pick_base: 0,
        }
    }

    /// How many addition instances this bundle draws.
    pub(super) fn addition_count(&self) -> u32 {
        self.additions.as_ref().map_or(0, |a| a.point_count)
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
