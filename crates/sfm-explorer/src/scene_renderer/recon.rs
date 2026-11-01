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
use ndarray::Array4;
use sfmtool_core::{Se3Transform, SfmrReconstruction};

use super::gpu_types::FALLBACK_POINT_SIZE;
use crate::filter_3d::KeptPoints;
use crate::scene::NodeTint;

/// The patch (surfel) half of a bundle: present only when the reconstruction
/// carries patch frames *and* bitmaps, so its fields need no individual
/// `Option`s.
pub(super) struct PatchResources {
    pub(crate) instance_buffer: wgpu::Buffer,
    /// Per-patch mask word, the same word the point buffer holds for the
    /// patch's point, as a second instance buffer stepping with
    /// `instance_buffer`. Rewritten where the overlay's deleted set or the
    /// viewport's marks change; never rebuilt for a point edit.
    pub(crate) alive_buffer: wgpu::Buffer,
    /// Which instance slot each point that carries a patch was packed into.
    /// The atlas and the instance buffer are compacted, so a patch's slot is
    /// not its point index, and this is what turns a deleted point index into
    /// the entry of `alive_buffer` to clear.
    pub(crate) slot_of_point: HashMap<u32, u32>,
    /// The bitmap column these tiles were written from, held by pointer.
    ///
    /// What decides whether the atlas survives a new base. A bulk edit rewrites
    /// poses, positions and patch frames and leaves the pixels alone, and
    /// materialisation keeps the column's `Arc` when no edit touched a patch --
    /// so pointer equality here is an exact answer to "are these the same
    /// tiles", and the answer is yes for every edit the viewer has.
    pub(crate) uploaded_bitmaps: Arc<Array4<u8>>,
    /// The point each instance slot was packed from, in slot order.
    ///
    /// The other half of that question. The atlas is compacted over the points
    /// that carry a patch, so it is only reusable if the same points pack into
    /// the same slots; this is what that comparison reads. `slot_of_point` is
    /// this inverted, and is what the mask writer reads instead.
    pub(crate) packed_points: Vec<u32>,
    /// The atlas itself. Nothing reads it after the bind group is built — it is
    /// held so the node *owns* its atlas: dropping the bundle is what returns
    /// that GPU memory, which is the whole point of per-node resources.
    #[allow(dead_code)]
    pub(crate) atlas_texture: wgpu::Texture,
    /// Per-recon `PatchUniforms` — the atlas grid it carries is per-recon, so
    /// the buffer is too, even though the camera half of it is global.
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) count: u32,
    pub(crate) atlas_cols: u32,
    pub(crate) atlas_rows: u32,
    pub(crate) patches_per_page: u32,
    /// The liveness vector `alive_buffer` was created from, in slot order.
    ///
    /// What reached the GPU before any mask write, which is the half a headless
    /// test cannot read back: the buffer is `VERTEX | COPY_DST` and never a
    /// copy source. Kept only under `cfg(test)`, so the shipped bundle carries
    /// nothing it does not draw from.
    #[cfg(test)]
    pub(crate) built_liveness: Vec<u32>,
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
    pub(crate) uniform_buffer: wgpu::Buffer,
    /// The point bind group over `uniform_buffer`.
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) point_instance_buffer: wgpu::Buffer,
    /// Per-addition liveness, as for the base's.
    pub(crate) point_alive_buffer: wgpu::Buffer,
    pub(crate) point_count: u32,
    /// The additions' surfels and their own atlas, when any addition carries a
    /// patch bitmap.
    pub(crate) patch: Option<PatchResources>,
}

/// The camera model proposal's preview on one node: a target per row of each
/// point buffer and per slot of each surfel buffer, and the blend.
///
/// Separate buffers stepping beside the instance buffers, as the deleted mask
/// is, because the instance buffers are `VERTEX` only and are never written;
/// so ending the preview restores nothing and only drops these. Present only
/// on the node a proposal's preview is on, which is drawn with the preview
/// variants of the point and patch pipelines.
pub(super) struct PreviewResources {
    /// The proposal generation the rows were written for. A new one is one
    /// upload of each buffer.
    pub(crate) generation: u64,
    /// The blend, mirrored into the node's `ReconUniforms::preview_s`.
    pub(crate) s: f32,
    /// One `vec4` per row of the base's point buffer.
    pub(crate) points: wgpu::Buffer,
    /// One `vec4` per row of the additions' point buffer, when there are
    /// additions.
    pub(crate) additions: Option<wgpu::Buffer>,
    /// One `SurfelPreview` per slot of the base's surfel buffer, when the node
    /// has surfels.
    pub(crate) patches: Option<wgpu::Buffer>,
    /// One per slot of the additions' surfel buffer.
    pub(crate) addition_patches: Option<wgpu::Buffer>,
    /// The surfel rows as they were written, slot by slot: the half a headless
    /// test cannot read back from a `VERTEX | COPY_DST` buffer.
    #[cfg(test)]
    pub(crate) built_patches: Vec<super::gpu_types::SurfelPreview>,
}

/// One node's display state as the renderer needs it: which layers to draw,
/// and whether the node captures picks.
///
/// Mirrored from [`SceneNode`](crate::scene::SceneNode) once per frame rather
/// than looked up per draw, so the draw loop and the uniform write — which see
/// only bundles — need no access to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeDisplay {
    /// **Effective** whole-node visibility: the node's master eye composed with
    /// the scene's solo override (`crate::scene::is_visible`). Off = the node
    /// contributes nothing to any pass, and nothing to the scene bounds.
    ///
    /// Composed once, by `app.rs`'s per-frame mirror, so the draw loop, the
    /// bounds union and the per-recon uniform write cannot disagree about what
    /// "visible" means.
    pub(crate) visible: bool,
    pub(crate) show_points: bool,
    /// Group eye: the node's camera frustums and image quads.
    pub(crate) show_camera_images: bool,
    pub(crate) show_patches: bool,
    /// Sub-toggle of `show_points`, applied in the point shader so instance
    /// indices (and therefore pick ids) stay unfiltered.
    pub(crate) show_points_at_infinity: bool,
    /// Off → the node's shaders emit `PICK_TAG_NONE`. It still renders, still
    /// occludes, and still answers the depth readback.
    pub(crate) interactive: bool,
    /// The node's comparison tint, mixed into its colors by every scene shader.
    pub(crate) tint: NodeTint,
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
    pub(crate) display: NodeDisplay,

    /// This node's similarity transform into the shared world space, mirrored
    /// from its scene node alongside `display`.
    ///
    /// Kept as the `Se3Transform` rather than only as the `model` matrix
    /// because two CPU-side consumers need the pieces: the bounding sphere
    /// (centre through the transform, radius times the scale) and the splat
    /// size (scaled with the node).
    pub(crate) transform: Se3Transform,

    /// This node's `ReconUniforms` slice: model matrix, point size, pick bases,
    /// pickable flag, tint. Written every frame by `update_uniforms`.
    pub(crate) uniform_buffer: wgpu::Buffer,

    /// The base this bundle's buffers were built from, or `None` before the
    /// first upload.
    ///
    /// The whole of the renderer's change detection: an upload runs when the
    /// node's base is a different `Arc` from this one, and a run of point edits
    /// -- which never writes through the base -- leaves it alone and re-uploads
    /// nothing. See `specs/gui/document-model.md`.
    pub(crate) uploaded_base: Option<Arc<SfmrReconstruction>>,

    /// Which base indexes the deleted mask below currently marks as gone. What
    /// a mask update diffs against, so the write is the size of the change.
    pub(crate) masked_deleted: HashSet<u32>,

    /// Which indexes the mask currently marks as highlighted -- the viewport
    /// calling a set of points out, which today is the points a camera being
    /// moved observes. Diffed the same way, and for the same reason.
    pub(crate) masked_highlighted: HashSet<u32>,
    /// Which indexes the mask currently marks as painted by the eraser brush.
    /// Diffed the same way.
    pub(crate) masked_painted: HashSet<u32>,
    /// Which indexes the mask currently marks as under the eraser brush's
    /// circle. Diffed the same way.
    pub(crate) masked_under_brush: HashSet<u32>,
    /// The Track Table's kept set the mask's filter bit was last written
    /// from, or `None` while no filter is on this node. Compared by pointer:
    /// the panel builds a new set only when its matched rows change, so the
    /// same `Arc` means the same bits, and a different one is a rewrite of
    /// every word.
    pub(crate) masked_kept: Option<Arc<KeptPoints>>,
    /// Whether the filter on this node hides the points it rejects rather
    /// than dims them, mirrored into `ReconUniforms::filter_mode`. Read only
    /// while `masked_kept` is set.
    pub(crate) filter_hides: bool,

    // ── points ──
    pub(crate) point_instance_buffer: Option<wgpu::Buffer>,
    /// Per-point mask word (`0` deleted, `1` alive, `2` to `4` alive and
    /// marked; see `upload::overlay`), as a second instance buffer stepping
    /// with `point_instance_buffer`. The overlay's deleted set and the
    /// viewport's marks reach the point shader through this and nothing else.
    pub(crate) point_alive_buffer: Option<wgpu::Buffer>,
    pub(crate) point_count: u32,
    /// Global point uniforms + this node's `ReconUniforms`.
    pub(crate) point_bind_group: wgpu::BindGroup,

    // ── frustums + image quads ──
    pub(crate) frustum_edge_buffer: Option<wgpu::Buffer>,
    pub(crate) frustum_edge_count: u32,
    pub(crate) frustum_image_count: u32,
    /// Per-image ABGR, cheap write path for selection/hover recolouring.
    pub(crate) frustum_color_buffer: Option<wgpu::Buffer>,
    pub(crate) frustum_bind_group: Option<wgpu::BindGroup>,
    pub(crate) image_quad_instance_buffer: Option<wgpu::Buffer>,
    pub(crate) image_quad_count: u32,
    pub(crate) distorted_quad_vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) distorted_quad_index_buffer: Option<wgpu::Buffer>,
    pub(crate) distorted_quad_index_count: u32,

    // ── thumbnails: per-recon atlas + bind group ──
    pub(crate) thumbnail_texture: Option<wgpu::Texture>,
    pub(crate) thumbnail_view: Option<wgpu::TextureView>,
    /// What this atlas was written from: the node's display column and the
    /// value's own column, by pointer, and the image list.
    ///
    /// The atlas is a function of those alone, so an edit that leaves them alone
    /// leaves the atlas correct, and every edit that is not about the image
    /// set does. See [`PatchResources::uploaded_bitmaps`], which is the same
    /// reuse key for the other atlas.
    pub(crate) uploaded_thumbnails: Option<super::upload::UploadedThumbnails>,
    /// Per-recon `ImageQuadUniforms` (view-projection + this atlas's grid).
    pub(crate) image_quad_uniform_buffer: Option<wgpu::Buffer>,
    /// Shared by the pinhole and distorted image-quad pipelines.
    pub(crate) image_quad_bind_group: Option<wgpu::BindGroup>,
    pub(crate) atlas_cols: u32,
    pub(crate) atlas_rows: u32,
    pub(crate) images_per_page: u32,

    // ── patches (optional) ──
    pub(crate) patch: Option<PatchResources>,

    // ── the overlay's additions (optional) ──
    /// The additions' instance buffers and their own patch atlas, when this
    /// version holds any.
    pub(crate) additions: Option<AdditionResources>,
    /// The addition set those buffers were built from, as `(points, tracks)`.
    ///
    /// A complete change signal, and O(1): additions are append-only along a
    /// node's history, and the history is linear because a new edit discards
    /// the redo tail. So two versions of one base whose addition sets have the
    /// same point and observation counts hold the same additions, and a count
    /// that moved is the only way the set can differ.
    pub(crate) uploaded_additions: (usize, usize),

    // ── the camera model proposal's preview (optional) ──
    /// The preview buffers and blend, on the node a proposal's preview is on.
    pub(crate) preview: Option<PreviewResources>,

    // ── per-recon derived scalars (formerly singletons on SceneRenderer) ──
    /// Auto-computed splat size (world space, before the global user scaling).
    pub(crate) auto_point_size: f32,
    /// Characteristic inter-camera distance (p90 of camera-centre NN
    /// distances), or `None` with fewer than 2 cameras.
    pub(crate) camera_nn_scale: Option<f32>,
    /// Bounding sphere (centre, radius) of this node's finite points, in the
    /// node's own coordinates. `None` until points have been uploaded, so an
    /// empty bundle cannot drag the union bounds toward the origin.
    pub(crate) bounds: Option<(Point3<f64>, f64)>,

    // ── pick bases (see `super::picking`) ──
    pub(crate) point_pick_base: u32,
    pub(crate) image_pick_base: u32,
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
            masked_painted: HashSet::new(),
            masked_under_brush: HashSet::new(),
            masked_kept: None,
            filter_hides: false,
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
            uploaded_thumbnails: None,
            image_quad_uniform_buffer: None,
            image_quad_bind_group: None,
            atlas_cols: 0,
            atlas_rows: 0,
            images_per_page: 0,
            patch: None,
            additions: None,
            uploaded_additions: (0, 0),
            preview: None,
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

    /// The marks the mask last written for this bundle holds.
    #[cfg(test)]
    pub(super) fn masked_marks(&self) -> super::upload::PointMarks<'_> {
        super::upload::PointMarks {
            highlighted: &self.masked_highlighted,
            painted: &self.masked_painted,
            under_brush: &self.masked_under_brush,
            kept: self.masked_kept.as_ref(),
            filter_hides: self.filter_hides,
        }
    }

    /// Whether this node's points and surfels take the second, dimmed draw:
    /// a filter is on it and dims what it rejects.
    pub(super) fn draws_dimmed(&self) -> bool {
        self.masked_kept.is_some() && !self.filter_hides
    }

    /// Every surfel atlas this bundle holds, in draw order: the base's, then
    /// the overlay's additions' own.
    ///
    /// One list with two readers -- the frame's uniform write and the patch
    /// pass -- because each `PatchResources` carries its own grid dimensions in
    /// its own uniform block, and the two have to cover the same atlases. A
    /// block left unwritten is a zero one, and a zero `view_proj` collapses
    /// every corner of every surfel in that atlas to one clipped vertex: an
    /// atlas the write misses and the draw does not is an atlas that draws
    /// nothing at all.
    pub(super) fn patch_atlases(&self) -> impl Iterator<Item = &PatchResources> {
        self.patch.iter().chain(
            self.additions
                .as_ref()
                .and_then(|additions| additions.patch.as_ref()),
        )
    }

    /// The seed this node contributes to the global `length_scale`: the point
    /// splat scale, capped by the inter-camera distance when there is one.
    ///
    /// Both inputs are measured in the node's own coordinates, so the node
    /// transform's scale converts them to the world-space quantity
    /// `length_scale` actually is. That is what lets a scaled node stop
    /// dominating the global frustum size once it has been aligned.
    pub(super) fn length_scale_seed(&self) -> f32 {
        sfmtool_core::analysis::scene_scale::length_scale(
            self.auto_point_size,
            self.camera_nn_scale,
        ) * self.transform.scale as f32
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
