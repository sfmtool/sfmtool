// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU scene renderer.
//!
//! Manages offscreen rendering of the 3D scene to a texture that is
//! displayed inside the egui UI. Uses a two-pass pipeline:
//!
//! 1. Point splat pass — renders instanced billboard quads to color + linear depth
//! 2. EDL post-process pass — applies Eye-Dome Lighting for depth-aware shading
//!
//! ## Per-reconstruction resources
//!
//! Everything belonging to one loaded reconstruction lives in a
//! [`ReconResources`] bundle keyed by [`ReconId`]; the fields on
//! [`SceneRenderer`] itself are the shared singletons. Each scene pass loops
//! over the bundles, binding that node's bind group (which carries its
//! [`ReconUniforms`] slice: model matrix, point size, pick bases) before the
//! existing instanced draw. See `specs/gui/gui-scene-graph.md`.

mod auto_point_size;
mod compass;
mod distorted_mesh;
mod gpu_types;
mod picking;
mod pipelines;
mod readback;
mod recon;
mod render;
mod sizing;
mod uniforms;
mod upload;

use std::collections::HashMap;

use nalgebra::Point3;
use sfmtool_core::Se3Transform;

use crate::scene::{ImageRef, PointRef, ReconId};
use gpu_types::*;
use picking::PickTables;
use recon::ReconResources;

// Re-export public constants so external modules can use `crate::scene_renderer::*`.
pub use gpu_types::{
    DEFAULT_FRUSTUM_SIZE_MULTIPLIER, DEFAULT_LENGTH_SCALE_MULTIPLIER,
    DEFAULT_TARGET_FOG_MULTIPLIER, DEFAULT_TARGET_SIZE_MULTIPLIER,
};
pub use picking::PickTarget;
pub use recon::NodeDisplay;

// ── SceneRenderer ───────────────────────────────────────────────────────

/// Manages GPU rendering of the 3D scene to an offscreen texture.
///
/// The final EDL-shaded texture is registered with egui for display as a
/// background image in the 3D viewport panel.
pub struct SceneRenderer {
    // ── Pass 1 render targets (recreated on resize) ──
    splat_color_view: Option<wgpu::TextureView>,
    linear_depth_view: Option<wgpu::TextureView>,
    hw_depth_view: Option<wgpu::TextureView>,
    pick_texture: Option<wgpu::Texture>,
    pick_texture_view: Option<wgpu::TextureView>,

    // ── Pass 2 render targets (recreated on resize) ──
    edl_output_view: Option<wgpu::TextureView>,

    // ── egui integration ──
    egui_texture_id: Option<egui::TextureId>,
    current_size: (u32, u32),

    // ── Pass 1 pipeline resources (created once) ──
    point_pipeline: Option<wgpu::RenderPipeline>,
    quad_vertex_buffer: Option<wgpu::Buffer>,
    point_uniform_buffer: Option<wgpu::Buffer>,
    point_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // ── Pass 2 pipeline resources ──
    edl_pipeline: Option<wgpu::RenderPipeline>,
    edl_bind_group_layout: Option<wgpu::BindGroupLayout>,
    edl_uniform_buffer: Option<wgpu::Buffer>,
    edl_sampler: Option<wgpu::Sampler>,
    edl_bind_group: Option<wgpu::BindGroup>, // recreated on resize

    // ── Target indicator pipeline resources ──
    target_edge_pipeline: Option<wgpu::RenderPipeline>,
    target_edge_buffer: Option<wgpu::Buffer>,
    target_edge_count: u32,
    target_star_pipeline: Option<wgpu::RenderPipeline>,
    target_star_buffer: Option<wgpu::Buffer>,
    target_star_vertex_count: u32,
    target_uniform_buffer: Option<wgpu::Buffer>,
    target_bind_group_layout: Option<wgpu::BindGroupLayout>,
    target_bind_group: Option<wgpu::BindGroup>, // recreated on resize

    // ── Track ray rendering (post-EDL, depth-aware) ──
    track_ray_pipeline: Option<wgpu::RenderPipeline>,
    track_ray_edge_buffer: Option<wgpu::Buffer>,
    track_ray_uniform_buffer: Option<wgpu::Buffer>,
    track_ray_bind_group_layout: Option<wgpu::BindGroupLayout>,
    track_ray_bind_group: Option<wgpu::BindGroup>, // recreated on resize
    track_ray_count: u32,

    // ── Frustum rendering ──
    frustum_pipeline: Option<wgpu::RenderPipeline>,
    frustum_uniform_buffer: Option<wgpu::Buffer>,
    frustum_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // ── Image quad rendering (pinhole: instanced, distorted: indexed) ──
    image_quad_pipeline: Option<wgpu::RenderPipeline>,
    image_quad_bind_group_layout: Option<wgpu::BindGroupLayout>,
    distorted_quad_pipeline: Option<wgpu::RenderPipeline>,
    image_quad_sampler: Option<wgpu::Sampler>,

    // ── Patch (surfel) rendering ──
    patch_pipeline: Option<wgpu::RenderPipeline>,
    patch_bind_group_layout: Option<wgpu::BindGroupLayout>,
    patch_sampler: Option<wgpu::Sampler>,

    // ── Background image (camera view mode) ──
    bg_image_distorted_pipeline: Option<wgpu::RenderPipeline>,
    bg_image_distorted_vertex_buffer: Option<wgpu::Buffer>,
    bg_image_distorted_index_buffer: Option<wgpu::Buffer>,
    bg_image_distorted_index_count: u32,
    bg_image_uniform_buffer: Option<wgpu::Buffer>,
    bg_image_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bg_image_bind_group: Option<wgpu::BindGroup>,
    bg_image_sampler: Option<wgpu::Sampler>,
    /// Full-resolution background image texture (single image, not array).
    bg_image_texture: Option<wgpu::Texture>,
    /// Which image is currently loaded into `bg_image_texture`. A ref rather
    /// than an index: keyed by index alone, a file replacement kept showing the
    /// old background for the same position in the new reconstruction.
    bg_image_loaded: Option<ImageRef>,

    // ── Per-reconstruction resources ──
    /// One bundle per loaded node. Uploads write into a bundle; a node leaving
    /// the scene drops one.
    recons: HashMap<ReconId, ReconResources>,
    /// The bundle keys in `ReconId` (= load) order, so draws and pick-base
    /// assignment are deterministic.
    recon_order: Vec<ReconId>,
    /// Sorted `(base, ReconId)` tables that decode a pick id back to a ref.
    pick_tables: PickTables,

    // ── GPU readback (5x5 region, shared by hover + click) ──
    /// The linear depth texture (kept for copy operations).
    linear_depth_texture: Option<wgpu::Texture>,
    /// Staging buffer for 5x5 depth region readback.
    depth_staging: Option<wgpu::Buffer>,
    /// Staging buffer for 5x5 pick ID region readback.
    pick_staging: Option<wgpu::Buffer>,
    /// Whether a readback copy was enqueued this frame.
    readback_pending: bool,
    /// Most recently read-back hover depth.
    hover_depth: Option<f32>,
    /// Most recently read-back hover pick, already decoded to a typed ref.
    hover_pick: Option<PickTarget>,
}

impl SceneRenderer {
    pub fn new() -> Self {
        Self {
            splat_color_view: None,
            linear_depth_view: None,
            hw_depth_view: None,
            pick_texture: None,
            pick_texture_view: None,
            edl_output_view: None,
            egui_texture_id: None,
            current_size: (0, 0),
            point_pipeline: None,
            quad_vertex_buffer: None,
            point_uniform_buffer: None,
            point_bind_group_layout: None,
            edl_pipeline: None,
            edl_bind_group_layout: None,
            edl_uniform_buffer: None,
            edl_sampler: None,
            edl_bind_group: None,
            target_edge_pipeline: None,
            target_edge_buffer: None,
            target_edge_count: 0,
            target_star_pipeline: None,
            target_star_buffer: None,
            target_star_vertex_count: 0,
            target_uniform_buffer: None,
            target_bind_group_layout: None,
            target_bind_group: None,
            track_ray_pipeline: None,
            track_ray_edge_buffer: None,
            track_ray_uniform_buffer: None,
            track_ray_bind_group_layout: None,
            track_ray_bind_group: None,
            track_ray_count: 0,
            frustum_pipeline: None,
            frustum_uniform_buffer: None,
            frustum_bind_group_layout: None,
            image_quad_pipeline: None,
            image_quad_bind_group_layout: None,
            distorted_quad_pipeline: None,
            image_quad_sampler: None,
            patch_pipeline: None,
            patch_bind_group_layout: None,
            patch_sampler: None,
            bg_image_distorted_pipeline: None,
            bg_image_distorted_vertex_buffer: None,
            bg_image_distorted_index_buffer: None,
            bg_image_distorted_index_count: 0,
            bg_image_uniform_buffer: None,
            bg_image_bind_group_layout: None,
            bg_image_bind_group: None,
            bg_image_sampler: None,
            bg_image_texture: None,
            bg_image_loaded: None,
            recons: HashMap::new(),
            recon_order: Vec::new(),
            pick_tables: PickTables::default(),
            linear_depth_texture: None,
            depth_staging: None,
            pick_staging: None,
            readback_pending: false,
            hover_depth: None,
            hover_pick: None,
        }
    }

    /// Returns the egui texture ID for the rendered scene, if available.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.egui_texture_id
    }

    /// The `length_scale` the loaded reconstructions suggest: the smallest
    /// per-node seed (splat scale, capped by that node's inter-camera
    /// distance). `None` when nothing has been uploaded yet.
    ///
    /// One global value for the whole scene is a known compromise of the
    /// multi-reconstruction design — two nodes at wildly different scales share
    /// one frustum size until they are aligned.
    pub fn length_scale_seed(&self) -> Option<f32> {
        self.recons
            .values()
            .map(|r| r.length_scale_seed())
            .min_by(f32::total_cmp)
    }

    /// The largest auto point size across loaded nodes, for the one global
    /// consumer that cannot be per-recon: the EDL pass, which shades the whole
    /// frame in one fullscreen draw.
    fn max_auto_point_size(&self) -> f32 {
        self.recons
            .values()
            .map(|r| r.auto_point_size)
            .max_by(f32::total_cmp)
            .unwrap_or(FALLBACK_POINT_SIZE)
    }

    /// Bounding sphere (centre, radius) of the whole scene: the union of the
    /// *visible* nodes' bounds. Drives the adaptive clip planes.
    ///
    /// Hidden nodes drop out so switching a reference reconstruction off also
    /// stops it dragging the clip planes and the grid scale around.
    pub fn scene_bounds(&self) -> (Point3<f64>, f64) {
        self.visible_bounds()
            .unwrap_or_else(|| self.any_bounds().unwrap_or((Point3::origin(), 1.0)))
    }

    /// The union of the visible nodes' bounds, or `None` when nothing visible
    /// has uploaded its points yet. Each node contributes its bounds *through
    /// its transform*, so an aligned node frames where it is drawn.
    fn visible_bounds(&self) -> Option<(Point3<f64>, f64)> {
        self.recons
            .values()
            .filter(|r| r.display.visible)
            .filter_map(|r| r.world_bounds())
            .reduce(union_sphere)
    }

    /// The union over every loaded node, visible or not — the fallback that
    /// keeps the camera somewhere sensible when every node is hidden.
    fn any_bounds(&self) -> Option<(Point3<f64>, f64)> {
        self.recons
            .values()
            .filter_map(|r| r.world_bounds())
            .reduce(union_sphere)
    }

    /// Mirror a node's Scene-panel display state onto its GPU bundle.
    ///
    /// A no-op for a node with no bundle yet: the bundle is created by its
    /// first upload and starts from [`NodeDisplay::default`], and the next
    /// frame's sync lands before it is ever drawn.
    pub fn set_node_display(&mut self, id: ReconId, display: NodeDisplay) {
        if let Some(bundle) = self.recons.get_mut(&id) {
            bundle.display = display;
        }
    }

    /// Mirror a node's similarity transform onto its GPU bundle.
    ///
    /// From here it reaches the shaders as the per-recon `model` matrix (and
    /// scales the node's splat size), the union scene bounds, and the
    /// `length_scale` seed. Same no-op-before-upload rule as
    /// [`SceneRenderer::set_node_display`].
    pub fn set_node_transform(&mut self, id: ReconId, transform: Se3Transform) {
        if let Some(bundle) = self.recons.get_mut(&id) {
            bundle.transform = transform;
        }
    }

    /// Decode a pick id read back from the GPU into the entity it addresses.
    pub fn decode_pick(&self, pick_id: u32) -> Option<PickTarget> {
        self.pick_tables.resolve(pick_id)
    }

    /// The global point index a ref maps to, for the shader's `u32` compare.
    fn global_point_index(&self, point: PointRef) -> Option<u32> {
        let bundle = self.recons.get(&point.recon)?;
        let local = point.point;
        (local < bundle.point_count).then(|| bundle.point_pick_base + local)
    }

    /// The global image index a ref maps to, for the shader's `u32` compare.
    fn global_image_index(&self, image: ImageRef) -> Option<u32> {
        let bundle = self.recons.get(&image.recon)?;
        let local = image.image;
        (local < bundle.frustum_image_count).then(|| bundle.image_pick_base + local)
    }

    /// Ensure a bundle exists for `id`, creating an empty one if not.
    ///
    /// Also creates the pipelines, since a bundle's point bind group needs the
    /// point pipeline's layout and uniform buffer. Returns nothing on purpose:
    /// callers reach for the bundle with a direct `self.recons` field borrow so
    /// they can hold the shared layouts and samplers at the same time.
    fn ensure_recon(&mut self, device: &wgpu::Device, id: ReconId) {
        self.ensure_pipelines(device);
        if !self.recons.contains_key(&id) {
            let bundle = ReconResources::new(
                device,
                self.point_bind_group_layout.as_ref().unwrap(),
                self.point_uniform_buffer.as_ref().unwrap(),
            );
            self.recons.insert(id, bundle);
            self.assign_pick_bases();
        }
    }

    /// Drop the bundles of every node that has left the scene.
    ///
    /// The single release path: closing one node and replacing the whole scene
    /// are the same operation seen from the renderer, and both have to reassign
    /// pick bases afterwards.
    pub fn retain_nodes(&mut self, alive: impl Fn(ReconId) -> bool) {
        let before = self.recons.len();
        self.recons.retain(|id, _| alive(*id));
        if self.recons.len() != before {
            self.assign_pick_bases();
        }
    }

    /// (Re)assign every node's contiguous slice of the two global pick index
    /// spaces, and rebuild the decode tables.
    ///
    /// Called whenever a node is added or removed, or its entity counts change.
    /// The bases live in each bundle and travel to the GPU in that node's
    /// uniform block, which `update_uniforms` rewrites every frame — so a
    /// reassignment costs nothing beyond this walk: **no instance buffer is
    /// ever rewritten for a base change.**
    fn assign_pick_bases(&mut self) {
        self.recon_order = {
            let mut ids: Vec<ReconId> = self.recons.keys().copied().collect();
            ids.sort_unstable();
            ids
        };

        self.pick_tables.clear();
        let (mut point_base, mut image_base) = (0u32, 0u32);
        for id in &self.recon_order {
            let bundle = self.recons.get_mut(id).expect("id came from the map");
            bundle.point_pick_base = point_base;
            bundle.image_pick_base = image_base;
            self.pick_tables.push(
                *id,
                point_base,
                bundle.point_count,
                image_base,
                bundle.frustum_image_count,
            );
            point_base = point_base.saturating_add(bundle.point_count);
            image_base = image_base.saturating_add(bundle.frustum_image_count);
        }

        if point_base > picking::PICK_INDEX_CAPACITY || image_base > picking::PICK_INDEX_CAPACITY {
            log::warn!(
                "loaded reconstructions exceed the {} entity pick capacity \
                 ({point_base} points, {image_base} images); picking will be wrong \
                 for the overflowing nodes",
                picking::PICK_INDEX_CAPACITY,
            );
        }
    }

    /// Ensure all render pipelines exist. Called once on first use.
    fn ensure_pipelines(&mut self, device: &wgpu::Device) {
        if self.point_pipeline.is_some() {
            return;
        }

        // ── Pass 1: Point splat pipeline ──
        let pt = pipelines::points::create(device);
        self.point_pipeline = Some(pt.pipeline);
        self.quad_vertex_buffer = Some(pt.quad_vertex_buffer);
        self.point_uniform_buffer = Some(pt.uniform_buffer);
        self.point_bind_group_layout = Some(pt.bind_group_layout);

        // ── Pass 2: EDL post-process pipeline ──
        let edl = pipelines::edl::create(device);
        self.edl_pipeline = Some(edl.pipeline);
        self.edl_bind_group_layout = Some(edl.bind_group_layout);
        self.edl_uniform_buffer = Some(edl.uniform_buffer);
        self.edl_sampler = Some(edl.sampler);

        // ── Target indicator pipeline ──
        let tgt = pipelines::target::create(device);
        self.target_edge_pipeline = Some(tgt.edge_pipeline);
        self.target_edge_buffer = Some(tgt.edge_buffer);
        self.target_edge_count = tgt.edge_count;
        self.target_star_pipeline = Some(tgt.star_pipeline);
        self.target_star_buffer = Some(tgt.star_buffer);
        self.target_star_vertex_count = tgt.star_vertex_count;
        self.target_uniform_buffer = Some(tgt.uniform_buffer);
        self.target_bind_group_layout = Some(tgt.bind_group_layout);

        // ── Track ray pipeline (post-EDL, depth-aware) ──
        let tr = pipelines::track_ray::create(device);
        self.track_ray_pipeline = Some(tr.pipeline);
        self.track_ray_uniform_buffer = Some(tr.uniform_buffer);
        self.track_ray_bind_group_layout = Some(tr.bind_group_layout);

        // ── Frustum wireframe pipeline ──
        let fr = pipelines::frustum::create(device);
        self.frustum_pipeline = Some(fr.pipeline);
        self.frustum_uniform_buffer = Some(fr.uniform_buffer);
        self.frustum_bind_group_layout = Some(fr.bind_group_layout);

        // ── Image quad pipeline ──
        let iq = pipelines::image_quad::create(device);
        self.image_quad_pipeline = Some(iq.pipeline);
        self.image_quad_bind_group_layout = Some(iq.bind_group_layout);

        // ── Patch surfel pipeline ──
        let pa = pipelines::patch::create(device);
        self.patch_pipeline = Some(pa.pipeline);
        self.patch_bind_group_layout = Some(pa.bind_group_layout);

        // Atlas samplers are shared: every node's thumbnail / patch atlas is
        // sampled the same way, so only the textures are per-recon.
        self.image_quad_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("thumbnail sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.patch_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("patch sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // ── Distorted image quad pipeline ──
        self.distorted_quad_pipeline = Some(pipelines::distorted_quad::create(
            device,
            self.image_quad_bind_group_layout.as_ref().unwrap(),
        ));

        // ── Background image pipeline (camera view mode) ──
        let bg = pipelines::bg_image::create(device);
        self.bg_image_bind_group_layout = Some(bg.bind_group_layout);
        self.bg_image_uniform_buffer = Some(bg.uniform_buffer);
        self.bg_image_sampler = Some(bg.sampler);
        self.bg_image_distorted_pipeline = Some(pipelines::bg_distorted::create(
            device,
            self.bg_image_bind_group_layout.as_ref().unwrap(),
        ));
    }
}

/// The smallest sphere containing both inputs.
fn union_sphere(a: (Point3<f64>, f64), b: (Point3<f64>, f64)) -> (Point3<f64>, f64) {
    let offset = b.0 - a.0;
    let distance = offset.norm();
    if distance + b.1 <= a.1 {
        return a; // b is inside a
    }
    if distance + a.1 <= b.1 {
        return b; // a is inside b
    }
    // Neither contains the other, so `distance > 0` and the new centre sits on
    // the segment joining them.
    let radius = (distance + a.1 + b.1) / 2.0;
    let t = (radius - a.1) / distance;
    (a.0 + offset * t, radius)
}

impl Default for SceneRenderer {
    fn default() -> Self {
        Self::new()
    }
}
