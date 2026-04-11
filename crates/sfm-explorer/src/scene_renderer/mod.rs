// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU scene renderer.
//!
//! Manages offscreen rendering of the 3D scene to a texture that is
//! displayed inside the egui UI. Uses a two-pass pipeline:
//!
//! 1. Point splat pass — renders instanced billboard quads to color + linear depth
//! 2. EDL post-process pass — applies Eye-Dome Lighting for depth-aware shading

mod auto_point_size;
mod distorted_mesh;
mod gpu_types;
mod pipelines;
mod readback;
mod render;
mod sizing;
mod uniforms;
mod upload;

use gpu_types::*;

// Re-export public constants so external modules can use `crate::scene_renderer::*`.
pub use gpu_types::{
    DEFAULT_FRUSTUM_SIZE_MULTIPLIER, DEFAULT_LENGTH_SCALE_MULTIPLIER,
    DEFAULT_TARGET_FOG_MULTIPLIER, DEFAULT_TARGET_SIZE_MULTIPLIER, PICK_INDEX_MASK,
    PICK_TAG_FRUSTUM, PICK_TAG_MASK, PICK_TAG_NONE, PICK_TAG_POINT,
};

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
    point_bind_group: Option<wgpu::BindGroup>,

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
    frustum_edge_buffer: Option<wgpu::Buffer>,
    frustum_uniform_buffer: Option<wgpu::Buffer>,
    frustum_bind_group: Option<wgpu::BindGroup>,
    frustum_edge_count: u32,

    // ── Image quad rendering (pinhole: instanced, distorted: indexed) ──
    image_quad_pipeline: Option<wgpu::RenderPipeline>,
    image_quad_bind_group_layout: Option<wgpu::BindGroupLayout>,
    image_quad_instance_buffer: Option<wgpu::Buffer>,
    image_quad_bind_group: Option<wgpu::BindGroup>,
    image_quad_count: u32,
    // Distorted image quad rendering (tessellated mesh)
    distorted_quad_pipeline: Option<wgpu::RenderPipeline>,
    distorted_quad_vertex_buffer: Option<wgpu::Buffer>,
    distorted_quad_index_buffer: Option<wgpu::Buffer>,
    distorted_quad_index_count: u32,
    thumbnail_texture: Option<wgpu::Texture>,
    image_quad_uniform_buffer: Option<wgpu::Buffer>,
    atlas_cols: u32,
    atlas_rows: u32,
    images_per_page: u32,

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
    /// Which camera index is currently loaded into `bg_image_texture`.
    bg_image_loaded_index: Option<usize>,

    // ── Point cloud data ──
    instance_buffer: Option<wgpu::Buffer>,
    point_count: u32,

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
    /// Most recently read-back hover pick ID (tag | index).
    hover_pick_id: u32,

    // ── Settings ──
    /// Auto-computed point size (median NN distance * 0.5). Updated on upload.
    auto_point_size: f32,

    /// Characteristic inter-camera distance (p90 of camera-center NN distances).
    /// `None` when fewer than 2 cameras or not yet computed.
    camera_nn_scale: Option<f32>,

    // ── Scene bounds ──
    /// Bounding sphere center of the reconstruction's 3D points.
    scene_center: nalgebra::Point3<f64>,
    /// Bounding sphere radius (80th percentile extent from center).
    scene_radius: f64,
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
            point_bind_group: None,
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
            frustum_edge_buffer: None,
            frustum_uniform_buffer: None,
            frustum_bind_group: None,
            frustum_edge_count: 0,
            image_quad_pipeline: None,
            image_quad_bind_group_layout: None,
            image_quad_instance_buffer: None,
            image_quad_bind_group: None,
            image_quad_count: 0,
            distorted_quad_pipeline: None,
            distorted_quad_vertex_buffer: None,
            distorted_quad_index_buffer: None,
            distorted_quad_index_count: 0,
            thumbnail_texture: None,
            image_quad_uniform_buffer: None,
            atlas_cols: 0,
            atlas_rows: 0,
            images_per_page: 0,
            bg_image_distorted_pipeline: None,
            bg_image_distorted_vertex_buffer: None,
            bg_image_distorted_index_buffer: None,
            bg_image_distorted_index_count: 0,
            bg_image_uniform_buffer: None,
            bg_image_bind_group_layout: None,
            bg_image_bind_group: None,
            bg_image_sampler: None,
            bg_image_texture: None,
            bg_image_loaded_index: None,
            instance_buffer: None,
            point_count: 0,
            linear_depth_texture: None,
            depth_staging: None,
            pick_staging: None,
            readback_pending: false,
            hover_depth: None,
            hover_pick_id: PICK_TAG_NONE,
            auto_point_size: FALLBACK_POINT_SIZE,
            camera_nn_scale: None,
            scene_center: nalgebra::Point3::origin(),
            scene_radius: 1.0,
        }
    }

    /// Returns the egui texture ID for the rendered scene, if available.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.egui_texture_id
    }

    /// Returns the auto-computed point size (world space, before user scaling).
    pub fn auto_point_size(&self) -> f32 {
        self.auto_point_size
    }

    /// Returns the characteristic inter-camera distance (p90 of camera NN distances),
    /// or `None` if fewer than 2 cameras.
    pub fn camera_nn_scale(&self) -> Option<f32> {
        self.camera_nn_scale
    }

    /// Returns the bounding sphere center of the scene.
    pub fn scene_center(&self) -> nalgebra::Point3<f64> {
        self.scene_center
    }

    /// Returns the bounding sphere radius of the scene.
    pub fn scene_radius(&self) -> f64 {
        self.scene_radius
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
        self.point_bind_group = Some(pt.bind_group);

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
        self.frustum_bind_group = Some(fr.bind_group);

        // ── Image quad pipeline ──
        let iq = pipelines::image_quad::create(device);
        self.image_quad_pipeline = Some(iq.pipeline);
        self.image_quad_bind_group_layout = Some(iq.bind_group_layout);

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

impl Default for SceneRenderer {
    fn default() -> Self {
        Self::new()
    }
}
