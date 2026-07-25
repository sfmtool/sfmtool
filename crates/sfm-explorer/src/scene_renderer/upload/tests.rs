// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Upload-path tests against a headless `wgpu` device.
//!
//! These exercise the CPU-side work each upload does — instance/edge counts,
//! atlas page-grid arithmetic, GPU-limit clamping, and the "skip rather than
//! draw something misleading" guards.
//!
//! The device comes from wgpu's `noop` backend: wgpu-core still runs its full
//! validation (limits, texture extents, copy sizes, pipeline and shader
//! checks) while wgpu-hal stubs the driver calls. That matters here — a
//! miscomputed atlas extent, or a `write_texture` whose data length disagrees
//! with its extent, is a validation error, so these tests fail on bad
//! arithmetic rather than silently accepting it.

use std::collections::HashMap;

use ndarray::{Array2, Array4};
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};
use sfmtool_core::reconstruction::ObservationSource;
use sfmtool_core::SfmrReconstruction;

use super::super::gpu_types::{
    BG_PINHOLE_SUBDIVISIONS, DISTORTION_SUBDIVISIONS, FISHEYE_SUBDIVISIONS, THUMBNAIL_SIZE,
};
use super::super::SceneRenderer;
use super::track_rays::track_ray_edges;
use crate::state::CachedSiftFeatures;

// ── Fixtures ────────────────────────────────────────────────────────────

/// A `wgpu::Device` on the noop backend, with `limits` as the *device* limits.
/// The noop adapter reports maximally permissive limits, so any request is
/// granted — which is what lets a test drive the clamping paths.
fn device_with_limits(limits: wgpu::Limits) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .expect("noop device")
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    device_with_limits(wgpu::Limits::default())
}

/// `SfmrReconstruction::demo` gives 8 pinhole images, `n` finite points, two
/// observations per point, and zeroed 128×128 thumbnails.
const DEMO_IMAGES: u32 = 8;

fn demo(points: usize) -> SfmrReconstruction {
    SfmrReconstruction::demo(points)
}

/// Replace the single camera so every image resolves to `model`.
fn with_camera_model(mut recon: SfmrReconstruction, model: CameraModel) -> SfmrReconstruction {
    recon.cameras = vec![CameraIntrinsics {
        model,
        width: 1920,
        height: 1080,
    }];
    recon
}

fn fisheye() -> CameraModel {
    CameraModel::OpenCVFisheye {
        focal_length_x: 800.0,
        focal_length_y: 800.0,
        principal_point_x: 960.0,
        principal_point_y: 540.0,
        radial_distortion_k1: 0.01,
        radial_distortion_k2: 0.0,
        radial_distortion_k3: 0.0,
        radial_distortion_k4: 0.0,
    }
}

fn radial_distorted() -> CameraModel {
    CameraModel::SimpleRadial {
        focal_length: 1000.0,
        principal_point_x: 960.0,
        principal_point_y: 540.0,
        radial_distortion_k1: 0.1,
    }
}

/// Attach patch frames + bitmaps. `present[i]` false leaves point `i`'s `u`
/// row all-zero, which is how the upload detects "no patch here".
///
/// `bitmap_rows` defaults to `present.len()`; passing fewer exercises the
/// short-array bound in the row scan. `bitmap_cols` defaults to `resolution`;
/// passing something else produces non-square tiles.
fn with_patches(
    mut recon: SfmrReconstruction,
    resolution: usize,
    present: &[bool],
    bitmap_rows: Option<usize>,
    bitmap_cols: Option<usize>,
) -> SfmrReconstruction {
    let n = present.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    for (i, &is_present) in present.iter().enumerate() {
        if is_present {
            u[[i, 0]] = 0.1;
            v[[i, 1]] = 0.1;
        }
    }
    let rows = bitmap_rows.unwrap_or(n);
    let cols = bitmap_cols.unwrap_or(resolution);
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);
    recon.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::zeros((rows, resolution, cols, 4)));
    recon
}

/// Swap the observation source to embedded keypoints, so track rays read
/// inline keypoints instead of the SIFT cache.
fn with_embedded_keypoints(mut recon: SfmrReconstruction) -> SfmrReconstruction {
    let obs_count = recon.tracks.len();
    recon.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::<f32>::from_elem((obs_count, 2), 100.0),
        image_file_hashes: vec![[0u8; 16]; recon.images.len()],
    };
    recon
}

/// A SIFT cache covering `images` images, each holding `features` entries.
fn sift_cache(images: usize, features: usize) -> HashMap<usize, CachedSiftFeatures> {
    (0..images)
        .map(|i| {
            (
                i,
                CachedSiftFeatures {
                    positions_xy: vec![[500.0, 300.0]; features],
                    affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]; features],
                    read_count: features,
                },
            )
        })
        .collect()
}

/// Bounding-box diagonal of the camera centres, recomputed here independently
/// of the implementation so the infinity-ray length assertion is a real check
/// rather than a restatement of the code under test.
fn camera_cloud_diagonal(recon: &SfmrReconstruction) -> f64 {
    let centres: Vec<_> = recon.images.iter().map(|im| im.camera_center()).collect();
    let axis_span = |f: fn(&nalgebra::Point3<f64>) -> f64| {
        let lo = centres.iter().map(f).fold(f64::INFINITY, f64::min);
        let hi = centres.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let (dx, dy, dz) = (axis_span(|p| p.x), axis_span(|p| p.y), axis_span(|p| p.z));
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn edge_length(e: &super::super::gpu_types::EdgeInstance) -> f64 {
    let d = [
        (e.endpoint_b[0] - e.endpoint_a[0]) as f64,
        (e.endpoint_b[1] - e.endpoint_a[1]) as f64,
        (e.endpoint_b[2] - e.endpoint_a[2]) as f64,
    ];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A uniquely-named temp directory removed on drop, so a failing assertion
/// cannot leak it into the next run (a leftover directory owned by another
/// user makes the image write fail for unrelated reasons).
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "sfm-explorer-upload-tests-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        TempDir(dir)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

// ── points ──────────────────────────────────────────────────────────────

#[test]
fn upload_points_counts_instances_and_derives_scene_scale() {
    let (device, _queue) = device();
    let recon = demo(64);
    let mut r = SceneRenderer::new();

    r.upload_points(&device, &recon);

    assert_eq!(r.point_count, 64);
    assert!(r.instance_buffer.is_some());
    // Demo points sit on a unit sphere offset to z+1, so both the splat size
    // and the bounding sphere must come out positive and finite.
    assert!(r.auto_point_size > 0.0 && r.auto_point_size.is_finite());
    assert!(r.scene_radius > 0.0 && r.scene_radius.is_finite());
    let nn = r
        .camera_nn_scale
        .expect("8 demo cameras give a nearest-neighbour scale");
    assert!(nn > 0.0 && nn.is_finite());
}

#[test]
fn upload_points_handles_an_empty_cloud() {
    let (device, _queue) = device();
    let mut recon = demo(8);
    recon.points.clear();
    let mut r = SceneRenderer::new();

    r.upload_points(&device, &recon);

    assert_eq!(r.point_count, 0);
}

// ── frustums ────────────────────────────────────────────────────────────

#[test]
fn upload_frustums_emits_eight_edges_per_pinhole_camera() {
    let (device, _queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_frustums(&device, &recon, 1.0, 1.0);

    // 4 apex→corner side edges + 4 base edges around the far face.
    assert_eq!(r.frustum_edge_count, DEMO_IMAGES * 8);
    assert_eq!(r.frustum_image_count, DEMO_IMAGES);
    assert!(r.frustum_color_buffer.is_some());
    // No thumbnail atlas yet, so no image quads are built.
    assert_eq!(r.image_quad_count, 0);
    assert_eq!(r.distorted_quad_index_count, 0);
}

#[test]
fn upload_frustums_builds_pinhole_image_quads_once_thumbnails_exist() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);
    r.upload_frustums(&device, &recon, 1.0, 1.0);

    assert_eq!(r.image_quad_count, DEMO_IMAGES);
    assert!(r.image_quad_instance_buffer.is_some());
    // Pinhole cameras take the flat-quad path, not the tessellated one.
    assert_eq!(r.distorted_quad_index_count, 0);
    assert!(r.distorted_quad_vertex_buffer.is_none());
}

#[test]
fn upload_frustums_tessellates_fisheye_cameras() {
    let (device, queue) = device();
    let recon = with_camera_model(demo(16), fisheye());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);
    r.upload_frustums(&device, &recon, 1.0, 1.0);

    // n×n grid: 4 side edges + 4 boundary walks of (n-1) segments each.
    let n = FISHEYE_SUBDIVISIONS + 1;
    assert_eq!(r.frustum_edge_count, DEMO_IMAGES * (4 + 4 * (n - 1)) as u32);
    // Two triangles per grid cell, six indices per cell.
    assert_eq!(
        r.distorted_quad_index_count,
        DEMO_IMAGES * ((n - 1) * (n - 1) * 6) as u32
    );
    // Fisheye takes the tessellated path exclusively.
    assert_eq!(r.image_quad_count, 0);
}

#[test]
fn upload_frustums_tessellates_distorted_cameras() {
    let (device, queue) = device();
    let recon = with_camera_model(demo(16), radial_distorted());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);
    r.upload_frustums(&device, &recon, 1.0, 1.0);

    let n = DISTORTION_SUBDIVISIONS + 1;
    assert_eq!(r.frustum_edge_count, DEMO_IMAGES * (4 + 4 * (n - 1)) as u32);
    assert_eq!(
        r.distorted_quad_index_count,
        DEMO_IMAGES * ((n - 1) * (n - 1) * 6) as u32
    );
    // A distorted camera must take the tessellated path exclusively, exactly
    // like fisheye — `has_distortion()` and `is_fisheye()` are separate
    // predicates, so a regression could send it down the pinhole quad path.
    assert_eq!(r.image_quad_count, 0);
}

#[test]
fn upload_frustums_replaces_quad_buffers_when_the_camera_model_changes() {
    let (device, queue) = device();
    let fisheye_recon = with_camera_model(demo(16), fisheye());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &fisheye_recon);
    r.upload_frustums(&device, &fisheye_recon, 1.0, 1.0);
    assert!(r.distorted_quad_index_count > 0);

    // Re-uploading a pinhole reconstruction must drop the stale distorted
    // buffers rather than leave them to be drawn against a new index count.
    let pinhole_recon = demo(16);
    r.upload_frustums(&device, &pinhole_recon, 1.0, 1.0);

    assert_eq!(r.distorted_quad_index_count, 0);
    assert!(r.distorted_quad_vertex_buffer.is_none());
    assert!(r.distorted_quad_index_buffer.is_none());
    assert_eq!(r.image_quad_count, DEMO_IMAGES);
}

#[test]
fn update_frustum_colors_is_a_no_op_before_any_upload() {
    let (_device, queue) = device();
    let r = SceneRenderer::new();
    // No color buffer yet — must return quietly rather than panic.
    r.update_frustum_colors(&queue, 8, Some(0), Some(1), &[2, 3]);
}

#[test]
fn update_frustum_colors_tolerates_out_of_range_indices() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();
    r.upload_frustums(&device, &recon, 1.0, 1.0);

    // Every index is past the end; each is individually bounds-checked, so
    // this must not panic or write out of range.
    r.update_frustum_colors(&queue, 8, Some(99), Some(99), &[99, 100]);
}

// ── thumbnails ──────────────────────────────────────────────────────────

#[test]
fn upload_thumbnails_packs_a_square_ish_atlas_grid() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);

    // cols = ceil(sqrt(8)) = 3, then rows = ceil(8/3) = 3.
    assert_eq!(r.atlas_cols, 3);
    assert_eq!(r.atlas_rows, 3);
    assert!(r.thumbnail_texture.is_some());
    assert!(r.image_quad_thumbnail_view.is_some());
    assert!(r.image_quad_uniform_buffer.is_some());
    // A page holds cols × (max_texture_dim / THUMBNAIL_SIZE) cells.
    let cells_per_axis = wgpu::Limits::default().max_texture_dimension_2d / THUMBNAIL_SIZE;
    assert_eq!(r.images_per_page, 3 * cells_per_axis);
}

#[test]
fn upload_thumbnails_clamps_to_the_gpu_texture_limits() {
    // 256px textures hold a 2×2 grid of 128px thumbnails, and only one array
    // layer is available — so at most 4 of the 8 demo images fit.
    let (device, queue) = device_with_limits(wgpu::Limits {
        max_texture_dimension_2d: 256,
        max_texture_array_layers: 1,
        ..wgpu::Limits::default()
    });
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);

    assert_eq!(r.atlas_cols, 2);
    assert_eq!(r.atlas_rows, 2);
    assert_eq!(r.images_per_page, 4);
    // The texture itself must stay inside the limits: wgpu-core would reject
    // the descriptor otherwise, so reaching here at all is the assertion.
    assert!(r.thumbnail_texture.is_some());
}

#[test]
fn upload_thumbnails_spills_onto_extra_atlas_pages() {
    // 512px textures hold a 4×4 grid, so 25 images need two array layers.
    let (device, queue) = device_with_limits(wgpu::Limits {
        max_texture_dimension_2d: 512,
        ..wgpu::Limits::default()
    });
    let mut recon = demo(16);
    let images = 25;
    let template = recon.images[0].clone();
    recon.images = vec![template; images];
    recon.thumbnails_y_x_rgb = Array4::zeros((images, 128, 128, 3));
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);

    // sqrt would ask for 5 columns, but a page is only 4 cells wide — the
    // texture-dimension budget is what caps the grid in practice. The
    // MAX_ATLAS_COLS cap (128) is looser than this on any real GPU: it binds
    // only above ~16k images *and* a >16384px texture limit, so it is left
    // untested rather than forced with an ~800MB thumbnail array.
    assert_eq!(r.atlas_cols, 4);
    assert_eq!(r.atlas_rows, 4);
    assert_eq!(r.images_per_page, 16);
    // 25 images over 16 per page = 2 pages, and all 25 still fit.
    assert!(r.thumbnail_texture.is_some());
}

#[test]
fn upload_thumbnails_skips_an_imageless_reconstruction() {
    let (device, queue) = device();
    let mut recon = demo(8);
    recon.images.clear();
    recon.thumbnails_y_x_rgb = Array4::zeros((0, 128, 128, 3));
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, &recon);

    assert!(r.thumbnail_texture.is_none());
}

// ── patches ─────────────────────────────────────────────────────────────

#[test]
fn upload_patches_counts_only_points_carrying_a_patch() {
    let (device, queue) = device();
    let present = [true, false, true, true, false, true, true];
    let recon = with_patches(demo(present.len()), 16, &present, None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    assert_eq!(r.patch_count, 5);
    assert!(r.patch_instance_buffer.is_some());
    assert!(r.patch_atlas_texture.is_some());
    // cols = ceil(sqrt(5)) = 3, rows = ceil(5/3) = 2.
    assert_eq!(r.patch_atlas_cols, 3);
    assert_eq!(r.patch_atlas_rows, 2);
}

#[test]
fn upload_patches_uploads_nothing_without_patch_arrays() {
    let (device, queue) = device();
    let recon = demo(8); // demo carries no patch frames
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    assert_eq!(r.patch_count, 0);
    assert!(r.patch_instance_buffer.is_none());
    assert!(r.patch_atlas_texture.is_none());
}

#[test]
fn upload_patches_skips_frames_without_bitmaps() {
    let (device, queue) = device();
    let mut recon = with_patches(demo(4), 16, &[true; 4], None, None);
    recon.patch_bitmaps_y_x_rgba = None; // frames present, bitmaps absent
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    // v1 renders textured patches only.
    assert_eq!(r.patch_count, 0);
}

#[test]
fn upload_patches_rejects_non_square_bitmaps() {
    let (device, queue) = device();
    // 16 rows × 17 columns — an in-memory reconstruction is not shape-verified.
    let recon = with_patches(demo(4), 16, &[true; 4], None, Some(17));
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    assert_eq!(r.patch_count, 0);
    assert!(r.patch_atlas_texture.is_none());
}

#[test]
fn upload_patches_rejects_zero_resolution_bitmaps() {
    let (device, queue) = device();
    let recon = with_patches(demo(4), 0, &[true; 4], None, Some(0));
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    assert_eq!(r.patch_count, 0);
}

#[test]
fn upload_patches_rejects_bitmaps_larger_than_the_texture_limit() {
    let (device, queue) = device_with_limits(wgpu::Limits {
        max_texture_dimension_2d: 256,
        ..wgpu::Limits::default()
    });
    let recon = with_patches(demo(2), 512, &[true; 2], None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    // Skipped rather than passed to wgpu, which would be a validation error.
    assert_eq!(r.patch_count, 0);
}

#[test]
fn upload_patches_bounds_the_row_scan_by_the_shortest_array() {
    let (device, queue) = device();
    // 6 points and 6 frame rows, but only 2 bitmap rows.
    let recon = with_patches(demo(6), 16, &[true; 6], Some(2), None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &recon);

    assert_eq!(r.patch_count, 2);
}

#[test]
fn upload_patches_clears_stale_patches_when_reloading_without_them() {
    let (device, queue) = device();
    let with = with_patches(demo(4), 16, &[true; 4], None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, &with);
    assert_eq!(r.patch_count, 4);

    r.upload_patches(&device, &queue, &demo(4));

    assert_eq!(r.patch_count, 0);
    assert!(r.patch_instance_buffer.is_none());
    assert!(r.patch_atlas_texture.is_none());
    assert!(r.patch_bind_group.is_none());
}

// ── track rays ──────────────────────────────────────────────────────────

#[test]
fn upload_track_rays_emits_one_ray_per_cached_observation() {
    let (device, _queue) = device();
    let recon = demo(4);
    // Point 3, not 0: demo sets feature_indexes[i] = i, so a non-zero point
    // makes the obs_start + k -> feature_indexes -> cache lookup chain
    // load-bearing rather than always resolving index 0.
    let cache = sift_cache(recon.images.len(), 8);
    let mut r = SceneRenderer::new();

    r.upload_track_rays(&device, &recon, 3, &cache);

    // Demo gives every point two observations.
    assert_eq!(r.track_ray_count, 2);
    assert!(r.track_ray_edge_buffer.is_some());
}

#[test]
fn track_rays_for_a_finite_point_stop_near_the_scene() {
    let recon = demo(4);
    let cache = sift_cache(recon.images.len(), 8);

    let edges = track_ray_edges(&recon, 0, &cache);

    // A finite point terminates each ray at the closest approach to it, which
    // lies inside the camera cloud — decisively shorter than the fixed
    // 2x-scene-extent an infinity point would produce (asserted below).
    assert_eq!(edges.len(), 2);
    let diagonal = camera_cloud_diagonal(&recon);
    for e in &edges {
        let len = edge_length(e);
        assert!(
            len > 0.0 && len < diagonal,
            "finite ray length {len} should be inside the scene (diagonal {diagonal})",
        );
    }
}

#[test]
fn upload_track_rays_skips_observations_with_no_cached_features() {
    let (device, _queue) = device();
    let recon = demo(4);
    let mut r = SceneRenderer::new();

    // Empty cache — a missing `.sift` companion must draw no ray at all
    // rather than a misleading one.
    r.upload_track_rays(&device, &recon, 0, &HashMap::new());

    assert_eq!(r.track_ray_count, 0);
    assert!(r.track_ray_edge_buffer.is_none());
}

#[test]
fn upload_track_rays_skips_feature_indexes_past_a_truncated_cache() {
    let (device, _queue) = device();
    let recon = demo(4);
    // Cache present but holding zero features, so every lookup misses.
    let cache = sift_cache(recon.images.len(), 0);
    let mut r = SceneRenderer::new();

    r.upload_track_rays(&device, &recon, 0, &cache);

    assert_eq!(r.track_ray_count, 0);
}

#[test]
fn upload_track_rays_reads_inline_keypoints_without_a_sift_cache() {
    let (device, _queue) = device();
    let recon = with_embedded_keypoints(demo(4));
    let mut r = SceneRenderer::new();

    r.upload_track_rays(&device, &recon, 0, &HashMap::new());

    // Embedded-patch reconstructions carry keypoints inline, so no cache is
    // consulted and the rays still build.
    assert_eq!(r.track_ray_count, 2);
}

#[test]
fn track_rays_for_a_point_at_infinity_run_to_twice_the_scene_extent() {
    let mut recon = demo(4);
    recon.points[0].w = 0.0;
    assert!(recon.points[0].is_at_infinity());
    let cache = sift_cache(recon.images.len(), 8);

    let edges = track_ray_edges(&recon, 0, &cache);

    // An infinity point's stored position is a unit direction at the origin,
    // which would project behind every camera and collapse each ray to zero
    // length. Instead each ray runs outward along its own bearing to
    // INFINITY_RAY_SCENE_MULTIPLE (2.0) x the camera-cloud diagonal.
    assert_eq!(edges.len(), 2);
    let expected = 2.0 * camera_cloud_diagonal(&recon);
    assert!(expected > 0.0);
    for e in &edges {
        let len = edge_length(e);
        assert!(
            (len - expected).abs() < 1e-3 * expected,
            "infinity ray length {len} should be {expected}",
        );
    }
}

#[test]
fn clear_track_rays_drops_the_buffer() {
    let (device, _queue) = device();
    let recon = demo(4);
    let cache = sift_cache(recon.images.len(), 8);
    let mut r = SceneRenderer::new();
    r.upload_track_rays(&device, &recon, 0, &cache);
    assert_eq!(r.track_ray_count, 2);

    r.clear_track_rays();

    assert_eq!(r.track_ray_count, 0);
    assert!(r.track_ray_edge_buffer.is_none());
}

// ── background image ────────────────────────────────────────────────────

/// Name of the background image written into the fixture directory.
const BG_NAME: &str = "bg.png";

/// Write an 8x4 PNG into a unique temp dir and point the reconstruction's
/// first image at it. The returned `TempDir` cleans up on drop, including
/// when a test panics.
fn recon_with_real_bg_image(tag: &str) -> (SfmrReconstruction, TempDir) {
    let dir = TempDir::new(tag);
    image::RgbImage::new(8, 4)
        .save(dir.path().join(BG_NAME))
        .expect("write test png");

    let mut recon = demo(4);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.images[0].name = BG_NAME.to_string();
    (recon, dir)
}

#[test]
fn upload_bg_image_loads_a_real_image_and_builds_its_mesh() {
    let (device, queue) = device();
    let (recon, dir) = recon_with_real_bg_image("load");
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, 0);

    let texture = r.bg_image_texture.as_ref().expect("bg texture");
    assert_eq!((texture.width(), texture.height()), (8, 4));
    assert_eq!(r.bg_image_loaded_index, Some(0));
    // Pinhole cameras get the coarsest background mesh: two triangles.
    let n = BG_PINHOLE_SUBDIVISIONS + 1;
    assert_eq!(
        r.bg_image_distorted_index_count,
        ((n - 1) * (n - 1) * 6) as u32
    );
    assert!(r.bg_image_distorted_vertex_buffer.is_some());

    drop(dir);
}

#[test]
fn upload_bg_image_skips_reloading_the_same_image() {
    let (device, queue) = device();
    let (recon, dir) = recon_with_real_bg_image("reload");
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, 0);
    assert_eq!(r.bg_image_texture.as_ref().map(|t| t.width()), Some(8));

    // Replace the file with a differently-sized image and ask for the same
    // index again. Deleting it instead would prove nothing: the load failure
    // path also returns before mutating any field, so the assertions below
    // would hold either way. A *readable* image of a new size is only ignored
    // if the already-loaded early-out actually fires.
    image::RgbImage::new(16, 8)
        .save(dir.path().join(BG_NAME))
        .expect("overwrite test png");
    r.upload_bg_image(&device, &queue, &recon, 0);

    let texture = r.bg_image_texture.as_ref().expect("bg texture");
    assert_eq!(
        (texture.width(), texture.height()),
        (8, 4),
        "the second call must have been skipped, not reloaded at the new size",
    );
    assert_eq!(r.bg_image_loaded_index, Some(0));

    drop(dir);
}

#[test]
fn upload_bg_image_ignores_an_unreadable_image() {
    let (device, queue) = device();
    // demo's workspace_dir is empty and its image names don't exist on disk.
    let recon = demo(4);
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, 0);

    assert!(r.bg_image_texture.is_none());
    assert_eq!(r.bg_image_loaded_index, None);
}

#[test]
fn upload_bg_image_ignores_an_out_of_range_index() {
    let (device, queue) = device();
    let recon = demo(4);
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, 999);

    assert!(r.bg_image_texture.is_none());
    assert_eq!(r.bg_image_loaded_index, None);
}

#[test]
fn clear_bg_image_resets_every_background_field() {
    let (device, queue) = device();
    let (recon, dir) = recon_with_real_bg_image("clear");
    let mut r = SceneRenderer::new();
    r.upload_bg_image(&device, &queue, &recon, 0);
    assert!(r.bg_image_texture.is_some());

    r.clear_bg_image();

    assert!(r.bg_image_texture.is_none());
    assert!(r.bg_image_bind_group.is_none());
    assert_eq!(r.bg_image_loaded_index, None);
    assert!(r.bg_image_distorted_vertex_buffer.is_none());
    assert!(r.bg_image_distorted_index_buffer.is_none());
    assert_eq!(r.bg_image_distorted_index_count, 0);

    drop(dir);
}
