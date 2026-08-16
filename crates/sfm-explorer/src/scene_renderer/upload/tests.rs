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
use sfmtool_core::{RotQuaternion, Se3Transform, SfmrReconstruction};

use super::super::gpu_types::{
    BG_PINHOLE_SUBDIVISIONS, DISTORTION_SUBDIVISIONS, FISHEYE_SUBDIVISIONS, THUMBNAIL_SIZE,
};
use super::super::picking::{PickTarget, PICK_TAG_FRUSTUM, PICK_TAG_NONE, PICK_TAG_POINT};
use super::super::recon::{NodeDisplay, ReconResources};
use super::super::uniforms::recon_uniforms;
use super::super::SceneRenderer;
use super::track_rays::track_ray_edges;
use crate::scene::{ImageRef, NodeTint, PointRef, ReconId, TINT_PALETTE};
use crate::state::CachedSiftFeatures;

// ── Fixtures ────────────────────────────────────────────────────────────

/// The reconstruction identity every fixture here shares. Cache keys and the
/// refs handed to the renderer have to agree on one, and a chosen id makes that
/// visible at the call site.
const RECON: ReconId = ReconId::from_raw(0);

/// A second reconstruction, for the tests that load two at once.
const OTHER: ReconId = ReconId::from_raw(1);

/// The node transform an unaligned node carries: what most of these tests want,
/// since they are about upload arithmetic rather than about where a node sits.
fn identity() -> Se3Transform {
    Se3Transform::identity()
}

/// The resource bundle `RECON`'s uploads land in.
fn bundle(r: &SceneRenderer) -> &ReconResources {
    bundle_of(r, RECON)
}

fn bundle_of(r: &SceneRenderer, id: ReconId) -> &ReconResources {
    r.recons.get(&id).expect("a bundle for this reconstruction")
}

/// How many patch surfels `RECON`'s bundle would draw.
fn patch_count(r: &SceneRenderer) -> u32 {
    bundle(r).patch.as_ref().map_or(0, |p| p.count)
}

/// The reconstruction's `n`-th point.
fn point(n: usize) -> PointRef {
    PointRef::new(RECON, n)
}

/// The reconstruction's `n`-th image.
fn image(n: usize) -> ImageRef {
    ImageRef::new(RECON, n)
}

/// A `wgpu::Device` on the noop backend, with `limits` as the *device* limits.
/// The noop adapter reports maximally permissive limits, so any request is
/// granted — which is what lets a test drive the clamping paths.
fn device_with_limits(limits: wgpu::Limits) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
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
fn sift_cache(images: usize, features: usize) -> HashMap<ImageRef, CachedSiftFeatures> {
    (0..images)
        .map(|i| {
            (
                image(i),
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

    r.upload_points(&device, RECON, &recon);

    let b = bundle(&r);
    assert_eq!(b.point_count, 64);
    assert!(b.point_instance_buffer.is_some());
    // Demo points sit on a unit sphere offset to z+1, so both the splat size
    // and the bounding sphere must come out positive and finite.
    assert!(b.auto_point_size > 0.0 && b.auto_point_size.is_finite());
    let (_, radius) = r.scene_bounds();
    assert!(radius > 0.0 && radius.is_finite());
    let nn = b
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

    r.upload_points(&device, RECON, &recon);

    assert_eq!(bundle(&r).point_count, 0);
}

// ── frustums ────────────────────────────────────────────────────────────

#[test]
fn upload_frustums_emits_eight_edges_per_pinhole_camera() {
    let (device, _queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    let b = bundle(&r);
    // 4 apex→corner side edges + 4 base edges around the far face.
    assert_eq!(b.frustum_edge_count, DEMO_IMAGES * 8);
    assert_eq!(b.frustum_image_count, DEMO_IMAGES);
    assert!(b.frustum_color_buffer.is_some());
    // No thumbnail atlas yet, so no image quads are built.
    assert_eq!(b.image_quad_count, 0);
    assert_eq!(b.distorted_quad_index_count, 0);
}

#[test]
fn upload_frustums_builds_pinhole_image_quads_once_thumbnails_exist() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &recon);
    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    let b = bundle(&r);
    assert_eq!(b.image_quad_count, DEMO_IMAGES);
    assert!(b.image_quad_instance_buffer.is_some());
    // Pinhole cameras take the flat-quad path, not the tessellated one.
    assert_eq!(b.distorted_quad_index_count, 0);
    assert!(b.distorted_quad_vertex_buffer.is_none());
}

#[test]
fn upload_frustums_tessellates_fisheye_cameras() {
    let (device, queue) = device();
    let recon = with_camera_model(demo(16), fisheye());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &recon);
    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    // n×n grid: 4 side edges + 4 boundary walks of (n-1) segments each.
    let n = FISHEYE_SUBDIVISIONS + 1;
    let b = bundle(&r);
    assert_eq!(b.frustum_edge_count, DEMO_IMAGES * (4 + 4 * (n - 1)) as u32);
    // Two triangles per grid cell, six indices per cell.
    assert_eq!(
        b.distorted_quad_index_count,
        DEMO_IMAGES * ((n - 1) * (n - 1) * 6) as u32
    );
    // Fisheye takes the tessellated path exclusively.
    assert_eq!(b.image_quad_count, 0);
}

#[test]
fn upload_frustums_tessellates_distorted_cameras() {
    let (device, queue) = device();
    let recon = with_camera_model(demo(16), radial_distorted());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &recon);
    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    let n = DISTORTION_SUBDIVISIONS + 1;
    let b = bundle(&r);
    assert_eq!(b.frustum_edge_count, DEMO_IMAGES * (4 + 4 * (n - 1)) as u32);
    assert_eq!(
        b.distorted_quad_index_count,
        DEMO_IMAGES * ((n - 1) * (n - 1) * 6) as u32
    );
    // A distorted camera must take the tessellated path exclusively, exactly
    // like fisheye — `has_distortion()` and `is_fisheye()` are separate
    // predicates, so a regression could send it down the pinhole quad path.
    assert_eq!(b.image_quad_count, 0);
}

#[test]
fn upload_frustums_replaces_quad_buffers_when_the_camera_model_changes() {
    let (device, queue) = device();
    let fisheye_recon = with_camera_model(demo(16), fisheye());
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &fisheye_recon);
    r.upload_frustums(&device, RECON, &fisheye_recon, 1.0, 1.0);
    assert!(bundle(&r).distorted_quad_index_count > 0);

    // Re-uploading a pinhole reconstruction must drop the stale distorted
    // buffers rather than leave them to be drawn against a new index count.
    let pinhole_recon = demo(16);
    r.upload_frustums(&device, RECON, &pinhole_recon, 1.0, 1.0);

    let b = bundle(&r);
    assert_eq!(b.distorted_quad_index_count, 0);
    assert!(b.distorted_quad_vertex_buffer.is_none());
    assert!(b.distorted_quad_index_buffer.is_none());
    assert_eq!(b.image_quad_count, DEMO_IMAGES);
}

#[test]
fn update_frustum_colors_is_a_no_op_before_any_upload() {
    let (_device, queue) = device();
    let r = SceneRenderer::new();
    // No bundle at all, let alone a color buffer — must return quietly rather
    // than panic.
    r.update_frustum_colors(&queue, RECON, 8, Some(0), Some(1), &[2, 3]);
}

#[test]
fn update_frustum_colors_ignores_a_reconstruction_that_is_not_loaded() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();
    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    // Colors are written into the *owning* node's buffer; an id with no bundle
    // must not fall through to whichever one happens to be loaded.
    r.update_frustum_colors(&queue, OTHER, 8, Some(0), None, &[]);
}

#[test]
fn update_frustum_colors_tolerates_out_of_range_indices() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();
    r.upload_frustums(&device, RECON, &recon, 1.0, 1.0);

    // Every index is past the end; each is individually bounds-checked, so
    // this must not panic or write out of range.
    r.update_frustum_colors(&queue, RECON, 8, Some(99), Some(99), &[99, 100]);
}

// ── thumbnails ──────────────────────────────────────────────────────────

#[test]
fn upload_thumbnails_packs_a_square_ish_atlas_grid() {
    let (device, queue) = device();
    let recon = demo(16);
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &recon);

    // cols = ceil(sqrt(8)) = 3, then rows = ceil(8/3) = 3.
    let b = bundle(&r);
    assert_eq!(b.atlas_cols, 3);
    assert_eq!(b.atlas_rows, 3);
    assert!(b.thumbnail_texture.is_some());
    assert!(b.thumbnail_view.is_some());
    assert!(b.image_quad_uniform_buffer.is_some());
    // A page holds cols × (max_texture_dim / THUMBNAIL_SIZE) cells.
    let cells_per_axis = wgpu::Limits::default().max_texture_dimension_2d / THUMBNAIL_SIZE;
    assert_eq!(b.images_per_page, 3 * cells_per_axis);
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

    r.upload_thumbnails(&device, &queue, RECON, &recon);

    let b = bundle(&r);
    assert_eq!(b.atlas_cols, 2);
    assert_eq!(b.atlas_rows, 2);
    assert_eq!(b.images_per_page, 4);
    // The texture itself must stay inside the limits: wgpu-core would reject
    // the descriptor otherwise, so reaching here at all is the assertion.
    assert!(b.thumbnail_texture.is_some());
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

    r.upload_thumbnails(&device, &queue, RECON, &recon);

    // sqrt would ask for 5 columns, but a page is only 4 cells wide — the
    // texture-dimension budget is what caps the grid in practice. The
    // MAX_ATLAS_COLS cap (128) is looser than this on any real GPU: it binds
    // only above ~16k images *and* a >16384px texture limit, so it is left
    // untested rather than forced with an ~800MB thumbnail array.
    let b = bundle(&r);
    assert_eq!(b.atlas_cols, 4);
    assert_eq!(b.atlas_rows, 4);
    assert_eq!(b.images_per_page, 16);
    // 25 images over 16 per page = 2 pages, and all 25 still fit.
    assert!(b.thumbnail_texture.is_some());
}

#[test]
fn upload_thumbnails_skips_an_imageless_reconstruction() {
    let (device, queue) = device();
    let mut recon = demo(8);
    recon.images.clear();
    recon.thumbnails_y_x_rgb = Array4::zeros((0, 128, 128, 3));
    let mut r = SceneRenderer::new();

    r.upload_thumbnails(&device, &queue, RECON, &recon);

    // Nothing to pack, so not even a bundle is created for it.
    assert!(r
        .recons
        .get(&RECON)
        .is_none_or(|b| b.thumbnail_texture.is_none()));
}

// ── patches ─────────────────────────────────────────────────────────────

#[test]
fn upload_patches_counts_only_points_carrying_a_patch() {
    let (device, queue) = device();
    let present = [true, false, true, true, false, true, true];
    let recon = with_patches(demo(present.len()), 16, &present, None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    let b = bundle(&r);
    assert_eq!(patch_count(&r), 5);
    let patch = b.patch.as_ref().expect("patch resources");
    // cols = ceil(sqrt(5)) = 3, rows = ceil(5/3) = 2, at 16px per tile.
    assert_eq!(patch.atlas_cols, 3);
    assert_eq!(patch.atlas_rows, 2);
    assert_eq!(
        (patch.atlas_texture.width(), patch.atlas_texture.height()),
        (3 * 16, 2 * 16),
    );
}

#[test]
fn upload_patches_uploads_nothing_without_patch_arrays() {
    let (device, queue) = device();
    let recon = demo(8); // demo carries no patch frames
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    let b = bundle(&r);
    assert_eq!(patch_count(&r), 0);
    assert!(b.patch.is_none());
}

#[test]
fn upload_patches_skips_frames_without_bitmaps() {
    let (device, queue) = device();
    let mut recon = with_patches(demo(4), 16, &[true; 4], None, None);
    recon.patch_bitmaps_y_x_rgba = None; // frames present, bitmaps absent
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    // v1 renders textured patches only.
    assert_eq!(patch_count(&r), 0);
}

#[test]
fn upload_patches_rejects_non_square_bitmaps() {
    let (device, queue) = device();
    // 16 rows × 17 columns — an in-memory reconstruction is not shape-verified.
    let recon = with_patches(demo(4), 16, &[true; 4], None, Some(17));
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    let b = bundle(&r);
    assert_eq!(patch_count(&r), 0);
    assert!(b.patch.is_none());
}

#[test]
fn upload_patches_rejects_zero_resolution_bitmaps() {
    let (device, queue) = device();
    let recon = with_patches(demo(4), 0, &[true; 4], None, Some(0));
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    assert_eq!(patch_count(&r), 0);
}

#[test]
fn upload_patches_rejects_bitmaps_larger_than_the_texture_limit() {
    let (device, queue) = device_with_limits(wgpu::Limits {
        max_texture_dimension_2d: 256,
        ..wgpu::Limits::default()
    });
    let recon = with_patches(demo(2), 512, &[true; 2], None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    // Skipped rather than passed to wgpu, which would be a validation error.
    assert_eq!(patch_count(&r), 0);
}

#[test]
fn upload_patches_bounds_the_row_scan_by_the_shortest_array() {
    let (device, queue) = device();
    // 6 points and 6 frame rows, but only 2 bitmap rows.
    let recon = with_patches(demo(6), 16, &[true; 6], Some(2), None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &recon);

    assert_eq!(patch_count(&r), 2);
}

#[test]
fn upload_patches_clears_stale_patches_when_reloading_without_them() {
    let (device, queue) = device();
    let with = with_patches(demo(4), 16, &[true; 4], None, None);
    let mut r = SceneRenderer::new();

    r.upload_patches(&device, &queue, RECON, &with);
    assert_eq!(patch_count(&r), 4);

    r.upload_patches(&device, &queue, RECON, &demo(4));

    let b = bundle(&r);
    assert_eq!(patch_count(&r), 0);
    // The whole patch half of the bundle goes, atlas and bind group with it.
    assert!(b.patch.is_none());
}

// ── resource bundles + pick bases ───────────────────────────────────────
//
// Two reconstructions at once is something the UI cannot do until phase 3 —
// which is exactly why the renderer has to be tested with two here.

/// Upload one node's points and frustums, the two counts pick bases are cut
/// from.
fn upload_node(r: &mut SceneRenderer, device: &wgpu::Device, id: ReconId, points: usize) {
    let recon = demo(points);
    r.upload_points(device, id, &recon);
    r.upload_frustums(device, id, &recon, 1.0, 1.0);
}

#[test]
fn two_nodes_upload_into_two_independent_bundles() {
    let (device, queue) = device();
    let mut r = SceneRenderer::new();

    let first = demo(12);
    let second = with_camera_model(demo(30), fisheye());
    r.upload_points(&device, RECON, &first);
    r.upload_thumbnails(&device, &queue, RECON, &first);
    r.upload_frustums(&device, RECON, &first, 1.0, 1.0);
    r.upload_points(&device, OTHER, &second);
    r.upload_thumbnails(&device, &queue, OTHER, &second);
    r.upload_frustums(&device, OTHER, &second, 1.0, 1.0);

    assert_eq!(r.recons.len(), 2);
    assert_eq!(bundle_of(&r, RECON).point_count, 12);
    assert_eq!(bundle_of(&r, OTHER).point_count, 30);
    // Each node keeps its own geometry: the pinhole node took the flat-quad
    // path while the fisheye node tessellated, and neither overwrote the other.
    assert_eq!(bundle_of(&r, RECON).image_quad_count, DEMO_IMAGES);
    assert_eq!(bundle_of(&r, RECON).distorted_quad_index_count, 0);
    assert_eq!(bundle_of(&r, OTHER).image_quad_count, 0);
    assert!(bundle_of(&r, OTHER).distorted_quad_index_count > 0);
}

#[test]
fn releasing_a_node_drops_only_its_bundle() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);

    r.retain_nodes(|id| id != RECON);

    assert_eq!(r.recons.len(), 1);
    assert!(!r.recons.contains_key(&RECON));
    assert_eq!(bundle_of(&r, OTHER).point_count, 30);
    // The survivor slides down to the bottom of both index spaces.
    assert_eq!(bundle_of(&r, OTHER).point_pick_base, 0);
    assert_eq!(bundle_of(&r, OTHER).image_pick_base, 0);
}

#[test]
fn retain_nodes_releases_everything_that_left_the_scene() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);

    r.retain_nodes(|id| id == OTHER);

    assert_eq!(r.recons.len(), 1);
    assert!(r.recons.contains_key(&OTHER));
}

#[test]
fn pick_bases_are_contiguous_and_non_overlapping() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);

    let first = bundle_of(&r, RECON);
    let second = bundle_of(&r, OTHER);
    // Bases are handed out in ReconId order, and each range starts exactly
    // where the previous one ended — no gaps, no overlap.
    assert_eq!(first.point_pick_base, 0);
    assert_eq!(second.point_pick_base, first.point_count);
    assert_eq!(first.image_pick_base, 0);
    assert_eq!(second.image_pick_base, first.frustum_image_count);
    // The two index spaces are cut independently: 12 points but only 8 images
    // in the first node, so the second node's bases differ.
    assert_eq!(second.point_pick_base, 12);
    assert_eq!(second.image_pick_base, DEMO_IMAGES);
}

#[test]
fn a_pick_id_decodes_back_to_the_node_that_drew_it() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);

    // Walk both ends of every range, which is where an off-by-one in the base
    // arithmetic would show up.
    for (id, points) in [(RECON, 12u32), (OTHER, 30u32)] {
        let b = bundle_of(&r, id);
        for local in [0, points - 1] {
            let pick_id = PICK_TAG_POINT | (b.point_pick_base + local);
            assert_eq!(
                r.decode_pick(pick_id),
                Some(PickTarget::Point(PointRef::new(id, local as usize))),
            );
        }
        for local in [0, DEMO_IMAGES - 1] {
            let pick_id = PICK_TAG_FRUSTUM | (b.image_pick_base + local);
            assert_eq!(
                r.decode_pick(pick_id),
                Some(PickTarget::Image(ImageRef::new(id, local as usize))),
            );
        }
    }

    // One past the last node's range belongs to nobody.
    let past_the_end = bundle_of(&r, OTHER).point_pick_base + 30;
    assert_eq!(r.decode_pick(PICK_TAG_POINT | past_the_end), None);
}

#[test]
fn a_non_interactive_node_writes_a_zero_pickable_flag_and_nothing_else_changes() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);

    // The reference node: display-only, but still visible and still uploaded.
    r.set_node_display(
        RECON,
        NodeDisplay {
            interactive: false,
            ..NodeDisplay::default()
        },
    );

    let quiet = recon_uniforms(bundle_of(&r, RECON), 1.0, true);
    let live = recon_uniforms(bundle_of(&r, OTHER), 1.0, true);
    assert_eq!(
        quiet.pickable, 0,
        "the interaction cursor did not reach the GPU"
    );
    assert_eq!(
        live.pickable, 1,
        "the other node stopped being pickable too"
    );
    // Everything else about the node is untouched: it still draws, at its own
    // splat size, out of its own slice of the pick space.
    assert_eq!(quiet.point_size, bundle_of(&r, RECON).auto_point_size);
    assert_eq!(quiet.point_pick_base, bundle_of(&r, RECON).point_pick_base);
    assert_eq!(quiet.image_pick_base, bundle_of(&r, RECON).image_pick_base);

    // What its shaders write with `pickable == 0` is the background value, and
    // that decodes to nothing — so a readback over the node produces neither
    // hover nor selection, while the interactive node still resolves.
    assert_eq!(r.decode_pick(PICK_TAG_NONE), None);
    let live_id = PICK_TAG_POINT | bundle_of(&r, OTHER).point_pick_base;
    assert_eq!(
        r.decode_pick(live_id),
        Some(PickTarget::Point(PointRef::new(OTHER, 0))),
    );
}

#[test]
fn the_infinity_toggle_is_the_and_of_the_global_switch_and_the_node_switch() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);

    for (global, node, expected) in [
        (true, true, 1.0),
        (true, false, 0.0),
        (false, true, 0.0),
        (false, false, 0.0),
    ] {
        r.set_node_display(
            RECON,
            NodeDisplay {
                show_points_at_infinity: node,
                ..NodeDisplay::default()
            },
        );
        let uniforms = recon_uniforms(bundle(&r), 1.0, global);
        assert_eq!(
            uniforms.show_infinity, expected,
            "global={global}, node={node}"
        );
    }
}

#[test]
fn pick_bases_stay_consistent_under_add_and_remove_churn() {
    let (device, _queue) = device();
    let third = ReconId::from_raw(2);
    let mut r = SceneRenderer::new();

    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 30);
    upload_node(&mut r, &device, third, 7);
    // Drop the middle node, then bring another one back — the churn that would
    // strand a stale base if reassignment were incremental.
    r.retain_nodes(|id| id != OTHER);
    upload_node(&mut r, &device, OTHER, 5);
    r.retain_nodes(|id| id != third);

    // Whatever the history, the invariant holds: bases in ReconId order,
    // contiguous from zero, each range exactly as long as its node's count.
    let mut ids: Vec<ReconId> = r.recons.keys().copied().collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![RECON, OTHER]);
    let (mut point_base, mut image_base) = (0, 0);
    for id in ids {
        let b = bundle_of(&r, id);
        assert_eq!(b.point_pick_base, point_base, "point base of {id:?}");
        assert_eq!(b.image_pick_base, image_base, "image base of {id:?}");
        point_base += b.point_count;
        image_base += b.frustum_image_count;
    }

    // And decode still lands in the right node afterwards.
    assert_eq!(
        r.decode_pick(PICK_TAG_POINT | 12),
        Some(PickTarget::Point(PointRef::new(OTHER, 0))),
    );
    assert_eq!(
        r.decode_pick(PICK_TAG_FRUSTUM | DEMO_IMAGES),
        Some(PickTarget::Image(ImageRef::new(OTHER, 0))),
    );
}

#[test]
fn the_length_scale_seed_is_the_smallest_across_loaded_nodes() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    assert_eq!(r.length_scale_seed(), None);

    upload_node(&mut r, &device, RECON, 12);
    let single = r.length_scale_seed().expect("one node seeds a scale");
    upload_node(&mut r, &device, OTHER, 4000);

    // A denser cloud has a smaller nearest-neighbour spacing, so adding it can
    // only lower the seed — and the union must never exceed either node's own.
    let both = r.length_scale_seed().expect("two nodes still seed a scale");
    assert!(
        both <= single,
        "{both} should not exceed the single-node {single}"
    );
}

// ── tint ────────────────────────────────────────────────────────────────

#[test]
fn a_nodes_tint_reaches_its_uniform_block_and_no_one_elses() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 12);

    // An untinted node writes the "original colors" convention: a == 0, which
    // makes the shaders' `mix(color, tint.rgb, tint.a)` the identity.
    assert_eq!(recon_uniforms(bundle(&r), 1.0, true).tint_color, [0.0; 4]);

    let color = &TINT_PALETTE[4]; // Blue
    r.set_node_display(
        RECON,
        NodeDisplay {
            tint: NodeTint::Tint(color),
            ..NodeDisplay::default()
        },
    );

    let tinted = recon_uniforms(bundle_of(&r, RECON), 1.0, true).tint_color;
    assert_eq!(
        tinted,
        [
            color.rgb[0] as f32 / 255.0,
            color.rgb[1] as f32 / 255.0,
            color.rgb[2] as f32 / 255.0,
            crate::scene::TINT_STRENGTH,
        ],
    );
    assert_eq!(
        recon_uniforms(bundle_of(&r, OTHER), 1.0, true).tint_color,
        [0.0; 4],
        "tinting one node tinted the other's uniform block",
    );
}

/// `frustum.wgsl` decides what to tint from the alpha in the per-image color
/// buffer: below full alpha is the node's own frustum color and takes the tint,
/// full alpha is a highlight and keeps the color it was chosen for. That is a
/// contract with this CPU-side writer, so assert it here.
#[test]
fn every_frustum_highlight_color_is_written_at_the_full_alpha_the_tint_exempts() {
    use super::frustums::{frustum_colors, FRUSTUM_ALPHA_DEFAULT, FRUSTUM_ALPHA_HIGHLIGHT};

    let alpha = |packed: u32| packed >> 24;
    let colors = frustum_colors(6, Some(1), Some(5), &[2, 3]);

    // Plain frustums: tintable.
    for &index in &[0usize, 4] {
        assert_eq!(alpha(colors[index]), FRUSTUM_ALPHA_DEFAULT);
        assert!(alpha(colors[index]) < FRUSTUM_ALPHA_HIGHLIGHT);
    }
    // The selected camera and the selected point's track cameras: highlights,
    // and so exempt from the tint.
    for &index in &[1usize, 2, 3] {
        assert_eq!(
            alpha(colors[index]),
            FRUSTUM_ALPHA_HIGHLIGHT,
            "image {index} is a highlight but would be tinted",
        );
    }
    // The camera being viewed through is discarded outright.
    assert_eq!(colors[5], 0);
}

// ── effective visibility (eye AND solo) ─────────────────────────────────

/// The draw filter and the bounds union read one flag, which `app.rs` composes
/// from the node's eye and the scene's solo. Hiding a node here is what soloing
/// its neighbour does.
#[test]
fn a_hidden_node_drops_out_of_every_pass_and_out_of_the_bounds() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 256);
    upload_node(&mut r, &device, OTHER, 256);
    // Move the second node well away, so a bounds union that still included it
    // could not possibly match the first node's alone.
    r.set_node_transform(OTHER, similarity(1.0));

    assert_eq!(r.drawn(|b| b.display.show_points).count(), 2);
    assert_eq!(r.drawn(|b| b.display.show_cameras).count(), 2);
    let both = r.scene_bounds();

    r.set_node_display(
        OTHER,
        NodeDisplay {
            visible: false,
            ..NodeDisplay::default()
        },
    );

    assert_eq!(
        r.drawn(|b| b.display.show_points).count(),
        1,
        "a node that is not visible was still drawn"
    );
    assert_eq!(r.drawn(|b| b.display.show_cameras).count(), 1);
    let solo_bounds = r.scene_bounds();
    let expected = bundle_of(&r, RECON)
        .world_bounds()
        .expect("uploaded bounds");
    assert!(
        (solo_bounds.0 - expected.0).norm() < 1e-9 && (solo_bounds.1 - expected.1).abs() < 1e-9,
        "the bounds still enclose the hidden node: {solo_bounds:?} vs {expected:?}",
    );
    assert!(solo_bounds.1 < both.1);

    // A node's own group eye still narrows what is drawn on top of that.
    r.set_node_display(
        RECON,
        NodeDisplay {
            show_points: false,
            ..NodeDisplay::default()
        },
    );
    assert_eq!(r.drawn(|b| b.display.show_points).count(), 0);
    assert_eq!(r.drawn(|b| b.display.show_cameras).count(), 1);
}

/// With everything hidden the union falls back to all loaded nodes rather than
/// collapsing to a unit sphere at the origin — switching the last eye off, or
/// soloing a node and then hiding it, must not fling the camera.
#[test]
fn the_all_hidden_fallback_still_frames_the_loaded_nodes() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 256);
    let visible = r.scene_bounds();

    r.set_node_display(
        RECON,
        NodeDisplay {
            visible: false,
            ..NodeDisplay::default()
        },
    );

    assert_eq!(r.drawn(|b| b.display.show_points).count(), 0);
    let hidden = r.scene_bounds();
    assert!((hidden.0 - visible.0).norm() < 1e-9 && (hidden.1 - visible.1).abs() < 1e-9);
}

// ── node transforms ─────────────────────────────────────────────────────

/// A similarity with all three parts non-trivial, mirroring `align/tests.rs`.
fn similarity(scale: f64) -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_axis_angle(nalgebra::Vector3::new(0.0, 0.0, 1.0), 0.5).unwrap(),
        nalgebra::Vector3::new(4.0, -2.5, 1.25),
        scale,
    )
}

#[test]
fn the_node_transform_reaches_the_gpu_as_the_model_matrix() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 12);
    upload_node(&mut r, &device, OTHER, 12);

    // The identity is what an unaligned node draws with, and is also the
    // baseline the aligned node below has to differ from.
    let before = recon_uniforms(bundle_of(&r, RECON), 1.0, true).model;
    assert_eq!(
        before,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );

    let t = similarity(2.0);
    r.set_node_transform(RECON, t.clone());
    let model = recon_uniforms(bundle_of(&r, RECON), 1.0, true).model;

    // Read the matrix back the way a vertex shader would — columns times the
    // point's components, plus the translation column — and check it against
    // the transform it was built from.
    let probe = nalgebra::Point3::new(0.7, -1.3, 2.1);
    let expected = t.apply_to_point(&probe);
    let mapped: Vec<f32> = (0..3)
        .map(|row| {
            model[0][row] * probe.x as f32
                + model[1][row] * probe.y as f32
                + model[2][row] * probe.z as f32
                + model[3][row]
        })
        .collect();
    for (got, want) in mapped.iter().zip([expected.x, expected.y, expected.z]) {
        assert!(
            (*got as f64 - want).abs() < 1e-5,
            "model matrix maps {probe:?} to {mapped:?}, not {expected:?}",
        );
    }
    // The last row stays affine, so `w` survives untouched and a direction fed
    // in with `w = 0` comes out rotated with the translation dropped.
    assert_eq!(
        [model[0][3], model[1][3], model[2][3], model[3][3]],
        [0.0, 0.0, 0.0, 1.0]
    );

    // And only the node it was set on moved.
    assert_eq!(
        recon_uniforms(bundle_of(&r, OTHER), 1.0, true).model,
        before
    );
}

#[test]
fn an_aligned_nodes_background_image_stays_in_front_of_the_camera() {
    use super::super::distorted_mesh::generate_bg_distorted_mesh;
    use super::super::uniforms::bg_image_uniforms;

    let recon = SfmrReconstruction::demo(64);
    let image = &recon.images[3];
    let camera = &recon.cameras[image.camera_index as usize];

    // A half-turn: the alignment between two solves that disagree about which
    // way the scene faces, and the one that makes a background image left in
    // the node's own frame end up directly behind the viewer.
    let t = Se3Transform::new(
        RotQuaternion::from_axis_angle(nalgebra::Vector3::new(0.0, 0.0, 1.0), std::f64::consts::PI)
            .unwrap(),
        nalgebra::Vector3::new(4.0, -2.5, 1.25),
        2.0,
    );

    // The viewport pose `enter_camera_view` builds for this image on this node.
    let (rotation, centre) = crate::viewer_3d::transformed_pose(image, &t);
    let mut viewport = crate::viewer_3d::ViewportCamera::default();
    viewport.camera.position = centre;
    viewport.camera.orientation = rotation;

    let uniforms = bg_image_uniforms(&viewport, 16.0 / 9.0, &t);
    let r = image.camera_to_world_rotation_flat();
    let (vertices, _) = generate_bg_distorted_mesh(camera, &r, BG_PINHOLE_SUBDIVISIONS);

    // `clip.w` is `-z_view` under the reversed-Z projection, so it is positive
    // exactly when a vertex is in front of the camera.
    let clip_w = |uniforms: &super::super::gpu_types::BgImageUniforms, d: [f32; 3]| {
        let world: Vec<f32> = (0..4)
            .map(|row| {
                uniforms.model[0][row] * d[0]
                    + uniforms.model[1][row] * d[1]
                    + uniforms.model[2][row] * d[2]
            })
            .collect();
        (0..4)
            .map(|i| uniforms.view_proj[i][3] * world[i])
            .sum::<f32>()
    };

    for v in &vertices {
        assert!(
            clip_w(&uniforms, v.position) > 0.0,
            "background vertex {:?} is behind the camera it is viewed through",
            v.position,
        );
    }

    // And it is the node transform that puts it there: the same rays without it
    // — the mesh drawn in the node's own frame — face the other way entirely.
    let unmodelled = super::super::gpu_types::BgImageUniforms {
        model: bg_image_uniforms(&viewport, 16.0 / 9.0, &Se3Transform::identity()).model,
        ..uniforms
    };
    for v in &vertices {
        assert!(
            clip_w(&unmodelled, v.position) < 0.0,
            "the fixture is meant to be a half-turn away from the camera",
        );
    }
}

#[test]
fn a_scaled_node_scales_its_splat_size_with_it() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 64);
    let native = recon_uniforms(bundle(&r), 1.0, true).point_size;

    r.set_node_transform(RECON, similarity(3.0));

    // The splat is billboarded in world space *after* the model matrix, so it
    // has to be scaled here or a magnified node would draw pinprick points.
    let scaled = recon_uniforms(bundle(&r), 1.0, true).point_size;
    assert!(
        (scaled - native * 3.0).abs() < 1e-6 * native.max(1.0),
        "{scaled} should be 3x the node-native {native}",
    );
    // The global HUD multiplier still multiplies on top of it.
    assert!(
        (recon_uniforms(bundle(&r), 2.0, true).point_size - scaled * 2.0).abs()
            < 1e-6 * native.max(1.0),
    );
}

#[test]
fn the_scene_bounds_follow_a_transformed_node() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 256);
    let (centre, radius) = r.scene_bounds();

    let t = similarity(2.0);
    r.set_node_transform(RECON, t.clone());

    let (moved_centre, moved_radius) = r.scene_bounds();
    let expected_centre = t.apply_to_point(&centre);
    assert!(
        (moved_centre - expected_centre).norm() < 1e-9,
        "bounds centre {moved_centre:?} should be the transformed {expected_centre:?}",
    );
    assert!((moved_radius - radius * 2.0).abs() < 1e-9);
}

#[test]
fn the_length_scale_seed_scales_with_the_node_transform() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 256);
    let native = r.length_scale_seed().expect("one node seeds a scale");

    r.set_node_transform(RECON, similarity(4.0));

    // `length_scale` is a world-space length; the node's own nearest-neighbour
    // spacings are not, so a scaled node has to re-seed it. This is what
    // dissolves the shared-frustum-size compromise once two nodes are aligned.
    let scaled = r.length_scale_seed().expect("still seeds a scale");
    assert!(
        (scaled - native * 4.0).abs() < 1e-4 * native,
        "{scaled} should be 4x the node-native {native}",
    );
}

#[test]
fn resetting_a_node_transform_puts_its_bounds_back() {
    let (device, _queue) = device();
    let mut r = SceneRenderer::new();
    upload_node(&mut r, &device, RECON, 256);
    let before = r.scene_bounds();

    r.set_node_transform(RECON, similarity(2.0));
    r.set_node_transform(RECON, Se3Transform::identity());

    let after = r.scene_bounds();
    assert!((after.0 - before.0).norm() < 1e-9);
    assert!((after.1 - before.1).abs() < 1e-9);
}

// ── track rays ──────────────────────────────────────────────────────────

#[test]
fn track_rays_are_built_through_the_owning_nodes_transform() {
    let recon = demo(4);
    let cache = sift_cache(recon.images.len(), 8);
    let t = similarity(2.0);

    let native = track_ray_edges(&recon, point(0), &cache, &identity());
    let moved = track_ray_edges(&recon, point(0), &cache, &t);

    // Track rays are drawn from a shared singleton buffer with no per-recon
    // `model` matrix, so the transform has to be applied on the CPU or the rays
    // would stay behind when the node they belong to moves.
    assert_eq!(native.len(), moved.len());
    for (a, b) in native.iter().zip(moved.iter()) {
        for (raw, mapped) in [(a.endpoint_a, b.endpoint_a), (a.endpoint_b, b.endpoint_b)] {
            let expected = t.apply_to_point(&nalgebra::Point3::new(
                raw[0] as f64,
                raw[1] as f64,
                raw[2] as f64,
            ));
            let d = (nalgebra::Point3::new(mapped[0] as f64, mapped[1] as f64, mapped[2] as f64)
                - expected)
                .norm();
            assert!(d < 1e-4, "endpoint {mapped:?} should be {expected:?}");
        }
    }
}

#[test]
fn upload_track_rays_emits_one_ray_per_cached_observation() {
    let (device, _queue) = device();
    let recon = demo(4);
    // Point 3, not 0: demo sets feature_indexes[i] = i, so a non-zero point
    // makes the obs_start + k -> feature_indexes -> cache lookup chain
    // load-bearing rather than always resolving index 0.
    let cache = sift_cache(recon.images.len(), 8);
    let mut r = SceneRenderer::new();

    r.upload_track_rays(&device, &recon, point(3), &cache, &identity());

    // Demo gives every point two observations.
    assert_eq!(r.track_ray_count, 2);
    assert!(r.track_ray_edge_buffer.is_some());
}

#[test]
fn track_rays_for_a_finite_point_stop_near_the_scene() {
    let recon = demo(4);
    let cache = sift_cache(recon.images.len(), 8);

    let edges = track_ray_edges(&recon, point(0), &cache, &identity());

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
    r.upload_track_rays(&device, &recon, point(0), &HashMap::new(), &identity());

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

    r.upload_track_rays(&device, &recon, point(0), &cache, &identity());

    assert_eq!(r.track_ray_count, 0);
}

#[test]
fn upload_track_rays_reads_inline_keypoints_without_a_sift_cache() {
    let (device, _queue) = device();
    let recon = with_embedded_keypoints(demo(4));
    let mut r = SceneRenderer::new();

    r.upload_track_rays(&device, &recon, point(0), &HashMap::new(), &identity());

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

    let edges = track_ray_edges(&recon, point(0), &cache, &identity());

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
    r.upload_track_rays(&device, &recon, point(0), &cache, &identity());
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

    r.upload_bg_image(&device, &queue, &recon, image(0));

    let texture = r.bg_image_texture.as_ref().expect("bg texture");
    assert_eq!((texture.width(), texture.height()), (8, 4));
    assert_eq!(r.bg_image_loaded, Some(image(0)));
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

    r.upload_bg_image(&device, &queue, &recon, image(0));
    assert_eq!(r.bg_image_texture.as_ref().map(|t| t.width()), Some(8));

    // Replace the file with a differently-sized image and ask for the same
    // index again. Deleting it instead would prove nothing: the load failure
    // path also returns before mutating any field, so the assertions below
    // would hold either way. A *readable* image of a new size is only ignored
    // if the already-loaded early-out actually fires.
    image::RgbImage::new(16, 8)
        .save(dir.path().join(BG_NAME))
        .expect("overwrite test png");
    r.upload_bg_image(&device, &queue, &recon, image(0));

    let texture = r.bg_image_texture.as_ref().expect("bg texture");
    assert_eq!(
        (texture.width(), texture.height()),
        (8, 4),
        "the second call must have been skipped, not reloaded at the new size",
    );
    assert_eq!(r.bg_image_loaded, Some(image(0)));

    drop(dir);
}

#[test]
fn upload_bg_image_ignores_an_unreadable_image() {
    let (device, queue) = device();
    // demo's workspace_dir is empty and its image names don't exist on disk.
    let recon = demo(4);
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, image(0));

    assert!(r.bg_image_texture.is_none());
    assert_eq!(r.bg_image_loaded, None);
}

#[test]
fn upload_bg_image_ignores_an_out_of_range_index() {
    let (device, queue) = device();
    let recon = demo(4);
    let mut r = SceneRenderer::new();

    r.upload_bg_image(&device, &queue, &recon, image(999));

    assert!(r.bg_image_texture.is_none());
    assert_eq!(r.bg_image_loaded, None);
}

#[test]
fn clear_bg_image_resets_every_background_field() {
    let (device, queue) = device();
    let (recon, dir) = recon_with_real_bg_image("clear");
    let mut r = SceneRenderer::new();
    r.upload_bg_image(&device, &queue, &recon, image(0));
    assert!(r.bg_image_texture.is_some());

    r.clear_bg_image();

    assert!(r.bg_image_texture.is_none());
    assert!(r.bg_image_bind_group.is_none());
    assert_eq!(r.bg_image_loaded, None);
    assert!(r.bg_image_distorted_vertex_buffer.is_none());
    assert!(r.bg_image_distorted_index_buffer.is_none());
    assert_eq!(r.bg_image_distorted_index_count, 0);

    drop(dir);
}
