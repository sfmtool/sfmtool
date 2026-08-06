// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Point Track Detail panel.
//!
//! egui needs no GPU to lay out a frame, so the whole panel runs through
//! `Context::run_ui` here — `show` really does prepare its data, paint the
//! header, and walk every row of the table. The assertions target what the
//! panel *decides* (which observations it prepared, what it measured for each,
//! which textures it cached, what it reports back to the dock) rather than
//! pixels; the painting itself is covered only in the sense that a layout or
//! borrow mistake would panic the frame.
//!
//! Note that `SfmrReconstruction::demo` is deliberately plain — flat image
//! names, an empty content hash, no patches — so tests that care about those
//! must enrich it first (`with_nested_image_paths`, `with_content_hash`,
//! `with_embedded_patches`) or they assert nothing.

use std::collections::HashMap;

use ndarray::{Array2, Array4};
use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::reconstruction::ObservationSource;
use sfmtool_core::SfmrReconstruction;

use super::table::format_feature_size;
use super::{PointTrackDetail, PointTrackDetailResponse};
use crate::platform::ScrollInput;
use crate::scene::{ImageRef, PointRef, ReconId};
use crate::state::CachedSiftFeatures;

// ── Fixtures ────────────────────────────────────────────────────────────

/// The reconstruction identity every fixture here shares. The panel keys its
/// caches by it, so the fixtures and the assertions have to agree on one.
const RECON: ReconId = ReconId::from_raw(1);

/// The reconstruction's `n`-th image.
fn image(n: usize) -> ImageRef {
    ImageRef::new(RECON, n)
}

/// A SIFT cache covering `images` images with `features` entries each. The
/// affine shape has column norms (half-axes) 2 and 4, so the panel's reported
/// full extents must be exactly `[8.0, 4.0]` — values no other code path
/// produces by accident.
fn sift_cache(images: usize, features: usize) -> HashMap<ImageRef, CachedSiftFeatures> {
    sift_cache_with_shape(images, features, [[2.0, 0.0], [0.0, 4.0]])
}

/// A SIFT cache whose every feature carries `affine_shape`, for tests that care
/// about how a particular shape is measured and printed.
fn sift_cache_with_shape(
    images: usize,
    features: usize,
    affine_shape: [[f32; 2]; 2],
) -> HashMap<ImageRef, CachedSiftFeatures> {
    (0..images)
        .map(|i| {
            (
                image(i),
                CachedSiftFeatures {
                    positions_xy: vec![[500.0, 300.0]; features],
                    affine_shapes: vec![affine_shape; features],
                    read_count: features,
                },
            )
        })
        .collect()
}

/// Give the images rig-style nested paths, so the name column has a parent
/// directory to keep. `demo` alone produces bare `image_000.jpg`, which makes
/// the truncation a no-op.
fn with_nested_image_paths(mut recon: SfmrReconstruction) -> SfmrReconstruction {
    for (i, image) in recon.images.iter_mut().enumerate() {
        image.name = format!("images/fisheye_left/image_{i:03}.jpg");
    }
    recon
}

/// Give the reconstruction a content hash. `demo` leaves it empty, which sends
/// the Point ID down its zero-fill fallback.
fn with_content_hash(mut recon: SfmrReconstruction, hash: &str) -> SfmrReconstruction {
    recon.content_hash.content_xxh128 = hash.to_string();
    recon
}

/// Swap the observation source to embedded keypoints and attach patch
/// half-vectors, so the panel takes the `keypoints_xy` branch and the "Patch"
/// column turns on.
///
/// The keypoints are each point's true projection rather than a constant:
/// `observation_affine_shape` anchors the patch frame where the keypoint's
/// back-projected ray meets the patch plane, so an arbitrary pixel puts the
/// anchor behind the camera and every reported feature size silently
/// degenerates to the `unwrap_or(0.0)` fallback.
fn with_embedded_patches(mut recon: SfmrReconstruction) -> SfmrReconstruction {
    let obs_count = recon.tracks.len();
    let n = recon.points.len();

    let mut keypoints = Array2::<f32>::zeros((obs_count, 2));
    for point_idx in 0..n {
        let start = recon.observation_offsets[point_idx];
        let position = recon.points[point_idx].position;
        for (k, obs) in recon.observations_for_point(point_idx).iter().enumerate() {
            let image = &recon.images[obs.image_index as usize];
            let camera = &recon.cameras[image.camera_index as usize];
            let p_cam = image.quaternion_wxyz.to_rotation_matrix() * position.coords
                + image.translation_xyz;
            if let Some((x, y)) = camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z]) {
                keypoints[[start + k, 0]] = x as f32;
                keypoints[[start + k, 1]] = y as f32;
            }
        }
    }

    // A patch spanning X and Z: the demo cameras ring the XY plane, so this is
    // never exactly edge-on to all of them.
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    for i in 0..n {
        u[[i, 0]] = 0.1;
        v[[i, 2]] = 0.1;
    }
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);
    recon.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::from_elem((n, 8, 8, 4), 200));
    recon.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; recon.images.len()],
    };
    recon
}

/// Full-resolution sources for `indices`, sized to the reconstruction's own
/// intrinsics so the patch warp samples inside them. Without these the
/// per-observation patch tiles never render — `ensure_rendered_patch` returns
/// early on a cache miss.
fn full_res_images(
    recon: &SfmrReconstruction,
    indices: &[usize],
) -> HashMap<ImageRef, Option<ImageU8>> {
    let camera = &recon.cameras[0];
    let (w, h) = (camera.width, camera.height);
    indices
        .iter()
        .map(|&i| {
            (
                image(i),
                Some(ImageU8::new(w, h, 3, vec![180u8; (w * h * 3) as usize])),
            )
        })
        .collect()
}

const VIEWPORT: egui::Vec2 = egui::vec2(1200.0, 800.0);

/// Drive one frame of the panel over a fixed-size viewport with `events`
/// delivered to egui, returning the response it hands back to the dock. The
/// panel is left in its post-frame state so tests can inspect what it prepared.
fn run_frame(
    panel: &mut PointTrackDetail,
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    selected_point: Option<usize>,
    sift_cache: &HashMap<ImageRef, CachedSiftFeatures>,
    full_res_cache: &HashMap<ImageRef, Option<ImageU8>>,
    events: Vec<egui::Event>,
) -> PointTrackDetailResponse {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    let mut response = None;
    let _ = ctx.run_ui(input, |ui| {
        response = Some(panel.show(
            ui,
            recon,
            RECON,
            selected_point,
            None,
            sift_cache,
            full_res_cache,
            &[],
            &ScrollInput::default(),
        ));
    });
    response.expect("the panel ran")
}

/// One frame with no input events and no full-res images.
fn show_once(
    panel: &mut PointTrackDetail,
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    selected_point: Option<usize>,
    sift_cache: &HashMap<ImageRef, CachedSiftFeatures>,
) -> PointTrackDetailResponse {
    run_frame(
        panel,
        ctx,
        recon,
        selected_point,
        sift_cache,
        &HashMap::new(),
        Vec::new(),
    )
}

/// Park the pointer at `pos` and optionally click there, returning the last
/// frame's response. Two frames are required: egui resolves hover and clicks
/// against the widget rects registered on the *previous* pass, so a single
/// frame never reports an interaction.
fn show_at_pointer(
    panel: &mut PointTrackDetail,
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    selected_point: Option<usize>,
    pos: egui::Pos2,
    clicks: usize,
) -> PointTrackDetailResponse {
    let cache = sift_cache(8, 16);
    let mut response = None;
    for frame in 0..2 {
        let mut events = vec![egui::Event::PointerMoved(pos)];
        if frame == 1 {
            for _ in 0..clicks {
                for pressed in [true, false] {
                    events.push(egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    });
                }
            }
        }
        response = Some(run_frame(
            panel,
            ctx,
            recon,
            selected_point,
            &cache,
            &HashMap::new(),
            events,
        ));
    }
    response.expect("two frames ran")
}

// ── Selection handling ──────────────────────────────────────────────────

#[test]
fn selecting_a_point_prepares_one_row_per_observation() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(3), &sift_cache(8, 16));

    // `demo` gives every point two observations, from adjacent cameras.
    let expected: Vec<usize> = recon
        .observations_for_point(3)
        .iter()
        .map(|o| o.image_index as usize)
        .collect();
    assert_eq!(expected.len(), 2);
    let prepared: Vec<usize> = panel.observations.iter().map(|o| o.image_index).collect();
    assert_eq!(prepared, expected);
    assert_eq!(panel.prepared_point, Some(PointRef::new(RECON, 3)));
}

#[test]
fn no_selection_clears_previously_prepared_state() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();
    let cache = sift_cache(8, 16);

    show_once(&mut panel, &ctx, &recon, Some(3), &cache);
    assert!(!panel.observations.is_empty());

    show_once(&mut panel, &ctx, &recon, None, &cache);
    assert!(panel.observations.is_empty());
    assert_eq!(panel.prepared_point, None);
}

#[test]
fn an_out_of_range_point_index_falls_back_to_the_placeholder() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    let response = show_once(&mut panel, &ctx, &recon, Some(999), &sift_cache(8, 16));

    assert_eq!(panel.prepared_point, None);
    assert!(panel.observations.is_empty());
    assert_eq!(response.select_image, None);
}

#[test]
fn changing_the_selection_reprepares_the_table() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();
    let cache = sift_cache(8, 16);

    show_once(&mut panel, &ctx, &recon, Some(0), &cache);
    let first: Vec<usize> = panel.observations.iter().map(|o| o.image_index).collect();

    show_once(&mut panel, &ctx, &recon, Some(4), &cache);
    let second: Vec<usize> = panel.observations.iter().map(|o| o.image_index).collect();

    // Point 0 is seen by cameras 0/1, point 4 by cameras 4/5.
    assert_eq!(first, vec![0, 1]);
    assert_eq!(second, vec![4, 5]);
    assert_eq!(panel.prepared_point, Some(PointRef::new(RECON, 4)));
}

// ── Per-observation data ────────────────────────────────────────────────

#[test]
fn feature_extents_are_the_doubled_affine_column_norms_larger_first() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(2), &sift_cache(8, 16));

    // The fixture's affine shape has column norms 2 and 4 — half-axes, so the
    // full extents the drawn quad spans are 4 and 8, reported larger first.
    for obs in &panel.observations {
        assert_eq!(obs.feature_extents, [8.0, 4.0]);
        assert_eq!(obs.feature_xy, [500.0, 300.0]);
    }
}

#[test]
fn the_larger_extent_comes_first_whichever_affine_column_is_longer() {
    let recon = SfmrReconstruction::demo(12);
    let ctx = egui::Context::default();

    // Same shape with its columns swapped: the ordering must come from the
    // norms, not from the column order.
    for shape in [[[2.0, 0.0], [0.0, 4.0]], [[4.0, 0.0], [0.0, 2.0]]] {
        let mut panel = PointTrackDetail::new();
        let cache = sift_cache_with_shape(8, 16, shape);
        show_once(&mut panel, &ctx, &recon, Some(2), &cache);
        for obs in &panel.observations {
            assert_eq!(obs.feature_extents, [8.0, 4.0], "shape {shape:?}");
        }
    }
}

#[test]
fn a_missing_sift_cache_leaves_the_feature_columns_empty() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    // No cache entries at all — the table must still draw, reporting zeros
    // rather than panicking on the absent features.
    show_once(&mut panel, &ctx, &recon, Some(2), &HashMap::new());

    assert_eq!(panel.observations.len(), 2);
    for obs in &panel.observations {
        assert_eq!(obs.feature_extents, [0.0, 0.0]);
        assert_eq!(obs.feature_xy, [0.0, 0.0]);
    }
}

#[test]
fn the_image_name_column_keeps_the_parent_directory() {
    let recon = with_nested_image_paths(SfmrReconstruction::demo(12));
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(1), &sift_cache(8, 16));

    assert_eq!(panel.observations.len(), 2);
    for obs in &panel.observations {
        let i = obs.image_index;
        // The leading directories are dropped, the parent kept, and the full
        // path is preserved separately for the row's hover tooltip.
        assert_eq!(
            obs.image_name,
            format!("\u{2026}/fisheye_left/image_{i:03}.jpg")
        );
        assert_eq!(
            obs.image_full_name,
            format!("images/fisheye_left/image_{i:03}.jpg")
        );
    }
}

#[test]
fn the_max_pairwise_angle_spans_the_observing_cameras() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(5), &sift_cache(8, 16));

    // Recomputed from the reconstruction rather than restated from the panel:
    // the angle subtended at the point by its two observing camera centres.
    let point = recon.points[5].position;
    let rays: Vec<_> = recon
        .observations_for_point(5)
        .iter()
        .map(|o| (point - recon.images[o.image_index as usize].camera_center()).normalize())
        .collect();
    let expected = rays[0].dot(&rays[1]).clamp(-1.0, 1.0).acos().to_degrees() as f32;

    assert!(expected > 1.0, "fixture geometry is degenerate: {expected}");
    assert!(
        (panel.max_angle_deg - expected).abs() < 1e-3,
        "panel reported {}, expected {expected}",
        panel.max_angle_deg
    );
}

// ── Pointer interaction ─────────────────────────────────────────────────

/// Sweep the pointer down the panel and collect which image each y reports as
/// hovered. Returns the distinct images in the order they first appear.
fn hovered_images_down_the_panel(
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut order = Vec::new();
    let mut first_y = Vec::new();
    for step in 0..100 {
        let y = step as f32 * 8.0;
        let mut panel = PointTrackDetail::new();
        let ctx = egui::Context::default();
        let response = show_at_pointer(
            &mut panel,
            &ctx,
            recon,
            Some(point_idx),
            egui::pos2(300.0, y),
            0,
        );
        if let Some(img) = response.hovered_image {
            if !order.contains(&img) {
                order.push(img);
                first_y.push(y);
            }
        }
    }
    (order, first_y)
}

#[test]
fn every_row_is_independently_hoverable() {
    let recon = SfmrReconstruction::demo(12);

    // Point 0 is observed by cameras 0 and 1, drawn in that order. Each must
    // own its own band of the panel — a single row swallowing the pointer, or
    // rows sharing a rect, would collapse this to one entry.
    let (hovered, first_y) = hovered_images_down_the_panel(&recon, 0);

    assert_eq!(hovered, vec![0, 1]);
    // The two bands are one row apart, which is what the fixed row height buys.
    let gap = first_y[1] - first_y[0];
    assert!(
        (gap - 56.0).abs() <= 8.0,
        "rows started {gap}px apart, expected ~56"
    );
}

#[test]
fn clicking_a_row_selects_its_image_and_double_clicking_enters_camera_view() {
    let recon = SfmrReconstruction::demo(12);
    let (_, first_y) = hovered_images_down_the_panel(&recon, 0);
    // Aim at the middle of the second row's band, well clear of its edges.
    let pos = egui::pos2(300.0, first_y[1] + 24.0);

    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();
    let single = show_at_pointer(&mut panel, &ctx, &recon, Some(0), pos, 1);
    assert_eq!(single.select_image, Some(1));
    assert_eq!(single.request_camera_view, None);

    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();
    let double = show_at_pointer(&mut panel, &ctx, &recon, Some(0), pos, 2);
    assert_eq!(double.select_image, Some(1));
    assert_eq!(double.request_camera_view, Some(1));
}

#[test]
fn the_panel_reports_whether_it_holds_the_pointer() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    let inside = show_at_pointer(
        &mut panel,
        &ctx,
        &recon,
        Some(0),
        egui::pos2(300.0, 200.0),
        0,
    );
    assert!(inside.has_pointer);

    // Off the bottom of the viewport entirely.
    let outside = show_at_pointer(
        &mut panel,
        &ctx,
        &recon,
        Some(0),
        egui::pos2(300.0, 5000.0),
        0,
    );
    assert!(!outside.has_pointer);
}

// ── Thumbnails and patch tiles ──────────────────────────────────────────

#[test]
fn each_observed_image_gets_a_cached_thumbnail() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(6), &sift_cache(8, 16));

    // One texture per distinct observed image proves every row drew its
    // thumbnail, not just the first.
    assert_eq!(panel.thumbnail_textures.len(), 2);
    assert!(panel.thumbnail_textures.contains_key(&image(6)));
    assert!(panel.thumbnail_textures.contains_key(&image(7)));
}

#[test]
fn a_reconstruction_without_patches_has_no_patch_column() {
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(0), &sift_cache(8, 16));

    assert!(panel.patch_frame.is_none());
    assert!(panel.stored_patch_texture.is_none());
    assert!(panel.rendered_patch_textures.is_empty());
}

#[test]
fn embedded_patches_enable_the_patch_column_and_header_tile() {
    let recon = with_embedded_patches(SfmrReconstruction::demo(12));
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(0), &HashMap::new());

    // The half-vectors give the point an oriented frame, which is what gates
    // the "Patch" column; the non-zero bitmap gives the header its tile.
    assert!(panel.patch_frame.is_some());
    assert!(panel.stored_patch_texture.is_some());
    // Keypoints now come from the reconstruction's inline array rather than
    // the SIFT cache — check the panel read the right row for each row.
    assert_eq!(panel.observations.len(), 2);
    let keypoints = recon.keypoints_xy().expect("embedded keypoints");
    let start = recon.observation_offsets[0];
    for (k, obs) in panel.observations.iter().enumerate() {
        assert_eq!(
            obs.feature_xy,
            [keypoints[[start + k, 0]], keypoints[[start + k, 1]]]
        );
        // Non-zero proves the extents came from projecting the patch frame into
        // the view, not from the `unwrap_or([0.0, 0.0])` fallback.
        assert!(
            obs.feature_extents[0] > 0.0,
            "extents were {:?}",
            obs.feature_extents
        );
    }
}

#[test]
fn patch_tiles_render_once_per_observed_image() {
    let recon = with_embedded_patches(SfmrReconstruction::demo(12));
    let full_res = full_res_images(&recon, &[0, 1]);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    run_frame(
        &mut panel,
        &ctx,
        &recon,
        Some(0),
        &HashMap::new(),
        &full_res,
        Vec::new(),
    );

    // Point 0 is observed by cameras 0 and 1; both have a source image, so
    // both rows warp one.
    assert_eq!(panel.rendered_patch_textures.len(), 2);
    assert!(panel.rendered_patch_textures.contains_key(&image(0)));
    assert!(panel.rendered_patch_textures.contains_key(&image(1)));
}

#[test]
fn a_missing_full_res_image_leaves_its_patch_tile_uncached() {
    let recon = with_embedded_patches(SfmrReconstruction::demo(12));
    // Only camera 0's source is available; camera 1 also observes point 0.
    let full_res = full_res_images(&recon, &[0]);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    run_frame(
        &mut panel,
        &ctx,
        &recon,
        Some(0),
        &HashMap::new(),
        &full_res,
        Vec::new(),
    );

    // Nothing is cached for the missing source, so the tile can render later
    // once the dock finishes pre-caching rather than being memoized as absent.
    assert_eq!(panel.rendered_patch_textures.len(), 1);
    assert!(panel.rendered_patch_textures.contains_key(&image(0)));
}

#[test]
fn an_all_zero_patch_bitmap_leaves_the_header_tile_empty() {
    let mut recon = with_embedded_patches(SfmrReconstruction::demo(12));
    let n = recon.points.len();
    recon.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::zeros((n, 8, 8, 4)));
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(0), &HashMap::new());

    assert!(panel.patch_frame.is_some());
    assert!(panel.stored_patch_texture.is_none());
}

// ── Panel lifecycle ─────────────────────────────────────────────────────

#[test]
fn the_point_id_carries_the_content_hash_prefix() {
    let recon = with_content_hash(SfmrReconstruction::demo(12), "deadbeefcafef00d");
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(0), &sift_cache(8, 16));

    assert_eq!(panel.hash_prefix, "deadbeef");
}

#[test]
fn a_short_content_hash_falls_back_to_zeros() {
    // `demo` leaves the hash empty, which is the case the fallback exists for.
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(0), &sift_cache(8, 16));

    assert_eq!(panel.hash_prefix, "00000000");
}

#[test]
fn clear_resets_every_cache() {
    let recon = with_embedded_patches(SfmrReconstruction::demo(12));
    let full_res = full_res_images(&recon, &[0, 1]);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    run_frame(
        &mut panel,
        &ctx,
        &recon,
        Some(0),
        &HashMap::new(),
        &full_res,
        Vec::new(),
    );
    // Every cache the panel owns is populated before we clear it, so each
    // assertion below has something to disprove.
    assert!(!panel.observations.is_empty());
    assert!(!panel.thumbnail_textures.is_empty());
    assert!(!panel.rendered_patch_textures.is_empty());
    assert!(panel.stored_patch_texture.is_some());

    panel.clear();

    assert_eq!(panel.prepared_point, None);
    assert!(panel.observations.is_empty());
    assert!(panel.thumbnail_textures.is_empty());
    assert!(panel.patch_frame.is_none());
    assert!(panel.stored_patch_texture.is_none());
    assert!(panel.rendered_patch_textures.is_empty());
    assert!(panel.hash_prefix.is_empty());
    assert_eq!(panel.scroll_offset_y, None);
}

// ── Size column formatting ──────────────────────────────────────────────

#[test]
fn a_circular_shape_prints_both_equal_extents() {
    // Both extents always show, so a circular feature reads `AxA` and the
    // reader never has to guess which display form they are looking at.
    assert_eq!(format_feature_size([14.0, 14.0]), "14.0x14.0");
    assert_eq!(format_feature_size([20.5, 19.5]), "20.5x19.5");
}

#[test]
fn an_oval_shape_prints_both_extents_larger_first() {
    assert_eq!(format_feature_size([20.3, 7.7]), "20.3x7.7");
    // Mildly oval too — no threshold decides between display forms.
    assert_eq!(format_feature_size([12.0, 10.0]), "12.0x10.0");
}

#[test]
fn a_degenerate_shape_prints_na_and_an_edge_on_one_shows_the_collapse() {
    assert_eq!(format_feature_size([0.0, 0.0]), "N/A");
    // A collapsed minor axis is infinitely oval, not circular — it must not
    // fall through to the averaging branch and report half its width.
    assert_eq!(format_feature_size([9.0, 0.0]), "9.0x0.0");
}

#[test]
fn the_printed_size_is_twice_the_affine_semi_axis() {
    // End to end: the fixture's affine columns are half-vectors of norm 2 and
    // 4, so the row must read 8x4 — the span of the quad drawn in the
    // viewport, not the 3.0 mean-radius the old column printed.
    let recon = SfmrReconstruction::demo(12);
    let mut panel = PointTrackDetail::new();
    let ctx = egui::Context::default();

    show_once(&mut panel, &ctx, &recon, Some(2), &sift_cache(8, 16));

    for obs in &panel.observations {
        assert_eq!(format_feature_size(obs.feature_extents), "8.0x4.0");
    }
}

// ── Shared metrics helpers ──────────────────────────────────────────────

#[test]
fn max_pairwise_angle_finds_the_widest_pair() {
    // Three rays: 0°, 45° and 90° from +X. The widest pair is the outer two.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rays = [[1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0]];
    let angle = super::compute_max_pairwise_angle(&rays);
    assert!((angle - 90.0).abs() < 1e-4, "angle was {angle}");
}

#[test]
fn max_pairwise_angle_of_fewer_than_two_rays_is_zero() {
    assert_eq!(super::compute_max_pairwise_angle(&[]), 0.0);
    assert_eq!(super::compute_max_pairwise_angle(&[[1.0, 0.0, 0.0]]), 0.0);
}

#[test]
fn point_diagnostics_are_undefined_for_a_missing_point() {
    let recon = SfmrReconstruction::demo(4);
    let (cond, z) = super::compute_point_diagnostics(&recon, 999);
    assert!(cond.is_nan());
    assert!(z.is_nan());
}

#[test]
fn point_diagnostics_are_finite_for_a_triangulated_point() {
    let recon = SfmrReconstruction::demo(12);
    let (cond, z) = super::compute_point_diagnostics(&recon, 5);
    assert!(cond.is_finite() && cond >= 1.0, "condition number {cond}");
    assert!(z.is_finite(), "inverse-depth z {z}");
}

#[test]
fn error_color_ramps_from_green_through_yellow_to_red() {
    let green = super::metrics::error_color(0.0);
    let yellow = super::metrics::error_color(1.0);
    let red = super::metrics::error_color(2.0);
    assert_eq!((green.r(), green.b()), (0, 0));
    assert_eq!((yellow.r(), yellow.g(), yellow.b()), (255, 255, 0));
    assert_eq!((red.r(), red.g(), red.b()), (255, 0, 0));
    // Anything past the top of the ramp stays red; NaN is the "N/A" gray.
    assert_eq!(super::metrics::error_color(50.0), red);
    assert_eq!(
        super::metrics::error_color(f32::NAN),
        egui::Color32::from_rgb(128, 128, 128)
    );
}
