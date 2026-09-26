// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the panel's pan/zoom view state.
//!
//! The property they pin down is that the view outlives the image it was set
//! on. Flipping between two images with `,` / `.`, between reconstructions with
//! `[` / `]`, or clicking a different thumbnail is how the panel gets used to
//! *compare* — and a view that snapped back to fit on every switch made the
//! comparison impossible to see. So each test zooms in, changes what the panel
//! is showing, and asks for the same region back.
//!
//! Real frames rather than direct calls to [`super::ImageDetail::rescale_view`]:
//! the extent the view is rescaled against is a product of the whole frame (the
//! texture's size, the panel's rect, the zoom), and the ordering of that against
//! input handling is exactly what a unit call would fake away.

use sfmtool_core::camera::remap::ImageU8;

use super::ImageDetail;

/// A feature ellipse is the affine applied to the unit circle, so a sheared
/// shape draws with its major axis along the left singular vector and not along
/// the first column: the two agree only for a similarity.
#[test]
fn a_feature_ellipse_is_the_affine_applied_to_the_unit_circle() {
    let center = egui::pos2(100.0, 100.0);
    // A pure shear: the first column points along +x, the ellipse's major axis
    // does not.
    let affine = [[10.0_f32, 8.0], [0.0, 10.0]];
    let points = super::overlay::ellipse_points(center, &affine, 1.0).expect("a drawable shape");
    for (i, p) in points.iter().enumerate() {
        let t = (i as f32) * std::f32::consts::TAU / 32.0;
        let (c, s) = (t.cos(), t.sin());
        let expected = egui::pos2(100.0 + 10.0 * c + 8.0 * s, 100.0 + 10.0 * s);
        assert!((p.x - expected.x).abs() < 1e-3 && (p.y - expected.y).abs() < 1e-3);
    }
    // The point farthest from the centre is not on the first column's line.
    let far = points
        .iter()
        .max_by(|a, b| a.distance(center).partial_cmp(&b.distance(center)).unwrap())
        .unwrap();
    assert!(
        (far.y - center.y).abs() > 1.0,
        "the major axis leans off +x: {far:?}"
    );
    assert!(super::overlay::ellipse_points(center, &[[0.05, 0.0], [0.0, 5.0]], 1.0).is_none());
}
use crate::scene::SceneNode;
use crate::state::{FeatureDisplaySettings, IntrinsicsDisplaySettings};

/// The panel size these frames run at, unless a test resizes it.
const PANEL: egui::Vec2 = egui::Vec2::new(900.0, 700.0);

/// One headless frame showing `image_index` of `node`, with `image` standing in
/// for the photograph's pixels.
fn frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    image_index: usize,
    image: &ImageU8,
    panel: egui::Vec2,
) {
    look_frame(detail, ctx, node, image_index, image, panel, None);
}

/// The same frame, carrying a look request for a place in `image_index`: what
/// the dock hands the panel on the frame after a row click elsewhere named a
/// feature, or after a tool asked for a view.
fn look_frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    image_index: usize,
    image: &ImageU8,
    panel: egui::Vec2,
    look: Option<super::Look>,
) {
    let feature_display = FeatureDisplaySettings::default();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, panel)),
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(image_index),
            look,
            None,
            None,
            super::BenchMenu::default(),
            &[],
            &crate::platform::ScrollInput::default(),
            None,
            Some(image),
            &feature_display,
            &mut intrinsics_display,
        );
    });
}

/// A node to page through: `demo` gives 8 images off one camera.
fn demo_node(path: &str) -> SceneNode {
    SceneNode::from_path(
        std::path::Path::new(path),
        sfmtool_core::SfmrReconstruction::demo(32),
    )
}

/// A flat grey stand-in for a photograph. Only its size matters here — it is
/// what the panel fits to the panel rect.
fn pixels(width: u32, height: u32) -> ImageU8 {
    ImageU8::new(width, height, 3, vec![90u8; (width * height * 3) as usize])
}

/// The normalized image coordinate sitting at the panel centre — the thing the
/// panel promises to hold fixed. See [`super::ImageDetail::rescale_view`].
fn anchor(detail: &ImageDetail) -> egui::Vec2 {
    let display = detail
        .last_display_size
        .expect("a frame that drew an image records its extent");
    egui::vec2(
        0.5 - detail.pan.x / display.x,
        0.5 - detail.pan.y / display.y,
    )
}

fn assert_close(actual: egui::Vec2, expected: egui::Vec2, what: &str) {
    assert!(
        (actual.x - expected.x).abs() < 1e-4 && (actual.y - expected.y).abs() < 1e-4,
        "{what}: {actual:?} != {expected:?}",
    );
}

#[test]
fn the_view_survives_switching_to_another_image() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(8, 8);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    // Set before the first frame, so the recorded extent and the zoom agree the
    // way they do after a real zoom gesture (which runs inside the frame).
    detail.zoom = 6.0;
    detail.pan = egui::vec2(120.0, -45.0);
    frame(&mut detail, &ctx, &node, 0, &image, PANEL);
    let (zoom, pan) = (detail.zoom, detail.pan);

    frame(&mut detail, &ctx, &node, 1, &image, PANEL);

    assert_eq!(detail.zoom, zoom, "zoom reset on an image switch");
    assert_eq!(detail.pan, pan, "pan reset on an image switch");
}

#[test]
fn the_view_survives_switching_to_another_reconstruction() {
    let first = demo_node("/runs/a.sfmr");
    let second = demo_node("/runs/b.sfmr");
    assert_ne!(first.id, second.id);
    let image = pixels(8, 8);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    detail.zoom = 6.0;
    detail.pan = egui::vec2(120.0, -45.0);
    frame(&mut detail, &ctx, &first, 3, &image, PANEL);
    let (zoom, pan) = (detail.zoom, detail.pan);

    frame(&mut detail, &ctx, &second, 3, &image, PANEL);

    assert_eq!(detail.zoom, zoom, "zoom reset on a reconstruction switch");
    assert_eq!(detail.pan, pan, "pan reset on a reconstruction switch");
}

#[test]
fn the_framed_region_survives_a_change_of_image_size() {
    let node = demo_node("/runs/demo.sfmr");
    let square = pixels(8, 8);
    let wide = pixels(16, 8);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    detail.zoom = 4.0;
    detail.pan = egui::vec2(100.0, 50.0);
    frame(&mut detail, &ctx, &node, 0, &square, PANEL);
    let before = anchor(&detail);
    let extent = detail.last_display_size.unwrap();

    frame(&mut detail, &ctx, &node, 1, &wide, PANEL);

    assert_ne!(
        detail.last_display_size.unwrap(),
        extent,
        "the two images should be displayed at different extents",
    );
    assert_close(anchor(&detail), before, "framed region moved");
}

#[test]
fn the_framed_region_survives_a_panel_resize() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(8, 8);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    detail.zoom = 4.0;
    detail.pan = egui::vec2(100.0, 50.0);
    frame(&mut detail, &ctx, &node, 0, &image, PANEL);
    let before = anchor(&detail);

    frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        egui::vec2(600.0, 500.0),
    );

    assert_close(anchor(&detail), before, "framed region moved");
}

#[test]
fn a_view_reset_forgets_the_extent_it_was_measured_against() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(8, 8);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    detail.zoom = 4.0;
    frame(&mut detail, &ctx, &node, 0, &image, PANEL);
    detail.reset_view();
    assert!(detail.last_display_size.is_none());

    // A fit view carries nothing, so the next frame must not rescale the fresh
    // pan against the zoomed-in extent it was reset from.
    frame(&mut detail, &ctx, &node, 0, &image, PANEL);
    assert_eq!(detail.zoom, 1.0);
    assert_eq!(detail.pan, egui::Vec2::ZERO);
}

// ── Revealing a feature ─────────────────────────────────────────────────
//
// A row click in the either mode of Track View panel selects an image
// *and* names a feature in it. Zoomed in, that feature can be nowhere on
// screen, so the panel is asked to bring it into view: by panning, never by
// zooming, and only when it has to.

/// The source image these tests reveal pixels of: big enough that a zoom of 4
/// shows only a part of it in [`PANEL`].
const SOURCE: egui::Vec2 = egui::Vec2::new(400.0, 300.0);

/// The row click's request: bring `pixel` into view, without touching the
/// zoom.
fn reveal(pixel: [f32; 2]) -> Option<super::Look> {
    Some(super::Look::Reveal { pixel })
}

/// Where `pixel` of [`SOURCE`] sits relative to the panel centre, in panel
/// pixels, for the view the last frame left behind. Zero is dead centre.
fn offset_of(detail: &ImageDetail, pixel: [f32; 2]) -> egui::Vec2 {
    let display = detail
        .last_display_size
        .expect("a frame that drew an image records its extent");
    let scale = display.x / SOURCE.x;
    detail.pan - display / 2.0 + egui::vec2(pixel[0], pixel[1]) * scale
}

/// As [`assert_close`], to a tolerance these panel-pixel magnitudes can hold:
/// the quantities here are differences of numbers in the thousands, where an
/// `f32` has no 1e-4 to give.
fn assert_near(actual: egui::Vec2, expected: egui::Vec2, what: &str) {
    assert!(
        (actual.x - expected.x).abs() < 0.05 && (actual.y - expected.y).abs() < 0.05,
        "{what}: {actual:?} != {expected:?}",
    );
}

#[test]
fn revealing_a_feature_out_of_view_centres_it_and_leaves_the_zoom_alone() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    // Zoomed in on the middle of the image; the feature is up in the corner,
    // several panel-widths away.
    detail.zoom = 4.0;
    let feature = [10.0, 10.0];
    look_frame(&mut detail, &ctx, &node, 0, &image, PANEL, reveal(feature));

    assert_eq!(detail.zoom, 4.0, "a reveal zoomed");
    assert_near(offset_of(&detail, feature), egui::Vec2::ZERO, "not centred");
}

#[test]
fn revealing_a_feature_already_in_view_moves_nothing() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    // A view off the image's own centre, and a feature a little way off the
    // *panel's* centre in it: 30 px across and 25 down, well inside the margin.
    detail.zoom = 4.0;
    detail.pan = egui::vec2(300.0, -200.0);
    let feature = [170.0, 175.0];
    look_frame(&mut detail, &ctx, &node, 0, &image, PANEL, reveal(feature));

    assert_eq!(
        detail.pan,
        egui::vec2(300.0, -200.0),
        "a feature on screen moved the view",
    );
}

#[test]
fn revealing_a_feature_at_fit_zoom_moves_nothing() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    // The very corner, which the margin would call out of view; but at fit
    // zoom the whole image is on screen and there is nothing to bring into it.
    look_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        PANEL,
        reveal([0.0, 0.0]),
    );

    assert_eq!(detail.zoom, 1.0);
    assert_eq!(detail.pan, egui::Vec2::ZERO, "a fitted image was panned");
}

#[test]
fn revealing_a_corner_feature_still_obeys_the_pan_clamp() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    // A panel small enough that the pan limit bites before the corner reaches
    // the centre: the limit is `(display + panel) / 2 - PAN_MARGIN`, and
    // centring a corner asks for `display / 2`.
    let panel = egui::vec2(80.0, 60.0);
    detail.zoom = 4.0;
    look_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        panel,
        reveal([SOURCE.x, SOURCE.y]),
    );

    let display = detail.last_display_size.expect("a frame drew");
    let limit = egui::vec2(
        (display.x + panel.x) / 2.0 - 50.0,
        (display.y + panel.y) / 2.0 - 50.0,
    );
    assert!(
        detail.pan.x >= -limit.x - 1e-3 && detail.pan.y >= -limit.y - 1e-3,
        "the reveal panned past the clamp: {:?} vs {limit:?}",
        detail.pan,
    );
    // Clamped, not ignored: it went as far as it is allowed to.
    assert_near(detail.pan, -limit, "the reveal stopped short of the clamp");
}

// ── Reading through the overlay ─────────────────────────────────────────

/// The embedded-features walk over a version that has deleted and added a
/// point: it is over the version's live indexes, not the base's rows.
#[test]
fn the_embedded_overlay_skips_a_deleted_point_and_shows_an_addition() {
    use crate::state::edits::tests::embedded_demo;

    let base = std::sync::Arc::new(embedded_demo(40));
    let loaded = sfmtool_core::EditedReconstruction::new(std::sync::Arc::clone(&base));

    // The image every feature below is counted in: the first one the point
    // whose fate we change is seen in.
    let image = loaded.track_image_indices(3)[0];
    let features_in = |edited: &sfmtool_core::EditedReconstruction| -> Vec<u32> {
        super::embedded_image_features(edited, image)
            .iter()
            .map(|f| f.point_index)
            .collect()
    };

    let before = features_in(&loaded);
    assert!(before.contains(&3));

    // Deleted: its observation leaves the overlay.
    let mut deleted = loaded.clone();
    deleted.delete_point(3).expect("a live point");
    let after = features_in(&deleted);
    assert!(!after.contains(&3), "a deleted point still drew a feature");
    assert_eq!(after.len(), before.len() - 1);

    // Modified: the old index is gone and the new one is there, still in the
    // same image, because the record carried its track over.
    let mut modified = loaded.clone();
    let record = modified.point(3).expect("a live point").to_record();
    let moved = modified.replace_point(3, record).expect("a live point");
    let after = features_in(&modified);
    assert!(!after.contains(&3), "the replaced index still drew");
    assert!(after.contains(&moved), "the addition drew no feature");
    assert_eq!(after.len(), before.len());
}

/// Deleting a point takes its features out of the overlay on the next frame.
///
/// The panel prepares the overlay once and keeps it across frames, and a point
/// edit renumbers nothing, so nothing else in the panel's cache key moves when
/// one lands. Before the version joined that key the overlay outlived the edit,
/// and all three of the symptoms that follow are the one stale list: the
/// deleted point's features went on drawing, a click on one selected an index
/// the version has no point at, and Track View -- which reads that
/// index *through* the version -- then showed nothing while a second delete
/// refused, because there was no live point there to delete.
///
/// A `sift_files` node, so the assertion also covers the half of the rebuild
/// that reads `image_feature_to_point`: that map belongs to the base and still
/// names the deleted row, so a rebuilt overlay is only correct if it maps
/// through the version.
#[test]
fn deleting_a_point_takes_its_features_out_of_the_overlay() {
    use crate::document::PointMap;
    use crate::state::OverlayMode;

    let mut node = demo_node("/runs/overlay.sfmr");
    // A point observed in image 0, named the way the panel names it: through
    // the base's per-image feature map, which is what the overlay reads.
    let (&feature, &point) = node.recon().point_set.image_feature_to_point[0]
        .iter()
        .next()
        .expect("the demo's first image tracks features");
    // Sized the way the panel sizes its own read: feature indexes are the
    // `.sift` file's, so they are not dense over a track's members.
    let count = node.recon().point_set.max_track_feature_index[0] as usize + 1;
    let sift = crate::state::CachedSiftFeatures {
        positions_xy: (0..count).map(|i| [40.0 + i as f32, 30.0]).collect(),
        affine_shapes: vec![[[6.0, 0.0], [0.0, 6.0]]; count],
        read_count: count,
    };
    assert!(
        (feature as usize) < count,
        "the feature index has to be inside the cache the frame is given",
    );

    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let image = pixels(1920, 1080);
    let display = FeatureDisplaySettings {
        overlay_mode: OverlayMode::Features,
        tracked_only: true,
        ..Default::default()
    };

    let offered = |detail: &ImageDetail| -> Vec<u32> {
        detail
            .feature_overlay
            .as_ref()
            .expect("a frame prepared the overlay")
            .features
            .iter()
            .map(|f| f.point_index)
            .collect()
    };

    overlay_frame(&mut detail, &ctx, &node, &sift, &image, &display);
    assert!(
        offered(&detail).contains(&point),
        "the point is in the overlay before it is deleted",
    );

    let mut next = node.history.current().clone();
    next.delete_point(point).expect("delete the live point");
    node.history
        .push(next, PointMap::Removed(vec![point]), "deleted".to_string());

    overlay_frame(&mut detail, &ctx, &node, &sift, &image, &display);
    assert!(
        !offered(&detail).contains(&point),
        "the deleted point is still offered by the overlay: {:?}",
        offered(&detail),
    );
}

/// One frame of the panel over a node that has a SIFT cache behind it.
///
/// Separate from [`frame`], which pages through images to test the view state
/// and hands the panel no features at all.
fn overlay_frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    sift: &crate::state::CachedSiftFeatures,
    image: &ImageU8,
    feature_display: &FeatureDisplaySettings,
) {
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(0),
            None,
            None,
            None,
            super::BenchMenu::default(),
            &[],
            &crate::platform::ScrollInput::default(),
            Some(sift),
            Some(image),
            feature_display,
            &mut intrinsics_display,
        );
    });
}

// ── The context menu's two bench entries ────────────────────────────────

use super::overlay::{
    add_bench_observation_entry, start_cluster_entry, ADD_BENCH_OBSERVATION_LABEL,
};
use super::{BenchMenu, START_CLUSTER_LABEL};

/// A node busy with a task refuses every step on it, and the entries say so in
/// that refusal's own words.
const BUSY: &str = "bull is busy: Evaluate track is still running.";

/// A track on the bench. The menu's rules read whether there is one and
/// nothing about what is in it, so an empty one says everything they can.
fn a_track() -> sfmtool_core::bench::EditableTrack {
    sfmtool_core::bench::EditableTrack::empty_cluster()
}

#[test]
fn starting_a_cluster_needs_only_a_pixel_and_a_node_that_is_not_busy() {
    assert_eq!(start_cluster_entry(BenchMenu::default()), Ok(()));
    assert_eq!(
        start_cluster_entry(BenchMenu {
            busy: Some(BUSY),
            active_track: None,
            lock: true,
            create_track: None,
        }),
        Err(BUSY.to_string()),
    );
}

#[test]
fn adding_to_the_bench_track_is_greyed_until_a_track_is_on_the_bench() {
    let track = a_track();
    let why = add_bench_observation_entry(BenchMenu::default())
        .expect_err("nothing is on the bench to add to");
    assert_eq!(
        why,
        "No track is being edited: tick Edit in Track View, or double-click a Bench item \
         in the Scene tree."
    );

    // The image the menu is open over is nowhere in the rule: a second
    // sighting in an image the track already holds joins as a candidate, and
    // it is the verdict that a track cannot hold twice.
    assert_eq!(
        add_bench_observation_entry(BenchMenu {
            busy: None,
            active_track: Some(&track),
            lock: true,
            create_track: None,
        }),
        Ok(()),
    );
    assert_eq!(
        add_bench_observation_entry(BenchMenu {
            busy: Some(BUSY),
            active_track: Some(&track),
            lock: true,
            create_track: None,
        }),
        Err(BUSY.to_string()),
    );
}

/// The two bench entries are in the menu, and are in it on a `sift_files`
/// node: a bench track is seeds in one image's pixels until it is committed, so
/// what backs the node's own observations does not decide it.
#[test]
fn the_context_menu_offers_the_two_bench_entries() {
    let track = a_track();
    let texts = context_menu_texts(BenchMenu {
        busy: None,
        active_track: Some(&track),
        lock: true,
        create_track: None,
    });
    for label in [START_CLUSTER_LABEL, ADD_BENCH_OBSERVATION_LABEL] {
        assert!(
            texts.iter().any(|t| t == label),
            "{label} is not in the menu: {texts:?}",
        );
    }

    // Greyed rather than absent: an entry with nothing on the bench still
    // names itself, and says why it cannot run on hover.
    let texts = context_menu_texts(BenchMenu::default());
    assert!(
        texts.iter().any(|t| t == ADD_BENCH_OBSERVATION_LABEL),
        "the greyed entry left the menu: {texts:?}",
    );
}

/// The texts the panel paints with its context menu open: one frame to load the
/// image and prepare the overlay, one that right-clicks the middle of it, and
/// one more, because the menu's entries are laid out on a later frame.
fn context_menu_texts(bench: BenchMenu<'_>) -> Vec<String> {
    let node = demo_node("/runs/menu.sfmr");
    let count = node.recon().point_set.max_track_feature_index[0] as usize + 1;
    let sift = crate::state::CachedSiftFeatures {
        positions_xy: (0..count).map(|i| [40.0 + i as f32, 30.0]).collect(),
        affine_shapes: vec![[[6.0, 0.0], [0.0, 6.0]]; count],
        read_count: count,
    };
    let image = pixels(1920, 1080);
    let display = FeatureDisplaySettings::default();
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();

    let at = egui::pos2(PANEL.x / 2.0, PANEL.y / 2.0);
    let click = vec![
        egui::Event::PointerMoved(at),
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Secondary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        },
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Secondary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        },
    ];

    let mut texts = Vec::new();
    for events in [Vec::new(), click, Vec::new()] {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
            events,
            ..Default::default()
        };
        texts = crate::test_support::painted_texts(&ctx, input, |ui| {
            detail.show(
                ui,
                node.edited(),
                node.id,
                node.history.current_version().serial,
                Some(0),
                None,
                None,
                None,
                bench,
                &[],
                &crate::platform::ScrollInput::default(),
                Some(&sift),
                Some(&image),
                &display,
                &mut intrinsics_display,
            );
        });
    }
    texts
}

// ── Edit on Bench, and what a double-click means ────────────────────────

use super::overlay::{create_track_here_entry, edit_on_bench_entry, feature_at, FeatureHit};
use crate::bench::track_at_pixel::{CREATE_TRACK_HERE_LABEL, CREATE_TRACK_HERE_SHORTCUT};
use crate::state::edits::PointGesture;
use crate::viewer_3d::EDIT_ON_BENCH_LABEL;

/// The menu's entries in the order they are drawn, of the four this panel
/// adds.
fn menu_entries(texts: &[String]) -> Vec<&str> {
    texts
        .iter()
        .map(String::as_str)
        .filter(|text| {
            [
                CREATE_TRACK_HERE_LABEL,
                EDIT_ON_BENCH_LABEL,
                START_CLUSTER_LABEL,
                ADD_BENCH_OBSERVATION_LABEL,
            ]
            .contains(text)
        })
        .collect()
}

/// `Create Track Here` stands at the top of the menu, directly above `Edit on
/// Bench`, which is above the two bench entries; and `Edit on Bench` is
/// **drawn** where it cannot run rather than hidden: the menu here is opened
/// over empty image, so nothing is under it to stage, while a track can be
/// created at any pixel.
#[test]
fn the_feature_menu_puts_create_track_here_directly_above_edit_on_bench() {
    let texts = context_menu_texts(BenchMenu::default());
    assert_eq!(
        menu_entries(&texts),
        [
            CREATE_TRACK_HERE_LABEL,
            EDIT_ON_BENCH_LABEL,
            START_CLUSTER_LABEL,
            ADD_BENCH_OBSERVATION_LABEL
        ],
    );
    // Its shortcut is painted beside it, as the menu bar paints a key's.
    assert!(
        texts.iter().any(|t| t == CREATE_TRACK_HERE_SHORTCUT),
        "the shortcut is not in the menu: {texts:?}"
    );
}

/// Greyed rather than hidden where it cannot run: the entry is still drawn,
/// in its place, and the entry's rule is the reason the state hands in, word
/// for word.
#[test]
fn create_track_here_is_greyed_with_the_reason_it_cannot_run() {
    assert_eq!(create_track_here_entry(BenchMenu::default()), Ok(()));
    for why in [
        crate::bench::track_at_pixel::NOT_POSED,
        crate::bench::track_at_pixel::NOT_EMBEDDED_PATCHES,
        BUSY,
    ] {
        let menu = BenchMenu {
            create_track: Some(why),
            ..BenchMenu::default()
        };
        assert_eq!(create_track_here_entry(menu), Err(why.to_string()));
        assert_eq!(
            menu_entries(&context_menu_texts(menu)).first(),
            Some(&CREATE_TRACK_HERE_LABEL),
            "the greyed entry left the menu or its place"
        );
    }
}

/// What one frame of clicks with `modifiers` held asked of the app: the pixel
/// Create Track Here was asked for at, and the point a click selected.
fn chord_click(
    detail: &mut ImageDetail,
    at: egui::Pos2,
    clicks: usize,
    modifiers: egui::Modifiers,
) -> (Option<[f32; 2]>, Option<usize>) {
    let (node, sift) = gesture_fixture();
    let image = pixels(1920, 1080);
    let ctx = egui::Context::default();
    let display = FeatureDisplaySettings::default();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let mut asked = (None, None);
    crate::platform::set_test_pointer_pos(Some(at));
    for frame in 0..2 {
        let mut events = vec![
            egui::Event::ModifiersChanged(modifiers),
            egui::Event::PointerMoved(at),
        ];
        if frame == 1 {
            for _ in 0..clicks {
                for pressed in [true, false] {
                    events.push(egui::Event::PointerButton {
                        pos: at,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers,
                    });
                }
            }
        }
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
            events,
            ..Default::default()
        };
        crate::test_support::run_frame_headless(&ctx, input, |ui| {
            let response = detail.show(
                ui,
                node.edited(),
                node.id,
                node.history.current_version().serial,
                Some(0),
                None,
                None,
                None,
                BenchMenu::default(),
                &[],
                &crate::platform::ScrollInput::default(),
                Some(&sift),
                Some(&image),
                &display,
                &mut intrinsics_display,
            );
            if frame == 1 {
                asked = (response.create_track_here, response.select_point);
            }
        });
    }
    crate::platform::set_test_pointer_pos(None);
    asked
}

/// Control and Shift held.
fn ctrl_shift() -> egui::Modifiers {
    egui::Modifiers {
        ctrl: true,
        shift: true,
        command: true,
        ..Default::default()
    }
}

/// A Control+Shift click on a feature asks for a track at the pixel clicked,
/// not at the feature, and is the whole of what the click means: no point is
/// selected, no bench item staged, nothing zoomed. The same click with no
/// modifier selects the feature's point and asks for no track.
#[test]
fn a_control_shift_click_asks_for_a_track_at_the_pixel_clicked() {
    let (node, _) = gesture_fixture();
    let image = pixels(1920, 1080);
    let (&feature, &point) = node.recon().point_set.image_feature_to_point[0]
        .iter()
        .next()
        .expect("the demo's first image tracks features");
    let [fx, fy] = feature_pixel(feature as usize);
    // A few source pixels off the feature, inside the click's hit radius, so
    // the answer tells the clicked pixel from the feature's.
    let clicked = [f64::from(fx) + 3.0, f64::from(fy) - 2.0];
    let at = to_panel(&image, clicked);

    let mut detail = ImageDetail::new();
    let (track_at, selected) = chord_click(&mut detail, at, 1, ctrl_shift());
    let track_at = track_at.expect("the chord asked for a track");
    let under = pixel_under(&detail, &image, at);
    assert!(
        (track_at[0] - under.x).abs() < 1e-3 && (track_at[1] - under.y).abs() < 1e-3,
        "asked at {track_at:?}, the pointer is over {under:?}"
    );
    assert!(
        (f64::from(track_at[0]) - clicked[0]).abs() < 0.5
            && (f64::from(track_at[1]) - clicked[1]).abs() < 0.5,
        "asked at {track_at:?} for a click at {clicked:?}: snapped to the feature?"
    );
    assert_eq!(selected, None, "the chord also selected a point");
    assert_eq!(detail.take_point_gesture(), None);
    assert_eq!(detail.zoom, 1.0);

    let mut detail = ImageDetail::new();
    let (track_at, selected) = chord_click(&mut detail, at, 1, egui::Modifiers::default());
    assert_eq!(track_at, None, "a plain click asked for a track");
    assert_eq!(selected, Some(point as usize));

    // Control alone, or Shift alone, is not the chord.
    for modifiers in [egui::Modifiers::CTRL, egui::Modifiers::SHIFT] {
        let mut detail = ImageDetail::new();
        assert_eq!(chord_click(&mut detail, at, 1, modifiers).0, None);
    }
}

/// A Control+Shift double-click on a feature is not Edit on Bench and not a
/// zoom: its second click is a second request the busy node will refuse, and
/// the panel asks for nothing else.
#[test]
fn a_control_shift_double_click_is_neither_edit_on_bench_nor_a_zoom() {
    let (node, _) = gesture_fixture();
    let image = pixels(1920, 1080);
    let (&feature, _) = node.recon().point_set.image_feature_to_point[0]
        .iter()
        .next()
        .expect("the demo's first image tracks features");
    let [fx, fy] = feature_pixel(feature as usize);
    let at = to_panel(&image, [f64::from(fx), f64::from(fy)]);

    let mut detail = ImageDetail::new();
    let _ = chord_click(&mut detail, at, 2, ctrl_shift());
    assert_eq!(
        detail.take_point_gesture(),
        None,
        "Edit on Bench was staged"
    );
    assert_eq!(
        detail.take_target_point(),
        None,
        "the chord moved the 3D view"
    );
    assert_eq!(detail.zoom, 1.0, "the chord zoomed");
}

/// What the entry needs, and the two ways of having nothing to stage told
/// apart: no feature at all, and a `.sift` keypoint the solve matched to no
/// point. Either is a sentence a reader can act on.
#[test]
fn edit_on_bench_needs_a_feature_with_a_point_behind_it() {
    assert_eq!(
        edit_on_bench_entry(BenchMenu::default(), Some(FeatureHit::Point(7))),
        Ok(7),
    );

    let why = edit_on_bench_entry(BenchMenu::default(), Some(FeatureHit::Unmatched))
        .expect_err("an unmatched keypoint names no point");
    assert!(why.contains("belongs to no 3D point"), "{why}");

    let why =
        edit_on_bench_entry(BenchMenu::default(), None).expect_err("empty image names no point");
    assert!(why.contains("no feature here"), "{why}");

    // A busy node refuses every step on it, in the state's own words, as the
    // two entries below this one do.
    assert_eq!(
        edit_on_bench_entry(
            BenchMenu {
                busy: Some(BUSY),
                active_track: None,
                lock: true,
                create_track: None,
            },
            Some(FeatureHit::Point(7)),
        ),
        Err(BUSY.to_string()),
    );
}

/// The hit test the entry and the double-click read: a tracked feature inside
/// the radius wins over an untracked one nearer the query, so the menu offers
/// the point a click there would have selected, and an untracked keypoint on
/// its own reports itself rather than nothing.
#[test]
fn the_hit_test_prefers_a_tracked_feature_and_still_reports_an_unmatched_one() {
    let feature = |x: f32, point: u32| super::DisplayFeature {
        position: [x, 0.0],
        affine_shape: [[4.0, 0.0], [0.0, 4.0]],
        point_index: point,
        max_track_angle_deg: f32::NAN,
        inverse_depth_z: f32::NAN,
        condition_number: f32::NAN,
    };
    let features = vec![feature(0.0, super::UNTRACKED), feature(3.0, 12)];
    let tree = super::build_feature_tree(&features);

    assert_eq!(
        feature_at(&features, &tree, &[0.5, 0.0], 8.0),
        Some(FeatureHit::Point(12)),
        "the untracked keypoint nearer the query swallowed the tracked one",
    );
    let alone = vec![feature(0.0, super::UNTRACKED)];
    let alone_tree = super::build_feature_tree(&alone);
    assert_eq!(
        feature_at(&alone, &alone_tree, &[0.5, 0.0], 8.0),
        Some(FeatureHit::Unmatched),
    );
    assert_eq!(feature_at(&features, &tree, &[80.0, 0.0], 8.0), None);
}

/// Where feature `i` of the gesture fixture sits in the source image: a grid
/// coarse enough that the click's 8-panel-pixel radius can only ever catch one
/// of them.
fn feature_pixel(i: usize) -> [f32; 2] {
    [
        200.0 + 100.0 * (i % 8) as f32,
        200.0 + 100.0 * (i / 8) as f32,
    ]
}

/// A `sift_files` node and the keypoint cache behind its image 0.
fn gesture_fixture() -> (SceneNode, crate::state::CachedSiftFeatures) {
    let node = demo_node("/runs/gestures.sfmr");
    let count = node.recon().point_set.max_track_feature_index[0] as usize + 1;
    let sift = crate::state::CachedSiftFeatures {
        positions_xy: (0..count).map(feature_pixel).collect(),
        affine_shapes: vec![[[6.0, 0.0], [0.0, 6.0]]; count],
        read_count: count,
    };
    (node, sift)
}

/// Two frames with the pointer at `at`: one to settle the panel's rects and
/// prepare the overlay, and one carrying `clicks` press/release pairs, which is
/// what egui counts a double-click out of.
fn click_frames(
    detail: &mut ImageDetail,
    node: &SceneNode,
    sift: &crate::state::CachedSiftFeatures,
    image: &ImageU8,
    at: egui::Pos2,
    clicks: usize,
) {
    let ctx = egui::Context::default();
    let display = FeatureDisplaySettings::default();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    crate::platform::set_test_pointer_pos(Some(at));
    for frame in 0..2 {
        let mut events = vec![egui::Event::PointerMoved(at)];
        if frame == 1 {
            for _ in 0..clicks {
                for pressed in [true, false] {
                    events.push(egui::Event::PointerButton {
                        pos: at,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    });
                }
            }
        }
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
            events,
            ..Default::default()
        };
        crate::test_support::run_frame_headless(&ctx, input, |ui| {
            detail.show(
                ui,
                node.edited(),
                node.id,
                node.history.current_version().serial,
                Some(0),
                None,
                None,
                None,
                BenchMenu::default(),
                &[],
                &crate::platform::ScrollInput::default(),
                Some(sift),
                Some(image),
                &display,
                &mut intrinsics_display,
            );
        });
    }
    crate::platform::set_test_pointer_pos(None);
}

/// The source pixel the panel is showing under `at`, read out of the view the
/// frame left behind. The anchor a zoom has to hold fixed.
fn pixel_under(detail: &ImageDetail, image: &ImageU8, at: egui::Pos2) -> egui::Vec2 {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let scale = (PANEL.x / w).min(PANEL.y / h) * detail.zoom;
    let display = egui::vec2(w * scale, h * scale);
    let origin = egui::pos2(PANEL.x / 2.0, PANEL.y / 2.0) + detail.pan - display / 2.0;
    (at - origin) / scale
}

/// Double-clicking a feature that observes a point is Edit on Bench: the panel
/// leaves behind the same request the viewport's menu entry reports, and the
/// view does not move. It also names the point for the 3D viewport to move its
/// target onto.
#[test]
fn double_clicking_a_feature_asks_for_its_point_on_the_bench() {
    let (node, sift) = gesture_fixture();
    let image = pixels(1920, 1080);
    let (&feature, &point) = node.recon().point_set.image_feature_to_point[0]
        .iter()
        .next()
        .expect("the demo's first image tracks features");
    let at = to_panel(&image, {
        let [x, y] = feature_pixel(feature as usize);
        [f64::from(x), f64::from(y)]
    });

    let mut detail = ImageDetail::new();
    click_frames(&mut detail, &node, &sift, &image, at, 2);

    assert_eq!(
        detail.take_point_gesture(),
        Some(PointGesture::EditOnBench(crate::scene::PointRef::new(
            node.id,
            point as usize,
        ))),
    );
    assert_eq!(detail.zoom, 1.0, "the gesture zoomed as well as staging");
    // One request, taken once: the frame drains it, and nothing is left for a
    // second bench item.
    assert_eq!(detail.take_point_gesture(), None);
    assert_eq!(
        detail.take_target_point(),
        Some(crate::scene::PointRef::new(node.id, point as usize)),
    );
    assert_eq!(detail.take_target_point(), None);
}

/// A single click on the same feature selects it and asks for nothing: the
/// gesture that stages a track is the double.
#[test]
fn a_single_click_on_a_feature_asks_for_nothing() {
    let (node, sift) = gesture_fixture();
    let image = pixels(1920, 1080);
    let (&feature, _) = node.recon().point_set.image_feature_to_point[0]
        .iter()
        .next()
        .expect("the demo's first image tracks features");
    let at = to_panel(&image, {
        let [x, y] = feature_pixel(feature as usize);
        [f64::from(x), f64::from(y)]
    });

    let mut detail = ImageDetail::new();
    click_frames(&mut detail, &node, &sift, &image, at, 1);

    assert_eq!(detail.take_point_gesture(), None);
    assert_eq!(detail.take_target_point(), None);
    assert_eq!(detail.zoom, 1.0, "a single click zoomed");
}

/// A double-click anywhere else zooms in one step, about the cursor: the pixel
/// under the pointer is the pixel still under it afterwards.
#[test]
fn double_clicking_off_a_feature_zooms_in_about_the_cursor() {
    let (node, sift) = gesture_fixture();
    let image = pixels(1920, 1080);
    // Well clear of the fixture's grid, and off-centre on both axes so a zoom
    // that ignored the cursor would move the pixel under it.
    let at = egui::pos2(700.0, 600.0);

    let mut detail = ImageDetail::new();
    click_frames(&mut detail, &node, &sift, &image, at, 1);
    let before = pixel_under(&detail, &image, at);
    let zoom = detail.zoom;

    click_frames(&mut detail, &node, &sift, &image, at, 2);

    assert_eq!(detail.take_point_gesture(), None, "empty image was staged");
    let ratio = detail.zoom / zoom;
    assert!(
        (ratio - std::f32::consts::SQRT_2).abs() < 1e-5,
        "the step was {ratio}x, not sqrt(2)",
    );
    let after = pixel_under(&detail, &image, at);
    assert!(
        (after.x - before.x).abs() < 1e-2 && (after.y - before.y).abs() < 1e-2,
        "the anchor pixel moved: {before:?} -> {after:?}",
    );
}

// ── The bench layer ─────────────────────────────────────────────────────

/// A node of [`projected_embedded_demo`] and the bench track its point 2 makes:
/// three observations, in images 0, 1 and 2, each at that point's exact
/// projection, standing on the patch the reconstruction stores for it.
fn bench_track_fixture() -> (SceneNode, sfmtool_core::bench::EditableTrack) {
    use sfmtool_core::bench::{create_track, Bench, CreateTrackOptions};

    let node = SceneNode::demo(crate::state::edits::tests::projected_embedded_demo(12));
    let (bench, report) = create_track(
        &Bench::new(),
        node.edited(),
        2,
        &CreateTrackOptions {
            version: 0,
            label: Some("bench-track".to_string()),
        },
    )
    .expect("point 2 is a live point with a track");
    let track = (**bench.track(&report.label).expect("just put on")).clone();
    (node, track)
}

/// Every path and line the frame painted in one of the bench's own colours.
///
/// The colour is what says a shape is the layer's: the overlays draw in greens,
/// greys and the colormaps, and nothing else in the panel draws in violet.
fn bench_shapes(
    node: &SceneNode,
    image_index: usize,
    bench: BenchMenu<'_>,
) -> Vec<Vec<egui::Pos2>> {
    let mut found = Vec::new();
    for clipped in &bench_frame(node, image_index, bench) {
        collect_bench_paths(&clipped.shape, &mut found);
    }
    found
}

/// Every bench-coloured **segment** the frame painted, with the colour it was
/// drawn in: the projection offsets, which are line segments rather than
/// paths and so invisible to [`bench_shapes`].
fn bench_segments(
    node: &SceneNode,
    image_index: usize,
    bench: BenchMenu<'_>,
) -> Vec<([egui::Pos2; 2], egui::Color32)> {
    let mut found = Vec::new();
    for clipped in &bench_frame(node, image_index, bench) {
        collect_bench_segments(&clipped.shape, &mut found);
    }
    found
}

/// One frame of the panel with `bench` on it, as the shapes it painted.
///
/// Two frames are run and the second one's shapes returned: the first loads
/// the image and prepares the overlay, and the layer draws over what is on
/// screen.
fn bench_frame(
    node: &SceneNode,
    image_index: usize,
    bench: BenchMenu<'_>,
) -> Vec<egui::epaint::ClippedShape> {
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let display = FeatureDisplaySettings::default();
    let image = pixels(640, 480);
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
        ..Default::default()
    };
    let mut shapes = Vec::new();
    for _ in 0..2 {
        let mut output = ctx.run_ui(input(), |ui| {
            detail.show(
                ui,
                node.edited(),
                node.id,
                node.history.current_version().serial,
                Some(image_index),
                None,
                None,
                None,
                bench,
                &[],
                &crate::platform::ScrollInput::default(),
                None,
                Some(&image),
                &display,
                &mut intrinsics_display,
            );
        });
        output.textures_delta.clear();
        shapes = output.shapes;
    }
    shapes
}

fn collect_bench_paths(shape: &egui::Shape, out: &mut Vec<Vec<egui::Pos2>>) {
    match shape {
        egui::Shape::Path(path) if is_bench_color(&path.stroke.color) => {
            out.push(path.points.clone());
        }
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_bench_paths(shape, out);
            }
        }
        _ => {}
    }
}

fn collect_bench_segments(shape: &egui::Shape, out: &mut Vec<([egui::Pos2; 2], egui::Color32)>) {
    match shape {
        egui::Shape::LineSegment { points, stroke }
            if [
                crate::bench::IN_COLOR,
                crate::bench::CANDIDATE_COLOR,
                crate::bench::OUT_COLOR,
            ]
            .contains(&stroke.color) =>
        {
            out.push((*points, stroke.color));
        }
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_bench_segments(shape, out);
            }
        }
        _ => {}
    }
}

fn is_bench_color(color: &egui::epaint::ColorMode) -> bool {
    matches!(
        color,
        egui::epaint::ColorMode::Solid(solid)
            if [
                crate::bench::IN_COLOR,
                crate::bench::CANDIDATE_COLOR,
                crate::bench::OUT_COLOR,
            ]
            .contains(solid)
    )
}

/// Where the panel puts one source pixel, for a frame drawn at the fixture's
/// panel size with the view unmoved. The panel fits the image and centres it,
/// which is the whole of the mapping at zoom 1.
fn to_panel(image: &sfmtool_core::camera::remap::ImageU8, pixel: [f64; 2]) -> egui::Pos2 {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let scale = (PANEL.x / w).min(PANEL.y / h);
    let origin = egui::pos2(
        PANEL.x / 2.0 - w * scale / 2.0,
        PANEL.y / 2.0 - h * scale / 2.0,
    );
    egui::pos2(
        origin.x + pixel[0] as f32 * scale,
        origin.y + pixel[1] as f32 * scale,
    )
}

/// The patch's outline is drawn where the frame's corners really project, and
/// it is a sampled curve rather than the four corners joined up.
#[test]
fn the_bench_layer_outlines_the_patch_where_its_corners_project() {
    use sfmtool_core::geometry::RigidTransform;

    let (node, track) = bench_track_fixture();
    let paths = bench_shapes(
        &node,
        0,
        BenchMenu {
            busy: None,
            active_track: Some(&track),
            lock: true,
            create_track: None,
        },
    );
    assert!(!paths.is_empty(), "the layer drew nothing");
    let outline = paths
        .iter()
        .max_by_key(|path| path.len())
        .expect("a path was drawn");
    assert!(
        outline.len() > 4,
        "the outline is the four corners rather than a sampled curve: {outline:?}",
    );

    // The corners, projected here rather than through the layer's own code.
    let frame = track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("a track from a point carries the stored patch as its frame");
    let table = &node.edited().base.image_table;
    let image = &table.images[0];
    let camera = &table.cameras[image.camera_index as usize];
    let q = image.quaternion_wxyz.quaternion();
    let pose = RigidTransform::from_wxyz_translation(
        [q.w, q.i, q.j, q.k],
        [
            image.translation_xyz.x,
            image.translation_xyz.y,
            image.translation_xyz.z,
        ],
    );
    let photograph = pixels(640, 480);
    // The outline is the frame re-anchored on this image's keypoint, as the
    // tile is rendered, so the corners are the anchored frame's.
    let keypoint = track
        .observations
        .iter()
        .find(|o| o.image == 0)
        .and_then(|o| o.track.as_ref())
        .and_then(|m| m.keypoint)
        .map(|k| [f64::from(k[0]), f64::from(k[1])])
        .expect("a track from a point carries its keypoints");
    let anchored = frame.anchored_at_keypoint(camera, &pose, keypoint);
    let frame = anchored.as_ref().unwrap_or(frame);
    for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let (xyz, w) = frame.corner_homogeneous(s, t);
        let p = pose.transform_point_homogeneous(xyz, w);
        let (u, v) = camera
            .ray_to_pixel([p.x, p.y, p.z])
            .expect("the demo's patch corners are in front of the camera");
        let expected = to_panel(&photograph, [u, v]);
        let nearest = outline
            .iter()
            .map(|point| (*point - expected).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 0.01,
            "the outline misses the corner ({s}, {t}) at {expected:?} by {nearest}",
        );
    }
}

/// The projection offset is drawn for **every** observation, whatever its
/// verdict, in that observation's own colour: it runs from the sighting's
/// keypoint to where the patch projects, which is the number the `Proj. off`
/// column carries.
#[test]
fn the_bench_layer_draws_the_projection_offset_for_every_verdict() {
    use sfmtool_core::bench::Verdict;
    use sfmtool_core::geometry::RigidTransform;

    let (node, track) = bench_track_fixture();
    // Where the patch's centre lands in this image, projected here rather
    // than through the layer's own code.
    let frame = track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("a track from a point carries the stored patch as its frame");
    let table = &node.edited().base.image_table;
    let image = &table.images[0];
    let camera = &table.cameras[image.camera_index as usize];
    let q = image.quaternion_wxyz.quaternion();
    let pose = RigidTransform::from_wxyz_translation(
        [q.w, q.i, q.j, q.k],
        [
            image.translation_xyz.x,
            image.translation_xyz.y,
            image.translation_xyz.z,
        ],
    );
    let centre = {
        let p = pose.transform_point_homogeneous(frame.center.coords, frame.w);
        let (u, v) = camera
            .ray_to_pixel([p.x, p.y, p.z])
            .expect("the demo's patch is in front of the camera");
        to_panel(&pixels(640, 480), [u, v])
    };
    let keypoint = track
        .observations
        .iter()
        .find(|o| o.image == 0)
        .and_then(|o| o.track.as_ref())
        .and_then(|m| m.keypoint)
        .map(|k| to_panel(&pixels(640, 480), [f64::from(k[0]), f64::from(k[1])]))
        .expect("a track from a point carries its keypoints");

    for (verdict, expected) in [
        (Verdict::In, crate::bench::IN_COLOR),
        (Verdict::Candidate, crate::bench::CANDIDATE_COLOR),
        (Verdict::Out, crate::bench::OUT_COLOR),
    ] {
        let mut judged = track.clone();
        for observation in &mut judged.observations {
            observation.verdict = verdict;
        }
        let segments = bench_segments(
            &node,
            0,
            BenchMenu {
                busy: None,
                active_track: Some(&judged),
                lock: true,
                create_track: None,
            },
        );
        assert_eq!(
            segments.len(),
            1,
            "{verdict:?} drew {} offset segments rather than one",
            segments.len()
        );
        let ([from, to], color) = segments[0];
        assert_eq!(color, expected, "{verdict:?} drew the segment in {color:?}");
        assert!(
            (from - keypoint).length() < 0.01,
            "{verdict:?}: the segment starts at {from:?}, not at the keypoint {keypoint:?}"
        );
        assert!(
            (to - centre).length() < 0.01,
            "{verdict:?}: the segment ends at {to:?}, not at the projection {centre:?}"
        );
    }
}

/// The layer is the *active* track's, and its marks are only in the images
/// that track observes: an empty bench draws nothing at all, and a photograph
/// outside the track draws no opaque mark (only the ghost, below).
#[test]
fn the_bench_layer_draws_nothing_without_a_track_and_no_mark_outside_it() {
    let (node, track) = bench_track_fixture();
    assert!(
        bench_shapes(&node, 0, BenchMenu::default()).is_empty(),
        "the layer drew with nothing on the bench",
    );
    assert!(
        ghost_shapes(&node, 0, BenchMenu::default()).is_empty(),
        "a ghost was drawn with nothing on the bench",
    );
    let seen: Vec<u32> = track.observations.iter().map(|o| o.image).collect();
    let unseen = (0..node.edited().image_count())
        .find(|i| !seen.contains(&(*i as u32)))
        .expect("the fixture's track does not span every image");
    assert!(
        bench_shapes(
            &node,
            unseen,
            BenchMenu {
                busy: None,
                active_track: Some(&track),
                lock: true,
                create_track: None,
            },
        )
        .is_empty(),
        "an opaque mark was drawn in an image the track does not observe",
    );
}

// ── The ghost outline ───────────────────────────────────────────────────

/// Every path the frame painted in the ghost outline's colour.
fn ghost_shapes(
    node: &SceneNode,
    image_index: usize,
    bench: BenchMenu<'_>,
) -> Vec<Vec<egui::Pos2>> {
    let ghost = egui::epaint::ColorMode::Solid(super::bench_track::ghost_color());
    let mut found = Vec::new();
    for clipped in &bench_frame(node, image_index, bench) {
        collect_paths_in(&clipped.shape, &ghost, &mut found);
    }
    found
}

fn collect_paths_in(
    shape: &egui::Shape,
    color: &egui::epaint::ColorMode,
    out: &mut Vec<Vec<egui::Pos2>>,
) {
    match shape {
        egui::Shape::Path(path) if &path.stroke.color == color => {
            out.push(path.points.clone());
        }
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_paths_in(shape, color, out);
            }
        }
        _ => {}
    }
}

/// The fixture's track with its patch turned to face the first image the track
/// has no observation in, and that image: the photograph a ghost is drawn in.
///
/// The demo's patches face the cameras that see them, so an image outside the
/// track sees each one from well off its normal. Turning the patch square to
/// that camera makes the ghost's view of it unambiguous without moving its
/// centre, and the images the track observes keep their sightings.
fn ghost_fixture() -> (SceneNode, sfmtool_core::bench::EditableTrack, usize) {
    let (node, track) = bench_track_fixture();
    let unseen = (0..node.edited().image_count())
        .find(|i| track.observations.iter().all(|o| o.image as usize != *i))
        .expect("the fixture's track does not span every image");
    let eye = node.edited().base.image_table.images[unseen].camera_center();
    let track = with_placement(&track, |placement| {
        let patch = placement
            .as_mut()
            .expect("a track from a point carries a patch");
        *patch = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
            patch.center,
            eye - patch.center,
            nalgebra::Vector3::z(),
            patch.half_extent,
        );
    });
    (node, track, unseen)
}

/// The fixture's track with its placement changed by `change`.
fn with_placement(
    track: &sfmtool_core::bench::EditableTrack,
    change: impl FnOnce(&mut Option<sfmtool_core::patch::cloud::OrientedPatch>),
) -> sfmtool_core::bench::EditableTrack {
    let mut changed = track.clone();
    match &mut changed.stage {
        sfmtool_core::bench::Stage::Track(payload) => change(&mut payload.placement),
        sfmtool_core::bench::Stage::Cluster(_) => panic!("the fixture is at the track stage"),
    }
    changed
}

/// In a photograph the track has no sighting in, the track stage draws the
/// patch's own square where that camera sees it, at the ghost opacity, and
/// nothing opaque: no dot, no segment, no member outline.
#[test]
fn the_bench_layer_ghosts_the_patch_in_an_image_the_track_does_not_observe() {
    let (node, track, unseen) = ghost_fixture();
    let photograph = pixels(640, 480);
    let bench = BenchMenu {
        busy: None,
        active_track: Some(&track),
        lock: true,
        create_track: None,
    };
    let ghosts = ghost_shapes(&node, unseen, bench);
    assert_eq!(ghosts.len(), 1, "one ghost outline, got {}", ghosts.len());
    let outline = &ghosts[0];
    assert!(
        outline.len() > 4,
        "the ghost is the four corners rather than a sampled curve: {outline:?}",
    );
    let alpha = super::bench_track::ghost_color().a();
    let expected_alpha = (255.0 * super::bench_track::GHOST_OPACITY).round() as u8;
    assert!(
        alpha.abs_diff(expected_alpha) <= 1,
        "the ghost is drawn at alpha {alpha}, not at the ghost opacity ({expected_alpha})",
    );

    // The patch itself, not re-anchored on anything: there is no sighting here.
    let patch = track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("a track from a point carries the stored patch");
    let (camera, pose) = crate::bench::geometry::view_of(&node.edited().base.image_table, unseen)
        .expect("the demo's images have cameras");
    for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let expected = to_panel(&photograph, patch_pixel(patch, &camera, &pose, s, t));
        let nearest = outline
            .iter()
            .map(|point| (*point - expected).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 0.01,
            "the ghost misses the corner ({s}, {t}) at {expected:?} by {nearest}",
        );
    }
    assert!(
        bench_shapes(&node, unseen, bench).is_empty(),
        "an opaque outline was drawn in an image the track does not observe",
    );
    assert!(
        bench_segments(&node, unseen, bench).is_empty(),
        "a projection-offset segment was drawn where there is no sighting",
    );
}

/// An image the track observes keeps its own drawing and gets no ghost, and
/// that holds for a `candidate` or an `out` sighting as much as an `in` one:
/// any observation makes the image the track's.
#[test]
fn a_member_image_draws_its_outline_and_no_ghost_whatever_the_verdict() {
    use sfmtool_core::bench::Verdict;

    let (node, track) = bench_track_fixture();
    for verdict in [Verdict::In, Verdict::Candidate, Verdict::Out] {
        let mut judged = track.clone();
        for observation in &mut judged.observations {
            observation.verdict = verdict;
        }
        let bench = BenchMenu {
            busy: None,
            active_track: Some(&judged),
            lock: true,
            create_track: None,
        };
        for image in judged.observations.iter().map(|o| o.image as usize) {
            assert!(
                ghost_shapes(&node, image, bench).is_empty(),
                "{verdict:?}: image {image} is the track's and drew a ghost",
            );
            assert!(
                !bench_shapes(&node, image, bench).is_empty(),
                "{verdict:?}: image {image} lost its member outline",
            );
        }
    }
}

/// The cluster stage has no shared geometry, so there is nothing to project
/// into a photograph the cluster has no sighting in.
#[test]
fn the_cluster_stage_draws_no_ghost() {
    use sfmtool_core::bench::{create_cluster, Bench, ClusterSeed};

    let node = SceneNode::demo(crate::state::edits::tests::projected_embedded_demo(12));
    let (bench, report) = create_cluster(
        &Bench::new(),
        &ClusterSeed::from_pixel(0, "image_0", [320.0, 240.0], 24.0),
    )
    .expect("a usable seed");
    let track = (**bench.track(&report.label).expect("just put on")).clone();
    let menu = BenchMenu {
        busy: None,
        active_track: Some(&track),
        lock: true,
        create_track: None,
    };
    for image in 1..node.edited().image_count() {
        assert!(
            ghost_shapes(&node, image, menu).is_empty(),
            "the cluster drew a ghost in image {image}",
        );
        assert!(
            bench_shapes(&node, image, menu).is_empty(),
            "the cluster drew in image {image}, which it has no sighting in",
        );
    }
}

/// A track-stage track with no placement has no square to project, and a
/// patch whose back is turned to the camera, or whose plane that camera sees
/// edge-on, is not one that photograph sees.
#[test]
fn no_ghost_is_drawn_without_a_placement_or_for_a_patch_it_cannot_see() {
    let (node, track, unseen) = ghost_fixture();
    let menu = |track| BenchMenu {
        busy: None,
        active_track: Some(track),
        lock: true,
        create_track: None,
    };
    assert!(
        !ghost_shapes(&node, unseen, menu(&track)).is_empty(),
        "the fixture should draw a ghost before anything is changed",
    );

    let unplaced = with_placement(&track, |placement| *placement = None);
    assert!(
        ghost_shapes(
            &node,
            unseen,
            BenchMenu {
                busy: None,
                active_track: Some(&unplaced),
                lock: true,
                create_track: None,
            },
        )
        .is_empty(),
        "a ghost was drawn for a track with no placement",
    );

    // Swapping the axes turns the normal over and leaves the square where it
    // was, so the only thing that changed is which face the camera sees.
    let turned = with_placement(&track, |placement| {
        let patch = placement.as_mut().expect("the fixture has a placement");
        std::mem::swap(&mut patch.u_axis, &mut patch.v_axis);
    });
    assert!(
        ghost_shapes(
            &node,
            unseen,
            BenchMenu {
                busy: None,
                active_track: Some(&turned),
                lock: true,
                create_track: None,
            },
        )
        .is_empty(),
        "a ghost was drawn for a patch whose back faces the camera",
    );

    // Turned a right angle about the vertical, the plane holds the line of
    // sight: the square is seen edge-on and projects to a line.
    let eye = node.edited().base.image_table.images[unseen].camera_center();
    let edge_on = with_placement(&track, |placement| {
        let patch = placement.as_mut().expect("the fixture has a placement");
        *patch = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
            patch.center,
            (eye - patch.center).cross(&nalgebra::Vector3::z()),
            nalgebra::Vector3::z(),
            patch.half_extent,
        );
    });
    assert!(
        ghost_shapes(&node, unseen, menu(&edge_on)).is_empty(),
        "a ghost was drawn for a patch seen edge-on",
    );
}

/// With the lock cleared the ghost is display only: a press on its corner takes
/// no handle, asks for no cursor, pans the photograph as a press on empty image
/// does, and edits nothing.
#[test]
fn with_the_lock_off_a_press_on_the_ghost_pans_and_edits_nothing() {
    let (node, track, unseen) = ghost_fixture();
    let patch = track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("a track from a point carries the stored patch");
    let (camera, pose) = crate::bench::geometry::view_of(&node.edited().base.image_table, unseen)
        .expect("the demo's images have cameras");
    let centre = patch_pixel(patch, &camera, &pose, 0.0, 0.0);
    let corner = patch_pixel(patch, &camera, &pose, 1.0, 1.0);
    let dragged = gesture(
        &node,
        unseen,
        &track,
        centre,
        corner,
        &[egui::vec2(30.0, 0.0)],
        false,
        false,
    );
    assert_eq!(dragged.edit, None, "a press on the ghost edited the patch");
    assert_eq!(
        dragged.cursor,
        egui::CursorIcon::Default,
        "the ghost asked for a cursor",
    );
    assert!(
        dragged.panned.length() > 1.0,
        "a drag from the ghost should pan the photograph, moved {:?}",
        dragged.panned,
    );
}

/// A cluster started from a pixel is drawn at the size that gesture asked for,
/// before anything has evaluated it.
///
/// The seed's shape is in keypoint-frame units and the cluster's radius is what
/// turns them into pixels, so a layer that read the shape without the radius
/// would draw the square several times over.
#[test]
fn the_bench_layer_draws_a_pixel_cluster_at_the_radius_it_was_started_with() {
    use sfmtool_core::bench::{create_cluster, Bench, ClusterSeed};

    let node = SceneNode::demo(crate::state::edits::tests::projected_embedded_demo(12));
    let pixel = [320.0, 240.0];
    let radius_px = 24.0;
    let (bench, report) = create_cluster(
        &Bench::new(),
        &ClusterSeed::from_pixel(0, "image_0", pixel, radius_px),
    )
    .expect("a usable seed");
    let track = (**bench.track(&report.label).expect("just put on")).clone();

    let paths = bench_shapes(
        &node,
        0,
        BenchMenu {
            busy: None,
            active_track: Some(&track),
            lock: true,
            create_track: None,
        },
    );
    let outline = paths
        .iter()
        .find(|path| path.len() == 4)
        .expect("the seed's square was drawn");
    let photograph = pixels(640, 480);
    for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let expected = to_panel(
            &photograph,
            [pixel[0] + s * radius_px, pixel[1] + t * radius_px],
        );
        let nearest = outline
            .iter()
            .map(|point| (*point - expected).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 0.01,
            "the seed's corner ({s}, {t}) should be at {expected:?}, nearest drawn is {nearest} away",
        );
    }
}

/// The same frame, giving back what the panel published about the view it
/// ended on -- which is what the dock puts on `AppState` for the wire's view
/// tools to read.
fn published_frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    image_index: usize,
    image: &ImageU8,
    panel: egui::Vec2,
    look: Option<super::Look>,
) -> super::ViewGeometry {
    let feature_display = FeatureDisplaySettings::default();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, panel)),
        ..Default::default()
    };
    let mut published = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        let response = detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(image_index),
            look,
            None,
            None,
            super::BenchMenu::default(),
            &[],
            &crate::platform::ScrollInput::default(),
            None,
            Some(image),
            &feature_display,
            &mut intrinsics_display,
        );
        published = response.view;
    });
    published.expect("a frame that drew an image publishes its view")
}

/// The centre of a published view's visible rectangle, in image pixels: the
/// place the panel is looking at, which is what every look aims.
fn looked_at(view: super::ViewGeometry) -> [f32; 2] {
    let [x0, y0, x1, y1] = view.visible_rect();
    [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
}

/// The reveal and the wire's own look are one function: a row click centres
/// the pixel without touching the zoom, and the panel publishes a view whose
/// visible rectangle is centred on it -- which is what
/// `get_image_detail_view` reports.
#[test]
fn the_published_view_is_centred_on_what_the_look_named() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    detail.zoom = 4.0;
    let feature = [10.0, 10.0];
    let revealed = published_frame(&mut detail, &ctx, &node, 0, &image, PANEL, reveal(feature));
    let at = looked_at(revealed);
    assert!(
        (at[0] - feature[0]).abs() < 0.05 && (at[1] - feature[1]).abs() < 0.05,
        "the published view is not centred on the revealed pixel: {at:?}"
    );
    assert_eq!(revealed.zoom, 4.0, "a reveal zoomed");
    assert_eq!(
        revealed.image_size,
        [SOURCE.x, SOURCE.y],
        "the published image size is not the photograph's"
    );
    assert_eq!(
        revealed.panel_size,
        [PANEL.x, PANEL.y],
        "the published panel size is not the panel's"
    );
}

/// The wire's forms go through the same door: a pixel with a zoom, a rectangle
/// fitted to the panel, and the whole photograph.
#[test]
fn a_look_at_a_pixel_a_rect_and_the_whole_photograph_all_land() {
    let node = demo_node("/runs/demo.sfmr");
    let image = pixels(SOURCE.x as u32, SOURCE.y as u32);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();

    let pixel = published_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        PANEL,
        Some(super::Look::Pixel {
            pixel: [300.0, 200.0],
            zoom: Some(8.0),
        }),
    );
    assert_eq!(pixel.zoom, 8.0);
    let at = looked_at(pixel);
    assert!(
        (at[0] - 300.0).abs() < 0.05 && (at[1] - 200.0).abs() < 0.05,
        "the pixel is off centre: {at:?}"
    );

    let rect = published_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        PANEL,
        Some(super::Look::Rect([100.0, 100.0, 300.0, 200.0])),
    );
    let [x0, y0, x1, y1] = rect.visible_rect();
    assert!(
        x1 - x0 >= 200.0 - 0.05 && y1 - y0 >= 100.0 - 0.05,
        "the rectangle does not fit: {:?}",
        rect.visible_rect()
    );
    let at = looked_at(rect);
    assert!(
        (at[0] - 200.0).abs() < 0.05 && (at[1] - 150.0).abs() < 0.05,
        "the rectangle is off centre: {at:?}"
    );

    let fitted = published_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        PANEL,
        Some(super::Look::Fit),
    );
    assert_eq!(fitted.zoom, 1.0);
    assert_eq!(fitted.pan, [0.0, 0.0]);
    let at = looked_at(fitted);
    assert!(
        (at[0] - SOURCE.x / 2.0).abs() < 0.05 && (at[1] - SOURCE.y / 2.0).abs() < 0.05,
        "the whole photograph is off centre: {at:?}"
    );
}

// ── The bench layer's handles ───────────────────────────────────────────

/// How far in the panel is zoomed for the handle tests.
///
/// The demo's patch is under three source pixels across, which at fit-to-panel
/// is a two-pixel outline: every handle would sit inside every other one's
/// reach, and a test that grabbed an edge would be asserting nothing. Zoomed,
/// the outline is tens of panel pixels wide, which is the size a person
/// actually works a patch at.
const HANDLE_ZOOM: f32 = 16.0;

/// Frame the panel on `center_on` at `HANDLE_ZOOM`, and give back the mapping
/// from source pixels to panel positions that framing produces.
///
/// Set before the first frame, as the view tests above do, so the recorded
/// extent and the zoom agree the way they do after a real gesture.
fn framed(
    detail: &mut ImageDetail,
    image: &ImageU8,
    center_on: [f64; 2],
) -> impl Fn([f64; 2]) -> egui::Pos2 {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let scale = panel_scale(image.width(), image.height());
    let display = egui::vec2(w * scale, h * scale);
    detail.zoom = HANDLE_ZOOM;
    detail.pan =
        display / 2.0 - egui::vec2(center_on[0] as f32 * scale, center_on[1] as f32 * scale);
    let center = egui::pos2(PANEL.x / 2.0, PANEL.y / 2.0);
    move |p: [f64; 2]| {
        egui::pos2(
            center.x + (p[0] - center_on[0]) as f32 * scale,
            center.y + (p[1] - center_on[1]) as f32 * scale,
        )
    }
}

/// What one gesture over the bench layer produced.
struct Dragged {
    /// The edit the panel published on release, if the pointer had hold of a
    /// handle.
    edit: Option<crate::bench::PatchEdit>,
    /// The cursor the panel asked for while the pointer hovered the handle,
    /// before any button went down.
    cursor: egui::CursorIcon,
    /// How far the **view** moved over the gesture, in panel px. Zero is the
    /// claim a handle drag makes: the photograph holds still for the whole of
    /// it.
    panned: egui::Vec2,
    /// The bench-coloured segments the last frame before the release painted:
    /// the preview, drawn while the handle is still held.
    held_segments: Vec<([egui::Pos2; 2], egui::Color32)>,
    /// Every path that same frame painted, with its colour: the outlines and
    /// the normal's arrow of the preview.
    held_paths: Vec<(Vec<egui::Pos2>, egui::epaint::ColorMode)>,
}

/// Drive one press-move-release over the bench layer and report what it
/// produced.
///
/// `press` is in source-image pixels; `steps` are pointer positions after it,
/// as offsets from the press **in panel pixels**, so a test can put one below
/// egui's own drag threshold and the next above it. That distinction is the
/// whole of what the press-decides-the-handle rule is about: the view pans on
/// the first pixel of motion, and egui does not call the gesture a drag until
/// several.
///
/// The frames are the gesture: one to load the photograph, one hovering (which
/// is where the cursor is read), one that presses, one per step, and one that
/// releases.
///
/// `lock` is Track View's *Lock*, handed to the panel as the dock hands it.
#[allow(clippy::too_many_arguments)]
fn gesture(
    node: &SceneNode,
    image_index: usize,
    track: &sfmtool_core::bench::EditableTrack,
    center_on: [f64; 2],
    press: [f64; 2],
    steps: &[egui::Vec2],
    escape: bool,
    lock: bool,
) -> Dragged {
    let image = pixels(1920, 1080);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let display = FeatureDisplaySettings::default();
    let panel = framed(&mut detail, &image, center_on);
    let from = panel(press);
    let was = detail.pan;

    let button = |pos: egui::Pos2, pressed: bool| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let escape_key = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let last = steps.last().map_or(from, |step| from + *step);
    let mut frames: Vec<Vec<egui::Event>> = vec![
        Vec::new(),
        vec![egui::Event::PointerMoved(from)],
        vec![egui::Event::PointerMoved(from), button(from, true)],
    ];
    for step in steps {
        frames.push(vec![egui::Event::PointerMoved(from + *step)]);
    }
    frames.push(if escape {
        vec![egui::Event::PointerMoved(last), escape_key]
    } else {
        vec![egui::Event::PointerMoved(last)]
    });
    frames.push(vec![egui::Event::PointerMoved(last), button(last, false)]);

    let mut edit = None;
    let mut cursor = egui::CursorIcon::Default;
    let mut held_segments = Vec::new();
    let mut held_paths = Vec::new();
    let release = frames.len() - 1;
    for (index, events) in frames.into_iter().enumerate() {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
            events,
            ..Default::default()
        };
        let mut response = None;
        let mut output = ctx.run_ui(input, |ui| {
            response = Some(detail.show(
                ui,
                node.edited(),
                node.id,
                node.history.current_version().serial,
                Some(image_index),
                None,
                None,
                None,
                BenchMenu {
                    busy: None,
                    active_track: Some(track),
                    lock,
                    create_track: None,
                },
                &[],
                &crate::platform::ScrollInput::default(),
                None,
                Some(&image),
                &display,
                &mut intrinsics_display,
            ));
        });
        output.textures_delta.clear();
        // The hover frame, before any button is down: what the layer asks for
        // there is the cursor a person sees over the handle.
        if index == 1 {
            cursor = output.platform_output.cursor_icon;
        }
        if index + 1 == release {
            for clipped in &output.shapes {
                collect_bench_segments(&clipped.shape, &mut held_segments);
                collect_all_paths(&clipped.shape, &mut held_paths);
            }
        }
        if let Some(from_panel) = response.and_then(|response| response.bench_edit) {
            edit = Some(from_panel);
        }
    }
    Dragged {
        edit,
        cursor,
        panned: detail.pan - was,
        held_segments,
        held_paths,
    }
}

fn collect_all_paths(
    shape: &egui::Shape,
    out: &mut Vec<(Vec<egui::Pos2>, egui::epaint::ColorMode)>,
) {
    match shape {
        egui::Shape::Path(path) => out.push((path.points.clone(), path.stroke.color.clone())),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_all_paths(shape, out);
            }
        }
        _ => {}
    }
}

/// The panel mapping [`gesture`] frames its view with, for a test that reads
/// the preview it painted: source pixels to panel positions.
fn gesture_panel(center_on: [f64; 2]) -> impl Fn([f64; 2]) -> egui::Pos2 {
    let mut detail = ImageDetail::new();
    framed(&mut detail, &pixels(1920, 1080), center_on)
}

/// One press-move-release between two places named in **source-image** pixels.
fn bench_drag(
    node: &SceneNode,
    image_index: usize,
    track: &sfmtool_core::bench::EditableTrack,
    center_on: [f64; 2],
    from: [f64; 2],
    to: [f64; 2],
    escape: bool,
) -> Dragged {
    let scale = panel_scale(1920, 1080);
    let step = egui::vec2(
        (to[0] - from[0]) as f32 * scale,
        (to[1] - from[1]) as f32 * scale,
    );
    gesture(
        node,
        image_index,
        track,
        center_on,
        from,
        &[step],
        escape,
        true,
    )
}

/// [`bench_drag`] with Track View's *Lock* cleared.
fn unlocked_drag(
    node: &SceneNode,
    image_index: usize,
    track: &sfmtool_core::bench::EditableTrack,
    center_on: [f64; 2],
    from: [f64; 2],
    to: [f64; 2],
) -> Dragged {
    let scale = panel_scale(1920, 1080);
    let step = egui::vec2(
        (to[0] - from[0]) as f32 * scale,
        (to[1] - from[1]) as f32 * scale,
    );
    gesture(
        node,
        image_index,
        track,
        center_on,
        from,
        &[step],
        false,
        false,
    )
}

/// Panel pixels per source pixel at [`HANDLE_ZOOM`].
fn panel_scale(width: u32, height: u32) -> f32 {
    (PANEL.x / width as f32).min(PANEL.y / height as f32) * HANDLE_ZOOM
}

/// The node, its bench and the track put on it, in a state that can push
/// versions: the panel publishes the edit and `AppState` is what turns it into
/// one, exactly as the dock does.
fn bench_state() -> (crate::state::AppState, crate::scene::ReconId, String) {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    (state, id, label)
}

/// The track on the bench, as an owned value.
fn on_bench(
    state: &crate::state::AppState,
    id: crate::scene::ReconId,
    label: &str,
) -> sfmtool_core::bench::EditableTrack {
    (**state.bench_track(id, label).expect("a track by that label")).clone()
}

/// The version labels the node holds, oldest first.
fn version_labels(state: &crate::state::AppState, id: crate::scene::ReconId) -> Vec<String> {
    state
        .node(id)
        .expect("loaded")
        .history
        .versions()
        .iter()
        .map(|version| version.label.clone())
        .collect()
}

/// The patch re-anchored on one sighting: the outline the layer draws there,
/// and the geometry every handle of it is placed by.
fn outline_at(
    node: &SceneNode,
    track: &sfmtool_core::bench::EditableTrack,
    observation: usize,
) -> (
    sfmtool_core::patch::cloud::OrientedPatch,
    sfmtool_core::camera::CameraIntrinsics,
    sfmtool_core::geometry::RigidTransform,
) {
    let sighting = &track.observations[observation];
    let table = &node.edited().base.image_table;
    let (camera, pose) = crate::bench::geometry::view_of(table, sighting.image as usize)
        .expect("the fixture's images have cameras");
    let frame = track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("a track from a point carries the stored patch");
    let anchored = crate::bench::geometry::anchored_frame(frame, &camera, &pose, sighting);
    (anchored, camera, pose)
}

/// Each sighting's offset from where the patch's centre projects, in its own
/// image's pixels.
///
/// **What a move of the patch must not disturb**: the gap between where a
/// photograph sees the patch's content and where the geometry puts its middle
/// is what the tile is cut on, so a step that reset every keypoint to the
/// centre's projection would zero all of them and scramble the correlation.
fn projection_offsets(
    node: &SceneNode,
    track: &sfmtool_core::bench::EditableTrack,
) -> Vec<[f64; 2]> {
    let frame = track
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a track from a point carries the stored patch");
    track
        .observations
        .iter()
        .map(|observation| {
            let (camera, pose) = crate::bench::geometry::view_of(
                &node.edited().base.image_table,
                observation.image as usize,
            )
            .expect("the fixture's images have cameras");
            let centre =
                crate::bench::geometry::project(&camera, &pose, frame.center.coords, frame.w)
                    .expect("the demo's patch is in front of every camera");
            let site = observation.site().expect("a sighting");
            [site[0] - centre[0], site[1] - centre[1]]
        })
        .collect()
}

/// Where a patch's `(s, t)` corner lands, in source-image px.
fn patch_pixel(
    patch: &sfmtool_core::patch::cloud::OrientedPatch,
    camera: &sfmtool_core::camera::CameraIntrinsics,
    pose: &sfmtool_core::geometry::RigidTransform,
    s: f64,
    t: f64,
) -> [f64; 2] {
    let (xyz, w) = patch.corner_homogeneous(s, t);
    crate::bench::geometry::project(camera, pose, xyz, w).expect("the demo's patch is in front")
}

#[test]
fn hovering_an_edge_of_the_outline_asks_for_the_resize_cursor_its_orientation_names() {
    let (state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let node = &state.scene[0];
    let (frame, camera, pose) = outline_at(node, &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);

    // The `+u` edge of this fixture's patch runs all but horizontally on
    // screen, so it is the edge you move up and down.
    let across = patch_pixel(&frame, &camera, &pose, 1.0, 0.0);
    let along = patch_pixel(&frame, &camera, &pose, 0.0, 1.0);
    let steepness = |edge: [f64; 2]| ((edge[1] - centre[1]) / (edge[0] - centre[0])).abs();
    assert!(
        steepness(across) < 0.4 && steepness(along) > 2.5,
        "the fixture's outline is meant to be all but axis-aligned: {across:?} {along:?}",
    );

    // Hovering the `+u` edge's midpoint, which runs vertically on screen.
    let dragged = bench_drag(node, 0, &track, centre, across, across, false);
    assert_eq!(dragged.cursor, egui::CursorIcon::ResizeHorizontal);
    // And the `+v` edge, which runs horizontally.
    let dragged = bench_drag(node, 0, &track, centre, along, along, false);
    assert_eq!(dragged.cursor, egui::CursorIcon::ResizeVertical);
    // A corner spins the patch, so it takes the resize cursor along the arc it
    // travels. On an all but axis-aligned outline its radius from the centre
    // runs diagonally, and the tangent is the other diagonal: a corner down
    // and to the right of the centre (the raster's `y` runs downward) travels
    // north-east to south-west, and one up and to the right travels
    // north-west to south-east. The outline is not square on screen, so the
    // radius is not at 45 degrees; what matters is that it sits well inside
    // the sector a diagonal cursor answers for, a slope between `tan 22.5°`
    // and `tan 67.5°`.
    for (s, t) in [(1.0, 1.0), (1.0, -1.0)] {
        let corner = patch_pixel(&frame, &camera, &pose, s, t);
        let radius = [corner[0] - centre[0], corner[1] - centre[1]];
        let slope = (radius[1] / radius[0]).abs();
        assert!(
            (0.5..2.0).contains(&slope),
            "the corner's radius is meant to run clearly diagonally: {radius:?}",
        );
        let tangent = if radius[0] * radius[1] > 0.0 {
            egui::CursorIcon::ResizeNeSw
        } else {
            egui::CursorIcon::ResizeNwSe
        };
        let dragged = bench_drag(node, 0, &track, centre, corner, corner, false);
        assert_eq!(dragged.cursor, tangent, "corner ({s}, {t})");
    }
    // The sighting's own dot moves it.
    let dragged = bench_drag(node, 0, &track, centre, centre, centre, false);
    assert_eq!(dragged.cursor, egui::CursorIcon::Move);
}

/// The dot at the track stage moves the **patch**, not one sighting: there is
/// one patch and every observation is a view of it, so the dot lands under the
/// pointer in the image it was dragged in and every other sighting goes to
/// where the moved centre projects in its own photograph.
#[test]
fn dragging_the_dot_slides_the_patch_and_every_sighting_follows_it() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let was = track.observations[0].site().expect("a sighting");
    let to = [was[0] + 1.5, was[1] + 2.0];
    let offsets = projection_offsets(&state.scene[0], &track);

    let edit = bench_drag(&state.scene[0], 0, &track, was, was, to, false)
        .edit
        .expect("the dot was dragged");
    // A thousandth of a source pixel: the pointer's position is an `f32` panel
    // coordinate on the way in and the source pixel is read back out of it, so
    // the round trip is exact only to that type's precision, not to the drag's.
    assert!(
        matches!(edit, crate::bench::PatchEdit::TranslateToPixel { viewpoint: sfmtool_core::bench::Viewpoint::Observation(0), pixel }
            if (pixel[0] - to[0]).abs() < 1e-3 && (pixel[1] - to[1]).abs() < 1e-3),
        "the drag named something else: {edit:?}",
    );

    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel the ray reaches");
    let labels = version_labels(&state, id);
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    let sentence = labels.last().expect("a version");
    assert!(
        sentence.starts_with(&format!("Moved {label} by ")),
        "the version's label does not name the slide: {sentence}",
    );

    let moved = on_bench(&state, id, &label);
    // The dot sits where the pointer left it.
    let site = moved.observations[0].site().expect("a sighting");
    assert!(
        (site[0] - to[0]).abs() < 1e-3 && (site[1] - to[1]).abs() < 1e-3,
        "the dot should sit at {to:?}, it sits at {site:?}",
    );
    // And the slide is in the patch's own plane: the normal is untouched and
    // the offset lies in it.
    let before_frame = track
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a frame");
    let after_frame = moved
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a frame");
    assert!((after_frame.normal() - before_frame.normal()).norm() < 1e-12);
    assert_eq!(after_frame.half_extent, before_frame.half_extent);
    let offset = after_frame.center - before_frame.center;
    assert!(
        offset.dot(&before_frame.normal()).abs() < 1e-12,
        "the patch left its own plane: {offset:?}",
    );
    // Every sighting moved with the patch and **kept its own offset** from
    // where the centre projects: the keypoints were carried along the plane,
    // not reset to the centre, because that offset is what the tiles are cut
    // on.
    for (offset, now) in offsets
        .iter()
        .zip(projection_offsets(&state.scene[0], &moved))
    {
        assert!(
            (now[0] - offset[0]).abs() < 1e-3 && (now[1] - offset[1]).abs() < 1e-3,
            "the slide scrambled a sighting's offset: {offset:?} became {now:?}",
        );
    }
    assert!(
        moved.observations.iter().all(|o| !o.pinned),
        "a translation is not a verdict"
    );
    // The outline moved in every image at once: no sighting is where it was.
    for (index, observation) in moved.observations.iter().enumerate() {
        assert_ne!(
            observation.site(),
            track.observations[index].site(),
            "image {} did not move with the patch",
            observation.image,
        );
    }
}

#[test]
fn dragging_an_edge_resizes_the_patch_so_it_reprojects_under_the_release_point() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let (frame, camera, pose) = outline_at(&state.scene[0], &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);
    let from = patch_pixel(&frame, &camera, &pose, 1.0, 0.0);
    let to = patch_pixel(&frame, &camera, &pose, 2.0, 0.0);
    let far_before = patch_pixel(&frame, &camera, &pose, -1.0, 0.0);
    let offsets = projection_offsets(&state.scene[0], &track);

    let edit = bench_drag(&state.scene[0], 0, &track, centre, from, to, false)
        .edit
        .expect("the edge was dragged");
    assert!(
        matches!(
            edit,
            crate::bench::PatchEdit::ResizeToPixel {
                viewpoint: sfmtool_core::bench::Viewpoint::Observation(0),
                edge: sfmtool_core::bench::Edge::PlusU,
                ..
            }
        ),
        "the drag named something else: {edit:?}",
    );

    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel the ray reaches");
    let labels = version_labels(&state, id);
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Resized {label} to ")),
        "the version's label does not name the resize: {:?}",
        labels.last(),
    );

    let after = on_bench(&state, id, &label);
    let resized = after
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a frame");
    assert_eq!(
        resized.half_extent[0], resized.half_extent[1],
        "a patch frame is square"
    );
    // A resize moves the centre, so every sighting is carried by the same
    // displacement and keeps its own offset, exactly as a slide's are; nothing
    // is pinned.
    for (offset, now) in offsets
        .iter()
        .zip(projection_offsets(&state.scene[0], &after))
    {
        assert!(
            (now[0] - offset[0]).abs() < 1e-3 && (now[1] - offset[1]).abs() < 1e-3,
            "the resize scrambled a sighting's offset: {offset:?} became {now:?}",
        );
    }
    assert!(after.observations.iter().all(|o| !o.pinned));
    // Against the outline as it is redrawn -- the patch re-anchored on the
    // sighting the edge was dragged in, which is what the person sees.
    let (redrawn, _, _) = outline_at(&state.scene[0], &after, 0);
    let landed = patch_pixel(&redrawn, &camera, &pose, 1.0, 0.0);
    assert!(
        (landed[0] - to[0]).abs() < 1e-3 && (landed[1] - to[1]).abs() < 1e-3,
        "the dragged edge should land on {to:?}, it landed on {landed:?}",
    );
    let far_after = patch_pixel(&redrawn, &camera, &pose, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-3 && (far_after[1] - far_before[1]).abs() < 1e-3,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );
}

#[test]
fn dragging_a_corner_turns_the_patch_and_pushes_one_version() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let (frame, camera, pose) = outline_at(&state.scene[0], &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);
    let from = patch_pixel(&frame, &camera, &pose, 1.0, 1.0);
    // A quarter turn in the patch's own plane: the `(1, 1)` corner's offset
    // rotated onto the `(-1, 1)` corner's.
    let to = patch_pixel(&frame, &camera, &pose, -1.0, 1.0);

    let edit = bench_drag(&state.scene[0], 0, &track, centre, from, to, false)
        .edit
        .expect("the corner was dragged");
    let crate::bench::PatchEdit::Spin { angle_rad } = edit else {
        panic!("the drag named something else: {edit:?}");
    };
    assert!(
        (angle_rad.to_degrees() - 90.0).abs() < 0.5,
        "a corner dragged onto its neighbour is a quarter turn, not {}",
        angle_rad.to_degrees(),
    );

    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a finite angle");
    let labels = version_labels(&state, id);
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Spun {label} by ")),
        "the version's label does not name the turn: {:?}",
        labels.last(),
    );
    let was = track
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a frame");
    let turned = on_bench(&state, id, &label);
    let turned = turned
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a frame");
    assert_eq!(turned.center, was.center, "a turn moves the patch nowhere");
    assert!((turned.u_axis.norm() - was.u_axis.norm()).abs() < 1e-12);
    assert!((turned.normal() - was.normal()).norm() < 1e-12);
}

#[test]
fn escape_cancels_a_drag_and_a_drag_that_ends_where_it_started_pushes_nothing() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let was = track.observations[0].site().expect("a sighting");
    let to = [was[0] + 1.5, was[1] + 2.0];

    let cancelled = bench_drag(&state.scene[0], 0, &track, was, was, to, true);
    assert!(
        cancelled.edit.is_none(),
        "escape left an edit behind: {:?}",
        cancelled.edit,
    );

    // And a gesture that ends where it began is a step that changes nothing,
    // which pushes no version -- the way a verdict an observation already holds
    // does.
    let still = bench_drag(&state.scene[0], 0, &track, was, was, was, false);
    let before = version_labels(&state, id).len();
    if let Some(edit) = still.edit {
        state
            .edit_bench_patch(id, &label, &edit)
            .expect("the pixel it already sits at");
    }
    assert_eq!(
        version_labels(&state, id).len(),
        before,
        "a drag that moved nothing pushed a version",
    );
}

/// **The press decides the handle, not the drag.**
///
/// egui calls a gesture a drag only after the pointer has left the press by
/// several pixels, while the view pans on whatever motion it is given, with no
/// threshold of its own. A layer that waited for `drag_started` therefore lost the
/// gesture twice over -- the photograph had already moved, so the handle was no
/// longer under the press position the hit test was given -- and what the
/// person got was a pan. This is that case, with a step deliberately below the
/// threshold in front of the real one.
#[test]
fn a_press_on_a_handle_takes_the_gesture_before_egui_would_call_it_a_drag() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let (frame, camera, pose) = outline_at(&state.scene[0], &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);
    let edge = patch_pixel(&frame, &camera, &pose, 1.0, 0.0);
    // Two panel pixels, then thirty: the first is under egui's drag threshold
    // and is exactly what used to be spent panning the photograph.
    let steps = [egui::vec2(2.0, 0.0), egui::vec2(30.0, 0.0)];

    let dragged = gesture(
        &state.scene[0],
        0,
        &track,
        centre,
        edge,
        &steps,
        false,
        true,
    );
    assert_eq!(
        dragged.panned,
        egui::Vec2::ZERO,
        "the photograph panned under a handle drag",
    );
    let edit = dragged.edit.expect("the edge was grabbed at the press");
    assert!(
        matches!(
            edit,
            crate::bench::PatchEdit::ResizeToPixel {
                viewpoint: sfmtool_core::bench::Viewpoint::Observation(0),
                edge: sfmtool_core::bench::Edge::PlusU,
                ..
            }
        ),
        "the gesture named something else: {edit:?}",
    );
    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel the ray reaches");
    let labels = version_labels(&state, id);
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Resized {label} to ")),
        "{:?}",
        labels.last(),
    );
}

/// With Track View's *Lock* cleared, the track stage's dot is that one
/// sighting's keypoint: the drag names [`crate::bench::PatchEdit::Sight`], the
/// release moves that keypoint and nothing else, and the patch, its size, its
/// turn and every other sighting stand where they stood.
#[test]
fn with_the_lock_off_the_dot_moves_one_sighting_and_leaves_the_patch_and_the_rest() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let was = track.observations[0].site().expect("a sighting");
    let to = [was[0] + 1.5, was[1] + 2.0];

    let dragged = unlocked_drag(&state.scene[0], 0, &track, was, was, to);
    assert_eq!(
        dragged.cursor,
        egui::CursorIcon::Move,
        "the dot is still a handle"
    );
    assert_eq!(dragged.panned, egui::Vec2::ZERO);
    let edit = dragged.edit.expect("the dot was dragged");
    assert!(
        matches!(edit, crate::bench::PatchEdit::Sight { observation: 0, pixel }
            if (pixel[0] - to[0]).abs() < 1e-3 && (pixel[1] - to[1]).abs() < 1e-3),
        "the drag named something else: {edit:?}",
    );

    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel on the photograph");
    let labels = version_labels(&state, id);
    assert_eq!(labels.len(), before + 1, "one gesture, one version");
    assert!(
        labels
            .last()
            .expect("a version")
            .starts_with(&format!("Moved observation 0 of {label} to (")),
        "the version's label does not name the one sighting: {:?}",
        labels.last(),
    );

    let moved = on_bench(&state, id, &label);
    let site = moved.observations[0].site().expect("a sighting");
    assert!(
        (site[0] - to[0]).abs() < 1e-3 && (site[1] - to[1]).abs() < 1e-3,
        "the dot should sit at {to:?}, it sits at {site:?}",
    );
    assert!(moved.observations[0].pinned, "a hand placement is a ruling");
    assert_eq!(
        moved.track().and_then(|p| p.placement.clone()),
        track.track().and_then(|p| p.placement.clone()),
        "the patch moved under an unlocked drag",
    );
    assert_eq!(
        moved.track().map(|p| p.position),
        track.track().map(|p| p.position)
    );
    for index in 1..track.observations.len() {
        assert_eq!(
            moved.observations[index], track.observations[index],
            "observation {index} moved under an unlocked drag of observation 0",
        );
    }
}

/// With the lock off, the outline's edges and corners are not handles: a
/// track-stage sighting has no size or turn of its own, and a resize or a spin
/// would move every sighting at once. A press on one pans the photograph, as a
/// press off the layer does, and edits nothing. The cluster stage's outline is
/// each sighting's own shape, so the lock leaves it a handle there.
#[test]
fn with_the_lock_off_the_outline_takes_no_drag_at_the_track_stage() {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let (frame, camera, pose) = outline_at(&state.scene[0], &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);
    let edge = patch_pixel(&frame, &camera, &pose, 1.0, 0.0);
    let corner = patch_pixel(&frame, &camera, &pose, 1.0, 1.0);
    let beyond = patch_pixel(&frame, &camera, &pose, 2.0, 0.0);
    for grabbed in [edge, corner] {
        let dragged = unlocked_drag(&state.scene[0], 0, &track, centre, grabbed, beyond);
        assert!(
            dragged.edit.is_none(),
            "an unlocked outline edited the track: {:?}",
            dragged.edit,
        );
        assert_ne!(
            dragged.panned,
            egui::Vec2::ZERO,
            "a press on an outline that is no handle is a pan"
        );
        assert_eq!(dragged.cursor, egui::CursorIcon::Default);
    }

    // And the same rule asked of the drag itself, for a drag already held.
    let drag = super::bench_track::Drag {
        image: 0,
        handle: super::bench_track::Handle::Edge {
            outline: sfmtool_core::bench::Viewpoint::Observation(0),
            edge: sfmtool_core::bench::Edge::PlusU,
        },
        from: edge,
        to: beyond,
        moved: true,
        cancelled: false,
    };
    let table = &state.scene[0].edited().base.image_table;
    assert!(super::bench_track::Layer::edit(table, &track, &drag, false).is_none());
    assert!(super::bench_track::Layer::edit(table, &track, &drag, true).is_some());

    // At the cluster stage every handle is one sighting's already, so the lock
    // changes nothing: the dot places that seed and a corner turns that shape,
    // whichever way the box is set.
    let seed = [120.0, 90.0];
    let cluster = state
        .start_bench_cluster(
            crate::scene::ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: seed,
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    let cluster = on_bench(&state, id, &cluster);
    let table = &state.scene[0].edited().base.image_table;
    for lock in [true, false] {
        let dot = super::bench_track::Drag {
            image: 0,
            handle: super::bench_track::Handle::Keypoint { observation: 0 },
            from: seed,
            to: [seed[0] + 2.0, seed[1]],
            moved: true,
            cancelled: false,
        };
        assert!(
            matches!(
                super::bench_track::Layer::edit(table, &cluster, &dot, lock),
                Some(crate::bench::PatchEdit::Sight { observation: 0, .. })
            ),
            "lock {lock}",
        );
        let corner = super::bench_track::Drag {
            handle: super::bench_track::Handle::Corner {
                outline: sfmtool_core::bench::Viewpoint::Observation(0),
                corner: 2,
            },
            from: [seed[0] + 4.0, seed[1] + 4.0],
            to: [seed[0] - 4.0, seed[1] + 4.0],
            ..dot
        };
        assert!(
            matches!(
                super::bench_track::Layer::edit(table, &cluster, &corner, lock),
                Some(crate::bench::PatchEdit::SpinShape { observation: 0, .. })
            ),
            "lock {lock}",
        );
    }
}

/// What is drawn while the dot is held is what the release leaves, in both
/// lock states. The one image's own view tells the two apart by the
/// projection-offset segment: locked, the patch slides and the sighting keeps
/// its offset from the patch's projection, which for this fixture is none; the
/// lock off, the patch stays and the segment runs the length of the drag.
#[test]
fn the_held_dot_previews_what_its_release_does_in_both_lock_states() {
    let (state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let was = track.observations[0].site().expect("a sighting");
    let to = [was[0] + 1.5, was[1] + 2.0];
    let panel_px = f64::from(panel_scale(1920, 1080));
    let drawn_px = (to[0] - was[0]).hypot(to[1] - was[1]) * panel_px;
    let longest = |dragged: &Dragged| {
        dragged
            .held_segments
            .iter()
            .map(|(points, _)| f64::from((points[1] - points[0]).length()))
            .fold(0.0f64, f64::max)
    };

    let locked = bench_drag(&state.scene[0], 0, &track, was, was, to, false);
    assert!(
        longest(&locked) < 0.05 * drawn_px,
        "a locked drag previews a sighting parted from the patch: {} of {drawn_px} px",
        longest(&locked),
    );
    let unlocked = unlocked_drag(&state.scene[0], 0, &track, was, was, to);
    assert!(
        (longest(&unlocked) - drawn_px).abs() < 0.05 * drawn_px,
        "an unlocked drag should preview a {drawn_px} px offset, it drew {}",
        longest(&unlocked),
    );

    // And each release is the track its preview drew: the offset in source px,
    // which is what the segment was drawing.
    for (dragged, offset) in [(&locked, 0.0), (&unlocked, drawn_px / panel_px)] {
        let edit = dragged.edit.expect("the dot was dragged");
        let (next, _) = crate::bench::geometry::apply(&track, state.scene[0].edited(), &edit)
            .expect("a pixel the step takes");
        let landed = projection_offsets(&state.scene[0], &next)[0];
        assert!(
            (landed[0].hypot(landed[1]) - offset).abs() < 0.05 * (drawn_px / panel_px),
            "{edit:?} released on an offset of {landed:?}, not {offset}",
        );
    }
}

/// A press that hits no handle is the pan it always was, and a press on a
/// handle that never moves is a click: neither edits the track.
#[test]
fn a_press_off_the_handles_still_pans_and_a_press_that_does_not_move_edits_nothing() {
    let (state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let (frame, camera, pose) = outline_at(&state.scene[0], &track, 0);
    let centre = patch_pixel(&frame, &camera, &pose, 0.0, 0.0);
    let steps = [egui::vec2(2.0, 0.0), egui::vec2(30.0, 0.0)];

    // Far outside the outline, which is a couple of source pixels across.
    let empty = [centre[0] + 40.0, centre[1] + 40.0];
    let dragged = gesture(
        &state.scene[0],
        0,
        &track,
        centre,
        empty,
        &steps,
        false,
        true,
    );
    assert!(dragged.edit.is_none(), "empty image edited the track");
    assert!(
        dragged.panned.x > 20.0 && dragged.panned.y == 0.0,
        "a press off the handles is the pan it always was: {:?}",
        dragged.panned,
    );

    // And a press on the edge that never moves leaves the track alone.
    let edge = patch_pixel(&frame, &camera, &pose, 1.0, 0.0);
    let still = gesture(&state.scene[0], 0, &track, centre, edge, &[], false, true);
    assert!(still.edit.is_none(), "a click edited the track");
    assert_eq!(still.panned, egui::Vec2::ZERO);
}

/// Every sighting of a point at infinity casts the same ray, so its widest pair
/// is zero degrees whatever the baseline. The three stored numbers are that
/// direction and not a place: subtracting a camera centre from them measures the
/// spread of rays to a point a unit from the world origin, which is a confident
/// wrong number and a different one from what Track View
/// reports for the same row.
#[test]
fn the_max_track_angle_of_a_bearing_is_zero_and_not_the_spread_about_the_origin() {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    let point = &mut recon.point_set.points[2];
    let finite = point.position;
    point.position = nalgebra::Point3::from(point.position.coords.normalize());
    point.w = 0.0;
    point.normal = nalgebra::Vector3::zeros();
    recon.rebuild_derived_fields();
    let bearing = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(recon));
    assert_eq!(super::compute_max_track_angle_deg(&bearing, 2), 0.0);

    // The same track as a place has a real spread, so the zero above is the
    // bearing's and not the fixture's.
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    recon.point_set.points[2].position = finite;
    recon.rebuild_derived_fields();
    let place = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(recon));
    assert!(
        super::compute_max_track_angle_deg(&place, 2) > 1.0,
        "the fixture's finite track should subtend a real angle"
    );
}

// ── The ghost's handles and the normal's ────────────────────────────────

/// `patch` turned so its normal leans `degrees` off the line of sight to
/// `eye`, about the world's vertical: a view that sees the square and its
/// normal both, neither edge-on nor end-on.
fn leaning(
    patch: &sfmtool_core::patch::cloud::OrientedPatch,
    eye: nalgebra::Point3<f64>,
    degrees: f64,
) -> sfmtool_core::patch::cloud::OrientedPatch {
    let turn =
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), degrees.to_radians());
    sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        patch.center,
        turn * (eye - patch.center),
        nalgebra::Vector3::z(),
        patch.half_extent,
    )
}

/// The first image the track has no observation in.
fn unseen_image(node: &SceneNode, track: &sfmtool_core::bench::EditableTrack) -> usize {
    (0..node.edited().image_count())
        .find(|i| track.observations.iter().all(|o| o.image as usize != *i))
        .expect("the fixture's track does not span every image")
}

/// The patch of a track-stage track.
fn placement(
    track: &sfmtool_core::bench::EditableTrack,
) -> sfmtool_core::patch::cloud::OrientedPatch {
    track
        .track()
        .and_then(|payload| payload.placement.clone())
        .expect("a track-stage track with a patch")
}

/// The bench state of [`bench_state`] with its patch tilted to lean 30 degrees
/// off the view of the first image it has no sighting in, and that image: a
/// ghost whose square and normal that camera both sees.
///
/// The tilt goes through the step a drag would push, so the bench holds it as
/// a version and a gesture after it is one more.
fn ghost_state() -> (crate::state::AppState, crate::scene::ReconId, String, usize) {
    let (mut state, id, label) = bench_state();
    let track = on_bench(&state, id, &label);
    let unseen = unseen_image(&state.scene[0], &track);
    let eye = state.scene[0].edited().base.image_table.images[unseen].camera_center();
    let want = leaning(&placement(&track), eye, 30.0).normal();
    state
        .edit_bench_patch(
            id,
            &label,
            &crate::bench::PatchEdit::Tilt {
                normal: [want.x, want.y, want.z],
            },
        )
        .expect("a finite normal");
    let track = on_bench(&state, id, &label);
    let (_, pose) =
        crate::bench::geometry::view_of(&state.scene[0].edited().base.image_table, unseen)
            .expect("the demo's images have cameras");
    let patch = placement(&track);
    let eye = pose.inverse_translation_origin();
    assert!(
        patch.is_front_facing(&pose)
            && !crate::bench::geometry::plane_is_edge_on(&patch, eye)
            && !crate::bench::geometry::normal_is_end_on(&patch, eye),
        "the fixture's ghost should see the square and its normal both",
    );
    (state, id, label, unseen)
}

/// Where a world point lands in one image, in its own px.
fn pixel_of(node: &SceneNode, image: usize, point: nalgebra::Point3<f64>) -> [f64; 2] {
    let (camera, pose) = crate::bench::geometry::view_of(&node.edited().base.image_table, image)
        .expect("the demo's images have cameras");
    crate::bench::geometry::project(&camera, &pose, point.coords, 1.0)
        .expect("in front of the camera")
}

/// The tip of the normal's arrow standing out of `square`.
fn tip_of(square: &sfmtool_core::patch::cloud::OrientedPatch) -> nalgebra::Point3<f64> {
    square.center
        + square.normal() * (crate::bench::geometry::NORMAL_LENGTH * square.half_extent[0])
}

/// Panel px per source px for [`layer_at`]: enough that the demo's small
/// patch spreads its handles well past one another's reach.
const LAYER_SCALE: f32 = 64.0;

/// The layer for `image` at [`LAYER_SCALE`], with the image's origin at the
/// panel's, so [`pos`] is where a source pixel lands.
fn layer_at(
    node: &SceneNode,
    image: usize,
    track: &sfmtool_core::bench::EditableTrack,
    lock: bool,
) -> Option<super::bench_track::Layer> {
    super::bench_track::Layer::build(
        &node.edited().base.image_table,
        image,
        track,
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1e6, 1e6)),
        LAYER_SCALE,
        lock,
    )
}

fn pos(pixel: [f64; 2]) -> egui::Pos2 {
    egui::pos2(pixel[0] as f32 * LAYER_SCALE, pixel[1] as f32 * LAYER_SCALE)
}

/// Locked, the ghost offers the patch-wide handles, each read against the
/// patch as it stands in this image: its centre, an edge, a corner, and the
/// normal's segment and arrowhead. Cleared, it offers none of them.
#[test]
fn the_locked_ghost_offers_the_patch_handles_and_the_unlocked_one_none() {
    use super::bench_track::Handle;
    use sfmtool_core::bench::{Edge, Viewpoint};

    let (state, id, label, unseen) = ghost_state();
    let node = &state.scene[0];
    let track = on_bench(&state, id, &label);
    let patch = placement(&track);
    let (camera, pose) = crate::bench::geometry::view_of(&node.edited().base.image_table, unseen)
        .expect("the demo's images have cameras");
    let at = |s, t| pos(patch_pixel(&patch, &camera, &pose, s, t));
    let here = Viewpoint::Image(unseen as u32);

    let locked = layer_at(node, unseen, &track, true).expect("a ghost");
    assert_eq!(locked.hit(at(0.0, 0.0)), Some(Handle::Centre));
    assert_eq!(
        locked.hit(at(1.0, 1.0)),
        Some(Handle::Corner {
            outline: here,
            corner: 2
        })
    );
    assert!(
        matches!(
            locked.hit(at(1.0, 0.0)),
            Some(Handle::Edge { outline, edge: Edge::PlusU }) if outline == here
        ),
        "the +u edge's midpoint is not its handle: {:?}",
        locked.hit(at(1.0, 0.0)),
    );
    let tip = pos(pixel_of(node, unseen, tip_of(&patch)));
    assert!(
        matches!(locked.hit(tip), Some(Handle::Arrowhead { outline, .. }) if outline == here),
        "the arrowhead is not a handle: {:?}",
        locked.hit(tip),
    );

    let unlocked = layer_at(node, unseen, &track, false).expect("a ghost, drawn");
    for place in [at(0.0, 0.0), at(1.0, 1.0), at(1.0, 0.0), tip] {
        assert_eq!(
            unlocked.hit(place),
            None,
            "the unlocked ghost offered a handle at {place:?}"
        );
    }

    // The cursors are the member layer's.
    let centre = patch_pixel(&patch, &camera, &pose, 0.0, 0.0);
    let dragged = gesture(node, unseen, &track, centre, centre, &[], false, true);
    assert_eq!(dragged.cursor, egui::CursorIcon::Move);
}

/// The ghost draws a hollow centre mark when it offers handles, so the dot has
/// something to grab, and keeps the ghost opacity whether or not it does.
#[test]
fn the_locked_ghost_draws_a_centre_mark_at_the_ghost_opacity() {
    let (state, id, label, unseen) = ghost_state();
    let node = &state.scene[0];
    let track = on_bench(&state, id, &label);
    let ghost = super::bench_track::ghost_color();
    let circles = |lock| {
        bench_frame(
            node,
            unseen,
            BenchMenu {
                busy: None,
                active_track: Some(&track),
                lock,
                create_track: None,
            },
        )
        .iter()
        .filter(|clipped| {
            matches!(&clipped.shape, egui::Shape::Circle(circle) if circle.stroke.color == ghost)
        })
        .count()
    };
    assert_eq!(circles(true), 1, "the locked ghost has one centre mark");
    assert_eq!(circles(false), 0, "the unlocked ghost has no centre mark");
    for lock in [true, false] {
        let menu = BenchMenu {
            busy: None,
            active_track: Some(&track),
            lock,
            create_track: None,
        };
        assert!(
            ghost_shapes(node, unseen, menu)
                .iter()
                .any(|path| path.len() > 4),
            "lock {lock}: the ghost outline is not drawn at the ghost colour",
        );
    }
}

/// A drag of the ghost's centre slides the patch until its own centre sits
/// under the release point in this image, as one version.
#[test]
fn dragging_the_ghost_centre_slides_the_patch_under_the_pointer() {
    let (mut state, id, label, unseen) = ghost_state();
    let track = on_bench(&state, id, &label);
    let patch = placement(&track);
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.scene[0].edited().base.image_table, unseen)
            .expect("the demo's images have cameras");
    let was = patch_pixel(&patch, &camera, &pose, 0.0, 0.0);
    let to = [was[0] + 1.5, was[1] + 2.0];
    let edit = bench_drag(&state.scene[0], unseen, &track, was, was, to, false)
        .edit
        .expect("the ghost's centre was dragged");
    assert!(
        matches!(edit, crate::bench::PatchEdit::TranslateToPixel {
            viewpoint: sfmtool_core::bench::Viewpoint::Image(image), pixel }
            if image as usize == unseen
                && (pixel[0] - to[0]).abs() < 1e-3 && (pixel[1] - to[1]).abs() < 1e-3),
        "the drag named something else: {edit:?}",
    );
    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel the ray reaches");
    assert_eq!(
        version_labels(&state, id).len(),
        before + 1,
        "one gesture, one version"
    );
    let moved = placement(&on_bench(&state, id, &label));
    let landed = patch_pixel(&moved, &camera, &pose, 0.0, 0.0);
    assert!(
        (landed[0] - to[0]).abs() < 1e-3 && (landed[1] - to[1]).abs() < 1e-3,
        "the patch's centre should land on {to:?}, it landed on {landed:?}",
    );
    assert!(
        (moved.normal() - patch.normal()).norm() < 1e-12,
        "a slide keeps the normal"
    );
}

/// A drag of the ghost's edge resizes the patch with the far edge held, both
/// read in this image, as one version; a drag of its corner spins it about its
/// normal.
#[test]
fn dragging_the_ghost_edge_and_corner_resizes_and_spins_the_patch() {
    let (mut state, id, label, unseen) = ghost_state();
    let track = on_bench(&state, id, &label);
    let patch = placement(&track);
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.scene[0].edited().base.image_table, unseen)
            .expect("the demo's images have cameras");
    let centre = patch_pixel(&patch, &camera, &pose, 0.0, 0.0);
    let from = patch_pixel(&patch, &camera, &pose, 1.0, 0.0);
    let to = patch_pixel(&patch, &camera, &pose, 1.6, 0.0);
    let far = patch_pixel(&patch, &camera, &pose, -1.0, 0.0);

    let dragged = bench_drag(&state.scene[0], unseen, &track, centre, from, to, false);
    let edit = dragged.edit.expect("the ghost's edge was dragged");
    assert!(
        matches!(edit, crate::bench::PatchEdit::ResizeToPixel {
            viewpoint: sfmtool_core::bench::Viewpoint::Image(image),
            edge: sfmtool_core::bench::Edge::PlusU, .. } if image as usize == unseen),
        "the drag named something else: {edit:?}",
    );
    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a pixel the ray reaches");
    assert_eq!(
        version_labels(&state, id).len(),
        before + 1,
        "one gesture, one version"
    );
    let grown = placement(&on_bench(&state, id, &label));
    let landed = patch_pixel(&grown, &camera, &pose, 1.0, 0.0);
    let held = patch_pixel(&grown, &camera, &pose, -1.0, 0.0);
    assert!(
        (landed[0] - to[0]).abs() < 1e-2 && (landed[1] - to[1]).abs() < 1e-2,
        "the dragged edge should land on {to:?}, it landed on {landed:?}",
    );
    assert!(
        (held[0] - far[0]).abs() < 1e-2 && (held[1] - far[1]).abs() < 1e-2,
        "the far edge moved from {far:?} to {held:?}",
    );
    // What was drawn while the edge was held is the square the release left.
    let panel = gesture_panel(centre);
    let ghost = egui::epaint::ColorMode::Solid(super::bench_track::ghost_color());
    let outline = dragged
        .held_paths
        .iter()
        .filter(|(_, color)| *color == ghost)
        .map(|(points, _)| points)
        .max_by_key(|points| points.len())
        .expect("the preview drew the ghost");
    for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let expected = panel(patch_pixel(&grown, &camera, &pose, s, t));
        let nearest = outline
            .iter()
            .map(|point| (*point - expected).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 0.05,
            "the preview misses the released corner ({s}, {t}) by {nearest}"
        );
    }

    let track = on_bench(&state, id, &label);
    let patch = placement(&track);
    let centre = patch_pixel(&patch, &camera, &pose, 0.0, 0.0);
    let from = patch_pixel(&patch, &camera, &pose, 1.0, 1.0);
    let to = patch_pixel(&patch, &camera, &pose, -1.0, 1.0);
    let edit = bench_drag(&state.scene[0], unseen, &track, centre, from, to, false)
        .edit
        .expect("the ghost's corner was dragged");
    let crate::bench::PatchEdit::Spin { angle_rad } = edit else {
        panic!("the drag named something else: {edit:?}");
    };
    assert!(
        (angle_rad.to_degrees() - 90.0).abs() < 0.5,
        "a corner dragged onto its neighbour is a quarter turn, not {}",
        angle_rad.to_degrees(),
    );
    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a finite angle");
    assert_eq!(
        version_labels(&state, id).len(),
        before + 1,
        "one gesture, one version"
    );
}

/// The normal is drawn out of the outline's centre in a member image (the
/// sighting's, the outline being anchored there) and out of the patch's own
/// centre in the ghost, as a segment and a two-barbed head.
#[test]
fn the_normal_is_drawn_in_member_images_and_in_the_locked_ghost() {
    let (state, id, label, unseen) = ghost_state();
    let node = &state.scene[0];
    let track = on_bench(&state, id, &label);
    let menu = |lock| BenchMenu {
        busy: None,
        active_track: Some(&track),
        lock,
        create_track: None,
    };
    // The segment is the one two-point path the layer draws.
    let segments = |image, lock, color: egui::Color32| -> Vec<Vec<egui::Pos2>> {
        let mut paths = Vec::new();
        for clipped in &bench_frame(node, image, menu(lock)) {
            collect_paths_in(
                &clipped.shape,
                &egui::epaint::ColorMode::Solid(color),
                &mut paths,
            );
        }
        paths.into_iter().filter(|path| path.len() <= 3).collect()
    };
    let photograph = pixels(640, 480);

    let member = track.observations[0].image as usize;
    let (square, _, _) = outline_at(node, &track, 0);
    let color = crate::bench::verdict_color(track.observations[0].verdict);
    for lock in [true, false] {
        let arrow = segments(member, lock, color);
        let segment = arrow
            .iter()
            .find(|path| path.len() == 2)
            .unwrap_or_else(|| panic!("lock {lock}: no normal in the member image: {arrow:?}"));
        let tail = to_panel(&photograph, pixel_of(node, member, square.center));
        let tip = to_panel(&photograph, pixel_of(node, member, tip_of(&square)));
        assert!(
            (segment[0] - tail).length() < 0.01,
            "the normal leaves {:?}, not {tail:?}",
            segment[0]
        );
        assert!(
            (segment[1] - tip).length() < 0.01,
            "the normal ends at {:?}, not {tip:?}",
            segment[1]
        );
        assert!(arrow.iter().any(|path| path.len() == 3), "no arrowhead");
    }

    let ghost = super::bench_track::ghost_color();
    let arrow = segments(unseen, true, ghost);
    let patch = placement(&track);
    let segment = arrow
        .iter()
        .find(|path| path.len() == 2)
        .expect("no normal in the locked ghost");
    let tail = to_panel(&photograph, pixel_of(node, unseen, patch.center));
    assert!(
        (segment[0] - tail).length() < 0.01,
        "the ghost's normal leaves {:?}",
        segment[0]
    );
    assert!(
        segments(unseen, false, ghost).is_empty(),
        "the unlocked ghost drew a normal",
    );
}

/// Seen down its own length the normal is not drawn and not offered, the bar
/// the 3D viewport refuses its segment at; a cluster has no normal at all; and
/// with the lock cleared a member image draws it and offers neither handle.
#[test]
fn the_normal_is_hidden_end_on_absent_at_the_cluster_stage_and_not_offered_unlocked() {
    use super::bench_track::Handle;

    let (node, track) = bench_track_fixture();
    let member = track.observations[0].image as usize;
    let eye = node.edited().base.image_table.images[member].camera_center();
    let end_on = with_placement(&track, |placement| {
        let patch = placement.as_mut().expect("the fixture has a placement");
        *patch = leaning(patch, eye, 0.0);
    });
    let layer = layer_at(&node, member, &end_on, true).expect("a member image");
    let (square, _, _) = outline_at(&node, &end_on, 0);
    let tip = pos(pixel_of(&node, member, tip_of(&square)));
    assert!(
        !matches!(
            layer.hit(tip),
            Some(Handle::Arrowhead { .. } | Handle::Normal { .. })
        ),
        "an end-on normal took a press",
    );
    let lines = |track: &sfmtool_core::bench::EditableTrack| {
        let paths = bench_shapes(
            &node,
            member,
            BenchMenu {
                busy: None,
                active_track: Some(track),
                lock: true,
                create_track: None,
            },
        );
        paths.into_iter().filter(|path| path.len() == 2).count()
    };
    assert_eq!(lines(&end_on), 0, "an end-on normal was drawn");

    // Leaning well off the view, the same image draws it and offers it, and
    // with the lock cleared it still draws it and offers nothing.
    let oblique = with_placement(&track, |placement| {
        let patch = placement.as_mut().expect("the fixture has a placement");
        *patch = leaning(patch, eye, 35.0);
    });
    assert_eq!(lines(&oblique), 1, "an oblique normal was not drawn");
    let (square, _, _) = outline_at(&node, &oblique, 0);
    let tip_px = pixel_of(&node, member, tip_of(&square));
    // Near the tip, where the segment has left the square: inside it an edge
    // it crosses is tried first and takes the press.
    let middle_px = {
        let a = pixel_of(&node, member, square.center);
        [
            a[0] * 0.08 + tip_px[0] * 0.92,
            a[1] * 0.08 + tip_px[1] * 0.92,
        ]
    };
    let (tip, middle) = (pos(tip_px), pos(middle_px));
    let locked = layer_at(&node, member, &oblique, true).expect("a member image");
    assert!(
        matches!(locked.hit(tip), Some(Handle::Arrowhead { .. })),
        "{:?}",
        locked.hit(tip)
    );
    assert!(
        matches!(locked.hit(middle), Some(Handle::Normal { .. })),
        "{:?}",
        locked.hit(middle)
    );
    let unlocked = layer_at(&node, member, &oblique, false).expect("a member image");
    for place in [tip, middle] {
        assert!(
            !matches!(
                unlocked.hit(place),
                Some(Handle::Arrowhead { .. } | Handle::Normal { .. })
            ),
            "an unlocked normal took a press at {place:?}",
        );
    }
    let drag = super::bench_track::Drag {
        image: member,
        handle: Handle::Normal {
            outline: sfmtool_core::bench::Viewpoint::Observation(0),
        },
        from: middle_px,
        to: tip_px,
        moved: true,
        cancelled: false,
    };
    let table = &node.edited().base.image_table;
    assert!(super::bench_track::Layer::edit(table, &oblique, &drag, false).is_none());
    assert!(super::bench_track::Layer::edit(table, &oblique, &drag, true).is_some());

    // The cluster stage has no shared geometry, so no normal.
    use sfmtool_core::bench::{create_cluster, Bench, ClusterSeed};
    let (bench, report) = create_cluster(
        &Bench::new(),
        &ClusterSeed::from_pixel(0, "image_0", [320.0, 240.0], 24.0),
    )
    .expect("a usable seed");
    let cluster = (**bench.track(&report.label).expect("just put on")).clone();
    assert_eq!(lines(&cluster), 0, "a cluster drew a normal");
}

/// Dragging the arrowhead tilts the patch, and the new normal leans toward
/// the pointer: its tip, projected, is nearer the release point than the old
/// one was. What was drawn while it was held is the arrow the release left.
/// Dragging the segment toward the tip pushes the patch out along its normal.
#[test]
fn dragging_the_arrowhead_tilts_toward_the_pointer_and_the_segment_moves_along_the_normal() {
    let (mut state, id, label, _) = ghost_state();
    let track = on_bench(&state, id, &label);
    let member = track.observations[0].image as usize;
    let node = &state.scene[0];
    let (square, _, pose) = outline_at(node, &track, 0);
    let eye = pose.inverse_translation_origin();
    assert!(
        !crate::bench::geometry::normal_is_end_on(&square, eye),
        "the fixture's normal should be seen from the side in the member image",
    );
    let centre = pixel_of(node, member, square.center);
    let tip = pixel_of(node, member, tip_of(&square));
    // Across the arrow on screen, a sixth of its length.
    let along = [tip[0] - centre[0], tip[1] - centre[1]];
    let to = [tip[0] - along[1] / 6.0, tip[1] + along[0] / 6.0];

    let dragged = bench_drag(node, member, &track, centre, tip, to, false);
    let edit = dragged.edit.expect("the arrowhead was dragged");
    let crate::bench::PatchEdit::Tilt { normal } = edit else {
        panic!("the drag named something else: {edit:?}");
    };
    let turned = nalgebra::Vector3::from(normal).normalize();
    assert!(
        turned.dot(&square.normal()) < 1.0 - 1e-9,
        "the normal did not turn"
    );
    let new_tip = {
        let length = crate::bench::geometry::NORMAL_LENGTH * square.half_extent[0];
        pixel_of(node, member, square.center + turned * length)
    };
    let distance = |a: [f64; 2]| (a[0] - to[0]).hypot(a[1] - to[1]);
    assert!(
        distance(new_tip) < distance(tip),
        "the tip moved away from the pointer: {tip:?} -> {new_tip:?}, pointer at {to:?}",
    );

    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a finite normal");
    assert_eq!(
        version_labels(&state, id).len(),
        before + 1,
        "one gesture, one version"
    );
    let tilted = on_bench(&state, id, &label);
    let node = &state.scene[0];
    // The preview's arrow is the released track's arrow.
    let (released, _, _) = outline_at(node, &tilted, 0);
    let panel = gesture_panel(centre);
    let expected = [
        panel(pixel_of(node, member, released.center)),
        panel(pixel_of(node, member, tip_of(&released))),
    ];
    let held = dragged
        .held_paths
        .iter()
        .find(|(points, _)| points.len() == 2)
        .map(|(points, _)| points.clone())
        .expect("the preview drew the normal");
    for (drawn, expected) in held.iter().zip(expected) {
        assert!(
            (*drawn - expected).length() < 0.05,
            "the preview drew the arrow at {drawn:?}, the release put it at {expected:?}",
        );
    }

    // The segment, grabbed part way along and pulled toward the tip.
    let (square, _, _) = outline_at(node, &tilted, 0);
    let centre = pixel_of(node, member, square.center);
    let tip = pixel_of(node, member, tip_of(&square));
    let from = [
        centre[0] * 0.5 + tip[0] * 0.5,
        centre[1] * 0.5 + tip[1] * 0.5,
    ];
    let to = [
        centre[0] * 0.3 + tip[0] * 0.7,
        centre[1] * 0.3 + tip[1] * 0.7,
    ];
    let edit = bench_drag(node, member, &tilted, centre, from, to, false)
        .edit
        .expect("the segment was dragged");
    let crate::bench::PatchEdit::Translate { by } = edit else {
        panic!("the drag named something else: {edit:?}");
    };
    assert!(by[0] == 0.0 && by[1] == 0.0 && by[2] > 0.0, "{by:?}");
    let before = version_labels(&state, id).len();
    state
        .edit_bench_patch(id, &label, &edit)
        .expect("a finite displacement");
    assert_eq!(
        version_labels(&state, id).len(),
        before + 1,
        "one gesture, one version"
    );
}
