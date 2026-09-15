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
    reveal_frame(detail, ctx, node, image_index, image, panel, None);
}

/// The same frame, carrying a reveal request for a pixel of `image_index`:
/// what the dock hands the panel on the frame after a row click elsewhere named
/// a feature.
fn reveal_frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    image_index: usize,
    image: &ImageU8,
    panel: egui::Vec2,
    reveal: Option<[f32; 2]>,
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
            reveal,
            None,
            None,
            super::BenchMenu::default(),
            &mut None,
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
// A row click in the Point Track Detail or Track Edit panel selects an image
// *and* names a feature in it. Zoomed in, that feature can be nowhere on
// screen, so the panel is asked to bring it into view: by panning, never by
// zooming, and only when it has to.

/// The source image these tests reveal pixels of: big enough that a zoom of 4
/// shows only a part of it in [`PANEL`].
const SOURCE: egui::Vec2 = egui::Vec2::new(400.0, 300.0);

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
    reveal_frame(&mut detail, &ctx, &node, 0, &image, PANEL, Some(feature));

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
    reveal_frame(&mut detail, &ctx, &node, 0, &image, PANEL, Some(feature));

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
    reveal_frame(&mut detail, &ctx, &node, 0, &image, PANEL, Some([0.0, 0.0]));

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
    reveal_frame(
        &mut detail,
        &ctx,
        &node,
        0,
        &image,
        panel,
        Some([SOURCE.x, SOURCE.y]),
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
/// the version has no point at, and the Point Track panel -- which reads that
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
            &mut None,
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
        }),
        Err(BUSY.to_string()),
    );
}

#[test]
fn adding_to_the_bench_track_is_greyed_until_a_track_is_on_the_bench() {
    let track = a_track();
    let why = add_bench_observation_entry(BenchMenu::default())
        .expect_err("nothing is on the bench to add to");
    assert!(why.contains("No track is on the bench"), "{why}");
    assert!(
        why.contains(START_CLUSTER_LABEL),
        "the refusal does not say how to start one: {why}",
    );

    // The image the menu is open over is nowhere in the rule: a second
    // sighting in an image the track already holds joins as a candidate, and
    // it is the verdict that a track cannot hold twice.
    assert_eq!(
        add_bench_observation_entry(BenchMenu {
            busy: None,
            active_track: Some(&track),
        }),
        Ok(()),
    );
    assert_eq!(
        add_bench_observation_entry(BenchMenu {
            busy: Some(BUSY),
            active_track: Some(&track),
        }),
        Err(BUSY.to_string()),
    );
}

/// The two bench entries are in the menu, and are in it on a `sift_files`
/// node, where neither point edit is defined: a bench track is seeds in one
/// image's pixels until it is committed, so what backs the node's own
/// observations does not decide it.
#[test]
fn the_context_menu_offers_the_two_bench_entries_beside_the_point_edits() {
    let track = a_track();
    let texts = context_menu_texts(BenchMenu {
        busy: None,
        active_track: Some(&track),
    });
    for label in [START_CLUSTER_LABEL, ADD_BENCH_OBSERVATION_LABEL] {
        assert!(
            texts.iter().any(|t| t == label),
            "{label} is not in the menu: {texts:?}",
        );
    }
    assert!(
        texts
            .iter()
            .any(|t| t == super::overlay::NOT_EMBEDDED_PATCHES),
        "the fixture is meant to be a node the two point edits are absent on: {texts:?}",
    );

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
                &mut None,
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

// ── The bench layer ─────────────────────────────────────────────────────

/// A node of [`projected_embedded_demo`] and the bench track its point 2 makes:
/// three observations, in images 0, 1 and 2, each at that point's exact
/// projection, with the stored patch as the surfel's frame.
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
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let display = FeatureDisplaySettings::default();
    let image = pixels(640, 480);
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, PANEL)),
        ..Default::default()
    };
    let mut found = Vec::new();
    // Two frames: the first loads the image and prepares the overlay, and the
    // layer draws over what is on screen.
    for _ in 0..2 {
        found.clear();
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
                &mut None,
                &[],
                &crate::platform::ScrollInput::default(),
                None,
                Some(&image),
                &display,
                &mut intrinsics_display,
            );
        });
        output.textures_delta.clear();
        for clipped in &output.shapes {
            collect_bench_paths(&clipped.shape, &mut found);
        }
    }
    found
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

fn is_bench_color(color: &egui::epaint::ColorMode) -> bool {
    matches!(
        color,
        egui::epaint::ColorMode::Solid(solid)
            if [
                super::bench_track::IN_COLOR,
                super::bench_track::CANDIDATE_COLOR,
                super::bench_track::OUT_COLOR,
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

/// The surfel's outline is drawn where the frame's corners really project, and
/// it is a sampled curve rather than the four corners joined up.
#[test]
fn the_bench_layer_outlines_the_surfel_where_its_corners_project() {
    use sfmtool_core::geometry::RigidTransform;

    let (node, track) = bench_track_fixture();
    let paths = bench_shapes(
        &node,
        0,
        BenchMenu {
            busy: None,
            active_track: Some(&track),
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
        .and_then(|payload| payload.frame.as_ref())
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

/// The layer is the *active* track's, and only in the images that track
/// observes: an empty bench and a photograph outside the track both draw
/// nothing.
#[test]
fn the_bench_layer_draws_nothing_without_a_track_or_outside_it() {
    let (node, track) = bench_track_fixture();
    assert!(
        bench_shapes(&node, 0, BenchMenu::default()).is_empty(),
        "the layer drew with nothing on the bench",
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
            },
        )
        .is_empty(),
        "the layer drew in an image the track does not observe",
    );
}
