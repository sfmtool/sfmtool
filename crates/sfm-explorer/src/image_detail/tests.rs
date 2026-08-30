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
    let feature_display = FeatureDisplaySettings::default();
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, panel)),
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        detail.show(
            ui,
            &node.recon,
            node.id,
            Some(image_index),
            None,
            None,
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
