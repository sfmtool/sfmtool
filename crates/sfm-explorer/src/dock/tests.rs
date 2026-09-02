// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the default layout and the tab bodies' own.

use super::*;

/// The Action Log opens docked beside the image strip, and *behind* it: the
/// viewer should come up on the strip, with the record one click away rather
/// than in front of what the user came to look at.
#[test]
fn the_action_log_shares_the_bottom_node_with_the_image_browser() {
    let dock_state = crate::default_dock_state();
    let leaf = dock_state
        .main_surface()
        .iter()
        .filter_map(|node| node.get_leaf())
        .find(|leaf| leaf.tabs.contains(&Tab::ActionLog))
        .expect("no leaf holds the Action Log");
    assert_eq!(leaf.tabs, vec![Tab::ImageBrowser, Tab::ActionLog]);
    assert_eq!(
        leaf.tabs[leaf.active.0],
        Tab::ImageBrowser,
        "the bottom node does not open on the image strip"
    );
}

/// A camera to hand the toolbar, so the intrinsics half of the row is drawn too.
fn demo_node() -> SceneNode {
    SceneNode::from_path(
        std::path::Path::new("/runs/demo.sfmr"),
        sfmtool_core::SfmrReconstruction::demo(16),
    )
}

/// The width the Image Detail panel is left believing it has, after its overlay
/// toolbar has been drawn into a dock cell `cell_width` wide.
fn available_after_toolbar(cell_width: f32, overlay_mode: OverlayMode) -> egui::Rect {
    let node = demo_node();
    let camera_ref = CameraRef::new(node.id, 0);
    let camera = &node.recon.cameras[0];
    let mut settings = FeatureDisplaySettings {
        overlay_mode,
        ..Default::default()
    };
    let mut intrinsics = IntrinsicsDisplaySettings::default();
    let mut detail = ImageDetail::new();

    let ctx = egui::Context::default();
    let input = egui::RawInput {
        // Room to the right of the cell for the toolbar to spill into, the way
        // a neighbouring dock cell would offer it.
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(2000.0, 700.0),
        )),
        ..Default::default()
    };
    let mut available = egui::Rect::NOTHING;
    crate::test_support::run_frame_headless(&ctx, input, |ui| {
        let cell = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(cell_width, 700.0));
        let mut cell_ui = ui.new_child(egui::UiBuilder::new().max_rect(cell));
        show_overlay_toolbar(
            &mut cell_ui,
            &mut settings,
            &mut intrinsics,
            &mut detail,
            Some((camera_ref, camera)),
        );
        available = cell_ui.available_rect_before_wrap();
    });
    available
}

/// The bug this exists for. The toolbar is one unwrapped row of controls, so in
/// a narrow dock cell it does not fit — and egui grows a `Ui`'s `max_rect` to
/// include any widget that overflowed it. Uncontained, a 400 px cell left the
/// panel below believing it had 726 px, reaching well into the dock cell next
/// door. `ImageDetail::show` lays the image out in that rect *and* asks whether
/// the pointer is inside it to decide whether a trackpad gesture is addressed to
/// it, so scrolling the Intrinsics panel beside it panned the image — for
/// exactly as far into that panel as the overhang reached, which is why widening
/// the Image Detail panel made the symptom go away.
#[test]
fn the_overlay_toolbar_never_widens_the_panel_it_is_drawn_in() {
    for mode in OverlayMode::ALL {
        for cell_width in [300.0, 400.0, 700.0, 1400.0] {
            let available = available_after_toolbar(cell_width, mode);
            assert!(
                available.right() <= cell_width + 0.5,
                "{mode:?} in a {cell_width}px cell left {}px of overhang",
                available.right() - cell_width,
            );
        }
    }
}

/// And it still takes the vertical space it needs, in the cell it was given.
#[test]
fn the_overlay_toolbar_leaves_the_rest_of_the_cell_below_it() {
    let available = available_after_toolbar(400.0, OverlayMode::Features);
    assert!(
        available.top() > 0.0,
        "the toolbar consumed no height: {available:?}",
    );
    assert!(
        available.bottom() >= 690.0 && available.left() <= 0.5,
        "the rest of the cell is not left to the image: {available:?}",
    );
}
