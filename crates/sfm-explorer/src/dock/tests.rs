// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the tab bodies' own drawing.

use super::*;

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
    let camera = &node.recon().image_table.cameras[0];
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
/// it, so scrolling the Camera Intrinsics panel beside it panned the image — for
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

// ── Track View's Surface Tiling Mode, as the dock applies it ────────────

/// Apply one Edited-mode Track View response through the dock, as a frame
/// does after drawing the panel.
fn apply_track_edit(
    state: &mut AppState,
    track_view: &mut TrackView,
    id: ReconId,
    response: TrackBodyResponse,
) {
    with_tabs(state, track_view, |tabs, ui| {
        tabs.apply_track_edit_response(ui, id, response)
    });
}

/// Run `apply` once over a dock built on `state` and `track_view`, inside a
/// headless frame, as a frame applies what its panels asked for.
fn with_tabs(
    state: &mut AppState,
    track_view: &mut TrackView,
    apply: impl FnOnce(&mut TabContext<'_>, &egui::Ui),
) {
    let mut viewer_3d = Viewer3D::new();
    let mut scene_graph = SceneGraphPanel::new();
    let mut image_browser = ImageBrowser::new();
    let mut image_detail = ImageDetail::new();
    let mut intrinsics_detail = IntrinsicsDetail::new();
    let scroll_input = platform::ScrollInput::default();
    let ctx = egui::Context::default();
    let mut apply = Some(apply);
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        let Some(apply) = apply.take() else {
            return;
        };
        let mut tabs = TabContext {
            state: &mut *state,
            viewer_3d: &mut viewer_3d,
            scene_graph: &mut scene_graph,
            image_browser: &mut image_browser,
            image_detail: &mut image_detail,
            track_view: &mut *track_view,
            intrinsics_detail: &mut intrinsics_detail,
            frame: Progress::none(),
            scene_texture_id: None,
            hover_depth: None,
            hover_pick: None,
            gesture_events: &[],
            scroll_input: &scroll_input,
            diagnostics: None,
            handler_ok: true,
        };
        apply(&mut tabs, ui);
    });
}

/// The toolbar's *Duplicate* in Surface Tiling Mode is the mode's duplicate:
/// the copy's verdicts are handed back to the thresholds, and once its
/// evaluation has settled the rows they turned `out` are dropped, as with
/// Ctrl+D.
#[test]
fn the_duplicate_button_in_surface_tiling_mode_unpins_the_copy_and_drops_its_out_rows() {
    use crate::bench::tests::{put_with_rows_the_bars_refuse, Refused};
    let (mut state, id) = crate::bench::tests::state();
    let (label, refused) = put_with_rows_the_bars_refuse(&mut state, id, Refused::Some);
    let mut track_view = TrackView::new();
    track_view.set_surface_tiling(true);

    let response = TrackBodyResponse {
        mode: Some(crate::track_view::BodyMode::Edited),
        duplicate: true,
        ..TrackBodyResponse::default()
    };
    apply_track_edit(&mut state, &mut track_view, id, response);
    // The frames after it: the copy's evaluation settles, and the drop runs.
    state.settle_bench_evaluation();
    state.drive_surface_tiling();

    let copy = state.focused_item_label(id).expect("the copy").to_string();
    assert_ne!(copy, label);
    assert!(state.surface_tiling.is_none(), "the drop is still waiting");
    let track = state.bench_track(id, &copy).expect("on the bench");
    assert!(track.observations.iter().all(|o| !o.pinned));
    assert_eq!(track.observations.len(), 3 - refused.len());
    assert!(
        track
            .observations
            .iter()
            .all(|o| !refused.contains(&o.image)),
        "a row the thresholds turned out was kept"
    );
    assert!(
        state
            .bench_track(id, &label)
            .expect("the original")
            .observations
            .iter()
            .all(|o| o.pinned),
        "the original was unpinned"
    );
}

/// Clearing the *Surface Tiling Mode* box ends the sequence under way, with
/// the stop row saying why.
#[test]
fn clearing_surface_tiling_mode_stops_the_sequence() {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    state
        .edit_bench_patch_by_gesture(
            id,
            &label,
            &crate::bench::PatchEdit::Translate {
                by: [0.01, 0.0, 0.0],
            },
            Some(crate::bench::SplitSettings::default()),
        )
        .expect("a move in the mode");
    assert!(
        state.surface_tiling.is_some(),
        "the move started no sequence"
    );
    let mut track_view = TrackView::new();

    let response = TrackBodyResponse {
        mode: Some(crate::track_view::BodyMode::Edited),
        surface_tiling_off: true,
        ..TrackBodyResponse::default()
    };
    apply_track_edit(&mut state, &mut track_view, id, response);

    assert!(state.surface_tiling.is_none());
    let last = state.action_log.entries().last().expect("a row");
    assert!(last.failed);
    assert_eq!(
        last.text,
        format!(
            "Surface tiling of {label} stopped at Fit (step 1 of 6): Surface Tiling Mode \
             was turned off"
        )
    );
}

/// An Image Detail drag of the patch: the dot of `row` slid one pixel along
/// its photograph's `x`, which with *Lock* ticked moves the patch and every
/// sighting with it.
fn detail_slide(state: &AppState, id: ReconId, label: &str, row: usize) -> crate::bench::PatchEdit {
    let track = state.bench_track(id, label).expect("on the bench");
    let [x, y] = track.observations[row].site().expect("a placed row");
    crate::bench::PatchEdit::TranslateToPixel {
        viewpoint: sfmtool_core::bench::Viewpoint::Observation(row),
        pixel: [x + 1.0, y],
    }
}

/// In Surface Tiling Mode a patch edit made with Image Detail's handles
/// starts the tiling sequence, as a 3D viewer gesture does, through the same
/// dock call; with the mode off the same edit starts none.
#[test]
fn an_image_detail_patch_edit_starts_the_sequence_only_in_surface_tiling_mode() {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    let mut track_view = TrackView::new();

    let edit = detail_slide(&state, id, &label, 0);
    with_tabs(&mut state, &mut track_view, |tabs, _| {
        tabs.apply_patch_gesture(id, &label, &edit)
    });
    let last = state.action_log.entries().last().expect("a row");
    assert!(
        last.text.starts_with(&format!("Moved {label} ")),
        "{}",
        last.text
    );
    assert!(state.surface_tiling.is_none(), "the mode is off");

    track_view.set_surface_tiling(true);
    let edit = detail_slide(&state, id, &label, 0);
    with_tabs(&mut state, &mut track_view, |tabs, _| {
        tabs.apply_patch_gesture(id, &label, &edit)
    });
    let texts: Vec<String> = state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect();
    let n = texts.len();
    assert!(
        texts[n - 2].starts_with(&format!("Moved {label} ")),
        "{texts:?}"
    );
    assert!(
        texts[n - 1].starts_with(&format!("Surface tiling {label} after the move")),
        "{texts:?}"
    );
    assert_eq!(
        state.surface_tiling_status(id, &label).as_deref(),
        Some("Step 1 of 6: Fit")
    );
}

/// In Surface Tiling Mode an Image Detail edit of one sighting alone moves no
/// patch, so it starts no sequence.
#[test]
fn an_image_detail_sighting_edit_starts_no_sequence() {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    state.settle_bench_evaluation();
    let mut track_view = TrackView::new();
    track_view.set_surface_tiling(true);
    let [x, y] = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations[0]
        .site()
        .expect("a placed row");
    let edit = crate::bench::PatchEdit::Sight {
        observation: 0,
        pixel: [x + 1.0, y],
    };

    with_tabs(&mut state, &mut track_view, |tabs, _| {
        tabs.apply_patch_gesture(id, &label, &edit)
    });

    let last = state.action_log.entries().last().expect("a row");
    assert!(!last.failed, "{}", last.text);
    assert!(state.surface_tiling.is_none(), "{}", last.text);
}
