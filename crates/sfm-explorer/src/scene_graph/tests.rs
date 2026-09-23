// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Scene Graph panel and the multi-node state machine
//! behind it.
//!
//! egui needs no GPU to lay out a frame, so the panel tests run the real thing
//! through `Context::run_ui` (the `track_view/view/tests.rs` pattern): the
//! tree is really built, `CollapsingState` really stores its expansion under
//! [`row_id`], and clicks are really delivered by pointer events. Clicks aim at
//! the rects the panel recorded on the previous frame
//! ([`SceneGraphPanel::hit_rect`]), because a collapsible virtualized tree has
//! no geometry a test could predict from outside.
//!
//! The rest — cache purge, selection fallback, the finer-selection invariant,
//! `[` / `]` stepping, label disambiguation — is `AppState` behaviour and needs
//! no frame at all.

use std::sync::Arc;

use eframe::egui;
use nalgebra::Vector3;
use sfmtool_core::reconstruction::ObservationSource;
use sfmtool_core::{RotQuaternion, Se3Transform, SfmrReconstruction};

use super::{row_id, SceneGraphPanel};
use crate::align::{AlignOptions, AlignSource};
use crate::resect::ResectFrom;
use crate::scene::{CameraRef, ImageRef, NodeTint, PointRef, SceneNode, TINT_PALETTE};
use crate::state::{AppState, CachedSiftFeatures};
use crate::viewer_3d::Viewer3D;

const VIEWPORT: egui::Vec2 = egui::vec2(320.0, 900.0);

// ── Fixtures ────────────────────────────────────────────────────────────

/// A demo reconstruction padded to `images` images named `<prefix>_<i>.jpg`, so
/// two nodes can be made to share image names or not, at will, and so a camera
/// list can be made longer than the panel shows at once.
///
/// Only the image *list* grows: nothing exercised here reads the per-image side
/// tables, and `SfmrReconstruction::demo` fixes the camera ring at 8.
fn recon_named(points: usize, images: usize, prefix: &str) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(points);
    let template = recon.image_table.images[0].clone();
    while recon.image_table.images.len() < images {
        recon.image_table.images.push(template.clone());
    }
    recon.image_table.images.truncate(images);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("{prefix}_{i:03}.jpg");
    }
    recon
}

/// A node loaded from `path`, the way `File > Open` would build it.
fn file_node(path: &str, images: usize, prefix: &str) -> SceneNode {
    SceneNode::from_path(std::path::Path::new(path), recon_named(32, images, prefix))
}

/// State holding `n` file-backed nodes that all share image names — the
/// comparison case: the same shoot solved several times.
fn shared_shoot(n: usize) -> AppState {
    let mut state = AppState::new();
    for i in 0..n {
        state.append_node(file_node(&format!("/runs/run_{i}.sfmr"), 8, "IMG"));
    }
    // Land on the first node, as if the user had clicked back to it.
    let first = state.scene[0].id;
    state.select_recon(first);
    state
}

/// A node whose eight images resolve to **two** cameras: the first four
/// through camera 0, the rest through camera 1.
///
/// `kerry_park` in miniature — a rig is what makes the difference between an
/// image and the intrinsics it was taken through visible at all, and a
/// one-camera node cannot tell the two selection fields apart.
fn two_camera_node(path: &str) -> SceneNode {
    let mut recon = recon_named(32, 8, "IMG");
    let second = recon.image_table.cameras[0].clone();
    recon.image_table.cameras.push(second);
    for image in recon.image_table.images.iter_mut().skip(4) {
        image.camera_index = 1;
    }
    recon.metadata.camera_count = recon.image_table.cameras.len() as u32;
    SceneNode::from_path(std::path::Path::new(path), recon)
}

/// [`two_camera_node`] with its second camera referenced by nothing — a state
/// a `.sfmr` allows and the tree has to show honestly rather than hide.
fn unused_camera_node(path: &str) -> SceneNode {
    let mut node = two_camera_node(path);
    for image in node.recon_mut().image_table.images.iter_mut() {
        image.camera_index = 0;
    }
    node
}

/// A node carrying `cameras` intrinsics records (at least two), for the
/// expand-by-default threshold, where only the count matters.
fn camera_count_node(path: &str, cameras: usize) -> SceneNode {
    let mut node = two_camera_node(path);
    let template = node.recon().image_table.cameras[0].clone();
    while node.recon().image_table.cameras.len() < cameras {
        node.recon_mut().image_table.cameras.push(template.clone());
    }
    node.recon_mut().metadata.camera_count = node.recon().image_table.cameras.len() as u32;
    node
}

/// A node whose reconstruction has *distinct* camera poses — unlike
/// [`recon_named`], which clones one image's pose for every row. Alignment fits
/// on camera centres, so it needs a real camera ring.
fn posed_node(path: &str) -> SceneNode {
    SceneNode::from_path(std::path::Path::new(path), SfmrReconstruction::demo(64))
}

/// A similarity with a rotation, a translation and a scale change, so a fit
/// that only handles some of the three cannot pass.
fn known_similarity() -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.3, -0.7, 0.6), 0.9).unwrap(),
        Vector3::new(4.0, -2.5, 1.25),
        2.0,
    )
}

/// `recon` with every point and camera pose put through `t`.
fn transformed(recon: &SfmrReconstruction, t: &Se3Transform) -> SfmrReconstruction {
    let mut out = recon.clone();
    for p in &mut out.point_set.points {
        p.position = t.apply_to_point(&p.position);
    }
    for image in &mut out.image_table.images {
        let (rotation, translation) = t.apply_to_camera_pose(
            &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
            &image.translation_xyz,
        );
        image.quaternion_wxyz = *rotation.as_nalgebra();
        image.translation_xyz = translation;
    }
    out
}

/// Two nodes: `run_a` as loaded, `run_b` the same scene under
/// [`known_similarity`]. The comparison case the whole feature exists for.
fn misaligned_pair() -> AppState {
    let mut state = AppState::new();
    let a = posed_node("/runs/run_a.sfmr");
    let b_recon = transformed(a.recon(), &known_similarity());
    state.append_node(a);
    let mut b = SceneNode::from_path(std::path::Path::new("/runs/run_b.sfmr"), b_recon);
    b.label = "run_b".to_string();
    state.append_node(b);
    state
}

/// Worst distance between the node's points *as displayed* and `target`'s.
fn worst_display_error(node: &SceneNode, target: &SceneNode, target_frame: &Se3Transform) -> f64 {
    node.recon()
        .point_set
        .points
        .iter()
        .zip(target.recon().point_set.points.iter())
        .map(|(s, t)| {
            (node.transform().apply_to_point(&s.position)
                - target_frame.apply_to_point(&t.position))
            .norm()
        })
        .fold(0.0, f64::max)
}

// ── Frame driving ───────────────────────────────────────────────────────

/// Run one frame of the panel with `events` delivered, returning its response.
fn run_frame(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
) -> super::SceneGraphResponse {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            response = Some(panel.show(ui, state));
        });
    });
    response.expect("the panel ran")
}

/// The strings one frame of the panel painted, at a panel `width` of the
/// caller's choosing.
///
/// The counts row elides on the width left over after the label, so a test of
/// that has to vary the width and read back what was actually drawn rather
/// than what was asked for.
fn painted_at_width(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    width: f32,
) -> Vec<String> {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, VIEWPORT.y),
        )),
        ..Default::default()
    };
    crate::test_support::painted_texts(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    })
}

/// The counts drawn on the one reconstruction row, at panel width `width`.
///
/// `None` once the panel is narrow enough that egui clips the row past the
/// three toggles and paints neither label nor counts — the end of what elision
/// has any say over.
fn counts_at_width(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    width: f32,
) -> Option<String> {
    // Two frames: egui settles a layout against the previous pass, so the
    // first frame at a new width is still measuring the old one.
    painted_at_width(panel, ctx, state, width);
    painted_at_width(panel, ctx, state, width)
        .into_iter()
        .find(|text| text.contains(" pts"))
}

/// A panel + context that have already settled: egui resolves hover and clicks
/// against the widget rects registered on the *previous* pass, so nothing can
/// be clicked until at least one frame has been laid out.
fn settled(state: &mut AppState) -> (SceneGraphPanel, egui::Context) {
    let mut panel = SceneGraphPanel::new();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, state, Vec::new());
    }
    (panel, ctx)
}

fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
    button(pos, egui::PointerButton::Primary, pressed)
}

fn button(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::default(),
    }
}

/// Right-click the element recorded under `id`, opening its context menu. The
/// menu's own rows are laid out (and recorded) on the frames that follow.
fn open_context_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    id: egui::Id,
) {
    let pos = panel
        .hit_rect(id)
        .unwrap_or_else(|| panic!("{id:?} was not drawn on the previous frame"))
        .center();
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, true)],
    );
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, false)],
    );
    // One more settling frame so the menu's contents register their rects.
    run_frame(panel, ctx, state, Vec::new());
}

/// Right-click a reconstruction row and open its `Align to ▸` submenu, leaving
/// the target buttons and the option radios laid out and clickable.
fn open_align_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    node: crate::scene::ReconId,
) {
    open_context_menu(panel, ctx, state, row_id(node, "node_label"));
    click(panel, ctx, state, row_id(node, "align_menu"));
    run_frame(panel, ctx, state, Vec::new());
}

/// Right-click at an explicit position rather than at a recorded element, for
/// the parts of a row that are not a widget of their own.
fn right_click_at(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
) {
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, true)],
    );
    run_frame(
        panel,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Secondary, false)],
    );
    run_frame(panel, ctx, state, Vec::new());
}

/// Every string painted with the pointer resting at `pos`, the tooltip it
/// raises included.
///
/// Two frames, because egui puts a tooltip up only once the pointer has stopped
/// moving and the frame carrying the move is the frame it was still moving in.
/// The delays go to zero first: a headless frame has no wall clock to pass, and
/// what is under test is which sentence comes up rather than how long a reader
/// waits for it.
fn hover_texts_at(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
) -> Vec<String> {
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        ..Default::default()
    };
    crate::test_support::painted_texts(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    })
}

/// Whether a reconstruction row's context menu is on screen: its entries lay
/// their own rects out only on the frames the menu is actually shown.
fn context_menu_open(panel: &SceneGraphPanel, node: crate::scene::ReconId) -> bool {
    panel.hit_rect(row_id(node, "reset_transform")).is_some()
}

/// Hover, press, release on the element recorded under `id` — the three frames
/// egui needs to register a click. Returns the response of the frame the click
/// landed in.
fn click(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    id: egui::Id,
) -> super::SceneGraphResponse {
    let pos = panel
        .hit_rect(id)
        .unwrap_or_else(|| panic!("{id:?} was not drawn on the previous frame"))
        .center();
    click_at(panel, ctx, state, pos)
}

fn click_at(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
) -> super::SceneGraphResponse {
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    run_frame(panel, ctx, state, vec![press(pos, true)]);
    run_frame(panel, ctx, state, vec![press(pos, false)])
}

/// Whether a row was drawn at all: `CollapsingState` is only stored for a
/// header that actually ran.
fn drawn(ctx: &egui::Context, id: egui::Id) -> bool {
    egui::collapsing_header::CollapsingState::load(ctx, id).is_some()
}

fn set_open(ctx: &egui::Context, id: egui::Id, open: bool) {
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ctx, id, false);
    state.set_open(open);
    state.store(ctx);
}

// ── Tree structure ──────────────────────────────────────────────────────

#[test]
fn every_loaded_node_gets_a_row_with_its_three_groups() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let (_panel, ctx) = settled(&mut state);

    for id in ids {
        assert!(drawn(&ctx, row_id(id, "node")), "node row missing");
        assert!(
            drawn(&ctx, row_id(id, "camera_images")),
            "Camera Images row missing"
        );
        assert!(
            drawn(&ctx, row_id(id, "intrinsics")),
            "Camera Intrinsics row missing"
        );
        assert!(drawn(&ctx, row_id(id, "points")), "Points row missing");
    }
}

#[test]
fn the_image_rows_appear_only_once_the_camera_images_group_is_expanded() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(id, "image_0")).is_none(),
        "the collapsed Camera Images group still laid out its rows"
    );
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "image_0")).is_some(),
        "expanding the Camera Images group did not draw any image rows"
    );
}

#[test]
fn the_image_list_lays_out_only_the_visible_rows() {
    // 400 images against a 220px-tall list: a non-virtualized list would lay
    // out every one of them.
    let mut state = AppState::new();
    state.append_node(file_node("/runs/big.sfmr", 400, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let laid_out = (0..400)
        .filter(|i| panel.hit_rect(row_id(id, &format!("image_{i}"))).is_some())
        .count();
    assert!(
        laid_out > 0 && laid_out < 60,
        "{laid_out} of 400 image rows were laid out; the list is not virtualized"
    );
}

#[test]
fn the_patches_row_appears_only_for_a_node_that_carries_patch_data() {
    use ndarray::{Array2, Array4};

    let mut state = shared_shoot(2);
    let plain = state.scene[0].id;
    let patched = state.scene[1].id;
    {
        let recon = state.scene[1].recon_mut();
        let n = recon.point_set.points.len();
        recon.point_set.patch_u_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
        recon.point_set.patch_v_halfvec_xyz = Some(Array2::<f32>::from_elem((n, 3), 0.1));
        recon.point_set.patch_bitmaps_y_x_rgba =
            Some(Arc::new(Array4::<u8>::from_elem((n, 8, 8, 4), 200)));
    }
    let (panel, _ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(patched, "patches_eye")).is_some(),
        "a node with patch bitmaps got no Patches row"
    );
    assert!(
        panel.hit_rect(row_id(plain, "patches_eye")).is_none(),
        "a node without patch bitmaps got a Patches row anyway"
    );
}

#[test]
fn the_infinity_mini_toggle_appears_only_when_the_node_has_points_at_infinity() {
    let mut state = shared_shoot(2);
    let none = state.scene[0].id;
    let some = state.scene[1].id;
    state.scene[1].recon_mut().metadata.infinity_point_count = 12;
    let (panel, _ctx) = settled(&mut state);

    assert!(panel.hit_rect(row_id(some, "points_infinity")).is_some());
    assert!(panel.hit_rect(row_id(none, "points_infinity")).is_none());
}

#[test]
fn the_panel_glyphs_are_available_in_the_bundled_fonts() {
    // A glyph egui does not bundle renders as a replacement box rather than
    // failing, so nothing else here would notice.
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(14.0);
    // The selection accent bar is deliberately absent: it is painted, not
    // written, precisely because no bundled proportional glyph would do.
    for glyph in [
        super::EYE_GLYPH,
        super::CURSOR_GLYPH,
        super::INFINITY_GLYPH,
        // A plain letter, so this one cannot fail — kept in the list because
        // the reason it is a letter is precisely that no bundled pictograph
        // says "only this one" (see `SOLO_GLYPH`).
        super::SOLO_GLYPH,
    ] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

// ── The bench groups ────────────────────────────────────────────────────

/// A node with a bench holding one track-stage item -- a point put on it -- and
/// one cluster-stage item started from a pixel, with the labels of the two.
///
/// The fixture is the bench module's own, photographs and all, because a stage
/// change reads them.
fn benched() -> (AppState, crate::scene::ReconId, String, String) {
    let (mut state, id) = crate::bench::tests::state();
    let point = crate::bench::tests::put_on_bench(&mut state, id);
    let cluster = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    (state, id, point, cluster)
}

/// The label of the group a bench row is drawn under, read off what the panel
/// painted: the rows are laid out under their group's body, so which group an
/// item is in is the count beside each title.
fn bench_titles(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
) -> Vec<String> {
    painted_at_width(panel, ctx, state, VIEWPORT.x)
        .into_iter()
        .filter(|text| text.starts_with("Bench "))
        .collect()
}

#[test]
fn the_bench_is_two_groups_one_per_stage() {
    let (mut state, id, point, cluster) = benched();
    let (mut panel, ctx) = settled(&mut state);

    assert!(drawn(&ctx, row_id(id, "bench_points")), "no Bench Points");
    assert!(
        drawn(&ctx, row_id(id, "bench_clusters")),
        "no Bench Clusters"
    );
    assert_eq!(
        bench_titles(&mut panel, &ctx, &mut state),
        ["Bench Points (1)", "Bench Clusters (1)"],
        "the two groups and their counts"
    );
    // Each item is drawn under the group its stage belongs to, which is what
    // the rows' own rects say.
    assert!(panel
        .hit_rect(row_id(id, &format!("bench_item_{point}")))
        .is_some());
    assert!(panel
        .hit_rect(row_id(id, &format!("bench_item_{cluster}")))
        .is_some());
}

/// The point of the split: an item taken down to the cluster stage leaves the
/// points group for the clusters group.
#[test]
fn an_item_changes_group_when_its_stage_changes() {
    let (mut state, id, point, _cluster) = benched();
    let (mut panel, ctx) = settled(&mut state);
    assert_eq!(
        bench_titles(&mut panel, &ctx, &mut state),
        ["Bench Points (1)", "Bench Clusters (1)"]
    );

    state
        .start_bench_stage(id, &point, sfmtool_core::bench::StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();

    assert_eq!(
        bench_titles(&mut panel, &ctx, &mut state),
        ["Bench Clusters (2)"],
        "the item did not move, or the emptied group was still drawn"
    );
    assert!(
        panel
            .hit_rect(row_id(id, &format!("bench_item_{point}")))
            .is_some(),
        "the item it moved is drawn under the group it moved to"
    );
}

/// An empty group is not drawn at all, which is what the one bench group did
/// with an empty bench -- and the other group is drawn all the same.
#[test]
fn a_group_with_nothing_in_it_is_not_drawn() {
    let (mut state, id) = crate::bench::tests::state();
    let (mut panel, ctx) = settled(&mut state);
    assert!(
        bench_titles(&mut panel, &ctx, &mut state).is_empty(),
        "an empty bench drew a group"
    );

    crate::bench::tests::put_on_bench(&mut state, id);
    assert_eq!(
        bench_titles(&mut panel, &ctx, &mut state),
        ["Bench Points (1)"],
        "a bench of one track drew a clusters group"
    );
    assert!(!drawn(&ctx, row_id(id, "bench_clusters")));
}

/// [`benched`], with a second node appended after it and selected, so that
/// what a gesture on the first node's Bench rows does to the node selection
/// can be seen.
fn benched_behind_another() -> (AppState, crate::scene::ReconId, String, String) {
    let (mut state, id, point, cluster) = benched();
    let other = state.append_node(SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    state.select_recon(other);
    assert_ne!(state.selected_recon, Some(id));
    (state, id, point, cluster)
}

/// The number of versions `id`'s history holds.
fn versions(state: &AppState, id: crate::scene::ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// A single click on a Bench row selects the node it is under and does nothing
/// to the bench: the node row's own pair, click to select and double-click to
/// act, is the pattern the bench rows follow, and a pass of clicks down the
/// tree pushes no versions.
#[test]
fn a_single_click_on_a_bench_row_selects_its_node_only() {
    let (mut state, id, point, cluster) = benched_behind_another();
    let (mut panel, ctx) = settled(&mut state);
    let before = versions(&state, id);
    let active =
        crate::bench::active_track_label(state.bench(id).expect("a bench")).map(str::to_string);
    assert_eq!(active.as_deref(), Some(cluster.as_str()));

    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, &format!("bench_item_{point}")),
    );
    assert_eq!(response.select_recon, Some(id));
    assert_eq!(response.edit_bench_item, None);
    assert_eq!(
        panel.take_bench_edit(),
        None,
        "a single click asked for a raise"
    );
    state.select_recon(id);

    assert_eq!(versions(&state, id), before, "a click pushed a version");
    assert_eq!(
        crate::bench::active_track_label(state.bench(id).expect("a bench")),
        active.as_deref(),
        "a click changed the active item"
    );
}

/// A double-click on a Bench row makes that item active, selects the node it
/// is under and raises Track View, from either group: the active item is one
/// across the two, because the bench's kinds are items, not stages.
#[test]
fn a_double_click_on_a_bench_row_edits_its_item() {
    let (_, _, point, _) = benched();
    // The cluster, put on last, is the active item, so the point row is the
    // one a double-click moves the activation to; and the point is made active
    // first for the cluster row's turn, so both are a real change.
    for (which, group) in [(0, "Bench Points"), (1, "Bench Clusters")] {
        let (mut state, id, point_label, cluster_label) = benched_behind_another();
        let label = if which == 0 {
            point_label
        } else {
            cluster_label
        };
        if which == 1 {
            state.activate_bench_item(id, &point).expect("on the bench");
            state.select_recon(state.scene[1].id);
        }
        let (mut panel, ctx) = settled(&mut state);
        state.hide_panel(crate::dock::Tab::TrackView);

        let pos = panel
            .hit_rect(row_id(id, &format!("bench_item_{label}")))
            .unwrap_or_else(|| panic!("the {group} row"))
            .center();
        click_at(&mut panel, &ctx, &mut state, pos);
        let response = click_at(&mut panel, &ctx, &mut state, pos);
        let (node, position) = response
            .edit_bench_item
            .unwrap_or_else(|| panic!("a {group} double-click reported nothing"));
        assert_eq!(node, id);
        assert_eq!(
            state.bench(id).expect("a bench").entries()[position].label,
            label,
            "the position names the item in the whole bench"
        );
        let before = versions(&state, id);
        let (node, position) = panel.take_bench_edit().expect("the panel kept the request");
        state.edit_bench_item_at(node, position);

        assert_eq!(state.selected_recon, Some(id), "the node was not selected");
        assert_eq!(
            crate::bench::active_track_label(state.bench(id).expect("a bench")),
            Some(label.as_str()),
            "{group}"
        );
        assert_eq!(
            versions(&state, id),
            before + 1,
            "one activation, one version"
        );
        assert!(state.is_panel_open(crate::dock::Tab::TrackView));
    }
}

/// A double-click on the item already active asked for the panel, and the
/// panel is what it gets: no version, and no no-effect row.
#[test]
fn a_double_click_on_the_active_item_only_raises_the_panel() {
    let (mut state, id, _point, cluster) = benched();
    let position = state
        .bench(id)
        .expect("a bench")
        .position(&cluster)
        .expect("on the bench");
    state.hide_panel(crate::dock::Tab::TrackView);
    let before = versions(&state, id);
    let rows = state.action_log.entries().count();

    state.edit_bench_item_at(id, position);

    assert_eq!(versions(&state, id), before, "a version was pushed");
    let bench_rows = state
        .action_log
        .entries()
        .skip(rows)
        .filter(|entry| entry.kind == crate::action_log::Kind::Bench)
        .count();
    assert_eq!(bench_rows, 0, "a bench row was written");
    assert!(state.is_panel_open(crate::dock::Tab::TrackView));
}

/// The frame takes the dock out of the state while a tab body draws, so the
/// raise the double-click asks for is kept by the panel past that swap and
/// applied by `app.rs` afterwards, as Image Detail's *Edit on Bench* is. Here
/// the double-click lands while the dock is swapped out, and the raise still
/// reaches the real dock once the request is drained.
#[test]
fn the_double_click_raise_survives_the_swapped_out_dock() {
    let (mut state, id, point, _cluster) = benched();
    let (mut panel, ctx) = settled(&mut state);
    state.hide_panel(crate::dock::Tab::TrackView);
    let pos = panel
        .hit_rect(row_id(id, &format!("bench_item_{point}")))
        .expect("the Bench Points row")
        .center();

    let dock = std::mem::replace(&mut state.dock, egui_dock::DockState::new(Vec::new()));
    click_at(&mut panel, &ctx, &mut state, pos);
    click_at(&mut panel, &ctx, &mut state, pos);
    let placeholder = std::mem::replace(&mut state.dock, dock);
    assert!(
        placeholder.find_tab(&crate::dock::Tab::TrackView).is_none(),
        "the panel raised inside the swap"
    );

    let (node, position) = panel.take_bench_edit().expect("the panel kept the request");
    state.edit_bench_item_at(node, position);
    assert!(state.is_panel_open(crate::dock::Tab::TrackView));
    assert_eq!(
        crate::bench::active_track_label(state.bench(id).expect("a bench")),
        Some(point.as_str())
    );
}

/// *Discard* is on a row of either group, and names the item the row is about.
#[test]
fn discard_works_from_either_group() {
    let (mut state, id, point, cluster) = benched();
    let (mut panel, ctx) = settled(&mut state);

    for label in [&cluster, &point] {
        open_context_menu(
            &mut panel,
            &ctx,
            &mut state,
            row_id(id, &format!("bench_item_{label}")),
        );
        let response = click(
            &mut panel,
            &ctx,
            &mut state,
            row_id(id, &format!("bench_discard_{label}")),
        );
        let (_, position) = response
            .discard_bench_item
            .unwrap_or_else(|| panic!("Discard on {label} reported nothing"));
        assert_eq!(
            &state.bench(id).expect("a bench").entries()[position].label,
            label
        );
        state
            .discard_bench_item(id, label)
            .expect("on the bench, as the dock would");
        run_frame(&mut panel, &ctx, &mut state, Vec::new());
    }
    assert!(state.bench(id).expect("a bench").is_empty());
}

/// Each group remembers its own expansion: collapsing the points group leaves
/// the clusters group's rows drawn.
#[test]
fn each_group_collapses_on_its_own() {
    let (mut state, id, point, cluster) = benched();
    let (mut panel, ctx) = settled(&mut state);

    set_open(&ctx, row_id(id, "bench_points"), false);
    // Several frames: a collapsing header animates shut and keeps drawing its
    // body while it does.
    for _ in 0..30 {
        run_frame(&mut panel, &ctx, &mut state, Vec::new());
    }

    assert!(
        panel
            .hit_rect(row_id(id, &format!("bench_item_{point}")))
            .is_none(),
        "a collapsed group still drew its rows"
    );
    assert!(
        panel
            .hit_rect(row_id(id, &format!("bench_item_{cluster}")))
            .is_some(),
        "collapsing one group closed the other"
    );
}

// ── Clicks and toggles ──────────────────────────────────────────────────

#[test]
fn clicking_a_reconstruction_row_reports_it_as_the_selection() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_label"));
    assert_eq!(response.select_recon, Some(second));
    // The panel reports; `dock.rs` applies. Nothing was selected behind its back.
    assert_ne!(state.selected_recon, Some(second));
}

#[test]
fn clicking_an_image_row_reports_the_image_it_names() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    assert_eq!(response.select_image, Some(ImageRef::new(id, 2)));
}

/// An image row keeps working once the virtualized list has scrolled off its
/// first slice — its identity is the image index, not its place in whatever
/// slice is currently rendered.
#[test]
fn an_image_row_still_selects_after_the_list_has_scrolled() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/long.sfmr", 200, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    // Scroll row 150 into view the only way the panel offers: a selection made
    // elsewhere.
    state.select_image(Some(ImageRef::new(id, 150)));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "image_150"));
    assert_eq!(response.select_image, Some(ImageRef::new(id, 150)));
}

#[test]
fn the_eye_and_cursor_toggles_write_through_to_the_node() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    assert!(state.scene[0].visible && state.scene[0].interactive);

    click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(!state.scene[0].visible, "the eye did not hide the node");
    assert!(
        state.scene[0].interactive,
        "the eye also flipped the interaction cursor"
    );

    click(&mut panel, &ctx, &mut state, row_id(id, "node_cursor"));
    assert!(
        !state.scene[0].interactive,
        "the cursor toggle did not turn interaction off"
    );
    assert!(
        !state.scene[0].visible,
        "the cursor toggle also moved the eye"
    );

    // And back again — a toggle that only worked once would still pass above.
    click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(state.scene[0].visible);
}

#[test]
fn the_group_eyes_drive_their_own_layers_only() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "camera_images_eye"),
    );
    assert!(!state.scene[0].show_camera_images);
    assert!(
        state.scene[0].show_points,
        "the Camera Images eye hid the points"
    );

    click(&mut panel, &ctx, &mut state, row_id(id, "points_eye"));
    assert!(!state.scene[0].show_points);
    assert!(
        state.scene[0].visible,
        "a group eye should not touch the master eye"
    );
}

#[test]
fn the_infinity_mini_toggle_drives_only_the_infinity_points() {
    let mut state = shared_shoot(1);
    state.scene[0].recon_mut().metadata.infinity_point_count = 12;
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    click(&mut panel, &ctx, &mut state, row_id(id, "points_infinity"));
    assert!(!state.scene[0].show_points_at_infinity);
    assert!(
        state.scene[0].show_points,
        "the ∞ toggle also hid the finite points"
    );
}

#[test]
fn the_selected_reconstruction_is_marked_in_the_tree() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    // The accent bar is painted, not written, so what is recorded for it is the
    // rect it was painted into — present only on the row that carries it.
    assert!(
        panel.hit_rect(row_id(first, "node_selected_bar")).is_some(),
        "the selected node got no marker"
    );
    assert!(
        panel
            .hit_rect(row_id(second, "node_selected_bar"))
            .is_none(),
        "an unselected node was marked too"
    );

    state.select_recon(second);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel
            .hit_rect(row_id(second, "node_selected_bar"))
            .is_some()
            && panel.hit_rect(row_id(first, "node_selected_bar")).is_none(),
        "the marker did not follow the selection"
    );
}

/// The space the bar occupies is reserved on every row, marked or not, so the
/// names line up down the tree and the selection does not shove one sideways.
#[test]
fn the_selection_marker_takes_no_room_from_the_row_it_is_not_on() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    let (panel, _ctx) = settled(&mut state);

    assert_eq!(
        panel.hit_rect(row_id(first, "node_label")).unwrap().left(),
        panel.hit_rect(row_id(second, "node_label")).unwrap().left(),
        "the selected row starts at a different x from the unselected one",
    );
}

#[test]
fn the_points_group_shows_the_selected_point_id_and_never_a_listing() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "points"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "point_selected")).is_none(),
        "a selection row appeared with nothing selected"
    );

    state.select_point(PointRef::new(id, 7));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "point_selected")).is_some(),
        "the selected point got no row"
    );

    // Clicking it re-selects the same point (useful after selecting elsewhere).
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "point_selected"));
    assert_eq!(response.select_point, Some(PointRef::new(id, 7)));
}

#[test]
fn hovering_an_image_row_reports_it_for_cross_panel_hover() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let pos = panel.hit_rect(row_id(id, "image_3")).unwrap().center();
    run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    let response = run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    assert!(response.has_pointer, "the panel did not claim the pointer");
    assert_eq!(response.hovered_image, Some(ImageRef::new(id, 3)));
}

#[test]
fn the_image_list_scrolls_to_a_selection_made_elsewhere_but_not_to_its_own() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/long.sfmr", 200, "IMG"));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "image_150")).is_none(),
        "row 150 is visible without scrolling, so this test proves nothing"
    );

    // A selection change from another panel scrolls the row into view.
    state.select_image(Some(ImageRef::new(id, 150)));
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        panel.hit_rect(row_id(id, "image_150")).is_some(),
        "the selected row was not scrolled into view"
    );

    // With the selection unchanged the list stays where the user left it: a
    // second frame must not re-apply the scroll.
    let before = panel.hit_rect(row_id(id, "image_150")).unwrap();
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(
        panel.hit_rect(row_id(id, "image_150")),
        Some(before),
        "the list kept scrolling itself with the selection unchanged"
    );
}

/// The other half of the interaction-cursor contract: switching a node's picks
/// off must not make it uninspectable. The tree is the control surface, so a
/// display-only node can always be selected deliberately.
///
/// (The GPU half — `pickable == 0` produces no hover or selection from the
/// readback — is asserted in `scene_renderer/upload/tests.rs`.)
#[test]
fn a_non_interactive_node_can_still_be_selected_from_the_tree() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    state.scene[1].interactive = false;
    let (mut panel, ctx) = settled(&mut state);
    set_open(&ctx, row_id(second, "camera_images"), true);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_label"));
    assert_eq!(response.select_recon, Some(second));

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "image_1"));
    assert_eq!(response.select_image, Some(ImageRef::new(second, 1)));
    assert!(
        !state.scene[1].interactive,
        "selecting from the tree quietly re-armed the node's picks"
    );
}

#[test]
fn an_empty_scene_draws_no_rows() {
    let mut state = AppState::new();
    let (panel, _ctx) = settled(&mut state);
    assert!(panel.hit_rect(egui::Id::new("anything")).is_none());
}

// ── The reconstruction row's context menu ───────────────────────────────

/// The row is one target all the way across, not just the name's own glyphs.
/// A right-click in the gap between the name and the counts is the natural one,
/// and it used to land on nothing at all.
#[test]
fn the_context_menu_opens_from_anywhere_along_the_reconstruction_row() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(b, "node_label")).expect("the row");
    // Far enough right to be past the name and short of the counts — bare row.
    let gap = egui::pos2(row.right() - row.width() / 3.0, row.center().y);
    right_click_at(&mut panel, &ctx, &mut state, gap);

    assert!(
        context_menu_open(&panel, b),
        "right-clicking the row away from its name opened no menu"
    );
}

/// And on the name itself, which is the part of the row a user actually aims
/// at. Worth its own test because a click there does *not* reach the row by
/// default: egui gives a bare label `Sense::click_and_drag()` so its text can be
/// selected, and a label drawn after the row wins every hit that lands on a
/// glyph. Aiming at the row's centre — as the tests around this one do — sails
/// straight past that, because the name is at its left end.
#[test]
fn the_context_menu_opens_on_the_reconstructions_name() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(b, "node_label")).expect("the row");
    // The name is drawn from the row's left edge, past the selection marker.
    let on_the_name = egui::pos2(row.left() + 12.0, row.center().y);

    // The same spot selects on a left click — checked first, because the menu
    // the right click opens lands under the pointer and would eat this one.
    let response = click_at(&mut panel, &ctx, &mut state, on_the_name);
    assert_eq!(
        response.select_recon,
        Some(b),
        "left-clicking the name did not select the reconstruction"
    );

    right_click_at(&mut panel, &ctx, &mut state, on_the_name);
    assert!(
        context_menu_open(&panel, b),
        "right-clicking the reconstruction's name opened no menu"
    );
}

/// Hovering the name says which file the node came from, in full. The label is
/// the stem alone, so a second run of the same capture loaded from another
/// directory reads identically until the hover says otherwise.
#[test]
fn hovering_the_reconstructions_name_names_its_file() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let path = state
        .node(id)
        .expect("loaded")
        .path
        .clone()
        .expect("a file");
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(id, "node_label")).expect("the row");
    let on_the_name = egui::pos2(row.left() + 12.0, row.center().y);
    let hovered = hover_texts_at(&mut panel, &ctx, &mut state, on_the_name);

    let expected = path.display().to_string();
    assert!(
        hovered.contains(&expected),
        "{expected:?} was not the tooltip: {hovered:?}"
    );
}

/// A node that came from no file says so rather than naming a path it has not
/// got.
#[test]
fn hovering_an_unsaved_reconstructions_name_says_it_came_from_no_file() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(id, "node_label")).expect("the row");
    let on_the_name = egui::pos2(row.left() + 12.0, row.center().y);
    let hovered = hover_texts_at(&mut panel, &ctx, &mut state, on_the_name);

    assert!(
        hovered
            .iter()
            .any(|t| t == "This reconstruction came from no file."),
        "the hover did not say the node has no file: {hovered:?}"
    );
}

/// The row spans the panel rather than wrapping the name, so there is no dead
/// strip along it for a click to fall into.
#[test]
fn the_reconstruction_row_is_as_wide_as_the_panel() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (panel, _ctx) = settled(&mut state);

    let row = panel.hit_rect(row_id(id, "node_label")).expect("the row");
    assert!(
        row.right() >= VIEWPORT.x - 20.0,
        "the row stops at {} in a {}-wide panel",
        row.right(),
        VIEWPORT.x,
    );
}

/// egui keys a popup on the id of the widget it hangs off. The row used to
/// carry an auto-generated one — a count of what had been laid out before it —
/// which the accent bar shifted the moment the row became the selection, so an
/// open menu lost its identity and silently stopped being drawn.
#[test]
fn the_context_menu_survives_the_rows_selection_changing_under_it() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let b = state.scene[1].id;
    state.select_recon(a);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert!(context_menu_open(&panel, b), "the menu never opened");

    // `Select` from this very menu does exactly this, so the menu has to be able
    // to outlive it.
    state.select_recon(b);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(
        context_menu_open(&panel, b),
        "the menu vanished when the row it belongs to became the selection"
    );
}

/// The other order: the row's selected state changed, and only *then* is the
/// menu asked for.
#[test]
fn the_context_menu_opens_on_a_row_whose_selection_just_changed() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let b = state.scene[1].id;
    state.select_recon(a);
    let (mut panel, ctx) = settled(&mut state);

    // Click the row and apply the selection the way `dock.rs` does.
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert_eq!(response.select_recon, Some(b), "the row did not select");
    state.select_recon(b);
    run_frame(&mut panel, &ctx, &mut state, Vec::new());

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    assert!(
        context_menu_open(&panel, b),
        "no menu on a row that had just been selected"
    );
}

/// Double-click still frames the node, and a right-click is not a selection.
#[test]
fn the_row_keeps_select_on_click_and_zoom_on_double_click() {
    let mut state = shared_shoot(2);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let pos = panel
        .hit_rect(row_id(second, "node_label"))
        .expect("the row")
        .center();
    click_at(&mut panel, &ctx, &mut state, pos);
    let response = click_at(&mut panel, &ctx, &mut state, pos);
    assert_eq!(
        response.zoom_to_node,
        Some(second),
        "double-clicking the row did not ask for a zoom-to-fit"
    );

    let response = run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![
            button(pos, egui::PointerButton::Secondary, true),
            button(pos, egui::PointerButton::Secondary, false),
        ],
    );
    assert_eq!(
        response.select_recon, None,
        "a right-click on the row selected it as well as opening its menu"
    );
}

/// The toggles keep their own clicks: the row-wide target must not swallow
/// them, and they are not a place the node's menu comes from either.
#[test]
fn the_eye_and_cursor_toggles_are_not_part_of_the_rows_target() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "node_eye"));
    assert!(!state.scene[0].visible, "the eye stopped hiding the node");
    assert_eq!(
        response.select_recon, None,
        "the row underneath the eye took the click too"
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_cursor"));
    assert!(
        !context_menu_open(&panel, id),
        "right-clicking a toggle opened the node's menu"
    );
}

// ── Align to… ───────────────────────────────────────────────────────────

#[test]
fn the_align_menu_lists_every_other_loaded_node() {
    let mut state = misaligned_pair();
    let a = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, a);

    assert!(
        panel.hit_rect(row_id(a, "align_to_run_b")).is_some(),
        "the other node is not offered as a target"
    );
    assert!(
        panel.hit_rect(row_id(a, "align_to_run_a")).is_none(),
        "a node was offered as a target for itself"
    );
}

#[test]
fn picking_a_target_reports_the_align_with_the_chosen_options() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    assert_eq!(
        response.align_node,
        Some((b, a, AlignOptions::default())),
        "the source, target or options did not reach the response"
    );
    // The panel reports; `dock.rs` applies. Nothing was fitted behind its back.
    assert!(!state.scene[1].has_transform());
}

#[test]
fn the_align_options_are_remembered_and_travel_with_the_request() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    click(&mut panel, &ctx, &mut state, row_id(b, "align_points"));
    click(&mut panel, &ctx, &mut state, row_id(b, "align_rigid"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    assert_eq!(
        response.align_node,
        Some((
            b,
            a,
            AlignOptions {
                source: AlignSource::Points,
                estimate_scale: false,
            }
        )),
    );
}

#[test]
fn the_point_mode_is_disabled_without_feature_indexes_in_both_nodes() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    // Node A carries embedded keypoints, so there is no feature index for a
    // point correspondence to be keyed on.
    let tracks = state.scene[0].recon().point_set.tracks.len();
    let images = state.scene[0].recon().image_table.images.len();
    state.scene[0].recon_mut().point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: ndarray::Array2::<f32>::from_elem((tracks, 2), 100.0),
        image_file_hashes: vec![[0u8; 16]; images],
    };
    let (mut panel, ctx) = settled(&mut state);

    open_align_menu(&mut panel, &ctx, &mut state, b);
    click(&mut panel, &ctx, &mut state, row_id(b, "align_points"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "align_to_run_a"));

    let (.., options) = response.align_node.expect("the target is still clickable");
    assert_eq!(
        options.source,
        AlignSource::Cameras,
        "the disabled Points radio still switched the mode"
    );
}

#[test]
fn reset_transform_is_offered_only_once_a_node_has_one() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "reset_transform"));
    assert_eq!(
        response.reset_transform, None,
        "an untransformed node offered a reset that would do nothing"
    );

    state.align_node(b, a, AlignOptions::default());
    let (mut panel, ctx) = settled(&mut state);
    open_context_menu(&mut panel, &ctx, &mut state, row_id(b, "node_label"));
    let response = click(&mut panel, &ctx, &mut state, row_id(b, "reset_transform"));
    assert_eq!(response.reset_transform, Some(b));
}

// ── Tint ────────────────────────────────────────────────────────────────

/// Open a reconstruction row's `Tint ▸` submenu, leaving its entries laid out
/// and clickable.
fn open_tint_menu(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
    node: crate::scene::ReconId,
) {
    open_context_menu(panel, ctx, state, row_id(node, "node_label"));
    click(panel, ctx, state, row_id(node, "tint_menu"));
    run_frame(panel, ctx, state, Vec::new());
}

#[test]
fn the_tint_menu_offers_original_and_the_whole_palette() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    open_tint_menu(&mut panel, &ctx, &mut state, id);

    assert!(
        panel.hit_rect(row_id(id, "tint_original")).is_some(),
        "the Tint menu offers no way back to the original colors"
    );
    for color in TINT_PALETTE.iter() {
        assert!(
            panel
                .hit_rect(row_id(id, &format!("tint_{}", color.name)))
                .is_some(),
            "the palette entry {:?} was not drawn",
            color.name,
        );
    }
}

#[test]
fn picking_a_tint_writes_it_to_the_node_and_picking_original_takes_it_off() {
    let mut state = shared_shoot(2);
    let id = state.scene[0].id;
    let chosen = &TINT_PALETTE[5]; // Vermillion
    let (mut panel, ctx) = settled(&mut state);
    assert_eq!(state.scene[0].tint, NodeTint::Original);

    open_tint_menu(&mut panel, &ctx, &mut state, id);
    click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, &format!("tint_{}", chosen.name)),
    );

    // A tint is per-node display state, so unlike `Solo` it is written straight
    // into the node rather than reported for `dock.rs` to apply.
    assert_eq!(state.scene[0].tint, NodeTint::Tint(chosen));
    assert_eq!(
        state.scene[1].tint,
        NodeTint::Original,
        "tinting one node tinted its neighbour"
    );

    // The menu is still standing, which is the point of it: a color is chosen
    // by looking at the viewport, so the next entry has to be one click away.
    click(&mut panel, &ctx, &mut state, row_id(id, "tint_original"));
    assert_eq!(state.scene[0].tint, NodeTint::Original);
}

/// The swatch is how a tinted node is identifiable in the tree rather than only
/// in the viewport. Painted, not written, so what is recorded is the rect.
#[test]
fn only_a_tinted_row_gets_a_swatch() {
    let mut state = shared_shoot(2);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.scene[0].tint = NodeTint::Tint(&TINT_PALETTE[1]);
    let (panel, _ctx) = settled(&mut state);

    assert!(
        panel.hit_rect(row_id(first, "node_tint_swatch")).is_some(),
        "the tinted node got no swatch"
    );
    assert!(
        panel.hit_rect(row_id(second, "node_tint_swatch")).is_none(),
        "an untinted node was given a swatch anyway"
    );
    // Reserved on every row, marked or not, so a tint does not shove the name
    // sideways (the accent bar's rule).
    assert_eq!(
        panel.hit_rect(row_id(first, "node_label")).unwrap().left(),
        panel.hit_rect(row_id(second, "node_label")).unwrap().left(),
    );
}

/// Tinting is a display change and nothing else: walking through the menu must
/// not move the selection, least of all onto the row the menu belongs to.
#[test]
fn working_the_tint_menu_leaves_the_selection_where_it_was() {
    let mut state = shared_shoot(2);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.select_image(Some(ImageRef::new(first, 2)));
    let (mut panel, ctx) = settled(&mut state);

    open_tint_menu(&mut panel, &ctx, &mut state, second);
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(second, &format!("tint_{}", TINT_PALETTE[0].name)),
    );

    assert_eq!(response.select_recon, None, "the tint menu selected a node");
    assert_eq!(state.selected_recon, Some(first));
    assert_eq!(state.selected_image, Some(ImageRef::new(first, 2)));
    assert_eq!(state.scene[1].tint, NodeTint::Tint(&TINT_PALETTE[0]));
}

#[test]
fn an_untinted_node_writes_the_original_colors_convention() {
    // `a == 0` is what every scene shader reads as "leave my colors alone".
    assert_eq!(NodeTint::Original.to_uniform(), [0.0; 4]);
    assert_eq!(NodeTint::Original.rgb(), None);

    let tint = NodeTint::Tint(&TINT_PALETTE[0]);
    let uniform = tint.to_uniform();
    assert_eq!(uniform[3], crate::scene::TINT_STRENGTH);
    assert!(
        uniform[3] > 0.0 && uniform[3] < 1.0,
        "a tint is a mix, not a repaint"
    );
    assert_eq!(tint.rgb(), Some(TINT_PALETTE[0].rgb));
}

/// The palette is meant to be told apart at a glance, so no two entries may be
/// near-identical — and none of them may be the background this viewer paints
/// on, which is what an eighth, black Okabe–Ito entry would have amounted to.
///
/// The floor is the palette's own closest pair, Orange and Vermillion, 82 apart
/// in summed channel distance; a crude metric, but enough to catch a duplicate
/// or a near-black entry added later.
#[test]
fn the_palette_entries_are_mutually_distinguishable() {
    for (i, a) in TINT_PALETTE.iter().enumerate() {
        let sum: u32 = a.rgb.iter().map(|&c| c as u32).sum();
        assert!(sum > 120, "{:?} is too dark to read as a tint", a.name);
        for b in TINT_PALETTE.iter().skip(i + 1) {
            let distance: u32 = (0..3)
                .map(|k| (a.rgb[k] as i32 - b.rgb[k] as i32).unsigned_abs())
                .sum();
            assert!(
                distance >= 82,
                "{:?} and {:?} are only {distance} apart",
                a.name,
                b.name,
            );
        }
    }
}

// ── Solo ────────────────────────────────────────────────────────────────

#[test]
fn the_solo_toggle_reports_the_row_it_sits_on() {
    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(second, "node_solo"));
    assert_eq!(response.toggle_solo, Some(second));
    // The panel reports; `dock.rs` applies. Solo is app state, not node state,
    // so nothing was soloed behind its back — and no eye moved either.
    assert_eq!(state.solo, None);
    assert!(state.scene.iter().all(|n| n.visible));
}

/// Solo lives on the row, so the row-wide click target must not swallow it and
/// it must not be a place the node's menu comes from — the eye's contract.
#[test]
fn the_solo_toggle_is_not_part_of_the_rows_target() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "node_solo"));
    assert_eq!(response.toggle_solo, Some(id));
    assert_eq!(
        response.select_recon, None,
        "the row underneath the solo toggle took the click too"
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_solo"));
    assert!(
        !context_menu_open(&panel, id),
        "right-clicking the solo toggle opened the node's menu"
    );
}

/// Clicking `S` on the soloed node ends the solo rather than re-soloing it.
#[test]
fn the_solo_toggle_reports_the_soloed_node_again_to_switch_it_off() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    state.toggle_solo(first);
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(first, "node_solo"));
    assert_eq!(response.toggle_solo, Some(first));
    state.toggle_solo(first); // what `dock.rs` does with it
    assert_eq!(state.solo, None);
}

// ── Solo state (no frame needed) ────────────────────────────────────────

/// Effective visibility is one rule — eye AND solo — and every consumer reads
/// it. Here it is directly.
#[test]
fn soloing_hides_every_other_node_without_touching_their_eyes() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    // One node the user had already hidden by hand, which is what makes
    // "restore what was there" a real requirement rather than "show everything".
    state.scene[2].visible = false;

    state.toggle_solo(second);
    assert!(!is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
    assert!(!is_visible(&state.scene[2], state.solo));
    // The eyes are untouched: solo overlays them rather than editing them.
    assert!(state.scene[0].visible && state.scene[1].visible && !state.scene[2].visible);

    state.toggle_solo(second);
    assert_eq!(state.solo, None);
    assert!(is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
    assert!(
        !is_visible(&state.scene[2], state.solo),
        "un-soloing revealed a node the user had hidden before the solo"
    );
}

#[test]
fn soloing_a_second_node_moves_the_solo_rather_than_adding_to_it() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let (a, b) = (state.scene[0].id, state.scene[1].id);

    state.toggle_solo(a);
    state.toggle_solo(b);

    assert_eq!(state.solo, Some(b));
    assert!(!is_visible(&state.scene[0], state.solo));
    assert!(is_visible(&state.scene[1], state.solo));
}

#[test]
fn an_eye_toggled_while_soloed_takes_effect_when_the_solo_ends() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(2);
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.toggle_solo(a);

    // Switching the hidden node's eye off while it is soloed away: it changes
    // nothing on screen now, and everything the moment the solo ends.
    state.scene[1].visible = false;
    assert!(!is_visible(&state.scene[1], state.solo));

    state.toggle_solo(a);
    assert!(is_visible(&state.scene[0], state.solo));
    assert!(!is_visible(&state.scene[1], state.solo));

    // And the soloed node's own eye still applies while it is soloed: solo says
    // "hide the others", not "force this one on".
    state.toggle_solo(b);
    assert!(!is_visible(&state.scene[1], state.solo));
}

#[test]
fn closing_the_soloed_node_ends_the_solo() {
    use crate::scene::is_visible;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    state.toggle_solo(second);

    state.close_node(second).expect("nothing is running");

    // A solo naming a node that is gone would hide the whole scene, with
    // nothing left on screen to explain why.
    assert_eq!(state.solo, None);
    assert!(state.scene.iter().all(|n| is_visible(n, state.solo)));
}

#[test]
fn closing_another_node_leaves_the_solo_alone() {
    let mut state = shared_shoot(3);
    let (first, second) = (state.scene[0].id, state.scene[1].id);
    state.toggle_solo(second);

    state.close_node(first).expect("nothing is running");
    assert_eq!(state.solo, Some(second));

    state.close_all().expect("nothing is running");
    assert_eq!(state.solo, None);
}

#[test]
fn opening_a_file_ends_a_solo_so_the_new_node_is_not_born_hidden() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    state.toggle_solo(first);

    state.append_node(file_node("/runs/new.sfmr", 8, "IMG"));

    assert_eq!(state.solo, None);
}

#[test]
fn stepping_carries_an_active_solo_to_the_node_it_lands_on() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    // No solo, no solo: stepping never starts one.
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.solo, None);

    state.toggle_solo(ids[1]);
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[2]));
    assert_eq!(
        state.solo,
        Some(ids[2]),
        "the solo stayed behind on the node we stepped away from, so `]` \
         appeared to do nothing at all"
    );
}

// ── Transforms (no frame needed) ────────────────────────────────────────

#[test]
fn every_node_loads_in_its_own_frame() {
    let state = misaligned_pair();
    assert!(state.scene.iter().all(|n| !n.has_transform()));
}

#[test]
fn aligning_puts_the_source_node_where_the_target_is_drawn() {
    for source in [AlignSource::Cameras, AlignSource::Points] {
        let mut state = misaligned_pair();
        let (a, b) = (state.scene[0].id, state.scene[1].id);

        state.align_node(
            b,
            a,
            AlignOptions {
                source,
                estimate_scale: true,
            },
        );

        let error =
            worst_display_error(&state.scene[1], &state.scene[0], &Se3Transform::identity());
        assert!(
            error < 1e-9,
            "{source:?}: worst displayed error {error} after aligning run_b onto run_a"
        );
        assert!(state.scene[1].has_transform());
        // The target is never modified.
        assert!(!state.scene[0].has_transform());
    }
}

#[test]
fn an_align_lands_in_the_targets_currently_displayed_frame() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    // Pretend run_a was itself aligned onto something earlier, so it is already
    // displayed somewhere other than its own coordinates.
    let target_frame = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), 1.1).unwrap(),
        Vector3::new(-7.0, 3.0, 0.5),
        0.25,
    );
    *state.scene[0].history.transform_mut() = target_frame.clone();

    state.align_node(b, a, AlignOptions::default());

    // `source.transform = target.transform ∘ T_fit`: run_b lands on top of run_a
    // *as drawn*, so aligning C→B after B→A chains as expected.
    let error = worst_display_error(&state.scene[1], &state.scene[0], &target_frame);
    assert!(error < 1e-9, "worst displayed error {error} after chaining");
}

#[test]
fn an_aligned_nodes_cameras_are_looked_through_where_they_are_drawn() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.align_node(b, a, AlignOptions::default());

    // The pose `enter_camera_view` builds its end state from. run_b's camera 3,
    // through run_b's transform, has to land on run_a's camera 3 — same photo,
    // two solves, one viewpoint — or "look through this camera" would show the
    // transformed scene from an untransformed viewpoint.
    let (rotation, centre) = crate::viewer_3d::transformed_pose(
        &state.scene[1].recon().image_table.images[3],
        state.scene[1].transform(),
    );
    let expected = &state.scene[0].recon().image_table.images[3];
    assert!(
        (centre - expected.camera_center()).norm() < 1e-9,
        "camera centre {centre:?} is not the target's {:?}",
        expected.camera_center(),
    );
    assert!(
        rotation.angle_to(&expected.quaternion_wxyz) < 1e-9,
        "the composed orientation is off by {} rad",
        rotation.angle_to(&expected.quaternion_wxyz),
    );
}

#[test]
fn the_status_message_reports_the_fit() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);

    state.align_node(b, a, AlignOptions::default());

    let status = state.status_message().expect("a status message");
    assert!(
        status.starts_with("Aligned run_b → run_a: ") && status.contains(" cameras, RMS "),
        "unexpected status line: {status}"
    );
}

#[test]
fn a_failed_align_leaves_the_transform_alone_and_says_why() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    for (i, image) in state.scene[1]
        .recon_mut()
        .image_table
        .images
        .iter_mut()
        .enumerate()
    {
        image.name = format!("unrelated_{i:03}.jpg");
    }

    state.align_node(b, a, AlignOptions::default());

    assert!(
        !state.scene[1].has_transform(),
        "a failed fit still moved the node"
    );
    let status = state.status_message().expect("a status message");
    assert!(
        status.starts_with("Align run_b → run_a failed: "),
        "unexpected status line: {status}"
    );
}

#[test]
fn aligning_a_node_to_itself_does_nothing() {
    let mut state = misaligned_pair();
    let b = state.scene[1].id;
    state.align_node(b, b, AlignOptions::default());
    assert!(!state.scene[1].has_transform());
}

#[test]
fn resetting_a_transform_returns_the_node_to_its_own_frame() {
    let mut state = misaligned_pair();
    let (a, b) = (state.scene[0].id, state.scene[1].id);
    state.align_node(b, a, AlignOptions::default());
    assert!(state.scene[1].has_transform());

    state
        .reset_node_transform(b)
        .expect("a node with a transform");

    assert!(!state.scene[1].has_transform());
    assert_eq!(state.scene[1].transform().scale, 1.0);
    // Both are versions of the node's framing, so the reset steps back to
    // the fit and the fit steps back to the node's own frame.
    state.undo(b).expect("undo the reset");
    assert!(state.scene[1].has_transform());
    state.undo(b).expect("undo the fit");
    assert!(!state.scene[1].has_transform());
    assert!(!state.scene[1].is_dirty());
}

// ── Node lifecycle (no frame needed) ────────────────────────────────────

#[test]
fn labels_are_disambiguated_when_two_files_share_a_stem() {
    let mut state = AppState::new();
    state.append_node(file_node("/a/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/b/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/c/run.sfmr", 8, "IMG"));
    state.append_node(file_node("/d/other.sfmr", 8, "IMG"));

    let labels: Vec<_> = state.scene.iter().map(|n| n.label.as_str()).collect();
    assert_eq!(labels, ["run", "run (2)", "run (3)", "other"]);
}

#[test]
fn closing_a_node_purges_its_caches_and_selection() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    for id in [first, second] {
        state.sift_cache.insert(
            ImageRef::new(id, 0),
            CachedSiftFeatures {
                positions_xy: vec![[0.0, 0.0]],
                affine_shapes: vec![[[1.0, 0.0], [0.0, 1.0]]],
                read_count: 1,
            },
        );
        state.full_res_cache.insert(ImageRef::new(id, 0), None);
    }
    state.select_image(Some(ImageRef::new(first, 3)));
    state.hovered_point = Some(PointRef::new(first, 1));

    state.close_node(first).expect("nothing is running");

    assert_eq!(state.scene.len(), 1);
    assert!(
        state.sift_cache.keys().all(|k| k.recon == second),
        "the SIFT cache kept entries for the closed node"
    );
    assert!(
        state.full_res_cache.keys().all(|k| k.recon == second),
        "the full-res cache kept entries for the closed node"
    );
    assert_eq!(state.selected_image, None, "selection outlived its node");
    assert_eq!(state.hovered_point, None, "hover outlived its node");
    // The other node's entries are untouched.
    assert!(state.sift_cache.contains_key(&ImageRef::new(second, 0)));
}

#[test]
fn closing_the_selected_node_falls_back_to_the_first_remaining() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[1]);

    state.close_node(ids[1]).expect("nothing is running");
    assert_eq!(state.selected_recon, Some(ids[0]));

    // Closing an unselected node leaves the selection alone.
    state.close_node(ids[2]).expect("nothing is running");
    assert_eq!(state.selected_recon, Some(ids[0]));

    // An empty scene means no selection at all.
    state.close_node(ids[0]).expect("nothing is running");
    assert_eq!(state.selected_recon, None);
    assert!(state.scene.is_empty());
}

#[test]
fn close_all_empties_the_scene_and_every_shared_cache() {
    let mut state = shared_shoot(2);
    let id = state.scene[0].id;
    state.full_res_cache.insert(ImageRef::new(id, 0), None);
    state.select_image(Some(ImageRef::new(id, 0)));

    state.close_all().expect("nothing is running");
    assert!(state.scene.is_empty());
    assert_eq!(state.selected_recon, None);
    assert_eq!(state.selected_image, None);
    assert!(state.full_res_cache.is_empty());
}

#[test]
fn selecting_a_reconstruction_clears_finer_selection_from_other_nodes() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    state.select_image(Some(ImageRef::new(first, 2)));
    state.selected_point = Some(PointRef::new(first, 5));
    // Hover is exempt — it is transient and may touch any visible node.
    state.hovered_image = Some(ImageRef::new(first, 9));

    state.select_recon(second);
    assert_eq!(state.selected_recon, Some(second));
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_point, None);
    assert_eq!(
        state.hovered_image,
        Some(ImageRef::new(first, 9)),
        "hover was cleared, but it is exempt from the invariant"
    );

    // Re-selecting the node the selection already lives in changes nothing.
    state.select_image(Some(ImageRef::new(second, 1)));
    state.select_recon(second);
    assert_eq!(state.selected_image, Some(ImageRef::new(second, 1)));
}

#[test]
fn selecting_an_image_or_point_selects_its_reconstruction_too() {
    let mut state = shared_shoot(2);
    let first = state.scene[0].id;
    let second = state.scene[1].id;
    state.select_recon(first);

    state.select_image(Some(ImageRef::new(second, 4)));
    assert_eq!(state.selected_recon, Some(second));

    state.select_point(PointRef::new(first, 4));
    assert_eq!(state.selected_recon, Some(first));
    assert_eq!(
        state.selected_image, None,
        "the image selection belonged to the node we just left"
    );
}

#[test]
fn opening_a_file_appends_a_node_rather_than_replacing_the_scene() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 8, "IMG"));
    state.append_node(file_node("/runs/b.sfmr", 8, "IMG"));
    assert_eq!(state.scene.len(), 2);
    assert_eq!(state.selected_recon, Some(state.scene[1].id));
}

// ── `[` / `]` stepping ──────────────────────────────────────────────────

/// Deliver one `[` or `]` press to `handle_recon_step`, the way `dock.rs` does.
fn step(viewer: &mut Viewer3D, ctx: &egui::Context, state: &mut AppState, forward: bool) {
    let key = if forward {
        egui::Key::CloseBracket
    } else {
        egui::Key::OpenBracket
    };
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }],
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, input, |ui| viewer.handle_recon_step(ui, state));
}

#[test]
fn bracket_keys_step_the_selected_reconstruction_in_tree_order() {
    let mut state = shared_shoot(3);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[1]));
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[2]));
    // Wraps.
    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(ids[0]));
    step(&mut viewer, &ctx, &mut state, false);
    assert_eq!(state.selected_recon, Some(ids[2]));
}

#[test]
fn stepping_carries_the_selection_to_the_same_named_image() {
    let mut state = shared_shoot(2);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    // Shift the second node's images by one, so the same name sits at a
    // different index — an index-based carry-over would land on the wrong photo.
    state.scene[1].recon_mut().image_table.images.rotate_left(1);
    let name = state.scene[0].recon().image_table.images[3].name.clone();
    let expected = state.scene[1]
        .recon()
        .image_table
        .images
        .iter()
        .position(|i| i.name == name)
        .expect("the name is present in the second node");
    assert_ne!(
        expected, 3,
        "the fixture did not actually shift the indices"
    );

    state.select_image(Some(ImageRef::new(ids[0], 3)));
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);

    assert_eq!(state.selected_recon, Some(ids[1]));
    assert_eq!(state.selected_image, Some(ImageRef::new(ids[1], expected)));
}

#[test]
fn stepping_clears_the_finer_selection_when_the_name_is_absent() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 16, "LEFT"));
    state.append_node(file_node("/runs/b.sfmr", 16, "RIGHT"));
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[0]);
    state.select_image(Some(ImageRef::new(ids[0], 2)));
    state.selected_point = Some(PointRef::new(ids[0], 5));

    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);

    assert_eq!(state.selected_recon, Some(ids[1]));
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_point, None);
}

#[test]
fn stepping_carries_the_camera_view_to_the_same_named_image() {
    let mut state = shared_shoot(2);
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    // Enter camera view on the first node's image 3, without the animation:
    // `switch_camera_view` needs an active view, so seed one directly.
    let image = ImageRef::new(ids[0], 3);
    viewer.camera_view = Some(crate::viewer_3d::CameraViewMode {
        image,
        r_world_from_cam: state.scene[0].recon().image_table.images[3]
            .quaternion_wxyz
            .inverse(),
    });
    state.select_image(Some(image));

    step(&mut viewer, &ctx, &mut state, true);
    let now = viewer.camera_view.as_ref().expect("still in camera view");
    assert_eq!(now.image.recon, ids[1], "the camera view stayed behind");
    assert_eq!(
        state.scene[1].recon().image_table.images[now.image.index()].name,
        state.scene[0].recon().image_table.images[3].name,
    );
}

#[test]
fn stepping_drops_a_camera_view_whose_image_has_no_counterpart() {
    let mut state = AppState::new();
    state.append_node(file_node("/runs/a.sfmr", 16, "LEFT"));
    state.append_node(file_node("/runs/b.sfmr", 16, "RIGHT"));
    let ids: Vec<_> = state.scene.iter().map(|n| n.id).collect();
    state.select_recon(ids[0]);
    let image = ImageRef::new(ids[0], 1);
    state.select_image(Some(image));
    let mut viewer = Viewer3D::new();
    viewer.camera_view = Some(crate::viewer_3d::CameraViewMode {
        image,
        r_world_from_cam: state.scene[0].recon().image_table.images[1]
            .quaternion_wxyz
            .inverse(),
    });

    let ctx = egui::Context::default();
    step(&mut viewer, &ctx, &mut state, true);
    assert!(
        viewer.camera_view.is_none(),
        "the camera view kept pointing into the node we stepped away from"
    );
}

#[test]
fn stepping_does_nothing_with_a_single_node_loaded() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    state.select_image(Some(ImageRef::new(id, 2)));
    let mut viewer = Viewer3D::new();
    let ctx = egui::Context::default();

    step(&mut viewer, &ctx, &mut state, true);
    assert_eq!(state.selected_recon, Some(id));
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 2)),
        "stepping to the only node cleared the selection"
    );
}

// ── Window title and stats ──────────────────────────────────────────────

#[test]
fn the_window_title_names_the_first_file_and_counts_the_rest() {
    let mut state = AppState::new();
    assert_eq!(state.window_title(), "SfM Explorer");

    state.append_node(file_node("/runs/run_a.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer - run_a.sfmr");

    state.append_node(file_node("/runs/run_b.sfmr", 8, "IMG"));
    state.append_node(file_node("/runs/run_c.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer - run_a.sfmr (+2)");
}

#[test]
fn demo_data_first_leaves_the_base_window_title_alone() {
    // `ui_basic` attaches to the window by this exact name on Windows.
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(8)));
    assert_eq!(state.window_title(), "SfM Explorer");
    state.append_node(file_node("/runs/run_a.sfmr", 8, "IMG"));
    assert_eq!(state.window_title(), "SfM Explorer");
}

#[test]
fn the_stats_overlay_sums_visible_nodes_and_leads_with_the_count() {
    use crate::viewer_3d::overlay::scene_stats_text;

    let mut state = shared_shoot(2);
    state.scene[0].recon_mut().metadata.infinity_point_count = 3;
    let one_node_points = state.scene[0].recon().point_set.points.len();
    let one_node_images = state.scene[0].recon().image_table.images.len();

    let text = scene_stats_text(&state.scene, None, false, 60.0);
    assert_eq!(
        text,
        format!(
            "2 reconstructions | {} points (3 at infinity) | {} images",
            2 * one_node_points,
            2 * one_node_images
        )
    );

    // Hiding a node takes it out of the totals — and with one left, the
    // reconstruction count leads no more.
    state.scene[1].visible = false;
    let text = scene_stats_text(&state.scene, None, false, 60.0);
    assert_eq!(
        text,
        format!("{one_node_points} points (3 at infinity) | {one_node_images} images")
    );
}

/// Solo reaches the totals the same way an eye does — the overlay describes
/// what is on screen, and the two switches are composed by one rule.
#[test]
fn the_stats_overlay_counts_only_the_soloed_node() {
    use crate::viewer_3d::overlay::scene_stats_text;

    let mut state = shared_shoot(3);
    let second = state.scene[1].id;
    let points = state.scene[1].recon().point_set.points.len();
    let images = state.scene[1].recon().image_table.images.len();

    state.toggle_solo(second);
    assert_eq!(
        scene_stats_text(&state.scene, state.solo, false, 60.0),
        format!("{points} points | {images} images"),
        "the stats line still counted the nodes the solo is hiding"
    );

    // A soloed node whose own eye is off is drawn by nobody.
    state.scene[1].visible = false;
    assert_eq!(
        scene_stats_text(&state.scene, state.solo, false, 60.0),
        "0 points | 0 images"
    );
}

#[test]
fn the_hover_overlay_names_the_reconstruction_only_when_several_are_loaded() {
    use crate::scene_renderer::PickTarget;
    use crate::viewer_3d::overlay::hover_overlay_text;

    let mut state = shared_shoot(1);
    let first = state.scene[0].id;
    let name = state.scene[0].recon().image_table.images[1].name.clone();

    let image_pick = Some(PickTarget::Image(ImageRef::new(first, 1)));
    let point_pick = Some(PickTarget::Point(PointRef::new(first, 88)));
    assert_eq!(
        hover_overlay_text(&state.scene, image_pick, None),
        format!("Camera: {name}")
    );
    assert_eq!(
        hover_overlay_text(&state.scene, point_pick, None),
        "Point3D #88"
    );

    state.append_node(file_node("/runs/run_b.sfmr", 8, "IMG"));
    assert_eq!(
        hover_overlay_text(&state.scene, image_pick, None),
        format!("Camera: run_0 / {name}")
    );
    assert_eq!(
        hover_overlay_text(&state.scene, point_pick, None),
        "Point3D run_0 #88"
    );
}

// ── The Camera Images group and the reconstruction row counts ──────────

#[test]
fn the_image_group_is_labelled_camera_images_and_counts_the_images() {
    // The rename is a bug fix: the row counts `recon.images.len()`, which is
    // images and not the intrinsics they share — eight against two here.
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let texts = painted_at_width(&mut panel, &ctx, &mut state, VIEWPORT.x);
    assert!(
        texts.iter().any(|t| t == "Camera Images (8)"),
        "no Camera Images group row was painted; got {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("Cameras (")),
        "the old Cameras label is still being painted: {texts:?}"
    );
}

#[test]
fn the_reconstruction_row_counts_points_images_and_cameras() {
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    assert_eq!(
        counts_at_width(&mut panel, &ctx, &mut state, 400.0).as_deref(),
        Some("32 pts · 8 imgs · 2 cams")
    );
}

#[test]
fn the_counts_drop_cameras_then_images_as_the_panel_narrows() {
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let full = "32 pts · 8 imgs · 2 cams".to_string();
    let two = "32 pts · 8 imgs".to_string();
    let one = "32 pts".to_string();

    // Swept rather than asserted at three chosen widths: where each count drops
    // out depends on the font, but the order they drop in does not.
    let mut seen: Vec<String> = Vec::new();
    let mut width = 400.0;
    while let Some(counts) = counts_at_width(&mut panel, &ctx, &mut state, width) {
        if seen.last() != Some(&counts) {
            seen.push(counts);
        }
        width -= 8.0;
    }
    assert_eq!(
        seen,
        vec![full.clone(), two, one],
        "the counts did not elide cameras, then images, exactly once each"
    );

    // And back: the elision is on available width, not a state the row keeps.
    assert_eq!(
        counts_at_width(&mut panel, &ctx, &mut state, 400.0),
        Some(full)
    );
}

// ── The Camera Intrinsics group ─────────────────────────────────────────

#[test]
fn the_intrinsics_group_labels_and_counts_the_cameras_and_lists_them() {
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let texts = painted_at_width(&mut panel, &ctx, &mut state, VIEWPORT.x);
    assert!(
        texts.iter().any(|t| t == "Camera Intrinsics (2)"),
        "no Camera Intrinsics group row was painted; got {texts:?}"
    );
    // Two cameras, so the group is open by itself and the rows are there to
    // read: index, model, size, focal length, and how many images use it.
    assert!(
        texts
            .iter()
            .any(|t| t == "#0  PINHOLE  1920×1080  f 1000.0  4 images"),
        "the first camera row is not what the spec draws; got {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t == "#1  PINHOLE  1920×1080  f 1000.0  4 images"),
        "the second camera row is missing; got {texts:?}"
    );
}

/// No eye on the group row: intrinsics have no geometry of their own to hide,
/// so the column is left blank rather than filled with a glyph that would
/// answer nothing.
#[test]
fn the_intrinsics_group_row_carries_no_eye() {
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let texts = painted_at_width(&mut panel, &ctx, &mut state, VIEWPORT.x);
    let eyes = texts
        .iter()
        .filter(|t| t.as_str() == super::EYE_GLYPH)
        .count();
    assert_eq!(
        eyes, 3,
        "expected exactly the node, Camera Images and Points eyes; got {texts:?}"
    );
}

#[test]
fn clicking_a_camera_row_reports_the_camera_it_names() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "intrinsics_1"));
    assert_eq!(response.select_camera, Some(CameraRef::new(id, 1)));
    // A camera click is not an image click: the row denotes a set of them.
    assert_eq!(response.select_image, None);
    assert_eq!(response.zoom_to_camera, None);
}

/// Double-click frames every image taken through that camera — the tree's
/// third double-click target, and consistent with the other two: it frames
/// what the row denotes.
#[test]
fn double_clicking_a_camera_row_asks_to_frame_its_images() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let pos = panel
        .hit_rect(row_id(id, "intrinsics_1"))
        .expect("the camera row")
        .center();
    click_at(&mut panel, &ctx, &mut state, pos);
    let response = click_at(&mut panel, &ctx, &mut state, pos);
    assert_eq!(response.zoom_to_camera, Some(CameraRef::new(id, 1)));
}

#[test]
fn the_intrinsics_group_is_open_at_a_handful_of_cameras_and_collapsed_beyond() {
    let mut few = AppState::new();
    let few_id = few.append_node(camera_count_node("/runs/rig.sfmr", 2));
    let (panel, _ctx) = settled(&mut few);
    assert!(
        panel.hit_rect(row_id(few_id, "intrinsics_0")).is_some(),
        "two cameras should list themselves rather than hide behind a triangle"
    );

    let mut many = AppState::new();
    let many_id = many.append_node(camera_count_node("/runs/solve.sfmr", 5));
    let (panel, _ctx) = settled(&mut many);
    assert!(
        panel.hit_rect(row_id(many_id, "intrinsics_0")).is_none(),
        "past a handful the group should stay out of the way like the image list"
    );
}

#[test]
fn a_camera_no_image_references_reads_zero_images() {
    let mut state = AppState::new();
    state.append_node(unused_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let texts = painted_at_width(&mut panel, &ctx, &mut state, VIEWPORT.x);
    assert!(
        texts.iter().any(|t| t.ends_with("0 images")),
        "an unreferenced camera should say so rather than be hidden; got {texts:?}"
    );
}

/// Hovering a camera row shows a tooltip and nothing else: cross-panel hover is
/// a two-field protocol, and a third field would have to be threaded through
/// every panel to preview a selection that is one click away.
#[test]
fn hovering_a_camera_row_raises_no_cross_panel_hover() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));
    let (mut panel, ctx) = settled(&mut state);

    let pos = panel
        .hit_rect(row_id(id, "intrinsics_1"))
        .expect("the camera row")
        .center();
    run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    let response = run_frame(
        &mut panel,
        &ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    assert!(response.has_pointer, "the panel did not claim the pointer");
    assert_eq!(response.hovered_image, None);
    assert_eq!(response.hovered_point, None);
}

// ── The cross-panel sibling highlight (no frame needed) ─────────────────

#[test]
fn the_sibling_set_is_the_images_that_share_the_selected_camera() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));
    state.select_camera(Some(CameraRef::new(id, 1)));

    let node = &state.scene[0];
    assert_eq!(
        crate::scene::camera_sibling_images(node, state.selected_camera),
        vec![4, 5, 6, 7],
        "the sibling set is not the second camera's images"
    );
}

/// Correct but uninformative: highlighting every image in a single-camera
/// reconstruction says nothing, so it is suppressed.
#[test]
fn the_sibling_highlight_is_suppressed_when_every_image_shares_the_camera() {
    let mut state = AppState::new();
    let id = state.append_node(unused_camera_node("/runs/rig.sfmr"));
    state.select_camera(Some(CameraRef::new(id, 0)));

    let node = &state.scene[0];
    assert!(
        crate::scene::camera_sibling_images(node, state.selected_camera).is_empty(),
        "a whole-node highlight should be suppressed"
    );
    // The unreferenced camera highlights nothing either — and specifically is
    // not caught by the suppression rule, which is about *every* image.
    assert!(crate::scene::camera_sibling_images(node, Some(CameraRef::new(id, 1))).is_empty());
}

#[test]
fn a_camera_selected_in_another_node_highlights_nothing_here() {
    let mut state = AppState::new();
    let first = state.append_node(two_camera_node("/runs/rig_a.sfmr"));
    state.append_node(two_camera_node("/runs/rig_b.sfmr"));

    let other = &state.scene[1];
    assert!(
        crate::scene::camera_sibling_images(other, Some(CameraRef::new(first, 1))).is_empty(),
        "a ref into another node must not index into this one"
    );
}

/// What a double-click frames: that camera's images, where they are *drawn*.
#[test]
fn the_camera_zoom_frames_only_its_own_images_through_the_node_transform() {
    let mut state = AppState::new();
    state.append_node(two_camera_node("/runs/rig.sfmr"));
    let transform = known_similarity();
    *state.scene[0].history.transform_mut() = transform.clone();

    let node = &state.scene[0];
    let centres = crate::scene::camera_world_centres(node, 1);
    assert_eq!(
        centres.len(),
        4,
        "framed the whole node rather than one lens"
    );
    let expected = transform.apply_to_point(&node.recon().image_table.images[4].camera_center());
    assert!((centres[0] - expected).norm() < 1e-9);

    // A camera nothing uses frames nothing, rather than a degenerate point.
    let mut unused = AppState::new();
    unused.append_node(unused_camera_node("/runs/rig.sfmr"));
    assert!(crate::scene::camera_world_centres(&unused.scene[0], 1).is_empty());
}

// ── The image / camera selection coupling (no frame needed) ────────────
//
// The truth table in `specs/gui/camera-intrinsics.md` § "The selection
// coupling", one test per row. All of it is `AppState`: the invariant is
// enforced in the two setters precisely so that no panel has to remember it.

#[test]
fn selecting_an_image_selects_the_camera_it_was_taken_through() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 5)));
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 1)));

    state.select_image(Some(ImageRef::new(id, 2)));
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 0)));
}

#[test]
fn reselecting_the_image_already_selected_changes_nothing() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    state.select_image(Some(ImageRef::new(id, 5)));
    assert_eq!(state.selected_image, Some(ImageRef::new(id, 5)));
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 1)));
}

#[test]
fn a_camera_can_be_selected_with_no_image_selected() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_camera(Some(CameraRef::new(id, 1)));
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 1)));
    assert_eq!(state.selected_image, None);
}

#[test]
fn selecting_the_camera_the_selected_image_uses_keeps_the_image() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    state.select_camera(Some(CameraRef::new(id, 1)));
    assert_eq!(
        state.selected_image,
        Some(ImageRef::new(id, 5)),
        "clicking the intrinsics of the image on screen threw the image away"
    );
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 1)));
}

#[test]
fn selecting_a_different_camera_clears_the_image() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    state.select_camera(Some(CameraRef::new(id, 0)));
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_camera, Some(CameraRef::new(id, 0)));
}

#[test]
fn clearing_the_image_keeps_the_camera_and_a_second_clear_takes_it() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    state.select_image(None);
    assert_eq!(state.selected_image, None);
    assert_eq!(
        state.selected_camera,
        Some(CameraRef::new(id, 1)),
        "dismissing the photograph also dismissed the lens"
    );

    // Esc a second time, finding no image, clears the camera.
    state.select_camera(None);
    assert_eq!(state.selected_camera, None);
}

#[test]
fn clearing_the_camera_out_from_under_a_selected_image_clears_the_image_too() {
    let mut state = AppState::new();
    let id = state.append_node(two_camera_node("/runs/rig.sfmr"));

    state.select_image(Some(ImageRef::new(id, 5)));
    state.select_camera(None);

    // Not a path any UI takes today — the Esc sequence clears the image first.
    // The invariant is that no caller *can* reach the forbidden state, though,
    // not that none currently tries, so the setter has to close it here rather
    // than trust the order it is called in.
    assert_eq!(state.selected_camera, None);
    assert_eq!(
        state.selected_image, None,
        "an image outlived the camera it was taken through"
    );
}

#[test]
fn selecting_another_reconstruction_filters_both_selections() {
    let mut state = AppState::new();
    let first = state.append_node(two_camera_node("/runs/rig_a.sfmr"));
    let second = state.append_node(two_camera_node("/runs/rig_b.sfmr"));

    state.select_image(Some(ImageRef::new(first, 5)));
    state.select_recon(second);
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_camera, None);

    // A selection inside the node being selected survives it.
    state.select_image(Some(ImageRef::new(second, 5)));
    state.select_recon(second);
    assert_eq!(state.selected_image, Some(ImageRef::new(second, 5)));
    assert_eq!(state.selected_camera, Some(CameraRef::new(second, 1)));
}

/// Closing a node and replacing one unwind its selections through the same
/// `forget_recon`, so the camera goes the way the image already did.
#[test]
fn closing_the_owning_node_clears_both_selections() {
    let mut state = AppState::new();
    let first = state.append_node(two_camera_node("/runs/rig_a.sfmr"));
    state.append_node(two_camera_node("/runs/rig_b.sfmr"));

    state.select_image(Some(ImageRef::new(first, 5)));
    state.close_node(first).expect("nothing is running");
    assert_eq!(state.selected_image, None);
    assert_eq!(state.selected_camera, None);

    // And the whole-scene path.
    let second = state.scene[0].id;
    state.select_camera(Some(CameraRef::new(second, 1)));
    state.close_all().expect("nothing is running");
    assert_eq!(state.selected_camera, None);
}

// ── Resect Image ────────────────────────────────────────────────────────

/// A node a resection can actually run on: the demo camera ring, with every
/// observation recomputed as an inline keypoint from the node's own geometry.
///
/// [`recon_named`] clones one pose for every image and `demo` keeps its
/// observations in `.sift` companions that no test has on disk — neither of
/// which a resection can work from. Here every point is observed by every
/// camera that can see it, at the pixel that camera actually projects it to.
pub(crate) fn resectable_node(path: &str) -> SceneNode {
    use sfmtool_core::reconstruction::{ObservationSource, TrackObservation};

    let mut recon = SfmrReconstruction::demo(120);
    let camera = recon.image_table.cameras[0].clone();
    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for (p, point) in recon.point_set.points.iter().enumerate() {
        let mut count = 0u32;
        for (i, image) in recon.image_table.images.iter().enumerate() {
            let local = image.quaternion_wxyz * point.position.coords + image.translation_xyz;
            let Some((u, v)) = camera.ray_to_pixel([local.x, local.y, local.z]) else {
                continue;
            };
            if u < 0.0 || v < 0.0 || u >= camera.width as f64 || v >= camera.height as f64 {
                continue;
            }
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push([u as f32, v as f32]);
            count += 1;
        }
        counts.push(count);
    }
    let images = recon.image_table.images.len();
    let mut keypoints_xy = ndarray::Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; images],
    };
    recon.metadata.feature_source = "embedded_patches".to_string();
    recon.point_set.tracks = tracks;
    recon.point_set.observation_counts = counts;
    recon.rebuild_derived_fields();
    SceneNode::from_path(std::path::Path::new(path), recon)
}

/// [`resectable_node`] with a square patch frame per point, which is what an
/// observation's footprint is read off.
///
/// The extent is a twentieth of a world unit against a demo scene a few units
/// across, so a point's frame projects to a handful of pixels: small enough
/// that the rule has something to compare and large enough that it is not a
/// collapsed measurement.
pub(crate) fn prunable_node(path: &str) -> SceneNode {
    let mut node = resectable_node(path);
    let recon = node.recon_mut();
    let n = recon.point_set.points.len();
    let mut u = ndarray::Array2::<f32>::zeros((n, 3));
    let mut v = ndarray::Array2::<f32>::zeros((n, 3));
    for p in 0..n {
        // Every other point an octave wider, so half the population has
        // something finer to be covered by.
        let half = if p % 2 == 0 { 0.05 } else { 0.0125 };
        u[[p, 0]] = half;
        v[[p, 1]] = half;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    node
}

/// A state holding one resectable node, with its image list expanded and the
/// panel settled, ready to be right-clicked.
fn resectable_scene() -> AppState {
    let mut state = AppState::new();
    state.append_node(resectable_node("/runs/run_a.sfmr"));
    state
}

/// Expand a node's Camera Images group and settle the panel on it.
fn with_image_list(
    state: &mut AppState,
) -> (SceneGraphPanel, egui::Context, crate::scene::ReconId) {
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(state);
    set_open(&ctx, row_id(id, "camera_images"), true);
    run_frame(&mut panel, &ctx, state, Vec::new());
    run_frame(&mut panel, &ctx, state, Vec::new());
    (panel, ctx, id)
}

#[test]
fn the_resect_entries_are_on_image_rows_and_not_on_the_reconstruction_row() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_0"));
    assert!(
        panel.hit_rect(row_id(id, "resect_0")).is_some(),
        "the image row's menu offered no Resect Image"
    );
    assert!(
        panel.hit_rect(row_id(id, "resect_matches_0")).is_some(),
        "the image row's menu offered no matches variant"
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        context_menu_open(&panel, id),
        "the reconstruction row's menu did not open"
    );
    assert!(
        panel.hit_rect(row_id(id, "resect_0")).is_none(),
        "Resect Image leaked onto the reconstruction row's menu"
    );
}

#[test]
fn choosing_resect_image_reports_the_image_and_the_source() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_2"));
    assert_eq!(
        response.resect_image,
        Some((ImageRef::new(id, 2), ResectFrom::Observations))
    );

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_matches_2"));
    assert_eq!(
        response.resect_image,
        Some((ImageRef::new(id, 2), ResectFrom::Matches))
    );
}

#[test]
fn resect_is_greyed_on_a_reconstruction_with_too_few_posed_images() {
    let mut state = AppState::new();
    // Three images: the target plus two others, one short of the floor.
    state.append_node(file_node("/runs/thin.sfmr", 3, "IMG"));
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_0"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_0"));
    assert_eq!(
        response.resect_image, None,
        "the greyed Resect Image still emitted the action"
    );
}

#[test]
fn resect_is_greyed_on_an_image_that_is_not_posed() {
    let mut state = shared_shoot(1);
    state.scene[0].recon_mut().image_table.images[1].translation_xyz =
        Vector3::new(f64::NAN, 0.0, 0.0);
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_1"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_1"));
    assert_eq!(
        response.resect_image, None,
        "an unposed image was resectable"
    );

    // Its posed neighbours are unaffected.
    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_0"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_0"));
    assert!(response.resect_image.is_some());
}

#[test]
fn the_matches_variant_is_greyed_without_feature_indexes() {
    // A resectable node carries embedded patches, which is exactly the case a
    // match row cannot be joined to.
    let mut state = resectable_scene();
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_0"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_matches_0"));
    assert_eq!(response.resect_image, None, "the matches variant was live");

    // The stored-observations entry beside it is unaffected — and the menu is
    // still standing, which is what `CloseOnClickOutside` is there for.
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "resect_0"));
    assert_eq!(
        response.resect_image,
        Some((ImageRef::new(id, 0), ResectFrom::Observations))
    );
}

#[test]
fn closing_a_node_forgets_the_matches_file_chosen_for_it() {
    let mut state = resectable_scene();
    let source = state.scene[0].id;
    state
        .resect_matches
        .insert(source, std::path::PathBuf::from("/runs/run_a.matches"));
    state.close_node(source).expect("nothing is running");
    assert!(state.resect_matches.is_empty());
}

// ── Move Camera ─────────────────────────────────────────────────────────

#[test]
fn the_image_row_offers_move_camera_and_reports_the_image_it_was_chosen_on() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx, id) = with_image_list(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    assert!(
        panel.hit_rect(row_id(id, "move_camera_2")).is_some(),
        "the image row's menu offered no Move Camera"
    );
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "move_camera_2"));
    assert_eq!(response.move_camera, Some(ImageRef::new(id, 2)));

    // It is an image's action, not a reconstruction's: the row above offers
    // alignment and closing, not a camera to take in hand.
    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel.hit_rect(row_id(id, "move_camera_2")).is_none(),
        "Move Camera leaked onto the reconstruction row's menu"
    );
}

// -- Convert to Embedded Patches ------------------------------------------

/// The entry is live on a `sift_files` node and reports the node it was opened
/// on.
#[test]
fn the_convert_entry_is_live_on_a_sift_files_node() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel.hit_rect(row_id(id, "to_embedded_patches")).is_some(),
        "the reconstruction row's menu offered no {}",
        super::menus::CONVERT_TO_EMBEDDED_PATCHES
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "to_embedded_patches"),
    );
    assert_eq!(response.convert_to_embedded_patches, Some(id));
}

/// On a node that already carries embedded patches it is drawn and dead: there
/// is no `.sift` left to copy a keypoint from.
#[test]
fn the_convert_entry_is_greyed_on_an_embedded_patches_node() {
    let mut state = resectable_scene();
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel.hit_rect(row_id(id, "to_embedded_patches")).is_some(),
        "the entry was hidden rather than greyed"
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "to_embedded_patches"),
    );
    assert_eq!(
        response.convert_to_embedded_patches, None,
        "an embedded_patches node offered to be converted again"
    );
}

/// And dead while an operation is running on that node, which is the state's
/// own refusal rather than a second rule.
#[test]
fn the_convert_entry_is_greyed_while_the_node_is_busy() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    // A worker held open, so the node is still busy when the menu is drawn.
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::TO_EMBEDDED_PATCHES,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "to_embedded_patches"),
    );
    assert_eq!(
        response.convert_to_embedded_patches, None,
        "the entry was live on a busy node"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
}

/// `Bundle Adjust...` is live on a node with a pixel per observation and one
/// posed lens, sits directly above `Retriangulate All Points`, and reports the
/// node it was opened on rather than the selection.
#[test]
fn the_bundle_adjust_entry_is_live_on_an_adjustable_node() {
    let mut state = resectable_scene();
    state.append_node(resectable_node("/runs/run_b.sfmr"));
    let selected = state.scene[0].id;
    let id = state.scene[1].id;
    state.selected_recon = Some(selected);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let entry = panel
        .hit_rect(row_id(id, "bundle_adjust"))
        .unwrap_or_else(|| {
            panic!(
                "the reconstruction row's menu offered no {}",
                super::menus::BUNDLE_ADJUST
            )
        });
    let below = panel
        .hit_rect(row_id(id, "retriangulate_all_points"))
        .expect("the retriangulation entry");
    assert!(
        below.top() >= entry.bottom() - 1.0,
        "the bundle adjust entry is not directly above the retriangulation"
    );
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "bundle_adjust"));
    assert_eq!(
        response.bundle_adjust,
        Some(id),
        "the entry reported the selection rather than the row it was opened on"
    );
}

/// On a node whose observations are `.sift` feature indexes it is drawn and
/// dead, on the edit's own gate: there is no pixel to reproject against.
#[test]
fn the_bundle_adjust_entry_is_greyed_without_inline_keypoints() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;
    assert!(crate::bundle_adjust_prompt::refusal(state.scene[0].edited()).is_some());

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel.hit_rect(row_id(id, "bundle_adjust")).is_some(),
        "the entry was hidden rather than greyed"
    );
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "bundle_adjust"));
    assert_eq!(
        response.bundle_adjust, None,
        "a node with no pixel per observation offered to be adjusted"
    );
}

/// And dead while an operation is running on that node.
#[test]
fn the_bundle_adjust_entry_is_greyed_while_the_node_is_busy() {
    let mut state = resectable_scene();
    let id = state.scene[0].id;
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::BUNDLE_ADJUST,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "bundle_adjust"));
    assert_eq!(
        response.bundle_adjust, None,
        "the entry was live on a busy node"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
}

/// `Retriangulate All Points` is live on a node whose observations carry a
/// pixel, and reports the node it was opened on.
#[test]
fn the_retriangulate_entry_is_live_on_a_node_with_keypoints() {
    let mut state = resectable_scene();
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel
            .hit_rect(row_id(id, "retriangulate_all_points"))
            .is_some(),
        "the reconstruction row's menu offered no {}",
        super::menus::RETRIANGULATE_ALL_POINTS
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "retriangulate_all_points"),
    );
    assert_eq!(response.retriangulate_all_points, Some(id));
}

/// On a node whose observations are `.sift` feature indexes it is drawn and
/// dead: there is no pixel to cast a ray through.
#[test]
fn the_retriangulate_entry_is_greyed_without_inline_keypoints() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel
            .hit_rect(row_id(id, "retriangulate_all_points"))
            .is_some(),
        "the entry was hidden rather than greyed"
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "retriangulate_all_points"),
    );
    assert_eq!(
        response.retriangulate_all_points, None,
        "a node with no pixel per observation offered to be retriangulated"
    );
}

/// And dead while an operation is running on that node, which is the state's
/// own refusal rather than a second rule.
#[test]
fn the_retriangulate_entry_is_greyed_while_the_node_is_busy() {
    let mut state = resectable_scene();
    let id = state.scene[0].id;
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::RETRIANGULATE_ALL_POINTS,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "retriangulate_all_points"),
    );
    assert_eq!(
        response.retriangulate_all_points, None,
        "the entry was live on a busy node"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
}

/// `Prune Covered Observations` is live on a node that carries a patch frame
/// per point, sits directly under `Retriangulate All Points`, and reports the
/// node it was opened on.
#[test]
fn the_prune_entry_is_live_on_a_node_with_patch_frames() {
    let mut state = AppState::new();
    state.append_node(prunable_node("/runs/run_a.sfmr"));
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let above = panel
        .hit_rect(row_id(id, "retriangulate_all_points"))
        .expect("the retriangulation entry");
    let entry = panel
        .hit_rect(row_id(id, "prune_covered_observations"))
        .unwrap_or_else(|| {
            panic!(
                "the reconstruction row's menu offered no {}",
                super::menus::PRUNE_COVERED_OBSERVATIONS
            )
        });
    assert!(
        entry.top() >= above.bottom() - 1.0,
        "the prune entry is not directly under the retriangulation"
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "prune_covered_observations"),
    );
    assert_eq!(response.prune_covered_observations, Some(id));
}

/// On a node whose points carry no patch frame it is drawn and dead: there is
/// no footprint to read.
#[test]
fn the_prune_entry_is_greyed_without_patch_frames() {
    let mut state = resectable_scene();
    let (mut panel, ctx) = settled(&mut state);
    let id = state.scene[0].id;

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    assert!(
        panel
            .hit_rect(row_id(id, "prune_covered_observations"))
            .is_some(),
        "the entry was hidden rather than greyed"
    );
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "prune_covered_observations"),
    );
    assert_eq!(
        response.prune_covered_observations, None,
        "a node with no patch frame offered to be pruned"
    );
}

/// And dead while an operation is running on that node, which is the state's
/// own refusal rather than a second rule.
#[test]
fn the_prune_entry_is_greyed_while_the_node_is_busy() {
    let mut state = AppState::new();
    state.append_node(prunable_node("/runs/run_a.sfmr"));
    let id = state.scene[0].id;
    let (open, held) = std::sync::mpsc::channel::<()>();
    state
        .start_background_task(
            crate::background::Operation::PRUNE_COVERED_OBSERVATIONS,
            id,
            Box::new(move |_progress| {
                let _ = held.recv();
                crate::background::Finished::Failed("nothing".to_string())
            }),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let response = click(
        &mut panel,
        &ctx,
        &mut state,
        row_id(id, "prune_covered_observations"),
    );
    assert_eq!(
        response.prune_covered_observations, None,
        "the entry was live on a busy node"
    );

    open.send(()).expect("the worker is waiting");
    state.finish_background_task();
}

// ── The SIFT Index row ──────────────────────────────────────────────────

/// Wide enough that the status text beside the row's name is not clipped away.
const INDEX_ROW_WIDTH: f32 = 600.0;

/// The strings one settled frame painted, at a width the row fits in.
fn index_row_texts(
    panel: &mut SceneGraphPanel,
    ctx: &egui::Context,
    state: &mut AppState,
) -> Vec<String> {
    painted_at_width(panel, ctx, state, INDEX_ROW_WIDTH);
    painted_at_width(panel, ctx, state, INDEX_ROW_WIDTH)
}

/// A node with `.sift` files and a built index, and the state holding it.
fn indexed(dir: &std::path::Path) -> (AppState, crate::scene::ReconId) {
    let (state, id, _) = crate::sift_index::tests::searchable(dir);
    (state, id)
}

#[test]
fn the_sift_index_row_says_none_on_a_node_with_no_index() {
    let mut state = shared_shoot(1);
    let (mut panel, ctx) = settled(&mut state);
    let texts = index_row_texts(&mut panel, &ctx, &mut state);
    assert!(
        texts.iter().any(|t| t == "SIFT Index"),
        "the row is not drawn: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == "none"),
        "the row does not say there is no index: {texts:?}"
    );
}

#[test]
fn the_sift_index_row_counts_the_descriptors_of_a_current_index() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = indexed(dir.path());
    let descriptors = state.sift_index(id).expect("built").feature_count();
    let (mut panel, ctx) = settled(&mut state);
    let texts = index_row_texts(&mut panel, &ctx, &mut state);
    let expected = format!("{descriptors} descriptors");
    assert!(
        texts.contains(&expected),
        "the row does not say {expected:?}: {texts:?}"
    );
}

#[test]
fn the_sift_index_row_says_stale_when_the_index_is() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = indexed(dir.path());
    // One image's features extracted again after the build.
    {
        let recon = state.node(id).expect("loaded").recon();
        crate::sift_index::tests::write_sift(
            &recon.sift_path_for_image(2),
            &recon.image_table.images[2].name,
            &vec![vec![5u8; 128]; 3],
            &[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        );
    }
    let path = state.sift_index(id).expect("built").path.clone();
    state.open_sift_index(id, Some(path)).expect("it opens");

    let (mut panel, ctx) = settled(&mut state);
    let texts = index_row_texts(&mut panel, &ctx, &mut state);
    assert!(
        texts.iter().any(|t| t == "stale"),
        "the row does not say the index is out of date: {texts:?}"
    );
}

/// The row's menu carries the three ways to give a node an index or take one
/// away, and the build reads *Rebuild* once there is one.
#[test]
fn the_sift_index_row_s_menu_offers_the_build_the_open_and_the_close() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = indexed(dir.path());
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "sift_index"));
    let texts = painted_at_width(&mut panel, &ctx, &mut state, INDEX_ROW_WIDTH);
    for entry in [
        super::menus::CLOSE_SIFT_INDEX,
        "Open...",
        super::REBUILD_SIFT_INDEX,
    ] {
        assert!(
            texts.iter().any(|t| t == entry),
            "{entry} is not in the row's menu: {texts:?}"
        );
    }

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "close_sift_index"));
    assert_eq!(response.close_sift_index, Some(id));
}

/// On a node with no index the same entry reads *Build*, and choosing it asks
/// the dock for one.
#[test]
fn the_sift_index_row_s_menu_builds_a_first_index() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = crate::sift_index::tests::state_in(dir.path());
    crate::sift_index::tests::with_sift_files(&state, id, [900.0, 500.0]);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "sift_index"));
    let texts = painted_at_width(&mut panel, &ctx, &mut state, INDEX_ROW_WIDTH);
    assert!(
        texts.iter().any(|t| t == super::BUILD_SIFT_INDEX),
        "the build entry is not in the row's menu: {texts:?}"
    );
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "build_sift_index"));
    assert_eq!(response.build_sift_index, Some(id));
}

/// The reconstruction row's own menu carries the build too, above *Convert to
/// Embedded Patches*: it is where a person looks first.
#[test]
fn the_reconstruction_row_s_menu_offers_the_build_above_the_conversion() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = crate::sift_index::tests::state_in(dir.path());
    crate::sift_index::tests::with_sift_files(&state, id, [900.0, 500.0]);
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "node_label"));
    let build = panel
        .hit_rect(row_id(id, "build_sift_index"))
        .expect("the build entry was not drawn on the reconstruction row's menu");
    let convert = panel
        .hit_rect(row_id(id, "to_embedded_patches"))
        .expect("the conversion entry was not drawn");
    assert!(
        build.min.y < convert.min.y,
        "the build sits below the conversion: {build:?} against {convert:?}"
    );

    let response = click(&mut panel, &ctx, &mut state, row_id(id, "build_sift_index"));
    assert_eq!(response.build_sift_index, Some(id));
}

/// A point over the row's `SIFT Index` label rather than over its status text.
/// The label is drawn from the row's left edge and the status follows it, so
/// the left end is the part a reader aims at and the part that answered nothing
/// while the label sensed its own clicks.
fn on_the_index_label(panel: &SceneGraphPanel, id: crate::scene::ReconId) -> egui::Pos2 {
    let row = panel
        .hit_rect(row_id(id, "sift_index"))
        .expect("the SIFT Index row was not drawn");
    egui::pos2(row.left() + 12.0, row.center().y)
}

/// The menu opens on the row's name, not only on the status text beside it.
#[test]
fn the_sift_index_row_s_menu_opens_on_its_name() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = indexed(dir.path());
    let (mut panel, ctx) = settled(&mut state);

    let on_the_name = on_the_index_label(&panel, id);
    right_click_at(&mut panel, &ctx, &mut state, on_the_name);
    assert!(
        panel.hit_rect(row_id(id, "build_sift_index")).is_some(),
        "right-clicking the row's name opened no menu"
    );
}

/// Hovering the name says which `.kdf` is open and how much is in it, over
/// the name rather than over the status text the label used to leave it to.
#[test]
fn hovering_the_sift_index_row_names_the_file_and_counts_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = indexed(dir.path());
    let index = state.sift_index(id).expect("built");
    let path = index.path.display().to_string();
    let expected = format!(
        "{path}\n{} descriptors of {} images",
        index.feature_count(),
        index.images
    );
    let (mut panel, ctx) = settled(&mut state);

    let on_the_name = on_the_index_label(&panel, id);
    let hovered = hover_texts_at(&mut panel, &ctx, &mut state, on_the_name);
    assert!(
        hovered.contains(&expected),
        "{expected:?} was not the tooltip: {hovered:?}"
    );
}

/// With no index there, the hover says so and then says where a build would
/// put one: a bare path would read as a file that is sitting there.
#[test]
fn hovering_a_node_with_no_index_says_where_a_build_would_write() {
    let mut state = shared_shoot(1);
    let id = state.scene[0].id;
    let path = state
        .sift_index_path(id)
        .expect("a saved node has an index path")
        .display()
        .to_string();
    let (mut panel, ctx) = settled(&mut state);

    let on_the_name = on_the_index_label(&panel, id);
    let hovered = hover_texts_at(&mut panel, &ctx, &mut state, on_the_name);
    let expected = format!("No SIFT index file is there.\nA build writes {path}");
    assert!(
        hovered.contains(&expected),
        "{expected:?} was not the tooltip: {hovered:?}"
    );
}

/// A node with nowhere to put one has no path to name, so the hover is the
/// sentence its menu entries are greyed with.
#[test]
fn hovering_an_unsaved_node_s_sift_index_row_says_to_save_first() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    let on_the_name = on_the_index_label(&panel, id);
    let hovered = hover_texts_at(&mut panel, &ctx, &mut state, on_the_name);
    assert!(
        hovered
            .iter()
            .any(|t| t.starts_with("Save demo first: the SIFT index is written beside")),
        "the hover did not say to save the node first: {hovered:?}"
    );
}

/// A node that has never been saved has nowhere to put an index, and both
/// menus say which thing to do about it rather than offering a build that
/// would fail.
#[test]
fn an_unsaved_node_s_build_entry_is_greyed_with_the_save_first_sentence() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let id = state.scene[0].id;
    let (mut panel, ctx) = settled(&mut state);

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "sift_index"));
    let response = click(&mut panel, &ctx, &mut state, row_id(id, "build_sift_index"));
    assert_eq!(
        response.build_sift_index, None,
        "the entry was live on a node with nowhere to write"
    );
    let why = state
        .build_sift_index_refusal(id)
        .expect("nowhere to write it");
    assert!(why.starts_with("Save demo first"), "{why}");
}
