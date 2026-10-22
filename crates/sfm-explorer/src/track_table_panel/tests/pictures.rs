// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Track Table's Patch column, the *Size* slider and
//! the texture cache the picture cells draw from.

use std::sync::Arc;

use ndarray::Array4;
use sfmtool_core::analysis::track_table::TrackTableColumn;

use super::super::pictures::{
    CellPicture, PictureColumn, PictureKey, PictureTextures, DEFAULT_PICTURE_SIZE,
    MIN_PICTURE_SIZE, PICTURE_PAD, TEXTURE_CAPACITY,
};
use super::super::{text_row_height, TrackTablePanel};
use super::{click, frame, input, painted_at, shown, state, POINTS};
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;
use crate::test_support::{painted_text_rects, painted_texts, run_frame_headless};

/// The point the bitmap fixture stores an all-zero bitmap for, which is how a
/// point with no stored patch is written.
const NO_PATCH: u32 = 3;

/// The fixture of [`state`] with an 8 × 8 RGBA patch bitmap per point, all
/// zero for [`NO_PATCH`].
fn state_with_bitmaps() -> (AppState, ReconId) {
    let (mut state, id) = state();
    let recon = state.scene[0].recon_mut();
    let bitmaps = Array4::from_shape_fn((POINTS, 8, 8, 4), |(p, y, x, c)| {
        if p as u32 == NO_PATCH {
            0
        } else {
            (10 + p + 7 * y + 3 * x + 40 * c) as u8
        }
    });
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    (state, id)
}

/// A window of 800 × 400 points, which shows a few of the fixture's rows at
/// the default picture size.
fn short(ctx: &egui::Context, panel: &mut TrackTablePanel, state: &mut AppState) {
    run_frame_headless(ctx, input(400.0), |ui| panel.show(ui, state));
}

/// A panel drawn in the short window, its table job run, and drawn again.
fn shown_short(state: &mut AppState) -> (egui::Context, TrackTablePanel) {
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    short(&ctx, &mut panel, state);
    state.settle_track_table_columns();
    short(&ctx, &mut panel, state);
    (ctx, panel)
}

/// The points of the rows the last frame drew, in order.
fn drawn_points(panel: &TrackTablePanel) -> Vec<u32> {
    panel.drawn().iter().map(|&(p, _)| p).collect()
}

/// The height of the rows the last frame drew.
fn drawn_row_height(panel: &TrackTablePanel) -> f32 {
    panel.drawn()[0].1.height()
}

/// A reconstruction with no patch bitmap, as a `sift_files` one usually is,
/// has no Patch column and no Size slider, and its rows are a line of text
/// high. One with bitmaps has the Patch column, shown by default and drawn
/// before the value columns, and its rows are the default picture size high.
#[test]
fn the_patch_column_shows_only_for_a_reconstruction_with_patch_bitmaps() {
    let (mut state, _) = state();
    let (ctx, mut panel) = shown(&mut state);
    let texts = painted_texts(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    assert!(!texts.iter().any(|t| t == "Patch"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Size"), "{texts:?}");
    assert!(panel.drawn_pictures.is_empty());
    let text_height = drawn_row_height(&panel);

    let (mut state, _) = state_with_bitmaps();
    let (ctx, mut panel) = shown(&mut state);
    let texts = painted_text_rects(&ctx, input(1200.0), |ui| panel.show(ui, &mut state));
    let left_of = |text: &str| {
        texts
            .iter()
            .find(|t| t.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .left()
    };
    assert!(left_of("Point") < left_of("Patch"));
    assert!(left_of("Patch") < left_of("Track"), "pictures come first");
    assert!(texts.iter().any(|t| t.text == "Size"));
    assert_eq!(
        drawn_row_height(&panel),
        DEFAULT_PICTURE_SIZE + 2.0 * PICTURE_PAD
    );
    assert!(drawn_row_height(&panel) > text_height);
    assert!(panel
        .drawn_pictures
        .iter()
        .any(|&(_, column, picture)| column == PictureColumn::Patch
            && matches!(picture, CellPicture::Texture(_))));

    // Unticked, the rows are text high again.
    panel.query.set_picture_shown(PictureColumn::Patch, false);
    frame(&ctx, &mut panel, &mut state);
    assert!(panel.drawn_pictures.is_empty());
    assert_eq!(drawn_row_height(&panel), text_height);
}

/// Pictures are made only for the rows drawn, once each: a frame that draws
/// the same rows makes none, and a point with an all-zero bitmap draws *no
/// patch*.
#[test]
fn pictures_are_made_only_for_the_visible_rows() {
    let (mut state, id) = state_with_bitmaps();
    let (ctx, mut panel) = shown_short(&mut state);
    let drawn = drawn_points(&panel);
    assert!(
        drawn.len() < POINTS / 2,
        "the window is short enough to leave most rows out: {drawn:?}"
    );
    let document = state.scene[0].history.current_version().document_serial;
    let made: Vec<u32> = panel.pictures.made.iter().map(|k| k.point).collect();
    assert_eq!(made, drawn, "one picture per drawn row and no other");
    assert!(panel
        .pictures
        .made
        .iter()
        .all(|k| k.node == id && k.document == document && k.column == PictureColumn::Patch));

    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    assert_eq!(
        panel.pictures.made.len(),
        drawn.len(),
        "drawing the same rows again makes nothing"
    );
    let no_patch = panel
        .drawn_pictures
        .iter()
        .find(|&&(p, _, _)| p == NO_PATCH)
        .expect("the row with no patch is drawn");
    assert_eq!(no_patch.2, CellPicture::Nothing);
    let texts = painted_texts(&ctx, input(400.0), |ui| panel.show(ui, &mut state));
    assert!(texts.iter().any(|t| t == "no patch"), "{texts:?}");
}

/// The texture cache is bounded by entry count: once it holds more than its
/// capacity it releases the pictures drawn longest ago, down to three
/// quarters of it, and never one the frame drew.
#[test]
fn the_texture_cache_releases_the_pictures_drawn_longest_ago() {
    let (state, id) = state();
    let document = state.scene[0].history.current_version().document_serial;
    let key = |point: u32| PictureKey {
        column: PictureColumn::Patch,
        node: id,
        point,
        document,
    };
    let image = || Some(egui::ColorImage::filled([2, 2], egui::Color32::GRAY));
    let ctx = egui::Context::default();
    let mut textures = PictureTextures::default();
    // One new picture per frame, and point 0 drawn on every frame.
    for point in 0..=TEXTURE_CAPACITY as u32 {
        run_frame_headless(&ctx, input(100.0), |ui| {
            textures.begin_frame();
            textures.picture(ui.ctx(), key(0), Some(&image));
            textures.picture(ui.ctx(), key(point), Some(&image));
            textures.end_frame();
        });
        assert!(textures.len() <= TEXTURE_CAPACITY, "frame {point}");
    }
    assert_eq!(textures.len(), TEXTURE_CAPACITY * 3 / 4);
    assert!(textures.holds(&key(0)), "a picture this frame drew is kept");
    assert!(textures.holds(&key(TEXTURE_CAPACITY as u32)));
    assert!(!textures.holds(&key(1)), "the longest unseen is released");

    // A cell with no picture held and nothing to make it from waits.
    let mut waiting = None;
    run_frame_headless(&ctx, input(100.0), |ui| {
        textures.begin_frame();
        waiting = Some(textures.picture(ui.ctx(), key(1), None));
        textures.end_frame();
    });
    assert_eq!(waiting, Some(CellPicture::Waiting));
    assert!(!textures.holds(&key(1)));
}

/// The Size slider sets the row height, and the rows at the top stay there
/// when it changes.
#[test]
fn the_size_slider_sets_the_row_height_and_keeps_the_top_row() {
    let (mut state, id) = state_with_bitmaps();
    let (ctx, mut panel) = shown_short(&mut state);
    state.select_point(PointRef {
        recon: id,
        point: 30,
    });
    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    let top = drawn_points(&panel)[0];
    assert!(top > 20, "scrolled down to the selection: {top}");

    for size in [MIN_PICTURE_SIZE, 128.0, DEFAULT_PICTURE_SIZE] {
        panel.picture_size.0 = size;
        short(&ctx, &mut panel, &mut state);
        assert_eq!(drawn_row_height(&panel), size + 2.0 * PICTURE_PAD);
        assert_eq!(drawn_points(&panel)[0], top, "size {size}");
        short(&ctx, &mut panel, &mut state);
        assert_eq!(drawn_points(&panel)[0], top, "size {size}, settled");
    }

    // Hiding the pictures goes to text rows. Those are short enough that the
    // scroll stops at the last row before the top row can reach the top, so
    // the top row stays on screen rather than at the top.
    panel.query.set_picture_shown(PictureColumn::Patch, false);
    short(&ctx, &mut panel, &mut state);
    let drawn = drawn_points(&panel);
    assert!(drawn.contains(&top), "{drawn:?}");
    assert_eq!(*drawn.last().unwrap(), POINTS as u32 - 1);
}

/// While the rows are of the version before an edit, their cells draw the
/// pictures already drawn for that version, dimmed, and a row that had none
/// drawn waits: nothing is made for a version the rows do not belong to. Once
/// the new table lands, the pictures are made for the new version.
#[test]
fn out_of_date_rows_make_no_pictures() {
    let (mut state, id) = state_with_bitmaps();
    let (ctx, mut panel) = shown_short(&mut state);
    let before = state.scene[0].history.current_version().document_serial;
    let drawn = drawn_points(&panel);
    let made = panel.pictures.made.len();
    state
        .delete_point(PointRef {
            recon: id,
            point: 0,
        })
        .expect("deletable");
    let after = state.scene[0].history.current_version().document_serial;
    assert_ne!(before, after);

    // Taller, so rows that had no picture drawn come into view.
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.pictures.made.len(), made, "nothing made while stale");
    for &(point, _, picture) in &panel.drawn_pictures {
        if drawn.contains(&point) && point != NO_PATCH {
            assert!(
                matches!(picture, CellPicture::Texture(_)),
                "#{point} keeps its old picture"
            );
        } else if !drawn.contains(&point) {
            assert_eq!(picture, CellPicture::Waiting, "#{point}");
        }
    }
    assert!(panel
        .drawn_pictures
        .iter()
        .any(|&(_, _, picture)| picture == CellPicture::Waiting));

    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let new: Vec<&PictureKey> = panel.pictures.made[made..].iter().collect();
    assert!(!new.is_empty());
    assert!(new.iter().all(|k| k.document == after && k.point != 0));
}

/// Hovering a Patch cell shows the patch with what it is; a click on it
/// selects the row's point.
#[test]
fn hovering_a_patch_says_what_it_is_and_a_click_selects_the_row() {
    let (mut state, id) = state_with_bitmaps();
    let (ctx, mut panel) = shown(&mut state);
    // A tooltip waits out `tooltip_delay`, which a headless frame has no wall
    // clock to pass.
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let heading = painted_at(&ctx, &mut panel, &mut state, "Patch");
    let &(point, row) = panel.drawn().iter().find(|(p, _)| *p == 5).unwrap();
    let at = egui::pos2(heading.x, row.center().y);
    let mut texts = Vec::new();
    for _ in 0..3 {
        let mut raw = input(1200.0);
        raw.events = vec![egui::Event::PointerMoved(at)];
        texts = painted_texts(&ctx, raw, |ui| panel.show(ui, &mut state));
    }
    let hover = format!("{}.", crate::patch_pictures::STORED_PATCH_HOVER);
    assert!(texts.contains(&hover), "{texts:?}");
    click(&ctx, &mut panel, &mut state, at);
    assert_eq!(state.selected_point, Some(PointRef { recon: id, point }));
    assert_eq!(
        panel.query.sort, None,
        "the Patch heading and cells do not sort"
    );
    // The value columns still sort.
    panel.query.click_heading(TrackTableColumn::RmsErrorPx);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.rows()[0], POINTS as u32 - 1);
}

/// Clicking the Patch heading does nothing: a picture column cannot be
/// sorted.
#[test]
fn the_patch_heading_does_not_sort() {
    let (mut state, _) = state_with_bitmaps();
    let (ctx, mut panel) = shown(&mut state);
    let heading = painted_at(&ctx, &mut panel, &mut state, "Patch");
    click(&ctx, &mut panel, &mut state, heading);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(panel.query.sort, None);
    assert_eq!(panel.rows(), (0..POINTS as u32).collect::<Vec<_>>());
}

/// The rows are a line of text high at the smallest picture size too: the
/// smallest row still holds a line of body text.
#[test]
fn the_smallest_row_holds_a_line_of_text() {
    let ctx = egui::Context::default();
    let mut height = 0.0;
    run_frame_headless(&ctx, input(100.0), |ui| height = text_row_height(ui));
    assert!(MIN_PICTURE_SIZE + 2.0 * PICTURE_PAD > height);
}

/// The Patch column is pinned with the Point column: scrolled sideways, its
/// heading stays at the left edge while the value columns' headings move.
#[test]
fn the_patch_column_stays_put_when_the_table_scrolls_sideways() {
    let (mut state, _) = state_with_bitmaps();
    let narrow = |time: f64| egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(360.0, 600.0),
        )),
        time: Some(time),
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let mut panel = TrackTablePanel::new();
    for column in [
        TrackTableColumn::X,
        TrackTableColumn::Y,
        TrackTableColumn::Z,
    ] {
        panel.query.set_shown(column, true);
    }
    run_frame_headless(&ctx, narrow(0.0), |ui| panel.show(ui, &mut state));
    state.settle_track_table_columns();
    run_frame_headless(&ctx, narrow(0.1), |ui| panel.show(ui, &mut state));
    let left_of = |texts: &[crate::test_support::PaintedText], text: &str| {
        texts
            .iter()
            .find(|t| t.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .left()
    };
    let before = painted_text_rects(&ctx, narrow(0.2), |ui| panel.show(ui, &mut state));
    let pictures_before = panel.drawn_pictures.len();
    let &(_, row) = panel.drawn().iter().find(|(p, _)| *p == 2).unwrap();
    let mut time = 0.3;
    for _ in 0..20 {
        let mut raw = narrow(time);
        raw.events = vec![
            egui::Event::PointerMoved(egui::pos2(300.0, row.center().y)),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(-40.0, 0.0),
                phase: egui::TouchPhase::Move,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        run_frame_headless(&ctx, raw, |ui| panel.show(ui, &mut state));
        time += 0.1;
    }
    let after = painted_text_rects(&ctx, narrow(time), |ui| panel.show(ui, &mut state));
    for pinned in ["Point", "Patch", "#2"] {
        assert_eq!(
            left_of(&before, pinned),
            left_of(&after, pinned),
            "{pinned} moved"
        );
    }
    let moved = left_of(&before, "Track") - left_of(&after, "Track");
    assert!(moved > 50.0, "the Track heading moved {moved}");
    assert_eq!(panel.drawn_pictures.len(), pictures_before);
}

/// A picture's hover view is three times the picture size, up to a cap that
/// keeps it inside a small window.
#[test]
fn the_hover_view_is_three_times_the_size_up_to_its_cap() {
    use super::super::pictures::{hover_size, MAX_HOVER_SIZE, MAX_PICTURE_SIZE};
    assert_eq!(hover_size(DEFAULT_PICTURE_SIZE), 3.0 * DEFAULT_PICTURE_SIZE);
    assert_eq!(hover_size(MIN_PICTURE_SIZE), 3.0 * MIN_PICTURE_SIZE);
    assert_eq!(hover_size(MAX_PICTURE_SIZE), MAX_HOVER_SIZE);
}
