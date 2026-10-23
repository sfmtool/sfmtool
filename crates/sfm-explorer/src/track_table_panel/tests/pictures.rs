// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Track Table's Patch column, the *Size* dropdown and
//! the header line it sits on, and the texture cache the picture cells draw
//! from.

use std::sync::Arc;

use ndarray::Array4;
use sfmtool_core::analysis::track_table::TrackTableColumn;

use super::super::pictures::{
    Absent, CellPicture, Made, PictureColumn, PictureKey, PictureTextures, DEFAULT_PICTURE_SIZE,
    PICTURE_PAD, PICTURE_SIZES, TEXTURE_CAPACITY,
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
/// has no Patch column and no Size dropdown, and its rows are a line of text
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
    assert_eq!(no_patch.2, CellPicture::Absent(Absent::NoPatch));
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
    let image = || Made::Image(egui::ColorImage::filled([2, 2], egui::Color32::GRAY));
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

/// Each size the Size dropdown offers sets the row height, and the rows at
/// the top stay there when it changes. At 16 pt the row is a text row high,
/// since a picture and its padding are shorter than a line of text.
#[test]
fn each_picture_size_sets_the_row_height_and_keeps_the_top_row() {
    let (mut state, id) = state_with_bitmaps();
    let (ctx, mut panel) = shown_short(&mut state);
    state.select_point(PointRef {
        recon: id,
        point: 20,
    });
    short(&ctx, &mut panel, &mut state);
    short(&ctx, &mut panel, &mut state);
    // Point 20, low enough that the rows below the top one fill the window
    // even at 16 pt, so the scroll can keep the top row at the top.
    let top = drawn_points(&panel)[0];
    assert!(top > 10, "scrolled down to the selection: {top}");

    let mut text_height = 0.0;
    run_frame_headless(&ctx, input(100.0), |ui| text_height = text_row_height(ui));
    for size in [16.0, 128.0, 32.0, DEFAULT_PICTURE_SIZE] {
        panel.picture_size.0 = size;
        short(&ctx, &mut panel, &mut state);
        assert_eq!(
            drawn_row_height(&panel),
            (size + 2.0 * PICTURE_PAD).max(text_height),
            "size {size}"
        );
        assert_eq!(drawn_points(&panel)[0], top, "size {size}");
        short(&ctx, &mut panel, &mut state);
        assert_eq!(drawn_points(&panel)[0], top, "size {size}, settled");
    }

    // Hiding the pictures goes to text rows, and the top row stays at the top.
    panel.query.set_picture_shown(PictureColumn::Patch, false);
    short(&ctx, &mut panel, &mut state);
    assert_eq!(drawn_row_height(&panel), text_height);
    assert_eq!(drawn_points(&panel)[0], top, "text rows");
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

/// The dropdown offers 16, 32, 64 and 128 pt, and 64 before another is
/// picked.
#[test]
fn the_size_dropdown_offers_four_sizes_and_starts_at_64() {
    assert_eq!(PICTURE_SIZES, [16.0, 32.0, 64.0, 128.0]);
    assert_eq!(DEFAULT_PICTURE_SIZE, 64.0);
    assert_eq!(TrackTablePanel::new().picture_size.0, DEFAULT_PICTURE_SIZE);
}

/// Every picture row holds a line of body text. At 16 pt the picture and its
/// padding are shorter than a text row, so the row is a text row high; at 32
/// pt and up the picture row is the taller.
#[test]
fn the_smallest_row_holds_a_line_of_text() {
    let (mut state, _) = state_with_bitmaps();
    let (ctx, mut panel) = shown_short(&mut state);
    let mut text_height = 0.0;
    run_frame_headless(&ctx, input(100.0), |ui| text_height = text_row_height(ui));
    assert!(
        16.0 + 2.0 * PICTURE_PAD < text_height,
        "16 pt needs the text row"
    );
    assert!(32.0 + 2.0 * PICTURE_PAD > text_height);
    for size in PICTURE_SIZES {
        panel.picture_size.0 = size;
        short(&ctx, &mut panel, &mut state);
        assert!(drawn_row_height(&panel) >= text_height, "size {size}");
    }
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
    use super::super::pictures::{hover_size, MAX_HOVER_SIZE};
    assert_eq!(hover_size(DEFAULT_PICTURE_SIZE), 3.0 * DEFAULT_PICTURE_SIZE);
    assert_eq!(hover_size(16.0), 48.0);
    assert_eq!(hover_size(128.0), MAX_HOVER_SIZE);
}

/// One frame of the panel in the short window with `events` delivered, and
/// the AccessKit nodes it built: what `get_widgets` reads a widget's role,
/// name, value and rectangle from.
fn accessible_frame(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    events: Vec<egui::Event>,
) -> Vec<egui::accesskit::Node> {
    let mut raw = input(400.0);
    raw.events = events;
    let mut output = ctx.run_ui(raw, |ui| panel.show(ui, state));
    output.textures_delta.clear();
    output
        .platform_output
        .accesskit_update
        .expect("AccessKit is on")
        .nodes
        .into_iter()
        .map(|(_, node)| node)
        .collect()
}

/// Press and release the primary button at `at`, then draw one more frame and
/// return its nodes.
fn accessible_click(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    at: egui::Pos2,
) -> Vec<egui::accesskit::Node> {
    for pressed in [true, false] {
        accessible_frame(
            ctx,
            panel,
            state,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
    }
    accessible_frame(ctx, panel, state, Vec::new())
}

/// The one node of `role`.
fn only(nodes: &[egui::accesskit::Node], role: egui::accesskit::Role) -> &egui::accesskit::Node {
    let mut found = nodes.iter().filter(|node| node.role() == role);
    let node = found.next().unwrap_or_else(|| panic!("no {role:?}"));
    assert!(found.next().is_none(), "more than one {role:?}");
    node
}

/// The centre of a node's rectangle.
fn centre(node: &egui::accesskit::Node) -> egui::Pos2 {
    let bounds = node.bounds().expect("the node has a rectangle");
    egui::pos2(
        ((bounds.x0 + bounds.x1) / 2.0) as f32,
        ((bounds.y0 + bounds.y1) / 2.0) as f32,
    )
}

/// The Size dropdown is named *Size* in the accessibility tree, with the size
/// it shows as its value, which is what `get_widgets` lists it by. A click on
/// it opens its four sizes, and a click on one picks it, as an agent does
/// through `click`.
#[test]
fn the_size_dropdown_is_named_and_a_click_on_an_entry_picks_its_size() {
    use egui::accesskit::Role;
    let (mut state, _) = state_with_bitmaps();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut panel = TrackTablePanel::new();
    accessible_frame(&ctx, &mut panel, &mut state, Vec::new());
    state.settle_track_table_columns();
    let nodes = accessible_frame(&ctx, &mut panel, &mut state, Vec::new());
    let dropdown = only(&nodes, Role::ComboBox);
    assert_eq!(dropdown.label(), Some("Size"));
    assert_eq!(dropdown.value(), Some("64 pt"));
    let width = |node: &egui::accesskit::Node| {
        let bounds = node.bounds().expect("the node has a rectangle");
        bounds.x1 - bounds.x0
    };
    let default_width = width(dropdown);
    assert!(
        default_width < egui::Spacing::default().combo_width as f64,
        "narrower than egui's default: {default_width}"
    );

    for (pick, text) in [(16.0, "16 pt"), (128.0, "128 pt"), (32.0, "32 pt")] {
        let nodes = accessible_frame(&ctx, &mut panel, &mut state, Vec::new());
        let at = centre(only(&nodes, Role::ComboBox));
        let nodes = accessible_click(&ctx, &mut panel, &mut state, at);
        // The update's nodes are in no drawing order, so the entries are
        // read top to bottom.
        let mut entries: Vec<(f64, &str)> = nodes
            .iter()
            .filter(|node| node.role() == Role::Button)
            .filter_map(|node| Some((node.bounds()?.y0, node.label()?)))
            .filter(|(_, name)| name.ends_with(" pt"))
            .collect();
        entries.sort_by(|a, b| a.0.total_cmp(&b.0));
        let entries: Vec<&str> = entries.into_iter().map(|(_, name)| name).collect();
        assert_eq!(entries, ["16 pt", "32 pt", "64 pt", "128 pt"]);
        let entry = nodes
            .iter()
            .find(|node| node.role() == Role::Button && node.label() == Some(text))
            .expect("the entry");
        let at = centre(entry);
        let nodes = accessible_click(&ctx, &mut panel, &mut state, at);
        assert_eq!(panel.picture_size.0, pick);
        let dropdown = only(&nodes, Role::ComboBox);
        assert_eq!(dropdown.label(), Some("Size"));
        assert_eq!(dropdown.value(), Some(text));
        assert_eq!(width(dropdown), default_width, "the same width at {text}");
    }
}

/// The strings of the header line, each with the middle of its row, painted
/// in a window `width` points wide.
fn header_rows(
    ctx: &egui::Context,
    panel: &mut TrackTablePanel,
    state: &mut AppState,
    width: f32,
) -> Vec<(String, f32)> {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, 400.0),
        )),
        ..Default::default()
    };
    let label = state.scene[0].label.clone();
    let header = [
        label.as_str(),
        "40 of 40 points",
        "Columns",
        "Size",
        "64 pt",
        "Seen in selected image",
    ];
    let texts = painted_text_rects(ctx, raw, |ui| panel.show(ui, state));
    header
        .iter()
        .map(|&text| {
            let painted = texts
                .iter()
                .find(|t| t.text == text)
                .unwrap_or_else(|| panic!("{text:?} was not painted"));
            (text.to_string(), painted.rect.center().y)
        })
        .collect()
}

/// With a long reconstruction label and an image selected, the header line,
/// from the label to the selected-image box, fits on one line at 700 pt, a
/// narrow panel. The box is named by its label in the accessibility tree,
/// which is the name `get_widgets` lists it under. In a window too narrow for
/// the line it wraps, and the *Size* label stays beside its dropdown.
#[test]
fn the_header_fits_on_one_line_and_size_stays_with_its_dropdown_when_it_wraps() {
    use egui::accesskit::Role;
    let (mut state, id) = state_with_bitmaps();
    state.scene[0].label = "seoul_bull_sculpture_ground_truth".into();
    let (ctx, mut panel) = shown(&mut state);
    state.select_image(Some(crate::scene::ImageRef::new(id, 0)));

    let rows = header_rows(&ctx, &mut panel, &mut state, 700.0);
    let first = rows[0].1;
    for (text, y) in &rows {
        assert!(
            (y - first).abs() < 4.0,
            "{text:?} is off the line: {rows:?}"
        );
    }

    let rows = header_rows(&ctx, &mut panel, &mut state, 450.0);
    let y = |text: &str| rows.iter().find(|(t, _)| t == text).unwrap().1;
    assert!(
        y("Seen in selected image") > y(&state.scene[0].label) + 4.0,
        "the line wraps at 450 pt: {rows:?}"
    );
    assert!((y("Size") - y("64 pt")).abs() < 4.0, "{rows:?}");

    ctx.enable_accesskit();
    accessible_frame(&ctx, &mut panel, &mut state, Vec::new());
    let nodes = accessible_frame(&ctx, &mut panel, &mut state, Vec::new());
    let checkbox = only(&nodes, Role::CheckBox);
    assert_eq!(checkbox.label(), Some("Seen in selected image"));
}
