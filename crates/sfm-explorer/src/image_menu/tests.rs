// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image menu is one menu in two places: a Scene tree image row and an
//! Image Browser thumbnail. These tests open it in both and compare, so the two
//! cannot drift apart.
//!
//! The strip is run headless the way the Scene tree is
//! (`scene_graph/tests.rs`): whole egui frames through `Context::run_ui`, with
//! clicks aimed at the rects the strip recorded on the previous frame.

use eframe::egui;

use super::{ImageMenuAction, ENTRIES};
use crate::image_browser::{menu_entry_id, thumbnail_id, ImageBrowser, ImageBrowserResponse};
use crate::scene::ImageRef;
use crate::scene_graph::row_id;
use crate::scene_graph::tests::{
    click, hover_texts_after_a_click, open_context_menu, resectable_shoot, shared_shoot,
    with_image_list,
};
use crate::state::AppState;

const STRIP: egui::Vec2 = egui::vec2(1600.0, 220.0);

/// One frame of the strip over the first node, with `events` delivered.
fn strip_frame(
    browser: &mut ImageBrowser,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
) -> ImageBrowserResponse {
    let id = state.scene[0].id;
    let menu = state.image_menu(id);
    let selected = state.selected_image_in(id);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), STRIP)),
        events,
        ..Default::default()
    };
    let mut response = None;
    let scroll = crate::platform::ScrollInput::default();
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            response = Some(browser.show(
                ui,
                state.scene[0].recon(),
                id,
                None,
                selected,
                &[],
                &[],
                &[],
                None,
                None,
                &[],
                &scroll,
                menu.as_ref(),
                &mut state.action_log,
            ));
        });
    });
    response.expect("the strip ran")
}

/// A strip settled over the first node, its thumbnails laid out.
fn settled_strip(state: &mut AppState) -> (ImageBrowser, egui::Context) {
    let mut browser = ImageBrowser::new();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        strip_frame(&mut browser, &ctx, state, Vec::new());
    }
    (browser, ctx)
}

fn button(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::default(),
    }
}

/// Hover, press, release `button` at `pos`, and one settling frame. Returns the
/// response of the frame the click landed in.
fn click_strip_at(
    browser: &mut ImageBrowser,
    ctx: &egui::Context,
    state: &mut AppState,
    pos: egui::Pos2,
    which: egui::PointerButton,
) -> ImageBrowserResponse {
    strip_frame(browser, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    strip_frame(browser, ctx, state, vec![button(pos, which, true)]);
    let landed = strip_frame(browser, ctx, state, vec![button(pos, which, false)]);
    strip_frame(browser, ctx, state, Vec::new());
    landed
}

/// Right-click thumbnail `index`, opening the image menu on it.
fn open_strip_menu(
    browser: &mut ImageBrowser,
    ctx: &egui::Context,
    state: &mut AppState,
    index: usize,
) -> ImageBrowserResponse {
    let pos = browser
        .hit_rect(thumbnail_id(index))
        .expect("the thumbnail was drawn")
        .center();
    click_strip_at(browser, ctx, state, pos, egui::PointerButton::Secondary)
}

/// Click the image menu's entry `key` open on thumbnail `index`.
fn click_strip_entry(
    browser: &mut ImageBrowser,
    ctx: &egui::Context,
    state: &mut AppState,
    key: &str,
    index: usize,
) -> ImageBrowserResponse {
    let pos = browser
        .hit_rect(menu_entry_id(key, index))
        .unwrap_or_else(|| panic!("{key} was not drawn"))
        .center();
    strip_frame(browser, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    strip_frame(
        browser,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Primary, true)],
    );
    strip_frame(
        browser,
        ctx,
        state,
        vec![button(pos, egui::PointerButton::Primary, false)],
    )
}

/// The entries' keys in the order they were laid out, top to bottom.
fn laid_out(rect_of: impl Fn(&str) -> Option<egui::Rect>) -> Vec<&'static str> {
    let mut drawn: Vec<(f32, &'static str)> = ENTRIES
        .iter()
        .filter_map(|&(key, _)| rect_of(key).map(|rect| (rect.center().y, key)))
        .collect();
    drawn.sort_by(|a, b| a.0.total_cmp(&b.0));
    drawn.into_iter().map(|(_, key)| key).collect()
}

#[test]
fn the_strip_and_the_tree_show_the_same_entries_in_the_same_order() {
    let every: Vec<&str> = ENTRIES.iter().map(|&(key, _)| key).collect();

    let mut state = shared_shoot(1);
    let (mut panel, ctx, id) = with_image_list(&mut state);
    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    let tree = laid_out(|key| panel.hit_rect(row_id(id, &format!("{key}_2"))));

    let (mut browser, strip_ctx) = settled_strip(&mut state);
    open_strip_menu(&mut browser, &strip_ctx, &mut state, 2);
    let strip = laid_out(|key| browser.hit_rect(menu_entry_id(key, 2)));

    assert_eq!(tree, every);
    assert_eq!(strip, every);
}

#[test]
fn an_entry_chosen_in_the_strip_is_the_action_the_tree_reports() {
    for (key, action) in [
        (super::RESECT, ImageMenuAction::Resect),
        (super::MOVE_CAMERA, ImageMenuAction::MoveCamera),
        (super::DELETE_IMAGE, ImageMenuAction::Delete),
    ] {
        let mut state = resectable_shoot();
        let (mut panel, ctx, id) = with_image_list(&mut state);
        open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
        let tree = click(
            &mut panel,
            &ctx,
            &mut state,
            row_id(id, &format!("{key}_2")),
        );

        let (mut browser, strip_ctx) = settled_strip(&mut state);
        open_strip_menu(&mut browser, &strip_ctx, &mut state, 2);
        let strip = click_strip_entry(&mut browser, &strip_ctx, &mut state, key, 2);

        assert_eq!(tree.image_menu, Some((ImageRef::new(id, 2), action)));
        assert_eq!(
            strip
                .menu_action
                .map(|(index, action)| (ImageRef::new(id, index), action)),
            tree.image_menu,
            "{key}"
        );
    }
}

#[test]
fn the_strip_greys_resect_with_the_tree_s_reason() {
    // No cluster-patches file: Resect Image is greyed in both, with one reason.
    let mut state = shared_shoot(1);
    let (mut panel, ctx, id) = with_image_list(&mut state);
    let refusal = state
        .resect_image_refusal(ImageRef::new(id, 2))
        .expect("no file is open");

    open_context_menu(&mut panel, &ctx, &mut state, row_id(id, "image_2"));
    let pos = panel
        .hit_rect(row_id(id, "resect_2"))
        .expect("drawn")
        .center();
    let tree_texts = hover_texts_after_a_click(&mut panel, &ctx, &mut state, pos);
    assert!(tree_texts.contains(&refusal), "{tree_texts:?}");

    let (mut browser, strip_ctx) = settled_strip(&mut state);
    open_strip_menu(&mut browser, &strip_ctx, &mut state, 2);
    // Hovered before it is clicked: egui puts no tooltip up over a widget
    // clicked more recently than the pointer moved.
    let pos = browser
        .hit_rect(menu_entry_id(super::RESECT, 2))
        .expect("the menu is still up")
        .center();
    strip_ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    // The pointer rests before it moves and after, for the reasons
    // `hover_texts_after_a_click` gives.
    for _ in 0..8 {
        strip_frame(&mut browser, &strip_ctx, &mut state, Vec::new());
    }
    strip_frame(
        &mut browser,
        &strip_ctx,
        &mut state,
        vec![egui::Event::PointerMoved(pos)],
    );
    for _ in 0..12 {
        strip_frame(&mut browser, &strip_ctx, &mut state, Vec::new());
    }
    let id0 = state.scene[0].id;
    let menu = state.image_menu(id0);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), STRIP)),
        ..Default::default()
    };
    let scroll = crate::platform::ScrollInput::default();
    let strip_texts = crate::test_support::painted_texts(&strip_ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            browser.show(
                ui,
                state.scene[0].recon(),
                id0,
                None,
                None,
                &[],
                &[],
                &[],
                None,
                None,
                &[],
                &scroll,
                menu.as_ref(),
                &mut state.action_log,
            );
        });
    });
    assert!(strip_texts.contains(&refusal), "{strip_texts:?}");

    let strip = click_strip_entry(&mut browser, &strip_ctx, &mut state, super::RESECT, 2);
    assert_eq!(strip.menu_action, None, "the greyed entry ran");
}

#[test]
fn a_right_click_on_a_thumbnail_selects_nothing() {
    let mut state = resectable_shoot();
    let (mut browser, ctx) = settled_strip(&mut state);
    let before = state.selected_image;
    let response = open_strip_menu(&mut browser, &ctx, &mut state, 3);
    assert_eq!(response.selection_changed, None);
    assert_eq!(state.selected_image, before);
    assert!(browser.hit_rect(menu_entry_id(super::RESECT, 3)).is_some());

    // A left click on a thumbnail still selects it, as it always has.
    let pos = browser.hit_rect(thumbnail_id(1)).expect("drawn").center();
    let response = click_strip_at(
        &mut browser,
        &ctx,
        &mut state,
        pos,
        egui::PointerButton::Primary,
    );
    assert_eq!(response.selection_changed, Some(Some(1)));
}

#[test]
fn a_right_click_off_the_thumbnails_opens_no_menu() {
    let mut state = resectable_shoot();
    let (mut browser, ctx) = settled_strip(&mut state);
    // The minibar, along the strip's bottom edge.
    let pos = egui::pos2(STRIP.x / 2.0, STRIP.y - 12.0);
    click_strip_at(
        &mut browser,
        &ctx,
        &mut state,
        pos,
        egui::PointerButton::Secondary,
    );
    for index in 0..8 {
        assert!(
            browser
                .hit_rect(menu_entry_id(super::RESECT, index))
                .is_none(),
            "a menu opened on image {index}"
        );
    }
}
