// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for where the 3D view's top text goes.

use eframe::egui;

use super::{camera_position_top, POSITION_THIRD_ROW_TOP, STATS_TOP, STATUS_TOP};
use crate::test_support::{painted_text_rects, PaintedText};
use crate::viewer_3d::Viewer3D;

#[test]
fn the_camera_position_stays_beside_stats_it_clears() {
    assert_eq!(camera_position_top(100.0, None, 200.0), STATS_TOP);
    assert_eq!(camera_position_top(100.0, Some(500.0), 200.0), STATS_TOP);
}

#[test]
fn the_camera_position_drops_to_the_status_row_when_that_is_clear() {
    assert_eq!(camera_position_top(300.0, None, 200.0), STATUS_TOP);
    assert_eq!(camera_position_top(300.0, Some(150.0), 200.0), STATUS_TOP);
}

#[test]
fn the_camera_position_takes_its_own_row_when_both_rows_are_taken() {
    assert_eq!(
        camera_position_top(300.0, Some(250.0), 200.0),
        POSITION_THIRD_ROW_TOP
    );
}

#[test]
fn text_that_only_touches_the_camera_position_moves_it_down() {
    // Closer than the gap counts as touching.
    assert_eq!(camera_position_top(195.0, None, 200.0), STATUS_TOP);
}

/// The top text one headless frame draws for an empty scene in a view
/// `width` wide, with `status` as the status message.
fn top_text(width: f32, status: Option<&str>) -> Vec<PaintedText> {
    let ctx = egui::Context::default();
    let viewer = Viewer3D::new();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, 400.0),
        )),
        ..Default::default()
    };
    painted_text_rects(&ctx, input, |ui| {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 400.0));
        viewer.draw_info_overlay(
            ui.painter(),
            rect,
            &[],
            None,
            None,
            false,
            true,
            status,
            None,
            None,
            60.0,
        );
    })
}

fn find<'a>(texts: &'a [PaintedText], prefix: &str) -> &'a PaintedText {
    texts
        .iter()
        .find(|t| t.text.starts_with(prefix))
        .unwrap_or_else(|| panic!("no text starting {prefix:?} in {texts:?}"))
}

#[test]
fn a_wide_view_keeps_the_camera_position_beside_the_stats() {
    let texts = top_text(1200.0, Some("Selected point 12"));
    let stats = find(&texts, "0 points");
    let position = find(&texts, "Pos:");
    assert_eq!(position.rect.top(), stats.rect.top());
}

#[test]
fn a_narrow_view_never_draws_the_camera_position_over_the_stats_or_status() {
    for status in [
        None,
        Some("Selected point 12"),
        Some("MCP: Proposed SFMTOOL_FISHEYE"),
    ] {
        for width in [140.0, 180.0, 220.0, 260.0, 300.0, 400.0] {
            let texts = top_text(width, status);
            let position = find(&texts, "Pos:");
            for other in texts.iter().filter(|t| !t.text.starts_with("Pos:")) {
                assert!(
                    !position.rect.intersects(other.rect),
                    "at {width} px with {status:?}: {position:?} touches {other:?}"
                );
            }
        }
    }
}

#[test]
fn a_narrow_view_moves_the_camera_position_under_the_stats() {
    let texts = top_text(180.0, None);
    let stats = find(&texts, "0 points");
    let position = find(&texts, "Pos:");
    assert!(position.rect.top() >= stats.rect.bottom(), "{texts:?}");
}
