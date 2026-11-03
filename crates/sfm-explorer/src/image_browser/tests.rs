// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The strip's labels and its hover tooltip on the checked-in ground truths,
//! drawn headless through whole egui frames.

use std::path::Path;

use eframe::egui;
use sfmtool_core::SfmrReconstruction;

use super::{thumbnail_id, ImageBrowser};
use crate::action_log::ActionLog;
use crate::elide::ELLIPSIS;
use crate::scene::ReconId;
use crate::test_support::PaintedText;

const SEOUL_BULL_V02: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth_v02.sfmr"
);
const KERRY_PARK_V01: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/kerry_park/kerry_park_ground_truth_v01.sfmr"
);

fn load(path: &str) -> SfmrReconstruction {
    SfmrReconstruction::load(Path::new(path), &sfmtool_core::progress::Progress::none())
        .expect("the ground truth loads")
}

/// One frame of the strip over `recon` in a window of `size`, with `events`
/// delivered; returns every string it painted.
fn frame(
    browser: &mut ImageBrowser,
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    id: ReconId,
    size: egui::Vec2,
    events: Vec<egui::Event>,
) -> Vec<PaintedText> {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        events,
        ..Default::default()
    };
    let scroll = crate::platform::ScrollInput::default();
    let mut log = ActionLog::default();
    crate::test_support::painted_text_rects(ctx, input, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            browser.show(
                ui,
                recon,
                id,
                None,
                None,
                &[],
                &[],
                &[],
                None,
                None,
                &[],
                &scroll,
                None,
                &mut log,
            );
        });
    })
}

/// Each thumbnail's rect and the label drawn under it, for every thumbnail on
/// screen, in image order.
fn labels(recon: &SfmrReconstruction, size: egui::Vec2) -> Vec<(usize, egui::Rect, PaintedText)> {
    let mut browser = ImageBrowser::new();
    let ctx = egui::Context::default();
    let id = ReconId::next();
    frame(&mut browser, &ctx, recon, id, size, Vec::new());
    let texts = frame(&mut browser, &ctx, recon, id, size, Vec::new());
    let mut out = Vec::new();
    for i in 0..recon.image_set.images.len() {
        let Some(thumb) = browser.hit_rect(thumbnail_id(i)) else {
            continue;
        };
        let under: Vec<&PaintedText> = texts
            .iter()
            .filter(|t| {
                (t.rect.center().x - thumb.center().x).abs() < 0.5
                    && t.rect.top() >= thumb.bottom()
                    && t.rect.top() <= thumb.bottom() + 4.0
            })
            .collect();
        assert_eq!(under.len(), 1, "image {i}: {under:?}");
        out.push((i, thumb, under[0].clone()));
    }
    out
}

/// No label is wider than its thumbnail or reaches its neighbour's.
fn assert_fit(drawn: &[(usize, egui::Rect, PaintedText)]) {
    for (i, thumb, text) in drawn {
        assert!(
            text.rect.width() <= thumb.width() + 0.01,
            "image {i}: {:?} is {} wide in a {} thumbnail",
            text.text,
            text.rect.width(),
            thumb.width()
        );
    }
    for pair in drawn.windows(2) {
        let (a, b) = (&pair[0].2, &pair[1].2);
        assert!(
            a.rect.right() <= b.rect.left(),
            "{:?} overlaps {:?}",
            a.text,
            b.text
        );
    }
}

#[test]
fn seoul_bull_thumbnails_are_labelled_with_their_numbers() {
    let recon = load(SEOUL_BULL_V02);
    let drawn = labels(&recon, egui::vec2(3000.0, 260.0));
    assert_eq!(drawn.len(), 17, "every thumbnail fits on screen");
    for (i, _, text) in &drawn {
        assert_eq!(text.text, format!("{ELLIPSIS}{:02}.jpg", i + 1));
    }
    assert_fit(&drawn);
}

#[test]
fn kerry_park_thumbnails_are_labelled_with_their_sensor_and_frame() {
    let recon = load(KERRY_PARK_V01);
    let drawn = labels(&recon, egui::vec2(9000.0, 220.0));
    assert_eq!(drawn.len(), 48, "every thumbnail fits on screen");
    for (i, _, text) in &drawn {
        let name = &recon.image_set.images[*i].name;
        let sensor = name.strip_prefix("fisheye_").expect("a rig name");
        assert_eq!(text.text, format!("{ELLIPSIS}{sensor}"));
    }
    assert_eq!(drawn[0].2.text, "\u{2026}left/frame_01.jpg");
    assert_fit(&drawn);
}

/// A strip too short for the names shortens each in the middle, keeping the
/// mark, the sensor's start and the extension, and still fits.
#[test]
fn narrow_kerry_park_thumbnails_shorten_their_labels() {
    let recon = load(KERRY_PARK_V01);
    let drawn = labels(&recon, egui::vec2(3000.0, 120.0));
    assert!(drawn.len() > 2, "{} on screen", drawn.len());
    for (i, thumb, text) in &drawn {
        assert!(thumb.width() < 90.0, "image {i}: {} wide", thumb.width());
        assert!(
            text.text.starts_with(ELLIPSIS),
            "image {i}: {:?}",
            text.text
        );
        assert_eq!(
            text.text.matches(ELLIPSIS).count(),
            2,
            "image {i}: {:?}",
            text.text
        );
        assert!(text.text.ends_with(".jpg"), "image {i}: {:?}", text.text);
    }
    assert_fit(&drawn);
}

/// Hovering a thumbnail shows the image's stored name, the path it is read
/// from, and its index, camera and pose.
#[test]
fn hovering_a_thumbnail_names_the_image_in_full() {
    let mut recon = load(SEOUL_BULL_V02);
    recon.image_set.images[4].translation_xyz.x = f64::NAN;
    let size = egui::vec2(3000.0, 260.0);
    let mut browser = ImageBrowser::new();
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let id = ReconId::next();
    let hover = |browser: &mut ImageBrowser, i: usize| -> Vec<String> {
        frame(browser, &ctx, &recon, id, size, Vec::new());
        let at = browser
            .hit_rect(thumbnail_id(i))
            .expect("on screen")
            .center();
        frame(
            browser,
            &ctx,
            &recon,
            id,
            size,
            vec![egui::Event::PointerMoved(at)],
        );
        for _ in 0..4 {
            frame(browser, &ctx, &recon, id, size, Vec::new());
        }
        frame(browser, &ctx, &recon, id, size, Vec::new())
            .into_iter()
            .map(|t| t.text)
            .collect()
    };

    let texts = hover(&mut browser, 3);
    let path = recon
        .workspace_dir
        .join("seoul_bull_sculpture_04.jpg")
        .display()
        .to_string();
    for line in [
        "seoul_bull_sculpture_04.jpg",
        &path,
        "image index 3 (0-based, 17 images), camera 0",
    ] {
        assert!(texts.iter().any(|t| t == line), "{line:?} in {texts:?}");
    }

    let texts = hover(&mut browser, 4);
    assert!(
        texts
            .iter()
            .any(|t| t == "image index 4 (0-based, 17 images), camera 0, unposed"),
        "{texts:?}"
    );
}
