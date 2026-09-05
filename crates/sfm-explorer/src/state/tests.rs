// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What one frame or one MCP call records about the Image Detail panel's
//! controls: [`super::record_image_detail_changes`] and the texts it writes.

use super::*;

/// The two settings structs at their defaults, as a frame or a tool call
/// starts from.
fn defaults() -> ImageDetailDisplay {
    ImageDetailDisplay {
        feature: FeatureDisplaySettings::default(),
        intrinsics: IntrinsicsDisplaySettings::default(),
    }
}

/// The entries one change records, as text.
///
/// One field per call, so that each text is read back on its own. Nothing
/// here folds — every field is its own run — so a multi-field call reads
/// back as many rows as it changed fields, which
/// `one_entry_per_changed_field` covers.
fn texts(change: impl FnOnce(&mut ImageDetailDisplay)) -> Vec<String> {
    let before = defaults();
    let mut after = before.clone();
    change(&mut after);
    let mut log = ActionLog::new();
    record_image_detail_changes(&mut log, &before, &after);
    log.entries().map(|entry| entry.text.clone()).collect()
}

/// Every row of the catalogue in `specs/gui/action-log.md`, in the words
/// the Image Detail panel's own controls use.
#[test]
fn every_control_records_the_text_the_catalogue_gives_it() {
    assert_eq!(
        texts(|d| d.feature.overlay_mode = OverlayMode::ReprojError),
        ["Overlay Reproj Error"]
    );
    assert_eq!(
        texts(|d| d.feature.overlay_mode = OverlayMode::None),
        ["Overlay None"]
    );
    assert_eq!(
        texts(|d| d.feature.max_features = Some(500)),
        ["Max features 500"]
    );
    assert_eq!(
        texts(|d| {
            d.feature.min_feature_size = Some(2.0);
            d.feature.max_feature_size = Some(40.0);
        }),
        ["Feature size 2.0–40.0 px"]
    );
    assert_eq!(
        texts(|d| d.feature.tracked_only = false),
        ["Tracked only off"]
    );
    assert_eq!(texts(|d| d.intrinsics.enabled = false), ["Intrinsics off"]);
    assert_eq!(
        texts(|d| d.intrinsics.axes = false),
        ["Intrinsics axes off"]
    );
    assert_eq!(
        texts(|d| d.intrinsics.rings = true),
        ["Intrinsics rings on"]
    );
    assert_eq!(
        texts(|d| d.intrinsics.distortion = false),
        ["Intrinsics distortion off"]
    );
    assert_eq!(
        texts(|d| d.intrinsics.distortion_scale = Some(10.0)),
        ["Distortion scale ×10"]
    );
    assert_eq!(texts(|d| d.intrinsics.grid_cols = 32), ["Grid density 32"]);
}

/// The two texts that only a *return* to a default produces.
#[test]
fn lifting_a_filter_records_the_word_for_no_filter() {
    let mut before = defaults();
    before.feature.max_features = Some(500);
    before.feature.min_feature_size = Some(2.0);
    before.feature.max_feature_size = Some(40.0);
    before.intrinsics.distortion_scale = Some(10.0);
    let after = defaults();

    let mut log = ActionLog::new();
    record_image_detail_changes(&mut log, &before, &after);
    // Three fields, three runs, three rows: nothing here folds.
    assert_eq!(log.revision(), 3);
    assert_eq!(log.entries().count(), 3);
    // Each of the three on its own, so every text is read back.
    for (change, text) in [
        (
            Box::new(|d: &mut ImageDetailDisplay| d.feature.max_features = None)
                as Box<dyn FnOnce(&mut ImageDetailDisplay)>,
            "Max features all",
        ),
        (
            Box::new(|d: &mut ImageDetailDisplay| {
                d.feature.min_feature_size = None;
                d.feature.max_feature_size = None;
            }),
            "Feature size filter off",
        ),
        (
            Box::new(|d: &mut ImageDetailDisplay| d.intrinsics.distortion_scale = None),
            "Distortion scale auto",
        ),
    ] {
        let mut after = before.clone();
        change(&mut after);
        let mut log = ActionLog::new();
        record_image_detail_changes(&mut log, &before, &after);
        assert_eq!(
            log.entries().map(|e| e.text.as_str()).collect::<Vec<_>>(),
            [text]
        );
    }
}

/// A diff is not a widget signal: nothing changed, nothing recorded — the
/// hazard `ActionLog::changed` documents cannot arise here.
#[test]
fn an_unchanged_frame_records_nothing() {
    let mut log = ActionLog::new();
    record_image_detail_changes(&mut log, &defaults(), &defaults());
    assert_eq!(log.entries().count(), 0);
    // A drag value moved with the filter off is not a change either: the
    // filter is the pair of options, and the toolbar's persisted values
    // are what it re-derives them from.
    let mut after = defaults();
    after.feature.min_feature_size_value = 3.0;
    record_image_detail_changes(&mut log, &defaults(), &after);
    assert_eq!(log.entries().count(), 0);
}

/// One entry per field, however many a single call changed — and each
/// field is its own run, so all three rows survive and only a repeat of
/// the *same* field folds into the row it already has.
#[test]
fn one_entry_per_changed_field() {
    let before = defaults();
    let mut after = before.clone();
    after.feature.overlay_mode = OverlayMode::TrackLength;
    after.feature.tracked_only = false;
    after.intrinsics.rings = true;
    let mut log = ActionLog::new();
    record_image_detail_changes(&mut log, &before, &after);
    assert_eq!(log.revision(), 3);
    assert_eq!(
        log.entries().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        [
            "Overlay Track Length",
            "Tracked only off",
            "Intrinsics rings on"
        ]
    );
    let mut later = after.clone();
    later.intrinsics.rings = false;
    record_image_detail_changes(&mut log, &after, &later);
    assert_eq!(
        log.entries().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        [
            "Overlay Track Length",
            "Tracked only off",
            "Intrinsics rings off"
        ],
        "the repeat did not fold into its own field's row"
    );
}

/// Every glyph these texts put in the Action Log is one egui bundles; one
/// it does not renders as a replacement box and nothing else would notice.
#[test]
fn the_texts_glyphs_are_available_in_the_bundled_fonts() {
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::FontId::proportional(12.0);
    for glyph in [TIMES, EN_DASH] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&font, glyph)),
            "{glyph:?} is not in egui's bundled fonts and would render as a box"
        );
    }
}

/// The wire spellings the MCP surface takes, round-tripping through the
/// enum they name — and exact, so a GUI label is not a mode name.
#[test]
fn every_overlay_mode_round_trips_through_its_wire_name() {
    for mode in OverlayMode::ALL {
        assert_eq!(OverlayMode::from_wire_name(mode.wire_name()), Some(mode));
        assert!(OverlayMode::all_wire_names().contains(mode.wire_name()));
    }
    assert_eq!(OverlayMode::from_wire_name("Features"), None);
    assert_eq!(OverlayMode::from_wire_name("reproj error"), None);
}
