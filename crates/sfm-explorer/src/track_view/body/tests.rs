// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View's body, most of them in Edited mode; the
//! Viewed-mode tests are in `tests/viewed.rs`.
//!
//! egui needs no GPU to lay out a frame, so the whole body runs through
//! `Context::run_ui` here: `show` really does draw the header, the toolbar,
//! the boxes and every row of the table. What the assertions target
//! is what the panel *decides* -- which rows it drew, what each says, what the
//! boxes paint, and what it reports back to the dock -- rather than pixels.

use sfmtool_core::bench::{StageKind, Thresholds, Verdict};
use sfmtool_core::camera::image::ImageU8;

use super::{TrackBody, TrackBodyResponse, LOCK_LABEL};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

/// The point the fixtures put on the bench: three observations, in images 0, 1
/// and 2, at their exact projections.
const POINT: u32 = 2;

const VIEWPORT: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// A state holding one `embedded_patches` node with a photograph cached for
/// every image, as the bench's own tests build it.
fn state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    cache_photographs(&mut state, id);
    (state, id)
}

/// A textured photograph in the node's cache for every one of its images, which
/// is what the tiles are warped out of.
///
/// A pattern rather than a flat field, with short periods that differ between
/// the axes, so two tiles cut at two places are two different pictures.
fn cache_photographs(state: &mut AppState, id: ReconId) {
    let node = crate::scene::node_by_id(&state.scene, id).expect("loaded");
    let camera = &node.recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    let images = node.image_count();
    for image in 0..images {
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.insert_photograph(id, image, ImageU8::new(w, h, 3, data));
    }
}

/// Drive one frame of the panel and hand back what it reported.
fn run_frame(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> TrackBodyResponse {
    run_frame_with(panel, ctx, state, Vec::new())
}

/// The same frame, with `events` delivered to egui.
fn run_frame_with(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> TrackBodyResponse {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        response = Some(panel.show(ui, state));
    });
    response.expect("the panel ran")
}

/// Park the pointer at `pos` for two frames, clicking there on the second when
/// `click`, and hand back the last frame's response. Two frames are required:
/// egui resolves hover and clicks against the rects the previous pass
/// registered, so a single frame reports no interaction.
fn at_pointer(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
    click: bool,
) -> TrackBodyResponse {
    let mut response = None;
    for frame in 0..2 {
        let mut events = vec![egui::Event::PointerMoved(pos)];
        if click && frame == 1 {
            for pressed in [true, false] {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
        response = Some(run_frame_with(panel, ctx, state, events));
    }
    response.expect("two frames ran")
}

/// The y at which the row for `image` answers the pointer, found by walking
/// down the panel: what sits above the table is the header, the toolbar and
/// the boxes, and a hard-coded offset would go stale the moment
/// one of them gains a line.
fn row_y(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState, image: usize) -> f32 {
    for step in 0..(VIEWPORT.y as usize / 8) {
        let y = step as f32 * 8.0;
        let response = at_pointer(panel, ctx, state, egui::pos2(400.0, y), false);
        if response.hovered_image == Some(image) {
            return y;
        }
    }
    panic!("no row of the table answered for image {image}");
}

/// The raw input one headless frame is driven with.
fn input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    }
}

/// The strings one frame of the panel painted, with `events` delivered.
fn painted(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> Vec<String> {
    crate::test_support::painted_texts(ctx, input(events), |ui| {
        panel.show(ui, state);
    })
}

/// A panel and a context that have laid the table out once, so the pointer has
/// rects to resolve against.
fn settled(state: &AppState) -> (TrackBody, egui::Context) {
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, state);
    (panel, ctx)
}

/// Right-click at `at` and leave the menu it opened laid out.
///
/// Three frames: one to register the rows, one that right-clicks, and one
/// more, because the menu's entries are laid out on a later frame.
fn open_row_menu(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState, at: egui::Pos2) {
    for frame in 0..3 {
        let mut events = vec![egui::Event::PointerMoved(at)];
        if frame == 1 {
            for pressed in [true, false] {
                events.push(egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Secondary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
        painted(panel, ctx, state, events);
    }
}

/// The strings the first table row's context menu painted.
fn row_menu(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    let y = row_y(panel, ctx, state, 0);
    let at = egui::pos2(400.0, y);
    open_row_menu(panel, ctx, state, at);
    painted(panel, ctx, state, vec![egui::Event::PointerMoved(at)])
}

/// Where an open menu drew the entry called `text`.
fn menu_entry_pos(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> egui::Pos2 {
    crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
    .into_iter()
    .find(|painted| painted.text == text)
    .unwrap_or_else(|| panic!("{text:?} was not painted"))
    .rect
    .center()
}

/// Put [`POINT`] on the bench and draw one frame over it.
fn on_the_bench() -> (AppState, ReconId, String, TrackBody, egui::Context) {
    let (mut state, id) = state();
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    (state, id, label, panel, ctx)
}

/// With nothing focused and no viewed track the body draws nothing: Track View
/// draws its empty state then, and the ways in are that state's.
#[test]
fn with_nothing_active_the_body_draws_no_rows_and_no_text() {
    let (state, _) = state();
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let response = run_frame(&mut panel, &ctx, &state);

    assert!(panel.rows().is_empty());
    assert_eq!(response, TrackBodyResponse::default());
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(texts.is_empty(), "{texts:?}");
}

/// Once the track is evaluated, the *Reference* column marks the reference
/// and the row the reference-view rule picks, and every row prints its
/// viewing angle and pair ZNCC with every reading on hover.
#[test]
fn the_reference_column_marks_the_row_the_rule_picks() {
    use super::reference::ReferenceMark;
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.reference.text == "-"),
        "nothing has measured the track yet"
    );
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let pick = crate::bench::reference_view_pick(&track).expect("the rule picks a row");
    let reference = crate::bench::reference_in_use(&track);
    let rows = panel.rows();
    for row in rows {
        let expected = match (reference == Some(row.observation), pick == row.observation) {
            (true, true) => ReferenceMark::Reference,
            (true, false) => ReferenceMark::ReferenceNotPick,
            (false, true) => ReferenceMark::Pick,
            (false, false) => ReferenceMark::None,
        };
        assert_eq!(row.reference.mark, expected, "{rows:?}");
    }
    let picked = &rows[pick];
    let word = if reference == Some(pick) {
        "reference\n"
    } else {
        "pick\n"
    };
    assert!(
        picked.reference.text.starts_with(word),
        "{}",
        picked.reference.text
    );
    for row in rows {
        let (_, second) = row
            .reference
            .text
            .split_once('\n')
            .expect("the standing over the readings");
        assert!(
            second.contains('\u{b0}') && second.ends_with('%'),
            "{second}"
        );
        let hover = row.reference.hover.as_deref().expect("a hover");
        assert!(hover.contains("Viewing angle"), "{hover}");
        assert!(hover.contains("Coverage"), "{hover}");
        assert!(hover.contains("Pair ZNCC per ninth"), "{hover}");
    }
}

/// The *Reference* cell's words and hover for each standing, for an `out` row
/// and for a track that could not be evaluated.
#[test]
fn the_reference_cell_names_the_test_that_turned_a_row_away() {
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};
    use sfmtool_core::patch::reference_view::{
        ReferenceFallback, ReferenceStanding, ReferenceTest,
    };
    use sfmtool_core::patch::self_similarity::{SelfSimilarityEllipse, SelfSimilarityEllipseUnits};

    let ellipse = SelfSimilarityEllipseUnits {
        grid_px: SelfSimilarityEllipse {
            axes: [1.5, 0.8],
            axes_is_at_least: [false, false],
            major_angle: 0.0,
            matrix: [[2.25, 0.0], [0.0, 0.64]],
        },
        image_px: None,
        patch: None,
    };
    let row = |rejected_by: Option<ReferenceTest>, fallback: ReferenceFallback| Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            viewing_angle_deg: Some(71.6),
            tilt_direction_deg: Some(-35.0),
            coverage: Some(1.0),
            clipped_share: Some(0.004),
            pair_zncc: Some(0.834),
            cell_deficit: Some(0.12),
            pair_zncc_grid: Some([[0.9, 0.8, f64::NAN], [0.7, 0.6, 0.5], [0.4, 0.3, 0.2]]),
            zncc_self_similarity_ellipse: Some(ellipse),
            reference_view: Some(ReferenceStanding {
                rejected_by,
                fallback,
            }),
            ..TrackMeasurement::default()
        }),
    };
    let current = crate::bench::live::Evaluation::Current;
    // The row is row 0 of a track with no reference, which the rule picks
    // where its standing says so.
    let rows_of = |o: &Observation| super::reference::ReferenceRows {
        pick: o
            .track
            .as_ref()
            .and_then(|m| m.reference_view)
            .filter(|s| s.is_reference())
            .map(|_| 0),
        ..Default::default()
    };
    let cell = |o: &Observation| {
        super::reference::reference_cell(0, o, StageKind::Track, &current, &rows_of(o))
    };

    let picked = cell(&row(None, ReferenceFallback::None));
    assert_eq!(picked.mark, super::reference::ReferenceMark::Pick);
    assert_eq!(picked.text, "pick\n72\u{b0}, 83%");
    let hover = picked.hover.expect("a hover");
    assert!(hover.contains("The reference view"), "{hover}");
    assert!(hover.contains("no patch bitmap yet"), "{hover}");
    assert!(
        hover.contains("Viewing angle 71.6\u{b0}, leaning -35\u{b0}"),
        "{hover}"
    );
    assert!(hover.contains("Clipped 0.4%"), "{hover}");
    assert!(hover.contains("   90   80    -"), "{hover}");

    for (test, word, why) in [
        (
            ReferenceTest::Coverage,
            "partial",
            "of its tile is on the photograph",
        ),
        (ReferenceTest::Clipped, "clipped", "is clipped, over the 5%"),
        (
            ReferenceTest::Angle,
            "oblique",
            "sees the patch at 71.6\u{b0}, over the 65\u{b0}",
        ),
        (
            ReferenceTest::Cells,
            "ninth differs",
            "0.12 below the track's typical agreement",
        ),
        (
            ReferenceTest::Agreement,
            "agrees less",
            "more than 15 points below",
        ),
        (
            ReferenceTest::Sharpness,
            "less sharp",
            "smaller self-similarity radius",
        ),
    ] {
        let rejected = cell(&row(Some(test), ReferenceFallback::None));
        assert_eq!(rejected.mark, super::reference::ReferenceMark::None);
        assert!(
            rejected.text.starts_with(&format!("{word}\n")),
            "{}",
            rejected.text
        );
        let hover = rejected.hover.expect("a hover");
        assert!(hover.starts_with("Not the reference view: "), "{hover}");
        assert!(hover.contains(why), "{test}: {hover}");
    }

    // A row turned away for sharpness with no self-similarity radius was not
    // compared, and the hover says that rather than naming a sharper row.
    let mut unmeasured = row(Some(ReferenceTest::Sharpness), ReferenceFallback::None);
    unmeasured
        .track
        .as_mut()
        .expect("a track slot")
        .zncc_self_similarity_ellipse = None;
    let hover = cell(&unmeasured).hover.expect("a hover");
    assert!(hover.contains("has no self-similarity radius"), "{hover}");
    assert!(!hover.contains("smaller self-similarity radius"), "{hover}");

    // A dropped test is said in the hover.
    let dropped = cell(&row(None, ReferenceFallback::WithoutAngle));
    assert!(dropped
        .hover
        .expect("a hover")
        .contains("dropped the 65\u{b0} angle limit"));

    // Once the 65° limit is dropped, a row turned away by the angle sees the
    // patch edge on or from behind.
    let mut behind = row(Some(ReferenceTest::Angle), ReferenceFallback::WithoutAngle);
    behind
        .track
        .as_mut()
        .expect("a track slot")
        .viewing_angle_deg = Some(111.0);
    let hover = cell(&behind).hover.expect("a hover");
    assert!(
        hover.contains("sees the patch at 111.0\u{b0}, edge on or from behind"),
        "{hover}"
    );

    // An `out` row carries its own readings and no standing.
    let mut out = row(None, ReferenceFallback::None);
    out.verdict = Verdict::Out;
    let slot = out.track.as_mut().expect("a track slot");
    slot.reference_view = None;
    slot.pair_zncc = None;
    let out_cell = cell(&out);
    assert_eq!(out_cell.text, "-\n72\u{b0}");
    assert!(out_cell
        .hover
        .expect("a hover")
        .starts_with("Not considered for the reference view"));

    // Nothing at the cluster stage or for a track that could not be evaluated.
    let refused = crate::bench::live::Evaluation::Refused("no frame".to_string());
    let picked = row(None, ReferenceFallback::None);
    let rows = rows_of(&picked);
    assert_eq!(
        super::reference::reference_cell(0, &picked, StageKind::Track, &refused, &rows).text,
        "-"
    );
    assert_eq!(
        super::reference::reference_cell(0, &picked, StageKind::Cluster, &current, &rows).text,
        "-"
    );
}

/// The *Reference* column's marks: green for a reference the rule picks too,
/// red for one it does not with the pick marked grey in its own cell, and only
/// the pick where the track has no reference, whether its bitmap is a fused
/// mean or it has none yet.
#[test]
fn the_reference_column_marks_the_reference_and_the_rule_s_pick() {
    use super::reference::{reference_cell, ReferenceMark, ReferenceRows};
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};
    use sfmtool_core::patch::reference_view::{
        ReferenceFallback, ReferenceStanding, ReferenceTest,
    };

    let row = |image: u32, rejected_by: Option<ReferenceTest>| Observation {
        image,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: true,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            viewing_angle_deg: Some(20.0),
            pair_zncc: Some(0.9),
            reference_view: Some(ReferenceStanding {
                rejected_by,
                fallback: ReferenceFallback::None,
            }),
            ..TrackMeasurement::default()
        }),
    };
    let current = crate::bench::live::Evaluation::Current;
    let rows = [
        row(10, Some(ReferenceTest::Sharpness)),
        row(11, Some(ReferenceTest::Agreement)),
        row(12, None),
    ];
    let cells = |marks: &ReferenceRows| {
        rows.iter()
            .enumerate()
            .map(|(i, o)| reference_cell(i, o, StageKind::Track, &current, marks))
            .collect::<Vec<_>>()
    };

    // The reference is the pick: one green cell.
    let agreed = ReferenceRows {
        reference: Some(2),
        reference_image: Some(12),
        pick: Some(2),
        pick_image: Some(12),
        has_bitmap: true,
    };
    let drawn = cells(&agreed);
    assert_eq!(drawn[2].mark, ReferenceMark::Reference);
    assert!(
        drawn[2].text.starts_with("reference\n"),
        "{}",
        drawn[2].text
    );
    assert_eq!(drawn[0].mark, ReferenceMark::None);
    assert!(
        drawn[0].text.starts_with("less sharp\n"),
        "{}",
        drawn[0].text
    );
    assert_eq!(
        super::table::reference_fill(drawn[2].mark),
        Some(super::table::KEEP_ON_FILL)
    );

    // It is not: the reference is red and the pick grey in its own cell, and
    // each hover says how to accept the pick.
    let held = ReferenceRows {
        reference: Some(0),
        reference_image: Some(10),
        ..agreed
    };
    let drawn = cells(&held);
    assert_eq!(drawn[0].mark, ReferenceMark::ReferenceNotPick);
    assert!(
        drawn[0].text.starts_with("reference\n"),
        "{}",
        drawn[0].text
    );
    let hover = drawn[0].hover.as_deref().expect("a hover");
    assert!(hover.contains("held by the row's pin"), "{hover}");
    assert!(hover.contains("the row of image 12 instead"), "{hover}");
    assert!(
        hover.contains("Unpinning this row, or Set as reference"),
        "{hover}"
    );
    assert_eq!(drawn[2].mark, ReferenceMark::Pick);
    assert!(drawn[2].text.starts_with("pick\n"), "{}", drawn[2].text);
    let hover = drawn[2].hover.as_deref().expect("a hover");
    assert!(
        hover.contains("reference is the row of image 10"),
        "{hover}"
    );
    assert_eq!(drawn[1].mark, ReferenceMark::None);
    assert_ne!(
        super::table::reference_fill(ReferenceMark::ReferenceNotPick),
        super::table::reference_fill(ReferenceMark::Pick)
    );

    // A reference where the rule picks no row is not red: there is no pick to
    // accept in its place.
    let alone = ReferenceRows {
        pick: None,
        pick_image: None,
        ..held
    };
    let drawn = cells(&alone);
    assert_eq!(drawn[0].mark, ReferenceMark::ReferenceWithoutPick);
    assert!(
        drawn[0].text.starts_with("reference\n"),
        "{}",
        drawn[0].text
    );
    assert_eq!(
        super::table::reference_fill(ReferenceMark::ReferenceWithoutPick),
        None
    );
    let hover = drawn[0].hover.as_deref().expect("a hover");
    assert!(hover.contains("picks no row"), "{hover}");

    // No reference: a fused mean, or no bitmap yet, marks only the pick.
    for has_bitmap in [true, false] {
        let none = ReferenceRows {
            reference: None,
            reference_image: None,
            has_bitmap,
            ..agreed
        };
        let drawn = cells(&none);
        let marks: Vec<_> = drawn.iter().map(|c| c.mark).collect();
        assert_eq!(
            marks,
            [
                ReferenceMark::None,
                ReferenceMark::None,
                ReferenceMark::Pick
            ]
        );
        let hover = drawn[2].hover.as_deref().expect("a hover");
        let says = if has_bitmap {
            "mean of the rows"
        } else {
            "no patch bitmap yet"
        };
        assert!(hover.contains(says), "{hover}");
    }

    // Sorting by the column puts the reference first, then the pick.
    let rank = |i: usize| super::reference::reference_rank(i, &rows[i], &held);
    assert!(
        rank(0) < rank(2) && rank(2) < rank(1),
        "{:?}",
        [rank(0), rank(1), rank(2)]
    );
}

/// At the track stage the *ZNCC* cell prints the plain score against the
/// stored bitmap, with the blur-matched one after an arrow where the bitmap was
/// blurred and the two print differently, and its hover gives the blur, the
/// sharper note, the reason for a missing score and the leave-one-out reading.
#[test]
fn the_zncc_cell_prints_the_score_against_the_bitmap() {
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement, Unmeasured};

    let row = |plain: f64, matched: f64, sigma: f64, sharper: bool| Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(plain),
            zncc_middle: Some(0.61),
            blur_matched_zncc: Some(matched),
            bitmap_blur_sigma: Some(sigma),
            sharper_than_bitmap: Some(sharper),
            loo_zncc: Some(0.88),
            loo_zncc_middle: Some(0.7),
            ..TrackMeasurement::default()
        }),
    };
    let current = crate::bench::live::Evaluation::Current;
    let text = |o: &Observation| super::measurements(o, StageKind::Track, &current)[0].clone();
    let hover =
        |o: &Observation, own: bool| super::reference::zncc_hover(o.track.as_ref().unwrap(), own);

    let blurred = row(0.504, 0.531, 0.83, false);
    assert_eq!(text(&blurred), "50% \u{23f5} 53% whole\n61% mid");
    let said = hover(&blurred, false);
    assert!(
        said.contains("ZNCC with the stored patch bitmap 50.4% whole"),
        "{said}"
    );
    assert!(said.contains("blurred by 0.83 grid px"), "{said}");
    assert!(
        said.contains("Leave-one-out ZNCC 88.0% whole, 70.0% middle"),
        "{said}"
    );

    // One number where the blur leaves the printed score as it was, and where
    // the pair was read plain.
    assert_eq!(text(&row(0.5, 0.502, 0.4, false)), "50% whole\n61% mid");
    let sharper = row(0.64, 0.64, 0.0, true);
    assert_eq!(text(&sharper), "64% whole\n61% mid");
    assert!(hover(&sharper, false).contains("could replace the reference"));

    // The reference's own row reads 100%.
    let mut own = row(1.0, 1.0, 0.0, false);
    own.track.as_mut().unwrap().zncc_middle = Some(1.0);
    assert_eq!(text(&own), "100% whole\n100% mid");
    assert!(hover(&own, true).contains("track's reference"));

    // A row with no score says why, in the hover and the status cell.
    let mut unscored = row(0.5, 0.5, 0.0, false);
    let slot = unscored.track.as_mut().unwrap();
    slot.zncc = None;
    slot.zncc_middle = None;
    slot.reason = Some(Unmeasured::NoBitmap);
    assert_eq!(text(&unscored), "-");
    let reason = Unmeasured::NoBitmap.to_string();
    assert!(hover(&unscored, false).contains(&reason));
    assert_eq!(
        super::measurements(&unscored, StageKind::Track, &current)[4],
        reason
    );
    // A row the localizer read and the bitmap scored is localized.
    assert_eq!(
        super::measurements(&blurred, StageKind::Track, &current)[4],
        "localized"
    );
}

/// After a fit stores a bitmap, the row it is rendered from reads 100% in the
/// *ZNCC* column and is the one row the *Reference* column marks as the
/// reference; every other measured row prints its score against it. There is
/// no *Bitmap* column.
#[test]
fn a_fitted_track_marks_the_one_row_its_bitmap_is_rendered_from() {
    use super::reference::ReferenceMark;
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state.settle_bench_evaluation();
    state
        .start_bench_fit(id, &label)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let payload = track.track().expect("the track stage");
    assert!(payload.bitmap.is_some(), "the fit stored no bitmap");
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(!texts.iter().any(|t| t == "Bitmap"), "{texts:?}");
    let rows = panel.rows();
    let marked: Vec<usize> = rows
        .iter()
        .filter(|r| {
            matches!(
                r.reference.mark,
                ReferenceMark::Reference | ReferenceMark::ReferenceNotPick
            )
        })
        .map(|r| r.observation)
        .collect();
    match payload.reference {
        Some(r) => {
            assert_eq!(marked, [r], "{rows:?}");
            for (i, observation) in track.observations.iter().enumerate() {
                let m = observation.track.as_ref().expect("measured");
                if i == r {
                    assert_eq!(m.zncc, Some(1.0));
                    assert!(rows[i].cells[0].starts_with("100% whole"), "{rows:?}");
                } else {
                    assert!(m.zncc.is_some(), "row {i} has no score");
                }
            }
        }
        None => assert!(marked.is_empty(), "{rows:?}"),
    }
}

/// *Set as reference* is offered on a track-stage row that is `in` and has a
/// keypoint, greyed on the reference its pin holds, and the step makes the
/// row the reference: the live evaluation renders the bitmap from it.
#[test]
fn set_as_reference_is_offered_on_an_in_row_with_a_keypoint() {
    use super::table::set_reference_offer;
    let (mut state, id, label, _panel, _ctx) = on_the_bench();
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let held = track.track().and_then(|p| p.reference);
    let other = (0..track.observations.len())
        .find(|&i| Some(i) != held)
        .expect("a second row");
    assert_eq!(set_reference_offer(&track, other), Some(Ok(())));
    if let Some(held) = held {
        assert!(matches!(set_reference_offer(&track, held), Some(Err(_))));
    }

    // Not on an out row, nor on one with no keypoint.
    let mut out = sfmtool_core::bench::EditableTrack::clone(&track);
    out.observations[other].verdict = Verdict::Out;
    assert_eq!(set_reference_offer(&out, other), None);
    let mut unplaced = sfmtool_core::bench::EditableTrack::clone(&track);
    unplaced.observations[other]
        .track
        .as_mut()
        .unwrap()
        .keypoint = None;
    assert_eq!(set_reference_offer(&unplaced, other), None);

    // The step: one version, the row pinned and held as the reference, and
    // after the live evaluation the bitmap is rendered from it.
    state
        .set_bench_reference(id, &label, other)
        .expect("an in row with a keypoint can be the reference");
    let after = state.bench_track(id, &label).expect("on the bench").clone();
    assert_eq!(after.held_reference(), Some(other));
    assert!(after.observations[other].pinned);
    state.settle_bench_evaluation();
    let after = state.bench_track(id, &label).expect("on the bench").clone();
    assert_eq!(crate::bench::reference_in_use(&after), Some(other));
    assert_eq!(
        after.observations[other]
            .track
            .as_ref()
            .and_then(|m| m.zncc),
        Some(1.0)
    );
    assert_eq!(
        set_reference_offer(&after, other).map(|o| o.is_ok()),
        Some(false)
    );
}

/// *Set as reference* in a row's menu: drawn on an `in` row that is not the
/// reference and answers a click with that row, greyed on the held
/// reference, and absent on an `out` row.
#[test]
fn set_as_reference_is_drawn_in_the_row_menu_and_clicked() {
    let (mut state, id, label, _panel, _ctx) = on_the_bench();
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let held = track
        .held_reference()
        .expect("a track from a point holds its reference");
    let other = (0..track.observations.len())
        .find(|&i| i != held)
        .expect("a second row");

    let menu_at = |state: &AppState, row: usize| {
        let (mut panel, ctx) = settled(state);
        let image = track.observations[row].image as usize;
        let y = row_y(&mut panel, &ctx, state, image);
        let at = egui::pos2(400.0, y);
        open_row_menu(&mut panel, &ctx, state, at);
        let texts = painted(&mut panel, &ctx, state, vec![egui::Event::PointerMoved(at)]);
        (panel, ctx, texts)
    };

    // Offered and clicked on another `in` row.
    let (mut panel, ctx, texts) = menu_at(&state, other);
    assert!(
        texts.iter().any(|t| t == super::SET_REFERENCE_LABEL),
        "{texts:?}"
    );
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::SET_REFERENCE_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.set_reference, Some(other));

    // Greyed on the reference its pin holds: drawn, but a click does nothing.
    let (mut panel, ctx, texts) = menu_at(&state, held);
    assert!(
        texts.iter().any(|t| t == super::SET_REFERENCE_LABEL),
        "{texts:?}"
    );
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::SET_REFERENCE_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.set_reference, None);

    // Absent on an `out` row.
    state
        .set_bench_verdict(id, &label, other, Verdict::Out)
        .expect("a verdict");
    state.settle_bench_evaluation();
    let (_, _, texts) = menu_at(&state, other);
    assert!(
        !texts.iter().any(|t| t == super::SET_REFERENCE_LABEL),
        "{texts:?}"
    );
}

/// Unpinning the row that holds the reference, where the rule picks another
/// row, logs that the verdicts wait for the new bitmap rather than a verdict
/// judged against the outgoing one; the live evaluation then renders from the
/// pick and the track's reference is the pick.
#[test]
fn unpinning_the_held_reference_logs_that_the_verdicts_wait_for_the_render() {
    let (mut state, id, label, _panel, _ctx) = on_the_bench();
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let pick = track.reference_view_pick().expect("the rule picks a row");
    let other = (0..track.observations.len())
        .find(|&i| i != pick)
        .expect("a second row");
    state
        .set_bench_reference(id, &label, other)
        .expect("an in row with a keypoint");
    state.settle_bench_evaluation();

    state
        .unpin_bench_verdicts(id, &label, &[other])
        .expect("a pinned row");
    let logged = state
        .action_log
        .entries()
        .last()
        .expect("a row")
        .text
        .clone();
    assert!(
        logged.contains("waiting for the bitmap to be rendered from"),
        "{logged}"
    );
    // The old bitmap stays until the render replaces it; no row has a score.
    let pending = state.bench_track(id, &label).expect("on the bench").clone();
    assert_eq!(crate::bench::reference_in_use(&pending), Some(other));
    assert!(pending
        .observations
        .iter()
        .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none())));

    state.settle_bench_evaluation();
    let after = state.bench_track(id, &label).expect("on the bench").clone();
    assert_eq!(crate::bench::reference_in_use(&after), Some(pick));
}
#[test]
fn the_table_has_a_row_per_observation_in_index_order() {
    let (_state, _id, _label, panel, _ctx) = on_the_bench();
    let rows = panel.rows();
    assert_eq!(rows.len(), 3, "the fixture's track has three observations");
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.observation, index, "rows are not in index order");
        assert_eq!(row.verdict, Verdict::In, "a track from a point is all in");
        assert!(row.pinned, "a track from a point arrives pinned");
    }
    assert_eq!(
        rows.iter().map(|row| row.image).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn a_verdict_shows_under_the_same_observation_index() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_eq!(rows.len(), 3, "the refused observation left the table");
    assert_eq!(rows[1].observation, 1);
    assert_eq!(rows[1].verdict, Verdict::Out);
    assert!(rows[1].pinned, "a verdict set by hand is not pinned");
    assert_eq!(rows[0].verdict, Verdict::In);
    assert_eq!(rows[2].verdict, Verdict::In);
}

/// The *Keep* cells show what the bars propose, a drag of the boxes moves no
/// pinned verdict, and an unpinned row's proposal is exactly what applying the
/// bars makes it.
#[test]
fn the_thresholds_propose_for_the_rows_and_move_no_pinned_verdict() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Measure the track, so the rows have numbers for the bars to judge.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.proposal.is_some()),
        "every row is measured: {:?}",
        panel.rows()
    );
    let pinned_before = panel.rows()[1].verdict;

    // A bar nothing can clear: every measured row would be refused.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = 1.01;
    });
    let rows = panel.rows();
    assert!(
        rows.iter().all(|row| row.proposal == Some(Verdict::Out)),
        "an unreachable bar proposed a row in: {rows:?}"
    );
    assert_eq!(
        rows[1].verdict, pinned_before,
        "the proposal moved a verdict the person set"
    );

    // And an unpinned row's proposal is exactly what applying the bars does.
    state
        .apply_bench_thresholds(id, &label, panel.thresholds())
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    for row in panel.rows().iter().filter(|row| !row.pinned) {
        assert_eq!(
            Some(row.verdict),
            row.proposal,
            "applying the bars produced a verdict the Keep cell did not show"
        );
    }
    assert_eq!(
        panel.rows()[1].verdict,
        pinned_before,
        "applying the bars moved the pinned verdict"
    );
}

/// Every row draws its own rendered tile, at both stages: the patch seen from
/// that observation at the track stage, and the kernel's own grid at the
/// cluster stage.
#[test]
fn every_row_draws_a_tile_at_either_stage() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.tile),
        "a track from a committed point draws no tile: {:?}",
        panel.rows(),
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.tile),
        "the cluster stage draws no tile: {:?}",
        panel.rows(),
    );
}

/// A row drawn before its photograph is decoded gets its tile and its crop on a
/// later frame once the photograph arrives, without the track moving: a
/// photograph not decoded yet is not remembered as a row with neither.
#[test]
fn a_tile_appears_when_its_photograph_arrives_after_the_first_frame() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| !row.tile && !row.crop),
        "a row drew a tile or a crop with no photograph decoded: {:?}",
        panel.rows(),
    );

    cache_photographs(&mut state, id);
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.tile && row.crop),
        "a row drawn before its photograph arrived stayed without a tile or a crop: {:?}",
        panel.rows(),
    );
}

/// A candidate a descriptor search has just added sits where the index's warp
/// put it -- its keypoint and its seed are that pixel, and nothing has read it
/// -- and its tile is cut around **that**, which is the whole of what says
/// whether the search found the right surface, rather than wherever the bare
/// projection of the point happens to land in the photograph.
#[test]
fn a_search_candidate_draws_its_tile_where_the_seed_put_it() {
    use sfmtool_core::bench::Stage;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, label) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    state
        .start_bench_descriptor_search(id, &label, 0, None, None)
        .expect("an index is open and the image has keypoints");
    state.finish_background_task();

    let track = state
        .bench_track(id, &label)
        .expect("still on the bench")
        .clone();
    let candidate = track.observations.len() - 1;
    let row = &track.observations[candidate];
    assert!(
        matches!(
            row.provenance,
            sfmtool_core::bench::Provenance::Search { .. }
        ),
        "the search added no candidate: {:?}",
        row.provenance
    );
    assert!(
        row.track.as_ref().is_none_or(|m| m.zncc.is_none()),
        "the candidate arrived already read, so this proves nothing"
    );
    let seed = row.cluster.as_ref().expect("a searched seed").seed_position;
    let site = row.site().expect("the candidate sits somewhere");
    assert!(
        (site[0] - seed[0]).abs() < 1e-3 && (site[1] - seed[1]).abs() < 1e-3,
        "the candidate sits at {site:?}, not at the seed {seed:?}"
    );
    let image = ImageRef::new(id, row.image as usize);
    let src = state
        .cached_photograph(image.recon, image.index())
        .expect("a cached photograph");

    let node = crate::scene::node_by_id(&state.scene, id).expect("loaded");
    let recon = node.recon();
    let drawn = super::tile::image(recon, &track, candidate, &src).expect("a tile");

    // The patch cut around where the observation sits, which is the same
    // picture the row draws once a reading has written that pixel as its
    // keypoint: where an observation is, is one question however it is
    // answered.
    let frame = match &track.stage {
        Stage::Track(payload) => payload.placement.clone().expect("a fitted patch"),
        Stage::Cluster(_) => panic!("a track put on from a point is at the track stage"),
    };
    let sfmr_image = &recon.image_table.images[row.image as usize];
    let camera = &recon.image_table.cameras[sfmr_image.camera_index as usize];
    let cam_from_world = crate::scene::cam_from_world(sfmr_image);
    let sampler = super::tile::tile_sampler(recon, &track, candidate);
    let at_the_seed =
        super::patch::patch_color_image(&frame, camera, &cam_from_world, Some(site), &src, sampler);
    assert_eq!(drawn, at_the_seed, "the tile is not cut around the seed");

    let at_the_projection =
        super::patch::patch_color_image(&frame, camera, &cam_from_world, None, &src, sampler);
    assert_ne!(
        drawn, at_the_projection,
        "the tile is the point's own projection rather than the sighting's place"
    );
}

#[test]
fn the_cells_follow_the_stage_the_track_is_in() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // The five cells are ZNCC, seed shift, projection error, self-similarity
    // and status. At the track stage the projection error has
    // numbers behind it; at the cluster stage there is no geometry behind an
    // observation and it is absent.
    assert_eq!(panel.rows()[0].cells[2], "-", "nothing has measured it yet");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_ne!(rows[0].cells[0], "-", "the ZNCC column is unmeasured");
    let (whole, middle) = rows[0].cells[0]
        .split_once('\n')
        .expect("the ZNCC cell shows the whole over the middle reading");
    let percent = |line: &str, part: &str| {
        let number = line
            .strip_suffix(part)
            .and_then(|rest| rest.strip_suffix("% "))
            .unwrap_or_else(|| panic!("{line:?} is not a percent named {part}"));
        assert!(number.parse::<f64>().is_ok(), "{line}");
    };
    percent(whole, "whole");
    percent(middle, "mid");
    assert_eq!(rows[0].cells[2], "-", "a cluster has no point to project");
    // Beside the ZNCC cell, the row draws its grid.
    assert!(rows[0].grids.zncc.is_some(), "no ZNCC grid drawn");
    // And beside the self-similarity cell, its grid and its ellipses.
    assert!(
        rows[0].cells[3].contains(" px whole\n") && rows[0].cells[3].ends_with(" px mid"),
        "{}",
        rows[0].cells[3]
    );
    assert!(
        rows[0].grids.radius.is_some(),
        "no self-similarity grid drawn"
    );
    assert!(
        rows[0].grids.radius_ellipse.is_some(),
        "no self-similarity ellipses"
    );
    // And the whole tile's surface plot.
    assert!(
        rows[0].self_similarity_plot,
        "no self-similarity surface plot"
    );
}

/// A row the reading refused to widen its window for says so in the Status
/// cell, in the reading's own sentence.
///
/// The seed's distance from the projection is what sizes the search window, and
/// the tile that window renders costs its square, so a sighting placed a long
/// way from the point is left out of the round with a reason rather than
/// allocated for. The cell is where the person meets that decision.
#[test]
fn a_row_seeded_far_from_the_projection_says_so_in_the_status_cell() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, 5),
            &crate::bench::Seed::Pixel {
                pixel: [24.0, 24.0],
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    let status = rows.last().expect("the row just added").cells[4].clone();
    assert!(
        status.contains("beyond the 64 px bound"),
        "the cell names the bound the seed passed: {status}"
    );
    assert!(
        rows.iter().filter(|row| row.cells[0] != "-").count() >= 2,
        "and the rows that could be read still were: {rows:?}"
    );
}

/// Evaluation is live, so there is no *Evaluate* button: the toolbar offers
/// *Fit*, which moves the track, and says where the evaluation of the track as
/// it stands is. There is no search radius control: the evaluation looks
/// within the shift bar.
#[test]
fn the_toolbar_offers_the_fit_and_says_where_the_evaluation_stands() {
    let (mut state, _, _, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        !texts.iter().any(|t| t == "search px"),
        "the search px box is still drawn: {texts:?}"
    );
    for label in ["Fit", super::MAX_SHIFT_LABEL, super::EVALUATING_LABEL] {
        assert!(
            texts.iter().any(|t| t == label),
            "{label} is not drawn: {texts:?}"
        );
    }
    assert!(
        !texts.iter().any(|t| t == "Evaluate"),
        "the Evaluate button is still drawn: {texts:?}"
    );

    state.settle_bench_evaluation();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts.iter().any(|t| t == super::EVALUATED_LABEL),
        "an evaluated track does not say so: {texts:?}"
    );
    assert!(!texts.iter().any(|t| t == super::EVALUATING_LABEL));
}

/// While an evaluation of the track's current inputs is on its way the rows do
/// not present the last evaluation's numbers as the track's: each status cell
/// says the row is being evaluated, until the evaluation lands.
#[test]
fn the_rows_read_evaluating_until_the_evaluation_of_the_current_inputs_lands() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert_eq!(
        panel.evaluation(),
        &crate::bench::live::Evaluation::Evaluating
    );
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] == super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );

    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.evaluation(), &crate::bench::live::Evaluation::Current);
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] != super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );

    // A step on the track puts every row back to evaluating.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] == super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );
}

/// A track that cannot be evaluated says why where the *Evaluate* button's
/// refusal used to be, and its rows print no numbers at all.
#[test]
fn a_track_that_cannot_be_evaluated_shows_the_reason_instead_of_values() {
    let (mut state, id) = state();
    let (bench, _) = sfmtool_core::bench::Bench::new().put(
        "bearing",
        sfmtool_core::bench::BenchItem::Track(std::sync::Arc::new(frameless_bearing_track())),
    );
    let index = state.node_index(id).expect("loaded");
    state.scene[index]
        .history
        .push_bench(std::sync::Arc::new(bench), "Put a bearing on the bench");
    state.focus_put_item(id, "bearing");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();

    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    let why = match panel.evaluation() {
        crate::bench::live::Evaluation::Refused(why) => why.clone(),
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(why.starts_with("Cannot evaluate bearing"), "{why}");
    assert!(texts.contains(&why), "the reason is not drawn: {texts:?}");
}

/// Edited mode draws the focused item and nothing else on the bench: no row of
/// item tabs, so the labels of the other items appear nowhere in what the frame
/// painted. The bench as a list is the Scene tree's.
#[test]
fn edit_mode_draws_the_focused_item_and_no_item_tabs() {
    let (mut state, id) = state();
    let first = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let second = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor")
        .label;
    let third = state
        .start_bench_cluster(
            ImageRef::new(id, 1),
            &crate::bench::Seed::Pixel {
                pixel: [60.0, 40.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor")
        .label;
    state.focus_bench_item(id, &second).expect("on the bench");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);

    // The cluster is the focused one, and it has the one observation it was
    // started with.
    assert_eq!(panel.rows().len(), 1);

    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        texts.contains(&second),
        "the focused item's header: {texts:?}"
    );
    for label in [&first, &third] {
        assert!(
            !texts.iter().any(|t| t.contains(label.as_str())),
            "{label} is not the focused item and was painted: {texts:?}"
        );
    }
}

/// A row is an *observation*, so clicking one names both an image and a place
/// in it: the response carries the pixel the Image Detail panel's bench layer
/// draws that observation's mark at, so the two cannot disagree about where the
/// view should land. It also picks the observation out, which the panel reports
/// rather than holds: the selection is the bench's.
#[test]
fn clicking_a_row_selects_its_image_and_reveals_the_observation() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let expected = crate::bench::observation_pixel(&track.observations[1])
        .expect("a track from a committed point carries its keypoints");

    // The second row, whose observation is in image 1.
    let y = row_y(&mut panel, &ctx, &state, 1);
    let response = at_pointer(&mut panel, &ctx, &state, egui::pos2(400.0, y + 8.0), true);

    assert_eq!(response.select_image, Some(1));
    assert_eq!(response.reveal_feature, Some(expected));
    assert_eq!(
        response.request_camera_view, None,
        "a single click asked for camera view"
    );
    assert_eq!(response.pick_row, Some((1, false)));

    let (observation, extend) = response.pick_row.expect("a pick");
    state.pick_bench_observation(id, &label, observation, extend);
    assert_eq!(state.selected_bench_observations(id, &label), [1]);
    state.pick_bench_observation(id, &label, 0, true);
    assert_eq!(state.selected_bench_observations(id, &label), [0, 1]);
    state.pick_bench_observation(id, &label, 1, true);
    assert_eq!(state.selected_bench_observations(id, &label), [0]);
}

/// A double-click on a row enters camera view for its image, as a Viewed-mode
/// row's does: the rows of both modes are observations of one track. The
/// row's observation goes with it, for the view to turn toward.
#[test]
fn double_clicking_a_row_asks_for_camera_view() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let y = row_y(&mut panel, &ctx, &state, 1);
    let pos = egui::pos2(400.0, y + 8.0);
    run_frame_with(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(pos)],
    );
    let mut events = vec![egui::Event::PointerMoved(pos)];
    for _ in 0..2 {
        for pressed in [true, false] {
            events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
    }
    let response = run_frame_with(&mut panel, &ctx, &state, events);
    assert_eq!(response.request_camera_view, Some(1));
    assert_eq!(response.select_image, Some(1));
    assert!(
        response.reveal_feature.is_some(),
        "the double-click did not say where the observation is"
    );
}

/// *Lock* starts ticked, which is the dot moving the whole patch, and toggling
/// it is a tool setting rather than a step: no version, no gesture reported,
/// and the box holds what it was set to from one frame to the next.
#[test]
fn the_lock_starts_ticked_and_toggling_it_pushes_no_version() {
    let (state, id, _, mut panel, ctx) = on_the_bench();
    assert!(TrackBody::new().lock(), "a new panel starts locked");
    assert!(panel.lock());
    let versions = state.node(id).expect("loaded").history.versions().len();

    let at = menu_entry_pos(&mut panel, &ctx, &state, LOCK_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(!panel.lock(), "a click on the box clears it");
    assert_eq!(
        TrackBodyResponse {
            mode: Some(super::BodyMode::Edited),
            has_pointer: response.has_pointer,
            hovered_image: response.hovered_image,
            ..TrackBodyResponse::default()
        },
        response,
        "toggling the lock asked the dock for a step",
    );
    run_frame(&mut panel, &ctx, &state);
    assert!(!panel.lock(), "the box keeps what it was set to");
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        versions,
        "toggling the lock pushed a version",
    );

    at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(panel.lock(), "a second click ticks it again");
}

/// At the cluster stage every handle is one sighting's already, so the box is
/// drawn but greyed: a click on it changes nothing.
#[test]
fn the_lock_is_greyed_at_the_cluster_stage() {
    let (mut state, id) = state();
    state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor");
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts.iter().any(|text| text == LOCK_LABEL),
        "the box is drawn at the cluster stage too: {texts:?}"
    );
    let at = menu_entry_pos(&mut panel, &ctx, &state, LOCK_LABEL);
    at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(panel.lock(), "a greyed box took the click");
}

/// Outside a drag the boxes hold no value of their own: whatever the panel
/// had is replaced by the track's bars on the next frame.
#[test]
fn the_boxes_show_the_focused_item_s_bars_outside_a_drag() {
    let (state, id, label, mut panel, ctx) = on_the_bench();
    panel.thresholds.min_zncc = 0.5;
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(panel.thresholds(), &track.thresholds);
}

/// The boxes show the **focused item's** bars, whoever moved them: a step
/// taken over the wire moves the track's, and the panel that paints the rows by
/// them has to be showing the same numbers or it proposes a rule the track does
/// not hold. An undo moves them back.
#[test]
fn the_boxes_follow_the_focused_item_s_own_thresholds() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Handed to the bars, so the painting below is every row's proposal.
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    let before = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .clone();

    let bars = Thresholds {
        min_zncc: 0.94,
        geometry_search_min_relative_zncc: 0.62,
        ..Thresholds::default()
    };
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(panel.thresholds(), &track.thresholds);
    assert_eq!(panel.thresholds().min_zncc, 0.94);
    // And the proposals are the track's rule rather than the box's old one.
    assert_eq!(
        panel
            .judged
            .iter()
            .map(|judged| judged.map(|j| j.proposal))
            .collect::<Vec<_>>(),
        sfmtool_core::bench::apply_thresholds(track)
            .0
            .observations
            .iter()
            .map(|o| Some(o.verdict))
            .collect::<Vec<_>>()
    );

    state.undo(id).expect("the thresholds step undoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &before, "an undo moves the boxes back");
    state.redo(id).expect("and redoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.94, "a redo moves them again");
}

/// The pointer events of one drag, as one list per frame: hover, press, two
/// moves and the release. egui resolves a press against the rects the previous
/// frame registered, so each event gets a frame of its own.
fn drag_frames(from: egui::Pos2, to: egui::Pos2) -> Vec<Vec<egui::Event>> {
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let mid = from + (to - from) * 0.5;
    vec![
        vec![egui::Event::PointerMoved(from)],
        vec![button(from, true)],
        vec![egui::Event::PointerMoved(mid)],
        vec![egui::Event::PointerMoved(to)],
        vec![button(to, false)],
        vec![],
    ]
}

/// Where to press to drag the box whose label is `label`, painted at `named`.
/// The geometry search box comes one gap to the right of its label; a box of
/// the threshold row comes one gap to the left of the unit written after it.
pub(super) fn box_point(label: &str, named: egui::Rect) -> egui::Pos2 {
    let spacing = egui::Spacing::default();
    let to_centre = spacing.item_spacing.x + 0.5 * spacing.interact_size.x;
    let x = if label == super::GEOMETRY_SEARCH_LABEL {
        named.right() + to_centre
    } else {
        named.left() - to_centre
    };
    egui::pos2(x, named.center().y)
}

/// Drag the shift box `by` points to the right (left when negative),
/// applying each frame's response the way the dock does, and hand back every
/// frame's response.
fn drag_max_shift(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &mut AppState,
    id: ReconId,
    label: &str,
    by: f32,
) -> Vec<TrackBodyResponse> {
    let texts = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    });
    let named = texts
        .iter()
        .find(|t| t.text == super::MAX_SHIFT_LABEL)
        .expect("the max shift box is drawn")
        .rect;
    let start = box_point(super::MAX_SHIFT_LABEL, named);
    let mut responses = Vec::new();
    for events in drag_frames(start, start + egui::vec2(by, 0.0)) {
        let response = run_frame_with(panel, ctx, state, events);
        if let Some(bars) = response.apply_thresholds.as_ref() {
            state
                .apply_bench_thresholds(id, label, bars)
                .expect("on the bench");
        }
        responses.push(response);
    }
    responses
}

fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// A box applies on its release: the drag's frames push nothing, the
/// release pushes exactly one version with one Action Log row, and the
/// track's bar is where the box was let go.
#[test]
fn releasing_a_threshold_box_applies_it_as_one_version() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let before = versions(&state, id);
    let bar = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .max_shift_px;
    assert_eq!(bar, sfmtool_core::bench::BENCH_MAX_SHIFT_PX);
    state.action_log.clear();

    let responses = drag_max_shift(&mut panel, &ctx, &mut state, id, &label, 100.0);
    let applied: Vec<&Thresholds> = responses
        .iter()
        .filter_map(|r| r.apply_thresholds.as_ref())
        .collect();
    assert_eq!(applied.len(), 1, "one release, one application");
    assert_eq!(versions(&state, id), before + 1, "one version for the drag");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_ne!(track.thresholds.max_shift_px, bar, "the drag moved the bar");
    assert_eq!(&track.thresholds, applied[0]);
    let rows: Vec<String> = state.action_log.entries().map(|e| e.text.clone()).collect();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].starts_with("Applied the thresholds to"), "{rows:?}");

    // The panel shows the track's bar after the release.
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &track.thresholds);

    // And an undo takes the bar and the box back together.
    state.undo(id).expect("undoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().max_shift_px, bar);
}

/// There is no button left to forget to press: the boxes are the whole
/// gesture.
#[test]
fn there_is_no_apply_thresholds_button() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        !texts.iter().any(|t| t == "Apply thresholds"),
        "the button is gone: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::MAX_SHIFT_LABEL),
        "{texts:?}"
    );
}

/// Each box of the threshold row stands in the column of the readings it
/// judges, under that column's heading and before the next one: the two ZNCC
/// bars stacked under *ZNCC*, whole over mid as the cell prints them, the
/// self-similarity bar under *Self-similarity* and the shift bar under
/// *Shift*. The geometry search bar judges no column and stays above the
/// table.
#[test]
fn each_threshold_box_stands_under_the_heading_of_what_it_judges() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    });
    let rect = |text: &str| {
        texts
            .iter()
            .find(|t| t.text == text)
            .unwrap_or_else(|| panic!("{text:?} is not drawn"))
            .rect
    };
    let heading = |text: &str| rect(text).left();
    for (label, under, before) in [
        (super::MIN_ZNCC_LABEL, "ZNCC", "Self-similarity"),
        (super::MIN_ZNCC_MIDDLE_LABEL, "ZNCC", "Self-similarity"),
        (
            super::MAX_SELF_SIMILARITY_LABEL,
            "Self-similarity",
            "Proj. err",
        ),
    ] {
        let press = box_point(label, rect(label));
        assert!(
            heading(under) <= press.x && rect(label).right() <= heading(before),
            "{label:?} is not in the {under:?} column"
        );
        assert!(
            rect(label).top() > rect(under).bottom(),
            "{label:?} is not under its heading"
        );
    }
    // The projection error and the shift boxes are both followed by `px`, in
    // that order from the left.
    assert_eq!(super::MAX_PROJECTION_ERROR_LABEL, super::MAX_SHIFT_LABEL);
    let mut px: Vec<egui::Rect> = texts
        .iter()
        .filter(|t| t.text == super::MAX_SHIFT_LABEL)
        .map(|t| t.rect)
        .collect();
    px.sort_by(|a, b| a.left().total_cmp(&b.left()));
    assert_eq!(px.len(), 2, "{px:?}");
    for (label, under, before) in [(px[0], "Proj. err", "Shift"), (px[1], "Shift", "Zoom")] {
        assert!(
            heading(under) <= label.left() && label.right() <= heading(before),
            "a px box is not in the {under:?} column"
        );
        assert!(label.top() > rect(under).bottom(), "not under {under:?}");
    }
    let (whole, mid) = (
        rect(super::MIN_ZNCC_LABEL),
        rect(super::MIN_ZNCC_MIDDLE_LABEL),
    );
    assert!(mid.top() >= whole.bottom(), "mid is not under whole");
    assert!(
        rect(super::GEOMETRY_SEARCH_LABEL).bottom() < rect("ZNCC").top(),
        "the geometry search bar is not above the table"
    );
}

/// The bar a box was let go at is the bar the next *Fit* bounds its walk
/// by, and a kept-at-seed row then offers *Accept walk*, which moves the
/// keypoint to the walked pixel as one version.
#[test]
fn a_fit_after_a_release_uses_the_new_bar_and_accept_walk_moves_the_keypoint() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // A bar of zero: any move at all is a walk past it.
    drag_max_shift(&mut panel, &ctx, &mut state, id, &label, -400.0);
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .thresholds
            .max_shift_px,
        0.0
    );

    // Before the fit no row has a walk to accept.
    let menu = row_menu(&mut panel, &ctx, &state);
    assert!(
        !menu.iter().any(|t| t == super::ACCEPT_WALK_LABEL),
        "{menu:?}"
    );

    state
        .start_bench_fit(id, &label)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let walked: Vec<usize> = (0..track.observations.len())
        .filter(|&i| {
            track.observations[i]
                .track
                .as_ref()
                .is_some_and(|m| m.walked_to.is_some())
        })
        .collect();
    assert!(
        !walked.is_empty(),
        "a zero bar keeps every moved sighting at its seed"
    );

    // Each kept-at-seed row offers the entry; the others do not.
    for (index, observation) in track.observations.iter().enumerate() {
        let offered = super::table::accepted_walk(observation).is_some();
        assert_eq!(offered, walked.contains(&index), "row {index}");
    }
    // A fresh panel, so the menu opened above is not lying over the rows.
    let (mut panel, ctx) = settled(&state);
    let first = walked[0];
    let image = track.observations[first].image as usize;
    let y = row_y(&mut panel, &ctx, &state, image);
    let at = egui::pos2(400.0, y);
    open_row_menu(&mut panel, &ctx, &state, at);
    let texts = painted(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(at)],
    );
    assert!(
        texts.iter().any(|t| t == super::ACCEPT_WALK_LABEL),
        "{texts:?}"
    );
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::ACCEPT_WALK_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.accept_walk, Some(first));

    let to = track.observations[first]
        .track
        .as_ref()
        .and_then(|m| m.walked_to)
        .expect("a walked pixel");
    let before = versions(&state, id);
    state
        .accept_bench_walk(id, &label, first)
        .expect("a walk to accept");
    assert_eq!(versions(&state, id), before + 1);
    let accepted = state.bench_track(id, &label).expect("on the bench");
    let row = &accepted.observations[first];
    assert!(row.pinned, "an accepted walk is a hand placement");
    let m = row.track.as_ref().expect("a track slot");
    assert_eq!(m.keypoint, Some([to[0] as f32, to[1] as f32]));
    assert_eq!(m.walked_to, None, "the walk is spent");
    assert!(super::table::accepted_walk(row).is_none());
}

/// A second track has its own bars, so focusing it moves the boxes.
#[test]
fn a_change_of_focused_item_reseats_the_boxes() {
    let (mut state, id, first, mut panel, ctx) = on_the_bench();
    state
        .apply_bench_thresholds(
            id,
            &first,
            &Thresholds {
                min_zncc: 0.94,
                ..Thresholds::default()
            },
        )
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.94);

    let second = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor")
        .label;
    assert_ne!(second, first);
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &Thresholds::default());
}

#[test]
fn the_column_headings_are_drawn_outside_the_scrolling_rows() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    let find = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .unwrap_or_else(|| {
                let all: Vec<&str> = painted.iter().map(|p| p.text.as_str()).collect();
                panic!("{text:?} was not painted: {all:?}")
            })
    };

    // The rows are clipped to the scroll area's viewport; a heading drawn
    // above that viewport is one that cannot scroll out of it. The name cell
    // is the row text furthest from any widget, so it is the one asked.
    let name = find("image_000.jpg");
    for (_, heading, _) in super::table::ColumnLayout::new().headers(super::BodyMode::Edited) {
        let painted = find(heading);
        assert!(
            painted.clip != name.clip,
            "the {heading:?} heading shares the rows' clip rectangle, so it scrolls with them",
        );
        assert!(
            painted.rect.max.y <= name.clip.min.y,
            "the {heading:?} heading sits inside the scrolling rows ({:?} against {:?})",
            painted.rect,
            name.clip,
        );
    }
}

/// Every heading says what its column holds, and the ZNCC heading says what
/// its two numbers are.
#[test]
fn every_heading_carries_hover_text() {
    for (_, heading, tip) in super::table::ColumnLayout::new().headers(super::BodyMode::Edited) {
        assert!(!tip.is_empty(), "the {heading:?} heading has no hover text");
    }
    let tip = super::table::ZNCC_TIP;
    assert!(
        tip.contains("whole patch") && tip.contains("middle"),
        "{tip}"
    );
    assert!(tip.contains("percent"), "{tip}");
    assert!(
        tip.contains("stored patch bitmap") && tip.contains("blur-matched"),
        "{tip}"
    );
}

#[test]
fn a_heading_stands_at_the_x_offset_its_column_is_drawn_at() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    let x_of = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .map(|painted| painted.rect.min.x)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
    };
    // Moving the heading out of the scroll area must not have moved it
    // sideways: the scroll bar takes width off the right edge, and every
    // column is measured from the left one.
    assert_eq!(
        x_of("Name"),
        x_of("image_000.jpg"),
        "the Name heading no longer stands over the name column",
    );
}

// ── The row's own menu and the SIFT index behind it ─────────────────────

/// The panel is about one track: nothing above the table names the node's
/// index, which lives on the Scene tree's own row.
#[test]
fn the_panel_draws_no_sift_index_row() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    for absent in ["Descriptor index", "SIFT Index", "Open..."] {
        assert!(
            !texts.iter().any(|t| t == absent),
            "{absent} is still drawn above the table: {texts:?}"
        );
    }
}

/// With no index beside the node, the entry a row offers is the build, because
/// a person who finds the search missing has no reason to look anywhere else.
#[test]
fn a_row_s_menu_offers_the_build_when_the_node_has_no_index() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts
            .iter()
            .any(|t| t == crate::index_files::BUILD_INDEX_FILES),
        "the build entry is not in the row's menu: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t == super::SEARCH_DESCRIPTORS_LABEL),
        "the search is offered with no index to run it against: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "geometry search wrongly depends on the SIFT index: {texts:?}"
    );
}

/// With a current index the entry is the search itself, live.
#[test]
fn a_row_s_menu_offers_the_search_against_a_current_index() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    let (mut panel, ctx) = settled(&state);

    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t == super::SEARCH_DESCRIPTORS_LABEL),
        "the search entry is not in the row's menu: {texts:?}"
    );
    assert!(
        !texts
            .iter()
            .any(|t| t == crate::index_files::REBUILD_INDEX_FILES),
        "a current index is offered a rebuild: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "the track-stage geometry entry is absent: {texts:?}"
    );
}

#[test]
fn geometry_search_is_track_stage_only_and_routes_its_own_response() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "the geometry entry is absent at the track stage: {texts:?}"
    );

    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::SEARCH_GEOMETRY_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.search_geometry, Some(0));
    assert_eq!(
        response.search_descriptors, None,
        "the geometry entry was routed as a SIFT query"
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("the framed track downgrades");
    state.finish_background_task();
    let (mut panel, ctx) = settled(&state);
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        !texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "a cluster-stage row offers geometry search: {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t == crate::index_files::BUILD_INDEX_FILES),
        "stage gating accidentally removed the SIFT/build action: {texts:?}"
    );
}

/// With a stale one the entry is the rebuild, and choosing it starts the build
/// and no search.
#[test]
fn a_row_s_menu_offers_the_rebuild_when_the_index_is_stale_and_starts_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    // The features of one image, extracted again after the build.
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
    state.open_sift_index(id, path).expect("it opens");
    assert_eq!(
        state.sift_index_state(id),
        crate::index_files::IndexFileState::Stale
    );

    let (mut panel, ctx) = settled(&state);
    let y = row_y(&mut panel, &ctx, &state, 0);
    let at = egui::pos2(400.0, y);
    open_row_menu(&mut panel, &ctx, &state, at);
    let texts = painted(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(at)],
    );
    assert!(
        texts
            .iter()
            .any(|t| t == crate::index_files::REBUILD_INDEX_FILES),
        "the rebuild entry is not in the row's menu: {texts:?}"
    );

    // Clicking it asks for the build, and asks for no search.
    let entry = menu_entry_pos(
        &mut panel,
        &ctx,
        &state,
        crate::index_files::REBUILD_INDEX_FILES,
    );
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert!(response.build_index_files, "the entry started no build");
    assert_eq!(
        response.search_descriptors, None,
        "the build does not run the search when it finishes, and does not run it now"
    );
}

/// *Duplicate* is the toolbar's own way to a second patch over neighbouring
/// ground: one version, a second item on the bench, and the copy focused.
#[test]
fn duplicate_puts_a_second_item_on_the_bench_and_focuses_it() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        texts.iter().any(|t| t == "Duplicate"),
        "the toolbar does not offer it: {texts:?}"
    );

    let before = state.node(id).expect("loaded").history.versions().len();
    let copy = state
        .duplicate_bench_item(id, &label)
        .expect("the label is on the bench");
    assert_eq!(copy, format!("{label} copy"));
    let node = state.node(id).expect("loaded");
    assert_eq!(
        node.history.versions().len(),
        before + 1,
        "one gesture, one version"
    );
    assert_eq!(
        node.history.versions().last().expect("a version").label,
        format!("Duplicated {label} as {copy}")
    );
    let bench = state.bench(id).expect("a loaded node has a bench");
    assert_eq!(bench.len(), 2, "the bench holds the original and the copy");
    assert_eq!(state.focused_item_label(id), Some(copy.as_str()));
    assert_eq!(
        state.bench_track(id, &copy).and_then(|track| track.origin),
        None,
        "a copy commits as a creation"
    );

    // And the panel shows the copy, and nothing of the original.
    run_frame(&mut panel, &ctx, &state);
    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        !texts.contains(&label),
        "the original was painted beside the copy: {texts:?}"
    );
    // The copy is the one being drawn: its header says it commits as a
    // creation, because it has no origin.
    assert!(texts.contains(&copy), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("new")),
        "the copy's header does not say it creates: {texts:?}"
    );
    assert_eq!(panel.rows().len(), 3, "the copy carries every sighting");
}

/// A track-stage track standing on a bearing: the payload a `w = 0` point
/// arrives on the bench as.
fn bearing_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{EditableTrack, Stage, TrackPayload};
    use sfmtool_core::patch::cloud::OrientedPatch;

    let direction = nalgebra::Point3::new(0.0, 0.0, 1.0);
    let mut track = EditableTrack::empty_cluster();
    track.stage = Stage::Track(TrackPayload {
        position: Some(direction),
        // The track's own flag is what the header reads; the frame's `w` agrees
        // with it wherever a frame exists.
        at_infinity: true,
        placement: Some(OrientedPatch::from_infinity_direction(
            direction,
            nalgebra::Vector3::y(),
            [0.03, 0.03],
        )),
        condition_number: Some(68_848.0),
        ..TrackPayload::default()
    });
    track
}

/// The same track standing on a place.
fn position_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{Stage, TrackPayload};
    use sfmtool_core::patch::cloud::OrientedPatch;

    let center = nalgebra::Point3::new(1.0, 2.0, 3.0);
    let mut track = bearing_track();
    track.stage = Stage::Track(TrackPayload {
        position: Some(center),
        placement: Some(OrientedPatch::from_center_normal(
            center,
            nalgebra::Vector3::z(),
            nalgebra::Vector3::y(),
            [0.1, 0.1],
        )),
        ..TrackPayload::default()
    });
    track
}

/// Everything the header and the stage's line under it painted for `track`,
/// joined.
fn header_text(track: &sfmtool_core::bench::EditableTrack) -> String {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        ..Default::default()
    };
    crate::test_support::painted_texts(&ctx, input, |ui| {
        super::show_header(ui, "item", track, None, None);
        super::show_headline(ui, track);
    })
    .join(" | ")
}

/// A track at infinity carries the infinity mark left of its label, and a
/// track on a place does not.
#[test]
fn the_header_marks_a_track_at_infinity() {
    let bearing = header_text(&bearing_track());
    assert!(
        bearing.starts_with(crate::track_view::INFINITY),
        "the mark is not first in the header: {bearing}"
    );
    let position = header_text(&position_track());
    assert!(
        !position.contains(crate::track_view::INFINITY),
        "a track on a place carries the mark: {position}"
    );
}

/// The *Split off* entry counts its rows in the singular for one.
#[test]
fn the_split_entry_counts_one_row_in_the_singular() {
    assert_eq!(super::split_entry_text(1), "Split off 1 row");
    assert_eq!(super::split_entry_text(3), "Split off 3 rows");
}

/// The headline gives a finite track's condition number and leaves it off a
/// track at infinity, where it measures the finite triangulation that failed
/// rather than the bearing, and so reads as a failed fit.
#[test]
fn the_headline_leaves_the_condition_number_off_a_bearing() {
    use sfmtool_core::bench::Stage;

    let bearing = header_text(&bearing_track());
    assert!(bearing.contains("at infinity"), "{bearing}");
    assert!(!bearing.contains("condition"), "{bearing}");

    let mut place = position_track();
    if let Stage::Track(payload) = &mut place.stage {
        payload.condition_number = Some(42.0);
    }
    let place = header_text(&place);
    assert!(place.contains("condition 42.0"), "{place}");
}

/// The track's own patch sits left of the toolbar, under the header, and the
/// controls start to its right. Before a fit fuses the observations the
/// demo's track, from a reconstruction that stores no bitmaps, has none and
/// the slot is empty; after it the slot shows the patch bitmap.
#[test]
fn the_track_s_patch_sits_left_of_the_toolbar() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let bitmap = |state: &AppState| {
        let track = state.bench_track(id, &label).expect("on the bench");
        let Stage::Track(payload) = &track.stage else {
            panic!("a track put on from a point is at the track stage");
        };
        payload.bitmap.is_some()
    };
    assert!(!bitmap(&state), "the demo stores bitmaps now");
    assert!(
        matches!(panel.track_patch, Some(None)),
        "a track with no bitmap drew one"
    );
    state
        .start_bench_fit(id, &label)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    assert!(bitmap(&state), "the fit fused no bitmap");
    let painted = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    });
    assert!(
        matches!(panel.track_patch, Some(Some(_))),
        "the track's patch was not uploaded"
    );
    let size = crate::track_view::body::STORED_PATCH_SIZE;
    let x_of = |text: &str| {
        painted
            .iter()
            .find(|p| p.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .min
            .x
    };
    let label_x = x_of(&label);
    assert!(
        x_of("Fit") >= label_x + size,
        "the toolbar overlaps the patch"
    );
    assert!(
        x_of("Crop") < label_x + size,
        "the table's first heading moved right with the toolbar"
    );
}

/// At the cluster stage the slot shows the template once one is cut, and is
/// an empty frame before that.
#[test]
fn the_cluster_stage_shows_its_template_in_the_slot() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    let Stage::Cluster(payload) = &track.stage else {
        panic!("the track moved to the cluster stage");
    };
    assert_eq!(
        payload.template.is_some(),
        matches!(panel.track_patch, Some(Some(_))),
        "the slot does not follow whether a template is cut"
    );
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    let Stage::Cluster(payload) = &track.stage else {
        panic!("still the cluster stage");
    };
    assert!(payload.template.is_some(), "the evaluation cut no template");
    assert!(
        matches!(panel.track_patch, Some(Some(_))),
        "the template is not shown"
    );
}

/// A patch bitmap of one, three or four channels becomes an opaque picture of
/// its colours, and an all-zero bitmap, which is how no patch is written,
/// becomes none.
#[test]
fn a_stored_patch_image_reads_any_channel_count() {
    use ndarray::Array3;

    let image = |channels: usize, fill: &[u8]| {
        let bitmap = Array3::from_shape_fn((2, 3, channels), |(_, _, c)| fill[c]);
        super::patch::stored_patch_image(bitmap.view())
    };
    let grey = image(1, &[40]).expect("a grey patch");
    assert_eq!(grey.size, [3, 2]);
    assert_eq!(grey.pixels[0], egui::Color32::from_rgb(40, 40, 40));
    let rgb = image(3, &[10, 20, 30]).expect("an RGB patch");
    assert_eq!(rgb.pixels[5], egui::Color32::from_rgb(10, 20, 30));
    let rgba = image(4, &[10, 20, 30, 7]).expect("an RGBA patch");
    assert_eq!(
        rgba.pixels[0],
        egui::Color32::from_rgb(10, 20, 30),
        "the confidence channel is not dropped for an opaque alpha"
    );
    assert!(image(4, &[0, 0, 0, 0]).is_none(), "an empty patch drew");
}

/// A bearing and a position are the same three numbers under different rules,
/// so the word in front of them is what tells a reader which they are looking
/// at. Printing a bearing as a position reads as a point one unit from the
/// world origin, which is the one thing it is not.
#[test]
fn the_header_names_a_bearing_a_bearing_and_a_position_a_position() {
    let bearing = header_text(&bearing_track());
    assert!(
        bearing.contains("Bearing (0.000, 0.000, 1.000)"),
        "the header should name the direction a bearing: {bearing}"
    );
    assert!(
        bearing.contains("at infinity"),
        "and say so in the panel's own words: {bearing}"
    );
    assert!(
        !bearing.contains("Position ("),
        "a bearing is not a position: {bearing}"
    );

    let position = header_text(&position_track());
    assert!(
        position.contains("Position (1.000, 2.000, 3.000)"),
        "{position}"
    );
    assert!(
        !position.contains("at infinity") && !position.contains("Bearing ("),
        "{position}"
    );
}

/// A sighting the fit refused to walk did **not** move where the correlation
/// wanted it, which is the one thing about the row a person reading "localized"
/// would get wrong. So the Status cell says it, and says how far.
#[test]
fn a_sighting_kept_at_its_seed_says_so_in_the_status_cell() {
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};

    let walked = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.41),
            walked_px: Some(19.4),
            ..TrackMeasurement::default()
        }),
    };
    let current = crate::bench::live::Evaluation::Current;
    let cells = super::measurements(&walked, StageKind::Track, &current);
    assert_eq!(cells[4], "walked 19 grid px, kept at seed");
    // With the ZNCC the fit scored at the walked peak, where it scored one.
    let mut scored = walked.clone();
    let slot = scored.track.as_mut().expect("a track slot");
    slot.walked_zncc = Some(0.873);
    slot.walked_zncc_middle = Some(0.412);
    assert_eq!(
        super::measurements(&scored, StageKind::Track, &current)[4],
        "walked 19 grid px (leave-one-out ZNCC 87% / 41% there), kept at seed"
    );

    // The same row without the flag is the ordinary scored row.
    let mut moved = walked.clone();
    moved.track.as_mut().expect("a track slot").walked_px = None;
    assert_eq!(
        super::measurements(&moved, StageKind::Track, &current)[4],
        "localized"
    );
}

/// The ZNCC cell prints the whole-patch reading, then the middle one, at both
/// stages; a row with no middle reading beside its ZNCC says so with `-`.
#[test]
fn the_zncc_cell_shows_the_whole_and_the_middle_reading() {
    use sfmtool_core::bench::{ClusterMeasurement, Observation, Provenance, TrackMeasurement};

    let current = crate::bench::live::Evaluation::Current;
    let mut cluster = ClusterMeasurement::from_seed([10.0, 12.0], [[1.0, 0.0], [0.0, 1.0]]);
    cluster.zncc = Some(0.923);
    cluster.zncc_middle = Some(0.614);
    let mut row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: Some(cluster),
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.95),
            zncc_middle: Some(0.2),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Cluster, &current)[0],
        "92% whole\n61% mid"
    );
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "95% whole\n20% mid"
    );
    // A committed track read back carries the stored ZNCC and no middle.
    row.track.as_mut().expect("a track slot").zncc_middle = None;
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "95% whole\n- mid"
    );
    row.track.as_mut().expect("a track slot").zncc = None;
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "-"
    );
}

#[test]
fn a_typed_percent_is_read_as_the_fraction_it_stands_for() {
    use super::parse_percent;
    assert_eq!(parse_percent("70"), Some(0.7));
    assert_eq!(parse_percent(" 85 % "), Some(0.85));
    assert_eq!(parse_percent("abc"), None);
}

#[test]
fn zncc_text_formats_both_readings() {
    use super::zncc_sentence;
    use super::zncc_text;
    assert_eq!(zncc_text(Some(0.918), Some(0.607)), "92% whole\n61% mid");
    assert_eq!(zncc_text(Some(0.5), None), "50% whole\n- mid");
    assert_eq!(zncc_text(Some(f64::NAN), Some(0.3)), "NaN whole\n30% mid");
    assert_eq!(zncc_text(Some(-0.35), Some(1.0)), "-35% whole\n100% mid");
    assert_eq!(zncc_text(None, Some(0.3)), "-");
    assert_eq!(zncc_sentence(Some(0.918), Some(0.607)), "92% / 61%");
    assert_eq!(zncc_sentence(Some(0.918), None), "92% / -");
}

/// A track-stage track standing on a bearing with **no patch**: the payload a
/// `w = 0` point of a node with no patch frames arrives on the bench as, which
/// is every row of a `sift_files` value.
fn frameless_bearing_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{EditableTrack, Stage, TrackPayload};

    let direction = nalgebra::Point3::new(0.778, -0.611, -0.146);
    let mut track = EditableTrack::empty_cluster();
    track.stage = Stage::Track(TrackPayload {
        position: Some(direction),
        at_infinity: true,
        placement: None,
        ..TrackPayload::default()
    });
    track
}

/// The header reads the **track's** flag and not its patch's `w`, so a bearing
/// that carries no patch still reads as a bearing. Reading the patch printed
/// `Position (0.778, -0.611, -0.146)` for a unit direction, which is a place one
/// unit from the world origin and the one thing that row is not.
#[test]
fn a_bearing_with_no_patch_still_reads_as_a_bearing() {
    let said = header_text(&frameless_bearing_track());
    assert!(
        said.contains("Bearing (0.778, -0.611, -0.146)"),
        "the header should name it a bearing: {said}"
    );
    assert!(said.contains("at infinity"), "{said}");
    assert!(!said.contains("Position ("), "{said}");
}

/// *Fit*, the two normal entries and the *Stage* toggle are greyed by what the
/// track is missing: each asks core's own half of their step's validation, so a button that cannot work
/// is not offered and the sentence a person reads is the one the step would
/// have refused with.
#[test]
fn the_photometric_entries_grey_with_their_own_sentence_on_a_frameless_track() {
    let track = frameless_bearing_track();
    let refusals = super::photometric_refusals(None, &track, StageKind::Cluster);
    for (what, refusal) in [
        ("Fit", &refusals.fit),
        ("Stage", &refusals.stage),
        ("Fit Normal", &refusals.normal),
    ] {
        let why = refusal
            .as_deref()
            .unwrap_or_else(|| panic!("{what} should be greyed on a track with no patch"));
        assert!(
            why.contains("no patch"),
            "{what} is greyed with {why:?}, which does not name what is missing"
        );
    }

    // And a whole track offers both.
    let (state, id) = state();
    let mut state = state;
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let whole = state.bench_track(id, &label).expect("on the bench");
    let refusals = super::photometric_refusals(None, whole, StageKind::Cluster);
    assert!(refusals.fit.is_none(), "{:?}", refusals.fit);
    assert!(refusals.stage.is_none(), "{:?}", refusals.stage);
    assert!(refusals.normal.is_none(), "{:?}", refusals.normal);

    // Busy wins over everything, as it did before.
    let refusals = super::photometric_refusals(Some("Busy."), whole, StageKind::Cluster);
    assert_eq!(refusals.fit.as_deref(), Some("Busy."));
    assert_eq!(refusals.stage.as_deref(), Some("Busy."));
    assert_eq!(refusals.normal.as_deref(), Some("Busy."));
}

/// The ZNCC grid's colours run red, yellow, green, from 50 to 100. A cell
/// with no reading has no colour.
#[test]
fn grid_cells_run_from_red_to_green() {
    use super::zncc_cell_color;
    let red = egui::Color32::from_rgb(220, 0, 40);
    let yellow = egui::Color32::from_rgb(220, 200, 40);
    let green = egui::Color32::from_rgb(0, 200, 40);
    assert_eq!(zncc_cell_color(0.2), Some(red));
    assert_eq!(zncc_cell_color(0.5), Some(red));
    assert_eq!(zncc_cell_color(0.75), Some(yellow));
    assert_eq!(zncc_cell_color(1.0), Some(green));
    assert_eq!(zncc_cell_color(f64::NAN), None);
}

#[test]
fn a_grid_s_hover_text_holds_its_nine_numbers() {
    use super::table::{grid_numbers, GridKind};
    let grid = [[0.92, 0.5, f64::NAN], [1.0, -0.2, 0.33], [0.0, 0.07, 0.8]];
    assert_eq!(
        grid_numbers(&grid, GridKind::Zncc),
        "   92    50     -
  100   -20    33
    0     7    80"
    );
    let radii = [[0.37, 1.0, 1.44], [2.0, 2.26, 3.0], [f64::NAN, 0.04, 3.0]];
    assert_eq!(
        grid_numbers(&radii, GridKind::SelfSimilarity),
        "  0.4   1.0   1.4
  2.0   2.3    3+
    -   0.0    3+"
    );
}

/// The self-similarity cell prints the whole and the middle radius to one
/// decimal, with `3+` for the largest radius the reading searches.
#[test]
fn the_self_similarity_cell_shows_the_whole_and_the_middle_radius() {
    use super::self_similarity_text;
    assert_eq!(
        self_similarity_text(Some(0.37), Some(1.44)),
        "0.4 px whole\n1.4 px mid"
    );
    assert_eq!(
        self_similarity_text(Some(3.0), Some(2.26)),
        "3+ px whole\n2.3 px mid"
    );
    assert_eq!(
        self_similarity_text(Some(1.0), Some(2.96)),
        "1.0 px whole\n3.0 px mid"
    );
    assert_eq!(self_similarity_text(Some(3.0), None), "3+ px whole\n- mid");
    assert_eq!(self_similarity_text(None, Some(1.0)), "-");

    // At both stages, from the fields the measurement carries.
    use sfmtool_core::bench::{ClusterMeasurement, Observation, Provenance, TrackMeasurement};
    let current = crate::bench::live::Evaluation::Current;
    let mut cluster = ClusterMeasurement::from_seed([10.0, 12.0], [[1.0, 0.0], [0.0, 1.0]]);
    cluster.zncc_self_similarity_radius = Some(3.0);
    cluster.zncc_self_similarity_radius_middle = Some(1.04);
    let row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: Some(cluster),
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.9),
            zncc_self_similarity_radius: Some(0.37),
            zncc_self_similarity_radius_middle: Some(1.44),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Cluster, &current)[3],
        "3+ px whole\n1.0 px mid"
    );
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[3],
        "0.4 px whole\n1.4 px mid"
    );
}

/// The self-similarity grid runs green under 1, yellow from 1 to 2, orange
/// from 2 to under the largest radius and red at it; a cell with no reading
/// has no colour.
#[test]
fn self_similarity_cells_run_from_green_to_red() {
    use super::self_similarity_cell_color;
    let green = egui::Color32::from_rgb(0, 200, 40);
    let yellow = egui::Color32::from_rgb(220, 200, 40);
    let orange = egui::Color32::from_rgb(220, 100, 40);
    let red = egui::Color32::from_rgb(220, 0, 40);
    assert_eq!(self_similarity_cell_color(0.0), Some(green));
    assert_eq!(self_similarity_cell_color(0.99), Some(green));
    assert_eq!(self_similarity_cell_color(1.0), Some(yellow));
    assert_eq!(self_similarity_cell_color(1.99), Some(yellow));
    assert_eq!(self_similarity_cell_color(2.0), Some(orange));
    assert_eq!(self_similarity_cell_color(2.99), Some(orange));
    assert_eq!(self_similarity_cell_color(3.0), Some(red));
    assert_eq!(self_similarity_cell_color(f64::NAN), None);
}

/// A self-similarity cell draws a line along its ellipse's major axis where
/// the ellipse is long and thin, its minor axis at most 0.71 of its major,
/// and nothing where it is rounder, a circle, or absent.
#[test]
fn a_self_similarity_cell_marks_its_major_axis() {
    use super::table::{ellipse_mark, CellMark};
    match ellipse_mark(&hover_ellipse([3.0, 0.3], [true, false], 0.0)) {
        CellMark::Line(half) => {
            assert!(half.y.abs() < 1e-6 && half.x.abs() > 3.0, "{half:?}");
        }
        other => panic!("expected a line, got {other:?}"),
    }
    match ellipse_mark(&hover_ellipse([1.0, 0.7], [false, false], 90.0)) {
        CellMark::Line(half) => assert!(half.x.abs() < 1e-6 && half.y.abs() > 3.0),
        other => panic!("expected a line, got {other:?}"),
    }
    assert_eq!(
        ellipse_mark(&hover_ellipse([1.0, 0.75], [false, false], 30.0)),
        CellMark::Nothing
    );
    let circle = sfmtool_core::patch::self_similarity::SelfSimilarityEllipse {
        major_angle: f64::NAN,
        ..hover_ellipse([3.0, 3.0], [true, true], 0.0)
    };
    assert_eq!(ellipse_mark(&circle), CellMark::Nothing);
    let none = sfmtool_core::patch::self_similarity::SelfSimilarityEllipse {
        axes: [f64::NAN; 2],
        major_angle: f64::NAN,
        ..circle
    };
    assert_eq!(ellipse_mark(&none), CellMark::Nothing);
}

/// The self-similarity heading says what its numbers and grid are, and which
/// bar judges them.
#[test]
fn the_self_similarity_heading_says_what_it_shows() {
    let headings = super::table::ColumnLayout::new().headers(super::BodyMode::Edited);
    assert!(headings
        .iter()
        .any(|&(_, heading, _)| heading == "Self-similarity"));
    let tip = super::table::SELF_SIMILARITY_TIP;
    assert!(tip.contains("middle") && tip.contains("3+"), "{tip}");
    assert!(tip.contains("box under this heading"), "{tip}");
}

// ---- Scrolling the table ------------------------------------------------------

/// A panel narrower than the table, so the table has somewhere to scroll
/// sideways.
const NARROW: egui::Vec2 = egui::vec2(600.0, 900.0);

/// Where, over the rows, the pointer is put to scroll or drag them: in the
/// *ZNCC* column of the second row, clear of every control.
const OVER_THE_ROWS: egui::Pos2 = egui::pos2(260.0, 300.0);

/// What the tests watch move: the self-similarity heading, the
/// self-similarity box's unit, the first row's self-similarity cell and the
/// geometry search box's label, which is above the table and must not move.
/// The self-similarity column rather than the ZNCC one because it stays in
/// sight after a drag of 200 points to the left.
struct Watched {
    heading: f32,
    bar: f32,
    cell: f32,
    above: f32,
}

/// One frame of the narrow panel with `events`, and where it drew what
/// [`Watched`] names.
fn narrow_frame(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> Watched {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), NARROW)),
        events,
        ..Default::default()
    };
    let texts = crate::test_support::painted_text_rects(ctx, input, |ui| {
        panel.show(ui, state);
    });
    let left = |found: Option<&crate::test_support::PaintedText>, what: &str| {
        found
            .unwrap_or_else(|| panic!("{what} is not drawn, or scrolled out of sight"))
            .rect
            .left()
    };
    Watched {
        heading: left(
            texts.iter().find(|t| t.text == "Self-similarity"),
            "the self-similarity heading",
        ),
        bar: left(
            texts
                .iter()
                .find(|t| t.text == super::MAX_SELF_SIMILARITY_LABEL),
            "the self-similarity box",
        ),
        cell: left(
            texts
                .iter()
                .find(|t| t.text.contains("px whole") && t.text.ends_with("px mid")),
            "a row's self-similarity cell",
        ),
        above: left(
            texts
                .iter()
                .find(|t| t.text == super::GEOMETRY_SEARCH_LABEL),
            "the geometry search box",
        ),
    }
}

/// The table moved left by the same distance, headings, threshold row and
/// rows together, and the controls above it stayed where they were. Returns
/// the distance.
fn moved_together(before: &Watched, after: &Watched, what: &str) -> f32 {
    let by = before.heading - after.heading;
    assert!(by > 20.0, "{what} did not scroll the table: {by}");
    assert!(
        (before.bar - after.bar - by).abs() < 0.5,
        "{what}: the threshold row did not move with the headings"
    );
    assert!(
        (before.cell - after.cell - by).abs() < 0.5,
        "{what}: the rows did not move with the headings"
    );
    assert_eq!(before.above, after.above, "{what} moved the controls above");
    by
}

/// A table wider than the panel scrolls sideways under a trackpad or a wheel,
/// the headings and the threshold row with the rows, and the controls above
/// it stay put. Run for both wheel units: a Windows precision touchpad
/// reaches the panel as a `Point` wheel, a mouse as a `Line` one.
#[test]
fn a_sideways_wheel_scrolls_the_table_and_its_headings_together() {
    for (unit, amount) in [
        (egui::MouseWheelUnit::Point, -240.0),
        (egui::MouseWheelUnit::Line, -3.0),
    ] {
        let (state, _, _, mut panel, ctx) = measured_on_the_bench();
        narrow_frame(&mut panel, &ctx, &state, Vec::new());
        let before = narrow_frame(&mut panel, &ctx, &state, Vec::new());
        narrow_frame(
            &mut panel,
            &ctx,
            &state,
            vec![
                egui::Event::PointerMoved(OVER_THE_ROWS),
                egui::Event::MouseWheel {
                    unit,
                    delta: egui::vec2(amount, 0.0),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        // egui smooths a wheel over several frames.
        for _ in 0..6 {
            narrow_frame(&mut panel, &ctx, &state, Vec::new());
        }
        let after = narrow_frame(&mut panel, &ctx, &state, Vec::new());
        moved_together(&before, &after, &format!("a {unit:?} wheel"));
    }
}

/// The pointer events of a drag with `button` from `from` to `to`, one list
/// per frame, as [`drag_frames`] does for the primary button.
fn button_drag_frames(
    button: egui::PointerButton,
    from: egui::Pos2,
    to: egui::Pos2,
) -> Vec<Vec<egui::Event>> {
    let press = |pos, pressed| egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let mid = from + (to - from) * 0.5;
    vec![
        vec![egui::Event::PointerMoved(from)],
        vec![press(from, true)],
        vec![egui::Event::PointerMoved(mid)],
        vec![egui::Event::PointerMoved(to)],
        vec![press(to, false)],
        vec![],
    ]
}

/// Drag with `button` from `from`, 200 points to the left, in the narrow
/// panel, and say where the watched things were before and after.
fn drag_the_table(button: egui::PointerButton, from: egui::Pos2) -> (Watched, Watched) {
    let (state, _, _, mut panel, ctx) = measured_on_the_bench();
    narrow_frame(&mut panel, &ctx, &state, Vec::new());
    let before = narrow_frame(&mut panel, &ctx, &state, Vec::new());
    for events in button_drag_frames(button, from, from - egui::vec2(200.0, 0.0)) {
        narrow_frame(&mut panel, &ctx, &state, events);
    }
    let after = narrow_frame(&mut panel, &ctx, &state, Vec::new());
    (before, after)
}

/// A middle-button drag moves the table with the pointer, over the rows and
/// over the headings alike.
#[test]
fn a_middle_drag_moves_the_table_with_the_pointer() {
    let (before, after) = drag_the_table(egui::PointerButton::Middle, OVER_THE_ROWS);
    let by = moved_together(&before, &after, "a middle drag over the rows");
    // At least as far as the pointer: a drag on the rows coasts on after
    // the button is let go.
    assert!(by >= 199.0, "the table moved {by}, less than the pointer");

    let on_the_headings = egui::pos2(OVER_THE_ROWS.x, heading_y());
    let (before, after) = drag_the_table(egui::PointerButton::Middle, on_the_headings);
    moved_together(&before, &after, "a middle drag over the headings");
}

/// A left-button drag begun on the rows, or on the headings, clear of a
/// control, moves the table as the middle button's does.
#[test]
fn a_left_drag_off_the_controls_moves_the_table() {
    let (before, after) = drag_the_table(egui::PointerButton::Primary, OVER_THE_ROWS);
    moved_together(&before, &after, "a left drag over the rows");

    let on_the_headings = egui::pos2(OVER_THE_ROWS.x, heading_y());
    let (before, after) = drag_the_table(egui::PointerButton::Primary, on_the_headings);
    moved_together(&before, &after, "a left drag over the headings");
}

/// A left-button drag begun on a *Keep* switch is the switch's, and leaves
/// the table where it is.
#[test]
fn a_left_drag_begun_on_a_switch_does_not_move_the_table() {
    let keep = super::table::ColumnLayout::new().keep_x();
    let on_a_switch = egui::pos2(keep + 16.0, OVER_THE_ROWS.y);
    let (before, after) = drag_the_table(egui::PointerButton::Primary, on_a_switch);
    assert_eq!(before.heading, after.heading, "the switch's drag scrolled");
    assert_eq!(before.cell, after.cell, "the switch's drag scrolled");
}

/// The height the headings are drawn at in the narrow panel.
fn heading_y() -> f32 {
    let (state, _, _, mut panel, ctx) = measured_on_the_bench();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), NARROW)),
        ..Default::default()
    };
    let texts = crate::test_support::painted_text_rects(&ctx, input, |ui| {
        panel.show(ui, &state);
    });
    texts
        .iter()
        .find(|t| t.text == "Name")
        .expect("the Name heading is drawn")
        .rect
        .center()
        .y
}

// ---- The self-similarity surface plot ----------------------------------------

/// A bowl `1 - k (dx² + dy²)` over the `7 × 7` square.
fn bowl(k: f64) -> Vec<f64> {
    (0..49)
        .map(|i| {
            let (dx, dy) = ((i % 7) as f64 - 3.0, (i / 7) as f64 - 3.0);
            1.0 - k * (dx * dx + dy * dy)
        })
        .collect()
}

#[test]
fn the_surface_plot_passes_through_the_measured_shifts() {
    use super::surface_plot::SurfacePlot;
    let surface = bowl(0.02);
    let plot = SurfacePlot::new(&surface, 0.05).expect("a textured surface");
    let step = (plot.n - 1) / 6;
    // The picture covers the whole square, corners included, as the radius
    // does.
    assert!(plot.values.iter().all(|v| v.is_finite()));
    for (i, &z) in surface.iter().enumerate() {
        let (x, y) = ((i % 7) * step, (i / 7) * step);
        assert!(
            (plot.values[y * plot.n + x] - z).abs() < 1e-9,
            "shift {i}: {} against {z}",
            plot.values[y * plot.n + x]
        );
    }
    // The shifts inside the contour at 0.95 are those with 0.02 d² <= 0.05,
    // d² <= 2.5: the four axis neighbours and the four diagonals.
    assert_eq!(plot.inside.len(), 8, "{:?}", plot.inside);
}

#[test]
fn a_bowl_s_contour_is_the_circle_at_its_level() {
    use super::surface_plot::SurfacePlot;
    let plot = SurfacePlot::new(&bowl(0.02), 0.05).expect("a textured surface");
    let contour = plot.contour();
    assert!(!contour.is_empty());
    // 1 - 0.02 d² = 0.95 at d = sqrt(2.5) px of shift, in a picture 6 px of
    // shift across.
    let expected = 2.5f32.sqrt() / 6.0;
    for [a, b] in contour {
        for p in [a, b] {
            let d = ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2)).sqrt();
            assert!(
                (d - expected).abs() < 0.02,
                "a contour point {d} from the centre"
            );
        }
    }
}

#[test]
fn the_surface_colours_jump_at_the_contour() {
    use super::surface_plot::surface_color;
    let below = surface_color(0.9499, 0.95);
    let above = surface_color(0.9501, 0.95);
    // Rec.601 luma, 0 to 255.
    let lightness = |c: egui::Color32| {
        0.299 * f64::from(c.r()) + 0.587 * f64::from(c.g()) + 0.114 * f64::from(c.b())
    };
    assert!(
        lightness(above) > lightness(below) + 80.0,
        "{below:?} to {above:?} is no jump"
    );
    // Below the level the ramp is muted and rises towards it.
    assert!(lightness(surface_color(0.5, 0.95)) < lightness(below));
    // Above it the colour lightens towards 1.
    assert!(lightness(surface_color(1.0, 0.95)) > lightness(above));
}

#[test]
fn a_surface_with_nothing_to_draw_has_no_plot() {
    use super::surface_plot::SurfacePlot;
    assert!(SurfacePlot::new(&[f64::NAN; 49], 0.05).is_none());
    assert!(SurfacePlot::new(&bowl(0.02)[..48], 0.05).is_none());
    assert!(SurfacePlot::new(&bowl(0.02), f64::INFINITY).is_none());
}

#[test]
fn a_ridge_s_contour_runs_to_the_edge_of_the_square() {
    use super::surface_plot::SurfacePlot;
    // A ridge along x: the ZNCC falls only across it.
    let surface: Vec<f64> = (0..49)
        .map(|i| {
            let (dx, dy) = ((i % 7) as f64 - 3.0, (i / 7) as f64 - 3.0);
            1.0 - 0.001 * dx * dx - 0.1 * dy * dy
        })
        .collect();
    let plot = SurfacePlot::new(&surface, 0.05).expect("a textured surface");
    assert!(plot.inside.contains(&[3, 0]) && plot.inside.contains(&[-3, 0]));
    assert!(!plot.inside.iter().any(|&[_, dy]| dy != 0));
}

/// The projection error cell prints the error in pixels over the same
/// residual as an angle in degrees, and says `-` for either that is not there.
#[test]
fn the_projection_error_cell_shows_pixels_and_degrees() {
    use super::projection_error_text;
    assert_eq!(
        projection_error_text(Some(0.654), Some(0.081)),
        "0.65 px\n0.08\u{b0}"
    );
    assert_eq!(projection_error_text(Some(1.5), None), "1.50 px\n-");
    assert_eq!(projection_error_text(None, None), "-");

    // Before the track is triangulated there is no point, so the error is
    // measured to the patch's centre.
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};
    let current = crate::bench::live::Evaluation::Current;
    let mut row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            projection_offset_px: Some(2.25),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[2],
        "2.25 px\n-"
    );
    // Once there is a point, the error to it, with its angle.
    let slot = row.track.as_mut().expect("a track slot");
    slot.reprojection_error = Some(0.5);
    slot.ray_angle_deg = Some(0.07);
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[2],
        "0.50 px\n0.07\u{b0}"
    );
}

/// A row's *Keep* switch turns a kept row out, and the whole cell takes the
/// click rather than the row behind it.
#[test]
fn clicking_the_keep_switch_turns_a_kept_row_out() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    assert_eq!(panel.rows()[1].verdict, Verdict::In);
    let y = row_y(&mut panel, &ctx, &state, 1);
    // Left of the switch itself, still in its cell.
    let x = super::table::ColumnLayout::new().keep_x() + 1.0;
    let response = at_pointer(&mut panel, &ctx, &state, egui::pos2(x, y + 8.0), true);
    assert_eq!(response.set_verdict, Some((1, Verdict::Out)));
    assert_eq!(
        response.pick_row, None,
        "the row behind the switch took the click"
    );
}

/// A verdict set by hand is handed back to the thresholds from the switch's
/// own menu and from the row's.
#[test]
fn a_pinned_row_offers_to_unpin_from_the_switch_and_from_the_row() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .set_bench_verdict(id, &label, 0, Verdict::Out)
        .expect("observation 0 exists");
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows()[0].pinned);

    let y = row_y(&mut panel, &ctx, &state, 0);
    let switch = super::table::ColumnLayout::new().keep_x() + 12.0;
    for x in [switch, 400.0] {
        let at = egui::pos2(x, y + 8.0);
        open_row_menu(&mut panel, &ctx, &state, at);
        let entry = menu_entry_pos(&mut panel, &ctx, &state, super::table::UNPIN_LABEL);
        let response = at_pointer(&mut panel, &ctx, &state, entry, true);
        assert_eq!(response.unpin_verdicts, Some(vec![0]), "from x = {x}");
    }

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    assert_eq!(versions(&state, id), before + 1);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(!track.observations[0].pinned);
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    assert_eq!(
        versions(&state, id),
        before + 1,
        "a second unpin is no effect"
    );
}

/// The header prints the ID of the point the track came from once: as the
/// label when the label is that ID, and beside a renamed label otherwise. A
/// point that is gone from the version is named by its old index instead.
#[test]
fn the_header_names_the_point_the_track_came_from() {
    let mut track = bearing_track();
    track.origin = Some(sfmtool_core::bench::Origin {
        version: 1,
        point: 12,
    });
    let ctx = egui::Context::default();
    let header = |label: &str, id: Option<&str>| {
        crate::test_support::painted_texts(&ctx, input(Vec::new()), |ui| {
            super::show_header(ui, label, &track, id, None);
        })
    };

    let same = header("pt3d_ab_12", Some("pt3d_ab_12"));
    assert_eq!(same.iter().filter(|t| t.contains("pt3d_ab_12")).count(), 1);
    assert!(!same.iter().any(|t| t.contains("from point")), "{same:?}");

    let renamed = header("my track", Some("pt3d_ab_12"));
    assert!(renamed.iter().any(|t| t == "pt3d_ab_12"), "{renamed:?}");

    let gone = header("my track", None);
    assert!(gone.iter().any(|t| t == "\u{b7} from point 12"), "{gone:?}");
}

/// A track put on the bench from a point knows its point's ID, which is what
/// the header's copy button copies.
#[test]
fn a_track_from_a_point_resolves_the_id_the_header_copies() {
    let (state, id, label, _, _) = on_the_bench();
    let node = state.node(id).expect("the node");
    let track = state.bench_track(id, &label).expect("on the bench");
    let index = state
        .resolved_origin(node, track)
        .expect("the point is live");
    assert_eq!(crate::scene::point_id(node, index as usize), label);
}

/// The pin beside the switch pins a verdict the thresholds set, as it stands,
/// and hands a pinned one back to the thresholds.
#[test]
fn the_pin_pins_a_verdict_and_unpins_a_pinned_one() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // A point's rows arrive pinned; this one is handed to the bars first.
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);
    let y = row_y(&mut panel, &ctx, &state, 1);
    let pin = egui::pos2(super::table::ColumnLayout::new().keep_x() + 52.0, y + 8.0);
    let response = at_pointer(&mut panel, &ctx, &state, pin, true);
    assert_eq!(response.set_verdict, Some((1, Verdict::In)));
    assert_eq!(
        response.pick_row, None,
        "the row behind the pin took the click"
    );

    state
        .set_bench_verdict(id, &label, 1, Verdict::In)
        .expect("observation 1 exists");
    let response = at_pointer(&mut panel, &ctx, &state, pin, true);
    assert_eq!(response.unpin_verdicts, Some(vec![1]));
    assert_eq!(response.set_verdict, None);
}

/// The Name column elides a long name in its middle, and hovering it shows
/// the name whole. Hovering the image's index shows the same name.
#[test]
fn hovering_a_name_shows_it_whole() {
    let cols = super::table::ColumnLayout::new();
    for (column, x) in [
        ("Name", cols.name_x(super::BodyMode::Edited)),
        ("Img", cols.image_x()),
    ] {
        let (state, id, _, mut panel, ctx) = on_the_bench();
        // A tooltip waits out `tooltip_delay` before it shows, which a
        // headless frame has no wall clock to pass.
        ctx.all_styles_mut(|style| {
            style.interaction.tooltip_delay = 0.0;
            style.interaction.tooltip_grace_time = 0.0;
        });
        let node = state.node(id).expect("the node");
        let name = node.recon().image_table.images[1].name.clone();
        let y = row_y(&mut panel, &ctx, &state, 1);
        // The Name cell prints the name itself where it fits, so the hover
        // is what adds one more copy of it.
        let shown = painted(&mut panel, &ctx, &state, Vec::new())
            .iter()
            .filter(|t| **t == name)
            .count();
        let at = egui::pos2(x + 10.0, y + 8.0);
        let response = at_pointer(&mut panel, &ctx, &state, at, false);
        assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
        // A tooltip shows after the pointer has rested, so a few more frames.
        let mut texts = Vec::new();
        for _ in 0..12 {
            texts = painted(
                &mut panel,
                &ctx,
                &state,
                vec![egui::Event::PointerMoved(at)],
            );
        }
        let hovered = texts.iter().filter(|t| **t == name).count();
        assert_eq!(
            hovered,
            shown + 1,
            "hovering {column} did not show {name:?}: {texts:?}"
        );
    }
}

/// The hover view of one observation's tile on the focused item, beside the
/// tile itself, both rendered from the node's cached photograph.
fn tile_and_context(
    state: &AppState,
    id: ReconId,
    label: &str,
    observation: usize,
) -> (egui::ColorImage, super::tile::TileContext) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let row = &track.observations[observation];
    let src = state
        .cached_photograph(id, row.image as usize)
        .expect("a cached photograph");
    let recon = state.node(id).expect("loaded").recon();
    let tile = super::tile::image(recon, &track, observation, &src).expect("a tile");
    let context = super::tile::context(recon, &track, observation, &src).expect("a context");
    (tile, context)
}

/// The largest difference in any channel between `tile` and the texels of
/// `context`'s picture inside its patch box, which must be the tile's size.
fn largest_difference_in_box(tile: &egui::ColorImage, context: &super::tile::TileContext) -> u8 {
    let [w, h] = tile.size;
    let min = context.patch_box.min;
    assert_eq!(
        (context.patch_box.width(), context.patch_box.height()),
        (w as f32, h as f32),
        "the patch box is not the tile's size in texels"
    );
    let side = context.image.size[0];
    let mut worst = 0u8;
    for y in 0..h {
        for x in 0..w {
            let a = tile.pixels[y * w + x].to_array();
            let (cx, cy) = (min.x as usize + x, min.y as usize + y);
            let b = context.image.pixels[cy * side + cx].to_array();
            for c in 0..3 {
                worst = worst.max(a[c].abs_diff(b[c]));
            }
        }
    }
    worst
}

/// A tile's hover view is the same picture over three times the patch's
/// width, at the tile's own sampling, at either stage: the picture is three
/// tiles across, the patch box is its middle third, the keypoint sits at the
/// box's centre, and the texels inside the box are the tile's.
#[test]
fn a_tile_s_hover_view_holds_the_tile_in_its_middle_at_either_stage() {
    let (mut state, id, label, _, _) = on_the_bench();
    let k = super::tile::CONTEXT_FACTOR as f32;
    let check = |state: &AppState, stage: &str| {
        let observations = state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations
            .len();
        for observation in 0..observations {
            let (tile, context) = tile_and_context(state, id, &label, observation);
            let side = context.image.size[0] as f32;
            assert_eq!(
                context.image.size,
                [tile.size[0] * 3, tile.size[1] * 3],
                "{stage}: the picture is not three tiles across"
            );
            assert_eq!(
                context.patch_box,
                egui::Rect::from_min_size(
                    egui::pos2(side / k, side / k),
                    egui::vec2(side / k, side / k)
                ),
                "{stage}: the patch box is not the middle third"
            );
            let keypoint = context.keypoint.expect("the keypoint meets the patch");
            assert!(
                keypoint.distance(context.patch_box.center()) < 1e-3,
                "{stage}: the keypoint {keypoint:?} is not the box's centre"
            );
            // The same sample positions, computed along a different sum, so a
            // bilinear read may round one level apart.
            let worst = largest_difference_in_box(&tile, &context);
            assert!(
                worst <= 2,
                "{stage}: observation {observation}'s box differs from its tile by {worst}"
            );
        }
    };
    check(&state, "track stage");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    check(&state, "cluster stage");
}

/// At the track stage the hover view marks where the track's point projects,
/// and it is placed consistently with the row's reprojection error: the
/// distance it states is the row's error, and its distance from the keypoint in
/// the picture, turned into the photograph's pixels by the patch's own width in
/// both, is that error too. A row sitting on its projection has the mark on
/// the keypoint.
#[test]
fn a_hover_view_marks_the_projection_as_far_off_as_the_row_s_error() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let frame = payload.placement.clone().expect("a patch");
    let position = payload.position.expect("a triangulated point");

    // A sighting of the point in a fourth image, placed a few pixels off
    // where the point projects.
    let image = 3;
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, image).expect("a view");
    let projected = camera
        .project_homogeneous(&pose, position.coords, frame.w)
        .expect("the demo cameras see every point");
    let off = [4.0, -3.0];
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, image),
            &crate::bench::Seed::Pixel {
                pixel: [projected[0] + off[0], projected[1] + off[1]],
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state.settle_bench_evaluation();

    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let added = track.observations.len() - 1;
    let error = track.observations[added]
        .track
        .as_ref()
        .and_then(|m| m.reprojection_error)
        .expect("the evaluation measured the new row's error");
    assert!(
        (error - 5.0).abs() < 0.05,
        "the row's error is {error}, not 5 px"
    );

    let (_, context) = tile_and_context(&state, id, &label, added);
    assert_eq!(
        context.projection_of,
        Some(super::tile::ProjectionOf::Point)
    );
    let stated = context.projection_px.expect("a projection distance");
    assert!(
        (stated - error).abs() < 1e-6,
        "the hover view states {stated} px, the row {error} px"
    );
    let keypoint = context.keypoint.expect("a keypoint");
    let mark = context.projection.expect("a projection mark");
    // The picture's own map back into the photograph: the widened frame the
    // picture was rendered through, a texel read as `(s, t)` on it, and that
    // place on the plane projected. The patch is seen at a slant here, so a
    // single pixels-per-texel scale would not do; the map is what scales an
    // offset in the picture into the photograph's pixels.
    let keypoint_px = [projected[0] + off[0], projected[1] + off[1]];
    let mut wide = frame
        .anchored_at_keypoint(&camera, &pose, keypoint_px)
        .expect("the keypoint meets the patch");
    wide.half_extent = wide
        .half_extent
        .map(|h| h * f64::from(super::tile::CONTEXT_FACTOR));
    let side = f64::from(context.image.size[0] as u32);
    let to_photograph = |at: egui::Pos2| {
        let s = 2.0 * f64::from(at.x) / side - 1.0;
        let t = 1.0 - 2.0 * f64::from(at.y) / side;
        let (xyz, w) = wide.corner_homogeneous(s, t);
        camera
            .project_homogeneous(&pose, xyz, w)
            .expect("on the plane")
    };
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    assert!(
        distance(to_photograph(keypoint), keypoint_px) < 0.01,
        "the keypoint in the picture is not the observation's pixel"
    );
    assert!(
        distance(to_photograph(mark), projected) < 0.01,
        "the mark is not where the point projects"
    );
    let measured = distance(to_photograph(keypoint), to_photograph(mark));
    assert!(
        (measured - error).abs() < 0.01,
        "the mark sits {measured} px from the keypoint in the photograph's pixels, \
         the row says {error}"
    );
    // And it is a visible distance in the picture, not a rounding.
    assert!(
        keypoint.distance(mark) > 1.0,
        "a 5 px error drew the mark {} texels off",
        keypoint.distance(mark)
    );

    // A row the fixture put at its exact projection has the mark on the dot.
    let (_, context) = tile_and_context(&state, id, &label, 0);
    let keypoint = context.keypoint.expect("a keypoint");
    let mark = context.projection.expect("a projection mark");
    assert!(
        keypoint.distance(mark) < 0.05,
        "a row on its projection has the mark {mark:?} off the keypoint {keypoint:?}"
    );
    assert!(context.projection_px.expect("a distance") < 0.01);
}

/// At the cluster stage there is no point, and the table's projection error
/// column is empty, so the hover view draws the picture, the box and the
/// keypoint with no projection mark, and its caption says why.
#[test]
fn a_cluster_stage_hover_view_has_no_projection() {
    let (mut state, id, label, _, _) = on_the_bench();
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let (_, context) = tile_and_context(&state, id, &label, 0);
    assert_eq!(context.projection, None);
    assert_eq!(context.projection_px, None);
    assert_eq!(context.projection_of, None);
    let caption = super::tile::context_caption(&context);
    assert!(
        caption.contains("no point to project"),
        "the caption does not say why there is no mark: {caption}"
    );
}

/// Resting the pointer on a row's tile shows the hover view, rendered for that
/// row alone, while the row keeps its hover; and a click on the tile is still
/// the row's click.
#[test]
fn hovering_a_tile_shows_it_in_context_and_keeps_the_row() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    // A tooltip waits out `tooltip_delay` before it shows, which a headless
    // frame has no wall clock to pass.
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let y = row_y(&mut panel, &ctx, &state, 1);
    // Inside the tile: the tile column's left edge plus half a tile, and far
    // enough below the top of the row to be within the tile's height.
    let x = super::table::ColumnLayout::new().tile_x() + super::table::TILE_SIZE / 2.0;
    let at = egui::pos2(x + 8.0, y + 20.0);
    let response = at_pointer(&mut panel, &ctx, &state, at, false);
    assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
    // egui shows a tooltip once the pointer has come to rest, so a few frames
    // with no movement in them.
    for _ in 0..12 {
        run_frame(&mut panel, &ctx, &state);
    }
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("\u{2022} Box: the patch")),
        "no hover view caption in {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("where the track's point projects")),
        "the caption does not name the projection: {texts:?}"
    );
    let rendered: Vec<usize> = panel.contexts.keys().copied().collect();
    assert_eq!(
        rendered,
        vec![1],
        "hover views rendered for rows not hovered"
    );

    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(
        response.pick_row,
        Some((1, false)),
        "a click on the tile did not pick the row"
    );
}

// ── The bars' colours and the Keep cell's proposal ──────────────────────

/// Set the boxes the way a drag in progress holds them: from the focused
/// track's bars with `change` applied, and held there for the next frame, which
/// is drawn.
fn with_boxes(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    id: ReconId,
    label: &str,
    change: impl FnOnce(&mut Thresholds),
) {
    let mut bars = state
        .bench_track(id, label)
        .expect("on the bench")
        .thresholds
        .clone();
    change(&mut bars);
    panel.thresholds = bars;
    panel.sliding = true;
    run_frame(panel, ctx, state);
}

/// [`POINT`] on the bench with its first evaluation landed, so every row has
/// readings for the bars to judge.
fn measured_on_the_bench() -> (AppState, ReconId, String, TrackBody, egui::Context) {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    (state, id, label, panel, ctx)
}

/// The whole and the middle ZNCC of observation `i` of the focused item.
fn zncc_readings(state: &AppState, id: ReconId, label: &str, i: usize) -> (f64, f64) {
    let track = state.bench_track(id, label).expect("on the bench");
    let m = track.observations[i]
        .track
        .as_ref()
        .expect("a track-stage reading");
    (
        m.zncc.expect("a whole ZNCC"),
        m.zncc_middle.expect("a middle ZNCC"),
    )
}

/// Each line of the ZNCC cell takes the colour of its own bar: the whole
/// reading can pass while the middle fails, and the reverse.
#[test]
fn each_line_of_the_zncc_cell_is_judged_by_its_own_bar() {
    use sfmtool_core::bench::BarCheck;

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let (whole, middle) = zncc_readings(&state, id, &label, 0);
    assert!(middle > 0.02, "the fixture's middle ZNCC is {middle}");

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole - 0.01;
        bars.min_zncc_middle = middle + 0.01;
    });
    assert_eq!(
        panel.rows()[0].checks[0],
        [BarCheck::Pass, BarCheck::Fail],
        "whole over its bar, middle under its own"
    );

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole + 0.01;
        bars.min_zncc_middle = middle - 0.01;
    });
    assert_eq!(
        panel.rows()[0].checks[0],
        [BarCheck::Fail, BarCheck::Pass],
        "whole under its bar, middle over its own"
    );
}

/// The readings no bar judges stay in the plain colour: the projection error
/// in degrees, the self-similarity middle radius, the status, the middle ZNCC while its
/// bar is off, a reading that is missing, and every reading of a row nothing
/// has measured or whose evaluation was refused.
#[test]
fn readings_no_bar_judges_are_drawn_plain() {
    use sfmtool_core::bench::{bar_checks, BarCheck, Observation, Provenance, TrackMeasurement};

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc_middle = 0.0;
    });
    let plain = [BarCheck::NotJudged; 2];
    for row in panel.rows() {
        assert_eq!(row.checks[0][1], BarCheck::NotJudged, "mid ZNCC, bar off");
        assert_eq!(row.checks[2][1], BarCheck::NotJudged, "Proj. err degrees");
        assert_eq!(row.checks[3][1], BarCheck::NotJudged, "self-similarity mid");
        assert_eq!(row.checks[4], plain, "Status");
        assert_ne!(
            row.checks[0][0],
            BarCheck::NotJudged,
            "ZNCC whole is judged"
        );
    }

    // A reading that is missing (`-`) is not judged, though it clears its bar.
    let row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            loo_zncc: Some(0.95),
            zncc: Some(0.95),
            ..TrackMeasurement::default()
        }),
    };
    let judged = super::Judgement {
        checks: bar_checks(&row, StageKind::Track, &Thresholds::default()).expect("measured"),
        proposal: Verdict::In,
    };
    let current = crate::bench::live::Evaluation::Current;
    let checks = super::table::cell_checks(Some(&judged), &current);
    assert_eq!(checks[0], [BarCheck::Pass, BarCheck::NotJudged]);
    assert_eq!(checks[1][0], BarCheck::NotJudged, "no shift reading");
    assert_eq!(checks[3][0], BarCheck::NotJudged, "no radius reading");

    // A row nothing has measured, and a row whose cells print no numbers.
    assert_eq!(super::table::cell_checks(None, &current), [plain; 5]);
    let refused = crate::bench::live::Evaluation::Refused("no photographs".to_string());
    assert_eq!(
        super::table::cell_checks(Some(&judged), &refused),
        [plain; 5]
    );
}

/// Dragging a box judges the readings against the box at once, and changes
/// nothing on the track until it is let go.
#[test]
fn dragging_a_box_changes_the_judgement_and_not_the_track() {
    use sfmtool_core::bench::BarCheck;

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let before = state.bench_track(id, &label).expect("on the bench").clone();
    let versions_before = versions(&state, id);

    // Bars every reading clears: each row is proposed in, and says why.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = -1.0;
        bars.min_zncc_middle = 0.0;
        bars.max_shift_px = 1.0e6;
        bars.max_zncc_self_similarity_radius = 1.0e6;
    });
    for row in panel.rows() {
        assert_eq!(row.checks[0][0], BarCheck::Pass, "{row:?}");
        assert_eq!(row.proposal, Some(Verdict::In), "{row:?}");
        assert!(
            row.keep_hover.contains("it clears every bar"),
            "{}",
            row.keep_hover
        );
    }

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = 1.01;
    });
    for row in panel.rows() {
        assert_eq!(row.checks[0][0], BarCheck::Fail, "{row:?}");
        assert_eq!(row.proposal, Some(Verdict::Out), "{row:?}");
        assert_eq!(row.verdict, Verdict::In, "the drag moved a verdict");
    }
    let after = state.bench_track(id, &label).expect("on the bench");
    assert!(
        std::sync::Arc::ptr_eq(&before, after),
        "the drag stepped the track"
    );
    assert_eq!(versions(&state, id), versions_before);
}

/// An unpinned row's *Keep* cell shows what applying the bars makes it, and a
/// pinned row's what unpinning it would: a switch that is on in a red cell is
/// a hand ruling against the bars, and its hover text says which bar.
#[test]
fn a_pinned_row_ruled_against_the_bars_shows_what_unpinning_would_give() {
    let (mut state, id, label, mut panel, ctx) = measured_on_the_bench();
    state
        .set_bench_verdict(id, &label, 0, Verdict::In)
        .expect("observation 0 exists");
    let (whole, _) = zncc_readings(&state, id, &label, 0);
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole + 0.001;
    });

    let row = &panel.rows()[0];
    assert!(row.pinned && row.verdict == Verdict::In, "{row:?}");
    assert_eq!(row.proposal, Some(Verdict::Out), "the bars fail it");
    assert!(
        row.keep_hover.contains("ZNCC whole is under the bar"),
        "{}",
        row.keep_hover
    );
    assert!(
        row.keep_hover.contains("Kept, set by hand."),
        "the switch's own text is kept: {}",
        row.keep_hover
    );

    // And that is what unpinning it gives.
    let bars = panel.thresholds().clone();
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(track.observations[0].verdict, Verdict::Out);
}

/// Of two sightings in one image that both clear every bar, the one the
/// painting does not take is proposed `out`, and its hover text says another
/// sighting in that image is kept.
#[test]
fn a_row_that_clears_every_bar_but_loses_its_image_says_so() {
    use sfmtool_core::bench::BarCheck;

    let (mut state, id, label, _, _) = on_the_bench();
    let pixel = crate::bench::observation_pixel(
        &state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[1],
    )
    .expect("a placed observation");
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, 1),
            &crate::bench::Seed::Pixel {
                pixel: [f64::from(pixel[0]), f64::from(pixel[1])],
                radius_px: None,
            },
        )
        .expect("a pixel in the photograph");
    state.settle_bench_evaluation();
    let (mut panel, ctx) = settled(&state);
    // Bars every measured reading clears.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = -1.0;
        bars.min_zncc_middle = 0.0;
        bars.max_shift_px = 1.0e6;
        bars.max_zncc_self_similarity_radius = 1.0e6;
    });

    let rows = panel.rows();
    assert_eq!(rows.len(), 4, "{rows:?}");
    let in_image_1: Vec<_> = rows.iter().filter(|row| row.image == 1).collect();
    assert!(
        in_image_1.iter().all(|row| row.proposal.is_some()
            && row.checks.iter().flatten().all(|&c| c != BarCheck::Fail)),
        "both sightings are measured and clear every bar: {in_image_1:?}"
    );
    let lost: Vec<_> = in_image_1
        .iter()
        .filter(|row| row.proposal == Some(Verdict::Out))
        .collect();
    assert_eq!(
        lost.len(),
        1,
        "one of the two keeps the image: {in_image_1:?}"
    );
    assert!(
        lost[0]
            .keep_hover
            .contains("another sighting in image 1 is kept"),
        "{}",
        lost[0].keep_hover
    );
}

/// The image's index stands at the table's left edge under *Img*, the crop
/// after it under *Crop*, the patch tile after that under *Patch*, and the
/// *Keep* column after them, none of them overlapping.
#[test]
fn the_crop_and_patch_columns_come_first_under_their_headings() {
    let cols = super::table::ColumnLayout::new();
    let size = super::table::TILE_SIZE;
    assert_eq!(cols.image_x(), 0.0);
    assert!(cols.crop_x() > cols.image_x(), "the crop overlaps Img");
    assert!(
        cols.tile_x() >= cols.crop_x() + size,
        "the patch overlaps the crop"
    );
    assert!(
        cols.keep_x() >= cols.tile_x() + size,
        "the Keep column overlaps the patch"
    );
    let headers = cols.headers(super::BodyMode::Edited);
    let firsts: Vec<(f32, &str)> = headers[..4].iter().map(|&(x, h, _)| (x, h)).collect();
    assert_eq!(
        firsts,
        vec![
            (cols.image_x(), "Img"),
            (cols.crop_x(), "Crop"),
            (cols.tile_x(), "Patch"),
            (cols.keep_x(), "Keep"),
        ]
    );
}

/// The headings are drawn at the size of the cells under them.
#[test]
fn the_headings_are_as_large_as_the_cells() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    });
    let height = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .map(|painted| painted.rect.height())
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
    };
    assert_eq!(height("Name"), height("image_000.jpg"));
}

/// Where the *Keep* heading's pin is drawn: over the rows' pin column, on the
/// heading's line.
fn heading_pin_pos(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> egui::Pos2 {
    let keep = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
    .into_iter()
    .find(|painted| painted.text == "Keep")
    .expect("the Keep heading was painted")
    .rect;
    egui::pos2(
        super::table::ColumnLayout::new().keep_x() + 52.0,
        keep.center().y,
    )
}

/// The pin in the *Keep* heading unpins every pinned row of the track in one
/// step, and once nothing is pinned a click asks to pin every row instead.
#[test]
fn the_keep_heading_pin_unpins_every_pinned_row_then_pins_them_all() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.pinned),
        "a point's rows arrive pinned"
    );
    let at = heading_pin_pos(&mut panel, &ctx, &state);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0, 1, 2]));
    assert_eq!(response.pick_row, None);

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows().iter().all(|row| !row.pinned));

    // Nothing is pinned now: a click asks to pin every row.
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, None);
    assert_eq!(response.pin_verdicts, Some(vec![0, 1, 2]));
    // And an unpin of rows none of which is pinned pushes no version.
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1);
}

/// With no row pinned, the heading pin pins every row at the verdict it has
/// now, in one step with one Action Log row, and a second click unpins them
/// all again.
#[test]
fn the_keep_heading_pin_pins_every_row_as_it_stands() {
    let (mut state, id, label, mut panel, ctx) = measured_on_the_bench();
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    // A whole-ZNCC bar just over the lowest reading turns that row out and
    // leaves the rest to the other bars: a mix of `in` and `out` with nothing
    // pinned.
    let lowest = (0..3)
        .map(|i| zncc_readings(&state, id, &label, i).0)
        .fold(f64::INFINITY, f64::min);
    let mut bars = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .clone();
    bars.min_zncc = lowest + 0.001;
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    let verdicts: Vec<Verdict> = panel.rows().iter().map(|row| row.verdict).collect();
    assert!(verdicts.contains(&Verdict::In) && verdicts.contains(&Verdict::Out));
    assert!(panel.rows().iter().all(|row| !row.pinned));

    let at = heading_pin_pos(&mut panel, &ctx, &state);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    let rows = response
        .pin_verdicts
        .expect("the pin asks to pin every row");
    assert_eq!(rows, vec![0, 1, 2]);
    assert_eq!(response.unpin_verdicts, None);

    let before = versions(&state, id);
    state.action_log.clear();
    state
        .pin_bench_verdicts(id, &label, &rows)
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    let log: Vec<String> = state.action_log.entries().map(|e| e.text.clone()).collect();
    let (kept, out) = (
        verdicts.iter().filter(|&&v| v == Verdict::In).count(),
        verdicts.iter().filter(|&&v| v == Verdict::Out).count(),
    );
    assert_eq!(log.len(), 1, "{log:?}");
    assert!(
        log[0].starts_with(&format!(
            "Pinned 3 verdicts in {label}: {kept} in, {out} out"
        )),
        "{log:?}"
    );
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows().iter().all(|row| row.pinned));
    assert_eq!(
        panel
            .rows()
            .iter()
            .map(|row| row.verdict)
            .collect::<Vec<_>>(),
        verdicts,
        "every verdict stands as it was"
    );
    // Pinning rows that are all pinned pushes nothing.
    state
        .pin_bench_verdicts(id, &label, &rows)
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1);

    // One undo takes the pins off again.
    state.undo(id).expect("the pin step undoes");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(track.observations.iter().all(|o| !o.pinned));
    state.redo(id).expect("and redoes");
    run_frame(&mut panel, &ctx, &state);

    // Now every row is pinned, so the same click unpins them all.
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0, 1, 2]));
    assert_eq!(response.pin_verdicts, None);
}

/// The heading pin's hover text says what a click does, with the count, and
/// why it is greyed. A headless frame shows no tooltip, so the text is asked of
/// the function that writes it.
#[test]
fn the_heading_pin_hover_counts_the_pins() {
    use super::table::{heading_pin_hover, heading_pin_name};
    assert_eq!(
        heading_pin_hover(12, 20, None),
        "Unpin all 12 pinned verdicts and let the bars decide"
    );
    assert_eq!(
        heading_pin_hover(1, 20, None),
        "Unpin the 1 pinned verdict and let the bars decide"
    );
    assert_eq!(
        heading_pin_hover(0, 12, None),
        "Pin all 12 verdicts as they stand"
    );
    assert_eq!(
        heading_pin_hover(0, 1, None),
        "Pin the 1 verdict as it stands"
    );
    assert_eq!(heading_pin_hover(4, 20, Some("busy")), "busy");
    assert_eq!(heading_pin_hover(0, 20, Some("busy")), "busy");
    assert!(heading_pin_hover(0, 0, None).contains("no observations"));
    assert_eq!(heading_pin_name(3), "Unpin all");
    assert_eq!(heading_pin_name(0), "Pin all");
}

/// The header counts the pinned rows beside the kept and the turned out.
#[test]
fn the_header_counts_kept_out_and_pinned() {
    let (mut state, id, label, _, _) = on_the_bench();
    let text = header_text(state.bench_track(id, &label).expect("on the bench"));
    assert!(text.contains("3 kept · 0 out · 3 pinned"), "{text}");
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    let text = header_text(state.bench_track(id, &label).expect("on the bench"));
    assert!(text.contains(" · 2 pinned"), "{text}");
}

/// On a row that is one of several selected, the row menu's unpin acts on the
/// selection: every pinned verdict among the selected rows in one step, with
/// the count in its label. The *Keep* switch's own menu stays per row.
#[test]
fn the_row_menu_unpins_every_pinned_row_of_a_selection() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    for (observation, extend) in [(0, false), (1, true), (2, true)] {
        state.pick_bench_observation(id, &label, observation, extend);
    }
    run_frame(&mut panel, &ctx, &state);

    let text = super::table::unpin_selection_label(2);
    assert_eq!(text, "Unpin 2 verdicts, let the thresholds decide");
    let y = row_y(&mut panel, &ctx, &state, 0);
    open_row_menu(&mut panel, &ctx, &state, egui::pos2(400.0, y + 8.0));
    let entry = menu_entry_pos(&mut panel, &ctx, &state, &text);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    let mut rows = response.unpin_verdicts.expect("the entry was taken");
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 2], "the pinned rows of the selection");

    let switch = egui::pos2(super::table::ColumnLayout::new().keep_x() + 12.0, y + 8.0);
    open_row_menu(&mut panel, &ctx, &state, switch);
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::table::UNPIN_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0]));

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(track.observations.iter().all(|o| !o.pinned));
}

// ── The crop column ─────────────────────────────────────────────────────

/// Every row draws the crop of its photograph beside its tile, at the track
/// stage and at the cluster stage.
#[test]
fn every_row_draws_a_crop_at_either_stage() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.crop),
        "a track-stage row drew no crop: {:?}",
        panel.rows(),
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.crop),
        "a cluster-stage row drew no crop: {:?}",
        panel.rows(),
    );
}

/// One observation's outline, its crop and the crop's hover view, all from
/// the node's cached photograph.
fn crop_and_context(
    state: &AppState,
    id: ReconId,
    label: &str,
    observation: usize,
) -> (
    super::crop::Outline,
    super::crop::CropPicture,
    super::crop::CropPicture,
) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let row = &track.observations[observation];
    let src = state
        .cached_photograph(id, row.image as usize)
        .expect("a cached photograph");
    let recon = state.node(id).expect("loaded").recon();
    let outline = super::crop::outline(recon, &track, observation).expect("an outline");
    let crop = super::crop::image(recon, &track, observation, &src).expect("a crop");
    let context = super::crop::context(recon, &track, observation, &src).expect("a context");
    (outline, crop, context)
}

/// The crop is square, and is the outline's bounding box widened by the margin,
/// rounded out to whole pixels and then widened evenly on its shorter side:
/// every sample of the outline lies at least the margin inside it, on the
/// longer side neither edge reaches a whole pixel further than that, and on
/// both the outline sits in the middle to within the rounding. At one texel per
/// pixel each texel is the photograph's own pixel. Checked at both stages.
#[test]
fn a_crop_holds_the_whole_outline_with_a_pixel_to_spare() {
    let (mut state, id, label, _, _) = on_the_bench();
    let margin = super::crop::CROP_MARGIN_PX;
    let check = |state: &AppState, stage: &str| {
        let track = state.bench_track(id, &label).expect("on the bench").clone();
        for observation in 0..track.observations.len() {
            let (outline, crop, _) = crop_and_context(state, id, &label, observation);
            let region = crop.crop;
            let landed: Vec<[f64; 2]> = outline.samples.iter().flatten().copied().collect();
            assert!(landed.len() >= 4, "{stage}: the outline did not land");
            assert_eq!(
                region.size[0], region.size[1],
                "{stage}: observation {observation}'s crop is not square"
            );
            let gaps: Vec<(f64, f64)> = (0..2)
                .map(|axis| {
                    let lo = landed.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min);
                    let hi = landed
                        .iter()
                        .map(|p| p[axis])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let start = region.min[axis] as f64;
                    (lo - start, start + region.size[axis] as f64 - hi)
                })
                .collect();
            for (axis, &(before, after)) in gaps.iter().enumerate() {
                assert!(
                    before >= margin && after >= margin,
                    "{stage}: observation {observation}'s outline is {before} and {after} px \
                     from the crop's edges on axis {axis}"
                );
                assert!(
                    (before - after).abs() < 2.0,
                    "{stage}: observation {observation}'s outline is off centre on axis \
                     {axis}: {before} vs {after} px"
                );
            }
            assert!(
                gaps.iter()
                    .any(|&(before, after)| before < margin + 1.0 && after < margin + 1.0),
                "{stage}: observation {observation}'s crop is wider than the outline on \
                 both axes: {gaps:?}"
            );
            // The demo's patches are a few dozen pixels across, so the crop is
            // read one texel per photograph pixel.
            assert_eq!(
                crop.image.size,
                [region.size[0] as usize, region.size[1] as usize],
                "{stage}: the crop is not one texel per pixel"
            );
            let src = state
                .cached_photograph(id, track.observations[observation].image as usize)
                .expect("a cached photograph");
            let (x, y) = (region.min[0] + 1, region.min[1] + 1);
            let texel = crop.image.pixels[crop.image.size[0] + 1].to_array();
            for (c, &value) in texel.iter().take(3).enumerate() {
                assert_eq!(
                    value,
                    src.level(0).get_pixel(x as u32, y as u32, c as u32),
                    "{stage}: a crop texel is not the photograph's pixel"
                );
            }
        }
    };
    check(&state, "track stage");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    check(&state, "cluster stage");
}

/// A crop's hover view is the crop centred in three times its width and height
/// of the photograph at the crop's own sampling: three crops across and down,
/// the box its middle third, the texels there the crop's, and the outline the
/// crop's moved by one crop width and height.
#[test]
fn a_crop_s_hover_view_holds_the_crop_in_its_middle_third() {
    let (state, id, label, _, _) = on_the_bench();
    let k = super::tile::CONTEXT_FACTOR as usize;
    let observations = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .len();
    for observation in 0..observations {
        let (_, crop, context) = crop_and_context(&state, id, &label, observation);
        let [w, h] = crop.image.size;
        assert_eq!(context.image.size, [w * k, h * k]);
        assert_eq!(context.crop, crop.crop, "the hover view names another crop");
        assert!(
            (context.keypoint - crop.keypoint - egui::vec2(w as f32, h as f32)).length() < 1e-3,
            "the hover view's dot is not the crop's moved by one crop"
        );
        for y in 0..h {
            for x in 0..w {
                let a = crop.image.pixels[y * w + x];
                let b = context.image.pixels[(y + h) * w * k + x + w];
                assert_eq!(a, b, "observation {observation}: texel ({x}, {y}) differs");
            }
        }
        for (a, b) in crop.outline.iter().zip(&context.outline) {
            let (a, b) = (a.expect("landed"), b.expect("landed"));
            assert!(
                (b - a - egui::vec2(w as f32, h as f32)).length() < 1e-3,
                "the hover view's outline is not the crop's moved by one crop"
            );
        }
    }
}

/// The caption gives the patch's two axes in the photograph's pixels. At the
/// track stage each is the axis's length through the lens, which for the
/// demo's cameras is the distance between the projections of the patch's
/// opposite edge midpoints; at the cluster stage it is the shape's column
/// times the template's width.
#[test]
fn a_crop_s_hover_view_gives_the_axes_in_photograph_pixels() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let patch = payload.placement.clone().expect("a patch");
    let row = &track.observations[0];
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, row.image as usize).expect("a view");
    let frame = crate::bench::geometry::anchored_frame(&patch, &camera, &pose, row);
    let across = |s: f64, t: f64| {
        let (xyz, w) = frame.corner_homogeneous(s, t);
        camera.project_homogeneous(&pose, xyz, w).expect("projects")
    };
    let chord = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    let want = [
        chord(across(-1.0, 0.0), across(1.0, 0.0)),
        chord(across(0.0, -1.0), across(0.0, 1.0)),
    ];
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    for (axis, (got, want)) in context.axes_px.iter().zip(want).enumerate() {
        let got = got.expect("the axis projects");
        assert!(
            (got - want).abs() < 0.01 * want,
            "axis {axis} is {got} px, the edge midpoints {want} px apart"
        );
    }
    let caption = super::crop::context_caption(&context);
    for axis in context.axes_px {
        let text = format!("{:.1} px", axis.expect("projects"));
        assert!(caption.contains(&text), "{text} is not in {caption}");
    }

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Cluster(payload) = &track.stage else {
        panic!("the track moved to the cluster stage");
    };
    let shape = track.observations[0].shape().expect("a cluster shape");
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    for (axis, got) in context.axes_px.iter().enumerate() {
        let want = 2.0 * payload.radius * shape[0][axis].hypot(shape[1][axis]);
        let got = got.expect("a cluster axis");
        assert!(
            (got - want).abs() < 1e-9,
            "cluster axis {axis}: {got} vs {want}"
        );
    }
}

/// Resting the pointer on a row's crop shows its hover view, rendered for
/// that row alone, while the row keeps its hover and its click.
#[test]
fn hovering_a_crop_shows_it_in_context_and_keeps_the_row() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let y = row_y(&mut panel, &ctx, &state, 1);
    let x = super::table::ColumnLayout::new().crop_x() + super::table::TILE_SIZE / 2.0;
    let at = egui::pos2(x + 8.0, y + 20.0);
    let response = at_pointer(&mut panel, &ctx, &state, at, false);
    assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
    for _ in 0..12 {
        run_frame(&mut panel, &ctx, &state);
    }
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("\u{2022} Outline: the patch")),
        "no crop hover caption in {texts:?}"
    );
    let rendered: Vec<usize> = panel.crop_contexts.keys().copied().collect();
    assert_eq!(
        rendered,
        vec![1],
        "crop hover views rendered for rows not hovered"
    );
    assert!(
        panel.contexts.is_empty(),
        "the tile's hover view was rendered"
    );

    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(
        response.pick_row,
        Some((1, false)),
        "a click on the crop did not pick the row"
    );
}

/// A crop's hover view marks where the observation sits and, at the track
/// stage, where the track's point projects, each at its own photograph pixel:
/// the picture is the photograph at one texel per pixel, so a mark's texel plus
/// the picture's corner is the pixel it marks. The distance the caption states
/// is the row's reprojection error. At the cluster stage there is a dot and no
/// ring.
#[test]
fn a_crop_s_hover_view_marks_the_keypoint_and_the_projection() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let frame = payload.placement.clone().expect("a patch");
    let position = payload.position.expect("a triangulated point");

    // A sighting of the point in a fourth image, 5 px off its projection.
    let image = 3;
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, image).expect("a view");
    let projected = camera
        .project_homogeneous(&pose, position.coords, frame.w)
        .expect("the demo cameras see every point");
    let keypoint = [projected[0] + 4.0, projected[1] - 3.0];
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, image),
            &crate::bench::Seed::Pixel {
                pixel: keypoint,
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let added = track.observations.len() - 1;
    let error = track.observations[added]
        .track
        .as_ref()
        .and_then(|m| m.reprojection_error)
        .expect("the evaluation measured the new row's error");

    let (_, _, context) = crop_and_context(&state, id, &label, added);
    let region = context.crop;
    let k = super::tile::CONTEXT_FACTOR as i64;
    let corner = [
        (region.min[0] - region.size[0] * (k / 2)) as f64,
        (region.min[1] - region.size[1] * (k / 2)) as f64,
    ];
    let to_photograph = |at: egui::Pos2| [corner[0] + f64::from(at.x), corner[1] + f64::from(at.y)];
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    assert!(
        distance(to_photograph(context.keypoint), keypoint) < 1e-3,
        "the dot is not on the observation"
    );
    let ring = context.projection.expect("a ring");
    assert!(
        distance(to_photograph(ring), projected) < 1e-3,
        "the ring is not on the point's projection"
    );
    assert_eq!(
        context.projection_of,
        Some(super::tile::ProjectionOf::Point)
    );
    let stated = context.projection_px.expect("a projection distance");
    assert!(
        (stated - error).abs() < 1e-6,
        "the hover view states {stated} px, the row {error} px"
    );
    let caption = super::crop::context_caption(&context);
    assert!(
        caption.contains(&format!("{stated:.2} px away along the dashed line")),
        "the caption does not state the distance: {caption}"
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    assert_eq!(context.projection, None);
    assert_eq!(context.projection_of, None);
    assert!(
        super::crop::context_caption(&context).contains("no point to project"),
        "the cluster caption does not say why there is no ring"
    );
}

// ── The tile's frame, and the name column ───────────────────────────────────

/// A row's tile is rendered through the point's frame re-anchored so that its
/// centre projects onto the observation's own keypoint, even when the keypoint
/// sits pixels off where the geometric frame lands; with no keypoint the
/// geometric frame is used as it is.
#[test]
fn the_tile_frame_is_anchored_on_the_observation_s_own_keypoint() {
    let (state, id) = state();
    let node = state.node(id).expect("loaded");
    let view = node.edited().point(POINT).expect("a live point");
    let frame = view.placement().expect("the point has a frame");
    for (k, obs) in view.observations().iter().enumerate() {
        let (camera, pose) =
            crate::bench::geometry::view_of(&node.recon().image_table, obs.image_index as usize)
                .expect("a posed image");
        let stored = view.keypoint_xy(k).expect("a stored keypoint");
        // Displaced from the projection, so the two anchors differ.
        let keypoint = [f64::from(stored[0]) + 6.0, f64::from(stored[1]) - 4.0];
        let centre = |frame: &sfmtool_core::patch::cloud::OrientedPatch| {
            camera
                .project_homogeneous(&pose, frame.center.coords, frame.w)
                .expect("the centre projects")
        };
        let geometric = centre(&frame);
        let miss = (geometric[0] - keypoint[0]).hypot(geometric[1] - keypoint[1]);
        assert!(miss > 1.0, "the geometric frame missed by only {miss} px");

        let anchored = super::patch::render_frame(&frame, &camera, &pose, Some(keypoint));
        let landed = centre(&anchored);
        assert!(
            (landed[0] - keypoint[0]).abs() < 1e-6 && (landed[1] - keypoint[1]).abs() < 1e-6,
            "the anchored frame is centred at {landed:?}, not {keypoint:?}"
        );

        // With no keypoint to anchor on, the stored frame is drawn as it is.
        let unanchored = super::patch::render_frame(&frame, &camera, &pose, None);
        assert_eq!(unanchored.center, frame.center);
    }
}

/// A keypoint whose ray cannot meet the patch -- here a direction patch
/// pointing behind the camera -- leaves the tile on the stored frame rather
/// than on no frame at all.
#[test]
fn a_keypoint_whose_ray_cannot_meet_the_patch_falls_back_to_the_geometric_frame() {
    use nalgebra::{Point3, Vector3};
    let (state, id) = state();
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.node(id).expect("loaded").recon().image_table, 0)
            .expect("a posed image");
    let behind = sfmtool_core::patch::cloud::OrientedPatch::from_infinity_direction(
        Point3::from(-(pose.to_rotation_matrix().transpose() * Vector3::new(0.0, 0.0, -1.0))),
        Vector3::new(0.0, 1.0, 0.0),
        [0.02, 0.02],
    );
    assert!(behind
        .anchored_at_keypoint(&camera, &pose, [320.0, 240.0])
        .is_none());
    let rendered = super::patch::render_frame(&behind, &camera, &pose, Some([320.0, 240.0]));
    assert_eq!(rendered.center, behind.center);
    assert_eq!(rendered.w, behind.w);
}

/// An image name too long for its column is cut in its middle, so the start of
/// the path and the end of the file name both stay in view.
#[test]
fn a_long_image_name_is_cut_in_its_middle() {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name =
            format!("images/a_capture_directory_with_a_long_name/fisheye_left/image_{i:03}.jpg");
    }
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    let cut: Vec<&String> = texts
        .iter()
        .filter(|t| t.starts_with("images/") && t.contains('\u{2026}'))
        .collect();
    assert_eq!(cut.len(), 3, "not one cut name per row: {texts:?}");
    for (i, name) in cut.iter().enumerate() {
        assert!(name.ends_with(&format!("_{i:03}.jpg")), "{name}");
    }
}

/// One frame's row switches and pins as AccessKit reports them: for each
/// widget labelled `Keep` or `Pin`, its label and whether it is enabled. Also
/// how many shapes the frame filled with the enabled switch's green.
fn row_switches(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
) -> (Vec<(String, bool)>, usize) {
    fn greens(shape: &egui::Shape, count: &mut usize) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| greens(s, count)),
            egui::Shape::Rect(rect) if rect.fill == super::table::KEEP_ON_FILL => *count += 1,
            _ => {}
        }
    }
    let mut output = ctx.run_ui(input(Vec::new()), |ui| {
        panel.show(ui, state);
    });
    output.textures_delta.clear();
    let update = output
        .platform_output
        .accesskit_update
        .expect("AccessKit is on");
    let switches = update
        .nodes
        .iter()
        .filter_map(|(_, node)| {
            let label = node.label()?;
            (node.role() == egui::accesskit::Role::CheckBox && (label == "Keep" || label == "Pin"))
                .then(|| (label.to_string(), !node.is_disabled()))
        })
        .collect();
    let mut green = 0;
    for clipped in &output.shapes {
        greens(&clipped.shape, &mut green);
    }
    (switches, green)
}

/// On a view-only bench the *Keep* switch and pin of every row are drawn
/// disabled, as every other edit control there is: AccessKit reports them
/// disabled, and no switch is filled the enabled green. On an editable bench
/// the same rows report them enabled, with the kept rows green.
#[test]
fn a_view_only_bench_draws_the_row_switches_disabled() {
    // A `sift_files` node: its bench is view-only.
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut sift, id) = crate::state::edits::tests::convertible_state(dir.path());
    sift.select_recon(id);
    sift.put_point_on_bench(PointRef::new(id, 0), None)
        .expect("a live point");
    assert!(sift.bench_view_only_refusal(id).is_some());
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    run_frame(&mut panel, &ctx, &sift);
    let (switches, green) = row_switches(&mut panel, &ctx, &sift);
    let rows = panel.rows().len();
    assert!(rows > 0, "the table drew no rows");
    assert_eq!(switches.len(), 2 * rows, "{switches:?}");
    assert!(
        switches.iter().all(|(_, enabled)| !enabled),
        "a row control on a view-only bench is enabled: {switches:?}"
    );
    assert_eq!(green, 0, "a greyed switch was filled the enabled green");

    // An `embedded_patches` node: its bench takes edits.
    let (mut state, id) = state();
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    assert!(state.bench_edit_refusal(id).is_none());
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    run_frame(&mut panel, &ctx, &state);
    let (switches, green) = row_switches(&mut panel, &ctx, &state);
    let rows = panel.rows().len();
    assert_eq!(switches.len(), 2 * rows, "{switches:?}");
    assert!(
        switches.iter().all(|(_, enabled)| *enabled),
        "a row control on an editable bench is disabled: {switches:?}"
    );
    let kept = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .iter()
        .filter(|o| o.verdict == Verdict::In)
        .count();
    assert!(kept > 0, "the fixture keeps no row");
    assert_eq!(green, kept, "each kept row's switch is green");
}

// ── The projection error bar ────────────────────────────────────────────────

/// The reprojection error in px of each observation of the focused item, by
/// observation index: the reading the *Proj. err* cell prints and its bar
/// judges.
fn projection_errors(state: &AppState, id: ReconId, label: &str) -> Vec<f64> {
    let track = state.bench_track(id, label).expect("on the bench");
    track
        .observations
        .iter()
        .map(|o| {
            let m = o.track.as_ref().expect("a track-stage reading");
            m.reprojection_error
                .or(m.projection_offset_px)
                .expect("a projection error")
        })
        .collect()
}

/// The projection error's bar colours the px line of the *Proj. err* cell and
/// nothing under it, and a row it fails says so in its *Keep* hover.
#[test]
fn the_projection_error_bar_judges_the_px_line() {
    use sfmtool_core::bench::BarCheck;

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let errors = projection_errors(&state, id, &label);
    let mut sorted = errors.clone();
    sorted.sort_by(f64::total_cmp);
    assert!(sorted[0] < sorted[2], "the fixture's errors are all equal");
    // Between the smallest and the largest, so the bar passes one and fails
    // another.
    let bar = 0.5 * (sorted[0] + sorted[2]);
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.max_projection_error_px = bar;
    });
    for row in panel.rows() {
        let error = errors[row.observation];
        let expected = if error <= bar {
            BarCheck::Pass
        } else {
            BarCheck::Fail
        };
        assert_eq!(row.checks[2], [expected, BarCheck::NotJudged], "{row:?}");
        assert_eq!(
            row.keep_hover.contains("Proj. err is over the bar"),
            expected == BarCheck::Fail,
            "{}",
            row.keep_hover
        );
    }
}

/// The bench's default bar is 3 px, and its box stands in the threshold row.
#[test]
fn the_projection_error_bar_defaults_to_3_px() {
    assert_eq!(Thresholds::default().max_projection_error_px, 3.0);
    assert!(super::table::PROJECTION_ERROR_TIP.contains("box under this heading"));
}

// ── Ordering the rows ───────────────────────────────────────────────────────

/// The images of the rows the table drew, top to bottom.
fn row_images(panel: &TrackBody) -> Vec<u32> {
    panel.rows().iter().map(|row| row.image).collect()
}

/// Click the heading that reads `heading`.
fn click_heading(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState, heading: &str) {
    let texts = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    });
    let at = texts
        .iter()
        .find(|t| t.text == heading)
        .unwrap_or_else(|| panic!("{heading:?} is not drawn"))
        .rect
        .center();
    at_pointer(panel, ctx, state, at, true);
    run_frame(panel, ctx, state);
}

/// The rows start in increasing order of image, and a click on *Img* reverses
/// that, and a second click puts it back.
#[test]
fn the_rows_start_in_increasing_order_of_image_and_a_click_reverses_it() {
    let (state, _id, _label, mut panel, ctx) = measured_on_the_bench();
    assert_eq!(panel.sort, super::table::TableSort::default());
    assert_eq!(row_images(&panel), [0, 1, 2]);

    click_heading(&mut panel, &ctx, &state, "Img");
    assert!(panel.sort.descending);
    assert_eq!(row_images(&panel), [2, 1, 0]);

    click_heading(&mut panel, &ctx, &state, "Img");
    assert!(!panel.sort.descending);
    assert_eq!(row_images(&panel), [0, 1, 2]);
}

/// A click on *Proj. err* orders the rows by the px they print, the largest
/// error first, and a second click puts the smallest first.
#[test]
fn a_click_on_proj_err_orders_the_rows_by_their_error() {
    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let errors = projection_errors(&state, id, &label);
    let mut by_error: Vec<u32> = (0..3).collect();
    by_error.sort_by(|&a, &b| errors[b as usize].total_cmp(&errors[a as usize]));

    click_heading(&mut panel, &ctx, &state, "Proj. err");
    assert_eq!(panel.sort.column, super::table::SortColumn::ProjectionError);
    assert!(panel.sort.descending);
    assert_eq!(row_images(&panel), by_error);

    click_heading(&mut panel, &ctx, &state, "Proj. err");
    by_error.reverse();
    assert_eq!(row_images(&panel), by_error);
}

/// A click on *Keep* orders the rows by how many bars they fail, most first,
/// and a second click puts the fewest first.
#[test]
fn a_click_on_keep_orders_the_rows_by_how_many_bars_they_fail() {
    use sfmtool_core::bench::BarCheck;

    let (state, _id, _label, mut panel, ctx) = measured_on_the_bench();
    let failed = |panel: &TrackBody| -> Vec<usize> {
        panel
            .rows()
            .iter()
            .map(|row| {
                row.checks
                    .iter()
                    .flatten()
                    .filter(|&&c| c == BarCheck::Fail)
                    .count()
            })
            .collect()
    };
    click_heading(&mut panel, &ctx, &state, "Keep");
    assert_eq!(panel.sort.column, super::table::SortColumn::Verdict);
    let decreasing = failed(&panel);
    assert!(
        decreasing.windows(2).all(|w| w[0] >= w[1]),
        "{decreasing:?}"
    );
    assert_ne!(
        decreasing.first(),
        decreasing.last(),
        "the fixture's rows all fail as many bars"
    );

    click_heading(&mut panel, &ctx, &state, "Keep");
    let increasing = failed(&panel);
    assert!(
        increasing.windows(2).all(|w| w[0] <= w[1]),
        "{increasing:?}"
    );
}

/// A click on another heading orders the rows worst first where a bar judges
/// the column and increasing where none does; a click on the same heading
/// reverses the order.
#[test]
fn a_heading_click_starts_worst_first_and_a_second_reverses() {
    use super::table::{SortColumn, TableSort};
    let start = TableSort::default();
    for (column, descending) in [
        (SortColumn::Image, false),
        (SortColumn::Verdict, true),
        (SortColumn::Zncc, false),
        (SortColumn::SelfSimilarity, true),
        (SortColumn::ProjectionError, true),
        (SortColumn::Shift, true),
        (SortColumn::Zoom, false),
        (SortColumn::Reference, false),
        (SortColumn::Status, false),
        (SortColumn::Name, false),
    ] {
        let from = TableSort {
            column: if column == SortColumn::Name {
                SortColumn::Image
            } else {
                SortColumn::Name
            },
            descending: !descending,
        };
        assert_eq!(
            from.clicked(column),
            TableSort { column, descending },
            "{column:?}"
        );
    }
    let zncc = start.clicked(SortColumn::Zncc);
    assert_eq!((zncc.column, zncc.descending), (SortColumn::Zncc, false));
    let reversed = zncc.clicked(SortColumn::Zncc);
    assert_eq!(
        (reversed.column, reversed.descending),
        (SortColumn::Zncc, true)
    );
    let name = reversed.clicked(SortColumn::Name);
    assert_eq!((name.column, name.descending), (SortColumn::Name, false));
}

/// A row with no reading sorts after every row with one, whichever way the
/// order runs, and rows that tie keep increasing order of image.
#[test]
fn a_row_with_no_reading_sorts_last_both_ways_and_ties_go_by_image() {
    use super::table::{sorted_rows, SortColumn, SortKey, TableSort};
    let keys = [
        Some(SortKey::Number(2.0, 0.0)),
        None,
        Some(SortKey::Number(1.0, 0.0)),
        Some(SortKey::Number(2.0, 0.0)),
    ];
    let images = [7, 0, 5, 3];
    let up = TableSort {
        column: SortColumn::Zncc,
        descending: false,
    };
    assert_eq!(sorted_rows(&keys, &images, up), [2, 3, 0, 1]);
    let down = TableSort {
        descending: true,
        ..up
    };
    assert_eq!(sorted_rows(&keys, &images, down), [3, 0, 2, 1]);

    let names = [
        Some(SortKey::Text("b".into())),
        Some(SortKey::Text("a".into())),
    ];
    assert_eq!(sorted_rows(&names, &[0, 1], up), [1, 0]);
}

/// Every heading that orders the rows says so in its hover text, and the
/// pictures and *From* order nothing.
#[test]
fn the_sortable_headings_are_the_ones_with_readings() {
    use super::table::SortColumn;
    for (_, heading, _) in super::table::ColumnLayout::new().headers(super::BodyMode::Edited) {
        let sortable = SortColumn::of_heading(heading).is_some();
        assert_eq!(
            sortable,
            !matches!(heading, "Crop" | "Patch" | "From"),
            "{heading:?}"
        );
    }
    assert_eq!(SortColumn::of_heading("Verdict"), Some(SortColumn::Verdict));
}

/// The *Verdict* text of an `out` row says how many bars it fails, and one that
/// fails none, which lost its image to another sighting, says `out` alone.
#[test]
fn the_verdict_text_counts_the_bars_an_out_row_fails() {
    use super::table::verdict_text;
    use super::Judgement;
    use sfmtool_core::bench::{BarCheck, BarChecks};
    let checks = |fail: usize| {
        let mut all = [BarCheck::Pass; 5];
        all[..fail].fill(BarCheck::Fail);
        BarChecks {
            min_zncc: all[0],
            min_zncc_middle: all[1],
            max_shift_px: all[2],
            max_zncc_self_similarity_radius: all[3],
            max_projection_error_px: all[4],
        }
    };
    let judged = |fail, proposal| Judgement {
        checks: checks(fail),
        proposal,
    };
    assert_eq!(verdict_text(None), "-");
    assert_eq!(verdict_text(Some(&judged(0, Verdict::In))), "in");
    assert_eq!(verdict_text(Some(&judged(0, Verdict::Out))), "out");
    assert_eq!(verdict_text(Some(&judged(2, Verdict::Out))), "out (2)");
}

// ── The tile's Jacobian and zoom ────────────────────────────────────────────

/// A pinhole camera at the origin, looking down `-Z` with `+Y` up, 640 by 480
/// with a focal length of 500 px and its principal point in the middle.
fn pinhole_at_origin() -> (
    sfmtool_core::camera::CameraIntrinsics,
    sfmtool_core::geometry::RigidTransform,
) {
    use sfmtool_core::camera::{CameraIntrinsics, CameraModel};
    let camera = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let pose = sfmtool_core::geometry::RigidTransform::from_wxyz_translation(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    );
    (camera, pose)
}

/// A square patch 1 unit wide, centred at `center`, facing the camera at the
/// origin and upright in it.
fn facing_patch(center: [f64; 3]) -> sfmtool_core::patch::cloud::OrientedPatch {
    use nalgebra::{Point3, Vector3};
    sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::new(center[0], center[1], center[2]),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    )
}

/// Each entry of `got` within `tolerance` of the same entry of `want`.
#[track_caller]
fn assert_jacobian_near(got: super::patch::PatchJacobian, want: [[f64; 2]; 2], tolerance: f64) {
    for (r, c, name) in [
        (0, 0, "dx/dcol"),
        (0, 1, "dx/drow"),
        (1, 0, "dy/dcol"),
        (1, 1, "dy/drow"),
    ] {
        assert!(
            (got.0[r][c] - want[r][c]).abs() < tolerance,
            "{name} is {}, not {}: {got:?}",
            got.0[r][c],
            want[r][c]
        );
    }
}

/// The patch resolution `R` of a reconstruction that stores no patch
/// bitmaps: the bench evaluation's own, 24, and not the 64 texels a tile is
/// drawn at.
fn fallback_resolution() -> usize {
    let r = sfmtool_core::bench::EvaluateOptions::default()
        .localize
        .resolution as usize;
    assert_eq!(r, 24);
    assert_ne!(r, super::patch::PATCH_RES as usize, "this would prove less");
    r
}

/// The Jacobian at the centre of a patch with no keypoint to anchor on, per
/// grid px at [`fallback_resolution`], as [`super::patch_jacobian`] computes
/// it for a row of a reconstruction with no patch bitmaps.
fn patch_centre_jacobian(
    patch: &sfmtool_core::patch::cloud::OrientedPatch,
    camera: &sfmtool_core::camera::CameraIntrinsics,
    pose: &sfmtool_core::geometry::RigidTransform,
) -> Option<super::patch::PatchJacobian> {
    sfmtool_core::camera::warp_map::patch_grid_jacobian(patch, camera, pose, fallback_resolution())
        .map(super::patch::PatchJacobian)
}

/// The warp map a tile rendered through `patch` with no keypoint is drawn
/// through, at the tile's display resolution.
fn tile_map(
    patch: &sfmtool_core::patch::cloud::OrientedPatch,
    camera: &sfmtool_core::camera::CameraIntrinsics,
    pose: &sfmtool_core::geometry::RigidTransform,
) -> sfmtool_core::camera::WarpMap {
    sfmtool_core::camera::WarpMap::from_patch(patch, camera, pose, super::patch::PATCH_RES)
}

/// A patch facing a pinhole camera square on is a pure scaling in the
/// photograph, so the Jacobian at the patch's centre is diagonal, and its
/// scale is the patch's width in pixels over the `R` grid px of its side: 1
/// unit at depth 4 under a 500 px focal length is 125 px, 125 / 24 photograph
/// pixels per grid px at the fallback `R` of 24, and the zoom is the
/// reciprocal, 24 / 125, the same in both directions. The 64 texels the tile
/// is drawn at do not enter it.
#[test]
fn a_fronto_parallel_patch_has_a_diagonal_jacobian_of_its_width_over_r() {
    let (camera, pose) = pinhole_at_origin();
    let patch = facing_patch([0.0, 0.0, -4.0]);
    let r = fallback_resolution();
    let map = sfmtool_core::camera::WarpMap::from_patch(&patch, &camera, &pose, r as u32);
    let jacobian = patch_centre_jacobian(&patch, &camera, &pose).expect("in front of the camera");

    let scale = 125.0 / r as f64;
    assert_jacobian_near(jacobian, [[scale, 0.0], [0.0, scale]], 1e-3);
    let [low, high] = jacobian.zoom_range().expect("a zoom");
    let zoom = 1.0 / scale;
    assert!((low - zoom).abs() < 1e-3 && (high - zoom).abs() < 1e-3);
    let mean = jacobian.mean_zoom().expect("a mean zoom");
    assert!((mean - zoom).abs() < 1e-3, "{mean}");

    // Printed as the cell prints it: both zooms, even where the two
    // directions agree.
    assert_eq!(super::table::zoom_text(Some(jacobian)), "0.19/0.19\u{d7}");

    // The straddling difference agrees with the per-px Jacobians of the
    // R-grid's warp, averaged over the four middle grid px, where the warp is
    // smooth.
    let mut map = map;
    map.compute_svd();
    let mut mean = [[0.0f64; 2]; 2];
    let (lo, hi) = (r as u32 / 2 - 1, r as u32 / 2);
    for (col, row) in [(lo, lo), (hi, lo), (lo, hi), (hi, hi)] {
        let j = map.get_jacobian(col, row);
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            mean[r][c] += f64::from(j[r][c]) / 4.0;
        }
    }
    assert_jacobian_near(jacobian, mean, 1e-3);
}

/// A patch turned within its own plane by a known angle has a Jacobian with
/// that turn in it, signs included. The patch faces the camera with its `u`
/// axis turned 30 degrees from `+X` towards `+Y`, which is up and to the right
/// in the photograph, whose `y` runs down. So one grid px right along a row of
/// the patch moves the photograph's `y` up (`dy/dcol` negative), and one grid
/// px down a column of the patch moves its `x` right (`dx/drow` positive):
/// `J = k [[cos, sin], [-sin, cos]]`, `k` being the fronto-parallel scale
/// of 125 / 24 per grid px.
#[test]
fn a_patch_turned_in_its_plane_has_off_diagonal_entries_of_the_turn() {
    use nalgebra::{Point3, Vector3};
    let (camera, pose) = pinhole_at_origin();
    let (s, c) = 30f64.to_radians().sin_cos();
    // From the camera, `up` turned 30 degrees towards `-X` makes `u = v x n`
    // turn the same way from `+X` towards `+Y`.
    let patch = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(-s, c, 0.0),
        [0.5, 0.5],
    );
    assert!((patch.u_axis - Vector3::new(c, s, 0.0)).norm() < 1e-12);

    let jacobian = patch_centre_jacobian(&patch, &camera, &pose).expect("a Jacobian");
    let k = 125.0 / fallback_resolution() as f64;
    assert_jacobian_near(jacobian, [[k * c, k * s], [-k * s, k * c]], 1e-3);
    assert!(jacobian.0[0][1] > 0.0 && jacobian.0[1][0] < 0.0);

    // A turn changes no zoom.
    let [low, high] = jacobian.zoom_range().expect("a zoom");
    assert!((low - 1.0 / k).abs() < 1e-3 && (high - 1.0 / k).abs() < 1e-3);
}

/// A 640 by 480 photograph with detail down to a few pixels, as a full
/// pyramid: what a mip level averages away differs from the full-resolution
/// pixels.
fn textured_pyramid() -> sfmtool_core::camera::image::ImageU8Pyramid {
    use sfmtool_core::camera::image::ImageU8Pyramid;
    let (w, h) = (640u32, 480u32);
    let data: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let v = ((x * 37 + y * 11) % 7 * 36) as u8;
            [v, 255 - v, ((x / 3 + y / 5) % 2 * 200) as u8]
        })
        .collect();
    let src = ImageU8::new(w, h, 3, data);
    ImageU8Pyramid::build(&src, ImageU8Pyramid::full_levels(w, h))
}

/// `image`, three channels, as the opaque RGBA picture a tile is.
fn opaque(image: &ImageU8) -> egui::ColorImage {
    let rgba: Vec<u8> = image
        .data()
        .chunks(3)
        .flat_map(|px| [px[0], px[1], px[2], 255])
        .collect();
    egui::ColorImage::from_rgba_unmultiplied(
        [image.width() as usize, image.height() as usize],
        &rgba,
    )
}

/// The sampler a facing view's tile is drawn with: the sampler rule leaves a
/// view facing the patch on it.
const MIP: sfmtool_core::camera::sampler::Sampler =
    sfmtool_core::camera::sampler::Sampler::BilinearMip;

/// Under `bilinear_mip` the tile is one bilinear sample per texel from the mip
/// level the warp's compression picks. A patch 250 px wide over 64 texels shrinks the
/// photograph about 3.9 times at every texel, which is level 2, so the whole
/// tile is plain bilinear on level 2 of the pyramid at the warp's coordinates
/// divided by 4. A patch 62.5 px wide shrinks nothing, so every texel reads
/// level 0 and the tile is plain bilinear on the photograph to the bit.
#[test]
fn a_tile_reads_the_mip_level_its_warp_shrinks_the_photograph_to() {
    use sfmtool_core::camera::remap::remap_bilinear;
    use sfmtool_core::camera::WarpMap;
    let (camera, pose) = pinhole_at_origin();
    let src = textured_pyramid();

    // 1 unit at depth 2 is 250 px over 64 texels, 3.9 px a texel.
    let near = facing_patch([0.0, 0.0, -2.0]);
    let mut map = tile_map(&near, &camera, &pose);
    map.compute_svd();
    let (w, h) = (map.width(), map.height());
    let mut at_level_2 = Vec::with_capacity(2 * (w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            assert!(
                map.is_valid(col, row),
                "texel ({col}, {row}) is off the photograph"
            );
            let (sigma_major, ..) = map.get_svd(col, row);
            assert!(
                (2f32.powf(1.5)..2f32.powf(2.5)).contains(&sigma_major),
                "texel ({col}, {row}) shrinks the photograph {sigma_major} times, not level 2"
            );
            let (x, y) = map.get(col, row);
            at_level_2.extend_from_slice(&[x / 4.0, y / 4.0]);
        }
    }
    let want = opaque(&remap_bilinear(
        src.level(2),
        &WarpMap::new(w, h, at_level_2),
    ));
    let tile = super::patch::patch_color_image(&near, &camera, &pose, None, &src, MIP);
    assert_eq!(tile, want, "the shrinking tile is not level 2 throughout");
    assert_ne!(
        tile,
        opaque(&remap_bilinear(src.level(0), &map)),
        "level 2 of this photograph is the photograph itself, so this proves less"
    );

    // At depth 8 it is 62.5 px over 64 texels: no texel spans more than a
    // pixel, so every texel reads level 0.
    let far = facing_patch([0.0, 0.0, -8.0]);
    let map = tile_map(&far, &camera, &pose);
    let tile = super::patch::patch_color_image(&far, &camera, &pose, None, &src, MIP);
    assert_eq!(
        tile,
        opaque(&remap_bilinear(src.level(0), &map)),
        "a tile that does not shrink is not plain bilinear"
    );
}

/// A patch tilted away from the camera is foreshortened along the tilt, so
/// the two singular directions give two zooms, and the cell prints the range,
/// least first.
#[test]
fn a_tilted_patch_prints_a_range_of_zooms() {
    use nalgebra::{Point3, Vector3};
    let (camera, pose) = pinhole_at_origin();
    // Turned 60 degrees about the vertical, so it is half as wide in the
    // photograph as it is tall.
    let (s, c) = 60f64.to_radians().sin_cos();
    let patch = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(s, 0.0, c),
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    );
    let jacobian = patch_centre_jacobian(&patch, &camera, &pose).expect("a Jacobian");
    let [low, high] = jacobian.zoom_range().expect("a zoom");
    assert!(
        (high / low - 2.0).abs() < 0.01,
        "the zooms {low} and {high} are not one twice the other"
    );
    // 125 px over 24 grid px up the patch, and half that across it.
    assert_eq!(super::table::zoom_text(Some(jacobian)), "0.19/0.38\u{d7}");
    let mean = jacobian.mean_zoom().expect("a mean zoom");
    assert!(((low * high).sqrt() - mean).abs() < 1e-9);
}

/// The sampler rule reads the same Jacobian the *Zoom* cell prints. The
/// facing patch at depth 4 compresses both axes alike (5.2 photograph px per
/// grid px) and stays on `bilinear_mip`; turned 60° it compresses one axis
/// twice as much as the other, `bilinear_mip` would read the other axis at
/// level 2, 4 / 2.6 = 1.5× too coarsely, and the view moves to the
/// anisotropic sampler, which draws a different tile.
#[test]
fn a_tilted_patch_is_drawn_with_the_anisotropic_sampler() {
    use nalgebra::{Point3, Vector3};
    use sfmtool_core::camera::sampler::Sampler;
    let (camera, pose) = pinhole_at_origin();
    let facing = facing_patch([0.0, 0.0, -4.0]);
    let jacobian = patch_centre_jacobian(&facing, &camera, &pose).expect("a Jacobian");
    assert_eq!(jacobian.sampler(), Sampler::BilinearMip);
    assert!(super::table::zoom_sampler_text(&jacobian).contains("bilinear_mip"));

    let (s, c) = 60f64.to_radians().sin_cos();
    let tilted = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(s, 0.0, c),
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    );
    let jacobian = patch_centre_jacobian(&tilted, &camera, &pose).expect("a Jacobian");
    assert!(
        jacobian.minor_axis_loss() >= 1.5,
        "{}",
        jacobian.minor_axis_loss()
    );
    assert_eq!(jacobian.sampler(), Sampler::Anisotropic);
    assert!(super::table::zoom_sampler_text(&jacobian).contains("anisotropic"));

    let src = textured_pyramid();
    let aniso =
        super::patch::patch_color_image(&tilted, &camera, &pose, None, &src, Sampler::Anisotropic);
    let mip = super::patch::patch_color_image(&tilted, &camera, &pose, None, &src, MIP);
    assert_ne!(aniso, mip, "the two samplers drew the same tile");
}

/// The *Zoom* cell's and heading's hover texts describe the sampler choice
/// they are given. The patch turned 60° loses 1.5 to 2 times along its less
/// compressed axis, so it moves to the anisotropic sampler at the default
/// threshold and stays on `bilinear_mip` at 2; the facing patch at depth 20
/// is compressed less than √2 and reads the full-resolution level; and under a
/// fixed choice every view names the fixed sampler.
#[test]
fn the_zoom_hover_texts_describe_the_sampler_choice() {
    use super::table::{zoom_sampler_text_for, zoom_tip};
    use nalgebra::{Point3, Vector3};
    use sfmtool_core::camera::sampler::{Sampler, SamplerChoice};
    let (camera, pose) = pinhole_at_origin();
    let (s, c) = 60f64.to_radians().sin_cos();
    let tilted = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::new(s, 0.0, c),
        Vector3::new(0.0, 1.0, 0.0),
        [0.5, 0.5],
    );
    let tilted = patch_centre_jacobian(&tilted, &camera, &pose).expect("a Jacobian");
    let far = patch_centre_jacobian(&facing_patch([0.0, 0.0, -20.0]), &camera, &pose)
        .expect("a Jacobian");

    let rule = SamplerChoice::per_view();
    let text = zoom_sampler_text_for(rule, &tilted);
    assert!(
        text.starts_with("Rendered with the anisotropic sampler") && text.contains("1.5\u{d7}"),
        "{text}"
    );
    let text = zoom_sampler_text_for(rule, &far);
    assert!(
        text.starts_with("Rendered with the bilinear_mip sampler")
            && text.contains("full-resolution level"),
        "{text}"
    );

    let strict = SamplerChoice::PerView {
        anisotropic_threshold: 2.0,
    };
    let text = zoom_sampler_text_for(strict, &tilted);
    assert!(
        text.starts_with("Rendered with the bilinear_mip sampler")
            && text.contains("under the 2\u{d7}"),
        "{text}"
    );
    assert!(zoom_tip(strict).contains("at least 2\u{d7} too coarsely"));

    let fixed = SamplerChoice::Fixed(Sampler::Bilinear);
    let text = zoom_sampler_text_for(fixed, &tilted);
    assert!(
        text.starts_with("Rendered with the bilinear sampler, which the bench renders every"),
        "{text}"
    );
    let tip = zoom_tip(fixed);
    assert!(
        tip.contains("every tile with the bilinear sampler"),
        "{tip}"
    );
    assert!(!tip.contains("1.5\u{d7}"), "{tip}");
}

/// The zoom is geometry alone: a patch whose centre projects has one even
/// where the middle of its tile is off the photograph. Only a patch whose
/// centre does not project, here behind the camera, has none, and the *Zoom*
/// cell prints `-` for it.
#[test]
fn a_tile_whose_middle_is_off_the_photograph_still_has_a_zoom() {
    let (camera, pose) = pinhole_at_origin();
    // 330 px right of the principal point, which is 10 px past the right edge;
    // the patch is 125 px wide, so its left 52.5 px are on the photograph.
    let off_the_side = facing_patch([2.64, 0.0, -4.0]);
    let map = tile_map(&off_the_side, &camera, &pose);
    assert!(
        !map.is_valid(map.width() / 2, map.height() / 2),
        "the middle of the tile is on the photograph, so this proves less"
    );
    let jacobian = patch_centre_jacobian(&off_the_side, &camera, &pose).expect("a Jacobian");
    // A fronto-parallel patch under a pinhole is a pure scaling wherever it
    // sits: 125 px across 24 grid px.
    let k = 125.0 / fallback_resolution() as f64;
    assert_jacobian_near(jacobian, [[k, 0.0], [0.0, k]], 1e-3);
    assert_eq!(super::table::zoom_text(Some(jacobian)), "0.19/0.19\u{d7}");

    let behind = facing_patch([0.0, 0.0, 4.0]);
    assert_eq!(patch_centre_jacobian(&behind, &camera, &pose), None);

    assert_eq!(super::table::zoom_text(None), "-");
}

/// A patch seen edge on has no zoom. Its plane passes through the camera, so
/// the four points the Jacobian is read across project onto one line, and the
/// finite difference leaves only a rounding residue across it, a smaller
/// singular value near 1e-14 of the larger, which would otherwise print as a
/// zoom of 10¹³. The cell prints `-` for it, as `get_bench_track` reports
/// `patch_zoom` as null.
#[test]
fn a_patch_seen_edge_on_has_no_zoom() {
    use nalgebra::{Point3, Vector3};
    use sfmtool_core::camera::warp_map::singular_values_2x2;
    let (camera, pose) = pinhole_at_origin();
    let center = Vector3::<f64>::new(0.7, -0.2, -3.0);
    // Normal to the line of sight, so the plane holds the camera centre.
    let normal = Vector3::new(3.0, 0.0, 0.7).normalize();
    assert!(normal.dot(&center).abs() < 1e-12);
    let edge_on = sfmtool_core::patch::cloud::OrientedPatch::from_center_normal(
        Point3::from(center),
        normal,
        Vector3::y(),
        [0.5, 0.5],
    );
    let jacobian = patch_centre_jacobian(&edge_on, &camera, &pose).expect("the centre projects");
    let [major, minor] = singular_values_2x2(jacobian.0);
    assert!(major > 0.1, "{jacobian:?}");
    assert!(minor <= major * 1e-9, "not edge on: {major} {minor}");
    assert_eq!(jacobian.zoom_range(), None, "{jacobian:?}");
    assert_eq!(jacobian.mean_zoom(), None);
    assert_eq!(super::table::zoom_text(Some(jacobian)), "-");
}

/// The zoom is read from the patch re-anchored where the observation sits, the
/// placement its tile is rendered through, not from the patch as stored. One
/// observation's keypoint is moved a few px off the patch's projection, and
/// its Jacobian is exactly that of the patch re-anchored on the moved
/// keypoint, which differs from the stored patch's.
#[test]
fn the_zoom_reads_the_patch_reanchored_on_the_keypoint() {
    use sfmtool_core::bench::{Stage, TrackMeasurement};
    use sfmtool_core::camera::warp_map::patch_grid_jacobian;
    let (state, id, label, _panel, _ctx) = on_the_bench();
    let mut track = (**state.bench_track(id, &label).expect("on the bench")).clone();
    let recon = state.node(id).expect("loaded").recon();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track made from a point is at the track stage");
    };
    let patch = payload.placement.clone().expect("a patch");
    let row = 1;
    let image = &recon.image_table.images[track.observations[row].image as usize];
    let camera = &recon.image_table.cameras[image.camera_index as usize];
    let pose = crate::scene::cam_from_world(image);
    let [x, y] = camera
        .project_homogeneous(&pose, patch.center.coords, patch.w)
        .expect("the patch projects");
    let moved = [x + 4.0, y - 3.0];
    track.observations[row].track = Some(TrackMeasurement {
        keypoint: Some([moved[0] as f32, moved[1] as f32]),
        ..TrackMeasurement::default()
    });
    let site = track.observations[row].site().expect("a site");

    // The reconstruction stores no patch bitmaps, so `R` is the fallback.
    let resolution = crate::bench::patch_resolution(recon) as usize;
    assert_eq!(resolution, fallback_resolution());
    let anchored = patch
        .anchored_at_keypoint(camera, &pose, site)
        .expect("the keypoint's ray meets the patch");
    let want = patch_grid_jacobian(&anchored, camera, &pose, resolution).expect("projects");
    let stored = patch_grid_jacobian(&patch, camera, &pose, resolution).expect("projects");
    assert!(
        (0..2).any(|r| (0..2).any(|c| (want[r][c] - stored[r][c]).abs() > 1e-6)),
        "re-anchoring changes nothing here, so this proves less: {want:?} {stored:?}"
    );

    let got = super::patch_jacobian(recon, &track, row).expect("a Jacobian");
    assert_eq!(got.0, want);
}

/// A reconstruction that stores no patch bitmaps has the bench evaluation's
/// resolution, 24, and every row's Jacobian is the patch-grid Jacobian at 24,
/// not at the 64 texels its tile is drawn at.
#[test]
fn with_no_patch_bitmaps_the_zoom_is_per_grid_px_of_the_evaluation() {
    use sfmtool_core::camera::warp_map::patch_grid_jacobian;
    let (state, id, label, _panel, _ctx) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let recon = state.node(id).expect("loaded").recon();
    assert!(recon.point_set.patch_bitmaps_y_x_rgba.is_none());
    assert_eq!(
        crate::bench::patch_resolution(recon) as usize,
        fallback_resolution()
    );
    let sfmtool_core::bench::Stage::Track(payload) = &track.stage else {
        panic!("a track made from a point is at the track stage");
    };
    let patch = payload.placement.as_ref().expect("a patch");
    for row in 0..track.observations.len() {
        let image = &recon.image_table.images[track.observations[row].image as usize];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let pose = crate::scene::cam_from_world(image);
        let site = track.observations[row].site().expect("a site");
        let anchored = patch
            .anchored_at_keypoint(camera, &pose, site)
            .unwrap_or_else(|| patch.clone());
        let at = |r: usize| patch_grid_jacobian(&anchored, camera, &pose, r).expect("projects");
        let got = super::patch_jacobian(recon, &track, row).expect("a Jacobian");
        assert_eq!(got.0, at(fallback_resolution()), "row {row}");
        assert_ne!(
            got.0,
            at(super::patch::PATCH_RES as usize),
            "row {row} is at the display resolution"
        );
    }
}

/// The zoom is per grid px at the reconstruction's own patch resolution, the
/// `patch_bitmap_resolution` an `.sfmr` declares: the same rows on the same
/// reconstruction with patch bitmaps 48 px a side have Jacobians half the
/// size, and so zooms twice as large, as they have at the fallback 24.
#[test]
fn the_zoom_is_per_grid_px_of_the_stored_patch_bitmaps() {
    let (state, id, label, _panel, _ctx) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let recon = state.node(id).expect("loaded").recon();
    let mut at_48 = recon.clone();
    let points = at_48.point_set.points.len();
    at_48.point_set.patch_bitmaps_y_x_rgba = Some(std::sync::Arc::new(ndarray::Array4::zeros((
        points, 48, 48, 4,
    ))));
    assert_eq!(crate::bench::patch_resolution(&at_48), 48);

    for row in 0..track.observations.len() {
        let at_24 = super::patch_jacobian(recon, &track, row).expect("a Jacobian");
        let got = super::patch_jacobian(&at_48, &track, row).expect("a Jacobian");
        let scale = at_24.0.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (got.0[r][c] - at_24.0[r][c] / 2.0).abs() < 1e-3 * scale,
                    "row {row}: {got:?} is not half of {at_24:?}"
                );
            }
        }
        let [low_24, high_24] = at_24.zoom_range().expect("a zoom");
        let [low, high] = got.zoom_range().expect("a zoom");
        assert!(
            (low / low_24 - 2.0).abs() < 1e-3,
            "row {row}: {low} {low_24}"
        );
        assert!(
            (high / high_24 - 2.0).abs() < 1e-3,
            "row {row}: {high} {high_24}"
        );
    }
}

/// Each row of the table prints the zoom read from the Jacobian of its patch
/// as its tile is re-anchored: the same Jacobian `get_bench_track` reports for
/// the row.
#[test]
fn each_row_prints_the_jacobian_and_zoom_of_its_own_patch() {
    let (state, id, label, panel, _ctx) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let recon = state.node(id).expect("loaded").recon();
    assert_eq!(panel.rows().len(), 3);
    for row in panel.rows() {
        assert!(row.tile, "row {} drew no tile", row.observation);
        let jacobian = super::patch_jacobian(recon, &track, row.observation);
        assert!(
            jacobian.is_some(),
            "row {} has no Jacobian",
            row.observation
        );
        assert_eq!(row.jacobian, jacobian, "row {}", row.observation);
        assert_eq!(row.zoom_text, super::table::zoom_text(jacobian));
        assert!(row.zoom_text.ends_with('\u{d7}'), "{:?}", row.zoom_text);
    }
}

/// The *Zoom* cell is geometry and needs no photograph: with none decoded, no
/// row has a tile, and every row still prints the zoom it prints once its
/// photograph is there.
#[test]
fn a_row_prints_its_zoom_before_its_photograph_is_decoded() {
    let (with_photographs, _id, _label, decoded, _ctx) = on_the_bench();

    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    drop(with_photographs);

    assert_eq!(panel.rows().len(), decoded.rows().len());
    for (row, with) in panel.rows().iter().zip(decoded.rows()) {
        assert!(
            !row.tile,
            "row {} drew a tile with no photograph",
            row.observation
        );
        assert!(
            row.jacobian.is_some(),
            "row {} has no Jacobian",
            row.observation
        );
        assert_eq!(row.jacobian, with.jacobian, "row {}", row.observation);
        assert_eq!(row.zoom_text, with.zoom_text, "row {}", row.observation);
    }
}

/// A click on *Zoom* orders the rows by their mean zoom, least first, since no
/// bar judges it; a second click puts the most first.
#[test]
fn a_click_on_zoom_orders_the_rows_by_their_mean_zoom() {
    let (state, _id, _label, mut panel, ctx) = measured_on_the_bench();
    let means = |panel: &TrackBody| -> Vec<f64> {
        panel
            .rows()
            .iter()
            .map(|row| row.jacobian.and_then(|j| j.mean_zoom()).expect("a zoom"))
            .collect()
    };
    let as_drawn = means(&panel);
    assert!(
        as_drawn.windows(2).any(|w| w[0] != w[1]),
        "the fixture's rows all zoom alike: {as_drawn:?}"
    );

    click_heading(&mut panel, &ctx, &state, "Zoom");
    assert_eq!(panel.sort.column, super::table::SortColumn::Zoom);
    assert!(!panel.sort.descending);
    let increasing = means(&panel);
    assert!(
        increasing.windows(2).all(|w| w[0] <= w[1]),
        "{increasing:?}"
    );

    click_heading(&mut panel, &ctx, &state, "Zoom");
    let decreasing = means(&panel);
    assert!(
        decreasing.windows(2).all(|w| w[0] >= w[1]),
        "{decreasing:?}"
    );
}

/// A row with no zoom sorts after every row with one, whichever way *Zoom*
/// orders the rows. The fixture's rows all have one, so one row's cached
/// Jacobian is set to none, as for a patch behind the camera; the cache keeps
/// it until the track moves, which nothing here does.
#[test]
fn a_row_with_no_zoom_sorts_last_both_ways() {
    let (state, _id, _label, mut panel, ctx) = measured_on_the_bench();
    let missing = 1;
    panel.jacobians.insert(missing, None);
    let order = |panel: &TrackBody| -> Vec<(usize, Option<f64>)> {
        panel
            .rows()
            .iter()
            .map(|row| (row.observation, row.jacobian.and_then(|j| j.mean_zoom())))
            .collect()
    };

    click_heading(&mut panel, &ctx, &state, "Zoom");
    assert!(!panel.sort.descending);
    let increasing = order(&panel);
    assert_eq!(increasing.last(), Some(&(missing, None)), "{increasing:?}");
    assert_eq!(panel.rows().last().expect("a row").zoom_text, "-");
    assert!(
        increasing[..2].windows(2).all(|w| w[0].1 <= w[1].1),
        "{increasing:?}"
    );

    click_heading(&mut panel, &ctx, &state, "Zoom");
    assert!(panel.sort.descending);
    let decreasing = order(&panel);
    assert_eq!(decreasing.last(), Some(&(missing, None)), "{decreasing:?}");
    assert!(
        decreasing[..2].windows(2).all(|w| w[0].1 >= w[1].1),
        "{decreasing:?}"
    );
}

/// The *Zoom* cell prints each zoom to two significant digits, judged after
/// rounding: a zoom just under 10 that rounds to 10 prints whole, one just
/// under 1 that rounds to 1 prints to one decimal, a small zoom keeps both its
/// digits, and one of 100 or more prints whole.
#[test]
fn the_zoom_cell_picks_its_format_after_rounding() {
    use super::patch::PatchJacobian;
    use super::table::zoom_text;
    // A diagonal Jacobian of `1 / zoom` along each axis.
    let zooms = |a: f64, b: f64| zoom_text(Some(PatchJacobian([[1.0 / a, 0.0], [0.0, 1.0 / b]])));
    assert_eq!(zooms(9.96, 9.96), "10/10\u{d7}");
    assert_eq!(zooms(9.94, 9.94), "9.9/9.9\u{d7}");
    assert_eq!(zooms(0.996, 0.996), "1.0/1.0\u{d7}");
    assert_eq!(zooms(0.994, 0.994), "0.99/0.99\u{d7}");
    assert_eq!(zooms(0.996, 9.96), "1.0/10\u{d7}");
    assert_eq!(zooms(25.0, 0.5), "0.50/25\u{d7}");
    assert_eq!(zooms(0.031, 0.0312), "0.031/0.031\u{d7}");
    assert_eq!(zooms(0.031, 123.4), "0.031/123\u{d7}");
}

// ── Viewed mode ─────────────────────────────────────────────────────────────

mod viewed;

/// An ellipse with semi-axes `[major, minor]`, their lower bounds, and its
/// major axis at `degrees`, with the matrix that goes with them.
fn hover_ellipse(
    axes: [f64; 2],
    axes_is_at_least: [bool; 2],
    degrees: f64,
) -> sfmtool_core::patch::self_similarity::SelfSimilarityEllipse {
    let (s, c) = degrees.to_radians().sin_cos();
    let (a2, b2) = (axes[0] * axes[0], axes[1] * axes[1]);
    let xy = (a2 - b2) * c * s;
    sfmtool_core::patch::self_similarity::SelfSimilarityEllipse {
        axes,
        axes_is_at_least,
        major_angle: degrees.to_radians(),
        matrix: [[a2 * c * c + b2 * s * s, xy], [xy, a2 * s * s + b2 * c * c]],
    }
}

/// The two readings the hover tests lay out: a whole core that locks, and a
/// middle that matched itself at the edge of the search along `x`.
fn hover_ellipses() -> (
    sfmtool_core::patch::self_similarity::SelfSimilarityEllipseUnits,
    sfmtool_core::patch::self_similarity::SelfSimilarityEllipseUnits,
) {
    use sfmtool_core::patch::self_similarity::{PatchEllipse, SelfSimilarityEllipseUnits};
    let (x, at_least) = (false, true);
    let whole = SelfSimilarityEllipseUnits {
        grid_px: hover_ellipse([0.42, 0.31], [x, x], 35.0),
        image_px: Some(hover_ellipse([0.853, 0.6], [x, x], 41.0)),
        patch: Some(PatchEllipse::Length(hover_ellipse(
            [0.0031, 0.0022],
            [x, x],
            145.0,
        ))),
    };
    let middle = SelfSimilarityEllipseUnits {
        grid_px: hover_ellipse([3.0, 0.8], [at_least, x], 0.0),
        image_px: Some(hover_ellipse([5.62, 1.6], [at_least, x], 2.0)),
        patch: Some(PatchEllipse::Length(hover_ellipse(
            [0.0291, 0.00781],
            [at_least, x],
            0.0,
        ))),
    };
    (whole, middle)
}

/// The self-similarity cell's hover lays out the whole and middle readings'
/// ellipses in three units, each as its semi-major × semi-minor axis and the
/// major axis's angle, with a `+` on a lower bound and `3+` for a grid length
/// at the largest radius, as the cell prints it.
#[test]
fn the_self_similarity_hover_shows_the_ellipse_in_three_units() {
    use super::self_similarity_ellipse_text;
    let (whole, middle) = hover_ellipses();
    let text =
        self_similarity_ellipse_text(Some(&whole), Some(&middle), Some("m")).expect("ellipses");
    assert_eq!(
        text,
        [
            "           whole                  mid",
            "grid px    0.42 \u{d7} 0.31 at 35\u{b0}     3+ \u{d7} 0.80 at 0\u{b0}",
            "image px   0.85 \u{d7} 0.60 at 41\u{b0}     5.6+ \u{d7} 1.6 at 2\u{b0}",
            "world      3.1 \u{d7} 2.2 mm at 145\u{b0}   29+ \u{d7} 7.8 mm at 0\u{b0}",
        ]
        .join("\n")
    );

    // With no unit on the file, the numbers are bare and the row says so.
    let bare = self_similarity_ellipse_text(Some(&whole), None, None).expect("an ellipse");
    assert!(
        bare.lines().any(|line| line.starts_with("scene units")
            && line.contains("0.0031 \u{d7} 0.0022 at 145\u{b0}")),
        "{bare}"
    );
    // A circle has no direction, so it prints no angle.
    let circle = sfmtool_core::patch::self_similarity::SelfSimilarityEllipseUnits {
        grid_px: sfmtool_core::patch::self_similarity::SelfSimilarityEllipse {
            major_angle: f64::NAN,
            ..hover_ellipse([3.0, 3.0], [true, true], 0.0)
        },
        image_px: None,
        patch: None,
    };
    let text = self_similarity_ellipse_text(Some(&circle), None, None).expect("an ellipse");
    assert!(
        text.lines()
            .any(|line| line.starts_with("grid px") && line.ends_with("3+ \u{d7} 3+   -")),
        "{text}"
    );
    // Nothing is printed where nothing was measured.
    assert_eq!(self_similarity_ellipse_text(None, None, Some("m")), None);
}

/// The world row of the self-similarity hover for a whole ellipse along the
/// patch of `whole` and a middle one of `middle`, each `[major, minor]`, in a
/// scene whose unit is `unit`, its cells split apart. A `true` beside a value
/// marks it a lower bound.
fn hover_world_row(
    unit: Option<&str>,
    whole: [(f64, bool); 2],
    middle: [(f64, bool); 2],
) -> Vec<String> {
    use sfmtool_core::patch::self_similarity::{PatchEllipse, SelfSimilarityEllipseUnits};
    let on_patch = |values: [(f64, bool); 2], degrees: f64| {
        Some(PatchEllipse::Length(hover_ellipse(
            values.map(|(value, _)| value),
            values.map(|(_, at_least)| at_least),
            degrees,
        )))
    };
    let (w, m) = hover_ellipses();
    let w = SelfSimilarityEllipseUnits {
        patch: on_patch(whole, 145.0),
        ..w
    };
    let m = SelfSimilarityEllipseUnits {
        patch: on_patch(middle, 0.0),
        ..m
    };
    let text = super::self_similarity_ellipse_text(Some(&w), Some(&m), unit).expect("ellipses");
    let row = text.lines().last().expect("a world row");
    row.split("   ")
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(str::to_string)
        .collect()
}

/// The world row prints every length in one unit, chosen from the largest of
/// them: a metric scene's in whichever of µm, mm, cm and m puts it in
/// [1, 1000) (its own unit where that does), a scene in feet in inches under
/// a foot, and bare scene units in scientific form under 0.001. The `+` on a
/// lower bound and the two significant digits survive the conversion.
#[test]
fn the_self_similarity_hover_scales_the_world_lengths_to_one_unit() {
    let (x, at_least) = (false, true);
    let row = |cells: [&str; 3]| cells.map(str::to_string).to_vec();
    // A metres scene prints in mm, the largest value setting the unit.
    assert_eq!(
        hover_world_row(
            Some("m"),
            [(0.004, x), (0.0031, x)],
            [(0.0291, at_least), (0.00781, x)]
        ),
        row([
            "world",
            "4.0 \u{d7} 3.1 mm at 145\u{b0}",
            "29+ \u{d7} 7.8 mm at 0\u{b0}"
        ])
    );
    // Under a millimetre, the lengths print in micrometres.
    assert_eq!(
        hover_world_row(
            Some("m"),
            [(0.00054, x), (0.00012, x)],
            [(0.00031, x), (0.00002, at_least)]
        ),
        row([
            "world",
            "540 \u{d7} 120 \u{b5}m at 145\u{b0}",
            "310 \u{d7} 20+ \u{b5}m at 0\u{b0}"
        ])
    );
    // A centimetre scene whose lengths fit in centimetres stays in them.
    assert_eq!(
        hover_world_row(
            Some("cm"),
            [(0.4, x), (0.31, x)],
            [(2.9, at_least), (0.78, x)]
        ),
        row([
            "world",
            "0.40 \u{d7} 0.31 cm at 145\u{b0}",
            "2.9+ \u{d7} 0.78 cm at 0\u{b0}"
        ])
    );
    // Lengths past a kilometre take the nearer end of the list, m.
    assert_eq!(
        hover_world_row(
            Some("mm"),
            [(4.0e6, x), (3.1e6, x)],
            [(7.8e5, x), (2.9e5, x)]
        ),
        row([
            "world",
            "4000 \u{d7} 3100 m at 145\u{b0}",
            "780 \u{d7} 290 m at 0\u{b0}"
        ])
    );
    // A scene in feet prints in inches while its largest length is under a
    // foot, and a scene in inches stays in inches.
    assert_eq!(
        hover_world_row(
            Some("ft"),
            [(0.5, at_least), (0.25, x)],
            [(0.1, x), (0.05, x)]
        ),
        row([
            "world",
            "6.0+ \u{d7} 3.0 in at 145\u{b0}",
            "1.2 \u{d7} 0.60 in at 0\u{b0}"
        ])
    );
    assert_eq!(
        hover_world_row(Some("ft"), [(1.5, x), (0.5, x)], [(0.1, x), (0.05, x)]),
        row([
            "world",
            "1.5 \u{d7} 0.50 ft at 145\u{b0}",
            "0.10 \u{d7} 0.050 ft at 0\u{b0}"
        ])
    );
    assert_eq!(
        hover_world_row(Some("in"), [(25.0, x), (0.5, x)], [(0.1, x), (0.05, x)]),
        row([
            "world",
            "25 \u{d7} 0.50 in at 145\u{b0}",
            "0.10 \u{d7} 0.050 in at 0\u{b0}"
        ])
    );
    // Bare scene units under 0.001 print in scientific form, every value.
    assert_eq!(
        hover_world_row(
            None,
            [(0.00054, x), (0.000123, at_least)],
            [(0.0009, x), (0.0000071, x)]
        ),
        row([
            "scene units",
            "5.4e-4 \u{d7} 1.2e-4+ at 145\u{b0}",
            "9.0e-4 \u{d7} 7.1e-6 at 0\u{b0}"
        ])
    );
    // From 0.001 up they stay plain.
    assert_eq!(
        hover_world_row(
            None,
            [(0.0031, x), (0.0004, x)],
            [(0.0078, at_least), (0.0012, x)]
        ),
        row([
            "scene units",
            "0.0031 \u{d7} 0.00040 at 145\u{b0}",
            "0.0078+ \u{d7} 0.0012 at 0\u{b0}"
        ])
    );
}

/// A patch at infinity reads its ellipse along the patch as angles: the last
/// row is labelled *angle* and each semi-axis carries a degree sign, a `+`
/// before it on a lower bound. A part with no ellipse prints `-` in every
/// row.
#[test]
fn the_self_similarity_hover_shows_a_bearing_in_degrees() {
    use super::self_similarity_ellipse_text;
    use sfmtool_core::patch::self_similarity::{PatchEllipse, SelfSimilarityEllipseUnits};
    let (whole, _) = hover_ellipses();
    let bearing = SelfSimilarityEllipseUnits {
        patch: Some(PatchEllipse::Angle(hover_ellipse(
            [0.5, 0.12],
            [true, false],
            145.0,
        ))),
        ..whole
    };
    let text = self_similarity_ellipse_text(Some(&bearing), None, Some("m")).expect("an ellipse");
    assert_eq!(
        text,
        [
            "           whole                    mid",
            "grid px    0.42 \u{d7} 0.31 at 35\u{b0}       -",
            "image px   0.85 \u{d7} 0.60 at 41\u{b0}       -",
            "angle      0.50+\u{b0} \u{d7} 0.12\u{b0} at 145\u{b0}   -",
        ]
        .join("\n")
    );
}

/// At the cluster stage there is no patch, so the last row prints `-` for
/// both parts, while the grid and image rows carry their numbers.
#[test]
fn the_self_similarity_hover_prints_a_dash_where_there_is_no_patch() {
    use super::self_similarity_ellipse_text;
    use sfmtool_core::patch::self_similarity::SelfSimilarityEllipseUnits;
    let (whole, middle) = hover_ellipses();
    let (whole, middle) = (
        SelfSimilarityEllipseUnits {
            patch: None,
            ..whole
        },
        SelfSimilarityEllipseUnits {
            patch: None,
            ..middle
        },
    );
    let text =
        self_similarity_ellipse_text(Some(&whole), Some(&middle), Some("m")).expect("ellipses");
    assert_eq!(
        text,
        [
            "           whole                mid",
            "grid px    0.42 \u{d7} 0.31 at 35\u{b0}   3+ \u{d7} 0.80 at 0\u{b0}",
            "image px   0.85 \u{d7} 0.60 at 41\u{b0}   5.6+ \u{d7} 1.6 at 2\u{b0}",
            "world      -                    -",
        ]
        .join("\n")
    );
}

/// A row evaluated at the track stage hovers the ellipses its measurement
/// carries, with every unit filled in, and the same at the cluster stage
/// without the patch.
#[test]
fn an_evaluated_row_hovers_the_ellipses_its_measurement_carries_at_both_stages() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    let track = state
        .bench_track(id, &label)
        .expect("the item on the bench");
    let rows = panel.rows();
    let row = &rows[0];
    let m = track.observations[row.observation]
        .track
        .as_ref()
        .expect("a track slot");
    let (whole, middle) = (
        m.zncc_self_similarity_ellipse.expect("an ellipse"),
        m.zncc_self_similarity_ellipse_middle.expect("an ellipse"),
    );
    assert!(whole.image_px.is_some() && whole.patch.is_some());
    let unit = crate::scene::node_by_id(&state.scene, id)
        .expect("the node")
        .recon()
        .metadata
        .world_space_unit
        .clone();
    assert_eq!(row.self_similarity_ellipse, [Some(whole), Some(middle)]);
    let hover = super::self_similarity_ellipse_text(Some(&whole), Some(&middle), unit.as_deref())
        .expect("the cell hovers its ellipses");
    let image = hover
        .lines()
        .find(|line| line.starts_with("image px"))
        .expect("an image row");
    assert!(!image.contains(" -"), "{hover}");
    assert!(
        hover.lines().any(
            |line| (line.starts_with("world") || line.starts_with("scene units"))
                && line.contains(" \u{d7} ")
        ),
        "{hover}"
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    let [whole, middle] = panel.rows()[0].self_similarity_ellipse;
    let hover =
        super::self_similarity_ellipse_text(whole.as_ref(), middle.as_ref(), unit.as_deref())
            .expect("a cluster row hovers its ellipses too");
    assert!(
        hover
            .lines()
            .any(|line| line.starts_with("image px") && !line.contains(" -")),
        "{hover}"
    );
    assert!(
        hover.lines().any(|line| line.contains("    -")
            && (line.starts_with("world") || line.starts_with("scene units"))),
        "{hover}"
    );
}

/// The row that holds the reference where unpinning it would move the bitmap
/// is measured, and the bars still propose nothing for it: its hover says why
/// rather than that nothing has measured it.
#[test]
fn the_held_reference_s_verdict_hover_says_the_bars_wait_for_the_render() {
    let waits = super::table::proposal_reason(None, 3, true);
    assert!(waits.contains("cannot say"), "{waits}");
    assert!(!waits.contains("Nothing at this stage"), "{waits}");
    let unmeasured = super::table::proposal_reason(None, 3, false);
    assert!(unmeasured.contains("Nothing at this stage has measured it"));
}
