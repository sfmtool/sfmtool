// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{progress_info, progress_note, progress_status, progress_warn};
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// The test sink
// ---------------------------------------------------------------------------

/// An owned mirror of [`Event`], because an `Event<'a>` borrows its text only
/// for the duration of the sink call.
#[derive(Debug, Clone, PartialEq)]
enum Owned {
    Enter(&'static str, u8),
    Leave(&'static str, u8, Option<String>),
    Message(Level, u8, String),
    Count(u64, Option<u64>, &'static str),
    Fraction(f32),
}

/// Somewhere for events to land, shaped like the collector the viewer will
/// have: appended events, and the latest status, which replaces.
#[derive(Default)]
struct Collector {
    events: Mutex<Vec<Owned>>,
    status: Mutex<Option<String>>,
}

impl Collector {
    fn new() -> Self {
        Self::default()
    }

    fn push(&self, event: Event<'_>) {
        let owned = match event {
            Event::Enter { phase, depth } => Owned::Enter(phase, depth),
            Event::Leave {
                phase, depth, note, ..
            } => Owned::Leave(phase, depth, note.map(str::to_string)),
            Event::Message { level, depth, text } => Owned::Message(level, depth, text.to_string()),
            Event::Count { done, total, unit } => Owned::Count(done, total, unit),
            Event::Fraction { of_whole } => Owned::Fraction(of_whole),
            Event::Status { text } => {
                *self.status.lock().unwrap() = Some(text.to_string());
                return;
            }
        };
        self.events.lock().unwrap().push(owned);
    }

    fn events(&self) -> Vec<Owned> {
        self.events.lock().unwrap().clone()
    }

    fn status(&self) -> Option<String> {
        self.status.lock().unwrap().clone()
    }

    fn fractions(&self) -> Vec<f32> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Fraction(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    /// `(name, depth)` of every phase boundary, with `Enter` and `Leave` told
    /// apart by the leading character.
    fn boundaries(&self) -> Vec<(String, u8)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Enter(name, depth) => Some((format!(">{name}"), depth)),
                Owned::Leave(name, depth, _) => Some((format!("<{name}"), depth)),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The type itself
// ---------------------------------------------------------------------------

#[test]
fn progress_is_send_sync_and_copy() {
    fn assert<T: Send + Sync + Copy>() {}
    assert::<Progress<'static>>();
}

#[test]
fn none_reports_nothing_and_builds_nothing() {
    let calls = AtomicUsize::new(0);
    let count = || {
        calls.fetch_add(1, Ordering::Relaxed);
        7
    };

    let progress = Progress::none();
    assert!(!progress.wants(Level::Info));
    assert!(!progress.wants(Level::Warn));
    assert!(!progress.is_detailed());
    assert!(!progress.is_cancelled());
    assert_eq!(progress.check_cancel(), Ok(()));

    progress_info!(progress, "{} points", count());
    progress_warn!(progress, "{} points", count());
    progress_status!(progress, "{} points", count());
    progress.count(1, Some(2), "point");
    progress.set_fraction(0.5);

    {
        let mut phase = progress.phase("solve");
        assert!(!phase.is_recording(), "no sink means nothing to record");
        progress_note!(phase, "{} points", count());
        let mut inner = phase.phase("linearise");
        progress_note!(inner, "{} points", count());
    }

    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

// ---------------------------------------------------------------------------
// Phases and depth
// ---------------------------------------------------------------------------

#[test]
fn nested_phases_enter_and_leave_in_order_at_their_own_depths() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    {
        let solve = progress.phase("solve");
        progress_info!(solve, "85 images");
        {
            let linearise = solve.phase("linearise");
            progress_info!(linearise, "12 blocks");
        }
        let _row_map = solve.phase("row map");
    }
    progress_info!(progress, "done");

    assert_eq!(
        collector.events(),
        vec![
            Owned::Enter("solve", 0),
            Owned::Message(Level::Info, 1, "85 images".to_string()),
            Owned::Enter("linearise", 1),
            Owned::Message(Level::Info, 2, "12 blocks".to_string()),
            Owned::Leave("linearise", 1, None),
            Owned::Enter("row map", 1),
            Owned::Leave("row map", 1, None),
            Owned::Leave("solve", 0, None),
            Owned::Message(Level::Info, 0, "done".to_string()),
        ]
    );
}

#[test]
fn depth_comes_from_the_value_not_from_the_sink() {
    // Two threads, each opening a phase inside one of its own, reporting into
    // one collector. A sink counting `Enter` against `Leave` would see the two
    // interleave and get every depth wrong; the depth is a field, so it does
    // not.
    const ROUNDS: usize = 200;

    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    std::thread::scope(|scope| {
        for (outer_name, inner_name) in [("outer a", "inner a"), ("outer b", "inner b")] {
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    let outer = progress.phase(outer_name);
                    let inner = outer.phase(inner_name);
                    progress_info!(inner, "{}", inner_name);
                }
            });
        }
    });

    let mut outers = 0;
    let mut inners = 0;
    let mut messages = 0;
    for event in collector.events() {
        match event {
            Owned::Enter(name, depth) | Owned::Leave(name, depth, _) => {
                if name.starts_with("outer") {
                    assert_eq!(depth, 0, "{name} is a top-level phase");
                    outers += 1;
                } else {
                    assert_eq!(depth, 1, "{name} is nested one deep");
                    inners += 1;
                }
            }
            Owned::Message(_, depth, _) => {
                assert_eq!(depth, 2, "a message sits inside the phase it was in");
                messages += 1;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(outers, 2 * 2 * ROUNDS);
    assert_eq!(inners, 2 * 2 * ROUNDS);
    assert_eq!(messages, 2 * ROUNDS);
}

#[test]
fn a_note_reaches_its_own_leave_and_nowhere_else() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    {
        let mut atlas = progress.phase("patch atlas");
        {
            let _thumbnails = atlas.phase("thumbnails");
        }
        progress_note!(atlas, "{} tiles", 46_231);
    }

    assert_eq!(
        collector.events(),
        vec![
            Owned::Enter("patch atlas", 0),
            Owned::Enter("thumbnails", 1),
            Owned::Leave("thumbnails", 1, None),
            Owned::Leave("patch atlas", 0, Some("46231 tiles".to_string())),
        ]
    );
}

#[test]
fn a_cancelled_phase_emits_no_leave() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    {
        let kept = progress.phase("kept");
        let abandoned = kept.phase("abandoned");
        abandoned.cancel();
    }

    assert_eq!(
        collector.events(),
        vec![
            Owned::Enter("kept", 0),
            Owned::Enter("abandoned", 1),
            Owned::Leave("kept", 0, None),
        ]
    );
}

#[test]
fn detail_phase_is_inert_when_detail_is_off() {
    let calls = AtomicUsize::new(0);
    let tiles = || {
        calls.fetch_add(1, Ordering::Relaxed);
        46_231
    };

    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);
    assert!(!progress.is_detailed());

    {
        let mut packing = progress.detail_phase("packing");
        assert!(!packing.is_recording());
        progress_note!(packing, "{} tiles", tiles());
        // The inner `Progress` stays usable, and stays at this depth, because
        // no phase was opened above it.
        progress_info!(packing, "still reported");
    }

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        collector.events(),
        vec![Owned::Message(Level::Info, 0, "still reported".to_string())]
    );
}

#[test]
fn detail_phase_records_when_detail_is_on() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink).detailed(true);
    assert!(progress.is_detailed());

    {
        let mut packing = progress.detail_phase("packing");
        assert!(packing.is_recording());
        progress_note!(packing, "{} tiles", 46_231);
        progress_info!(packing, "inside");
    }

    assert_eq!(
        collector.events(),
        vec![
            Owned::Enter("packing", 0),
            Owned::Message(Level::Info, 1, "inside".to_string()),
            Owned::Leave("packing", 0, Some("46231 tiles".to_string())),
        ]
    );
}

#[test]
fn two_detail_settings_run_side_by_side() {
    let loud = Collector::new();
    let quiet = Collector::new();
    let loud_sink = |event: Event<'_>| loud.push(event);
    let quiet_sink = |event: Event<'_>| quiet.push(event);
    let detailed = Progress::to(&loud_sink).detailed(true);
    let plain = Progress::to(&quiet_sink);

    std::thread::scope(|scope| {
        scope.spawn(move || {
            for _ in 0..200 {
                let _stage = detailed.detail_phase("stage");
            }
        });
        scope.spawn(move || {
            for _ in 0..200 {
                let _stage = plain.detail_phase("stage");
            }
        });
    });

    assert_eq!(loud.events().len(), 400);
    assert!(quiet.events().is_empty());
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[test]
fn macros_do_not_evaluate_their_arguments_when_nothing_is_listening() {
    // The one property a plain method cannot have: `format_args!` defers the
    // formatting but still evaluates what it interpolates.
    let calls = AtomicUsize::new(0);
    let points = || {
        calls.fetch_add(1, Ordering::Relaxed);
        44_912
    };

    let silent = Progress::none();
    progress_info!(silent, "{} points", points());
    progress_warn!(silent, "{} points", points());
    progress_status!(silent, "{} points", points());
    {
        let mut phase = silent.phase("solve");
        progress_note!(phase, "{} points", points());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let listening = Progress::to(&sink);
    progress_info!(listening, "{} points", points());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        collector.events(),
        vec![Owned::Message(Level::Info, 0, "44912 points".to_string())]
    );
}

#[test]
fn a_note_is_not_built_for_a_phase_that_is_not_recording() {
    let calls = AtomicUsize::new(0);
    let tiles = || {
        calls.fetch_add(1, Ordering::Relaxed);
        17
    };

    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    {
        let mut detail = progress.detail_phase("detail");
        progress_note!(detail, "{} tiles", tiles());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    {
        let mut overview = progress.phase("overview");
        progress_note!(overview, "{} tiles", tiles());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        collector.events(),
        vec![
            Owned::Enter("overview", 0),
            Owned::Leave("overview", 0, Some("17 tiles".to_string())),
        ]
    );
}

#[test]
fn a_status_replaces_while_a_message_appends() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    for name in ["dino_41.jpg", "dino_42.jpg", "dino_43.jpg"] {
        progress_status!(progress, "{name}");
    }
    assert_eq!(collector.status().as_deref(), Some("dino_43.jpg"));
    assert!(collector.events().is_empty());

    for i in 0..3 {
        progress_info!(progress, "line {i}");
    }
    assert_eq!(
        collector.events(),
        vec![
            Owned::Message(Level::Info, 0, "line 0".to_string()),
            Owned::Message(Level::Info, 0, "line 1".to_string()),
            Owned::Message(Level::Info, 0, "line 2".to_string()),
        ]
    );
    assert_eq!(collector.status().as_deref(), Some("dino_43.jpg"));
}

#[test]
fn a_warning_carries_its_level() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    progress_warn!(progress, "{} views below the ZNCC floor", 3);

    assert_eq!(
        collector.events(),
        vec![Owned::Message(
            Level::Warn,
            0,
            "3 views below the ZNCC floor".to_string()
        )]
    );
}

// ---------------------------------------------------------------------------
// Nesting arithmetic
// ---------------------------------------------------------------------------

#[test]
fn a_child_maps_its_own_range_onto_its_share() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    let [materialise, solve, row_map] = progress.split([0.05, 0.90, 0.05]);
    materialise.set_fraction(1.0);
    solve.set_fraction(0.0);
    solve.set_fraction(0.5);
    solve.set_fraction(1.0);
    row_map.set_fraction(1.0);

    assert_eq!(collector.fractions(), vec![0.05, 0.05, 0.5, 0.95, 1.0]);
}

#[test]
fn relative_weights_normalise_to_the_same_split() {
    let tenths = Progress::none().split([1.0, 9.0]);
    let fractions = Progress::none().split([0.1, 0.9]);
    for (a, b) in tenths.iter().zip(fractions.iter()) {
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.extent, b.extent);
    }
    assert_eq!(tenths[0].offset, 0.0);
    assert_eq!(tenths[1].offset, 0.1);
    assert_eq!(tenths[1].extent, 0.9);
}

#[test]
fn children_tile_their_parent_without_overlap_or_gap() {
    let parent = Progress::none().split([2.0, 1.0, 1.0])[1];
    let children = parent.split([1.0, 3.0]);

    assert_eq!(children[0].offset, parent.offset);
    assert_eq!(
        children[0].offset + children[0].extent,
        children[1].offset,
        "the second child starts where the first ends"
    );
    let end = children[1].offset + children[1].extent;
    assert!(
        (end - (parent.offset + parent.extent)).abs() < 1e-6,
        "the last child ends where the parent does"
    );
}

#[test]
fn split_evenly_gives_equal_shares() {
    let shares: Vec<_> = Progress::none().split_evenly(4).collect();
    assert_eq!(shares.len(), 4);
    for (i, share) in shares.iter().enumerate() {
        assert_eq!(share.offset, i as f32 * 0.25);
        assert_eq!(share.extent, 0.25);
    }
    assert_eq!(Progress::none().split_evenly(0).count(), 0);
}

#[test]
fn a_degenerate_weight_splits_evenly_rather_than_panicking() {
    for weights in [[0.0, 0.0], [-1.0, -2.0], [f32::NAN, f32::NAN]] {
        let children = Progress::none().split(weights);
        assert_eq!(children[0].offset, 0.0);
        assert_eq!(children[0].extent, 0.5);
        assert_eq!(children[1].offset, 0.5);
        assert_eq!(children[1].extent, 0.5);
    }

    // A single bad weight among good ones is worth nothing, and the rest still
    // tile the range.
    let children = Progress::none().split([-1.0, 1.0]);
    assert_eq!(children[0].extent, 0.0);
    assert_eq!(children[1].offset, 0.0);
    assert_eq!(children[1].extent, 1.0);
}

#[test]
fn three_levels_of_splitting_compose() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    let second_half = progress.split([1.0, 1.0])[1]; // 0.5 .. 1.0
    let last_quarter = second_half.split([1.0, 1.0])[1]; // 0.75 .. 1.0
    let deepest = last_quarter.split([1.0, 1.0, 1.0, 1.0])[2]; // 0.875 .. 0.9375

    deepest.set_fraction(0.0);
    deepest.set_fraction(1.0);

    assert_eq!(collector.fractions(), vec![0.875, 0.9375]);
}

#[test]
fn a_child_cannot_report_into_its_siblings_range() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    let [first, second] = progress.split([1.0, 1.0]);
    first.set_fraction(5.0);
    first.set_fraction(-3.0);
    first.set_fraction(f32::NAN);
    second.set_fraction(9.0);

    assert_eq!(collector.fractions(), vec![0.5, 0.0, 0.0, 1.0]);
    for f in collector.fractions() {
        assert!((0.0..=1.0).contains(&f));
    }
}

#[test]
fn a_phase_keeps_the_range_and_a_split_keeps_the_depth() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    let [_first, second] = progress.split([1.0, 1.0]);
    {
        let inside = second.phase("inside");
        inside.set_fraction(0.5);
        let [_a, b] = inside.split([1.0, 1.0]);
        let _nested = b.phase("nested");
    }

    assert_eq!(
        collector.boundaries(),
        vec![
            (">inside".to_string(), 0),
            (">nested".to_string(), 1),
            ("<nested".to_string(), 1),
            ("<inside".to_string(), 0),
        ]
    );
    assert_eq!(collector.fractions(), vec![0.75]);
}

// ---------------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------------

#[test]
fn a_known_total_moves_the_bar_and_an_unknown_one_does_not() {
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink);

    let [_first, second] = progress.split([1.0, 1.0]);
    second.count(3, Some(4), "image");
    second.count(9, None, "image");
    second.count(1, Some(0), "image");

    assert_eq!(
        collector.events(),
        vec![
            Owned::Count(3, Some(4), "image"),
            Owned::Fraction(0.875),
            Owned::Count(9, None, "image"),
            Owned::Count(1, Some(0), "image"),
        ]
    );
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

#[test]
fn check_cancel_is_err_exactly_when_is_cancelled() {
    let flag = AtomicBool::new(false);
    let progress = Progress::none().cancelled_by(&flag);

    assert!(!progress.is_cancelled());
    assert_eq!(progress.check_cancel(), Ok(()));

    flag.store(true, Ordering::Relaxed);
    assert!(progress.is_cancelled());
    assert_eq!(progress.check_cancel(), Err(Cancelled));

    // A phase's inner `Progress` carries the flag too.
    let phase = progress.phase("solve");
    assert!(phase.is_cancelled());

    assert_eq!(Cancelled.to_string(), "the operation was cancelled");
}

#[test]
fn cancellation_propagates_three_calls_deep_through_the_question_mark() {
    fn innermost(progress: &Progress<'_>) -> Result<u32, Cancelled> {
        progress.check_cancel()?;
        Ok(3)
    }
    fn middle(progress: &Progress<'_>) -> Result<u32, Cancelled> {
        Ok(innermost(progress)? + 1)
    }
    fn outermost(progress: &Progress<'_>) -> Result<u32, Cancelled> {
        Ok(middle(progress)? + 1)
    }

    let flag = AtomicBool::new(false);
    let progress = Progress::none().cancelled_by(&flag);
    assert_eq!(outermost(&progress), Ok(5));

    flag.store(true, Ordering::Relaxed);
    assert_eq!(outermost(&progress), Err(Cancelled));
}

#[test]
fn a_rayon_loop_stops_and_hands_back_what_it_had() {
    use rayon::prelude::*;

    const ITEMS: u64 = 10_000;
    const BEFORE_CANCELLING: usize = 100;

    let flag = AtomicBool::new(false);
    let collector = Collector::new();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink).cancelled_by(&flag);

    let seen = AtomicUsize::new(0);
    let done = Mutex::new(Vec::new());

    let result: Result<(), Cancelled> = (0..ITEMS).into_par_iter().try_for_each(|item| {
        // The parallel case the whole design exists for: this closure runs on
        // rayon's workers and on the calling thread, and every one of them
        // reads the same flag and writes to the same sink.
        progress.check_cancel()?;
        if seen.fetch_add(1, Ordering::Relaxed) >= BEFORE_CANCELLING {
            flag.store(true, Ordering::Relaxed);
        }
        progress.count(item, Some(ITEMS), "item");
        done.lock().unwrap().push(item);
        Ok(())
    });

    assert_eq!(result, Err(Cancelled));
    let done = done.into_inner().unwrap();
    assert!(
        done.len() >= BEFORE_CANCELLING,
        "the work before the cancellation is kept: {} items",
        done.len()
    );
    assert!(
        (done.len() as u64) < ITEMS,
        "the loop stopped early: {} of {ITEMS} items",
        done.len()
    );

    // Every worker reported into the one sink, and nothing it reported left
    // the range the loop was given.
    let fractions = collector.fractions();
    assert_eq!(fractions.len(), done.len());
    assert!(fractions.iter().all(|f| (0.0..=1.0).contains(f)));
}
