// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the collector: what it keeps, what it only holds the
//! newest of, and how repeated phases fold.
//!
//! Everything here is driven through a real [`Progress`] rather than by
//! building [`Detail`] values by hand, because what is being asserted is that
//! the flat `Enter` / `Leave` stream folds correctly, and a hand-built row
//! would assert nothing about the stream.

use std::sync::Mutex;
use std::time::Duration;

use sfmtool_core::progress::Level;
use sfmtool_core::{progress_info, progress_status, progress_warn};

use super::{Collector, Detail};

/// A collector with detailed phases off, which is what an operation gets until
/// somebody asks for more.
fn collector() -> Collector {
    Collector::new(false)
}

/// Every row as `(name, depth, runs)`, for the rows that are phases.
fn phases(detail: &[Detail]) -> Vec<(&'static str, u8, u32)> {
    detail
        .iter()
        .filter_map(|row| match row {
            Detail::Phase {
                name, depth, runs, ..
            } => Some((*name, *depth, *runs)),
            Detail::Message { .. } => None,
        })
        .collect()
}

/// Every row as a short string: a phase by name, a message by its text.
fn rows(detail: &[Detail]) -> Vec<String> {
    detail
        .iter()
        .map(|row| match row {
            Detail::Phase { name, .. } => (*name).to_string(),
            Detail::Message { text, .. } => text.clone(),
        })
        .collect()
}

/// What one named row cost.
fn took(detail: &[Detail], name: &str) -> Duration {
    detail
        .iter()
        .find_map(|row| match row {
            Detail::Phase {
                name: found, took, ..
            } if *found == name => Some(*took),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no phase named {name} in {:?}", rows(detail)))
}

// -- What lands, and what does not --------------------------------------

#[test]
fn a_phase_and_a_message_land_in_the_order_they_happened() {
    let collector = collector();
    {
        let progress = collector.progress();
        let outer = progress.phase("solve");
        progress_info!(outer, "8 images");
        let _inner = outer.phase("linearise");
    }

    let detail = collector.take();
    assert_eq!(rows(&detail), ["solve", "8 images", "linearise"]);
    assert_eq!(phases(&detail), [("solve", 0, 1), ("linearise", 1, 1)]);
    let Detail::Message { level, depth, .. } = &detail[1] else {
        panic!("the middle row is not a message: {:?}", rows(&detail));
    };
    assert_eq!(*level, Level::Info);
    assert_eq!(*depth, 1, "the message did not carry its phase's depth");
}

#[test]
fn taking_the_detail_leaves_the_collector_empty() {
    let collector = collector();
    drop(collector.phase("materialise"));
    assert_eq!(rows(&collector.take()), ["materialise"]);
    assert!(
        collector.take().is_empty(),
        "the rows were handed out twice"
    );

    // And it is usable again afterwards, with no memory of the fold it did.
    drop(collector.phase("materialise"));
    assert_eq!(phases(&collector.take()), [("materialise", 0, 1)]);
}

/// A status is live state: it says what the operation is doing *now*, so by the
/// time an entry exists the answer is "finished" and there is nothing to keep.
#[test]
fn a_status_replaces_and_is_never_part_of_the_detail() {
    let collector = collector();
    let progress = collector.progress();
    for name in ["dino_42.jpg", "dino_43.jpg"] {
        progress_status!(progress, "{name}");
    }

    assert_eq!(collector.status().as_deref(), Some("dino_43.jpg"));
    assert!(collector.take().is_empty(), "a status reached the entry");
}

#[test]
fn a_count_and_a_fraction_are_the_newest_only_and_never_part_of_the_detail() {
    let collector = collector();
    let progress = collector.progress();
    progress.count(3, Some(20), "iteration");
    progress.count(7, Some(20), "iteration");

    let count = collector.count().expect("a count was reported");
    assert_eq!(
        (count.done, count.total, count.unit),
        (7, Some(20), "iteration")
    );
    assert!(
        collector.fraction().is_some(),
        "a known total moves the bar"
    );
    assert!(collector.take().is_empty(), "a count reached the entry");
}

/// The kernels guarantee only that a call stays inside the range it was given.
/// That the bar never runs backwards is the collector's, and this is it.
#[test]
fn the_fraction_never_runs_backwards() {
    let collector = collector();
    let progress = collector.progress();
    progress.set_fraction(0.8);
    progress.set_fraction(0.3);
    assert_eq!(collector.fraction(), Some(0.8));
}

// -- Folding ------------------------------------------------------------

#[test]
fn children_sharing_a_name_under_one_parent_fold_with_the_count_and_the_sum() {
    let collector = collector();
    {
        let solve = collector.phase("solve");
        for _ in 0..3 {
            let round = solve.phase("round");
            drop(round.phase("linearise"));
            drop(round.phase("normal equations"));
        }
    }

    let detail = collector.take();
    assert_eq!(
        phases(&detail),
        [
            ("solve", 0, 1),
            ("round", 1, 3),
            ("linearise", 2, 3),
            ("normal equations", 2, 3),
        ]
    );
    assert!(
        took(&detail, "round") >= took(&detail, "linearise"),
        "a folded row's time is not the sum of its runs"
    );
}

#[test]
fn a_phase_that_ran_once_carries_no_count() {
    let collector = collector();
    drop(collector.phase("row map"));
    let detail = collector.take();
    let [(_, _, runs)] = phases(&detail)[..] else {
        panic!("expected one row, got {:?}", rows(&detail));
    };
    assert_eq!(
        runs, 1,
        "a phase that ran once was counted as a run of many"
    );
}

/// The name is a key only within one enclosing phase, so two parents that each
/// run a `refine` are two rows and not one.
#[test]
fn the_same_name_under_two_parents_stays_two_rows() {
    let collector = collector();
    {
        let localize = collector.phase("localize");
        drop(localize.phase("refine"));
    }
    {
        let resect = collector.phase("resect");
        drop(resect.phase("refine"));
    }

    let detail = collector.take();
    assert_eq!(
        phases(&detail),
        [
            ("localize", 0, 1),
            ("refine", 1, 1),
            ("resect", 0, 1),
            ("refine", 1, 1),
        ]
    );
}

#[test]
fn folded_rows_are_ordered_by_first_appearance() {
    let collector = collector();
    let progress = collector.progress();
    for _ in 0..2 {
        drop(progress.phase("beta"));
        drop(progress.phase("alpha"));
    }

    assert_eq!(
        phases(&collector.take()),
        [("beta", 0, 2), ("alpha", 0, 2)],
        "the second name overtook the first"
    );
}

#[test]
fn a_message_between_two_foldable_phases_keeps_its_place() {
    let collector = collector();
    let progress = collector.progress();
    drop(progress.phase("alpha"));
    progress_warn!(progress, "3 views below the floor");
    drop(progress.phase("beta"));
    drop(progress.phase("alpha"));
    drop(progress.phase("beta"));

    assert_eq!(
        rows(&collector.take()),
        ["alpha", "3 views below the floor", "beta"]
    );
}

/// A note survives a trip through the loop that says nothing, so a `reused` is
/// not erased by the next run.
#[test]
fn a_silent_run_leaves_an_earlier_note_standing() {
    let collector = collector();
    let progress = collector.progress();
    {
        let mut first = progress.phase("patch atlas");
        first.note(format_args!("reused"));
    }
    drop(progress.phase("patch atlas"));

    let detail = collector.take();
    let [Detail::Phase {
        note,
        note_last,
        runs,
        ..
    }] = &detail[..]
    else {
        panic!("expected one row, got {:?}", rows(&detail));
    };
    assert_eq!(note.as_deref(), Some("reused"));
    assert_eq!(*note_last, None, "silence was read as a disagreement");
    assert_eq!(*runs, 2);
}

/// One run's words are not true of a row that folded three of them, so the row
/// keeps both ends instead of asserting whichever ran last.
#[test]
fn runs_that_said_different_things_leave_both_ends_on_the_row() {
    let collector = collector();
    let progress = collector.progress();
    for trim in ["trim 50 px", "trim 20 px", "trim 4 px"] {
        let mut round = progress.phase("round");
        round.note(format_args!("{trim}"));
    }

    let detail = collector.take();
    let [Detail::Phase {
        note,
        note_last,
        runs,
        ..
    }] = &detail[..]
    else {
        panic!("expected one row, got {:?}", rows(&detail));
    };
    assert_eq!(
        note.as_deref(),
        Some("trim 50 px"),
        "the first end was lost"
    );
    assert_eq!(
        note_last.as_deref(),
        Some("trim 4 px"),
        "the last end was lost"
    );
    assert_eq!(*runs, 3);
}

/// A row whose runs came back to what the first one said carries one note,
/// since both ends agree and `first ... last` would read as a span that is not
/// there.
#[test]
fn runs_that_came_back_to_the_first_note_carry_one() {
    let collector = collector();
    let progress = collector.progress();
    for note in ["reused", "17 images", "reused"] {
        let mut thumbnails = progress.phase("thumbnails");
        thumbnails.note(format_args!("{note}"));
    }

    let detail = collector.take();
    let [Detail::Phase {
        note, note_last, ..
    }] = &detail[..]
    else {
        panic!("expected one row, got {:?}", rows(&detail));
    };
    assert_eq!(note.as_deref(), Some("reused"));
    assert_eq!(*note_last, None);
}

// -- The mirror ---------------------------------------------------------

/// A record the capture logger below kept: level, target, text.
static MIRRORED: Mutex<Vec<(log::Level, String, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        MIRRORED.lock().expect("the capture buffer").push((
            record.level(),
            record.target().to_string(),
            record.args().to_string(),
        ));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture;

/// A `RUST_LOG` capture of a session is the stream of what happened, and this
/// is the piece that makes a kernel's messages useful outside the window.
#[test]
fn messages_are_mirrored_to_log_at_the_matching_level() {
    log::set_logger(&CAPTURE).expect("nothing else installs a logger in these tests");
    log::set_max_level(log::LevelFilter::Trace);

    let collector = collector();
    let progress = collector.progress();
    progress_info!(progress, "mirror-test: 8 images");
    progress_warn!(progress, "mirror-test: 3 views dropped");

    let mirrored: Vec<_> = MIRRORED
        .lock()
        .expect("the capture buffer")
        .iter()
        .filter(|(_, _, text)| text.starts_with("mirror-test: "))
        .cloned()
        .collect();
    assert_eq!(
        mirrored,
        [
            (
                log::Level::Info,
                "sfm_explorer::progress".to_string(),
                "mirror-test: 8 images".to_string(),
            ),
            (
                log::Level::Warn,
                "sfm_explorer::progress".to_string(),
                "mirror-test: 3 views dropped".to_string(),
            ),
        ]
    );
}

/// `Phase::cancel` says the phase did not run, so it leaves no row. The
/// opening `Enter` has already reached the collector by then, which is why
/// this needs saying: the row is claimed on the way in and dropped on the way
/// out. Work the cancelled phase did contain keeps its own row.
#[test]
fn a_cancelled_phase_leaves_no_row_but_its_children_keep_theirs() {
    let collector = collector();
    {
        let abandoned = collector.phase("decode views");
        drop(abandoned.phase("read"));
        abandoned.cancel();
    }
    let detail = collector.take();
    assert_eq!(
        phases(&detail)
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<Vec<_>>(),
        ["read"],
        "{:?}",
        rows(&detail)
    );
}
