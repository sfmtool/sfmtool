// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The rule's cases, ported from the NumPy stage it was lifted out of.

use std::sync::atomic::AtomicBool;

use super::*;

/// The refine radius the fixture's scales are read at, so its numbers are the
/// NumPy stage's own: a row of unit scale `s` states a radius of `8 s` and a
/// footprint of `2.5 s`.
const REFINE_RADIUS: f64 = 8.0;

/// The half-extent a footprint is read at, in unit scales.
const FOOTPRINT: f64 = 2.5;

/// `(owner, x, y, unit scale)`, repeated on every image of the fixture.
///
/// The three pairs are the three answers the rule has to give: a fine row
/// inside the coarse row's drawn footprint, one outside it but well inside the
/// disk the radius itself spans, and a same-scale neighbour sitting on top of
/// its coarse partner.
const FEATURES: [(i64, f64, f64, f64); 6] = [
    (0, 100.0, 100.0, 4.0),
    (1, 103.0, 100.0, 1.0),
    (2, 300.0, 300.0, 4.0),
    (3, 315.0, 300.0, 1.0),
    (4, 500.0, 500.0, 4.0),
    (5, 502.0, 500.0, 3.0),
];

/// The rows of one fixture, owned so the borrows in [`CoveredRows`] have
/// something to point at.
struct Rows {
    image: Vec<i64>,
    owner: Vec<i64>,
    xy: Vec<f64>,
    reach: Vec<f64>,
    radius: Vec<f64>,
}

impl Rows {
    /// `(image, owner, x, y, unit scale)` rows, read at [`REFINE_RADIUS`].
    fn new(rows: &[(i64, i64, f64, f64, f64)], scale: f64) -> Self {
        let mut out = Rows {
            image: Vec::new(),
            owner: Vec::new(),
            xy: Vec::new(),
            reach: Vec::new(),
            radius: Vec::new(),
        };
        for &(image, owner, x, y, unit) in rows {
            out.image.push(image);
            out.owner.push(owner);
            out.xy.push(x);
            out.xy.push(y);
            out.reach.push(FOOTPRINT * unit * scale);
            out.radius.push(REFINE_RADIUS * unit * scale);
        }
        out
    }

    fn view(&self) -> CoveredRows<'_> {
        CoveredRows {
            image_of_row: &self.image,
            owner_of_row: &self.owner,
            xy_px: &self.xy,
            reach_px: &self.reach,
            radius_px: &self.radius,
            protected: None,
        }
    }
}

/// [`FEATURES`] on each of `images`.
fn spread(images: &[i64]) -> Vec<(i64, i64, f64, f64, f64)> {
    let mut out = Vec::new();
    for &image in images {
        for &(owner, x, y, unit) in FEATURES.iter() {
            out.push((image, owner, x, y, unit));
        }
    }
    out
}

/// The fixture on three images, with every owner free to be kept.
fn read(rows: &Rows, owner_count: usize) -> CoveredByFiner {
    covered_by_finer(
        rows.view(),
        owner_count,
        &CoveredOptions {
            min_observations: 1,
            ..CoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("well-formed rows")
}

/// The rows of one owner, as flags.
fn of_owner(rows: &Rows, out: &CoveredByFiner, owner: i64) -> Vec<bool> {
    rows.owner
        .iter()
        .enumerate()
        .filter(|(_, &o)| o == owner)
        .map(|(row, _)| out.flagged[row])
        .collect()
}

// ─── containment ──────────────────────────────────────────────────────────

/// The disk the rule reads is the **reach**, not the radius: owner 0's fine
/// neighbour sits 3 px away, inside `2.5 * 4`; owner 2's sits 15 px away,
/// outside it and well inside `8 * 4`.
#[test]
fn the_footprint_the_rule_reads_is_the_reach_and_not_the_radius() {
    let rows = Rows::new(&spread(&[0, 1, 2]), 1.0);
    let out = read(&rows, 6);
    assert_eq!(of_owner(&rows, &out, 0), [true, true, true]);
    assert_eq!(of_owner(&rows, &out, 2), [false, false, false]);
    // The two lengths really are different on this fixture: owner 2's
    // separation sits between them.
    assert!(rows.reach[4] < 15.0 && 15.0 < rows.radius[4]);
}

/// The fine row is never the one retired.
#[test]
fn the_fine_row_is_never_the_one_retired() {
    let rows = Rows::new(&spread(&[0, 1, 2]), 1.0);
    let out = read(&rows, 6);
    for fine in [1, 3, 5] {
        assert!(!of_owner(&rows, &out, fine).iter().any(|&f| f));
    }
}

/// A same-scale neighbour retires nothing, and it is the scale test that
/// spares it rather than the distance: 2 px is well inside the drawn disk, and
/// the pair is counted as contained.
#[test]
fn a_same_scale_neighbour_retires_nothing() {
    let rows = Rows::new(&spread(&[0]), 1.0);
    let out = read(&rows, 6);
    assert!(!of_owner(&rows, &out, 4).iter().any(|&f| f));
    // Owner 4's row sees owner 5's, and owner 0's sees owner 1's.
    assert_eq!(out.census.pairs_contained, 2);
    assert_eq!(out.census.pairs_finer, 1);
}

/// A row is never its own candidate, so one row alone retires nothing.
#[test]
fn a_row_never_covers_itself() {
    let rows = Rows::new(&[(0, 0, 100.0, 100.0, 4.0)], 1.0);
    let out = read(&rows, 1);
    assert_eq!(out.flagged, [false]);
    assert_eq!(out.census.pairs_contained, 0);
}

/// Two rows of one owner never pair, however close and however different their
/// radii: nothing covers itself, and an owner is one thing.
#[test]
fn two_rows_of_one_owner_never_pair() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (0, 0, 1.0, 0.0, 1.0)], 1.0);
    let out = read(&rows, 1);
    assert!(!out.flagged.iter().any(|&f| f));
    assert_eq!(out.census.pairs_contained, 0);
}

/// Rows of different images never pair, even at identical pixels.
#[test]
fn rows_of_different_images_never_pair() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (1, 1, 0.0, 0.0, 1.0)], 1.0);
    let out = read(&rows, 2);
    assert!(!out.flagged.iter().any(|&f| f));
    assert_eq!(out.census.pairs_contained, 0);
}

// ─── the scale test ───────────────────────────────────────────────────────

/// The ratio bar is read where the band edge is: exactly one octave apart is
/// finer, and a hair under it is not.
#[test]
fn the_ratio_bar_is_read_where_the_band_edge_is() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (0, 1, 1.0, 0.0, 2.0)], 1.0);
    let out = read(&rows, 2);
    assert_eq!(out.flagged, [true, false]);

    let near = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (0, 1, 1.0, 0.0, 2.000_000_1)], 1.0);
    let out = read(&near, 2);
    assert!(!out.flagged.iter().any(|&f| f));
    // It is the scale test alone that spared it: the pair is still contained.
    assert_eq!(out.census.pairs_contained, 1);
    assert_eq!(out.census.pairs_finer, 0);
}

/// A covering row under the fine-radius floor says nothing, and the floor is
/// off at its default.
#[test]
fn a_covering_row_under_the_floor_says_nothing() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (0, 1, 1.0, 0.0, 0.05)], 1.0);
    // The fine row's radius is 0.4 px: finer than an octave, and a collapsed
    // measurement rather than a feature.
    assert!((rows.radius[1] - 0.4).abs() < 1e-12);
    assert_eq!(read(&rows, 2).flagged, [true, false]);

    let out = covered_by_finer(
        rows.view(),
        2,
        &CoveredOptions {
            min_fine_radius_px: 1.0,
            min_observations: 1,
            ..CoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("well-formed rows");
    assert!(!out.flagged.iter().any(|&f| f));
    assert_eq!(out.census.pairs_contained, 1);
    assert_eq!(out.census.pairs_finer, 0);
}

/// A row whose radius is not stated is neither coarser nor finer than
/// anything, so it retires nothing and nothing retires it.
#[test]
fn a_row_of_unstated_radius_pairs_with_nothing() {
    let mut rows = Rows::new(&[(0, 0, 0.0, 0.0, 4.0), (0, 1, 1.0, 0.0, 1.0)], 1.0);
    rows.radius[1] = f64::NAN;
    let out = read(&rows, 2);
    assert!(!out.flagged.iter().any(|&f| f));
    assert_eq!(out.census.pairs_contained, 0);
}

// ─── protection ───────────────────────────────────────────────────────────

/// A protected row is never retired, and still covers.
#[test]
fn a_protected_row_is_spared_and_still_covers() {
    // Three rows in a line: a wide one, a middling one it covers, and a fine
    // one that covers the middling one.
    let rows = Rows::new(
        &[
            (0, 0, 0.0, 0.0, 8.0),
            (0, 1, 1.0, 0.0, 4.0),
            (0, 2, 2.0, 0.0, 1.0),
        ],
        1.0,
    );
    assert_eq!(read(&rows, 3).flagged, [true, true, false]);

    let protected = [false, true, false];
    let out = covered_by_finer(
        CoveredRows {
            protected: Some(&protected),
            ..rows.view()
        },
        3,
        &CoveredOptions {
            min_observations: 1,
            ..CoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("well-formed rows");
    // Row 1 is spared, and row 0 is still retired by it.
    assert_eq!(out.flagged, [true, false, false]);
    assert_eq!(out.census.rows_spared, 1);
    // The pair that would have retired it is still counted: what protection
    // refuses is the retirement, not the reading.
    assert_eq!(out.census.pairs_finer, read(&rows, 3).census.pairs_finer);
    assert_eq!(read(&rows, 3).census.rows_spared, 0);
}

// ─── the sweep ────────────────────────────────────────────────────────────

/// An owner left under the bar goes whole, and its survivors with it.
#[test]
fn an_owner_under_the_bar_goes_whole() {
    // Owner 0 loses one row of three and keeps two; owner 1 loses one of two,
    // so it and its survivor go; owner 2 loses none.
    let rows = Rows::new(
        &[
            (0, 0, 0.0, 0.0, 1.0),
            (1, 0, 0.0, 0.0, 1.0),
            (2, 0, 0.0, 0.0, 8.0),
            (2, 1, 1.0, 0.0, 1.0),
            (3, 1, 0.0, 0.0, 8.0),
            (3, 2, 1.0, 0.0, 1.0),
            (4, 2, 0.0, 0.0, 1.0),
            (5, 2, 0.0, 0.0, 1.0),
        ],
        1.0,
    );
    let out = covered_by_finer(
        rows.view(),
        3,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    // Image 2 holds owner 0's wide row beside owner 1's fine one, and image 3
    // owner 1's wide row beside owner 2's, so both wide rows are retired.
    assert_eq!(
        out.flagged,
        [false, false, true, false, true, false, false, false]
    );
    assert_eq!(out.keep_owner, [true, false, true]);
    assert_eq!(
        out.keep_row,
        [true, true, false, false, false, true, true, true]
    );
    assert_eq!(out.census.rows_flagged, 2);
    assert_eq!(out.census.rows_removed, 3);
    assert_eq!(out.census.owners_kept, 2);
    assert_eq!(out.census.owners_dropped_by_sweep, 1);
    assert_eq!(out.census.owners_dropped_all_covered, 0);
}

/// An owner no row names is claimed about in neither direction: it is not kept
/// and it is in neither drop count.
#[test]
fn an_owner_with_no_rows_is_in_no_drop_count() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0), (1, 0, 0.0, 0.0, 1.0)], 1.0);
    let out = covered_by_finer(
        rows.view(),
        3,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    assert_eq!(out.keep_owner, [true, false, false]);
    assert_eq!(out.census.owners_kept, 1);
    assert_eq!(out.census.owners_dropped_all_covered, 0);
    assert_eq!(out.census.owners_dropped_by_sweep, 0);
}

// ─── the whole rule ───────────────────────────────────────────────────────

/// `(image, owner, x, y, unit scale)` for the end-to-end case: the six
/// three-row owners, plus owner 6 on two images with owner 7 covering it on
/// one of them, so the sweep takes both.
fn end_to_end() -> Vec<(i64, i64, f64, f64, f64)> {
    let mut rows = spread(&[0, 1, 2]);
    rows.push((0, 6, 700.0, 100.0, 4.0));
    rows.push((1, 6, 700.0, 100.0, 4.0));
    rows.push((0, 7, 702.0, 100.0, 1.0));
    rows
}

/// The counts the NumPy stage reported on this set, verdict for verdict.
#[test]
fn the_rule_retires_the_covered_coarse_rows() {
    let rows = Rows::new(&end_to_end(), 1.0);
    let out = covered_by_finer(
        rows.view(),
        8,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    assert_eq!(out.census.rows, 21);
    assert_eq!(out.census.pairs_contained, 7);
    assert_eq!(out.census.pairs_finer, 4);
    assert_eq!(out.census.rows_flagged, 4);
    assert_eq!(out.census.rows_removed, 6);
    assert_eq!(out.census.owners_dropped_all_covered, 1);
    assert_eq!(out.census.owners_dropped_by_sweep, 2);
    assert_eq!(out.census.owners_kept, 5);
    assert_eq!(
        out.keep_owner,
        [false, true, true, true, true, true, false, false]
    );
}

/// The same six owners at a fortieth of their scale: every footprint is then
/// under a pixel, no centre falls inside another, and the rule fires nowhere
/// even though every radius RATIO is what it was.
#[test]
fn nothing_covered_retires_nothing() {
    let rows = Rows::new(&spread(&[0, 1, 2]), 0.025);
    let out = covered_by_finer(
        rows.view(),
        6,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    assert_eq!(out.census.pairs_contained, 0);
    assert_eq!(out.census.rows_flagged, 0);
    assert_eq!(out.census.rows_removed, 0);
    assert_eq!(out.census.owners_kept, 6);
    assert!(out.keep_row.iter().all(|&k| k));
}

// ─── determinism ──────────────────────────────────────────────────────────

/// Two readings of one input give one answer.
#[test]
fn two_readings_agree() {
    let rows = Rows::new(&end_to_end(), 1.0);
    let a = covered_by_finer(
        rows.view(),
        8,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    let b = covered_by_finer(
        rows.view(),
        8,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    assert_eq!(a, b);
}

/// The order the rows arrive in cannot change the verdict: a row is retired by
/// the existence of a cover, and every cover is measured against the rows as
/// they stand.
#[test]
fn the_row_order_cannot_change_the_verdict() {
    let forward = Rows::new(&end_to_end(), 1.0);
    let mut reversed_rows = end_to_end();
    reversed_rows.reverse();
    let reversed = Rows::new(&reversed_rows, 1.0);

    let a = covered_by_finer(
        forward.view(),
        8,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    let b = covered_by_finer(
        reversed.view(),
        8,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("well-formed rows");
    let back: Vec<bool> = b.flagged.iter().rev().copied().collect();
    assert_eq!(a.flagged, back);
    assert_eq!(a.census, b.census);
    assert_eq!(a.keep_owner, b.keep_owner);
}

// ─── refusals ─────────────────────────────────────────────────────────────

/// Rows that disagree on how many there are are refused, and the refusal says
/// what each input stated.
#[test]
fn disagreeing_lengths_are_refused() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0), (0, 1, 1.0, 0.0, 1.0)], 1.0);
    let error = covered_by_finer(
        CoveredRows {
            radius_px: &rows.radius[..1],
            ..rows.view()
        },
        2,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect_err("the radius column is short");
    assert!(matches!(
        error,
        CoveredByFinerError::LengthMismatch { radii: 1, .. }
    ));
    assert!(error.to_string().contains("radius_px 1"));
}

/// A protection mask of the wrong length is refused the same way.
#[test]
fn a_short_protection_mask_is_refused() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0), (0, 1, 1.0, 0.0, 1.0)], 1.0);
    let protected = [true];
    let error = covered_by_finer(
        CoveredRows {
            protected: Some(&protected),
            ..rows.view()
        },
        2,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect_err("the mask is short");
    assert!(matches!(
        error,
        CoveredByFinerError::LengthMismatch {
            protected: Some(1),
            ..
        }
    ));
}

/// A row naming an owner outside the declared space is refused by row.
#[test]
fn an_owner_out_of_range_is_refused_by_row() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0), (0, 4, 1.0, 0.0, 1.0)], 1.0);
    let error = covered_by_finer(
        rows.view(),
        2,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect_err("owner 4 of 2");
    assert_eq!(
        error,
        CoveredByFinerError::OwnerOutOfRange {
            row: 1,
            owner: 4,
            owner_count: 2,
        }
    );
    assert!(error.to_string().contains("row 1"));
}

/// A ratio below one would make every neighbour finer than itself, and a
/// non-finite floor would admit or refuse everything silently.
#[test]
fn an_unusable_threshold_is_refused() {
    let rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0)], 1.0);
    let error = covered_by_finer(
        rows.view(),
        1,
        &CoveredOptions {
            ratio: 0.5,
            ..CoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect_err("a ratio below one");
    assert_eq!(error, CoveredByFinerError::BadRatio(0.5));

    let error = covered_by_finer(
        rows.view(),
        1,
        &CoveredOptions {
            min_fine_radius_px: f64::NAN,
            ..CoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect_err("a floor that is not a number");
    assert!(matches!(error, CoveredByFinerError::BadMinFineRadius(_)));
}

/// A negative reach names no disk, and the enumeration's refusal is carried
/// out by name.
#[test]
fn a_negative_reach_is_refused() {
    let mut rows = Rows::new(&[(0, 0, 0.0, 0.0, 1.0), (0, 1, 1.0, 0.0, 1.0)], 1.0);
    rows.reach[1] = -1.0;
    let error = covered_by_finer(
        rows.view(),
        2,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect_err("a negative reach");
    assert!(matches!(error, CoveredByFinerError::Reach(_)));
    assert!(error.to_string().contains("row 1"));
}

/// A rule asked to stop decides nothing.
#[test]
fn a_cancelled_reading_decides_nothing() {
    let rows = Rows::new(&end_to_end(), 1.0);
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    let error = covered_by_finer(rows.view(), 8, &CoveredOptions::default(), &progress)
        .expect_err("it was asked to stop");
    assert_eq!(error, CoveredByFinerError::Cancelled);
}

/// No rows at all is an answer, not a refusal.
#[test]
fn no_rows_is_an_answer() {
    let rows = Rows::new(&[], 1.0);
    let out = covered_by_finer(
        rows.view(),
        0,
        &CoveredOptions::default(),
        &Progress::none(),
    )
    .expect("an empty set is well formed");
    assert_eq!(out.census, CoveredCensus::default());
    assert!(out.flagged.is_empty());
    assert!(out.keep_owner.is_empty());
}
