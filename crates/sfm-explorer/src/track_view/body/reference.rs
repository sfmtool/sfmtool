// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *Reference* column: which row the track's patch bitmap is rendered
//! from, which row the reference-view rule picks, what the rule decided about
//! each row, the per-view readings it decided on, and the sentence that says
//! why a row is not its pick; and the *ZNCC* cell's hover at the track stage,
//! which says how the row scores against that bitmap.
//!
//! The rule picks one `in` row
//! (`sfmtool_core::patch::reference_view::choose_reference_view`). The
//! reference in use is the row the stored bitmap is rendered from
//! (`TrackPayload::reference`, where the track has a bitmap). While that row
//! is pinned every render keeps it, whichever row the rule picks, so the two
//! can differ; unpinning the row, or *Set as reference* on the pick, makes
//! them agree. Each row's ZNCC is its score against the bitmap, plain and
//! blur-matched (`sfmtool_core::patch::stored_bitmap`).

use sfmtool_core::bench::{EditableTrack, Observation, StageKind, TrackMeasurement};
use sfmtool_core::patch::reference_view::{
    ReferenceFallback, ReferenceStanding, ReferenceTest, REFERENCE_AGREEMENT_MARGIN,
    REFERENCE_FACING_LIMIT_DEG, REFERENCE_MAX_CELL_DEFICIT, REFERENCE_MAX_CLIPPED_SHARE,
    REFERENCE_MAX_VIEWING_ANGLE_DEG, REFERENCE_MIN_COVERAGE,
};

use crate::bench::live::Evaluation;

/// The *Reference* heading's hover text.
pub(super) const REFERENCE_TIP: &str = "The track's reference, the row its patch bitmap is \
    rendered from, and which row the reference-view rule would pick from the current \
    readings.\n\n\
    The reference reads reference, on a green cell where the rule picks it too and on a red \
    cell where it does not. Then the rule's pick reads pick, on a grey cell: unpinning the \
    reference's row, or Set as reference on the pick, makes the pick the reference. While the \
    reference's row is pinned every render keeps it; a track put on the bench from a point has \
    every row pinned. Where the bitmap is the mean of the rows that are in, or there is no \
    bitmap yet, only the pick is marked.\n\n\
    A candidate has at least 99% of its tile on the photograph, at most 5% of the photograph \
    under the tile clipped to black or white, a viewing angle of at most 65\u{b0}, and no ninth \
    of the tile where it agrees with the other rows more than 0.3 below the track's typical \
    agreement there. Of the candidates whose pair ZNCC, the median of its ZNCCs with the \
    other rows that are in, is within 15 points of the best candidate's, the rule picks the one with \
    the smallest self-similarity radius. When no row passes, it drops the 65\u{b0} limit, then \
    the check of the ninths, then the coverage and clipping tests; a row that sees the patch \
    edge on or from behind, at 90\u{b0} or more, is never picked.\n\n\
    The first line is reference or pick, or for any other row the test that turned it away: \
    partial, clipped, oblique, ninth differs, agrees less, or less sharp, read against the \
    rule's pick. The second is the viewing angle, the angle between the patch's normal and the \
    direction to the camera, and the pair ZNCC the rule read. Hover a cell for every reading. \
    An out row is not considered.\n\n\
    Sorting by this column puts the reference first, then the pick, then the rows nearest to \
    being picked.";

/// How a row's *Reference* cell is marked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceMark {
    /// Neither the reference nor the rule's pick.
    None,
    /// The reference in use, which the rule picks too: a green cell.
    Reference,
    /// The reference in use, which the rule does not pick: a red cell. Its
    /// row's pin holds it.
    ReferenceNotPick,
    /// The reference in use where the rule picks no row: a cell with no fill,
    /// since there is no pick to accept in its place.
    ReferenceWithoutPick,
    /// The rule's pick where it is not the reference in use, or where the
    /// track has none: a grey cell.
    Pick,
}

/// The rows the *Reference* column marks, read off the track once per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ReferenceRows {
    /// The reference in use and its image: the row the stored bitmap is
    /// rendered from ([`crate::bench::reference_in_use`]), or `None` where the
    /// track has no bitmap or its bitmap is a fused mean.
    pub(crate) reference: Option<usize>,
    /// The reference's image index, for the hover.
    pub(crate) reference_image: Option<u32>,
    /// The row the reference-view rule picked at the last evaluation
    /// ([`crate::bench::reference_view_pick`]).
    pub(crate) pick: Option<usize>,
    /// The pick's image index, for the hover.
    pub(crate) pick_image: Option<u32>,
    /// Whether the track has a patch bitmap: with one and no reference, the
    /// bitmap is a fused mean.
    pub(crate) has_bitmap: bool,
}

impl ReferenceRows {
    /// The rows `track` marks.
    pub(crate) fn of(track: &EditableTrack) -> Self {
        let image = |i: Option<usize>| i.and_then(|i| track.observations.get(i)).map(|o| o.image);
        let reference = crate::bench::reference_in_use(track);
        let pick = crate::bench::reference_view_pick(track);
        Self {
            reference,
            reference_image: image(reference),
            pick,
            pick_image: image(pick),
            has_bitmap: track
                .track()
                .is_some_and(|p| p.committable_bitmap().is_some()),
        }
    }

    /// How row `index` is marked.
    pub(crate) fn mark(&self, index: usize) -> ReferenceMark {
        if self.reference == Some(index) {
            if self.pick == Some(index) {
                ReferenceMark::Reference
            } else if self.pick.is_none() {
                ReferenceMark::ReferenceWithoutPick
            } else {
                ReferenceMark::ReferenceNotPick
            }
        } else if self.pick == Some(index) {
            ReferenceMark::Pick
        } else {
            ReferenceMark::None
        }
    }
}

/// What one row's *Reference* cell draws.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReferenceCell {
    /// The cell's two lines: `reference`, `pick` or the test that turned the
    /// row away, then the viewing angle and the pair ZNCC; or `-` where
    /// nothing has measured the row.
    pub(crate) text: String,
    /// The hover text with what the mark means, every reading and the reason,
    /// or `None` where the cell is `-`.
    pub(crate) hover: Option<String>,
    /// How the cell is marked: the reference, the rule's pick, or neither.
    pub(crate) mark: ReferenceMark,
}

impl ReferenceCell {
    fn empty() -> Self {
        Self {
            text: "-".to_string(),
            hover: None,
            mark: ReferenceMark::None,
        }
    }
}

/// The *Reference* cell of row `index`, `observation`, at `stage`, as the
/// evaluation's state lets it print: nothing at the cluster stage, which has
/// no patch, or where the track could not be evaluated.
pub(super) fn reference_cell(
    index: usize,
    observation: &Observation,
    stage: StageKind,
    evaluation: &Evaluation,
    rows: &ReferenceRows,
) -> ReferenceCell {
    if stage != StageKind::Track
        || matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_))
    {
        return ReferenceCell::empty();
    }
    let mark = rows.mark(index);
    let Some(m) = observation.track.as_ref() else {
        return ReferenceCell::empty();
    };
    if mark == ReferenceMark::None && m.viewing_angle_deg.is_none() && m.reference_view.is_none() {
        return ReferenceCell::empty();
    }
    let word = match (mark, m.reference_view) {
        (
            ReferenceMark::Reference
            | ReferenceMark::ReferenceNotPick
            | ReferenceMark::ReferenceWithoutPick,
            _,
        ) => "reference",
        (ReferenceMark::Pick, _) => "pick",
        (ReferenceMark::None, None) => "-",
        (ReferenceMark::None, Some(standing)) => standing_word(standing),
    };
    let angle = m
        .viewing_angle_deg
        .map_or_else(|| "-".to_string(), |a| format!("{a:.0}\u{b0}"));
    let second = match m.pair_zncc {
        Some(pair) => format!("{angle}, {:.0}%", 100.0 * pair),
        None => angle,
    };
    let mut hover = mark_sentences(mark, observation.pinned, rows);
    if !hover.is_empty() {
        hover.push_str("\n\n");
    }
    hover.push_str(&reference_hover(m));
    ReferenceCell {
        text: format!("{word}\n{second}"),
        hover: Some(hover),
        mark,
    }
}

/// The lines over a marked cell's readings that say what the mark means and
/// how to change it; empty for an unmarked row.
fn mark_sentences(mark: ReferenceMark, pinned: bool, rows: &ReferenceRows) -> String {
    let image = |i: Option<u32>| i.map_or_else(|| "-".to_string(), |i| i.to_string());
    match mark {
        ReferenceMark::None => String::new(),
        ReferenceMark::Reference => format!(
            "The track's reference: the patch bitmap is this row's render, and the \
             reference-view rule picks this row too.{}",
            if pinned {
                " Its pin holds it."
            } else {
                " Its row is not pinned, so the reference follows the rule's pick at every \
                 render."
            }
        ),
        ReferenceMark::ReferenceWithoutPick => format!(
            "The track's reference: the patch bitmap is this row's render. The \
             reference-view rule picks no row.{}",
            if pinned { " Its pin holds it." } else { "" }
        ),
        ReferenceMark::ReferenceNotPick => {
            let pick = match rows.pick {
                Some(_) => format!(
                    "The reference-view rule picks the row of image {} instead. Unpinning \
                     this row, or Set as reference on that row, makes it the reference.",
                    image(rows.pick_image)
                ),
                None => "The reference-view rule picks no row.".to_string(),
            };
            format!(
                "The track's reference: the patch bitmap is this row's render, held by the \
                 row's pin. {pick}"
            )
        }
        ReferenceMark::Pick => match rows.reference {
            Some(_) => format!(
                "The reference-view rule's pick. The track's reference is the row of image {}, \
                 held by its pin. Unpinning that row, or Set as reference on this one, makes \
                 this row the reference.",
                image(rows.reference_image)
            ),
            None if rows.has_bitmap => "The reference-view rule's pick. The patch bitmap is \
                 the mean of the rows that are in, the render of no row, so there is no \
                 reference to mark. Set as reference on this row renders the bitmap from it."
                .to_string(),
            None => "The reference-view rule's pick. The track has no patch bitmap yet; the \
                 next render makes one."
                .to_string(),
        },
    }
}

/// The blur-matched score the *ZNCC* cell prints after the plain one, or
/// `None` where it prints one number: where the pair was read plain, and where
/// the two print the same.
pub(super) fn blur_matched_shown(m: &TrackMeasurement) -> Option<f64> {
    let plain = m.zncc.filter(|v| v.is_finite())?;
    m.bitmap_blur_sigma.filter(|&s| s > 0.0)?;
    let matched = m.blur_matched_zncc.filter(|v| v.is_finite())?;
    let shown = |v: f64| format!("{:.0}", 100.0 * v);
    (shown(plain) != shown(matched)).then_some(matched)
}

/// The *ZNCC* cell's hover text at the track stage: the row's score against
/// the stored patch bitmap, the blur-matched score and the blur's width or
/// the note that the row is sharper than the bitmap, the reason where there is
/// no score, and the localizer's leave-one-out reading beside them.
/// `is_reference` says the bitmap is this row's own render.
pub(super) fn zncc_hover(m: &TrackMeasurement, is_reference: bool) -> String {
    let percent = |v: f64| {
        if v.is_finite() {
            format!("{:.1}%", 100.0 * v)
        } else {
            "NaN".to_string()
        }
    };
    let mut lines: Vec<String> = Vec::new();
    match m.zncc {
        Some(_) if is_reference => lines.push(
            "This row is the track's reference: the patch bitmap is its render at its \
             keypoint, so its score against the bitmap is 100% and is not computed."
                .to_string(),
        ),
        Some(plain) => {
            lines.push(format!(
                "ZNCC with the stored patch bitmap {} whole, {} middle, read plain: the \
                 min ZNCC bars judge these.",
                percent(plain),
                m.zncc_middle.map_or_else(|| "-".to_string(), percent)
            ));
            let blurred = m.bitmap_blur_sigma.filter(|&s| s > 0.0);
            match (blurred, m.blur_matched_zncc) {
                (Some(sigma), Some(matched)) => lines.push(format!(
                    "Blur-matched {}: the bitmap blurred by {sigma:.2} grid px to this row's \
                     sharpness first. No bar judges it: a view out of focus scores as well \
                     blur-matched as a sharp one.",
                    percent(matched)
                )),
                _ if m.sharper_than_bitmap == Some(true) => lines.push(
                    "This row's tile is sharper than the bitmap along every direction, so \
                     neither is blurred: it could replace the reference."
                        .to_string(),
                ),
                _ => lines.push(
                    "Read plain only: the bitmap is not sharper than this row's tile along \
                     every direction by enough to blur it."
                        .to_string(),
                ),
            }
        }
        None => lines.push(match m.reason {
            Some(reason) => format!("No score against the stored patch bitmap: {reason}."),
            None => "No score against the stored patch bitmap.".to_string(),
        }),
    }
    lines.push(match m.loo_zncc {
        Some(loo) => format!(
            "Leave-one-out ZNCC {} whole, {} middle: the localizer's reading against the \
             consensus of the other rows, at the correlation peak the shift is measured to. \
             No bar judges it.",
            percent(loo),
            m.loo_zncc_middle.map_or_else(|| "-".to_string(), percent)
        ),
        None => "No leave-one-out ZNCC: the localizer could not read this row.".to_string(),
    });
    lines.join("\n")
}

/// The lines of a per-ninth grid, each value in percent, `-` where a ninth
/// has no reading.
fn grid_lines(grid: [[f64; 3]; 3]) -> Vec<String> {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|v| {
                    if v.is_finite() {
                        format!("{:>5.0}", 100.0 * v)
                    } else {
                        format!("{:>5}", "-")
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .collect()
}

/// The first line of a cell: `reference`, or the word for the test that
/// turned the row away.
fn standing_word(standing: ReferenceStanding) -> &'static str {
    match standing.rejected_by {
        None => "reference",
        Some(ReferenceTest::Coverage) => "partial",
        Some(ReferenceTest::Clipped) => "clipped",
        Some(ReferenceTest::Angle) => "oblique",
        Some(ReferenceTest::Cells) => "ninth differs",
        Some(ReferenceTest::Agreement) => "agrees less",
        Some(ReferenceTest::Sharpness) => "less sharp",
    }
}

/// Where row `index`, `observation`, stands for ordering by the column: the
/// reference in use first, then the rule's pick, then the rows nearest to
/// being picked, and last the rows a candidate test turned away, in the
/// reverse of the order the rule applies them. `None` for a row with nothing
/// to order by.
pub(super) fn reference_rank(
    index: usize,
    observation: &Observation,
    rows: &ReferenceRows,
) -> Option<f64> {
    if rows.reference == Some(index) {
        return Some(-1.0);
    }
    observation
        .track
        .as_ref()
        .and_then(|m| m.reference_view)
        .map(standing_rank)
}

/// [`reference_rank`] of a row the rule stood somewhere on.
fn standing_rank(standing: ReferenceStanding) -> f64 {
    match standing.rejected_by {
        None => 0.0,
        Some(ReferenceTest::Sharpness) => 1.0,
        Some(ReferenceTest::Agreement) => 2.0,
        Some(ReferenceTest::Cells) => 3.0,
        Some(ReferenceTest::Angle) => 4.0,
        Some(ReferenceTest::Clipped) => 5.0,
        Some(ReferenceTest::Coverage) => 6.0,
    }
}

/// The cell's hover text: the decision and why, then every reading.
pub(super) fn reference_hover(m: &TrackMeasurement) -> String {
    let percent = |v: f64| format!("{:.1}%", 100.0 * v);
    let mut lines: Vec<String> = Vec::new();
    match m.reference_view {
        None => lines.push(
            "Not considered for the reference view: the rule reads only the rows that are in."
                .to_string(),
        ),
        Some(standing) => {
            lines.push(match standing.rejected_by {
                None => "The reference view: the rule picks this row's tile.".to_string(),
                Some(test) => format!(
                    "Not the reference view: {}.",
                    rejection(test, standing.fallback, m)
                ),
            });
            if let Some(dropped) = fallback_sentence(standing.fallback) {
                lines.push(dropped.to_string());
            }
        }
    }
    lines.push(String::new());
    if let Some(angle) = m.viewing_angle_deg {
        lines.push(match m.tilt_direction_deg {
            Some(tilt) => format!(
                "Viewing angle {angle:.1}\u{b0}, leaning {tilt:.0}\u{b0} from the patch's u \
                 towards its v."
            ),
            None => format!("Viewing angle {angle:.1}\u{b0}, facing the patch."),
        });
    }
    if let Some(coverage) = m.coverage {
        lines.push(format!("Coverage {} of the tile.", percent(coverage)));
    }
    if let Some(clipped) = m.clipped_share {
        lines.push(format!(
            "Clipped {} of the photograph under the tile.",
            percent(clipped)
        ));
    }
    if let Some(pair) = m.pair_zncc {
        lines.push(format!(
            "Pair ZNCC {:.0}%, the median with the other rows that are in.",
            100.0 * pair
        ));
    }
    if let Some(deficit) = m.cell_deficit {
        lines.push(format!(
            "Cell deficit {deficit:.2}: how far it falls below the track's typical agreement \
             in its worst ninth."
        ));
    }
    if let Some(grid) = m.pair_zncc_grid {
        lines.push("Pair ZNCC per ninth, the median with the other rows that are in:".to_string());
        lines.extend(grid_lines(grid));
    }
    lines.join("\n")
}

/// Why `test` turned a row away under `fallback`, with the row's own reading
/// against the threshold.
fn rejection(test: ReferenceTest, fallback: ReferenceFallback, m: &TrackMeasurement) -> String {
    let or_dash = |v: Option<f64>, f: &dyn Fn(f64) -> String| v.map_or("-".to_string(), f);
    match test {
        ReferenceTest::Coverage => format!(
            "{} of its tile is on the photograph, under the {:.0}% a reference needs",
            or_dash(m.coverage, &|c| format!("{:.1}%", 100.0 * c)),
            100.0 * REFERENCE_MIN_COVERAGE
        ),
        ReferenceTest::Clipped => format!(
            "{} of the photograph under its tile is clipped, over the {:.0}% a reference may have",
            or_dash(m.clipped_share, &|c| format!("{:.1}%", 100.0 * c)),
            100.0 * REFERENCE_MAX_CLIPPED_SHARE
        ),
        ReferenceTest::Angle if fallback != ReferenceFallback::None => format!(
            "it sees the patch at {}, edge on or from behind: at or past the {:.0}\u{b0} that \
             no reference may reach",
            or_dash(m.viewing_angle_deg, &|a| format!("{a:.1}\u{b0}")),
            REFERENCE_FACING_LIMIT_DEG
        ),
        ReferenceTest::Angle => format!(
            "it sees the patch at {}, over the {:.0}\u{b0} a reference may be at",
            or_dash(m.viewing_angle_deg, &|a| format!("{a:.1}\u{b0}")),
            REFERENCE_MAX_VIEWING_ANGLE_DEG
        ),
        ReferenceTest::Cells => format!(
            "in one ninth of the tile it agrees with the other rows {} below the track's \
             typical agreement there, over the {} allowed",
            or_dash(m.cell_deficit, &|d| format!("{d:.2}")),
            REFERENCE_MAX_CELL_DEFICIT
        ),
        ReferenceTest::Agreement => format!(
            "its pair ZNCC is more than {:.0} points below the best candidate's",
            100.0 * REFERENCE_AGREEMENT_MARGIN
        ),
        // The rule compares the self-similarity ellipse's semi-major axis. A row
        // without one was not compared at all, and where no row left had one
        // the rule picked nothing.
        ReferenceTest::Sharpness
            if !m
                .zncc_self_similarity_ellipse
                .is_some_and(|e| e.grid_px.axes[0].is_finite()) =>
        {
            "it passed every other test, but it has no self-similarity radius, so the rule \
             could not compare its sharpness with the other candidates'"
                .to_string()
        }
        ReferenceTest::Sharpness => "it passed every test, and a candidate with a smaller \
             self-similarity radius did too"
            .to_string(),
    }
}

/// The sentence for a fallback, or `None` when every test applied.
fn fallback_sentence(fallback: ReferenceFallback) -> Option<&'static str> {
    match fallback {
        ReferenceFallback::None => None,
        ReferenceFallback::WithoutAngle => {
            Some("No row passed every test, so the rule dropped the 65\u{b0} angle limit.")
        }
        ReferenceFallback::WithoutAngleOrCells => Some(
            "No row passed every test, so the rule dropped the 65\u{b0} angle limit and the \
             check of the ninths.",
        ),
        ReferenceFallback::WithoutAny => Some(
            "No row passed the coverage and clipping tests, so every row that faces the patch \
             was a candidate.",
        ),
    }
}
