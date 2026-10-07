// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *Reference* column: what the reference-view rule decided about each
//! row, the per-view readings it decided on, and the sentence that says why a
//! row is not the reference.
//!
//! The rule picks the one `in` row whose tile could stand as the patch bitmap
//! (`sfmtool_core::patch::reference_view::choose_reference_view_with`). The
//! column reports it; nothing on the track depends on it.

use sfmtool_core::bench::{Observation, StageKind, TrackMeasurement};
use sfmtool_core::patch::reference_view::{
    PairZnccReading, ReferenceFallback, ReferenceRuleInputs, ReferenceStanding, ReferenceTest,
    REFERENCE_FACING_LIMIT_DEG, REFERENCE_MAX_CLIPPED_SHARE, REFERENCE_MAX_VIEWING_ANGLE_DEG,
    REFERENCE_MIN_COVERAGE,
};

use crate::bench::live::Evaluation;

/// The *Reference* heading's hover text.
pub(super) const REFERENCE_TIP: &str = "Which row's tile could stand as the patch bitmap, by \
    the reference-view rule, and the readings it decides on. The row it picks reads \
    reference, on a green cell.\n\n\
    A candidate has at least 99% of its tile on the photograph, at most 5% of the photograph \
    under the tile clipped to black or white, a viewing angle of at most 65\u{b0}, and no ninth \
    of the tile where it agrees with the other rows more than 0.25 below the track's typical \
    agreement there. Of the candidates whose pair ZNCC, the median of its ZNCCs with the \
    other rows that are in, is within 15 points of the best candidate's, the rule picks the one with \
    the smallest self-similarity radius. When no row passes, it drops the 65\u{b0} limit, then \
    the check of the ninths, then the coverage and clipping tests; a row that sees the patch \
    edge on or from behind, at 90\u{b0} or more, is never picked.\n\n\
    Both agreements are blur-matched: before two rows' tiles are correlated, the sharper one \
    is blurred to the other's sharpness, along each direction in which their self-similarity \
    ellipses differ by more than a quarter, so a sharp row is not counted as disagreeing for \
    the detail the blurrier rows lack. The hover gives the plain readings too.\n\n\
    The first line is the pick, or the test that turned the row away: partial, clipped, \
    oblique, ninth differs, agrees less, or less sharp. The second is the viewing angle, \
    the angle between the patch's normal and the direction to the camera, and the pair \
    ZNCC the rule read. Hover a cell for every reading. An out row is not considered.\n\n\
    The rule reports a view; the patch bitmap is fused from every in row.";

/// What one row's *Reference* cell draws.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReferenceCell {
    /// The cell's two lines: the standing, then the viewing angle and the pair
    /// ZNCC, or `-` where nothing has measured the row.
    pub(crate) text: String,
    /// The hover text with every reading and the reason, or `None` where the
    /// cell is `-`.
    pub(crate) hover: Option<String>,
    /// Whether the rule picks this row.
    pub(crate) is_reference: bool,
}

impl ReferenceCell {
    fn empty() -> Self {
        Self {
            text: "-".to_string(),
            hover: None,
            is_reference: false,
        }
    }
}

/// The *Reference* cell of `observation` at `stage`, as the evaluation's
/// state lets it print: nothing at the cluster stage, which has no patch, or
/// where the track could not be evaluated.
pub(super) fn reference_cell(
    observation: &Observation,
    stage: StageKind,
    evaluation: &Evaluation,
) -> ReferenceCell {
    if stage != StageKind::Track
        || matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_))
    {
        return ReferenceCell::empty();
    }
    let Some(m) = observation.track.as_ref() else {
        return ReferenceCell::empty();
    };
    if m.viewing_angle_deg.is_none() && m.reference_view.is_none() {
        return ReferenceCell::empty();
    }
    let word = match m.reference_view {
        None => "-",
        Some(standing) => standing_word(standing),
    };
    let angle = m
        .viewing_angle_deg
        .map_or_else(|| "-".to_string(), |a| format!("{a:.0}\u{b0}"));
    let second = match agreement_read(m) {
        Some(pair) => format!("{angle}, {:.0}%", 100.0 * pair),
        None => angle,
    };
    ReferenceCell {
        text: format!("{word}\n{second}"),
        hover: Some(reference_hover(m)),
        is_reference: m.reference_view.is_some_and(|s| s.is_reference()),
    }
}

/// Which readings the rule read for the row's track: the plain ones where it
/// has not decided on the row.
fn inputs_of(m: &TrackMeasurement) -> ReferenceRuleInputs {
    m.reference_view
        .map_or(ReferenceRuleInputs::PLAIN, |standing| standing.inputs)
}

/// The pair ZNCC the rule's agreement test read for the row.
fn agreement_read(m: &TrackMeasurement) -> Option<f64> {
    match inputs_of(m).agreement {
        PairZnccReading::Plain => m.pair_zncc,
        PairZnccReading::BlurMatched => m.blur_matched_pair_zncc,
    }
}

/// The cell deficit the rule's cell check read for the row.
fn cell_deficit_read(m: &TrackMeasurement) -> Option<f64> {
    match inputs_of(m).cells {
        PairZnccReading::Plain => m.cell_deficit,
        PairZnccReading::BlurMatched => m.blur_matched_cell_deficit,
    }
}

/// `"the rule reads it"` where `reading` is the one `read` names.
fn read_note(read: PairZnccReading, reading: PairZnccReading) -> &'static str {
    if read == reading {
        " The rule reads it."
    } else {
        ""
    }
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

/// Where a row stands for ordering by the column: the reference first, then
/// the rows nearest to being picked, and last the rows a candidate test
/// turned away, in the reverse of the order the rule applies them.
pub(super) fn reference_rank(standing: ReferenceStanding) -> f64 {
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
    let inputs = inputs_of(m);
    if let Some(pair) = m.pair_zncc {
        lines.push(format!(
            "Pair ZNCC {:.0}%, the median with the other rows that are in.{}",
            100.0 * pair,
            read_note(inputs.agreement, PairZnccReading::Plain)
        ));
    }
    if let Some(pair) = m.blur_matched_pair_zncc {
        lines.push(format!(
            "Blur-matched pair ZNCC {:.0}%: the same with the sharper tile of each pair blurred \
             to the other's sharpness first.{}",
            100.0 * pair,
            read_note(inputs.agreement, PairZnccReading::BlurMatched)
        ));
    }
    if let Some(deficit) = m.cell_deficit {
        lines.push(format!(
            "Cell deficit {deficit:.2}: how far it falls below the track's typical agreement \
             in its worst ninth.{}",
            read_note(inputs.cells, PairZnccReading::Plain)
        ));
    }
    if let Some(deficit) = m.blur_matched_cell_deficit {
        lines.push(format!(
            "Blur-matched cell deficit {deficit:.2}.{}",
            read_note(inputs.cells, PairZnccReading::BlurMatched)
        ));
    }
    if let Some(grid) = m.pair_zncc_grid {
        lines.push("Pair ZNCC per ninth, the median with the other rows that are in:".to_string());
        lines.extend(grid_lines(grid));
    }
    if let Some(grid) = m.blur_matched_pair_zncc_grid {
        lines.push("Blur-matched pair ZNCC per ninth:".to_string());
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
            or_dash(cell_deficit_read(m), &|d| format!("{d:.2}")),
            inputs_of(m).max_cell_deficit()
        ),
        ReferenceTest::Agreement => format!(
            "its {}pair ZNCC is more than {:.0} points below the best candidate's",
            match inputs_of(m).agreement {
                PairZnccReading::Plain => "",
                PairZnccReading::BlurMatched => "blur-matched ",
            },
            100.0 * inputs_of(m).agreement_margin()
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
