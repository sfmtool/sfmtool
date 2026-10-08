// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *Reference* and *Bitmap* columns: what the reference-view rule decided
//! about each row, the per-view readings it decided on, and the sentence that
//! says why a row is not the reference; and how each row scores against the
//! track's stored patch bitmap.
//!
//! The rule picks the one `in` row whose tile is stored as the patch bitmap
//! (`sfmtool_core::patch::reference_view::choose_reference_view`); a fit or a
//! render of the bitmap stores that row's tile and names it
//! (`TrackPayload::reference`). The *Bitmap* column reports each row's ZNCC
//! with that bitmap, plain and blur-matched
//! (`sfmtool_core::patch::stored_bitmap`).

use sfmtool_core::bench::{Observation, StageKind, TrackMeasurement};
use sfmtool_core::patch::reference_view::{
    ReferenceFallback, ReferenceStanding, ReferenceTest, REFERENCE_AGREEMENT_MARGIN,
    REFERENCE_FACING_LIMIT_DEG, REFERENCE_MAX_CELL_DEFICIT, REFERENCE_MAX_CLIPPED_SHARE,
    REFERENCE_MAX_VIEWING_ANGLE_DEG, REFERENCE_MIN_COVERAGE,
};

use crate::bench::live::Evaluation;

/// The *Reference* heading's hover text.
pub(super) const REFERENCE_TIP: &str = "Which row's tile is stored as the patch bitmap, by \
    the reference-view rule, and the readings it decides on. The row it picks reads \
    reference, on a green cell.\n\n\
    A candidate has at least 99% of its tile on the photograph, at most 5% of the photograph \
    under the tile clipped to black or white, a viewing angle of at most 65\u{b0}, and no ninth \
    of the tile where it agrees with the other rows more than 0.3 below the track's typical \
    agreement there. Of the candidates whose pair ZNCC, the median of its ZNCCs with the \
    other rows that are in, is within 15 points of the best candidate's, the rule picks the one with \
    the smallest self-similarity radius. When no row passes, it drops the 65\u{b0} limit, then \
    the check of the ninths, then the coverage and clipping tests; a row that sees the patch \
    edge on or from behind, at 90\u{b0} or more, is never picked.\n\n\
    The first line is the pick, or the test that turned the row away: partial, clipped, \
    oblique, ninth differs, agrees less, or less sharp. The second is the viewing angle, \
    the angle between the patch's normal and the direction to the camera, and the pair \
    ZNCC the rule read. Hover a cell for every reading. An out row is not considered.\n\n\
    A fit stores the picked row's tile as the patch bitmap, and the Bitmap column marks \
    the row the stored bitmap is the tile of.";

/// The *Bitmap* heading's hover text.
pub(super) const BITMAP_TIP: &str = "How each row's tile scores against the track's patch \
    bitmap, which is the tile of the reference row.\n\n\
    The row the bitmap is the tile of reads bitmap; its score is 100% and is not computed. \
    Every other row reads its ZNCC with the bitmap, over the samples both have on the \
    photograph. Where the bitmap is sharper than the row's tile along every direction (the \
    short axis of the row's self-similarity ellipse is at least a quarter longer than the \
    long axis of the bitmap's), the bitmap alone is blurred by a round blur until its long \
    axis reaches the row's short axis (up to 2 grid px) and correlated again: the second line \
    gives that blur-matched score after an arrow. A row sharper than the bitmap is read \
    plain and its second line reads sharper: its tile could replace the reference.\n\n\
    A bitmap stored before the reference was recorded, or a mean of the rows, names no row, \
    and every row is scored.";

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
    let second = match m.pair_zncc {
        Some(pair) => format!("{angle}, {:.0}%", 100.0 * pair),
        None => angle,
    };
    ReferenceCell {
        text: format!("{word}\n{second}"),
        hover: Some(reference_hover(m)),
        is_reference: m.reference_view.is_some_and(|s| s.is_reference()),
    }
}

/// What one row's *Bitmap* cell draws.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BitmapCell {
    /// The cell's two lines: `bitmap` for the row the bitmap is the tile of,
    /// otherwise the plain score over the blur-matched one or `sharper`; `-`
    /// where the row has no score.
    pub(crate) text: String,
    /// The hover text with the scores and the blur, or `None` where the cell
    /// is `-`.
    pub(crate) hover: Option<String>,
    /// Whether the stored bitmap is this row's tile.
    pub(crate) is_bitmap: bool,
}

/// The *Bitmap* cell of row `index`, `observation`, of a track whose bitmap is
/// the tile of row `bitmap_row`, as the evaluation's state lets it print.
pub(super) fn bitmap_cell(
    index: usize,
    observation: &Observation,
    bitmap_row: Option<usize>,
    stage: StageKind,
    evaluation: &Evaluation,
) -> BitmapCell {
    let empty = BitmapCell {
        text: "-".to_string(),
        hover: None,
        is_bitmap: false,
    };
    if stage != StageKind::Track
        || matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_))
    {
        return empty;
    }
    let Some(m) = observation.track.as_ref() else {
        return empty;
    };
    if bitmap_row == Some(index) {
        return BitmapCell {
            text: "bitmap\n100%".to_string(),
            hover: Some(
                "The patch bitmap is this row's tile, rendered at its keypoint: its score is \
                 100% and is not computed."
                    .to_string(),
            ),
            is_bitmap: true,
        };
    }
    let Some(plain) = m.bitmap_zncc else {
        return empty;
    };
    let blurred = m.bitmap_blur_sigma.filter(|&s| s > 0.0);
    let second = match (blurred, m.blur_matched_bitmap_zncc) {
        // The arrow is U+23F5, which egui's bundled fonts draw; U+2192 draws
        // as a box.
        (Some(_), Some(matched)) => format!("\u{23f5} {:.0}%", 100.0 * matched),
        _ if m.sharper_than_bitmap == Some(true) => "sharper".to_string(),
        _ => String::new(),
    };
    let mut hover = vec![format!(
        "ZNCC with the patch bitmap {:.1}%, as stored.",
        100.0 * plain
    )];
    match (blurred, m.blur_matched_bitmap_zncc) {
        (Some(sigma), Some(matched)) => hover.push(format!(
            "Blur-matched {:.1}%: the bitmap blurred by {sigma:.2} grid px to this row's \
             sharpness first.",
            100.0 * matched
        )),
        _ if m.sharper_than_bitmap == Some(true) => hover.push(
            "This row's tile is sharper than the bitmap along every direction, so neither is \
             blurred: it could replace the reference."
                .to_string(),
        ),
        _ => hover.push(
            "Read plain: the bitmap is not sharper than this row's tile along every direction \
             by enough to blur it."
                .to_string(),
        ),
    }
    BitmapCell {
        text: format!("{:.0}%\n{second}", 100.0 * plain),
        hover: Some(hover.join("\n")),
        is_bitmap: false,
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
