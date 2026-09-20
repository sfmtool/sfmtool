// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Retiring a coarse observation that a finer one, on another owner, covers.
//!
//! One rule over flat rows. A ROW is one observation: the image it was seen in,
//! the OWNER it belongs to (a point, a cluster, whatever the caller tracks), its
//! pixel position, the REACH of its drawn footprint, and its own feature RADIUS.
//! The reach and the radius are two different lengths and the rule needs both:
//! containment is asked at the reach, and "finer" is asked of the radius.
//!
//! A row is retired where another row **in the same image, on another owner**
//! has its centre inside the first row's footprint and a radius at least `ratio`
//! times smaller. The coarse side is the one retired, never the fine one. The
//! verdict is by existence, so the order rows arrive in cannot change it.
//!
//! An owner left with fewer than `min_observations` surviving rows is dropped
//! and its survivors go with it, because a caller that would not keep such an
//! owner must not be handed rows belonging to one.
//!
//! The enumeration is [`crate::spatial::keypoint_reach`], stated once and shared
//! with the other rules that ask the same neighbourhood question.
//!
//! See `specs/core/analysis/covered-by-finer.md` for the design.

use std::fmt;

use crate::progress::{Cancelled, Progress};
use crate::spatial::keypoint_reach::{pairs_within_reach, KeypointReachError, KeypointRows};

/// One row per observation, over whatever set of them the caller tracks.
#[derive(Debug, Clone, Copy)]
pub struct CoveredRows<'a> {
    /// `n` image index per row. Rows of different images are never paired.
    pub image_of_row: &'a [i64],
    /// `n` owner per row: the point, cluster or track the row belongs to. A row
    /// never covers another row of its own owner.
    pub owner_of_row: &'a [i64],
    /// `n * 2` pixel positions, `[x, y, x, y, ...]`.
    pub xy_px: &'a [f64],
    /// `n` footprint radius per row, in pixels: the disk containment is asked
    /// within. A reach that is not finite asks nothing and still covers.
    pub reach_px: &'a [f64],
    /// `n` feature radius per row, in pixels: the length "finer" is measured on.
    pub radius_px: &'a [f64],
    /// `n` rows that are never retired, or `None` for none of them. A protected
    /// row still covers other rows; what it is spared is being covered.
    pub protected: Option<&'a [bool]>,
}

/// The rule's thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoveredOptions {
    /// How many times finer the covering row's radius has to be. `2.0` is one
    /// octave, and the comparison is non-strict, so a pair exactly one octave
    /// apart is finer.
    pub ratio: f64,
    /// A covering row whose radius is below this says nothing: it is a
    /// collapsed measurement rather than a finer feature. `0.0` is off, which
    /// is the default, and off admits every radius the ratio admits.
    pub min_fine_radius_px: f64,
    /// How many surviving rows an owner needs to be kept.
    pub min_observations: usize,
}

impl Default for CoveredOptions {
    fn default() -> Self {
        Self {
            ratio: 2.0,
            min_fine_radius_px: 0.0,
            min_observations: 2,
        }
    }
}

/// What one reading of the rule saw and did.
///
/// The three owner counts are over owners that hold at least one row. An owner
/// the caller declared but that no row names is in none of them: nothing was
/// read about it, so nothing is claimed about it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoveredCensus {
    /// Rows read.
    pub rows: usize,
    /// Pairs where the candidate lies inside the row's footprint, is on another
    /// owner, and is strictly smaller. The population the scale test is over.
    pub pairs_contained: usize,
    /// Of those, the pairs that also pass the scale test: at least `ratio`
    /// finer, and the fine side at or above `min_fine_radius_px`. Counted
    /// whether or not the coarse side was protected.
    pub pairs_finer: usize,
    /// Rows the rule retired.
    pub rows_flagged: usize,
    /// Protected rows a passing pair would otherwise have retired. What
    /// [`CoveredRows::protected`] actually bought, as opposed to how many rows
    /// carried the mark.
    pub rows_spared: usize,
    /// Rows removed in all: the retired ones, plus the survivors of an owner
    /// the sweep dropped.
    pub rows_removed: usize,
    /// Owners dropped because every row they held was retired.
    pub owners_dropped_all_covered: usize,
    /// Owners dropped although a row of theirs survived the rule: the sweep
    /// left them under `min_observations`.
    pub owners_dropped_by_sweep: usize,
    /// Owners the sweep keeps.
    pub owners_kept: usize,
}

/// The rule's verdict, per row and per owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoveredByFiner {
    /// Per row, whether a finer row covers it.
    pub flagged: Vec<bool>,
    /// Per row, whether it survives: not flagged, and its owner survives the
    /// sweep.
    pub keep_row: Vec<bool>,
    /// Per owner, whether it survives the sweep.
    pub keep_owner: Vec<bool>,
    /// What the reading saw.
    pub census: CoveredCensus,
}

/// What the rule refuses to answer. Every variant names what did not hold.
#[derive(Debug, Clone, PartialEq)]
pub enum CoveredByFinerError {
    /// The per-row inputs disagree on how many rows there are.
    LengthMismatch {
        /// Rows the image index states.
        images: usize,
        /// Rows the owner column states.
        owners: usize,
        /// Rows the position array states.
        positions: usize,
        /// Rows the reach array states.
        reaches: usize,
        /// Rows the radius array states.
        radii: usize,
        /// Rows the protection mask states, where one was given.
        protected: Option<usize>,
    },
    /// A row names an owner outside the declared owner space.
    OwnerOutOfRange {
        /// The offending row.
        row: usize,
        /// The owner it named.
        owner: i64,
        /// How many owners the caller declared.
        owner_count: usize,
    },
    /// The scale ratio is not a usable multiple.
    BadRatio(f64),
    /// The fine-radius floor is not a usable length.
    BadMinFineRadius(f64),
    /// The enumeration underneath refused.
    Reach(KeypointReachError),
    /// The caller asked the rule to stop, and it did.
    Cancelled,
}

impl fmt::Display for CoveredByFinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                images,
                owners,
                positions,
                reaches,
                radii,
                protected,
            } => write!(
                f,
                "image_of_row states {images} rows, owner_of_row {owners}, xy_px {positions}, \
                 reach_px {reaches}, radius_px {radii}{}",
                match protected {
                    Some(p) => format!(" and protected {p}"),
                    None => String::new(),
                }
            ),
            Self::OwnerOutOfRange {
                row,
                owner,
                owner_count,
            } => write!(
                f,
                "row {row} names owner {owner}, and {owner_count} owners were declared"
            ),
            Self::BadRatio(ratio) => write!(
                f,
                "the scale ratio must be finite and at least 1, and {ratio} is not"
            ),
            Self::BadMinFineRadius(floor) => write!(
                f,
                "the fine-radius floor must be finite and not negative, and {floor} is not"
            ),
            Self::Reach(e) => write!(f, "{e}"),
            Self::Cancelled => write!(
                f,
                "the rule was asked to stop before it had a verdict, so nothing was decided"
            ),
        }
    }
}

impl std::error::Error for CoveredByFinerError {}

impl From<KeypointReachError> for CoveredByFinerError {
    fn from(e: KeypointReachError) -> Self {
        Self::Reach(e)
    }
}

impl From<Cancelled> for CoveredByFinerError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Which rows a finer row covers, and what survives once the owners that fall
/// below `min_observations` go.
///
/// `owner_count` is the owner space the rows index: `keep_owner` has one entry
/// per owner in it, so a caller holding a per-owner array reads the verdict
/// straight off its own indexing.
///
/// The rule is three tests over the pairs the enumeration produces, all of them
/// on the pair alone, which is why the answer does not depend on the order the
/// rows arrive in:
///
/// - the candidate's centre lies inside the row's own footprint;
/// - the candidate is on another owner, so nothing covers itself;
/// - the candidate is at least [`CoveredOptions::ratio`] times finer, and is
///   itself no finer than [`CoveredOptions::min_fine_radius_px`].
///
/// The **coarse** row is what a passing pair retires. A row named by
/// [`CoveredRows::protected`] is never retired, and still covers.
///
/// `progress` names the three stages (the enumeration, the rule, the sweep) and
/// is how the call is asked to stop; a stopped call decides nothing. Pass
/// `&Progress::none()` to report nothing and never stop.
///
/// # Example
///
/// ```
/// use sfmtool_core::analysis::covered_by_finer::{
///     covered_by_finer, CoveredOptions, CoveredRows,
/// };
/// use sfmtool_core::progress::Progress;
///
/// // Two rows of one image: a wide one at the origin and a fine one 1 px away,
/// // on different owners and exactly one octave apart.
/// let out = covered_by_finer(
///     CoveredRows {
///         image_of_row: &[0, 0],
///         owner_of_row: &[0, 1],
///         xy_px: &[0.0, 0.0, 1.0, 0.0],
///         reach_px: &[10.0, 2.5],
///         radius_px: &[4.0, 2.0],
///         protected: None,
///     },
///     2,
///     &CoveredOptions {
///         min_observations: 1,
///         ..CoveredOptions::default()
///     },
///     &Progress::none(),
/// )
/// .expect("well-formed rows");
/// assert_eq!(out.flagged, [true, false]);
/// ```
///
/// # Errors
///
/// [`CoveredByFinerError`] states which precondition did not hold, or that the
/// call was cancelled.
pub fn covered_by_finer(
    rows: CoveredRows<'_>,
    owner_count: usize,
    options: &CoveredOptions,
    progress: &Progress<'_>,
) -> Result<CoveredByFiner, CoveredByFinerError> {
    let n = rows.image_of_row.len();
    if rows.owner_of_row.len() != n
        || rows.xy_px.len() != 2 * n
        || rows.reach_px.len() != n
        || rows.radius_px.len() != n
        || rows.protected.is_some_and(|p| p.len() != n)
    {
        return Err(CoveredByFinerError::LengthMismatch {
            images: n,
            owners: rows.owner_of_row.len(),
            positions: rows.xy_px.len() / 2,
            reaches: rows.reach_px.len(),
            radii: rows.radius_px.len(),
            protected: rows.protected.map(<[bool]>::len),
        });
    }
    if !(options.ratio.is_finite() && options.ratio >= 1.0) {
        return Err(CoveredByFinerError::BadRatio(options.ratio));
    }
    if !(options.min_fine_radius_px.is_finite() && options.min_fine_radius_px >= 0.0) {
        return Err(CoveredByFinerError::BadMinFineRadius(
            options.min_fine_radius_px,
        ));
    }
    for (row, &owner) in rows.owner_of_row.iter().enumerate() {
        if owner < 0 || owner as usize >= owner_count {
            return Err(CoveredByFinerError::OwnerOutOfRange {
                row,
                owner,
                owner_count,
            });
        }
    }
    progress.check_cancel()?;

    let [p_pairs, p_rule, p_sweep] = progress.split([0.70, 0.20, 0.10]);
    let pairs = {
        let _phase = p_pairs.phase("enumerate footprints");
        pairs_within_reach(KeypointRows {
            image_of_row: rows.image_of_row,
            xy_px: rows.xy_px,
            reach_px: rows.reach_px,
        })?
    };
    progress.check_cancel()?;

    let mut flagged = vec![false; n];
    let mut spared = vec![false; n];
    let mut pairs_contained = 0usize;
    let mut pairs_finer = 0usize;
    {
        let mut phase = p_rule.phase("read the rule");
        for k in 0..pairs.len() {
            let big = pairs.row[k] as usize;
            let small = pairs.candidate[k] as usize;
            // The containment is the enumeration's own and is restated here
            // rather than assumed: it keeps the rule readable as the three
            // tests it is, and costs one comparison per pair. A NaN radius on
            // either side fails the size test and takes the pair out, which is
            // the answer wanted -- a row whose size is unstated is neither
            // coarser nor finer than anything.
            let contained = pairs.distance_px[k] <= rows.reach_px[big]
                && rows.radius_px[small] < rows.radius_px[big]
                && rows.owner_of_row[small] != rows.owner_of_row[big];
            if !contained {
                continue;
            }
            pairs_contained += 1;
            let finer = rows.radius_px[big] >= options.ratio * rows.radius_px[small]
                && rows.radius_px[small] >= options.min_fine_radius_px;
            if !finer {
                continue;
            }
            pairs_finer += 1;
            if rows.protected.is_some_and(|p| p[big]) {
                spared[big] = true;
            } else {
                flagged[big] = true;
            }
        }
        crate::progress_note!(
            phase,
            "{} pairs contained, {pairs_finer} a band finer",
            pairs_contained
        );
    }
    progress.check_cancel()?;

    let _phase = p_sweep.phase("sweep the owners");
    let mut rows_of_owner = vec![0usize; owner_count];
    let mut flagged_of_owner = vec![0usize; owner_count];
    for (row, &owner) in rows.owner_of_row.iter().enumerate() {
        rows_of_owner[owner as usize] += 1;
        flagged_of_owner[owner as usize] += usize::from(flagged[row]);
    }
    let keep_owner: Vec<bool> = (0..owner_count)
        .map(|o| rows_of_owner[o] - flagged_of_owner[o] >= options.min_observations)
        .collect();
    let keep_row: Vec<bool> = (0..n)
        .map(|row| !flagged[row] && keep_owner[rows.owner_of_row[row] as usize])
        .collect();

    let rows_flagged = flagged.iter().filter(|&&f| f).count();
    let rows_removed = keep_row.iter().filter(|&&k| !k).count();
    let mut owners_dropped_all_covered = 0usize;
    let mut owners_dropped_by_sweep = 0usize;
    for o in 0..owner_count {
        if keep_owner[o] || rows_of_owner[o] == 0 {
            continue;
        }
        if flagged_of_owner[o] == rows_of_owner[o] {
            owners_dropped_all_covered += 1;
        } else {
            owners_dropped_by_sweep += 1;
        }
    }
    Ok(CoveredByFiner {
        flagged,
        keep_row,
        census: CoveredCensus {
            rows: n,
            pairs_contained,
            pairs_finer,
            rows_flagged,
            rows_spared: spared.iter().filter(|&&s| s).count(),
            rows_removed,
            owners_dropped_all_covered,
            owners_dropped_by_sweep,
            owners_kept: keep_owner.iter().filter(|&&k| k).count(),
        },
        keep_owner,
    })
}

#[cfg(test)]
mod tests;
