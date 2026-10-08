// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-cell columns of `cluster_patches/`: what the piecewise refinement
//! read in each of the nine cells of every member's patch.
//!
//! Format version 8 adds four optional entries to the section, present
//! together or absent together: `member_cell_shift_px`, `member_cell_zncc`,
//! `member_cell_status` and `member_cell_iterations`. The status column
//! indexes the `member_cell_status_names` legend in the section's metadata,
//! the same way `member_status` indexes `member_status_names`. See
//! `specs/formats/matches-file-format.md` § "Cluster Patches".

use std::fmt;
use std::str::FromStr;

use ndarray::{Array1, Array3, Array4, Axis};

use crate::entries;
use crate::types::{
    legend_names, normalize_codes, parse_legend, ClusterMemberStatus, MatchesError,
};

/// The first format version whose `cluster_patches/` section may carry the
/// per-cell columns and their `member_cell_status_names` legend.
pub(crate) const MEMBER_CELLS_VERSION: u32 = 8;

/// The metadata key of the legend `member_cell_status` indexes.
pub(crate) const MEMBER_CELL_STATUS_NAMES_KEY: &str = "member_cell_status_names";

/// What the piecewise refinement concluded about one cell of a member — the
/// meaning behind a `cluster_patches/member_cell_status` code.
///
/// The stored code is an index into the file's `member_cell_status_names`
/// legend. [`crate::read_matches`] resolves every code through that legend and
/// hands back the canonical numbering these discriminants define, and
/// [`crate::write_matches`] states the canonical legend ([`Self::NAMES`]).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClusterCellStatus {
    /// The cell's shift was measured and the member's affine shape was fitted
    /// to it.
    Fitted = 0,
    /// Refused because the cell does not pin a shift: the reference is flat
    /// over it, or its ZNCC over the shift search is too flat at the peak.
    RefusedCurvature = 1,
    /// Refused because its ZNCC at the best shift is below the bar: the cell
    /// lies over a different surface in this view.
    RefusedZncc = 2,
    /// Not registered, or registered without effect: the member is not
    /// kept, a sample the search needs could not be read, no cell of the
    /// member survived, or the refined shape could not be accepted (the
    /// render failed, the fitted update was not finite or reflected, the
    /// whole-patch ZNCC or shift read again at the refined shape failed the
    /// member's gates, or the refined shape's support left the frame). In
    /// every case but the first the member keeps its whole-patch shape, and
    /// its `member_cell_iterations` count includes the pass that failed.
    NotAttempted = 3,
    /// Refused because its best shift lies on the search bound, so the
    /// optimum is at or past the bound.
    RefusedBound = 4,
}

impl ClusterCellStatus {
    /// Every cell status this format defines, in canonical order: a status's
    /// position here is its discriminant.
    pub const ALL: [ClusterCellStatus; 5] = [
        Self::Fitted,
        Self::RefusedCurvature,
        Self::RefusedZncc,
        Self::NotAttempted,
        Self::RefusedBound,
    ];

    /// The canonical `member_cell_status_names` legend, one name per entry of
    /// [`Self::ALL`].
    pub const NAMES: [&'static str; 5] = [
        "fitted",
        "refused_curvature",
        "refused_zncc",
        "not_attempted",
        "refused_bound",
    ];

    /// The canonical lowercase name the legend states.
    pub fn as_str(&self) -> &'static str {
        Self::NAMES[*self as usize]
    }

    /// Decode a canonical discriminant; `None` when out of range.
    pub fn from_u8(value: u8) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }
}

impl FromStr for ClusterCellStatus {
    type Err = MatchesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|status| status.as_str() == s)
            .ok_or_else(|| MatchesError::InvalidFormat(format!("Unknown ClusterCellStatus: {s:?}")))
    }
}

impl fmt::Display for ClusterCellStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The per-cell columns of `cluster_patches/`, member-parallel like the rest
/// of the section. Cells are indexed `[member, row, col]` from the top-left
/// cell of the member's patch grid.
///
/// Only a member whose status is `kept` carries readings. Every other member's
/// row is `NaN` shifts, `NaN` ZNCCs, [`ClusterCellStatus::NotAttempted`]
/// throughout and `0` iterations.
#[derive(Debug, Clone)]
pub struct MemberCellData {
    /// `(M, 3, 3, 2)` each cell's displacement `[x, y]` from where the
    /// member's affine shape places it, in patch grid px; `NaN` where no
    /// shift was measured.
    pub shift_px: Array4<f32>,
    /// `(M, 3, 3)` each cell's ZNCC against the reference at its best shift;
    /// `NaN` where nothing was read.
    pub zncc: Array3<f32>,
    /// `(M, 3, 3)` [`ClusterCellStatus`] discriminants, in the canonical
    /// numbering.
    pub status: Array3<u8>,
    /// `(M,)` renders the refinement made for the member; `0` for a member it
    /// did not run on.
    pub iterations: Array1<u8>,
}

impl MemberCellData {
    /// Columns for `member_count` members none of which was refined: `NaN`
    /// shifts and ZNCCs, every cell not attempted, no iterations.
    pub fn not_attempted(member_count: usize) -> MemberCellData {
        MemberCellData {
            shift_px: Array4::from_elem((member_count, 3, 3, 2), f32::NAN),
            zncc: Array3::from_elem((member_count, 3, 3), f32::NAN),
            status: Array3::from_elem((member_count, 3, 3), ClusterCellStatus::NotAttempted as u8),
            iterations: Array1::zeros(member_count),
        }
    }

    /// The rows of `members`, in that order.
    pub(crate) fn select(&self, members: &[usize]) -> MemberCellData {
        MemberCellData {
            shift_px: self.shift_px.select(Axis(0), members),
            zncc: self.zncc.select(Axis(0), members),
            status: self.status.select(Axis(0), members),
            iterations: self.iterations.select(Axis(0), members),
        }
    }

    /// The first way these columns disagree with the section's
    /// `member_status`, or `None` when they are consistent: every column sized
    /// by the member count, every status a defined cell status, and a member
    /// that is not `kept` carrying no readings.
    pub(crate) fn validation_error(&self, member_status: &[u8]) -> Option<String> {
        let member_count = member_status.len();
        for (name, shape, expected) in [
            (
                "member_cell_shift_px",
                self.shift_px.shape(),
                &[member_count, 3, 3, 2][..],
            ),
            ("member_cell_zncc", self.zncc.shape(), &[member_count, 3, 3]),
            (
                "member_cell_status",
                self.status.shape(),
                &[member_count, 3, 3],
            ),
            (
                "member_cell_iterations",
                self.iterations.shape(),
                &[member_count],
            ),
        ] {
            if shape != expected {
                return Some(format!("{name} shape {shape:?} != {expected:?}"));
            }
        }
        let status: Vec<u8> = self.status.iter().copied().collect();
        if let Some((i, &code)) = status
            .iter()
            .enumerate()
            .find(|(_, &code)| ClusterCellStatus::from_u8(code).is_none())
        {
            return Some(format!(
                "{} = {code} is not a valid ClusterCellStatus discriminant",
                locate_cell(i)
            ));
        }
        let shift_px: Vec<f32> = self.shift_px.iter().copied().collect();
        let zncc: Vec<f32> = self.zncc.iter().copied().collect();
        let iterations: Vec<u8> = self.iterations.iter().copied().collect();
        not_kept_with_readings(member_status, &status, &shift_px, &zncc, &iterations)
    }
}

/// Name the cell at flat index `i` of a `(M, 3, 3)` column.
pub(crate) fn locate_cell(i: usize) -> String {
    format!("member_cell_status[{}][{}][{}]", i / 9, (i / 3) % 3, i % 3)
}

/// The first member that is not `kept` yet carries a cell reading (a status
/// other than not attempted, a displacement or ZNCC that is not `NaN`, or a
/// non-zero iteration count), as a message.
///
/// `status` is the `(M, 3, 3)` canonical cell codes, `shift_px` the
/// `(M, 3, 3, 2)` displacements and `zncc` the `(M, 3, 3)` ZNCCs, all in
/// row-major order, and `iterations` the `(M,)` counts, each already checked
/// to be sized by the member count.
pub(crate) fn not_kept_with_readings(
    member_status: &[u8],
    status: &[u8],
    shift_px: &[f32],
    zncc: &[f32],
    iterations: &[u8],
) -> Option<String> {
    let not_attempted = ClusterCellStatus::NotAttempted as u8;
    for (m, &member) in member_status.iter().enumerate() {
        if member == ClusterMemberStatus::Kept as u8 {
            continue;
        }
        let name = ClusterMemberStatus::from_u8(member)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let not_kept = format!("but member {m} is {name}, not kept, and carries no cell readings");
        let cells = &status[m * 9..m * 9 + 9];
        if let Some(i) = cells.iter().position(|&c| c != not_attempted) {
            return Some(format!(
                "{} is {}, {not_kept}",
                locate_cell(m * 9 + i),
                ClusterCellStatus::from_u8(cells[i])
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
            ));
        }
        let shifts = &shift_px[m * 18..m * 18 + 18];
        if let Some(i) = shifts.iter().position(|v| !v.is_nan()) {
            return Some(format!(
                "member_cell_shift_px[{m}][{}][{}][{}] is {}, {not_kept}",
                i / 6,
                (i / 2) % 3,
                i % 2,
                shifts[i]
            ));
        }
        let znccs = &zncc[m * 9..m * 9 + 9];
        if let Some(i) = znccs.iter().position(|v| !v.is_nan()) {
            return Some(format!(
                "member_cell_zncc[{m}][{}][{}] is {}, {not_kept}",
                i / 3,
                i % 3,
                znccs[i]
            ));
        }
        if iterations[m] != 0 {
            return Some(format!(
                "member_cell_iterations[{m}] is {}, {not_kept}",
                iterations[m]
            ));
        }
    }
    None
}

/// The legend a file's `member_cell_status` codes index, as the canonical
/// [`ClusterCellStatus`] code each stored code stands for, or `None` when the
/// file carries no per-cell columns.
///
/// A file below [`MEMBER_CELLS_VERSION`] must not carry the legend. The legend
/// rides inside `cluster_patches/metadata.json`, which is hashed into the
/// section digest with the columns it describes.
pub(crate) fn read_member_cell_status_legend(
    cp_meta: &serde_json::Value,
    version: u32,
) -> Result<Option<Vec<u8>>, String> {
    let Some(names) = legend_names(cp_meta, MEMBER_CELL_STATUS_NAMES_KEY)? else {
        return Ok(None);
    };
    if version < MEMBER_CELLS_VERSION {
        return Err(format!(
            "version {version} file carries cluster_patches/metadata.json \
             {MEMBER_CELL_STATUS_NAMES_KEY} (introduced in version {MEMBER_CELLS_VERSION})"
        ));
    }
    parse_legend(
        MEMBER_CELL_STATUS_NAMES_KEY,
        &names,
        &ClusterCellStatus::NAMES,
    )
    .map(Some)
}

/// Rewrite stored `member_cell_status` codes in place onto the canonical
/// numbering, resolving each through `legend`.
pub(crate) fn normalize_cell_statuses(codes: &mut [u8], legend: &[u8]) -> Result<(), String> {
    normalize_codes(codes, legend, locate_cell)
}

/// The per-cell entries for `member_count` members, each as the name prefix
/// that identifies the column and the full name at that count, in the
/// lexicographic order the section hash takes them.
pub(crate) fn cell_entry_names(member_count: usize) -> [(&'static str, String); 4] {
    [
        (
            entries::cluster_patches_member_cell_iterations_prefix(),
            entries::cluster_patches_member_cell_iterations(member_count),
        ),
        (
            entries::cluster_patches_member_cell_shift_px_prefix(),
            entries::cluster_patches_member_cell_shift_px(member_count),
        ),
        (
            entries::cluster_patches_member_cell_status_prefix(),
            entries::cluster_patches_member_cell_status(member_count),
        ),
        (
            entries::cluster_patches_member_cell_zncc_prefix(),
            entries::cluster_patches_member_cell_zncc(member_count),
        ),
    ]
}

/// Check the archive's per-cell entries against the legend: with the legend,
/// all four columns are present at the member count's shape; without it, none
/// is. `names` is every entry name in the archive.
///
/// Returns every problem found, so the verifier can report them all; the
/// reader stops at the first.
pub(crate) fn cell_entry_errors<'a>(
    names: impl Iterator<Item = &'a str> + Clone,
    member_count: usize,
    legend_present: bool,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (prefix, expected) in cell_entry_names(member_count) {
        let column = &prefix["cluster_patches/".len()..prefix.len() - 1];
        let mut found = names.clone().filter(|n| n.starts_with(prefix));
        match (legend_present, found.next()) {
            (true, None) => errors.push(format!(
                "cluster_patches/metadata.json carries {MEMBER_CELL_STATUS_NAMES_KEY} but the \
                 {column} column is missing"
            )),
            (true, Some(name)) if name != expected => errors.push(format!(
                "{name} does not match the member count: expected {expected}"
            )),
            (false, Some(name)) => errors.push(format!(
                "{name} is present but cluster_patches/metadata.json carries no \
                 {MEMBER_CELL_STATUS_NAMES_KEY} to read the per-cell columns through"
            )),
            _ => {}
        }
    }
    errors
}
