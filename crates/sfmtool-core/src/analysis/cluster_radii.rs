// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How coarse each cluster of a cluster backbone is, and the coarsest-N cut
//! ordered by it.
//!
//! A cluster member stores an absolute 2x2 affine shape: the map from the
//! detector's canonical unit frame onto that member's image pixels. Its two
//! column norms are the member's image-space extent, so the refine radius times
//! their MEAN is the member's feature radius in pixels, and a cluster's radius
//! is its widest member's. That single reading is what
//! [`crate::analysis::source_clusters`] bands against and what the coarsest-N
//! cut orders by, so both take it from here rather than restating it.
//!
//! The reading is `f32` wide, in and out: `f32` is what a `.matches` backbone
//! stores the shapes at, and the computation gives nothing back for a wider
//! accumulator to hold onto -- each radius is two squares of one member's own
//! shape summed under a square root, with no cancellation, and a cluster takes
//! a MAX over its members rather than a sum, so no error accumulates along a
//! cluster however many members it has. What consumes a radius orders or bands
//! by it, and `f32` resolves the gap between two clusters far below the
//! precision either of those decisions is taken at.
//!
//! See `specs/core/analysis/source-clusters.md` for the design.

use std::borrow::Cow;
use std::cmp::Ordering;

use matches_format::MatchesData;
use rayon::prelude::*;

use crate::geometry::focal_vote::contiguous;

/// What [`cluster_radii_from_matches`] and [`coarsest_cluster_ids_from_matches`]
/// refuse: a property of the `.matches` file, never of the radius reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterRadiiError {
    /// The file stores the pairwise backbone, so it holds no clusters to
    /// measure.
    NoClusters,
    /// The cluster backbone carries no member affine shapes -- a file written
    /// before format version 6 made the member geometry mandatory.
    NoAffineShapes,
    /// The file carries no `cluster_patches/` section, so there is no refine
    /// radius the stored shapes are expressed against.
    NoClusterPatches,
    /// The `cluster_patches/` refine options record neither `patch_size` nor
    /// the legacy `radius`, so the shapes have no scale.
    NoRefineRadius,
}

impl std::fmt::Display for ClusterRadiiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoClusters => write!(
                f,
                "this .matches file carries no clusters/ section: a cluster radius needs \
                 the cluster backbone, not pairwise matches"
            ),
            Self::NoAffineShapes => write!(
                f,
                "this .matches file's clusters carry no member affine shapes, so there is \
                 no extent to read a radius off"
            ),
            Self::NoClusterPatches => write!(
                f,
                "this .matches file carries no cluster_patches/ section: the refine radius \
                 the stored shapes are expressed against lives there"
            ),
            Self::NoRefineRadius => write!(
                f,
                "this .matches file's cluster_patches/ refine options record neither \
                 patch_size nor radius, so the stored shapes have no scale"
            ),
        }
    }
}

impl std::error::Error for ClusterRadiiError {}

/// Each member's feature radius: half the refine radius times the sum of its
/// stored affine's two column norms, which is the refine radius times their
/// mean.
///
/// `member_affine_shapes` is flat, four values per member, each 2x2 in
/// row-major order, at the `f32` width a `.matches` backbone stores them at. A
/// caller holding them widened narrows them back at its own boundary, which is
/// exact for every value that came off such a file.
pub fn member_radii(member_affine_shapes: &[f32], refine_radius: f32) -> Vec<f32> {
    let half = 0.5 * refine_radius;
    member_affine_shapes
        .par_chunks_exact(4)
        .map(|a| {
            // Column norms of the row-major 2x2 [a0 a1; a2 a3].
            let (a0, a1, a2, a3) = (a[0], a[1], a[2], a[3]);
            let c0 = (a0 * a0 + a2 * a2).sqrt();
            let c1 = (a1 * a1 + a3 * a3).sqrt();
            half * (c0 + c1)
        })
        .collect()
}

/// Each cluster's radius: the widest of its own members' ([`member_radii`]), so
/// "radius" means what it meant when the cluster's admission was drawn.
///
/// `cluster_starts` is the CSR index over the members (`n_clusters + 1`
/// offsets); cluster `c` owns members `cluster_starts[c]..cluster_starts[c+1]`.
/// A cluster with no members reads `0.0`. Offsets are clamped to the member
/// count, and members no cluster's range covers contribute to nothing, so a
/// caller reading a well-formed backbone gets every member counted exactly
/// once.
///
/// ```
/// use sfmtool_core::analysis::cluster_radii::cluster_radii;
///
/// // Two clusters over three members, each shape a scaled identity.
/// let starts = [0u32, 2, 3];
/// let shapes: [f32; 12] = [
///     1.0, 0.0, 0.0, 1.0, // member 0: radius 6 * 1.0
///     0.5, 0.0, 0.0, 0.5, // member 1: radius 6 * 0.5
///     2.0, 0.0, 0.0, 2.0, // member 2: radius 6 * 2.0
/// ];
/// assert_eq!(cluster_radii(&starts, &shapes, 6.0), vec![6.0, 12.0]);
/// ```
pub fn cluster_radii(
    cluster_starts: &[u32],
    member_affine_shapes: &[f32],
    refine_radius: f32,
) -> Vec<f32> {
    let row_radius = member_radii(member_affine_shapes, refine_radius);
    let n_member = row_radius.len();
    let n_cl = cluster_starts.len().saturating_sub(1);
    (0..n_cl)
        .into_par_iter()
        .map(|c| {
            let hi = (cluster_starts[c + 1] as usize).min(n_member);
            // `min(hi)` rather than an assert: a non-monotonic offset pair is an
            // empty cluster here, the same reading the CSR walk elsewhere gives.
            let lo = (cluster_starts[c] as usize).min(hi);
            let mut widest = 0.0f32;
            for &r in &row_radius[lo..hi] {
                if r > widest {
                    widest = r;
                }
            }
            widest
        })
        .collect()
}

/// Ids of the `n` coarsest clusters: radius descending, id ascending on ties,
/// returned sorted ASCENDING.
///
/// The ascending return is what composes as a restriction, while the descending
/// ordering behind it decides membership. Fewer than `n` clusters returns all
/// of them.
///
/// ```
/// use sfmtool_core::analysis::cluster_radii::coarsest_cluster_ids;
///
/// let starts = [0u32, 1, 2, 3];
/// // Clusters 0 and 2 tie at the same radius; the smaller id is taken first.
/// let shapes: [f32; 12] = [
///     2.0, 0.0, 0.0, 2.0, //
///     1.0, 0.0, 0.0, 1.0, //
///     2.0, 0.0, 0.0, 2.0, //
/// ];
/// assert_eq!(coarsest_cluster_ids(&starts, &shapes, 6.0, 2), vec![0, 2]);
/// ```
pub fn coarsest_cluster_ids(
    cluster_starts: &[u32],
    member_affine_shapes: &[f32],
    refine_radius: f32,
    n: usize,
) -> Vec<u32> {
    coarsest_by_radius(
        &cluster_radii(cluster_starts, member_affine_shapes, refine_radius),
        n,
    )
}

/// The coarsest-N cut over radii already in hand; see [`coarsest_cluster_ids`].
///
/// The comparator is a total order over the whole `f32` range so the cut is a
/// function of the radii alone: descending, `NaN` last (a shape carrying `NaN`
/// has no extent and must not displace a cluster that does), ties in ascending
/// id order.
fn coarsest_by_radius(radius: &[f32], n: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..radius.len() as u32).collect();
    let n = n.min(order.len());
    if n == 0 {
        return Vec::new();
    }
    // The id tiebreak makes this a STRICT total order -- no two distinct ids
    // compare equal -- so the set of the n first elements under it is unique
    // and an unstable selection returns exactly the cut a full sort would.
    let cmp = |a: &u32, b: &u32| {
        let (ra, rb) = (radius[*a as usize], radius[*b as usize]);
        match (ra.is_nan(), rb.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => rb.partial_cmp(&ra).expect("neither side is NaN"),
        }
        .then(a.cmp(b))
    };
    if n < order.len() {
        order.select_nth_unstable_by(n - 1, cmp);
        order.truncate(n);
    }
    order.sort_unstable();
    order
}

/// Every cluster's radius, read off a parsed `.matches` file in one call.
///
/// The file states everything the reading needs: the `clusters/` backbone
/// supplies the CSR index and the member affine shapes, and the
/// `cluster_patches/` section supplies the refine radius those shapes are
/// expressed against. It is [`cluster_radii`] with the reading done for the
/// caller, and produces identical bits.
///
/// ```no_run
/// use sfmtool_core::analysis::cluster_radii::cluster_radii_from_matches;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let matches = matches_format::read_matches("clusters.matches".as_ref())?;
/// let radius = cluster_radii_from_matches(&matches)?;
/// println!("{} clusters, widest {:?}", radius.len(), radius.iter().cloned().fold(0.0, f32::max));
/// # Ok(())
/// # }
/// ```
pub fn cluster_radii_from_matches(matches: &MatchesData) -> Result<Vec<f32>, ClusterRadiiError> {
    let input = RadiiInput::read(matches)?;
    Ok(cluster_radii(
        &input.cluster_starts,
        &input.member_affine_shapes,
        input.refine_radius,
    ))
}

/// The coarsest-N cut over a parsed `.matches` file; see
/// [`coarsest_cluster_ids`] and [`cluster_radii_from_matches`].
pub fn coarsest_cluster_ids_from_matches(
    matches: &MatchesData,
    n: usize,
) -> Result<Vec<u32>, ClusterRadiiError> {
    Ok(coarsest_by_radius(&cluster_radii_from_matches(matches)?, n))
}

/// The three things a radius reading takes off a `.matches` file, borrowed
/// where the file's own storage is already contiguous.
struct RadiiInput<'a> {
    /// The cluster backbone's CSR index.
    cluster_starts: Cow<'a, [u32]>,
    /// The member affine shapes, flat, four per member in row-major order.
    member_affine_shapes: Cow<'a, [f32]>,
    /// The refine radius the `cluster_patches/` section records, at the width
    /// the reading runs at.
    refine_radius: f32,
}

impl<'a> RadiiInput<'a> {
    fn read(matches: &'a MatchesData) -> Result<Self, ClusterRadiiError> {
        let clusters = matches
            .clusters
            .as_ref()
            .ok_or(ClusterRadiiError::NoClusters)?;
        let shapes = clusters
            .member_affine_shapes
            .as_ref()
            .ok_or(ClusterRadiiError::NoAffineShapes)?;
        let refine_radius = matches
            .cluster_patches
            .as_ref()
            .ok_or(ClusterRadiiError::NoClusterPatches)?
            .refine_radius()
            .ok_or(ClusterRadiiError::NoRefineRadius)?;
        // The file records the refine radius as a JSON number, so it arrives
        // `f64` wide however it was written; the reading is `f32`.
        let refine_radius = refine_radius as f32;
        Ok(Self {
            cluster_starts: contiguous(&clusters.cluster_starts),
            member_affine_shapes: match shapes.as_slice() {
                Some(flat) => Cow::Borrowed(flat),
                None => Cow::Owned(shapes.iter().copied().collect()),
            },
            refine_radius,
        })
    }
}

#[cfg(test)]
mod tests;
