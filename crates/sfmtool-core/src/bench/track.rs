// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The [`EditableTrack`] value: the observations being considered for one
//! track, what has been measured about each, what the person has decided about
//! each, and which of the two representations the track is currently in.
//!
//! `specs/core/bench/editable-track.md` is the design. Everything here is a
//! plain value: `Clone`, no interior mutability, no handle to any device, cache
//! or window. The steps that produce a new one live in
//! [`steps`](super::steps) and [`commit`](mod@super::commit).

use nalgebra::Point3;
use ndarray::Array3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{ClusterRefineParams, MemberStatus};
use crate::patch::reference_view::ReferenceStanding;
use crate::patch::self_similarity::{SelfSimilarityEllipse, SelfSimilarityEllipseUnits};
use crate::patch::view_selection::ViewSelectParams;

/// Where an observation came from.
///
/// Shown to the person, and read by exactly one step: a commit deletes the
/// points that [`Provenance::Point`] observations were pulled from. No kernel
/// reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// The committed track the item was put on the bench from.
    Origin,
    /// A `.sift` feature the person or a caller named directly: the detected
    /// keypoint a cluster was started on, or one added to a track by index.
    Descriptor {
        /// The feature's index in its image's `.sift` file.
        feature: u32,
    },
    /// An image a descriptor search found the patch in.
    ///
    /// Separate from [`Provenance::Descriptor`] because the two name different
    /// things. A descriptor provenance names **one detected feature**, which
    /// the observation sits exactly on. A search's observation sits wherever
    /// the image's affine warp puts the pixel that was searched from, which is
    /// in general no feature at all; what stands behind it is the number of
    /// correspondences that agreed on that warp, and that is what is worth
    /// showing beside the row.
    Search {
        /// Correspondences that voted for the warp this observation was placed
        /// by. It ranks the search's candidates against one another, and it is
        /// an admission the photometry then judges.
        inliers: u32,
    },
    /// An image the view sweep proposed.
    Sweep,
    /// A pixel the person pointed at.
    Pixel,
    /// An observation of another point, pulled in. A commit that keeps it
    /// absorbs that point.
    Point {
        /// The point it was pulled from, by the index it had when it was
        /// pulled.
        point: u32,
    },
}

/// Whether the track keeps one observation.
///
/// A measurement is a report and never a decision: the thresholds propose a
/// verdict and the steps apply the proposal to the observations nobody has
/// ruled on by hand, and [`Observation::pinned`] says when one was set by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The observation belongs to the track. The kernels run over these, and a
    /// commit writes exactly these.
    In,
    /// The track does not keep the observation: it was added and not yet
    /// measured, the thresholds did not take it, or the person refused it,
    /// which a pin says. It stays in the list so a search does not propose it
    /// again and so the refusal is visible.
    Out,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::In => write!(f, "in"),
            Verdict::Out => write!(f, "out"),
        }
    }
}

/// What the cluster stage has measured about one observation.
///
/// The seed is what the observation was put on the track with and is always
/// present; everything below it is the refinement's answer and is `None` until
/// an evaluation has run.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterMeasurement {
    /// Where the seed put the observation, in that image's pixels.
    pub seed_position: [f64; 2],
    /// The seed's affine shape: the detector's canonical **keypoint frame**
    /// mapped onto this image's pixels, the same `S` the `.matches`
    /// cluster-patches section stores.
    ///
    /// A shape is a scale and not a size on its own. The patch is the square
    /// `[-r, r]^2` of keypoint-frame units, where `r` is
    /// [`ClusterPayload::radius`], so the sighting's pixel half-width along a
    /// column is `r * ||column||` and a shape read without that radius says
    /// nothing about how large the patch is.
    pub seed_shape: [[f64; 2]; 2],
    /// Where the refinement put the observation, in that image's pixels.
    pub position: Option<[f64; 2]>,
    /// The refined absolute affine shape, in the same convention as
    /// [`Self::seed_shape`]: keypoint frame to pixels, over the same
    /// `[-r, r]^2` square.
    pub shape: Option<[[f64; 2]; 2]>,
    /// The windowed ZNCC the refinement achieved against the template.
    pub zncc: Option<f64>,
    /// The **middle ZNCC** beside [`Self::zncc`]: the same samples at the
    /// refinement's final map, correlated against the template over only the
    /// middle square of the grid (the rows and columns `R/4 .. R - R/4`, the
    /// middle `12 × 12` of a `24 × 24` grid). A high `zncc` that the middle does
    /// not share is carried by the parts of the patch away from its centre.
    /// `None` wherever `zncc` is, and where the template's middle is flat.
    pub zncc_middle: Option<f64>,
    /// The **ZNCC grid** beside [`Self::zncc`]: the same samples against the
    /// same template, read over each cell of a three-by-three split of the
    /// grid (rows and columns cut at `R/3` and `R - R/3`, `8 × 8` cells of a
    /// `24 × 24` grid) with every pixel weighted equally, `grid[row][col]`
    /// from the top-left cell. It says where in the grid an agreement or a
    /// disagreement is. `None` wherever `zncc` is; a single cell is
    /// `NaN` where the template is flat over it.
    pub zncc_grid: Option<[[f64; 3]; 3]>,
    /// How far the refinement moved off the seed, in **patch-grid px**: the
    /// drift in the seed's own keypoint frame, scaled to the template's grid
    /// (`resolution` samples across `2 · radius` keypoint-frame units), so it is
    /// measured in the unit [`Thresholds::max_shift_px`] and the self-similarity
    /// radius are.
    pub shift_px: Option<f64>,
    /// The ZNCC self-similarity radius of the observation's own tile, in
    /// template-grid px: the semi-major axis of the ellipse of the shifts at which
    /// the `R×R` tile still matches itself within the tolerance a true match
    /// between two views allows, `0 ..= r`, with `r` read as "`r` or more",
    /// read the overlap way with no pixels from outside the tile
    /// (see `specs/core/patch/zncc-self-similarity-radius.md`). `None` wherever
    /// the tile could not be sampled.
    pub zncc_self_similarity_radius: Option<f64>,
    /// The ZNCC self-similarity radius of the middle square of the same tile,
    /// the rows and columns `R/4 .. R - R/4`. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_radius_middle: Option<f64>,
    /// The ZNCC self-similarity radius of each cell of the ZNCC grid's
    /// three-by-three split of the same tile, `grid[row][col]` from the
    /// top-left cell. `None` wherever `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_radius_grid: Option<[[f64; 3]; 3]>,
    /// The ellipse of the whole tile's self-similarity region
    /// ([`SelfSimilarityEllipseUnits`]): in grid px, and in source-image px
    /// through the seed shape's map from the grid to the image. A cluster
    /// member has no patch, so `patch` is always `None`. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_ellipse: Option<SelfSimilarityEllipseUnits>,
    /// The same for the middle square. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_ellipse_middle: Option<SelfSimilarityEllipseUnits>,
    /// The ellipse of each cell of [`Self::zncc_self_similarity_radius_grid`],
    /// in grid px (`x` column-right, `y` row-down). `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_ellipse_grid: Option<[[SelfSimilarityEllipse; 3]; 3]>,
    /// The whole tile's ZNCC against itself at every shift of the `(2r + 1)²`
    /// square, row-major from `(dx, dy) = (-r, -r)`: `1` at the centre, all
    /// `NaN` when the tile has no texture. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_surface: Option<Vec<f64>>,
    /// The tolerance the tile was judged by, `ε + mean_c (n / s_c)²`: a shift
    /// whose ZNCC deficit is at or under it is indistinguishable from the
    /// true position, so `1 - tolerance` is the level of
    /// [`Self::zncc_self_similarity_surface`] the ellipse is read at. `None`
    /// wherever `zncc_self_similarity_radius` is, and where the tile has no
    /// texture.
    pub zncc_self_similarity_tolerance: Option<f64>,
    /// The refinement's own verdict on the observation, in the `member_status`
    /// legend.
    pub status: Option<MemberStatus>,
}

impl ClusterMeasurement {
    /// A measurement that is a seed and nothing else.
    pub fn from_seed(position: [f64; 2], shape: [[f64; 2]; 2]) -> Self {
        Self {
            seed_position: position,
            seed_shape: shape,
            position: None,
            shape: None,
            zncc: None,
            zncc_middle: None,
            zncc_grid: None,
            shift_px: None,
            zncc_self_similarity_radius: None,
            zncc_self_similarity_radius_middle: None,
            zncc_self_similarity_radius_grid: None,
            zncc_self_similarity_ellipse: None,
            zncc_self_similarity_ellipse_middle: None,
            zncc_self_similarity_ellipse_grid: None,
            zncc_self_similarity_surface: None,
            zncc_self_similarity_tolerance: None,
            status: None,
        }
    }

    /// Where the observation is: the refined position when there is one, and
    /// the seed otherwise.
    pub fn best_position(&self) -> [f64; 2] {
        self.position.unwrap_or(self.seed_position)
    }
}

/// Why an observation carries no measurement at the track stage.
///
/// An evaluation drops nothing: it turns off the localizer's own gates, so
/// every observation it can read comes back with a number. What is left is the observation it
/// cannot read at all, and this says which of those it was, in one short sentence, so a row without
/// a ZNCC never reads as an unexplained refusal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unmeasured {
    /// Nothing says where the observation sits in its photograph: it carries
    /// neither a keypoint nor a cluster seed.
    NoSeed,
    /// Where it sits is off its photograph's sensor, so there is no tile to cut
    /// around it.
    OffSensor,
    /// The track's point does not project into this view, so there is no anchor
    /// to render a tile about.
    NoProjection,
    /// The view's ray runs near-parallel to the patch plane, where nothing pins
    /// an in-plane position.
    Grazing {
        /// `|d_hat . n_hat|`, the cosine the grazing cutoff judges.
        cosine: f64,
    },
    /// Its seed sits further from the point's projection than the reading is
    /// willing to widen its window for.
    ///
    /// The window the localizer searches is anchored at the projection and has
    /// to reach the seed, so a seed a thousand px out asks for a tile a
    /// thousand px wide -- and the tile's cost is that number **squared**, per
    /// view. Past the bound the answer is that the sighting is somewhere else
    /// entirely, which is a thing to say about the row rather than a thing to
    /// allocate for.
    SeedTooFar {
        /// How far the seed sits from the projection, in patch-grid px.
        offset_px: f64,
        /// The bound it passed, in the same units.
        bound_px: f64,
    },
    /// Its round had no reference render to align it to: the round held no
    /// other observation that could be read, or the reference observation's
    /// own tile left its photograph or carried no texture.
    NoReference,
    /// The correlation could not be scored: the tile around the observation
    /// runs off the photograph, or no channel of it carries texture.
    Unscorable,
    /// The track has no patch bitmap to score the observation against, as
    /// before its first render, or one on another grid than the tile's.
    NoBitmap,
    /// The track holds a bitmap that the next render replaces
    /// ([`TrackPayload::bitmap_pending`]), so no row is scored against it
    /// until that render has run.
    BitmapPending,
}

impl std::fmt::Display for Unmeasured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unmeasured::NoSeed => write!(f, "nothing says where it sits"),
            Unmeasured::OffSensor => write!(f, "it sits off the photograph"),
            Unmeasured::NoProjection => write!(f, "the point misses this view"),
            Unmeasured::Grazing { cosine } => write!(f, "its ray grazes the patch ({cosine:.2})"),
            Unmeasured::SeedTooFar {
                offset_px,
                bound_px,
            } => write!(
                f,
                "its seed sits {offset_px:.0} px from the projection, beyond the \
                 {bound_px:.0} px bound"
            ),
            Unmeasured::NoReference => write!(f, "there is no reference render to align it to"),
            Unmeasured::Unscorable => write!(f, "its tile could not be scored"),
            Unmeasured::NoBitmap => write!(f, "there is no bitmap to score it against"),
            Unmeasured::BitmapPending => write!(
                f,
                "the bitmap is to be rendered again before the row is scored"
            ),
        }
    }
}

/// What the track stage has measured about one observation.
///
/// A track put on the bench from a committed point arrives with
/// [`Self::keypoint`] and [`Self::blur_matched_zncc`] read off the stored
/// columns; the rest is what an evaluation computes.
///
/// The row's **score** is its tile's ZNCC with the track's stored patch
/// bitmap, read twice: plain ([`Self::plain_zncc`], with
/// [`Self::plain_zncc_middle`] and [`Self::plain_zncc_grid`] beside it), and
/// blur-matched ([`Self::blur_matched_zncc`], with
/// [`Self::blur_matched_zncc_middle`] and [`Self::blur_matched_zncc_grid`]),
/// which the bars judge.
/// The bitmap is the render of the track's reference observation, which is
/// also the template the localizer aligns every other row to, so the score
/// and the alignment read one comparison. The localizer's own reading of the
/// row is the distance to its correlation peak, [`Self::seed_shift_px`].
///
/// The two distances are different questions, and both are here because a
/// person reading a row has to tell them apart: [`Self::seed_shift_px`] is
/// about the **observation** -- how far the correlation peak sits from where
/// the sighting is -- and [`Self::projection_offset_px`] is about the
/// **point** -- how far the sighting sits from where the position puts it. A
/// mis-triangulated point gives every row a large offset while the shifts stay
/// at zero.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackMeasurement {
    /// Where the observation sits, in that image's pixels: where the localizer
    /// put it, or, until a fit moves it, the pixel it was added or placed at
    /// by hand. This is the pixel a commit writes.
    ///
    /// An evaluation never writes it: it reads the track as it stands, and this
    /// pixel is the thing it reads.
    pub keypoint: Option<[f32; 2]>,
    /// The **score**: the observation's tile, rendered at its keypoint, read
    /// against the track's patch bitmap as stored ([`TrackPayload::bitmap`]),
    /// plain, windowed, over the samples with data in both
    /// ([`BitmapScorer`](crate::patch::stored_bitmap::BitmapScorer)). `1` for
    /// the observation the bitmap is the render of
    /// ([`TrackPayload::reference`]), which is not computed. No bar judges
    /// it: the bars read [`Self::blur_matched_zncc`], and this one is kept
    /// beside it to show what blur matching changed. `None` where the track
    /// has no bitmap, the tile could not be rendered, or the pair could not be
    /// read, and then [`Self::reason`] says which; also `None` on a row read
    /// back from a committed point, which carries only the blur-matched score.
    pub plain_zncc: Option<f64>,
    /// The **middle ZNCC** beside [`Self::plain_zncc`]: the same tile against the
    /// same bitmap, read over only the middle square of the tile (the rows and
    /// columns `R/4 .. R - R/4`, the middle `12 × 12` of a `24 × 24` tile). A
    /// high `plain_zncc` that the middle does not share is carried by the parts of
    /// the tile away from the keypoint: a small near object in front of a
    /// textured background, a pixel at a depth edge, a texture that repeats
    /// along the epipolar line. `1` on the bitmap's own row. `None` wherever
    /// `plain_zncc` is and where the middle cannot be read.
    pub plain_zncc_middle: Option<f64>,
    /// The **ZNCC grid** beside [`Self::plain_zncc`]: the same tile against the same
    /// bitmap, read over each cell of a three-by-three split of the tile (rows
    /// and columns cut at `R/3` and `R - R/3`, `8 × 8` cells of a `24 × 24`
    /// tile) with every sample weighted equally, `grid[row][col]` from the
    /// top-left cell. It says where in the tile an agreement or a disagreement
    /// is. Every cell `1` on the bitmap's own row. `None` wherever `plain_zncc` is;
    /// a single cell is `NaN` where it cannot be read.
    pub plain_zncc_grid: Option<[[f64; 3]; 3]>,
    /// How far the localizer's correlation peak sits from the observation's
    /// own keypoint, in **patch-grid px** on the patch's plane: the
    /// observation's own evidence, and what [`Thresholds::max_shift_px`] paints
    /// on. The peak is the shift, within the search radius of the keypoint, at
    /// which the observation's tile best matches the render of the track's
    /// reference observation, so `0` on the reference's own row. In the unit
    /// of the self-similarity radius, so the two compare directly: a shift
    /// inside the radius is within what the patch cannot tell apart.
    ///
    /// Present is also the statement that the localizer read the row: `None`
    /// where it could not ([`Self::reason`] says why), and such a row is
    /// unmeasured to the bars ([`bar_checks`](super::steps::bar_checks)) even
    /// where it has a score against the bitmap.
    pub seed_shift_px: Option<f64>,
    /// How far the observation's keypoint sits from the point's projection, in
    /// source-image px: the number that says how far the **point** is off,
    /// rather than the sighting.
    pub projection_offset_px: Option<f64>,
    /// The reprojection error against the triangulated position, in px.
    pub reprojection_error: Option<f64>,
    /// The angle between this observation's own ray and the direction from its
    /// camera to the triangulated position, in degrees: the reprojection
    /// residual stated as an angle, which is what makes it comparable across
    /// lenses and depths. The same number Track View's
    /// *Angle* column shows for a committed track.
    pub ray_angle_deg: Option<f64>,
    /// The ZNCC self-similarity radius of the observation's own tile, in
    /// grid px: the semi-major axis of the ellipse of the shifts at which
    /// the `R×R` tile still matches itself within the tolerance a true match
    /// between two views allows, `0 ..= r`, with `r` read as "`r` or more",
    /// read the overlap way with no pixels from outside the tile
    /// (see `specs/core/patch/zncc-self-similarity-radius.md`). `None` wherever
    /// the tile could not be sampled.
    pub zncc_self_similarity_radius: Option<f64>,
    /// The ZNCC self-similarity radius of the middle square of the same tile,
    /// the rows and columns `R/4 .. R - R/4`. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_radius_middle: Option<f64>,
    /// The ZNCC self-similarity radius of each cell of the ZNCC grid's
    /// three-by-three split of the same tile, `grid[row][col]` from the
    /// top-left cell. `None` wherever `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_radius_grid: Option<[[f64; 3]; 3]>,
    /// The ellipse of the whole tile's self-similarity region
    /// ([`SelfSimilarityEllipseUnits`]): in grid px, in source-image px
    /// through the Jacobian of the tile's warp at its centre, and along the
    /// patch's `u` and `v` axes in the scene's world-space unit (degrees for
    /// a patch at infinity), all read through the keypoint-anchored placement
    /// the tile was rendered through. `None` wherever
    /// `zncc_self_similarity_radius` is; its `image_px` is `None` where the
    /// tile's centre does not project.
    pub zncc_self_similarity_ellipse: Option<SelfSimilarityEllipseUnits>,
    /// The same for the middle square. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_ellipse_middle: Option<SelfSimilarityEllipseUnits>,
    /// The ellipse of each cell of [`Self::zncc_self_similarity_radius_grid`],
    /// in grid px (`x` column-right, `y` row-down). `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_ellipse_grid: Option<[[SelfSimilarityEllipse; 3]; 3]>,
    /// The whole tile's ZNCC against itself at every shift of the `(2r + 1)²`
    /// square, row-major from `(dx, dy) = (-r, -r)`: `1` at the centre, all
    /// `NaN` when the tile has no texture. `None` wherever
    /// `zncc_self_similarity_radius` is.
    pub zncc_self_similarity_surface: Option<Vec<f64>>,
    /// The tolerance the tile was judged by, `ε + mean_c (n / s_c)²`: a shift
    /// whose ZNCC deficit is at or under it is indistinguishable from the
    /// true position, so `1 - tolerance` is the level of
    /// [`Self::zncc_self_similarity_surface`] the ellipse is read at. `None`
    /// wherever `zncc_self_similarity_radius` is, and where the tile has no
    /// texture.
    pub zncc_self_similarity_tolerance: Option<f64>,
    /// The **viewing angle** at the keypoint, in degrees: the angle between
    /// the patch's normal and the direction from the keypoint-anchored patch's
    /// centre to the camera, `0` facing the patch and `90` edge on
    /// ([`viewing_angle`](crate::patch::normal_refine::viewing_angle)).
    /// `None` wherever the tile could not be rendered.
    pub viewing_angle_deg: Option<f64>,
    /// The **tilt direction**, in degrees: the angle in the patch's plane,
    /// from its `u` axis towards its `v` axis, of the ray from the camera,
    /// the direction the view foreshortens the patch along. `None` where
    /// [`Self::viewing_angle_deg`] is, and for a view within
    /// [`MIN_TILT_ANGLE_DEG`](crate::patch::normal_refine::MIN_TILT_ANGLE_DEG)
    /// of facing the patch.
    pub tilt_direction_deg: Option<f64>,
    /// The **coverage** of the observation's own `R×R` tile: the share of its
    /// samples the warp places on the photograph, `0 ..= 1`. `None` wherever
    /// the tile could not be rendered.
    pub coverage: Option<f64>,
    /// The **clipped share**: the share of the photograph's own pixels inside
    /// the tile's outline that are `0` or `255` in any colour channel
    /// ([`clipped_share`](crate::patch::reference_view::clipped_share)).
    /// `None` wherever the tile could not be rendered or no sample of it lands
    /// on the photograph.
    pub clipped_share: Option<f64>,
    /// The **pair ZNCC**: the median of this observation's pairwise ZNCCs
    /// with the track's other `in` observations, from member coherence's
    /// matrix ([`member_zncc_matrix`](crate::patch::member_coherence::member_zncc_matrix)),
    /// the agreement the reference-view rule's margin is read on. `None` for
    /// an `out` observation, and where the matrix has no reading for it.
    pub pair_zncc: Option<f64>,
    /// The **pair ZNCC grid**: per cell of the ZNCC grid's three-by-three
    /// split, `grid[row][col]` from the top-left cell, the median over the
    /// track's other `in` observations of the pair's ZNCC in that cell
    /// ([`CellAgreement::pair_zncc_grid`](crate::patch::reference_view::CellAgreement::pair_zncc_grid)).
    /// A cell is `NaN` where no other observation could be correlated with
    /// this one there. `None` for an `out` observation.
    pub pair_zncc_grid: Option<[[f64; 3]; 3]>,
    /// The **cell deficit**: the largest amount by which
    /// [`Self::pair_zncc_grid`] falls below the track's typical agreement in
    /// the same cell, over the cells where that typical agreement is at least
    /// [`REFERENCE_MIN_JUDGED_CELL_ZNCC`](crate::patch::reference_view::REFERENCE_MIN_JUDGED_CELL_ZNCC);
    /// `0` where no cell is. `None` for an `out` observation and where it has
    /// no reading in any judged cell.
    pub cell_deficit: Option<f64>,
    /// What the reference-view rule decided about this observation: picked as
    /// the **reference view**, or the test that turned it away, and which
    /// tests the rule dropped for the track
    /// ([`choose_reference_view`](crate::patch::reference_view::choose_reference_view)).
    /// `None` for an `out` observation, which the rule does not consider.
    pub reference_view: Option<ReferenceStanding>,
    /// The **blur-matched score**: [`Self::plain_zncc`] after the bitmap, and only
    /// the bitmap, is blurred to this observation's sharpness where it is
    /// sharper along every direction by at least the ratio of 1.25; the plain
    /// score where it is not. `1` on the bitmap's own row. This is the reading
    /// [`Thresholds::min_zncc`] judges and the painting ranks rows by, the one
    /// Track at Pixel's median gate reads, and the one a commit writes as the
    /// observation's `observation_confidence`. A track put on the bench from a
    /// committed point carries that column read back here until its first
    /// evaluation. Otherwise `None` wherever [`Self::plain_zncc`] is.
    pub blur_matched_zncc: Option<f64>,
    /// [`Self::plain_zncc_middle`] read against the same blurred bitmap as
    /// [`Self::blur_matched_zncc`]; the plain middle score where the pair is
    /// read plain. The reading [`Thresholds::min_zncc_middle`] judges. `None`
    /// wherever [`Self::plain_zncc_middle`] is.
    pub blur_matched_zncc_middle: Option<f64>,
    /// [`Self::plain_zncc_grid`] read against the same blurred bitmap as
    /// [`Self::blur_matched_zncc`]; the plain grid where the pair is read
    /// plain. `None` wherever [`Self::plain_zncc`] is.
    pub blur_matched_zncc_grid: Option<[[f64; 3]; 3]>,
    /// The width of the round blur, in grid px, the bitmap was blurred by for
    /// [`Self::blur_matched_zncc`]; `0` where the pair was read plain.
    /// `None` wherever [`Self::plain_zncc`] is.
    pub bitmap_blur_sigma: Option<f64>,
    /// Whether this observation's tile is sharper than the bitmap along every
    /// direction (its self-similarity semi-major axis shorter than the
    /// bitmap's semi-minor axis), so a candidate to replace the reference.
    /// Such a pair is read plain. `None` wherever [`Self::plain_zncc`] is, and for
    /// the reference observation itself.
    pub sharper_than_bitmap: Option<bool>,
    /// How far the last fit's correlation peak sat from this sighting's seed,
    /// when that was further than [`Thresholds::max_shift_px`] and the seed was
    /// therefore kept, in patch-grid px.
    ///
    /// **Present is the whole statement**: this sighting did not move, and the
    /// number says how far the kernel wanted to take it. The fit's kernels run
    /// gate-free so that nothing is dropped, and this is the one bound on what
    /// they may *write* -- a correlation that walked a sighting onto a similar
    /// detail elsewhere in the photograph would otherwise feed the
    /// re-triangulation a place the person never pointed at. The row still casts
    /// its ray, from the seed, and the reading that follows scores it there like
    /// any other.
    ///
    /// Only a fit sets and clears it; an evaluation leaves it alone, because the
    /// statement is about what a fit did rather than about what the photographs
    /// show.
    pub walked_px: Option<f64>,
    /// Where the last fit's kernels would have put this sighting, in source-image
    /// px, when [`Self::walked_px`] says the fit refused it: the refined keypoint
    /// the bar turned away.
    ///
    /// Kept so the refusal can be overruled. A person who looks at the two
    /// places and finds the walked one right accepts it by putting the sighting
    /// there with [`sight_observation`](super::steps::sight_observation), which
    /// pins it and drops this measurement like any hand placement. Set and
    /// cleared with [`Self::walked_px`], and only by a fit.
    pub walked_to: Option<[f64; 2]>,
    /// The plain score against the stored bitmap of the tile rendered at
    /// [`Self::walked_to`], read the way [`Self::plain_zncc`] is read at the
    /// keypoint, kept beside [`Self::walked_blur_matched_zncc`], which is the
    /// one judged. Every
    /// reading that scores the rows against the bitmap reads it where
    /// [`Self::walked_to`] is set, so the two scores are always against the
    /// same bitmap; `None` where `walked_to` is not set, where the track has
    /// no bitmap to score against, and where the tile could not be read.
    pub walked_plain_zncc: Option<f64>,
    /// The middle ZNCC beside [`Self::walked_plain_zncc`], read the way
    /// [`Self::plain_zncc_middle`] is.
    pub walked_plain_zncc_middle: Option<f64>,
    /// The ZNCC grid beside [`Self::walked_plain_zncc`], read the way
    /// [`Self::plain_zncc_grid`] is.
    pub walked_plain_zncc_grid: Option<[[f64; 3]; 3]>,
    /// The blur-matched score of the tile at [`Self::walked_to`], read the way
    /// [`Self::blur_matched_zncc`] is: the bitmap blurred to that tile's
    /// sharpness where the ratio selects it. The agreement the walk would have
    /// bought, judged by the bars as [`Self::blur_matched_zncc`] is, to set
    /// beside it, which is read with the sighting kept at its seed. `None`
    /// wherever [`Self::walked_plain_zncc`] is.
    pub walked_blur_matched_zncc: Option<f64>,
    /// The middle score beside [`Self::walked_blur_matched_zncc`], read the
    /// way [`Self::blur_matched_zncc_middle`] is.
    pub walked_blur_matched_zncc_middle: Option<f64>,
    /// The grid beside [`Self::walked_blur_matched_zncc`], read the way
    /// [`Self::blur_matched_zncc_grid`] is.
    pub walked_blur_matched_zncc_grid: Option<[[f64; 3]; 3]>,
    /// Why a reading is missing, when an evaluation has read the row: where
    /// the localizer could not read the observation ([`Self::seed_shift_px`]
    /// is `None`), which of its refusals it was, whether or not the row has a
    /// score against the bitmap; otherwise why there is no score
    /// ([`Self::plain_zncc`]), and `None` where there is one. So a row without a
    /// score never reads as an unexplained blank.
    pub reason: Option<Unmeasured>,
}

impl TrackMeasurement {
    /// Clear the scores of the tile at [`Self::walked_to`], plain and
    /// blur-matched.
    pub(crate) fn clear_walked_scores(&mut self) {
        self.walked_plain_zncc = None;
        self.walked_plain_zncc_middle = None;
        self.walked_plain_zncc_grid = None;
        self.walked_blur_matched_zncc = None;
        self.walked_blur_matched_zncc_middle = None;
        self.walked_blur_matched_zncc_grid = None;
    }
}

/// One observation of an editable track: an image, a place in it, what has been
/// measured about it at each stage, and the verdict.
///
/// Observations are appended and no step on the track renumbers them, so an
/// index into [`EditableTrack::observations`] is stable across every step and
/// an evaluation that finishes late still lands on the observation it
/// measured. Deleting an image from the reconstruction does renumber them
/// ([`EditableTrack::delete_image`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// The image, as an index into the node's image table.
    pub image: u32,
    /// Where it came from.
    pub provenance: Provenance,
    /// The person's decision.
    pub verdict: Verdict,
    /// Whether the verdict was set by hand. A pinned verdict is left alone by
    /// [`apply_thresholds`](super::steps::apply_thresholds) and by an
    /// evaluation's repaint; an unpinned one is whatever the bars propose.
    pub pinned: bool,
    /// The cluster stage's slot, filled for an observation the track carried
    /// through that stage.
    pub cluster: Option<ClusterMeasurement>,
    /// The track stage's slot, filled for an observation the track carried
    /// through that stage.
    pub track: Option<TrackMeasurement>,
}

impl Observation {
    /// An unpinned `out` observation in `image`, seeded for the cluster stage
    /// at `position` with `shape`, measured at neither stage. Its first
    /// evaluation takes it in when it clears the thresholds.
    pub fn seeded(
        image: u32,
        provenance: Provenance,
        position: [f64; 2],
        shape: [[f64; 2]; 2],
    ) -> Self {
        Self {
            image,
            provenance,
            verdict: Verdict::Out,
            pinned: false,
            cluster: Some(ClusterMeasurement::from_seed(position, shape)),
            track: None,
        }
    }

    /// Where the observation currently sits in its photograph: the keypoint a
    /// track-stage measurement carries, else the cluster stage's refined
    /// position or the seed it started from, else nothing.
    ///
    /// The same order the evaluation's own seeding walks -- a measured position
    /// wins over the seed it was measured from -- so a fresh observation, which
    /// has only a seed, is placed where the step that proposed it put it rather
    /// than nowhere. One rule in one place, because everything that draws,
    /// names or moves a sighting has to agree about where it is. `None` is the
    /// state [`Unmeasured::NoSeed`] names.
    pub fn site(&self) -> Option<[f64; 2]> {
        if let Some(keypoint) = self.track.as_ref().and_then(|m| m.keypoint) {
            return Some([f64::from(keypoint[0]), f64::from(keypoint[1])]);
        }
        Some(self.cluster.as_ref()?.best_position())
    }

    /// The affine shape the observation is read at -- keypoint-frame units to
    /// this image's pixels -- where its cluster slot carries one.
    ///
    /// The refined shape when there is one, else the seed's. `None` for an
    /// observation that has only a track-stage keypoint, whose shape is the
    /// patch's rather than its own.
    pub fn shape(&self) -> Option<[[f64; 2]; 2]> {
        let cluster = self.cluster.as_ref()?;
        Some(cluster.shape.unwrap_or(cluster.seed_shape))
    }
}

/// The cluster stage's own data: a `.matches` cluster with its cluster-patches
/// section, in memory.
///
/// There is no pose, no position and no normal here. What makes the
/// observations one thing is that they all register onto one template cut from
/// one of them.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterPayload {
    /// Which observation the template is cut around, as an index into
    /// [`EditableTrack::observations`].
    pub reference: usize,
    /// The template's half-width, in keypoint-frame units: the patch every
    /// observation's shape is read over is the square `[-radius, radius]^2` of
    /// those units.
    ///
    /// **This is the cluster's one scale.** The seed and refined shapes in
    /// [`ClusterMeasurement`] are maps from keypoint-frame units to pixels and
    /// carry no size of their own, so what says how large the patches are is
    /// this number and nothing else. An evaluation runs the refinement kernel
    /// at exactly this radius rather than at
    /// [`ClusterRefineParams::radius`](crate::patch::cluster_refine::ClusterRefineParams::radius),
    /// so the meaning of a seed cannot change under it; everything drawn --
    /// the overlay's parallelogram, the Track View tile -- is read at it too.
    pub radius: f64,
    /// The cut itself. `None` until an evaluation cuts it, because the cut is a
    /// function of the reference's pixels and the bench holds no photographs.
    pub template: Option<ClusterTemplate>,
}

impl Default for ClusterPayload {
    /// A cluster cut around observation 0, at the refinement kernel's own
    /// template radius, with no template yet.
    ///
    /// The radius is read from the kernel's parameter type rather than written
    /// out again, so a bench cluster and a batch pass start from one scale.
    fn default() -> Self {
        Self {
            reference: 0,
            radius: ClusterRefineParams::default().radius,
            template: None,
        }
    }
}

/// The template the cluster stage registers its observations onto.
///
/// The samples are the reference observation's own tile on the template grid,
/// as the refinement's sampler reads it, which is the tile a panel draws. The
/// correlation the cascade runs z-normalizes it inside the kernel, over the
/// window and without the pixels that window drops, so what is kept here is the
/// picture rather than the kernel's working copy of it.
///
/// The cut's half-width is [`ClusterPayload::radius`] and is not repeated here:
/// one number says how large the cluster's patches are, and a template that
/// carried a second copy of it could disagree with the seeds it was cut from.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterTemplate {
    /// The `(resolution, resolution, channels)` samples.
    pub samples: Array3<f32>,
}

/// The track stage's own data: an `embedded_patches` point that is not in the
/// reconstruction yet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackPayload {
    /// Where the track's point stands. `None` for a track whose observations
    /// have not been triangulated, which a commit refuses.
    pub position: Option<Point3<f64>>,
    /// Whether [`Self::position`] is a bearing rather than a place: the point's
    /// own `w == 0`.
    ///
    /// **The flag is the authority and the frame is not.** The same three
    /// numbers are a place or a unit direction depending on this one bit, and a
    /// track can carry the bit with no patch at all -- a point put on the bench
    /// from a `sift_files` reconstruction has no patch frame to read a `w` off,
    /// and its bearings are still bearings. So everything that says which of the
    /// two it is holding -- the panel's word in front of the coordinate, the
    /// wire's choice of `direction` over `position`, the commit's `w` -- reads
    /// this, and [`Self::placement`]'s own `w` is kept equal to it wherever both
    /// exist, for the renderer and the kernels that project corners.
    pub at_infinity: bool,
    /// The patch the localizer registers against. Its centre is
    /// [`Self::position`] when both are present, and its `w` agrees with
    /// [`Self::at_infinity`].
    pub placement: Option<OrientedPatch>,
    /// The `(R, R, C)` patch bitmap: the render of the reference observation
    /// [`Self::reference`] names, or, where it names none, a mean of the `in`
    /// observations' renders (a bitmap fused before the reference was recorded,
    /// or one the reference-view rule found no view for).
    pub bitmap: Option<Array3<u8>>,
    /// The track's **reference observation**, as an index into the track's
    /// observations: the observation [`Self::bitmap`] is, or is to be,
    /// rendered from. With a bitmap, the bitmap is that observation's render.
    /// `None` means the track has no reference observation: a bitmap beside
    /// it is not the render of an observation the track has (a fused mean, or
    /// the render of an observation since removed from the point). Without a
    /// bitmap it is the reference the track carries until the next render,
    /// read from the point's stored reference or kept from a bitmap a step
    /// made stale.
    ///
    /// It follows its observation through every step that reorders or removes
    /// other observations. A step that makes the bitmap stale while the
    /// reference observation stays `in` -- one that moves the patch, or
    /// sights the reference observation at another keypoint -- drops the
    /// bitmap and keeps the reference (`drop_stale_bitmap`). A step that
    /// removes the reference observation from the track, splits it off, or
    /// turns it `out` drops both (`drop_bitmap`).
    ///
    /// **The pin of its row holds it.** Every render of a new bitmap on the
    /// bench -- the live evaluation's, a fit's, a normal step's -- renders
    /// from this reference where its row is `in`, carries a keypoint and is
    /// pinned ([`EditableTrack::held_reference`]), whichever row the
    /// reference-view rule would pick. Otherwise (`None`, or a row that is
    /// unpinned, not `in` or without a keypoint) the render runs the rule over
    /// the `in` rows and sets the bitmap and this reference together, so while
    /// the reference row is unpinned the reference follows the rule's pick at
    /// every render. The rule's pick that an evaluation reports per row
    /// (`TrackMeasurement::reference_view`) can therefore differ from this
    /// reference, the one in use, while the reference row is pinned, between
    /// an unpin that hands the reference on and the render
    /// ([`Self::bitmap_pending`]), and where the pick alternated between rows
    /// within one evaluation.
    /// [`set_reference`](super::steps::set_reference) makes a row the
    /// reference and pins it. A track put on the bench from a point has every
    /// row pinned, so the point's stored reference stays until the person
    /// unpins its row. A reference read from a column rendered for display is
    /// the rule's pick on the file's track, and is held the same way. A point
    /// stored with no reference (`-1`) beside a bitmap that names no row
    /// comes on with `None`, and the first evaluation that renders the bitmap
    /// ([`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap))
    /// and reads a pick the rule reached other than through its last fallback
    /// renders the bitmap from that pick and sets it here, whatever the pins
    /// ([`bitmap_target`](super::evaluate::bitmap_target)). A commit
    /// writes this reference as the point's reference observation, beside the
    /// bitmap it is the render of where the reconstruction stores bitmaps.
    pub reference: Option<usize>,
    /// Whether [`Self::bitmap`] is a **bitmap for judging**: a render made
    /// only so that the bars have something to score the rows against,
    /// because no `in` row could hold the reference. The live evaluation
    /// ([`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap))
    /// makes one where the render from the `in` rows gives no bitmap (every
    /// row is `out`, or fewer than two `in` rows carry a keypoint), from the
    /// reference-view rule's pick among every row that carries a keypoint,
    /// `in` or `out`. Without it, a track whose `in` set emptied could never
    /// be scored again, and loosening a bar could bring no row back.
    ///
    /// It names no reference ([`Self::reference`] is `None`), a step that
    /// turns rows `out` leaves it in place, and a commit treats it as no
    /// bitmap ([`Self::committable_bitmap`]). Once a row is `in` again, the
    /// next render replaces it with a bitmap rendered by the usual rule.
    pub bitmap_for_judging: bool,
    /// Whether [`Self::bitmap`] is kept only until the next render replaces
    /// it: an unpin handed the reference to the reference-view rule
    /// ([`unpin_verdicts`](super::steps::unpin_verdicts)), whose pick is
    /// another row, so every score read against this bitmap is about to be
    /// replaced. While it is set, no row is scored against the bitmap (each
    /// says [`Unmeasured::BitmapPending`]) and the bars judge none. The next
    /// render
    /// ([`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap))
    /// replaces the bitmap and clears it, and a step that holds the reference
    /// again by pinning its row ([`pin_verdicts`](super::steps::pin_verdicts),
    /// [`set_verdict`](super::steps::set_verdict)) clears it with nothing
    /// rendered, since the bitmap is that row's render. A commit in between
    /// writes this bitmap with the reference it is the render of.
    pub bitmap_pending: bool,
    /// The colour the point carries, used when there is no bitmap to read one
    /// from.
    pub color: [u8; 3],
    /// Confidence in the frame's normal, in the stored column's byte scale.
    pub normal_confidence: Option<u8>,
    /// The last triangulation's condition number.
    pub condition_number: Option<f64>,
}

impl TrackPayload {
    /// Drop the bitmap and the reference observation, together, for a step
    /// after which the reference observation is no longer one the track
    /// keeps `in`.
    pub(crate) fn drop_bitmap(&mut self) {
        self.bitmap = None;
        self.bitmap_for_judging = false;
        self.bitmap_pending = false;
        self.reference = None;
    }

    /// Drop the bitmap and keep the reference observation, for a step after
    /// which the bitmap no longer shows what the reference observation sees
    /// but that observation is still on the track.
    pub fn drop_stale_bitmap(&mut self) {
        self.bitmap = None;
        self.bitmap_for_judging = false;
        self.bitmap_pending = false;
    }

    /// The bitmap a commit writes: [`Self::bitmap`], or `None` where the track
    /// has none or its bitmap is one for judging only
    /// ([`Self::bitmap_for_judging`]).
    pub fn committable_bitmap(&self) -> Option<&Array3<u8>> {
        self.bitmap.as_ref().filter(|_| !self.bitmap_for_judging)
    }

    /// Drop the bitmap with its reference observation when that observation
    /// is not one of `observations`' `in` rows: a reference is an observation
    /// the track keeps. A bitmap for judging names no reference, so it stays.
    pub fn drop_bitmap_unless_in(&mut self, observations: &[Observation]) {
        if let Some(r) = self.reference {
            if observations.get(r).is_none_or(|o| o.verdict != Verdict::In) {
                self.drop_bitmap();
            }
        }
    }
}

/// Which of the two representations a track is in, and that representation's
/// own data.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// A set of image patches that register onto one template, with no geometry
    /// behind them.
    Cluster(ClusterPayload),
    /// A patch at a position, with a keypoint per observation.
    Track(TrackPayload),
}

/// A stage without its data, for a caller that only wants to say which one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// [`Stage::Cluster`].
    Cluster,
    /// [`Stage::Track`].
    Track,
}

impl std::fmt::Display for StageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageKind::Cluster => write!(f, "cluster"),
            StageKind::Track => write!(f, "track"),
        }
    }
}

impl Stage {
    /// Which stage this is, without its data.
    pub fn kind(&self) -> StageKind {
        match self {
            Stage::Cluster(_) => StageKind::Cluster,
            Stage::Track(_) => StageKind::Track,
        }
    }
}

/// The point an editable track was put on the bench from.
///
/// The serial is opaque here: it is whatever the caller numbers its versions
/// with, and core neither mints nor interprets it. What core does with the
/// origin is decide whether a commit replaces a point or creates one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    /// The version the point was read out of, as the caller numbers versions.
    pub version: u64,
    /// The index the point had in that version.
    pub point: u32,
}

/// The bars the threshold painting judges an observation against.
///
/// [`Self::geometry_search_min_relative_zncc`]'s default is read from view
/// selection's own parameter type rather than written out again, so the bench and the batch
/// pass start from the same bar and moving it is the person choosing to
/// differ. The others are the bench's own. [`BENCH_MAX_SHIFT_PX`]: on the bench the bar is also how far a
/// fit may move a sighting, and the cluster refinement's 3 px turned away walks
/// a person wanted. [`BENCH_CLUSTER_MIN_ZNCC`] and
/// [`BENCH_CLUSTER_MIN_ZNCC_MIDDLE`]: the cluster refinement's `0.85` judges
/// the score it reached by fitting a whole affine warp, and turned out
/// sightings a person would keep. [`BENCH_MIN_ZNCC`] and
/// [`BENCH_MIN_ZNCC_MIDDLE`]: the track stage judges another score, a row's
/// tile against the stored bitmap, and has bars measured for it.
/// [`BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`]: the keypoint localizer's member
/// gate has the same default, but the bench runs that kernel with its gate off
/// and judges the radius by this bar instead.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// The ZNCC an observation has to reach at the track stage: the
    /// blur-matched score of its tile against the stored bitmap
    /// ([`TrackMeasurement::blur_matched_zncc`]).
    /// The default is [`BENCH_MIN_ZNCC`]. The cluster stage judges another
    /// score, and has its own bar, [`Self::cluster_min_zncc`].
    pub min_zncc: f64,
    /// The middle ZNCC an observation has to reach at the track stage
    /// ([`TrackMeasurement::blur_matched_zncc_middle`]). It turns out a sighting whose
    /// whole-patch agreement is carried by the patch's surroundings rather
    /// than its middle.
    ///
    /// `0` turns the bar off. The default is [`BENCH_MIN_ZNCC_MIDDLE`]; its
    /// doc gives what the measurement found. An
    /// observation with no middle reading, because its middle is flat or it
    /// was read back from a committed point, has nothing to judge and clears
    /// the bar, as a row with no self-similarity reading clears
    /// [`Self::max_zncc_self_similarity_radius`].
    pub min_zncc_middle: f64,
    /// The achieved template ZNCC an observation has to reach at the cluster
    /// stage ([`ClusterMeasurement::zncc`]). The default is
    /// [`BENCH_CLUSTER_MIN_ZNCC`].
    pub cluster_min_zncc: f64,
    /// The middle ZNCC an observation has to reach at the cluster stage
    /// ([`ClusterMeasurement::zncc_middle`]), judged as
    /// [`Self::min_zncc_middle`] is at the track stage: `0` turns it off, and
    /// a row with no middle reading clears it. The default is
    /// [`BENCH_CLUSTER_MIN_ZNCC_MIDDLE`].
    pub cluster_min_zncc_middle: f64,
    /// How far the correlation peak may sit from where the observation sits, in
    /// **patch-grid px**: [`ClusterMeasurement::shift_px`] at the cluster stage
    /// and [`TrackMeasurement::seed_shift_px`] at the track stage.
    ///
    /// At the track stage it is also how far from each observation the
    /// evaluation looks for the peak, so one number says how far a sighting may
    /// be from where the correlation wants it, how far the evaluation looks, and
    /// how far a fit may walk it.
    ///
    /// Both are the observation's **own** evidence. The distance to the
    /// point's projection is judged by its own bar,
    /// [`Self::max_projection_error_px`].
    ///
    /// At the track stage it is also the bound on a fit's walk: a sighting the
    /// fit's kernels would move further than this from where it sat keeps its
    /// place ([`TrackMeasurement::walked_px`]).
    pub max_shift_px: f64,
    /// The largest ZNCC self-similarity radius an observation's own tile may
    /// have, in patch-grid px: [`ClusterMeasurement::zncc_self_similarity_radius`]
    /// at the cluster stage and [`TrackMeasurement::zncc_self_similarity_radius`]
    /// at the track stage. It turns out a sighting whose patch can slide over
    /// itself further than this and still match, such as a straight edge or a
    /// flat patch, whose position a match cannot pin.
    ///
    /// The radius reads at most the largest shift searched, `3` by default,
    /// which stands for "that far or further", so a bar at or above it turns
    /// nothing out. The default is [`BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`].
    /// An observation with no reading, because its tile could not be sampled
    /// or it was read back from a committed point, has nothing to judge and
    /// clears the bar.
    pub max_zncc_self_similarity_radius: f64,
    /// The largest reprojection error an observation may have, in
    /// **source-image px**: [`TrackMeasurement::reprojection_error`] where the
    /// track is triangulated and [`TrackMeasurement::projection_offset_px`]
    /// before it is, the number Track View's *Proj. err* cell prints. Judged at
    /// the track stage alone, since the cluster stage has no point to project.
    ///
    /// Unlike the other bars it judges the **point** as much as the sighting:
    /// a mis-triangulated point gives every row a large error, and this bar
    /// then proposes turning them all out, including the observations that
    /// would move the point back. A row that fails this bar alone, beside rows
    /// that fail it too, says the point is off rather than the sighting. The
    /// default is [`BENCH_MAX_PROJECTION_ERROR_PX`].
    ///
    /// `0` turns the bar off, as it does [`Self::min_zncc_middle`]: no
    /// reading clears a bar of `0` but an exact projection, so the value says
    /// nothing a bar could mean. An observation with no reading has nothing to
    /// judge and clears the bar.
    pub max_projection_error_px: f64,
    /// The fraction of the track's own self-agreement a candidate's ZNCC has to
    /// reach for a geometry search ([`search_geometry`](super::search_geometry))
    /// to admit it. It judges no observation of the track: the evaluation, the
    /// painting and [`apply_thresholds`](super::apply_thresholds) do not read it.
    pub geometry_search_min_relative_zncc: f64,
}

/// The bench's default [`Thresholds::max_shift_px`], in patch-grid px.
///
/// The keypoint localizer's own search radius, so a track's evaluation looks
/// as far as the batch localizer does. Separate from the cluster refinement's
/// own `max_shift_px` (3 source-image px), which stays the batch pass's bar.
pub const BENCH_MAX_SHIFT_PX: f64 = 6.0;

/// The bench's default [`Thresholds::max_projection_error_px`], in
/// source-image px.
pub const BENCH_MAX_PROJECTION_ERROR_PX: f64 = 3.0;

/// The bench's default [`Thresholds::min_zncc`], the track stage's bar on the
/// blur-matched score against the stored bitmap: `0.70`.
///
/// Below the cluster refinement's own `min_zncc` (0.85), which stays the batch
/// pass's bar: a correct sighting's tile reads lower against one view's render
/// than a fitted template does. Measured on 150 tracks from each of eight
/// reconstructions (two ground truths, six solves whose members are taken as
/// correct), every view aligned to the reference render. Wrong views were
/// planted by pasting an unwarped square of another place's pixels over a
/// row's keypoint, and true members were blurred, which are views to keep.
/// About three in four of the views planted in observed images fail a
/// geometry bar and are out whatever this bar is, so the objective counts only rows that clear
/// every geometry bar: the mean over tracks of the share of members kept and
/// the share of wrong views in images the track observes turned out. Held
/// out by reconstruction, the whole bar alone picks `0.70` on all eight
/// folds, on the blur-matched score and on the plain one. Against the plain
/// `0.65` it replaced, the blur-matched `0.70` with the middle bar of
/// [`BENCH_MIN_ZNCC_MIDDLE`] loses 2.1% of the blurred members that clear the
/// geometry bars rather than 5.2%, and turns out 94.0% of the wrong views
/// rather than 90.3%, and loses 5.8% of the members rather than 3.1%. The spec of
/// the editable track (specs/core/bench/editable-track.md) gives the tables.
pub const BENCH_MIN_ZNCC: f64 = 0.70;

/// The bench's default [`Thresholds::min_zncc_middle`]: `0.50`, on the
/// blur-matched middle score.
///
/// The maintainer's choice from the measurement behind [`BENCH_MIN_ZNCC`],
/// where the middle bar changes little: beside a whole bar of `0.70` the
/// objective is 93.48 over all the data at a middle bar of `0.50`, 93.55 with
/// it off and 93.74 at `0.30`, and held out by reconstruction 92.94, 92.98 and
/// 93.20. A middle bar of `0.50` turns out a few more of the wrong views most
/// similar to the true content, which the whole bar cannot tell apart, at the
/// cost of 0.6% of the members. The middle reading covers a quarter of the
/// samples, so on a correct sighting it scatters more and reads lower than the
/// whole-patch one.
pub const BENCH_MIN_ZNCC_MIDDLE: f64 = 0.50;

/// The bench's default [`Thresholds::cluster_min_zncc`], the cluster stage's
/// bar on the achieved template ZNCC.
///
/// `0.7`, below the cluster refinement's own `min_zncc` (0.85), which stays
/// the batch pass's bar and turned out sightings a person would keep. The
/// measurement behind [`BENCH_MIN_ZNCC`] read the track stage only; this bar
/// is not measured.
pub const BENCH_CLUSTER_MIN_ZNCC: f64 = 0.7;

/// The bench's default [`Thresholds::cluster_min_zncc_middle`]: `0.7`, the
/// same as [`BENCH_CLUSTER_MIN_ZNCC`], and likewise not measured.
pub const BENCH_CLUSTER_MIN_ZNCC_MIDDLE: f64 = 0.7;

/// The bench's default [`Thresholds::max_zncc_self_similarity_radius`], in
/// patch-grid px.
///
/// `2.5`, the same bar as the keypoint localizer's member gate
/// ([`DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`](crate::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS)),
/// which the user chose from a sweep on seoul_bull and kerry_park. It sits
/// under the largest shift the reading searches (`3`), so a tile that still
/// matches itself at the edge of the search, such as a straight edge or a flat
/// area, is turned out; a corner or a busy texture reads under `1`.
pub const BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS: f64 = 2.5;

// The bench and the localizer start from the same bar.
const _: () = assert!(
    BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS
        == crate::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS
);

// The middle bar, where it is on, sits no higher than the whole bar: the
// middle reading reads lower than the whole one on a correct sighting.
const _: () = assert!(BENCH_MIN_ZNCC_MIDDLE <= BENCH_MIN_ZNCC);

impl Thresholds {
    /// The whole and middle ZNCC bars at `stage`: [`Self::min_zncc`] and
    /// [`Self::min_zncc_middle`] at the track stage, and
    /// [`Self::cluster_min_zncc`] and [`Self::cluster_min_zncc_middle`] at the
    /// cluster stage, which judge another score.
    pub fn zncc_bars(&self, stage: StageKind) -> (f64, f64) {
        match stage {
            StageKind::Track => (self.min_zncc, self.min_zncc_middle),
            StageKind::Cluster => (self.cluster_min_zncc, self.cluster_min_zncc_middle),
        }
    }
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_zncc: BENCH_MIN_ZNCC,
            min_zncc_middle: BENCH_MIN_ZNCC_MIDDLE,
            cluster_min_zncc: BENCH_CLUSTER_MIN_ZNCC,
            cluster_min_zncc_middle: BENCH_CLUSTER_MIN_ZNCC_MIDDLE,
            max_shift_px: BENCH_MAX_SHIFT_PX,
            max_zncc_self_similarity_radius: BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS,
            max_projection_error_px: BENCH_MAX_PROJECTION_ERROR_PX,
            geometry_search_min_relative_zncc: ViewSelectParams::default().min_relative_zncc,
        }
    }
}

/// A track being worked on: everything that has been tried against it, and
/// nothing about the window it is being looked at in.
///
/// The observations are the list; the stage says which kernels apply and what a
/// commit can do; the origin says whether a commit replaces a point or creates
/// one; the thresholds are the bars the painting proposes verdicts against.
///
/// One `in` observation per image is the invariant every step that sets a
/// verdict holds: a track cannot observe an image twice, so a second
/// observation in an image already held is shown and scored but cannot be
/// turned `in` until the other is turned `out`.
#[derive(Debug, Clone, PartialEq)]
pub struct EditableTrack {
    /// The observations, in the order they were added. No step on the track
    /// renumbers them; a split and the deletion of an image from the
    /// reconstruction ([`Self::delete_image`]) are the two operations that
    /// do.
    pub observations: Vec<Observation>,
    /// Which representation the track is in, and that representation's data.
    pub stage: Stage,
    /// The point this track was put on the bench from, when there was one.
    pub origin: Option<Origin>,
    /// The bars the painting judges against.
    pub thresholds: Thresholds,
    /// Whether the verdicts were set by the repaint of the evaluation that
    /// took the readings, so the readings were taken under other verdicts.
    /// See [`RepaintMark`].
    pub repaint: RepaintMark,
}

/// That an evaluation's repaint set a track's verdicts after its readings were
/// taken, which the next evaluation of the same value reads without repainting.
///
/// An evaluation paints the bars' verdicts onto every unpinned observation
/// ([`evaluate`](super::evaluate::evaluate)). The track-stage ZNCC of a row is
/// scored against the rows that are `in`, so a repaint that changes the `in`
/// set leaves readings taken under the old one, and the next evaluation reads
/// the rows again under the new one. Were that evaluation to repaint too, a row
/// its new reading drops below a bar would be turned out, the one after would
/// read under that set and could turn it back, and so on. So the evaluation
/// whose repaint moved a verdict leaves this mark on the track it returns, and
/// an evaluation of a track carrying it reads without repainting. The table
/// then settles after one more evaluation, and a row left out of step with the
/// bars shows as a verdict the bars disagree with until the next edit.
///
/// [`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap)
/// is the one exception: it repaints a marked track when its bitmap moves,
/// because the new bitmap gives new scores for the bars to judge. A chain of
/// such evaluations can run more than one extra evaluation; it is bounded
/// because each one that renders does so from a row the mark does not already
/// record (`rendered_from` below), so the chain is no longer than the number
/// of distinct rows the bitmap can be rendered from.
///
/// **The mark belongs to the one value the evaluation returned.** Every step
/// makes its new track by cloning the old one, and a clone does not carry the
/// mark, so "only the repaint has changed since the readings" is exactly
/// "this is still the value that evaluation returned". A caller that holds the
/// track behind an `Arc`, as the bench does, keeps the mark until the next step.
/// The mark also records the verdicts and pins the repaint left, and is honoured
/// only while the track still carries them, so a verdict changed in place
/// rather than by a step is repainted as any other change is.
///
/// Equality ignores it: two tracks that differ only in the mark are the same
/// track.
#[derive(Debug, Default)]
pub struct RepaintMark {
    /// The verdict and pin of each observation as the repaint left them, or
    /// `None` for no mark.
    left: Option<Vec<(Verdict, bool)>>,
    /// The rows the stored bitmap has been rendered from by the evaluations
    /// since the last step (`None` for a render left to the reference-view
    /// rule's own choice), so an evaluation of the marked track does not move
    /// the bitmap back to one of them
    /// ([`evaluate_rendering_bitmap`](super::evaluate::evaluate_rendering_bitmap)).
    rendered_from: Vec<Option<usize>>,
}

impl RepaintMark {
    /// The mark for `track`'s verdicts and pins as they stand.
    pub(super) fn of(track: &EditableTrack) -> Self {
        Self::after_renders(track, Vec::new())
    }

    /// The mark for `track`'s verdicts and pins as they stand, after
    /// evaluations that rendered the bitmap from the rows `rendered_from`.
    pub(super) fn after_renders(track: &EditableTrack, rendered_from: Vec<Option<usize>>) -> Self {
        Self {
            left: Some(verdicts_and_pins(track)),
            rendered_from,
        }
    }

    /// The rows the bitmap has been rendered from since the last step.
    pub(super) fn rendered_from(&self) -> &[Option<usize>] {
        &self.rendered_from
    }

    /// The same mark, for a value that differs from the marked one in nothing
    /// a reading or a verdict depends on: a step that re-renders the stored
    /// bitmap where the patch stands carries the mark across with this.
    pub(super) fn carried(&self) -> Self {
        Self {
            left: self.left.clone(),
            rendered_from: self.rendered_from.clone(),
        }
    }
}

/// A clone is the start of a new value a step is about to change, so it does
/// not carry the mark.
impl Clone for RepaintMark {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for RepaintMark {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

fn verdicts_and_pins(track: &EditableTrack) -> Vec<(Verdict, bool)> {
    track
        .observations
        .iter()
        .map(|o| (o.verdict, o.pinned))
        .collect()
}

impl EditableTrack {
    /// An empty track at the cluster stage, with the default thresholds and no
    /// origin.
    ///
    /// Its reference names observation 0, which does not exist yet; the first
    /// observation added takes that place.
    pub fn empty_cluster() -> Self {
        Self {
            observations: Vec::new(),
            stage: Stage::Cluster(ClusterPayload::default()),
            origin: None,
            thresholds: Thresholds::default(),
            repaint: RepaintMark::default(),
        }
    }

    /// Whether the track's verdicts were set by the repaint of the evaluation
    /// that took its readings, and nothing has changed since: the state an
    /// evaluation reads without repainting ([`RepaintMark`]).
    pub fn repainted(&self) -> bool {
        self.repaint
            .left
            .as_ref()
            .is_some_and(|left| *left == verdicts_and_pins(self))
    }

    /// Which stage the track is in.
    pub fn stage_kind(&self) -> StageKind {
        self.stage.kind()
    }

    /// How many observations carry each verdict, as `(in, out)`.
    pub fn verdict_counts(&self) -> (usize, usize) {
        let kept = self
            .observations
            .iter()
            .filter(|o| o.verdict == Verdict::In)
            .count();
        (kept, self.observations.len() - kept)
    }

    /// The indexes of the `in` observations, ascending.
    pub fn in_observations(&self) -> Vec<usize> {
        self.observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.verdict == Verdict::In)
            .map(|(i, _)| i)
            .collect()
    }

    /// The index of the `in` observation in `image`, when the track holds one.
    pub fn in_observation_of_image(&self, image: u32) -> Option<usize> {
        self.observations
            .iter()
            .position(|o| o.image == image && o.verdict == Verdict::In)
    }

    /// A copy of this track whose origin is the point at `point` in version
    /// `version`.
    ///
    /// What a caller re-seats a committed track with, so a second commit of it
    /// replaces what the first wrote.
    pub fn with_origin(&self, version: u64, point: u32) -> Self {
        let mut next = self.clone();
        next.origin = Some(Origin { version, point });
        next
    }

    /// This track as it reads after image `image` is deleted from its
    /// reconstruction, which moves every later image down by one.
    ///
    /// The observations in `image` are dropped, and every observation in a
    /// later image is renumbered to the index that photograph holds after the
    /// delete, so each observation still names the photograph it was sighted
    /// in. What a dropped observation measured goes with it; every other
    /// observation keeps its measurements and its verdict.
    ///
    /// A cluster's reference follows its observation. When the reference
    /// itself was dropped, it is pointed at the first observation left and the
    /// template is dropped, because the template is a cut around the old
    /// reference. A track-stage reference follows its observation the same
    /// way; when it was dropped, the patch bitmap, its render, is dropped with
    /// it ([`TrackPayload::reference`]).
    ///
    /// Returns `None` when the track observes no image at or past `image`,
    /// since then nothing about it changes. Otherwise returns the new track and
    /// where each observation went: entry `i` is the new index of observation
    /// `i`, or `None` for one that was dropped. The new track may have no
    /// observations left; what to do with it then is the caller's choice
    /// ([`Bench::delete_image`](super::Bench::delete_image) discards it).
    pub fn delete_image(&self, image: u32) -> Option<(EditableTrack, Vec<Option<usize>>)> {
        if self.observations.iter().all(|o| o.image < image) {
            return None;
        }
        let mut next = self.clone();
        let mut map = Vec::with_capacity(self.observations.len());
        next.observations.clear();
        for observation in &self.observations {
            if observation.image == image {
                map.push(None);
                continue;
            }
            let mut kept = observation.clone();
            if kept.image > image {
                kept.image -= 1;
            }
            map.push(Some(next.observations.len()));
            next.observations.push(kept);
        }
        let mut dropped = false;
        match &mut next.stage {
            Stage::Cluster(payload) => match map.get(payload.reference).copied().flatten() {
                Some(reference) => payload.reference = reference,
                None => {
                    payload.reference = 0;
                    payload.template = None;
                }
            },
            // The row the bitmap is the tile of follows its observation. A
            // bitmap whose observation was dropped is the tile of a sighting
            // the track no longer has, so it goes with its reference and every
            // score read against it, and the live evaluation renders one from
            // the rows that remain.
            Stage::Track(payload) => {
                // A bitmap for judging names no row and can be the tile of a
                // dropped observation, so it goes where any row went.
                if payload.bitmap_for_judging && map.iter().any(Option::is_none) {
                    payload.drop_bitmap();
                    dropped = true;
                }
                if let Some(r) = payload.reference {
                    match map.get(r).copied().flatten() {
                        Some(kept) => payload.reference = Some(kept),
                        None => {
                            payload.drop_bitmap();
                            dropped = true;
                        }
                    }
                }
            }
        }
        if dropped {
            super::evaluate::clear_bitmap_scores(&mut next, Unmeasured::NoBitmap);
        }
        Some((next, map))
    }

    /// The cluster payload, or `None` at the track stage.
    pub fn cluster(&self) -> Option<&ClusterPayload> {
        match &self.stage {
            Stage::Cluster(payload) => Some(payload),
            Stage::Track(_) => None,
        }
    }

    /// The track payload, or `None` at the cluster stage.
    pub fn track(&self) -> Option<&TrackPayload> {
        match &self.stage {
            Stage::Track(payload) => Some(payload),
            Stage::Cluster(_) => None,
        }
    }

    /// The row the reference-view rule picked when the track was last read:
    /// the one whose
    /// [`reference_view`](TrackMeasurement::reference_view) standing is the
    /// reference. `None` at the cluster stage, before an evaluation, and where
    /// the rule picked none. Where the reference's row is pinned this can
    /// differ from the reference in use ([`TrackPayload::reference`]).
    pub fn reference_view_pick(&self) -> Option<usize> {
        self.track()?;
        self.observations.iter().position(|o| {
            o.track
                .as_ref()
                .and_then(|m| m.reference_view)
                .is_some_and(|standing| standing.is_reference())
        })
    }

    /// The reference observation the next render of the bitmap renders from
    /// whatever the reference-view rule picks: the track's reference
    /// ([`TrackPayload::reference`]) where its row is `in`, carries a keypoint
    /// and is **pinned**. `None` at the cluster stage and where the track
    /// holds no reference that way; a render then takes the rule's pick.
    pub fn held_reference(&self) -> Option<usize> {
        let r = self.track()?.reference?;
        let row = self.observations.get(r)?;
        let keyed = row.track.as_ref().is_some_and(|m| m.keypoint.is_some());
        (row.pinned && row.verdict == Verdict::In && keyed).then_some(r)
    }

    /// Whether the track's bitmap is kept only until the next render
    /// replaces it ([`TrackPayload::bitmap_pending`]) while the reference is
    /// not held again by its row's pin: no row is scored against it.
    pub fn bitmap_pending(&self) -> bool {
        self.track()
            .is_some_and(|p| p.bitmap_pending && p.bitmap.is_some())
            && self.held_reference().is_none()
    }

    /// Clear [`TrackPayload::bitmap_pending`] where the reference is held
    /// again by its row's pin: the bitmap is that row's render, so nothing is
    /// pending. Every step that pins a row calls this, so the field is set only
    /// while a render is still to replace the bitmap.
    pub(crate) fn end_pending_if_held(&mut self) {
        if self.held_reference().is_some() {
            if let Stage::Track(payload) = &mut self.stage {
                payload.bitmap_pending = false;
            }
        }
    }
}
