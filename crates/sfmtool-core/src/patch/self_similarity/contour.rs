// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The contour a ZNCC self-similarity radius is read from, and its reach: how
//! far the contour extends, in three units. Those are grid px, source-image
//! px, and lengths along the patch's own axes in the scene's world-space unit
//! (degrees for a patch at infinity).
//!
//! The radius says how far, in any direction, a match could land and still be
//! indistinguishable from the true position. Read along the patch's `u` and `v`
//! axes in world space, the same contour bounds where on the patch the point
//! could be, per axis. See `specs/core/patch/zncc-self-similarity-radius.md`,
//! "The contour and its reach".

use super::{crossing_points, radius_of_points, SelfSimilarity};
use crate::camera::warp_map::singular_values_2x2;
use crate::patch::cloud::OrientedPatch;

/// One point of the contour where a self-similarity surface falls through its
/// level `1 − tolerance`, in the grid's axes (`x` column-right, `y` row-down),
/// in grid px from the centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContourPoint {
    /// `(dx, dy)` from the centre.
    pub offset: [f64; 2],
    /// `None` for a crossing interpolated along a grid edge. `Some([ex, ey])`
    /// for a shift at or above the level whose neighbour one grid step
    /// `[ex, ey]` away has no reading, because it lies past the square's
    /// border or its reading is not finite: the crossing lies somewhere past
    /// the shift in that direction, so the shift's own offset is a lower bound
    /// for it along that axis.
    pub open_towards: Option<[i8; 2]>,
}

impl ContourPoint {
    /// Whether the point is open: the crossing it stands for lies further out
    /// than the point itself, possibly much further.
    pub fn is_open(&self) -> bool {
        self.open_towards.is_some()
    }
}

/// A length read off the contour, and whether it is only a lower bound: "this
/// far or further", the reading the radius prints as `3+`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedLength {
    /// The length the readings show.
    pub value: f64,
    /// Whether the true length may be larger than [`Self::value`], because
    /// the region at the level runs off the square of shifts searched along
    /// this length's axis, runs off along the other axis without holding its
    /// width, borders a gap (a neighbour with no reading) that reaches further,
    /// or reaches the largest radius searched, the cap.
    ///
    /// `false` where the value is exact within the square searched, with a
    /// region that runs off the square and holds its width taken to keep it
    /// past the border. A repeating texture whose period exceeds the square
    /// can match itself again further out, and that is not seen.
    pub at_least: bool,
}

/// How far the contour reaches along the patch's `u` and `v` axes, `[u, v]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchAxisReach {
    /// A finite patch (`w == 1`): a length in the unit of the patch's
    /// half-extents, the scene's world-space unit (`world_space_unit`), or
    /// scene units where the scene names none.
    Length([BoundedLength; 2]),
    /// A patch at infinity (`w == 0`), which has no length: the angle at the
    /// eye, in degrees.
    Angle([BoundedLength; 2]),
}

impl PatchAxisReach {
    /// The two reaches, `[u, v]`, whatever their unit.
    pub fn values(&self) -> [BoundedLength; 2] {
        match *self {
            PatchAxisReach::Length(values) | PatchAxisReach::Angle(values) => values,
        }
    }
}

/// The contour of one [`SelfSimilarity`] reading: every point where its
/// surface falls through the level, as the radius reads them
/// ([`SelfSimilarity::contour`]), and where the readings leave it open.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfSimilarityContour {
    points: Vec<ContourPoint>,
    max_radius: usize,
    openings: Openings,
}

/// Where the region at the level reaches a shift with no reading, and so
/// where the contour may lie further out than its points.
#[derive(Debug, Clone, PartialEq, Default)]
struct Openings {
    /// The region may continue past the square's border: a shift at the level
    /// on the border, or a connected set of shifts with no reading next to
    /// one at the level that reaches the border.
    escapes: bool,
    /// Per grid axis, `[x, y]`, whether an escape may carry the contour
    /// further along that axis than the readings show (the rule is
    /// `holds_its_width`'s).
    axis_open: [bool; 2],
    /// The shifts bounding where an enclosed gap could put the contour: every
    /// shift of each connected set of shifts with no reading next to one at
    /// the level and not reaching the border, and every read shift next to
    /// one of them. Any crossing in such a gap lies on a grid edge between two
    /// of them.
    enclosed: Vec<[f64; 2]>,
}

impl SelfSimilarity {
    /// The contour this reading's radius is read from, or `None` for a reading
    /// that has none: no data (the overlap reading's `NaN` tolerance), or a
    /// surface that is not a `(2r + 1)²` square for an odd side.
    ///
    /// A template with no textured channel, whose tolerance is infinite and
    /// whose every shift counts, has as its contour every shift of the square's
    /// border, each open towards the outside: it matched itself at the edge of
    /// the search in every direction, and reads `r` or more along each axis.
    pub fn contour(&self) -> Option<SelfSimilarityContour> {
        let len = self.surface.len();
        let side = (len as f64).sqrt().round() as usize;
        if side * side != len || side.is_multiple_of(2) || self.tolerance.is_nan() {
            return None;
        }
        let r = (side - 1) / 2;
        let (points, openings) = if self.tolerance == f64::INFINITY {
            // Every shift is at the level; only the border has unread
            // neighbours.
            let flat = vec![1.0; len];
            (crossing_points(&flat, r, 0.0), openings(&flat, r, 0.0))
        } else {
            let level = 1.0 - self.tolerance;
            (
                crossing_points(&self.surface, r, level),
                openings(&self.surface, r, level),
            )
        };
        Some(SelfSimilarityContour {
            points,
            max_radius: r,
            openings,
        })
    }
}

impl SelfSimilarityContour {
    /// The contour's points, row-major by the shift they were read from.
    pub fn points(&self) -> &[ContourPoint] {
        &self.points
    }

    /// The largest radius the reading searched, `r`, in grid px.
    pub fn max_radius(&self) -> usize {
        self.max_radius
    }

    /// The radius, in grid px: exactly [`SelfSimilarity::radius`], the largest
    /// distance of a point from the centre capped at `r`. At least, and so
    /// [`BoundedLength::at_least`], where it reaches `r`, where the region at
    /// the level runs off the square, or where a gap with no reading beside
    /// that region reaches further from the centre than the value.
    pub fn radius(&self) -> BoundedLength {
        let value = radius_of_points(&self.points, self.max_radius);
        let r = self.max_radius as f64;
        let open = &self.openings;
        BoundedLength {
            value,
            at_least: value >= r
                || open.escapes
                || open
                    .enclosed
                    .iter()
                    .any(|&c| norm_sq(c).sqrt().min(r) > value),
        }
    }

    /// How far the contour reaches along each grid axis, in grid px:
    /// `[max |dx|, max |dy|]` over its points.
    ///
    /// An axis's value is a lower bound ([`BoundedLength::at_least`]) where
    /// the region at the level may extend further along it than the readings
    /// show. That is where the region runs off the square along this axis;
    /// where it runs off along the other axis and its width along this one
    /// grows over the last two lines before the border, by more than
    /// `WIDTH_SLACK`, or cannot be read there; and where a gap with no reading
    /// beside the region reaches further along the axis than the value. See
    /// the spec's "Lower bounds" for the rule.
    pub fn grid_axes(&self) -> [BoundedLength; 2] {
        let open = &self.openings;
        std::array::from_fn(|k| {
            let value = self
                .points
                .iter()
                .map(|p| p.offset[k].abs())
                .fold(0.0, f64::max);
            BoundedLength {
                value,
                at_least: open.axis_open[k] || open.enclosed.iter().any(|c| c[k].abs() > value),
            }
        })
    }

    /// The radius in source-image px, through `jacobian`, the image px per
    /// grid px at the grid's centre (`[[dx/dcol, dx/drow], [dy/dcol,
    /// dy/drow]]`, as
    /// [`crate::camera::warp_map::patch_grid_jacobian`] returns it): the largest
    /// `|J · d|` over the contour's points `d`, each first brought in to `r`
    /// from the centre where it lies further, the cap the radius takes, so
    /// that under `J = s·I` this is `s` times [`Self::radius`].
    ///
    /// A lower bound wherever a point reaches the cap, the region at the level
    /// runs off the square, or a gap with no reading beside that region could
    /// put a point further out through `J` than the value, so that it is never
    /// reported exact where the contour may reach further in the photograph.
    ///
    /// `None` for a Jacobian that is not finite or is singular.
    pub fn image_radius(&self, jacobian: [[f64; 2]; 2]) -> Option<BoundedLength> {
        let [[a, b], [c, d]] = jacobian;
        let det = a * d - b * c;
        if !(jacobian.iter().flatten().all(|v| v.is_finite()) && det != 0.0 && det.is_finite()) {
            return None;
        }
        let r = self.max_radius as f64;
        let image = |p: [f64; 2]| (a * p[0] + b * p[1]).hypot(c * p[0] + d * p[1]);
        let mut value = 0.0f64;
        let mut capped = false;
        for point in &self.points {
            let length = norm_sq(point.offset).sqrt();
            capped |= length >= r;
            let p = if length > r {
                point.offset.map(|v| v * r / length)
            } else {
                point.offset
            };
            value = value.max(image(p));
        }
        // Within the disc of radius `r` the largest `|J · d|` is `r` times
        // the larger singular value, so a gap beyond it is bounded by that;
        // inside it, by the gap's own shifts, a crossing lying on an edge
        // between two of them.
        let widest = r * singular_values_2x2(jacobian)[0];
        let open = &self.openings;
        let at_least = capped
            || open.escapes
            || open.enclosed.iter().any(|&c| {
                let bound = if norm_sq(c).sqrt() > r {
                    widest
                } else {
                    image(c)
                };
                bound > value
            });
        Some(BoundedLength { value, at_least })
    }

    /// How far the contour reaches along `placement`'s `u` and `v` axes, for a
    /// reading of a tile whose `resolution × resolution` core is `placement`:
    /// one grid px along `x` is `2·half_extent[0] / resolution` along `u`, and
    /// along `y` is `2·half_extent[1] / resolution` along `−v`, so
    /// [`Self::grid_axes`] scales to `[u, v]` with its lower bounds kept.
    ///
    /// A finite patch reads as a length in the scene's world-space unit. A
    /// patch at infinity (`w == 0`) is tangent to the unit sphere around its
    /// direction, so an offset `a` along an axis from the centre is the
    /// direction at `atan(a)` from the centre's, which reads as an angle in
    /// degrees (`a` radians, to first order).
    ///
    /// `None` for a `resolution` of 0 or a half-extent that is not positive and
    /// finite.
    pub fn patch_axis_reach(
        &self,
        placement: &OrientedPatch,
        resolution: usize,
    ) -> Option<PatchAxisReach> {
        let half = placement.half_extent;
        if resolution == 0 || !half.iter().all(|h| h.is_finite() && *h > 0.0) {
            return None;
        }
        let grid = self.grid_axes();
        let scaled: [BoundedLength; 2] = std::array::from_fn(|k| BoundedLength {
            value: grid[k].value * 2.0 * half[k] / resolution as f64,
            at_least: grid[k].at_least,
        });
        Some(if placement.w == 0.0 {
            PatchAxisReach::Angle(scaled.map(|length| BoundedLength {
                value: length.value.atan().to_degrees(),
                ..length
            }))
        } else {
            PatchAxisReach::Length(scaled)
        })
    }
}

/// Where the region at `level` of the `(2r + 1)²` `surface` reaches a shift
/// with no reading, and how far that leaves the contour open.
///
/// Only gaps next to a shift at the level count: a shift with no reading whose
/// read neighbours are all below the level is treated as below it, as the
/// radius treats it.
fn openings(surface: &[f64], r: usize, level: f64) -> Openings {
    const STEPS: [[i64; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    let side = 2 * r + 1;
    let ri = r as i64;
    let inside = |[dx, dy]: [i64; 2]| dx.abs() <= ri && dy.abs() <= ri;
    let index = |[dx, dy]: [i64; 2]| ((dy + ri) as usize) * side + (dx + ri) as usize;
    let reading = |cell: [i64; 2]| -> Option<f64> {
        if !inside(cell) {
            return None;
        }
        let z = surface[index(cell)];
        z.is_finite().then_some(z)
    };
    let mut out = Openings::default();
    let mut gathered = vec![false; side * side];
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if !reading([dx, dy]).is_some_and(|z| z >= level) {
                continue;
            }
            for [ex, ey] in STEPS {
                let next = [dx + ex, dy + ey];
                if !inside(next) {
                    // The region runs off the square along axis `j`.
                    out.escapes = true;
                    let j = if ex != 0 { 0 } else { 1 };
                    out.axis_open[j] = true;
                    if !holds_its_width(&reading, level, r, [dx, dy], j) {
                        out.axis_open[1 - j] = true;
                    }
                } else if reading(next).is_none() && !gathered[index(next)] {
                    // A gap inside the square: gather it, and the read shifts
                    // around it, which bound any crossing in it.
                    gathered[index(next)] = true;
                    let mut stack = vec![next];
                    while let Some(cell) = stack.pop() {
                        out.enclosed.push(cell.map(|v| v as f64));
                        for [fx, fy] in STEPS {
                            let beside = [cell[0] + fx, cell[1] + fy];
                            if !inside(beside) {
                                out.escapes = true;
                                out.axis_open = [true, true];
                            } else if reading(beside).is_some() {
                                out.enclosed.push(beside.map(|v| v as f64));
                            } else if !gathered[index(beside)] {
                                gathered[index(beside)] = true;
                                stack.push(beside);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Whether the region at the level that runs off the square at `cell`, along
/// grid axis `j`, may be taken to grow no wider along the other axis `k`
/// beyond the square: its extent along `k` on the border line (`d_j = ±r`)
/// lies within its extent on the line just inside it (`d_j = ±(r − 1)`), each
/// read as the crossings interpolated at the two ends of the run of shifts at
/// the level, as the contour reads them. The readings say nothing past the
/// border, so a region that keeps or narrows its width over its last two lines,
/// to within [`WIDTH_SLACK`], is extrapolated to keep doing so; one that
/// widens, shifts sideways, or has a run end that cannot be read (at the
/// square's corner, or beside a shift with no reading) is not.
fn holds_its_width(
    reading: &impl Fn([i64; 2]) -> Option<f64>,
    level: f64,
    r: usize,
    cell: [i64; 2],
    j: usize,
) -> bool {
    if r == 0 {
        return false;
    }
    let k = 1 - j;
    let border = cell[j];
    let inner = border - border.signum();
    let at = |line: i64, along: i64| {
        let mut c = [0; 2];
        c[j] = line;
        c[k] = along;
        reading(c)
    };
    let Some((border_low, border_high, a, b)) = run_extent(&at, border, cell[k], level) else {
        return false;
    };
    if !(a..=b).all(|t| at(inner, t).is_some_and(|z| z >= level)) {
        return false;
    }
    let Some((inner_low, inner_high, ..)) = run_extent(&at, inner, a, level) else {
        return false;
    };
    border_low >= inner_low - WIDTH_SLACK && border_high <= inner_high + WIDTH_SLACK
}

/// How far, in grid px, a run end on the border line may lie outside the one
/// on the line inside it and still count as holding its width.
///
/// The kernels sum in `f32`, so a surface that is the same on every line in
/// exact arithmetic, a ridge of uniform stripes, reads crossings that differ
/// from line to line by about 1e-7 grid px, and more where the surface falls
/// through the level shallowly, since a crossing's error is the reading's
/// divided by the step it is interpolated across. 1e-4 is a thousand times
/// the rounding seen on such a ridge, and a region that widens by that much
/// per line would take ten thousand lines past the border to widen by one
/// grid px, so no widening that could change a reported reach is excused.
const WIDTH_SLACK: f64 = 1e-4;

/// The run of shifts at the level along one line through `start`, as `(low,
/// high, a, b)`: the shifts `a..=b` along the line, and the crossings at the
/// two ends, interpolated as the contour's are. `None` where a neighbour past
/// either end has no reading.
fn run_extent(
    at: &impl Fn(i64, i64) -> Option<f64>,
    line: i64,
    start: i64,
    level: f64,
) -> Option<(f64, f64, i64, i64)> {
    let above = |t: i64| at(line, t).is_some_and(|z| z >= level);
    let (mut a, mut b) = (start, start);
    while above(a - 1) {
        a -= 1;
    }
    while above(b + 1) {
        b += 1;
    }
    let (za, below_a) = (at(line, a)?, at(line, a - 1)?);
    let (zb, below_b) = (at(line, b)?, at(line, b + 1)?);
    let low = a as f64 - (za - level) / (za - below_a);
    let high = b as f64 + (zb - level) / (zb - below_b);
    Some((low, high, a, b))
}

/// One self-similarity reading's contour, measured in each unit that can be
/// computed for it: its reach.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfSimilarityReach {
    /// The radius in grid px, [`SelfSimilarityContour::radius`]: the value is
    /// exactly [`SelfSimilarity::radius`].
    pub grid_radius: BoundedLength,
    /// `[max |dx|, max |dy|]` in grid px, [`SelfSimilarityContour::grid_axes`].
    pub grid_axes: [BoundedLength; 2],
    /// The radius in source-image px,
    /// [`SelfSimilarityContour::image_radius`]; `None` without a usable
    /// Jacobian.
    pub image_radius: Option<BoundedLength>,
    /// The reach along the patch's `u` and `v` axes,
    /// [`SelfSimilarityContour::patch_axis_reach`]; `None` without a
    /// placement.
    pub patch_axes: Option<PatchAxisReach>,
}

impl SelfSimilarityReach {
    /// Measure `reading`'s contour: in grid px always, in image px through
    /// `jacobian` (image px per grid px,
    /// [`crate::camera::warp_map::patch_grid_jacobian`]) where there
    /// is one, and along `placement`'s axes where there is one. `placement`
    /// and `resolution` describe the tile's `R × R` core, the placement the
    /// tile was rendered through. `None` where the reading has no contour
    /// ([`SelfSimilarity::contour`]).
    ///
    /// ```
    /// # use sfmtool_core::camera::CameraIntrinsics;
    /// # use sfmtool_core::geometry::RigidTransform;
    /// # use sfmtool_core::patch::cloud::OrientedPatch;
    /// use sfmtool_core::camera::warp_map::patch_grid_jacobian;
    /// use sfmtool_core::patch::self_similarity::{
    ///     PatchAxisReach, SelfSimilarity, SelfSimilarityReach,
    /// };
    /// # fn bounds(reading: &SelfSimilarity, placement: &OrientedPatch,
    /// #           camera: &CameraIntrinsics, pose: &RigidTransform) {
    /// // `reading` is the whole core of a tile rendered through `placement`
    /// // with a 24 × 24 core.
    /// let jacobian = patch_grid_jacobian(placement, camera, pose, 24);
    /// if let Some(reach) = SelfSimilarityReach::read(reading, jacobian, Some(placement), 24) {
    ///     if let Some(PatchAxisReach::Length([u, v])) = reach.patch_axes {
    ///         // The point could sit up to u.value along u and v.value along
    ///         // v from where it is, in the scene's world-space unit.
    ///         let _ = (u, v);
    ///     }
    /// }
    /// # }
    /// ```
    pub fn read(
        reading: &SelfSimilarity,
        jacobian: Option<[[f64; 2]; 2]>,
        placement: Option<&OrientedPatch>,
        resolution: usize,
    ) -> Option<Self> {
        let contour = reading.contour()?;
        Some(Self {
            grid_radius: contour.radius(),
            grid_axes: contour.grid_axes(),
            image_radius: jacobian.and_then(|j| contour.image_radius(j)),
            patch_axes: placement.and_then(|p| contour.patch_axis_reach(p, resolution)),
        })
    }
}

fn norm_sq(v: [f64; 2]) -> f64 {
    v[0] * v[0] + v[1] * v[1]
}
