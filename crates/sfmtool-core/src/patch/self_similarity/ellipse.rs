// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The ellipse a ZNCC self-similarity reading summarises its region by: the
//! ellipse with the same second moments per unit area, about the true position
//! `d = 0`, as the region of shifts where the self-similarity surface is at or
//! above its level `1 − tolerance`. Its semi-major axis is the radius.
//!
//! The region is the surface's part at or above the level, the surface taken
//! as bilinear between the whole-pixel shifts. Its area and second moments
//! are integrated cell by cell over the square of shifts (see
//! [`region_moments`]). The ellipse maps through any linear map as an ellipse,
//! so the same reading is measured in source-image px through the Jacobian of
//! the tile's warp and along the patch's own axes through its half-extents.
//! See `specs/core/patch/zncc-self-similarity-radius.md`, "The region, its
//! ellipse and the radius", "The ellipse in other units" and "Lower bounds".

use crate::patch::cloud::OrientedPatch;

/// The ellipse of a self-similarity reading's region: centred on the true
/// position, with the same second moments per unit area about it as the region
/// of shifts that match the template as well as a true match between two views
/// would. `dᵀ E⁻¹ d = 1` is its boundary, with `E` [`Self::matrix`].
///
/// In grid px it is in the grid's frame (`x` column-right, `y` row-down);
/// [`Self::mapped`] takes it into another frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfSimilarityEllipse {
    /// `[semi-major, semi-minor]`, each capped at the largest radius searched
    /// in grid px, so a length of `max_radius` keeps its meaning, "this far or
    /// further". `NaN` for a reading with no data.
    pub axes: [f64; 2],
    /// Per axis, whether the true length may be larger than
    /// [`Self::axes`] says: the region at the level runs off the square of
    /// shifts searched, a gap with no reading beside it could hide more of
    /// it, or the length reached the cap. `false` for a reading with no data.
    pub axes_is_at_least: [bool; 2],
    /// The angle of the major axis from the frame's first axis towards its
    /// second, in radians in `[0, π)`. `NaN` where it has no direction: the
    /// two axes equal (a circle, a flat template), or no data.
    pub major_angle: f64,
    /// `E`, symmetric, in the frame's units squared: `4·M` for the region's
    /// second-moment matrix `M` per unit area, rebuilt from the capped axes
    /// where an axis is capped. Its eigenvalues are the squared semi-axes.
    pub matrix: [[f64; 2]; 2],
}

/// The ellipse along the patch's `u` and `v` axes ([`SelfSimilarityEllipse::on_patch`]),
/// with its angle from `u` towards `v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchEllipse {
    /// A finite patch (`w == 1`): lengths in the unit of the patch's
    /// half-extents, the scene's world-space unit (`world_space_unit`), or
    /// scene units where the scene names none.
    Length(SelfSimilarityEllipse),
    /// A patch at infinity (`w == 0`), which has no length: each semi-axis is
    /// the angle at the eye it spans, in degrees, and the matrix is rebuilt
    /// from those angles.
    Angle(SelfSimilarityEllipse),
}

impl PatchEllipse {
    /// The ellipse, whatever its unit.
    pub fn ellipse(&self) -> &SelfSimilarityEllipse {
        match self {
            PatchEllipse::Length(e) | PatchEllipse::Angle(e) => e,
        }
    }
}

/// One reading's ellipse in each unit that can be computed for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfSimilarityEllipseUnits {
    /// In grid px, the reading's own [`super::SelfSimilarity::ellipse`].
    pub grid_px: SelfSimilarityEllipse,
    /// In source-image px through the Jacobian of the tile's warp at its
    /// centre ([`SelfSimilarityEllipse::mapped`]); `None` without a usable
    /// Jacobian.
    pub image_px: Option<SelfSimilarityEllipse>,
    /// Along the patch's `u` and `v` ([`SelfSimilarityEllipse::on_patch`]);
    /// `None` without a placement.
    pub patch: Option<PatchEllipse>,
}

impl SelfSimilarityEllipseUnits {
    /// Measure `reading`'s ellipse: in grid px always, in image px through
    /// `jacobian` (image px per grid px,
    /// [`crate::camera::warp_map::patch_grid_jacobian`]) where there is one,
    /// and along `placement`'s axes where there is one. `placement` and
    /// `resolution` describe the `R × R` bitmap read: the placement it was
    /// rendered through, and `R`. `None` for a reading with no data.
    ///
    /// ```
    /// # use sfmtool_core::camera::CameraIntrinsics;
    /// # use sfmtool_core::geometry::RigidTransform;
    /// # use sfmtool_core::patch::cloud::OrientedPatch;
    /// use sfmtool_core::camera::warp_map::patch_grid_jacobian;
    /// use sfmtool_core::patch::self_similarity::{
    ///     PatchEllipse, SelfSimilarity, SelfSimilarityEllipseUnits,
    /// };
    /// # fn bounds(reading: &SelfSimilarity, placement: &OrientedPatch,
    /// #           camera: &CameraIntrinsics, pose: &RigidTransform) {
    /// // `reading` is the whole of a 24 × 24 bitmap rendered through
    /// // `placement`.
    /// let jacobian = patch_grid_jacobian(placement, camera, pose, 24);
    /// if let Some(units) = SelfSimilarityEllipseUnits::read(reading, jacobian, Some(placement), 24) {
    ///     if let Some(PatchEllipse::Length(on_patch)) = units.patch {
    ///         // A match could land up to on_patch.axes[0] from the point
    ///         // along the patch, in the scene's world-space unit, in the
    ///         // direction on_patch.major_angle from u towards v.
    ///         let _ = on_patch;
    ///     }
    /// }
    /// # }
    /// ```
    pub fn read(
        reading: &super::SelfSimilarity,
        jacobian: Option<[[f64; 2]; 2]>,
        placement: Option<&OrientedPatch>,
        resolution: usize,
    ) -> Option<Self> {
        let grid_px = reading.ellipse;
        if !grid_px.axes.iter().all(|a| a.is_finite()) {
            return None;
        }
        Some(Self {
            grid_px,
            image_px: jacobian.and_then(|j| grid_px.mapped(j)),
            patch: placement.and_then(|p| grid_px.on_patch(p, resolution)),
        })
    }
}

impl SelfSimilarityEllipse {
    /// The ellipse with semi-axes `axes` (`[major, minor]`), their flags, and
    /// its major axis at `major_angle`, its matrix rebuilt from them: how a
    /// stored reading, which keeps the axes and the angle, is read back. With
    /// a `NaN` angle the axes are taken as equal and the larger is used for
    /// both in the matrix.
    pub fn from_axes(axes: [f64; 2], axes_is_at_least: [bool; 2], major_angle: f64) -> Self {
        Self {
            axes,
            axes_is_at_least,
            major_angle,
            matrix: matrix_of(axes, major_angle),
        }
    }

    /// The ellipse of a reading with no data: `NaN` throughout.
    pub(super) fn no_data() -> Self {
        Self {
            axes: [f64::NAN; 2],
            axes_is_at_least: [false; 2],
            major_angle: f64::NAN,
            matrix: [[f64::NAN; 2]; 2],
        }
    }

    /// The ellipse of a template with no textured channel, every shift of
    /// which counts: a circle of the largest radius searched, `r` or more
    /// along both axes, with no direction.
    pub(super) fn flat(r: usize) -> Self {
        let r = r as f64;
        Self {
            axes: [r, r],
            axes_is_at_least: [true, true],
            major_angle: f64::NAN,
            matrix: [[r * r, 0.0], [0.0, r * r]],
        }
    }

    /// The ellipse taken through the linear map `map`, `L E Lᵀ`: the same
    /// region seen in another frame. Through `jacobian`, the image px per grid
    /// px at the tile's centre (`[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]`, as
    /// [`crate::camera::warp_map::patch_grid_jacobian`] returns it), it is the
    /// ellipse in source-image px, its angle from the image's `x` towards its
    /// `y` (row-down).
    ///
    /// Its axes are lower bounds as the grid's are, carried through the map:
    /// the new major axis is one where either grid axis is. The new minor axis
    /// is one where the grid minor axis is, and also where only the grid major
    /// axis is and the map does not carry the grid's major direction onto the
    /// new major direction: lengthening the ellipse along a direction that
    /// does not map onto its new major axis lengthens its new minor axis too.
    ///
    /// `None` for a map that is not finite or is singular, one so large that
    /// `L E Lᵀ` overflows, or an ellipse with no data.
    pub fn mapped(&self, map: [[f64; 2]; 2]) -> Option<SelfSimilarityEllipse> {
        let [[a, b], [c, d]] = map;
        let det = a * d - b * c;
        if !(map.iter().flatten().all(|v| v.is_finite()) && det != 0.0 && det.is_finite())
            || !self.matrix.iter().flatten().all(|v| v.is_finite())
        {
            return None;
        }
        let m = self.matrix;
        // L E.
        let le = [
            [a * m[0][0] + b * m[1][0], a * m[0][1] + b * m[1][1]],
            [c * m[0][0] + d * m[1][0], c * m[0][1] + d * m[1][1]],
        ];
        // (L E) Lᵀ, symmetric.
        let xx = le[0][0] * a + le[0][1] * b;
        let xy = le[0][0] * c + le[0][1] * d;
        let yy = le[1][0] * c + le[1][1] * d;
        let matrix = [[xx, xy], [xy, yy]];
        // A map large enough to overflow `L E Lᵀ` gives no ellipse, rather
        // than an infinite axis, or a `NaN` that `axes_of` would read as 0.
        if !matrix.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        let (axes, major_angle) = axes_of(matrix);
        let [major_open, minor_open] = self.axes_is_at_least;
        let keeps_major = || {
            if self.major_angle.is_nan() {
                return false;
            }
            let (s, co) = self.major_angle.sin_cos();
            let w = [a * co + b * s, c * co + d * s];
            let len = w[0].hypot(w[1]);
            if len == 0.0 {
                return false;
            }
            // n ⟂ L·v: the new ellipse's extent along n does not change when
            // the grid ellipse grows along v, and bounds its minor axis.
            let n = [-w[1] / len, w[0] / len];
            let across = n[0] * n[0] * xx + 2.0 * n[0] * n[1] * xy + n[1] * n[1] * yy;
            // The allowance scales with the ellipse, so the test reads the same
            // in any unit: a relative 1e-9 of the minor axis, and 1e-12 of the
            // major axis for a minor axis at or near 0.
            across.max(0.0).sqrt() <= axes[1] * (1.0 + 1e-9) + 1e-12 * axes[0]
        };
        Some(SelfSimilarityEllipse {
            axes,
            axes_is_at_least: [
                major_open || minor_open,
                minor_open || (major_open && !keeps_major()),
            ],
            major_angle,
            matrix,
        })
    }

    /// The ellipse along `placement`'s `u` and `v` axes, for a reading of a
    /// `resolution × resolution` bitmap rendered through `placement`: one grid
    /// px along `x` is `2·half_extent[0] / resolution` along `u`, and along
    /// `y` is `2·half_extent[1] / resolution` along `−v`
    /// (`WarpMap::from_patch` steps the rows down `v`), so it is
    /// [`Self::mapped`] through `diag(2·h₀/R, −2·h₁/R)`, its angle from `u`
    /// towards `v`.
    ///
    /// A finite patch reads as lengths in the scene's world-space unit. A
    /// patch at infinity (`w == 0`) is tangent to the unit sphere around its
    /// direction, so an offset `a` along the patch from its centre is the
    /// direction at `atan(a)` from the centre's: each semi-axis reads as that
    /// angle in degrees (`a` radians, to first order).
    ///
    /// `None` for a `resolution` of 0, a half-extent that is not positive and
    /// finite, or an ellipse with no data.
    pub fn on_patch(&self, placement: &OrientedPatch, resolution: usize) -> Option<PatchEllipse> {
        let half = placement.half_extent;
        if resolution == 0 || !half.iter().all(|h| h.is_finite() && *h > 0.0) {
            return None;
        }
        let r = resolution as f64;
        let along = self.mapped([[2.0 * half[0] / r, 0.0], [0.0, -2.0 * half[1] / r]])?;
        Some(if placement.w == 0.0 {
            let axes = along.axes.map(|a| a.atan().to_degrees());
            PatchEllipse::Angle(SelfSimilarityEllipse {
                axes,
                matrix: matrix_of(axes, along.major_angle),
                ..along
            })
        } else {
            PatchEllipse::Length(along)
        })
    }
}

/// How far, in grid px, a length bounding what the readings cannot see may
/// exceed the value read and still leave it exact.
///
/// The kernels sum in `f32`, so a surface that is the same on every line in
/// exact arithmetic, a ridge of uniform stripes, reads crossings that differ
/// from line to line by about 1e-7 grid px, and more where the surface falls
/// through the level shallowly, since a crossing's error is the reading's
/// divided by the step it is interpolated across. 1e-4 is a thousand times
/// the rounding seen on such a ridge, and a region that widens by that much
/// per line would take ten thousand lines past the border to widen by one
/// grid px, so no widening that could change a reported length is excused.
const LENGTH_SLACK: f64 = 1e-4;

/// The ellipse of the region of the `(2r + 1)²` `surface` at or above `level`,
/// with its lower bounds.
pub(super) fn fit_ellipse(surface: &[f64], r: usize, level: f64) -> SelfSimilarityEllipse {
    let moments = region_moments(surface, r, level);
    let open = openings(surface, r, level);
    let m = if moments.area > 0.0 {
        [
            [moments.sxx / moments.area, moments.sxy / moments.area],
            [moments.sxy / moments.area, moments.syy / moments.area],
        ]
    } else {
        [[0.0; 2]; 2]
    };
    let e = m.map(|row| row.map(|v| 4.0 * v));
    let (axes, major_angle) = axes_of(e);
    let [a, b] = axes;

    // The major axis: the region may continue past the border, or a gap may
    // hide area further from the centre than the ellipse reaches. λ_max of a
    // mixture is at most the larger of the parts', and the unseen part's is
    // at most its largest |d|².
    let major_open = open.escapes || open.gap_points.iter().any(|c| 2.0 * norm_sq(*c).sqrt() > a);

    // The minor axis: λ_min of the true region is at most uᵀ M u for any unit
    // u, and that is at most the larger of the seen region's and the unseen
    // part's mean (u·d)². Exact where some u keeps that bound within the
    // minor axis.
    let minor_bound = |u: [f64; 2], continuation: f64| {
        let seen = u[0] * u[0] * m[0][0] + 2.0 * u[0] * u[1] * m[0][1] + u[1] * u[1] * m[1][1];
        let unseen = open
            .gap_points
            .iter()
            .map(|c| (u[0] * c[0] + u[1] * c[1]).powi(2))
            .fold(continuation, f64::max);
        2.0 * seen.max(unseen).max(0.0).sqrt()
    };
    let mut directions: Vec<([f64; 2], f64)> = Vec::new();
    if !open.escapes {
        if !major_angle.is_nan() {
            let (s, c) = major_angle.sin_cos();
            directions.push(([-s, c], 0.0));
        }
        directions.push(([1.0, 0.0], 0.0));
        directions.push(([0.0, 1.0], 0.0));
    } else {
        // A run-off along one grid axis only, holding its width, is taken to
        // continue past the border with the border line's run: across it, the
        // continuation's mean square is that run's.
        for k in 0..2 {
            if open.axis_open[1 - k] && !open.axis_open[k] {
                let mut u = [0.0; 2];
                u[k] = 1.0;
                directions.push((u, open.continuation_sq[k]));
            }
        }
    }
    let minor_open = !directions
        .iter()
        .any(|&(u, continuation)| minor_bound(u, continuation) <= b + LENGTH_SLACK);

    let cap = r as f64;
    let capped = [a.min(cap), b.min(cap)];
    let axes_is_at_least = [major_open || a >= cap, minor_open || b >= cap];
    let matrix = if a > cap || b > cap {
        matrix_of(capped, major_angle)
    } else {
        e
    };
    SelfSimilarityEllipse {
        axes: capped,
        axes_is_at_least,
        major_angle,
        matrix,
    }
}

/// The semi-axes `[major, minor]` of the ellipse with matrix `e`, the square
/// roots of its eigenvalues, and the angle of its major axis in `[0, π)`,
/// `NaN` where the two are equal to within `1e-12` of their size.
fn axes_of(e: [[f64; 2]; 2]) -> ([f64; 2], f64) {
    let half_trace = 0.5 * (e[0][0] + e[1][1]);
    let spread = (0.25 * (e[0][0] - e[1][1]).powi(2) + e[0][1] * e[0][1]).sqrt();
    let axes = [
        (half_trace + spread).max(0.0).sqrt(),
        (half_trace - spread).max(0.0).sqrt(),
    ];
    let angle = if half_trace > 0.0 && spread > 1e-12 * half_trace {
        let theta = 0.5 * (2.0 * e[0][1]).atan2(e[0][0] - e[1][1]);
        let theta = if theta < 0.0 {
            theta + std::f64::consts::PI
        } else {
            theta
        };
        // A tiny negative angle can round to π, which is the same axis as 0.
        if theta >= std::f64::consts::PI {
            0.0
        } else {
            theta
        }
    } else {
        f64::NAN
    };
    (axes, angle)
}

/// The matrix of the ellipse with semi-axes `[major, minor]` and its major
/// axis at `angle`; with no angle, the axes are taken as equal and the larger
/// is used for both.
fn matrix_of([a, b]: [f64; 2], angle: f64) -> [[f64; 2]; 2] {
    if angle.is_nan() {
        let s = a.max(b);
        return [[s * s, 0.0], [0.0, s * s]];
    }
    let (s, c) = angle.sin_cos();
    let (a2, b2) = (a * a, b * b);
    let xy = (a2 - b2) * c * s;
    [[a2 * c * c + b2 * s * s, xy], [xy, a2 * s * s + b2 * c * c]]
}

/// The area and second moments about the centre of a region.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct RegionMoments {
    pub(super) area: f64,
    pub(super) sxx: f64,
    pub(super) sxy: f64,
    pub(super) syy: f64,
}

impl RegionMoments {
    /// Add a region whose moments `local` are about the point `origin` and in
    /// axes parallel to the centre's.
    fn add_at(&mut self, origin: [f64; 2], local: &LocalMoments) {
        let [ox, oy] = origin;
        self.area += local.area;
        self.sxx += local.spp + 2.0 * ox * local.sp + ox * ox * local.area;
        self.syy += local.sqq + 2.0 * oy * local.sq + oy * oy * local.area;
        self.sxy += local.spq + ox * local.sq + oy * local.sp + ox * oy * local.area;
    }
}

/// Area, first and second moments of a region about a local origin, in axes
/// `(p, q)` parallel to the grid's.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LocalMoments {
    pub(super) area: f64,
    pub(super) sp: f64,
    pub(super) sq: f64,
    pub(super) spp: f64,
    pub(super) spq: f64,
    pub(super) sqq: f64,
}

impl LocalMoments {
    /// The unit square `[0, 1]²`.
    const UNIT_SQUARE: Self = Self {
        area: 1.0,
        sp: 0.5,
        sq: 0.5,
        spp: 1.0 / 3.0,
        spq: 0.25,
        sqq: 1.0 / 3.0,
    };

    /// The part of the unit cell `[0, 1]²` where the bilinear surface through
    /// the corner values `[v00, v10, v11, v01]` (counter-clockwise from the
    /// origin, `p` first) is at or above `level`.
    ///
    /// Along each line of constant `q` the surface is linear in `p`, `A(q) +
    /// B(q)·p` with `A` and `B` linear in `q`, so the part at or above the
    /// level is one interval of `p`, `[p*(q), 1]` or `[0, p*(q)]` with `p*`
    /// clamped to the cell, and its moments along `p` are exact. The `q`
    /// where `p*` meets the cell's sides or `B` changes sign split the cell
    /// into pieces. On a piece where the interval is the whole side or empty,
    /// the moments across `q` are exact too. On one where `p*` lies inside the
    /// cell, `p* = (level − A)/B` is a rational function of `q` whose pole,
    /// where `B = 0`, lies outside the piece but may lie close to it, where the
    /// level's curve bends sharply round a saddle. That piece is integrated by
    /// [`GAUSS_LEGENDRE`]'s eight nodes over spans halved towards the pole until
    /// each is no longer than its distance from it, which keeps the rule's
    /// error under about `1e-10` of a cell's area.
    fn bilinear_part(values: [f64; 4], level: f64) -> Self {
        Self::bilinear_part_graded(values, level, 1.0)
    }

    /// [`Self::bilinear_part`] with its spans halved until each is no longer
    /// than `1 / grading` of its distance from the pole.
    pub(super) fn bilinear_part_graded(values: [f64; 4], level: f64, grading: f64) -> Self {
        let [v00, v10, v11, v01] = values;
        let (a0, a1) = (v00, v01 - v00);
        let (b0, b1) = (v10 - v00, (v11 - v01) - (v10 - v00));
        let mut breaks = [0.0f64; 5];
        let mut count = 0;
        breaks[count] = 0.0;
        count += 1;
        // p* = 0, p* = 1, and B = 0: each a linear equation in q.
        for (c0, c1) in [(a0 - level, a1), (a0 + b0 - level, a1 + b1), (b0, b1)] {
            if c1 != 0.0 {
                let q = -c0 / c1;
                if q > 0.0 && q < 1.0 {
                    breaks[count] = q;
                    count += 1;
                }
            }
        }
        breaks[count] = 1.0;
        count += 1;
        let breaks = &mut breaks[..count];
        breaks.sort_by(f64::total_cmp);
        let pole = (b1 != 0.0).then(|| -b0 / b1);

        // The interval of p at or above the level along the line at q.
        let interval = |q: f64| {
            let (a, b) = (a0 + a1 * q, b0 + b1 * q);
            let (lo, hi) = if b > 0.0 {
                (((level - a) / b).clamp(0.0, 1.0), 1.0)
            } else if b < 0.0 {
                (0.0, ((level - a) / b).clamp(0.0, 1.0))
            } else if a >= level {
                (0.0, 1.0)
            } else {
                (0.0, 0.0)
            };
            (lo, hi.max(lo))
        };
        let mut out = Self::default();
        // A stack of spans still to integrate; halving pushes two and pops one,
        // so it holds at most one more span than the halvings so far.
        let mut spans = [(0.0f64, 0.0f64, 0u32); MAX_HALVINGS as usize + 2];
        for piece in breaks.windows(2) {
            let (q0, q1) = (piece[0], piece[1]);
            if q1 <= q0 {
                continue;
            }
            match interval(0.5 * (q0 + q1)) {
                (lo, hi) if lo == hi => continue,
                (lo, hi) if lo == 0.0 && hi == 1.0 => {
                    let (d1, d2, d3) = (q1 - q0, q1 * q1 - q0 * q0, q1 * q1 * q1 - q0 * q0 * q0);
                    out.area += d1;
                    out.sp += 0.5 * d1;
                    out.spp += d1 / 3.0;
                    out.sq += 0.5 * d2;
                    out.spq += 0.25 * d2;
                    out.sqq += d3 / 3.0;
                    continue;
                }
                _ => {}
            }
            spans[0] = (q0, q1, 0);
            let mut top = 1usize;
            while top > 0 {
                top -= 1;
                let (qa, qb, depth) = spans[top];
                let near = pole.map_or(f64::INFINITY, |qp| (qa - qp).max(qp - qb).max(0.0));
                if grading * (qb - qa) > near && depth < MAX_HALVINGS {
                    let mid = 0.5 * (qa + qb);
                    spans[top] = (mid, qb, depth + 1);
                    spans[top + 1] = (qa, mid, depth + 1);
                    top += 2;
                    continue;
                }
                let (half, centre) = (0.5 * (qb - qa), 0.5 * (qa + qb));
                for (node, weight) in GAUSS_LEGENDRE {
                    let q = centre + half * node;
                    let wgt = half * weight;
                    let (lo, hi) = interval(q);
                    let length = hi - lo;
                    let first = 0.5 * (hi * hi - lo * lo);
                    out.area += wgt * length;
                    out.sp += wgt * first;
                    out.spp += wgt * (hi * hi * hi - lo * lo * lo) / 3.0;
                    out.sq += wgt * q * length;
                    out.spq += wgt * q * first;
                    out.sqq += wgt * q * q * length;
                }
            }
        }
        out
    }
}

/// The eight-node Gauss–Legendre rule on `[−1, 1]`, `(node, weight)`.
const GAUSS_LEGENDRE: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
];

/// How many times a span of [`LocalMoments::bilinear_part`] is halved towards
/// the pole at most: a span of `2⁻⁴⁰` of a cell holds no area a reading could
/// see.
const MAX_HALVINGS: u32 = 40;

/// The area and second moments about `d = 0` of the region of the
/// `(2r + 1)²` `surface` at or above `level`, the surface taken as bilinear
/// between the whole-pixel shifts.
///
/// Each cell of the square of shifts, a unit square between four
/// neighbouring shifts, adds its part of the region. A cell wholly at or
/// above the level adds the whole square, exactly, and one wholly below adds
/// nothing. A cell the level crosses adds [`LocalMoments::bilinear_part`]. A
/// cell with a corner that has no finite reading adds nothing: [`openings`]
/// accounts for what it may hide.
pub(super) fn region_moments(surface: &[f64], r: usize, level: f64) -> RegionMoments {
    let side = 2 * r + 1;
    let ri = r as i64;
    let mut out = RegionMoments::default();
    let at = |dx: i64, dy: i64| surface[((dy + ri) as usize) * side + (dx + ri) as usize];
    for y in -ri..ri {
        for x in -ri..ri {
            // Counter-clockwise from (x, y): v00, v10, v11, v01.
            let v = [at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)];
            if !v.iter().all(|z| z.is_finite()) {
                continue;
            }
            let count = v.iter().filter(|&&z| z >= level).count();
            let part = match count {
                0 => continue,
                4 => LocalMoments::UNIT_SQUARE,
                _ => LocalMoments::bilinear_part(v, level),
            };
            out.add_at([x as f64, y as f64], &part);
        }
    }
    out
}

/// Where the region at the level reaches a shift with no reading, and so
/// where the true region may be larger than the one integrated.
#[derive(Debug, Clone, PartialEq, Default)]
struct Openings {
    /// The region may continue past the square's border: a shift at the level
    /// on the border, or a connected set of shifts with no reading next to
    /// one at the level that reaches the border.
    escapes: bool,
    /// Per grid axis, `[x, y]`, whether an escape may carry the region
    /// further along that axis than the readings show: the region runs off
    /// along it, or runs off along the other axis without holding its width
    /// (`holds_its_width`), or through a gap that reaches the border.
    axis_open: [bool; 2],
    /// Per grid axis `k`, over the run-offs along the other axis that hold
    /// their width, the largest mean square of `d_k` across the border line's
    /// run, which a continuation with that run past the border has.
    continuation_sq: [f64; 2],
    /// The corners of every cell a gap with no reading next to a shift at the
    /// level, diagonals included, could hide part of the region in, where the gap does not reach
    /// the border: the shifts within one step, diagonals included, of each
    /// shift of the gap.
    gap_points: Vec<[f64; 2]>,
}

/// Where the region at `level` of the `(2r + 1)²` `surface` reaches a shift
/// with no reading, and how far that leaves it open.
///
/// Only gaps next to a shift at the level, diagonals included, count: a shift
/// with no reading whose eight read neighbours are all below the level is
/// treated as below it.
fn openings(surface: &[f64], r: usize, level: f64) -> Openings {
    const STEPS: [[i64; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const NEIGHBOURS: [[i64; 2]; 8] = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ];
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
                    match holds_its_width(&reading, level, r, [dx, dy], j) {
                        Some((low, high)) => {
                            let k = 1 - j;
                            let mean_sq = if high > low {
                                (high.powi(3) - low.powi(3)) / (3.0 * (high - low))
                            } else {
                                low * low
                            };
                            out.continuation_sq[k] = out.continuation_sq[k].max(mean_sq);
                        }
                        None => out.axis_open[1 - j] = true,
                    }
                }
            }
            // A gap at any of the eight neighbours shares a cell with this
            // shift, and that cell adds nothing to the region.
            for [ex, ey] in NEIGHBOURS {
                let next = [dx + ex, dy + ey];
                if inside(next) && reading(next).is_none() && !gathered[index(next)] {
                    // A gap inside the square: gather it, and the shifts
                    // around it, the corners of the cells it could hide part
                    // of the region in.
                    gathered[index(next)] = true;
                    let mut stack = vec![next];
                    while let Some(cell) = stack.pop() {
                        for fy in -1..=1 {
                            for fx in -1..=1 {
                                let corner = [cell[0] + fx, cell[1] + fy];
                                if inside(corner) {
                                    out.gap_points.push(corner.map(|v| v as f64));
                                }
                            }
                        }
                        for [fx, fy] in STEPS {
                            let beside = [cell[0] + fx, cell[1] + fy];
                            if !inside(beside) {
                                out.escapes = true;
                                out.axis_open = [true, true];
                            } else if reading(beside).is_none() && !gathered[index(beside)] {
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
/// beyond the square, and if so the border line's run, `(low, high)` along
/// `k`: its extent along `k` on the border line (`d_j = ±r`) lies within its
/// extent on the line just inside it (`d_j = ±(r − 1)`), each read as the
/// crossings interpolated at the two ends of the run of shifts at the level.
/// The readings say nothing past the border, so a region that keeps or narrows
/// its width over its last two lines, to within [`LENGTH_SLACK`], is
/// extrapolated to continue with the border line's run; one that widens,
/// shifts sideways, or has a run end that cannot be read (at the square's
/// corner, or beside a shift with no reading) is not.
fn holds_its_width(
    reading: &impl Fn([i64; 2]) -> Option<f64>,
    level: f64,
    r: usize,
    cell: [i64; 2],
    j: usize,
) -> Option<(f64, f64)> {
    if r == 0 {
        return None;
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
    let (border_low, border_high, a, b) = run_extent(&at, border, cell[k], level)?;
    if !(a..=b).all(|t| at(inner, t).is_some_and(|z| z >= level)) {
        return None;
    }
    let (inner_low, inner_high, ..) = run_extent(&at, inner, a, level)?;
    (border_low >= inner_low - LENGTH_SLACK && border_high <= inner_high + LENGTH_SLACK)
        .then_some((border_low, border_high))
}

/// The run of shifts at the level along one line through `start`, as `(low,
/// high, a, b)`: the shifts `a..=b` along the line, and the crossings at the
/// two ends, interpolated linearly along the grid edge past each end. `None`
/// where a neighbour past either end has no reading.
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

fn norm_sq(v: [f64; 2]) -> f64 {
    v[0] * v[0] + v[1] * v[1]
}
