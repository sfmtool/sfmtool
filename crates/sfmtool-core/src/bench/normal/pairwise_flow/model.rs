// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pair model: the flow from one crop to another that a candidate plane
//! through the point predicts, by exact ray-plane geometry on the round's
//! common plane, and its derivative with respect to the candidate's tilt.
//!
//! For a sample of crop `i` at grid coordinates `q`:
//!
//! 1. the sample's position `P` on the round's plane gives a ray from `C_i`;
//! 2. the ray meets the candidate plane, through `X` with normal `n`, at `Y`;
//! 3. the ray from `C_j` to `Y` meets the round's plane at a point whose grid
//!    coordinates `q'`, less `q`, are the predicted flow.
//!
//! A sample whose ray at step 2 or 3 is parallel to the plane it should meet,
//! or meets it behind the camera it leaves from, has no prediction. Where the
//! candidate is the round's plane every predicted flow is zero.
//!
//! Every quantity is `f64`: the model is evaluated a few thousand times per
//! fit iteration into 4×4 normal equations, and its derivative near edge-on
//! tilts subtracts nearly equal terms, which `f32` would leave with too few
//! digits; the cost at that size is negligible.

use nalgebra::{Matrix2x3, Vector3};

/// A ray meeting a plane at a cosine below this, relative to the two
/// vectors' lengths, is taken as parallel to it.
const MIN_INCIDENCE: f64 = 1e-9;

/// The round's plane through the point and the grid the crops share on it.
///
/// A sample at grid coordinates `(x, y)`, in grid samples from the crop's
/// centre, is the world point `point + spacing · (x · x_axis + y · y_axis)`.
/// `x` counts the crop's columns and `y` its rows, as
/// [`WarpMap::from_patch`](crate::camera::WarpMap::from_patch) lays them
/// out: `x_axis` is the crop's `u` and `y_axis` its `-v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CommonPlane {
    /// The point `X` the plane passes through, at the crop's centre.
    pub(crate) point: Vector3<f64>,
    /// The plane's unit normal.
    pub(crate) normal: Vector3<f64>,
    /// The unit in-plane direction of the grid's columns.
    pub(crate) x_axis: Vector3<f64>,
    /// The unit in-plane direction of the grid's rows.
    pub(crate) y_axis: Vector3<f64>,
    /// World units per grid sample.
    pub(crate) spacing: f64,
}

impl CommonPlane {
    /// The world point at grid coordinates `q`.
    pub(crate) fn world(&self, q: [f64; 2]) -> Vector3<f64> {
        self.point + (self.x_axis * q[0] + self.y_axis * q[1]) * self.spacing
    }

    /// The grid coordinates of the world point `p`, which is taken to lie on
    /// the plane.
    pub(crate) fn grid(&self, p: Vector3<f64>) -> [f64; 2] {
        let d = p - self.point;
        [
            d.dot(&self.x_axis) / self.spacing,
            d.dot(&self.y_axis) / self.spacing,
        ]
    }
}

/// Where a ray from `origin` through `through` meets the plane through
/// `point` with normal `normal`, as the ray's parameter `t` (the meeting is
/// `origin + t · (through − origin)`); `None` where the ray is parallel to
/// the plane or meets it behind `origin`.
fn meet(
    origin: Vector3<f64>,
    through: Vector3<f64>,
    point: Vector3<f64>,
    normal: Vector3<f64>,
) -> Option<f64> {
    let d = through - origin;
    let along = normal.dot(&d);
    if along.abs() <= MIN_INCIDENCE * d.norm() * normal.norm() {
        return None;
    }
    let t = normal.dot(&(point - origin)) / along;
    (t.is_finite() && t > 0.0).then_some(t)
}

/// The flow from crop `i` to crop `j` that a candidate plane predicts, for
/// one pair read at one pyramid level.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairModel {
    /// The round's plane the crops were rendered on.
    pub(crate) plane: CommonPlane,
    /// The centre of the camera the flow starts from, `C_i`.
    pub(crate) from: Vector3<f64>,
    /// The centre of the camera the flow ends in, `C_j`.
    pub(crate) to: Vector3<f64>,
    /// Grid samples per sample at the pair's level, `2^ℓ`: the flow is
    /// predicted in samples at that level.
    pub(crate) level_scale: f64,
}

impl PairModel {
    /// The predicted flow at grid coordinates `q` for the candidate plane of
    /// unit normal `n` through the point, in samples at the pair's level;
    /// `None` where either ray misses.
    pub(crate) fn flow(&self, q: [f64; 2], n: Vector3<f64>) -> Option<[f64; 2]> {
        self.trace(q, n).map(|traced| traced.flow)
    }

    /// [`Self::flow`] and its derivative with respect to turning `n` by small
    /// angles toward the unit vectors `a` and `b`, both perpendicular to `n`:
    /// the columns of the 2×2 matrix are `∂flow/∂α` and `∂flow/∂β`, in
    /// samples at the pair's level per radian.
    ///
    /// With `D = P − C_i`, `Y = C_i + t·D` and `t = n·(X − C_i) / n·D`, a
    /// turn `dn` moves `t` by `(X − Y)·dn / n·D` and so `Y` by `D` times
    /// that. With `E = Y − C_j`, `Z = C_j + t'·E` and
    /// `t' = n₀·(X − C_j) / n₀·E`, a move `dY` moves `Z` by
    /// `t'·(dY − E·(n₀·dY) / n₀·E)`, and `q'` by its in-plane components over
    /// the spacing.
    pub(crate) fn flow_and_jacobian(
        &self,
        q: [f64; 2],
        n: Vector3<f64>,
        a: Vector3<f64>,
        b: Vector3<f64>,
    ) -> Option<([f64; 2], [[f64; 2]; 2])> {
        let traced = self.trace(q, n)?;
        let plane = &self.plane;
        let d = traced.p - self.from;
        let e = traced.y - self.to;
        let dt = (plane.point - traced.y) / n.dot(&d);
        let n0_e = plane.normal.dot(&e);
        let scale = 1.0 / (plane.spacing * self.level_scale);
        let in_plane = Matrix2x3::from_rows(&[plane.x_axis.transpose(), plane.y_axis.transpose()]);
        let column = |turn: Vector3<f64>| -> [f64; 2] {
            let dy = d * dt.dot(&turn);
            let dz = (dy - e * (plane.normal.dot(&dy) / n0_e)) * traced.t_to;
            let dq = in_plane * dz * scale;
            [dq[0], dq[1]]
        };
        let (ja, jb) = (column(a), column(b));
        Some((traced.flow, [[ja[0], jb[0]], [ja[1], jb[1]]]))
    }

    fn trace(&self, q: [f64; 2], n: Vector3<f64>) -> Option<Traced> {
        let plane = &self.plane;
        let p = plane.world(q);
        let t = meet(self.from, p, plane.point, n)?;
        let y = self.from + (p - self.from) * t;
        let t_to = meet(self.to, y, plane.point, plane.normal)?;
        let z = self.to + (y - self.to) * t_to;
        let to = plane.grid(z);
        let flow = [
            (to[0] - q[0]) / self.level_scale,
            (to[1] - q[1]) / self.level_scale,
        ];
        Some(Traced { p, y, t_to, flow })
    }
}

/// One sample traced through the pair model.
struct Traced {
    /// The sample's position on the round's plane.
    p: Vector3<f64>,
    /// Where the ray from `C_i` meets the candidate plane.
    y: Vector3<f64>,
    /// The parameter of the ray from `C_j` through `y` at the round's plane.
    t_to: f64,
    /// The predicted flow, samples at the pair's level.
    flow: [f64; 2],
}

/// Two unit vectors perpendicular to the unit `n` and to each other, the
/// first along `hint` projected off `n` (or along a world axis where `hint`
/// is within a degree of `n`), the second `n × first`.
pub(crate) fn tangent_basis(n: Vector3<f64>, hint: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let mut a = hint - n * n.dot(&hint);
    if a.norm() < 0.017 * hint.norm().max(f64::MIN_POSITIVE) {
        let axis = if n.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        a = axis - n * n.dot(&axis);
    }
    let a = a.normalize();
    (a, n.cross(&a))
}

#[cfg(test)]
mod tests;
