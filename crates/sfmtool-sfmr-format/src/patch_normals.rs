// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The normals a version 16 file derives from the patch frame.
//!
//! From [`SFMR_DERIVED_PATCH_NORMALS_VERSION`](crate::SFMR_DERIVED_PATCH_NORMALS_VERSION),
//! in a file with both `points3d/normals_xyz` and the patch frame, the normal
//! of a finite point (`w ≠ 0`) whose `u` is non-zero is not stored: its row
//! holds `(0, 0, 0)` and a reader fills it with `normalize(u × v)`, computed
//! in `f32` by [`patch_normal`]. A writer sets the value it is handed to the
//! derived normals in place, so after a write the value is what a reader of
//! the file gets back. Rows of points at infinity, of points without a patch,
//! and of points whose frame is degenerate (`u × v` zero or not finite) are
//! stored as before; the format does not otherwise constrain a frame, so a
//! degenerate one is kept rather than refused. See `specs/formats/sfmr-file-format.md` § "Per-point patch
//! frame".

/// Whether a file of `version` derives the normals of its finite patched
/// points from the frame.
pub(crate) const fn derives_patch_normals(version: u32) -> bool {
    version >= crate::SFMR_DERIVED_PATCH_NORMALS_VERSION
}

/// `normalize(u × v)` in `f32`, or `None` where the cross product is zero or
/// not finite.
///
/// The cross product is `(uy·vz − uz·vy, uz·vx − ux·vz, ux·vy − uy·vx)`, its
/// length `sqrt(nx² + ny² + nz²)` summed in that order, and each component is
/// divided by the length: one fixed sequence of `f32` operations, so every
/// reader and writer gets the same bits.
pub(crate) fn patch_normal(u: [f32; 3], v: [f32; 3]) -> Option<[f32; 3]> {
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let normal = [n[0] / len, n[1] / len, n[2] / len];
    (len > 0.0 && len.is_finite() && normal.iter().all(|c| c.is_finite())).then_some(normal)
}

/// The per-point frame and positions the derived normals are read from: the
/// `w` of each point and the `(P, 3)` row-major `u` and `v`.
pub(crate) struct PatchFrameRows<'a> {
    pub(crate) w: &'a [f64],
    pub(crate) u: &'a [f32],
    pub(crate) v: &'a [f32],
}

impl PatchFrameRows<'_> {
    fn row(a: &[f32], i: usize) -> [f32; 3] {
        [a[3 * i], a[3 * i + 1], a[3 * i + 2]]
    }

    /// Point `i`'s derived normal, `Some` where it is finite (`w ≠ 0`), its
    /// `u` is non-zero (the format's test for a point that has a patch) and
    /// its `u × v` gives a normal ([`patch_normal`]). `None` for a point at
    /// infinity, a point without a patch, and a degenerate frame, whose
    /// normal is stored.
    pub(crate) fn derived(&self, i: usize) -> Option<[f32; 3]> {
        let u = Self::row(self.u, i);
        (self.w[i] != 0.0 && u != [0.0; 3])
            .then(|| patch_normal(u, Self::row(self.v, i)))
            .flatten()
    }

    /// Set every derived row of `normals`, `(P, 3)` row-major, to its derived
    /// normal: what a writer does to the value it is handed, and a reader to
    /// the zero rows it read. Every other row is left as it is.
    pub(crate) fn fill(&self, normals: &mut [f32]) {
        for i in 0..self.w.len() {
            if let Some(n) = self.derived(i) {
                normals[3 * i..3 * i + 3].copy_from_slice(&n);
            }
        }
    }

    /// `normals` as a version 16 file stores them: every derived row `(0, 0,
    /// 0)`, every other row as it is.
    pub(crate) fn stored(&self, normals: &[f32]) -> Vec<f32> {
        let mut out = normals.to_vec();
        for i in 0..self.w.len() {
            if self.derived(i).is_some() {
                out[3 * i..3 * i + 3].fill(0.0);
            }
        }
        out
    }

    /// Check the stored `normals` of a version 16 file: every derived row is
    /// `(0, 0, 0)`.
    pub(crate) fn check_stored(&self, normals: &[f32]) -> Result<(), String> {
        for i in 0..self.w.len() {
            let row = Self::row(normals, i);
            if self.derived(i).is_some() && row.iter().any(|&c| c != 0.0) {
                return Err(format!(
                    "points3d/normals_xyz row {i} is {row:?}, but point {i} is finite and has \
                     a patch frame that gives a normal, so a version {} file stores (0, 0, 0) \
                     there and derives the normal from the frame",
                    crate::SFMR_DERIVED_PATCH_NORMALS_VERSION
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_is_the_normalised_cross_product() {
        assert_eq!(
            patch_normal([2.0, 0.0, 0.0], [0.0, 0.5, 0.0]),
            Some([0.0, 0.0, 1.0])
        );
        let n = patch_normal([0.1, 0.2, 0.0], [0.0, 0.3, 0.4]).unwrap();
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "{n:?}");
        assert!(n[0] > 0.0 && n[1] < 0.0 && n[2] > 0.0, "{n:?}");
    }

    #[test]
    fn a_degenerate_frame_gives_no_normal() {
        assert_eq!(patch_normal([1.0, 0.0, 0.0], [0.0; 3]), None);
        assert_eq!(patch_normal([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), None);
        assert_eq!(patch_normal([f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]), None);
        assert_eq!(
            patch_normal([f32::MAX, 0.0, 0.0], [0.0, f32::MAX, 0.0]),
            None
        );
    }
}
