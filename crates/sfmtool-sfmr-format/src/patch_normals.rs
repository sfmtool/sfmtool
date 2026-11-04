// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The normals a version 16 file derives from the patch frame.
//!
//! From [`SFMR_DERIVED_PATCH_NORMALS_VERSION`](crate::SFMR_DERIVED_PATCH_NORMALS_VERSION),
//! in a file with both normals and the patch frame, the normal of a finite
//! point (`w ≠ 0`) whose frame gives one is not stored: a reader computes it
//! as `normalize(u × v)` in `f32` by [`patch_normal`]. The file stores only
//! the normals of the other finite points, the underived normals, in point
//! order in `points3d/underived_normals_xyz.{K}.3.float32.zst`, and no entry
//! at all when there are none; a point at infinity's normal is `(0, 0, 0)`
//! and not stored. A writer sets the value it is handed to the normals a
//! reader gives back, so after a write the value is what a reader of the file
//! gets back. A point whose frame is degenerate (`u × v` zero or not finite)
//! stores its normal; the format does not otherwise constrain a frame, so a
//! degenerate one is kept rather than refused. See
//! `specs/formats/sfmr-file-format.md` § "Derived normals (version 16+)".

/// Whether a file of `version` derives the normals of its finite patched
/// points from the frame.
pub(crate) const fn derives_patch_normals(version: u32) -> bool {
    version >= crate::SFMR_DERIVED_PATCH_NORMALS_VERSION
}

/// The prefix of the `points3d/underived_normals_xyz` entry, whatever its K.
pub(crate) const UNDERIVED_NORMALS_PREFIX: &str = "points3d/underived_normals_xyz.";

/// Why the normals entries of a file are refused, or `None` where they are
/// the ones its version and flags call for.
///
/// `names` is every entry name in the archive. `underived_count` is `Some(K)`
/// for a file of version 16 or later with both `has_normals` and
/// `has_uv_frames`, `K` its finite points whose normal the frame does not
/// give, and `None` for any other file. Such a file stores
/// `points3d/underived_normals_xyz.{K}.3.float32.zst` exactly when `K > 0`
/// and no full `points3d/normals_xyz` column; any other file stores no
/// underived entry. The reader and the verifier both ask this.
pub(crate) fn normals_entries_error<'a>(
    version: u32,
    point_count: usize,
    underived_count: Option<usize>,
    names: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    let names: Vec<&str> = names.into_iter().collect();
    let underived: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with(UNDERIVED_NORMALS_PREFIX))
        .collect();
    let Some(k) = underived_count else {
        return underived.first().map(|found| {
            format!(
                "{found} is stored only by a file of version {} or later with both normals \
                 and the patch frame, and this is a version {version} file without them",
                crate::SFMR_DERIVED_PATCH_NORMALS_VERSION
            )
        });
    };
    let full = crate::entries::points3d_normals(false, point_count);
    if names.contains(&full.as_str()) {
        return Some(format!(
            "a version {version} file with the patch frame stores the normals its frame does \
             not give in points3d/underived_normals_xyz, and this one holds the full column \
             {full}"
        ));
    }
    let expected = (k > 0).then(|| crate::entries::points3d_underived_normals(k));
    if let Some(found) = underived.iter().find(|n| Some(n.to_string()) != expected) {
        return Some(format!(
            "{found} does not match the {k} finite points whose normal the patch frame does \
             not give{}",
            expected
                .as_ref()
                .map_or(", which a file stores no entry for".to_string(), |e| {
                    format!(", which a file stores as {e}")
                })
        ));
    }
    match &expected {
        Some(e) if !underived.contains(&e.as_str()) => Some(format!(
            "this file has {k} finite points whose normal the patch frame does not give, and \
             no {e} to hold them"
        )),
        _ => None,
    }
}

/// `normalize(u × v)` in `f32`, or `None` where the cross product is zero or
/// not finite.
///
/// The cross product is `(uy·vz − uz·vy, uz·vx − ux·vz, ux·vy − uy·vx)`, its
/// length `sqrt(nx² + ny² + nz²)` summed in that order, and each component is
/// divided by the length: one fixed sequence of `f32` operations, so every
/// reader and writer gets the same bits.
///
/// A version 16 file derives the normal of a finite point with a patch by
/// this function, and a caller that fills a missing normal in memory as a
/// write would calls it too, so the two agree to the bit.
pub fn patch_normal(u: [f32; 3], v: [f32; 3]) -> Option<[f32; 3]> {
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

    /// Set the rows of `normals`, `(P, 3)` row-major, that a version 16 file
    /// does not store to what a reader gives back for them: each derived row
    /// to its derived normal, and each point at infinity's row to `(0, 0, 0)`.
    /// What a writer does to the value it is handed. Every other row is left
    /// as it is.
    pub(crate) fn fill(&self, normals: &mut [f32]) {
        for i in 0..self.w.len() {
            if let Some(n) = self.derived(i) {
                normals[3 * i..3 * i + 3].copy_from_slice(&n);
            } else if self.w[i] == 0.0 {
                normals[3 * i..3 * i + 3].fill(0.0);
            }
        }
    }

    /// The points whose normals a version 16 file stores, in point order:
    /// the finite points whose normal is not derived, because they have no
    /// patch (`u` zero) or a frame that gives no normal.
    pub(crate) fn underived(&self) -> Vec<usize> {
        (0..self.w.len())
            .filter(|&i| self.w[i] != 0.0 && self.derived(i).is_none())
            .collect()
    }

    /// The rows of `normals`, `(P, 3)` row-major, that a version 16 file
    /// stores: those of [`Self::underived`], in point order, `(K, 3)`.
    pub(crate) fn underived_rows(&self, normals: &[f32]) -> Vec<f32> {
        self.underived()
            .into_iter()
            .flat_map(|i| Self::row(normals, i))
            .collect()
    }

    /// The `(P, 3)` normals a reader gives back from `stored`, the `(K, 3)`
    /// row-major rows of [`Self::underived`]: row `k` of `stored` for the
    /// `k`-th of those points, the derived normal for a point with a usable
    /// frame, and `(0, 0, 0)` for a point at infinity. A message where
    /// `stored` does not hold one row per such point.
    pub(crate) fn rebuild(&self, stored: &[f32]) -> Result<Vec<f32>, String> {
        let underived = self.underived();
        if stored.len() != 3 * underived.len() {
            return Err(format!(
                "points3d/underived_normals_xyz holds {} values, not the {} of the {} finite \
                 points whose normal the patch frame does not give",
                stored.len(),
                3 * underived.len(),
                underived.len()
            ));
        }
        let mut normals = vec![0.0f32; 3 * self.w.len()];
        self.fill(&mut normals);
        for (k, i) in underived.into_iter().enumerate() {
            normals[3 * i..3 * i + 3].copy_from_slice(&stored[3 * k..3 * k + 3]);
        }
        Ok(normals)
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
