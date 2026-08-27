// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Selection queries over a built [`ClusterCovisibility`]: the isolation
//! sweep order, banded thinning, and reach.
//!
//! Separated from construction because these answer a different question —
//! *which* images to keep rather than *how covisible* they are. See
//! `specs/core/features/covisibility-selection.md`.

use super::ClusterCovisibility;

impl ClusterCovisibility {
    /// The thinning sweep order: decreasing isolation — largest
    /// nearest-covisible-partner displacement first (an image's isolation is
    /// the *smallest* sampled mean displacement to any partner; no sampled
    /// partner means infinitely isolated), ties and the no-positions case
    /// falling back to construction order (ascending index).
    fn sweep_order(&self) -> Vec<u32> {
        let n = self.num_images;
        let mut order: Vec<u32> = (0..n as u32).collect();
        if let Some(tables) = &self.displacement {
            let isolation: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .filter(|&j| tables.count[i * n + j] > 0)
                        .map(|j| tables.mean[i * n + j])
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            order.sort_by(|&a, &b| {
                isolation[b as usize]
                    .partial_cmp(&isolation[a as usize])
                    .expect("isolation values are never NaN")
                    .then(a.cmp(&b))
            });
        }
        order
    }

    /// [`Self::thin`] against a precomputed sweep order.
    fn thin_in_order(&self, order: &[u32], tau: f64) -> Vec<u32> {
        let n = self.num_images;
        let mut kept: Vec<u32> = Vec::new();
        for &i in order {
            if kept.is_empty() {
                kept.push(i);
                continue;
            }
            let row = &self.counts[i as usize * n..(i as usize + 1) * n];
            let best = kept
                .iter()
                .map(|&k| row[k as usize])
                .max()
                .expect("kept is non-empty") as f64;
            if tau / 8.0 <= best && best < tau {
                kept.push(i);
            }
        }
        kept.sort_unstable();
        kept
    }

    /// Redundancy-thinned subset (sorted ascending): a greedy sweep in
    /// decreasing isolation (see the spec's Thinning section) keeps an image
    /// only when its best shared-cluster count against the already-kept set
    /// falls in the band `[tau/8, tau)` — images above the band duplicate a
    /// kept viewpoint, images below it are disconnected from the skeleton.
    /// The first swept image is always kept.
    pub fn thin(&self, tau: f64) -> Vec<u32> {
        self.thin_in_order(&self.sweep_order(), tau)
    }

    /// Thin to approximately `target` images: binary-search `tau` (the kept
    /// count grows monotonically with `tau`) over `[1, median row peak]` and
    /// return the subset whose size lands closest to `target` (sorted
    /// ascending; earlier iterations win exact-distance ties).
    pub fn thin_to(&self, target: usize) -> Vec<u32> {
        let n = self.num_images;
        if n == 0 {
            return Vec::new();
        }
        // Median (numpy-style: mean of the middle two for even n) of the
        // per-image peak covisibility.
        let mut peaks: Vec<u32> = (0..n)
            .map(|i| {
                self.counts[i * n..(i + 1) * n]
                    .iter()
                    .copied()
                    .max()
                    .expect("rows are non-empty")
            })
            .collect();
        peaks.sort_unstable();
        let med_peak = if n % 2 == 1 {
            peaks[n / 2] as f64
        } else {
            (peaks[n / 2 - 1] as f64 + peaks[n / 2] as f64) / 2.0
        };

        let order = self.sweep_order();
        let (mut lo, mut hi) = (1.0f64, med_peak);
        let mut best: Option<Vec<u32>> = None;
        for _ in 0..25 {
            let mid = (lo + hi) / 2.0;
            let keep = self.thin_in_order(&order, mid);
            let closer = best
                .as_ref()
                .is_none_or(|b| keep.len().abs_diff(target) < b.len().abs_diff(target));
            let below = keep.len() < target;
            if closer {
                best = Some(keep);
            }
            if below {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        best.expect("25 iterations always produce a candidate")
    }

    /// Fraction of all images sharing at least `min_shared` clusters with at
    /// least one image of `images` (subset members count as reached). An
    /// empty subset reaches nothing (`0.0`). Panics if any index is out of
    /// range.
    pub fn reach(&self, images: &[u32], min_shared: u32) -> f64 {
        let n = self.num_images;
        if images.is_empty() || n == 0 {
            return 0.0;
        }
        let mut connected = vec![false; n];
        for &s in images {
            assert!((s as usize) < n, "subset image index out of range");
            connected[s as usize] = true;
        }
        for (i, slot) in connected.iter_mut().enumerate() {
            *slot = *slot
                || images
                    .iter()
                    .any(|&s| self.counts[i * n + s as usize] >= min_shared);
        }
        connected.iter().filter(|&&c| c).count() as f64 / n as f64
    }
}
