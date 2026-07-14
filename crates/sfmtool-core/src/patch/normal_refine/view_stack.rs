// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The multi-view patch render substrate ([`PatchViewStack`]): render every kept
//! view of a patch once, then score it ([`PatchViewStack::score`]) and fuse the
//! representative RGBA texture ([`PatchViewStack::fuse`]) without re-rendering.

use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear, remap_bilinear_mip};
use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;

use super::consensus::{consensus_phi_with_weights, ConsensusScratch};
use super::level::LevelContext;
use super::params::{NormalRefineParams, ProjectedImage, Sampler, MAX_ANISOTROPY};
use super::prof;
use super::support::view_render_patch;
use super::znorm::znormalize_into;

/// Contrast scale (in 0–255 colour units) of the per-pixel agreement confidence:
/// the weighted cross-view colour RMS deviation `σ` at which agreement decays to
/// `e^{-1/2}`. Larger tolerates more cross-view colour spread before the alpha
/// drops.
pub(in crate::patch) const AGREEMENT_SIGMA: f64 = 24.0;

/// Map a rendered source pixel to RGB, replicating a single channel to grey and
/// taking the first three of a multi-channel image (matching the thumbnail RGB
/// convention).
fn sample_rgb(img: &crate::camera::remap::ImageU8, col: u32, row: u32, channels: u32) -> [f64; 3] {
    match channels {
        0 => [0.0; 3],
        1 | 2 => {
            let g = img.get_pixel(col, row, 0) as f64;
            [g, g, g]
        }
        _ => [
            img.get_pixel(col, row, 0) as f64,
            img.get_pixel(col, row, 1) as f64,
            img.get_pixel(col, row, 2) as f64,
        ],
    }
}

/// The appearance of one oriented patch as each observing camera sees it: every
/// kept view warped into the patch's `R×R` `(s, t)` grid, with per-pixel
/// validity. Rendering is the dominant cost of refinement, so this is the single
/// substrate the operations that all need "the patch from each view" read from,
/// instead of re-rendering:
///
/// - photoconsistency scoring and the robust IRLS view weights ([`Self::score`],
///   over a frozen masked support);
/// - the fused representative texture ([`Self::fuse`], over the full grid).
///
/// It is the natural input for future per-view operations too — cross-validation
/// strips, the per-pixel robust template (`patch-normal-refinement.md` item 7),
/// per-view patch export, and GPU textured-surfel rendering.
///
/// Shared with the sibling `keypoint_subpixel` module (via the
/// `normal_refine` re-export), which fuses each point's representative bitmap
/// at the final refined keypoints.
pub(in crate::patch) struct PatchViewStack {
    resolution: u32,
    /// One full `R×R` render per kept view.
    images: Vec<crate::camera::remap::ImageU8>,
    /// Full-grid per-pixel validity per kept view (row-major, length `R²`),
    /// parallel to `images`.
    valid: Vec<Vec<bool>>,
    /// Channels shared across the consensus: the min over kept views.
    channels: usize,
}

impl PatchViewStack {
    /// Render `patch` into every view in `kept` (full grid + per-pixel validity).
    pub(in crate::patch) fn render(
        patch: &OrientedPatch,
        views: &[ProjectedImage<'_>],
        kept: &[usize],
        resolution: u32,
        sampler: Sampler,
        view_keypoints: Option<&[Option<[f64; 2]>]>,
    ) -> Self {
        let r = resolution as usize;
        let npix = r * r;
        prof::count(&prof::N_RENDER, kept.len() as u64);
        // The consensus space is the channel count shared by every kept view (its
        // min), exactly as `normalized_stack`; `0` for an empty stack.
        let channels = kept
            .iter()
            .map(|&vi| views[vi].pyramid.level(0).channels() as usize)
            .min()
            .unwrap_or(0);
        let mut images = Vec::with_capacity(kept.len());
        let mut valid = Vec::with_capacity(kept.len());
        for &vi in kept {
            let view = &views[vi];
            let rpatch = view_render_patch(patch, view, view_keypoints.and_then(|k| k[vi]));
            let mut map = prof::WARP.time(|| {
                WarpMap::from_patch(&rpatch, view.camera, view.cam_from_world, resolution)
            });
            let mut vmask = vec![false; npix];
            for row in 0..resolution {
                for col in 0..resolution {
                    vmask[(row * resolution + col) as usize] = map.is_valid(col, row);
                }
            }
            let img = match sampler {
                Sampler::Anisotropic => {
                    prof::SVD.time(|| map.compute_svd());
                    prof::REMAP
                        .time(|| remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY))
                }
                Sampler::BilinearMip => {
                    prof::SVD.time(|| map.compute_svd());
                    prof::REMAP.time(|| remap_bilinear_mip(view.pyramid, &map))
                }
                Sampler::Bilinear => {
                    prof::REMAP.time(|| remap_bilinear(view.pyramid.level(0), &map))
                }
            };
            images.push(img);
            valid.push(vmask);
        }
        Self {
            resolution,
            images,
            valid,
            channels,
        }
    }

    /// Gather the masked `pixels` into the flat consensus layout
    /// `[(view*channels + channel)*n + pixel]` (`n = pixels.len()`). `None` when a
    /// masked pixel is invalid in any kept view (the frame-edge rejection that
    /// keeps `Φ` comparable over the frozen mask) or no channel is present —
    /// matching [`normalized_stack`].
    fn gather(&self, pixels: &[usize]) -> Option<Vec<f32>> {
        let n_views = self.images.len();
        let channels = self.channels;
        if channels == 0 || n_views == 0 {
            return None;
        }
        let r = self.resolution as usize;
        let n = pixels.len();
        // Reject if any masked pixel falls out of frame in any kept view.
        for vmask in &self.valid {
            if pixels.iter().any(|&p| !vmask[p]) {
                prof::count(&prof::N_REJECT, 1);
                return None;
            }
        }
        let mut raw = vec![0f32; n_views * channels * n];
        prof::ZNORM.time(|| {
            for (vk, img) in self.images.iter().enumerate() {
                for c in 0..channels {
                    let base = (vk * channels + c) * n;
                    for (ki, &p) in pixels.iter().enumerate() {
                        raw[base + ki] =
                            img.get_pixel((p % r) as u32, (p / r) as u32, c as u32) as f32;
                    }
                }
            }
        });
        Some(raw)
    }

    /// Consensus `Φ` over the frozen support `ctx` and the per-view weights that
    /// produced it. Mirrors [`eval_phi`] (the same render → z-normalize →
    /// consensus), but reads the already-rendered images and also returns the
    /// weights, so the representative fusion reuses them. `None` when the support
    /// can't be scored.
    /// `view_priors` (if given, in the stack's view order — the `ctx.kept` order
    /// the stack was rendered in) is the multiplicative obliquity view-weight (A)
    /// for the robust consensus; `None` runs it prior-free.
    pub(super) fn score(
        &self,
        ctx: &LevelContext,
        params: &NormalRefineParams,
        view_priors: Option<&[f64]>,
    ) -> Option<(f64, Vec<f64>)> {
        let raw = self.gather(&ctx.pixels)?;
        let n = ctx.pixels.len();
        let total_weight: f64 = ctx.weights.iter().sum();
        if total_weight <= 0.0 {
            return None;
        }
        let sqrt_weights: Vec<f32> = ctx.weights.iter().map(|&w| w.sqrt() as f32).collect();
        let mut xs = Vec::new();
        let kept = prof::ZNORM.time(|| {
            znormalize_into(
                &raw,
                self.images.len(),
                self.channels,
                n,
                &ctx.weights,
                total_weight,
                &sqrt_weights,
                &mut xs,
            )
        })?;
        let mut sc = ConsensusScratch::default();
        prof::CONSENSUS.time(|| {
            consensus_phi_with_weights(
                &xs,
                self.images.len(),
                kept,
                n,
                params.objective,
                view_priors,
                &mut sc,
            )
        })
    }

    /// Fuse the kept views into an `R×R` RGBA representative: per-pixel cross-view
    /// weighted-mean colour (RGB) and a cross-view *agreement* confidence (alpha).
    /// `weights` are the per-view consensus weights (parallel to the stack's
    /// views); only views that cover a pixel contribute, renormalized there. Alpha
    /// is `0` where no kept view covers a pixel and for a pixel seen by a single
    /// view (no cross-view evidence). Returns the flat `R·R·4` row-major texture.
    pub(in crate::patch) fn fuse(&self, weights: &[f64], sigma: f64) -> Vec<u8> {
        let r = self.resolution as usize;
        let npix = r * r;
        let mut out = vec![0u8; npix * 4];
        for p in 0..npix {
            let col = (p % r) as u32;
            let row = (p / r) as u32;
            // One pass: weighted colour sum and sum-of-squares over covering views.
            let mut wsum = 0.0;
            let mut s = [0.0f64; 3];
            let mut s2 = [0.0f64; 3];
            let mut n_cover = 0u32;
            for (vk, img) in self.images.iter().enumerate() {
                let w = weights.get(vk).copied().unwrap_or(1.0);
                if !self.valid[vk][p] || w <= 0.0 {
                    continue;
                }
                n_cover += 1;
                wsum += w;
                let rgb = sample_rgb(img, col, row, img.channels());
                for c in 0..3 {
                    s[c] += w * rgb[c];
                    s2[c] += w * rgb[c] * rgb[c];
                }
            }
            if n_cover == 0 || wsum <= 0.0 {
                continue; // no coverage: leave RGB and alpha zero
            }
            let mean = [s[0] / wsum, s[1] / wsum, s[2] / wsum];
            // Weighted per-channel variance E[x²]−E[x]², averaged over RGB; the
            // fp difference can dip slightly negative, so floor at 0.
            let var = (0..3)
                .map(|c| (s2[c] / wsum - mean[c] * mean[c]).max(0.0))
                .sum::<f64>()
                / 3.0;
            let rms = var.sqrt();
            let agreement = (-(rms * rms) / (2.0 * sigma * sigma)).exp();
            let coverage = 1.0 - 1.0 / n_cover as f64; // 1 view -> 0, 2 -> 0.5, ...
            let alpha = (255.0 * agreement * coverage).round().clamp(0.0, 255.0) as u8;
            let base_idx = p * 4;
            out[base_idx] = mean[0].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 1] = mean[1].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 2] = mean[2].round().clamp(0.0, 255.0) as u8;
            out[base_idx + 3] = alpha;
        }
        out
    }
}
