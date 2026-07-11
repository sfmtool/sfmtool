// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust SIFT feature detector and descriptor.
//!
//! Reference: David G. Lowe, "Distinctive Image Features from Scale-Invariant
//! Keypoints," IJCV 60(2):91-110, 2004.
//!
//! This crate implements SIFT directly (no OpenCV/COLMAP round-trip), structured
//! to mirror the optical-flow module's house conventions: it operates on
//! [`GrayImage`] (reused from [`crate::features::optical_flow`]), uses separable Gaussian
//! blur with an SSE2 inner loop plus scalar fallback, and parallelizes with
//! rayon. See `specs/core/sift.md` for the authoritative design.
//!
//! The public interface follows COLMAP's conventions: keypoint coordinates use
//! the pixel-center convention (the upper-left pixel's center is `(0.5, 0.5)`),
//! and each keypoint's geometry is a 2x2 affine-shape matrix
//! `[[a11, a12], [a21, a22]]`. SIFT produces similarity-only keypoints, which map
//! onto that matrix as a scaled rotation
//! `[[s·cosθ, -s·sinθ], [s·sinθ, s·cosθ]]`.
//!
//! The full pipeline is implemented here: the scale space (Gaussian + DoG
//! pyramids) and image-to-gray conversion, keypoint detection and sub-pixel
//! localization, orientation assignment, and the 128-D descriptor, with the
//! orientation/descriptor sampling and the Gaussian blur SIMD-accelerated (see
//! [`simd`] and the parallelism/SIMD section of `specs/core/sift.md`).

mod descriptor;
mod detect;
pub mod gray;
mod orientation;
mod scale_space;
mod simd;

pub use gray::{gray_from_rgb, parse_gray_formula, GrayFormula, DEFAULT_GRAY_FORMULA};
pub use scale_space::ScaleSpace;

// Reuse the optical-flow grayscale image type rather than duplicating it; the
// coordinate convention (pixel centers at col+0.5, row+0.5) matches what SIFT
// needs.
pub use crate::features::optical_flow::GrayImage;

/// Env-gated stage timing for the keypoint-finding pipeline. Enabled by setting
/// the `SFMTOOL_SIFT_TIMING` environment variable; effectively zero-cost otherwise
/// (one cached bool check plus a few `Instant::now()` per image). When on, each
/// chain build (`ScaleSpace::build_chain`), each `detect_keypoints` call (whose
/// `octaves=detected/total` field reports the cap-aware early stop), and each
/// descriptor batch (`extract_sift_partial`) prints one `SIFT_TIMING ...` line
/// to stderr with per-stage wall-clock milliseconds, for offline aggregation.
pub(crate) static SIFT_TIMING: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_SIFT_TIMING").is_some());

/// Env-gated per-operator detail logging for scale-space build. Enabled by
/// setting the `SFMTOOL_SIFT_OPS` environment variable. When on, each scale-space
/// operator (`upsample`, `blur`, `decimate`) prints one `SIFT_OP ...`
/// line to stderr with its image dimensions, pixel count, and wall-clock time,
/// for offline aggregation. Independent of `SFMTOOL_SIFT_TIMING`.
pub(crate) static SIFT_OPS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_SIFT_OPS").is_some());

/// SIFT algorithm parameters. [`SiftParams::default`] returns Lowe (2004) values.
///
/// Value-domain parameters (`contrast_threshold`, descriptor clamp) assume the
/// gray image is on a `[0, 1]` scale, which is what the default image-to-gray
/// formula produces.
#[derive(Debug, Clone)]
pub struct SiftParams {
    /// Number of intervals `s` per octave. The Gaussian stack has `s + 3` levels
    /// and the DoG stack `s + 2`, so extrema are searched at `s` levels. Lowe's
    /// experimentally-optimal value is 3.
    pub octave_layers: u32,
    /// Base blur `σ` of the first level of octave 0. Default 1.6.
    pub sigma: f64,
    /// Gaussian blur kernel half-width in units of `σ`: each separable blur uses
    /// `radius = ceil(blur_radius_factor · σ)`, i.e. `2·radius + 1` taps. The
    /// default 2.25 keeps ~97.6% of the kernel mass and was chosen empirically as
    /// the narrowest setting that leaves descriptor agreement with COLMAP intact
    /// (3.0 is Lowe's wider ~99.7%-mass setting). Lower values trade a small blur
    /// error (the truncated kernel is renormalized) for fewer taps, which speeds
    /// up the blur — the dominant scale-space cost — most on the widest kernels.
    pub blur_radius_factor: f64,
    /// Assumed blur `σ_in` already present in the source image (anti-aliasing
    /// minimum). Default 0.5.
    pub input_sigma: f64,
    /// Whether to upsample the input 2x (bilinear) before octave 0, which
    /// increases the count of stable keypoints ~4x. Default true.
    pub double_image: bool,
    /// Contrast threshold `C`: discard extrema with `|D(x̂)| < C`. Lowe's paper
    /// uses 0.03, but that is ~2–4x stricter than the effective per-layer
    /// threshold the common implementations ship. This value is on the same
    /// scale as OpenCV's `contrastThreshold / nOctaveLayers` and COLMAP's
    /// `peak_threshold`; the default 0.0067 (≈ 0.02/3) matches COLMAP's density.
    pub contrast_threshold: f64,
    /// Edge threshold `r`: discard if the ratio of principal curvatures exceeds
    /// `r` (`Tr(H)²/Det(H) ≥ (r+1)²/r`). Default 10.
    pub edge_threshold: f64,
    /// Maximum number of features to keep per image, or `None` for unlimited.
    /// When detection yields more candidates than this, the largest-scale ones
    /// are kept (COLMAP's `max_num_features` selection). The cap is applied
    /// before orientation and description, so those stages only process the
    /// retained set. Default `Some(8192)`, matching COLMAP's default.
    pub max_num_features: Option<usize>,
    /// Number of bins `n_ori` in the orientation histogram. Default 36.
    pub orientation_bins: u32,
    /// Secondary-orientation peak ratio: emit an extra keypoint for every local
    /// peak within this fraction of the dominant peak. Default 0.8.
    pub peak_ratio: f64,
    /// Descriptor width `d`: the descriptor is a `d × d` array of histograms.
    /// Fixed at 4 — the descriptor produces a 128-D (`d·d·b`) vector, so the
    /// extraction entry points reject any other value.
    pub descriptor_width: u32,
    /// Orientation bins `b` per descriptor histogram. Fixed at 8 (so `d·d·b =
    /// 128`); other values are rejected (see `descriptor_width`).
    pub descriptor_bins: u32,
    /// Descriptor magnification `m_descr`: sample spacing per subregion in units
    /// of `σ_kp`. Default 3. Honored by descriptor computation.
    pub descriptor_magnification: f64,
    /// Descriptor component cap applied after L2 normalization, before renorm.
    /// Default 0.2. Honored by descriptor computation.
    pub descriptor_clamp: f64,
    /// The image-to-gray conversion formula (see [`gray`]). The default is the
    /// BT.709 luma matching COLMAP. This affects feature output and the meaning
    /// of value-domain thresholds, so it is part of the parameters.
    pub image_to_gray: GrayFormula,
}

impl Default for SiftParams {
    fn default() -> Self {
        Self {
            octave_layers: 3,
            sigma: 1.6,
            blur_radius_factor: 2.25,
            input_sigma: 0.5,
            double_image: true,
            contrast_threshold: 0.0067,
            edge_threshold: 10.0,
            max_num_features: Some(8192),
            orientation_bins: 36,
            peak_ratio: 0.8,
            descriptor_width: 4,
            descriptor_bins: 8,
            descriptor_magnification: 3.0,
            descriptor_clamp: 0.2,
            image_to_gray: GrayFormula::default(),
        }
    }
}

/// A detected, localized, and oriented SIFT keypoint.
///
/// Coordinates are in full-resolution image space using COLMAP's pixel-center
/// convention (the upper-left pixel's center is `(0.5, 0.5)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftKeypoint {
    /// Refined x coordinate, full-resolution, pixel-center convention.
    pub x: f32,
    /// Refined y coordinate, full-resolution, pixel-center convention.
    pub y: f32,
    /// Affine shape `[[a11, a12], [a21, a22]]` (COLMAP convention). For SIFT this
    /// is a scaled rotation: `[[s·cosθ, -s·sinθ], [s·sinθ, s·cosθ]]`.
    pub affine_shape: [[f32; 2]; 2],
    /// Octave index the keypoint was detected in (used to pick the pyramid level
    /// for description). Octave 0 is the (optionally doubled) base octave.
    pub octave: i32,
    /// Continuous sub-level (layer) within the octave.
    pub layer: f32,
    /// Contrast response `|D(x̂)|`, for ranking / feature-count caps.
    pub response: f32,
}

impl SiftKeypoint {
    /// Construct a keypoint from its similarity geometry, filling the affine
    /// shape as a scaled rotation `[[s·cosθ, -s·sinθ], [s·sinθ, s·cosθ]]`.
    ///
    /// `scale` is the keypoint size `s` (the affine-shape column norm), and
    /// `orientation` is in radians.
    pub fn from_similarity(
        x: f32,
        y: f32,
        scale: f32,
        orientation: f32,
        octave: i32,
        layer: f32,
        response: f32,
    ) -> Self {
        let (sin, cos) = orientation.sin_cos();
        let affine_shape = [[scale * cos, -scale * sin], [scale * sin, scale * cos]];
        Self {
            x,
            y,
            affine_shape,
            octave,
            layer,
            response,
        }
    }

    /// The keypoint scale (size), recovered from the affine shape as the average
    /// of its two column norms:
    /// `0.5 · (sqrt(a11² + a21²) + sqrt(a12² + a22²))`.
    pub fn scale(&self) -> f32 {
        let [[a11, a12], [a21, a22]] = self.affine_shape;
        let col0 = (a11 * a11 + a21 * a21).sqrt();
        let col1 = (a12 * a12 + a22 * a22).sqrt();
        0.5 * (col0 + col1)
    }

    /// The keypoint orientation in radians, recovered from the affine shape as
    /// `atan2(a21, a11)`.
    pub fn orientation(&self) -> f32 {
        let [[a11, _], [a21, _]] = self.affine_shape;
        a21.atan2(a11)
    }
}

/// A block of 128-D unsigned-byte SIFT descriptors, one row per keypoint.
///
/// Stored as `Vec<[u8; 128]>` (rather than an `ndarray::Array2`) because a
/// descriptor is naturally a fixed-size 128-byte row, the per-keypoint type used
/// throughout description, and this avoids a dependency edge in the public API.
#[derive(Debug, Clone, Default)]
pub struct Descriptors {
    rows: Vec<[u8; 128]>,
}

impl Descriptors {
    /// Wrap a vector of descriptor rows.
    pub fn from_rows(rows: Vec<[u8; 128]>) -> Self {
        Self { rows }
    }

    /// The descriptor rows.
    pub fn rows(&self) -> &[[u8; 128]] {
        &self.rows
    }

    /// The number of descriptors.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no descriptors.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// The result of [`detect_keypoints`]: the oriented keypoint pool plus the
/// retained scale space needed to describe any of them later.
pub struct Detection {
    /// Oriented keypoints (a single localized candidate may yield several when
    /// it has multiple dominant orientations).
    pub keypoints: Vec<SiftKeypoint>,
    /// The Gaussian scale space (with gradients) retained for lazy description.
    /// (The DoG is never stored — detection fuses it per stripe in cache.)
    pub scale_space: ScaleSpace,
}

/// The result of [`extract_sift`]: keypoints with their descriptors.
pub struct SiftFeatures {
    /// The oriented keypoints.
    pub keypoints: Vec<SiftKeypoint>,
    /// Parallel descriptors (`descriptors.rows()[i]` describes `keypoints[i]`).
    pub descriptors: Descriptors,
}

/// Detect, localize, and orient SIFT keypoints in `image`.
///
/// Builds the [`ScaleSpace`] (the Gaussian pyramid; gradients are sampled on the
/// fly), runs extrema detection + subpixel localization (`detect`, which fuses
/// the DoG per stripe in cache), then assigns orientation(s) (`orientation`). The
/// returned [`Detection`] retains the Gaussian pyramid so the caller can describe
/// any subset of keypoints later.
pub fn detect_keypoints(image: &GrayImage, params: &SiftParams) -> Detection {
    let timing = *SIFT_TIMING;

    let t = std::time::Instant::now();
    // Minimal pyramid skeleton (levels 0..=s per octave); each octave's last
    // two levels are built lazily right before that octave is detected, so a
    // skipped octave never pays for them.
    let mut scale_space = ScaleSpace::build_chain(image, params);
    let t_build = t.elapsed();

    // Stages 3–5 interleaved per octave, coarse→fine, with a cap-aware early
    // stop. Octave scale ranges are disjoint and ordered — octave o's
    // localized candidates span σ·k^[0.5, s+0.5]·2^o strictly (the localizer
    // clamps the integer layer to 1..=s and requires |offset| < 0.5) — so the
    // running top-`cap` candidate pool built coarse-to-fine equals the global
    // top-`cap` of a full scan (COLMAP `max_num_features`: keep the
    // largest-scale candidates, under a *total* order — scale, then response,
    // then octave/layer/y/x — so the retained set is deterministic when
    // scales tie; see the reproducible-`.sift` contract in
    // `specs/core/sift.md`). Each octave's surviving candidates are oriented
    // as soon as they are admitted (orientation preserves scale, so the
    // final keypoint ranking follows the candidate ranking), which lets two
    // guards stop the walk with output identical to scanning everything:
    //
    // 1. **Output-fixed:** ≥ `cap` keypoints exist and the cap-th largest
    //    keypoint scale strictly clears the next octave's `max_scale_bound` —
    //    every later keypoint would be cut by the final sort+truncate.
    // 2. **Candidate-set-fixed:** the candidate pool is full and its minimum
    //    scale strictly clears the bound — every later candidate would be cut
    //    before orientation (this also covers the corner where orientation
    //    yields no peak for some candidates, matching the full scan's
    //    behavior of orienting only the top-`cap` candidates).
    //
    // The bound carries a rounding margin; in the (never observed) event an
    // f32 seam collision makes a fine candidate tie the pool's minimum, the
    // slow path re-selects over the merged pool and re-orients it, so the
    // result is *always* the full scan's, not merely almost-always. On a
    // large image with the default cap the walk stops above the finest
    // octaves, which hold most of the pixels (with `double_image`, octaves 0
    // and 1 are ~94% of the pyramid) — skipping their detection scan and
    // their last two Gaussian levels.
    let t = std::time::Instant::now();
    let mut t_orient = std::time::Duration::ZERO;
    let mut pool: Vec<detect::LocalizedKeypoint> = Vec::new();
    let mut keypoints: Vec<SiftKeypoint> = Vec::new();
    let total_octaves = scale_space.num_octaves();
    let mut detected_octaves = 0usize;
    use std::cmp::Ordering::Equal;
    let candidate_order = |a: &detect::LocalizedKeypoint, b: &detect::LocalizedKeypoint| {
        b.scale
            .partial_cmp(&a.scale)
            .unwrap_or(Equal)
            .then(b.response.partial_cmp(&a.response).unwrap_or(Equal))
            .then(a.octave.cmp(&b.octave))
            .then(a.layer.partial_cmp(&b.layer).unwrap_or(Equal))
            .then(a.y.partial_cmp(&b.y).unwrap_or(Equal))
            .then(a.x.partial_cmp(&b.x).unwrap_or(Equal))
    };
    for o in (0..total_octaves).rev() {
        if let Some(cap) = params.max_num_features {
            if cap == 0 {
                // Degenerate cap: nothing can be kept, so nothing needs
                // detecting (the final truncate would empty the output anyway).
                break;
            }
            let bound = scale_space.max_scale_bound(o);
            if keypoints.len() >= cap {
                // Guard 1: the cap-th largest keypoint scale (select_nth is
                // O(n); the order it leaves behind is irrelevant — every
                // downstream selection and sort uses a total order).
                let (_, kth, _) = keypoints.select_nth_unstable_by(cap - 1, |a, b| {
                    b.scale().partial_cmp(&a.scale()).unwrap_or(Equal)
                });
                if kth.scale() > bound {
                    break;
                }
            }
            if pool.len() >= cap {
                // Guard 2: the pool minimum (the cap-th candidate).
                let pool_min = pool.iter().map(|c| c.scale).fold(f32::INFINITY, f32::min);
                if pool_min > bound {
                    break;
                }
            }
        }
        scale_space.extend_octave(o);
        let mut new = detect::detect_octave(&scale_space, params, o);
        detected_octaves += 1;
        // Dropping candidates from `new` (never orienting them) is sound only
        // when every pooled (coarser) candidate strictly outranks every new
        // one — verified by the same margin-guarded bound the break guards
        // use. If an f32 seam tie makes that uncertain (the pool is full and
        // guard 2 did not break, or a crossing octave overflows the cap while
        // the pool minimum does not clear the bound), fall back to a merged
        // re-selection and a from-scratch re-orientation, which reproduces
        // the full scan's single global cap exactly.
        let seam_merge = match params.max_num_features {
            Some(cap) if pool.len() >= cap => true,
            Some(cap) if pool.len() + new.len() > cap => {
                let pool_min = pool.iter().map(|c| c.scale).fold(f32::INFINITY, f32::min);
                pool_min <= scale_space.max_scale_bound(o)
            }
            _ => false,
        };
        if seam_merge {
            // Slow path (f32 seam collision only): merge, re-select the
            // top-`cap`, and re-orient the whole pool from scratch.
            let cap = params.max_num_features.unwrap();
            pool.append(&mut new);
            if pool.len() > cap {
                pool.select_nth_unstable_by(cap, candidate_order);
                pool.truncate(cap);
            }
            let to = std::time::Instant::now();
            keypoints = orientation::assign_orientations(&scale_space, &pool, params);
            t_orient += to.elapsed();
        } else {
            if let Some(cap) = params.max_num_features {
                // Admit only what fits: the crossing octave contributes its
                // top-(cap − len) candidates under the same total order the
                // full scan's single cap would use (sound here: the pool is
                // empty or its minimum strictly clears the octave's bound).
                let room = cap - pool.len();
                if new.len() > room {
                    new.select_nth_unstable_by(room, candidate_order);
                    new.truncate(room);
                }
            }
            let to = std::time::Instant::now();
            keypoints.extend(orientation::assign_orientations(&scale_space, &new, params));
            t_orient += to.elapsed();
            pool.append(&mut new);
        }
    }
    drop(pool);
    let t_detect = t.elapsed() - t_orient;

    // No DoG pyramid to free — detection computed it per stripe in cache. The
    // Gaussian pyramid is retained for orientation and description.

    // The `.sift` format requires keypoints sorted by descending feature size
    // (size = the affine-shape column-norm average, `SiftKeypoint::scale`). Sort
    // here so every consumer of `detect_keypoints` (and `extract_sift`) sees that
    // ordering. A total-order tie-break (response, octave, layer, y, x,
    // orientation) makes the output order — and thus the `.sift` bytes —
    // reproducible across runs and thread counts, independent of the upstream
    // (parallel) collection order. The derived comparison keys (`scale()` /
    // `orientation()` — a sqrt and an atan2) are computed once per keypoint
    // rather than per comparison; the comparator over the cached keys is the
    // same total order, so the result is unchanged.
    let t = std::time::Instant::now();
    let mut keyed: Vec<(f32, f32, SiftKeypoint)> = keypoints
        .into_iter()
        .map(|kp| (kp.scale(), kp.orientation(), kp))
        .collect();
    keyed.sort_by(|a, b| {
        let (a_scale, a_ori, a) = (a.0, a.1, &a.2);
        let (b_scale, b_ori, b) = (b.0, b.1, &b.2);
        b_scale
            .partial_cmp(&a_scale)
            .unwrap_or(Equal)
            .then(b.response.partial_cmp(&a.response).unwrap_or(Equal))
            .then(a.octave.cmp(&b.octave))
            .then(a.layer.partial_cmp(&b.layer).unwrap_or(Equal))
            .then(a.y.partial_cmp(&b.y).unwrap_or(Equal))
            .then(a.x.partial_cmp(&b.x).unwrap_or(Equal))
            .then(a_ori.partial_cmp(&b_ori).unwrap_or(Equal))
    });
    let mut keypoints: Vec<SiftKeypoint> = keyed.into_iter().map(|(_, _, kp)| kp).collect();
    // Orientation assignment can emit several keypoints per candidate, pushing
    // the count back above the cap; enforce it as a hard output limit. The sort
    // above already placed the largest-scale keypoints first.
    if let Some(cap) = params.max_num_features {
        keypoints.truncate(cap);
    }
    let t_sort = t.elapsed();

    if timing {
        eprintln!(
            "SIFT_TIMING detect build_ms={:.3} detect_ms={:.3} orient_ms={:.3} sort_ms={:.3} \
             n_kp={} octaves={}/{}",
            t_build.as_secs_f64() * 1e3,
            t_detect.as_secs_f64() * 1e3,
            t_orient.as_secs_f64() * 1e3,
            t_sort.as_secs_f64() * 1e3,
            keypoints.len(),
            detected_octaves,
            total_octaves,
        );
    }

    Detection {
        keypoints,
        scale_space,
    }
}

/// Compute descriptors for `keypoints` against an already-built `scale_space`.
///
/// `magnification` is the subregion spacing in units of `σ_kp` and `clamp` is
/// the post-normalization component cap (see [`SiftParams`]).
/// `descriptors.rows()[i]` corresponds to `keypoints[i]`.
pub fn compute_descriptors(
    scale_space: &ScaleSpace,
    keypoints: &[SiftKeypoint],
    magnification: f32,
    clamp: f32,
) -> Descriptors {
    descriptor::compute_descriptors(scale_space, keypoints, magnification, clamp)
}

/// Compute the descriptor for a single keypoint. Pure function of its inputs
/// (see [`compute_descriptors`] for `magnification` / `clamp`).
pub fn compute_descriptor(
    scale_space: &ScaleSpace,
    keypoint: &SiftKeypoint,
    magnification: f32,
    clamp: f32,
) -> [u8; 128] {
    descriptor::compute_descriptor(scale_space, keypoint, magnification, clamp)
}

/// One-shot convenience: detect every keypoint and describe all of them.
///
/// Builds the scale space once, runs detection, then describes every keypoint.
pub fn extract_sift(image: &GrayImage, params: &SiftParams) -> SiftFeatures {
    extract_sift_partial(image, params, None)
}

/// Like [`extract_sift`], but describe only the first `max_described` keypoints.
///
/// Detection still finds (and returns) every keypoint; only descriptor
/// computation is limited to the prefix `[0, k)` with `k =
/// min(max_described, keypoints.len())`. Because keypoints are sorted by
/// descending feature size, the described prefix is the top-`k` by size. `None`
/// describes all of them (equivalent to [`extract_sift`]).
///
/// This makes "detect a large pool cheaply, describe only a small working set"
/// a single operation: detection cost is dominated by extrema-finding /
/// localization (independent of the feature cap), while descriptor cost scales
/// with `k`. The returned `descriptors` then has `k` rows for `keypoints.len()`
/// keypoints — the caller is responsible for treating it as a described prefix.
pub fn extract_sift_partial(
    image: &GrayImage,
    params: &SiftParams,
    max_described: Option<usize>,
) -> SiftFeatures {
    let detection = detect_keypoints(image, params);
    let k = max_described.map_or(detection.keypoints.len(), |m| {
        m.min(detection.keypoints.len())
    });
    let t = std::time::Instant::now();
    let descriptors = compute_descriptors(
        &detection.scale_space,
        &detection.keypoints[..k],
        params.descriptor_magnification as f32,
        params.descriptor_clamp as f32,
    );
    if *SIFT_TIMING {
        eprintln!(
            "SIFT_TIMING describe describe_ms={:.3} n_desc={}",
            t.elapsed().as_secs_f64() * 1e3,
            descriptors.len(),
        );
    }
    SiftFeatures {
        keypoints: detection.keypoints,
        descriptors,
    }
}

#[cfg(test)]
mod tests;
