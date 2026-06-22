// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust SIFT feature detector and descriptor.
//!
//! Reference: David G. Lowe, "Distinctive Image Features from Scale-Invariant
//! Keypoints," IJCV 60(2):91-110, 2004.
//!
//! This crate implements SIFT directly (no OpenCV/COLMAP round-trip), structured
//! to mirror the optical-flow module's house conventions: it operates on
//! [`GrayImage`] (reused from [`crate::optical_flow`]), uses separable Gaussian
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
pub use crate::optical_flow::GrayImage;

/// Env-gated stage timing for the keypoint-finding pipeline. Enabled by setting
/// the `SFMTOOL_SIFT_TIMING` environment variable; effectively zero-cost otherwise
/// (one cached bool check plus a few `Instant::now()` per image). When on, each
/// `detect_keypoints` call and each `ScaleSpace::build` prints one
/// `SIFT_TIMING ...` line to stderr with per-stage wall-clock milliseconds, for
/// offline aggregation.
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
    let scale_space = ScaleSpace::build(image, params);
    let t_build = t.elapsed();

    // Stage 3+4: scale-space extrema + subpixel localization (un-oriented).
    let t = std::time::Instant::now();
    let mut localized = detect::detect_and_localize(&scale_space, params);
    // Feature-count cap (COLMAP `max_num_features`): keep the largest-scale
    // candidates. Applied here, before orientation and description, so neither
    // of those (atan2-heavy) stages processes candidates that will be discarded.
    // `select_nth_unstable_by` partitions in O(n) so the first `cap` entries are
    // the largest scale. The comparator is a *total* order (scale, then response,
    // then octave/layer/y/x) so the retained set is deterministic when scales tie
    // — otherwise an unstable partition could keep a different subset run-to-run,
    // breaking the reproducible-`.sift` contract (see `specs/core/sift.md`).
    if let Some(cap) = params.max_num_features {
        if localized.len() > cap {
            use std::cmp::Ordering::Equal;
            localized.select_nth_unstable_by(cap, |a, b| {
                b.scale
                    .partial_cmp(&a.scale)
                    .unwrap_or(Equal)
                    .then(b.response.partial_cmp(&a.response).unwrap_or(Equal))
                    .then(a.octave.cmp(&b.octave))
                    .then(a.layer.partial_cmp(&b.layer).unwrap_or(Equal))
                    .then(a.y.partial_cmp(&b.y).unwrap_or(Equal))
                    .then(a.x.partial_cmp(&b.x).unwrap_or(Equal))
            });
            localized.truncate(cap);
        }
    }
    let t_detect = t.elapsed();

    // No DoG pyramid to free — detection computed it per stripe in cache. The
    // Gaussian pyramid is retained for orientation and description.

    // Stage 5: orientation assignment (may emit multiple keypoints per candidate).
    let t = std::time::Instant::now();
    let mut keypoints = orientation::assign_orientations(&scale_space, &localized, params);
    let t_orient = t.elapsed();

    // The `.sift` format requires keypoints sorted by descending feature size
    // (size = the affine-shape column-norm average, `SiftKeypoint::scale`). Sort
    // here so every consumer of `detect_keypoints` (and `extract_sift`) sees that
    // ordering. A total-order tie-break (response, octave, layer, y, x,
    // orientation) makes the output order — and thus the `.sift` bytes —
    // reproducible across runs and thread counts, independent of the upstream
    // (parallel) collection order.
    let t = std::time::Instant::now();
    keypoints.sort_by(|a, b| {
        use std::cmp::Ordering::Equal;
        b.scale()
            .partial_cmp(&a.scale())
            .unwrap_or(Equal)
            .then(b.response.partial_cmp(&a.response).unwrap_or(Equal))
            .then(a.octave.cmp(&b.octave))
            .then(a.layer.partial_cmp(&b.layer).unwrap_or(Equal))
            .then(a.y.partial_cmp(&b.y).unwrap_or(Equal))
            .then(a.x.partial_cmp(&b.x).unwrap_or(Equal))
            .then(
                a.orientation()
                    .partial_cmp(&b.orientation())
                    .unwrap_or(Equal),
            )
    });
    // Orientation assignment can emit several keypoints per candidate, pushing
    // the count back above the cap; enforce it as a hard output limit. The sort
    // above already placed the largest-scale keypoints first.
    if let Some(cap) = params.max_num_features {
        keypoints.truncate(cap);
    }
    let t_sort = t.elapsed();

    if timing {
        eprintln!(
            "SIFT_TIMING detect build_ms={:.3} detect_ms={:.3} orient_ms={:.3} sort_ms={:.3} n_kp={}",
            t_build.as_secs_f64() * 1e3,
            t_detect.as_secs_f64() * 1e3,
            t_orient.as_secs_f64() * 1e3,
            t_sort.as_secs_f64() * 1e3,
            keypoints.len(),
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
    let descriptors = compute_descriptors(
        &detection.scale_space,
        &detection.keypoints[..k],
        params.descriptor_magnification as f32,
        params.descriptor_clamp as f32,
    );
    SiftFeatures {
        keypoints: detection.keypoints,
        descriptors,
    }
}

#[cfg(test)]
mod tests;
