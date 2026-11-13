// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The one conversion of HDR video pixels (BT.2100 HLG or PQ, BT.2020
//! primaries) to 8-bit sRGB: [`hdr_ycbcr_to_srgb`].
//!
//! Every pixel goes through the same fixed function of its own value, with no
//! analysis of the frame, so frames converted at different times, from
//! different videos, have the same exposure. The recipe, its constants and
//! why it is built this way are in `specs/core/camera/hdr-to-srgb.md`; in
//! short: narrow-range Y'CbCr to R'G'B', the HLG inverse OETF and OOTF for a
//! 1000 cd/m² display (or the PQ EOTF) to display light, one exposure scale,
//! BT.2020 to BT.709 primaries, negative components set to 0, the BT.2390
//! EETF on `max(R, G, B)` from 1000 to 203 cd/m², 203 cd/m² as the sRGB
//! scale's 1.0, then the sRGB encoding rounded to 8 bits.
//!
//! **Why single precision.** The input is a 16-bit code per channel and the
//! output an 8-bit one, so the result needs about three significant digits.
//! Each pixel's value passes through a fixed chain of about twenty operations
//! (a 3x3 matrix, `exp` or two `powf`, one more `powf` for the HLG OOTF and
//! four for the roll-off), with no sum over pixels; each adds at most a few
//! units of `2^-24` of relative error, so the chain stays below `10^-5` of the
//! value, under a hundredth of an 8-bit level even at the steepest part of the
//! sRGB curve, where one level is about 0.4% of the linear value. Display
//! light spans 0 to 10,000 cd/m², well inside `f32`'s range. An `f32` lane is
//! twice as many values per vector instruction as `f64`. The 255 rounding
//! thresholds of the sRGB step are computed once in `f64` and stored as `f32`,
//! so the rounding of each value is a comparison with fixed numbers rather
//! than one more `powf`.

use std::sync::OnceLock;

use rayon::prelude::*;

use super::ImageU8;

/// The name of the recipe [`hdr_ycbcr_to_srgb`] applies, recorded beside the
/// frames it writes, with [`HDR_TO_SRGB_RECIPE_VERSION`].
pub const HDR_TO_SRGB_RECIPE: &str = "sfmtool-hdr-to-srgb";

/// The version of the recipe. A change to any constant or step of the
/// conversion, anything that changes an output code, takes a new version.
pub const HDR_TO_SRGB_RECIPE_VERSION: u32 = 1;

/// The largest exposure change, in stops either way, the conversion takes.
/// `2^16` keeps every scaled display light (at most 10,000 cd/m² before the
/// scale) far inside `f32`'s range, and is far beyond any useful exposure.
pub const HDR_EXPOSURE_STOPS_LIMIT: f32 = 16.0;

/// An HDR transfer function of ITU-R BT.2100.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrTransfer {
    /// Hybrid log-gamma, `arib-std-b67` in FFmpeg's naming: relative scene
    /// light, as phones record HDR video.
    Hlg,
    /// Perceptual quantizer, `smpte2084` in FFmpeg's naming: absolute display
    /// light up to 10,000 cd/m², as in HDR10.
    Pq,
}

/// The settings of one conversion. The same options give the same output for
/// the same pixel, wherever it is and whichever frame it is in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrToSrgbOptions {
    /// The transfer function the input's R'G'B' is encoded with.
    pub transfer: HdrTransfer,
    /// The exposure change in stops: display light is multiplied by
    /// `2^exposure_stops` before the highlight roll-off. 0 is the recipe's
    /// own anchor, HDR reference white (203 cd/m²) as the sRGB scale's 1.0.
    /// At most [`HDR_EXPOSURE_STOPS_LIMIT`] either way.
    pub exposure_stops: f32,
}

/// Why [`hdr_ycbcr_to_srgb`] refused its input.
#[derive(Clone, Debug, PartialEq)]
pub enum HdrToSrgbError {
    /// The sample count is not `width * height * 3`, or that product does
    /// not fit in memory.
    SampleCount {
        width: u32,
        height: u32,
        samples: usize,
    },
    /// The exposure is not finite or is beyond [`HDR_EXPOSURE_STOPS_LIMIT`].
    Exposure(f32),
}

impl std::fmt::Display for HdrToSrgbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HdrToSrgbError::SampleCount {
                width,
                height,
                samples,
            } => write!(
                f,
                "a {width} x {height} image of 3 channels has {} samples, not {samples}",
                (*width as u128) * (*height as u128) * 3
            ),
            HdrToSrgbError::Exposure(stops) => write!(
                f,
                "the exposure must be a finite number of stops from \
                 -{HDR_EXPOSURE_STOPS_LIMIT} to {HDR_EXPOSURE_STOPS_LIMIT}, not {stops}"
            ),
        }
    }
}

impl std::error::Error for HdrToSrgbError {}

// BT.2100 non-constant-luminance Y'CbCr, BT.2020 luma weights.
const KR: f32 = 0.2627;
const KB: f32 = 0.0593;
const KG: f32 = 0.6780;
// R' = Y' + 2(1 - KR) Cr, B' = Y' + 2(1 - KB) Cb, and
// G' = Y' - (2 KB (1 - KB) / KG) Cb - (2 KR (1 - KR) / KG) Cr, the form in
// which zero chroma gives R' = G' = B' = Y' exactly.
const CR_TO_R: f32 = 1.4746;
const CB_TO_B: f32 = 1.8814;
const CB_TO_G: f32 = (0.0593 * 1.8814 / 0.6780) as f32;
const CR_TO_G: f32 = (0.2627 * 1.4746 / 0.6780) as f32;

// HLG inverse OETF.
const HLG_A: f32 = 0.178_832_77_f64 as f32;
const HLG_B: f32 = 0.284_668_92_f64 as f32; // 1 - 4a
const HLG_C: f32 = 0.559_910_73_f64 as f32; // 0.5 - a ln(4a)
/// The nominal peak of the display the HLG OOTF renders for, cd/m², and its
/// system gamma, `1.2 + 0.42 log10(L_W / 1000)` = 1.2.
const HLG_DISPLAY_PEAK: f32 = 1000.0;
const HLG_SYSTEM_GAMMA: f32 = 1.2;

// PQ (SMPTE ST 2084).
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;
const PQ_PEAK: f64 = 10000.0;

/// BT.2020 to BT.709 linear RGB, both with a D65 white, from the two sets of
/// primaries' chromaticities (ITU-R BT.2087): the off-diagonal entries of
/// each row, `OFF_DIAGONAL[i] = [M[i][j] for the two j != i in order]`. Each
/// row of the matrix sums to 1, so `(M F)_i = F_i + sum_j M[i][j] (F_j - F_i)`
/// over `j != i`, the form used: a grey (`F_j = F_i`) stays exactly the same
/// grey in `f32`.
///
/// The full matrix, to the ten digits of the derivation:
/// `[[1.6604910021, -0.5876411388, -0.0728498633],
///   [-0.1245504745, 1.1328998971, -0.0083494226],
///   [-0.0181507634, -0.1005788980, 1.1187296614]]`.
const OFF_DIAGONAL: [[f32; 2]; 3] = [
    [-0.587_641_138_8_f64 as f32, -0.072_849_863_3_f64 as f32],
    [-0.124_550_474_5_f64 as f32, -0.008_349_422_6_f64 as f32],
    [-0.018_150_763_4_f64 as f32, -0.100_578_898_0_f64 as f32],
];

/// HDR reference white (ITU-R BT.2408), cd/m²: the light the sRGB scale's 1.0
/// stands for, and the target peak of the roll-off.
const REFERENCE_WHITE: f64 = 203.0;
/// The source peak of the roll-off, cd/m²: the HLG OOTF's display peak, and
/// the light a PQ video's highlights are taken to end at.
const SOURCE_PEAK: f64 = 1000.0;

/// PQ inverse EOTF: display light in cd/m² to the PQ signal in `[0, 1]`.
fn pq_inverse_eotf(nits: f64) -> f64 {
    let y = (nits / PQ_PEAK).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)).powf(PQ_M2)
}

/// PQ EOTF: a PQ signal in `[0, 1]` to display light in cd/m².
fn pq_eotf(signal: f64) -> f64 {
    let p = signal.powf(1.0 / PQ_M2);
    PQ_PEAK * ((p - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * p)).powf(1.0 / PQ_M1)
}

/// The same two functions in `f32`, for the per-pixel work.
fn pq_inverse_eotf_f32(nits: f32) -> f32 {
    let y = (nits / PQ_PEAK as f32).powf(PQ_M1 as f32);
    ((PQ_C1 as f32 + PQ_C2 as f32 * y) / (1.0 + PQ_C3 as f32 * y)).powf(PQ_M2 as f32)
}

fn pq_eotf_f32(signal: f32) -> f32 {
    let p = signal.powf(1.0 / PQ_M2 as f32);
    PQ_PEAK as f32
        * ((p - PQ_C1 as f32).max(0.0) / (PQ_C2 as f32 - PQ_C3 as f32 * p)).powf(1.0 / PQ_M1 as f32)
}

/// The constants of the BT.2390 EETF from [`SOURCE_PEAK`] to
/// [`REFERENCE_WHITE`], blacks at 0, in the PQ domain normalised by the
/// source peak.
struct RollOff {
    /// `PQ⁻¹(SOURCE_PEAK)`.
    source_peak_pq: f32,
    /// `maxLum = PQ⁻¹(REFERENCE_WHITE) / PQ⁻¹(SOURCE_PEAK)`.
    max_lum: f32,
    /// The knee, `KS = 1.5 maxLum - 0.5`.
    knee: f32,
    /// The light at the knee, cd/m²: below it a pixel is unchanged.
    knee_nits: f32,
}

fn roll_off() -> &'static RollOff {
    static ROLL_OFF: OnceLock<RollOff> = OnceLock::new();
    ROLL_OFF.get_or_init(|| {
        let source_peak_pq = pq_inverse_eotf(SOURCE_PEAK);
        let max_lum = pq_inverse_eotf(REFERENCE_WHITE) / source_peak_pq;
        let knee = 1.5 * max_lum - 0.5;
        RollOff {
            source_peak_pq: source_peak_pq as f32,
            max_lum: max_lum as f32,
            knee: knee as f32,
            knee_nits: pq_eotf(knee * source_peak_pq) as f32,
        }
    })
}

/// The BT.2390 EETF applied to `m`, the largest component of a pixel in
/// cd/m², returning the light it maps to (at most [`REFERENCE_WHITE`]).
fn eetf(m: f32, r: &RollOff) -> f32 {
    if m < r.knee_nits {
        return m;
    }
    let e1 = (pq_inverse_eotf_f32(m) / r.source_peak_pq).min(1.0);
    if e1 < r.knee {
        return m;
    }
    let t = (e1 - r.knee) / (1.0 - r.knee);
    let (t2, t3) = (t * t, t * t * t);
    let e2 = (2.0 * t3 - 3.0 * t2 + 1.0) * r.knee
        + (t3 - 2.0 * t2 + t) * (1.0 - r.knee)
        + (-2.0 * t3 + 3.0 * t2) * r.max_lum;
    pq_eotf_f32(e2 * r.source_peak_pq)
}

/// The 255 linear values at which the 8-bit sRGB code steps up:
/// `thresholds[k]` is the linear `y` whose sRGB encoding is `(k + 1/2) / 255`,
/// so code `n` is the count of thresholds at or below `y`, which rounds
/// `255 · sRGB(y)` half up.
fn srgb_thresholds() -> &'static [f32; 255] {
    static THRESHOLDS: OnceLock<[f32; 255]> = OnceLock::new();
    THRESHOLDS.get_or_init(|| {
        let mut thresholds = [0f32; 255];
        for (k, threshold) in thresholds.iter_mut().enumerate() {
            let v = (k as f64 + 0.5) / 255.0;
            let y = if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            };
            *threshold = y as f32;
        }
        thresholds
    })
}

/// The fixed state of one conversion.
struct Recipe {
    transfer: HdrTransfer,
    exposure: f32,
    roll_off: &'static RollOff,
    thresholds: &'static [f32; 255],
}

impl Recipe {
    /// One pixel: 16-bit narrow-range Y'CbCr codes to 8-bit sRGB.
    fn pixel(&self, ycbcr: [u16; 3]) -> [u8; 3] {
        // 1. Narrow-range Y'CbCr to R'G'B' in [0, 1].
        let y = (ycbcr[0] as f32 / 256.0 - 16.0) / 219.0;
        let cb = (ycbcr[1] as f32 / 256.0 - 128.0) / 224.0;
        let cr = (ycbcr[2] as f32 / 256.0 - 128.0) / 224.0;
        let r = y + CR_TO_R * cr;
        let g = y - CB_TO_G * cb - CR_TO_G * cr;
        let b = y + CB_TO_B * cb;
        let signal = [r, g, b].map(|v| v.clamp(0.0, 1.0));

        // 2. Display light, BT.2020 RGB, cd/m².
        let light = match self.transfer {
            HdrTransfer::Hlg => {
                let scene = signal.map(|e| {
                    if e <= 0.5 {
                        e * e / 3.0
                    } else {
                        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
                    }
                });
                let ys = KR * scene[0] + KG * scene[1] + KB * scene[2];
                let gain = if ys > 0.0 {
                    HLG_DISPLAY_PEAK * ys.powf(HLG_SYSTEM_GAMMA - 1.0)
                } else {
                    0.0
                };
                scene.map(|e| gain * e)
            }
            HdrTransfer::Pq => signal.map(pq_eotf_f32),
        };

        // 3. Exposure, 4. BT.709 primaries, 5. negatives to 0.
        let f = light.map(|v| v * self.exposure);
        let [[r_g, r_b], [g_r, g_b], [b_r, b_g]] = OFF_DIAGONAL;
        let mut rgb = [
            f[0] + r_g * (f[1] - f[0]) + r_b * (f[2] - f[0]),
            f[1] + g_r * (f[0] - f[1]) + g_b * (f[2] - f[1]),
            f[2] + b_r * (f[0] - f[2]) + b_g * (f[1] - f[2]),
        ]
        .map(|v| v.max(0.0));

        // 6. The roll-off on max(R, G, B), the same factor on each channel.
        let m = rgb[0].max(rgb[1]).max(rgb[2]);
        if m > 0.0 {
            let factor = eetf(m, self.roll_off) / m;
            if factor != 1.0 {
                rgb = rgb.map(|v| v * factor);
            }
        }

        // 7. 203 cd/m² is 1.0, 8. sRGB code rounded half up.
        rgb.map(|v| {
            let y = v / REFERENCE_WHITE as f32;
            self.thresholds.partition_point(|&t| t <= y) as u8
        })
    }
}

/// Convert one frame of HDR video to 8-bit sRGB.
///
/// `planes` is three planes of `width * height` 16-bit codes each, row by
/// row: Y', then Cb, then Cr, in BT.2100 narrow range at 16 bits (a 10-bit
/// code shifted left by 6: black 4096, white 60160, zero chroma 32768). That
/// is the layout of FFmpeg's `yuv444p16le`. The result is a 3-channel
/// [`ImageU8`] of the same size in sRGB. The rows are converted in parallel;
/// the output does not depend on the thread count, since each pixel is
/// converted alone.
///
/// Returns [`HdrToSrgbError::SampleCount`] when `planes` does not hold
/// `width * height * 3` samples and [`HdrToSrgbError::Exposure`] for an
/// exposure outside `±HDR_EXPOSURE_STOPS_LIMIT` or not finite.
pub fn hdr_ycbcr_to_srgb(
    planes: &[u16],
    width: u32,
    height: u32,
    options: &HdrToSrgbOptions,
) -> Result<ImageU8, HdrToSrgbError> {
    let stops = options.exposure_stops;
    if !stops.is_finite() || stops.abs() > HDR_EXPOSURE_STOPS_LIMIT {
        return Err(HdrToSrgbError::Exposure(stops));
    }
    let samples = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3));
    if samples != Some(planes.len()) {
        return Err(HdrToSrgbError::SampleCount {
            width,
            height,
            samples: planes.len(),
        });
    }
    let recipe = Recipe {
        transfer: options.transfer,
        exposure: stops.exp2(),
        roll_off: roll_off(),
        thresholds: srgb_thresholds(),
    };
    let mut data = vec![0u8; planes.len()];
    let width = width as usize;
    let plane_len = planes.len() / 3;
    let (y_plane, chroma) = planes.split_at(plane_len);
    let (cb_plane, cr_plane) = chroma.split_at(plane_len);
    data.par_chunks_mut((width * 3).max(1))
        .enumerate()
        .for_each(|(row, out)| {
            let start = row * width;
            let rows = (
                &y_plane[start..start + width],
                &cb_plane[start..start + width],
                &cr_plane[start..start + width],
            );
            for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                o.copy_from_slice(&recipe.pixel([rows.0[x], rows.1[x], rows.2[x]]));
            }
        });
    let width = width as u32;
    Ok(ImageU8::new(width, height, 3, data))
}

#[cfg(test)]
mod tests;
