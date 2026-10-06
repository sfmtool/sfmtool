// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The AVX2 kernel of [`super::remap_aniso_with_pyramid`]:
//! eight output pixels of a row at once.
//!
//! A group of eight consecutive pixels goes through the kernel when every one
//! of them is valid, compressed (`sigma_major > 1`) and reads the same two
//! pyramid levels; each lane keeps its own position, footprint direction,
//! sample count and level blend. Any other group, and the row's last pixels
//! short of a group, go through the scalar [`super::aniso_pixel`].
//! On the patch tiles the sampler rule moves, neighbouring pixels share their
//! levels almost everywhere, so nearly every group takes the kernel.
//!
//! **The kernel is bit-identical to the scalar path.** Each lane does the
//! scalar path's `f32` operations in the same order: the same products and
//! left-to-right sums for a bilinear sample, no fused multiply-add, the
//! samples along the major axis summed in order with a lane that has taken
//! all its samples adding `0.0`, and the same blend and rounding. The test
//! `aniso_avx2_matches_scalar_bit_for_bit` holds it to that.
//!
//! The corners are fetched with 32-bit gathers at byte offsets, one per
//! corner for all of a pixel's channels: the four bytes read end at the
//! pixel's last channel (or start at the buffer's first byte), so a gather
//! never reads outside the image, and each channel is shifted out of the
//! word.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::{aniso_footprint, aniso_pixel, AnisoTally};
#[cfg(target_arch = "x86_64")]
use crate::camera::image::{ImageU8, ImageU8Pyramid};
#[cfg(target_arch = "x86_64")]
use crate::camera::warp_map::WarpMap;

/// Whether this CPU runs the kernel: x86-64 with AVX2.
pub(crate) fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The largest coordinate magnitude, in level pixels, the kernel takes: below
/// it `floor` and the conversion to `i32` agree with the scalar path's
/// saturating `as i32`. A group with a sample past it goes scalar.
#[cfg(target_arch = "x86_64")]
const MAX_COORD: f32 = 16_777_216.0;

/// Fill one output row of an anisotropic remap.
///
/// # Safety
///
/// The CPU has AVX2 ([`available`]), and `pyramid` has 1 to 4 channels.
/// `row_data` is the row's `map.width() * channels` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn fill_row(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    row: u32,
    row_data: &mut [u8],
    max_anisotropy: u32,
    tally: &mut AnisoTally,
) {
    let out_w = map.width();
    let c = pyramid.level(0).channels() as usize;
    let num_levels = pyramid.num_levels();
    let mut col = 0u32;
    while col + 8 <= out_w {
        let done = group(
            pyramid,
            map,
            row,
            col,
            c,
            num_levels,
            max_anisotropy,
            row_data,
            tally,
        );
        if !done {
            for k in col..col + 8 {
                aniso_pixel(pyramid, map, k, row, max_anisotropy, row_data, tally);
            }
        }
        col += 8;
    }
    for k in col..out_w {
        aniso_pixel(pyramid, map, k, row, max_anisotropy, row_data, tally);
    }
}

/// The eight pixels from `col0`, through the kernel, or `false` (writing
/// nothing) where they do not all qualify.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn group(
    pyramid: &ImageU8Pyramid,
    map: &WarpMap,
    row: u32,
    col0: u32,
    c: usize,
    num_levels: usize,
    max_anisotropy: u32,
    row_data: &mut [u8],
    tally: &mut AnisoTally,
) -> bool {
    let mut sx = [0f32; 8];
    let mut sy = [0f32; 8];
    let mut smaj = [0f32; 8];
    let mut dx = [0f32; 8];
    let mut dy = [0f32; 8];
    let mut frac = [0f32; 8];
    let mut nf = [0f32; 8];
    let mut ni = [0u32; 8];
    let mut levels = None;
    for k in 0..8 {
        let col = col0 + k as u32;
        let (x, y) = map.get(col, row);
        if x.is_nan() || y.is_nan() {
            return false;
        }
        let (sigma_major, sigma_minor, major_dx, major_dy) = map.get_svd(col, row);
        if sigma_major.is_nan() || sigma_major <= 1.0 {
            return false;
        }
        let fp = aniso_footprint(sigma_major, sigma_minor, num_levels, max_anisotropy);
        match levels {
            None => levels = Some((fp.level_lo, fp.level_hi)),
            Some(l) if l == (fp.level_lo, fp.level_hi) => {}
            Some(_) => return false,
        }
        sx[k] = x;
        sy[k] = y;
        smaj[k] = sigma_major;
        dx[k] = major_dx;
        dy[k] = major_dy;
        frac[k] = fp.frac;
        nf[k] = fp.n as f32;
        ni[k] = fp.n;
    }
    let (level_lo, level_hi) = levels.expect("eight lanes");
    let need_hi = frac.iter().any(|&f| f > 0.0);
    let n_max = *ni.iter().max().expect("eight lanes");

    let v_sx = _mm256_loadu_ps(sx.as_ptr());
    let v_sy = _mm256_loadu_ps(sy.as_ptr());
    let v_smaj = _mm256_loadu_ps(smaj.as_ptr());
    let v_dx = _mm256_loadu_ps(dx.as_ptr());
    let v_dy = _mm256_loadu_ps(dy.as_ptr());
    let v_n = _mm256_loadu_ps(nf.as_ptr());
    let half = _mm256_set1_ps(0.5);
    let lo = pyramid.level(level_lo);
    let hi = pyramid.level(level_hi);
    let scale_lo = _mm256_set1_ps((1u32 << level_lo) as f32);
    let scale_hi = _mm256_set1_ps((1u32 << level_hi) as f32);

    let mut sum_lo = [_mm256_setzero_ps(); 4];
    let mut sum_hi = [_mm256_setzero_ps(); 4];
    for i in 0..n_max {
        let v_i = _mm256_set1_ps(i as f32);
        // Lanes that have taken all their samples add nothing.
        let active = _mm256_cmp_ps::<_CMP_LT_OQ>(v_i, v_n);
        // t = (i + 0.5) / n - 0.5, sample = s + t * sigma_major * dir, with
        // dir the major direction in the source image.
        let t = _mm256_sub_ps(_mm256_div_ps(_mm256_add_ps(v_i, half), v_n), half);
        let ts = _mm256_mul_ps(t, v_smaj);
        let x = _mm256_add_ps(v_sx, _mm256_mul_ps(ts, v_dx));
        let y = _mm256_add_ps(v_sy, _mm256_mul_ps(ts, v_dy));
        if !accumulate(
            lo,
            _mm256_div_ps(x, scale_lo),
            _mm256_div_ps(y, scale_lo),
            active,
            c,
            &mut sum_lo,
        ) {
            return false;
        }
        if need_hi
            && !accumulate(
                hi,
                _mm256_div_ps(x, scale_hi),
                _mm256_div_ps(y, scale_hi),
                active,
                c,
                &mut sum_hi,
            )
        {
            return false;
        }
    }

    // Blend, round and write, as the scalar path does per channel.
    let v_frac = _mm256_loadu_ps(frac.as_ptr());
    let one_minus = _mm256_sub_ps(_mm256_set1_ps(1.0), v_frac);
    let zero = _mm256_setzero_ps();
    let max = _mm256_set1_ps(255.0);
    let mut out = [[0i32; 8]; 4];
    for ch in 0..c {
        let avg_lo = _mm256_div_ps(sum_lo[ch], v_n);
        let avg_hi = _mm256_div_ps(sum_hi[ch], v_n);
        let val = _mm256_add_ps(
            _mm256_mul_ps(avg_lo, one_minus),
            _mm256_mul_ps(avg_hi, v_frac),
        );
        let rounded = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(val, half), zero), max);
        _mm256_storeu_si256(
            out[ch].as_mut_ptr() as *mut __m256i,
            _mm256_cvttps_epi32(rounded),
        );
    }
    let pixels = &mut row_data[col0 as usize * c..(col0 as usize + 8) * c];
    for (k, pixel) in pixels.chunks_exact_mut(c).enumerate() {
        for (ch, slot) in pixel.iter_mut().enumerate() {
            *slot = out[ch][k] as u8;
        }
    }

    tally.sampled += 8;
    tally.multi += 8;
    tally.simd_groups += 1;
    for k in 0..8 {
        tally.sum_n += ni[k] as u64;
        tally.taps += c as u64 * ni[k] as u64 * if frac[k] > 0.0 { 2 } else { 1 };
    }
    true
}

/// Add the bilinear samples of `img` at the eight `(x, y)` (level pixels) to
/// `sums`, per channel, in the lanes `active` sets, with the products and
/// sums of the scalar `sample_bilinear_u8`. `false`, adding nothing, where a
/// coordinate is past [`MAX_COORD`] or not a number.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn accumulate(
    img: &ImageU8,
    x: __m256,
    y: __m256,
    active: __m256,
    c: usize,
    sums: &mut [__m256; 4],
) -> bool {
    // A gather reads a whole 32-bit word, which an image of fewer than four
    // bytes does not hold.
    if img.data.len() < 4 {
        return false;
    }
    let gx = _mm256_sub_ps(x, _mm256_set1_ps(0.5));
    let gy = _mm256_sub_ps(y, _mm256_set1_ps(0.5));
    // Every lane in range (a NaN fails the comparison).
    let limit = _mm256_set1_ps(MAX_COORD);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let in_x = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(gx, abs_mask), limit);
    let in_y = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(gy, abs_mask), limit);
    if _mm256_movemask_ps(_mm256_and_ps(in_x, in_y)) != 0xff {
        return false;
    }
    let x0f = _mm256_floor_ps(gx);
    let y0f = _mm256_floor_ps(gy);
    let fx = _mm256_sub_ps(gx, x0f);
    let fy = _mm256_sub_ps(gy, y0f);
    let x0 = _mm256_cvttps_epi32(x0f);
    let y0 = _mm256_cvttps_epi32(y0f);
    let one = _mm256_set1_epi32(1);
    let x1 = _mm256_add_epi32(x0, one);
    let y1 = _mm256_add_epi32(y0, one);
    let zero_i = _mm256_setzero_si256();
    let w_max = _mm256_set1_epi32(img.width as i32 - 1);
    let h_max = _mm256_set1_epi32(img.height as i32 - 1);
    let cx0 = _mm256_min_epi32(_mm256_max_epi32(x0, zero_i), w_max);
    let cx1 = _mm256_min_epi32(_mm256_max_epi32(x1, zero_i), w_max);
    let cy0 = _mm256_min_epi32(_mm256_max_epi32(y0, zero_i), h_max);
    let cy1 = _mm256_min_epi32(_mm256_max_epi32(y1, zero_i), h_max);

    // Byte offset of each corner's first channel.
    let stride = _mm256_set1_epi32((img.width as usize * c) as i32);
    let chans = _mm256_set1_epi32(c as i32);
    let row0 = _mm256_mullo_epi32(cy0, stride);
    let row1 = _mm256_mullo_epi32(cy1, stride);
    let col0 = _mm256_mullo_epi32(cx0, chans);
    let col1 = _mm256_mullo_epi32(cx1, chans);
    let corners = [
        _mm256_add_epi32(row0, col0),
        _mm256_add_epi32(row0, col1),
        _mm256_add_epi32(row1, col0),
        _mm256_add_epi32(row1, col1),
    ];

    // One 32-bit gather per corner: the word ending at the corner's last
    // channel, or the image's first word for a corner within its first bytes.
    let data = img.data.as_ptr() as *const i32;
    let back = _mm256_set1_epi32(4 - c as i32);
    let mut words = [_mm256_setzero_si256(); 4];
    let mut shifts = [_mm256_setzero_si256(); 4];
    for (k, &first) in corners.iter().enumerate() {
        let start = _mm256_max_epi32(_mm256_sub_epi32(first, back), zero_i);
        words[k] = _mm256_i32gather_epi32::<1>(data, start);
        shifts[k] = _mm256_slli_epi32::<3>(_mm256_sub_epi32(first, start));
    }

    // The weights, grouped as `(1 - fx) * (1 - fy) * v00` in the scalar sum.
    let onef = _mm256_set1_ps(1.0);
    let ofx = _mm256_sub_ps(onef, fx);
    let ofy = _mm256_sub_ps(onef, fy);
    let w = [
        _mm256_mul_ps(ofx, ofy),
        _mm256_mul_ps(fx, ofy),
        _mm256_mul_ps(ofx, fy),
        _mm256_mul_ps(fx, fy),
    ];
    let byte = _mm256_set1_epi32(0xff);
    for (ch, sum) in sums.iter_mut().take(c).enumerate() {
        let ch_shift = _mm256_set1_epi32(8 * ch as i32);
        let mut v = [_mm256_setzero_ps(); 4];
        for (k, value) in v.iter_mut().enumerate() {
            let s = _mm256_add_epi32(shifts[k], ch_shift);
            *value = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(words[k], s), byte));
        }
        let sample = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(w[0], v[0]), _mm256_mul_ps(w[1], v[1])),
                _mm256_mul_ps(w[2], v[2]),
            ),
            _mm256_mul_ps(w[3], v[3]),
        );
        *sum = _mm256_add_ps(*sum, _mm256_and_ps(sample, active));
    }
    true
}
