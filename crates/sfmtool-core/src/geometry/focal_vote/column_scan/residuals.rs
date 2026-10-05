// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-correspondence residual kernels of the column scan: the one-sided
//! epipolar residual and the rotation residual, each in `f64` and in `f32`.
//!
//! Each residual is a dispatcher, a scalar reference, and on `x86_64` an AVX2
//! kernel that repeats the scalar arithmetic lane for lane. All of the column
//! scan's `unsafe` code is in this module, and the parity tests beside it
//! compare each AVX2 kernel bit for bit against its scalar twin.

use nalgebra::{Matrix3, Vector3};

/// Whether `idx[..len]` is the consecutive run `idx[0] .. idx[0] + len`.
///
/// Every index list the column scan hands a residual kernel is **strictly
/// increasing** — it is `0..n`, or a filter over it — and a strictly
/// increasing run of `len` values spans at least `len - 1`, with equality only
/// when it steps by one throughout. So the first and last entries settle it,
/// and the vectorized dispatch does not scan six hundred indices at each of
/// the hundreds of thousands of calls to find out whether it may run.
#[cfg(target_arch = "x86_64")]
#[inline]
fn consecutive_run(idx: &[usize], len: usize) -> bool {
    len > 0 && idx.len() >= len && idx[len - 1] == idx[0] + len - 1
}

/// One side's rays at a candidate focal, structure-of-arrays in `f32`.
///
/// Three separate arrays, not packed triples: a lane load is then one
/// `loadu_ps` with no transpose, where the `f64` kernels pay four shuffle µops
/// per four points to keep their rays in the `Vec<Vector3<f64>>` layout their
/// other callers share.
pub(super) struct RaysF32 {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

impl RaysF32 {
    pub(super) fn zeros(n: usize) -> Self {
        Self {
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
        }
    }

    /// Store ray `i` narrowed, and hand back the widened value for the `f64`
    /// buffer — the one call that keeps the two representations one ray.
    #[inline]
    pub(super) fn set(&mut self, i: usize, v: Vector3<f64>) -> Vector3<f64> {
        let (x, y, z) = (v.x as f32, v.y as f32, v.z as f32);
        self.x[i] = x;
        self.y[i] = y;
        self.z[i] = z;
        Vector3::new(f64::from(x), f64::from(y), f64::from(z))
    }
}

/// [`epipolar_residuals`] at single precision (see the single-precision
/// section header in the parent module).
///
/// Dispatches to an 8-lane AVX2 kernel where the CPU has it;
/// `epipolar_residuals_f32_scalar` is both the fallback and its op-for-op
/// twin.
pub(super) fn epipolar_residuals_f32(
    e: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    side_two: bool,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let n = r1.x.len();
        if crate::geometry::simd::avx2_enabled() && r2.x.len() >= n && out.len() >= n {
            // SAFETY: avx2 confirmed available, and both ray sets and `out`
            // cover the `n` points the kernel reads and writes.
            unsafe { epipolar_residuals_f32_avx2(e, r1, r2, side_two, out) };
            return;
        }
    }
    epipolar_residuals_f32_scalar(e, r1, r2, side_two, out);
}

/// The `3×3` matrix the one-sided residual multiplies its source rays by,
/// narrowed and row-major: `E` for side two, `Eᵀ` for side one.
fn side_matrix_f32(e: &Matrix3<f64>, side_two: bool) -> [[f32; 3]; 3] {
    let at = |r: usize, c: usize| {
        if side_two {
            e[(r, c)] as f32
        } else {
            e[(c, r)] as f32
        }
    };
    [
        [at(0, 0), at(0, 1), at(0, 2)],
        [at(1, 0), at(1, 1), at(1, 2)],
        [at(2, 0), at(2, 1), at(2, 2)],
    ]
}

/// Scalar twin of [`epipolar_residuals_f32`].
fn epipolar_residuals_f32_scalar(
    e: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    side_two: bool,
    out: &mut [f32],
) {
    let m = side_matrix_f32(e, side_two);
    let (src, other) = if side_two { (r1, r2) } else { (r2, r1) };
    for (i, o) in out.iter_mut().enumerate().take(src.x.len()) {
        let (sx, sy, sz) = (src.x[i], src.y[i], src.z[i]);
        let nx = (m[0][0] * sx + m[0][1] * sy) + m[0][2] * sz;
        let ny = (m[1][0] * sx + m[1][1] * sy) + m[1][2] * sz;
        let nz = (m[2][0] * sx + m[2][1] * sy) + m[2][2] * sz;
        let nn = ((nx * nx + ny * ny) + nz * nz).sqrt().max(1e-15);
        let d = (nx * other.x[i] + ny * other.y[i]) + nz * other.z[i];
        *o = (d.abs() / nn).min(1.0);
    }
}

/// Eight correspondences per iteration of [`epipolar_residuals_f32`].
///
/// # Safety
/// Requires the `avx2` target feature, ray sets covering `r1.x.len()` points
/// and an `out` of at least that length (all guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn epipolar_residuals_f32_avx2(
    e: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    side_two: bool,
    out: &mut [f32],
) {
    use crate::geometry::simd::{dot3_ps, row3_ps};
    use std::arch::x86_64::*;

    let n = r1.x.len();
    let (src, other) = if side_two { (r1, r2) } else { (r2, r1) };
    let m = {
        let s = side_matrix_f32(e, side_two);
        let b = |r: usize, c: usize| _mm256_set1_ps(s[r][c]);
        [
            [b(0, 0), b(0, 1), b(0, 2)],
            [b(1, 0), b(1, 1), b(1, 2)],
            [b(2, 0), b(2, 1), b(2, 2)],
        ]
    };
    let floor = _mm256_set1_ps(1e-15);
    let one = _mm256_set1_ps(1.0);
    // `andnot` against the sign bit is what `f32::abs` compiles to.
    let sign = _mm256_set1_ps(-0.0);

    let blocks = n / 8;
    for b in 0..blocks {
        let k = b * 8;
        let sx = _mm256_loadu_ps(src.x.as_ptr().add(k));
        let sy = _mm256_loadu_ps(src.y.as_ptr().add(k));
        let sz = _mm256_loadu_ps(src.z.as_ptr().add(k));
        let ox = _mm256_loadu_ps(other.x.as_ptr().add(k));
        let oy = _mm256_loadu_ps(other.y.as_ptr().add(k));
        let oz = _mm256_loadu_ps(other.z.as_ptr().add(k));

        let nx = row3_ps(&m[0], sx, sy, sz);
        let ny = row3_ps(&m[1], sx, sy, sz);
        let nz = row3_ps(&m[2], sx, sy, sz);

        // Value first in both `max` and `min`, matching `f32::max`/`f32::min`.
        let nn = _mm256_max_ps(_mm256_sqrt_ps(dot3_ps(nx, ny, nz, nx, ny, nz)), floor);
        let d = _mm256_andnot_ps(sign, dot3_ps(nx, ny, nz, ox, oy, oz));
        let r = _mm256_min_ps(_mm256_div_ps(d, nn), one);
        _mm256_storeu_ps(out.as_mut_ptr().add(k), r);
    }

    for (i, o) in out.iter_mut().enumerate().take(n).skip(blocks * 8) {
        let s = side_matrix_f32(e, side_two);
        let (sx, sy, sz) = (src.x[i], src.y[i], src.z[i]);
        let nx = (s[0][0] * sx + s[0][1] * sy) + s[0][2] * sz;
        let ny = (s[1][0] * sx + s[1][1] * sy) + s[1][2] * sz;
        let nz = (s[2][0] * sx + s[2][1] * sy) + s[2][2] * sz;
        let nn = ((nx * nx + ny * ny) + nz * nz).sqrt().max(1e-15);
        let d = (nx * other.x[i] + ny * other.y[i]) + nz * other.z[i];
        *o = (d.abs() / nn).min(1.0);
    }
}

/// [`rotation_residuals`] at single precision, through the cross-product-norm
/// form: `θ = asin|R r₁ × r₂|`, folded to `π − θ` where the dot is negative.
///
/// The dot alone cannot carry the answer in `f32` — at the inlier band's
/// `1e-4..1e-3` radians, `cos θ = 1 − θ²/2` differs from `1` by less than
/// single-precision epsilon and the recovered angle is worthless (measured
/// median error 38%). The cross-product norm IS `sin θ`, so the small angle
/// arrives in its own leading digits.
///
/// Dispatches to an 8-lane AVX2 kernel over consecutive `idx` runs — the hot
/// shape, since the RANSAC scoring loops pass all `n` points at every one of
/// the ~128 minimal samples; the arbitrary-subset callers and the ragged tail
/// take the scalar twin.
pub(super) fn rotation_residuals_f32(
    rot: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    idx: &[usize],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let len = out.len().min(idx.len());
        if crate::geometry::simd::avx2_enabled()
            && consecutive_run(idx, len)
            && idx[0] + len <= r1.x.len().min(r2.x.len())
        {
            // SAFETY: avx2 confirmed available, and the consecutive run
            // `idx[0] .. idx[0] + len` lies inside both ray sets while `out`
            // covers `len` elements.
            unsafe { rotation_residuals_f32_avx2(rot, r1, r2, idx[0], &mut out[..len]) };
            return;
        }
    }
    rotation_residuals_f32_scalar(rot, r1, r2, idx, out);
}

/// The rotation narrowed to `f32`, row-major.
fn rot_f32(rot: &Matrix3<f64>) -> [[f32; 3]; 3] {
    let b = |r: usize, c: usize| rot[(r, c)] as f32;
    [
        [b(0, 0), b(0, 1), b(0, 2)],
        [b(1, 0), b(1, 1), b(1, 2)],
        [b(2, 0), b(2, 1), b(2, 2)],
    ]
}

/// One point's angle under [`rotation_residuals_f32`], the scalar arithmetic
/// the vector kernel repeats lane for lane.
#[inline]
fn rotation_angle_f32(m: &[[f32; 3]; 3], a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let px = (m[0][0] * a.0 + m[0][1] * a.1) + m[0][2] * a.2;
    let py = (m[1][0] * a.0 + m[1][1] * a.1) + m[1][2] * a.2;
    let pz = (m[2][0] * a.0 + m[2][1] * a.1) + m[2][2] * a.2;
    let c = (px * b.0 + py * b.1) + pz * b.2;
    let cx = py * b.2 - pz * b.1;
    let cy = pz * b.0 - px * b.2;
    let cz = px * b.1 - py * b.0;
    let s = ((cx * cx + cy * cy) + cz * cz).sqrt().min(1.0);
    let ang = crate::geometry::acos_poly::asin_poly_scalar_f32(s);
    if c < 0.0 {
        std::f32::consts::PI - ang
    } else {
        ang
    }
}

/// Scalar twin of [`rotation_residuals_f32`].
fn rotation_residuals_f32_scalar(
    rot: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    idx: &[usize],
    out: &mut [f32],
) {
    let m = rot_f32(rot);
    for (o, &i) in out.iter_mut().zip(idx.iter()) {
        *o = rotation_angle_f32(&m, (r1.x[i], r1.y[i], r1.z[i]), (r2.x[i], r2.y[i], r2.z[i]));
    }
}

/// Eight correspondences per iteration of [`rotation_residuals_f32`], starting
/// at `base`.
///
/// # Safety
/// Requires the `avx2` target feature, and both ray sets must cover
/// `base .. base + out.len()` (all guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rotation_residuals_f32_avx2(
    rot: &Matrix3<f64>,
    r1: &RaysF32,
    r2: &RaysF32,
    base: usize,
    out: &mut [f32],
) {
    use crate::geometry::simd::{asin_ps, broadcast_mat3_ps, dot3_ps, row3_ps};
    use std::arch::x86_64::*;

    let n = out.len();
    let m = broadcast_mat3_ps(rot);
    let one = _mm256_set1_ps(1.0);
    let pi = _mm256_set1_ps(std::f32::consts::PI);
    let zero = _mm256_setzero_ps();

    let blocks = n / 8;
    for blk in 0..blocks {
        let k = base + blk * 8;
        let ax = _mm256_loadu_ps(r1.x.as_ptr().add(k));
        let ay = _mm256_loadu_ps(r1.y.as_ptr().add(k));
        let az = _mm256_loadu_ps(r1.z.as_ptr().add(k));
        let bx = _mm256_loadu_ps(r2.x.as_ptr().add(k));
        let by = _mm256_loadu_ps(r2.y.as_ptr().add(k));
        let bz = _mm256_loadu_ps(r2.z.as_ptr().add(k));

        let px = row3_ps(&m[0], ax, ay, az);
        let py = row3_ps(&m[1], ax, ay, az);
        let pz = row3_ps(&m[2], ax, ay, az);

        let c = dot3_ps(px, py, pz, bx, by, bz);
        let cx = _mm256_sub_ps(_mm256_mul_ps(py, bz), _mm256_mul_ps(pz, by));
        let cy = _mm256_sub_ps(_mm256_mul_ps(pz, bx), _mm256_mul_ps(px, bz));
        let cz = _mm256_sub_ps(_mm256_mul_ps(px, by), _mm256_mul_ps(py, bx));
        // Value first, matching `f32::min`: a NaN norm yields `1.0`.
        let s = _mm256_min_ps(_mm256_sqrt_ps(dot3_ps(cx, cy, cz, cx, cy, cz)), one);
        let ang = asin_ps(s);
        // `asin` covers `[0, π/2]`; the obtuse half is `π − asin|sin θ|`, and
        // the sign of the dot is what says which half a point is in.
        let neg = _mm256_cmp_ps(c, zero, _CMP_LT_OQ);
        let r = _mm256_blendv_ps(ang, _mm256_sub_ps(pi, ang), neg);
        _mm256_storeu_ps(out.as_mut_ptr().add(blk * 8), r);
    }

    let m = rot_f32(rot);
    for (j, o) in out.iter_mut().enumerate().skip(blocks * 8) {
        let i = base + j;
        *o = rotation_angle_f32(&m, (r1.x[i], r1.y[i], r1.z[i]), (r2.x[i], r2.y[i], r2.z[i]));
    }
}

/// Angular epipolar residual of every correspondence against `e`, one-sided.
///
/// `side_two` measures the angle between each image-2 ray and the epipolar
/// plane `E·x₁`; otherwise the image-1 ray is measured against `Eᵀ·x₂`. The two
/// are genuinely different measurements — a symmetric residual would score the
/// swapped correspondences identically, because the epipolar matrix of the swap
/// is exactly the transpose.
///
/// Dispatches to a bit-identical AVX2 kernel where the CPU has it (see
/// [`crate::geometry::simd`]); `epipolar_residuals_scalar` is both the fallback
/// and the reference the parity tests compare against.
pub(crate) fn epipolar_residuals(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    side_two: bool,
    out: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let n = r1.len();
        if crate::geometry::simd::avx2_enabled() && r2.len() >= n && out.len() >= n {
            // SAFETY: avx2 confirmed available, and both ray slices and `out`
            // cover the `n` points the kernel reads and writes.
            unsafe { epipolar_residuals_avx2(e, r1, r2, side_two, out) };
            return;
        }
    }
    epipolar_residuals_scalar(e, r1, r2, side_two, out);
}

/// Scalar reference for [`epipolar_residuals`], and its fallback where AVX2 is
/// unavailable or switched off.
fn epipolar_residuals_scalar(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    side_two: bool,
    out: &mut [f64],
) {
    let et = e.transpose();
    for i in 0..r1.len() {
        let (n, other) = if side_two {
            (e * r1[i], r2[i])
        } else {
            (et * r2[i], r1[i])
        };
        let nn = n.norm().max(1e-15);
        out[i] = (n.dot(&other).abs() / nn).min(1.0);
    }
}

/// Four correspondences per iteration of [`epipolar_residuals`], one per lane.
///
/// Every lane repeats the scalar sequence exactly: the `3×3` matvec in
/// nalgebra's `gemv` order (`(m₀·x + m₁·y) + m₂·z` per row), the length-3 dot
/// products in nalgebra's unrolled `U3` order (`(a + b) + c`), then
/// `sqrt`/`max`/`abs`/`div`/`min`. The side branch and the transpose are
/// loop-invariant and hoisted out; nothing else moves.
///
/// # Safety
/// Requires the `avx2` target feature, `r2.len() >= r1.len()` and
/// `out.len() >= r1.len()` (all guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn epipolar_residuals_avx2(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    side_two: bool,
    out: &mut [f64],
) {
    use crate::geometry::simd::{broadcast_mat3, dot3, load_vec3x4, row3};
    use std::arch::x86_64::*;

    let n = r1.len();
    let et = e.transpose();
    let (m, src, other) = if side_two { (e, r1, r2) } else { (&et, r2, r1) };
    let m = broadcast_mat3(m);
    let floor = _mm256_set1_pd(1e-15);
    let one = _mm256_set1_pd(1.0);
    // `andnot` against the sign bit is what `f64::abs` compiles to.
    let sign = _mm256_set1_pd(-0.0);

    let blocks = n / 4;
    for b in 0..blocks {
        let (sx, sy, sz) = load_vec3x4(src.as_ptr().add(b * 4) as *const f64);
        let (ox, oy, oz) = load_vec3x4(other.as_ptr().add(b * 4) as *const f64);

        let nx = row3(&m[0], sx, sy, sz);
        let ny = row3(&m[1], sx, sy, sz);
        let nz = row3(&m[2], sx, sy, sz);

        // `norm().max(1e-15)`: value first, so a NaN norm yields the floor,
        // matching `f64::max`.
        let nn = _mm256_max_pd(_mm256_sqrt_pd(dot3(nx, ny, nz, nx, ny, nz)), floor);
        let d = _mm256_andnot_pd(sign, dot3(nx, ny, nz, ox, oy, oz));
        // `.min(1.0)`: value first again, so a NaN quotient yields `1.0`.
        let r = _mm256_min_pd(_mm256_div_pd(d, nn), one);
        _mm256_storeu_pd(out.as_mut_ptr().add(b * 4), r);
    }

    let rem = blocks * 4;
    if rem < n {
        epipolar_residuals_scalar(e, &r1[rem..], &r2[rem..n], side_two, &mut out[rem..]);
    }
}

/// Angle between each rotated ray and its measured partner.
///
/// The AVX2 path fills `out` with clamped cosines and then takes the angles
/// with the vector `acos` polynomial, whose scalar twin
/// [`crate::geometry::acos_poly::acos_poly_scalar`] serves the ragged tail and
/// the fallback below — one arithmetic in every arm, so the dispatch stays a
/// pure performance switch. The vectorized half is taken only when `idx` is a
/// consecutive run, which is the hot shape: the RANSAC scoring loops in
/// [`super::rotation_support_at`] and [`crate::geometry::relative_pose`] pass all `n`
/// points at every one of the ~128 minimal samples, while the arbitrary-subset
/// callers ([`super::fit_rotation`]'s trimmed support) run three times per grid point.
///
/// `idx` must be strictly increasing, which is what lets [`consecutive_run`]
/// settle that dispatch in constant time; every caller builds it as `0..n` or
/// as a filter over `0..n`.
pub(crate) fn rotation_residuals(
    rot: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    idx: &[usize],
    out: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let len = out.len().min(idx.len());
        if crate::geometry::simd::avx2_enabled()
            && consecutive_run(idx, len)
            && idx[0] + len <= r1.len().min(r2.len())
        {
            // SAFETY: avx2 confirmed available, and the consecutive run
            // `idx[0] .. idx[0] + len` lies inside both ray slices while `out`
            // covers `len` elements.
            unsafe { rotation_cosines_avx2(rot, &r1[idx[0]..], &r2[idx[0]..], &mut out[..len]) };
            if crate::geometry::acos_poly::libm_acos_enabled() {
                for o in out[..len].iter_mut() {
                    *o = o.acos();
                }
            } else {
                // SAFETY: avx2 confirmed available just above.
                unsafe { acos_slice_avx2(&mut out[..len]) };
            }
            return;
        }
    }
    rotation_residuals_scalar(rot, r1, r2, idx, out);
}

/// Scalar reference for [`rotation_residuals`], and its fallback where AVX2 is
/// unavailable, switched off, or `idx` is not a consecutive run.
fn rotation_residuals_scalar(
    rot: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    idx: &[usize],
    out: &mut [f64],
) {
    if crate::geometry::acos_poly::libm_acos_enabled() {
        for (o, &i) in out.iter_mut().zip(idx.iter()) {
            *o = (rot * r1[i]).dot(&r2[i]).clamp(-1.0, 1.0).acos();
        }
        return;
    }
    for (o, &i) in out.iter_mut().zip(idx.iter()) {
        *o = crate::geometry::acos_poly::acos_poly_scalar(
            (rot * r1[i]).dot(&r2[i]).clamp(-1.0, 1.0),
        );
    }
}

/// `clamp((rot·r1ᵢ) · r2ᵢ, −1, 1)` for the first `out.len()` points, four per
/// iteration — [`rotation_residuals`] without the `acos`.
///
/// The clamp is [`f64::clamp`]'s, which propagates a NaN rather than quieting
/// it, so the constants go *first* into `min`/`max` here (the opposite of the
/// `f64::min` convention in [`epipolar_residuals_avx2`]); see
/// [`crate::geometry::simd`].
///
/// # Safety
/// Requires the `avx2` target feature, and `r1` and `r2` must both cover
/// `out.len()` points (all guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rotation_cosines_avx2(
    rot: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    out: &mut [f64],
) {
    use crate::geometry::simd::{broadcast_mat3, dot3, load_vec3x4, row3};
    use std::arch::x86_64::*;

    let n = out.len();
    let m = broadcast_mat3(rot);
    let lo = _mm256_set1_pd(-1.0);
    let hi = _mm256_set1_pd(1.0);

    let blocks = n / 4;
    for b in 0..blocks {
        let (ax, ay, az) = load_vec3x4(r1.as_ptr().add(b * 4) as *const f64);
        let (bx, by, bz) = load_vec3x4(r2.as_ptr().add(b * 4) as *const f64);
        let px = row3(&m[0], ax, ay, az);
        let py = row3(&m[1], ax, ay, az);
        let pz = row3(&m[2], ax, ay, az);
        let c = dot3(px, py, pz, bx, by, bz);
        let c = _mm256_min_pd(hi, _mm256_max_pd(lo, c));
        _mm256_storeu_pd(out.as_mut_ptr().add(b * 4), c);
    }
    for i in blocks * 4..n {
        out[i] = (rot * r1[i]).dot(&r2[i]).clamp(-1.0, 1.0);
    }
}

/// In-place `acos` over a slice of clamped cosines, four lanes at a time.
///
/// The ragged tail takes [`crate::geometry::acos_poly::acos_poly_scalar`], the
/// bit-identical twin of the vector form, so a slice length that is not a
/// multiple of four changes nothing but the instruction count.
///
/// # Safety
/// Requires the `avx2` target feature (guarded by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn acos_slice_avx2(out: &mut [f64]) {
    use std::arch::x86_64::*;
    let blocks = out.len() / 4;
    for b in 0..blocks {
        let p = out.as_mut_ptr().add(b * 4);
        _mm256_storeu_pd(p, crate::geometry::simd::acos_pd(_mm256_loadu_pd(p)));
    }
    for o in out[blocks * 4..].iter_mut() {
        *o = crate::geometry::acos_poly::acos_poly_scalar(*o);
    }
}

#[cfg(test)]
mod tests;
