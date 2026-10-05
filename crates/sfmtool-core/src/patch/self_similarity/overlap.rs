// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlap reading of the ZNCC self-similarity radius: the reading of a
//! bitmap as it is, with no pixels from outside it.
//!
//! At each shift `d` the template is correlated with the window moved by `d`
//! over only the samples `k` where `k` and `k + d` both lie inside the bitmap
//! and both carry data. The template's mean and spread, and the moved
//! window's, are taken over that overlap, per channel, so every `z(d)` is a
//! ZNCC on one scale. Where the moved window stays inside the bitmap and every
//! sample carries data, the overlap is the whole template.
//!
//! Two routes compute the same sums. Where every sample carries data, the
//! overlap at each shift is a rectangle: the **dense** route reads its moments
//! from summed-area tables and its cross sums from the SIMD kernel run over a
//! copy of the bitmap around the template, centred by the template's own mean
//! and padded with `r` zeros, where a moved sample past the edge adds nothing
//! to `Σ t·u`. Where some sample carries no data, the **masked**
//! route visits every sample of every overlap.

use super::kernels;
use super::{
    flat_reading, no_reading, read_surface, PatchTile, SelfSimilarity, SelfSimilarityParams,
    SelfSimilarityParts,
};
use super::{grid_bounds, middle_span, Kernel, FLAT_FLOOR, FLAT_NORM_SQ_EPS};

/// One template rectangle `Ω = (x, y, w, h)` inside `tile`, read the overlap
/// way: at each shift, only the samples of `Ω` whose moved sample lies inside
/// the tile, and where both carry data, are correlated. `data` holds one flag
/// per sample, row-major `width × height`, `true` where the sample carries
/// data; `None` means every sample does.
///
/// A template with `max_radius` px of tile around it on every side, every
/// sample carrying data, has the whole template as its overlap at every shift.
///
/// A template with no sample carrying data has no reading: its radius, ellipse,
/// tolerance and surface are all `NaN`.
///
/// # Panics
///
/// Panics if the tile is malformed, `data` does not hold one flag per sample,
/// or the template is empty or does not lie inside the tile; the message names
/// the sizes.
pub fn zncc_self_similarity_radius(
    tile: &PatchTile<'_>,
    data: Option<&[bool]>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity {
    radius_with(tile, data, template, params, Route::Auto)
}

/// A whole `R×R` bitmap, its middle square and the nine cells of the ZNCC
/// grid's split, each read the overlap way, as
/// [`zncc_self_similarity_radius`] reads a template. The middle and the
/// cells take their shifted windows from the rest of the bitmap where it
/// reaches, so the middle and the centre cell have their whole template as
/// the overlap at every shift when every sample carries data.
///
/// # Panics
///
/// Panics if the tile is malformed or not square, its side is under 3, or `data` does not hold one flag per sample; the message names the
/// sizes.
pub fn zncc_self_similarity_parts(
    bitmap: &PatchTile<'_>,
    data: Option<&[bool]>,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts {
    parts_with(bitmap, data, params, Route::Auto)
}

/// Which route computes the sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Route {
    /// The dense route with the dispatching kernel where every sample carries
    /// data, the masked route otherwise.
    Auto,
    /// The dense route with the scalar reference kernel, for the equivalence
    /// tests; every sample must carry data.
    #[cfg_attr(not(test), allow(dead_code))]
    DenseScalar,
    /// The masked route whatever the flags, for the equivalence tests.
    #[cfg_attr(not(test), allow(dead_code))]
    Masked,
}

/// The route a reading takes, with what it needs.
enum Reader<'a> {
    Dense(Kernel),
    Masked(Option<&'a [bool]>),
}

impl<'a> Reader<'a> {
    fn new(tile: &PatchTile<'_>, data: Option<&'a [bool]>, route: Route) -> Self {
        let n = tile.width * tile.height;
        if let Some(flags) = data {
            assert_eq!(
                flags.len(),
                n,
                "zncc_self_similarity overlap: a {}×{} tile needs {n} data flags, not {}",
                tile.width,
                tile.height,
                flags.len()
            );
        }
        let all_data = data.is_none_or(|flags| flags.iter().all(|&d| d));
        match route {
            Route::Auto if all_data => Reader::Dense(Kernel::Dispatch),
            Route::Auto | Route::Masked => Reader::Masked(data),
            Route::DenseScalar => {
                assert!(all_data, "the dense route needs every sample to carry data");
                Reader::Dense(Kernel::Scalar)
            }
        }
    }
}

pub(super) fn radius_with(
    tile: &PatchTile<'_>,
    data: Option<&[bool]>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
    route: Route,
) -> SelfSimilarity {
    tile.validate();
    let reader = Reader::new(tile, data, route);
    let [x, y, w, h] = template;
    assert!(
        w > 0 && h > 0,
        "zncc_self_similarity_radius: the template is {w}×{h}, it needs at least one pixel"
    );
    assert!(
        x + w <= tile.width && y + h <= tile.height,
        "zncc_self_similarity_radius: the {w}×{h} template at ({x}, {y}) does not fit \
         in the {}×{} tile",
        tile.width,
        tile.height
    );
    let r = params.max_radius as usize;
    let sums = match reader {
        Reader::Dense(kernel) => Dense::new(tile).sums(template, r, kernel),
        Reader::Masked(flags) => Masked::new(tile, flags).sums(template, r),
    };
    judge(&sums, tile.channels, r, params)
}

pub(super) fn parts_with(
    bitmap: &PatchTile<'_>,
    data: Option<&[bool]>,
    params: &SelfSimilarityParams,
    route: Route,
) -> SelfSimilarityParts {
    bitmap.validate();
    let resolution = bitmap.width;
    assert!(
        bitmap.height == resolution && resolution >= 3,
        "zncc_self_similarity_parts: the bitmap is {}×{}, it needs to be square and at \
         least 3×3",
        bitmap.width,
        bitmap.height
    );
    let reader = Reader::new(bitmap, data, route);
    let r = params.max_radius as usize;
    let bounds = grid_bounds(resolution as u32);
    let cell_rect = |row: usize, col: usize| {
        [
            bounds[col],
            bounds[row],
            bounds[col + 1] - bounds[col],
            bounds[row + 1] - bounds[row],
        ]
    };
    let middle = middle_span(resolution as u32);
    let middle_rect = [middle.start, middle.start, middle.len(), middle.len()];

    let (cells, middle_sums): ([[Vec<Sums>; 3]; 3], Vec<Sums>) = match reader {
        Reader::Dense(kernel) => {
            let dense = Dense::new(bitmap);
            (
                std::array::from_fn(|row| {
                    std::array::from_fn(|col| dense.sums(cell_rect(row, col), r, kernel))
                }),
                dense.sums(middle_rect, r, kernel),
            )
        }
        Reader::Masked(flags) => {
            let masked = Masked::new(bitmap, flags);
            (
                std::array::from_fn(|row| {
                    std::array::from_fn(|col| masked.sums(cell_rect(row, col), r))
                }),
                masked.sums(middle_rect, r),
            )
        }
    };
    // The nine cells tile the bitmap and the sums add, so the whole bitmap's
    // sums are the cells' added, shift by shift.
    let mut whole = vec![Sums::default(); middle_sums.len()];
    for cell in cells.iter().flatten() {
        for (total, sums) in whole.iter_mut().zip(cell) {
            total.add(sums);
        }
    }
    let channels = bitmap.channels;
    SelfSimilarityParts {
        whole: judge(&whole, channels, r, params),
        middle: judge(&middle_sums, channels, r, params),
        grid: std::array::from_fn(|row| {
            std::array::from_fn(|col| judge(&cells[row][col], channels, r, params))
        }),
    }
}

/// Every shift of the `(2r + 1)²` square, row-major from `(−r, −r)`.
fn shifts(r: usize) -> impl Iterator<Item = (i64, i64)> {
    let ri = r as i64;
    (-ri..=ri).flat_map(move |dy| (-ri..=ri).map(move |dx| (dx, dy)))
}

/// The sums of one template over one shift's overlap: the overlap's sample
/// count and, per channel, `Σ t`, `Σ u`, `Σ t²`, `Σ u²` and `Σ t·u` of the
/// template values `t = I(k)` and the moved values `u = I(k + d)`.
#[derive(Debug, Clone, Copy, Default)]
struct Sums {
    count: f64,
    channel: [[f64; 5]; 3],
}

impl Sums {
    fn add(&mut self, other: &Sums) {
        self.count += other.count;
        for (a, b) in self.channel.iter_mut().zip(&other.channel) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }

    /// Channel `c`'s centred sums of squares of the template and the moved
    /// window over the overlap, and their centred cross sum.
    fn centred(&self, c: usize) -> (f64, f64, f64) {
        let [st, su, stt, suu, stu] = self.channel[c];
        let n = self.count;
        (
            (stt - st * st / n).max(0.0),
            (suu - su * su / n).max(0.0),
            stu - st * su / n,
        )
    }
}

/// The most template rows one `f32` kernel call accumulates before its sums
/// are added into the `f64` totals, which bounds the `f32` rounding a large
/// template accumulates.
const BAND_ROWS: usize = 8;

/// How many multiples of `f64::EPSILON · √N · S` a centred sum of squares
/// taken from the summed-area tables must exceed to count as spread, where
/// `N` is the tile's sample count and `S` its channel's sum of squares over
/// the whole tile.
///
/// A rectangle's `Σ v²` and `Σ v` are differences of table entries as large
/// as `S`, so a window constant at a level far from the tile's mean keeps a
/// residue of a few `ε · S` after `Σ v² − (Σ v)² / n`. At a level of 60000
/// in a 24×24 tile that is about `1e-5`, which the absolute flat test would
/// take for texture.
const SAT_ROUNDING_ULPS: f64 = 64.0;

/// A tile ready for the dense route: each channel centred by its own mean in
/// `f64`, and the summed-area tables of the centred values and their squares.
///
/// Each template's cross sums are taken on a copy of the plane around it,
/// centred again by the template's own mean and padded with zeros past the
/// tile's edge (see [`Dense::sums`]).
struct Dense {
    /// `channels` planes of `width × height`, each centred by its own mean.
    values: Vec<f64>,
    width: usize,
    height: usize,
    channels: usize,
    /// Per channel, the `(width + 1) × (height + 1)` summed-area table of the
    /// centred values, and of their squares.
    sat1: Vec<f64>,
    sat2: Vec<f64>,
    /// Per channel, the centred sum of squares at or under which a
    /// rectangle's is the tables' rounding rather than spread (see
    /// [`SAT_ROUNDING_ULPS`]).
    rounding: [f64; 3],
}

impl Dense {
    fn new(tile: &PatchTile<'_>) -> Self {
        let (width, height, channels) = (tile.width, tile.height, tile.channels);
        let n = width * height;
        let sat_side = (width + 1) * (height + 1);
        let mut values = vec![0.0f64; channels * n];
        let mut sat1 = vec![0.0f64; channels * sat_side];
        let mut sat2 = vec![0.0f64; channels * sat_side];
        let mut rounding = [0.0f64; 3];
        for c in 0..channels {
            let plane = &tile.values[c * n..][..n];
            let mean = plane.iter().map(|&v| f64::from(v)).sum::<f64>() / n.max(1) as f64;
            let centred = &mut values[c * n..][..n];
            let s1 = &mut sat1[c * sat_side..][..sat_side];
            let s2 = &mut sat2[c * sat_side..][..sat_side];
            for y in 0..height {
                let (mut row1, mut row2) = (0.0f64, 0.0f64);
                for x in 0..width {
                    let v = f64::from(plane[y * width + x]) - mean;
                    centred[y * width + x] = v;
                    row1 += v;
                    row2 += v * v;
                    let at = (y + 1) * (width + 1) + x + 1;
                    s1[at] = s1[at - (width + 1)] + row1;
                    s2[at] = s2[at - (width + 1)] + row2;
                }
            }
            rounding[c] = SAT_ROUNDING_ULPS * f64::EPSILON * (n as f64).sqrt() * s2[sat_side - 1];
        }
        Self {
            values,
            width,
            height,
            channels,
            sat1,
            sat2,
            rounding,
        }
    }

    /// `Σ v` and `Σ v²` of channel `c`'s centred values over the rectangle
    /// `[x, y, w, h]` of the tile.
    fn rect_sums(&self, c: usize, x: usize, y: usize, w: usize, h: usize) -> (f64, f64) {
        let side = (self.width + 1) * (self.height + 1);
        let at = |sat: &[f64], xx: usize, yy: usize| sat[c * side + yy * (self.width + 1) + xx];
        let sum = |sat: &[f64]| {
            at(sat, x + w, y + h) - at(sat, x, y + h) - at(sat, x + w, y) + at(sat, x, y)
        };
        (sum(&self.sat1), sum(&self.sat2))
    }

    /// The template `rect`'s sums at every shift of the square, row-major from
    /// `(−r, −r)`, in the units of the tile's own mean.
    ///
    /// The kernel runs in `f32` on a copy of the plane `r` px around the
    /// template, centred by the template's own mean `a` per channel, with
    /// zeros past the tile's edge and 8 more zero columns per row for the
    /// kernel's widest load. The zeros are what make its cross sums overlap
    /// sums: a moved sample past the edge adds nothing to `Σ t·u`. Centring by
    /// the template's own mean keeps the `f32` products on the scale of the
    /// template's own spread, however far its mean is from the tile's. The
    /// cross sum is then taken back to the tile's units in `f64`: with
    /// `t = t′ + a` and `u = u′ + a`, `Σ t·u = Σ t′·u′ + a (Σ t + Σ u) − n a²`
    /// over the `n` samples of the overlap. The moments come from the tables
    /// over the overlap rectangle, so the zeros never enter them. A centred
    /// sum of squares, the template's or the moved window's, that is no more
    /// than the tables' rounding is set to exactly zero, so a constant window
    /// reads as flat however far its level is from the tile's mean.
    fn sums(&self, rect: [usize; 4], r: usize, kernel: Kernel) -> Vec<Sums> {
        let [x, y, w, h] = rect;
        let side = 2 * r + 1;
        let shift_count = side * side;
        let n = self.width * self.height;
        let stride = w + 2 * r + 8;
        let rows = h + 2 * r;
        let (width, height) = (self.width as i64, self.height as i64);
        let (left, top_row) = (x as i64 - r as i64, y as i64 - r as i64);
        let (x0, x1) = (left.max(0), (left + (w + 2 * r) as i64).min(width));
        // Per channel, the template's own mean and the raw cross sums at its
        // `(2r + 1)²` shifts, the kernel run over bands of at most
        // `BAND_ROWS` template rows.
        let mut offsets = [0.0f64; 3];
        let mut cross = vec![0.0f64; self.channels * shift_count];
        let mut band = vec![0.0f32; shift_count];
        let mut plane = vec![0.0f32; stride * rows];
        for c in 0..self.channels {
            let a = self.rect_sums(c, x, y, w, h).0 / (w * h) as f64;
            offsets[c] = a;
            let values = &self.values[c * n..][..n];
            plane.fill(0.0);
            for j in 0..rows {
                let ty = top_row + j as i64;
                if !(0..height).contains(&ty) {
                    continue;
                }
                let row = &values[ty as usize * self.width..][..self.width];
                for tx in x0..x1 {
                    plane[j * stride + (tx - left) as usize] = (row[tx as usize] - a) as f32;
                }
            }
            let mut top = 0;
            while top < h {
                let band_rows = BAND_ROWS.min(h - top);
                let band_rect = [r, r + top, w, band_rows];
                match kernel {
                    Kernel::Dispatch => {
                        kernels::cross_sums(&plane, stride, band_rect, r, &mut band)
                    }
                    Kernel::Scalar => {
                        kernels::cross_sums_scalar(&plane, stride, band_rect, r, &mut band)
                    }
                }
                for (total, &value) in cross[c * shift_count..][..shift_count]
                    .iter_mut()
                    .zip(&band)
                {
                    *total += f64::from(value);
                }
                top += band_rows;
            }
        }
        shifts(r)
            .enumerate()
            .map(|(index, (dx, dy))| {
                // The template columns and rows whose moved sample is inside
                // the tile.
                let x0 = (x as i64).max(-dx);
                let x1 = ((x + w) as i64).min(width - dx);
                let y0 = (y as i64).max(-dy);
                let y1 = ((y + h) as i64).min(height - dy);
                let mut sums = Sums::default();
                if x1 <= x0 || y1 <= y0 {
                    return sums;
                }
                let (ow, oh) = ((x1 - x0) as usize, (y1 - y0) as usize);
                let (tx, ty) = (x0 as usize, y0 as usize);
                let (ux, uy) = ((x0 + dx) as usize, (y0 + dy) as usize);
                let count = (ow * oh) as f64;
                sums.count = count;
                for c in 0..self.channels {
                    let (st, stt) = self.rect_sums(c, tx, ty, ow, oh);
                    let (su, suu) = self.rect_sums(c, ux, uy, ow, oh);
                    // `Sums::centred` takes `s² / n` in the same order, so a
                    // snapped sum of squares centres to exactly zero.
                    let snap = |s: f64, ss: f64| {
                        if ss - s * s / count <= self.rounding[c] {
                            s * s / count
                        } else {
                            ss
                        }
                    };
                    let (stt, suu) = (snap(st, stt), snap(su, suu));
                    let a = offsets[c];
                    let stu = cross[c * shift_count + index] + a * (st + su) - count * a * a;
                    sums.channel[c] = [st, su, stt, suu, stu];
                }
                sums
            })
            .collect()
    }
}

/// A tile ready for the masked route: each channel centred by its own mean
/// over the samples that carry data, in `f64`, and one data flag per sample.
struct Masked {
    values: Vec<f64>,
    data: Vec<bool>,
    channels: usize,
    width: usize,
    height: usize,
}

impl Masked {
    fn new(tile: &PatchTile<'_>, data: Option<&[bool]>) -> Self {
        let (width, height, channels) = (tile.width, tile.height, tile.channels);
        let n = width * height;
        let data = data.map_or_else(|| vec![true; n], <[bool]>::to_vec);
        let with_data = data.iter().filter(|&&d| d).count();
        let mut values = vec![0.0f64; channels * n];
        for c in 0..channels {
            let plane = &tile.values[c * n..][..n];
            let mean = if with_data == 0 {
                0.0
            } else {
                plane
                    .iter()
                    .zip(&data)
                    .filter(|(_, &d)| d)
                    .map(|(&v, _)| f64::from(v))
                    .sum::<f64>()
                    / with_data as f64
            };
            for (out, &v) in values[c * n..][..n].iter_mut().zip(plane) {
                *out = f64::from(v) - mean;
            }
        }
        Self {
            values,
            data,
            channels,
            width,
            height,
        }
    }

    /// The template `rect`'s sums at every shift of the square, row-major from
    /// `(−r, −r)`.
    fn sums(&self, rect: [usize; 4], r: usize) -> Vec<Sums> {
        shifts(r)
            .map(|(dx, dy)| self.sums_at(rect, dx, dy))
            .collect()
    }

    /// The sums of the template `rect` over the overlap at shift `(dx, dy)`.
    fn sums_at(&self, rect: [usize; 4], dx: i64, dy: i64) -> Sums {
        let [x0, y0, w, h] = rect;
        let (width, height) = (self.width as i64, self.height as i64);
        // The template columns and rows whose moved sample is inside the tile.
        let xs = (x0 as i64).max(-dx)..((x0 + w) as i64).min(width - dx);
        let ys = (y0 as i64).max(-dy)..((y0 + h) as i64).min(height - dy);
        let n = self.width * self.height;
        let mut sums = Sums::default();
        for y in ys {
            for x in xs.clone() {
                let k = (y * width + x) as usize;
                let m = ((y + dy) * width + x + dx) as usize;
                if !(self.data[k] && self.data[m]) {
                    continue;
                }
                sums.count += 1.0;
                for c in 0..self.channels {
                    let t = self.values[c * n + k];
                    let u = self.values[c * n + m];
                    let s = &mut sums.channel[c];
                    s[0] += t;
                    s[1] += u;
                    s[2] += t * t;
                    s[3] += u * u;
                    s[4] += t * u;
                }
            }
        }
        sums
    }
}

/// Judge one template from its sums at every shift of the square (row-major
/// from `(−r, −r)`, the centre's being the template's own samples that carry
/// data).
///
/// The channels judged are those textured over the whole template, at the
/// centre, and the tolerance comes from their spreads there. At each other
/// shift a channel whose template spread over that shift's overlap is under
/// the flat floor is left out as well; a shift with no channel left, or with no
/// overlap, scores 0.
fn judge(
    sums: &[Sums],
    channels: usize,
    r: usize,
    params: &SelfSimilarityParams,
) -> SelfSimilarity {
    let side = 2 * r + 1;
    let shift_count = side * side;
    let centre = &sums[r * side + r];
    if centre.count == 0.0 {
        return no_reading(r);
    }
    let floor_sq = FLAT_FLOOR * FLAT_FLOOR;
    let textured: Vec<(usize, f64)> = (0..channels)
        .filter_map(|c| {
            let (norm, _, _) = centre.centred(c);
            (norm / centre.count >= floor_sq).then_some((c, norm))
        })
        .collect();
    if textured.is_empty() {
        return flat_reading(r);
    }
    let noise_term = textured
        .iter()
        .map(|&(_, norm)| params.noise * params.noise * centre.count / norm)
        .sum::<f64>()
        / textured.len() as f64;
    let tolerance = params.relative_tolerance + noise_term;

    let mut surface = vec![f64::NAN; shift_count];
    for (index, (dx, dy)) in shifts(r).enumerate() {
        if dx == 0 && dy == 0 {
            surface[index] = 1.0;
            continue;
        }
        let s = &sums[index];
        let mut total = 0.0f64;
        let mut used = 0usize;
        if s.count > 0.0 {
            for &(c, _) in &textured {
                let (norm, window_norm, cross) = s.centred(c);
                if norm / s.count < floor_sq {
                    continue;
                }
                used += 1;
                if window_norm >= FLAT_NORM_SQ_EPS {
                    total += cross / (norm * window_norm).sqrt();
                }
            }
        }
        surface[index] = if used == 0 { 0.0 } else { total / used as f64 };
    }
    read_surface(surface, r, tolerance)
}
