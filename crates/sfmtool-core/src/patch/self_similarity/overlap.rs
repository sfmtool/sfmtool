// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlap reading of the ZNCC self-similarity radius, for a bitmap with no
//! ring of pixels around it.
//!
//! At each shift `d` the template is correlated with the window moved by `d`
//! over only the samples `k` where `k` and `k + d` both lie inside the bitmap
//! and both carry data. The template's mean and spread, and the moved
//! window's, are taken over that overlap, per channel, so every `z(d)` is a
//! ZNCC on the same scale as the ringed reading's. Where the moved window stays
//! inside the bitmap and every sample carries data, the overlap is the whole
//! template and the reading is the ringed one.

use super::{grid_bounds, middle_span, FLAT_FLOOR, FLAT_NORM_SQ_EPS};
use super::{read_surface, PatchTile, SelfSimilarity, SelfSimilarityParams, SelfSimilarityParts};

/// One template rectangle `Ω = (x, y, w, h)` inside `tile`, read the overlap
/// way: at each shift, only the samples of `Ω` whose moved sample lies inside
/// the tile, and where both carry data, are correlated. `data` holds one flag
/// per sample, row-major `width × height`, `true` where the sample carries
/// data; `None` means every sample does.
///
/// A template with no sample carrying data has no reading: its radius, slide,
/// tolerance and surface are all `NaN`.
///
/// # Panics
///
/// Panics if the tile is malformed, `data` does not hold one flag per sample,
/// or the template is empty or does not lie inside the tile; the message names
/// the sizes.
pub fn zncc_self_similarity_radius_overlap(
    tile: &PatchTile<'_>,
    data: Option<&[bool]>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity {
    let prepared = OverlapPrepared::new(tile, data);
    let [x, y, w, h] = template;
    assert!(
        w > 0 && h > 0,
        "zncc_self_similarity_radius_overlap: the template is {w}×{h}, it needs at least one pixel"
    );
    assert!(
        x + w <= tile.width && y + h <= tile.height,
        "zncc_self_similarity_radius_overlap: the {w}×{h} template at ({x}, {y}) does not fit \
         in the {}×{} tile",
        tile.width,
        tile.height
    );
    let r = params.max_radius as usize;
    let sums: Vec<Sums> = shifts(r)
        .map(|(dx, dy)| prepared.sums(template, dx, dy))
        .collect();
    judge(&sums, prepared.channels, r, params)
}

/// A whole `R×R` bitmap, its middle square and the nine cells of the ZNCC
/// grid's split, each read the overlap way, as
/// [`zncc_self_similarity_radius_overlap`] reads a template. The middle and the
/// cells take their shifted windows from the rest of the bitmap where it
/// reaches, so the middle and the centre cell read exactly as the ringed
/// [`zncc_self_similarity_parts`](super::zncc_self_similarity_parts) reads them
/// when every sample carries data.
///
/// # Panics
///
/// Panics if the tile is malformed or not square, `resolution` (its side) is
/// under 3, or `data` does not hold one flag per sample; the message names the
/// sizes.
pub fn zncc_self_similarity_parts_overlap(
    bitmap: &PatchTile<'_>,
    data: Option<&[bool]>,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts {
    let resolution = bitmap.width;
    assert!(
        bitmap.height == resolution && resolution >= 3,
        "zncc_self_similarity_parts_overlap: the bitmap is {}×{}, it needs to be square and at \
         least 3×3",
        bitmap.width,
        bitmap.height
    );
    let prepared = OverlapPrepared::new(bitmap, data);
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

    let count = (2 * r + 1) * (2 * r + 1);
    let mut cells: [[Vec<Sums>; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(count)));
    let mut middle_sums = Vec::with_capacity(count);
    let mut whole = Vec::with_capacity(count);
    for (dx, dy) in shifts(r) {
        // The nine cells tile the bitmap and the sums add, so the whole
        // bitmap's sums are the cells' added.
        let mut total = Sums::default();
        for (row, cells_row) in cells.iter_mut().enumerate() {
            for (col, cell) in cells_row.iter_mut().enumerate() {
                let sums = prepared.sums(cell_rect(row, col), dx, dy);
                total.add(&sums);
                cell.push(sums);
            }
        }
        whole.push(total);
        middle_sums.push(prepared.sums(middle_rect, dx, dy));
    }
    let channels = prepared.channels;
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

/// A tile ready to read the overlap way: each channel centred by its own mean
/// over the samples that carry data, in `f64`, and one data flag per sample.
struct OverlapPrepared {
    values: Vec<f64>,
    data: Vec<bool>,
    channels: usize,
    width: usize,
    height: usize,
}

impl OverlapPrepared {
    fn new(tile: &PatchTile<'_>, data: Option<&[bool]>) -> Self {
        tile.validate();
        let (width, height, channels) = (tile.width, tile.height, tile.channels);
        let n = width * height;
        let data = match data {
            Some(flags) => {
                assert_eq!(
                    flags.len(),
                    n,
                    "zncc_self_similarity overlap: a {width}×{height} tile needs {n} data flags, \
                     not {}",
                    flags.len()
                );
                flags.to_vec()
            }
            None => vec![true; n],
        };
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

    /// The sums of the template `rect` over the overlap at shift `(dx, dy)`.
    fn sums(&self, rect: [usize; 4], dx: i64, dy: i64) -> Sums {
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
        return SelfSimilarity {
            radius: f64::NAN,
            slide: [f64::NAN; 2],
            tolerance: f64::NAN,
            surface: vec![f64::NAN; shift_count],
        };
    }
    let floor_sq = FLAT_FLOOR * FLAT_FLOOR;
    let textured: Vec<(usize, f64)> = (0..channels)
        .filter_map(|c| {
            let (norm, _, _) = centre.centred(c);
            (norm / centre.count >= floor_sq).then_some((c, norm))
        })
        .collect();
    if textured.is_empty() {
        return SelfSimilarity {
            radius: r as f64,
            slide: [0.0; 2],
            tolerance: f64::INFINITY,
            surface: vec![f64::NAN; shift_count],
        };
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
