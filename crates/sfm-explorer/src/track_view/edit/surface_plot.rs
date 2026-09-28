// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The self-similarity surface plot: a patch's ZNCC against itself at every
//! whole-pixel shift, drawn as a heatmap with the contour the radius is read
//! at.
//!
//! The bench carries the surface as a `(2r + 1)²` square of ZNCC values
//! round the disk of shifts the score searches, and the tolerance `τ` the
//! patch was judged by. A shift is indistinguishable from the true position
//! where the ZNCC is at or above `1 - τ`, so that level is the one drawn. The
//! square is interpolated to a finer grid for the picture and the contour, and
//! the colour ramp jumps at the level so the region inside the contour reads as
//! one bright shape: a small ring round the centre is a patch that locks, a
//! long one is a patch that slides along it, and one that runs to the edge of
//! the disk is a patch that slides further than the score looks.

/// Interpolated samples per pixel of shift. At `r = 3` the picture is
/// `6 · 16 + 1 = 97` samples across.
const SAMPLES_PER_PIXEL: usize = 16;

/// How far outside the disk of searched shifts the picture still shows the
/// interpolated surface, in pixels of shift, so the disk's edge is not cut
/// through the middle of a lattice point's neighbourhood.
const DISK_MARGIN: f64 = 0.35;

/// A surface ready to draw: the interpolated values, the level the radius is
/// read at, and which shifts are indistinguishable.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SurfacePlot {
    /// Interpolated ZNCC, row-major, `n × n`, `NaN` outside the disk.
    pub values: Vec<f64>,
    /// Samples across the picture.
    pub n: usize,
    /// The radius of the disk of searched shifts, `r`.
    pub r: usize,
    /// The ZNCC level of the contour, `1 - τ`.
    pub level: f64,
    /// The indistinguishable shifts, `(dx, dy)` with `dx² + dy² ≤ r²`, the
    /// centre left out.
    pub inside: Vec<[i32; 2]>,
}

impl SurfacePlot {
    /// The plot of a measured surface, or `None` when there is nothing to
    /// draw: a surface that is not a square of odd side, or one missing a
    /// reading, as the `NaN` surface of a core with no texture is.
    pub(super) fn new(surface: &[f64], tolerance: f64) -> Option<SurfacePlot> {
        let side = (surface.len() as f64).sqrt().round() as usize;
        if side * side != surface.len()
            || side.is_multiple_of(2)
            || side < 3
            || !tolerance.is_finite()
        {
            return None;
        }
        let r = side / 2;
        if surface.iter().any(|z| !z.is_finite()) {
            return None;
        }
        let level = 1.0 - tolerance;
        let n = (side - 1) * SAMPLES_PER_PIXEL + 1;
        let mut values = upsample(surface, side, n);
        let reach = r as f64 + DISK_MARGIN;
        for (k, v) in values.iter_mut().enumerate() {
            let (x, y) = (k % n, k / n);
            let dx = x as f64 / SAMPLES_PER_PIXEL as f64 - r as f64;
            let dy = y as f64 / SAMPLES_PER_PIXEL as f64 - r as f64;
            if dx.hypot(dy) > reach {
                *v = f64::NAN;
            }
        }
        let mut inside = Vec::new();
        for (k, &z) in surface.iter().enumerate() {
            let dx = (k % side) as i32 - r as i32;
            let dy = (k / side) as i32 - r as i32;
            if (dx, dy) != (0, 0) && z.is_finite() && z >= level {
                inside.push([dx, dy]);
            }
        }
        Some(SurfacePlot {
            values,
            n,
            r,
            level,
            inside,
        })
    }

    /// The picture, one pixel per interpolated sample, transparent outside
    /// the disk.
    pub(super) fn image(&self) -> egui::ColorImage {
        let pixels = self
            .values
            .iter()
            .map(|&z| {
                if z.is_finite() {
                    surface_color(z, self.level)
                } else {
                    egui::Color32::TRANSPARENT
                }
            })
            .collect();
        egui::ColorImage::new([self.n, self.n], pixels)
    }

    /// The contour at [`Self::level`], as line segments in the unit square
    /// the picture is drawn in (`[0, 0]` its top-left corner).
    pub(super) fn contour(&self) -> Vec<[[f32; 2]; 2]> {
        let scale = 1.0 / (self.n - 1) as f32;
        contour_segments(&self.values, self.n, self.level)
            .into_iter()
            .map(|seg| seg.map(|[x, y]| [x * scale, y * scale]))
            .collect()
    }

    /// Where the shift `(dx, dy)` falls in the unit square the picture is
    /// drawn in.
    pub(super) fn shift_position(&self, dx: i32, dy: i32) -> [f32; 2] {
        let side = (2 * self.r) as f32;
        [
            (dx + self.r as i32) as f32 / side,
            (dy + self.r as i32) as f32 / side,
        ]
    }
}

/// The colour of ZNCC `z` against the contour `level`: below it a muted ramp
/// from dark to mid slate, and at it a jump to bright amber that lightens
/// towards pale yellow at `1`. The jump is what makes the region inside the
/// contour read as one shape.
pub(super) fn surface_color(z: f64, level: f64) -> egui::Color32 {
    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| {
        let t = t.clamp(0.0, 1.0);
        let c: [f64; 3] = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
        egui::Color32::from_rgb(c[0] as u8, c[1] as u8, c[2] as u8)
    };
    const DEEP: [f64; 3] = [22.0, 30.0, 44.0];
    const SLATE: [f64; 3] = [64.0, 82.0, 112.0];
    const AMBER: [f64; 3] = [236.0, 164.0, 44.0];
    const PALE: [f64; 3] = [255.0, 244.0, 196.0];
    if z < level {
        // The last 0.4 of ZNCC below the level carries the ramp; anything
        // lower is the deepest shade.
        lerp(DEEP, SLATE, (z - (level - 0.4)) / 0.4)
    } else {
        lerp(AMBER, PALE, (z - level) / (1.0 - level).max(1e-6))
    }
}

/// Catmull-Rom interpolation of a `side × side` square of values to `n × n`
/// samples spanning the same extent, the first and last samples on the
/// square's corners. A neighbour past the square's edge repeats the edge's
/// value. Separable: rows, then columns.
fn upsample(values: &[f64], side: usize, n: usize) -> Vec<f64> {
    let at = |row: &[f64], x: f64| {
        let i = (x.floor() as isize).clamp(0, side as isize - 2) as usize;
        let t = x - i as f64;
        let p = |k: isize| row[(i as isize + k).clamp(0, side as isize - 1) as usize];
        let (p0, p1, p2, p3) = (p(-1), p(0), p(1), p(2));
        0.5 * (2.0 * p1
            + (p2 - p0) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t * t)
    };
    let step = (side - 1) as f64 / (n - 1) as f64;
    // Across each row of the square first.
    let wide: Vec<f64> = (0..side)
        .flat_map(|row| {
            let src = &values[row * side..][..side];
            (0..n).map(move |x| at(src, x as f64 * step))
        })
        .collect();
    // Then down each column of that.
    let mut out = vec![0.0; n * n];
    let mut column = vec![0.0; side];
    for x in 0..n {
        for (row, c) in column.iter_mut().enumerate() {
            *c = wide[row * n + x];
        }
        for y in 0..n {
            out[y * n + x] = at(&column, y as f64 * step);
        }
    }
    out
}

/// Marching squares: the segments of the `level` contour of an `n × n` grid,
/// in grid coordinates (`[x, y]`, `[0, 0]` the first sample). A cell with a
/// `NaN` corner is skipped. The two ambiguous cases are split by the cell's
/// centre value.
pub(super) fn contour_segments(values: &[f64], n: usize, level: f64) -> Vec<[[f32; 2]; 2]> {
    let mut out = Vec::new();
    for y in 0..n.saturating_sub(1) {
        for x in 0..n.saturating_sub(1) {
            let v = [
                values[y * n + x],
                values[y * n + x + 1],
                values[(y + 1) * n + x + 1],
                values[(y + 1) * n + x],
            ];
            if v.iter().any(|z| !z.is_finite()) {
                continue;
            }
            let above = v.map(|z| z >= level);
            let case = above
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &a)| acc | (u8::from(a) << i));
            if case == 0 || case == 15 {
                continue;
            }
            // The crossing on each edge, corners in order top-left, top-right,
            // bottom-right, bottom-left.
            let corner = [[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            let cross = |a: usize, b: usize| {
                let t = ((level - v[a]) / (v[b] - v[a])).clamp(0.0, 1.0);
                let p = [
                    corner[a][0] + (corner[b][0] - corner[a][0]) * t,
                    corner[a][1] + (corner[b][1] - corner[a][1]) * t,
                ];
                [(x as f64 + p[0]) as f32, (y as f64 + p[1]) as f32]
            };
            let top = || cross(0, 1);
            let right = || cross(1, 2);
            let bottom = || cross(2, 3);
            let left = || cross(3, 0);
            let centre_above = v.iter().sum::<f64>() / 4.0 >= level;
            match case {
                1 | 14 => out.push([left(), top()]),
                2 | 13 => out.push([top(), right()]),
                3 | 12 => out.push([left(), right()]),
                4 | 11 => out.push([right(), bottom()]),
                6 | 9 => out.push([top(), bottom()]),
                7 | 8 => out.push([left(), bottom()]),
                5 => {
                    // Top-left and bottom-right above.
                    if centre_above {
                        out.push([left(), bottom()]);
                        out.push([top(), right()]);
                    } else {
                        out.push([left(), top()]);
                        out.push([right(), bottom()]);
                    }
                }
                10 => {
                    // Top-right and bottom-left above.
                    if centre_above {
                        out.push([left(), top()]);
                        out.push([right(), bottom()]);
                    } else {
                        out.push([top(), right()]);
                        out.push([left(), bottom()]);
                    }
                }
                _ => unreachable!("cases 0 and 15 have no crossing"),
            }
        }
    }
    out
}

/// A plot with its picture uploaded and its contour traced, ready to paint
/// every frame without recomputing either.
pub(super) struct DrawnPlot {
    pub plot: SurfacePlot,
    pub contour: Vec<[[f32; 2]; 2]>,
    pub texture: egui::TextureHandle,
}

impl DrawnPlot {
    /// Upload `plot`'s picture under `name` and trace its contour.
    pub(super) fn new(ctx: &egui::Context, plot: SurfacePlot, name: String) -> DrawnPlot {
        let texture = ctx.load_texture(name, plot.image(), egui::TextureOptions::LINEAR);
        let contour = plot.contour();
        DrawnPlot {
            plot,
            contour,
            texture,
        }
    }

    /// Draw into `rect`: the heatmap, the contour over it, a dot on each
    /// indistinguishable shift and a ring on the centre. `fade` greys it while
    /// an evaluation is on its way, as it does the numbers.
    pub(super) fn paint(&self, painter: &egui::Painter, rect: egui::Rect, fade: f32) {
        paint(
            painter,
            rect,
            &self.plot,
            &self.contour,
            self.texture.id(),
            fade,
        );
    }
}

fn paint(
    painter: &egui::Painter,
    rect: egui::Rect,
    plot: &SurfacePlot,
    contour: &[[[f32; 2]; 2]],
    texture: egui::TextureId,
    fade: f32,
) {
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    painter.image(texture, rect, uv, egui::Color32::WHITE.gamma_multiply(fade));
    let at = |p: [f32; 2]| rect.min + egui::vec2(p[0] * rect.width(), p[1] * rect.height());
    let line = (rect.width() / 40.0).clamp(1.0, 2.0);
    for &[a, b] in contour {
        painter.line_segment(
            [at(a), at(b)],
            egui::Stroke::new(line, egui::Color32::WHITE.gamma_multiply(fade)),
        );
    }
    let dot = (rect.width() / 50.0).clamp(0.8, 3.0);
    for &[dx, dy] in &plot.inside {
        painter.circle_filled(
            at(plot.shift_position(dx, dy)),
            dot,
            egui::Color32::from_black_alpha(200).gamma_multiply(fade),
        );
    }
    painter.circle_stroke(
        at(plot.shift_position(0, 0)),
        dot * 1.6,
        egui::Stroke::new(
            1.0,
            egui::Color32::from_black_alpha(220).gamma_multiply(fade),
        ),
    );
}
