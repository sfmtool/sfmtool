// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Decoding a rectangle of a JPEG from only the restart intervals that hold
//! it: a tile of a mip level file ([`MipLevelFile::decode_region`]) and a
//! region of a region-readable original ([`OriginalRegions`]).
//!
//! Both assemble a small JPEG from the file's own tables, a frame of another
//! size and the intervals the rectangle needs, with the restart markers
//! between them numbered again from `RST0`, and decode it with
//! `jpeg-decoder`. Each interval starts with the DC predictions reset, so it
//! decodes the same wherever it is placed.
//!
//! [`MipLevelFile::decode_region`]: super::MipLevelFile::decode_region

use std::ops::Range;
use std::sync::Arc;

use super::region::{read_header, region_layout, JpegHeader, NotRegionReadable};
use crate::camera::image::{jpeg_error, ImageU8};

/// The largest side of a JPEG frame.
const MAX_FRAME_SIDE: usize = 65535;

/// A file's bytes with its restart index: where each restart interval's
/// entropy-coded data is, and the bytes a small JPEG made from some of them
/// starts with. Made once per file.
#[derive(Clone, Debug)]
pub(super) struct RestartIndex {
    bytes: Arc<[u8]>,
    /// Start-of-image, then every segment a decode reads up to and including
    /// the scan header: the tables, the frame, the restart interval.
    prefix: Arc<[u8]>,
    /// The offset in `prefix` of the frame's height, which its width follows.
    size_at: usize,
    /// Each interval's bytes in `bytes`, in scan order, without its marker.
    intervals: Arc<[Range<usize>]>,
}

impl RestartIndex {
    /// The index of `bytes`, whose header walk is `header` and whose scan
    /// indexed into `intervals`.
    pub(super) fn new(
        bytes: Arc<[u8]>,
        header: &JpegHeader,
        intervals: Vec<Range<usize>>,
    ) -> Result<Self, String> {
        let mut prefix = vec![0xFF, 0xD8];
        let mut size_at = None;
        for segment in &header.decode_segments {
            let Some(data) = bytes.get(segment.clone()) else {
                return Err("a segment runs past the end of the file".to_string());
            };
            // The frame header's body starts 4 bytes into its segment, after
            // the marker and the length; the height follows the precision.
            if segment.start + 4 == header.frame_at {
                size_at = Some(prefix.len() + 5);
            }
            prefix.extend_from_slice(data);
        }
        let Some(size_at) = size_at.filter(|&at| at + 4 <= prefix.len()) else {
            return Err("it has no frame header".to_string());
        };
        if intervals.iter().any(|i| i.end > bytes.len()) {
            return Err("an interval runs past the end of the file".to_string());
        }
        Ok(Self {
            bytes,
            prefix: prefix.into(),
            size_at,
            intervals: intervals.into(),
        })
    }

    /// The file's bytes.
    pub(super) fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    /// The bytes the index holds: the file's, the prefix and the intervals'
    /// ranges.
    pub(super) fn held_bytes(&self) -> u64 {
        (self.bytes.len()
            + self.prefix.len()
            + self.intervals.len() * std::mem::size_of::<Range<usize>>()) as u64
    }

    /// The number of restart intervals.
    pub(super) fn len(&self) -> usize {
        self.intervals.len()
    }

    /// A JPEG of `width` x `height` whose scan is the intervals `chosen`, in
    /// that order, separated by `RST0` to `RST7` in sequence from the first.
    /// Every index in `chosen` is below [`len`](Self::len).
    fn assemble(&self, width: u16, height: u16, chosen: &[usize]) -> Vec<u8> {
        let size: usize = chosen.iter().map(|&i| self.intervals[i].len() + 2).sum();
        let mut jpeg = Vec::with_capacity(self.prefix.len() + size + 2);
        jpeg.extend_from_slice(&self.prefix);
        jpeg[self.size_at..self.size_at + 2].copy_from_slice(&height.to_be_bytes());
        jpeg[self.size_at + 2..self.size_at + 4].copy_from_slice(&width.to_be_bytes());
        for (k, &i) in chosen.iter().enumerate() {
            if k > 0 {
                jpeg.extend_from_slice(&[0xFF, 0xD0 + ((k - 1) % 8) as u8]);
            }
            jpeg.extend_from_slice(&self.bytes[self.intervals[i].clone()]);
        }
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        jpeg
    }
}

/// `Err` saying why `x0..x1` by `y0..y1` is not a non-empty rectangle of a
/// `width` x `height` image.
pub(super) fn check_rectangle(
    width: u32,
    height: u32,
    (x0, y0, x1, y1): (u32, u32, u32, u32),
) -> Result<(), String> {
    if x0 >= x1 || y0 >= y1 || x1 > width || y1 > height {
        return Err(format!(
            "pixels {x0}..{x1} by {y0}..{y1} are not a non-empty rectangle of \
             a {width} x {height} image"
        ));
    }
    Ok(())
}

/// What a tile decode of a mip level file needs from its record.
pub(super) struct TileGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
    /// 1 or 3; a mip level file is 4:4:4, so an MCU is 8 x 8 pixels.
    pub(super) components: usize,
    /// `N`, the restart interval in MCUs.
    pub(super) restart_interval: usize,
}

/// Pixels `x0..x1` by `y0..y1` of a mip level file, equal to the same pixels
/// of its whole decode. The rectangle is checked by the caller.
///
/// With `M` MCUs per row and `N` per interval, the MCU at row `r`, column `c`
/// is in interval `(r·M + c) div N`, so an interval may run from the end of
/// one MCU row into the next. Each needed interval becomes one MCU row of a
/// small frame `8·N` wide, and its MCUs are copied to their places from
/// there; the level's last interval, which holds fewer than `N` MCUs when
/// the level's MCU count is not a multiple of `N`, is decoded in a frame of
/// its own, as wide as it is. Each frame is decoded to its samples as coded
/// and converted to RGB with [`ycbcr_to_rgb`], as the whole decode is.
pub(super) fn decode_mip_level_region(
    index: &RestartIndex,
    geometry: &TileGeometry,
    (x0, y0, x1, y1): (u32, u32, u32, u32),
) -> Result<ImageU8, String> {
    let n = geometry.restart_interval;
    let per_row = geometry.width.div_ceil(8) as usize;
    let mcus = per_row * geometry.height.div_ceil(8) as usize;
    if n == 0 || 8 * n > MAX_FRAME_SIDE {
        return Err(format!(
            "a restart interval of {n} MCUs cannot be decoded alone"
        ));
    }
    if index.len() != mcus.div_ceil(n) {
        return Err("its restart index does not fit its size".to_string());
    }
    // The intervals holding the MCUs the rectangle covers, in order.
    let (c0, c1) = ((x0 / 8) as usize, x1.div_ceil(8) as usize);
    let mut needed: Vec<usize> = Vec::new();
    for row in (y0 / 8) as usize..y1.div_ceil(8) as usize {
        let first = (row * per_row + c0) / n;
        let last = (row * per_row + c1 - 1) / n;
        let from = needed.last().map_or(first, |&l| first.max(l + 1));
        needed.extend(from..=last);
    }
    let short = mcus % n;
    let last = (short != 0 && needed.last() == Some(&(index.len() - 1))).then(|| needed.pop());

    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let mut out = vec![0u8; w * h * 3];
    let mut place = |chosen: &[usize], mcus_per_interval: usize| -> Result<(), String> {
        let frame_width = 8 * mcus_per_interval;
        let frame_height = 8 * chosen.len();
        let jpeg = index.assemble(frame_width as u16, frame_height as u16, chosen);
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        decoder.set_color_transform(jpeg_decoder::ColorTransform::None);
        let samples = decoder.decode().map_err(|e| {
            format!(
                "restart intervals {} to {} do not decode: {e}",
                chosen[0],
                chosen[chosen.len() - 1]
            )
        })?;
        let line = frame_width * geometry.components;
        if samples.len() != line * frame_height {
            return Err("its intervals decode to the wrong number of samples".to_string());
        }
        for (j, &interval) in chosen.iter().enumerate() {
            for m in 0..mcus_per_interval {
                let g = interval * n + m;
                let (mx, my) = ((g % per_row) as u32 * 8, (g / per_row) as u32 * 8);
                let (xa, xb) = (mx.max(x0), (mx + 8).min(x1));
                let (ya, yb) = (my.max(y0), (my + 8).min(y1));
                if xa >= xb {
                    continue;
                }
                for y in ya..yb {
                    let row = &samples[(8 * j + (y - my) as usize) * line..][..line];
                    let target = ((y - y0) as usize * w + (xa - x0) as usize) * 3;
                    let column = 8 * m + (xa - mx) as usize;
                    let pixels = (xb - xa) as usize;
                    let out = &mut out[target..target + 3 * pixels];
                    // A line holds each component's samples in turn.
                    if geometry.components == 1 {
                        for (p, &v) in out
                            .as_chunks_mut::<3>()
                            .0
                            .iter_mut()
                            .zip(&row[column..column + pixels])
                        {
                            *p = [v, v, v];
                        }
                    } else {
                        let fw = frame_width;
                        for (k, p) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                            let c = column + k;
                            *p = ycbcr_to_rgb(row[c], row[fw + c], row[2 * fw + c]);
                        }
                    }
                }
            }
        }
        Ok(())
    };
    for chosen in needed.chunks(MAX_FRAME_SIDE / 8) {
        place(chosen, n)?;
    }
    if let Some(Some(last)) = last {
        place(&[last], short)?;
    }
    Ok(ImageU8::new(w as u32, h as u32, 3, out))
}

/// `jpeg-decoder` 0.3.2's scalar YCbCr to RGB conversion (`ycbcr_to_rgb` in
/// its `decoder.rs`, from libjpeg-turbo's `jdcolext.c`): 20-bit fixed point,
/// rounded at half. Its SIMD conversion rounds differently and leaves the
/// last 8 to 15 pixels of each line to this formula, so a tile, whose lines
/// end elsewhere, matches the whole decode only when both use this one.
pub(super) fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    const SHIFT: i32 = 20;
    const HALF: i32 = 1 << (SHIFT - 1);
    let fixed = |x: f32| (x * (1 << SHIFT) as f32 + 0.5) as i32;
    let clamp = |v: i32| (v >> SHIFT).clamp(0, 255) as u8;
    let y = i32::from(y) * (1 << SHIFT) + HALF;
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    [
        clamp(y + fixed(1.40200) * cr),
        clamp(y - fixed(0.34414) * cb - fixed(0.71414) * cr),
        clamp(y + fixed(1.77200) * cb),
    ]
}

/// Region reads of a region-readable original: a rectangle decoded from the
/// restart intervals that hold it, equal to the same pixels of
/// `ImageU8::read_rgb` of the file. Made once per file, from its bytes, by
/// the region test.
#[derive(Clone, Debug)]
pub struct OriginalRegions {
    index: RestartIndex,
    width: u32,
    height: u32,
    components: u8,
    /// An MCU's width and height in pixels.
    mcu_size: (u32, u32),
    restart_interval: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
}

impl OriginalRegions {
    /// The region reader of the original whose bytes are `bytes`.
    ///
    /// # Errors
    ///
    /// The first rule of the region test the bytes fail.
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, NotRegionReadable> {
        let layout = region_layout(&bytes)?;
        // The region test read the same header; it cannot fail now.
        let header = read_header(&bytes).map_err(NotRegionReadable::Scan)?;
        let (h, v) = layout.luma_sampling;
        let mcu_size = if layout.components == 3 {
            (8 * u32::from(h), 8 * u32::from(v))
        } else {
            (8, 8)
        };
        let index =
            RestartIndex::new(bytes, &header, layout.intervals).map_err(NotRegionReadable::Scan)?;
        Ok(Self {
            index,
            width: layout.width,
            height: layout.height,
            components: layout.components,
            mcu_size,
            restart_interval: u32::from(layout.restart_interval),
            mcus_per_row: layout.mcus_per_row,
            mcu_rows: layout.mcu_rows,
        })
    }

    /// The original's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The original's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The original's bytes.
    pub fn bytes(&self) -> &Arc<[u8]> {
        self.index.bytes()
    }

    /// The bytes held: the original's and its restart index's, what the
    /// photograph cache counts against its budget.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.index.held_bytes()
    }

    /// Pixels `x0..x1` by `y0..y1` of the original as RGB, equal to the same
    /// pixels of `ImageU8::read_rgb` of the file.
    ///
    /// Each small JPEG keeps the original's layout: its columns are MCU
    /// columns from an interval boundary to an interval boundary, its rows
    /// the rectangle's MCU rows with a margin, and it is decoded to RGB by
    /// the `jpeg-decoder` path `read_rgb` takes, then cropped. One MCU above,
    /// below and to the left holds the neighbours chroma upsampling reads,
    /// and one MCU on the right also keeps the cropped pixels out of the last
    /// 8 px of a line, which the decoder converts with its scalar formula
    /// rather than its SIMD one: a frame a whole number of MCUs wide leaves it
    /// exactly those. Such a frame ends at least one MCU before the image's
    /// last, so in the whole decode, whose scalar part is the last 8 to 15 px,
    /// the cropped pixels are converted by the SIMD formula too. Where a
    /// frame reaches the image's right or bottom edge it ends at that edge,
    /// so the edge is padded as in the whole decode, and, the frame starting
    /// on a multiple of 8 px, the scalar part of each line falls on the same
    /// pixels in both.
    ///
    /// The columns are decoded in strips, each a frame no wider than
    /// [`DECODED_ON_THE_CALLING_THREAD`] px, whenever the restart interval is
    /// narrow enough for frames that wide to cover them with their margins,
    /// so the decode runs on the calling thread and starts none of its own;
    /// otherwise in one frame as wide as the rectangle and its margins.
    ///
    /// # Errors
    ///
    /// An `ImageError::Parameter` for a rectangle that is empty or not inside
    /// the image, and an `ImageError::Decoding` when the intervals it needs
    /// do not decode.
    pub fn decode_region(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<ImageU8, ::image::ImageError> {
        use ::image::error::{ParameterError, ParameterErrorKind};
        check_rectangle(self.width, self.height, (x0, y0, x1, y1)).map_err(|why| {
            ::image::ImageError::Parameter(ParameterError::from_kind(ParameterErrorKind::Generic(
                why,
            )))
        })?;
        let mh = self.mcu_size.1;
        let rows = (
            (y0 / mh).saturating_sub(1),
            (y1.div_ceil(mh) + 1).min(self.mcu_rows),
        );
        let strips = self
            .strips(x0, x1)
            .unwrap_or_else(|| vec![Strip::one_frame(self, x0, x1)]);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut out = vec![0; w * h * 3];
        for strip in strips {
            self.decode_strip(&strip, rows, (x0, y0, w, h), &mut out)?;
        }
        Ok(ImageU8::new(w as u32, h as u32, 3, out))
    }

    /// Pixel columns `x0..x1` split into strips, each decoded by a frame at
    /// most [`DECODED_ON_THE_CALLING_THREAD`] px wide that holds its columns
    /// with their margins, and none reaching past the one frame of
    /// [`Strip::one_frame`]; `None` when the restart interval is too wide for
    /// frames that narrow to cover them.
    fn strips(&self, x0: u32, x1: u32) -> Option<Vec<Strip>> {
        let (mw, r) = (self.mcu_size.0, self.restart_interval);
        let per_frame = DECODED_ON_THE_CALLING_THREAD / mw / r * r;
        let last = Strip::one_frame(self, x0, x1).columns.1;
        let mut strips = Vec::new();
        let mut x = x0;
        while x < x1 {
            let lo = (x / mw).saturating_sub(1) / r * r;
            let hi = (lo + per_frame).min(last);
            let end = if hi == self.mcus_per_row {
                x1
            } else {
                x1.min(hi.checked_sub(1)? * mw)
            };
            if end <= x {
                return None;
            }
            strips.push(Strip {
                columns: (lo, hi),
                pixels: (x, end),
            });
            x = end;
        }
        Some(strips)
    }

    /// Decode the frame of `strip`'s MCU columns and MCU rows `rows`, and
    /// copy its pixel columns, rows `y0..y0 + h`, into `out`, the `w` x `h`
    /// RGB rectangle whose left edge is `x0`.
    fn decode_strip(
        &self,
        strip: &Strip,
        (row_lo, row_hi): (u32, u32),
        (x0, y0, w, h): (u32, u32, usize, usize),
        out: &mut [u8],
    ) -> Result<(), ::image::ImageError> {
        let (mw, mh) = self.mcu_size;
        let r = self.restart_interval;
        let (col_lo, col_hi) = strip.columns;
        let (from, to) = strip.pixels;
        let frame_width = if col_hi == self.mcus_per_row {
            self.width - col_lo * mw
        } else {
            (col_hi - col_lo) * mw
        };
        let frame_height = if row_hi == self.mcu_rows {
            self.height - row_lo * mh
        } else {
            (row_hi - row_lo) * mh
        };
        let per_row = (self.mcus_per_row / r) as usize;
        let chosen: Vec<usize> = (row_lo as usize..row_hi as usize)
            .flat_map(|row| {
                (col_lo / r) as usize + row * per_row..(col_hi / r) as usize + row * per_row
            })
            .collect();
        #[cfg(test)]
        tests::note_frame_width(frame_width);
        // Both sides are at most the original's, which fit a JPEG frame.
        let jpeg = self
            .index
            .assemble(frame_width as u16, frame_height as u16, &chosen);
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
        if self.components == 3 {
            decoder.set_color_transform(jpeg_decoder::ColorTransform::YCbCr);
        }
        let pixels = decoder.decode().map_err(jpeg_error)?;
        let channels = usize::from(self.components);
        let line = frame_width as usize * channels;
        if pixels.len() != line * frame_height as usize {
            return Err(jpeg_error(jpeg_decoder::Error::Format(
                "the region's intervals decode to the wrong number of samples".to_string(),
            )));
        }
        let span = (to - from) as usize;
        let (left, top) = ((from - col_lo * mw) as usize, (y0 - row_lo * mh) as usize);
        for y in 0..h {
            let row = &pixels[(top + y) * line + left * channels..][..span * channels];
            let at = (y * w + (from - x0) as usize) * 3;
            let target = &mut out[at..at + span * 3];
            if channels == 1 {
                for (p, &v) in target.as_chunks_mut::<3>().0.iter_mut().zip(row) {
                    *p = [v, v, v];
                }
            } else {
                target.copy_from_slice(row);
            }
        }
        Ok(())
    }
}

/// One frame of a region read of an original: the MCU columns it decodes,
/// from one interval boundary to another or the image's right edge, and the
/// pixel columns of the rectangle it gives.
struct Strip {
    columns: (u32, u32),
    pixels: (u32, u32),
}

impl Strip {
    /// The one frame that gives pixel columns `x0..x1` with their margins:
    /// from the interval boundary at or before the MCU left of `x0` to the
    /// one at or after the MCU right of `x1`, or the image's right edge.
    fn one_frame(original: &OriginalRegions, x0: u32, x1: u32) -> Self {
        let (mw, r) = (original.mcu_size.0, original.restart_interval);
        let lo = (x0 / mw).saturating_sub(1) / r * r;
        let hi = (x1.div_ceil(mw) + 1)
            .div_ceil(r)
            .saturating_mul(r)
            .min(original.mcus_per_row);
        Self {
            columns: (lo, hi),
            pixels: (x0, x1),
        }
    }
}

/// The widest frame, in pixels, that `jpeg-decoder` 0.3.2 decodes on the
/// calling thread. For a wider frame its `Decoder::decode` starts one OS
/// thread per colour component (`select_worker` in its `decoder.rs`, which
/// compares the frame's width squared with `128 * 128`). While a kernel keeps
/// every core busy those threads wait their turn to run, and the thread that
/// asked for the decode waits for them: a tile of a 16320 x 12240 original
/// that decodes in 3 ms alone took about 200 ms inside `sfm xform
/// --refine-normals`.
pub(crate) const DECODED_ON_THE_CALLING_THREAD: u32 = 128;

#[cfg(test)]
mod tests;
