// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The region test: whether an original JPEG's layout lets a rectangle of it
//! be decoded on its own, to exactly the pixels of the whole decode. It reads
//! the file's bytes and nothing else, so every reader decides the same way.
//!
//! Also the marker walk up to the first scan that the mip level files' own
//! check reads their frame, restart interval and record with.

use std::fmt;
use std::ops::Range;

use super::MAX_REGION_INTERVAL_WIDTH;

/// What a region read of a region-readable original needs: the frame, its
/// sampling, its restart interval and the byte range of every interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionLayout {
    /// The frame's size in pixels.
    pub width: u32,
    pub height: u32,
    /// 1 (grey) or 3 (YCbCr).
    pub components: u8,
    /// The luma component's horizontal and vertical sampling factors, each 1
    /// or 2; 1 x 1 for a grey file whatever its header says, since its scan is
    /// not interleaved.
    pub luma_sampling: (u8, u8),
    /// `R`, the restart interval in MCUs, which divides `mcus_per_row`.
    pub restart_interval: u16,
    /// MCUs per MCU row and MCU rows.
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    /// The byte range in the file of each restart interval's entropy-coded
    /// data, in scan order, without its restart marker: `mcu_rows ·
    /// (mcus_per_row / R)` of them.
    pub intervals: Vec<Range<usize>>,
}

/// Why an original cannot be read by region: the first rule of the region
/// test it fails, with what was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotRegionReadable {
    /// Rule 1: the bytes do not start with the JPEG start-of-image marker.
    NotJpeg,
    /// Rule 2: the frame is not baseline or extended sequential Huffman with
    /// 8-bit samples and a height that is not 0, or there is none.
    Frame(String),
    /// Rule 3: not one or three components, or three that the decoder does
    /// not read as YCbCr.
    Colour(String),
    /// Rule 4: chroma not sampled 1 x 1, or luma not `H x V` with each 1 or 2.
    Sampling(String),
    /// Rule 5: not exactly one scan holding every component in frame order,
    /// spectral selection 0 to 63, no successive approximation.
    Scan(String),
    /// Rule 6: no `DRI` before the scan, an interval of 0, an interval that
    /// does not divide the MCUs per row, or one wider than 1024 px.
    RestartInterval(String),
    /// Rule 7: the scan's restart intervals are not exactly the expected
    /// number, each non-empty, separated by `RST0`..`RST7` in sequence and
    /// followed by end-of-image.
    Intervals(String),
}

impl fmt::Display for NotRegionReadable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJpeg => write!(f, "not a JPEG"),
            Self::Frame(why) => write!(f, "frame: {why}"),
            Self::Colour(why) => write!(f, "colour: {why}"),
            Self::Sampling(why) => write!(f, "sampling: {why}"),
            Self::Scan(why) => write!(f, "scan: {why}"),
            Self::RestartInterval(why) => write!(f, "restart interval: {why}"),
            Self::Intervals(why) => write!(f, "restart intervals: {why}"),
        }
    }
}

impl std::error::Error for NotRegionReadable {}

/// One component of a frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameComponent {
    pub(super) id: u8,
    pub(super) h: u8,
    pub(super) v: u8,
}

/// A frame header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Frame {
    /// The SOF marker's second byte, `0xC0` to `0xCF`.
    pub(super) marker: u8,
    pub(super) precision: u8,
    pub(super) width: u16,
    pub(super) height: u16,
    pub(super) components: Vec<FrameComponent>,
}

/// A scan header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ScanHeader {
    pub(super) components: Vec<u8>,
    pub(super) ss: u8,
    pub(super) se: u8,
    pub(super) ah_al: u8,
}

/// What the marker segments before the first scan say, read as
/// `jpeg-decoder` 0.3.2 reads them.
#[derive(Clone, Debug, Default)]
pub(super) struct JpegHeader {
    pub(super) frame: Option<Frame>,
    /// The last `DRI` before the scan.
    pub(super) restart_interval: Option<u16>,
    /// An APP0 segment starting `JFIF\0`.
    pub(super) jfif: bool,
    /// The transform byte of the last APP14 segment starting `Adobe\0`.
    pub(super) adobe_transform: Option<u8>,
    /// The bodies of the APP15 segments, in file order.
    pub(super) app15: Vec<Range<usize>>,
    pub(super) scan: Option<ScanHeader>,
    /// The offset of the first byte of entropy-coded data after the scan
    /// header; 0 when there is no scan.
    pub(super) scan_data: usize,
    /// Every whole segment, its marker included, that a decode of the scan
    /// reads: all but the APPn and COM segments, in file order, ending with
    /// the scan header.
    pub(super) decode_segments: Vec<Range<usize>>,
    /// The offset of the frame header's body (its precision byte); 0 when
    /// there is none.
    pub(super) frame_at: usize,
}

/// The position just after the next marker at or after `at`, and its second
/// byte, finding it as `jpeg-decoder` does: bytes up to the next `FF` are
/// skipped, fill `FF`s are skipped, and `FF 00` is not a marker.
fn next_marker(bytes: &[u8], at: &mut usize) -> Option<u8> {
    loop {
        while *bytes.get(*at)? != 0xFF {
            *at += 1;
        }
        *at += 1;
        let mut byte = *bytes.get(*at)?;
        while byte == 0xFF {
            *at += 1;
            byte = *bytes.get(*at)?;
        }
        *at += 1;
        if byte != 0x00 {
            return Some(byte);
        }
    }
}

/// Read the marker segments of `bytes` from just after start-of-image up to
/// and including the first scan header. An error says why the walk stopped
/// early: a segment that runs past the end, an end-of-image before any scan,
/// or the end of the bytes.
pub(super) fn read_header(bytes: &[u8]) -> Result<JpegHeader, String> {
    let mut header = JpegHeader::default();
    let mut at = 2;
    loop {
        let Some(marker) = next_marker(bytes, &mut at) else {
            return Err("the file ends before its first scan".to_string());
        };
        match marker {
            0x01 | 0xD0..=0xD8 => continue,
            0xD9 => return Err("end-of-image before any scan".to_string()),
            _ => {}
        }
        let Some(&[high, low]) = bytes.get(at..at + 2) else {
            return Err("the file ends inside a segment's length".to_string());
        };
        let length = usize::from(u16::from_be_bytes([high, low]));
        if length < 2 {
            return Err(format!("a segment FF{marker:02X} of length {length}"));
        }
        let body = at + 2..at + length;
        let Some(data) = bytes.get(body.clone()) else {
            return Err(format!(
                "segment FF{marker:02X} runs past the end of the file"
            ));
        };
        if !matches!(marker, 0xE0..=0xEF | 0xFE) {
            header.decode_segments.push(at - 2..at + length);
        }
        at += length;
        match marker {
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                if header.frame.is_some() {
                    return Err("a second frame header".to_string());
                }
                if data.len() < 6 {
                    return Err("a frame header too short to read".to_string());
                }
                let count = usize::from(data[5]);
                let Some(list) = data.get(6..6 + 3 * count) else {
                    return Err("a frame header too short for its components".to_string());
                };
                header.frame_at = body.start;
                header.frame = Some(Frame {
                    marker,
                    precision: data[0],
                    height: u16::from_be_bytes([data[1], data[2]]),
                    width: u16::from_be_bytes([data[3], data[4]]),
                    components: list
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|c| FrameComponent {
                            id: c[0],
                            h: c[1] >> 4,
                            v: c[1] & 0x0F,
                        })
                        .collect(),
                });
            }
            0xDD => {
                let Some(&[high, low]) = data.get(..2) else {
                    return Err("a DRI segment too short to read".to_string());
                };
                header.restart_interval = Some(u16::from_be_bytes([high, low]));
            }
            0xE0 if data.len() >= 5 && data[..5] == *b"JFIF\0" => header.jfif = true,
            0xEE if data.len() >= 12 && data[..6] == *b"Adobe\0" => {
                header.adobe_transform = Some(data[11]);
            }
            0xEF => header.app15.push(body),
            0xDA => {
                let count = usize::from(*data.first().unwrap_or(&0));
                let Some(list) = data.get(1..1 + 2 * count) else {
                    return Err("a scan header too short for its components".to_string());
                };
                let Some(&[ss, se, ah_al]) = data.get(1 + 2 * count..4 + 2 * count) else {
                    return Err("a scan header too short to read".to_string());
                };
                header.scan = Some(ScanHeader {
                    components: list.as_chunks::<2>().0.iter().map(|c| c[0]).collect(),
                    ss,
                    se,
                    ah_al,
                });
                header.scan_data = at;
                return Ok(header);
            }
            _ => {}
        }
    }
}

/// The region test of the spec `specs/core/camera/mipmaps.md`: the layout a
/// region decode needs, or the first rule the bytes fail.
pub fn region_layout(bytes: &[u8]) -> Result<RegionLayout, NotRegionReadable> {
    use NotRegionReadable as N;
    // Rule 1.
    if bytes.get(..2) != Some(&[0xFF, 0xD8]) {
        return Err(N::NotJpeg);
    }
    let header = read_header(bytes);
    // Rule 2.
    let frame = match &header {
        Ok(JpegHeader { frame: Some(f), .. }) => f.clone(),
        Ok(_) => return Err(N::Frame("no frame header before the scan".to_string())),
        Err(why) => match header_frame_before_failure(bytes) {
            Some(f) => f,
            None => return Err(N::Frame(why.clone())),
        },
    };
    if !matches!(frame.marker, 0xC0 | 0xC1) {
        return Err(N::Frame(format!(
            "SOF{} is not baseline or extended sequential Huffman",
            frame.marker - 0xC0
        )));
    }
    if frame.precision != 8 {
        return Err(N::Frame(format!("{}-bit samples", frame.precision)));
    }
    if frame.height == 0 || frame.width == 0 {
        return Err(N::Frame(format!(
            "a frame of {} x {}",
            frame.width, frame.height
        )));
    }
    let header = header.map_err(N::Scan)?;
    // Rule 3.
    let c = &frame.components;
    match c.len() {
        1 => {}
        3 => {
            let ids = (c[0].id, c[1].id, c[2].id);
            let transform = match ids {
                (1, 2, 3) => "YCbCr",
                (1, 34, 35) => "big-gamut YCC",
                (82, 71, 66) => "RGB",
                (114, 103, 98) => "big-gamut RGB",
                _ if header.jfif => "YCbCr",
                _ => match header.adobe_transform {
                    Some(0) => "RGB",
                    Some(1) | None => "YCbCr",
                    Some(2) => "YCCK",
                    Some(_) => "an invalid Adobe transform",
                },
            };
            if transform != "YCbCr" {
                return Err(N::Colour(format!("the decoder reads it as {transform}")));
            }
        }
        n => return Err(N::Colour(format!("{n} components"))),
    }
    // Rule 4.
    let luma = c[0];
    if !(1..=2).contains(&luma.h) || !(1..=2).contains(&luma.v) {
        return Err(N::Sampling(format!("luma sampled {} x {}", luma.h, luma.v)));
    }
    if let Some(chroma) = c[1..].iter().find(|x| (x.h, x.v) != (1, 1)) {
        return Err(N::Sampling(format!(
            "chroma sampled {} x {}",
            chroma.h, chroma.v
        )));
    }
    // Rule 5 (the scan header; a second scan is found while indexing).
    let Some(scan) = &header.scan else {
        return Err(N::Scan("no scan".to_string()));
    };
    let frame_ids: Vec<u8> = c.iter().map(|x| x.id).collect();
    if scan.components != frame_ids {
        return Err(N::Scan(format!(
            "the scan holds components {:?} of the frame's {:?}",
            scan.components, frame_ids
        )));
    }
    if (scan.ss, scan.se, scan.ah_al) != (0, 63, 0) {
        return Err(N::Scan(format!(
            "spectral selection {} to {} with approximation {:#04x}",
            scan.ss, scan.se, scan.ah_al
        )));
    }
    // Rule 6.
    let (mcu_w, mcu_h) = if c.len() == 3 {
        (8 * u32::from(luma.h), 8 * u32::from(luma.v))
    } else {
        (8, 8)
    };
    let (width, height) = (u32::from(frame.width), u32::from(frame.height));
    let mcus_per_row = width.div_ceil(mcu_w);
    let mcu_rows = height.div_ceil(mcu_h);
    let r = match header.restart_interval {
        None => return Err(N::RestartInterval("no DRI segment".to_string())),
        Some(0) => return Err(N::RestartInterval("an interval of 0".to_string())),
        Some(r) => r,
    };
    if mcus_per_row % u32::from(r) != 0 {
        return Err(N::RestartInterval(format!(
            "{r} MCUs does not divide the {mcus_per_row} MCUs of a row"
        )));
    }
    if u32::from(r) * mcu_w > MAX_REGION_INTERVAL_WIDTH {
        return Err(N::RestartInterval(format!(
            "an interval of {} px is wider than {MAX_REGION_INTERVAL_WIDTH}",
            u32::from(r) * mcu_w
        )));
    }
    // Rule 7.
    let expected = mcu_rows as usize * (mcus_per_row / u32::from(r)) as usize;
    let intervals = index_intervals(bytes, header.scan_data, expected)?;
    Ok(RegionLayout {
        width,
        height,
        components: c.len() as u8,
        luma_sampling: if c.len() == 3 {
            (luma.h, luma.v)
        } else {
            (1, 1)
        },
        restart_interval: r,
        mcus_per_row,
        mcu_rows,
        intervals,
    })
}

/// The frame header of a file whose walk to the scan failed after it, so a
/// progressive or 12-bit frame is reported by rule 2 even when the rest of
/// the header does not read.
fn header_frame_before_failure(bytes: &[u8]) -> Option<Frame> {
    let mut at = 2;
    loop {
        let marker = next_marker(bytes, &mut at)?;
        if matches!(marker, 0x01 | 0xD0..=0xD8) {
            continue;
        }
        if marker == 0xD9 || marker == 0xDA {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([*bytes.get(at)?, *bytes.get(at + 1)?]));
        let data = bytes.get(at + 2..at + length.max(2))?;
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            if data.len() < 6 {
                return None;
            }
            return Some(Frame {
                marker,
                precision: data[0],
                height: u16::from_be_bytes([data[1], data[2]]),
                width: u16::from_be_bytes([data[3], data[4]]),
                components: Vec::new(),
            });
        }
        at += length.max(2);
    }
}

/// Index the entropy-coded data from `start`: exactly `expected` non-empty
/// intervals separated by `RST0`..`RST7` in sequence modulo 8, then
/// end-of-image. Bytes after end-of-image are ignored.
pub(super) fn index_intervals(
    bytes: &[u8],
    start: usize,
    expected: usize,
) -> Result<Vec<Range<usize>>, NotRegionReadable> {
    use NotRegionReadable as N;
    // Each interval takes at least three bytes (one of data and a marker), so
    // a header declaring more than the file can hold does not reserve them.
    let mut intervals = Vec::with_capacity(expected.min(bytes.len() / 3));
    let mut begin = start;
    let mut i = start;
    loop {
        let Some(offset) = bytes
            .get(i..)
            .and_then(|rest| rest.iter().position(|&b| b == 0xFF))
        else {
            return Err(N::Intervals(format!(
                "the file ends after {} of {expected} intervals",
                intervals.len()
            )));
        };
        let ff = i + offset;
        let Some(&next) = bytes.get(ff + 1) else {
            return Err(N::Intervals("the file ends inside a marker".to_string()));
        };
        match next {
            // A stuffed byte, or a fill byte before a marker.
            0x00 => i = ff + 2,
            0xFF => i = ff + 1,
            0xD0..=0xD7 | 0xD9 => {
                if ff == begin {
                    return Err(N::Intervals(format!(
                        "interval {} is empty",
                        intervals.len()
                    )));
                }
                intervals.push(begin..ff);
                if next == 0xD9 {
                    if intervals.len() != expected {
                        return Err(N::Intervals(format!(
                            "{} intervals where {expected} were expected",
                            intervals.len()
                        )));
                    }
                    return Ok(intervals);
                }
                let number = usize::from(next - 0xD0);
                if intervals.len() >= expected {
                    return Err(N::Intervals(format!(
                        "more than the {expected} intervals expected"
                    )));
                }
                if number != (intervals.len() - 1) % 8 {
                    return Err(N::Intervals(format!(
                        "RST{number} after interval {} where RST{} belongs",
                        intervals.len() - 1,
                        (intervals.len() - 1) % 8
                    )));
                }
                begin = ff + 2;
                i = begin;
            }
            0xC4 | 0xDA | 0xDB | 0xDD => {
                return Err(N::Scan(
                    "a second scan, or tables for one, after the first".to_string(),
                ))
            }
            other => {
                return Err(N::Intervals(format!(
                    "marker FF{other:02X} inside the scan"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests;
