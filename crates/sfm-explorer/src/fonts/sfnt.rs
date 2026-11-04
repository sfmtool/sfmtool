// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Just enough of the OpenType file format to check a font file from the
//! operating system before egui is given it.
//!
//! egui panics when a face in its font definitions does not open, so a file
//! read from disk is opened here first, with every read bounds-checked. A
//! face this module opens is one egui opens too: the checks are the ones
//! egui's font reader makes (a collection header or a table directory that
//! fits in the file, and a known `sfnt` version), and stricter, since the
//! table records and the `cmap` table must also fit, and the `head` table
//! must fit and give a `unitsPerEm` other than `0`, which egui divides by.
//! The character map read
//! here is what tells whether a face has the glyphs the viewer is after,
//! without building egui's fonts to ask.

/// One face of a font file: its index in the file, its weight and the
/// characters it maps to a glyph.
#[derive(Debug, Clone)]
pub(super) struct Face {
    /// The face's index in its file: `0` for a plain font file, its place in
    /// the collection for a `.ttc`.
    pub(super) index: u32,
    /// The `OS/2` table's `usWeightClass`, `400` for a regular weight, or
    /// `None` when the face has no `OS/2` table.
    pub(super) weight: Option<u16>,
    /// The mapped code points, as sorted, disjoint, inclusive ranges.
    ranges: Vec<(u32, u32)>,
}

impl Face {
    /// Whether the face maps `code` to a glyph.
    pub(super) fn maps(&self, code: u32) -> bool {
        let i = self.ranges.partition_point(|&(_, end)| end < code);
        self.ranges.get(i).is_some_and(|&(start, _)| start <= code)
    }

    /// The mapped code points, as sorted, disjoint, inclusive ranges.
    pub(super) fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }
}

/// Every face of the font file `data` that opens, or `None` when the file is
/// not a font file or a collection of them.
///
/// A collection whose header opens keeps those of its faces that open, so
/// one bad face does not cost the others.
pub(super) fn faces(data: &[u8]) -> Option<Vec<Face>> {
    let offsets = directory_offsets(data)?;
    let faces: Vec<Face> = offsets
        .into_iter()
        .enumerate()
        .filter_map(|(index, offset)| face(data, u32::try_from(index).ok()?, offset))
        .collect();
    (!faces.is_empty()).then_some(faces)
}

/// Where each face's table directory starts: `[0]` for a plain font file,
/// the collection header's offsets for a `.ttc`.
fn directory_offsets(data: &[u8]) -> Option<Vec<usize>> {
    if data.get(0..4)? != b"ttcf" {
        return Some(vec![0]);
    }
    let count = be32(data, 8)?;
    (0..count)
        .map(|i| {
            let at = usize::try_from(i).ok()?.checked_mul(4)?.checked_add(12)?;
            usize::try_from(be32(data, at)?).ok()
        })
        .collect()
}

/// The face whose table directory starts at `offset`, if it opens.
fn face(data: &[u8], index: u32, offset: usize) -> Option<Face> {
    const TRUETYPE: u32 = 0x0001_0000;
    const CFF: u32 = u32::from_be_bytes(*b"OTTO");
    const APPLE_TRUETYPE: u32 = u32::from_be_bytes(*b"true");
    let version = be32(data, offset)?;
    if ![TRUETYPE, CFF, APPLE_TRUETYPE].contains(&version) {
        return None;
    }
    let num_tables = usize::from(be16(data, offset.checked_add(4)?)?);
    let records = offset.checked_add(12)?;
    data.get(records..records.checked_add(num_tables.checked_mul(16)?)?)?;
    let table = |tag: &[u8; 4]| -> Option<&[u8]> {
        (0..num_tables).find_map(|i| {
            let record = records + i * 16;
            if data.get(record..record + 4)? != tag {
                return None;
            }
            let start = usize::try_from(be32(data, record + 8)?).ok()?;
            let length = usize::try_from(be32(data, record + 12)?).ok()?;
            data.get(start..start.checked_add(length)?)
        })
    };
    // egui scales every glyph by `unitsPerEm` from `head`, so a face whose
    // `head` is missing, cut short or gives `0` would draw at an infinite
    // scale.
    let units_per_em = be16(table(b"head")?, 18)?;
    if units_per_em == 0 {
        return None;
    }
    let ranges = charmap(table(b"cmap")?)?;
    let weight = table(b"OS/2").and_then(|os2| be16(os2, 4));
    Some(Face {
        index,
        weight,
        ranges,
    })
}

/// The code points `cmap` maps to a glyph other than `0`, from its best
/// Unicode subtable: a format 12 one for the whole of Unicode when there is
/// one, else a format 4 one for the Basic Multilingual Plane.
fn charmap(cmap: &[u8]) -> Option<Vec<(u32, u32)>> {
    let count = usize::from(be16(cmap, 2)?);
    let mut full = None;
    let mut bmp = None;
    for i in 0..count {
        let record = 4 + i * 8;
        let platform = be16(cmap, record)?;
        let encoding = be16(cmap, record + 2)?;
        let offset = usize::try_from(be32(cmap, record + 4)?).ok()?;
        let unicode = platform == 0 || (platform == 3 && (encoding == 1 || encoding == 10));
        if !unicode {
            continue;
        }
        let Some(subtable) = cmap.get(offset..) else {
            continue;
        };
        match be16(subtable, 0) {
            Some(12) if full.is_none() => full = Some(subtable),
            Some(4) if bmp.is_none() => bmp = Some(subtable),
            _ => {}
        }
    }
    let mut codes = match (full, bmp) {
        (Some(subtable), _) => format12(subtable)?,
        (None, Some(subtable)) => format4(subtable)?,
        (None, None) => return None,
    };
    codes.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(codes.len());
    for (start, end) in codes {
        match merged.last_mut() {
            Some(last) if start <= last.1.saturating_add(1) => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    Some(merged)
}

/// The mapped ranges of a format 12 subtable (segmented coverage).
fn format12(subtable: &[u8]) -> Option<Vec<(u32, u32)>> {
    let groups = usize::try_from(be32(subtable, 12)?).ok()?;
    let mut ranges = Vec::new();
    for i in 0..groups {
        let group = i.checked_mul(12)?.checked_add(16)?;
        let start = be32(subtable, group)?;
        let end = be32(subtable, group + 4)?;
        let first_glyph = be32(subtable, group + 8)?;
        if end < start || end > 0x10_FFFF {
            continue;
        }
        // A group whose first glyph is `0` maps its first code to no glyph.
        let start = if first_glyph == 0 { start + 1 } else { start };
        if start <= end {
            ranges.push((start, end));
        }
    }
    Some(ranges)
}

/// The mapped ranges of a format 4 subtable (segment mapping to delta
/// values).
fn format4(subtable: &[u8]) -> Option<Vec<(u32, u32)>> {
    let segments = usize::from(be16(subtable, 6)? / 2);
    let ends = 14;
    let starts = ends + segments * 2 + 2;
    let deltas = starts + segments * 2;
    let range_offsets = deltas + segments * 2;
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for s in 0..segments {
        let end = be16(subtable, ends + s * 2)?;
        let start = be16(subtable, starts + s * 2)?;
        let delta = be16(subtable, deltas + s * 2)?;
        let range_offset_at = range_offsets + s * 2;
        let range_offset = usize::from(be16(subtable, range_offset_at)?);
        if end < start || start == 0xFFFF {
            continue;
        }
        for code in start..=end {
            let glyph = if range_offset == 0 {
                code.wrapping_add(delta)
            } else {
                let at = range_offset_at + range_offset + usize::from(code - start) * 2;
                match be16(subtable, at) {
                    Some(0) | None => 0,
                    Some(glyph) => glyph.wrapping_add(delta),
                }
            };
            if glyph != 0 {
                let code = u32::from(code);
                match ranges.last_mut() {
                    Some(last) if last.1 + 1 == code => last.1 = code,
                    _ => ranges.push((code, code)),
                }
            }
        }
    }
    Some(ranges)
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    let bytes = data.get(at..at.checked_add(2)?)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
