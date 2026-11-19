// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One mip level file: how it is encoded, and how it is read and checked.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use super::record::MipLevelRecord;
use super::region::{index_intervals, read_header};
use super::tile::{
    check_rectangle, decode_mip_level_region, ycbcr_to_rgb, RestartIndex, TileGeometry,
};
use super::{MipmapError, PyramidLayout};
use crate::camera::image::ImageU8;

/// What a mip level file must say to be current for one photograph and
/// level, so far as the file alone can tell: test 1 and the file's part of
/// tests 2 and 3 of when a mip level file is current. The original's
/// length, time and hash are checked by [`MipmapSource`](super::MipmapSource).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedMipLevel {
    /// The photograph's path relative to its workspace root.
    pub source: String,
    /// The photograph's size from its header.
    pub source_width: u32,
    pub source_height: u32,
    /// The level the file holds, the number after `mip` in its name.
    pub level: usize,
    /// The level's size.
    pub width: u32,
    pub height: u32,
    /// The quality in the file's name.
    pub jpeg_quality: u8,
}

impl ExpectedMipLevel {
    /// What the mip level file of `level` of the photograph at `source` with
    /// `layout` must say, at the quality the layout's options give that
    /// level. `None` for an in-core layout, which has no mip level files.
    pub fn new(layout: &PyramidLayout, source: &str, level: usize) -> Option<Self> {
        let options = layout.options().filter(|_| layout.is_out_of_core())?;
        let (width, height) = layout.level_size(level);
        let (source_width, source_height) = layout.level_size(0);
        Some(Self {
            source: source.to_string(),
            source_width,
            source_height,
            level,
            width,
            height,
            jpeg_quality: options.quality_for_level(level),
        })
    }
}

/// Encode one level as a mip level file: a baseline JPEG at `quality`,
/// 4:4:4 (or one component when `grey`), standard Huffman tables, a restart
/// interval of the record's `restart_interval_mcus` (a build's is
/// [`MIPMAP_RESTART_INTERVAL_MCUS`](super::MIPMAP_RESTART_INTERVAL_MCUS)),
/// and `record` in an APP15 segment.
pub(super) fn encode_mip_level(
    image: &ImageU8,
    grey: bool,
    quality: u8,
    record: &MipLevelRecord,
) -> Result<Vec<u8>, MipmapError> {
    let (Ok(width), Ok(height)) = (u16::try_from(image.width()), u16::try_from(image.height()))
    else {
        return Err(MipmapError::TooLarge {
            path: PathBuf::new(),
            width: image.width(),
            height: image.height(),
        });
    };
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, quality);
    encoder.set_sampling_factor(SamplingFactor::F_1_1);
    encoder.set_restart_interval(record.restart_interval_mcus);
    // Optimized tables would put each component in a scan of its own, which
    // a tile cannot be cut from.
    encoder.set_optimized_huffman_tables(false);
    encoder
        .add_app_segment(15, record.to_segment())
        .map_err(|e| MipmapError::Encode(e.to_string()))?;
    let result = if grey {
        let luma: Vec<u8> = image
            .data()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();
        encoder.encode(&luma, width, height, ColorType::Luma)
    } else {
        encoder.encode(image.data(), width, height, ColorType::Rgb)
    };
    result.map_err(|e| MipmapError::Encode(e.to_string()))?;
    Ok(bytes)
}

/// One mip level file's bytes, record and restart index, read and checked
/// once.
///
/// The whole file is held in memory, so a reader sees one version of it even
/// when another writer replaces the file.
#[derive(Clone, Debug)]
pub struct MipLevelFile {
    /// Where it was read from; empty for bytes that were not read from a
    /// file.
    path: PathBuf,
    index: RestartIndex,
    record: MipLevelRecord,
}

impl MipLevelFile {
    /// Read the file at `path` and check it against `expected`, indexing its
    /// restart intervals.
    ///
    /// # Errors
    ///
    /// [`MipmapError::Io`] when it cannot be read (`NotFound` when it does
    /// not exist), and [`MipmapError::Stale`] naming the first test it
    /// fails.
    pub fn open(path: &Path, expected: &ExpectedMipLevel) -> Result<Self, MipmapError> {
        let bytes = std::fs::read(path).map_err(|error| MipmapError::Io {
            path: path.to_path_buf(),
            error,
        })?;
        Self::check(path, bytes.into(), expected).map_err(|why| MipmapError::Stale {
            path: path.to_path_buf(),
            why,
        })
    }

    /// [`open`](Self::open) for bytes already in memory, such as those a build
    /// has just encoded. A [`MipmapError`] from it or from its decodes has an
    /// empty path.
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, MipmapError> {
        Self::check(Path::new(""), bytes, expected).map_err(|why| MipmapError::Stale {
            path: PathBuf::new(),
            why,
        })
    }

    /// The file's record.
    pub fn record(&self) -> &MipLevelRecord {
        &self.record
    }

    /// The file's bytes.
    pub fn bytes(&self) -> &Arc<[u8]> {
        self.index.bytes()
    }

    fn check(path: &Path, bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, String> {
        if bytes.get(..2) != Some(&[0xFF, 0xD8]) {
            return Err("it is not a JPEG".to_string());
        }
        let header = read_header(&bytes)?;
        let Some(segment) = header.app15.first() else {
            return Err("it has no record".to_string());
        };
        let record = MipLevelRecord::from_segment(&bytes[segment.clone()])?;
        // Test 1: the record agrees with the name and the frame.
        if record.level as usize != expected.level {
            return Err(format!(
                "its record says level {} and its name level {}",
                record.level, expected.level
            ));
        }
        if record.jpeg_quality != expected.jpeg_quality {
            return Err(format!(
                "its record says quality {} and its name quality {}",
                record.jpeg_quality, expected.jpeg_quality
            ));
        }
        let Some(frame) = &header.frame else {
            return Err("it has no frame header".to_string());
        };
        if frame.marker != 0xC0
            || (u32::from(frame.width), u32::from(frame.height)) != (record.width, record.height)
            || frame.components.len() != usize::from(record.components)
            || !matches!(record.components, 1 | 3)
            || frame.components.iter().any(|c| (c.h, c.v) != (1, 1))
        {
            return Err("its frame is not the one its record describes".to_string());
        }
        let r = match header.restart_interval {
            Some(r) if r > 0 && r == record.restart_interval_mcus => r,
            _ => return Err("it has not the restart interval its record names".to_string()),
        };
        // Test 2, the file's part: the original's path up to letter case,
        // and the level's size.
        if record.source.to_lowercase() != expected.source.to_lowercase() {
            return Err(format!(
                "it was made from {}, not {}",
                record.source, expected.source
            ));
        }
        if (record.width, record.height) != (expected.width, expected.height) {
            return Err(format!(
                "it is {} x {}, and level {} is {} x {}",
                record.width, record.height, expected.level, expected.width, expected.height
            ));
        }
        // Test 3.
        if (record.source_width, record.source_height)
            != (expected.source_width, expected.source_height)
        {
            return Err(format!(
                "it was made from a {} x {} original, and this one is {} x {}",
                record.source_width,
                record.source_height,
                expected.source_width,
                expected.source_height
            ));
        }
        // Whole: every restart interval present, in order, then end-of-image.
        let mcus = record.width.div_ceil(8) as usize * record.height.div_ceil(8) as usize;
        let intervals = index_intervals(&bytes, header.scan_data, mcus.div_ceil(usize::from(r)))
            .map_err(|e| format!("its scan is damaged or cut short ({e})"))?;
        let index = RestartIndex::new(bytes, &header, intervals)?;
        Ok(Self {
            path: path.to_path_buf(),
            index,
            record,
        })
    }

    fn stale(&self, why: String) -> MipmapError {
        MipmapError::Stale {
            path: self.path.clone(),
            why,
        }
    }

    /// The whole level as RGB, decoded with `jpeg-decoder` returning the
    /// samples as coded and converted to RGB here with the fixed-point BT.601
    /// formula of its scalar path, so [`decode_region`](Self::decode_region)
    /// gives the same pixels. A grey file gives its value in all three
    /// channels.
    ///
    /// # Errors
    ///
    /// [`MipmapError::Stale`] when the scan does not decode.
    pub fn decode(&self) -> Result<ImageU8, MipmapError> {
        let mut decoder = jpeg_decoder::Decoder::new(&self.bytes()[..]);
        decoder.set_color_transform(jpeg_decoder::ColorTransform::None);
        let samples = decoder
            .decode()
            .map_err(|e| self.stale(format!("it does not decode: {e}")))?;
        let (width, height) = (self.record.width as usize, self.record.height as usize);
        let components = usize::from(self.record.components);
        if samples.len() != width * height * components {
            return Err(self.stale("it decodes to the wrong number of samples".to_string()));
        }
        let mut data = Vec::with_capacity(width * height * 3);
        if components == 1 {
            data.extend(samples.iter().flat_map(|&y| [y, y, y]));
        } else {
            // Each line holds its Y samples, then its Cb, then its Cr.
            for line in samples.chunks_exact(3 * width) {
                let (y, chroma) = line.split_at(width);
                let (cb, cr) = chroma.split_at(width);
                for x in 0..width {
                    data.extend(ycbcr_to_rgb(y[x], cb[x], cr[x]));
                }
            }
        }
        Ok(ImageU8::new(self.record.width, self.record.height, 3, data))
    }

    /// Pixels `x0..x1` by `y0..y1` of the level as RGB, decoded from only
    /// the restart intervals that hold them, equal to the same pixels of
    /// [`decode`](Self::decode).
    ///
    /// # Errors
    ///
    /// [`MipmapError::OutOfRange`] for a rectangle that is empty or not
    /// inside the level, and [`MipmapError::Stale`] when the intervals it
    /// needs do not decode.
    pub fn decode_region(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<ImageU8, MipmapError> {
        let rectangle = (x0, y0, x1, y1);
        check_rectangle(self.record.width, self.record.height, rectangle).map_err(|why| {
            MipmapError::OutOfRange {
                path: self.path.clone(),
                why,
            }
        })?;
        let geometry = TileGeometry {
            width: self.record.width,
            height: self.record.height,
            components: usize::from(self.record.components),
            restart_interval: usize::from(self.record.restart_interval_mcus),
        };
        decode_mip_level_region(&self.index, &geometry, rectangle).map_err(|why| self.stale(why))
    }
}
