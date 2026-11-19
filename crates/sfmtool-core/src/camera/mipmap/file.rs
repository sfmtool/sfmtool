// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One mip level file: how it is encoded, and how it is read and checked.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use super::record::MipLevelRecord;
use super::region::{index_intervals, read_header};
use super::{MipmapError, PyramidLayout, MIPMAP_RESTART_INTERVAL_MCUS};
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
/// interval of [`MIPMAP_RESTART_INTERVAL_MCUS`], and `record` in an APP15
/// segment.
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
    encoder.set_restart_interval(MIPMAP_RESTART_INTERVAL_MCUS);
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

/// One mip level file's bytes and record, read and checked once.
///
/// The whole file is held in memory, so a reader sees one version of it even
/// when another writer replaces the file.
#[derive(Clone, Debug)]
pub struct MipLevelFile {
    bytes: Arc<[u8]>,
    record: MipLevelRecord,
}

impl MipLevelFile {
    /// Read the file at `path` and check it against `expected`.
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
        Self::check(bytes.into(), expected).map_err(|why| MipmapError::Stale {
            path: path.to_path_buf(),
            why,
        })
    }

    /// [`open`](Self::open) for bytes already in memory, such as those a build
    /// has just encoded. A [`MipmapError::Stale`] from it has an empty path.
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, MipmapError> {
        Self::check(bytes, expected).map_err(|why| MipmapError::Stale {
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
        &self.bytes
    }

    fn check(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, String> {
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
        index_intervals(&bytes, header.scan_data, mcus.div_ceil(usize::from(r)))
            .map_err(|e| format!("its scan is damaged or cut short ({e})"))?;
        Ok(Self { bytes, record })
    }

    /// The whole level as RGB, decoded with `jpeg-decoder` returning the
    /// samples as coded and converted to RGB here with the fixed-point BT.601
    /// formula of its scalar path, so a region decode can give the same
    /// pixels. A grey file gives its value in all three channels.
    pub fn decode(&self) -> Result<ImageU8, MipmapError> {
        let stale = |why: String| MipmapError::Stale {
            path: PathBuf::new(),
            why,
        };
        let mut decoder = jpeg_decoder::Decoder::new(&self.bytes[..]);
        decoder.set_color_transform(jpeg_decoder::ColorTransform::None);
        let samples = decoder
            .decode()
            .map_err(|e| stale(format!("it does not decode: {e}")))?;
        let (width, height) = (self.record.width, self.record.height);
        let pixels = width as usize * height as usize;
        let components = usize::from(self.record.components);
        if samples.len() != pixels * components {
            return Err(stale(
                "it decodes to the wrong number of samples".to_string(),
            ));
        }
        let mut data = Vec::with_capacity(pixels * 3);
        if components == 1 {
            for &y in &samples {
                data.extend([y, y, y]);
            }
        } else {
            for &[y, cb, cr] in samples.as_chunks::<3>().0 {
                data.extend(ycbcr_to_rgb(y, cb, cr));
            }
        }
        Ok(ImageU8::new(width, height, 3, data))
    }
}

/// `jpeg-decoder` 0.3.2's scalar YCbCr to RGB conversion (`ycbcr_to_rgb` in
/// its `decoder.rs`, from libjpeg-turbo's `jdcolext.c`): 20-bit fixed point,
/// rounded at half.
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
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
