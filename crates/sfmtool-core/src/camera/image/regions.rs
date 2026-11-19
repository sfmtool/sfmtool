// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! View-region pyramids: an [`ImageU8Pyramid`] that holds, of each level, only
//! the rectangle a render reads ([`ImageU8Pyramid::from_regions`]).

use std::fmt;

use super::{ImageU8, ImageU8Pyramid};

/// One level of a pyramid that may hold only a rectangle of the full level.
pub struct PyramidLevel {
    /// The pixels held: the whole level, or a rectangle of it.
    pub image: ImageU8,
    /// The held rectangle's top-left pixel in the full level; `[0, 0]` for a
    /// whole level.
    pub origin: [u32; 2],
    /// The full level's size.
    pub full_size: [u32; 2],
}

/// Why [`ImageU8Pyramid::from_regions`] refused its levels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyramidRegionsError {
    /// No level is held.
    NoLevels,
    /// A held level has no pixels, or no channels.
    Empty { level: usize },
    /// A held rectangle reaches past its full level.
    OutsideLevel { level: usize },
    /// Two held levels have different channel counts.
    Channels { level: usize },
    /// A full level's size is not the finer held level's size halved.
    FullSize { level: usize },
}

impl fmt::Display for PyramidRegionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLevels => write!(f, "a region pyramid needs at least one held level"),
            Self::Empty { level } => write!(f, "the region of level {level} has no pixels"),
            Self::OutsideLevel { level } => {
                write!(f, "the region of level {level} reaches past the level")
            }
            Self::Channels { level } => write!(
                f,
                "the region of level {level} has a different channel count from the others"
            ),
            Self::FullSize { level } => write!(
                f,
                "the full size of level {level} is not the finer level's size halved"
            ),
        }
    }
}

impl std::error::Error for PyramidRegionsError {}

impl ImageU8Pyramid {
    /// A view-region pyramid: of each level, the rectangle a render reads, or
    /// `None` for a level it does not read.
    ///
    /// The samplers read it at full-level coordinates and give the values a
    /// whole pyramid gives wherever every tap they take, after clamping to
    /// the full level, lies inside the held rectangle; `required_regions` in
    /// [`sampler`](crate::camera::sampler) computes such rectangles. A tap
    /// outside the held rectangle reads the rectangle's nearest pixel, and a
    /// level not held is a 1 x 1 black image with the origin `[0, 0]` and the
    /// full size `[0, 0]`.
    ///
    /// Refuses no held level, a held level with no pixels or channels, a
    /// rectangle past its full level, channel counts that differ, and full
    /// sizes that are not each other halved as [`ImageU8::downsample_2x`]
    /// halves a level.
    pub fn from_regions(levels: Vec<Option<PyramidLevel>>) -> Result<Self, PyramidRegionsError> {
        let mut channels = None;
        let mut finer: Option<(usize, [u32; 2])> = None;
        for (l, level) in levels.iter().enumerate() {
            let Some(level) = level else { continue };
            let image = &level.image;
            if image.width == 0 || image.height == 0 || image.channels == 0 {
                return Err(PyramidRegionsError::Empty { level: l });
            }
            let fits = |origin: u32, size: u32, full: u32| {
                origin.checked_add(size).is_some_and(|end| end <= full)
            };
            if !fits(level.origin[0], image.width, level.full_size[0])
                || !fits(level.origin[1], image.height, level.full_size[1])
            {
                return Err(PyramidRegionsError::OutsideLevel { level: l });
            }
            if *channels.get_or_insert(image.channels) != image.channels {
                return Err(PyramidRegionsError::Channels { level: l });
            }
            if let Some((k, size)) = finer {
                let shift = (l - k) as u32;
                let halved = [
                    size[0].checked_shr(shift).unwrap_or(0),
                    size[1].checked_shr(shift).unwrap_or(0),
                ];
                if halved != level.full_size {
                    return Err(PyramidRegionsError::FullSize { level: l });
                }
            }
            finer = Some((l, level.full_size));
        }
        let Some(channels) = channels else {
            return Err(PyramidRegionsError::NoLevels);
        };
        let n = levels.len();
        let mut images = Vec::with_capacity(n);
        let mut origins = Vec::with_capacity(n);
        let mut full_sizes = Vec::with_capacity(n);
        for level in levels {
            match level {
                Some(level) => {
                    images.push(level.image);
                    origins.push(level.origin);
                    full_sizes.push(level.full_size);
                }
                None => {
                    images.push(ImageU8::from_channels(1, 1, channels));
                    origins.push([0, 0]);
                    full_sizes.push([0, 0]);
                }
            }
        }
        Ok(Self {
            levels: images,
            origins,
            full_sizes,
        })
    }
}
