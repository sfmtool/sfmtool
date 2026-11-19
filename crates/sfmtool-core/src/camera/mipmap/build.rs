// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Making a photograph's mipmap, writing its files, and checking whether the
//! files on disk are current.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::file::{encode_mip_level, ExpectedMipLevel, MipLevelFile};
use super::location::{mipmap_location, MipmapLocation, CACHEDIR_TAG};
use super::record::{MipLevelRecord, MipmapSource, ENCODER, MIP_LEVEL_RECORD_VERSION};
use super::region::region_layout;
use super::{
    MipmapError, MipmapOptions, PyramidLayout, MIPMAP_DIR_NAME, MIPMAP_RESTART_INTERVAL_MCUS,
};
use crate::camera::image::{image_dimensions, temporary_path, ImageU8};
use crate::progress::Progress;

/// A photograph's layout under its workspace's `mipmap_options`, and where
/// its mipmap is when it is out of core (`None` in core).
///
/// The layout's options are `None` for a photograph in no workspace or in a
/// workspace whose file has no `mipmap_options`, so such a photograph is in
/// core whatever its size.
///
/// # Errors
///
/// [`MipmapError::Workspace`] for an invalid `mipmap_options`;
/// [`MipmapError::Source`] when the photograph's header cannot be read; and
/// [`MipmapError::TooLarge`] for a photograph with a side over
/// [`MAX_JPEG_SIDE`](super::MAX_JPEG_SIDE), in every workspace and with no
/// workspace, decided from the header alone.
pub fn photograph_layout(
    photograph: &Path,
    tile_size: u32,
) -> Result<(PyramidLayout, Option<MipmapLocation>), MipmapError> {
    let (width, height) = image_dimensions(photograph).map_err(|error| MipmapError::Source {
        path: photograph.to_path_buf(),
        error,
    })?;
    let location = mipmap_location(photograph);
    let options = match &location {
        Some(location) => crate::workspace::read_mipmap_options(location.root())
            .map_err(MipmapError::Workspace)?,
        None => None,
    };
    let layout =
        PyramidLayout::for_size(width, height, options.as_ref(), tile_size).ok_or_else(|| {
            MipmapError::TooLarge {
                path: photograph.to_path_buf(),
                width,
                height,
            }
        })?;
    let location = location.filter(|_| layout.is_out_of_core());
    Ok((layout, location))
}

/// Check the mipmap on disk of the photograph `source` describes, at
/// `location` with `layout` (an out-of-core layout): every stored level from
/// the coarsest, then `-mip0` when the files say the original is not
/// region-readable. Returns the region answer the files record.
///
/// # Errors
///
/// [`MipmapError::Stale`] naming the first file that is missing or fails a
/// test of when a mip level file is current, and why;
/// [`MipmapError::NotOutOfCore`] for an in-core layout.
pub fn check_mipmap(
    source: &MipmapSource,
    location: &MipmapLocation,
    layout: &PyramidLayout,
) -> Result<bool, MipmapError> {
    let not_out_of_core = || MipmapError::NotOutOfCore {
        path: location.root().join(location.relative()),
    };
    let options = layout
        .options()
        .filter(|_| layout.is_out_of_core())
        .ok_or_else(not_out_of_core)?;
    let open = |level: usize| -> Result<MipLevelFile, MipmapError> {
        let path = location.level_path(level, options.quality_for_level(level));
        let expected =
            ExpectedMipLevel::new(layout, source.relative(), level).ok_or_else(not_out_of_core)?;
        let file = MipLevelFile::open(&path, &expected).map_err(|e| match e {
            MipmapError::Io { path, error } => MipmapError::Stale {
                why: format!("it cannot be read: {error}"),
                path,
            },
            e => e,
        })?;
        source
            .check(file.record())
            .map_err(|why| MipmapError::Stale { path, why })?;
        Ok(file)
    };
    let k = layout.stored_levels();
    let mut answer = None;
    for level in (1..=k).rev() {
        let file = open(level)?;
        if file.record().stored_levels as usize != k {
            return Err(MipmapError::Stale {
                path: location.level_path(level, options.jpeg_quality),
                why: format!(
                    "its record says the mipmap stores {} levels, not {k}",
                    file.record().stored_levels
                ),
            });
        }
        let region_readable = file.record().region_readable_original;
        // Test 5: one region answer across the mipmap.
        if answer.is_some_and(|a| a != region_readable) {
            return Err(MipmapError::Stale {
                path: location.level_path(level, options.jpeg_quality),
                why: "its record's region answer differs from the coarser levels'".to_string(),
            });
        }
        answer = Some(region_readable);
    }
    let region_readable = match answer {
        Some(answer) => answer,
        // No reduced level to record the answer: ask the original.
        None => {
            let path = location.root().join(location.relative());
            let bytes = std::fs::read(&path).map_err(|error| MipmapError::Io { path, error })?;
            region_layout(&bytes).is_ok()
        }
    };
    // Test 6.
    if !region_readable {
        let file = open(0)?;
        if file.record().region_readable_original {
            return Err(MipmapError::Stale {
                path: location.level_path(0, options.full_resolution_jpeg_quality),
                why: "its record's region answer differs from the reduced levels'".to_string(),
            });
        }
    }
    Ok(region_readable)
}

/// A photograph's mipmap as a build made it.
#[derive(Debug)]
pub struct BuiltMipmap {
    /// The photograph's layout under the options the build was given.
    pub layout: PyramidLayout,
    /// The region test's answer for the original.
    pub region_readable: bool,
    /// Every stored level's file, in level order: `-mip0` first when the
    /// original is not region-readable, then levels 1 to `K`. Held in
    /// memory, so a caller decodes these rather than reading the files back.
    pub files: Vec<MipLevelFile>,
    /// `None` when every file was written; otherwise why the files could not
    /// be written, and these bytes are the in-memory equivalent, which a
    /// caller logs as a warning.
    pub not_written: Option<String>,
}

impl BuiltMipmap {
    /// The file of `level`, if the mipmap stores it.
    pub fn level(&self, level: usize) -> Option<&MipLevelFile> {
        self.files
            .iter()
            .find(|f| f.record().level as usize == level)
    }
}

/// Make a photograph's mipmap from one decode, writing what can be written:
/// levels 1 to `K`, coarsest first, and `-mip0` last when the original is not
/// region-readable.
///
/// The photograph's header is read first, so a photograph in core under
/// `options` or too large is refused without being read whole or decoded.
/// `decoded` is `ImageU8::read_rgb(photograph)` when the caller already holds
/// it, and is decoded here from the bytes otherwise. Each reduced level is
/// box-filtered from the exact level above it in memory, never from a decoded
/// file, and encoded on its own. Each file is written to a temporary file in
/// its directory and renamed into place; when that rename fails, a target
/// that is now current (another writer made it) counts as written, and
/// otherwise the build keeps the encoded bytes as the in-memory equivalent and
/// writes nothing further, saying why in [`BuiltMipmap::not_written`]. A
/// mipmap directory made by the build gets a `CACHEDIR.TAG`.
///
/// Reports the phases `decode`, `reduce` and `encode` through `progress`,
/// counting levels, and polls it for cancellation between levels.
///
/// # Errors
///
/// [`MipmapError::NotOutOfCore`] when the photograph is in no workspace or
/// is in core under `options`; [`MipmapError::Io`] when the original cannot
/// be read; [`MipmapError::Source`] when its header cannot be read, it does
/// not decode, or its decode (or `decoded`) is not its header's size;
/// [`MipmapError::TooLarge`] for a side over
/// [`MAX_JPEG_SIDE`](super::MAX_JPEG_SIDE); and [`MipmapError::Cancelled`],
/// after which no further file is written and none is left partial.
pub fn build_mipmaps(
    photograph: &Path,
    decoded: Option<&ImageU8>,
    options: &MipmapOptions,
    tile_size: u32,
    progress: &Progress<'_>,
) -> Result<BuiltMipmap, MipmapError> {
    let not_out_of_core = || MipmapError::NotOutOfCore {
        path: photograph.to_path_buf(),
    };
    let location = mipmap_location(photograph).ok_or_else(not_out_of_core)?;
    let source_error = |error| MipmapError::Source {
        path: photograph.to_path_buf(),
        error,
    };
    // The header decides whether there is a mipmap to make, before the
    // photograph is read whole or decoded.
    let (width, height) = image_dimensions(photograph).map_err(source_error)?;
    let layout = PyramidLayout::for_size(width, height, Some(options), tile_size).ok_or(
        MipmapError::TooLarge {
            path: photograph.to_path_buf(),
            width,
            height,
        },
    )?;
    if !layout.is_out_of_core() {
        return Err(not_out_of_core());
    }
    let io = |error| MipmapError::Io {
        path: photograph.to_path_buf(),
        error,
    };
    let (meta, bytes) = {
        let mut file = std::fs::File::open(photograph).map_err(io)?;
        let meta = file.metadata().map_err(io)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(io)?;
        (meta, bytes)
    };

    let mut phase = progress.phase("decode");
    let owned;
    let level0 = match decoded {
        Some(image) => image,
        None => {
            owned = ImageU8::decode_rgb(&bytes).map_err(source_error)?;
            &owned
        }
    };
    drop(phase);
    if (level0.width(), level0.height()) != (width, height) {
        return Err(source_error(::image::ImageError::Parameter(
            ::image::error::ParameterError::from_kind(
                ::image::error::ParameterErrorKind::DimensionMismatch,
            ),
        )));
    }
    progress.check_cancel()?;
    let source = MipmapSource::from_bytes(
        photograph,
        location.relative(),
        &meta,
        &bytes,
        (width, height),
    );
    let region_readable = region_layout(&bytes).is_ok();
    drop(bytes);

    let k = layout.stored_levels();
    phase = progress.phase("reduce");
    let mut reduced: Vec<ImageU8> = Vec::with_capacity(k);
    for level in 1..=k {
        let next = reduced.last().unwrap_or(level0).downsample_2x();
        reduced.push(next);
        phase.count(level as u64, Some(k as u64), "levels");
        progress.check_cancel()?;
    }
    drop(phase);

    let grey = level0
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .all(|&[r, g, b]| r == g && g == b);
    let (relative, source_size, source_modified_ns, source_xxh128) =
        source.record_fields().map_err(|why| MipmapError::Stale {
            path: photograph.to_path_buf(),
            why,
        })?;
    let record_for = |level: usize, image: &ImageU8| MipLevelRecord {
        version: MIP_LEVEL_RECORD_VERSION,
        source: relative.clone(),
        source_size,
        source_modified_ns,
        source_xxh128: source_xxh128.clone(),
        source_width: width,
        source_height: height,
        level: level as u32,
        out_of_core_long_side: options.out_of_core_long_side,
        region_readable_original: region_readable,
        width: image.width(),
        height: image.height(),
        stored_levels: k as u32,
        jpeg_quality: options.quality_for_level(level),
        components: if grey { 1 } else { 3 },
        restart_interval_mcus: MIPMAP_RESTART_INTERVAL_MCUS,
        encoder: ENCODER.to_string(),
        sfmtool_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    // Coarsest first, then `-mip0`, the largest, last.
    let mut order: Vec<usize> = (1..=k).rev().collect();
    if !region_readable {
        order.push(0);
    }
    let total = order.len() as u64;
    phase = progress.phase("encode");
    let mut not_written = None;
    let mut files = Vec::with_capacity(order.len());
    for (done, level) in order.into_iter().enumerate() {
        progress.check_cancel()?;
        let image = if level == 0 {
            level0
        } else {
            &reduced[level - 1]
        };
        let record = record_for(level, image);
        let encoded: Arc<[u8]> =
            encode_mip_level(image, grey, record.jpeg_quality, &record)?.into();
        let expected =
            ExpectedMipLevel::new(&layout, &relative, level).ok_or_else(not_out_of_core)?;
        let file = MipLevelFile::from_bytes(encoded, &expected)?;
        if not_written.is_none() {
            let target = location.level_path(level, record.jpeg_quality);
            if let Err(why) = write_mip_level_file(&location, &target, &file, &expected, &source) {
                not_written = Some(why);
            }
        }
        files.push(file);
        phase.count(done as u64 + 1, Some(total), "levels");
    }
    drop(phase);
    files.sort_by_key(|f| f.record().level);
    Ok(BuiltMipmap {
        layout,
        region_readable,
        files,
        not_written,
    })
}

/// Write `file` to `target` through a temporary file in its directory,
/// renamed into place, making the directory, and the mipmap directory's
/// `CACHEDIR.TAG`, when missing. A failed rename over a target that is now
/// current for `source` counts as written. `Err` says why the file could not
/// be written.
fn write_mip_level_file(
    location: &MipmapLocation,
    target: &Path,
    file: &MipLevelFile,
    expected: &ExpectedMipLevel,
    source: &MipmapSource,
) -> Result<(), String> {
    let mipmaps = location.root().join(MIPMAP_DIR_NAME);
    std::fs::create_dir_all(location.dir())
        .map_err(|e| format!("{} could not be made: {e}", location.dir().display()))?;
    write_cachedir_tag(&mipmaps)?;
    let temporary = temporary_path(target);
    let written = std::fs::write(&temporary, file.bytes().as_ref())
        .and_then(|()| std::fs::rename(&temporary, target));
    let _ = std::fs::remove_file(&temporary);
    match written {
        Ok(()) => Ok(()),
        Err(e) => {
            let current = MipLevelFile::open(target, expected)
                .is_ok_and(|other| source.check(other.record()).is_ok());
            if current {
                Ok(())
            } else {
                Err(format!("{} could not be written: {e}", target.display()))
            }
        }
    }
}

/// Write `CACHEDIR.TAG` in the mipmap directory unless it is there.
fn write_cachedir_tag(mipmaps: &Path) -> Result<(), String> {
    let tag: PathBuf = mipmaps.join("CACHEDIR.TAG");
    if tag.is_file() {
        return Ok(());
    }
    let temporary = temporary_path(&tag);
    let written =
        std::fs::write(&temporary, CACHEDIR_TAG).and_then(|()| std::fs::rename(&temporary, &tag));
    let _ = std::fs::remove_file(&temporary);
    match written {
        Ok(()) => Ok(()),
        // Another writer put it there first.
        Err(_) if tag.is_file() => Ok(()),
        Err(e) => Err(format!("{} could not be written: {e}", tag.display())),
    }
}
