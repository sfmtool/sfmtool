// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnail a reconstruction carries for an image, made from its
//! photograph.
//!
//! A `.sfmr` thumbnail row is the source photograph resized to
//! [`THUMBNAIL_SIZE`] x [`THUMBNAIL_SIZE`] by area averaging, stretched to the
//! square (see `specs/formats/sfmr-file-format.md`). [`resize_area`] is that
//! resize for any Rust caller. It matches OpenCV's `INTER_AREA` in method, each
//! output pixel the coverage-weighted mean of the source pixels under it, but
//! not bit for bit, so a caller that must reproduce stored rows exactly (such
//! as `sfm xform --add-thumbnails`) uses the extractors' own OpenCV path
//! instead. The viewer uses this one for display thumbnails.
//!
//! [`verified_sift_thumbnail`] reads the row an image's `.sift` already holds,
//! checked against the reconstruction's record of that file. A caller filling
//! in an absent column tries it before the photograph, since it is the same row
//! already reduced.

use ndarray::Array3;

use super::embed::decode_xxh128_hex;
use crate::{SfmrReconstruction, THUMBNAIL_SIZE};

/// One output sample's source taps along one axis: `(source index, weight)`,
/// the weights summing to one.
fn axis_taps(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|d| {
            let start = d as f64 * scale;
            let end = (d + 1) as f64 * scale;
            let mut taps = Vec::new();
            let mut s = start.floor() as usize;
            while (s as f64) < end && s < src_len {
                let lo = start.max(s as f64);
                let hi = end.min((s + 1) as f64);
                if hi > lo {
                    taps.push((s, ((hi - lo) / scale) as f32));
                }
                s += 1;
            }
            taps
        })
        .collect()
}

/// Resize an interleaved 8-bit image by area averaging.
///
/// `pixels` is `height` rows of `width` pixels of `channels` bytes each, row
/// major. Returns `dst_height` rows of `dst_width` pixels in the same layout.
/// Each output pixel is the mean of the source area it covers, each source
/// pixel weighted by the fraction of it inside that area, rounded to the
/// nearest integer. Upscaling an axis degenerates to sampling the covering
/// source pixel.
///
/// # Panics
///
/// Panics if `pixels.len() != width * height * channels`, or any dimension is
/// zero.
pub fn resize_area(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    assert!(
        width > 0 && height > 0 && channels > 0 && dst_width > 0 && dst_height > 0,
        "resize_area needs non-empty images"
    );
    assert_eq!(
        pixels.len(),
        width * height * channels,
        "pixels must hold width * height * channels bytes"
    );
    let x_taps = axis_taps(width, dst_width);
    let y_taps = axis_taps(height, dst_height);

    // Horizontal pass into f32 rows, then the vertical pass per output row.
    let mut horizontal = vec![0f32; height * dst_width * channels];
    for y in 0..height {
        let src_row = &pixels[y * width * channels..(y + 1) * width * channels];
        let dst_row = &mut horizontal[y * dst_width * channels..(y + 1) * dst_width * channels];
        for (dx, taps) in x_taps.iter().enumerate() {
            for c in 0..channels {
                let mut acc = 0f32;
                for &(sx, w) in taps {
                    acc += w * f32::from(src_row[sx * channels + c]);
                }
                dst_row[dx * channels + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; dst_height * dst_width * channels];
    for (dy, taps) in y_taps.iter().enumerate() {
        let dst_row = &mut out[dy * dst_width * channels..(dy + 1) * dst_width * channels];
        for (i, value) in dst_row.iter_mut().enumerate() {
            let mut acc = 0f32;
            for &(sy, w) in taps {
                acc += w * horizontal[sy * dst_width * channels + i];
            }
            *value = (acc + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// The `THUMBNAIL_SIZE` x `THUMBNAIL_SIZE` RGB thumbnail row of an RGB
/// photograph `width` x `height`, by [`resize_area`].
///
/// # Panics
///
/// As [`resize_area`], with three channels.
pub fn thumbnail_from_rgb(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    resize_area(pixels, width, height, 3, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
}

/// Image `index`'s thumbnail from its `.sift` file, when that file verifiably
/// belongs to the image.
///
/// The `.sift` is the one [`SfmrReconstruction::sift_path_for_image`] names. In
/// a `sift_files` reconstruction it belongs when its stored content hash is the
/// image's `sift_content_hashes` entry; in an `embedded_patches` one, when its
/// recorded `image_file_xxh128` is the image's `image_file_hashes` entry, the
/// hash that says it was extracted from the photograph the reconstruction was
/// built from. `None` when there is no such file, it cannot be read, or it does
/// not match.
///
/// A `.sift` thumbnail is already the 128 x 128 row a reconstruction carries,
/// made by the extractor from the photograph, so reading it is a decompression
/// of 48 KiB where the photograph is a full decode and a resize. That is why a
/// caller filling in a missing column tries it first and the photograph second.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::progress::Progress;
/// use sfmtool_core::reconstruction::thumbnail::verified_sift_thumbnail;
/// use sfmtool_core::SfmrReconstruction;
/// # fn run(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
/// let recon = SfmrReconstruction::load(path, &Progress::none())?;
/// let found = (0..recon.image_count())
///     .filter(|&i| verified_sift_thumbnail(&recon, i).is_some())
///     .count();
/// println!("{found} of {} thumbnails are in the .sift files", recon.image_count());
/// # Ok(())
/// # }
/// ```
pub fn verified_sift_thumbnail(recon: &SfmrReconstruction, index: usize) -> Option<Array3<u8>> {
    if index >= recon.image_table.images.len() || recon.workspace_dir.as_os_str().is_empty() {
        return None;
    }
    let path = recon.sift_path_for_image(index);
    if !path.is_file() {
        return None;
    }
    let (metadata, content_hash, thumbnail) =
        sfmtool_sift_format::read_sift_thumbnail(&path).ok()?;
    let belongs = if let Some(hashes) = recon.image_file_hashes() {
        decode_xxh128_hex(&metadata.image_file_xxh128) == Some(*hashes.get(index)?)
    } else if let Some(hashes) = recon.sift_content_hashes() {
        decode_xxh128_hex(&content_hash.content_xxh128) == Some(*hashes.get(index)?)
    } else {
        false
    };
    belongs.then_some(thumbnail)
}

/// The grey a display thumbnail row is filled with when neither the image's
/// `.sift` nor its photograph can supply one.
///
/// A neutral mid-grey rather than black: black reads as a dark photograph,
/// while a flat grey reads as "no picture here".
pub const PLACEHOLDER_GREY: u8 = 128;

/// Where a display thumbnail row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailSource {
    /// Read from the image's verified `.sift` ([`verified_sift_thumbnail`]).
    Sift,
    /// Decoded from the photograph and resized by area averaging.
    Photograph,
    /// Neither could supply one; the row is flat [`PLACEHOLDER_GREY`].
    Placeholder,
}

/// Image `index`'s display thumbnail row for a reconstruction that carries no
/// thumbnail column: `THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3` RGB bytes, row major.
///
/// The image's `.sift` thumbnail when a `.sift` verifiably belongs to it,
/// otherwise its photograph (`workspace_dir` joined with the image name)
/// decoded and resized by [`thumbnail_from_rgb`], otherwise flat
/// [`PLACEHOLDER_GREY`]. SfM Explorer's open and `sfm web-export` both fill a
/// missing column with this, so the two show the same picture.
///
/// # Panics
///
/// Panics if `index` is past the image table.
pub fn display_thumbnail_row(
    recon: &SfmrReconstruction,
    index: usize,
) -> (Vec<u8>, ThumbnailSource) {
    if let Some(thumbnail) = verified_sift_thumbnail(recon, index) {
        let pixels = thumbnail.as_standard_layout().iter().copied().collect();
        return (pixels, ThumbnailSource::Sift);
    }
    let path = recon
        .workspace_dir
        .join(&recon.image_table.images[index].name);
    if path.is_file() {
        if let Ok(image) = crate::camera::remap::ImageU8::read_rgb(&path) {
            let (width, height) = (image.width() as usize, image.height() as usize);
            if width > 0 && height > 0 {
                return (
                    thumbnail_from_rgb(image.data(), width, height),
                    ThumbnailSource::Photograph,
                );
            }
        }
    }
    (
        vec![PLACEHOLDER_GREY; THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3],
        ThumbnailSource::Placeholder,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_downscale_is_the_block_mean() {
        // 4x2 single-channel image to 2x1: each output is the mean of a 2x2 block.
        let pixels = [0u8, 10, 100, 200, 20, 30, 50, 50];
        let out = resize_area(&pixels, 4, 2, 1, 2, 1);
        assert_eq!(out, vec![15, 100]);
    }

    #[test]
    fn a_fractional_downscale_weights_by_coverage() {
        // 3 pixels to 2: the middle one is split half and half.
        let pixels = [0u8, 90, 180];
        let out = resize_area(&pixels, 3, 1, 1, 2, 1);
        // (0 * 1 + 90 * 0.5) / 1.5 = 30, (90 * 0.5 + 180 * 1) / 1.5 = 150.
        assert_eq!(out, vec![30, 150]);
    }

    #[test]
    fn a_flat_image_stays_flat_and_channels_stay_apart() {
        let pixels: Vec<u8> = (0..270 * 480).flat_map(|_| [7u8, 128, 250]).collect();
        let out = thumbnail_from_rgb(&pixels, 270, 480);
        assert_eq!(out.len(), THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3);
        assert!(out.chunks(3).all(|p| p == [7, 128, 250]));
    }

    #[test]
    fn the_same_size_is_the_identity() {
        let pixels: Vec<u8> = (0..5 * 4 * 3).map(|i| (i * 37 % 256) as u8).collect();
        assert_eq!(resize_area(&pixels, 5, 4, 3, 5, 4), pixels);
    }

    /// A `.sift` for `name` under `workspace`, whose thumbnail is flat `value`,
    /// recorded as extracted from a photograph hashing to `image_hash`.
    fn write_sift_for(workspace: &std::path::Path, name: &str, value: u8, image_hash: &str) {
        use ndarray::{Array2, Array3};
        use sfmtool_sift_format::*;
        let data = SiftData {
            feature_tool_metadata: FeatureToolMetadata {
                feature_tool: "sfmtool".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
            },
            metadata: SiftMetadata {
                version: 1,
                image_name: name.into(),
                image_file_xxh128: image_hash.into(),
                image_file_size: 1,
                image_width: 64,
                image_height: 64,
                feature_count: 0,
            },
            content_hash: SiftContentHash::default(),
            positions_xy: Array2::zeros((0, 2)),
            affine_shapes: Array3::zeros((0, 2, 2)),
            descriptors: Array2::zeros((0, 128)),
            thumbnail_y_x_rgb: Array3::from_elem((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), value),
        };
        write_sift(&workspace.join(format!("{name}.sift")), &data, 3).unwrap();
    }

    #[test]
    fn a_sift_thumbnail_is_read_only_when_its_hash_matches() {
        let dir = tempfile::tempdir().unwrap();
        let mut recon = SfmrReconstruction::demo(12);
        recon.workspace_dir = dir.path().to_path_buf();
        let names: Vec<String> = recon
            .image_table
            .images
            .iter()
            .map(|image| image.name.clone())
            .collect();
        let image_hash = "00112233445566778899aabbccddeeff";
        write_sift_for(dir.path(), &names[0], 40, image_hash);
        write_sift_for(dir.path(), &names[1], 90, image_hash);

        // A sift_files reconstruction names each .sift by its content hash:
        // image 0's record matches its file, image 1's does not.
        let stored = |name: &str| {
            let path = dir.path().join(format!("{name}.sift"));
            let (_, _, hash) = sfmtool_sift_format::read_sift_metadata(&path).unwrap();
            decode_xxh128_hex(&hash.content_xxh128).unwrap()
        };
        let first = stored(&names[0]);
        if let crate::ObservationSource::SiftFiles {
            sift_content_hashes,
            ..
        } = &mut recon.point_set.observations
        {
            sift_content_hashes[0] = first;
        }
        let row = verified_sift_thumbnail(&recon, 0).expect("a matching .sift");
        assert!(row.iter().all(|&b| b == 40));
        assert!(verified_sift_thumbnail(&recon, 1).is_none(), "hash differs");
        assert!(verified_sift_thumbnail(&recon, 2).is_none(), "no file");
        assert!(verified_sift_thumbnail(&recon, 99).is_none(), "no image");

        // An embedded_patches reconstruction names the photograph instead.
        let mut embedded = recon.clone();
        let n = embedded.image_table.images.len();
        let mut hashes = vec![[0u8; 16]; n];
        hashes[1] = decode_xxh128_hex(image_hash).unwrap();
        embedded.point_set.observations = crate::ObservationSource::EmbeddedPatches {
            keypoints_xy: ndarray::Array2::zeros((embedded.point_set.tracks.len(), 2)),
            image_file_hashes: hashes,
        };
        let row = verified_sift_thumbnail(&embedded, 1).expect("the photograph's .sift");
        assert!(row.iter().all(|&b| b == 90));
        assert!(verified_sift_thumbnail(&embedded, 0).is_none());
    }
}
