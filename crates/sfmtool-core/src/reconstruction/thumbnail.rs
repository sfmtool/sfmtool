// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnail a reconstruction carries for an image, made from its
//! photograph.
//!
//! A `.sfmr` thumbnail row is the source photograph resized to
//! [`THUMBNAIL_SIZE`] x [`THUMBNAIL_SIZE`] by area averaging, stretched to the
//! square (see `specs/formats/sfmr-file-format.md`). [`resize_area`] is an area
//! resize for any size. It matches OpenCV's `INTER_AREA` in method, each output
//! pixel the coverage-weighted mean of the source pixels under it, but not bit
//! for bit. [`thumbnail_from_rgb`] is the thumbnail resize itself: for a
//! photograph at least [`THUMBNAIL_SIZE`] on each side it follows OpenCV's
//! `INTER_AREA` arithmetic step for step, so its row is the one the Python SIFT
//! extractors and `sfm xform --add-thumbnails` make with `cv2.resize`, byte for
//! byte. The Rust `.sift` writer
//! [`build_sift_files`](crate::features::sift::build_sift_files) and the
//! viewer's display thumbnails use it.
//!
//! [`verified_sift_thumbnail`] reads the row an image's `.sift` already holds,
//! checked against the reconstruction's record of that file. A caller filling
//! in an absent column tries it before the photograph, since it is the same row
//! already reduced.

use ndarray::Array3;

use crate::{SfmrReconstruction, THUMBNAIL_SIZE};
use sfmtool_archive_io::parse_hash_bytes;

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
/// photograph `width` x `height`.
///
/// When neither side is shorter than [`THUMBNAIL_SIZE`], the row equals
/// OpenCV's `cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)` byte
/// for byte, which is how the Python extractors make a `.sift` thumbnail. An
/// integer reduction on both axes is the block sum times `1/area`, rounded half
/// to even, except that a 2 x 2 reduction rounds half up as OpenCV's vector path
/// does; any other reduction is OpenCV's coverage-weighted sum in `f32`, in its
/// order of operations. A photograph shorter than the thumbnail on a side is
/// enlarged on that axis, which OpenCV does by bilinear interpolation; such a
/// photograph goes through [`resize_area`] instead and does not match OpenCV.
///
/// # Panics
///
/// As [`resize_area`], with three channels.
pub fn thumbnail_from_rgb(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    if width >= THUMBNAIL_SIZE && height >= THUMBNAIL_SIZE {
        assert_eq!(
            pixels.len(),
            width * height * 3,
            "pixels must hold width * height * channels bytes"
        );
        reduce_like_inter_area(pixels, width, height, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    } else {
        resize_area(pixels, width, height, 3, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    }
}

/// OpenCV's `cvRound` of an `f32`, saturated to a byte: round half to even.
fn round_to_u8(value: f32) -> u8 {
    value.round_ties_even().clamp(0.0, 255.0) as u8
}

/// One axis of OpenCV's area reduction: `(dst index, src index, weight)` in
/// the order `computeResizeAreaTab` lists them, for a reduction by `scale`
/// (`src_len / dst_len` as OpenCV computes it, at least 1).
fn inter_area_taps(src_len: usize, dst_len: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut taps = Vec::with_capacity(src_len * 2);
    for d in 0..dst_len {
        let fs1 = d as f64 * scale;
        let fs2 = fs1 + scale;
        let cell = scale.min(src_len as f64 - fs1);
        let s2 = (fs2.floor() as usize).min(src_len - 1);
        let s1 = (fs1.ceil() as usize).min(s2);
        if s1 as f64 - fs1 > 1e-3 {
            taps.push((d, s1 - 1, ((s1 as f64 - fs1) / cell) as f32));
        }
        for s in s1..s2 {
            taps.push((d, s, (1.0 / cell) as f32));
        }
        if fs2 - s2 as f64 > 1e-3 {
            taps.push((d, s2, ((fs2 - s2 as f64).min(1.0).min(cell) / cell) as f32));
        }
    }
    taps
}

/// A 3-channel reduction (`dst <= src` on both axes) with OpenCV's `INTER_AREA`
/// arithmetic; see [`thumbnail_from_rgb`] for what that is.
fn reduce_like_inter_area(
    pixels: &[u8],
    width: usize,
    height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    const CN: usize = 3;
    // cv::resize takes the scale as the reciprocal of dst / src, which is not
    // always the same double as src / dst.
    let scale_x = 1.0 / (dst_width as f64 / width as f64);
    let scale_y = 1.0 / (dst_height as f64 / height as f64);
    let iscale_x = scale_x.round();
    let iscale_y = scale_y.round();
    let mut out = vec![0u8; dst_width * dst_height * CN];

    if (scale_x - iscale_x).abs() < f64::EPSILON && (scale_y - iscale_y).abs() < f64::EPSILON {
        // OpenCV's `resizeAreaFast`: the integer block sum times 1/area.
        let (sx, sy) = (iscale_x as usize, iscale_y as usize);
        let inv_area = 1.0f32 / (sx * sy) as f32;
        for dy in 0..dst_height {
            for dx in 0..dst_width {
                for c in 0..CN {
                    let mut sum = 0u32;
                    for y in dy * sy..(dy + 1) * sy {
                        let row = &pixels[y * width * CN..(y + 1) * width * CN];
                        for x in dx * sx..(dx + 1) * sx {
                            sum += u32::from(row[x * CN + c]);
                        }
                    }
                    out[(dy * dst_width + dx) * CN + c] = if sx == 2 && sy == 2 {
                        // The 2 x 2 vector path: (sum + 2) >> 2.
                        ((sum + 2) >> 2) as u8
                    } else {
                        round_to_u8(sum as f32 * inv_area)
                    };
                }
            }
        }
        return out;
    }

    // OpenCV's `ResizeArea_Invoker`: each source row is reduced horizontally
    // into `buf`, then added into its output row's `sum` with that row's
    // weight, in the order the tables list the taps.
    let x_taps = inter_area_taps(width, dst_width, scale_x);
    let y_taps = inter_area_taps(height, dst_height, scale_y);
    let row_len = dst_width * CN;
    let mut buf = vec![0f32; row_len];
    let mut sum = vec![0f32; row_len];
    let mut current = y_taps.first().map_or(0, |t| t.0);
    for &(dy, sy, beta) in &y_taps {
        let src = &pixels[sy * width * CN..(sy + 1) * width * CN];
        buf.fill(0.0);
        for &(dx, sx, alpha) in &x_taps {
            for c in 0..CN {
                buf[dx * CN + c] += f32::from(src[sx * CN + c]) * alpha;
            }
        }
        if dy != current {
            let dst = &mut out[current * row_len..(current + 1) * row_len];
            for ((d, s), b) in dst.iter_mut().zip(sum.iter_mut()).zip(&buf) {
                *d = round_to_u8(*s);
                *s = beta * b;
            }
            current = dy;
        } else {
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s += beta * b;
            }
        }
    }
    let dst = &mut out[current * row_len..(current + 1) * row_len];
    for (d, s) in dst.iter_mut().zip(&sum) {
        *d = round_to_u8(*s);
    }
    out
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
    if index >= recon.image_set.images.len() || recon.workspace_dir.as_os_str().is_empty() {
        return None;
    }
    let path = recon.sift_path_for_image(index);
    if !path.is_file() {
        return None;
    }
    let (metadata, content_hash, thumbnail) =
        sfmtool_sift_format::read_sift_thumbnail(&path).ok()?;
    let belongs = if let Some(hashes) = recon.image_file_hashes() {
        parse_hash_bytes(&metadata.image_file_xxh128).ok() == Some(*hashes.get(index)?)
    } else if let Some(hashes) = recon.sift_content_hashes() {
        parse_hash_bytes(&content_hash.content_xxh128).ok() == Some(*hashes.get(index)?)
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
/// Panics if `index` is past the image set.
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
        .join(&recon.image_set.images[index].name);
    if path.is_file() {
        if let Ok(image) = crate::camera::image::ImageU8::read_rgb(&path) {
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

    /// `w` x `h` RGB noise from a 32-bit xorshift, one byte per step from its
    /// top eight bits.
    fn noise(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..w * h * 3)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn a_thumbnail_equals_opencv_inter_area_byte_for_byte() {
        // The hashes are xxh3_64 of `cv2.resize(img, (128, 128),
        // interpolation=cv2.INTER_AREA)` (OpenCV 5.0.0) for the same noise,
        // made in numpy with this generator: the fractional reduction
        // (270 x 480, 2040 x 1536, 641 x 129), the integer one (384 x 512), the
        // 2 x 2 vector path (256 x 256) and the identity (128 x 128).
        let cases = [
            (270, 480, "4fee12c3d05c540c"),
            (2040, 1536, "4b3bee0740a62628"),
            (384, 512, "da01e889f60f1dad"),
            (256, 256, "5ba3dc6192213ca9"),
            (128, 128, "e24a9a816e37998d"),
            (641, 129, "2ff2ab4aef589a29"),
        ];
        for (w, h, expected) in cases {
            let seed = 0x9E37_79B9u32 ^ (w * 31 + h) as u32;
            let row = thumbnail_from_rgb(&noise(w, h, seed), w, h);
            let hash = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&row));
            assert_eq!(hash, expected, "{w} x {h}");
        }
    }

    #[test]
    fn a_thumbnail_of_a_small_photograph_still_fills_the_square() {
        // Shorter than the thumbnail on one side: the resize_area path.
        let pixels = noise(300, 100, 7);
        let row = thumbnail_from_rgb(&pixels, 300, 100);
        assert_eq!(row, resize_area(&pixels, 300, 100, 3, 128, 128));
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
            .image_set
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
            parse_hash_bytes(&hash.content_xxh128).unwrap()
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
        let n = embedded.image_set.images.len();
        let mut hashes = vec![[0u8; 16]; n];
        hashes[1] = parse_hash_bytes(image_hash).unwrap();
        embedded.point_set.observations = crate::ObservationSource::EmbeddedPatches {
            keypoints_xy: ndarray::Array2::zeros((embedded.point_set.tracks.len(), 2)),
            image_file_hashes: hashes,
        };
        let row = verified_sift_thumbnail(&embedded, 1).expect("the photograph's .sift");
        assert!(row.iter().all(|&b| b == 90));
        assert!(verified_sift_thumbnail(&embedded, 0).is_none());
    }
}
