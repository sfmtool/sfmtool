// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic photographs for the mipmap tests and the photograph cache's
//! tests of reading a mipmap: a few hundred pixels each, so no test reads or
//! makes a large photograph.

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use crate::camera::image::ImageU8;

/// A `width` x `height` RGB image with smooth gradients and some texture, the
/// same for the same size.
pub(crate) fn synthetic_rgb(width: u32, height: u32) -> ImageU8 {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let texture = ((x * 7 + y * 13) % 23) as u8;
            data.push((x * 255 / width.max(1)) as u8 ^ texture);
            data.push((y * 255 / height.max(1)) as u8);
            data.push(((x + y) % 256) as u8 / 2 + texture);
        }
    }
    ImageU8::new(width, height, 3, data)
}

/// How a test JPEG is written.
#[derive(Clone, Copy)]
pub(crate) struct JpegOptions {
    pub(crate) sampling: SamplingFactor,
    /// 0 writes no `DRI`.
    pub(crate) restart_interval: u16,
    pub(crate) progressive: bool,
    pub(crate) grey: bool,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self {
            sampling: SamplingFactor::F_2_2,
            restart_interval: 0,
            progressive: false,
            grey: false,
        }
    }
}

/// `image` encoded as a JPEG by `jpeg-encoder` at quality 90.
pub(crate) fn encode_jpeg(image: &ImageU8, options: JpegOptions) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 90);
    encoder.set_sampling_factor(options.sampling);
    encoder.set_progressive(options.progressive);
    if options.restart_interval > 0 {
        encoder.set_restart_interval(options.restart_interval);
    }
    let (w, h) = (image.width() as u16, image.height() as u16);
    if options.grey {
        let grey: Vec<u8> = image
            .data()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[1])
            .collect();
        encoder.encode(&grey, w, h, ColorType::Luma).unwrap();
    } else {
        encoder.encode(image.data(), w, h, ColorType::Rgb).unwrap();
    }
    bytes
}

/// `image` encoded as a PNG.
pub(crate) fn encode_png(image: &ImageU8) -> Vec<u8> {
    use ::image::ImageEncoder;
    let mut bytes = Vec::new();
    ::image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            image.data(),
            image.width(),
            image.height(),
            ::image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    bytes
}

/// A temporary workspace for the tests of checking and pruning mipmaps: a
/// directory with a `.sfm-workspace.json`, with `mipmap_options` of a 64 px
/// threshold and `quality` for both qualities when `quality` is given.
pub(crate) struct TestWorkspace {
    pub(crate) dir: tempfile::TempDir,
}

impl TestWorkspace {
    pub(crate) fn new(quality: Option<u8>) -> Self {
        let ws = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        ws.set_quality(quality);
        ws
    }

    /// The root, spelled as `find_workspace_for_path` spells it.
    pub(crate) fn root(&self) -> std::path::PathBuf {
        crate::workspace::find_workspace_for_path(&self.dir.path().join("x")).unwrap()
    }

    /// Write the workspace file, with `mipmap_options` at `quality` or none.
    pub(crate) fn set_quality(&self, quality: Option<u8>) {
        write_workspace_file(self.dir.path(), quality);
    }

    /// Write `bytes` at `relative` under the root and return its path.
    pub(crate) fn write(&self, relative: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = self.root().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// The path of `relative` under the mipmap directory.
    pub(crate) fn mipmap(&self, relative: &str) -> std::path::PathBuf {
        self.root().join(super::MIPMAP_DIR_NAME).join(relative)
    }
}

/// Write a version 2 `.sfm-workspace.json` in `dir`, with `mipmap_options`
/// of a 64 px threshold and `quality` for both qualities when given.
pub(crate) fn write_workspace_file(dir: &std::path::Path, quality: Option<u8>) {
    let object = quality.map_or(String::new(), |q| {
        format!(
            r#", "mipmap_options": {{"out_of_core_long_side": 64, "jpeg_quality": {q},
                "full_resolution_jpeg_quality": {q}}}"#
        )
    });
    std::fs::write(
        dir.join(crate::workspace::WORKSPACE_FILE_NAME),
        format!(r#"{{"version": 2{object}}}"#),
    )
    .unwrap();
}

/// Make the mipmap of `photograph` under its workspace's options, with
/// photograph tiles of `tile_size`.
pub(crate) fn make_mipmap(photograph: &std::path::Path, tile_size: u32) {
    let (_, location) = super::photograph_layout(photograph, tile_size).unwrap();
    let root = location.expect("out of core").root().to_path_buf();
    let options = crate::workspace::read_mipmap_options(&root)
        .unwrap()
        .unwrap();
    super::build_mipmaps(
        photograph,
        None,
        &options,
        tile_size,
        &crate::progress::Progress::none(),
        &mut |_| {},
    )
    .unwrap();
}
