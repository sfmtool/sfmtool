// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene: a textured ground plane with a cube standing on it,
//! photographed by five pinhole cameras, so that a reconstruction can be
//! written down from a scene whose every number is known.
//!
//! [`DemoScene::render`] renders the five photographs from
//! [`DemoSceneOptions`], encodes each as a JPEG and decodes it again;
//! [`DemoScene::reconstruction`] writes down the scene's `embedded_patches`
//! reconstruction, and [`DemoScene::write_workspace`] writes the photographs,
//! a workspace marker and that reconstruction to a directory. The scene's
//! geometry, its cameras and its texture are fixed, and are described in
//! `specs/drafts/demo-scene.md`; the options are the parts a caller has a
//! reason to change. The children are [`texture`], the surfaces and their
//! tinted Perlin-noise texture; [`render`], the camera rotations and the ray
//! tracer; [`points`], the stored poses, the point sampling and the
//! visibility test; and [`value`], the reconstruction's columns.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::camera::image::{ImageU8, DEFAULT_JPEG_QUALITY};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::{Cancelled, Progress};

mod points;
mod render;
mod texture;
mod value;

#[cfg(test)]
pub(crate) use texture::value_noise;

/// How the demo scene is rendered and sampled. `Default` is the scene
/// `specs/drafts/demo-scene.md` describes.
///
/// [`DemoScene::render`] reads the image size, the focal length, the
/// supersampling and the seed; the point spacings, the patch half-extent, the
/// bitmap side and the obliquity cut are the sampling's, and are checked by
/// `render` so that a scene that renders also samples.
#[derive(Debug, Clone, PartialEq)]
pub struct DemoSceneOptions {
    /// The photographs' width in pixels: 640.
    pub image_width: u32,
    /// The photographs' height in pixels: 480.
    pub image_height: u32,
    /// The pinhole focal length in pixels, the same on both axes: 560.
    pub focal_length_px: f64,
    /// Samples per pixel along each axis; the renderer takes the square of
    /// this many rays per pixel: 2 (4 rays).
    pub supersampling: u32,
    /// The spacing of the point grid on the ground, in metres: 0.25. The
    /// grid over the 5 m square the points are sampled on has
    /// `round(5 / ground_point_spacing)` rows and columns.
    pub ground_point_spacing: f64,
    /// The spacing of the point grid on each cube face, in metres: 0.125.
    /// The grid on a 1 m face has `round(1 / face_point_spacing)` rows and
    /// columns.
    pub face_point_spacing: f64,
    /// Each point's patch half-extent, in metres: 0.09.
    pub patch_half_extent: f64,
    /// The patch bitmaps' side `R`: 24.
    pub patch_bitmap_resolution: u32,
    /// A camera that views a point's patch more than this many degrees off
    /// its normal does not observe it: 80.
    pub max_obliquity_deg: f64,
    /// Picks the surfaces' textures and the points' offsets from the grid.
    pub seed: u64,
}

impl Default for DemoSceneOptions {
    fn default() -> Self {
        Self {
            image_width: 640,
            image_height: 480,
            focal_length_px: 560.0,
            supersampling: 2,
            ground_point_spacing: 0.25,
            face_point_spacing: 0.125,
            patch_half_extent: 0.09,
            patch_bitmap_resolution: 24,
            max_obliquity_deg: 80.0,
            seed: 0,
        }
    }
}

impl DemoSceneOptions {
    /// The number of points [`DemoScene::reconstruction`] keeps under the
    /// default options: 524.
    pub const DEFAULT_POINT_COUNT: u32 = 524;

    /// The default options with both point spacings scaled by
    /// `sqrt(DEFAULT_POINT_COUNT / point_count)`, so the scene keeps about
    /// `point_count` points: the points on a surface number about its area
    /// over the spacing squared.
    ///
    /// # Errors
    ///
    /// [`DemoSceneError::Options`] for a `point_count` of 0.
    ///
    /// ```
    /// use sfmtool_core::reconstruction::DemoSceneOptions;
    ///
    /// let options = DemoSceneOptions::for_point_count(4 * DemoSceneOptions::DEFAULT_POINT_COUNT)?;
    /// assert_eq!(options.face_point_spacing, 0.0625);
    /// # Ok::<(), sfmtool_core::reconstruction::DemoSceneError>(())
    /// ```
    pub fn for_point_count(point_count: u32) -> Result<DemoSceneOptions, DemoSceneError> {
        if point_count == 0 {
            return Err(DemoSceneError::Options(
                "point_count is 0; it must be at least 1".to_owned(),
            ));
        }
        let scale = (f64::from(Self::DEFAULT_POINT_COUNT) / f64::from(point_count)).sqrt();
        let default = DemoSceneOptions::default();
        Ok(DemoSceneOptions {
            ground_point_spacing: default.ground_point_spacing * scale,
            face_point_spacing: default.face_point_spacing * scale,
            ..default
        })
    }

    /// `Ok` when every option is in its range, and otherwise the first one
    /// that is not, named with its range.
    fn validate(&self) -> Result<(), DemoSceneError> {
        let fail = |message: String| Err(DemoSceneError::Options(message));
        for (name, value) in [
            ("image_width", self.image_width),
            ("image_height", self.image_height),
        ] {
            if !(1..=u32::from(u16::MAX)).contains(&value) {
                return fail(format!(
                    "{name} is {value}; it must be from 1 to 65535, the sides a JPEG holds"
                ));
            }
        }
        if self.supersampling == 0 {
            return fail("supersampling is 0; it must be at least 1".to_owned());
        }
        if self.patch_bitmap_resolution == 0 {
            return fail("patch_bitmap_resolution is 0; it must be at least 1".to_owned());
        }
        for (name, value) in [
            ("focal_length_px", self.focal_length_px),
            ("ground_point_spacing", self.ground_point_spacing),
            ("face_point_spacing", self.face_point_spacing),
            ("patch_half_extent", self.patch_half_extent),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return fail(format!("{name} is {value}; it must be finite and positive"));
            }
        }
        if !(0.0..=90.0).contains(&self.max_obliquity_deg) {
            return fail(format!(
                "max_obliquity_deg is {}; it must be from 0 to 90",
                self.max_obliquity_deg
            ));
        }
        Ok(())
    }
}

/// The rendered scene: its camera, poses and photographs, in memory.
pub struct DemoScene {
    options: DemoSceneOptions,
    image_names: Vec<String>,
    rendered_photographs: Vec<ImageU8>,
    encoded_photographs: Vec<Vec<u8>>,
    photographs: Vec<ImageU8>,
}

impl DemoScene {
    /// Render the scene's photographs, one per camera, in a phase `render`
    /// that counts rows; then encode each as a JPEG at
    /// [`DEFAULT_JPEG_QUALITY`] and decode it again, in a phase `encode` that
    /// counts images.
    ///
    /// The render depends only on `options`: the rows are rendered in
    /// parallel, each by one thread and each pixel's samples summed in a fixed
    /// order, so the thread count does not change a pixel, and the arithmetic
    /// is `+`, `−`, `×`, `÷`, `sqrt` and `floor`, which IEEE 754 rounds the
    /// same on every platform.
    ///
    /// # Errors
    ///
    /// [`DemoSceneError::Options`] for an option out of its range,
    /// [`DemoSceneError::Cancelled`] when `progress` is cancelled, and
    /// [`DemoSceneError::Photograph`] when a photograph cannot be encoded or
    /// decoded.
    ///
    /// ```
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::reconstruction::{DemoScene, DemoSceneOptions};
    ///
    /// let options = DemoSceneOptions {
    ///     image_width: 64,
    ///     image_height: 48,
    ///     focal_length_px: 56.0,
    ///     ..DemoSceneOptions::default()
    /// };
    /// let scene = DemoScene::render(&options, &Progress::none())?;
    /// assert_eq!(scene.image_names()[0], "images/demo_0.jpg");
    /// assert_eq!(scene.photographs()[4].width(), 64);
    /// # Ok::<(), sfmtool_core::reconstruction::DemoSceneError>(())
    /// ```
    pub fn render(
        options: &DemoSceneOptions,
        progress: &Progress<'_>,
    ) -> Result<DemoScene, DemoSceneError> {
        options.validate()?;
        let camera = demo_camera(options);
        let (width, height) = (options.image_width, options.image_height);
        let row_bytes = width as usize * 3;
        let total_rows = (render::CAMERA_CENTRES.len() as u64) * u64::from(height);

        let rendered_photographs = {
            let phase = progress.phase("render");
            let rows_progress: &Progress<'_> = &phase;
            let rows_done = AtomicU64::new(0);
            let mut photographs = Vec::with_capacity(render::CAMERA_CENTRES.len());
            for centre in render::CAMERA_CENTRES {
                let rotation = render::camera_rotation(centre);
                let mut data = vec![0u8; row_bytes * height as usize];
                data.par_chunks_mut(row_bytes).enumerate().try_for_each(
                    |(row, out)| -> Result<(), Cancelled> {
                        rows_progress.check_cancel()?;
                        render::render_row(
                            &camera,
                            centre,
                            &rotation,
                            options.supersampling,
                            options.seed,
                            row as u32,
                            out,
                        );
                        let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                        rows_progress.count(done, Some(total_rows), "rows");
                        Ok(())
                    },
                )?;
                photographs.push(ImageU8::new(width, height, 3, data));
            }
            photographs
        };

        let (encoded_photographs, photographs) = {
            let phase = progress.phase("encode");
            let count = rendered_photographs.len() as u64;
            let mut encoded = Vec::with_capacity(rendered_photographs.len());
            let mut decoded = Vec::with_capacity(rendered_photographs.len());
            for (i, rendered) in rendered_photographs.iter().enumerate() {
                phase.check_cancel()?;
                let bytes = rendered.encode(::image::ImageFormat::Jpeg, DEFAULT_JPEG_QUALITY)?;
                decoded.push(ImageU8::decode_rgb(&bytes)?);
                encoded.push(bytes);
                phase.count(i as u64 + 1, Some(count), "images");
            }
            (encoded, decoded)
        };

        Ok(DemoScene {
            options: options.clone(),
            image_names: (0..render::CAMERA_CENTRES.len())
                .map(|i| format!("images/demo_{i}.jpg"))
                .collect(),
            rendered_photographs,
            encoded_photographs,
            photographs,
        })
    }

    /// The options the scene was rendered with.
    pub fn options(&self) -> &DemoSceneOptions {
        &self.options
    }

    /// Each photograph's image name, `images/demo_0.jpg` to `images/demo_4.jpg`.
    pub fn image_names(&self) -> &[String] {
        &self.image_names
    }

    /// The renderer's exact output, RGB, in image order, before JPEG.
    pub fn rendered_photographs(&self) -> &[ImageU8] {
        &self.rendered_photographs
    }

    /// The photographs as a reader of the written files decodes them: each
    /// JPEG of [`encoded_photographs`](Self::encoded_photographs) decoded by
    /// [`ImageU8::decode_rgb`].
    pub fn photographs(&self) -> &[ImageU8] {
        &self.photographs
    }

    /// Each photograph's JPEG file, encoded from
    /// [`rendered_photographs`](Self::rendered_photographs) by
    /// [`ImageU8::encode`] at [`DEFAULT_JPEG_QUALITY`]. The bytes are the
    /// same on every run of one encoder path, and differ between the JPEG
    /// encoder's AVX2 path, which an x86 CPU with AVX2 takes, and its scalar
    /// one.
    pub fn encoded_photographs(&self) -> &[Vec<u8>] {
        &self.encoded_photographs
    }

    /// Write the scene as a workspace in `dir`, which must be an empty
    /// directory or not exist, and must not lie inside another workspace:
    /// the JPEGs of [`encoded_photographs`](Self::encoded_photographs) under
    /// `images/`, a `.sfm-workspace.json` for the `sfmtool` feature tool with
    /// its default options, and [`reconstruction`](Self::reconstruction)'s
    /// value as `demo.sfmr` at the workspace root, which records the
    /// workspace as its own directory, `.`. Returns the `.sfmr` path.
    ///
    /// The value is built first, in the phases
    /// [`reconstruction`](Self::reconstruction) reports, and then the files
    /// are written in a phase `write` that counts them, so nothing is written
    /// when the build is cancelled. The `.sfmr` needs nothing from disk: the
    /// value carries every column the format stores.
    ///
    /// # Errors
    ///
    /// [`DemoSceneError::Directory`] when `dir` is not a directory, holds
    /// files or lies inside another workspace; the errors of
    /// [`reconstruction`](Self::reconstruction); [`DemoSceneError::Photograph`]
    /// when a JPEG cannot be written, [`DemoSceneError::Io`] when a directory
    /// or the workspace file cannot, and [`DemoSceneError::Save`] when the
    /// `.sfmr` cannot.
    pub fn write_workspace(
        &self,
        dir: &Path,
        progress: &Progress<'_>,
    ) -> Result<PathBuf, DemoSceneError> {
        check_workspace_dir(dir)?;
        let recon = self.reconstruction(progress)?;
        let phase = progress.phase("write");
        let total = self.encoded_photographs.len() as u64 + 2;
        std::fs::create_dir_all(dir.join("images")).map_err(DemoSceneError::Io)?;
        let mut written = 0u64;
        for (name, bytes) in self.image_names.iter().zip(&self.encoded_photographs) {
            phase.check_cancel()?;
            std::fs::write(dir.join(name), bytes)
                .map_err(|e| DemoSceneError::Photograph(::image::ImageError::IoError(e)))?;
            written += 1;
            phase.count(written, Some(total), "files");
        }
        phase.check_cancel()?;
        crate::workspace::write_sfmtool_workspace_marker(
            dir,
            &crate::workspace::SfmtoolWorkspaceOptions::default(),
        )
        .map_err(DemoSceneError::Io)?;
        written += 1;
        phase.count(written, Some(total), "files");
        phase.check_cancel()?;
        let path = dir.join(DEMO_SFMR_FILE_NAME);
        recon.save(&path, &phase).map_err(DemoSceneError::Save)?;
        phase.count(total, Some(total), "files");
        Ok(path)
    }
}

/// The name of the `.sfmr` [`DemoScene::write_workspace`] writes at the
/// workspace root.
const DEMO_SFMR_FILE_NAME: &str = "demo.sfmr";

/// `Ok` when `dir` does not exist or is an empty directory, and lies inside
/// no workspace, as `sfm ws init` refuses to nest one.
fn check_workspace_dir(dir: &Path) -> Result<(), DemoSceneError> {
    if dir.exists() {
        if !dir.is_dir() {
            return Err(DemoSceneError::Directory(format!(
                "{} is not a directory",
                dir.display()
            )));
        }
        let mut entries = std::fs::read_dir(dir).map_err(DemoSceneError::Io)?;
        if entries.next().is_some() {
            return Err(DemoSceneError::Directory(format!(
                "{} holds files; give an empty directory or one that does not exist",
                dir.display()
            )));
        }
    }
    if let Some(workspace) = crate::workspace::find_workspace_for_path(dir) {
        return Err(DemoSceneError::Directory(format!(
            "{} lies inside the workspace {}, and a workspace is not nested in another",
            dir.display(),
            workspace.display()
        )));
    }
    Ok(())
}

/// The scene's one camera: an undistorted pinhole of the options' size and
/// focal length, with the principal point at the image's centre.
fn demo_camera(options: &DemoSceneOptions) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: options.focal_length_px,
            focal_length_y: options.focal_length_px,
            principal_point_x: f64::from(options.image_width) / 2.0,
            principal_point_y: f64::from(options.image_height) / 2.0,
        },
        width: options.image_width,
        height: options.image_height,
        rolling_shutter: None,
    }
}

/// Why the demo scene could not be rendered, built or written.
#[derive(Debug)]
pub enum DemoSceneError {
    /// An option out of its range, naming it and the range.
    Options(String),
    /// `progress` was cancelled.
    Cancelled,
    /// The directory [`DemoScene::write_workspace`] was given holds files, is
    /// not a directory, or lies inside another workspace.
    Directory(String),
    /// A photograph could not be encoded, decoded, read or written.
    Photograph(::image::ImageError),
    /// The `.sfmr` could not be written.
    Save(sfmtool_sfmr_format::SfmrError),
    /// A directory or file of the workspace could not be created or written.
    Io(std::io::Error),
}

impl std::fmt::Display for DemoSceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Options(message) => write!(f, "invalid demo scene options: {message}"),
            Self::Cancelled => f.write_str("the demo scene was cancelled"),
            Self::Directory(message) => {
                write!(f, "cannot write the demo scene's workspace: {message}")
            }
            Self::Photograph(e) => write!(f, "a demo scene photograph failed: {e}"),
            Self::Save(e) => write!(f, "the demo scene's .sfmr could not be written: {e}"),
            Self::Io(e) => write!(f, "the demo scene's workspace could not be written: {e}"),
        }
    }
}

impl std::error::Error for DemoSceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Photograph(e) => Some(e),
            Self::Save(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<Cancelled> for DemoSceneError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

impl From<::image::ImageError> for DemoSceneError {
    fn from(e: ::image::ImageError) -> Self {
        Self::Photograph(e)
    }
}

#[cfg(test)]
mod tests;
