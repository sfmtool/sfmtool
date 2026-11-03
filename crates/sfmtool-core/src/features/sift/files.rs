// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Build the `.sift` files of a set of images with the `sfmtool` feature tool:
//! what `sfm sift --extract` does in a workspace whose `feature_tool` is
//! `sfmtool`, done in Rust.
//!
//! [`build_sift_files`] writes each image's file where the workspace puts it,
//! `<image dir>/<feature_prefix_dir>/<image name>.sift`, with the same
//! `feature_tool_metadata`, `metadata`, features and thumbnail the Python
//! extractor writes, and skips an image whose `.sift` is at least as new as the
//! image. The files are `.sift` version 2: each records the conversion its
//! options name as `feature_options.image_to_gray` and the image's two
//! readings ([`image_readings_from_rgb`](super::image_readings_from_rgb)),
//! taken from the decode the thumbnail is made from. It refuses version 1
//! settings and a folder that holds version 1 files, since the two versions
//! never share a folder (see `specs/workspace/workspace.md`).
//!
//! It runs several images at once in the way that extractor does (see
//! `specs/cli/image-feature/sift-command.md` § Extraction-orchestration
//! pipelining): `K` threads each decode an image, hash it and make its
//! thumbnail, then run the extraction and the write on the rayon pool, so the
//! threads compete with the pool's threads for no cores while they wait on it.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use sfmtool_sift_format::{
    FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata, SIFT_FORMAT_VERSION,
    SIFT_READINGS_VERSION,
};

use super::image_readings::image_readings_from_gray;
use super::{extract_sift, gray_from_rgb, FeatureOptionsError, SiftParams};
use crate::camera::image::{image_dimensions, ImageU8};
use crate::progress::{Cancelled, Progress};
use crate::reconstruction::thumbnail::thumbnail_from_rgb;
use crate::{progress_info, progress_status, progress_warn, THUMBNAIL_SIZE};

/// The file that marks a directory as a workspace and holds its feature
/// configuration.
pub const WORKSPACE_CONFIG_FILE: &str = ".sfm-workspace.json";

/// The environment variable that sets how many images [`build_sift_files`]
/// runs at once when [`BuildSiftFilesOptions::concurrent_images`] is `None`:
/// a whole number, raised to 1 if lower. The Python `sfmtool` backend of
/// `sfm sift` reads the same variable.
pub const CONCURRENT_IMAGES_ENV_VAR: &str = "SFMTOOL_SIFT_EXTRACT_WORKERS";

pub use sfmtool_sift_format::SIFT_ZSTD_LEVEL;

/// About how many bytes one image in flight holds per source pixel: the decoded
/// RGB frame plus the `f32` Gaussian pyramid of the doubled image the extraction
/// keeps through description. Rounded up, since guessing high only gives up a
/// few concurrent images while guessing low risks running out of memory. The
/// Python backend uses the same figure.
const BYTES_PER_SOURCE_PIXEL: u64 = 192;

/// The feature tool [`build_sift_files`] extracts with, as a workspace's
/// `feature_tool` names it.
pub const SFMTOOL_FEATURE_TOOL: &str = crate::workspace::SFMTOOL_TOOL;

/// The `feature_type` a workspace of the [`SFMTOOL_FEATURE_TOOL`] records, and
/// the start of the last component of its `feature_prefix_dir`.
pub const SFMTOOL_FEATURE_TYPE: &str = "sift-sfmtool";

/// How [`build_sift_files`] extracts and where it writes.
#[derive(Debug, Clone)]
pub struct BuildSiftFilesOptions {
    /// The `feature_options` object recorded for the `sfmtool` tool, read into
    /// [`SiftParams`] by [`SiftParams::from_feature_options`] and stored as it
    /// is in each file's `feature_tool_metadata`.
    pub feature_options: serde_json::Value,
    /// Where each image's `.sift` goes, relative to the image's own directory:
    /// the workspace's `feature_prefix_dir`, such as
    /// `features/sift-sfmtool-<hash>`. An absolute path puts every file in that
    /// one directory.
    pub feature_prefix_dir: PathBuf,
    /// How many images to decode and extract at once. `None` takes
    /// [`CONCURRENT_IMAGES_ENV_VAR`] when it is set, and otherwise the core
    /// count, capped so that the images in flight fit in half of physical
    /// memory at the first image's size.
    pub concurrent_images: Option<usize>,
}

impl BuildSiftFilesOptions {
    /// The options `sfm ws init --feature-tool sfmtool` writes into a new
    /// workspace when given no other choice: the `sfmtool` tool's default
    /// `feature_options`, with its `image_to_gray`, and the
    /// `feature_prefix_dir` `features/sift-sfmtool-<hash>` named from them by
    /// [`feature_prefix_dir`](crate::workspace::feature_prefix_dir), with the
    /// default concurrency.
    ///
    /// The options are the ones `get_default_sfmtool_feature_options` in
    /// `sfmtool.sift.extract_sfmtool` returns, and a test keeps the prefix
    /// equal to the one `sfm ws init` names from them, so a caller that
    /// switches a reconstruction's feature tool to `sfmtool` writes its files
    /// where a workspace initialized that way puts them. The two names need
    /// not agree for correctness: a folder name is read back from the record
    /// that holds it, never computed again.
    pub fn sfmtool_defaults() -> Self {
        let mut feature_options = serde_json::json!({
            "octave_layers": 3,
            "sigma": 1.6,
            "blur_radius_factor": 2.25,
            "input_sigma": 0.5,
            "double_image": true,
            "contrast_threshold": 0.0067,
            "edge_threshold": 10.0,
            "max_num_features": 8192,
            "orientation_bins": 36,
            "peak_ratio": 0.8,
            "descriptor_width": 4,
            "descriptor_bins": 8,
            "descriptor_magnification": 3.0,
            "descriptor_clamp": 0.2,
        });
        feature_options["image_to_gray"] =
            crate::workspace::tool_image_to_gray(SFMTOOL_FEATURE_TOOL)
                .expect("the sfmtool tool has a conversion")
                .to_json();
        let feature_prefix_dir = crate::workspace::feature_prefix_dir(
            SFMTOOL_FEATURE_TOOL,
            SFMTOOL_FEATURE_TYPE,
            &feature_options,
        );
        Self {
            feature_options,
            feature_prefix_dir: PathBuf::from(feature_prefix_dir),
            concurrent_images: None,
        }
    }

    /// The `.sift` path of the image at `image_path`:
    /// `<image dir>/<feature_prefix_dir>/<image name>.sift`.
    pub fn sift_path_for_image(&self, image_path: &Path) -> PathBuf {
        let name = image_path.file_name().unwrap_or_default().to_string_lossy();
        image_path
            .parent()
            .unwrap_or(Path::new(""))
            .join(&self.feature_prefix_dir)
            .join(format!("{name}.sift"))
    }
}

/// What [`build_sift_files`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildSiftFilesReport {
    /// Each image's `.sift` path, in the order of the images.
    pub sift_paths: Vec<PathBuf>,
    /// How many `.sift` files were written.
    pub extracted: usize,
    /// How many images were skipped because their `.sift` was up to date.
    pub skipped: usize,
}

/// Why [`build_sift_files`] stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildSiftFilesError {
    /// The feature options are version 1 settings: they carry no
    /// `image_to_gray`, and no `.sift` file is written from version 1
    /// settings.
    Version1Settings,
    /// A folder the files would go in holds a version 1 `.sift` file, and a
    /// version 2 file never goes beside one.
    Version1Files {
        /// The folder.
        folder: PathBuf,
    },
    /// The feature options do not describe a valid extraction.
    FeatureOptions(FeatureOptionsError),
    /// An image could not be read, decoded or hashed. When several fail, this
    /// is the first in the order of the images.
    Image {
        /// The image.
        path: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// A `.sift` file or its directory could not be written.
    Write {
        /// The `.sift` file.
        path: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// The caller asked the build to stop. The files written before it stopped
    /// stay, and a later build skips them.
    Cancelled,
}

impl fmt::Display for BuildSiftFilesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version1Settings => write!(
                f,
                "the feature options hold version 1 settings, with no 'image_to_gray', and no \
                 .sift file is written from version 1 settings; run 'sfm ws upgrade', which \
                 moves the workspace to version 2 and a new folder"
            ),
            Self::Version1Files { folder } => write!(
                f,
                "the folder {} holds version 1 .sift files, and version 2 files are never \
                 written beside them; run 'sfm ws upgrade', which moves the workspace to \
                 version 2 and a new folder",
                folder.display()
            ),
            Self::FeatureOptions(e) => write!(f, "invalid sfmtool feature options: {e}"),
            Self::Image { path, problem } => {
                write!(f, "failed to load image {}: {problem}", path.display())
            }
            Self::Write { path, problem } => {
                write!(f, "failed to write {}: {problem}", path.display())
            }
            Self::Cancelled => write!(
                f,
                "the build was cancelled; the .sift files written before it stopped are kept"
            ),
        }
    }
}

impl std::error::Error for BuildSiftFilesError {}

impl From<FeatureOptionsError> for BuildSiftFilesError {
    fn from(e: FeatureOptionsError) -> Self {
        Self::FeatureOptions(e)
    }
}

impl From<Cancelled> for BuildSiftFilesError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Extract SIFT features from each image of `image_paths` with the `sfmtool`
/// feature tool and write its `.sift` file, skipping an image whose `.sift`
/// file is already at least as new as the image.
///
/// This is `sfm sift --extract` for a workspace whose `feature_tool` is
/// `sfmtool`: given a workspace's `feature_options` and `feature_prefix_dir`,
/// the files are the ones that command writes, at the same paths
/// ([`BuildSiftFilesOptions::sift_path_for_image`]), with the same content. It
/// never reads the workspace file: the caller hands it the options and the
/// folder. Each file is `.sift` version 2 and records
/// `feature_tool: "sfmtool"`, `feature_type: "sift"` and
/// `options.feature_options` as given; its metadata names the image by its file
/// name, records the image file's XXH128 and size, and holds the image's
/// sharpness and gradient structure tensor ([`image_readings_from_rgb`](super::image_readings_from_rgb) of the
/// decode, through the options' `image_to_gray`); its features are sorted by
/// descending size; and its thumbnail is [`thumbnail_from_rgb`] of the same
/// decode, which equals OpenCV's for any photograph at least
/// [`THUMBNAIL_SIZE`] on each side.
///
/// When there is something to write, it first refuses version 1 settings
/// (options with no `image_to_gray`) and a folder that already holds a
/// version 1 `.sift` file. When every file is up to date it writes nothing and
/// refuses nothing.
///
/// Reports through `progress` a `"build_sift_files"` phase, a message saying
/// how many images were skipped, a status naming each image as it starts, and
/// a count of the images written out of those to write. The cancel flag is
/// read before each image starts and again before its extraction and its write.
///
/// ```no_run
/// use std::path::{Path, PathBuf};
/// use sfmtool_core::features::sift::{build_sift_files, BuildSiftFilesOptions};
/// use sfmtool_core::Progress;
///
/// let workspace = Path::new("my-workspace");
/// let options = BuildSiftFilesOptions::sfmtool_defaults();
/// let images = vec![workspace.join("images/0001.jpg"), workspace.join("images/0002.jpg")];
/// let report = build_sift_files(&images, &options, &Progress::none())?;
/// println!("wrote {} .sift files, {} were up to date", report.extracted, report.skipped);
/// # Ok::<(), sfmtool_core::features::sift::BuildSiftFilesError>(())
/// ```
///
/// # Errors
///
/// [`BuildSiftFilesError::FeatureOptions`] before anything is touched when
/// the options are invalid. [`BuildSiftFilesError::Version1Settings`] and
/// [`BuildSiftFilesError::Version1Files`] before anything is written, as
/// above. [`BuildSiftFilesError::Image`] when an image's readings are not
/// finite under the conversion, naming the image and the conversion, as well
/// as for an image that cannot be read. [`BuildSiftFilesError::Image`] or
/// [`BuildSiftFilesError::Write`] when an image or a file fails: no further
/// image is started, the ones in flight finish, and the error is the first
/// failing image's in input order. [`BuildSiftFilesError::Cancelled`] when the
/// caller cancels. In each case the files already written stay.
pub fn build_sift_files(
    image_paths: &[PathBuf],
    options: &BuildSiftFilesOptions,
    progress: &Progress,
) -> Result<BuildSiftFilesReport, BuildSiftFilesError> {
    let progress = progress.phase("build_sift_files");
    let params = SiftParams::from_feature_options(&options.feature_options)?;
    let sift_paths: Vec<PathBuf> = image_paths
        .iter()
        .map(|path| options.sift_path_for_image(path))
        .collect();

    // As `sfm sift`: keep the images whose `.sift` is missing or older than
    // the image.
    let mut todo = Vec::new();
    for (index, (image, sift)) in image_paths.iter().zip(&sift_paths).enumerate() {
        let image_modified = std::fs::metadata(image)
            .and_then(|m| m.modified())
            .map_err(|e| BuildSiftFilesError::Image {
                path: image.clone(),
                problem: e.to_string(),
            })?;
        let up_to_date = std::fs::metadata(sift)
            .and_then(|m| m.modified())
            .is_ok_and(|sift_modified| sift_modified >= image_modified);
        if !up_to_date {
            todo.push(index);
        }
    }
    let skipped = image_paths.len() - todo.len();
    if skipped > 0 {
        progress_info!(
            progress,
            "{skipped} of {} images already have an up-to-date .sift file",
            image_paths.len()
        );
    }
    let report = |extracted| BuildSiftFilesReport {
        sift_paths: sift_paths.clone(),
        extracted,
        skipped,
    };
    if todo.is_empty() {
        return Ok(report(0));
    }
    // Something will be written, so the two versions must not mix: no file
    // from version 1 settings, and none beside a version 1 file.
    if !crate::workspace::has_image_to_gray(&options.feature_options) {
        return Err(BuildSiftFilesError::Version1Settings);
    }
    let mut folders: Vec<&Path> = todo
        .iter()
        .filter_map(|&i| sift_paths[i].parent())
        .collect();
    folders.sort();
    folders.dedup();
    for folder in &folders {
        if folder_holds_version_1_files(folder) {
            return Err(BuildSiftFilesError::Version1Files {
                folder: folder.to_path_buf(),
            });
        }
    }
    for folder in &folders {
        std::fs::create_dir_all(folder).map_err(|e| BuildSiftFilesError::Write {
            path: folder.to_path_buf(),
            problem: format!("could not create the folder: {e}"),
        })?;
    }
    progress.check_cancel()?;

    let workers = options
        .concurrent_images
        .unwrap_or_else(|| default_concurrent_images(&image_paths[todo[0]], &progress))
        .clamp(1, todo.len());
    let total = todo.len() as u64;
    progress.count(0, Some(total), "image");

    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let failures: Mutex<Vec<(usize, BuildSiftFilesError)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                if failed.load(Ordering::Relaxed) || progress.is_cancelled() {
                    return;
                }
                let slot = next.fetch_add(1, Ordering::Relaxed);
                let Some(&index) = todo.get(slot) else {
                    return;
                };
                let image = &image_paths[index];
                progress_status!(
                    progress,
                    "Extracting SIFT features from {}",
                    image.file_name().unwrap_or_default().to_string_lossy()
                );
                match build_one(image, &sift_paths[index], options, &params, &progress) {
                    Ok(()) => {
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        progress.count(n, Some(total), "image");
                    }
                    Err(BuildSiftFilesError::Cancelled) => return,
                    Err(e) => {
                        failed.store(true, Ordering::Relaxed);
                        failures.lock().unwrap().push((index, e));
                        return;
                    }
                }
            });
        }
    });

    let failures = failures.into_inner().unwrap();
    if let Some((_, error)) = failures.into_iter().min_by_key(|(index, _)| *index) {
        return Err(error);
    }
    progress.check_cancel()?;
    Ok(report(done.into_inner() as usize))
}

/// Decode, hash, extract and write one image.
fn build_one(
    image_path: &Path,
    sift_path: &Path,
    options: &BuildSiftFilesOptions,
    params: &SiftParams,
    progress: &Progress,
) -> Result<(), BuildSiftFilesError> {
    let image_error = |problem: String| BuildSiftFilesError::Image {
        path: image_path.to_path_buf(),
        problem,
    };
    let image = ImageU8::read_rgb(image_path).map_err(|e| image_error(e.to_string()))?;
    let (width, height) = (image.width(), image.height());
    let thumbnail = thumbnail_from_rgb(image.data(), width as usize, height as usize);
    let (image_file_xxh128, image_file_size) =
        xxh128_of_file(image_path).map_err(|e| image_error(e.to_string()))?;
    progress.check_cancel()?;

    // The readings, the extraction and the write go on the rayon pool, which
    // this thread waits on rather than adding to; `rayon::scope` runs its body
    // on a pool thread. See the module docs.
    let (readings, features) = rayon::scope(|_| {
        // One grey image for the detector and the readings both.
        let gray = gray_from_rgb(width, height, image.data(), &params.image_to_gray);
        drop(image);
        let readings = image_readings_from_gray(&gray, &params.image_to_gray)
            .map_err(|e| image_error(e.to_string()))?;
        Ok::<_, BuildSiftFilesError>((readings, extract_sift(&gray, params)))
    })?;
    progress.check_cancel()?;

    let count = features.keypoints.len();
    let mut positions = Vec::with_capacity(count * 2);
    let mut shapes = Vec::with_capacity(count * 4);
    for kp in &features.keypoints {
        positions.extend_from_slice(&[kp.x, kp.y]);
        let [[a11, a12], [a21, a22]] = kp.affine_shape;
        shapes.extend_from_slice(&[a11, a12, a21, a22]);
    }
    let descriptors: Vec<u8> = features.descriptors.rows().concat();
    let shape_error = |e: ndarray::ShapeError| BuildSiftFilesError::Write {
        path: sift_path.to_path_buf(),
        problem: e.to_string(),
    };
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: SFMTOOL_FEATURE_TOOL.to_owned(),
            feature_type: "sift".to_owned(),
            feature_options: options.feature_options.clone(),
        },
        metadata: SiftMetadata {
            version: SIFT_FORMAT_VERSION,
            image_name: image_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            image_file_xxh128,
            image_file_size,
            image_width: width,
            image_height: height,
            feature_count: count as u32,
            sharpness: Some(f64::from(readings.sharpness)),
            gradient_structure_tensor_xx_yy_xy: Some(
                readings.gradient_structure_tensor_xx_yy_xy.map(f64::from),
            ),
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::from_shape_vec((count, 2), positions).map_err(shape_error)?,
        affine_shapes: Array3::from_shape_vec((count, 2, 2), shapes).map_err(shape_error)?,
        descriptors: Array2::from_shape_vec((count, 128), descriptors).map_err(shape_error)?,
        thumbnail_y_x_rgb: Array3::from_shape_vec((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), thumbnail)
            .map_err(shape_error)?,
    };
    rayon::scope(|_| sfmtool_sift_format::write_sift(sift_path, &data, SIFT_ZSTD_LEVEL)).map_err(
        |e| BuildSiftFilesError::Write {
            path: sift_path.to_path_buf(),
            problem: e.to_string(),
        },
    )
}

/// Whether `folder` holds a `.sift` file older than version 2. A file whose
/// metadata cannot be read says nothing about the folder's version and is
/// passed over; a folder that is not there holds nothing.
fn folder_holds_version_1_files(folder: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.extension().is_some_and(|e| e == "sift")
            && sfmtool_sift_format::read_sift_metadata(&path)
                .is_ok_and(|(_, metadata, _)| metadata.version < SIFT_READINGS_VERSION)
    })
}

/// The XXH128 of a file's bytes as 32 lowercase hex digits, and its size.
fn xxh128_of_file(path: &Path) -> std::io::Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        size += n as u64;
    }
    Ok((format!("{:032x}", hasher.digest128()), size))
}

/// How many images to run at once when the caller does not say: the
/// [`CONCURRENT_IMAGES_ENV_VAR`] value when it is a whole number, otherwise the
/// core count capped by memory at the size of `first_image`.
fn default_concurrent_images(first_image: &Path, progress: &Progress) -> usize {
    // An empty value counts as unset, as it does for the Python backend.
    if let Some(value) = std::env::var(CONCURRENT_IMAGES_ENV_VAR)
        .ok()
        .filter(|v| !v.is_empty())
    {
        match value.trim().parse::<i64>() {
            Ok(n) => return n.max(1) as usize,
            Err(_) => progress_warn!(
                progress,
                "Ignoring non-integer {CONCURRENT_IMAGES_ENV_VAR}={value:?}; using the default \
                 concurrency."
            ),
        }
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let pixels = image_dimensions(first_image)
        .ok()
        .map(|(w, h)| u64::from(w) * u64::from(h));
    let budget = crate::camera::photograph_cache::physical_memory_bytes()
        .unwrap_or(4 << 30)
        .div_euclid(2)
        .max(1);
    concurrent_images_for(cores, pixels, budget)
}

/// [`default_concurrent_images`] without the environment: `cores`, capped so
/// that that many images of `pixels` source pixels fit in `budget` bytes, or
/// at most 4 when the size is unknown. At least 1.
fn concurrent_images_for(cores: usize, pixels: Option<u64>, budget: u64) -> usize {
    match pixels.filter(|&p| p > 0) {
        None => cores.clamp(1, 4),
        Some(pixels) => {
            let by_memory = (budget / (pixels * BYTES_PER_SOURCE_PIXEL)).max(1);
            cores.min(by_memory as usize).max(1)
        }
    }
}

#[cfg(test)]
mod tests;
