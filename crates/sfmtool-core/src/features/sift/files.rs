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
//! image. It runs several images at once in the way that extractor does (see
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
use sfmtool_sift_format::{FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata};

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

/// The zstd level `.sift` files are written at, the one `sfm sift` uses.
pub const SIFT_ZSTD_LEVEL: i32 = 5;

/// About how many bytes one image in flight holds per source pixel: the decoded
/// RGB frame plus the `f32` Gaussian pyramid of the doubled image the extraction
/// keeps through description. Rounded up, since guessing high only gives up a
/// few concurrent images while guessing low risks running out of memory. The
/// Python backend uses the same figure.
const BYTES_PER_SOURCE_PIXEL: u64 = 192;

/// The feature tool [`build_sift_files`] extracts with.
const FEATURE_TOOL: &str = "sfmtool";

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
    /// The options `sfm sift --extract` uses in the workspace at
    /// `workspace_dir`, read from its [`WORKSPACE_CONFIG_FILE`]: its
    /// `feature_options` and `feature_prefix_dir`, with the default
    /// concurrency.
    ///
    /// # Errors
    ///
    /// [`BuildSiftFilesError::Workspace`] when the file is missing, is not JSON
    /// or lacks `feature_tool`, `feature_options` or `feature_prefix_dir`, and
    /// [`BuildSiftFilesError::UnsupportedFeatureTool`] when the workspace's
    /// feature tool is not `sfmtool`, since this crate implements no other.
    pub fn from_workspace(workspace_dir: &Path) -> Result<Self, BuildSiftFilesError> {
        let path = workspace_dir.join(WORKSPACE_CONFIG_FILE);
        let workspace_error = |problem: String| BuildSiftFilesError::Workspace {
            path: path.clone(),
            problem,
        };
        let text = std::fs::read_to_string(&path).map_err(|e| {
            workspace_error(if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "the workspace is not initialized; initialize it with 'sfm ws init {}'",
                    workspace_dir.display()
                )
            } else {
                format!("could not be read: {e}")
            })
        })?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| workspace_error(format!("is not valid JSON: {e}")))?;
        let field = |name: &str| {
            config
                .get(name)
                .ok_or_else(|| workspace_error(format!("has no '{name}' field")))
        };
        let feature_tool = field("feature_tool")?
            .as_str()
            .ok_or_else(|| workspace_error("has a 'feature_tool' that is not a string".into()))?;
        if !feature_tool.eq_ignore_ascii_case(FEATURE_TOOL) {
            return Err(BuildSiftFilesError::UnsupportedFeatureTool {
                feature_tool: feature_tool.to_owned(),
            });
        }
        let feature_options = field("feature_options")?.clone();
        let feature_prefix_dir = field("feature_prefix_dir")?.as_str().ok_or_else(|| {
            workspace_error("has a 'feature_prefix_dir' that is not a string".into())
        })?;
        Ok(Self {
            feature_options,
            feature_prefix_dir: PathBuf::from(feature_prefix_dir),
            concurrent_images: None,
        })
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

/// Why [`build_sift_files`] or [`BuildSiftFilesOptions::from_workspace`]
/// stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildSiftFilesError {
    /// The workspace's [`WORKSPACE_CONFIG_FILE`] could not be used.
    Workspace {
        /// The file.
        path: PathBuf,
        /// What is wrong with it, as the end of a sentence about the file.
        problem: String,
    },
    /// The workspace extracts with a tool other than `sfmtool`.
    UnsupportedFeatureTool {
        /// The tool the workspace names.
        feature_tool: String,
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
            Self::Workspace { path, problem } => {
                write!(f, "workspace configuration {}: {problem}", path.display())
            }
            Self::UnsupportedFeatureTool { feature_tool } => write!(
                f,
                "the workspace extracts features with '{feature_tool}', and only the \
                 'sfmtool' feature tool can build .sift files here; run 'sfm sift --extract' \
                 for that tool"
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
/// `sfmtool`: take the options from
/// [`BuildSiftFilesOptions::from_workspace`] and the files are the ones that
/// command writes, at the same paths
/// ([`BuildSiftFilesOptions::sift_path_for_image`]), with the same content. Each
/// file records `feature_tool: "sfmtool"`, `feature_type: "sift"` and
/// `options.feature_options` as given; its metadata names the image by its file
/// name and records the image file's XXH128 and size; its features are sorted
/// by descending size; and its thumbnail is [`thumbnail_from_rgb`] of the
/// decode, which equals OpenCV's for any photograph at least
/// [`THUMBNAIL_SIZE`] on each side.
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
/// let options = BuildSiftFilesOptions::from_workspace(workspace)?;
/// let images = vec![workspace.join("images/0001.jpg"), workspace.join("images/0002.jpg")];
/// let report = build_sift_files(&images, &options, &Progress::none())?;
/// println!("wrote {} .sift files, {} were up to date", report.extracted, report.skipped);
/// # Ok::<(), sfmtool_core::features::sift::BuildSiftFilesError>(())
/// ```
///
/// # Errors
///
/// [`BuildSiftFilesError::FeatureOptions`] before anything is touched when
/// the options are invalid. [`BuildSiftFilesError::Image`] or
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

    // As `sfm sift`: make every output directory, then keep the images whose
    // `.sift` is missing or older than the image.
    let mut todo = Vec::new();
    for (index, (image, sift)) in image_paths.iter().zip(&sift_paths).enumerate() {
        if let Some(dir) = sift.parent() {
            std::fs::create_dir_all(dir).map_err(|e| BuildSiftFilesError::Write {
                path: sift.clone(),
                problem: format!("could not create its directory: {e}"),
            })?;
        }
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

    // The extraction and the write go on the rayon pool, which this thread
    // waits on rather than adding to; `rayon::scope` runs its body on a pool
    // thread. See the module docs.
    let features = rayon::scope(|_| {
        let gray = gray_from_rgb(width, height, image.data(), &params.image_to_gray);
        drop(image);
        extract_sift(&gray, params)
    });
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
            feature_tool: FEATURE_TOOL.to_owned(),
            feature_type: "sift".to_owned(),
            feature_options: options.feature_options.clone(),
        },
        metadata: SiftMetadata {
            version: 1,
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
