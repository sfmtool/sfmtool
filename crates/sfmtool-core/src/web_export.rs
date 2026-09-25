// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A reconstruction as static web files: the data half of `sfm web-export`.
//!
//! See `specs/cli/visualization/web-export-command.md`. [`build_web_export`]
//! turns a reconstruction into the `scene.json` text and the JPEG atlas pages a
//! browser viewer draws, with every camera-model computation done here: each
//! camera's lens is sampled into a grid of far-surface points, so the browser
//! draws triangles, lines and points and never evaluates a lens.
//! [`write_web_export`] writes those files into a directory. The viewer page and
//! its script are not made here; the Python command copies them in beside the
//! data.
//!
//! ```no_run
//! use sfmtool_core::progress::Progress;
//! use sfmtool_core::web_export::{write_web_export, WebExportOptions};
//! use sfmtool_core::SfmrReconstruction;
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let recon = SfmrReconstruction::load("scene.sfmr".as_ref(), &Progress::none())?;
//! let options = WebExportOptions {
//!     max_points: Some(50_000),
//!     ..WebExportOptions::default()
//! };
//! let report = write_web_export(&recon, "site".as_ref(), &options, &Progress::none())?;
//! for (name, bytes) in &report.files {
//!     println!("{name}: {bytes} bytes");
//! }
//! # Ok(())
//! # }
//! ```

pub mod atlas;

use std::path::Path;

use base64::Engine as _;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use serde_json::{json, Map, Value};

use crate::analysis::scene_scale::{
    compute_auto_point_size, compute_camera_nn_scale, compute_scene_bounds, length_scale,
};
use crate::camera::frustum::compute_distorted_frustum_grid;
use crate::patch::display_bitmaps::render_display_patch_bitmaps;
use crate::progress::{Cancelled, Progress};
use crate::reconstruction::thumbnail::{display_thumbnail_row, resize_area, ThumbnailSource};
use crate::{progress_note, SfmrReconstruction, THUMBNAIL_SIZE};

use atlas::{AtlasLayout, RgbPage};

/// The `format` number `scene.json` carries; a viewer reads the layout it
/// names.
pub const FORMAT_VERSION: u32 = 1;

/// The largest atlas page edge, in texels: the texture size every current
/// phone's WebGL2 supports.
pub const MAX_PAGE_SIZE: usize = 4096;

/// Decoded atlas bytes above which the report carries a warning: a phone may
/// refuse that much texture memory or reload the tab.
pub const DECODED_ATLAS_WARNING_BYTES: u64 = 256 << 20;

/// Vertices along each edge of a distorted or fisheye camera's far-surface
/// grid. A pinhole camera without distortion takes 2, its four corners.
pub const LENS_GRID_SIZE: usize = 9;

/// Frustum length per length scale, SfM Explorer's default
/// `frustum_size_multiplier`.
pub const FRUSTUM_LENGTH_FACTOR: f64 = 0.5;

/// Vertical field of view of the framing view, in degrees.
const FRAMING_FOV_DEG: f64 = 50.0;

/// What to put in the export.
#[derive(Debug, Clone)]
pub struct WebExportOptions {
    /// Write the patch atlas. Off, every point is drawn as a splat.
    pub patches: bool,
    /// Write the thumbnail atlas. Off, frustums are wireframes only.
    pub thumbnails: bool,
    /// Resample patch tiles to this edge, in texels; `None` keeps the bitmaps'
    /// own size.
    pub patch_size: Option<usize>,
    /// JPEG quality of both atlases, 1 to 100.
    pub jpeg_quality: u8,
    /// Keep only this many points, those with the most observations.
    pub max_points: Option<usize>,
    /// Name of the image whose camera the page opens looking through, instead
    /// of framing the scene.
    pub start_image: Option<String>,
    /// The file name `scene.json` names as its source.
    pub source_name: Option<String>,
    /// What `scene.json` names as having written it.
    pub generator: String,
    /// Largest atlas page edge, in texels. [`MAX_PAGE_SIZE`] unless a caller
    /// has reason to pack smaller pages.
    pub max_page_size: usize,
}

impl Default for WebExportOptions {
    fn default() -> Self {
        Self {
            patches: true,
            thumbnails: true,
            patch_size: None,
            jpeg_quality: 85,
            max_points: None,
            start_image: None,
            source_name: None,
            generator: format!("sfmtool-core {}", env!("CARGO_PKG_VERSION")),
            max_page_size: MAX_PAGE_SIZE,
        }
    }
}

/// Where the thumbnail atlas rows came from.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ThumbnailCounts {
    /// Rows from the file's own thumbnail column.
    pub file: usize,
    /// Rows from each image's verified `.sift`.
    pub sift: usize,
    /// Rows decoded and resized from the photograph.
    pub photographs: usize,
    /// Grey rows for images neither could supply.
    pub placeholders: usize,
}

/// What an export holds, and anything the caller should be told about it.
#[derive(Debug, Clone, Default)]
pub struct WebExportReport {
    /// Points written.
    pub points: usize,
    /// Of those, points at infinity.
    pub points_at_infinity: usize,
    /// Points `max_points` left out.
    pub points_left_out: usize,
    /// Points drawn as a textured patch.
    pub patches: usize,
    /// Patch tile edge written, in texels, without the border; 0 without an
    /// atlas.
    pub patch_size: usize,
    /// Whether the patch bitmaps were rendered from the photographs because
    /// the file carries frames and no bitmaps.
    pub patch_bitmaps_rendered: bool,
    /// Cameras written.
    pub cameras: usize,
    /// Where the thumbnail rows came from; all zero without a thumbnail atlas.
    pub thumbnails: ThumbnailCounts,
    /// Bytes the atlas pages take once decoded to RGBA, as a GPU holds them.
    pub decoded_atlas_bytes: u64,
    /// Every file written or to be written, by name, with its size in bytes.
    pub files: Vec<(String, u64)>,
    /// Things the caller should pass on, in plain sentences.
    pub warnings: Vec<String>,
}

/// One file of an export, held in memory.
#[derive(Debug, Clone)]
pub struct WebExportFile {
    /// File name within the output directory.
    pub name: String,
    /// Its bytes.
    pub bytes: Vec<u8>,
}

/// A whole export in memory: `scene.json` and the atlas pages.
#[derive(Debug, Clone)]
pub struct WebExport {
    /// The files, `scene.json` first.
    pub files: Vec<WebExportFile>,
    /// What the files hold.
    pub report: WebExportReport,
}

/// Why an export could not be made.
#[derive(Debug)]
pub enum WebExportError {
    /// `start_image` names no image of the reconstruction.
    UnknownStartImage(String),
    /// An option is out of range.
    InvalidOption(String),
    /// An atlas page could not be encoded.
    Encode(image::ImageError),
    /// A file could not be written.
    Io(std::io::Error),
    /// The progress was cancelled.
    Cancelled,
}

impl std::fmt::Display for WebExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownStartImage(name) => {
                write!(
                    f,
                    "--start-image {name:?} names no image of the reconstruction"
                )
            }
            Self::InvalidOption(message) => f.write_str(message),
            Self::Encode(e) => write!(f, "could not encode an atlas page: {e}"),
            Self::Io(e) => write!(f, "could not write the export: {e}"),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for WebExportError {}

impl From<Cancelled> for WebExportError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Build the export of `recon` in memory.
///
/// Thumbnails come from the file's column when it has one, and otherwise from
/// each image's `.sift` or photograph through
/// [`display_thumbnail_row`]; the atlas is left out when no image yields one.
/// Patch bitmaps come from the file when it has them, and a file with patch
/// frames and inline keypoints but no bitmaps has them rendered from the
/// photographs by [`render_display_patch_bitmaps`]. A point with no frame, no
/// bitmap, an all-zero bitmap or a place at infinity is drawn as a splat.
///
/// # Errors
///
/// [`WebExportError::UnknownStartImage`] and
/// [`WebExportError::InvalidOption`] for options that do not fit `recon`;
/// [`WebExportError::Encode`] when a page cannot be encoded;
/// [`WebExportError::Cancelled`] when `progress` is cancelled.
pub fn build_web_export(
    recon: &SfmrReconstruction,
    options: &WebExportOptions,
    progress: &Progress<'_>,
) -> Result<WebExport, WebExportError> {
    if options.jpeg_quality == 0 || options.jpeg_quality > 100 {
        return Err(WebExportError::InvalidOption(format!(
            "JPEG quality must be 1 to 100, got {}",
            options.jpeg_quality
        )));
    }
    if options.patch_size == Some(0) {
        return Err(WebExportError::InvalidOption(
            "the patch size must be at least 1".into(),
        ));
    }
    let images = &recon.image_table.images;
    let start_index = match &options.start_image {
        Some(name) => Some(
            images
                .iter()
                .position(|image| &image.name == name)
                .ok_or_else(|| WebExportError::UnknownStartImage(name.clone()))?,
        ),
        None => None,
    };
    let [thumb_progress, bitmap_progress, rest] = progress.split([1.0, 4.0, 1.0]);
    let mut report = WebExportReport::default();

    // The points kept, in their order in the file.
    let points = &recon.point_set.points;
    let kept = kept_points(&recon.point_set.observation_counts, options.max_points);
    report.points = kept.len();
    report.points_left_out = points.len() - kept.len();
    report.points_at_infinity = kept.iter().filter(|&&p| points[p].is_at_infinity()).count();

    // Scale, from the whole cloud, as the viewer measures it.
    let (centre, radius) = compute_scene_bounds(points);
    let point_size = compute_auto_point_size(points);
    let camera_nn = compute_camera_nn_scale(images);
    let scene_length = length_scale(point_size, camera_nn) as f64;
    let frustum_length = scene_length * FRUSTUM_LENGTH_FACTOR;

    // Patch tiles.
    let rendered_bitmaps;
    let bitmaps = if !options.patches || recon.point_set.patch_u_halfvec_xyz.is_none() {
        None
    } else if let Some(bitmaps) = &recon.point_set.patch_bitmaps_y_x_rgba {
        Some(bitmaps.as_ref())
    } else {
        let mut phase = bitmap_progress.phase("patch bitmaps");
        rendered_bitmaps = render_display_patch_bitmaps(recon, &phase)?;
        report.patch_bitmaps_rendered = rendered_bitmaps.is_some();
        if rendered_bitmaps.is_none() {
            progress_note!(phase, "no photographs to render them from; drawing splats");
        }
        rendered_bitmaps.as_ref()
    };
    let patch_source: Vec<usize> = match bitmaps {
        Some(bitmaps) => kept
            .iter()
            .copied()
            .filter(|&p| has_patch(recon, bitmaps, p))
            .collect(),
        None => Vec::new(),
    };
    let bitmap_size = bitmaps.map_or(0, |b| b.shape()[1]);
    let patch_size = options.patch_size.unwrap_or(bitmap_size);
    let patch_layout = AtlasLayout::new(patch_source.len(), patch_size, options.max_page_size);
    if patch_layout.is_none() && !patch_source.is_empty() {
        return Err(WebExportError::InvalidOption(format!(
            "a {patch_size}-texel patch tile does not fit a {}-texel page",
            options.max_page_size
        )));
    }

    // Thumbnail rows.
    let thumbnail_rows = if options.thumbnails && !images.is_empty() {
        let mut phase = thumb_progress.phase("thumbnails");
        let rows = thumbnail_rows(recon, &mut report.thumbnails, &phase)?;
        progress_note!(
            phase,
            "{} from the file, {} from .sift files, {} from photographs, {} placeholders",
            report.thumbnails.file,
            report.thumbnails.sift,
            report.thumbnails.photographs,
            report.thumbnails.placeholders
        );
        rows
    } else {
        None
    };
    let thumb_layout = thumbnail_rows
        .as_ref()
        .and_then(|rows| AtlasLayout::new(rows.len(), THUMBNAIL_SIZE, options.max_page_size));

    let mut phase = rest.phase("atlases");
    let mut files = vec![WebExportFile {
        name: "scene.json".into(),
        bytes: Vec::new(),
    }];

    // The patch atlas pages.
    let mut patch_cell = vec![u32::MAX; kept.len()];
    let patches_json = match (patch_layout, bitmaps) {
        (Some(layout), Some(bitmaps)) => {
            let slot_of_kept: std::collections::HashMap<usize, usize> =
                kept.iter().enumerate().map(|(i, &p)| (p, i)).collect();
            for (k, &p) in patch_source.iter().enumerate() {
                let cell = layout.place(k);
                patch_cell[slot_of_kept[&p]] = ((cell.page as u32) << 24) | cell.cell as u32;
            }
            let pages = layout.pack(|k| {
                // The row's texels in (y, x, channel) order, alpha left out:
                // it is cross-view confidence, not transparency.
                let rgb: Vec<u8> = bitmaps
                    .index_axis(ndarray::Axis(0), patch_source[k])
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 4 != 3)
                    .map(|(_, &b)| b)
                    .collect();
                if patch_size == bitmap_size {
                    rgb
                } else {
                    resize_area(&rgb, bitmap_size, bitmap_size, 3, patch_size, patch_size)
                }
            });
            report.patches = patch_source.len();
            report.patch_size = patch_size;
            Some(encode_atlas(
                "patches",
                &layout,
                &pages,
                options,
                &mut files,
                &mut report,
            )?)
        }
        _ => None,
    };
    phase.check_cancel()?;

    // The thumbnail atlas pages.
    let thumbs_json = match (thumb_layout, &thumbnail_rows) {
        (Some(layout), Some(rows)) => {
            let pages = layout.pack(|k| rows[k].clone());
            Some(encode_atlas(
                "thumbs",
                &layout,
                &pages,
                options,
                &mut files,
                &mut report,
            )?)
        }
        _ => None,
    };
    progress_note!(phase, "{} pages", files.len() - 1);

    // The point arrays, positions relative to the centre.
    let mut block = ByteBlock::default();
    let positions: Vec<f32> = kept
        .iter()
        .flat_map(|&p| relative_position(&points[p].position, points[p].w, &centre))
        .collect();
    block.push_f32("position", 3, &positions);
    if patches_json.is_some() {
        let u = recon
            .point_set
            .patch_u_halfvec_xyz
            .as_ref()
            .expect("frames");
        let v = recon
            .point_set
            .patch_v_halfvec_xyz
            .as_ref()
            .expect("frames");
        let half = |column: &ndarray::Array2<f32>| -> Vec<f32> {
            kept.iter()
                .zip(&patch_cell)
                .flat_map(|(&p, &cell)| {
                    if cell == u32::MAX {
                        [0.0; 3]
                    } else {
                        [column[[p, 0]], column[[p, 1]], column[[p, 2]]]
                    }
                })
                .collect()
        };
        block.push_f32("patch_u", 3, &half(u));
        block.push_f32("patch_v", 3, &half(v));
        block.push_u32("patch_cell", &patch_cell);
    }
    let rgbw: Vec<u8> = kept
        .iter()
        .flat_map(|&p| {
            let c = points[p].color;
            [c[0], c[1], c[2], u8::from(!points[p].is_at_infinity())]
        })
        .collect();
    block.push_u8("color_w", 4, &rgbw);
    let observations: Vec<u16> = kept
        .iter()
        .map(|&p| recon.point_set.observation_counts[p].min(u16::MAX as u32) as u16)
        .collect();
    block.push_u16("observations", &observations);
    if report.points_left_out > 0 {
        let indexes: Vec<u32> = kept.iter().map(|&p| p as u32).collect();
        block.push_u32("source_index", &indexes);
    }

    // The cameras, and their far-surface grids in a block of their own.
    let mut grids = ByteBlock::default();
    let mut grid_vertices: Vec<f32> = Vec::new();
    let mut cameras = Vec::with_capacity(images.len());
    let per_page = thumb_layout.map(|layout| layout.per_page());
    for (i, image) in images.iter().enumerate() {
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let c = image.camera_center();
        let rel = [c.x - centre.x, c.y - centre.y, c.z - centre.z];
        let grid = lens_grid(image, camera, &centre, frustum_length);
        let first = grid_vertices.len() / 3;
        grid_vertices.extend(grid.positions.iter().map(|&x| x as f32));
        let q = image.quaternion_wxyz.inverse();
        let mut entry = Map::new();
        entry.insert("name".into(), json!(image.name));
        entry.insert("center".into(), json!(rel));
        entry.insert("rotation_wxyz".into(), json!([q.w, q.i, q.j, q.k]));
        entry.insert("grid".into(), json!([first, grid.size]));
        if let Some(per_page) = per_page {
            entry.insert("thumb".into(), json!([i / per_page, i % per_page]));
        }
        cameras.push(Value::Object(entry));
    }
    grids.push_f32("vertex", 3, &grid_vertices);
    report.cameras = images.len();

    let view = match start_index {
        Some(i) => start_view(recon, i, &centre, radius),
        None => framing_view(recon, &centre, radius),
    };

    let mut scene = Map::new();
    scene.insert("format".into(), json!(FORMAT_VERSION));
    scene.insert("generator".into(), json!(options.generator));
    scene.insert(
        "source".into(),
        json!({
            "name": options.source_name,
            "content_xxh128": (!recon.content_hash.content_xxh128.is_empty())
                .then_some(&recon.content_hash.content_xxh128),
        }),
    );
    scene.insert("center".into(), json!([centre.x, centre.y, centre.z]));
    scene.insert("radius".into(), json!(radius));
    scene.insert("point_size".into(), json!(point_size));
    scene.insert("length_scale".into(), json!(scene_length));
    scene.insert("frustum_length".into(), json!(frustum_length));
    scene.insert("view".into(), view);
    scene.insert(
        "points".into(),
        json!({
            "count": report.points,
            "at_infinity": report.points_at_infinity,
            "left_out": report.points_left_out,
            "with_patch": report.patches,
            "arrays": block.arrays,
        }),
    );
    scene.insert(
        "points_b64".into(),
        json!(base64::engine::general_purpose::STANDARD.encode(&block.bytes)),
    );
    scene.insert("cameras".into(), Value::Array(cameras));
    scene.insert("frustum_arrays".into(), Value::Object(grids.arrays));
    scene.insert(
        "frustums_b64".into(),
        json!(base64::engine::general_purpose::STANDARD.encode(&grids.bytes)),
    );
    scene.insert(
        "atlases".into(),
        json!({ "patches": patches_json, "thumbnails": thumbs_json }),
    );
    let scene_bytes = serde_json::to_vec(&Value::Object(scene)).expect("JSON of plain values");
    report
        .files
        .insert(0, ("scene.json".into(), scene_bytes.len() as u64));
    files[0].bytes = scene_bytes;

    if report.decoded_atlas_bytes > DECODED_ATLAS_WARNING_BYTES {
        report.warnings.push(format!(
            "the atlases take {} MB of GPU memory once decoded, more than the {} MB a phone \
             can be relied on to hold; --patch-size 12 quarters the patch atlas, and \
             --max-points leaves points out",
            report.decoded_atlas_bytes >> 20,
            DECODED_ATLAS_WARNING_BYTES >> 20
        ));
    }
    Ok(WebExport { files, report })
}

/// Build the export of `recon` and write its files into `out_dir`, creating
/// the directory when it is missing. Files already there under other names are
/// left alone; the caller decides what an existing directory may hold.
///
/// # Errors
///
/// As [`build_web_export`], and [`WebExportError::Io`] when a file cannot be
/// written.
pub fn write_web_export(
    recon: &SfmrReconstruction,
    out_dir: &Path,
    options: &WebExportOptions,
    progress: &Progress<'_>,
) -> Result<WebExportReport, WebExportError> {
    let export = build_web_export(recon, options, progress)?;
    std::fs::create_dir_all(out_dir).map_err(WebExportError::Io)?;
    for file in &export.files {
        std::fs::write(out_dir.join(&file.name), &file.bytes).map_err(WebExportError::Io)?;
    }
    Ok(export.report)
}

/// The indexes of the points to write, ascending: every point, or the
/// `max_points` with the most observations, ties going to the lower index.
fn kept_points(observation_counts: &[u32], max_points: Option<usize>) -> Vec<usize> {
    let count = observation_counts.len();
    match max_points {
        Some(max) if max < count => {
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by_key(|&p| (std::cmp::Reverse(observation_counts[p]), p));
            order.truncate(max);
            order.sort_unstable();
            order
        }
        _ => (0..count).collect(),
    }
}

/// Whether point `p` is drawn as a textured patch: a finite point with a
/// nonzero frame and a bitmap row that is not all zero.
fn has_patch(recon: &SfmrReconstruction, bitmaps: &ndarray::Array4<u8>, p: usize) -> bool {
    let set = &recon.point_set;
    let (Some(u), Some(v)) = (&set.patch_u_halfvec_xyz, &set.patch_v_halfvec_xyz) else {
        return false;
    };
    if p >= bitmaps.shape()[0] || set.points[p].is_at_infinity() {
        return false;
    }
    let nonzero = |column: &ndarray::Array2<f32>| column.row(p).iter().any(|&x| x != 0.0);
    nonzero(u)
        && nonzero(v)
        && bitmaps
            .index_axis(ndarray::Axis(0), p)
            .iter()
            .any(|&b| b != 0)
}

/// A point's position as the viewer stores it: a place relative to `centre`,
/// or a direction as it is.
fn relative_position(position: &Point3<f64>, w: f64, centre: &Point3<f64>) -> [f32; 3] {
    if w == 0.0 {
        [position.x as f32, position.y as f32, position.z as f32]
    } else {
        [
            (position.x - centre.x) as f32,
            (position.y - centre.y) as f32,
            (position.z - centre.z) as f32,
        ]
    }
}

/// One RGB thumbnail row per image, `THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3`
/// bytes each: the file's column, or the display rows. `None` when every row
/// would be a placeholder.
fn thumbnail_rows(
    recon: &SfmrReconstruction,
    counts: &mut ThumbnailCounts,
    progress: &Progress<'_>,
) -> Result<Option<Vec<Vec<u8>>>, Cancelled> {
    let total = recon.image_table.images.len();
    if let Some(column) = &recon.image_table.thumbnails_y_x_rgb {
        counts.file = total;
        let rows = column
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        return Ok(Some(rows));
    }
    let landed = std::sync::atomic::AtomicUsize::new(0);
    let built: Vec<(Vec<u8>, ThumbnailSource)> = (0..total)
        .into_par_iter()
        .map(|i| {
            if progress.is_cancelled() {
                return (Vec::new(), ThumbnailSource::Placeholder);
            }
            let row = display_thumbnail_row(recon, i);
            let n = landed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress.count(n as u64, Some(total as u64), "images");
            row
        })
        .collect();
    progress.check_cancel()?;
    for (_, source) in &built {
        match source {
            ThumbnailSource::Sift => counts.sift += 1,
            ThumbnailSource::Photograph => counts.photographs += 1,
            ThumbnailSource::Placeholder => counts.placeholders += 1,
        }
    }
    if counts.placeholders == total {
        *counts = ThumbnailCounts::default();
        return Ok(None);
    }
    Ok(Some(built.into_iter().map(|(row, _)| row).collect()))
}

/// Encode an atlas's pages as `<stem>-<page>.jpg` files, record them in the
/// report, and return the atlas's `scene.json` entry.
fn encode_atlas(
    stem: &str,
    layout: &AtlasLayout,
    pages: &[RgbPage],
    options: &WebExportOptions,
    files: &mut Vec<WebExportFile>,
    report: &mut WebExportReport,
) -> Result<Value, WebExportError> {
    let encoded: Vec<Result<Vec<u8>, image::ImageError>> = pages
        .par_iter()
        .map(|page| page.to_jpeg(options.jpeg_quality))
        .collect();
    let mut page_entries = Vec::with_capacity(pages.len());
    for (index, (page, bytes)) in pages.iter().zip(encoded).enumerate() {
        let bytes = bytes.map_err(WebExportError::Encode)?;
        let name = format!("{stem}-{index}.jpg");
        report.files.push((name.clone(), bytes.len() as u64));
        report.decoded_atlas_bytes += (page.width * page.height * 4) as u64;
        page_entries.push(json!({ "file": name, "width": page.width, "height": page.height }));
        files.push(WebExportFile { name, bytes });
    }
    Ok(json!({
        "pages": page_entries,
        "size": layout.size,
        "tile": layout.tile,
        "border": 1,
        "cols": layout.cols,
        "per_page": layout.per_page(),
    }))
}

/// A camera's far-surface grid, relative to the scene centre.
struct LensGrid {
    /// Vertices along each edge.
    size: usize,
    /// `size * size` vertices, row major from the image's top-left pixel
    /// corner, three coordinates each.
    positions: Vec<f64>,
}

/// Sample `camera`'s lens into its far-surface grid at `frustum_length`, in
/// world coordinates relative to `centre`.
///
/// A pinhole camera without distortion is its four far-plane corners. Any
/// other is a [`LENS_GRID_SIZE`] square grid of its pixels' rays, on the far
/// plane for a perspective lens and on the sphere of radius `frustum_length`
/// for a fisheye, as SfM Explorer draws them
/// ([`compute_distorted_frustum_grid`]).
fn lens_grid(
    image: &crate::SfmrImage,
    camera: &crate::CameraIntrinsics,
    centre: &Point3<f64>,
    frustum_length: f64,
) -> LensGrid {
    let c = image.camera_center();
    let apex = [c.x - centre.x, c.y - centre.y, c.z - centre.z];
    let size = if camera.has_distortion() || camera.model.is_fisheye() {
        LENS_GRID_SIZE
    } else {
        2
    };
    let grid = compute_distorted_frustum_grid(
        &apex,
        &image.camera_to_world_rotation_flat(),
        camera,
        frustum_length,
        size - 1,
    );
    LensGrid {
        size,
        positions: grid.positions,
    }
}

/// The view that frames the scene: looking at the centre from above the side
/// the cameras are on, far enough back that the points' bounding sphere, or
/// most of the cameras when they are farther out, fills the view.
fn framing_view(recon: &SfmrReconstruction, centre: &Point3<f64>, radius: f64) -> Value {
    let images = &recon.image_table.images;
    let offsets: Vec<Vector3<f64>> = images.iter().map(|i| i.camera_center() - centre).collect();
    let mut reach = radius;
    let mut side = Vector3::zeros();
    if !offsets.is_empty() {
        let mut distances: Vec<f64> = offsets.iter().map(|o| o.norm()).collect();
        distances.sort_by(f64::total_cmp);
        reach = reach.max(distances[distances.len() * 9 / 10]);
        side = offsets.iter().sum::<Vector3<f64>>() / offsets.len() as f64;
    }
    side.z = 0.0;
    let side = if side.norm() > 1e-9 * reach.max(1e-12) {
        side.normalize()
    } else {
        Vector3::new(0.0, -1.0, 0.0)
    };
    let elevation = 30f64.to_radians();
    let direction = side * elevation.cos() + Vector3::z() * elevation.sin();
    let distance = reach / (FRAMING_FOV_DEG.to_radians() / 2.0).sin();
    let eye = direction * distance;
    json!({
        "eye": [eye.x, eye.y, eye.z],
        "target": [0.0, 0.0, 0.0],
        "fov": FRAMING_FOV_DEG,
        "start_image": Value::Null,
    })
}

/// The view through image `index`'s camera: the eye at its centre, the target
/// on its optical axis as far out as the scene centre lies along it, and its
/// vertical field of view, held to 20 to 100 degrees.
fn start_view(
    recon: &SfmrReconstruction,
    index: usize,
    centre: &Point3<f64>,
    radius: f64,
) -> Value {
    let image = &recon.image_table.images[index];
    let camera = recon.image_table.camera_for_image(index);
    let eye = image.camera_center() - centre;
    let forward = image.quaternion_wxyz.inverse() * Vector3::new(0.0, 0.0, -1.0);
    let along = (-eye).dot(&forward);
    let depth = if along > 0.05 * radius { along } else { radius };
    let target = eye + forward * depth;
    let (w, h) = (camera.width as f64, camera.height as f64);
    let top = camera.pixel_to_ray(w / 2.0, 0.0);
    let bottom = camera.pixel_to_ray(w / 2.0, h);
    let cos = top[0] * bottom[0] + top[1] * bottom[1] + top[2] * bottom[2];
    let fov = cos.clamp(-1.0, 1.0).acos().to_degrees();
    let fov = if fov.is_finite() {
        fov.clamp(20.0, 100.0)
    } else {
        FRAMING_FOV_DEG
    };
    json!({
        "eye": [eye.x, eye.y, eye.z],
        "target": [target.x, target.y, target.z],
        "fov": fov,
        "start_image": image.name,
    })
}

/// Little-endian arrays packed one after another, each starting on a 4-byte
/// boundary, with a `scene.json` entry per array naming where it is.
#[derive(Default)]
struct ByteBlock {
    bytes: Vec<u8>,
    arrays: Map<String, Value>,
}

impl ByteBlock {
    fn describe(&mut self, name: &str, kind: &str, components: usize, count: usize) {
        self.arrays.insert(
            name.into(),
            json!({
                "offset": self.bytes.len(),
                "type": kind,
                "components": components,
                "count": count,
            }),
        );
    }

    fn pad(&mut self) {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
    }

    fn push_f32(&mut self, name: &str, components: usize, values: &[f32]) {
        self.describe(name, "f32", components, values.len() / components);
        self.bytes
            .extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }

    fn push_u32(&mut self, name: &str, values: &[u32]) {
        self.describe(name, "u32", 1, values.len());
        self.bytes
            .extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }

    fn push_u16(&mut self, name: &str, values: &[u16]) {
        self.describe(name, "u16", 1, values.len());
        self.bytes
            .extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.pad();
    }

    fn push_u8(&mut self, name: &str, components: usize, values: &[u8]) {
        self.describe(name, "u8", components, values.len() / components);
        self.bytes.extend_from_slice(values);
        self.pad();
    }
}

#[cfg(test)]
mod tests;
