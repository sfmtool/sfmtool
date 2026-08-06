// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared application state.

use crate::scene::{node_by_id, ImageRef, PointRef, ReconId, SceneNode};
use crate::scene_renderer::{
    DEFAULT_FRUSTUM_SIZE_MULTIPLIER, DEFAULT_LENGTH_SCALE_MULTIPLIER,
    DEFAULT_TARGET_FOG_MULTIPLIER, DEFAULT_TARGET_SIZE_MULTIPLIER,
};
use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::SfmrReconstruction;
use std::collections::HashMap;

/// The window title with no file loaded. `ui_basic`'s Windows attach path
/// finds our window by this exact name, so it is also what the tests match on.
pub const WINDOW_TITLE_BASE: &str = "SfM Explorer";

/// Which overlay to draw on the image detail panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlayMode {
    /// No feature overlay — clean image only.
    #[default]
    None,
    /// SIFT keypoint ellipses + center dots.
    Features,
    /// Colored circles by reprojection error.
    ReprojError,
    /// Colored circles by track length (observation count).
    TrackLength,
    /// Colored circles by the max pairwise angle (degrees) between
    /// world-space rays from observing cameras to the 3D point, i.e. the
    /// track's widest triangulation baseline. High = well-triangulated,
    /// low = unreliable.
    MaxTrackAngle,
    /// Colored circles by the inverse-depth z-score (`depth / σ_depth`) of the
    /// 3D point's triangulation. High = depth well-resolved, low = depth
    /// statistically indistinguishable from infinity. Unlike the max angle this
    /// is scale-free and does not inflate with view count, so it stays right in
    /// the distant / near-infinity regime.
    DepthReliability,
    /// Colored circles by the condition number of the triangulation's normal
    /// matrix (log scale). Low = well-conditioned depth, high = ill-conditioned
    /// / near-degenerate. A cheap geometric proxy that scales with track length.
    ConditionNumber,
}

impl OverlayMode {
    pub const ALL: [OverlayMode; 7] = [
        OverlayMode::None,
        OverlayMode::Features,
        OverlayMode::ReprojError,
        OverlayMode::TrackLength,
        OverlayMode::MaxTrackAngle,
        OverlayMode::DepthReliability,
        OverlayMode::ConditionNumber,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OverlayMode::None => "None",
            OverlayMode::Features => "Features",
            OverlayMode::ReprojError => "Reproj Error",
            OverlayMode::TrackLength => "Track Length",
            OverlayMode::MaxTrackAngle => "Max Track Angle",
            OverlayMode::DepthReliability => "Depth Reliability",
            OverlayMode::ConditionNumber => "Condition Number",
        }
    }
}

/// Scene-level settings controlling which features are displayed and how.
pub struct FeatureDisplaySettings {
    /// Which overlay mode is active.
    pub overlay_mode: OverlayMode,
    /// Maximum number of features to display per image. None = unlimited.
    /// Since features are sorted by decreasing size, this shows the N largest.
    pub max_features: Option<usize>,
    /// Minimum feature size threshold in pixels. None = no threshold.
    pub min_feature_size: Option<f32>,
    /// Maximum feature size threshold in pixels. None = no threshold.
    pub max_feature_size: Option<f32>,
    /// Drag value for the min size slider (persists when checkbox is unchecked).
    pub min_feature_size_value: f32,
    /// Drag value for the max size slider (persists when checkbox is unchecked).
    pub max_feature_size_value: f32,
    /// If true, only show features that have an associated 3D point.
    pub tracked_only: bool,
}

impl Default for FeatureDisplaySettings {
    fn default() -> Self {
        Self {
            overlay_mode: OverlayMode::Features,
            max_features: None,
            min_feature_size: None,
            max_feature_size: None,
            min_feature_size_value: 0.0,
            max_feature_size_value: 50.0,
            tracked_only: true,
        }
    }
}

/// Global application state shared across all views.
pub struct AppState {
    /// The loaded reconstructions, in load order.
    ///
    /// Phase-1 invariant: **at most one** node. `File > Open` and the demo
    /// dialog both replace the whole vector ([`AppState::set_single_node`]);
    /// appending arrives with the Scene Graph panel in phase 3.
    pub scene: Vec<SceneNode>,

    /// The reconstruction that file- and sequence-shaped UI follows (Image
    /// Browser strip, animation, `,`/`.` stepping). `Some` whenever `scene` is
    /// non-empty.
    pub selected_recon: Option<ReconId>,

    /// Currently selected image.
    pub selected_image: Option<ImageRef>,

    /// Currently selected 3D point.
    pub selected_point: Option<PointRef>,

    /// Transient hover state: image under cursor (from GPU pick or browser).
    /// Updated every frame; cleared when pointer leaves the source panel.
    pub hovered_image: Option<ImageRef>,

    /// Transient hover state: 3D point under cursor (from GPU pick or detail).
    /// Updated every frame; cleared when pointer leaves the source panel.
    pub hovered_point: Option<PointRef>,

    /// Feature overlay display settings (shared across images).
    pub feature_display: FeatureDisplaySettings,

    /// Whether to show 3D points.
    pub show_points: bool,

    /// Whether to show camera frustums.
    pub show_camera_images: bool,

    /// Whether to show the ground plane grid.
    pub show_grid: bool,

    /// Whether to show patch surfels (textured oriented quads). Only has an
    /// effect when the loaded reconstruction carries patch frames + bitmaps.
    pub show_patches: bool,

    /// Global opacity multiplier for patch surfel color.
    pub patch_opacity: f32,

    /// Log2 multiplier on the stored patch half-extents, mirroring
    /// `point_size_log2`. Actual multiplier = 2^patch_size_log2.
    pub patch_size_log2: f32,

    /// Coverage cutoff on the patch bitmap alpha (per-pixel cross-view
    /// confidence): fragments below this are discarded so ragged patch edges
    /// drop out. Defaults to `0.0` — render every texel opaque, discarding
    /// nothing — since the alpha look reads better fully filled; raise it to
    /// carve ragged edges away.
    pub patch_alpha_cutoff: f32,

    /// Status message shown in the UI (e.g. loading errors).
    pub status_message: Option<String>,

    /// Log2 multiplier on the auto-computed point size.
    /// 0.0 = use auto size, positive = larger, negative = smaller.
    /// Actual multiplier = 2^point_size_log2.
    pub point_size_log2: f32,

    /// Whether to draw points at infinity (`w = 0`). Independent of
    /// `show_points`, because a skyline of directions is often exactly the part
    /// of a reconstruction you want to look at without — or only without.
    pub show_points_at_infinity: bool,

    /// On-screen splat radius (pixels) for points at infinity. A direction has
    /// no distance, so infinity points are sized in pixels rather than world
    /// units like finite points.
    pub infinity_point_px: f32,

    /// Whether the bottom-right navigation cheat sheet is painted.
    pub show_controls_help: bool,

    /// Whether the top-left scene stats include the frame rate.
    pub show_fps: bool,

    /// EDL line thickness in pixels. Controls how far the neighbor samples
    /// reach, which determines the width of depth-discontinuity edges.
    pub edl_line_thickness: f32,

    /// Target indicator size multiplier (radius = multiplier * length_scale).
    pub target_size_multiplier: f32,

    /// Target indicator fog multiplier (fog_distance = multiplier * length_scale).
    pub target_fog_multiplier: f32,

    /// World-space length scale. Represents characteristic scene size.
    /// Initialized to `DEFAULT_LENGTH_SCALE_MULTIPLIER * auto_point_size` on
    /// point upload, then independently adjustable via UI.
    pub length_scale: f32,

    /// Frustum stub depth as a fraction of `length_scale`.
    pub frustum_size_multiplier: f32,

    /// Cached SIFT feature positions and affine shapes per image.
    /// Shared by ImageDetail (for drawing features) and track ray upload
    /// (for computing true observation ray directions).
    /// Cleared when the scene changes.
    pub sift_cache: HashMap<ImageRef, CachedSiftFeatures>,

    /// Full-resolution source images decoded to CPU pixels (RGB `ImageU8`).
    /// `None` = decode failed (don't retry). Shared by ImageDetail (builds its
    /// GPU texture from this) and PointTrackDetail (CPU-samples it to render
    /// per-observation patch tiles). Cleared when the scene changes.
    pub full_res_cache: HashMap<ImageRef, Option<ImageU8>>,

    /// Whether the "Load Demo Data" dialog is currently open.
    pub show_demo_dialog: bool,

    /// Number of points configured in the demo-data dialog (preserved across opens).
    pub demo_num_points: usize,
}

/// Cached SIFT positions and affine shapes for one image (no descriptors).
pub struct CachedSiftFeatures {
    /// Feature positions (x, y) in image pixel coordinates. Length = read_count.
    pub positions_xy: Vec<[f32; 2]>,
    /// Affine shape matrices [[a11, a12], [a21, a22]]. Length = read_count.
    pub affine_shapes: Vec<[[f32; 2]; 2]>,
    /// How many features were read from the file (the read_count used).
    pub read_count: usize,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            scene: Vec::new(),
            selected_recon: None,
            selected_image: None,
            selected_point: None,
            hovered_image: None,
            hovered_point: None,
            feature_display: FeatureDisplaySettings::default(),
            show_points: true,
            show_camera_images: true,
            show_grid: true,
            show_patches: true,
            patch_opacity: 1.0,
            patch_size_log2: 0.0,
            patch_alpha_cutoff: 0.0,
            status_message: None,
            point_size_log2: 0.0,
            show_points_at_infinity: true,
            infinity_point_px: 3.0,
            show_controls_help: true,
            show_fps: true,
            edl_line_thickness: 2.4,
            target_size_multiplier: DEFAULT_TARGET_SIZE_MULTIPLIER,
            target_fog_multiplier: DEFAULT_TARGET_FOG_MULTIPLIER,
            length_scale: DEFAULT_LENGTH_SCALE_MULTIPLIER * 0.03, // fallback until points loaded
            frustum_size_multiplier: DEFAULT_FRUSTUM_SIZE_MULTIPLIER,
            sift_cache: HashMap::new(),
            full_res_cache: HashMap::new(),
            show_demo_dialog: false,
            demo_num_points: 1000,
        }
    }

    /// Install `node` as the whole scene, replacing whatever was loaded.
    ///
    /// The single node-creation path: file loads and demo data both come
    /// through here, so neither can forget the cache and selection resets that
    /// a new reconstruction requires. (The demo path used to skip them, which
    /// left the SIFT and full-res caches keyed to the *previous* file.)
    ///
    /// Phase 3 replaces this with append-on-open plus per-node close.
    pub fn set_single_node(&mut self, node: SceneNode) {
        self.status_message = None;
        self.selected_recon = Some(node.id);
        self.scene = vec![node];
        self.selected_image = None;
        self.selected_point = None;
        self.hovered_image = None;
        self.hovered_point = None;
        self.sift_cache.clear();
        self.full_res_cache.clear();
    }

    /// Load a reconstruction from an .sfmr file.
    pub fn load_file(&mut self, path: &std::path::Path) {
        match SfmrReconstruction::load(path) {
            Ok(recon) => {
                log::info!(
                    "Loaded {} points, {} images from {}",
                    recon.point_count(),
                    recon.image_count(),
                    path.display()
                );
                self.set_single_node(SceneNode::from_path(path, recon));
            }
            Err(e) => {
                let msg = format!("Failed to load {}: {}", path.display(), e);
                log::error!("{}", msg);
                self.status_message = Some(msg);
            }
        }
    }

    /// Replace the scene with generated demo data.
    pub fn load_demo(&mut self, num_points: usize) {
        self.set_single_node(SceneNode::demo(SfmrReconstruction::demo(num_points)));
    }

    /// The node the panels follow, if any.
    ///
    /// Borrows all of `self`; where a caller also needs `&mut` access to a
    /// cache, go through [`crate::scene::node_by_id`] with `self.selected_recon`
    /// so only the `scene` field is borrowed.
    pub fn selected_node(&self) -> Option<&SceneNode> {
        node_by_id(&self.scene, self.selected_recon?)
    }

    /// Look up a loaded node by id.
    pub fn node(&self, id: ReconId) -> Option<&SceneNode> {
        node_by_id(&self.scene, id)
    }

    /// The selected image's local index, when it belongs to `recon`.
    pub fn selected_image_in(&self, recon: ReconId) -> Option<usize> {
        self.selected_image?.index_in(recon)
    }

    /// The selected point's local index, when it belongs to `recon`.
    pub fn selected_point_in(&self, recon: ReconId) -> Option<usize> {
        self.selected_point?.index_in(recon)
    }

    /// The hovered image's local index, when it belongs to `recon`.
    pub fn hovered_image_in(&self, recon: ReconId) -> Option<usize> {
        self.hovered_image?.index_in(recon)
    }

    /// The hovered point's local index, when it belongs to `recon`.
    pub fn hovered_point_in(&self, recon: ReconId) -> Option<usize> {
        self.hovered_point?.index_in(recon)
    }

    /// The window title for the current state: the base name alone until a file
    /// is loaded, then `"SfM Explorer - <file>"`.
    ///
    /// Demo data leaves this at the base title — it came from no file, so
    /// naming one would be a lie.
    pub fn window_title(&self) -> String {
        match self.scene.first().and_then(|node| node.file_name()) {
            Some(name) => format!("{WINDOW_TITLE_BASE} - {name}"),
            None => WINDOW_TITLE_BASE.to_string(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Get cached SIFT features for an image, loading from disk if needed.
///
/// This is a free function (not a method on `AppState`) so the caller can borrow
/// `sift_cache` mutably while simultaneously borrowing other `AppState` fields
/// (like `reconstruction`) immutably.
///
/// Reads up to `read_count` features from the `.sift` file. If a cached entry
/// exists with at least `read_count` features, returns it directly.
pub fn ensure_sift_cached<'a>(
    cache: &'a mut HashMap<ImageRef, CachedSiftFeatures>,
    recon: &SfmrReconstruction,
    image: ImageRef,
    read_count: usize,
) -> Option<&'a CachedSiftFeatures> {
    let image_idx = image.index();

    // Check if we already have enough features cached
    if cache
        .get(&image)
        .is_some_and(|c| c.read_count >= read_count)
    {
        return cache.get(&image);
    }

    // Load from disk
    let sift_path = recon.sift_path_for_image(image_idx);
    let sift_data = match sift_format::read_sift_partial(&sift_path, read_count) {
        Ok(d) => d,
        Err(e) => {
            log::warn!(
                "Failed to read SIFT data from {}: {}",
                sift_path.display(),
                e
            );
            return None;
        }
    };

    let n = sift_data.positions_xy.nrows();
    let mut positions_xy = Vec::with_capacity(n);
    let mut affine_shapes = Vec::with_capacity(n);
    for i in 0..n {
        positions_xy.push([
            sift_data.positions_xy[[i, 0]],
            sift_data.positions_xy[[i, 1]],
        ]);
        affine_shapes.push([
            [
                sift_data.affine_shapes[[i, 0, 0]],
                sift_data.affine_shapes[[i, 0, 1]],
            ],
            [
                sift_data.affine_shapes[[i, 1, 0]],
                sift_data.affine_shapes[[i, 1, 1]],
            ],
        ]);
    }
    cache.insert(
        image,
        CachedSiftFeatures {
            positions_xy,
            affine_shapes,
            read_count: n,
        },
    );
    cache.get(&image)
}

/// Get the cached full-resolution image for an image index, decoding from disk
/// if needed.
///
/// This is a free function (not a method on `AppState`) so the caller can borrow
/// `full_res_cache` mutably while simultaneously borrowing other `AppState`
/// fields (like `reconstruction`) immutably.
///
/// Images are decoded to 3-channel RGB [`ImageU8`]. A failed decode is memoized
/// as `None` so missing files aren't re-opened every frame.
pub fn ensure_full_res_cached<'a>(
    cache: &'a mut HashMap<ImageRef, Option<ImageU8>>,
    recon: &SfmrReconstruction,
    image: ImageRef,
) -> Option<&'a ImageU8> {
    cache
        .entry(image)
        .or_insert_with(|| {
            recon.images.get(image.index()).and_then(|im| {
                let path = recon.workspace_dir.join(&im.name);
                match image::open(&path) {
                    Ok(dyn_image) => {
                        let rgb = dyn_image.to_rgb8();
                        Some(ImageU8::new(rgb.width(), rgb.height(), 3, rgb.into_raw()))
                    }
                    Err(e) => {
                        log::warn!("Failed to load full-res image {}: {}", path.display(), e);
                        None
                    }
                }
            })
        })
        .as_ref()
}
