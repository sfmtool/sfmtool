// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared application state.

#![allow(dead_code)]

use crate::scene_renderer::{
    DEFAULT_FRUSTUM_SIZE_MULTIPLIER, DEFAULT_LENGTH_SCALE_MULTIPLIER,
    DEFAULT_TARGET_FOG_MULTIPLIER, DEFAULT_TARGET_SIZE_MULTIPLIER,
};
use sfmtool_core::SfmrReconstruction;
use std::collections::HashMap;

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
}

impl OverlayMode {
    pub const ALL: [OverlayMode; 4] = [
        OverlayMode::None,
        OverlayMode::Features,
        OverlayMode::ReprojError,
        OverlayMode::TrackLength,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OverlayMode::None => "None",
            OverlayMode::Features => "Features",
            OverlayMode::ReprojError => "Reproj Error",
            OverlayMode::TrackLength => "Track Length",
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
    /// The currently loaded reconstruction.
    pub reconstruction: Option<SfmrReconstruction>,

    /// Currently selected image index.
    pub selected_image: Option<usize>,

    /// Currently selected 3D point index.
    pub selected_point: Option<usize>,

    /// Transient hover state: image index under cursor (from GPU pick or browser).
    /// Updated every frame; cleared when pointer leaves the source panel.
    pub hovered_image: Option<usize>,

    /// Transient hover state: 3D point index under cursor (from GPU pick or detail).
    /// Updated every frame; cleared when pointer leaves the source panel.
    pub hovered_point: Option<usize>,

    /// Feature overlay display settings (shared across images).
    pub feature_display: FeatureDisplaySettings,

    /// Whether to show 3D points.
    pub show_points: bool,

    /// Whether to show camera frustums.
    pub show_camera_images: bool,

    /// Whether to show the ground plane grid.
    pub show_grid: bool,

    /// Status message shown in the UI (e.g. loading errors).
    pub status_message: Option<String>,

    /// Whether point cloud data needs to be uploaded to the GPU.
    /// Set to true when a reconstruction is loaded; cleared after upload.
    pub points_need_upload: bool,

    /// Log2 multiplier on the auto-computed point size.
    /// 0.0 = use auto size, positive = larger, negative = smaller.
    /// Actual multiplier = 2^point_size_log2.
    pub point_size_log2: f32,

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

    /// Cached SIFT feature positions and affine shapes per image index.
    /// Shared by ImageDetail (for drawing features) and track ray upload
    /// (for computing true observation ray directions).
    /// Cleared when the reconstruction changes.
    pub sift_cache: HashMap<usize, CachedSiftFeatures>,
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
            reconstruction: None,
            selected_image: None,
            selected_point: None,
            hovered_image: None,
            hovered_point: None,
            feature_display: FeatureDisplaySettings::default(),
            show_points: true,
            show_camera_images: true,
            show_grid: true,
            status_message: None,
            points_need_upload: false,
            point_size_log2: 0.0,
            edl_line_thickness: 2.4,
            target_size_multiplier: DEFAULT_TARGET_SIZE_MULTIPLIER,
            target_fog_multiplier: DEFAULT_TARGET_FOG_MULTIPLIER,
            length_scale: DEFAULT_LENGTH_SCALE_MULTIPLIER * 0.03, // fallback until points loaded
            frustum_size_multiplier: DEFAULT_FRUSTUM_SIZE_MULTIPLIER,
            sift_cache: HashMap::new(),
        }
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
                self.status_message = None;
                self.reconstruction = Some(recon);
                self.selected_image = None;
                self.selected_point = None;
                self.hovered_image = None;
                self.hovered_point = None;
                self.points_need_upload = true;
                self.sift_cache.clear();
            }
            Err(e) => {
                let msg = format!("Failed to load {}: {}", path.display(), e);
                log::error!("{}", msg);
                self.status_message = Some(msg);
            }
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
    cache: &'a mut HashMap<usize, CachedSiftFeatures>,
    recon: &SfmrReconstruction,
    image_idx: usize,
    read_count: usize,
) -> Option<&'a CachedSiftFeatures> {
    // Check if we already have enough features cached
    if cache
        .get(&image_idx)
        .is_some_and(|c| c.read_count >= read_count)
    {
        return cache.get(&image_idx);
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
        image_idx,
        CachedSiftFeatures {
            positions_xy,
            affine_shapes,
            read_count: n,
        },
    );
    cache.get(&image_idx)
}