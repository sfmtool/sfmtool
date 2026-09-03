// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared application state.

use egui_dock::DockState;

use crate::action_log::{ActionLog, Kind};
use crate::align::{self, AlignOptions};
use crate::dock::Tab;
use crate::goto_point::{self, GotoPointDialog};
use crate::layout::Layout;
use crate::resect::{self, ResectFrom};
use crate::scene::{node_by_id, unique_label, CameraRef, ImageRef, PointRef, ReconId, SceneNode};
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

/// Scene-level state of the intrinsics overlay layer, drawn on the Image
/// Detail panel independently of [`FeatureDisplaySettings::overlay_mode`].
///
/// A **layer**, not an eighth [`OverlayMode`]: the questions worth asking about
/// a lens are the joint ones — do the keypoints crowd the distorted rim, is the
/// reprojection-error heatmap hot precisely where the displacement field is
/// largest — and an exclusive mode turns each of those into flipping back and
/// forth from memory. So the layer composes with whichever feature mode is
/// active, including [`OverlayMode::None`], and its state lives here rather
/// than in [`FeatureDisplaySettings`], whose name and contents are about
/// *feature* display. See `specs/gui/camera-intrinsics.md` § "Image Detail:
/// the Intrinsics overlay layer".
pub struct IntrinsicsDisplaySettings {
    /// Draw the layer at all. Off by default: it is a diagnostic, and the
    /// panel's default view is the photograph.
    pub enabled: bool,
    /// Draw the angular axes through the principal point.
    pub axes: bool,
    /// Draw iso-angle rings at the same angular ladder as the axis ticks.
    pub rings: bool,
    /// Draw the distortion displacement field. Ignored when the model has no
    /// distortion.
    pub distortion: bool,
    /// Displacement arrow exaggeration. `None` = auto.
    pub distortion_scale: Option<f32>,
    /// Grid density of the arrow field, arrows across the image width.
    pub grid_cols: usize,
}

impl Default for IntrinsicsDisplaySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            axes: true,
            rings: false,
            distortion: true,
            distortion_scale: None,
            grid_cols: 16,
        }
    }
}

impl IntrinsicsDisplaySettings {
    /// The exaggerations the auto scale chooses among, and the settings popup
    /// offers by hand.
    pub const SCALE_LADDER: [f32; 7] = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0];

    /// The grid densities the settings popup offers.
    pub const GRID_LADDER: [usize; 5] = [8, 12, 16, 24, 32];
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
    /// The loaded reconstructions, in load order — which is also tree order in
    /// the Scene Graph panel, and the order `[` / `]` step through.
    ///
    /// `File > Open`, the demo dialog and the CLI all *append* through
    /// [`AppState::append_node`]; nodes leave one at a time through
    /// [`AppState::close_node`] or wholesale through [`AppState::close_all`].
    pub scene: Vec<SceneNode>,

    /// The reconstruction that file- and sequence-shaped UI follows (Image
    /// Browser strip, animation, `,`/`.` stepping). `Some` whenever `scene` is
    /// non-empty.
    pub selected_recon: Option<ReconId>,

    /// The soloed reconstruction: while `Some`, only that node is drawn.
    ///
    /// A **view mode layered over** the per-node eyes, not a bulk edit of them:
    /// `SceneNode::visible` is never written by soloing, so un-soloing restores
    /// exactly the visibility the user had — including nodes they had already
    /// hidden by hand — and an eye toggled while soloed takes effect the moment
    /// the solo ends. Mutating the eyes and restoring them later would be
    /// lossy (a hidden node soloed and un-soloed would come back visible) and
    /// would need a saved copy that any other path touching `visible` could
    /// desync. One `Option` cannot desync with anything.
    ///
    /// At most one node at a time, so soloing B while A is soloed simply moves
    /// the solo to B. Composed with the eyes by [`crate::scene::is_visible`].
    pub solo: Option<ReconId>,

    /// Currently selected image.
    ///
    /// Set only through [`AppState::select_image`], which is what keeps
    /// `selected_camera` in step with it.
    pub selected_image: Option<ImageRef>,

    /// The selected camera intrinsics, or `None`.
    ///
    /// Coupled to `selected_image`: whenever `selected_image` is `Some`, this
    /// is `Some` and names that image's camera. See
    /// [`AppState::select_camera`].
    ///
    /// **Stored, not derived.** A camera can be selected with no image
    /// selected at all — that is the whole point of picking one out of the
    /// tree — so deriving it from `selected_image` would blank it the moment
    /// the user clicked anything else. What is derived is the *constraint*
    /// between the two, and that lives in the two setters.
    pub selected_camera: Option<CameraRef>,

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

    /// Intrinsics overlay layer settings (shared across images). A sibling of
    /// `feature_display`, not part of it: the two layers compose rather than
    /// excluding one another.
    pub intrinsics_display: IntrinsicsDisplaySettings,

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

    /// Every action taken in this session, by whoever took it.
    ///
    /// A field of the state rather than a panel's own, so that every method
    /// here which owns a change also owns the record of it — and so the
    /// viewport status line can be a one-row window onto the log
    /// ([`AppState::status_message`]) instead of a second piece of state that
    /// can disagree with it. See [`crate::action_log`].
    pub action_log: ActionLog,

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

    /// Bumped whenever any node's transform is set or reset.
    ///
    /// A node transform is a world-space change, so it invalidates the same
    /// derived state a fresh upload does: the union scene bounds, the global
    /// `length_scale`, and the frustum geometry sized from it. Comparing one
    /// counter against the previous frame's is how `app.rs` notices, without
    /// having to diff a `Vec<Se3Transform>`.
    pub transform_epoch: u64,

    /// Whether the "Load Demo Data" dialog is currently open.
    pub show_demo_dialog: bool,

    /// Number of points configured in the demo-data dialog (preserved across opens).
    pub demo_num_points: usize,

    /// The "Go to Point" dialog: the typed query, its last error, and whether
    /// it is showing. See [`crate::goto_point`].
    pub goto_point: GotoPointDialog,

    /// The `.matches` file each node's matches-backed resection reads, chosen
    /// once per source node and remembered for the session. See
    /// [`crate::resect`].
    pub resect_matches: HashMap<ReconId, std::path::PathBuf>,

    /// The last `.matches` file parsed, kept so repeated resections against the
    /// same file pay for the read once. One entry, not a map: a reviewer works
    /// through one capture at a time, and these files are large.
    resect_matches_cache: Option<(std::path::PathBuf, matches_format::MatchesData)>,

    /// The live MCP endpoint, or `None` when the viewer was started without
    /// `--mcp`. See [`crate::mcp`].
    #[cfg(feature = "mcp")]
    pub mcp: Option<McpStatus>,

    /// The window as it was last observed, or `None` before there is a window
    /// to observe.
    ///
    /// A snapshot rather than a live handle, so that reading the window is a
    /// plain field access from a method that has no `winit::Window` in reach —
    /// Panels ▸ Save Layout…, and the MCP `apply` seam, which is handed
    /// `(&mut AppState, &mut Viewer3D)` — and so the same reads work headlessly
    /// in a test. The frame refreshes it at the top of *every* frame
    /// ([`AppState::observe_window`]), and applying a window layout refreshes
    /// it again after the change, so nothing here can be a frame behind what
    /// was just asked for. See [`crate::window`].
    pub(crate) window: Option<crate::window::WindowInfo>,

    /// The rectangle the window had when it was last observed as `normal`.
    ///
    /// Remembered rather than read, because `winit` reports only the window's
    /// *current* rectangle: once it is maximized, the rectangle it will restore
    /// to is unreadable, and that is the one a saved layout needs. See
    /// `specs/gui/panel-layout.md` § "The window layout file".
    pub(crate) window_normal_rect: Option<crate::window::NormalRect>,

    /// Which panels are docked where.
    ///
    /// State rather than a field of `App`, so that a layout operation is an
    /// `AppState` method with the Action Log in reach — and so the MCP `apply`
    /// seam, which is handed `(&mut AppState, &mut Viewer3D)` and no `App`,
    /// can drive the layout headlessly. `DockState<Tab>` is plain data, with no
    /// GPU or window behind it. See [`crate::layout`].
    pub(crate) dock: DockState<Tab>,
}

/// What the viewer says about a live MCP endpoint.
///
/// A window that something else can drive should never look like one that
/// nothing can, so this is announced in two places at once: a window-title
/// suffix ([`AppState::window_title`]) and a header line on the Scene panel.
/// It lives on `AppState` rather than beside the server so both of those can
/// read it without knowing the transport exists.
#[cfg(feature = "mcp")]
pub struct McpStatus {
    /// The port actually bound, which is not necessarily the one asked for:
    /// `--mcp 0` takes an ephemeral one.
    pub port: u16,
    /// How many tool calls have been applied. Shown live, so a human can see
    /// the agent working.
    pub requests: u64,
}

#[cfg(feature = "mcp")]
impl McpStatus {
    pub fn new(port: u16) -> Self {
        Self { port, requests: 0 }
    }

    /// The URL a human pastes into a client config.
    pub fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.port)
    }
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
            solo: None,
            selected_image: None,
            selected_camera: None,
            selected_point: None,
            hovered_image: None,
            hovered_point: None,
            feature_display: FeatureDisplaySettings::default(),
            intrinsics_display: IntrinsicsDisplaySettings::default(),
            show_points: true,
            show_camera_images: true,
            show_grid: true,
            show_patches: true,
            patch_opacity: 1.0,
            patch_size_log2: 0.0,
            patch_alpha_cutoff: 0.0,
            action_log: ActionLog::new(),
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
            transform_epoch: 0,
            show_demo_dialog: false,
            demo_num_points: 1000,
            goto_point: GotoPointDialog::default(),
            resect_matches: HashMap::new(),
            resect_matches_cache: None,
            #[cfg(feature = "mcp")]
            mcp: None,
            window: None,
            window_normal_rect: None,
            dock: Layout::default().to_dock(),
        }
    }

    /// Append `node` to the scene and select it.
    ///
    /// The single node-arrival path: file loads, reloads, the CLI and demo data
    /// all come through here, so none of them can forget that arriving is also
    /// a selection change. Selecting the new node clears the image and point
    /// selection per the finer-selection invariant — you opened this file to
    /// look at it, and no panel should be left showing another file's row.
    pub fn append_node(&mut self, mut node: SceneNode) -> ReconId {
        node.label = unique_label(&self.scene, &node.label);
        let id = node.id;
        self.scene.push(node);
        // Muted: arriving *is* a selection change, but the caller's own entry
        // ("Opened x from …", "Loaded demo data", "Resected …") is the action,
        // and a `Selected reconstruction x` beneath it would say nothing more.
        self.action_log.mute();
        self.select_recon(id);
        self.action_log.unmute();
        self.hovered_image = None;
        self.hovered_point = None;
        // Arriving also ends any solo: you opened this file to look at it, and
        // a solo left over from before would hide it the moment it loaded.
        self.solo = None;
        id
    }

    /// Load a reconstruction from an .sfmr file, **appending** it as a node.
    ///
    /// Opening a path that is already loaded reloads that node in place instead
    /// — the predictable interpretation of "open this again", and it doubles as
    /// a refresh.
    ///
    /// A failure is **returned, not logged**: the File menu records it as
    /// `Failed to load …`, the MCP drain as `open_reconstruction failed: …`,
    /// and one failure that logged itself as well would appear twice, in two
    /// vocabularies. Success is logged here, because there the text is the same
    /// whoever asked.
    pub fn load_file(&mut self, path: &std::path::Path) -> Result<ReconId, String> {
        if let Some(id) = self
            .scene
            .iter()
            .find(|n| n.path.as_deref() == Some(path))
            .map(|n| n.id)
        {
            return self.reload_node(id);
        }
        match SfmrReconstruction::load(path) {
            Ok(recon) => {
                log::info!(
                    "Loaded {} points, {} images from {}",
                    recon.point_count(),
                    recon.image_count(),
                    path.display()
                );
                // Recorded after the append, which is what deduplicates the
                // label: the entry should name the node as the tree does
                // (`global (2)`), not the file stem the node arrived with.
                let id = self.append_node(SceneNode::from_path(path, recon));
                let label = self.label_of(id);
                self.action_log.record(
                    Kind::File,
                    format!("Opened {label} from {}", path.display()),
                );
                Ok(id)
            }
            Err(e) => {
                let msg = format!("Failed to load {}: {}", path.display(), e);
                log::error!("{}", msg);
                Err(msg)
            }
        }
    }

    /// Append a node of generated demo data.
    pub fn load_demo(&mut self, num_points: usize) {
        self.append_node(SceneNode::demo(SfmrReconstruction::demo(num_points)));
        self.action_log.record(Kind::File, "Loaded demo data");
    }

    /// A node's label, or a placeholder if it has already left the scene.
    fn label_of(&self, id: ReconId) -> String {
        self.node(id)
            .map(|node| node.label.clone())
            .unwrap_or_default()
    }

    /// Re-read a node's file from disk, keeping its place in tree order, its
    /// label and its display settings.
    ///
    /// The refreshed node gets a **new** [`ReconId`]. A reload can change every
    /// entity count, so every index-keyed cache entry for the old id is wrong;
    /// a new id makes all of them unreachable rather than merely stale, which
    /// is the same guarantee that makes closing a node safe. Returns the new id,
    /// or the message for a demo node (no file to re-read) or a failed read.
    ///
    /// Like [`AppState::load_file`], a failure is returned rather than logged:
    /// the caller is the one that knows whether it was asked for as a reload or
    /// as an `open_reconstruction` of a path that happened to be loaded.
    pub fn reload_node(&mut self, id: ReconId) -> Result<ReconId, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let path = self.scene[index].path.clone().ok_or_else(|| {
            format!(
                "{} was generated, not loaded from a file, so there is nothing to re-read.",
                self.scene[index].label
            )
        })?;
        let recon = match SfmrReconstruction::load(&path) {
            Ok(recon) => recon,
            Err(e) => {
                let msg = format!("Failed to reload {}: {}", path.display(), e);
                log::error!("{}", msg);
                return Err(msg);
            }
        };
        let mut node = SceneNode::from_path(&path, recon);
        node.label = self.scene[index].label.clone();
        node.copy_display_from(&self.scene[index]);
        let new_id = node.id;
        let was_selected = self.selected_recon == Some(id);
        // A reload mints a fresh id, so the solo — which names an id rather
        // than a position — has to be re-pointed or refreshing the soloed node
        // would silently hide it along with everything else.
        let was_solo = self.solo == Some(id);
        self.scene[index] = node;
        self.forget_recon(id);
        if was_selected || self.selected_recon.is_none() {
            self.selected_recon = Some(new_id);
        }
        if was_solo {
            self.solo = Some(new_id);
        }
        let label = self.label_of(new_id);
        self.action_log
            .record(Kind::File, format!("Reloaded {label}"));
        Ok(new_id)
    }

    /// Remove a node from the scene and unwind everything that pointed into it.
    ///
    /// The renderer releases its bundle separately, from `retain_nodes` on the
    /// next frame; the camera view is dropped by the same frame's check in
    /// `app.rs`, which covers *every* way a node can leave the scene.
    pub fn close_node(&mut self, id: ReconId) {
        let Some(label) = self.node(id).map(|node| node.label.clone()) else {
            return;
        };
        self.scene.retain(|n| n.id != id);
        self.forget_recon(id);
        if self.selected_recon == Some(id) {
            // Fall back to the first remaining node; an empty scene means no
            // selection, and panels show their empty-state text.
            self.selected_recon = self.scene.first().map(|n| n.id);
        }
        if self.solo == Some(id) {
            // Closing the soloed node ends the solo rather than promoting the
            // next one: a solo naming a node that is gone would hide the whole
            // scene, with nothing left on screen to explain why.
            self.solo = None;
        }
        self.resect_matches.remove(&id);
        self.action_log
            .record(Kind::File, format!("Closed {label}"));
    }

    /// Clear the whole scene.
    ///
    /// One entry, not one per node: `Close All` is a single action, and a
    /// twelve-node scene should not push twelve lines through the log for it.
    pub fn close_all(&mut self) {
        let closed = self.scene.len();
        self.scene.clear();
        self.selected_recon = None;
        self.solo = None;
        self.selected_image = None;
        self.selected_camera = None;
        self.selected_point = None;
        self.hovered_image = None;
        self.hovered_point = None;
        self.sift_cache.clear();
        self.full_res_cache.clear();
        self.resect_matches.clear();
        self.resect_matches_cache = None;
        if closed > 0 {
            self.action_log
                .record(Kind::File, format!("Closed all ({closed})"));
        }
    }

    /// Drop every cache entry and every selection/hover ref belonging to `id`.
    ///
    /// Does *not* touch `scene` or `selected_recon` — the callers differ on
    /// what should happen to those (close falls back, reload re-points).
    fn forget_recon(&mut self, id: ReconId) {
        self.sift_cache.retain(|image, _| image.recon != id);
        self.full_res_cache.retain(|image, _| image.recon != id);
        self.selected_image = self.selected_image.filter(|i| i.recon != id);
        self.selected_camera = self.selected_camera.filter(|c| c.recon != id);
        self.selected_point = self.selected_point.filter(|p| p.recon != id);
        self.hovered_image = self.hovered_image.filter(|i| i.recon != id);
        self.hovered_point = self.hovered_point.filter(|p| p.recon != id);
    }

    /// Select a reconstruction directly.
    ///
    /// Image and point selections belonging to *other* nodes are cleared: the
    /// invariant is that all finer selection state lives inside the selected
    /// reconstruction, so no two panels ever show different files' selections.
    /// Hover is exempt — it is transient and may touch any visible node.
    pub fn select_recon(&mut self, id: ReconId) {
        let moved = self.selected_recon != Some(id);
        self.selected_recon = Some(id);
        self.selected_image = self.selected_image.filter(|i| i.recon == id);
        self.selected_camera = self.selected_camera.filter(|c| c.recon == id);
        self.selected_point = self.selected_point.filter(|p| p.recon == id);
        if moved {
            let label = self.label_of(id);
            self.action_log
                .record(Kind::Selection, format!("Selected reconstruction {label}"));
        }
    }

    /// Solo `id`, or end the solo if it is already the soloed node.
    ///
    /// Solo is not selection: it changes what is *drawn* and nothing else, so
    /// this leaves `selected_recon` and every finer selection exactly as they
    /// were. Soloing a second node moves the solo rather than adding to it —
    /// "show only this one" has one answer.
    pub fn toggle_solo(&mut self, id: ReconId) {
        self.set_solo((self.solo != Some(id)).then_some(id));
    }

    /// Solo one node, or end the solo — the *set* form, which the MCP `set_solo`
    /// tool needs because an agent issuing a toggle cannot know the outcome
    /// without reading the scene first, and a retried call would undo itself.
    ///
    /// [`AppState::toggle_solo`] resolves the click into a request and comes
    /// through here, so the two cannot disagree about what soloing does or
    /// about what the log says it did.
    pub fn set_solo(&mut self, solo: Option<ReconId>) {
        if self.solo == solo {
            return;
        }
        self.solo = solo;
        let text = match solo {
            Some(id) => format!("Soloed {}", self.label_of(id)),
            None => "Ended the solo".to_string(),
        };
        self.action_log.record(Kind::Scene, text);
    }

    /// The camera an image uses, when the image ref still resolves.
    pub fn camera_of(&self, image: ImageRef) -> Option<CameraRef> {
        let node = self.node(image.recon)?;
        let camera = node.recon.images.get(image.index())?.camera_index as usize;
        Some(CameraRef::new(image.recon, camera))
    }

    /// Select an image — and with it the reconstruction that owns it and the
    /// camera it was taken through.
    ///
    /// The one place an image selection is *set* — the recon-scoped paths
    /// (`select_recon`, `forget_recon`, `close_all`) only ever clear it — so
    /// no caller can leave `selected_camera` naming another image's lens.
    /// Passing `None`
    /// deselects the image and **keeps** the camera: dismissing a photograph
    /// says nothing about the lens, and collapsing the intrinsics with it
    /// would be a surprise. A second `Esc`, finding no image, clears the
    /// camera through [`AppState::select_camera`].
    pub fn select_image(&mut self, image: Option<ImageRef>) {
        let moved = self.selected_image != image;
        let had_one = self.selected_image.is_some();
        self.selected_image = image;
        let Some(image) = image else {
            if moved && had_one {
                self.action_log.record(Kind::Selection, "Deselected image");
            }
            return;
        };
        self.selected_recon = Some(image.recon);
        self.selected_camera = self.camera_of(image);
        self.selected_point = self.selected_point.filter(|p| p.recon == image.recon);
        if moved {
            let text = format!(
                "Selected image {} in {}",
                self.image_name(image),
                self.label_of(image.recon)
            );
            self.action_log.record(Kind::Selection, text);
        }
    }

    /// An image's `.sfmr` name, or a placeholder when the ref no longer
    /// resolves. Only ever used to build a log entry.
    fn image_name(&self, image: ImageRef) -> String {
        self.node(image.recon)
            .and_then(|node| node.recon.images.get(image.index()))
            .map(|i| i.name.clone())
            .unwrap_or_else(|| format!("#{}", image.index()))
    }

    /// Select a camera, and with it the reconstruction that owns it.
    ///
    /// Clears `selected_image` unless the selected image already uses this
    /// camera — asking for a *different* lens is a statement that the image on
    /// screen is no longer what is being looked at, while asking for the one
    /// it already uses is not, and clearing there would throw an image away
    /// for clicking the row that is already highlighted for it.
    ///
    /// Written from the Scene panel's Camera Intrinsics rows, and the only
    /// place `selected_camera` is set to a camera: everything else — the
    /// recon-scoped filters, [`AppState::select_image`] — either clears it or
    /// derives it from an image, so the invariant has one door.
    pub fn select_camera(&mut self, camera: Option<CameraRef>) {
        let moved = self.selected_camera != camera;
        let had_one = self.selected_camera.is_some() || self.selected_image.is_some();
        self.selected_camera = camera;
        let Some(camera) = camera else {
            // Clearing the camera clears the image with it. An image implies
            // its camera, so leaving one selected without the other is exactly
            // the state the invariant forbids — and the guarantee is that no
            // caller *can* reach it, not that no caller currently tries.
            self.selected_image = None;
            if had_one {
                self.action_log
                    .record(Kind::Selection, "Deselected camera intrinsics");
            }
            return;
        };
        self.selected_recon = Some(camera.recon);
        let image_uses_it = self
            .selected_image
            .is_some_and(|image| self.camera_of(image) == Some(camera));
        if !image_uses_it {
            self.selected_image = None;
        }
        self.selected_point = self.selected_point.filter(|p| p.recon == camera.recon);
        if moved {
            let text = format!(
                "Selected camera intrinsics #{} in {}",
                camera.index(),
                self.label_of(camera.recon)
            );
            self.action_log.record(Kind::Selection, text);
        }
    }

    /// Drop every selection at once, as one action.
    ///
    /// One entry rather than three: `clear_selection` is a single request, and
    /// the image, the intrinsics and the point going together is what it means
    /// rather than three things that happened to coincide.
    pub fn clear_selection(&mut self) {
        if self.selected_image.is_none()
            && self.selected_camera.is_none()
            && self.selected_point.is_none()
        {
            return;
        }
        self.action_log.mute();
        // `select_camera(None)` clears the image with it, so "everything" is
        // that followed by the point.
        self.select_camera(None);
        self.selected_point = None;
        self.action_log.unmute();
        self.action_log.record(Kind::Selection, "Cleared selection");
    }

    /// Drop the point selection alone.
    pub fn deselect_point(&mut self) {
        if self.selected_point.take().is_some() {
            self.action_log.record(Kind::Selection, "Deselected point");
        }
    }

    /// Open the Go to Point dialog, prefilled with the selected point's ID.
    ///
    /// The one entry point for all three ways in (the Go menu, Ctrl/Cmd+G, the
    /// Point Track panel's button), so none of them can forget the prefill —
    /// which is what makes the dialog double as a place to read or copy the
    /// current point's ID rather than only to type a new one.
    pub fn open_goto_point(&mut self) {
        let prefill = goto_point::selected_point_id(&self.scene, self.selected_point);
        self.goto_point.open(prefill);
    }

    /// Select a 3D point, and with it the reconstruction that owns it.
    pub fn select_point(&mut self, point: PointRef) {
        let moved = self.selected_point != Some(point);
        self.selected_point = Some(point);
        self.selected_recon = Some(point.recon);
        self.selected_image = self.selected_image.filter(|i| i.recon == point.recon);
        self.selected_camera = self.selected_camera.filter(|c| c.recon == point.recon);
        if moved {
            let id = self
                .node(point.recon)
                .map(|node| crate::scene::point_id(&node.recon, point.index()))
                .unwrap_or_else(|| format!("#{}", point.index()));
            self.action_log
                .record(Kind::Selection, format!("Selected point {id}"));
        }
    }

    /// Fit `source`'s transform so it lands on top of `target`, and report the
    /// outcome in the status message.
    ///
    /// The fit maps the source's *native* coordinates onto the target's native
    /// coordinates; what the node stores is that composed into the target's
    /// **currently displayed** frame — `source.transform = target.transform ∘
    /// T_fit`, so aligning C→B after B→A chains as expected. The target node is
    /// never touched, and on any failure neither is the source: the transform is
    /// left exactly as it was and only the status line changes.
    ///
    /// The fit runs synchronously. By-cameras is trivially small; by-points is a
    /// bounded RANSAC over the correspondences (see [`crate::align`]).
    pub fn align_node(&mut self, source: ReconId, target: ReconId, options: AlignOptions) {
        if source == target {
            return;
        }
        let (Some(si), Some(ti)) = (
            self.scene.iter().position(|n| n.id == source),
            self.scene.iter().position(|n| n.id == target),
        ) else {
            return;
        };
        let (source_label, target_label) =
            (self.scene[si].label.clone(), self.scene[ti].label.clone());
        let fit =
            align::align_reconstructions(&self.scene[si].recon, &self.scene[ti].recon, options);
        match fit {
            Ok(fit) => {
                // `compose` applies the receiver first: the fit takes the source
                // into the target's own coordinates, then the target's transform
                // takes those into world space.
                self.scene[si].transform = fit.transform.compose(&self.scene[ti].transform);
                self.transform_epoch += 1;
                let message = align::success_message(&source_label, &target_label, &fit);
                self.action_log.record(Kind::Scene, message);
            }
            Err(reason) => {
                let message = align::failure_message(&source_label, &target_label, &reason);
                self.action_log.fail(Kind::Scene, message);
            }
        }
    }

    /// Re-estimate one image's pose against the rest of its reconstruction, and
    /// show the answer as a new node beside the source.
    ///
    /// The source node is never modified under any outcome. On success the
    /// derived node is named `<source> (resected <image>)`, inherits the
    /// source's current transform (so it lands exactly on top of it), becomes
    /// the selected reconstruction with the resected image selected in it, and
    /// carries the marker that says which image moved. A second resection of the
    /// same image from the same source **replaces** the earlier derived node,
    /// in place, rather than adding a third.
    ///
    /// A refused *estimate* still produces the node — with the stored pose
    /// retained, so the reviewer can see the held-out re-triangulation on its
    /// own — and reports the refusal. A resection that could not be attempted
    /// at all produces no node and only a status line.
    ///
    /// Runs synchronously; see `specs/gui/resect-image.md`, "Performance".
    ///
    /// The outcome is one Action Log entry, and the node arrival and selection
    /// change inside are muted: a resection is one action, and its result — not
    /// its mechanics — is what the log and the status line should carry.
    pub fn resect_image(&mut self, source: ReconId, image: usize, from: ResectFrom) {
        self.action_log.mute();
        let outcome = self.resect_image_inner(source, image, from);
        self.action_log.unmute();
        match outcome {
            Some(Ok(message)) => self.action_log.record(Kind::Scene, message),
            Some(Err(message)) => self.action_log.fail(Kind::Scene, message),
            None => {}
        }
    }

    /// The resection itself. `None` when it could not be attempted at all
    /// (no such node, no such image); otherwise the message the outer method
    /// records, as success or refusal.
    fn resect_image_inner(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
    ) -> Option<Result<String, String>> {
        let index = self.scene.iter().position(|n| n.id == source)?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon
            .images
            .get(image)
            .map(|i| i.name.clone())?;
        let basename = resect::basename(&name).to_string();
        if from == ResectFrom::Matches {
            if let Err(reason) = self.load_resect_matches(source) {
                return Some(Err(resect::failure_message(&basename, &label, &reason)));
            }
        }

        // Both borrows are shared, and the outcome owns its reconstruction — so
        // nothing here is still borrowed when the scene is written below.
        let outcome = {
            let matches = match from {
                ResectFrom::Observations => None,
                ResectFrom::Matches => self.resect_matches_cache.as_ref().map(|(_, data)| data),
            };
            let kind = match matches {
                Some(data) => resect::ResectSource::Matches(data),
                None => resect::ResectSource::StoredObservations,
            };
            // The panel's action is one image, which is the set primitive on a
            // one-element set.
            resect::resect_images(
                &self.scene[index].recon,
                &[image],
                kind,
                &resect::ResectImageOptions::default(),
            )
        };
        let mut resected = match outcome {
            Ok(resected) => resected,
            Err(error) => {
                return Some(Err(resect::failure_message(
                    &basename,
                    &label,
                    &error.to_string(),
                )));
            }
        };
        let report = resected.reports.pop().expect("one target, one report");
        // A refused *estimate* still produces the node, so the message is
        // decided here and carried out past the scene edit below.
        let message = match &report.refusal {
            Some(reason) => Err(resect::failure_message(&basename, &label, reason)),
            None => Ok(resect::success_message(&basename, &label, &report)),
        };

        let derived_label = format!("{label} (resected {basename})");
        let mut node = SceneNode::derived(derived_label.clone(), resected.reconstruction);
        // The derived node lands in the source's *displayed* frame, so it sits
        // exactly on top of it and the two can be compared with every existing
        // affordance.
        node.transform = self.scene[index].transform.clone();
        let new_id = node.id;
        // The derived node's name is its provenance: the same source and image
        // produce the same name, which is how a repeat finds the node it
        // replaces.
        match self
            .scene
            .iter()
            .position(|n| n.path.is_none() && n.label == derived_label)
        {
            // Replaced in place, keeping its position in tree order and its
            // label: this is the same question asked again, not a third answer.
            Some(slot) => {
                let old = self.scene[slot].id;
                node.label = self.scene[slot].label.clone();
                node.copy_display_from(&self.scene[slot]);
                // After the display copy, which brought the *old* derived
                // node's frame with it: the source may have been aligned since.
                node.transform = self.scene[index].transform.clone();
                self.scene[slot] = node;
                self.forget_recon(old);
                if self.solo == Some(old) {
                    self.solo = Some(new_id);
                }
                self.selected_recon = Some(new_id);
            }
            // A first resection arrives like any other node — through the one
            // arrival path, which owns label disambiguation and what a new node
            // does to the selection and the solo.
            None => {
                self.append_node(node);
            }
        }
        self.hovered_image = None;
        self.hovered_point = None;
        // The point of the action is to look at the image that moved, so the
        // point track detail opens on it immediately.
        self.select_image(Some(ImageRef::new(new_id, image)));
        Some(message)
    }

    /// Make sure [`AppState::resect_matches_cache`] holds the `.matches` file
    /// chosen for `source`, reading it if it does not. `Err` carries the reason
    /// for the status line.
    fn load_resect_matches(&mut self, source: ReconId) -> Result<(), String> {
        let path = self
            .resect_matches
            .get(&source)
            .cloned()
            .ok_or_else(|| "no .matches file chosen".to_string())?;
        if self
            .resect_matches_cache
            .as_ref()
            .is_some_and(|(cached, _)| *cached == path)
        {
            return Ok(());
        }
        match matches_format::read_matches(&path) {
            Ok(data) => {
                self.resect_matches_cache = Some((path, data));
                Ok(())
            }
            Err(e) => {
                // A path that cannot be read is not a path worth remembering:
                // the next attempt should ask again rather than fail the same
                // way silently.
                self.resect_matches.remove(&source);
                Err(format!("could not read {}: {e}", path.display()))
            }
        }
    }

    /// Return a node to its own frame.
    pub fn reset_node_transform(&mut self, id: ReconId) {
        let Some(node) = self.scene.iter_mut().find(|n| n.id == id) else {
            return;
        };
        node.transform = sfmtool_core::Se3Transform::identity();
        let label = node.label.clone();
        self.transform_epoch += 1;
        self.action_log
            .record(Kind::Scene, format!("Reset transform of {label}"));
    }

    /// Look up a loaded node by id.
    pub fn node(&self, id: ReconId) -> Option<&SceneNode> {
        node_by_id(&self.scene, id)
    }

    /// The scene and the log at once.
    ///
    /// A split borrow, for the two callers that write a node's display state
    /// and record what they wrote in the same breath — the Scene panel's eyes
    /// and tint, and the MCP `set_reconstruction_display` tool. Going through
    /// `&mut self` twice would borrow the whole state twice.
    pub fn scene_and_log(&mut self) -> (&mut [SceneNode], &mut ActionLog) {
        (&mut self.scene, &mut self.action_log)
    }

    /// What the viewport status line shows: the text of the most recent
    /// non-query Action Log entry, prefixed `MCP: ` when an agent took it.
    ///
    /// Derived rather than stored, so a successful action cannot leave a stale
    /// error on screen and a refusal cannot go unreported. See
    /// [`crate::action_log::ActionLog::status_line`].
    pub fn status_message(&self) -> Option<String> {
        self.action_log.status_line()
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
    /// is loaded, then `"SfM Explorer - <file>"`, and with several loaded
    /// `"SfM Explorer - <first> (+N-1)"`.
    ///
    /// A first node with no file name — demo data — leaves this at the base
    /// title however many nodes follow it: it came from no file, so naming one
    /// would be a lie, and the base title is load-bearing for `ui_basic`'s
    /// Windows attach path.
    pub fn window_title(&self) -> String {
        let extra = self.scene.len().saturating_sub(1);
        #[allow(unused_mut)]
        let mut title = match self.scene.first().and_then(|node| node.file_name()) {
            Some(name) if extra > 0 => format!("{WINDOW_TITLE_BASE} - {name} (+{extra})"),
            Some(name) => format!("{WINDOW_TITLE_BASE} - {name}"),
            None => WINDOW_TITLE_BASE.to_string(),
        };
        // A window something else can drive should never look like one nothing
        // can. Appended rather than prefixed so `ui_basic`'s Windows attach
        // path, which matches on the leading base title, is unaffected.
        #[cfg(feature = "mcp")]
        if let Some(mcp) = &self.mcp {
            title.push_str(&format!(" [MCP :{}]", mcp.port));
        }
        title
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
