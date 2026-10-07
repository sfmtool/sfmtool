// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! 3D reconstruction viewer.
//!
//! Renders point clouds and camera frustums with orbit/pan/zoom camera
//! navigation, keyboard fly mode, and animated transitions.

/// Crate-visible because the scene renderer draws what it builds: the figure
/// is world geometry with no GPU in it, and the pass that uploads it lives in
/// [`crate::scene_renderer`].
pub(crate) mod bench_track;
mod camera;
/// Crate-visible because the MCP surface reads and writes the fields the HUD
/// draws, through the same list.
pub(crate) mod display;
mod framing;
mod hud;
mod input;
/// Crate-visible so the overlay text builders can be asserted on directly —
/// they are the user-facing wording of the scene stats and hover lines.
pub(crate) mod overlay;
mod righting;

#[cfg(test)]
mod tests;

pub use camera::{best_fit_fov, ViewportCamera};

use eframe::egui::{self, Color32, Pos2, Rect, Sense};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use sfmtool_core::{Camera, Se3Transform};

use crate::action_log::{ActionLog, Kind};
use crate::platform::GestureEvent;
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::PointGesture;

/// Drag/gesture zoom speed: maps pixel deltas to zoom amount.
const DRAG_ZOOM_SPEED: f64 = 0.13125;

/// Trackpad Ctrl+scroll zoom speed: maps trackpad point deltas to zoom amount.
const TRACKPAD_ZOOM_SPEED: f64 = 0.00375;

/// Mouse wheel zoom speed: maps line deltas to zoom amount.
const MOUSE_WHEEL_ZOOM_SPEED: f64 = 0.75;

/// Rotation speed of the target indicator in radians per second.
const TARGET_ROTATION_SPEED: f64 = std::f64::consts::PI / 6.0; // 30 deg/sec

/// Duration of animated camera transitions in seconds.
const CAMERA_TRANSITION_DURATION: f64 = 0.2;

/// The middle of the viewport, as a fraction of each axis, that a point must
/// already be inside for [`Viewer3D::turn_and_move_target_to`] to move to it by
/// a pan alone.
const PAN_ONLY_FRACTION: f64 = 2.0 / 3.0;

/// The middle of the viewport, as a fraction of each axis, that
/// [`Viewer3D::turn_and_move_target_to`] turns the camera to bring a point
/// inside before panning to it.
const TURN_INTO_FRACTION: f64 = 0.5;

/// How many aim-and-level steps the turn takes at most. Each step's error is a
/// small fraction of the last one's, so a handful is enough.
const TURN_STEPS: usize = 16;

/// Whether normalized device coordinates `ndc` are inside the middle
/// `fraction` of the viewport on both axes.
fn within(ndc: [f64; 2], fraction: f64) -> bool {
    ndc[0].abs() <= fraction && ndc[1].abs() <= fraction
}

/// Where a point on `bearing` from the camera lands in a viewport under
/// `orientation`, in normalized device coordinates (`[-1, 1]` on each axis,
/// `+y` up), or `None` when the bearing does not point in front of the camera.
/// `tan_y` is the tangent of half the vertical field of view.
///
/// A bearing rather than a point, so that a finite point (its offset from the
/// camera), a point at infinity (its direction) and a feature (the ray through
/// its pixel) are asked the same question: where a point lands depends only on
/// its direction from the camera.
fn ndc_of(
    orientation: UnitQuaternion<f64>,
    bearing: Vector3<f64>,
    tan_y: f64,
    aspect: f64,
) -> Option<[f64; 2]> {
    let in_camera = orientation * bearing;
    let depth = -in_camera.z;
    if depth <= 1e-10 {
        return None;
    }
    Some([
        in_camera.x / (depth * tan_y * aspect),
        in_camera.y / (depth * tan_y),
    ])
}

/// The orientation, level with `up`, that a camera at `start` turns to in place
/// to bring what lies on `bearing` inside the middle `fraction` of the viewport
/// on both axes, turning little more than that needs.
///
/// Each step aims the bearing at the nearest place inside that region by the
/// shortest-arc rotation, then levels the camera again, which moves it a
/// little; the steps repeat until it is inside. The levelling can leave one
/// axis a little way inside the edge rather than on it. A bearing behind the
/// camera is looked along directly, since no nearest place in the region is
/// defined for it.
///
/// A free function of the start state rather than a method on the viewport,
/// because a camera view can be framed before the viewport is there: the turn
/// that brings a feature into a photograph's view starts from that
/// photograph's pose and lens.
fn orientation_bringing_into(
    start: UnitQuaternion<f64>,
    up: Vector3<f64>,
    bearing: Vector3<f64>,
    tan_y: f64,
    aspect: f64,
    fraction: f64,
) -> UnitQuaternion<f64> {
    let mut orientation = start;
    for _ in 0..TURN_STEPS {
        let Some(ndc) = ndc_of(orientation, bearing, tan_y, aspect) else {
            return Camera::orientation_from_forward(bearing.normalize(), up);
        };
        if within(ndc, fraction) {
            break;
        }
        // Aim a little inside the edge, so that the levelling does not leave
        // the bearing just outside it.
        let edge = fraction * (1.0 - 1e-3);
        let aim = Vector3::new(
            ndc[0].clamp(-edge, edge) * tan_y * aspect,
            ndc[1].clamp(-edge, edge) * tan_y,
            -1.0,
        );
        let Some(turn) = UnitQuaternion::rotation_between(&(orientation * bearing), &aim) else {
            break;
        };
        let forward = (turn * orientation).inverse() * Vector3::new(0.0, 0.0, -1.0);
        orientation = Camera::orientation_from_forward(forward, up);
    }
    orientation
}

/// The viewport camera's whole state, as [`Viewer3D::pose`] reads it and
/// [`Viewer3D::ease_from`] eases away from.
pub(crate) struct ViewPose {
    position: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    distance: f64,
    fov: f64,
    world_up: Vector3<f64>,
    camera_view: Option<CameraViewMode>,
}

/// Animated camera transition for smooth navigation.
///
/// Interpolates camera state over ~200ms using slerp (orientation) + lerp
/// (position, distance, FOV) with smoothstep easing. Used for orbit target
/// changes, zoom-to-fit, and camera view transitions.
struct CameraTransition {
    start_position: Point3<f64>,
    end_position: Point3<f64>,
    start_orientation: UnitQuaternion<f64>,
    end_orientation: UnitQuaternion<f64>,
    start_distance: f64,
    end_distance: f64,
    start_fov: f64,
    end_fov: f64,
    start_world_up: Vector3<f64>,
    end_world_up: Vector3<f64>,
    /// When the ease began, on egui's clock. `None` for one started outside a
    /// frame, by the wire, which has no clock to read: the next frame that
    /// animates it stamps its own time here, so the ease runs its whole length
    /// however long the window sat idle before that frame.
    start_time: Option<f64>,
    /// Camera view mode to activate when the transition completes.
    pending_camera_view: Option<CameraViewMode>,
    /// Whether to trigger a target flash on completion.
    flash_on_complete: bool,
}

impl CameraTransition {
    /// Returns the interpolation progress with smoothstep easing, or None if complete.
    fn progress(&self, current_time: f64) -> Option<f64> {
        let elapsed = current_time - self.start_time.unwrap_or(current_time);
        if elapsed >= CAMERA_TRANSITION_DURATION {
            return None; // transition complete
        }
        let t = elapsed / CAMERA_TRANSITION_DURATION;
        // Smoothstep: 3t² - 2t³ (ease-in/ease-out)
        Some(t * t * (3.0 - 2.0 * t))
    }
}

/// One reconstruction camera's pose in the **shared world space**, given the
/// owning node's transform: `(world-to-camera rotation, camera centre)`.
///
/// The centre is the transform applied to the stored centre; the rotation loses
/// the transform's rotation on the world side, exactly as
/// [`Se3Transform::apply_to_camera_pose`] derives it (`q' = q · conj(q_world)`).
/// The uniform scale does not touch the rotation.
pub(crate) fn transformed_pose(
    image: &sfmtool_core::SfmrImage,
    transform: &Se3Transform,
) -> (UnitQuaternion<f64>, Point3<f64>) {
    let centre = transform.apply_to_point(&image.camera_center());
    let rotation = image.quaternion_wxyz * transform.rotation.as_nalgebra().inverse();
    (rotation, centre)
}

/// `Looking through IMG_0007.jpg` — the entry a deliberate camera-view entry
/// writes, wherever it came from (a double-click, `Z`, the Scene tree, the
/// track panel).
fn record_camera_view(log: &mut ActionLog, image: ImageRef, node: &SceneNode) {
    let name = node
        .recon()
        .image_table
        .images
        .get(image.index())
        .map(|i| i.name.as_str())
        .unwrap_or("?");
    log.record(Kind::View, format!("Looking through {name}"));
}

/// Computed end state for a camera view switch.
struct SwitchCameraViewState {
    position: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    distance: f64,
    world_up: Vector3<f64>,
    camera_view: CameraViewMode,
}

/// Where entering camera view mode on one image lands the viewport camera.
///
/// [`SwitchCameraViewState`] plus a field of view: *switching* between two
/// camera views preserves the current FOV so the free-look framing survives the
/// step, while *entering* one fits the image's own intrinsics into the viewport.
struct EnterCameraViewState {
    position: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    distance: f64,
    world_up: Vector3<f64>,
    fov: f64,
    camera_view: CameraViewMode,
}

/// Camera view mode state: active when viewing through a selected camera.
///
/// Stores the SfM camera's world-from-camera rotation so the background mesh
/// can be rendered with the correct relative rotation during free-look navigation.
#[derive(Clone)]
pub struct CameraViewMode {
    /// The image being viewed through (stable across selection changes). A ref
    /// rather than an index: the background image and the hidden-frustum test
    /// both key off it, and a bare index would follow a file replacement onto
    /// whatever now sits at that position.
    pub image: ImageRef,
    /// World-from-camera rotation of the SfM camera being viewed.
    /// Used to compute the relative view rotation for the BG mesh.
    pub r_world_from_cam: UnitQuaternion<f64>,
}

/// 3D viewer state and rendering.
pub struct Viewer3D {
    /// Viewport camera.
    pub camera: ViewportCamera,
    /// Whether the view has been initialized to frame the data.
    pub view_initialized: bool,
    /// Camera view mode — active when viewing through a selected camera.
    pub camera_view: Option<CameraViewMode>,
    /// The camera being moved by hand, when one is: camera view with the camera
    /// coming along. Held beside [`Viewer3D::camera_view`] because it is the
    /// same mode with one bit flipped, and it is only ever entered from it.
    /// See [`crate::camera_lock`].
    pub camera_lock: Option<crate::camera_lock::CameraLock>,
    /// Last known panel size in physical pixels, used by SceneRenderer
    /// to create offscreen textures at the correct resolution.
    pub panel_size: [u32; 2],
    /// Mouse position in texture pixels (set each frame from hover pos).
    pub hover_pixel: Option<[u32; 2]>,
    /// Whether the Alt key is currently held.
    pub alt_held: bool,
    /// Supernova effect activation level (0.0 = off, 1.0 = fully on).
    pub supernova_active: f32,
    /// Target's view-space position [x, y, z] for the supernova effect (z is positive = in front).
    pub supernova_view_pos: [f32; 3],
    /// Elapsed time for supernova wave animation (seconds).
    pub supernova_time: f32,
    /// Current rotation angle of the target indicator (radians).
    pub target_indicator_rotation: f64,
    /// Timestamp of the last target change for the flash animation.
    pub target_flash_start: Option<f64>,
    /// Whether the target indicator should be visible this frame.
    pub target_indicator_visible: bool,
    /// Flash animation radius scale (1.0 = normal, >1.0 during flash).
    pub target_indicator_radius_scale: f32,
    /// Flash animation alpha scale (1.0 = normal, >1.0 during flash).
    pub target_indicator_alpha_scale: f32,
    /// Pending click request: screen pixel position [x, y] in texture pixels.
    /// Both depth and entity pick are read back from this single click.
    pub pending_click: Option<[u32; 2]>,
    /// Whether the pending click was Alt+Click (sets orbit target from depth).
    pub pending_click_is_alt: bool,
    /// Whether the pending click was a double-click (enters camera view mode).
    pub pending_click_is_double: bool,
    /// Pixels per point at the time of the click request.
    pick_ppp: f32,
    /// Screen rect at the time of the click request.
    pick_rect: Rect,
    /// Whether any WASD/RF/QE fly key is currently held.
    fly_keys_held: bool,
    /// Whether a drag was initiated while fly keys were held (locks nodal pan for the drag).
    fly_drag_locked: bool,
    /// Active animated camera transition (orbit target, zoom-to-fit, camera view).
    target_transition: Option<CameraTransition>,
    /// Whether the viewport HUD is expanded. Open at launch — the controls are
    /// the point of the panel, and a viewport that starts by hiding them just
    /// trades a menu round-trip for a click. Never persisted across runs.
    pub hud_open: bool,
    /// Whether the HUD was opened by a click on the gear while the viewport
    /// was too small for the expanded panel. The size rule decides only
    /// whether the HUD opens on its own; an explicit click opens it at any
    /// size, and it stays open until the close button is clicked. Never
    /// persisted across runs.
    hud_opened_by_click: bool,
    /// Whether the view turns itself back to Z-up whenever it is not looking
    /// through a camera: the HUD's **Maintain Z-up**. On at launch; Q and E
    /// turn it off, since rolling the view is a request for a view that is not
    /// level. Never persisted across runs. See [`Self::right_toward_z_up`].
    pub maintain_z_up: bool,
    /// How fast [`Self::right_toward_z_up`] turned `world_up` on the last
    /// frame, in radians per second, and zero when it did not turn it. The only
    /// state the turn carries between frames.
    righting_speed: f64,
    /// Screen rect the HUD occupied on the last frame it was built — the gear
    /// when collapsed, gear plus panel when expanded. Every viewport input path
    /// that cannot rely on egui's layer arbitration (scroll, gestures, pinch)
    /// excludes it geometrically. `None` until the HUD has been built once.
    pub hud_rect: Option<Rect>,
    /// What the viewport's context menu stands on, recorded on the frame the
    /// secondary click landed and read on every later frame the menu is laid
    /// out over -- by which time the pointer has moved off whatever the user
    /// named, and whatever the pick reports under it is somebody else's.
    menu_target: Option<MenuTarget>,
    /// What the menu asked of the app about a point, drained by `app.rs` after
    /// the frame.
    ///
    /// A single slot, because the two things it carries happen on different
    /// frames: a menu opens on the frame of the click and an entry is chosen on
    /// a later one.
    pub point_menu: Option<PointGesture>,
    /// Which reframe the patch menu asked for, and on which node, drained by
    /// `app.rs` after the frame for the reason [`Self::point_menu`] is.
    pub(crate) patch_menu: Option<(ReconId, crate::display_transform::PatchReframe)>,
    /// The figure the focused item draws in the scene, as of the last
    /// frame this panel was shown, or `None` when there is nothing to draw.
    ///
    /// Built here rather than in the upload because it is a reading of the
    /// viewport camera -- the arrowhead's barbs turn to face the eye -- and the
    /// eye is this panel's. `app.rs` uploads it and the pass draws it, on the
    /// frame-behind cadence every other camera-derived value already runs on.
    pub(crate) bench_figure: Option<bench_track::Figure>,
    /// The bench figure's handle the pointer has hold of, while it has one.
    ///
    /// Held here rather than derived each frame for the reason the Image Detail
    /// panel's is: a drag is a gesture and not a state of the pointer -- what is
    /// being dragged was decided when the button went down, and the pointer
    /// wanders off the handle the moment it starts moving.
    pub(crate) bench_drag: Option<bench_track::Drag>,
    /// What a gesture over the bench figure asked of the app, drained by
    /// `dock.rs` after the frame.
    ///
    /// Held here rather than carried out in the viewport for the reason the
    /// point menu's gesture is: both need the state mutably, and the state is
    /// borrowed out of for the whole of this call.
    pub(crate) bench_gesture: Option<bench_track::BenchGesture>,
    /// Where each of the context menu's entries was drawn on the frame just
    /// past, empty on a frame with no menu up.
    ///
    /// Recorded in the production path rather than behind a test flag, for the
    /// reason the Scene tree records its own row rects: what a headless test
    /// aims at is then the very layout the window produces, rather than a
    /// second layout built to be aimed at.
    pub(crate) menu_entry_rects: Vec<(&'static str, Rect)>,
}

/// What the entry that stages a point's track on the bench is called, in both
/// menus that offer it and in the tests that aim at them.
///
/// The Image Detail panel's feature menu offers the same entry under the same
/// name ([`crate::image_detail`]), so the two cannot drift.
pub const EDIT_ON_BENCH_LABEL: &str = "Edit on Bench";

/// What the HUD's checkbox for [`Viewer3D::maintain_z_up`] is called, and the
/// word its Action Log entries open with, whether the checkbox or Q and E
/// changed it.
pub const MAINTAIN_Z_UP_LABEL: &str = "Maintain Z-up";

/// What the entry that re-solves one point is called, in the menu and in the
/// tests that aim at it.
pub const RETRIANGULATE_POINT_LABEL: &str = "Retriangulate Point";

/// What the patch menu's entry that adopts the patch's whole frame is called,
/// in the menu and in the tests that aim at it.
pub const SET_TO_ORIGIN_LABEL: &str = "Set to Origin";

/// What the patch menu's entry that tips the patch's normal onto `+Z` is
/// called.
pub const ALIGN_NORMAL_TO_Z_LABEL: &str = "Align Normal to Z";

/// What the patch menu's entry that moves the patch's centre to the origin is
/// called.
pub const TRANSLATE_TO_ORIGIN_LABEL: &str = "Translate to Origin";

/// What the patch menu's entry that drops the patch's centre onto `z = 0` is
/// called.
pub const TRANSLATE_TO_XY_PLANE_LABEL: &str = "Translate to XY Plane";

/// Why the patch menu's entries are greyed on a track at infinity.
const PATCH_AT_INFINITY_HINT: &str =
    "This track is at infinity: its patch is a bearing with no place, so it has no centre to move.";

/// What the viewport's context menu stands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuTarget {
    /// The focused item's patch, on the node whose figure was drawn.
    Patch(ReconId),
    /// A 3D point the pick reported under the cursor.
    Point(PointRef),
}

impl Default for Viewer3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Viewer3D {
    /// Creates a new 3D viewer.
    pub fn new() -> Self {
        Self {
            camera: ViewportCamera::default(),
            view_initialized: false,
            camera_view: None,
            camera_lock: None,
            panel_size: [0, 0],
            hover_pixel: None,
            alt_held: false,
            supernova_active: 0.0,
            supernova_view_pos: [0.0, 0.0, 5.0],
            supernova_time: 0.0,
            target_indicator_rotation: 0.0,
            target_flash_start: None,
            target_indicator_visible: false,
            target_indicator_radius_scale: 1.0,
            target_indicator_alpha_scale: 1.0,
            pending_click: None,
            pending_click_is_alt: false,
            pending_click_is_double: false,
            pick_ppp: 1.0,
            pick_rect: Rect::NOTHING,
            fly_keys_held: false,
            fly_drag_locked: false,
            target_transition: None,
            hud_open: true,
            hud_opened_by_click: false,
            maintain_z_up: true,
            righting_speed: 0.0,
            hud_rect: None,
            menu_target: None,
            point_menu: None,
            patch_menu: None,
            bench_figure: None,
            bench_drag: None,
            bench_gesture: None,
            menu_entry_rects: Vec::new(),
        }
    }

    /// The viewport's context menu: right-click the focused item's patch, a
    /// point's dot or a point's own patch, and this is what opens.
    ///
    /// **A right *click* opens it and a right *drag* does not.** The viewport's
    /// right-drag is its zoom, read from the raw platform button state in
    /// [`Self::handle_drag`], and that state says nothing about press and
    /// release. So the menu hangs off egui's own `clicked_by(Secondary)`, whose
    /// drag threshold is what tells a click from a drag, exactly as the Image
    /// Detail overlay's menu does.
    ///
    /// **One menu with two sets of entries**, never two popups: both would hang
    /// off the one response this panel produces, and a popup takes its identity
    /// from that response. Which set is decided when the click lands, and **the
    /// patch wins**: the figure is drawn on top of the cloud, so a right-click
    /// the square covers ([`bench_track::Handles::covers`]) means the square,
    /// for the reason a primary click one of its handles catches does not reach
    /// the points under it. Otherwise the point the pick reported is the
    /// target.
    ///
    /// The target is latched in [`Self::menu_target`] because the entries are
    /// laid out on later frames by which time the pointer has moved. A
    /// secondary click on anything else clears it, so no menu opens over empty
    /// space.
    fn show_viewport_menu(
        &mut self,
        response: &egui::Response,
        rect: Rect,
        hover_pick: Option<crate::scene_renderer::PickTarget>,
        busy: Option<&str>,
        bench: Option<&bench_track::BenchTrack<'_>>,
    ) {
        if response.clicked_by(egui::PointerButton::Secondary) {
            let on_patch = bench
                .zip(self.bench_figure.as_ref())
                .zip(response.interact_pointer_pos())
                .filter(|((_, figure), pos)| {
                    bench_track::Handles::project(figure, &self.camera, rect, None).covers(*pos)
                })
                .map(|((bench, _), _)| bench.node);
            self.menu_target = match (on_patch, hover_pick) {
                (Some(node), _) => Some(MenuTarget::Patch(node)),
                (None, Some(crate::scene_renderer::PickTarget::Point(point))) => {
                    Some(MenuTarget::Point(point))
                }
                _ => None,
            };
            // Selecting on open rather than on choice: the menu is about this
            // point, and the panels beside the viewport should be saying so
            // while it stands open.
            if let Some(MenuTarget::Point(point)) = self.menu_target {
                self.point_menu = Some(PointGesture::Opened(point));
            }
        }
        self.menu_entry_rects.clear();
        match self.menu_target {
            None => {}
            Some(MenuTarget::Point(point)) => self.show_point_entries(response, point, busy),
            Some(MenuTarget::Patch(node)) => {
                // The patch as it stands now. A menu left up across an undo that
                // took the patch away has nothing to stand on, and closes.
                let placement = bench
                    .filter(|bench| bench.node == node)
                    .and_then(|bench| bench_track::placement_of(bench.track));
                match placement {
                    Some(placement) => {
                        let refusal = busy.map(str::to_string).or_else(|| {
                            (placement.w == 0.0).then(|| PATCH_AT_INFINITY_HINT.to_string())
                        });
                        self.show_patch_entries(response, node, refusal.as_deref());
                    }
                    None => self.menu_target = None,
                }
            }
        }
    }

    /// The four reframes, on the node whose patch the menu opened on.
    ///
    /// Greyed with `refusal` rather than hidden when they cannot run, as every
    /// other entry that makes a version is: a busy node, or a track at infinity,
    /// whose patch is a bearing with no place.
    fn show_patch_entries(
        &mut self,
        response: &egui::Response,
        node: ReconId,
        refusal: Option<&str>,
    ) {
        use crate::display_transform::PatchReframe;
        let mut rects = Vec::with_capacity(PatchReframe::ALL.len());
        crate::context_menu::on_secondary_click(response).show(|ui| {
            for mode in PatchReframe::ALL {
                let button = egui::Button::new(mode.label());
                let entry = match refusal {
                    None => ui.add(button).on_hover_text(mode.hint()),
                    Some(why) => ui.add_enabled(false, button).on_disabled_hover_text(why),
                };
                rects.push((mode.label(), entry.rect));
                if entry.clicked() {
                    self.patch_menu = Some((node, mode));
                    ui.close();
                }
            }
        });
        self.menu_entry_rects = rects;
    }

    /// The point's two entries.
    fn show_point_entries(
        &mut self,
        response: &egui::Response,
        point: PointRef,
        busy: Option<&str>,
    ) {
        let mut rects = Vec::with_capacity(2);
        crate::context_menu::on_secondary_click(response).show(|ui| {
            let mut entry = |ui: &mut egui::Ui, text: &'static str, hint: &str| -> bool {
                let button = egui::Button::new(text);
                let (response, clicked) = match busy {
                    None => {
                        let response = ui.add(button);
                        let clicked = response.clicked();
                        (response.on_hover_text(hint), clicked)
                    }
                    Some(why) => (
                        ui.add_enabled(false, button).on_disabled_hover_text(why),
                        false,
                    ),
                };
                rects.push((text, response.rect));
                clicked
            };
            if entry(
                ui,
                EDIT_ON_BENCH_LABEL,
                "Put this point's track on the bench and open Track View on it.",
            ) {
                self.point_menu = Some(PointGesture::EditOnBench(point));
                ui.close();
            }
            if entry(
                ui,
                RETRIANGULATE_POINT_LABEL,
                "Re-solve this point from its own observations at these poses and these \
                 lenses, as one version.",
            ) {
                self.point_menu = Some(PointGesture::Retriangulate(point));
                ui.close();
            }
        });
        self.menu_entry_rects = rects;
    }

    /// Shows the 3D viewer UI and renders the reconstruction.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        // The selected node: its reconstruction is what the viewport's own
        // bindings work within, and its transform is what puts that
        // reconstruction where it is drawn.
        node: &SceneNode,
        // Every loaded node, for the overlays that describe the whole scene:
        // the stats line and the hover text's reconstruction label.
        scene: &[SceneNode],
        // The soloed node, if any — the stats line counts what is drawn, and
        // a solo is half of what decides that.
        solo: Option<ReconId>,
        // The image selection, which `Z` looks through and the overlays draw.
        // Read-only: nothing the viewport draws moves it, and the keys that do
        // move it go through `AppState::select_image` at the app level.
        selected_image: Option<ImageRef>,
        show_grid: bool,
        length_scale: f32,
        status_message: Option<&str>,
        gesture_events: &[GestureEvent],
        scroll_input: &crate::platform::ScrollInput,
        show_controls_help: bool,
        show_fps: bool,
        // The HUD's **Target indicator**: draw the orbit target all the time,
        // not only while Alt is held.
        show_target_indicator: bool,
        scene_texture_id: Option<egui::TextureId>,
        hover_depth: Option<f32>,
        hover_pick: Option<crate::scene_renderer::PickTarget>,
        // Why an edit of the selected node is refused right now, or `None`:
        // what the point menu greys its entries with. Read before the node is
        // borrowed out of the scene, because the answer is the whole state's.
        busy: Option<&str>,
        // The node's bench, as much of it as this layer draws: the focused
        // item, the value its marks unproject through and the node's
        // transform. Read out by the dock for the reason `busy` above is --
        // the state is borrowed mutably further down this same call.
        bench: Option<bench_track::BenchTrack<'_>>,
        // The viewport's own keyboard bindings — `Z` and `Home` — are
        // discrete commands, so they record what they did. Taken as a separate
        // `&mut` rather than through `AppState` because `node` and `scene`
        // above are borrowed out of the same state for the whole call.
        log: &mut ActionLog,
    ) {
        let reconstruction = node.recon();
        // Allocate the entire available space for the 3D view.
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        self.alt_held = ui.input(|i| i.modifiers.alt);

        let current_time = ui.input(|i| i.time);

        // Keyboard arbitration: a HUD `DragValue` in text-entry mode owns the
        // keyboard, and WASD typed into it must not also fly the camera. egui
        // only reports focus for widgets that actually consume text (text edits
        // and `DragValue`s being typed into), so clicking a HUD checkbox does
        // not disarm the fly keys.
        let keyboard_free = !ui.ctx().egui_wants_keyboard_input();

        // Check fly key state early — used by drag handling and supernova suppression.
        // With Ctrl/Cmd down the letter belongs to a shortcut (Ctrl+S saves,
        // Ctrl+D duplicates), and the `key_down` egui still reports for it must
        // not also move the camera. Shift stays free: it is the fly sprint.
        self.fly_keys_held = keyboard_free
            && ui.input(|i| !(i.modifiers.command || i.modifiers.ctrl))
            && ui.input(|i| {
                i.key_down(egui::Key::W)
                    || i.key_down(egui::Key::A)
                    || i.key_down(egui::Key::S)
                    || i.key_down(egui::Key::D)
                    || i.key_down(egui::Key::R)
                    || i.key_down(egui::Key::F)
                    || i.key_down(egui::Key::Q)
                    || i.key_down(egui::Key::E)
            });
        let fly_keys_held = self.fly_keys_held;

        // Animate supernova activation (200ms fade)
        let dt = ui.input(|i| i.stable_dt);
        self.supernova_time = current_time as f32;
        let fade_speed = 5.0; // 1.0 / 0.2 seconds
        if self.alt_held || show_target_indicator {
            self.supernova_active = (self.supernova_active + dt * fade_speed).min(1.0);
        } else {
            self.supernova_active = (self.supernova_active - dt * fade_speed).max(0.0);
        }
        if self.supernova_active > 0.0 && self.supernova_active < 1.0 {
            ui.ctx().request_repaint(); // keep animating the fade
        }

        // Animate camera transition (smooth movement for orbit target, zoom, camera view)
        self.animate_transition(ui, current_time);

        // Initialize view to frame all points on first show
        if !self.view_initialized && !reconstruction.point_set.points.is_empty() {
            let aspect = rect.width() as f64 / rect.height() as f64;
            let points = crate::scene::FitPoints::of(node);
            if let Some(end) = self.camera.compute_fit(&points, aspect, self.maintain_z_up) {
                self.camera.apply_fit(&end);
            }
            self.view_initialized = true;
        }

        // --- The bench figure's handles, before the viewport's own input ---
        //
        // A drag that began on a handle is an edit of the track and must not
        // also orbit, pan or zoom: the pointer can only mean one of the two, and
        // what it means was decided where the button went down.
        let bench_owns_pointer = self.update_bench_drag(ui, &response, rect, bench.as_ref());

        // Handle all input.
        //
        // Drag and click need no explicit HUD guard: the HUD is an `egui::Area`
        // on a layer above this panel, and egui's hit test keeps only the
        // top-most layer under the pointer, so `response.dragged()`,
        // `clicked()` and `hovered()` are all false while the pointer is over
        // it. Verified against egui 0.34 in `hud/tests.rs`.
        self.handle_drag(ui, &response, rect, fly_keys_held, bench_owns_pointer);

        // Scroll, pinch and platform gestures cannot rely on that. They gate on
        // `platform::pointer_in_rect`, a raw geometric containment test that on
        // Windows reads the OS cursor position and knows nothing about egui
        // layers, so the HUD has to be excluded by geometry.
        let over_hud = self
            .hud_rect
            .is_some_and(|hud| crate::platform::pointer_in_rect(ui.ctx(), hud));
        let pointer_over = !over_hud && crate::platform::pointer_in_rect(ui.ctx(), rect);
        if pointer_over {
            self.handle_scroll(rect, scroll_input, fly_keys_held);
        }
        self.handle_pinch(ui, pointer_over);

        let gesture_events = if pointer_over { gesture_events } else { &[] };
        self.handle_gestures(ui, gesture_events, rect, fly_keys_held);
        self.handle_fly_keys(ui, fly_keys_held, log);
        if keyboard_free {
            self.handle_keyboard(ui, rect, node, selected_image, log);
        }
        // The figure is on top, so a click one of its marks catches does not
        // also reach the points under it: two selections from one click would
        // be two answers to one gesture.
        if !bench_owns_pointer {
            self.handle_click(ui, &response, rect);
        }
        self.show_viewport_menu(&response, rect, hover_pick, busy, bench.as_ref());

        // After this frame's input, so the step starts from the view the input
        // left and the frame draws the result.
        if self.right_toward_z_up(dt as f64) {
            ui.ctx().request_repaint();
        }

        // Record mouse position in texture pixels for GPU depth readback
        let ppp = ui.ctx().pixels_per_point();
        self.hover_pixel = response.hover_pos().map(|pos| {
            let px = ((pos.x - rect.left()) * ppp) as u32;
            let py = ((pos.y - rect.top()) * ppp) as u32;
            [px, py]
        });

        // Compute target view-space position for supernova effect
        let target = self.camera.target();
        let view_pos = self.camera.world_to_view(&target);
        self.supernova_view_pos = [
            view_pos.x as f32,
            view_pos.y as f32,
            -view_pos.z as f32, // positive = in front of camera
        ];

        // Track panel size in physical pixels for the scene renderer
        let ppp = ui.ctx().pixels_per_point();
        self.panel_size = [(rect.width() * ppp) as u32, (rect.height() * ppp) as u32];

        // Background: GPU-rendered scene texture or solid color fallback
        if let Some(tex_id) = scene_texture_id {
            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            let mut mesh = egui::Mesh::with_texture(tex_id);
            mesh.add_rect_with_uv(rect, uv, Color32::WHITE);
            painter.add(egui::Shape::mesh(mesh));
        } else {
            painter.rect_filled(rect, 0.0, Color32::from_rgb(30, 30, 35));
        }

        // Draw grid if enabled
        if show_grid {
            self.draw_grid(&painter, rect, length_scale);
        }

        // Draw axis indicator in corner
        self.draw_axis_indicator(&painter, rect);

        // Update target indicator state for GPU rendering
        self.update_target_indicator_state(ui, show_target_indicator);

        // The focused item, as the scene geometry the next frame's
        // upload draws. After the camera has been moved by this frame's input,
        // because the arrowhead is squared to the eye.
        //
        // While a handle is held what is drawn is the **preview**: the track the
        // release would push, through the same function the release goes
        // through, so there is one answer rather than a drawn guess and a
        // pushed result.
        let eye = self.camera.position();
        let drag = self.bench_drag;
        self.bench_figure = bench.and_then(|bench| {
            let previewed = drag
                .filter(|drag| drag.node == bench.node)
                .and_then(|drag| drag.edit(bench_track::placement_of(bench.track)?))
                .and_then(|edit| {
                    crate::bench::geometry::apply(bench.track, bench.edited, &edit).ok()
                })
                .map(|(next, _)| next);
            let shown = bench_track::BenchTrack {
                track: previewed.as_ref().unwrap_or(bench.track),
                ..bench
            };
            bench_track::figure(&shown, eye)
        });

        // The lock banner, over the scene and under nothing: it is what the
        // viewport is in the middle of.
        self.draw_lock_banner(&painter, rect, node);

        // Draw info overlay
        let fps = 1.0 / ui.input(|i| i.predicted_dt as f64);
        self.draw_info_overlay(
            &painter,
            rect,
            scene,
            solo,
            show_controls_help,
            show_fps,
            status_message,
            hover_depth,
            hover_pick,
            fps,
        );
    }

    /// Animates the camera transition (smooth movement for orbit target, zoom, camera view).
    fn animate_transition(&mut self, ui: &egui::Ui, current_time: f64) {
        if let Some(transition) = self.target_transition.as_mut() {
            transition.start_time.get_or_insert(current_time);
        }
        if let Some(ref transition) = self.target_transition {
            if let Some(t) = transition.progress(current_time) {
                self.camera.camera.position = transition.start_position
                    + (transition.end_position - transition.start_position) * t;
                self.camera.camera.orientation = transition
                    .start_orientation
                    .slerp(&transition.end_orientation, t);
                self.camera.camera.target_distance = transition.start_distance
                    + (transition.end_distance - transition.start_distance) * t;
                self.camera.fov =
                    transition.start_fov + (transition.end_fov - transition.start_fov) * t;
                self.camera.world_up = transition
                    .start_world_up
                    .lerp(&transition.end_world_up, t)
                    .normalize();
                ui.ctx().request_repaint(); // keep animating
            }
        }
        // Handle transition completion separately to take ownership
        if self
            .target_transition
            .as_ref()
            .is_some_and(|t| t.progress(current_time).is_none())
        {
            let transition = self.target_transition.take().unwrap();
            let flash = transition.flash_on_complete;
            self.land_transition(transition);
            if flash {
                self.target_flash_start = Some(current_time);
            }
        }
    }

    /// Assign a transition's end state to the camera.
    fn land_transition(&mut self, transition: CameraTransition) {
        self.camera.camera.position = transition.end_position;
        self.camera.camera.orientation = transition.end_orientation;
        self.camera.camera.target_distance = transition.end_distance;
        self.camera.fov = transition.end_fov;
        self.camera.world_up = transition.end_world_up;
        self.camera_view = transition.pending_camera_view;
    }

    /// Where the viewport camera stands now: what [`Self::ease_from`] eases
    /// away from.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn pose(&self) -> ViewPose {
        ViewPose {
            position: self.camera.camera.position,
            orientation: self.camera.camera.orientation,
            distance: self.camera.camera.target_distance,
            fov: self.camera.fov,
            world_up: self.camera.world_up,
            camera_view: self.camera_view.clone(),
        }
    }

    /// Put the camera back at `from` and ease from there to where it stands
    /// now, with the transition the gestures use.
    ///
    /// What the wire's `set_view` does when asked to animate: the form has
    /// already put the camera at its end state, by the same code that answers
    /// it instantly, so the animated and the instant call end in one place.
    /// The ease starts on the next frame the viewer draws.
    ///
    /// A camera view being entered or left is dropped for the length of the
    /// ease and taken up at its end, as [`Self::enter_camera_view`] does. One
    /// that holds on both ends of the ease, looking through the same image, is
    /// kept throughout, so a turn or a change of field of view inside camera
    /// view does not leave it for 200 ms.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn ease_from(&mut self, from: ViewPose) {
        let end = self.pose();
        let same_view = match (&from.camera_view, &end.camera_view) {
            (Some(a), Some(b)) => a.image == b.image,
            _ => false,
        };
        self.camera.camera.position = from.position;
        self.camera.camera.orientation = from.orientation;
        self.camera.camera.target_distance = from.distance;
        self.camera.fov = from.fov;
        self.camera.world_up = from.world_up;
        self.camera_view = if same_view {
            end.camera_view.clone()
        } else {
            None
        };
        self.target_transition = Some(CameraTransition {
            start_position: from.position,
            end_position: end.position,
            start_orientation: from.orientation,
            end_orientation: end.orientation,
            start_distance: from.distance,
            end_distance: end.distance,
            start_fov: from.fov,
            end_fov: end.fov,
            start_world_up: from.world_up,
            end_world_up: end.world_up,
            start_time: None,
            pending_camera_view: end.camera_view,
            flash_on_complete: false,
        });
    }

    /// Land the transition in progress at once, where it would have ended, with
    /// no ease and no flash of the target indicator.
    ///
    /// What the MCP surface's `set_view` does after a gesture's own method has
    /// started one, so the agent's view is the gesture's end state computed by
    /// the gesture's own code, and a screenshot taken straight afterward shows
    /// it. Nothing happens when no transition is running.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn finish_transition(&mut self) {
        if let Some(transition) = self.target_transition.take() {
            self.land_transition(transition);
        }
    }

    /// Applies a depth pick result to set the orbit target with smooth animation.
    ///
    /// Called from `main.rs` after the depth readback completes. Instead of
    /// snapping instantly, starts an animated transition (slerp + lerp with
    /// smoothstep easing over ~200ms).
    pub fn apply_pick_result(&mut self, depth: f32, click_pixel: [u32; 2], current_time: f64) {
        if depth <= 0.0 {
            return;
        }

        let [px, py] = click_pixel;

        // Convert texture pixels back to screen coordinates for unprojection
        let screen_x = self.pick_rect.left() + px as f32 / self.pick_ppp;
        let screen_y = self.pick_rect.top() + py as f32 / self.pick_ppp;

        let world_point = self
            .camera
            .unproject(screen_x, screen_y, depth as f64, self.pick_rect);

        // Compute the end state: orientation and distance for the new target
        let direction = world_point - self.camera.camera.position;
        let new_distance = direction.norm();
        if new_distance < 1e-10 {
            return; // target too close to camera, would produce NaN
        }
        let new_forward = direction / new_distance;
        let new_distance = new_distance.max(0.1);

        let end_orientation = Camera::orientation_from_forward(new_forward, self.camera.world_up);

        self.start_transition(
            self.camera.camera.position,
            end_orientation,
            new_distance,
            self.camera.fov,
            self.camera.world_up,
            None,
            true,
            current_time,
        );
    }

    /// Move the orbit target onto `point` by the smallest camera motion that
    /// does it, with the same animated transition as [`Self::apply_pick_result`].
    ///
    /// The orientation is kept, so the motion is a pan parallel to the view
    /// plane: the camera moves by the component of the offset to `point` that
    /// is perpendicular to the view direction, and the orbit distance becomes
    /// the remaining component, the point's depth. The point slides to the
    /// centre of the viewport and keeps its apparent size. What a double-click
    /// on a point does after putting it on the bench.
    ///
    /// Returns whether a transition was started. It is not when a camera is in
    /// hand, since every viewport motion then moves the held camera and a
    /// double-click is not a request to move it; nor when `point` is not in
    /// front of the camera, where no pan brings it onto the view axis.
    pub(crate) fn move_target_to(&mut self, point: Point3<f64>, current_time: f64) -> bool {
        if self.camera_lock.is_some() {
            return false;
        }
        self.pan_target_onto(point, self.camera.camera.orientation, current_time)
    }

    /// Move the orbit target onto `point`, turning the camera first when the
    /// point is far from the middle of the viewport, with one animated
    /// transition to the end state.
    ///
    /// A point already inside the middle [`PAN_ONLY_FRACTION`] of the viewport
    /// on both axes is moved to as [`Self::move_target_to`] moves to it. Any
    /// other point, including one behind the camera, first has the camera turn
    /// in place until the point is inside the middle [`TURN_INTO_FRACTION`],
    /// and the pan is then made with that orientation. What a double-click on a
    /// tracked feature in Image Detail does: the point may be anywhere in the
    /// scene, and a pan alone across a large angle would carry the camera far
    /// from where it was.
    ///
    /// Returns whether a transition was started; not when a camera is in hand,
    /// for the reason [`Self::move_target_to`] gives.
    pub(crate) fn turn_and_move_target_to(
        &mut self,
        point: Point3<f64>,
        current_time: f64,
    ) -> bool {
        if self.camera_lock.is_some() {
            return false;
        }
        let aspect = self.panel_aspect().unwrap_or(16.0 / 9.0);
        let bearing = point - self.camera.camera.position;
        let orientation = self.camera.camera.orientation;
        let in_middle = self
            .viewport_ndc(orientation, bearing, aspect)
            .is_some_and(|ndc| within(ndc, PAN_ONLY_FRACTION));
        let orientation = if in_middle {
            orientation
        } else {
            self.orientation_bringing_into(bearing, aspect, TURN_INTO_FRACTION)
        };
        self.pan_target_onto(point, orientation, current_time)
    }

    /// Turn the camera in place until `direction`, a point at infinity's
    /// bearing in the shared world space, is inside the middle
    /// [`TURN_INTO_FRACTION`] of the viewport on both axes, with the animated
    /// transition the other target moves use.
    ///
    /// What a double-click on a point at infinity does, in the viewport and in
    /// Image Detail alike. A point at infinity is a direction rather than a
    /// place, so there is no target to put on it and no pan that changes where
    /// it is drawn: the camera turns, level with `world_up`, by about as little
    /// as brings the bearing into the middle, and the position and orbit
    /// distance stay. A turn in place is a free look, so camera view is kept,
    /// as nodal pan keeps it.
    ///
    /// Returns whether a transition was started: not when the bearing is
    /// already inside the middle, and not when a camera is in hand, for the
    /// reason [`Self::move_target_to`] gives.
    pub(crate) fn turn_toward_bearing(
        &mut self,
        direction: Vector3<f64>,
        current_time: f64,
    ) -> bool {
        if self.camera_lock.is_some() {
            return false;
        }
        let aspect = self.panel_aspect().unwrap_or(16.0 / 9.0);
        let orientation = self.camera.camera.orientation;
        if self
            .viewport_ndc(orientation, direction, aspect)
            .is_some_and(|ndc| within(ndc, TURN_INTO_FRACTION))
        {
            return false;
        }
        let end_orientation = self.orientation_bringing_into(direction, aspect, TURN_INTO_FRACTION);
        let world_up = self.camera.world_up;
        self.start_transition(
            self.camera.camera.position,
            end_orientation,
            self.camera.camera.target_distance,
            self.camera.fov,
            world_up,
            self.camera_view.clone(),
            false,
            current_time,
        );
        true
    }

    /// Start the transition that ends with `orientation` and with the orbit
    /// target on `point`, reached by a pan from the current position: the
    /// camera moves by the component of its offset to `point` that is
    /// perpendicular to the end view direction, and the orbit distance becomes
    /// the rest. Returns false, starting nothing, when `point` is not in front
    /// of the camera under `orientation`.
    fn pan_target_onto(
        &mut self,
        point: Point3<f64>,
        orientation: UnitQuaternion<f64>,
        current_time: f64,
    ) -> bool {
        let position = self.camera.camera.position;
        let forward = orientation.inverse() * Vector3::new(0.0, 0.0, -1.0);
        let offset = point - position;
        let depth = offset.dot(&forward);
        if depth <= 1e-10 {
            return false;
        }
        let end_position = position + (offset - forward * depth);
        // A pan is a step away from the pose being looked through.
        self.leave_camera_view();
        self.start_transition(
            end_position,
            orientation,
            depth,
            self.camera.fov,
            self.camera.world_up,
            None,
            true,
            current_time,
        );
        true
    }

    /// Where a point on `bearing` from the camera lands in the viewport under
    /// `orientation`, with the viewport's own field of view: [`ndc_of`] for
    /// the lens on screen now.
    fn viewport_ndc(
        &self,
        orientation: UnitQuaternion<f64>,
        bearing: Vector3<f64>,
        aspect: f64,
    ) -> Option<[f64; 2]> {
        let tan_y = (self.camera.vertical_fov(aspect) / 2.0).tan();
        ndc_of(orientation, bearing, tan_y, aspect)
    }

    /// [`orientation_bringing_into`] from the camera's orientation and up as
    /// they are now, through the viewport's own field of view.
    fn orientation_bringing_into(
        &self,
        bearing: Vector3<f64>,
        aspect: f64,
        fraction: f64,
    ) -> UnitQuaternion<f64> {
        let tan_y = (self.camera.vertical_fov(aspect) / 2.0).tan();
        orientation_bringing_into(
            self.camera.camera.orientation,
            self.camera.world_up,
            bearing,
            tan_y,
            aspect,
            fraction,
        )
    }

    /// Starts a smooth animated transition to the given camera end state.
    ///
    /// Captures the current camera state as the start and interpolates over
    /// ~200ms. If a transition is already in progress, it is replaced (the
    /// current interpolated state becomes the new start).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_transition(
        &mut self,
        end_position: Point3<f64>,
        end_orientation: UnitQuaternion<f64>,
        end_distance: f64,
        end_fov: f64,
        end_world_up: Vector3<f64>,
        pending_camera_view: Option<CameraViewMode>,
        flash_on_complete: bool,
        current_time: f64,
    ) {
        self.target_transition = Some(CameraTransition {
            start_position: self.camera.camera.position,
            end_position,
            start_orientation: self.camera.camera.orientation,
            end_orientation,
            start_distance: self.camera.camera.target_distance,
            end_distance,
            start_fov: self.camera.fov,
            end_fov,
            start_world_up: self.camera.world_up,
            end_world_up,
            start_time: Some(current_time),
            pending_camera_view,
            flash_on_complete,
        });
    }

    /// Leave camera view, unless a camera is being moved.
    ///
    /// Every navigation path that would otherwise drop camera view goes through
    /// this, which is the whole of what the lock does to navigation: while a
    /// camera is in hand the viewport *is* the camera, so an input that would
    /// have left it behind moves it instead. See [`crate::camera_lock`].
    pub(crate) fn leave_camera_view(&mut self) {
        if self.camera_lock.is_none() {
            self.camera_view = None;
        }
    }

    /// One frame of **Maintain Z-up**: turn `world_up` a step of `dt` seconds
    /// toward +Z, and roll the view to stay level with it.
    ///
    /// The view direction, the camera position and the orbit target are kept;
    /// only the roll about the view direction changes, as `Home` changes it but
    /// eased over a fraction of a second. The step is chosen from the angle left
    /// and the last frame's speed alone ([`righting::step`]), so any other
    /// motion of the view between two frames is taken as found.
    ///
    /// Nothing turns, and the speed is forgotten, while the setting is off,
    /// while looking through a camera, whose own up is part of what camera view
    /// shows, and while an animated transition runs, since a transition sets
    /// `world_up` itself on every frame. A turn that one of these interrupts
    /// starts again from rest.
    ///
    /// Returns whether the view is still turning, so the caller keeps frames
    /// coming.
    pub(crate) fn right_toward_z_up(&mut self, dt: f64) -> bool {
        if !self.maintain_z_up
            || self.camera_view.is_some()
            || self.camera_lock.is_some()
            || self.target_transition.is_some()
        {
            self.righting_speed = 0.0;
            return false;
        }
        if self.camera.world_up == Vector3::z() {
            self.righting_speed = 0.0;
            return false;
        }
        let forward = self.camera.camera.forward();
        let (up, speed) = righting::step(self.camera.world_up, forward, self.righting_speed, dt);
        self.camera.world_up = up;
        self.righting_speed = speed;
        // Looking straight along the new up there is no roll to level to, and
        // re-deriving the orientation there would pick an arbitrary one, so the
        // orientation is left as it was.
        if forward.cross(&up).norm() > 1e-9 {
            self.camera.set_orientation_from_forward(forward);
        }
        up != Vector3::z()
    }

    /// Cancels any in-progress camera transition, snapping to the current interpolated state.
    pub(crate) fn cancel_transition(&mut self) {
        self.target_transition = None;
    }

    /// Frame `points`, with the same animated transition the `Z` key uses.
    ///
    /// Shared by `Z` (which frames the selected reconstruction) and the Scene
    /// panel's per-node `Zoom to Fit`, so the two cannot drift apart.
    ///
    /// Points at infinity are framed by direction rather than by position
    /// (`ViewportCamera::compute_fit`), and a panorama's turn is levelled when
    /// Maintain Z-up is on.
    pub(crate) fn zoom_to_fit_points(
        &mut self,
        points: &crate::scene::FitPoints,
        aspect: f64,
        current_time: f64,
    ) {
        if points.is_empty() || aspect <= 0.0 || aspect.is_nan() {
            return;
        }
        if let Some(end) = self.camera.compute_fit(points, aspect, self.maintain_z_up) {
            self.start_transition(
                end.position,
                end.orientation,
                end.target_distance,
                self.camera.fov,
                end.world_up,
                None,
                false,
                current_time,
            );
        }
    }

    /// The aspect ratio of the viewport as last laid out, or `None` before the
    /// 3D panel has been shown once.
    pub fn panel_aspect(&self) -> Option<f64> {
        let [w, h] = self.panel_size;
        (w > 0 && h > 0).then(|| w as f64 / h as f64)
    }

    /// Enter camera view mode with a smooth animated transition.
    ///
    /// Computes the target camera pose from the image's extrinsics **through the
    /// node's transform**, deactivates the current camera view (if any), and
    /// starts an animated transition. Camera view mode is activated when the
    /// transition completes.
    ///
    /// Composing the transform is what makes "look through this camera" show the
    /// transformed scene from the transformed camera: an aligned node's cameras
    /// are drawn where its points are, so the viewpoint has to move with them.
    pub fn enter_camera_view(
        &mut self,
        image_ref: ImageRef,
        node: &SceneNode,
        current_time: f64,
        log: &mut ActionLog,
    ) {
        let end = self.compute_camera_view(image_ref, node);
        record_camera_view(log, image_ref, node);

        // Deactivate current camera view during transition
        self.camera_view = None;

        self.start_transition(
            end.position,
            end.orientation,
            end.distance,
            end.fov,
            end.world_up,
            Some(end.camera_view),
            false,
            current_time,
        );
    }

    /// Enter camera view mode **immediately**, with no animated transition.
    ///
    /// Same end state as [`Self::enter_camera_view`], assigned rather than eased
    /// toward. What the MCP surface's `set_view` uses -- an agent that sets the
    /// view and screenshots straight afterward would otherwise photograph the
    /// middle of the ease -- and what `Move Camera` on a Scene Graph image row
    /// uses, the lock being entered *from* camera view rather than towards it.
    pub fn jump_to_camera_view(&mut self, image_ref: ImageRef, node: &SceneNode) {
        let end = self.compute_camera_view(image_ref, node);
        self.cancel_transition();
        self.camera.camera.position = end.position;
        self.camera.camera.orientation = end.orientation;
        self.camera.camera.target_distance = end.distance;
        self.camera.world_up = end.world_up;
        self.camera.fov = end.fov;
        self.camera_view = Some(end.camera_view);
    }

    /// The camera state that looking through `image_ref` lands on.
    ///
    /// Split out of [`Self::enter_camera_view`] so the animated and immediate
    /// entries share one derivation: the two differ only in whether the state
    /// is eased toward or assigned, and a second copy of this arithmetic would
    /// be a second answer to "where does this camera look from".
    fn compute_camera_view(&self, image_ref: ImageRef, node: &SceneNode) -> EnterCameraViewState {
        let reconstruction = node.recon();
        let img_idx = image_ref.index();
        let image = &reconstruction.image_table.images[img_idx];
        let camera = &reconstruction.image_table.cameras[image.camera_index as usize];

        let (world_pose, end_position) = transformed_pose(image, node.transform());
        let r_world_from_cam = world_pose.inverse();

        // The image quaternion is a canonical +Y-up / −Z-forward camera
        // orientation (OpenGL-style), which is exactly the viewport camera
        // convention, so it is used directly with no convention bridge.
        let end_orientation = world_pose;

        let end_distance = reconstruction
            .image_table
            .depth_statistics
            .images
            .get(img_idx)
            .and_then(|stats| stats.observed.median_z)
            // A depth is a length in the node's own coordinates, so a scaled
            // node's median depth has to scale with it.
            .map(|z| z * node.transform().scale)
            .unwrap_or(self.camera.camera.target_distance);

        // world_up = up direction of the end orientation
        let end_world_up = end_orientation.inverse() * Vector3::new(0.0, 1.0, 0.0);

        let end_fov = if !camera.model.is_fisheye() {
            let (fx, fy) = camera.focal_lengths();
            let vfov_cam = (camera.height as f64 / (2.0 * fy)).atan() * 2.0;
            let hfov_cam = (camera.width as f64 / (2.0 * fx)).atan() * 2.0;
            let aspect = if self.panel_size[0] > 0 && self.panel_size[1] > 0 {
                self.panel_size[0] as f64 / self.panel_size[1] as f64
            } else {
                16.0 / 9.0
            };
            best_fit_fov(vfov_cam, hfov_cam, aspect)
        } else {
            self.camera.fov
        };

        EnterCameraViewState {
            position: end_position,
            orientation: end_orientation,
            distance: end_distance,
            world_up: end_world_up,
            fov: end_fov,
            camera_view: CameraViewMode {
                image: image_ref,
                r_world_from_cam,
            },
        }
    }

    /// Computes the end state for switching camera view, preserving relative orientation.
    ///
    /// Returns `None` if not currently in camera view.
    fn compute_switch_camera_view(
        &self,
        new_image_ref: ImageRef,
        node: &SceneNode,
    ) -> Option<SwitchCameraViewState> {
        let old_r_world_from_cam = self.camera_view.as_ref()?.r_world_from_cam;

        let reconstruction = node.recon();
        let new_img_idx = new_image_ref.index();
        let new_image = &reconstruction.image_table.images[new_img_idx];
        let (new_qwxyz, position) = transformed_pose(new_image, node.transform());

        // Compute new orientation preserving relative viewing direction.
        //   new_orientation = orientation * old_r_world_from_cam * new_qwxyz
        let orientation = self.camera.camera.orientation * old_r_world_from_cam * new_qwxyz;
        let distance = reconstruction
            .image_table
            .depth_statistics
            .images
            .get(new_img_idx)
            .and_then(|stats| stats.observed.median_z)
            .map(|z| z * node.transform().scale)
            .unwrap_or(self.camera.camera.target_distance);

        // Transform world_up through the relative rotation between cameras
        let r_old_cam_from_world = old_r_world_from_cam.inverse();
        let r_world_from_new_cam = new_qwxyz.inverse();
        let up_in_old_cam = r_old_cam_from_world * self.camera.world_up;
        let world_up = (r_world_from_new_cam * up_in_old_cam).normalize();

        Some(SwitchCameraViewState {
            position,
            orientation,
            distance,
            world_up,
            camera_view: CameraViewMode {
                image: new_image_ref,
                r_world_from_cam: r_world_from_new_cam,
            },
        })
    }

    /// Switch from one camera view to another instantly, preserving relative orientation.
    ///
    /// Used by `,`/`.` keys and by animation playback for rapid camera
    /// switching. **Records nothing**, unlike the two deliberate entries above:
    /// this one follows a selection step, whose own `Selected image …` entry
    /// already says which image — and a `Looking through …` between every two
    /// of those would break the coalescing that keeps a scrub to one line.
    pub fn switch_camera_view(&mut self, new_image: ImageRef, node: &SceneNode) {
        let Some(state) = self.compute_switch_camera_view(new_image, node) else {
            return;
        };
        self.camera.camera.position = state.position;
        self.camera.camera.orientation = state.orientation;
        self.camera.camera.target_distance = state.distance;
        self.camera.world_up = state.world_up;
        self.camera_view = Some(state.camera_view);
    }

    /// Switch from one camera view to another with a smooth animated transition.
    ///
    /// Used by double-click on frustum/image strip when already in camera view.
    pub fn animated_switch_camera_view(
        &mut self,
        new_image: ImageRef,
        node: &SceneNode,
        current_time: f64,
        log: &mut ActionLog,
    ) {
        let Some(state) = self.compute_switch_camera_view(new_image, node) else {
            self.enter_camera_view(new_image, node, current_time, log);
            return;
        };
        record_camera_view(log, new_image, node);
        self.camera_view = None;
        self.start_transition(
            state.position,
            state.orientation,
            state.distance,
            self.camera.fov,
            state.world_up,
            Some(state.camera_view),
            false,
            current_time,
        );
    }

    /// Look through `image_ref` and turn the view until the feature at `pixel`
    /// in that photograph is inside the middle [`TURN_INTO_FRACTION`] of the
    /// viewport, with one animated transition.
    ///
    /// What a double-click on a Track View row does, in either mode. The row
    /// names an observation rather than a whole image, so the view it opens
    /// shows where that observation is: looking straight through the camera
    /// can leave the feature near the edge of a photograph wider than the
    /// viewport, or off it. Other double-clicks on an image (a frustum, a
    /// thumbnail, a Scene tree row) name no feature and keep
    /// [`Self::enter_camera_view`] and [`Self::animated_switch_camera_view`].
    ///
    /// The view the turn starts from is the one those two land on: the
    /// camera's own pose when entering camera view, and the relative
    /// orientation kept when switching between cameras. The turn is level with
    /// that view's up and uses its field of view, and the feature's bearing is
    /// the ray the lens model maps its pixel from
    /// (`CameraIntrinsics::pixel_to_ray`), put through the camera's world pose.
    /// A feature already inside the middle 1/2 turns nothing, and the result is
    /// a camera view looked around in, as a free look leaves one.
    pub(crate) fn look_through_toward_feature(
        &mut self,
        image_ref: ImageRef,
        node: &SceneNode,
        pixel: [f32; 2],
        current_time: f64,
        log: &mut ActionLog,
    ) {
        let end = self.feature_view(image_ref, node, pixel);
        record_camera_view(log, image_ref, node);
        self.camera_view = None;
        self.start_transition(
            end.position,
            end.orientation,
            end.distance,
            end.fov,
            end.world_up,
            Some(end.camera_view),
            false,
            current_time,
        );
    }

    /// Look through `image_ref` turned toward the feature at `pixel`
    /// **immediately**, with no animated transition.
    ///
    /// Same end state as [`Self::look_through_toward_feature`], assigned
    /// rather than eased toward, for the reason [`Self::jump_to_camera_view`]
    /// gives: what the MCP surface's `set_view` `bench_observation` form uses.
    /// It records nothing, so the call's own Action Log row is the only one.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn jump_through_toward_feature(
        &mut self,
        image_ref: ImageRef,
        node: &SceneNode,
        pixel: [f32; 2],
    ) {
        let end = self.feature_view(image_ref, node, pixel);
        self.cancel_transition();
        self.camera.camera.position = end.position;
        self.camera.camera.orientation = end.orientation;
        self.camera.camera.target_distance = end.distance;
        self.camera.world_up = end.world_up;
        self.camera.fov = end.fov;
        self.camera_view = Some(end.camera_view);
    }

    /// The camera state that looking through `image_ref` turned toward the
    /// feature at `pixel` lands on, shared by the animated and the immediate
    /// entry so the two cannot disagree on where the view ends up.
    fn feature_view(
        &self,
        image_ref: ImageRef,
        node: &SceneNode,
        pixel: [f32; 2],
    ) -> EnterCameraViewState {
        let (position, orientation, distance, fov, world_up, camera_view) =
            match self.compute_switch_camera_view(image_ref, node) {
                Some(s) => (
                    s.position,
                    s.orientation,
                    s.distance,
                    self.camera.fov,
                    s.world_up,
                    s.camera_view,
                ),
                None => {
                    let e = self.compute_camera_view(image_ref, node);
                    (
                        e.position,
                        e.orientation,
                        e.distance,
                        e.fov,
                        e.world_up,
                        e.camera_view,
                    )
                }
            };

        let reconstruction = node.recon();
        let image = &reconstruction.image_table.images[image_ref.index()];
        let lens = &reconstruction.image_table.cameras[image.camera_index as usize];
        let ray = lens.pixel_to_ray(f64::from(pixel[0]), f64::from(pixel[1]));
        let (world_pose, _) = transformed_pose(image, node.transform());
        let bearing = world_pose.inverse() * Vector3::new(ray[0], ray[1], ray[2]);

        let aspect = self.panel_aspect().unwrap_or(16.0 / 9.0);
        let tan_y = (camera::vertical_fov(fov, aspect) / 2.0).tan();
        let orientation = orientation_bringing_into(
            orientation,
            world_up,
            bearing,
            tan_y,
            aspect,
            TURN_INTO_FRACTION,
        );

        EnterCameraViewState {
            position,
            orientation,
            distance,
            world_up,
            fov,
            camera_view,
        }
    }

    /// Updates target indicator state for GPU rendering.
    ///
    /// The indicator is drawn while Alt is held, because Alt+click sets the
    /// target and the indicator shows where it stands before the click; while
    /// `show_target_indicator`, the HUD's **Target indicator**, is on; and for
    /// the flash after the target moves.
    ///
    /// Advances the rotation animation and computes flash animation state.
    /// The actual rendering is done by `SceneRenderer::render_target_indicator`.
    fn update_target_indicator_state(&mut self, ui: &egui::Ui, show_target_indicator: bool) {
        self.target_indicator_visible =
            self.alt_held || show_target_indicator || self.target_flash_start.is_some();

        if !self.target_indicator_visible {
            return;
        }

        // Advance rotation animation
        let dt = ui.input(|i| i.stable_dt) as f64;
        self.target_indicator_rotation += TARGET_ROTATION_SPEED * dt;
        if self.target_indicator_rotation > std::f64::consts::TAU {
            self.target_indicator_rotation -= std::f64::consts::TAU;
        }

        // Flash animation: expand radius and brighten on target change
        let current_time = ui.input(|i| i.time);
        let (radius_scale, alpha_scale) = if let Some(flash_start) = self.target_flash_start {
            let elapsed = current_time - flash_start;
            if elapsed > 0.3 {
                self.target_flash_start = None;
                (1.0_f32, 1.0_f32)
            } else {
                let t = (elapsed / 0.3) as f32;
                let ease_out = 1.0 - (1.0 - t) * (1.0 - t); // quadratic ease-out
                (1.0 + 0.5 * (1.0 - ease_out), 1.0 + 1.0 * (1.0 - ease_out))
            }
        } else {
            (1.0, 1.0)
        };

        self.target_indicator_radius_scale = radius_scale;
        self.target_indicator_alpha_scale = alpha_scale;

        // Request continuous repaint for rotation animation
        ui.ctx().request_repaint();
    }
}
