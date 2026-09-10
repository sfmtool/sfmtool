// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Move Camera lock: camera view with the camera coming along.
//!
//! See `specs/gui/edits/move-camera.md`. Camera view already puts the viewport
//! at an image's pose with its photograph attached; the lock flips one bit, so
//! that every navigation input moves the **camera** rather than leaving it
//! behind. The pending pose therefore *is* the viewport pose, and nearly all of
//! the interaction is "do not leave camera view" plus the arithmetic here:
//! reading the viewport back as a pose in the node's own frame, measuring what
//! it would cost, and handing that to the edit.
//!
//! The lock is viewport state rather than document state, which is the line
//! camera view itself sits on: every panel, and the MCP read surface, shows the
//! stored pose until the commit.

use std::collections::HashSet;

use eframe::egui;
use nalgebra::{Point3, Vector3};
use sfmtool_core::reconstruction::move_camera::{
    edited_image_reprojection_samples, residual_quantiles_px, ReprojectionSample,
};
use sfmtool_core::{CameraIntrinsics, RotQuaternion, Se3Transform, SfmrReconstruction};

use crate::action_log::Kind;
use crate::scene::{ImageRef, SceneNode};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

#[cfg(test)]
mod tests;

/// Rotation below which a commit is inside the dead band, in degrees.
const DEAD_BAND_DEG: f64 = 0.01;

/// Translation below which a commit is inside the dead band, as a fraction of
/// the capture's own extent.
const DEAD_BAND_EXTENT_FRACTION: f64 = 1e-6;

/// One held lock: which camera is in hand, where it stood, and what the
/// residual readout measures against.
///
/// Everything here is fixed for the lock's life. The pending pose is not a
/// field: it is the viewport's, read back through [`pending_pose`] whenever it
/// is wanted, so there is one statement of where the camera is rather than two
/// that can disagree.
pub(crate) struct CameraLock {
    /// The image being moved.
    pub(crate) image: ImageRef,
    /// Where it stood when the lock was entered, world-from-camera in the
    /// node's **own** frame. What `Escape` puts back and the dead band
    /// measures against.
    pub(crate) stored: Se3Transform,
    /// The lens the residual readout projects through.
    camera: CameraIntrinsics,
    /// This image's observations as of the lock: the point where the value has
    /// it and the pixel that saw it. Fixed, because nothing in the value moves
    /// while the lock is held.
    samples: Vec<ReprojectionSample>,
    /// The residual pair under the stored pose, which the banner shows beside
    /// the pending one.
    stored_residual: Option<[f64; 2]>,
    /// The points this image observes, which are drawn in the hover tint for
    /// the lock's duration: they are the ones the commit will move.
    pub(crate) highlighted: HashSet<u32>,
    /// The capture's own extent -- the radius of its camera cloud -- which the
    /// dead band's translation floor is a fraction of.
    extent: f64,
}

impl CameraLock {
    /// The image's basename, as every label and log line names it.
    fn basename<'a>(&self, node: &'a SceneNode) -> &'a str {
        node.recon()
            .image_table
            .images
            .get(self.image.index())
            .map(|image| crate::resect::basename(&image.name))
            .unwrap_or("?")
    }
}

/// Why the lock cannot be entered on the viewer as it stands, or `None` when it
/// can.
///
/// The Edit menu's gate and the edit's own precondition, so the entry and the
/// lock cannot disagree about when a camera can be taken in hand.
pub(crate) fn refusal(state: &AppState, viewer: &Viewer3D) -> Option<String> {
    if viewer.camera_lock.is_some() {
        return Some("A camera is already being moved; commit or cancel it first".to_string());
    }
    let Some(view) = viewer.camera_view.as_ref() else {
        return Some("Look through a camera first -- select an image and press Z".to_string());
    };
    let Some(node) = state.node(view.image.recon) else {
        return Some("That reconstruction is no longer loaded".to_string());
    };
    let Some(image) = node.recon().image_table.images.get(view.image.index()) else {
        return Some("That image is no longer in the reconstruction".to_string());
    };
    let posed = image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
        && image.translation_xyz.iter().all(|c| c.is_finite());
    if !posed {
        return Some("That image carries no pose to move".to_string());
    }
    None
}

/// Take the camera being looked through in hand.
///
/// Snaps the viewport to that camera's **stored** pose, keeping the field of
/// view: a free-look offset the reviewer happened to be holding is not an edit
/// anybody made, and the lens is not what is being moved. Returns the sentence
/// saying why not when the lock cannot be entered.
pub(crate) fn enter(viewer: &mut Viewer3D, state: &mut AppState) -> Result<(), String> {
    if let Some(why) = refusal(state, viewer) {
        return Err(format!("Cannot move the camera: {why}."));
    }
    let image = viewer
        .camera_view
        .as_ref()
        .expect("checked by refusal")
        .image;
    let node = state
        .node(image.recon)
        .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
    let edited = node.history.current();
    let recon = node.recon();
    let stored = sfmtool_core::reconstruction::move_camera::pose_of(recon, image.index());
    let samples = edited_image_reprojection_samples(edited, image.index());
    let camera = recon.image_table.camera_for_image(image.index()).clone();
    let stored_residual = residual_quantiles_px(&camera, &stored, &samples);
    let highlighted: HashSet<u32> = edited
        .live_indexes()
        .filter(|&index| {
            edited.point(index).is_some_and(|view| {
                view.observations()
                    .iter()
                    .any(|o| o.image_index == image.image)
            })
        })
        .collect();
    let lock = CameraLock {
        image,
        stored,
        camera,
        samples,
        stored_residual,
        highlighted,
        extent: capture_extent(recon),
    };
    let basename = lock.basename(node).to_string();
    let label = node.label.clone();

    snap_to(viewer, &lock.stored, node);
    viewer.camera_lock = Some(lock);
    state.action_log.record(
        Kind::View,
        format!("Moving the camera of {basename} ({label})"),
    );
    Ok(())
}

/// The pose the viewport is holding, in the node's **own** frame.
///
/// The one place a display transform crosses into an edit short of
/// bake-transform, and it crosses by being divided out: the viewport pose is
/// composed with the inverse of the node's `Align to…` transform, so a camera
/// moved in an aligned node lands where the reviewer put it on screen and the
/// value never learns the display transform existed.
pub(crate) fn pending_pose(viewer: &Viewer3D, node: &SceneNode) -> Se3Transform {
    // The viewport's orientation is the world-to-camera rotation in the frame
    // the node is *drawn* in, and `transformed_pose` builds that as
    // `q_node · q_transform⁻¹`; multiplying it back by the transform's rotation
    // is the division.
    let cam_from_world = viewer.camera.camera.orientation * *node.transform.rotation.as_nalgebra();
    let centre = match node.transform.inverse() {
        Ok(inverse) => inverse.apply_to_point(&viewer.camera.camera.position),
        Err(_) => viewer.camera.camera.position,
    };
    Se3Transform::new(
        RotQuaternion::from_nalgebra(cam_from_world.inverse()),
        centre.coords,
        1.0,
    )
}

/// How far the pending pose stands from the stored one: `(degrees, distance)`
/// in the node's own units.
pub(crate) fn displacement(lock: &CameraLock, pending: &Se3Transform) -> (f64, f64) {
    let rotation = pending
        .rotation
        .as_nalgebra()
        .rotation_to(lock.stored.rotation.as_nalgebra())
        .angle()
        .to_degrees();
    let translation = (pending.translation - lock.stored.translation).norm();
    (rotation, translation)
}

/// Whether a commit of `pending` would be inside the dead band: the lock was
/// held and nothing was moved.
pub(crate) fn in_dead_band(lock: &CameraLock, pending: &Se3Transform) -> bool {
    let (rotation, translation) = displacement(lock, pending);
    let floor = if lock.extent.is_finite() && lock.extent > 0.0 {
        DEAD_BAND_EXTENT_FRACTION * lock.extent
    } else {
        0.0
    };
    rotation <= DEAD_BAND_DEG && translation <= floor
}

/// The residual pair the banner shows: the median and 90th percentile under the
/// pending pose, and the same two under the stored one.
///
/// `None` in either slot where the value carries no inline keypoints, which is
/// the `n/a` the banner prints.
pub(crate) fn residuals(
    lock: &CameraLock,
    pending: &Se3Transform,
) -> (Option<[f64; 2]>, Option<[f64; 2]>) {
    (
        residual_quantiles_px(&lock.camera, pending, &lock.samples),
        lock.stored_residual,
    )
}

/// Commit the held lock: one version carrying the pose the viewport is at.
///
/// A commit inside the dead band pushes nothing and says so; anything else
/// hands the pose, divided out of the node's transform, to
/// [`AppState::move_camera`]. The lock is released either way.
///
/// `Ok(Some(node))` is "a version was pushed on that node", which is what tells
/// the caller to drop the panel caches that describe its geometry;
/// `Ok(None)` is the dead band.
pub(crate) fn commit(
    viewer: &mut Viewer3D,
    state: &mut AppState,
) -> Result<Option<crate::scene::ReconId>, String> {
    let Some(lock) = viewer.camera_lock.take() else {
        return Err("No camera is being moved.".to_string());
    };
    let Some(node) = state.node(lock.image.recon) else {
        return Err("That reconstruction is no longer loaded.".to_string());
    };
    let pending = pending_pose(viewer, node);
    if in_dead_band(&lock, &pending) {
        let text = format!("{} ({}) was not moved", lock.basename(node), node.label);
        state.action_log.record(Kind::View, text);
        return Ok(None);
    }
    state
        .move_camera(lock.image, &pending)
        .map(|()| Some(lock.image.recon))
}

/// Release the held lock and put the viewport back where the camera stands.
///
/// No version is pushed and nothing is recorded as an edit: the value did not
/// change.
pub(crate) fn cancel(viewer: &mut Viewer3D, state: &mut AppState) {
    let Some(lock) = viewer.camera_lock.take() else {
        return;
    };
    let Some(node) = state.node(lock.image.recon) else {
        return;
    };
    let text = format!(
        "Cancelled the camera move of {} ({})",
        lock.basename(node),
        node.label
    );
    snap_to(viewer, &lock.stored, node);
    state.action_log.record(Kind::View, text);
}

/// End a lock that something else is about to end anyway.
///
/// Commits when the pose has moved past the dead band and drops the lock
/// silently otherwise, which is the rule the delete edits set: an edit has a
/// history behind it, so undo is the answer to an accidental one, whereas
/// cancelled work has no undo.
pub(crate) fn exit_implicitly(
    viewer: &mut Viewer3D,
    state: &mut AppState,
) -> Option<crate::scene::ReconId> {
    let lock = viewer.camera_lock.as_ref()?;
    // A node that has left the scene took the camera with it, and there is
    // nothing to commit the pose *to*. Dropping the lock is what keeps a closed
    // node from leaving one held on a reconstruction nobody can see.
    if state.node(lock.image.recon).is_none() {
        viewer.camera_lock = None;
        return None;
    }
    match commit(viewer, state) {
        Ok(moved) => moved,
        Err(message) => {
            state.action_log.fail(Kind::Edit, message);
            None
        }
    }
}

/// The Move Camera keys, which are viewport keys held here because they need
/// the state the viewport is not handed: `M` and `Enter` commit, `Escape`
/// cancels, and `M` with no lock takes the camera in hand.
///
/// Gated by the caller on egui's own keyboard arbitration, exactly as the
/// viewport's other bindings are. Reports the node a commit pushed a version
/// on, so the caller can drop the panel caches describing its geometry.
pub(crate) fn handle_keys(
    ui: &egui::Ui,
    viewer: &mut Viewer3D,
    state: &mut AppState,
) -> Option<crate::scene::ReconId> {
    let (m, commits, escape, stepping) = ui.input(|i| {
        (
            i.key_pressed(egui::Key::M),
            i.key_pressed(egui::Key::Enter),
            i.key_pressed(egui::Key::Escape),
            i.key_pressed(egui::Key::Comma) || i.key_pressed(egui::Key::Period),
        )
    });
    // A lock whose node has been closed under it is dropped rather than acted
    // on: its camera is gone, and every key below would be about nothing.
    if viewer
        .camera_lock
        .as_ref()
        .is_some_and(|lock| state.node(lock.image.recon).is_none())
    {
        viewer.camera_lock = None;
    }
    if viewer.camera_lock.is_none() {
        if m {
            if let Err(message) = enter(viewer, state) {
                state.action_log.fail(Kind::View, message);
            }
        }
        return None;
    }
    if m || commits {
        match commit(viewer, state) {
            Ok(moved) => moved,
            Err(message) => {
                state.action_log.fail(Kind::Edit, message);
                None
            }
        }
    } else if escape {
        cancel(viewer, state);
        None
    } else if stepping {
        // The step itself happens inside the viewport, on this same frame and
        // after this call: what the lock owes it is to be gone by then.
        exit_implicitly(viewer, state)
    } else {
        None
    }
}

/// End a lock held on `node`, for something that is about to edit that node.
///
/// An edit landing under a held lock would leave the reviewer holding a camera
/// whose stored pose had moved beneath them, so the lock goes first -- as a
/// commit when it has been moved, and silently otherwise.
pub(crate) fn exit_implicitly_for(
    viewer: &mut Viewer3D,
    state: &mut AppState,
    node: crate::scene::ReconId,
) -> Option<crate::scene::ReconId> {
    if viewer
        .camera_lock
        .as_ref()
        .is_some_and(|lock| lock.image.recon == node)
    {
        return exit_implicitly(viewer, state);
    }
    None
}

/// End a held lock when `[` or `]` is about to step the viewport onto another
/// reconstruction, which is a step away from the camera in hand.
pub(crate) fn exit_implicitly_on_recon_step(
    ui: &egui::Ui,
    viewer: &mut Viewer3D,
    state: &mut AppState,
) -> Option<crate::scene::ReconId> {
    viewer.camera_lock.as_ref()?;
    let stepping = ui
        .input(|i| i.key_pressed(egui::Key::OpenBracket) || i.key_pressed(egui::Key::CloseBracket));
    stepping.then(|| exit_implicitly(viewer, state)).flatten()
}

/// Put the viewport back at the pose the version now at the cursor holds.
///
/// A move of the history cursor -- an undo, a redo, a jump -- can change the
/// pose of the very camera the viewport is looking through, and camera view
/// follows the value rather than remembering where a hand left it, exactly as
/// it follows a `,` / `.` switch. No lock is ever held here: a commit is what
/// releases one.
pub(crate) fn resnap_camera_view(viewer: &mut Viewer3D, state: &AppState) {
    if viewer.camera_lock.is_some() {
        return;
    }
    let Some(image) = viewer.camera_view.as_ref().map(|view| view.image) else {
        return;
    };
    let Some(node) = state.node(image.recon) else {
        return;
    };
    if image.index() >= node.recon().image_table.images.len() {
        return;
    }
    let pose = sfmtool_core::reconstruction::move_camera::pose_of(node.recon(), image.index());
    snap_to(viewer, &pose, node);
}

/// The model transform the camera-view background is drawn with while a lock is
/// held, or `None` when there is none.
///
/// The background mesh's vertices are the image's rays pre-rotated by the
/// **stored** camera-to-world rotation, so putting the photograph at the
/// pending pose is a rotation of the node's model matrix rather than a rebuilt
/// mesh: `R_pending · R_stored⁻¹`, which at the moment of entry is the node's
/// own transform and nothing more. The mesh is drawn with `w = 0`, so only the
/// rotation reaches it -- which is also why translating the camera leaves the
/// photograph alone, a background at infinity having no parallax to show.
pub(crate) fn background_transform(viewer: &Viewer3D, node: &SceneNode) -> Option<Se3Transform> {
    let lock = viewer.camera_lock.as_ref()?;
    if lock.image.recon != node.id {
        return None;
    }
    let world_from_camera_stored = *lock.stored.rotation.as_nalgebra();
    let rotation = (world_from_camera_stored * viewer.camera.camera.orientation).inverse();
    Some(Se3Transform::new(
        RotQuaternion::from_nalgebra(rotation),
        node.transform.translation,
        node.transform.scale,
    ))
}

/// The lock banner's lines: what is in hand, what it costs, and how to end it.
///
/// A pure function of the two residual pairs so the wording is assertable
/// without a viewport.
pub(crate) fn banner_lines(
    basename: &str,
    label: &str,
    pending: Option<[f64; 2]>,
    stored: Option<[f64; 2]>,
) -> Vec<String> {
    let pair = |r: Option<[f64; 2]>| match r {
        Some([median, p90]) => format!("{median:.2} / {p90:.2} px"),
        None => "n/a".to_string(),
    };
    vec![
        format!("Moving {basename} ({label})"),
        format!("residual {}, stored {}", pair(pending), pair(stored)),
        "M or Enter commits, Esc cancels".to_string(),
    ]
}

/// Put the viewport exactly at `pose`, which is in the node's own frame.
///
/// The field of view and the orbit distance are left alone: the snap is about
/// where the camera is, and the lens is not what is being moved.
fn snap_to(viewer: &mut Viewer3D, pose: &Se3Transform, node: &SceneNode) {
    let cam_from_world =
        pose.rotation.as_nalgebra().inverse() * node.transform.rotation.as_nalgebra().inverse();
    let centre = node
        .transform
        .apply_to_point(&Point3::from(pose.translation));
    viewer.cancel_transition();
    viewer.camera.camera.position = centre;
    viewer.camera.camera.orientation = cam_from_world;
    viewer.camera.world_up = cam_from_world.inverse() * Vector3::new(0.0, 1.0, 0.0);
    if let Some(view) = viewer.camera_view.as_mut() {
        view.r_world_from_cam = cam_from_world.inverse();
    }
}

/// The capture's own extent: the radius of its camera cloud about its centroid.
///
/// The scale the dead band's translation floor is a fraction of. Over camera
/// centres rather than over structure because it is a camera that is being
/// moved, and because it is `O(images)` on a keystroke.
fn capture_extent(recon: &SfmrReconstruction) -> f64 {
    let centres: Vec<Point3<f64>> = recon
        .image_table
        .images
        .iter()
        .map(|image| image.camera_center())
        .filter(|centre| centre.coords.iter().all(|c| c.is_finite()))
        .collect();
    if centres.is_empty() {
        return 0.0;
    }
    let mut centroid = Vector3::zeros();
    for centre in &centres {
        centroid += centre.coords;
    }
    centroid /= centres.len() as f64;
    centres
        .iter()
        .map(|centre| (centre.coords - centroid).norm())
        .fold(0.0, f64::max)
}
