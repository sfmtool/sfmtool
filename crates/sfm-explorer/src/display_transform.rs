// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Setting a node's **display transform**, and **baking** it into the
//! reconstruction.
//!
//! See `specs/gui/scene-graph.md` § "The transform" and
//! `specs/gui/edits/bake-transform.md`. A display transform is view state that
//! the timeline remembers: every change to it is a version of its own whose
//! value half is untouched (a **reframe**), which is why `Ctrl+Z` steps back out
//! of one and the node never goes dirty over it. The bake is the one operation
//! that crosses into the data, and it crosses as one version that states all
//! three halves at once: the transformed value, the transformed bench, and the
//! identity.
//!
//! Four of the ways to set a transform read the bench's active patch
//! ([`PatchReframe`]), which the 3D viewport's context menu and the wire both
//! offer. The arithmetic is here, apart from either, so the menu, the wire and
//! the tests share one answer.

use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use sfmtool_core::bench::{Bench, BenchItem, EditableTrack, Stage};
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::{EditedReconstruction, RotQuaternion, Se3Transform};

use crate::action_log::{version_step_text, Kind};
use crate::document::PointMap;
use crate::progress::Collector;
use crate::scene::ReconId;
use crate::state::edits::version_before;
use crate::state::AppState;
use crate::viewer_3d::{
    ALIGN_NORMAL_TO_Z_LABEL, SET_TO_ORIGIN_LABEL, TRANSLATE_TO_ORIGIN_LABEL,
    TRANSLATE_TO_XY_PLANE_LABEL,
};

#[cfg(test)]
mod tests;

/// One of the four ways the world's frame can be put onto a patch.
///
/// They differ only in how much of the patch's frame the world adopts: all of
/// it, its normal, its centre, or its height. Each is a map acting in **world**
/// space ([`PatchReframe::map`]), with a scale of `1`: they re-frame the scene
/// and never resize it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PatchReframe {
    /// The patch's frame becomes the world frame: `X` along `u`, `Y` along `v`,
    /// `Z` along the outward normal, and the centre at the origin.
    SetToOrigin,
    /// The scene tips about the patch's centre by the shortest rotation taking
    /// its normal to `+Z`.
    AlignNormalToZ,
    /// The scene moves so the patch's centre is at the origin.
    TranslateToOrigin,
    /// The scene moves along world `Z` alone, so the patch's centre sits at
    /// `z = 0`.
    TranslateToXyPlane,
}

impl PatchReframe {
    /// All four, in the order the menu lists them.
    pub(crate) const ALL: [Self; 4] = [
        Self::SetToOrigin,
        Self::AlignNormalToZ,
        Self::TranslateToOrigin,
        Self::TranslateToXyPlane,
    ];

    /// The menu entry's label.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SetToOrigin => SET_TO_ORIGIN_LABEL,
            Self::AlignNormalToZ => ALIGN_NORMAL_TO_Z_LABEL,
            Self::TranslateToOrigin => TRANSLATE_TO_ORIGIN_LABEL,
            Self::TranslateToXyPlane => TRANSLATE_TO_XY_PLANE_LABEL,
        }
    }

    /// What the menu entry says on hover.
    pub(crate) fn hint(self) -> &'static str {
        match self {
            Self::SetToOrigin => {
                "Draw the scene in this patch's own frame: its centre at the origin, its normal \
                 along +Z."
            }
            Self::AlignNormalToZ => {
                "Tip the scene so this patch faces +Z, turning about the patch so it stays where \
                 it is."
            }
            Self::TranslateToOrigin => "Move the scene so this patch's centre is at the origin.",
            Self::TranslateToXyPlane => {
                "Drop the scene along Z so this patch's centre sits on the ground plane."
            }
        }
    }

    /// The wire's spelling: the menu label, snake-cased.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Self::SetToOrigin => "set_to_origin",
            Self::AlignNormalToZ => "align_normal_to_z",
            Self::TranslateToOrigin => "translate_to_origin",
            Self::TranslateToXyPlane => "translate_to_xy_plane",
        }
    }

    /// The entry a wire spelling names, or `None` for one that names none.
    pub(crate) fn from_wire_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.wire_name() == name)
    }

    /// The sentence the Action Log records and the Edit History panel lists.
    ///
    /// The bench item's label names the patch, because that is what the Bench
    /// group and Track View both show, so a person reading the log back can
    /// find the square the sentence is about.
    fn sentence(self, node: &str, patch: &str) -> String {
        match self {
            Self::SetToOrigin => format!("Set {node} to the frame of patch {patch}"),
            Self::AlignNormalToZ => format!("Aligned {node}'s patch {patch} normal to +Z"),
            Self::TranslateToOrigin => format!("Translated {node}'s patch {patch} to the origin"),
            Self::TranslateToXyPlane => {
                format!("Translated {node}'s patch {patch} to the XY plane")
            }
        }
    }

    /// The map this entry applies, acting on **world** space, for a patch
    /// whose world frame is `frame`.
    ///
    /// Set to Origin is `Mᵀ` and not `M`, with `M = [u v n]` as columns: a
    /// world point is `c + M·(a, b, d)`, so its coordinates on the patch's axes
    /// are `Mᵀ(p − c)`. Align Normal to Z turns about `c` rather than about the
    /// world origin, so the patch stays where it is and everything else tips
    /// around it.
    pub(crate) fn map(self, frame: &PatchFrame) -> Se3Transform {
        let c = frame.centre.coords;
        match self {
            Self::SetToOrigin => {
                let m = Matrix3::from_columns(&[frame.u, frame.v, frame.n]);
                // Renormalised, so the quaternion is unit to the last bit and
                // not merely to the matrix's own rounding.
                let q = *RotQuaternion::from_rotation_matrix(m.transpose()).as_nalgebra();
                let rotation = RotQuaternion::new(q.w, q.i, q.j, q.k);
                let translation = -rotation.rotate_vector(&c);
                Se3Transform::new(rotation, translation, 1.0)
            }
            Self::AlignNormalToZ => {
                // `rotation_between` declines only for a normal at `-Z`, where
                // the half turn's axis is not determined by the two vectors. The
                // patch's own `u` is the axis taken there: it leaves `u` alone
                // and sends `v` to `-v`, so the heading a person can see in the
                // square survives.
                let turn = UnitQuaternion::rotation_between(&frame.n, &Vector3::z())
                    .unwrap_or_else(|| {
                        UnitQuaternion::from_axis_angle(
                            &nalgebra::Unit::new_normalize(frame.u),
                            std::f64::consts::PI,
                        )
                    });
                let rotation = RotQuaternion::from_nalgebra(turn);
                let translation = c - rotation.rotate_vector(&c);
                Se3Transform::new(rotation, translation, 1.0)
            }
            Self::TranslateToOrigin => Se3Transform::new(RotQuaternion::identity(), -c, 1.0),
            Self::TranslateToXyPlane => {
                Se3Transform::new(RotQuaternion::identity(), Vector3::new(0.0, 0.0, -c.z), 1.0)
            }
        }
    }
}

/// A patch as it is drawn: its centre and its orthonormal, right-handed axes in
/// **world** coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PatchFrame {
    /// The centre, through the node's display transform.
    pub(crate) centre: Point3<f64>,
    /// The in-plane `u` axis, rotated and unit.
    pub(crate) u: Vector3<f64>,
    /// The in-plane `v` axis, rotated and unit.
    pub(crate) v: Vector3<f64>,
    /// The outward normal, `u × v`.
    pub(crate) n: Vector3<f64>,
}

impl PatchFrame {
    /// `placement`, which is in the reconstruction's own coordinates, as the
    /// node draws it under `transform`.
    ///
    /// The transform's uniform scale is dropped from the axes: it stretches no
    /// direction relative to another, so the rotation alone carries them and
    /// they stay orthonormal and right-handed.
    ///
    /// A stored frame is orthonormal only to the `f32` it was read from, and a
    /// matrix that is not quite a rotation makes a quaternion that is not quite
    /// unit, whose turn then disagrees with itself by far more than rounding.
    /// So the axes are squared up here, `u` kept and `v` rebuilt from the
    /// normal `OrientedPatch::normal` defines, which moves each by no more than
    /// the stored frame was already off.
    pub(crate) fn of(placement: &OrientedPatch, transform: &Se3Transform) -> Self {
        let u = transform
            .rotation
            .rotate_vector(&placement.u_axis)
            .normalize();
        let n = u
            .cross(&transform.rotation.rotate_vector(&placement.v_axis))
            .normalize();
        Self {
            centre: transform.apply_to_point(&placement.center),
            u,
            v: n.cross(&u),
            n,
        }
    }
}

/// The bench with every track-stage placement put through `transform`, which is
/// what a bake does to the bench half of the version it pushes.
///
/// The placement and the track's position are the bench's only world-space
/// geometry; everything else on an item is a pixel, an angle or a count. So the
/// centre goes through the whole similarity, the axes are rotated, and the
/// half-extent is scaled, which is what `apply_se3_transform` does to the point
/// set's own patch half-vectors. A place at infinity is a direction and keeps
/// only the rotation, for the same reason it does there. A cluster has nothing
/// to move and keeps its `Arc`.
pub(crate) fn bake_bench(bench: &Bench, transform: &Se3Transform) -> Bench {
    let mut baked = bench.clone();
    for entry in bench.entries() {
        let BenchItem::Track(track) = &entry.item;
        if let Some(moved) = bake_track(track, transform) {
            baked = baked
                .replace(&entry.label, BenchItem::Track(Arc::new(moved)))
                .expect("the label was read off this bench");
        }
    }
    baked
}

/// `track` with its world-space geometry put through `transform`, or `None`
/// for a cluster-stage item, which has none.
fn bake_track(track: &EditableTrack, transform: &Se3Transform) -> Option<EditableTrack> {
    let Stage::Track(payload) = &track.stage else {
        return None;
    };
    let place = |at: &Point3<f64>, at_infinity: bool| -> Point3<f64> {
        if at_infinity {
            let turned = transform.rotation.rotate_vector(&at.coords);
            let norm = turned.norm();
            Point3::from(if norm > 0.0 { turned / norm } else { turned })
        } else {
            transform.apply_to_point(at)
        }
    };
    let mut payload = payload.clone();
    payload.position = payload
        .position
        .map(|position| place(&position, payload.at_infinity));
    payload.placement = payload.placement.map(|patch| {
        let at_infinity = patch.w == 0.0;
        let scale = if at_infinity { 1.0 } else { transform.scale };
        OrientedPatch {
            center: place(&patch.center, at_infinity),
            u_axis: transform.rotation.rotate_vector(&patch.u_axis),
            v_axis: transform.rotation.rotate_vector(&patch.v_axis),
            half_extent: patch.half_extent.map(|half| half * scale),
            w: patch.w,
        }
    });
    Some(EditableTrack {
        stage: Stage::Track(payload),
        ..track.clone()
    })
}

/// How far `transform` moves things, in the vocabulary the camera move's row
/// already uses: the turn in degrees, the translation's length in scene units,
/// and the scale only when it is not `1`.
fn magnitude_text(transform: &Se3Transform) -> String {
    let mut text = format!(
        "{:.1} deg, {:.3} scene units",
        transform.rotation.angle().to_degrees(),
        transform.translation.norm()
    );
    if transform.scale != 1.0 {
        text.push_str(&format!(", x{:.3}", transform.scale));
    }
    text
}

/// Whether `transform` is the identity, compared exactly, which is the rule
/// [`crate::scene::SceneNode::has_transform`] states.
fn is_identity(transform: &Se3Transform) -> bool {
    transform.scale == 1.0
        && transform.translation == Vector3::zeros()
        && transform.rotation == RotQuaternion::identity()
}

impl AppState {
    /// Set `id`'s display transform to `transform`, as one version of the node.
    ///
    /// A **reframe**: the value is untouched, so the node does not go dirty,
    /// and `Ctrl+Z` steps back out of it. Setting the identity is `Reset
    /// Transform`, and records that entry's sentence whoever asked for it; it is
    /// refused on a node already at the identity, which is the rule that greys
    /// the menu entry. Records its own outcome; the `Err` carries the refusal for a caller
    /// that answers someone.
    pub fn set_node_transform(
        &mut self,
        id: ReconId,
        transform: Se3Transform,
    ) -> Result<(), String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| crate::state::NOT_LOADED.to_string());
        let outcome = label.and_then(|label| {
            let text = if is_identity(&transform) {
                format!("Reset transform of {label}")
            } else {
                format!("Set transform of {label}: {}", magnitude_text(&transform))
            };
            let at_identity = !self.node(id).expect("just found").has_transform();
            if at_identity && is_identity(&transform) {
                return Err(format!("{label} is already in its own frame."));
            }
            self.push_reframe(id, transform, text)
        });
        outcome.inspect_err(|why| self.action_log.fail(Kind::Scene, why.clone()))
    }

    /// Return `id` to its own frame, as one version: `Reset Transform`.
    pub fn reset_node_transform(&mut self, id: ReconId) -> Result<(), String> {
        self.set_node_transform(id, Se3Transform::identity())
    }

    /// Put the world's frame onto the bench's active patch of `id`, in the way
    /// `mode` names, as one version of the node.
    ///
    /// The map acts in world space and is composed **after** the transform the
    /// node already carries, which maps the node's own coordinates into that
    /// world: `next = current ∘ map`, the same shape `Align to…` writes. The
    /// node keeps the scale it had. Refused, and recorded as refused, when the
    /// node is busy or its bench has no patch in the world to read.
    pub(crate) fn reframe_on_patch(
        &mut self,
        id: ReconId,
        mode: PatchReframe,
    ) -> Result<(), String> {
        let outcome = self
            .patch_reframe(id, mode)
            .and_then(|(next, text)| self.push_reframe(id, next, text));
        outcome.inspect_err(|why| self.action_log.fail(Kind::Scene, why.clone()))
    }

    /// The transform `mode` would leave `id` under and the sentence that says
    /// so, or the reason there is no patch to read.
    fn patch_reframe(
        &self,
        id: ReconId,
        mode: PatchReframe,
    ) -> Result<(Se3Transform, String), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let refuse =
            |why: String| format!("Cannot reframe {} on its active patch: {why}", node.label);
        let bench = node.history.current_bench();
        let item = crate::bench::active_track_label(bench)
            .ok_or_else(|| refuse("nothing on its bench is active.".to_string()))?;
        let track = bench.track(item).expect("the active label names an item");
        let payload = match &track.stage {
            Stage::Track(payload) => payload,
            Stage::Cluster(_) => {
                return Err(refuse(format!(
                    "{item} is a cluster, which has no patch in the world until it is a track."
                )))
            }
        };
        let placement = payload
            .placement
            .as_ref()
            .ok_or_else(|| refuse(format!("{item} has no patch frame yet.")))?;
        if placement.w == 0.0 {
            return Err(refuse(format!(
                "{item} is at infinity, a bearing with no place, so its patch has no centre to \
                 move."
            )));
        }
        let current = node.transform();
        let map = mode.map(&PatchFrame::of(placement, current));
        Ok((current.compose(&map), mode.sentence(&node.label, item)))
    }

    /// Push `transform` onto `id` as a reframe labelled `text`, and record it.
    ///
    /// The one place a transform other than the bake's identity reaches a
    /// history, so every way of setting one refuses a busy node the same way and
    /// writes the same kind of row.
    pub(crate) fn push_reframe(
        &mut self,
        id: ReconId,
        transform: Se3Transform,
        text: String,
    ) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let node = self
            .scene
            .iter_mut()
            .find(|n| n.id == id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let serial = node.history.push_transform(transform, text.clone());
        let parent = version_before(node, serial);
        self.action_log
            .record(Kind::Scene, version_step_text(&text, parent, serial));
        Ok(())
    }

    /// Why `Bake Transform` is refused on `id` right now, or `None`.
    ///
    /// The same two gates the menu entry greys on, busy first and then a node
    /// at the identity, in the words a caller that asked anyway is answered in.
    fn bake_refusal(&self, id: ReconId) -> Option<String> {
        if let Some(why) = self.busy_refusal(id) {
            return Some(why);
        }
        let Some(node) = self.node(id) else {
            return Some(crate::state::NOT_LOADED.to_string());
        };
        (!node.has_transform()).then(|| {
            format!(
                "Cannot bake the transform of {}: it is already in its own frame.",
                node.label
            )
        })
    }

    /// Write `id`'s display transform into its reconstruction and return the
    /// node to its own frame, leaving the drawn scene exactly where it is.
    ///
    /// A **bulk edit** through `SfmrReconstruction::apply_se3_transform`, the
    /// one whole-reconstruction similarity `sfm xform` also reaches, with the
    /// bench's placements put through the same transform. One version states
    /// all three halves, the transformed value, the transformed bench and the
    /// identity, so an undo puts back the value and the framing together and
    /// the picture does not move either way. The row map is the identity: the
    /// transform maps the point vector in place, keeping its count, its order
    /// and its ids.
    ///
    /// Recorded as an `Edit` from the instant below, so the row carries the
    /// cost of the transform rather than of writing the row.
    pub fn bake_node_transform(&mut self, id: ReconId) -> Result<(), String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        match self.bake_node_transform_inner(id, &collector) {
            Ok(text) => {
                self.action_log
                    .record_done(Kind::Edit, started, text, collector.take());
                Ok(())
            }
            Err(why) => {
                self.action_log.fail(Kind::Edit, why.clone());
                Err(why)
            }
        }
    }

    /// The bake itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn bake_node_transform_inner(
        &mut self,
        id: ReconId,
        collector: &Collector,
    ) -> Result<String, String> {
        if let Some(why) = self.bake_refusal(id) {
            return Err(why);
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .expect("the refusal above covers a node that is gone");
        let node = &self.scene[index];
        let transform = node.transform().clone();

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the transform can read directly.
        let edited = node.history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let (baked, bench) = {
            let _phase = collector.phase("transform");
            let source = materialised.as_ref().unwrap_or(&edited.base);
            (
                source.apply_se3_transform(&transform),
                bake_bench(node.history.current_bench(), &transform),
            )
        };
        let identity = PointMap::Removed(Vec::new());
        let map = match mat_map {
            Some(folded) => PointMap::Chain(vec![folded, identity]),
            None => identity,
        };

        let text = format!(
            "Baked transform of {}: {}",
            node.label,
            magnitude_text(&transform)
        );
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push_pair(
                Some(EditedReconstruction::new(Arc::new(baked))),
                Arc::new(bench),
                Some(Se3Transform::identity()),
                map,
                text.clone(),
                None,
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(id);
        Ok(version_step_text(&text, parent, serial))
    }
}

/// Why `Bake Transform` and `Reset Transform` are greyed on a node at the
/// identity: there is nothing to bake and nothing to reset.
pub(crate) const IN_OWN_FRAME_HINT: &str = "This reconstruction is already in its own frame";
