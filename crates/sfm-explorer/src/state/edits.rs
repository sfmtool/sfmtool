// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s edits: the operations that give a node a new version, and the
//! three that move its cursor -- undo, redo, and the Edit History panel's jump,
//! which is the two of them repeated.
//!
//! See `specs/gui/document-model.md` and `specs/gui/edit-history.md`. Each edit
//! is a function from the value at the node's cursor to the next value, pushed
//! onto that node's history with the map that says what it did to point
//! indexes. Nothing here mutates a reconstruction in place: a point edit writes
//! the *overlay* the new version owns, and a bulk edit builds a whole new base
//! out of the old one.
//!
//! They are here together because two of them are the two shapes every other
//! one follows:
//!
//! - [`AppState::delete_selected_point`] is the **point edit**. It calls
//!   `EditedReconstruction::delete_point` on a clone of the current value, so
//!   the base is the same `Arc` and nothing but the deleted set changed.
//! - [`AppState::delete_image`] is the **bulk edit**. It materialises the
//!   current value when the overlay is not empty, runs
//!   `SfmrReconstruction::subset_by_image_indices` over it, and the output is
//!   the next version's base with an empty overlay, under the row map
//!   `RowMap::by_scan` reads off that call's input and output.

use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::Kind;
use crate::document::{CreatedPoints, PointMap, VersionSerial};
use crate::progress::Collector;
use crate::resect::ResectFrom;
use crate::scene::{ImageRef, PointRef, ReconId};

use super::AppState;

/// How many pyramid levels the photometric fit's sampler needs. The kernels'
/// own callers build the same number.
const PYRAMID_LEVELS: usize = 6;

/// Decoded images for one edit, owning what a [`ProjectedImage`] borrows.
///
/// One entry per image of the node, because the patch kernels index their view
/// slice by image index; the entries the call does not read are one-pixel
/// placeholders.
pub(super) struct DecodedViews {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramids: Vec<ImageU8Pyramid>,
}

impl DecodedViews {
    /// The borrowed form a patch kernel takes.
    pub(super) fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cameras
            .iter()
            .zip(&self.poses)
            .zip(&self.pyramids)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests;

impl AppState {
    /// Delete the selected point from the node it belongs to.
    ///
    /// A point edit: the version's overlay gains one deleted index, the base is
    /// untouched, and every other index still means what it meant. Returns the
    /// message the caller reports, or `Err` when there is nothing to delete.
    pub fn delete_selected_point(&mut self) -> Result<(), String> {
        let point = self
            .selected_point
            .ok_or_else(|| "No point is selected.".to_string())?;
        self.delete_point(point)
    }

    /// Delete one point, named by ref.
    pub fn delete_point(&mut self, point: PointRef) -> Result<(), String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let node = &mut self.scene[index];
        let mut next = node.history.current().clone();
        next.delete_point(point.point)
            .map_err(|e| format!("Cannot delete that point: {e}"))?;
        let label = node.label.clone();
        let text = format!("Deleted point {} in {label}", point.point);
        let serial = node
            .history
            .push(next, PointMap::Removed(vec![point.point]), text.clone());
        let parent = version_before(node, serial);
        self.follow_selection_forward(point.recon);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Add an observation of the selected point in `image`, at `pixel`.
    ///
    /// A point edit, and the first one that *creates* structure: the point is
    /// deleted from the base and re-added with the new sighting in its track, so
    /// its index moves and the version's map records the move. The base is the
    /// same `Arc`.
    ///
    /// The photometric fit needs the photographs, which a reconstruction value
    /// does not carry, so the track's images and the clicked one are decoded on
    /// demand through the node's full-resolution cache and turned into pyramids
    /// for this call. A handful of images per edit; nothing pre-decodes the
    /// table.
    pub fn add_observation(&mut self, point: PointRef, image: ImageRef) -> Result<(), String> {
        let pixel = self
            .pending_observation_pixel
            .ok_or_else(|| "No pixel was named for the observation.".to_string())?;
        self.add_observation_at(point, image, pixel)
    }

    /// [`AppState::add_observation`] at an explicit pixel.
    pub fn add_observation_at(
        &mut self,
        point: PointRef,
        image: ImageRef,
        pixel: [f32; 2],
    ) -> Result<(), String> {
        if point.recon != image.recon {
            return Err("The point and the image belong to different reconstructions.".to_string());
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;

        // Which images the fit needs: the track's, plus the one being added to.
        // The gate is the menu's own, so the entry and the edit cannot disagree
        // about when this can run, and it is checked before anything is decoded.
        let (label, image_name, needed) = {
            let node = &self.scene[index];
            let edited = node.history.current();
            match crate::image_detail::add_observation_entry(
                edited,
                image.index(),
                Some(point.index()),
            ) {
                None => {
                    return Err(
                        "Adding an observation needs an embedded_patches reconstruction."
                            .to_string(),
                    )
                }
                Some(Err(why)) => return Err(why.to_string()),
                Some(Ok(())) => {}
            }
            let mut needed = edited.track_image_indices(point.point);
            needed.push(image.index());
            let name = node
                .recon()
                .image_table
                .images
                .get(image.index())
                .map(|im| im.name.clone())
                .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
            (node.label.clone(), name, needed)
        };
        let decoded = self.decode_views_for(point.recon, &needed)?;

        let node = &mut self.scene[index];
        let edited = node.history.current();
        let views = decoded.views();
        let (next, report) = sfmtool_core::add_observation(
            edited,
            point.point,
            image.image,
            pixel,
            &views,
            &sfmtool_core::AddObservationOptions::default(),
        )
        .map_err(|e| format!("Cannot add that observation: {e}"))?;

        let moved = report.point;
        let text = format!(
            "Added observation of point {} in {image_name} ({label})",
            point.point
        );
        let serial = node.history.push(
            next,
            PointMap::Replaced(vec![(point.point, moved)]),
            text.clone(),
        );
        let parent = version_before(node, serial);
        // The selection stays on the point, which has taken a new index; the
        // map is what moves it there.
        self.follow_selection_forward(point.recon);
        self.pending_observation_pixel = None;
        self.action_log.record(
            Kind::Edit,
            format!(
                "{text}: ZNCC {:.3}, {:.2} px from the click ({parent} → {serial})",
                report.zncc, report.shift_px
            ),
        );
        Ok(())
    }

    /// Remove the observation of `point` in `image` from its track.
    ///
    /// A point edit, and the one that takes structure out of a track without
    /// taking the track's point with it: the point is deleted from the base and
    /// re-added with the sighting gone, so its index moves and the version's map
    /// records the move. The base is the same `Arc`, and no photographs are
    /// read -- the rays the re-triangulation needs are the poses and lenses the
    /// value already carries.
    ///
    /// The last observation is the exception: with nothing left to see the
    /// point, the point goes, and the version's map is the one
    /// [`AppState::delete_point`] pushes.
    pub fn remove_observation(&mut self, point: PointRef, image: ImageRef) -> Result<(), String> {
        if point.recon != image.recon {
            return Err("The point and the image belong to different reconstructions.".to_string());
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let node = &mut self.scene[index];
        let edited = node.history.current();
        // The gate is the menu's own, so the entry and the edit cannot disagree
        // about when this can run.
        crate::image_detail::remove_observation_entry(edited, image.index(), Some(point.index()))
            .map_err(|why| why.to_string())?;
        let image_name = node
            .recon()
            .image_table
            .images
            .get(image.index())
            .map(|im| im.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let (next, report) = sfmtool_core::remove_observation(edited, point.point, image.image)
            .map_err(|e| format!("Cannot remove that observation: {e}"))?;

        let label = node.label.clone();
        let text = format!(
            "Removed observation of point {} in {image_name} ({label})",
            point.point
        );
        // The point survived and took a new index, or it was the track's last
        // sighting and the point went with it. The map is what the selection
        // follows in either case.
        let map = match report.point {
            Some(moved) => PointMap::Replaced(vec![(point.point, moved)]),
            None => PointMap::Removed(vec![point.point]),
        };
        let serial = node.history.push(next, map, text.clone());
        let parent = version_before(node, serial);
        self.follow_selection_forward(point.recon);
        let outcome = if report.deleted {
            "the point had no other observation and is deleted".to_string()
        } else if report.to_infinity {
            "one observation left, so the point is a bearing at infinity".to_string()
        } else {
            format!("{} observations left", report.observation_count)
        };
        self.action_log.record(
            Kind::Edit,
            format!("{text}: {outcome} ({parent} → {serial})"),
        );
        Ok(())
    }

    /// Create a 3D point at `pixel` in `image`, with a patch of `radius_px`.
    ///
    /// A point edit, and the first one that creates a point rather than moving
    /// one: the version's overlay gains an addition the base has no row for, so
    /// the version is pushed with `push_creating` and the point's id is minted
    /// against the point edit's own content hash rather than against a base's.
    /// The point is created at infinity -- one sighting fixes a bearing and no
    /// distance -- and adding a second observation to it re-triangulates it.
    ///
    /// The image is decoded on demand, as add-observation decodes the ones its
    /// fit needs: the colour and the patch bitmap come out of the photograph.
    pub fn create_point(
        &mut self,
        image: ImageRef,
        pixel: [f32; 2],
        radius_px: f32,
    ) -> Result<(), String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let (label, image_name) = {
            let node = &self.scene[index];
            if node.history.current().has_feature_indexes() {
                return Err(
                    "Creating a point needs an embedded_patches reconstruction.".to_string()
                );
            }
            let name = node
                .recon()
                .image_table
                .images
                .get(image.index())
                .map(|im| im.name.clone())
                .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
            (node.label.clone(), name)
        };
        let decoded = self.decode_views_for(image.recon, &[image.index()])?;

        let node = &mut self.scene[index];
        let edited = node.history.current();
        let views = decoded.views();
        let (next, report) = sfmtool_core::create_point(
            edited,
            image.image,
            pixel,
            radius_px,
            &views,
            &sfmtool_core::CreatePointOptions::default(),
        )
        .map_err(|e| format!("Cannot create that point: {e}"))?;

        // The point is in no base, so its id is the point edit's hash and its
        // place among that edit's creations. The hash is over the record as it
        // now stands, read back out of the value the edit produced.
        let record = next.point(report.point).expect("just created").to_record();
        let created = next
            .point_edit_hash(std::slice::from_ref(&record))
            .ok()
            .map(|hash| CreatedPoints {
                hash,
                indexes: vec![report.point],
            });

        let text = format!("Created point in {image_name} ({label}), radius {radius_px:.1} px");
        let serial = node.history.push_creating(
            next,
            PointMap::Created(vec![report.point]),
            text.clone(),
            created,
        );
        let parent = version_before(node, serial);
        // The selection moves to the point that was just made: it is what the
        // user is now looking at, and what the next edit acts on.
        self.select_point(PointRef::new(image.recon, report.point as usize));
        self.create_point_prompt = None;
        self.create_point_radius = Some(radius_px);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// The radius the Create 3D Point prompt offers for `image`, in that
    /// image's pixels.
    ///
    /// Data-derived, because the scale a point's patch wants is the scale the
    /// reconstruction already works at in that view: the median pixel radius of
    /// the patches this image's own observations project to, or the median over
    /// every observation of the node when this image has none, or
    /// [`FALLBACK_PATCH_RADIUS_PX`] when the node has no patch frames at all.
    pub fn create_point_default_radius(&self, image: ImageRef) -> f32 {
        let Some(node) = self.node(image.recon) else {
            return FALLBACK_PATCH_RADIUS_PX;
        };
        let edited = node.history.current();
        median_projected_radius(edited, Some(image.index()))
            .or_else(|| median_projected_radius(edited, None))
            .unwrap_or(FALLBACK_PATCH_RADIUS_PX)
    }

    /// Delete one image, and with it its observations and any track that is
    /// left with none.
    ///
    /// A bulk edit: the node's next version is a whole new base, so every
    /// image index at or after the deleted one moves down by one and the
    /// surviving points are renumbered. The caller drops its panel-local caches
    /// for the node afterwards, exactly as it does when a node is closed.
    pub fn delete_image(&mut self, image: ImageRef) -> Result<(), String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let removed = image.image;
        let node = &self.scene[index];
        let edited = node.history.current();
        if removed as usize >= edited.image_count() {
            return Err("That image is no longer in the reconstruction.".to_string());
        }
        if edited.image_count() == 1 {
            return Err(format!(
                "{} has only one image; deleting it would leave nothing.",
                node.label
            ));
        }
        let name = node.recon().image_table.images[removed as usize]
            .name
            .clone();

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, and the subset can read that directly.
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &edited.base,
        };

        let keep: Vec<u32> = (0..source.image_count() as u32)
            .filter(|&i| i != removed)
            .collect();
        let subset = source
            .subset_by_image_indices(&keep, true)
            .map_err(|e| format!("Cannot delete that image: {e}"))?;

        // Where each image of `source` went, which is the keep list read the
        // other way round: the deleted one goes nowhere, and everything past it
        // moves down by one.
        let mut image_map: Vec<Option<u32>> = vec![None; source.image_count()];
        for (new, &old) in keep.iter().enumerate() {
            image_map[old as usize] = Some(new as u32);
        }
        // The subset says nothing about which points it dropped, so the map is
        // read off its input and its output.
        let subset_map = RowMap::by_scan(source, &subset, Some(&image_map))
            .map_err(|e| format!("Cannot delete that image: {e}"))?;

        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(subset_map));
        let map = PointMap::Chain(steps);

        let label = node.label.clone();
        let text = format!("Deleted image {name} from {label}");
        let node = &mut self.scene[index];
        let serial = node.history.push(
            EditedReconstruction::new(Arc::new(subset)),
            map,
            text.clone(),
        );
        let parent = version_before(node, serial);
        self.follow_selection_forward(image.recon);
        // Every image index at or past the deleted one moved, so an image,
        // camera or cached texture named by one of them is now a statement
        // about a different image. The node keeps its identity; what it held
        // about images does not.
        self.forget_images_of(image.recon);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Re-estimate `image`'s pose against the rest of `source` and install the
    /// answer as `source`'s next version, rather than as a node beside it.
    ///
    /// A bulk edit: the resection re-poses one image and re-triangulates the
    /// points it observes, so the next version is a whole new base and the map
    /// is the one `RowMap::by_scan` reads off the call's input and output. The
    /// image table does not move -- a resection re-poses an image, it does not
    /// remove one -- so image indexes, the image and camera selections, and the
    /// decoded pixels keyed by them all still mean what they meant.
    ///
    /// A refused *estimate* pushes no version. The derived-node variant keeps
    /// such an answer, because a held-out re-triangulation beside the original
    /// is worth looking at; installed as the original it would be a version that
    /// moved the points and left the pose alone. See
    /// [`AppState::resect_image`] for that variant and
    /// `specs/gui/resect-image.md` for both.
    ///
    /// Records its own outcome, success or refusal, as one Action Log entry, in
    /// the vocabulary the derived-node variant reports in; the `Err` is for the
    /// caller to know the node's caches are still good, not to be logged again.
    pub fn resect_image_in_place(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
    ) -> Result<(), String> {
        match self.resect_in_place_inner(source, image, from) {
            Ok(message) => {
                self.action_log.record(Kind::Edit, message);
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn resect_in_place_inner(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
    ) -> Result<String, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == source)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon()
            .image_table
            .images
            .get(image)
            .map(|i| i.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let basename = crate::resect::basename(&name).to_string();
        if from == ResectFrom::Matches {
            self.load_resect_matches(source)
                .map_err(|why| crate::resect::failure_message(&basename, &label, &why))?;
        }

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the resection can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source_value: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        let outcome = self.with_resect_source(from, |kind| {
            crate::resect::resect_image_in_place(
                source_value,
                image,
                kind,
                &crate::resect::ResectImageOptions::default(),
            )
        });
        let (resected, report) = outcome.map_err(|error| {
            crate::resect::failure_message(&basename, &label, &error.to_string())
        })?;

        // The resection may drop a point it could neither re-triangulate nor
        // hold out, and says nothing about which; the map is read off its input
        // and its output. The image table is untouched, so no image map.
        let scan = RowMap::by_scan(source_value, &resected, None)
            .map_err(|e| crate::resect::failure_message(&basename, &label, &e.to_string()))?;
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let text = format!("Resected {basename} in place ({label})");
        let node = &mut self.scene[index];
        let serial = node.history.push(
            EditedReconstruction::new(Arc::new(resected)),
            map,
            text.clone(),
        );
        let parent = version_before(node, serial);
        self.follow_selection_forward(source);
        Ok(format!(
            "{text}: {} ({parent} → {serial})",
            crate::resect::outcome_summary(&report)
        ))
    }

    /// Put `image` at `world_from_camera`, and install the answer as its node's
    /// next version.
    ///
    /// The pose is in the **node's own frame**, never the displayed one: the
    /// viewport divides its own pose by the node's `Align to…` transform before
    /// calling, so this method and the offline binding take the same thing (see
    /// `crate::camera_lock::pending_pose`).
    ///
    /// A bulk edit: a pose lives in the base, and the tracks the image observes
    /// are re-triangulated around it, so the next version is a whole new base
    /// under the row map `RowMap::by_scan` reads off the call's input and
    /// output. The move deletes and creates no points, so that scan produces
    /// the identity map -- it is still run rather than assumed, because the map
    /// is a fact about the two values and not about this method.
    ///
    /// The image table does not move, so image indexes and the selections and
    /// decodes keyed by them all still mean what they meant; what the panels
    /// cached *about* the geometry is the caller's to drop, as it is after the
    /// resection and the adjustment. Records its own outcome as one Action Log
    /// entry; the `Err` is for the caller to know the node's caches are still
    /// good, not to be logged again.
    pub fn move_camera(
        &mut self,
        image: ImageRef,
        world_from_camera: &sfmtool_core::Se3Transform,
    ) -> Result<(), String> {
        match self.move_camera_inner(image, world_from_camera) {
            Ok(message) => {
                self.action_log.record(Kind::Edit, message);
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn move_camera_inner(
        &mut self,
        image: ImageRef,
        world_from_camera: &sfmtool_core::Se3Transform,
    ) -> Result<String, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon()
            .image_table
            .images
            .get(image.index())
            .map(|i| i.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let basename = crate::resect::basename(&name).to_string();
        let refuse = |why: String| format!("Cannot move {basename} ({label}): {why}");

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the move can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        let (moved, report) = sfmtool_core::move_camera(source, image.index(), world_from_camera)
            .map_err(|e| refuse(e.to_string()))?;
        // The move deletes and creates no points, so this scan is the identity
        // map -- read off the two values rather than asserted. The image table
        // is untouched, so no image map.
        let scan = RowMap::by_scan(source, &moved, None).map_err(|e| refuse(e.to_string()))?;
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let mut text = format!(
            "Moved camera {basename} ({label}): {:.2} deg, {}",
            report.rotation_deg,
            match report.translation_scene {
                Some(scene) => format!("{scene:.3} scene units"),
                None => format!("{:.4}", report.translation),
            }
        );
        if report.retriangulated > 0 {
            text.push_str(&format!(", {} points re-solved", report.retriangulated));
        }
        let node = &mut self.scene[index];
        let serial = node.history.push(
            EditedReconstruction::new(Arc::new(moved)),
            map,
            text.clone(),
        );
        let parent = version_before(node, serial);
        self.follow_selection_forward(image.recon);
        let residual = match (report.residual_before_px, report.residual_after_px) {
            (Some(before), Some(after)) => {
                format!(", residual {:.1} → {:.1} px", before[0], after[0])
            }
            _ => String::new(),
        };
        Ok(format!("{text}{residual} ({parent} → {serial})"))
    }

    /// Bundle-adjust `id`'s current value, and install the answer as its next
    /// version.
    ///
    /// A bulk edit: every posed image's pose, every point's position and, when
    /// the options release it, the shared focal move together, so the next
    /// version is a whole new base under the row map `RowMap::by_scan` reads off
    /// the call's input and output. The map is not decoration here -- a point
    /// the solve leaves unsupported is deleted, and the map is what carries a
    /// selection over that.
    ///
    /// Runs **synchronously** on the GUI thread, as every other edit does. The
    /// window is unresponsive while it solves.
    ///
    /// The image table does not move, so image indexes and the selections keyed
    /// by them still mean what they meant. Records its own outcome as one Action
    /// Log entry; the `Err` is for the caller to know the node's caches are
    /// still good, not to be logged again.
    ///
    /// The entry carries what the solve reported: this call's own stages, and
    /// underneath them the four the kernel names for itself. It is recorded
    /// with [`crate::action_log::ActionLog::record_done`] from the instant
    /// below, so the row says how long the adjustment took rather than how long
    /// writing the row took.
    pub fn bundle_adjust(
        &mut self,
        id: ReconId,
        options: &sfmtool_core::BundleAdjustOptions,
    ) -> Result<(), String> {
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left: read here, as
        // the operation starts, so that a change to it takes effect on the
        // next operation and re-times nothing already recorded.
        let collector = Collector::new(self.action_log.detailed_timing());
        match self.bundle_adjust_inner(id, options, &collector) {
            Ok(message) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn bundle_adjust_inner(
        &mut self,
        id: ReconId,
        options: &sfmtool_core::BundleAdjustOptions,
        collector: &Collector,
    ) -> Result<String, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let refuse = |why: String| format!("Bundle adjust of {label} refused: {why}");
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the adjustment can run.
        if let Some(why) = crate::bundle_adjust_prompt::refusal(self.scene[index].history.current())
        {
            return Err(refuse(why));
        }

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the solve can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        // The kernel's own four stages nest directly under this call's, since
        // the collector's `Progress` is at the top of the operation.
        let (adjusted, report) =
            sfmtool_core::bundle_adjust(source, options, &collector.progress())
                .map_err(|e| refuse(e.to_string()))?;
        // The solve drops the points it left unsupported and says how many, not
        // which; the map is read off its input and its output. The image table
        // is untouched, so no image map.
        let scan = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source, &adjusted, None).map_err(|e| refuse(e.to_string()))?
        };
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let mut text = format!("Bundle adjusted {label}");
        if report.focal_released {
            text.push_str(", focal released");
        }
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(adjusted)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(id);
        let focal = if report.focal_released {
            format!(
                ", focal {:.1} → {:.1}",
                report.focal_before, report.focal_after
            )
        } else {
            String::new()
        };
        let deleted = if report.points_deleted > 0 {
            format!(", {} points deleted", report.points_deleted)
        } else {
            String::new()
        };
        Ok(format!(
            "{text}: {} images, {} points, {} observations, median residual {:.3} → {:.3} px{focal}{deleted} ({parent} → {serial})",
            report.images,
            report.points,
            report.observations,
            report.median_residual_before,
            report.median_residual_after,
        ))
    }

    /// Step `id`'s cursor back one version.
    pub fn undo(&mut self, id: ReconId) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &mut self.scene[index];
        let undone_label = node.history.current_version().label.clone();
        let Some((undone, now)) = node.history.undo() else {
            return Err(format!("Nothing to undo in {}.", node.label));
        };
        self.follow_selection_backward(id, undone);
        self.forget_images_of(id);
        self.action_log.record(
            Kind::Edit,
            format!("Undo: {undone_label} ({undone} → {now})"),
        );
        Ok(())
    }

    /// Step `id`'s cursor forward one version.
    pub fn redo(&mut self, id: ReconId) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &mut self.scene[index];
        let Some((from, redone)) = node.history.redo() else {
            return Err(format!("Nothing to redo in {}.", node.label));
        };
        let redone_label = node.history.current_version().label.clone();
        self.follow_selection_forward(id);
        self.forget_images_of(id);
        self.action_log.record(
            Kind::Edit,
            format!("Redo: {redone_label} ({from} → {redone})"),
        );
        Ok(())
    }

    /// Move `id`'s cursor straight to the version with serial `serial`.
    ///
    /// The move is one action to the user and one Action Log entry, and it is
    /// the sequence of undos or redos that separates the two versions to
    /// everything that follows a map: the cursor is walked one step at a time
    /// and the selection is put through each step's map in turn, so a jump over
    /// three edits lands the selection exactly where three undos would have.
    /// Refused as a whole when any version it would pass through, the
    /// destination included, has had its value released by the budget: there is
    /// nothing there to show.
    pub fn jump_to_version(&mut self, id: ReconId, serial: VersionSerial) -> Result<(), String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &self.scene[index];
        let Some(target) = node.history.position_of(serial) else {
            return Err(format!("{serial} is not a version of {}.", node.label));
        };
        let cursor = node.history.cursor();
        if target == cursor {
            return Err(format!("{serial} is already what {} shows.", node.label));
        }
        let from = node.history.current_version().serial;
        let span = target.min(cursor)..=target.max(cursor);
        if node.history.versions()[span]
            .iter()
            .any(|v| v.value.is_none())
        {
            return Err(format!(
                "Cannot go to {serial}: its value, or one on the way to it, was released to keep {} inside the history budget.",
                node.label
            ));
        }
        while self.scene[index].history.cursor() != target {
            let node = &mut self.scene[index];
            let stepping_back = target < node.history.cursor();
            let Some((left, _)) = node.history.step_towards(target) else {
                break;
            };
            if stepping_back {
                self.follow_selection_backward(id, left);
            } else {
                self.follow_selection_forward(id);
            }
        }
        let node = &self.scene[index];
        let label = node.history.current_version().label.clone();
        let to = node.history.current_version().serial;
        self.forget_images_of(id);
        self.action_log
            .record(Kind::Edit, format!("Go to: {label} ({from} → {to})"));
        Ok(())
    }

    /// Whether `id` has a version to step back to.
    pub fn can_undo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_undo())
    }

    /// Whether `id` has a version to step forward to.
    pub fn can_redo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_redo())
    }

    /// Follow the selected point through the step that produced the version now
    /// at `id`'s cursor: a surviving point keeps its place, a deleted one
    /// clears the selection.
    pub(super) fn follow_selection_forward(&mut self, id: ReconId) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let moved = match node.history.map_into_cursor() {
            Some(map) => map.forward(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Follow the selected point back through the step `undone` was made by.
    fn follow_selection_backward(&mut self, id: ReconId, undone: crate::document::VersionSerial) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let parent = node.history.current_version().serial;
        let moved = match node.history.map_between(parent, undone) {
            Some(map) => map.inverse(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Decode the images `needed` names and hand back a view per image of the
    /// node, ready for a patch kernel.
    ///
    /// Every entry has to exist because the kernels index `views` by image
    /// index, but only the ones a call reads have to be real: an image outside
    /// `needed` gets a one-pixel placeholder, which nothing samples. Decoding
    /// goes through the node's full-resolution cache, so an image the panels
    /// have already shown is not read twice.
    fn decode_views_for(&mut self, id: ReconId, needed: &[usize]) -> Result<DecodedViews, String> {
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        for &img_idx in needed {
            let AppState {
                scene,
                full_res_cache,
                ..
            } = self;
            let recon = scene[index].recon();
            if crate::state::ensure_full_res_cached(
                full_res_cache,
                recon,
                ImageRef::new(id, img_idx),
            )
            .is_none()
            {
                let name = recon.image_table.images[img_idx].name.clone();
                return Err(format!("Cannot read {name}."));
            }
        }
        let recon = self.scene[index].recon();
        let placeholder = ImageU8::new(1, 1, 3, vec![0u8; 3]);
        let mut cameras = Vec::with_capacity(recon.image_count());
        let mut poses = Vec::with_capacity(recon.image_count());
        let mut pyramids = Vec::with_capacity(recon.image_count());
        for (i, im) in recon.image_table.images.iter().enumerate() {
            cameras.push(recon.image_table.cameras[im.camera_index as usize].clone());
            let q = im.quaternion_wxyz;
            poses.push(RigidTransform::from_wxyz_translation(
                [q.w, q.i, q.j, q.k],
                [
                    im.translation_xyz.x,
                    im.translation_xyz.y,
                    im.translation_xyz.z,
                ],
            ));
            let source = needed
                .contains(&i)
                .then(|| self.full_res_cache.get(&ImageRef::new(id, i)))
                .flatten()
                .and_then(|slot| slot.as_ref())
                .unwrap_or(&placeholder);
            pyramids.push(ImageU8Pyramid::build(source, PYRAMID_LEVELS));
        }
        Ok(DecodedViews {
            cameras,
            poses,
            pyramids,
        })
    }

    /// Drop everything this state holds that is keyed by an image of `id`, and
    /// clear the image and camera selections in it.
    ///
    /// What a bulk edit owes: it renumbers the image table, so a cached decode
    /// or a selected index would silently become a statement about a different
    /// image. The panels' own texture caches are dropped by the caller, which
    /// is where they are reachable.
    fn forget_images_of(&mut self, id: ReconId) {
        self.sift_cache.retain(|image, _| image.recon != id);
        self.full_res_cache.retain(|image, _| image.recon != id);
        self.selected_image = self.selected_image.filter(|i| i.recon != id);
        self.selected_camera = self.selected_camera.filter(|c| c.recon != id);
        self.hovered_image = self.hovered_image.filter(|i| i.recon != id);
        self.hovered_point = self.hovered_point.filter(|p| p.recon != id);
    }
}

/// The patch radius offered when a node says nothing about what one should be:
/// no patch frames, or none whose projection is readable. Eight pixels is the
/// order of a SIFT keypoint's own support at the scales these captures are
/// detected at, which is the size a hand-placed point is usually after.
pub const FALLBACK_PATCH_RADIUS_PX: f32 = 8.0;

/// How many observations the median is taken over before the walk stops. A
/// median of a few hundred samples is the same number as a median of a million,
/// and the walk runs while a menu is open.
const RADIUS_SAMPLE_CAP: usize = 512;

/// The median pixel radius of the patches `edited`'s observations project to,
/// over one image or over every image, or `None` when none of them projects.
///
/// The radius of one observation is the mean of its projected affine shape's
/// two axis lengths, which is the ellipse the Image Detail panel draws around
/// that keypoint: what the prompt offers is the size of the patches already on
/// screen beside the click.
fn median_projected_radius(edited: &EditedReconstruction, image: Option<usize>) -> Option<f32> {
    let mut radii: Vec<f64> = Vec::new();
    'points: for index in edited.live_indexes() {
        let Some(view) = edited.point(index) else {
            continue;
        };
        for (k, obs) in view.observations().iter().enumerate() {
            let at = obs.image_index as usize;
            if image.is_some_and(|wanted| wanted != at) {
                continue;
            }
            let Some(keypoint) = view.keypoint_xy(k) else {
                continue;
            };
            let Some(shape) = edited.observation_affine_shape(index, at, keypoint) else {
                continue;
            };
            let u = f64::from(shape[0][0]).hypot(f64::from(shape[1][0]));
            let v = f64::from(shape[0][1]).hypot(f64::from(shape[1][1]));
            let radius = 0.5 * (u + v);
            if radius.is_finite() && radius > 0.0 {
                radii.push(radius);
            }
            if radii.len() >= RADIUS_SAMPLE_CAP {
                break 'points;
            }
        }
    }
    if radii.is_empty() {
        return None;
    }
    Some(sfmtool_core::numeric::median_in_place(&mut radii) as f32)
}

/// The serial of the version `serial` was made from, for the log entry.
fn version_before(
    node: &crate::scene::SceneNode,
    serial: crate::document::VersionSerial,
) -> crate::document::VersionSerial {
    let versions = node.history.versions();
    let position = versions
        .iter()
        .position(|v| v.serial == serial)
        .expect("the version was just pushed");
    versions[position.saturating_sub(1)].serial
}
