# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle adjustment transformation."""

import tempfile
from pathlib import Path

import numpy as np
import pycolmap

from .._sfmtool import SfmrReconstruction


class BundleAdjustTransform:
    """Apply bundle adjustment to refine camera poses and 3D points."""

    def __init__(
        self,
        refine_focal_length: bool = True,
        refine_principal_point: bool = False,
        refine_extra_params: bool = True,
    ):
        self.refine_focal_length = refine_focal_length
        self.refine_principal_point = refine_principal_point
        self.refine_extra_params = refine_extra_params

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        from ..colmap.io import save_colmap_binary

        print("  Running bundle adjustment...")

        # Bundle adjustment is finite-only. Materialise any points at infinity
        # to finite landmarks for the solve, then reclassify afterwards so
        # points whose depth is still unconstrained return to w = 0.
        n_infinity = int(np.count_nonzero(recon.point_is_at_infinity))
        if n_infinity:
            print(f"    Materializing {n_infinity} point(s) at infinity")
        ba_input = recon.materialize_points_at_infinity()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            colmap_dir = temp_path / "colmap"
            # In-pipeline pycolmap round trip: S-only (D3). Export flips the
            # camera frames to COLMAP but leaves the world untouched; the
            # re-import in _reconstruction_to_data flips the cameras back.
            save_colmap_binary(ba_input, colmap_dir, apply_world_rotation=False)

            reconstruction = pycolmap.Reconstruction()
            reconstruction.read_binary(str(colmap_dir))

            ba_config = pycolmap.BundleAdjustmentOptions()
            ba_config.refine_focal_length = self.refine_focal_length
            ba_config.refine_principal_point = self.refine_principal_point
            ba_config.refine_extra_params = self.refine_extra_params

            print(f"    Optimizing {len(reconstruction.points3D)} points...")
            pycolmap.bundle_adjustment(reconstruction, ba_config)
            reconstruction.update_point_3d_errors()

            refined = self._reconstruction_to_data(reconstruction, ba_input)

        result = refined.classify_points_at_infinity()
        n_after = int(np.count_nonzero(result.point_is_at_infinity))
        if n_after:
            print(f"    Reclassified {n_after} point(s) as at infinity")
            # The errors read back from the (materialised) BA solve describe the
            # finite landmark, not the w = 0 bearing the point reverted to.
            # Recompute only those points from the feature positions; finite
            # points keep the errors the BA solve produced.
            result.recompute_infinity_point_errors()
        return result

    def _reconstruction_to_data(
        self,
        reconstruction: pycolmap.Reconstruction,
        original_recon: SfmrReconstruction,
    ) -> SfmrReconstruction:
        from ..camera.cameras import pycolmap_camera_to_intrinsics
        from ..colmap.io import _extract_rig_frame_data

        sorted_camera_ids = sorted(reconstruction.cameras.keys())
        cameras = [
            pycolmap_camera_to_intrinsics(reconstruction.cameras[cam_id])
            for cam_id in sorted_camera_ids
        ]
        camera_id_to_index = {
            cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)
        }

        name_to_idx = {name: idx for idx, name in enumerate(original_recon.image_names)}

        image_names = original_recon.image_names
        camera_indexes = np.zeros(len(image_names), dtype=np.uint32)
        quaternions_wxyz = np.zeros((len(image_names), 4), dtype=np.float64)
        translations = np.zeros((len(image_names), 3), dtype=np.float64)

        image_id_to_index = {}
        for image_id, image in reconstruction.images.items():
            idx = name_to_idx.get(image.name)
            if idx is not None:
                image_id_to_index[image_id] = idx
                camera_indexes[idx] = camera_id_to_index[image.camera_id]
                cam_from_world = image.cam_from_world()
                quat_xyzw = cam_from_world.rotation.quat
                quaternions_wxyz[idx] = [
                    quat_xyzw[3],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                ]
                translations[idx] = cam_from_world.translation

        point_ids = sorted(reconstruction.points3D.keys())

        # Bundle adjustment refines geometry; it must not add or drop points.
        # The result is rebuilt with clone_with_changes, which reindexes points
        # by position but carries the per-point patch frames/bitmaps through
        # unchanged — so a changed point count would silently misalign those
        # arrays (and the observation bookkeeping) with the new points. Fail
        # loudly instead: if this ever fires, pycolmap dropped a point and the
        # readback needs to remap the per-point arrays accordingly.
        if len(point_ids) != original_recon.point_count:
            raise RuntimeError(
                "Bundle adjustment changed the point count "
                f"({original_recon.point_count} -> {len(point_ids)}); this is "
                "unexpected (BA refines points, it does not add or remove them) "
                "and would misalign per-point patch data. Aborting rather than "
                "producing a corrupt reconstruction."
            )

        positions = np.array(
            [reconstruction.points3D[pid].xyz for pid in point_ids], dtype=np.float64
        )
        colors = np.array(
            [reconstruction.points3D[pid].color for pid in point_ids], dtype=np.uint8
        )
        errors = np.array(
            [reconstruction.points3D[pid].error for pid in point_ids], dtype=np.float32
        )

        # An embedded_patches reconstruction has no external .sift files; its 2D
        # observations live inline as keypoints_xy. Recover each observation's
        # keypoint from the COLMAP point2D we exported so the refined result can
        # be rebuilt in embedded_patches mode too.
        is_embedded = original_recon.feature_source == "embedded_patches"

        track_image_indexes_list = []
        track_feature_indexes_list = []
        track_point_indexes_list = []
        track_keypoints_xy_list = []
        observation_counts = np.zeros(len(point_ids), dtype=np.uint32)

        for new_pid, old_pid in enumerate(point_ids):
            point3d = reconstruction.points3D[old_pid]
            observation_counts[new_pid] = len(point3d.track.elements)

            for element in point3d.track.elements:
                image = reconstruction.images[element.image_id]
                image_idx = name_to_idx.get(image.name)
                if image_idx is not None:
                    track_image_indexes_list.append(image_idx)
                    track_feature_indexes_list.append(element.point2D_idx)
                    track_point_indexes_list.append(new_pid)
                    if is_embedded:
                        track_keypoints_xy_list.append(
                            image.points2D[element.point2D_idx].xy
                        )

        track_image_indexes = np.array(track_image_indexes_list, dtype=np.uint32)
        track_feature_indexes = np.array(track_feature_indexes_list, dtype=np.uint32)
        track_point_indexes = np.array(track_point_indexes_list, dtype=np.uint32)

        # The poses read back are in COLMAP camera frame (S-only export);
        # flip the camera frames back to canonical. Points were exported with
        # the world untouched, so they are already canonical.
        from ..colmap.convention import flip_camera_pose_s

        quaternions_wxyz, translations = flip_camera_pose_s(
            quaternions_wxyz, translations
        )

        rig_frame_data = _extract_rig_frame_data(
            reconstruction, camera_id_to_index, image_id_to_index
        )

        kwargs = dict(
            cameras=cameras,
            camera_indexes=camera_indexes,
            quaternions_wxyz=quaternions_wxyz,
            translations=translations,
            positions=positions,
            colors=colors,
            errors=errors,
            track_image_indexes=track_image_indexes,
            track_feature_indexes=track_feature_indexes,
            track_point_indexes=track_point_indexes,
            observation_counts=observation_counts,
        )
        if is_embedded:
            # Rebuild in embedded_patches mode: the inline keypoints replace the
            # (unused) sift feature indices. image_file_hashes is carried over
            # from the original reconstruction by clone_with_changes.
            kwargs["feature_source"] = "embedded_patches"
            kwargs["keypoints_xy"] = np.array(
                track_keypoints_xy_list, dtype=np.float32
            ).reshape(-1, 2)
        if rig_frame_data is not None:
            kwargs["rig_frame_data"] = rig_frame_data

        return original_recon.clone_with_changes(**kwargs)

    def description(self) -> str:
        opts = []
        if self.refine_focal_length:
            opts.append("focal")
        if self.refine_principal_point:
            opts.append("pp")
        if self.refine_extra_params:
            opts.append("extra")
        opt_str = ",".join(opts) if opts else "none"
        return f"Bundle adjustment (refine: {opt_str})"
