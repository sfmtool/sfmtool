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
        from .._colmap_io import save_colmap_binary

        print("  Running bundle adjustment...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            colmap_dir = temp_path / "colmap"
            save_colmap_binary(recon, colmap_dir)

            reconstruction = pycolmap.Reconstruction()
            reconstruction.read_binary(str(colmap_dir))

            ba_config = pycolmap.BundleAdjustmentOptions()
            ba_config.refine_focal_length = self.refine_focal_length
            ba_config.refine_principal_point = self.refine_principal_point
            ba_config.refine_extra_params = self.refine_extra_params

            print(f"    Optimizing {len(reconstruction.points3D)} points...")
            pycolmap.bundle_adjustment(reconstruction, ba_config)
            reconstruction.update_point_3d_errors()

            return self._reconstruction_to_data(reconstruction, recon)

    def _reconstruction_to_data(
        self,
        reconstruction: pycolmap.Reconstruction,
        original_recon: SfmrReconstruction,
    ) -> SfmrReconstruction:
        from .._cameras import pycolmap_camera_to_intrinsics

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

        for _image_id, image in reconstruction.images.items():
            idx = name_to_idx.get(image.name)
            if idx is not None:
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
        positions = np.array(
            [reconstruction.points3D[pid].xyz for pid in point_ids], dtype=np.float64
        )
        colors = np.array(
            [reconstruction.points3D[pid].color for pid in point_ids], dtype=np.uint8
        )
        errors = np.array(
            [reconstruction.points3D[pid].error for pid in point_ids], dtype=np.float32
        )

        track_image_indexes_list = []
        track_feature_indexes_list = []
        track_point_ids_list = []
        observation_counts = np.zeros(len(point_ids), dtype=np.uint32)

        for new_pid, old_pid in enumerate(point_ids):
            point3d = reconstruction.points3D[old_pid]
            observation_counts[new_pid] = len(point3d.track.elements)

            for element in point3d.track.elements:
                image_name = reconstruction.images[element.image_id].name
                image_idx = name_to_idx.get(image_name)
                if image_idx is not None:
                    track_image_indexes_list.append(image_idx)
                    track_feature_indexes_list.append(element.point2D_idx)
                    track_point_ids_list.append(new_pid)

        track_image_indexes = np.array(track_image_indexes_list, dtype=np.uint32)
        track_feature_indexes = np.array(track_feature_indexes_list, dtype=np.uint32)
        track_point_ids = np.array(track_point_ids_list, dtype=np.uint32)

        return original_recon.replace(
            cameras=cameras,
            camera_indexes=camera_indexes,
            quaternions_wxyz=quaternions_wxyz,
            translations=translations,
            positions=positions,
            colors=colors,
            errors=errors,
            track_image_indexes=track_image_indexes,
            track_feature_indexes=track_feature_indexes,
            track_point_ids=track_point_ids,
            observation_counts=observation_counts,
        )

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
