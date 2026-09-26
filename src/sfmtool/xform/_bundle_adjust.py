# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle adjustment transformation.

A reconstruction whose cameras are all COLMAP models goes through pycolmap. One
with a camera of an sfmtool spline model (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`),
which pycolmap does not know, goes through sfmtool's own reconstruction-level
bundle adjustment with the focal and the lens distortion released (the spline,
and k1 on any SIMPLE_RADIAL_FISHEYE camera beside it).
"""

import tempfile
from pathlib import Path

import click
import numpy as np
import pycolmap

from .._sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
from ._switch_camera_model import format_monotone_constraint, format_outermost_keypoint

# The camera models only sfmtool's own bundle adjustment can refine.
SPLINE_MODELS = ("SFMTOOL_FISHEYE", "SFMTOOL_PINHOLE")


class BundleAdjustTransform:
    """Apply bundle adjustment to refine camera poses and 3D points.

    Args:
        coeff_count: Refit every spline camera to this many spline coefficients
            before the solve (``--bundle-adjust coeffs=N``).
        spline_domain_deg: Refit every spline camera on a domain ending at this
            incidence angle, in degrees, before the solve, in the same refit as
            ``coeff_count`` (``--bundle-adjust domain=DEG``).

    Only a reconstruction with a spline camera takes either, which the sfmtool
    path adjusts; given for one without, ``apply`` raises ``click.UsageError``.
    """

    def __init__(
        self,
        refine_focal_length: bool = True,
        refine_principal_point: bool = False,
        refine_extra_params: bool = True,
        coeff_count: int | None = None,
        spline_domain_deg: float | None = None,
    ):
        self.refine_focal_length = refine_focal_length
        self.refine_principal_point = refine_principal_point
        self.refine_extra_params = refine_extra_params
        self.coeff_count = coeff_count
        self.spline_domain_deg = spline_domain_deg

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if any(c.model in SPLINE_MODELS for c in recon.cameras):
            return self._apply_sfmtool(recon)
        if self.coeff_count is not None or self.spline_domain_deg is not None:
            raise click.UsageError(
                "--bundle-adjust coeffs= and domain= apply only to a reconstruction with a "
                f"{' or '.join(SPLINE_MODELS)} camera, and this one has none "
                "(switch a camera with --camera-model first)"
            )
        return self._apply_pycolmap(recon)

    def _apply_sfmtool(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        """Adjust through sfmtool's own solve, releasing each camera's focal and,
        where its model has one the solve can free, its lens distortion.

        The solve needs a pixel per observation (inline keypoints) and a camera
        model whose focal it can release for every posed image; it raises
        ``ValueError`` naming what does not hold. The principal point stays
        where it is, as it does in every sfmtool solve.
        """
        print(
            "  Running bundle adjustment (sfmtool; a camera has a spline model, "
            "which pycolmap cannot refine)..."
        )
        adjusted, report = EditedReconstruction(recon).bundle_adjust(
            opt_f=self.refine_focal_length,
            opt_distortion=self.refine_focal_length and self.refine_extra_params,
            spline_coeff_count=self.coeff_count,
            spline_domain_deg=self.spline_domain_deg,
        )
        print(
            f"    {report['images']} images, {report['points']} points, "
            f"{report['observations']} observations; median residual "
            f"{report['median_residual_before']:.3f} -> "
            f"{report['median_residual_after']:.3f} px"
        )
        if report["points_deleted"]:
            print(f"    Deleted {report['points_deleted']} unsupported point(s)")
        for camera in report["cameras"]:
            released = [
                name
                for name, flag in (
                    ("focal", camera["focal_released"]),
                    ("distortion", camera["distortion_released"]),
                )
                if flag
            ]
            print(
                f"    Camera {camera['camera']} ({camera['images']} images): focal "
                f"{camera['focal_before']:.3f} -> {camera['focal_after']:.3f}; "
                f"released: {', '.join(released) or 'none'}"
            )
            refit = camera["spline_refit"]
            if refit is not None:
                domain = ""
                if abs(refit["domain_after_deg"] - refit["domain_before_deg"]) > 1e-9:
                    domain = (
                        f", domain {refit['domain_before_deg']:.1f}° -> "
                        f"{refit['domain_after_deg']:.1f}°"
                    )
                print(
                    f"      spline refitted {refit['coeffs_before']} -> "
                    f"{refit['coeffs_after']} coefficients{domain} before the solve: "
                    f"rms {refit['rms_px']:.4f} px, max {refit['max_px']:.4f} px "
                    "from the old curve over its domain"
                )
                held = format_monotone_constraint(refit["monotone_constraint"])
                if held:
                    print(f"        {held}")
        result = adjusted.materialize()[0]
        # How far out the photographs reach, under the adjusted cameras: the
        # outermost observation, and the outermost feature detected in the
        # images' .sift files where they can be read.
        for outermost in result.outermost_keypoints(
            cameras=[c["camera"] for c in report["cameras"]]
        ):
            line = format_outermost_keypoint(outermost)
            if line:
                print(f"    Camera {outermost['camera']} {line}")
        return result

    def _apply_pycolmap(self, recon: SfmrReconstruction) -> SfmrReconstruction:
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
            # Solved reconstruction: claim the native models back (see
            # `pycolmap_camera_to_intrinsics`).
            pycolmap_camera_to_intrinsics(
                reconstruction.cameras[cam_id], claim_native=True
            )
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
        params = ""
        if self.coeff_count is not None:
            params += f", coeffs={self.coeff_count}"
        if self.spline_domain_deg is not None:
            params += f", domain={self.spline_domain_deg:g}"
        return f"Bundle adjustment (refine: {opt_str}{params})"
