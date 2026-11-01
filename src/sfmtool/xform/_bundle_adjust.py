# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Bundle adjustment transformation.

A reconstruction whose cameras are all COLMAP models goes through pycolmap. One
with a camera of an sfmtool spline model (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`),
which pycolmap does not know, goes through sfmtool's own reconstruction-level
bundle adjustment with the focal and the lens distortion released (the spline,
and k1 on any SIMPLE_RADIAL_FISHEYE camera beside it), and the principal point
too where ``release_principal_point`` is set. So does one with a non-zero
per-image intrinsic adjustment: pycolmap has no per-image intrinsics, and
sfmtool's adjustment holds each image's adjustment and writes it back. An
image's radial correction or rolling-shutter rates sit on an
`SFMTOOL_PINHOLE` camera, so a reconstruction with either always takes
sfmtool's path, which holds both and each camera's scanlines setting; the
pycolmap path's COLMAP export refuses them.

Either path releases every camera's lens by default. ``cameras=`` limits the
release to the cameras it names and holds the rest. A per-image release
(``image_focal=``, ``image_principal_point=``, ``image_radial_bspline=``,
``image_rolling_shutter=``, with ``scanlines=`` choosing the rates' scanlines)
applies to the released cameras and always takes sfmtool's path, since
pycolmap has no per-image intrinsics.

sfmtool's adjustment solves every free point in inverse depth and stores it as a
position or a direction by the point-or-bearing test at the end of the solve,
so points cross between the two; ``cross=off`` keeps each point in the
representation it has. The pycolmap path solves every point finite and
reclassifies afterwards, and has no such switch. The adjustment refines the
spline coefficients a camera has; a new coefficient count or spline domain is a
refit of the camera, ``--camera-model SFMTOOL_FISHEYE,coeffs=N,spline_domain=DEG``.
"""

import tempfile
from pathlib import Path

import click
import numpy as np
import pycolmap

from .._sfmtool.reconstruction import (
    EditedReconstruction,
    SfmrReconstruction,
    distortion_is_releasable,
)
from ..camera.cameras import _adjusted_image_mask
from ._switch_camera_model import format_outermost_keypoint

# The camera models only sfmtool's own bundle adjustment can refine.
SPLINE_MODELS = ("SFMTOOL_FISHEYE", "SFMTOOL_PINHOLE")

# The values of ``scanlines=``: the `RollingShutterScanlinesChoice` names.
SCANLINES_CHOICES = ("ROWS", "COLUMNS", "UNKNOWN")


def format_image_prior_widths(
    camera: dict, focal_px: float, short_side_px: float
) -> str | None:
    """One line giving the widths of the prior a camera's per-image release
    ran under, or ``None`` when the camera released nothing per image.

    ``per-image widths focal 0.42 %, radial distortion 0.00531 of the focal
    (1.85 px), principal point 1.31 % of the short side (20.12 px), rolling
    shutter 4.92 mrad per short side, derived, 2 images beyond two widths``,
    naming only the parameters ``camera`` (a ``report["cameras"]`` entry)
    released per image; ``fixed`` in place of ``derived`` when the widths were
    given, and ``no image beyond two widths`` when none ended there.
    ``focal_px`` and ``short_side_px`` turn the radial and principal-point
    widths into pixels.
    """
    spread = camera["image_adjustments"]
    if spread is None:
        return None
    prior = spread["prior"]
    parts = []
    if camera["image_focal_released"]:
        parts.append(f"focal {100.0 * prior['focal']:.2f} %")
    if camera["image_radial_bspline_released"]:
        width = prior["radial_bspline"]
        parts.append(
            f"radial distortion {width:.5f} of the focal ({width * focal_px:.2f} px)"
        )
    if camera["image_principal_point_released"]:
        width = prior["principal_point"]
        parts.append(
            f"principal point {100.0 * width:.2f} % of the short side "
            f"({width * short_side_px:.2f} px)"
        )
    if camera["image_rolling_shutter_released"]:
        parts.append(
            f"rolling shutter {1000.0 * prior['rolling_shutter']:.2f} mrad per "
            "short side"
        )
    beyond = len({n["image"] for n in spread["near_prior"]})
    beyond_text = {0: "no image", 1: "1 image"}.get(beyond, f"{beyond} images")
    how = "derived" if spread["prior_derived"] else "fixed"
    return (
        f"per-image widths {', '.join(parts)}, {how}, {beyond_text} beyond two widths"
    )


def format_scanlines(camera: dict) -> str | None:
    """One line saying which scanlines a camera's rolling-shutter rates were
    solved on, or ``None`` when the camera did not release them.

    ``rolling shutter on rows`` where the scanlines were given or stored;
    where they were raced, ``rolling shutter on columns chosen: cost 1823.6
    with rows, 1817.6 with columns over 1272 observations, both solves run to
    the end``, or, when the trailing solve was stopped early, ``rolling
    shutter on rows chosen: columns stopped at round 1, iteration 12: cost
    16552.7 with rows, 17506.1 with columns (gap 953.4 over 24918
    observations)``, the two costs at the pair the stop was decided at.
    """
    setting = camera["rolling_shutter_scanlines"]
    if not camera["image_rolling_shutter_released"] or setting is None:
        return None
    line = f"rolling shutter on {setting.lower()}"
    race = camera["scanlines_race"]
    if race is None:
        return line
    costs = (
        f"cost {race['rows_cost']:.1f} with rows, "
        f"{race['columns_cost']:.1f} with columns"
    )
    stop = race["stop"]
    if stop is None:
        return (
            f"{line} chosen: {costs} over {race['observations']} observations, "
            "both solves run to the end"
        )
    return (
        f"{line} chosen: {stop['stopped'].lower()} stopped at round "
        f"{stop['round']}, iteration {stop['iteration']}: {costs} "
        f"(gap {stop['gap']:.1f} over {race['observations']} observations)"
    )


def format_free_point_decision(decision: dict | None) -> str | None:
    """One line saying what the storage decision of a bundle adjustment did,
    or ``None`` when the adjustment kept every point's representation.

    ``Free points decided at 0.412 px: 3 to finite, 25 to directions``, with
    ``, 2 not scored: too few kept observations`` when the final round kept
    too few observations of some free points to score them, and
    ``; the final round did not converge`` when it stopped on its iteration
    budget or settled with a distortion held at its step guard, since a level
    read from such a round still carries pose error.
    """
    if decision is None:
        return None
    if not decision["decided"] or decision["sigma_px"] is None:
        return (
            "Free points not decided: the final round kept no observation of a "
            "finite point"
        )
    line = (
        f"Free points decided at {decision['sigma_px']:.3f} px: "
        f"{decision['to_finite']} to finite, {decision['to_direction']} to directions"
    )
    if decision["unscored"]:
        line += f", {decision['unscored']} not scored: too few kept observations"
    if not decision["converged"]:
        line += "; the final round did not converge"
    return line


class BundleAdjustTransform:
    """Apply bundle adjustment to refine camera poses and 3D points.

    Args:
        release_focal: Whether a released camera's focal may move.
        release_principal_point: Whether a released camera's principal point
            may move, on either path. ``--bundle-adjust`` leaves it ``False``.
        release_distortion: Whether a released camera's lens distortion may
            move. sfmtool's adjustment releases it only together with the
            focal, and only where the camera model has distortion it can free.
        cameras: Camera indexes whose lens is released
            (``--bundle-adjust cameras=0+1``); every other camera is held, its
            intrinsics unchanged. ``None``, the default, releases every camera.
        cross: Whether sfmtool's adjustment lets free points cross between a
            position and a direction (``--bundle-adjust cross=off`` clears it).
            The pycolmap path refuses ``cross=False``, since it has no way to
            keep each point's representation.
        image_focal: Release each image's focal scale on every released camera
            (``image_focal=on``).
        image_principal_point: Release each image's principal-point offset
            (``image_principal_point=on``).
        image_radial_bspline: Release each image's radial correction
            (``image_radial_bspline=on``), on an ``SFMTOOL_PINHOLE`` camera.
        image_rolling_shutter: Release each image's rolling-shutter rates
            (``image_rolling_shutter=on``), on an ``SFMTOOL_PINHOLE`` camera.
        scanlines: The scanlines the rates are solved on, ``"ROWS"``,
            ``"COLUMNS"`` or ``"UNKNOWN"`` (solve with each and keep the lower
            cost). ``None``, the default, is each camera's stored setting, or
            ``"UNKNOWN"`` where it has none. Given only with
            ``image_rolling_shutter``.

    Any per-image release runs sfmtool's adjustment, which refuses one on a
    camera that cannot take it, naming the camera.

    Raises:
        ValueError: For a ``scanlines`` value that is not one of the three, or
            one given without ``image_rolling_shutter``.
    """

    def __init__(
        self,
        release_focal: bool = True,
        release_principal_point: bool = False,
        release_distortion: bool = True,
        cameras: list[int] | None = None,
        cross: bool = True,
        image_focal: bool = False,
        image_principal_point: bool = False,
        image_radial_bspline: bool = False,
        image_rolling_shutter: bool = False,
        scanlines: str | None = None,
    ):
        if scanlines is not None:
            if scanlines not in SCANLINES_CHOICES:
                raise ValueError(
                    f"scanlines must be one of {', '.join(SCANLINES_CHOICES)}, "
                    f"got {scanlines!r}"
                )
            if not image_rolling_shutter:
                raise ValueError(
                    "scanlines= chooses the scanlines of the rolling-shutter "
                    "rates, which are released only with image_rolling_shutter=on"
                )
        self.release_focal = release_focal
        self.release_principal_point = release_principal_point
        self.release_distortion = release_distortion
        self.cameras = None if cameras is None else sorted(set(cameras))
        self.cross = cross
        self.image_focal = image_focal
        self.image_principal_point = image_principal_point
        self.image_radial_bspline = image_radial_bspline
        self.image_rolling_shutter = image_rolling_shutter
        self.scanlines = scanlines

    @property
    def releases_per_image(self) -> bool:
        """Whether any per-image release is asked for."""
        return (
            self.image_focal
            or self.image_principal_point
            or self.image_radial_bspline
            or self.image_rolling_shutter
        )

    def _released(self, recon: SfmrReconstruction) -> list[bool]:
        """Whether each camera of the table has its lens released: all of them
        without ``cameras=``, only the named ones with it."""
        count = len(recon.cameras)
        if self.cameras is None:
            return [True] * count
        bad = [c for c in self.cameras if not 0 <= c < count]
        if bad:
            raise click.UsageError(
                f"--bundle-adjust cameras= names camera(s) {bad}, and this "
                f"reconstruction has {count} camera(s)"
            )
        return [c in self.cameras for c in range(count)]

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        # Checked before either path runs, so a bad index is refused alike.
        self._released(recon)
        if any(c.model in SPLINE_MODELS for c in recon.cameras):
            return self._apply_sfmtool(recon)
        if self.releases_per_image:
            print(
                "  Running bundle adjustment (sfmtool; a per-image release, "
                "which pycolmap cannot hold)..."
            )
            return self._apply_sfmtool(recon, announce=False)
        if _adjusted_image_mask(recon).any():
            print(
                "  Running bundle adjustment (sfmtool; an image has a per-image "
                "intrinsic adjustment, which pycolmap cannot hold)..."
            )
            return self._apply_sfmtool(recon, announce=False)
        if not self.cross:
            raise click.UsageError(
                "--bundle-adjust cross=off applies to sfmtool's adjustment, which "
                "runs when a camera has a spline model, an image has a "
                "per-image intrinsic adjustment or a per-image release is asked "
                "for; pycolmap's adjustment solves "
                "every point finite and reclassifies the points at infinity "
                "afterwards"
            )
        return self._apply_pycolmap(recon)

    def _apply_sfmtool(
        self, recon: SfmrReconstruction, announce: bool = True
    ) -> SfmrReconstruction:
        """Adjust through sfmtool's own solve, releasing each camera's focal and,
        where its model has one the solve can free, its lens distortion.

        The solve needs a pixel per observation (inline keypoints) and a camera
        model whose focal it can release for every posed image; it raises
        ``ValueError`` naming what does not hold. A released camera's
        principal point moves too where ``release_principal_point`` is set,
        on any model.
        """
        if announce:
            print(
                "  Running bundle adjustment (sfmtool; a camera has a spline "
                "model, which pycolmap cannot refine)..."
            )
        # Each released camera frees its focal, its principal point when asked
        # and, where its model has any the solve can free, its lens distortion,
        # and each per-image field asked for; a held camera frees none of them.
        releases = [
            {
                "focal": released and self.release_focal,
                "principal_point": released and self.release_principal_point,
                "distortion": released
                and self.release_focal
                and self.release_distortion
                and distortion_is_releasable(camera),
                "image_focal": released and self.image_focal,
                "image_principal_point": released and self.image_principal_point,
                "image_radial_bspline": released and self.image_radial_bspline,
                "image_rolling_shutter": released and self.image_rolling_shutter,
                "rolling_shutter_scanlines": self.scanlines
                if released and self.image_rolling_shutter
                else None,
            }
            for camera, released in zip(recon.cameras, self._released(recon))
        ]
        adjusted, report = EditedReconstruction(recon).bundle_adjust(
            releases=releases, free_points_cross=self.cross
        )
        print(
            f"    {report['images']} images, {report['points']} points, "
            f"{report['observations']} observations; median residual "
            f"{report['median_residual_before']:.3f} -> "
            f"{report['median_residual_after']:.3f} px"
        )
        line = format_free_point_decision(report["free_point_decision"])
        if line:
            print(f"    {line}")
        if report["points_deleted"]:
            print(f"    Deleted {report['points_deleted']} unsupported point(s)")
        lenses = recon.cameras
        for camera in report["cameras"]:
            released = [
                name
                for name, flag in (
                    ("focal", camera["focal_released"]),
                    ("principal point", camera["principal_point_released"]),
                    ("distortion", camera["distortion_released"]),
                    ("focal per image", camera["image_focal_released"]),
                    (
                        "principal point per image",
                        camera["image_principal_point_released"],
                    ),
                    (
                        "radial distortion per image",
                        camera["image_radial_bspline_released"],
                    ),
                    (
                        "rolling shutter per image",
                        camera["image_rolling_shutter_released"],
                    ),
                )
                if flag
            ]
            principal_point = ""
            if camera["principal_point_released"]:
                (bx, by), (ax, ay) = (
                    camera["principal_point_before"],
                    camera["principal_point_after"],
                )
                principal_point = (
                    f"; principal point ({bx:.1f}, {by:.1f}) -> ({ax:.1f}, {ay:.1f})"
                )
            print(
                f"    Camera {camera['camera']} ({camera['images']} images): focal "
                f"{camera['focal_before']:.3f} -> {camera['focal_after']:.3f}"
                f"{principal_point}; released: {', '.join(released) or 'none'}"
            )
            lens = lenses[camera["camera"]]
            for line in (
                format_image_prior_widths(
                    camera, camera["focal_after"], min(lens.width, lens.height)
                ),
                format_scanlines(camera),
            ):
                if line:
                    print(f"    Camera {camera['camera']} {line}")
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
            ba_config.refine_focal_length = self.release_focal
            ba_config.refine_principal_point = self.release_principal_point
            ba_config.refine_extra_params = self.release_distortion

            print(f"    Optimizing {len(reconstruction.points3D)} points...")
            if self.cameras is None:
                pycolmap.bundle_adjustment(reconstruction, ba_config)
            else:
                self._solve_holding_cameras(
                    reconstruction, ba_config, self._released(recon)
                )
            reconstruction.update_point_3d_errors()

            refined = self._reconstruction_to_data(reconstruction, ba_input)

        # COLMAP's per-point error is the mean of the observation errors; the
        # stored column is their root mean square, so score every point again.
        # The `.sift` files were already read for the export, and an
        # embedded_patches result carries its keypoints inline.
        refined.recompute_point_errors()

        # Every point comes back from the solve finite. The reclassification
        # demotes those whose rays ask for no depth, at the noise the adjusted
        # reconstruction measures, and recomputes their errors as bearings;
        # the others keep the errors just recomputed.
        result, summary = refined.classify_points_at_infinity()
        if summary["demoted"]:
            print(f"    Reclassified {summary['demoted']} point(s) as at infinity")
        return result

    @staticmethod
    def _solve_holding_cameras(
        reconstruction: pycolmap.Reconstruction,
        options: pycolmap.BundleAdjustmentOptions,
        released: list[bool],
    ) -> None:
        """What ``pycolmap.bundle_adjustment`` runs, with the intrinsics of every
        camera ``released`` marks ``False`` held constant.

        The configuration is the one COLMAP's own bundle adjustment controller
        builds: every registered image, and the gauge fixed by two camera poses.
        """
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in reconstruction.reg_image_ids():
            config.add_image(image_id)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        # The export numbers the cameras in camera index order, which is how the
        # readback maps them to indexes too.
        for camera_id, free in zip(sorted(reconstruction.cameras.keys()), released):
            if not free:
                config.set_constant_cam_intrinsics(camera_id)
        pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()
        for index, free in enumerate(released):
            print(f"    Camera {index}: {'released' if free else 'held'}")

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
            rms_reprojection_errors=errors,
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
        if self.release_focal:
            opts.append("focal")
        if self.release_principal_point:
            opts.append("principal_point")
        if self.release_distortion:
            opts.append("distortion")
        opt_str = ",".join(opts) if opts else "none"
        params = ""
        if self.cameras is not None:
            params += ", cameras=" + "+".join(str(c) for c in self.cameras)
        if not self.cross:
            params += ", cross=off"
        for key in (
            "image_focal",
            "image_principal_point",
            "image_radial_bspline",
            "image_rolling_shutter",
        ):
            if getattr(self, key):
                params += f", {key}=on"
        if self.scanlines is not None:
            params += f", scanlines={self.scanlines.lower()}"
        return f"Bundle adjustment (release: {opt_str}{params})"
