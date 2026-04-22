# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch the camera model of every camera in a reconstruction.

Useful as a step before bundle adjustment when you want to optimize a
higher-order distortion model — e.g. upgrading `SIMPLE_RADIAL` to `RADIAL`
so bundle adjustment has a `k2` parameter to refine.
"""

from .._cameras import _CAMERA_PARAM_NAMES
from .._sfmtool import CameraIntrinsics, SfmrReconstruction


class SwitchCameraModelTransform:
    """Convert every camera in the reconstruction to a different COLMAP model.

    Parameters shared between the source and target models (focal length,
    principal point, any distortion coefficient whose name is identical in
    both) are carried over. Parameters that only exist in the target model
    initialize to zero. Parameters that only exist in the source model are
    dropped.

    When the focal-length representation differs between source and target
    (single `focal_length` vs split `focal_length_x` / `focal_length_y`),
    values are translated: single → split duplicates the value, split →
    single averages `fx` and `fy` (a warning is printed if they differ).
    """

    def __init__(self, target_model: str):
        target_upper = target_model.upper()
        if target_upper not in _CAMERA_PARAM_NAMES:
            supported = ", ".join(sorted(_CAMERA_PARAM_NAMES.keys()))
            raise ValueError(
                f"Unknown camera model '{target_model}'. Supported: {supported}"
            )
        self.target_model = target_upper

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        new_cameras = []
        for i, camera in enumerate(recon.cameras):
            new_cameras.append(self._convert_camera(camera, i))

        # Report what changed.
        source_models = {c.model for c in recon.cameras}
        if source_models == {self.target_model}:
            print(
                f"  All {len(recon.cameras)} camera(s) already use model "
                f"{self.target_model}; re-initializing to zero-padded params"
            )
        else:
            src_list = ", ".join(sorted(source_models))
            print(
                f"  Switched {len(recon.cameras)} camera(s) from "
                f"[{src_list}] to {self.target_model}"
            )

        return recon.clone_with_changes(cameras=new_cameras)

    def _convert_camera(
        self, source: CameraIntrinsics, camera_index: int
    ) -> CameraIntrinsics:
        src_dict = source.to_dict()
        src_params = src_dict["parameters"]
        target_param_names = _CAMERA_PARAM_NAMES[self.target_model]

        new_params: dict[str, float] = {}
        for name in target_param_names:
            if name in src_params:
                new_params[name] = float(src_params[name])
                continue

            # Handle focal-length representation mismatch.
            if (
                name == "focal_length"
                and {
                    "focal_length_x",
                    "focal_length_y",
                }
                <= src_params.keys()
            ):
                fx = float(src_params["focal_length_x"])
                fy = float(src_params["focal_length_y"])
                if abs(fx - fy) > 1e-6 * max(abs(fx), abs(fy), 1.0):
                    print(
                        f"  Warning: camera {camera_index} has asymmetric focal "
                        f"lengths (fx={fx:.4f}, fy={fy:.4f}); averaging to a "
                        f"single focal_length for {self.target_model}"
                    )
                new_params[name] = 0.5 * (fx + fy)
                continue
            if (
                name in ("focal_length_x", "focal_length_y")
                and "focal_length" in src_params
            ):
                new_params[name] = float(src_params["focal_length"])
                continue

            # Parameter not present in source — default to zero.
            new_params[name] = 0.0

        return CameraIntrinsics.from_dict(
            {
                "model": self.target_model,
                "width": source.width,
                "height": source.height,
                "parameters": new_params,
            }
        )

    def description(self) -> str:
        return f"Switch camera model to {self.target_model}"
