# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformation pipeline for SfM reconstructions."""

from .._sfmtool.reconstruction import SfmrReconstruction
from ._align_to import AlignToTransform
from ._align_to_input import AlignToInputTransform
from ._apply import apply_transforms
from ._bundle_adjust import BundleAdjustTransform
from ._filter_by_image_range import (
    ExcludeGlobFilter,
    ExcludeRangeFilter,
    IncludeGlobFilter,
    IncludeRangeFilter,
)
from ._filter_by_localizability import FilterByLocalizabilityTransform
from ._filter_by_patch_size import FilterByPatchSizeTransform
from ._filter_by_reprojection_error import FilterByReprojectionErrorTransform
from ._find_points_at_infinity import (
    ClassifyPointsAtInfinityTransform,
    FindPointsAtInfinityTransform,
)
from ._interface import Transform
from ._localize_keypoints import LocalizeKeypointsTransform
from ._point_filters import (
    RemoveIsolatedPointsFilter,
    RemoveLargeFeaturesFilter,
    RemoveNarrowTracksFilter,
    RemoveShortTracksFilter,
)
from ._refine_keypoints import RefineKeypointsTransform
from ._refine_normals import RefineNormalsTransform
from ._rotate import RotateTransform
from ._scale import ScaleTransform
from ._scale_by_measurements import ScaleByMeasurementsTransform
from ._select_by_distribution import SelectByDistributionFilter
from ._similarity import SimilarityTransform
from ._switch_camera_model import SwitchCameraModelTransform
from ._to_embedded_patches import ToEmbeddedPatchesTransform
from ._translate import TranslateTransform

__all__ = [
    "SfmrReconstruction",
    "Transform",
    "RotateTransform",
    "TranslateTransform",
    "ScaleTransform",
    "ScaleByMeasurementsTransform",
    "RemoveShortTracksFilter",
    "RemoveNarrowTracksFilter",
    "RemoveIsolatedPointsFilter",
    "RemoveLargeFeaturesFilter",
    "FilterByReprojectionErrorTransform",
    "FilterByLocalizabilityTransform",
    "FilterByPatchSizeTransform",
    "FindPointsAtInfinityTransform",
    "ClassifyPointsAtInfinityTransform",
    "IncludeRangeFilter",
    "ExcludeRangeFilter",
    "IncludeGlobFilter",
    "ExcludeGlobFilter",
    "SelectByDistributionFilter",
    "BundleAdjustTransform",
    "LocalizeKeypointsTransform",
    "RefineKeypointsTransform",
    "RefineNormalsTransform",
    "AlignToTransform",
    "AlignToInputTransform",
    "SimilarityTransform",
    "SwitchCameraModelTransform",
    "ToEmbeddedPatchesTransform",
    "apply_transforms",
]
