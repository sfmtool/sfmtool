# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Translation transformation."""

import numpy as np

from .._sfmtool import Se3Transform, SfmrReconstruction


class TranslateTransform:
    """Translate reconstruction by a vector."""

    def __init__(self, translation: np.ndarray):
        self.translation = np.array(translation, dtype=float)
        self.transform = Se3Transform(
            translation=self.translation,
            scale=1.0,
        )

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return self.transform @ recon

    def description(self) -> str:
        t = self.translation
        return f"Translate by ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"
