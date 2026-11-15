# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-readable summaries of file path lists.

This module is the single seam onto the `deadline` (AWS Deadline Cloud client)
dependency: it is the only place in the package that imports from it, and the
whole of the use is the two pure path-formatting functions re-exported here,
with the `PathFormat` enum their ``path_format`` argument takes. Import them
from here rather than from `deadline` directly, so that replacing, vendoring or
stubbing that dependency stays a one-file change.

By default both functions write paths with the host's separator, so on Windows
they turn every ``/`` into a backslash. Image names always use ``/``, so a caller
summarizing image names passes ``path_format=PathFormat.POSIX``.
"""

from deadline.job_attachments.api import (
    summarize_path_list,
    summarize_paths_by_sequence,
)
from deadline.job_attachments.models import PathFormat

__all__ = ["PathFormat", "summarize_path_list", "summarize_paths_by_sequence"]
