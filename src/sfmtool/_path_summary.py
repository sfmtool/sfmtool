# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-readable summaries of file path lists.

This module is the single seam onto the `deadline` (AWS Deadline Cloud client)
dependency: it is the only place in the package that imports from it, and the
whole of the use is the two pure path-formatting functions re-exported here.
Import them from here rather than from `deadline` directly, so that replacing,
vendoring or stubbing that dependency stays a one-file change.
"""

from deadline.job_attachments.api import (
    summarize_path_list,
    summarize_paths_by_sequence,
)

__all__ = ["summarize_path_list", "summarize_paths_by_sequence"]
