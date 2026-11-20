# Dropping the format versions older than sfmtool 0.4's (amendment)

**Status:** Draft

Amends [`../formats/format-versions.md`](../formats/format-versions.md), which
points here for the one step left, and
[`kdf-file-format.md`](../formats/kdf-file-format.md).

sfmtool 0.4.1's release notes say: "0.4.1 still reads the older `.sfmr`,
`.matches` and `.sift` versions so that `sfm ws upgrade` can convert them. The
next release will not." The next release is 0.5. This draft is that removal,
done as [format-versions.md § Dropping a version](../formats/format-versions.md#dropping-a-version)
says, one PR per format, with `.camrig` version 1 going in the same release.
`.sift` version 1, `.sfmr` versions 1 to 15, `.matches` versions 1 to 9,
`.camrig` version 1 and the version 1 workspace file are no longer read
([sift-file-format.md § Format versions](../formats/sift-file-format.md#format-versions),
[sfmr-file-format.md § Format versions](../formats/sfmr-file-format.md#format-versions),
[matches-file-format.md § Format versions](../formats/matches-file-format.md#format-versions),
[camrig-file-format.md § Format versions](../formats/camrig-file-format.md#format-versions),
[workspace.md § Versions](../workspace/workspace.md#versions)), each refused
with a message naming sfmtool 0.4.1, and `sfm ws upgrade` refuses a version 1
workspace and rewrites no file until a format next changes version
([ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md)). The only
work this draft still holds is the `.kdf` spec's shape, below.

## `.kdf`: the spec's Format versions section (one PR)

sfmtool 0.4 and 0.4.1 write `.sfmr` 16, `.matches` 10, `.sift` 2, `.camrig` 2
and `.kdf` 3, so 0.5's read window is exactly those versions. Every release,
0.3 to 0.4.1, wrote `.kdf` version 3, so 0.5's window is version 3 alone,
which is what its reader already reads (versions 1 and 2 were written only by
development builds). Nothing is dropped. Its spec's "Version" subsection still
becomes a "Format versions" section near the top, with a Version History at
the end, in a PR of its own, so all five specs share the shape.
