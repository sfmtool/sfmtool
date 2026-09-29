#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Give a command a display to open windows on, then run it: the Linux `ui-test`
# pixi task runs the `ui_basic` UI tests through this.
#
#   scripts/display_env.sh cargo test -p sfm-explorer --features ui-tests --test ui_basic -- --test-threads=1
#
# The viewer opens a real window, so it needs an X server or a Wayland
# compositor. On a desktop one is already running and this script only runs
# the command. On a headless box (a container, a CI runner, an SSH session)
# neither `DISPLAY` nor `WAYLAND_DISPLAY` is set, and this starts Xvfb, points
# `DISPLAY` at it, and stops it when the command exits.
#
# The viewer also needs a Vulkan driver, which this cannot provide: on a
# machine without a GPU driver install Mesa's lavapipe (`mesa-vulkan-drivers`
# on Debian/Ubuntu). See `specs/gui/architecture.md` § "Linux".

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

if [ "$(uname -s)" != "Linux" ] || [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    exec "$@"
fi

display="${SFMTOOL_XVFB_DISPLAY:-:99}"
echo "display_env: starting Xvfb on $display"
Xvfb "$display" -screen 0 1920x1200x24 -ac >/dev/null 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
export DISPLAY="$display"

# Wait for the server's socket rather than a fixed time, up to 10s.
socket="/tmp/.X11-unix/X${display#:}"
for _ in $(seq 100); do
    [ -S "$socket" ] && break
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "display_env: Xvfb exited; is it installed (apt install xvfb)?" >&2
        exit 1
    fi
    sleep 0.1
done

"$@"
