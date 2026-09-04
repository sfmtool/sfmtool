#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Stand up the Linux accessibility stack the `ui_basic` UI tests read the
# viewer's tree through, then run a command under it.
#
#   scripts/a11y_env.sh pixi run ui-test
#   scripts/a11y_env.sh cargo test -p sfm-explorer --test ui_basic -- --test-threads=1
#
# xa11y talks to AT-SPI2 over the session D-Bus, and AT-SPI2 is a desktop
# service, not a library: on a machine that already runs GNOME/KDE the bus, the
# registry and a window manager are all live and this script is unnecessary —
# run the tests directly. On a headless box (a container, a CI runner, an SSH
# session) none of the four pieces below exists, and the failure is silent: a
# query returns an empty tree rather than an error, so the viewer looks like it
# has no UI at all. This is the shell equivalent of what `xa11y/setup-a11y`
# does for the `ui-test-linux` CI job.
#
# A window manager is started with the display for fidelity to a real desktop,
# not because the suite needs one: xa11y's own CI guidance flags egui/AccessKit
# as a toolkit whose tree can depend on a window being mapped and activated,
# but this viewer publishes its whole tree under a bare Xvfb and the suite
# passes without a WM. It costs a second, so it stays.
#
# Anything already in the environment is left alone, so an outer
# `xa11y/setup-a11y` (or a real desktop) wins over what this would start.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

if [ "$(uname -s)" != "Linux" ]; then
    # Windows needs nothing and macOS needs a TCC grant this cannot give, so on
    # both the honest thing is to run the command unchanged.
    exec "$@"
fi

# Track what we start so the daemons die with the command rather than leaking
# into the user's session.
pids=()
cleanup() {
    for pid in ${pids+"${pids[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

if [ -z "${DISPLAY:-}" ]; then
    display="${SFMTOOL_A11Y_DISPLAY:-:99}"
    echo "a11y_env: starting Xvfb on $display"
    Xvfb "$display" -screen 0 1920x1200x24 -ac >/dev/null 2>&1 &
    pids+=("$!")
    export DISPLAY="$display"
    sleep 1

    # Only worth starting a window manager for a display we own: on someone's
    # real desktop there already is one, and a second would fight it.
    if command -v fluxbox >/dev/null 2>&1; then
        echo "a11y_env: starting fluxbox"
        fluxbox >/dev/null 2>&1 &
        pids+=("$!")
        sleep 1
    fi
fi

if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    echo "a11y_env: starting a D-Bus session"
    # dbus-launch, not dbus-run-session: the bus has to outlive this shell's
    # setup and be inherited by both the test process and the viewer it spawns.
    eval "$(dbus-launch --sh-syntax)"
    pids+=("${DBUS_SESSION_BUS_PID}")
fi

export NO_AT_BRIDGE=0 AT_SPI_CLIENT=true ACCESSIBILITY_ENABLED=1

# The daemons live under /usr/libexec on Debian/Ubuntu and on PATH elsewhere.
find_atspi() {
    if [ -x "/usr/libexec/$1" ]; then
        echo "/usr/libexec/$1"
    elif command -v "$1" >/dev/null 2>&1; then
        command -v "$1"
    fi
}

# A ping is the check *and*, where at-spi2-core installed its
# `org.a11y.Bus.service` file, the fix: the name is D-Bus activatable, so the
# session bus starts the launcher (and through it the registry) to answer.
# Starting them by hand is the fallback for an install without that file.
if ! dbus-send --session --dest=org.a11y.Bus --print-reply /org/a11y/bus \
        org.freedesktop.DBus.Peer.Ping >/dev/null 2>&1; then
    launcher="$(find_atspi at-spi-bus-launcher)"
    registry="$(find_atspi at-spi2-registryd)"
    if [ -z "$launcher" ] || [ -z "$registry" ]; then
        echo "a11y_env: at-spi2-core is not installed — the accessibility tree" \
             "will be empty (apt install at-spi2-core)" >&2
    else
        echo "a11y_env: starting the AT-SPI bus and registry"
        "$launcher" --launch-immediately >/dev/null 2>&1 &
        pids+=("$!")
        sleep 1
        "$registry" >/dev/null 2>&1 &
        pids+=("$!")
        sleep 1
    fi
fi

# Newer at-spi2-core defaults these on and the properties can take a moment to
# register, so a failure here is a note rather than an error.
for prop in IsEnabled ScreenReaderEnabled; do
    dbus-send --session --print-reply --dest=org.a11y.Bus /org/a11y/bus \
        org.freedesktop.DBus.Properties.Set string:org.a11y.Status \
        string:"$prop" variant:boolean:true >/dev/null 2>&1 ||
        echo "a11y_env: org.a11y.Status.$prop not set (continuing)"
done

"$@"
