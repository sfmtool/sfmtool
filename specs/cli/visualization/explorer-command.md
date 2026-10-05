# `sfm explorer` Command

## Purpose

`sfm explorer` opens SfM Explorer, the sfmtool desktop viewer, from the Python
command line. The viewer shows one or more reconstructions (`.sfmr` files) in a
3D window with panels for the cameras, images, points and tracks, so a person
can look at a solve and find where it went wrong. The viewer is compiled into
the `sfmtool._sfmtool` extension module, so it comes with `pip install
sfmtool`. The command runs it in the `sfm` process itself, on the main
thread, with the arguments given, waits for the window to close, and exits with
the viewer's exit status. The viewer itself is
specified under [../../gui/](../../gui/README.md); start with
[architecture.md](../../gui/architecture.md) and
[user-experience.md](../../gui/user-experience.md).

## Command Syntax

```bash
sfm explorer [--mcp [PORT]] [--no-default-layout] [FILE ...]
```

The command is implemented in
[explorer.py](../../../src/sfmtool/_commands/explorer.py) and registered under
the Visualization category in [cli.py](../../../src/sfmtool/cli.py).

## Arguments and options

| Argument / option | Type | Default | Meaning |
|-------------------|------|---------|---------|
| `FILE ...` | paths, zero or more | none | Files to load. Each must exist, or Click refuses the command before the viewer starts. Each file becomes its own node in the viewer's scene graph, in the order given, so several reconstructions can be compared in one 3D space ([scene-graph.md](../../gui/scene-graph.md)). With no files the viewer opens empty. |
| `--mcp [PORT]` | int, 0 to 65535 | off; `8787` when given without a number | Host a Model Context Protocol endpoint on `127.0.0.1:PORT`, so an agent can drive the viewer. `0` takes an ephemeral port, which the viewer prints at startup. See [mcp-server.md](../../gui/mcp-server.md). |
| `--no-default-layout` | flag | off | Start with the stock panel grid, ignoring a layout saved at `~/.sfm-explorer-default-layout.json` ([panel-layout.md](../../gui/panel-layout.md) § "The default layout file"). |

The default port is the constant `DEFAULT_MCP_PORT` in `explorer.py`. It
repeats `DEFAULT_MCP_PORT` in
[cli.rs](../../../crates/sfm-explorer/src/cli.rs) rather than asking the viewer
for it, because Click needs the value to build `--help`.

`--mcp` takes its port from the next argument whenever there is one, because
Click options with an optional value consume the following word. So
`sfm explorer --mcp scene.sfmr` fails with "'scene.sfmr' is not a valid
integer". Put the files first (`sfm explorer scene.sfmr --mcp`), give the port
(`sfm explorer --mcp 8787 scene.sfmr`), or end the options with `--`
(`sfm explorer --mcp -- scene.sfmr`). The viewer's own command line accepts
`--mcp scene.sfmr` and reads it as the default port and a file.

A port outside 0 to 65535 is refused by Click with a usage error (exit status
2) before the viewer starts. The viewer refuses it too, with exit status 2, as
it takes a word of digits after `--mcp` as the port whether or not it fits;
checking it in the command gives the error in the same form as the command's
other usage errors.

## How the viewer is launched

The command builds the viewer's command line: `--mcp PORT` (when `--mcp` was
given), then `--no-default-layout` (when given), then the files. `run_viewer`
in `explorer.py` then restores the default `SIGINT` handler when Python's own
handler is installed, calls `sfmtool._sfmtool.run_explorer` with that command
line on the main thread, and the command calls `sys.exit` with the status it
returns.

Python's `SIGINT` handler only sets a flag for the interpreter to act on, and
the interpreter does not run while the viewer holds the main thread, so with it
installed Ctrl+C would do nothing. The default handler ends the process, as it
does for any other program. Python installs its handler only over an inherited
default, so an inherited "ignore" (a background job of a non-interactive shell)
is left as it is. `run_viewer` puts Python's handler back when the viewer
returns. On Windows, Ctrl+C or Ctrl+Break in the console ends `sfm explorer`
with status `0xC000013A` (`STATUS_CONTROL_C_EXIT`).

Before the call `run_viewer` flushes `sys.stdout` and `sys.stderr`. The `sfm`
entry point (`main` in [cli.py](../../../src/sfmtool/cli.py)) replaces both
with line-buffered UTF-8 files on the same file descriptors, and the viewer
writes to the process's stdout and stderr through Rust's own handles, not
through those Python objects, so the flush keeps anything Python wrote first
ahead of the viewer's output.

`run_explorer` is a root-level function of the extension, in
[lib.rs](../../../crates/sfmtool-py/src/lib.rs) of `sfmtool-py`. It releases
the GIL and calls `sfm_explorer::run_with_args`, which parses the viewer's
command line, opens the window, and returns when the window closes.

`run_with_args` does not end the process. It returns `Result<(), RunError>`,
and a `RunError` carries the message to show and the exit status to end with:
status 2 when the command line does not parse (an unknown option, or `--mcp=`
followed by something that is not a port number) or asks for `--mcp` in a build
without the `mcp` feature, and status 1 when the MCP endpoint cannot bind its
port or the event loop, the window or its GPU device cannot be created (for
example with no display, or no Vulkan driver on Linux). `run_explorer` prints
the message to stderr and returns the status, or returns 0 when the window
closed or `--help` printed the usage, and `sfm explorer` exits with that
status.

`run_explorer` has to be called on the process's main thread, and once per
process, and it changes the process; the command can call it in the `sfm`
process because running the viewer is the last thing the command does, and the
process ends when it returns:

- It creates a `winit` event loop, which `winit` creates only on the
  process's main thread, on every platform, and only once per process. A call
  from another thread panics. Because of the once-per-process rule, a second
  `run_explorer` call in one process that gets as far as creating the event
  loop returns status 1 with a message saying the viewer has already run in
  this process. No test makes that second call, since the first would have to
  open a window. `sfm explorer` makes one call.
- It sets process-wide state that outlives the call: it initializes the global
  `env_logger` logger, unless one is already installed, and on Windows it sets
  the process's DPI awareness. An MCP endpoint's server thread also keeps
  running after the window closes, until the process ends.

The command imports nothing it does not use. `sfmtool/__init__.py` binds the
names of its Python submodules on first use, and `sfm` imports a command's
module only when that command runs (see
[cli/README.md](../README.md) § "How `sfm` loads its commands"), so `sfm
explorer` loads Click, the extension and `explorer.py`, and not numpy, OpenCV
or pycolmap.

`pixi run gui` runs the same viewer from a source checkout through a different
entry point: the `sfm-explorer` crate's own `sfm-explorer` binary,
[main.rs](../../../crates/sfm-explorer/src/main.rs), whose `main` calls
`sfm_explorer::run`, which passes `std::env::args()` without the program name
to `run_with_args`. Both accept the same command line, and the viewer's
`--help` text and its unknown-option error name neither program: the usage line
is `[OPTIONS] [FILE.sfmr ...]` under a heading naming SfM Explorer.

A panic in `run_with_args`, which is a bug in the viewer rather than a failure
to start, reaches Python as an exception: `sfm explorer` prints a Python
traceback ending in `pyo3_runtime.PanicException` and exits with status 1.

### What is in the wheel

The viewer, including its `mcp` feature (`sfmtool-py` depends on
`sfm-explorer` with default features), is part of the `_sfmtool` extension
module that maturin builds, so a wheel, the editable install from
`maturin develop` or `pixi install`, and a build from the sdist all contain it.
After Rust changes to the viewer, `pixi run maturin develop --release` rebuilds
it for `sfm explorer`, as for any other part of the extension.

## Options the viewer takes that this command does not

The viewer's command line also takes `--demo` (load generated demo data) and
`-h` / `--help`. `sfm explorer` does not forward either: `sfm explorer --help`
prints the Click help for this command, and `--demo` is refused as an unknown
option. Run the `sfm-explorer` binary for those, or call
`sfmtool._sfmtool.run_explorer(["--demo"])` from Python.

## Usage Examples

```bash
# Open one reconstruction
sfm explorer sfmr/solve.sfmr

# Compare two reconstructions in one 3D space
sfm explorer sfmr/solve-a.sfmr sfmr/solve-b.sfmr

# Let an agent drive the viewer on port 9000
sfm explorer --mcp 9000 sfmr/solve.sfmr

# Start from the stock panel grid
sfm explorer --no-default-layout sfmr/solve.sfmr
```

## Testing

[test_explorer_command.py](../../../tests/test_explorer_command.py) checks,
without opening a window and with `run_explorer` replaced in `explorer.py`,
that the command passes its options and files to `run_explorer` in the order
above, passes the default port for a bare `--mcp`, refuses an out-of-range port
and a missing file without calling it, exits with the status it returns, and
has the default `SIGINT` handler installed while it runs. It checks in a child
interpreter that `sfm explorer --help` imports none of numpy, OpenCV and
pycolmap. It also calls the real `run_explorer` in the test process with
`--help`, an unknown option and `--mcp 70000`, and checks that it returns 0, 2
and 2 without ending the process. The
viewer is tested in the `sfm-explorer` crate: its command-line
parser in [cli/tests.rs](../../../crates/sfm-explorer/src/cli/tests.rs), the
errors `run_with_args` returns before it creates an event loop in
[tests.rs](../../../crates/sfm-explorer/src/tests.rs), and the
running window in the `ui_basic` integration tests described in
[architecture.md](../../gui/architecture.md) § "Testing". Those window-opening
tests run the standalone `sfm-explorer` binary, so no automated test opens the
window through `sfm explorer`.

## Non-goals

The command does not forward the viewer's `--demo` option, and it does not run
the viewer in a separate process.
