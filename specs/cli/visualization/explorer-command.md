# `sfm explorer` Command

## Purpose

`sfm explorer` opens SfM Explorer, the sfmtool desktop viewer, from the Python
command line. The viewer shows one or more reconstructions (`.sfmr` files) in a
3D window with panels for the cameras, images, points and tracks, so a person
can look at a solve and find where it went wrong. The viewer is compiled into
the `sfmtool._sfmtool` extension module, so it comes with `pip install
sfmtool`. The command runs it in a child Python process
(`python -m sfmtool._explorer`) with the arguments given, waits for the window
to close, and exits with the child's exit status. The viewer itself is
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
| `--mcp [PORT]` | int | off; `8787` when given without a number | Host a Model Context Protocol endpoint on `127.0.0.1:PORT`, so an agent can drive the viewer. `0` takes an ephemeral port, which the viewer prints at startup. See [mcp-server.md](../../gui/mcp-server.md). |
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

## How the viewer is launched

The command builds the viewer's command line: `--mcp PORT` (when `--mcp` was
given), then `--no-default-layout` (when given), then the files. It runs
`[sys.executable, "-m", "sfmtool._explorer", *that]` with `subprocess.run`,
so the child is the same Python interpreter as the command, and calls
`sys.exit` with the child's return code. `viewer_command` in `explorer.py`
builds that list, and `VIEWER_MODULE` names the module.

[_explorer.py](../../../src/sfmtool/_explorer.py) is the child's program. It
restores the default `SIGINT` handler, so Ctrl+C ends the viewer as it ends any
other program, and calls `sfmtool._sfmtool.run_explorer(sys.argv[1:])`.
`run_explorer` is a root-level function of the extension, in
[lib.rs](../../../crates/sfmtool-py/src/lib.rs) of `sfmtool-py`. It releases
the GIL and calls `sfm_explorer::run_with_args`, which parses the viewer's
command line, opens the window, and returns when the window closes.

The viewer runs in a child process rather than in the `sfm` process because it
needs a process to itself:

- It creates a `winit` event loop, which macOS allows only on the process's
  main thread, and which `winit` allows only once per process.
- It ends the process with `std::process::exit` on an option it does not
  recognize (status 2), on `--mcp` in a build without the `mcp` feature
  (status 2), and when the MCP endpoint cannot bind its port (status 1).
- It initializes the global `env_logger` logger and, on Windows, sets the
  process's DPI awareness.

`pixi run gui` runs the same viewer from a source checkout through a different
entry point: the `sfm-explorer` crate's own `sfm-explorer` binary,
[main.rs](../../../crates/sfm-explorer/src/main.rs), whose `main` calls
`sfm_explorer::run`, which passes `std::env::args()` without the program name
to `run_with_args`. Both accept the same command line.

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
option. Run `python -m sfmtool._explorer --demo` or
`python -m sfmtool._explorer --help`, or the `sfm-explorer` binary, for those.

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
without opening a window, that the command passes its options and files to
`python -m sfmtool._explorer` in the order above and exits with the child's
status, and that `python -m sfmtool._explorer` reaches the viewer in the built
extension: `--help` prints the viewer's usage and exits 0, and an unknown option
exits 2. The viewer is tested in the `sfm-explorer` crate: its command-line
parser in [cli/tests.rs](../../../crates/sfm-explorer/src/cli/tests.rs), and the
running window in the `ui_basic` integration tests described in
[architecture.md](../../gui/architecture.md) § "Testing".

## Non-goals

The command does not run the viewer in its own process, and it does not look
for a viewer executable on the `PATH`.
