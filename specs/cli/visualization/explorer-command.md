# `sfm explorer` Command

## Purpose

`sfm explorer` opens SfM Explorer, the sfmtool desktop viewer, from the Python
command line. The viewer shows one or more reconstructions (`.sfmr` files) in a
3D window with panels for the cameras, images, points and tracks, so a person
can look at a solve and find where it went wrong. The command does not contain
the viewer: it finds a separate executable named `launch-sfm-explorer` on the
`PATH`, runs it as a child process with the arguments given, waits for the
window to close, and exits with the viewer's exit status. The viewer itself is
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

The command looks up `launch-sfm-explorer` with `shutil.which`, so only the
`PATH` is searched. When it is not found, the command stops with a Click error
that names the missing executable and the build command
`pixi run cargo build --release -p sfmtool-py`. When it is found, the command
passes `--mcp PORT` (when `--mcp` was given), then `--no-default-layout` (when
given), then the files, and calls `sys.exit` with the child's return code.

`launch-sfm-explorer` is a binary target of the `sfmtool-py` crate,
[launch-sfm-explorer.rs](../../../crates/sfmtool-py/src/bin/launch-sfm-explorer.rs),
whose `main` calls `sfm_explorer::run`. `pixi run gui` runs a different binary,
`sfm-explorer`, built from the `sfm-explorer` crate's
[main.rs](../../../crates/sfm-explorer/src/main.rs), whose `main` also calls
`sfm_explorer::run`. The two executables run the same viewer code and accept the
same command line, which `sfm_explorer::run` parses.

### Where `launch-sfm-explorer` comes from

maturin builds only the `_sfmtool` extension module from `sfmtool-py`, not its
binary targets, so neither a wheel built by `maturin build` nor the editable
install from `maturin develop` or `pixi install` contains
`launch-sfm-explorer`. A source checkout gets it from
`pixi run cargo build --release -p sfmtool-py`, which writes
`target/release/launch-sfm-explorer`; that directory then has to be on the
`PATH` for `sfm explorer` to find it. Without that, `pixi run gui` runs the
viewer directly.

## Options the viewer takes that this command does not

The viewer's command line also takes `--demo` (load generated demo data) and
`-h` / `--help`. `sfm explorer` does not forward either: `sfm explorer --help`
prints the Click help for this command, and `--demo` is refused as an unknown
option. Run the viewer binary directly for those.

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

No automated test runs this command. The viewer it launches is tested in the
`sfm-explorer` crate: its command-line parser in
[cli/tests.rs](../../../crates/sfm-explorer/src/cli/tests.rs), and the running
window in the `ui_basic` integration tests described in
[architecture.md](../../gui/architecture.md) § "Testing".

## Non-goals

The command does not build or locate the viewer anywhere other than the `PATH`;
`pixi run gui` builds and runs it from a source checkout.
