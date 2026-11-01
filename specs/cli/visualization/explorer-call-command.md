# `sfm explorer-call` Command

## Purpose

`sfm explorer-call` calls one tool on a running SfM Explorer, the sfmtool
desktop viewer, from the command line. A viewer started with `sfm explorer
--mcp` hosts a Model Context Protocol (MCP) endpoint, a small HTTP server on
the local machine through which a program can read what the viewer shows,
move its selection and camera, edit a loaded reconstruction and take
screenshots ([mcp-server.md](../../gui/mcp-server.md)). An agent registered
with that endpoint calls its tools directly; this command is for everything
else that wants to call one: a script, a skill, or an agent whose session did
not register the endpoint. It prints the reply, writes any screenshot to a
file, and exits with a status that says whether the call was answered.

The command is not a general MCP client. It knows the viewer's own tool
catalog and argument parser, compiled into the same `sfmtool._sfmtool`
extension that runs the viewer for `sfm explorer`, and it checks each call
against them before it connects: a tool name that is not in the catalog is
refused with the nearest names, and arguments the viewer would refuse are
refused with the viewer's own message. So a mistyped call fails at once, with
nothing sent to the window a person may be watching, and the check is the
viewer's own code rather than a copy of it in Python.

## Command Syntax

```bash
sfm explorer-call [--mcp PORT] TOOL [ARGS]            # one call
sfm explorer-call [--mcp PORT] TOOL --batch PATH      # one call per line of PATH
sfm explorer-call --list-tools                        # the catalog, offline
sfm explorer-call --schema TOOL                       # one tool's entry, offline
```

The command is implemented in
[explorer_call.py](../../../src/sfmtool/_commands/explorer_call.py) and
registered under the Visualization category in
[cli.py](../../../src/sfmtool/cli.py), after `explorer`. The catalog and the
parse it checks against are the bindings `sfmtool.explorer.tool_catalog` and
`sfmtool.explorer.validate_tool_call`, in
[explorer.rs](../../../crates/sfmtool-py/src/explorer.rs) of `sfmtool-py`,
which call `sfm_explorer::tool_catalog` and `sfm_explorer::validate_tool_call`
in [call_check.rs](../../../crates/sfm-explorer/src/mcp/call_check.rs). Those
two are the viewer crate's only public MCP items, and they exist only in a
build with its `mcp` feature, which `sfmtool-py` always turns on.

## Arguments and options

| Argument / option | Type | Default | Meaning |
|-------------------|------|---------|---------|
| `TOOL` | tool name | none | The tool to call, as `tools/list` names it. Required except with `--list-tools`. |
| `ARGS` | JSON object | `{}` | The tool's arguments: inline, `@FILE` to read them from a file, or `-` to read them from standard input. A file or standard input avoids quoting JSON for a shell, which in Windows PowerShell 5.1 strips the double quotes from an argument passed to a program. A file is read as UTF-8 or, with a byte order mark, as UTF-16, and a leading byte order mark is dropped from a file or from standard input, as PowerShell 5.1 writes one; in PowerShell, quote `'@FILE'`, since a bare `@name` is splatting. |
| `--mcp PORT` | int, 1 to 65535 | `8787` | The port of the endpoint on `127.0.0.1`. The default is `DEFAULT_MCP_PORT` from [explorer.py](../../../src/sfmtool/_commands/explorer.py), the port `sfm explorer --mcp` takes when given no number. |
| `--list-tools` | flag | off | Print each tool of this sfmtool's catalog as `name (kind): first sentence of its description`, then the count. Connects to nothing. |
| `--schema` | flag | off | Print `TOOL`'s `tools/list` entry from this sfmtool's catalog as indented JSON: its name, description, `inputSchema` and annotations. Connects to nothing. |
| `--validate-tool-names-only` | flag | off | Skip the checks against this sfmtool's catalog and parser; check only that `TOOL` is in the running viewer's `tools/list`, and let the viewer check the arguments. For a viewer whose catalog differs (§ "When the viewer was built from other sources"). |
| `--batch PATH` | path or `-` | none | Call `TOOL` once for each non-blank line of `PATH`, a JSON-lines file of argument objects, or `-` for standard input. Replaces `ARGS`. |
| `--field PATH` | dotted path | none | Print one field of the reply instead of the whole reply: object keys and list indexes joined by dots, such as `track.0.camera_image`. A string is printed as it is, anything else as JSON. With `--batch`, each line's `reply` is that field. |
| `--out-dir DIR` | directory | `.` | Where an image in the reply is written. Created when missing. |
| `--name STEM` | file stem | the tool name | The file stem of an image in the reply. |
| `--wait-for-endpoint SECONDS` | number | `0` | Keep trying to reach the endpoint for up to `SECONDS`, for a viewer that is still starting. |
| `--wait-until-finished` | flag | off | Only with `get_background_tasks`: repeat the call once a second until its reply shows no task running or waiting, then print that last reply. With `operation_id` it waits for that one task. |

`--list-tools` and `--schema` read only the catalog compiled into this
sfmtool, so they describe the viewer `sfm explorer` runs, and they work with
no viewer running.

## What the command checks, and where

A call goes through these steps, in order, and stops at the first refusal:

1. **ARGS is read** and must be one JSON object; anything else is refused.
2. **The tool name is looked up** in this sfmtool's catalog. A name that is not
   there is refused with up to three close names (`get_scen` gives "Did you
   mean 'get_scene'?"), or a pointer to `--list-tools` when none is close.
3. **The arguments are parsed** by `validate_tool_call`, which runs
   `tools::parse`, the function the endpoint's `tools/call` runs before it
   hands a command to the viewer. A refusal there is printed with the message
   the endpoint would have sent as its JSON-RPC `invalid_params` error, word
   for word. The parse is the whole of what is checked here: the argument
   names (every schema is closed), their types and ranges, the choices among
   alternatives, and the forms of handles such as a point id. It reads no
   application state.
4. **The endpoint is reached**, and its `tools/list` is compared with the
   catalog (§ "When the viewer was built from other sources").
5. **The call is sent** as a `tools/call`.

What the parse cannot know is decided only by the running viewer, and comes
back as the viewer's refusal (exit status 3): whether a `reconstruction_label`
names a loaded reconstruction, whether an index or a point id is in range,
whether a panel is open to be screenshotted, whether a bench item exists, and
every other check against what is loaded.

When step 2 or 3 refuses a call and an endpoint answers on the port within two
seconds, the command also fetches its `tools/list`; if that differs from the
catalog, the refusal ends with a note saying so and naming
`--validate-tool-names-only`, since the call may be right for that viewer.

### When the viewer was built from other sources

The viewer on the port need not be the one this sfmtool's extension holds. A
contributor who changes the MCP tools runs `pixi run gui`, or a
`cargo build --release` binary, which is built from the working tree, while
`sfm explorer-call` uses the extension that `pixi run maturin develop` built
last. A check against an older or newer parser would refuse calls that viewer
accepts, or pass calls it refuses.

So before each call the command fetches the viewer's `tools/list` and compares
each entry with the catalog, on the four fields the catalog states: `name`,
`description`, `inputSchema` and `annotations`. Order is not compared, and
neither is any other field a server adds for the protocol revision it answers
at. When the two differ the call is refused before it is sent, with a message
that names the tools only the viewer has, the tools only this sfmtool has, and
the tools whose entries differ with the fields that differ (eight names of
each at most, then a count), and that offers two ways on: rebuild the
extension with `pixi run maturin develop --release` when the viewer is from
this checkout, or pass `--validate-tool-names-only`.

`tools/list` is the comparison because nothing cheaper identifies the
catalog: the `serverInfo` the endpoint reports on `initialize` carries the
crate version, which does not change when a tool does. The list is a single
loopback request, a few hundred kilobytes, before each call.

With `--validate-tool-names-only` the command skips steps 2 and 3 and the
comparison, fetches the viewer's `tools/list`, refuses a `TOOL` that is not in
it (with close names from that list), and sends the call. The viewer then
checks the arguments, and a refusal of them comes back as a JSON-RPC error
(exit status 3) with the same message step 3 would have printed from the
viewer's own parser.

## The exchange

The endpoint is JSON-RPC 2.0 over HTTP POST to `http://127.0.0.1:PORT/mcp`.
It is stateless, so the command sends no `initialize` and keeps no session:
each `tools/list` and `tools/call` is a POST of its own, with
`Accept: application/json, text/event-stream`. A reply that arrives as a
server-sent event is read from its `data:` lines. Each request may take up to
600 seconds, since a few tools answer only on a later frame.

## Output and exit status

A text reply is printed whole, as the viewer sent it; for a tool that answers
with JSON that is one line of compact JSON. Each image in the reply is
decoded and written to `--out-dir` as `STEM.png`, then `STEM_1.png` and on for
further images, and `Wrote image <absolute path>` is printed for each before
the text. `screenshot` answers with a caption line, the image, and with
`include_widgets` the widget listing; all its text blocks are printed.

| Status | Meaning |
|--------|---------|
| 0 | The viewer answered the call. |
| 1 | The command refused the call before sending it: ARGS is not a JSON object, an unknown tool, arguments the parser refuses, a different catalog on the port, or a `--field` the reply does not have. |
| 2 | A usage error, such as `TOOL` missing or `--wait-until-finished` with another tool. |
| 3 | The viewer refused the call: a tool result with `isError`, or a JSON-RPC error. The message is printed after "The viewer refused the call:". |
| 4 | No endpoint answered on the port, or what answered is not one. |

Messages for statuses 1 to 4 go to standard error, after `Error:`.

With `--batch` the command prints one JSON line per argument line, in order:
`{"arguments": {...}, "reply": ...}` for an answer, with the first text block
parsed as JSON where it is JSON and `--field` applied, and
`{"arguments": {...}, "error": "..."}` for a call refused by the parser here
(which is not sent) or by the viewer, or for a reply that lacks the `--field`. A line whose reply carries images also
has `"images"`, the paths written, with the stem `STEM_<line>` where `<line>`
counts argument lines from 1. The tool name and the catalog are checked once,
before the first call, and stop the whole batch as they stop a single call;
per-line refusals do not change the exit status. A batch exists so that a
probe that reads hundreds of indexes runs in one process with one catalog
check, rather than starting `sfm` once per index.

## Loading

The command module imports Click, the standard library modules for HTTP and
JSON, and the `explorer` extension submodule; it imports none of numpy,
OpenCV and pycolmap (see [cli/README.md](../README.md) § "How `sfm` loads its
commands").

## Usage Examples

```bash
# What the viewer offers, and what one tool takes
sfm explorer-call --list-tools
sfm explorer-call --schema get_point

# Read the scene of a viewer on port 9000
sfm explorer-call --mcp 9000 get_scene

# One field of a reply
sfm explorer-call --mcp 9000 get_point '{"point": 12}' --field track

# A screenshot of one panel, written to shots/track_view.png
sfm explorer-call --mcp 9000 screenshot '{"panel_name": "track_view"}' --out-dir shots --name track_view

# Arguments from a file, which needs no shell quoting
sfm explorer-call --mcp 9000 set_view @view.json

# Start a bundle adjustment, then wait until it and anything queued behind it finish
sfm explorer-call --mcp 9000 bundle_adjust '{"reconstruction_label": "bull"}'
sfm explorer-call --mcp 9000 get_background_tasks --wait-until-finished

# Wait for a viewer that is still starting
sfm explorer-call --mcp 9000 --wait-for-endpoint 60 get_scene

# A viewer built from a newer checkout than this sfmtool
sfm explorer-call --mcp 9000 get_scene --validate-tool-names-only

# Many reads in one process
sfm explorer-call --mcp 9000 get_point --batch points.jsonl > replies.jsonl
```

## Testing

[test_explorer_call_command.py](../../../tests/test_explorer_call_command.py)
checks the offline modes, the refusals made before connecting (an unknown tool
with its suggestion, a bad argument whose message equals
`validate_tool_call`'s, ARGS that is not an object), a closed port's status 4,
and that the command imports none of numpy, OpenCV and pycolmap. Against a
small JSON-RPC server in the test process that serves this sfmtool's catalog,
or a changed copy of it, it checks ARGS read inline, from `@FILE` and from
standard input, `--field`, an image written under `--out-dir` and `--name`, a
viewer refusal's status 3, the refusal of a different catalog and the note on
a local refusal, `--validate-tool-names-only` in both directions,
`--wait-until-finished` and `--batch`.
[test_explorer_rust_bindings.py](../../../tests/rust_bindings/explorer/test_explorer_rust_bindings.py)
checks the two bindings, and the viewer crate's
`the_exported_catalog_and_validation_match_what_the_endpoint_serves`, in
[tests/server.rs](../../../crates/sfm-explorer/src/mcp/tests/server.rs),
checks over a real socket that the exported catalog is the endpoint's
`tools/list` and that the exported parse gives the endpoint's error messages.
No automated test calls a real viewer window through the command, since the
windowed `ui_basic` suite runs the viewer from Rust and has no Python
environment to run `sfm` in.

## Non-goals

The command does not call a tool the catalog (or, with
`--validate-tool-names-only`, the running viewer) does not list, does not
speak to any MCP server other than SfM Explorer's, does not keep a session
across calls, and does not read the running viewer's own schemas for
`--list-tools` or `--schema`.
