# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import difflib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

from .._sfmtool.explorer import tool_catalog, validate_tool_call
from .explorer import DEFAULT_MCP_PORT

# How long one request to the endpoint may take. Generous because a few tools
# answer on a later frame or after a worker finishes, and the viewer gives up on
# its own GUI thread after ten seconds anyway.
_REQUEST_TIMEOUT_S = 600.0

# How long the check behind a local refusal waits for an endpoint, when it
# looks for a viewer with a different catalog to explain the refusal by.
_HINT_TIMEOUT_S = 2.0

# The four fields of a `tools/list` entry the local catalog states. A server
# may add others, depending on the protocol revision it answers at.
_COMPARED_FIELDS = ("name", "description", "inputSchema", "annotations")

_FALLBACK = (
    "pass --validate-tool-names-only to check only that the tool is in the "
    "viewer's own list and let the viewer check the arguments"
)


class _ViewerRefused(click.ClickException):
    """The viewer answered the call with a refusal."""

    exit_code = 3


class _NoEndpoint(click.ClickException):
    """No endpoint answered on the port."""

    exit_code = 4


@click.command("explorer-call")
@click.option(
    "--mcp",
    "mcp_port",
    type=click.IntRange(1, 65535),
    default=DEFAULT_MCP_PORT,
    show_default=True,
    metavar="PORT",
    help="The port of the viewer's MCP endpoint on 127.0.0.1, as given to "
    "`sfm explorer --mcp`.",
)
@click.option(
    "--list-tools",
    is_flag=True,
    help="Print each tool's name, kind and the first sentence of its "
    "description, from this sfmtool's catalog, without connecting.",
)
@click.option(
    "--schema",
    "show_schema",
    is_flag=True,
    help="Print TOOL's tools/list entry, with its description and input "
    "schema, from this sfmtool's catalog, without connecting.",
)
@click.option(
    "--validate-tool-names-only",
    is_flag=True,
    help="Check only that TOOL is in the running viewer's tool list, and let "
    "the viewer check the arguments. For a viewer built from other sources "
    "than this sfmtool, whose catalog differs.",
)
@click.option(
    "--batch",
    "batch_path",
    metavar="PATH",
    help="Call TOOL once for each line of PATH, a JSON-lines file of argument "
    "objects, or - for standard input, and print one JSON line per call.",
)
@click.option(
    "--field",
    "field_path",
    metavar="PATH",
    help="Print one field of the reply instead of the whole reply: keys and "
    "list indexes joined by dots, such as track.0.camera_image.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Where an image in the reply is written.",
)
@click.option(
    "--name",
    "image_name",
    metavar="STEM",
    help="File stem for an image in the reply. Defaults to the tool name.",
)
@click.option(
    "--wait-for-endpoint",
    type=click.FloatRange(min=0),
    default=0.0,
    metavar="SECONDS",
    help="Keep trying to reach the endpoint for up to SECONDS, for a viewer "
    "that is still starting.",
)
@click.option(
    "--wait-until-finished",
    is_flag=True,
    help="With get_background_tasks: repeat the call once a second until its "
    "reply shows no task running or waiting, then print that reply.",
)
@click.argument("tool", required=False)
@click.argument("arguments", required=False, metavar="[ARGS]")
def explorer_call(
    mcp_port,
    list_tools,
    show_schema,
    validate_tool_names_only,
    batch_path,
    field_path,
    out_dir,
    image_name,
    wait_for_endpoint,
    wait_until_finished,
    tool,
    arguments,
):
    """Call one tool on a running SfM Explorer's MCP endpoint.

    ARGS is the tool's arguments as one JSON object: given inline, as @FILE to
    read it from a file, or as - to read it from standard input. It defaults
    to {}. Reading it from a file or from standard input avoids quoting JSON
    for a shell.

    Before it connects, the command checks the call against the viewer's own
    tool catalog and argument parser, as compiled into this sfmtool: a tool
    name the catalog does not have is refused with the nearest names, and
    arguments that do not parse are refused with the message the viewer would
    send. It then compares the running viewer's tool list with that catalog,
    and refuses the call when they differ; --validate-tool-names-only checks
    only the tool name, against the running viewer's list.

    A text reply is printed whole. An image in the reply is written to
    --out-dir as STEM.png (STEM_1.png and on for more) and its path printed.
    The exit status is 0 when the viewer answered, 1 when the command refused
    the call, 2 for a usage error, 3 when the viewer refused the call, and 4
    when no endpoint answered. See specs/cli/visualization/explorer-call-command.md.
    """
    catalog = tool_catalog()

    if list_tools:
        if tool is not None or show_schema:
            raise click.UsageError("--list-tools takes neither TOOL nor --schema.")
        for entry in catalog:
            description = entry["advertised"]["description"]
            click.echo(
                f"{entry['name']} ({entry['kind']}): {_first_sentence(description)}"
            )
        click.echo(f"({len(catalog)} tools)")
        return

    if tool is None:
        raise click.UsageError("Name a TOOL, or pass --list-tools.")

    if show_schema:
        if arguments is not None:
            raise click.UsageError("--schema takes no ARGS.")
        entry = _find_local(catalog, tool, mcp_port, check_viewer=False)
        click.echo(json.dumps(entry["advertised"], indent=2, ensure_ascii=False))
        return

    if wait_until_finished and tool != "get_background_tasks":
        raise click.UsageError(
            "--wait-until-finished works only with get_background_tasks."
        )
    if batch_path is not None:
        if arguments is not None:
            raise click.UsageError(
                "--batch reads the arguments; give no ARGS beside it."
            )
        if wait_until_finished:
            raise click.UsageError(
                "--batch and --wait-until-finished do not go together."
            )
        calls = _read_batch(batch_path)
    else:
        calls = [_read_arguments(arguments)]

    endpoint = _Endpoint(mcp_port)
    # Each call's local refusal, or None. A batch keeps going past a refused
    # line, which is reported in its output; a single call stops here.
    refusals = []
    if not validate_tool_names_only:
        _find_local(catalog, tool, mcp_port, check_viewer=True)
        for call_arguments in calls:
            refusals.append(_local_refusal(tool, call_arguments))
        if batch_path is None and refusals[0] is not None:
            raise click.ClickException(refusals[0] + _mismatch_hint(endpoint, catalog))
    else:
        refusals = [None] * len(calls)

    served = endpoint.tools(wait_s=wait_for_endpoint)
    if validate_tool_names_only:
        names = [entry["name"] for entry in served]
        if tool not in names:
            raise click.ClickException(
                f"The viewer on port {mcp_port} has no tool named {tool!r}."
                + _suggestion(tool, names)
            )
    else:
        difference = _catalog_difference(catalog, served)
        if difference:
            raise click.ClickException(
                f"The viewer on port {mcp_port} serves a different tool catalog "
                f"from the one in this sfmtool, so the call cannot be checked "
                f"here. {difference} If the viewer was built from this "
                f"checkout, rebuild the extension with `pixi run maturin develop "
                f"--release`; otherwise {_FALLBACK}."
            )

    if batch_path is not None:
        stem = image_name or tool
        for line, (call_arguments, refusal) in enumerate(zip(calls, refusals), 1):
            record = {"arguments": call_arguments}
            if refusal is not None:
                record["error"] = refusal
            else:
                try:
                    result = endpoint.call(tool, call_arguments)
                except _ViewerRefused as error:
                    record["error"] = error.message
                else:
                    images = _save_images(result, out_dir, f"{stem}_{line}")
                    if images:
                        record["images"] = [str(path) for path in images]
                    text = _texts(result)
                    if result.get("isError"):
                        record["error"] = "\n".join(text)
                    else:
                        record["reply"] = _field(_parse_reply(text), field_path)
            click.echo(json.dumps(record, ensure_ascii=False))
        return

    (call_arguments,) = calls
    while True:
        result = endpoint.call(tool, call_arguments)
        if not (wait_until_finished and not result.get("isError")):
            break
        if not _still_working(_parse_reply(_texts(result))):
            break
        time.sleep(1.0)

    for path in _save_images(result, out_dir, image_name or tool):
        click.echo(f"Wrote image {path}")
    text = _texts(result)
    if result.get("isError"):
        raise _ViewerRefused("The viewer refused the call: " + "\n".join(text))
    if field_path is None:
        click.echo("\n".join(text))
        return
    value = _field(_parse_reply(text), field_path)
    click.echo(
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    )


# ── Reading the call ─────────────────────────────────────────────────────


def _read_arguments(arguments):
    """The ARGS argument as a dict."""
    if arguments is None:
        return {}
    if arguments == "-":
        text, source = sys.stdin.read(), "standard input"
    elif arguments.startswith("@"):
        path = Path(arguments[1:])
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise click.ClickException(f"Could not read ARGS from {path}: {error}")
        source = str(path)
    else:
        text, source = arguments, "ARGS"
    return _parse_object(text, source)


def _read_batch(batch_path):
    """Each non-blank line of the --batch file as a dict."""
    if batch_path == "-":
        text, source = sys.stdin.read(), "standard input"
    else:
        try:
            text = Path(batch_path).read_text(encoding="utf-8")
        except OSError as error:
            raise click.ClickException(f"Could not read {batch_path}: {error}")
        source = batch_path
    return [
        _parse_object(line, f"line {number} of {source}")
        for number, line in enumerate(text.splitlines(), 1)
        if line.strip()
    ]


def _parse_object(text, source):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise click.ClickException(f"{source} is not JSON: {error}")
    if not isinstance(value, dict):
        raise click.ClickException(
            f"{source} must be a JSON object of arguments, not {type(value).__name__}."
        )
    return value


# ── Checking the call against this sfmtool's catalog ─────────────────────


def _find_local(catalog, tool, mcp_port, check_viewer):
    """TOOL's catalog entry, or a refusal naming the nearest names."""
    for entry in catalog:
        if entry["name"] == tool:
            return entry
    names = [entry["name"] for entry in catalog]
    message = (
        f"There is no tool named {tool!r} in this sfmtool's catalog."
        + _suggestion(tool, names)
    )
    if check_viewer:
        message += _mismatch_hint(_Endpoint(mcp_port), catalog)
    raise click.ClickException(message)


def _local_refusal(tool, call_arguments):
    """The viewer parser's message for these arguments, or None."""
    try:
        return validate_tool_call(tool, call_arguments)
    except (ValueError, TypeError) as error:
        return str(error)


def _suggestion(tool, names):
    close = difflib.get_close_matches(tool, names, n=3, cutoff=0.6)
    if not close:
        return " --list-tools lists them."
    return " Did you mean " + " or ".join(repr(name) for name in close) + "?"


def _catalog_difference(catalog, served):
    """A sentence or two naming where the served tool list and the local
    catalog differ, or "" when they agree."""
    local = {entry["name"]: entry["advertised"] for entry in catalog}
    remote = {entry.get("name"): entry for entry in served}
    only_remote = [name for name in remote if name not in local]
    only_local = [name for name in local if name not in remote]
    changed = []
    for name, advertised in local.items():
        if name in remote:
            fields = [
                field
                for field in _COMPARED_FIELDS
                if remote[name].get(field) != advertised.get(field)
            ]
            if fields:
                changed.append(f"{name} ({', '.join(fields)})")
    parts = []
    if only_remote:
        parts.append(f"Only the viewer has {_names(only_remote)}.")
    if only_local:
        parts.append(f"Only this sfmtool has {_names(only_local)}.")
    if changed:
        parts.append(f"The two differ in {_names(changed)}.")
    return " ".join(parts)


def _names(names, most=8):
    shown = ", ".join(names[:most])
    if len(names) > most:
        shown += f" and {len(names) - most} more"
    return shown


def _mismatch_hint(endpoint, catalog):
    """For a call refused here: when a viewer answers on the port and its
    catalog differs, a sentence saying so and naming the fallback."""
    try:
        served = endpoint.tools(wait_s=0.0, timeout_s=_HINT_TIMEOUT_S)
    except click.ClickException:
        return ""
    difference = _catalog_difference(catalog, served)
    if not difference:
        return ""
    return (
        f"\nThe viewer on port {endpoint.port} serves a different tool catalog "
        f"from the one in this sfmtool. {difference} If the call is meant for "
        f"that viewer, {_FALLBACK}."
    )


# ── The endpoint ─────────────────────────────────────────────────────────


class _Endpoint:
    """JSON-RPC over HTTP to one viewer's MCP endpoint.

    The endpoint is stateless, so each request is a POST of its own, with no
    `initialize` before it and no session.
    """

    def __init__(self, port):
        self.port = port
        self.url = f"http://127.0.0.1:{port}/mcp"
        self._id = 0

    def tools(self, wait_s, timeout_s=_REQUEST_TIMEOUT_S):
        """The `tools/list` entries, trying for up to `wait_s` seconds."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                return self.rpc("tools/list", timeout_s=timeout_s)["tools"]
            except _NoEndpoint:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.25)

    def call(self, tool, arguments):
        """The `tools/call` result: `{"content": [...], "isError": ...}`."""
        return self.rpc("tools/call", {"name": tool, "arguments": arguments})

    def rpc(self, method, params=None, timeout_s=_REQUEST_TIMEOUT_S):
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise _NoEndpoint(
                f"No SfM Explorer MCP endpoint answered at {self.url}: {reason}. "
                f"Start one with `sfm explorer --mcp {self.port}`, or pass "
                f"--wait-for-endpoint SECONDS while it starts."
            )
        if raw.startswith(("event:", "data:")):
            raw = "\n".join(
                line[5:] for line in raw.splitlines() if line.startswith("data:")
            )
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError:
            raise _NoEndpoint(
                f"What answered at {self.url} is not an MCP endpoint: {raw[:200]!r}"
            )
        if "error" in reply:
            error = reply["error"]
            message = error.get("message") if isinstance(error, dict) else None
            raise _ViewerRefused(
                "The viewer refused the call: " + (message or json.dumps(error))
            )
        return reply["result"]


# ── The reply ────────────────────────────────────────────────────────────


def _texts(result):
    return [c["text"] for c in result.get("content", []) if c.get("type") == "text"]


def _parse_reply(texts):
    """The first text block, as JSON where it is JSON."""
    if not texts:
        return None
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return texts[0]


def _save_images(result, out_dir, stem):
    images = [c for c in result.get("content", []) if c.get("type") == "image"]
    if images:
        out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, content in enumerate(images):
        path = out_dir / (f"{stem}.png" if index == 0 else f"{stem}_{index}.png")
        path.write_bytes(base64.b64decode(content["data"]))
        paths.append(path.resolve())
    return paths


def _field(value, field_path):
    """The field `field_path` names inside `value`, or `value` with none."""
    if field_path is None:
        return value
    for part in field_path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and _is_index(part, len(value)):
            value = value[int(part)]
        else:
            raise click.ClickException(
                f"The reply has no field {field_path!r}: no {part!r}."
            )
    return value


def _is_index(part, length):
    try:
        return -length <= int(part) < length
    except ValueError:
        return False


def _still_working(reply):
    """Whether a `get_background_tasks` reply shows a task running or waiting."""
    if not isinstance(reply, dict):
        return False
    if "running" in reply:
        # One task, by its operation_id.
        return bool(reply["running"])
    return any(
        isinstance(queue, dict) and (queue.get("running") or queue.get("waiting"))
        for queue in reply.values()
    )


def _first_sentence(description):
    text = " ".join(description.split())
    end = text.find(". ")
    return text if end < 0 else text[: end + 1]
