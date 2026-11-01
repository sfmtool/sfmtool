# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``sfm explorer-call`` that need no viewer.

The offline parts (the tool list, a schema, the refusals made before
connecting) run against no endpoint at all. The HTTP side runs against a small
JSON-RPC server in this process that serves this sfmtool's own catalog, or a
changed copy of it, as its ``tools/list`` and answers ``tools/call`` with
canned results. A live call to a real viewer is described in
``specs/cli/visualization/explorer-call-command.md`` § "Testing".
"""

import base64
import copy
import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.explorer import tool_catalog, validate_tool_call


def _closed_port():
    """A port nothing listens on, so a connection to it is refused."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(args, input=None):
    return CliRunner().invoke(main, ["explorer-call", *args], input=input)


class _FakeViewer:
    """A JSON-RPC endpoint that records each `tools/call` it is sent."""

    def __init__(self, tools=None, answer=None):
        self.tools = (
            tools if tools is not None else [t["advertised"] for t in tool_catalog()]
        )
        self.answer = answer or (
            lambda name, arguments: {
                "content": [{"type": "text", "text": json.dumps(arguments)}]
            }
        )
        self.calls = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body["method"] == "tools/list":
                    reply = {"result": {"tools": fake.tools}}
                else:
                    params = body["params"]
                    fake.calls.append((params["name"], params["arguments"]))
                    reply = fake.answer(params["name"], params["arguments"])
                    if "error" not in reply:
                        reply = {"result": reply}
                data = json.dumps({"jsonrpc": "2.0", "id": body["id"], **reply})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


# ── Offline ──────────────────────────────────────────────────────────────


def test_list_tools_prints_every_tool_with_its_kind():
    result = _run(["--list-tools"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    catalog = tool_catalog()
    assert lines[-1] == f"({len(catalog)} tools)"
    assert len(lines) == len(catalog) + 1
    assert lines[0].startswith(f"{catalog[0]['name']} ({catalog[0]['kind']}): ")
    assert any(line.startswith("save_reconstruction (save): ") for line in lines)


def test_schema_prints_the_tool_s_advertised_entry():
    result = _run(["--schema", "get_point"])
    assert result.exit_code == 0, result.output
    entry = next(t for t in tool_catalog() if t["name"] == "get_point")
    assert json.loads(result.output) == entry["advertised"]


def test_an_unknown_tool_is_refused_with_the_nearest_names():
    result = _run(["--mcp", str(_closed_port()), "get_scen"])
    assert result.exit_code == 1
    assert "There is no tool named 'get_scen'" in result.output
    assert "Did you mean 'get_scene'" in result.output
    result = _run(["--schema", "zzzz"])
    assert result.exit_code == 1
    assert "--list-tools lists them" in result.output


def test_a_bad_argument_is_refused_with_the_viewer_s_own_message():
    arguments = {"offset": "soon"}
    expected = validate_tool_call("list_camera_images", arguments)
    assert expected
    result = _run(
        ["--mcp", str(_closed_port()), "list_camera_images", json.dumps(arguments)]
    )
    assert result.exit_code == 1
    assert result.output == f"Error: {expected}\n"


def test_args_that_are_not_a_json_object_are_refused():
    result = _run(["get_scene", "[1]"])
    assert result.exit_code == 1
    assert "must be a JSON object" in result.output
    result = _run(["get_scene", "{nope"])
    assert result.exit_code == 1
    assert "ARGS is not JSON" in result.output


def test_wait_until_finished_needs_get_background_tasks():
    result = _run(["get_scene", "--wait-until-finished"])
    assert result.exit_code == 2
    assert "only with get_background_tasks" in result.output


def test_no_endpoint_exits_4():
    port = _closed_port()
    result = _run(["--mcp", str(port), "get_scene"])
    assert result.exit_code == 4
    assert f"No SfM Explorer MCP endpoint answered at http://127.0.0.1:{port}/mcp" in (
        result.output
    )


def test_explorer_call_help_imports_no_heavy_library():
    code = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from sfmtool.cli import main\n"
        "r = CliRunner().invoke(main, ['explorer-call', '--list-tools'])\n"
        "assert r.exit_code == 0, r.output\n"
        "print(sorted(m for m in ('numpy', 'cv2', 'pycolmap') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


# ── Over HTTP, against a fake viewer ─────────────────────────────────────


@pytest.mark.parametrize("form", ["inline", "file", "stdin"])
def test_args_are_read_inline_from_a_file_or_from_stdin(form, tmp_path):
    arguments = {"point": "pt3d_ab_12"}
    text = json.dumps(arguments)
    with _FakeViewer() as viewer:
        if form == "inline":
            result = _run(["--mcp", str(viewer.port), "get_point", text])
        elif form == "file":
            path = tmp_path / "args.json"
            path.write_text(text, encoding="utf-8")
            result = _run(["--mcp", str(viewer.port), "get_point", f"@{path}"])
        else:
            result = _run(["--mcp", str(viewer.port), "get_point", "-"], input=text)
    assert result.exit_code == 0, result.output
    assert viewer.calls == [("get_point", arguments)]
    assert json.loads(result.output) == arguments


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_args_written_by_windows_powershell_are_read(encoding, tmp_path):
    """Windows PowerShell 5.1 writes files as UTF-8 or UTF-16 with a byte order
    mark, and puts one before what it pipes to a program."""
    arguments = {"point": 12}
    text = json.dumps(arguments)
    path = tmp_path / "args.json"
    path.write_text(text, encoding=encoding)
    with _FakeViewer() as viewer:
        from_file = _run(["--mcp", str(viewer.port), "get_point", f"@{path}"])
        from_stdin = _run(
            ["--mcp", str(viewer.port), "get_point", "-"], input="\ufeff" + text
        )
    assert from_file.exit_code == 0, from_file.output
    assert from_stdin.exit_code == 0, from_stdin.output
    assert viewer.calls == [("get_point", arguments)] * 2


def test_no_args_sends_an_empty_object():
    with _FakeViewer() as viewer:
        result = _run(["--mcp", str(viewer.port), "get_scene"])
    assert result.exit_code == 0, result.output
    assert viewer.calls == [("get_scene", {})]


def test_field_prints_one_field_of_the_reply():
    reply = {"track": [{"camera_image": "a.jpg"}, {"camera_image": "b.jpg"}]}

    def answer(name, arguments):
        return {"content": [{"type": "text", "text": json.dumps(reply)}]}

    with _FakeViewer(answer=answer) as viewer:
        port = str(viewer.port)
        one = _run(["--mcp", port, "get_point", '{"point": 1}', "--field", "track.1"])
        text = _run(
            [
                "--mcp",
                port,
                "get_point",
                '{"point": 1}',
                "--field",
                "track.0.camera_image",
            ]
        )
        missing = _run(["--mcp", port, "get_point", '{"point": 1}', "--field", "nope"])
    assert json.loads(one.output) == {"camera_image": "b.jpg"}
    assert text.output == "a.jpg\n"
    assert missing.exit_code == 1 and "no field 'nope'" in missing.output


def test_an_image_is_written_to_out_dir_under_name(tmp_path):
    png = b"\x89PNG fake bytes"

    def answer(name, arguments):
        return {
            "content": [
                {"type": "text", "text": "2×2 px. The window."},
                {
                    "type": "image",
                    "data": base64.b64encode(png).decode(),
                    "mimeType": "image/png",
                },
            ]
        }

    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "screenshot",
                "--out-dir",
                str(tmp_path / "shots"),
                "--name",
                "window",
            ]
        )
    assert result.exit_code == 0, result.output
    path = tmp_path / "shots" / "window.png"
    assert path.read_bytes() == png
    assert result.output == f"Wrote image {path.resolve()}\n2×2 px. The window.\n"


def test_a_viewer_refusal_exits_3():
    def answer(name, arguments):
        return {
            "content": [
                {"type": "text", "text": "No reconstruction is labelled gamma."}
            ],
            "isError": True,
        }

    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "select_reconstruction",
                '{"reconstruction_label": "gamma"}',
            ]
        )
    assert result.exit_code == 3
    assert "The viewer refused the call: No reconstruction is labelled gamma." in (
        result.output
    )


def test_a_different_catalog_is_refused_and_names_the_difference_and_fallback():
    tools = copy.deepcopy([t["advertised"] for t in tool_catalog()])
    tools = [t for t in tools if t["name"] != "get_point"]
    tools[0]["description"] += " Changed."
    tools.append(
        {
            "name": "get_everything",
            "description": "New.",
            "inputSchema": {"type": "object"},
        }
    )
    first = tools[0]["name"]
    with _FakeViewer(tools=tools) as viewer:
        refused = _run(["--mcp", str(viewer.port), "get_scene"])
        names_only = _run(
            ["--mcp", str(viewer.port), "get_everything", "--validate-tool-names-only"]
        )
        unknown_there = _run(
            ["--mcp", str(viewer.port), "get_point", "--validate-tool-names-only"]
        )
        # A call refused here gets a note that the viewer differs.
        unknown_here = _run(["--mcp", str(viewer.port), "get_everything"])
    assert refused.exit_code == 1
    assert "serves a different tool catalog" in refused.output
    assert "Only the viewer has get_everything." in refused.output
    assert "Only this sfmtool has get_point." in refused.output
    assert f"The two differ in {first} (description)." in refused.output
    assert "--validate-tool-names-only" in refused.output

    assert names_only.exit_code == 0, names_only.output
    assert unknown_there.exit_code == 1
    assert "The viewer on port" in unknown_there.output
    assert "has no tool named 'get_point'" in unknown_there.output
    assert viewer.calls == [("get_everything", {})]

    assert unknown_here.exit_code == 1
    assert "There is no tool named 'get_everything'" in unknown_here.output
    assert "serves a different tool catalog" in unknown_here.output
    assert "--validate-tool-names-only" in unknown_here.output


def test_names_only_lets_the_viewer_check_the_arguments():
    def answer(name, arguments):
        return {"error": {"code": -32602, "message": "list_camera_images needs ..."}}

    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "list_camera_images",
                '{"offset": "soon"}',
                "--validate-tool-names-only",
            ]
        )
    assert result.exit_code == 3
    assert "The viewer refused the call: list_camera_images needs ..." in result.output
    assert viewer.calls == [("list_camera_images", {"offset": "soon"})]


def test_wait_until_finished_repeats_the_call_until_nothing_runs():
    replies = iter(
        [
            {"slow": {"running": {"operation": "Bundle adjust"}, "waiting": []}},
            {"slow": {"running": None, "waiting": [{"operation": "Fit"}]}},
            {"slow": {"running": None, "waiting": [], "last_finished": {"id": 3}}},
        ]
    )

    def answer(name, arguments):
        return {"content": [{"type": "text", "text": json.dumps(next(replies))}]}

    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "get_background_tasks",
                "--wait-until-finished",
                "--field",
                "slow.last_finished",
            ]
        )
    assert result.exit_code == 0, result.output
    assert len(viewer.calls) == 3
    assert json.loads(result.output) == {"id": 3}


def test_batch_calls_once_per_line_and_reports_each(tmp_path):
    def answer(name, arguments):
        if arguments["point"] == 2:
            return {
                "content": [{"type": "text", "text": "no live point 2"}],
                "isError": True,
            }
        return {
            "content": [
                {"type": "text", "text": json.dumps({"index": arguments["point"]})}
            ]
        }

    lines = '{"point": 1}\n\n{"point": 2}\n{"point": "a b c"}\n'
    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "get_point",
                "--batch",
                "-",
                "--field",
                "index",
            ],
            input=lines,
        )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines()]
    assert records[0] == {"arguments": {"point": 1}, "reply": 1}
    assert records[1] == {"arguments": {"point": 2}, "error": "no live point 2"}
    # Refused here, so never sent.
    assert records[2]["error"] == validate_tool_call("get_point", {"point": "a b c"})
    assert [c[1] for c in viewer.calls] == [{"point": 1}, {"point": 2}]


def test_batch_reports_a_missing_field_on_its_line():
    def answer(name, arguments):
        reply = {"index": 1} if arguments["point"] == 1 else {"other": 0}
        return {"content": [{"type": "text", "text": json.dumps(reply)}]}

    with _FakeViewer(answer=answer) as viewer:
        result = _run(
            [
                "--mcp",
                str(viewer.port),
                "get_point",
                "--batch",
                "-",
                "--field",
                "index",
            ],
            input='{"point": 1}\n{"point": 2}\n',
        )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines()]
    assert records[0] == {"arguments": {"point": 1}, "reply": 1}
    assert records[1]["error"] == "The reply has no field 'index': no 'index'."
