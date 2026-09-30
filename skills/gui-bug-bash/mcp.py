# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Call an SfM Explorer's MCP endpoint without an MCP client.

    python mcp.py --port P get_scene
    python mcp.py --port P get_point '{"point": 12}' --print "d['track']"
    python mcp.py --port P screenshot '{"panel_name": "track_view"}' --name tv1
    python mcp.py --port P wait            # until the background task is done
    python mcp.py --port P tools           # names and first lines
    python mcp.py --port P tools --schema resize_bench_patch

The endpoint is stateless JSON-RPC over HTTP, so this works whether or not the
session registered the server. Text is printed whole, as UTF-8; image content
is written to PNG files in --out-dir and their paths printed. --print takes a
Python expression over the parsed reply `d` (with `json` in scope) and prints
its value instead of the reply. Import `Client` to make many calls from one
process: a loop of a thousand reads is seconds that way, and minutes as a
thousand processes.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class McpError(Exception):
    """A JSON-RPC error, or a tool reply with isError set."""

    def __init__(self, message: str, tool_error: bool):
        super().__init__(message)
        self.tool_error = tool_error


class Client:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout_s: float = 600):
        self.url = f"http://{host}:{port}/mcp"
        self.timeout_s = timeout_s
        self._id = 0

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        raw = urllib.request.urlopen(req, timeout=self.timeout_s).read().decode()
        if raw.startswith(("event:", "data:")):
            raw = "\n".join(
                line[5:] for line in raw.splitlines() if line.startswith("data:")
            )
        reply = json.loads(raw)
        if "error" in reply:
            raise McpError(json.dumps(reply["error"]), tool_error=False)
        return reply["result"]

    def call_raw(self, tool: str, args: dict | None = None) -> dict:
        """The tool's result: {"content": [...], "isError": bool}."""
        return self.rpc("tools/call", {"name": tool, "arguments": args or {}})

    def call(self, tool: str, args: dict | None = None):
        """The tool's first text content, parsed as JSON where it is JSON.

        Raises McpError on a JSON-RPC error or a refusal.
        """
        result = self.call_raw(tool, args)
        texts = [c["text"] for c in result.get("content", []) if c["type"] == "text"]
        if result.get("isError"):
            raise McpError(texts[0] if texts else "(no text)", tool_error=True)
        if not texts:
            return None
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]

    def tools(self) -> list[dict]:
        return self.rpc("tools/list")["tools"]

    def wait(self, poll_s: float = 1.0, timeout_s: float = 3600) -> dict:
        """Poll get_background_task until nothing is running; return its reply."""
        deadline = time.time() + timeout_s
        while True:
            task = self.call("get_background_task")
            if not task or not task.get("running"):
                return task
            if time.time() > deadline:
                raise TimeoutError(f"still running after {timeout_s}s: {task}")
            time.sleep(poll_s)


def save_images(result: dict, out_dir: Path, name: str) -> list[Path]:
    paths = []
    images = [c for c in result.get("content", []) if c["type"] == "image"]
    for i, c in enumerate(images):
        path = out_dir / (f"{name}.png" if i == 0 else f"{name}_{i}.png")
        path.write_bytes(base64.b64decode(c["data"]))
        paths.append(path)
    return paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SFM_EXPLORER_MCP_PORT", "8787")),
        help="endpoint port (default: $SFM_EXPLORER_MCP_PORT or 8787)",
    )
    p.add_argument("--out-dir", default=".", help="where screenshots are written")
    p.add_argument("--name", help="file stem for images (default: the tool name)")
    p.add_argument("--print", dest="expr", help="Python expression over the reply d")
    p.add_argument("--schema", help="with `tools`: print this tool's whole entry")
    p.add_argument("tool", help="a tool name, or `tools` or `wait`")
    p.add_argument("args", nargs="?", default="{}", help="JSON object of arguments")
    a = p.parse_args()
    client = Client(a.port)

    try:
        if a.tool == "tools":
            tools = client.tools()
            if a.schema:
                entry = next((t for t in tools if t["name"] == a.schema), None)
                if entry is None:
                    raise SystemExit(f"no tool named {a.schema}")
                print(json.dumps(entry, indent=2, ensure_ascii=False))
            else:
                for t in tools:
                    first = t["description"].split(". ")[0].replace("\n", " ")
                    print(f"{t['name']}: {first[:140]}")
                print(f"({len(tools)} tools)")
            return
        if a.tool == "wait":
            d = client.wait()
        else:
            result = client.call_raw(a.tool, json.loads(a.args))
            for path in save_images(result, Path(a.out_dir), a.name or a.tool):
                print(f"IMAGE -> {path.resolve()}")
            texts = [c["text"] for c in result["content"] if c["type"] == "text"]
            if result.get("isError"):
                print("TOOL ERROR: " + "\n".join(texts))
                sys.exit(2)
            if a.expr is None:
                print("\n".join(texts))
                return
            d = json.loads(texts[0])
    except McpError as e:
        print(f"RPC ERROR: {e}")
        sys.exit(2)
    except urllib.error.URLError as e:
        print(f"no endpoint on port {a.port}: {e.reason}")
        sys.exit(3)

    if a.expr is not None:
        value = eval(a.expr, {"json": json}, {"d": d})
        print(
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    else:
        print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    main()
