# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the viewer's advertised tool schemas against their own descriptions.

    python probes/schema_lint.py
    python probes/schema_lint.py --tools-json saved-tools-list.json

With no argument it lints the catalog compiled into the sfmtool extension
(`sfmtool.explorer.tool_catalog`), which needs no running viewer. It is the
running viewer's own list when both were built from the same sources;
`viewer.py launch` reports whether they were.

A client that validates arguments against `inputSchema` can only form the
calls the schema allows, so a schema that disagrees with what the server
accepts is a bug even when the server is lenient. The checks:

- a `required` name that is not a property;
- a required property whose description says it is one of several
  alternatives ("in place of", "not both", "exactly one", "Give this or"),
  which the schema cannot express as required;
- a property with no description.

Findings are leads: read the tool's description, then call it with the
smallest argument set the description allows to confirm the server accepts
what the schema forbids.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ALTERNATIVE = re.compile(
    r"in place of|not both|exactly one|give this or|instead of|"
    r"one of two ways|either .{0,40} or ",
    re.IGNORECASE,
)


def lint(tools: list[dict]) -> list[str]:
    findings = []
    for tool in tools:
        name = tool["name"]
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        for r in required:
            if r not in props:
                findings.append(f"{name}: required '{r}' is not a property")
        alternatives = [
            r
            for r in required
            if r in props and ALTERNATIVE.search(props[r].get("description", ""))
        ]
        if alternatives:
            findings.append(
                f"{name}: required {alternatives} are described as alternatives "
                f"(all required: {required})"
            )
        for p, spec in props.items():
            if not spec.get("description") and "properties" not in spec:
                findings.append(f"{name}: property '{p}' has no description")
    return findings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tools-json", help="a saved tools/list reply or its result")
    a = p.parse_args()
    if a.tools_json is None:
        from sfmtool.explorer import tool_catalog

        tools = [entry["advertised"] for entry in tool_catalog()]
    else:
        d = json.loads(Path(a.tools_json).read_text(encoding="utf-8"))
        tools = d.get("result", d)["tools"]
    findings = lint(tools)
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) over {len(tools)} tools")


if __name__ == "__main__":
    main()
