// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── The protocol, over a real socket ────────────────────────────────────

/// A running server with an ordinary thread standing in for the GUI.
///
/// This is what [`super::super::serve`] taking a wake *closure* rather than an
/// `EventLoopProxy` buys: the transport can be driven end to end with no event
/// loop, no window and no GPU, so the handshake, the tool list, a real tool
/// call and the `Origin` rejection are all covered by a normal `cargo test`.
struct RunningServer {
    address: std::net::SocketAddr,
    /// Kept for the life of the test. The stand-in GUI loop ends when the
    /// server's sender is dropped, which happens when the task does.
    _gui: std::thread::JoinHandle<()>,
}

fn running_server() -> RunningServer {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<super::super::Request>();
    let address = super::super::serve(0, tx, Default::default(), || {})
        .expect("an ephemeral port is bindable");

    // The stand-in for `App::drain_mcp`: one owner of the state, applying one
    // command at a time. Exactly the discipline the real frame keeps, which is
    // the point — the transport must not need anything more than this.
    let gui = std::thread::spawn(move || {
        let (mut state, mut viewer) = two_reconstructions();
        while let Some(super::super::Request { command, reply }) = rx.blocking_recv() {
            let answer = match apply(&mut state, &mut viewer, command) {
                Outcome::Done(answer) => answer,
                // No frame here to render, so the one deferred tool says so
                // rather than hanging the caller until the timeout.
                Outcome::Deferred(_) => Err(ToolError::new("no frame in this harness")),
            };
            let _ = reply.send(answer);
        }
    });

    RunningServer { address, _gui: gui }
}

/// POST one JSON-RPC body to the endpoint and return `(status, body)`.
///
/// Hand-written HTTP/1.1 rather than an HTTP client dev-dependency: a POST with
/// a JSON body and a connection-close response is a dozen lines, and the point
/// of the test is the bytes on the wire.
fn post(server: &RunningServer, body: &str, extra_headers: &[(&str, &str)]) -> (u16, String) {
    use std::io::{Read as _, Write as _};

    let mut stream =
        std::net::TcpStream::connect(server.address).expect("the endpoint is listening");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .expect("a read timeout is settable");
    let mut request = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n\
         Connection: close\r\n",
        server.address.port(),
        body.len()
    );
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    request.push_str(body);
    stream
        .write_all(request.as_bytes())
        .expect("the request is writable");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("the response is readable");
    let response = String::from_utf8_lossy(&response).into_owned();
    let status = response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {response:?}"));
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default();
    (status, body)
}

/// The JSON object of a response body, whether it arrived as
/// `application/json`, chunked, or inside an SSE frame.
fn rpc_body(body: &str) -> Value {
    let json = body
        .lines()
        .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
        .find(|line| line.starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON in {body:?}"));
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{e} in {json:?}"))
}

/// The JSON-RPC `result`, asserting there was no `error` beside it.
fn rpc_result(body: &str) -> Value {
    let parsed = rpc_body(body);
    assert_eq!(parsed["error"], Value::Null, "JSON-RPC error in {parsed}");
    parsed["result"].clone()
}

const PROTOCOL_VERSION: &str = "2025-06-18";

fn initialize_body() -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{PROTOCOL_VERSION}","capabilities":{{}},"clientInfo":{{"name":"sfm-explorer-test","version":"0"}}}}}}"#
    )
}

#[test]
fn the_endpoint_completes_a_handshake_and_advertises_its_tools() {
    let server = running_server();

    let (status, body) = post(&server, &initialize_body(), &[]);
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert!(
        result["capabilities"]["tools"].is_object(),
        "the server declares the tools capability: {result}"
    );
    // The viewer introduces itself, not the SDK. `ServerConfig::new` fills
    // `serverInfo` from `rmcp`'s own build environment, so a handler that does
    // not set it announces `{"name":"rmcp","version":"3.4.0"}` — which is what
    // a client lists the server as and what a human reads in a failure report.
    assert_eq!(
        result["serverInfo"]["name"],
        json!("sfm-explorer"),
        "{result}"
    );
    assert_eq!(
        result["serverInfo"]["version"],
        json!(env!("CARGO_PKG_VERSION")),
        "{result}"
    );
    assert_eq!(
        result["serverInfo"]["title"],
        json!("SfM Explorer"),
        "{result}"
    );
    assert!(
        result["instructions"]
            .as_str()
            .expect("instructions")
            .contains("get_scene"),
        "the instructions point an agent at the first call it should make"
    );

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let listed: Vec<String> = rpc_result(&body)["tools"]
        .as_array()
        .expect("a tool array")
        .iter()
        .map(|tool| tool["name"].as_str().expect("a name").to_string())
        .collect();
    let advertised: Vec<String> = tools::catalog()
        .iter()
        .map(|spec| spec.name.to_string())
        .collect();
    assert_eq!(listed, advertised);
}

#[test]
fn a_tool_call_reaches_the_gui_thread_and_comes_back() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_scene","arguments":{}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert_ne!(result["isError"], json!(true), "{result}");
    let labels: Vec<&str> = result["structuredContent"]["scene"]
        .as_array()
        .expect("a scene array")
        .iter()
        .map(|node| node["label"].as_str().expect("a label"))
        .collect();
    assert_eq!(labels, ["alpha", "beta"]);
}

/// A viewer refusal is a tool-level error the agent can read and act on, not a
/// transport failure it has to guess at.
#[test]
fn a_viewer_refusal_arrives_as_an_is_error_result() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"select_reconstruction","arguments":{"reconstruction_label":"gamma"}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert_eq!(result["isError"], json!(true), "{result}");
    let text = result["content"][0]["text"].as_str().expect("a message");
    assert!(text.contains("gamma") && text.contains("alpha"), "{text}");
}

/// A page the user has open must not be able to drive their viewer. The
/// allowlist is the endpoint's own loopback origins, so any other `Origin` is
/// rejected before the request reaches a tool.
#[test]
fn a_foreign_origin_is_rejected() {
    let server = running_server();

    let (status, _) = post(
        &server,
        &initialize_body(),
        &[("Origin", "http://evil.example")],
    );
    assert_eq!(status, 403);

    // …while the endpoint's own origin is fine, and so is no origin at all,
    // which is what a real MCP client sends.
    let own = format!("http://127.0.0.1:{}", server.address.port());
    let (status, _) = post(&server, &initialize_body(), &[("Origin", &own)]);
    assert_eq!(status, 200);
    let (status, _) = post(&server, &initialize_body(), &[]);
    assert_eq!(status, 200);
}

/// Arguments that do not fit the advertised schema are the *client's* problem,
/// so they come back as a JSON-RPC error rather than as a tool result.
#[test]
fn a_malformed_argument_is_a_protocol_error() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list_camera_images","arguments":{"offset":"soon"}}}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(status, 200, "{body}");
    let parsed = rpc_body(&body);
    let message = parsed["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("expected a JSON-RPC error in {parsed}"));
    assert!(message.contains("offset"), "{message}");
}

/// The newest revision a real client asks for. The server may negotiate *down*
/// from this — `rmcp` 3.2's own `LATEST` is 2025-11-25 — which is the point:
/// the test asks the way a current client asks and then works with whatever
/// comes back, so an SDK bump changes what is exercised without breaking it.
const NEWEST_PROTOCOL_VERSION: &str = "2026-07-28";

/// `tools/list` carries `ttlMs` and `cacheScope` at the revision a real client
/// negotiates.
///
/// SEP-2549 made both mandatory on a list result. `rmcp` models them as
/// `Option` so one type can serve the older revisions too, which means a
/// handler that simply does not set them compiles, passes a 2025-06-18
/// conformance check, and is then **rejected outright** by a current client:
/// the server shows as connected, its tool list fails schema validation, and
/// its tools are absent for the whole session with only a message about
/// `ttlMs` to explain why. That is exactly how this was found — by attaching
/// Claude Code to it — and this test is why it cannot come back.
#[test]
fn the_tool_list_carries_the_cache_hints_a_current_client_requires() {
    let server = running_server();
    let (status, body) = post(
        &server,
        &format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{NEWEST_PROTOCOL_VERSION}","capabilities":{{}},"clientInfo":{{"name":"t","version":"0"}}}}}}"#
        ),
        &[],
    );
    assert_eq!(status, 200, "{body}");
    let negotiated = rpc_result(&body)["protocolVersion"]
        .as_str()
        .expect("a negotiated version")
        .to_string();

    let (status, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", &negotiated)],
    );
    assert_eq!(status, 200, "{body}");
    let result = rpc_result(&body);
    assert!(
        result["ttlMs"].is_number(),
        "ttlMs must be present and numeric at {negotiated}: {result}"
    );
    assert!(
        matches!(result["cacheScope"].as_str(), Some("public" | "private")),
        "cacheScope must be present and public/private at {negotiated}: {result}"
    );
    assert!(!result["tools"].as_array().expect("tools").is_empty());
}

/// The catalog is not advertised as cacheable. It is fixed within one task
/// but changes across a rebuild, and the viewer exists to be rebuilt — a client
/// holding a cached list across a relaunch would call tools the new binary no
/// longer has.
#[test]
fn the_tool_list_is_not_cacheable() {
    let server = running_server();
    post(&server, &initialize_body(), &[]);
    let (_, body) = post(
        &server,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
        &[("MCP-Protocol-Version", PROTOCOL_VERSION)],
    );
    assert_eq!(rpc_result(&body)["ttlMs"], json!(0));
}
