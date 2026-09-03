// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The HTTP side: an `rmcp` server handler mounted on `axum`, on a
//! current-thread `tokio` runtime on one dedicated thread.
//!
//! ## Why the SDK
//!
//! MCP has run 2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25 → 2026-07-28,
//! adding and then removing sessions, the GET stream, `Last-Event-ID`
//! resumability and server-initiated requests along the way, and clients in the
//! field speak several of those. Keeping up with that is [`rmcp`]'s job, and it
//! is what buys a viewer that still works after somebody upgrades their client.
//! The current revision is simple enough to hand-roll — a single POST endpoint,
//! no sessions, no handshake, and no SSE needed for a server that answers every
//! request with `application/json` — and this configures exactly that shape,
//! leaving the older eras to the SDK behind it.
//!
//! ## Why HTTP on loopback rather than stdio
//!
//! The value of this surface is attaching to the window the human already has
//! open, which is what a listening socket gives: the viewer runs, an agent
//! connects and disconnects as it likes, and the human watches the same window
//! throughout. That is the shape desktop applications with in-process MCP
//! servers use. Stdio has the client launch the server as a child process,
//! which suits a tool with no life of its own.
//!
//! ## What this module may not do
//!
//! Touch application state. It has a channel and a wake callback, and that is
//! the whole of its access to the viewer. The wake is a closure rather than the
//! `EventLoopProxy` it actually wraps, so nothing here depends on winit — which
//! is what lets `mcp::tests` drive a real server over a real socket with an
//! ordinary thread standing in for the GUI.

use std::future::IntoFuture as _;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use rmcp::model::{
    CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{RoleServer, ServerHandler};
use tokio::sync::{mpsc, oneshot};

use super::tools::{self, ToolKind};
use super::{Reply, Request, ToolOutput};

/// How long a tool call waits for the GUI thread before giving up.
///
/// The GUI thread can legitimately stop pumping — a modal `rfd` file dialog is
/// open, or the user is dragging the window on Windows — and an agent must get
/// "the viewer is busy" rather than a hung connection.
const APPLY_TIMEOUT: Duration = Duration::from_secs(10);

/// Why the endpoint could not be brought up.
///
/// Fatal to startup, deliberately: two viewers on one port is the common
/// mistake, and a viewer that silently came up *without* the endpoint the agent
/// was told to use is worse than one that refused to start.
#[derive(Debug)]
pub(crate) enum ServeError {
    /// The port is taken, or otherwise unbindable.
    Bind { port: u16, source: std::io::Error },
    /// The tokio runtime or its thread could not be created.
    Runtime(std::io::Error),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServeError::Bind { port, source } => write!(
                f,
                "could not bind the MCP endpoint on 127.0.0.1:{port}: {source}\n\
                 Another viewer may already have that port. Use --mcp <other port>, or --mcp 0 \
                 for an ephemeral one."
            ),
            ServeError::Runtime(source) => write!(f, "could not start the MCP server: {source}"),
        }
    }
}

impl std::error::Error for ServeError {}

/// Bind the endpoint and start serving.
///
/// Returns once the socket is **bound and listening**, or with the bind error;
/// the runtime lives on its own thread from there. Binding before the thread is
/// spawned is what lets a port collision be reported to the caller at all — a
/// bind inside the spawned task could only log.
pub(crate) fn serve(
    port: u16,
    tx: mpsc::UnboundedSender<Request>,
    wake: impl Fn() + Send + Sync + 'static,
) -> Result<SocketAddr, ServeError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .thread_name("sfm-explorer-mcp")
        .build()
        .map_err(ServeError::Runtime)?;

    // Loopback only, and there is no flag to make it otherwise: this endpoint
    // hands out read access to any .sfmr path the process can read and control
    // of a window on the user's desktop. Both are appropriate for a tool the
    // user explicitly started with a flag, and neither is appropriate for
    // anything reachable from off the machine.
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, port)))
        .map_err(|source| ServeError::Bind { port, source })?;
    let address = listener
        .local_addr()
        .map_err(|source| ServeError::Bind { port, source })?;

    let handler = Viewer {
        tx,
        wake: Arc::new(wake),
    };
    let config = StreamableHttpServerConfig::default()
        // Every request is answered on its own, so there is nothing for a
        // session to hold: sessions are gone from 2026-07-28 anyway, and the
        // stateless path is the one this surface is designed around.
        .with_legacy_session_mode(false)
        .with_json_response(true)
        // Present-and-not-loopback Origin gets a 403, per the transport spec.
        // This is what stops a web page the user has open from driving their
        // viewer through DNS rebinding; a real MCP client sends no Origin at
        // all and is unaffected.
        .with_allowed_origins([
            format!("http://127.0.0.1:{}", address.port()),
            format!("http://localhost:{}", address.port()),
            format!("http://[::1]:{}", address.port()),
        ]);
    let service: StreamableHttpService<Viewer, LocalSessionManager> =
        StreamableHttpService::new(move || Ok(handler.clone()), Default::default(), config);
    let router = axum::Router::new().nest_service("/mcp", service);

    std::thread::Builder::new()
        .name("sfm-explorer-mcp".into())
        .spawn(move || {
            if let Err(e) = runtime.block_on(axum::serve(listener, router).into_future()) {
                log::error!("MCP server stopped: {e}");
            }
        })
        .map_err(ServeError::Runtime)?;

    Ok(address)
}

/// The MCP server handler: a channel to the GUI thread, and the wake that
/// makes it look.
///
/// `Arc` because `rmcp` builds a fresh handler per request from a factory
/// closure, so this is cloned constantly and the wake itself is not `Clone`.
#[derive(Clone)]
struct Viewer {
    tx: mpsc::UnboundedSender<Request>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl Viewer {
    /// Hand one command to the GUI thread and wait for its answer.
    async fn dispatch(&self, command: super::Command) -> Result<Reply, ErrorData> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Request {
                command,
                reply: reply_tx,
            })
            .map_err(|_| ErrorData::internal_error("The viewer has shut down.", None))?;
        // Wake the GUI thread. `App::user_event` requests a redraw, and
        // `run_ui_and_paint` drains the queue before it does anything else, so
        // an idle viewer answers on the frame this wakes.
        (self.wake)();

        match tokio::time::timeout(APPLY_TIMEOUT, reply_rx).await {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(_)) => Err(ErrorData::internal_error(
                "The viewer dropped the request without answering.",
                None,
            )),
            Err(_) => Err(ErrorData::internal_error(
                format!(
                    "The viewer did not answer within {} seconds. It may be showing a modal \
                     dialog, or be mid-drag.",
                    APPLY_TIMEOUT.as_secs()
                ),
                None,
            )),
        }
    }
}

impl ServerHandler for Viewer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Drives a running SfM Explorer window: the scene graph of loaded .sfmr \
                 reconstructions, the selection, the 3D viewport camera, and a screenshot of \
                 what is on screen. Start with get_scene — the reconstruction labels it reports \
                 are the handles every other tool takes. This is a state surface, not a data \
                 one: no tool returns point clouds or track tables in bulk, so read the .sfmr \
                 file for data. A human is watching this window; every change made here is \
                 visible to them and noted in the viewer's status line.",
        )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        // `ttlMs` and `cacheScope` are **required** from protocol revision
        // 2026-07-28 (SEP-2549), and `rmcp` models them as `Option` only so the
        // same type can serve the older revisions — so a handler that leaves
        // them unset emits a list that a 2026-07-28 client rejects outright,
        // with its tools absent for the whole session.
        //
        // `ttl_ms: 0` is "do not cache", and it is the honest answer rather
        // than the timid one. The catalog is a compile-time constant, so it
        // cannot change while a viewer runs — but it changes across a *rebuild*,
        // which is the normal state of affairs for a tool whose whole purpose is
        // being iterated on, and a client holding a cached list across a
        // relaunch would call tools the new binary no longer has. Twenty-three tools
        // are cheap to re-fetch; a stale list is not cheap to debug.
        Ok(
            ListToolsResult::with_all_items(tools::catalog().iter().map(advertise).collect())
                .with_ttl_ms(0)
                // No authorization contexts to share across: one local user, one
                // process, on loopback.
                .with_cache_scope(CacheScope::Private),
        )
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tools::catalog()
            .iter()
            .find(|spec| spec.name == name)
            .map(advertise)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        // Two levels of failure, and the distinction matters to a client. A
        // request the tool table cannot make sense of is a *protocol* error —
        // the arguments do not fit the advertised schema. Everything the viewer
        // itself refuses is a tool-level error with `isError: true`: the
        // request was fine, the answer is no.
        let command = tools::parse(&request.name, request.arguments.as_ref())
            .map_err(|e| ErrorData::invalid_params(e.0, None))?;

        Ok(match self.dispatch(command).await? {
            // `structured` emits the value twice: once as `structuredContent`
            // for a client that reads it as data, and once as a text block for
            // one that reads it as text. Both, because which of the two a
            // client surfaces to its model is the client's decision.
            Ok(ToolOutput::Json(value)) => CallToolResult::structured(value),
            Ok(ToolOutput::Png {
                bytes,
                width,
                height,
                caption,
            }) => CallToolResult::success(vec![
                ContentBlock::text(format!("{width}×{height} px. {caption}")),
                ContentBlock::image(
                    base64::engine::general_purpose::STANDARD.encode(&bytes),
                    "image/png",
                ),
            ]),
            Err(refusal) => CallToolResult::error(vec![ContentBlock::text(refusal.0)]),
        }
        .into())
    }
}

/// One [`tools::ToolSpec`] as the MCP `Tool` a client sees.
///
/// Every tool is annotated `openWorldHint: false` — this surface talks to one
/// process on this machine and nothing else — and the writes
/// `destructiveHint: false`, because nothing here touches a file on disk:
/// `close_reconstruction` unloads a reconstruction, it does not delete one.
fn advertise(spec: &tools::ToolSpec) -> Tool {
    let read_only = spec.kind == ToolKind::Read;
    let mut annotations = ToolAnnotations::new()
        .read_only(read_only)
        .open_world(false);
    if !read_only {
        annotations = annotations.destructive(false);
    }
    Tool::new(
        spec.name,
        spec.description,
        Arc::new(
            spec.schema
                .as_object()
                .cloned()
                .expect("every tool schema is a JSON object"),
        ),
    )
    .annotate(annotations)
}
