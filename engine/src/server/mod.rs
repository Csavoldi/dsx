//! OpenAI / Anthropic compatible HTTP API over one engine thread, built on DeepSeek's `deepseek-recipe` crates:
//! request -> Conversation -> V4.1 prompt -> token ids -> engine -> token ids -> StreamProcessor -> protocol events.
//! Text only: image inputs are rejected.

pub mod engine;
pub mod mock;
#[cfg(feature = "cuda")]
pub mod real;
pub mod sample;
pub mod webui;

use std::pin::{Pin, pin};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::{Stream, StreamExt};

pub use engine::{Engine, EngineHandle, Event, Finish, GenParams, GenStats};

/// Tokens that end generation (never forwarded to the parser).
pub const STOP_TOKENS: [&str; 2] = ["<｜end▁of▁sentence｜>", "<｜User｜>"];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// reported by /v1/models and used when a request names no model
    pub model_name: String,
    /// context window: prompt + generated tokens
    pub ctx: usize,
    /// generation cap when the request sets none
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    /// SSE comment interval while nothing else is sent (long prefills)
    pub keepalive: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_name: "deepseek-v4.1-flash".into(),
            ctx: 32768,
            default_max_tokens: 8192,
            default_temperature: 1.0,
            default_top_p: 1.0,
            keepalive: Duration::from_secs(10),
        }
    }
}

pub struct AppState {
    pub engine: EngineHandle,
    pub tok: Arc<Tokenizer>,
    pub cfg: ServerConfig,
    stop_ids: Vec<u32>,
}

impl AppState {
    pub fn new(engine: EngineHandle, tok: Arc<Tokenizer>, cfg: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let stop_ids = STOP_TOKENS
            .iter()
            .map(|t| tok.token_to_id(t).ok_or_else(|| anyhow::anyhow!("tokenizer has no {t} token: not a DeepSeek V4.1 tokenizer?")))
            .collect::<anyhow::Result<_>>()?;
        Ok(Arc::new(Self { engine, tok, cfg, stop_ids }))
    }
}

pub fn router(st: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/props", get(webui::props))
        .route("/slots", get(webui::slots))
        .route("/tools", get(webui::no_tools).post(webui::no_tools))
        .route("/v1/streams/lookup", post(webui::no_streams))
        .fallback(get(webui::asset))
        .with_state(st)
}

pub async fn serve(listener: tokio::net::TcpListener, st: Arc<AppState>) -> std::io::Result<()> {
    axum::serve(listener, router(st)).await
}

// ---- errors

#[derive(Clone, Copy, PartialEq)]
enum Proto {
    OpenAi,
    Anthropic,
}

struct ApiError(Proto, ConversionError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.1.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = match self.0 {
            Proto::OpenAi => serde_json::to_value(self.1.into_response()).unwrap_or_default(),
            Proto::Anthropic => {
                let kind = if status.is_client_error() { "invalid_request_error" } else { "api_error" };
                json!({"type": "error", "error": {"type": kind, "message": self.1.to_string()}})
            }
        };
        (status, axum::Json(body)).into_response()
    }
}

fn parse<T: DeserializeOwned>(body: &[u8]) -> Result<T, ConversionError> {
    serde_json::from_slice(body).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

/// parse, filling in this server's model name when the request has none (llama.cpp's web UI omits it in
/// single-model mode; there is only one model here anyway)
fn parse_with_model<T: DeserializeOwned>(st: &AppState, body: &[u8]) -> Result<T, ConversionError> {
    let mut v: serde_json::Value = parse(body)?;
    if let Some(o) = v.as_object_mut() {
        if !o.get("model").is_some_and(|m| m.as_str().is_some_and(|s| !s.is_empty())) {
            o.insert("model".into(), json!(st.cfg.model_name));
        }
    }
    serde_json::from_value(v).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

// ---- routes

async fn chat_completions(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let req: ChatCompletionRequest = parse_with_model(&st, &body).map_err(e)?;
    let (seed, include_usage) = (req.seed, req.include_usage());
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    let streaming = conv.stream;
    run::<ChatCompletionRequest>(st, conv, "chatcmpl-", seed, |g| if streaming { g.with_include_usage(include_usage) } else { g })
        .await
        .map_err(e)
}

async fn messages(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<MessagesRequest>(st, conv, "msg_", None, |g| g).await.map_err(e)
}

async fn responses(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let req: ResponsesRequest = parse_with_model(&st, &body).map_err(e)?;
    let custom = req.custom_tool_names();
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<ResponsesRequest>(st, conv, "resp_", None, move |g| g.with_custom_tool_names(custom)).await.map_err(e)
}

/// Anthropic token counting: the exact prompt length this server would prefill.
async fn count_tokens(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    let ids = encode(&st, &conv).map_err(e)?;
    Ok(axum::Json(json!({"input_tokens": ids.len()})).into_response())
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    let created = now();
    axum::Json(json!({
        "object": "list",
        "data": [{"id": st.cfg.model_name, "object": "model", "created": created, "owned_by": "dsx",
                  "display_name": st.cfg.model_name, "type": "model", "context_length": st.cfg.ctx}],
    }))
    .into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    axum::Json(json!({"status": "ok", "model": st.cfg.model_name, "pending": st.engine.pending()})).into_response()
}

// ---- the shared request path

type Gen<T> = <<T as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator;
type Chunk<T> = <Gen<T> as ChunkGenerator>::Chunk;

/// Render + tokenize the conversation (V4.1 template). Rejects images.
pub fn encode(st: &AppState, conv: &ConversationRequest) -> Result<Vec<u32>, ConversionError> {
    let rendered = DeepseekV41Encoding::new().render_conversation(&conv.conversation);
    if !rendered.image_sources.is_empty() {
        return Err(ConversionError::bad_request(format!(
            "image input is not supported by this server (text only); the request has {} image(s)",
            rendered.image_sources.len()
        )));
    }
    tracing::debug!(prompt = rendered.prompt, "rendered prompt");
    let enc = st.tok.encode(rendered.prompt.as_str(), false).map_err(|e| ConversionError::internal(format!("tokenizer: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

async fn run<T>(
    st: Arc<AppState>,
    conv: ConversationRequest,
    id_prefix: &str,
    seed: Option<u64>,
    configure: impl FnOnce(Gen<T>) -> Gen<T>,
) -> Result<Response, ConversionError>
where
    T: ProtocolRequest,
    Chunk<T>: Serialize + Send,
{
    let prompt = encode(&st, &conv)?;
    let cfg = &st.cfg;
    if prompt.len() >= cfg.ctx {
        return Err(ConversionError::bad_request(format!(
            "prompt is {} tokens; this server's context window is {} tokens",
            prompt.len(),
            cfg.ctx
        )));
    }
    let io = &conv.inference_options;
    let room = cfg.ctx - prompt.len();
    let params = GenParams {
        max_new: io.max_tokens.map(|m| m as usize).unwrap_or(cfg.default_max_tokens).min(room),
        temperature: io.temperature.unwrap_or(cfg.default_temperature),
        top_p: io.top_p.unwrap_or(cfg.default_top_p),
        seed,
        stop_token_ids: st.stop_ids.clone(),
    };
    let id = engine::request_id(id_prefix);
    let model = conv.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| cfg.model_name.clone());
    let c = &conv.conversation;
    tracing::info!(
        %id, %model, stream = conv.stream, thinking = c.thinking_mode, effort = ?c.reasoning_effort,
        messages = c.messages.len(), tools = c.tools.len(), temperature = params.temperature, top_p = params.top_p,
        "request"
    );
    let generator = configure(T::chunk_generator(&conv, id.clone(), model.clone()));
    let processor = StreamProcessor::new(generator, conv.parsing_options).with_tokenizer(st.tok.clone());
    let n_prompt = prompt.len();
    let rx = st.engine.submit(id.clone(), prompt, params).map_err(|e| ConversionError::internal(e.to_string()))?;
    let failed = Arc::new(Mutex::new(None::<String>));
    let timing = Arc::new(Mutex::new(Timing::new()));
    let inference = inference_stream(rx, n_prompt, failed.clone(), timing.clone());

    if !conv.stream {
        let mut resp = T::Response::new(id, model, now(), n_prompt, 0);
        let mut chunks = pin!(processor.process(inference));
        while let Some(chunk) = chunks.next().await {
            resp.append(chunk.map_err(|e| ConversionError::internal(e.to_string()))?);
        }
        if let Some(err) = failed.lock().unwrap().take() {
            return Err(ConversionError::internal(format!("engine error: {err}")));
        }
        return Ok(axum::Json(resp).into_response());
    }

    let events = processor.process(inference).map(|chunk| {
        let chunk = chunk.map_err(|e| std::io::Error::other(e.to_string()))?;
        let data = serde_json::to_string(&chunk).map_err(std::io::Error::other)?;
        Ok::<_, std::io::Error>(match <T::Response as ProtocolResponse>::chunk_event_type(&chunk) {
            Some(ev) => format!("event: {ev}\ndata: {data}\n\n"),
            None => format!("data: {data}\n\n"),
        })
    });
    // llama.cpp extension for chat completions: a last chunk with `timings` (its web UI shows tok/s from it)
    let llama_timings = id_prefix == "chatcmpl-";
    let (tid, tmodel) = (id.clone(), model.clone());
    let timings = tokio_stream::iter([()]).filter_map(move |_| {
        llama_timings.then(|| {
            let t = timing.lock().unwrap().json(n_prompt);
            Ok(format!("data: {}\n\n", json!({"id": tid, "object": "chat.completion.chunk", "model": tmodel, "choices": [], "timings": t})))
        })
    });
    let done = <T::Response as ProtocolResponse>::done_message().map(|m| Ok(format!("data: {m}\n\n")));
    let events = events.chain(timings).chain(tokio_stream::iter(done));
    // SSE comments keep idle connections (long prefill, slow decode) alive; parsers ignore them
    let events = events.timeout(cfg.keepalive).map(|r| r.unwrap_or_else(|_| Ok(": keep-alive\n\n".to_string())));
    let mut res = Response::new(Body::from_stream(events));
    let h = res.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Ok(res)
}

/// Engine events -> recipe inference chunks. Dropping the stream drops the receiver, which cancels generation.
fn inference_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    n_prompt: usize,
    failed: Arc<Mutex<Option<String>>>,
    timing: Arc<Mutex<Timing>>,
) -> Pin<Box<dyn Stream<Item = InferenceChunk> + Send>> {
    let ready = InferenceChunk::Ready {
        system_fingerprint: Some("dsx".into()),
        prompt_usage: PromptUsage { prompt_tokens: n_prompt, prompt_cache_hit_tokens: 0 },
    };
    let rest = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map_while(move |ev| match ev {
        Event::Token(t) => {
            timing.lock().unwrap().token();
            Some(InferenceChunk::Token { token_id: t })
        }
        Event::Done { finish: Finish::Stop } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }),
        Event::Done { finish: Finish::Length } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }),
        Event::Done { finish: Finish::Cancelled } => None,
        Event::Error(e) => {
            *failed.lock().unwrap() = Some(e);
            None
        }
    });
    Box::pin(tokio_stream::iter([ready]).chain(rest))
}

/// Request timing as seen by the HTTP layer: prompt = submit -> first token, predicted = first -> last token.
struct Timing {
    start: std::time::Instant,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    n: usize,
}

impl Timing {
    fn new() -> Self {
        Self { start: std::time::Instant::now(), first: None, last: None, n: 0 }
    }
    fn token(&mut self) {
        let t = std::time::Instant::now();
        self.first.get_or_insert(t);
        self.last = Some(t);
        self.n += 1;
    }
    /// llama.cpp's `timings` object
    fn json(&self, n_prompt: usize) -> serde_json::Value {
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let prompt_ms = self.first.map_or(0.0, |f| ms(self.start, f));
        // the first token comes with the prefill; decode speed is over the tokens after it
        let predicted_ms = match (self.first, self.last) { (Some(f), Some(l)) => ms(f, l), _ => 0.0 };
        let dn = self.n.saturating_sub(1);
        json!({
            "prompt_n": n_prompt, "prompt_ms": prompt_ms,
            "prompt_per_second": if prompt_ms > 0.0 { n_prompt as f64 * 1e3 / prompt_ms } else { 0.0 },
            "predicted_n": self.n, "predicted_ms": predicted_ms,
            "predicted_per_second": if predicted_ms > 0.0 { dn as f64 * 1e3 / predicted_ms } else { 0.0 },
        })
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
