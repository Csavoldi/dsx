//! Nanbeige protocol adapter around the unchanged DSX engine worker.
use anyhow::{Result, anyhow, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use dsx::{
    nanbeige::{Engine, Nanbeige},
    server::{EngineHandle, Event as EngineEvent, Finish, GenParams},
};
use serde_json::{Value, json};
use std::{convert::Infallible, path::PathBuf, sync::Arc, time::Duration};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::UnboundedReceiverStream;
#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "/home/lxe/nanbeige-runner/models/Nanbeige4.3-33B-NVFP4"
    )]
    model: PathBuf,
    #[arg(long, default_value_t = 8192)]
    ctx: usize,
    #[arg(long, default_value_t = 16)]
    kv_bits: usize,
    #[arg(long)]
    host_embed: bool,
    #[arg(long)]
    head_q8: bool,
    #[arg(long, default_value_t = 0)]
    head_bits: usize,
    #[arg(long, default_value_t = 0)]
    lookup: usize,
    #[arg(long, default_value_t = 8095)]
    port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long)]
    no_graphs: bool,
    #[arg(long)]
    render: Option<PathBuf>,
}
struct App {
    engine: EngineHandle,
    tok: Tokenizer,
    ctx: usize,
}
fn python_json(v: &Value) -> String {
    match v {
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        _ => v.to_string(),
    }
}
fn render(req: &Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter(
        "tojson",
        |v: minijinja::Value| -> std::result::Result<minijinja::Value, minijinja::Error> {
            let v: Value = serde_json::to_value(v).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })?;
            Ok(minijinja::Value::from_safe_string(python_json(&v)))
        },
    );
    env.add_template("chat", include_str!("../../templates/nanbeige.jinja"))?;
    let mut c = req.clone();
    if let Some(messages) = c.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    let f = if call.get("function").is_some() {
                        &mut call["function"]
                    } else {
                        call
                    };
                    if let Some(raw) = f.get("arguments").and_then(Value::as_str) {
                        let args: Value = serde_json::from_str(raw)?;
                        ensure!(args.is_object(), "tool arguments must be an object");
                        f["arguments"] = args;
                    }
                }
            }
        }
    }
    let map = c
        .as_object_mut()
        .ok_or_else(|| anyhow!("request must be an object"))?;
    map.insert("add_generation_prompt".into(), json!(true));
    let kw = req
        .get("chat_template_kwargs")
        .cloned()
        .unwrap_or(json!({}));
    map.insert(
        "enable_thinking".into(),
        kw.get("enable_thinking").cloned().unwrap_or(json!(false)),
    );
    map.insert(
        "preserve_thinking".into(),
        kw.get("preserve_thinking").cloned().unwrap_or(json!(true)),
    );
    map.insert("tool_call_format".into(), json!("xml"));
    Ok(env.get_template("chat")?.render(c)?)
}
fn calls(text: &str, tools: &Value) -> Result<Vec<Value>> {
    let fre =
        regex::Regex::new(r"(?s)<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>")?;
    let pre = regex::Regex::new(r"(?s)<parameter=([^>]+)>\n?(.*?)\n?</parameter>")?;
    let mut out = vec![];
    for cap in fre.captures_iter(text) {
        let name = cap[1].trim();
        let mut args = serde_json::Map::new();
        for p in pre.captures_iter(&cap[2]) {
            let key = p[1].trim();
            let raw = p[2].to_string();
            let ty = tools
                .as_array()
                .and_then(|ts| ts.iter().find(|t| t["function"]["name"] == name))
                .and_then(|t| t["function"]["parameters"]["properties"][key]["type"].as_str());
            let value = if ty == Some("string") {
                json!(raw)
            } else {
                serde_json::from_str(&raw).unwrap_or(json!(raw))
            };
            args.insert(key.into(), value);
        }
        out.push(json!({"id":format!("call_nb_{}",out.len()),"type":"function","function":{"name":name,"arguments":serde_json::to_string(&args)?}}));
    }
    if text.contains("<tool_call>") {
        ensure!(!out.is_empty(), "incomplete or malformed XML tool call");
    }
    Ok(out)
}
struct ParserState {
    thinking: bool,
    pending: String,
    content: String,
    reasoning: String,
}
impl ParserState {
    fn push(&mut self, piece: &str, finalize: bool) -> Vec<(&'static str, String)> {
        self.pending.push_str(piece);
        let mut out = vec![];
        loop {
            let marker = if self.thinking {
                "</think>"
            } else {
                "<tool_call>"
            };
            let found = self.pending.find(marker);
            let mut n = found.unwrap_or_else(|| {
                if finalize {
                    self.pending.len()
                } else {
                    self.pending.len().saturating_sub(16)
                }
            });
            while !self.pending.is_char_boundary(n) {
                n -= 1;
            }
            if n > 0 {
                let text = self.pending[..n].to_string();
                self.pending.drain(..n);
                if self.thinking {
                    self.reasoning.push_str(&text);
                    out.push(("reasoning_content", text));
                } else {
                    self.content.push_str(&text);
                    out.push(("content", text));
                }
            }
            if found.is_some() && self.thinking {
                self.pending.drain(..marker.len());
                self.thinking = false;
                continue;
            }
            break;
        }
        out
    }
}
fn chunk(id: &str, delta: Value, finish: Value) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","created":0,"model":"nanbeige4.3-33b","choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}
async fn chat(State(app): State<Arc<App>>, Json(req): Json<Value>) -> Response {
    match start(app, req).await {
        Ok(r) => r,
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"message":e.to_string(),"type":"invalid_request_error"}})),
        )
            .into_response(),
    }
}
async fn start(app: Arc<App>, req: Value) -> Result<Response> {
    ensure!(req["messages"].is_array(), "messages must be an array");
    ensure!(
        req.get("n").and_then(Value::as_u64).unwrap_or(1) == 1,
        "only n=1 is supported"
    );
    let prompt = render(&req)?;
    let ids = app
        .tok
        .encode(prompt, false)
        .map_err(|e| anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    ensure!(ids.len() < app.ctx, "prompt exceeds context");
    let max = req
        .get("max_tokens")
        .or(req.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let max = max.min(app.ctx - ids.len());
    let p = GenParams {
        max_new: max,
        temperature: req["temperature"].as_f64().unwrap_or(1.) as f32,
        top_p: req["top_p"].as_f64().unwrap_or(0.95) as f32,
        seed: req["seed"].as_u64(),
        stop_token_ids: vec![166101],
        hesitation: None,
        thinking_close: None,
        images: Arc::new(vec![]),
    };
    let id = dsx::server::engine::request_id("chatcmpl-nb");
    let prompt_tokens = ids.len();
    let mut events = app.engine.submit(id.clone(), ids, p)?;
    let thinking = req["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .unwrap_or(false);
    let stream = req["stream"].as_bool().unwrap_or(false);
    let tools = req.get("tools").cloned().unwrap_or(json!([]));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<std::result::Result<Event, Infallible>>();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let send = |v: Value| {
            if stream {
                let _ = tx.send(Ok(Event::default().data(v.to_string())));
            }
        };
        send(chunk(&id, json!({"role":"assistant"}), Value::Null));
        let mut parser = ParserState {
            thinking,
            pending: String::new(),
            content: String::new(),
            reasoning: String::new(),
        };
        let mut decoder = app.tok.decode_stream(false);
        let started = std::time::Instant::now();
        let mut first_token = None;
        let mut count: usize = 0;
        let mut finish = "length";
        let mut error = None;
        let mut model_stats = dsx::server::GenStats::default();
        while let Some(e) = events.recv().await {
            if stream && tx.is_closed() {
                break;
            }
            match e {
                EngineEvent::Token(t) => {
                    first_token.get_or_insert_with(std::time::Instant::now);
                    count += 1;
                    match decoder.step(t) {
                        Ok(Some(piece)) => {
                            for (k, v) in parser.push(&piece, false) {
                                send(chunk(&id, json!({k:v}), Value::Null));
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error = Some(e.to_string());
                            break;
                        }
                    }
                }
                EngineEvent::Done { finish: f, stats } => {
                    model_stats=stats;
                    finish = if f == Finish::Stop { "stop" } else { "length" };
                    break;
                }
                EngineEvent::Error(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        // Keep complete XML calls out of content; send them as structured tool_calls.
        let tooltext = parser
            .pending
            .find("<tool_call>")
            .map(|n| parser.pending[n..].to_string())
            .unwrap_or_default();
        if tooltext.is_empty() {
            for (k, v) in parser.push("", true) {
                send(chunk(&id, json!({k:v}), Value::Null));
            }
        }
        let mut tc = match calls(&tooltext, &tools) {
            Ok(v) => v,
            Err(e) => {
                error = Some(e.to_string());
                vec![]
            }
        };
        for (index, call) in tc.iter_mut().enumerate() {
            call["id"] = json!(format!("call_{id}_{index}"));
        }
        if !tc.is_empty() {
            finish = "tool_calls";
            let deltas: Vec<Value> = tc
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let mut v = v.clone();
                    v["index"] = json!(i);
                    v
                })
                .collect();
            send(chunk(&id, json!({"tool_calls":deltas}), Value::Null));
        }
        if let Some(e) = error {
            send(json!({"error":{"message":e}}));
            let _ = result_tx.send(Err(e));
        } else {
            let prefill = first_token
                .map(|t| t.duration_since(started).as_secs_f64())
                .unwrap_or(started.elapsed().as_secs_f64());
            let decode = first_token.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.);
            let uncached=prompt_tokens.saturating_sub(model_stats.cached_prompt_tokens);
            let model_prefill=model_stats.prefill.map(|d|d.as_secs_f64()).unwrap_or(prefill);
            let timings = json!({"prompt_n":uncached,"cached_n":model_stats.cached_prompt_tokens,"ttft_ms":prefill*1000.,"prompt_ms":model_prefill*1000.,"prompt_per_second":uncached as f64/model_prefill.max(1e-9),"predicted_n":count,"predicted_ms":decode*1000.,"predicted_per_second":(count.saturating_sub(1)) as f64/decode.max(1e-9)});
            let mut final_chunk = chunk(&id, json!({}), json!(finish));
            final_chunk["timings"] = timings.clone();
            send(final_chunk);
            let usage = json!({"prompt_tokens":prompt_tokens,"prompt_tokens_details":{"cached_tokens":model_stats.cached_prompt_tokens},"completion_tokens":count,"total_tokens":prompt_tokens+count});
            if stream {
                send(
                    json!({"id":id,"object":"chat.completion.chunk","model":"nanbeige4.3-33b","choices":[],"usage":usage}),
                );
            }
            let mut message = json!({"role":"assistant","content":parser.content});
            if thinking {
                message["reasoning_content"] = json!(parser.reasoning);
            }
            if !tc.is_empty() {
                message["tool_calls"] = json!(tc);
            }
            let _=result_tx.send(Ok(json!({"id":id,"object":"chat.completion","model":"nanbeige4.3-33b","choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":usage,"timings":timings})));
        }
        if stream {
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
    });
    if stream {
        Ok(Sse::new(UnboundedReceiverStream::new(rx))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
            .into_response())
    } else {
        match result_rx.await? {
            Ok(v) => Ok(Json(v).into_response()),
            Err(e) => Err(anyhow!(e)),
        }
    }
}
async fn props(State(s): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"default_generation_settings":{"id":0,"id_task":-1,"n_ctx":s.ctx,"is_processing":s.engine.pending()>0,"params":{"n_predict":-1,"temperature":1.,"top_k":0,"top_p":0.95,"min_p":0.,"max_tokens":4096,"stream":true,"stop":[],"samplers":["top_p","temperature"],"chat_format":"Nanbeige","reasoning_format":"deepseek","reasoning_in_content":false}},"total_slots":1,"model_path":"nanbeige4.3-33b","model_alias":"nanbeige4.3-33b","role":"model","modalities":{"vision":false,"audio":false,"video":false},"chat_template":include_str!("../../templates/nanbeige.jinja"),"bos_token":"<|im_start|>","eos_token":"<|im_end|>","build_info":"DSX Nanbeige SM120","ui_settings":{}}),
    )
}
async fn slots(State(s): State<Arc<App>>) -> Json<Value> {
    Json(json!([{"id":0,"n_ctx":s.ctx,"is_processing":s.engine.pending()>0}]))
}
#[tokio::main]
async fn main() -> Result<()> {
    let a = Args::parse();
    if let Some(p) = &a.render {
        println!("{}", render(&serde_json::from_slice(&std::fs::read(p)?)?)?);
        return Ok(());
    }
    tracing_subscriber::fmt().with_env_filter("info").init();
    let tok = Tokenizer::from_file(a.model.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let ctx = a.ctx;
    let model = a.model.clone();
    let ng = a.no_graphs;
    let kv_bits = a.kv_bits;
    let lookup = a.lookup;
    let head_bits = if a.head_q8 { 8 } else { a.head_bits };
    let host_embed = a.host_embed;
    let engine = EngineHandle::spawn(move || {
        let mut m = Nanbeige::load_with_cache(&model, ctx, kv_bits, host_embed, head_bits)?;
        m.graphs_enabled = !ng;
        let mut engine = Engine::new(m);
        engine.lookup = lookup;
        Ok(Box::new(engine))
    })?;
    let app = Arc::new(App { engine, tok, ctx });
    let router=Router::new().route("/health",get(||async{Json(json!({"status":"ok","engine":"DSX Nanbeige"}))})).route("/v1/models",get(||async{Json(json!({"object":"list","data":[{"id":"nanbeige4.3-33b","object":"model","owned_by":"local"}]}))})).route("/v1/chat/completions",post(chat)).route("/props",get(props)).route("/slots",get(slots)).route("/tools",get(dsx::server::webui::no_tools)).route("/v1/chat/completions/streams",get(dsx::server::webui::no_streams)).fallback(dsx::server::webui::asset).with_state(app);
    let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port)).await?;
    eprintln!("DSX Nanbeige listening on {}:{}", a.host, a.port);
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_boundaries() {
        let mut p = ParserState {
            thinking: true,
            pending: String::new(),
            reasoning: String::new(),
            content: String::new(),
        };
        for part in ["Reason", "ing.\n</thi", "nk>\nHello ", "世界."] {
            p.push(part, false);
        }
        p.push("", true);
        assert_eq!(p.reasoning, "Reasoning.\n");
        assert_eq!(p.content, "\nHello 世界.");
    }
    #[test]
    fn xml_tool_values() {
        let t = json!([{"function":{"name":"write","parameters":{"properties":{"content":{"type":"string"},"n":{"type":"integer"}}}}}]);
        let c=calls("<tool_call>\n<function=write>\n<parameter=content>\n123\n</parameter>\n<parameter=n>\n7\n</parameter>\n</function>\n</tool_call>",&t).unwrap();
        let args: Value =
            serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"content":"123","n":7}));
    }
    #[test]
    fn template_history() {
        let r = json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello","reasoning_content":"thought"},{"role":"user","content":"again"}],"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true}});
        let s = render(&r).unwrap();
        assert!(s.contains("<think>\nthought\n</think>"));
        assert!(s.ends_with("<think>\n\n</think>\n\n"));
    }
}
