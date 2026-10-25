//! Render a saved OpenAI coding request with the serving template and the model's own tokenizer.
use anyhow::{Result, anyhow, ensure};
use deepseek_recipe::{
    openai::ChatCompletionRequest,
    request::{ConversionOptions, ProtocolRequest},
};
use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: validation-encode MODEL.gguf REQUEST.json OUTPUT.ids"
    );
    let gg = dsx::gguf::Gguf::open(std::path::Path::new(&args[0]))?;
    let text = gg
        .kv
        .get("dsx.tokenizer_json")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("model has no embedded tokenizer"))?;
    let tok = tokenizers::Tokenizer::from_bytes(text.as_bytes())
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let req: ChatCompletionRequest = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let conv = req
        .convert(ConversionOptions::default())
        .map_err(|e| anyhow!("request: {e:?}"))?;
    let mut rendered = DeepseekV41Encoding::new().render_conversation(&conv.conversation);
    rendered.prompt = dsx_core::server::reasoning::apply_score(rendered.prompt, conv.conversation.thinking_mode, std::env::var("DSX_REASONING_SCORE").ok().as_deref()).map_err(|e| anyhow!("reasoning score: {e:?}"))?;
    ensure!(
        rendered.image_sources.is_empty(),
        "text-only validation fixture required"
    );
    let ids = tok
        .encode(rendered.prompt.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    std::fs::write(
        &args[2],
        ids.get_ids()
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )?;
    println!(
        "{}",
        serde_json::json!({"tokens":ids.len(),"prompt_bytes":rendered.prompt.len(),"tokenizer":"GGUF dsx.tokenizer_json","template":"DeepseekV41Encoding"})
    );
    Ok(())
}
