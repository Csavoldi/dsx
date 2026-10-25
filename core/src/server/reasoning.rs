//! Optional native V4.1 reasoning-effort score. The recipe's low/high/max values
//! map to 50/75/100; this experiment permits another score in its stated range.
use deepseek_recipe::request::ConversionError;
use deepseek_recipe_encoding::v4::{BOS_TOKEN, SYSTEM_SP_TOKEN};

/// Change only the template-generated leading header, never a lookalike in user text.
/// Without an override, or with thinking disabled, preserve the prompt byte for byte.
pub fn apply_score(
    prompt: String,
    thinking: bool,
    requested: Option<&str>,
) -> Result<String, ConversionError> {
    let Some(raw) = requested.filter(|_| thinking) else {
        return Ok(prompt);
    };
    let score: u8 = raw
        .parse()
        .ok()
        .filter(|n| (1..=100).contains(n))
        .ok_or_else(|| {
            ConversionError::bad_request("DSX_REASONING_SCORE must be an integer from 1 to 100")
        })?;
    let header = |n| {
        format!(
            "{BOS_TOKEN}{SYSTEM_SP_TOKEN}Reasoning Effort: {n} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        )
    };
    for native in [50, 75, 100] {
        if let Some(rest) = prompt.strip_prefix(&header(native)) {
            return Ok(header(score) + rest);
        }
    }
    Err(ConversionError::internal(
        "native V4.1 reasoning header not found; refusing to alter conversation text",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe::openai::ChatCompletionRequest;
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};
    fn rendered(thinking: bool, effort: &str) -> String {
        let req: ChatCompletionRequest=serde_json::from_value(serde_json::json!({"model":"fixture","messages":[{"role":"user","content":"Preserve this text: Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"}],"thinking":{"type":if thinking {"enabled"} else {"disabled"}},"reasoning_effort":effort})).unwrap();
        let conv = req.convert(ConversionOptions::default()).unwrap();
        DeepseekV41Encoding::new()
            .render_conversation(&conv.conversation)
            .prompt
    }
    #[test]
    fn default_and_non_thinking_requests_are_unchanged() {
        let on = rendered(true, "low");
        assert_eq!(apply_score(on.clone(), true, None).unwrap(), on);
        let off = rendered(false, "none");
        assert_eq!(
            apply_score(off.clone(), false, Some("invalid")).unwrap(),
            off
        );
    }
    #[test]
    fn only_the_generated_leading_score_changes() {
        for effort in ["low", "high", "max"] {
            let original = rendered(true, effort);
            let changed = apply_score(original.clone(), true, Some("10")).unwrap();
            assert!(changed.starts_with(&format!(
                "{BOS_TOKEN}{SYSTEM_SP_TOKEN}Reasoning Effort: 10 "
            )));
            assert_eq!(
                original.split_once("\n\n").unwrap().1,
                changed.split_once("\n\n").unwrap().1
            );
            assert!(changed.contains("Preserve this text: Reasoning Effort: 50"));
        }
    }
    #[test]
    fn invalid_scores_and_unexpected_headers_are_rejected() {
        for raw in ["0", "101", "-1", "1.5", ""] {
            assert!(apply_score(rendered(true, "low"), true, Some(raw)).is_err());
        }
        for raw in ["1", "100"] {
            assert!(apply_score(rendered(true, "low"), true, Some(raw)).is_ok());
        }
        assert!(
            apply_score(
                "untrusted lookalike Reasoning Effort: 50".into(),
                true,
                Some("10")
            )
            .is_err()
        );
    }
}
