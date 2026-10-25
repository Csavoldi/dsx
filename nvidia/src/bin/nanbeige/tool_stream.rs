//! Translate XML tool calls into append-only OpenAI argument deltas.
use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone, Copy, Default)]
enum Phase {
    #[default]
    Call,
    Function,
    Parameter,
    String { leading: bool },
    Json { start: usize },
    EndCall,
}

#[derive(Default)]
pub struct ToolStream {
    cursor: usize,
    index: usize,
    phase: Phase,
    name: String,
    first: bool,
}

fn header<'a>(text: &'a str, prefix: &str) -> Result<Option<(&'a str, usize)>> {
    if prefix.starts_with(text) { return Ok(None); }
    ensure!(text.starts_with(prefix), "malformed XML tool header");
    Ok(text.find('>').map(|end| (&text[prefix.len()..end], end + 1)))
}

impl ToolStream {
    fn argument(&self, text: String, out: &mut Vec<Value>) {
        if !text.is_empty() {
            out.push(json!({"index":self.index,"function":{"arguments":text}}));
        }
    }

    // `text` is the retained XML suffix, beginning at the first <tool_call>.
    // The cursor advances only across complete UTF-8 characters and parsed tags.
    pub fn push(&mut self, text: &str, tools: &Value, id: &str) -> Result<Vec<Value>> {
        const CLOSE: &str = "</parameter>";
        let mut out = vec![];
        loop {
            if matches!(self.phase, Phase::Function | Phase::Parameter | Phase::EndCall) {
                self.cursor += text[self.cursor..].len() - text[self.cursor..].trim_start().len();
            }
            let rest = &text[self.cursor..];
            match self.phase {
                Phase::Call => {
                    let Some(start) = rest.find("<tool_call>") else { break; };
                    self.cursor += start + "<tool_call>".len();
                    self.phase = Phase::Function;
                }
                Phase::Function => {
                    let Some((name, end)) = header(rest, "<function=")? else { break; };
                    self.name = name.trim().to_owned();
                    ensure!(!self.name.is_empty(), "empty XML function name");
                    out.push(json!({"index":self.index,"id":format!("call_{id}_{}",self.index),"type":"function","function":{"name":self.name,"arguments":"{"}}));
                    self.cursor += end;
                    self.first = true;
                    self.phase = Phase::Parameter;
                }
                Phase::Parameter => {
                    if rest.len() < "</function>".len() && "</function>".starts_with(rest) { break; }
                    if rest.starts_with("</function>") {
                        self.argument("}".into(), &mut out);
                        self.cursor += "</function>".len();
                        self.phase = Phase::EndCall;
                        continue;
                    }
                    let Some((key, end)) = header(rest, "<parameter=")? else { break; };
                    let key = key.trim();
                    ensure!(!key.is_empty(), "empty XML parameter name");
                    let string = tools.as_array()
                        .and_then(|ts| ts.iter().find(|t| t["function"]["name"] == self.name))
                        .and_then(|t| t["function"]["parameters"]["properties"][key]["type"].as_str()) == Some("string");
                    self.argument(format!("{}{}:{}", if self.first {""} else {","}, serde_json::to_string(key)?, if string {"\""} else {""}), &mut out);
                    self.first = false;
                    self.cursor += end;
                    self.phase = if string { Phase::String { leading: true } } else { Phase::Json { start: self.cursor } };
                }
                Phase::String { leading } => {
                    if rest.is_empty() { break; }
                    if leading {
                        if rest.starts_with('\n') { self.cursor += 1; }
                        self.phase = Phase::String { leading: false };
                        continue;
                    }
                    let close = rest.find(CLOSE);
                    let end = if let Some(n) = close {
                        self.cursor + n - usize::from(rest[..n].ends_with('\n'))
                    } else {
                        // Retain a possible split closing tag and one delimiter newline.
                        let mut n = text.len().saturating_sub(CLOSE.len() + 1).max(self.cursor);
                        while !text.is_char_boundary(n) { n -= 1; }
                        n
                    };
                    if end > self.cursor {
                        let escaped = serde_json::to_string(&text[self.cursor..end])?;
                        self.argument(escaped[1..escaped.len()-1].to_owned(), &mut out);
                    }
                    if let Some(n) = close {
                        self.cursor += n + CLOSE.len();
                        self.argument("\"".into(), &mut out);
                        self.phase = Phase::Parameter;
                    } else {
                        self.cursor = end;
                        break;
                    }
                }
                Phase::Json { start } => {
                    let Some(n) = rest.find(CLOSE) else { break; };
                    let raw = &text[start..self.cursor+n];
                    let raw = raw.strip_prefix('\n').unwrap_or(raw);
                    let raw = raw.strip_suffix('\n').unwrap_or(raw);
                    let value: Value = serde_json::from_str(raw).unwrap_or_else(|_|json!(raw));
                    self.argument(serde_json::to_string(&value)?, &mut out);
                    self.cursor += n + CLOSE.len();
                    self.phase = Phase::Parameter;
                }
                Phase::EndCall => {
                    if rest.len() < "</tool_call>".len() && "</tool_call>".starts_with(rest) { break; }
                    ensure!(rest.starts_with("</tool_call>"), "missing XML tool-call close");
                    self.cursor += "</tool_call>".len();
                    self.index += 1;
                    self.phase = Phase::Call;
                }
            }
        }
        Ok(out)
    }

    pub fn finish(&self, expected: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Call) && self.index == expected, "incomplete XML tool call");
        Ok(())
    }
}
