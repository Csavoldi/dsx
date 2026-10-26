//! Incremental equivalent of Nanbeige's Replace -> ByteFallback -> Fuse -> Strip.
//! ByteFallback invalidates an entire consecutive byte-token run if any byte is
//! invalid UTF-8. Hold that run until a non-byte token (or EOF), so a later bad
//! byte cannot rewrite text we have already streamed.
use anyhow::{Result, anyhow, ensure};
use tokenizers::Tokenizer;

pub fn validate(tok: &Tokenizer) -> Result<()> {
    let actual=serde_json::to_value(tok.get_decoder())?;
    let expected=serde_json::json!({"type":"Sequence","decoders":[
        {"type":"Replace","pattern":{"String":"▁"},"content":" "},
        {"type":"ByteFallback"},{"type":"Fuse"},
        {"type":"Strip","content":" ","start":1,"stop":0}]});
    ensure!(actual==expected,"unsupported Nanbeige decoder pipeline: {actual}");
    Ok(())
}
pub struct Decoder<'a> {
    tok: &'a Tokenizer,
    bytes: Vec<u8>,
    at_start: bool,
}
impl<'a> Decoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {Self {tok,bytes:vec![],at_start:true}}
    fn flush_bytes(&mut self) -> String {
        let bytes=std::mem::take(&mut self.bytes);
        match String::from_utf8(bytes) {Ok(s)=>s,Err(e)=>"�".repeat(e.as_bytes().len())}
    }
    fn emit(&mut self, mut text: String) -> Option<String> {
        if self.at_start && !text.is_empty() {
            self.at_start=false;
            if text.starts_with(' ') {text.remove(0);}
        }
        (!text.is_empty()).then_some(text)
    }
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        let token=self.tok.id_to_token(id).ok_or_else(||anyhow!("unknown token ID {id}"))?.replace('▁'," ");
        if token.len()==6 && token.starts_with("<0x") && token.ends_with('>') {
            if let Ok(byte)=u8::from_str_radix(&token[3..5],16) {self.bytes.push(byte);return Ok(None);}
        }
        let mut text=self.flush_bytes();text.push_str(&token);Ok(self.emit(text))
    }
    pub fn finish(&mut self) -> Option<String> {let text=self.flush_bytes();self.emit(text)}
}
#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Tokenizer {
        let decoder=serde_json::json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]});
        let v=serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":decoder,"model":{"type":"WordLevel","vocab":{"<unk>":0,"<0xE2>":1,"<0x96>":2,"<0xAF>":3,"<0xFF>":4,"k":5,"▁":6,"▁hello":7,"⣿":8,"<0x20>":9},"unk_token":"<unk>"}});
        Tokenizer::from_bytes(serde_json::to_vec(&v).unwrap()).unwrap()
    }
    fn stream(tok: &Tokenizer, ids: &[u32]) -> String {
        let mut d=Decoder::new(tok);let mut out=String::new();
        for &id in ids {out.push_str(&d.step(id).unwrap().unwrap_or_default());}
        out.push_str(&d.finish().unwrap_or_default());out
    }
    #[test]
    fn malformed_byte_run_reproduces_native_error_and_matches_full_decode() {
        let t=tokenizer();validate(&t).unwrap();let ids=[1,2,3,4,5];
        let mut native=t.decode_stream(false);for &id in &ids[..4] {native.step(id).unwrap();}
        assert!(native.step(5).unwrap_err().to_string().contains("Invalid prefix"));
        assert_eq!(stream(&t,&ids),"����k");assert_eq!(stream(&t,&ids),t.decode(&ids,false).unwrap());
    }
    #[test]
    fn spaces_unicode_long_runs_and_eof_match_reference() {
        let t=tokenizer();
        for ids in [vec![],vec![6],vec![6,6,7],vec![9,6,7],vec![7,1,2,3,8],vec![1,2],vec![6,1,2,3],vec![1,2,3].repeat(300),[vec![1,2,3].repeat(300),vec![4,5]].concat()] {
            assert_eq!(stream(&t,&ids),t.decode(&ids,false).unwrap(),"{ids:?}");
        }
    }
}
