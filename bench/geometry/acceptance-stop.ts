/** Finish a benchmark task after a certified artifact and the agent's own display check. */
import { readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const root = process.env.GAUNTLET_RUN!;
  const message = "The functional acceptance checks passed for the current renderer, and you have checked its display. The task is complete. Give a brief final summary now; do not refine the appearance or call further tools.";
  const read = (name: string) => JSON.parse(readFileSync(join(root, name), "utf8"));
  const hash = () => createHash("sha256").update(readFileSync(join(root,"workspace/render.py"))).digest("hex");
  const judgeVersion = Number(process.env.GAUNTLET_JUDGE_VERSION ?? "5");
  let accepted = false;
  let feedbackHash = "";
  function ready() {
    if (accepted) return true;
    try {
      const check = read("acceptance.json");
      const self = read("self-check.json");
      const current = hash();
      if (!check.pass || check.judge_version !== judgeVersion || check.source_sha256 !== current || self.source_sha256 !== current) return false;
      accepted = true;
      pi.setActiveTools([]);
      writeFileSync(join(root,"accepted-stop.json"),JSON.stringify({time:Date.now()/1000,source_sha256:current,judge_version:check.judge_version,self_check_time:self.time},null,2)+"\n");
      return true;
    } catch { return false; }
  }
  function functionalFeedback(): string | undefined {
    try {
      const check=read("latest-check.json"), self=read(judgeVersion>=8 ? "self-check-attempt.json" : "self-check.json"), current=hash();
      if (check.judge_version!==judgeVersion || check.source_sha256!==current || self.source_sha256!==current || feedbackHash===current || check.pass) return;
      const failures:string[]=[];
      for (const phase of check.renderer?.phases ?? []) {
        if (phase.max_dots===0) failures.push("No visible braille dots were rendered.");
        const cert=phase.frame_certification;
        if (judgeVersion>=7 && cert?.supported_full_frame && (judgeVersion<8 || phase.max_dots!==0)) {
          if (phase.typical_width_fraction>.95 || phase.typical_height_fraction>.95) failures.push("The solid reaches the terminal boundary; leave a clear visible margin.");
          if (phase.typical_width_fraction<.2 || phase.typical_height_fraction<.2) failures.push("The solid is too small to occupy a substantial part of the terminal.");
        }
        if (cert?.supported_full_frame && cert.max_ghost_fraction>.015) failures.push("Dots from previous frames remain visible as the object moves.");
        if (cert?.supported_full_frame && cert.shaded_frame_fraction<.2) failures.push("Visible face interiors do not show clearly differentiated shading.");
      }
      if (!failures.length) return;
      feedbackHash=current;
      const feedback="Functional acceptance check: "+[...new Set(failures)].join(" ")+" Repair these functional failures, self-check once, then finish. Do not polish unrelated appearance.";
      return feedback;
    } catch { return; }
  }
  pi.on("tool_result", (event) => {
    let content = event.content;
    let normalized = false;
    try {
      const pane=read("started.json").pane;
      const command=String(event.input.command ?? "");
      const output=event.content.filter((x:any)=>x.type==="text").map((x:any)=>x.text).join("\n");
      if (process.env.GAUNTLET_NORMALIZE_CAPTURE_BLANKS === "1" && !event.isError && event.toolName === "bash" && command.includes("tmux") && command.includes("capture-pane") && command.includes(pane) && output.includes("\u2800")) {
        content = event.content.map((item:any) => item.type === "text" ? {...item,text:item.text.replaceAll("\u2800"," ")} : item);
        const changed = content.filter((x:any)=>x.type === "text").map((x:any)=>x.text).join("\n");
        appendFileSync(join(root,"capture-normalization.jsonl"),JSON.stringify({time:Date.now()/1000,tool_call_id:event.toolCallId,command,raw_content:event.content,normalized_content:content,blank_cells:Array.from(output).filter(c=>c==="\u2800").length,raw_sha256:createHash("sha256").update(output).digest("hex"),normalized_sha256:createHash("sha256").update(changed).digest("hex")})+"\n");
        normalized = true;
      }
      const visibleDots=Array.from(output).filter(c=>c.charCodeAt(0)>0x2800 && c.charCodeAt(0)<=0x28ff).length;
      if (judgeVersion>=8 && !event.isError && event.toolName === "bash" && command.includes("tmux") && command.includes("capture-pane") && command.includes(pane)) {
        writeFileSync(join(root,"self-check-attempt.json"),JSON.stringify({time:Date.now()/1000,source_sha256:hash(),tool_call_id:event.toolCallId,visible_dots:visibleDots})+"\n");
      }
      if (visibleDots>=20 && !output.includes("Traceback (most recent call last)") && !event.isError && event.toolName === "bash" && command.includes("tmux") && command.includes("capture-pane") && command.includes(pane)) {
        writeFileSync(join(root,"self-check.json"),JSON.stringify({time:Date.now()/1000,source_sha256:hash(),tool_call_id:event.toolCallId})+"\n");
      }
    } catch { /* A missing artifact is not accepted. */ }
    if (ready()) return {content:[...content,{type:"text",text:message}]};
    if (judgeVersion>=8) {
      const feedback = functionalFeedback();
      if (feedback) {
        appendFileSync(join(root,"functional-feedback.jsonl"),JSON.stringify({time:Date.now()/1000,source_sha256:hash(),tool_call_id:event.toolCallId,delivery:"persisted-tool-result",text:feedback})+"\n");
        return {content:[...content,{type:"text",text:feedback}]};
      }
    }
    if (normalized) return {content};
  });
  pi.on("tool_call", () => {
    if (ready()) return {block:true,reason:message};
  });
  pi.on("before_provider_request", (event) => {
    const payload=event.payload as Record<string,unknown>;
    const messages=payload.messages as Array<unknown> | undefined;
    if (!Array.isArray(messages)) return;
    if (ready()) return {...payload,tools:[],tool_choice:"none",messages:[...messages,{role:"user",content:message}]};
    if (judgeVersion>=8) return; // Persist repair notes in tool results, not one-request payload edits.
    const feedback = functionalFeedback();
    if (feedback) return {...payload,messages:[...messages,{role:"user",content:feedback}]};
  });
}
