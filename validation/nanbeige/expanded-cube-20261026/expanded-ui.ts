import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    if (ctx.hasUI) {
      ctx.ui.setToolsExpanded(true);
      ctx.ui.setStatus("expanded-recording", "Tools expanded | thinking display visible");
    }
  });
}
