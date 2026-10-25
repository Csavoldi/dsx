import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Advisory only. Tools still execute, and every reminder remains in the session.
// This follows the official DeepSeek harness's exact-call counts (3, 5, 8).
function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return "{" + Object.keys(object).sort().map(k => JSON.stringify(k) + ":" + canonical(object[k])).join(",") + "}";
  }
  return JSON.stringify(value) ?? "undefined";
}

export default function (pi: ExtensionAPI) {
  let previous = "";
  let repetitions = 0;
  pi.on("before_agent_start", async () => { previous = ""; repetitions = 0; });
  pi.on("tool_result", async event => {
    const key = canonical([event.toolName, event.input]);
    repetitions = key === previous ? repetitions + 1 : 1;
    previous = key;
    if (![3, 5, 8].includes(repetitions)) return;
    pi.sendMessage({
      customType: "repeat-tool-reminder",
      content: `You have made the same tool call with identical arguments ${repetitions} times in a row. Review the returned result before acting again. If the task is complete, give your final response. If work remains, identify what must change and choose an action that makes progress instead of repeating the same call.`,
      display: true,
      details: { toolName: event.toolName, repetitions, policy: "exact-consecutive-call-3-5-8" },
    }, { deliverAs: "steer", triggerTurn: false });
  });
}
