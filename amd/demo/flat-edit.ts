import { createEditToolDefinition, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// DSML escapes top-level string parameters. Keeping code strings out of nested
// JSON avoids requiring the model to manually escape every code quote/newline.
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "edit",
    label: "edit",
    description: "Replace one unique exact text block in a file. oldText must match existing text exactly, including whitespace. Use separate calls for separate replacements.",
    promptSnippet: "Make precise file edits with exact text replacement",
    promptGuidelines: ["Use edit with top-level path, oldText, and newText strings. Keep oldText as small as possible while remaining unique."],
    parameters: Type.Object({
      path: Type.String({ description: "File path, absolute or relative to the current directory" }),
      oldText: Type.String({ description: "Unique existing text to replace, exactly as it appears" }),
      newText: Type.String({ description: "Replacement text" }),
    }),
    async execute(id, args, signal, onUpdate, ctx) {
      const original = createEditToolDefinition(ctx.cwd);
      return original.execute(id, { path: args.path, edits: [{ oldText: args.oldText, newText: args.newText }] }, signal, onUpdate, ctx);
    },
  });
}
