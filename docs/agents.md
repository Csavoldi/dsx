# Coding agents

For the [pi](https://github.com/badlogic/pi-mono) coding agent, add this to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "dsx": {
      "baseUrl": "http://localhost:8090/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4.1-flash",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 131072,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": "none",
            "minimal": null,
            "low": "low",
            "medium": null,
            "high": "high",
            "xhigh": null,
            "max": "max"
          }
        }
      ]
    }
  }
}
```

```bash
pi --provider dsx --model deepseek-v4.1-flash:off
```

With `--api-key`: drop `"authHeader": false`, set `"apiKey"` to the key and point `baseUrl` at the server's URL.

Set the provider context window to the server’s `--ctx` value. The example uses 128K; set both to `262144` for a 256K session.

Use `deepseek-v4.1-flash:low` to enable low reasoning. `maxTokens` caps each response, including reasoning tokens; increase it for longer tasks while leaving room for the prompt within the context window.
