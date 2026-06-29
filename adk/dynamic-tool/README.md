# dynamic-tool

Minimal ADK sample for trying request-scoped tool availability from `/run`
`stateDelta`.

Run:

```bash
uv run adk api_server --auto_create_session
```

Then call `/run` with `stateDelta`. A custom `BaseToolset` reads
`stateDelta.temp:tool_mode` from `ReadonlyContext.state` and exposes exactly
one transform tool for that run.

```bash
curl -sS http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{
    "appName": "dynamic_tool",
    "userId": "u1",
    "sessionId": "s1",
    "newMessage": {
      "role": "user",
      "parts": [{"text": "Transform hello adk"}]
    },
    "stateDelta": {
      "temp:tool_mode": "reverse"
    }
  }'
```

Supported `temp:tool_mode` values are `upper`, `lower`, `reverse`, and `title`.

For example, `reverse` exposes only the `reverse_text` tool, while `upper`
exposes only the `upper_text` tool. The user message does not choose the mode.

Without `stateDelta.temp:tool_mode`, the toolset exposes `missing_tool_mode`,
which returns an error instead of choosing a default. That keeps this sample
honest: the extra parameter is provided only by the API request.
