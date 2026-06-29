# dynamic-tool

Minimal ADK sample for trying request-scoped tool behavior from `/run`
`stateDelta`.

Run:

```bash
uv run adk api_server --auto_create_session
```

Then call `/run` with `stateDelta`. The transform mode is not exposed as a
chat tool and should not be inferred from the user message.

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

Without `stateDelta.temp:tool_mode`, the tool returns an error instead of
choosing a default mode. That keeps this sample honest: the extra parameter is
provided only by the API request.
