`uv run adk api_server --auto_create_session`

```console
curl -X POST \
  http://localhost:8000/run \
  --json '{
    "appName": "assistant",
    "userId": "user-001",
    "sessionId": "session-001",
    "newMessage": {
      "role": "user",
      "parts": [
        {
          "fileData": {
            "fileUri": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
            "mimeType": "application/pdf"
          }
        },
        {
          "text": "このPDFの目次を教えてください。"
        }
      ]
    }
  }'
```
