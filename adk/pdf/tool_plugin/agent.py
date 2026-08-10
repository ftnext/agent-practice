from urllib.parse import urlparse

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.multimodal_tool_results_plugin import (
    MultimodalToolResultsPlugin,
)
from google.genai import types


def get_gcs_file_content(uri: str):
    """
    Get the content of a file from Google Cloud Storage.

    Args:
        uri: The URI of the file in Google Cloud Storage.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        return f"[Error] Only gs:// URIs are supported: {uri}"
    if not parsed.path or parsed.path == "/":
        return f"[Error] The GCS object path is missing: {uri}"
    if not uri.lower().endswith(".pdf"):
        return f"[Error] Only PDF files are supported: {uri}"

    return [types.Part.from_uri(file_uri=uri, mime_type="application/pdf")]


root_agent = Agent(
    name="assistant",
    model="gemini-3.6-flash",
    static_instruction="""You are an AI assistant.

ユーザが「gs://bucket-name/object-name について要約して」のようにGCSのURI込みで依頼したとき、
まずツール get_gcs_file_content(uri="gs://bucket-name/object-name") と呼び出すこと。""",
    instruction="Please analyze the user's question and provide helpful insights. Reference the materials provided by `get_gcs_file_content` tool.",
    tools=[get_gcs_file_content],
)

app = App(
    name="multimodal_tool_results",
    root_agent=root_agent,
    plugins=[MultimodalToolResultsPlugin()],
)
