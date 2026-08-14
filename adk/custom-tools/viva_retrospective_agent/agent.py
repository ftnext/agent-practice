from google.adk import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

root_agent = Agent(
    model='gemini-3.7-flash',
    name='ふりかえりマスター',
    instruction='ふりかえりのファシリテーターに対して、ふりかえりMCPサーバを活用してよりよいふりかえりを実施できるように導いてください。',
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="https://hurikaeri-site.viva-tweet-x.workers.dev/mcp",
            )
        ),
    ],
)
