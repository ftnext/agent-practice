from google.adk.agents.llm_agent import Agent
from google.adk.apps import App
from google.adk.tools import google_search

search_agent = Agent(
    model='gemini-3-pro-preview',
    name='search_agent',
    description='A helpful assistant that can search Google.',
    instruction="""\
You are a helpful assistant with access to Google Search.

If the user asks a question that requires current information or facts, use the 'google_search' tool with English query.
Provide the answer clearly based on the search results and always cite your sources by including URLs from the search results.
""",
    tools=[google_search],
)

root_agent = search_agent
app = App(name="search_agent", root_agent=root_agent)
