from google.adk.agents.llm_agent import Agent
from google.adk.apps import App
from google.adk.tools import google_search

search_agent = Agent(
    model='gemini-3.1-pro-preview',
    name='search_agent',
    description='A helpful assistant that can search Google.',
    instruction="""\
You are a helpful assistant with access to Google Search.

If the user asks a question that requires current information or facts, use the 'google_search' tool with English query.
For current news questions, answer in Japanese using a concise bullet list.
For each news item, include the announcement timing and at least one source URL from the search results.
Do not include a news item if you cannot provide a source URL for it.
""",
    tools=[google_search],
)

root_agent = search_agent
app = App(name="search_agent", root_agent=root_agent)
