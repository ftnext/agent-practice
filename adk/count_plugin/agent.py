from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.tools import ToolContext


async def hello_world(tool_context: ToolContext, query: str) -> None:
    print(f"Hello world: query is [{query}]")


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="hello_world",
    description="Prints hello world with user query.",
    instruction="Use hello_world tool to print hello world and user query.",
    tools=[hello_world],
)

app = App(name="count_plugin", root_agent=root_agent)
