from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types


class ParrotAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_message = input("Please enter a message: ")
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=user_message)],
            ),
        )


loop_agent = LoopAgent(
    name="loop_agent",
    description="Loop agent that runs a sub-agent in a loop",
    max_iterations=None,
    sub_agents=[
        ParrotAgent(
            name="parrot_agent",
            description="Parrot agent that echoes user input",
        )
    ],
)


root_agent = loop_agent
