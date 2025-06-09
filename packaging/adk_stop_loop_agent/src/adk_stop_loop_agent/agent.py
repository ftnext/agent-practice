from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types


class CounterAgent(BaseAgent):
    counter: int = 0
    stop_at: int = 2

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        self.counter += 1
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=f"Counter: {self.counter}")],
            ),
        )

        if self.counter == self.stop_at:
            self.counter = 0
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="Send STOP signal")]
                ),
                actions=EventActions(escalate=True),
            )


loop_agent = LoopAgent(
    name="loop_agent",
    description="Loop agent that runs a sub-agent in a loop",
    max_iterations=None,
    sub_agents=[
        CounterAgent(
            name="counter_agent",
            description="Counter agent that counts up to a stop signal",
        )
    ],
)


root_agent = loop_agent
