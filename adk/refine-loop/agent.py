from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types


class HumanAsAgent(BaseAgent):
    output_key: str

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_message = input("Please enter a message: ")
        ctx.session.state[self.output_key] = user_message
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=user_message)],
            ),
        )


class CritiqueAgent(LlmAgent):
    review_end_phrase: str

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        async for event in super()._run_async_impl(ctx):
            if event.content.parts[0].text.strip() == self.review_end_phrase:
                event.actions.escalate = True
            yield event


critique_agent = CritiqueAgent(
    name="critique_agent",
    model="gemini-2.0-flash",
    description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
    instruction="""あなたは与えられた文がフォーマルかどうかをレビューします。

**レビュー対象の文:**
```
{current_sentence}
```

**タスク:**
文をレビューし、フォーマルでない場合は、改善点を提案してください。
十分にフォーマルな場合は、「No major issues found.」とだけ応答してください。
""",
    review_end_phrase="No major issues found.",
    output_key="current_sentence",  # Overwrite
)


loop_agent = LoopAgent(
    name="loop_agent",
    description="Loop agent that runs a sub-agent in a loop",
    max_iterations=None,
    sub_agents=[
        HumanAsAgent(
            name="human_as_agent",
            description="Human responds as an agent",
            output_key="current_sentence",
        ),
        critique_agent,
    ],
)


root_agent = loop_agent
