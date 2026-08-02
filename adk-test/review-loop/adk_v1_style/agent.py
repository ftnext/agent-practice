from google.adk import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools import exit_loop


write_draft = Agent(
    name="write_draft",
    description="Writes and revises a short announcement.",
    instruction="""
    Write a short announcement for the user's request.
    Revise the announcement according to this review feedback when present:
    {review_feedback?}
    Return only the announcement.
    """,
    output_key="current_draft",
)


review_draft = Agent(
    name="review_draft",
    description="Reviews whether an announcement uses a professional tone.",
    instruction="""
    Review the announcement in {current_draft} for a professional tone.
    If it needs revision, return concise, actionable feedback.
    If it is ready, call the exit_loop tool and do not return any text.
    """,
    output_key="review_feedback",
    tools=[exit_loop],
)


root_agent = LoopAgent(
    name="root_agent",
    sub_agents=[write_draft, review_draft],
)
