from __future__ import annotations

from typing import Literal

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from pydantic import BaseModel
from pydantic import Field


class Review(BaseModel):
    """Structured result returned by the draft reviewer."""

    grade: Literal["approved", "needs_revision"] = Field(
        description="Whether the announcement is ready or needs revision."
    )
    feedback: str = Field(
        description="Actionable revision feedback, or a short approval note."
    )


def process_input(node_input: str) -> Event:
    """Stores the request so it remains available across loop iterations."""
    return Event(state={"announcement_request": node_input})


write_draft = Agent(
    name="write_draft",
    instruction="""
    Write a short announcement for this request: {announcement_request}
    Revise the announcement according to this review when present:
    {review?}
    Return only the announcement.
    """,
    output_key="current_draft",
)


review_draft = Agent(
    name="review_draft",
    instruction="""
    Review the announcement in {current_draft} for a professional tone.
    Return approved if it is ready. Otherwise return needs_revision and
    concise, actionable feedback.
    """,
    output_schema=Review,
    output_key="review",
)


def route_review(node_input: Review) -> Event:
    """Routes drafts that need revision back to the writer."""
    return Event(route=node_input.grade)


root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", process_input, write_draft, review_draft, route_review),
        (route_review, {"needs_revision": write_draft}),
    ],
)
