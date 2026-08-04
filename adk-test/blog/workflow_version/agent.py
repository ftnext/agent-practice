"""A blog-writing workflow with explicit plan and draft review loops."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from google.adk import Agent, Event, Workflow
from pydantic import BaseModel, Field

_MAX_REVIEW_ATTEMPTS = 3


class Review(BaseModel):
  """A structured quality review used to route a revision loop."""

  grade: Literal["ok", "retry"] = Field(
      description="Whether the content passes or needs another attempt.",
  )
  feedback: str = Field(
      description=(
          "Concrete revision instructions, or a short approval explanation."
      ),
  )


class PromotionSuggestions(BaseModel):
  """Promotional copy generated without rewriting the approved article."""

  alternate_titles: list[str] = Field(min_length=3, max_length=3)
  social_hooks: list[str] = Field(min_length=2, max_length=2)


class BlogPackage(BaseModel):
  """The approved article and its promotional copy."""

  article: str
  alternate_titles: list[str]
  social_hooks: list[str]


def process_input(node_input: str) -> Event:
  """Stores the requested topic and initializes review counters."""
  return Event(
      state={
          "request": node_input,
          "outline_review_attempts": 0,
          "article_review_attempts": 0,
      }
  )


plan_blog = Agent(
    name="BlogPlanner",
    description="Creates a practical, skimmable Markdown outline.",
    instruction="""
    You are a technical content strategist. Produce a clear Markdown outline
    for this request: {request}
    - Title
    - Short intro
    - 4–6 main sections (each with 2–3 bullets)
    - Conclusion
    If codebase context exists in state, weave in specific sections/snippets:
    {codebase_context?}
    If outline validation feedback exists, address it:
    {outline_validation_result?}
    Return only the Markdown outline.
    """,
    output_key="blog_outline",
)


review_outline = Agent(
    name="OutlineValidationChecker",
    description="Checks whether the outline is usable.",
    instruction="""
    Check the outline in state:
    {blog_outline}
    If it has a title, intro, 4–6 sections, and a conclusion, set grade to "ok".
    Otherwise set grade to "retry" and list missing pieces in feedback.
    """,
    output_schema=Review,
    output_key="outline_validation_result",
)


def route_outline(node_input: Review, outline_review_attempts: int) -> Event:
  """Routes an outline to revision, writing, or a bounded failure."""
  attempts = outline_review_attempts + 1
  if node_input.grade == "ok":
    route = "ok"
  elif attempts < _MAX_REVIEW_ATTEMPTS:
    route = "retry"
  else:
    route = "failed"
  return Event(
      route=route,
      state={"outline_review_attempts": attempts},
  )


write_blog = Agent(
    name="BlogWriter",
    description="Writes a technical article from the approved outline.",
    instruction="""
    Write a complete Markdown article from the outline:
    {blog_outline}
    Guidelines:
    - Audience: software engineers; skip basics and focus on practical insight.
    - Explain both the 'how' and 'why'.
    - Include concise code snippets when helpful.
    - Follow the outline’s structure (H2/H3).
    - Output only the final article in Markdown (no fence around the whole post).
    If blog post validation feedback exists, address it:
    {blog_validation_result?}
    """,
    output_key="blog_post",
)


review_blog = Agent(
    name="BlogPostValidationChecker",
    description="Checks the article for structure and technical clarity.",
    instruction="""
    Check the blog post:
    {blog_post}
    Check it for: intro, clear sections matching the outline, conclusion, and
    technical clarity. If it passes, set grade to "ok". Else set grade to
    "retry" with the specific fixes in feedback.
    """,
    output_schema=Review,
    output_key="blog_validation_result",
)


def route_blog(node_input: Review, article_review_attempts: int) -> Event:
  """Routes an article to revision, publication, or a bounded failure."""
  attempts = article_review_attempts + 1
  if node_input.grade == "ok":
    route = "ok"
  elif attempts < _MAX_REVIEW_ATTEMPTS:
    route = "retry"
  else:
    route = "failed"
  return Event(
      route=route,
      state={"article_review_attempts": attempts},
  )


suggest_promotion = Agent(
    name="Blogger",
    description="Creates promotional copy for the approved article.",
    instruction="""
    The full draft is below:
    {blog_post}
    End with exactly 3 alternate titles and 2 tweet-length hooks. Do not rewrite
    the full draft.
    """,
    output_schema=PromotionSuggestions,
    output_key="promotion_suggestions",
)


def publish_blog(
    node_input: PromotionSuggestions,
    blog_post: str,
) -> Iterator[Event]:
  """Displays and returns the approved article with promotional suggestions."""
  package = BlogPackage(
      article=blog_post,
      alternate_titles=node_input.alternate_titles,
      social_hooks=node_input.social_hooks,
  )
  titles = "\n".join(f"- {title}" for title in package.alternate_titles)
  hooks = "\n".join(f"- {hook}" for hook in package.social_hooks)
  yield Event(
      message=(
          f"{package.article}\n\n"
          f"## Alternate titles\n\n{titles}\n\n"
          f"## Social hooks\n\n{hooks}"
      )
  )
  yield Event(output=package.model_dump())


def report_outline_failure(outline_validation_result: Review) -> Event:
  """Reports that the outline never met the quality gate."""
  message = (
      "The outline did not pass review after "
      f"{_MAX_REVIEW_ATTEMPTS} attempts: "
      f"{outline_validation_result.feedback}"
  )
  return Event(message=message, output=message)


def report_article_failure(blog_validation_result: Review) -> Event:
  """Reports that the article never met the quality gate."""
  message = (
      "The article did not pass review after "
      f"{_MAX_REVIEW_ATTEMPTS} attempts: {blog_validation_result.feedback}"
  )
  return Event(message=message, output=message)


root_agent = Workflow(
    name="BloggerWorkflow",
    edges=[
        ("START", process_input, plan_blog, review_outline, route_outline),
        (
            route_outline,
            {
                "retry": plan_blog,
                "ok": write_blog,
                "failed": report_outline_failure,
            },
        ),
        (write_blog, review_blog, route_blog),
        (
            route_blog,
            {
                "retry": write_blog,
                "ok": suggest_promotion,
                "failed": report_article_failure,
            },
        ),
        (suggest_promotion, publish_blog),
    ],
)
