"""The original LoopAgent sample with its state and loop bugs corrected."""
# Fix https://github.com/smithakolan/awesome-ai-agents/blob/056e5206adaa3cae3b71be070b65647d4984415e/build-ai-agent-google-adk/blogger/agent.py

import datetime

from google.adk import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools import agent_tool
from google.adk.tools import exit_loop

blog_planner = Agent(
    name="BlogPlanner",
    description="Creates a practical, skimmable outline in Markdown.",
    instruction="""
    You are a technical content strategist. Produce a clear Markdown outline
    with:
    - Title
    - Short intro
    - 4–6 main sections (each with 2–3 bullets)
    - Conclusion
    If codebase context exists in state, weave in specific sections/snippets:
    {codebase_context?}
    If outline validation feedback exists, address it:
    {outline_validation_result?}
    Return only the outline in Markdown.
    """,
    output_key="blog_outline",
)

outline_validation_checker = Agent(
    name="OutlineValidationChecker",
    description="Validates that the outline is usable.",
    instruction="""
    Check the outline in state:
    {blog_outline}
    If it has a title, intro, 4–6 sections, and a conclusion, call the
    `exit_loop` tool. Otherwise respond exactly "retry" and list missing pieces.
    """,
    output_key="outline_validation_result",
    tools=[exit_loop],
)

robust_blog_planner = LoopAgent(
    name="RobustBlogPlanner",
    description="Retries planning if validation fails.",
    sub_agents=[blog_planner, outline_validation_checker],
    max_iterations=3,
)

blog_writer = Agent(
    name="BlogWriter",
    description="Writes a technical blog post from the outline.",
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

blog_post_validation_checker = Agent(
    name="BlogPostValidationChecker",
    description="Validates the final post.",
    instruction="""
    Check the blog post:
    {blog_post}

    Check it for: intro, clear sections matching the outline, conclusion, and
    technical clarity. If it passes, call the `exit_loop` tool. Else respond
    "retry" with the specific fixes.
    """,
    output_key="blog_validation_result",
    tools=[exit_loop],
)

robust_blog_writer = LoopAgent(
    name="RobustBlogWriter",
    description="Retries writing if validation fails.",
    sub_agents=[blog_writer, blog_post_validation_checker],
    max_iterations=3,
)

planner_tool = agent_tool.AgentTool(agent=robust_blog_planner)
writer_tool = agent_tool.AgentTool(agent=robust_blog_writer)

root_agent = Agent(
    name="Blogger",
    description="Minimal multi-agent blogger that plans and writes.",
    instruction=f"""
    If the user gives a topic:
    1) Call the planner tool to generate the outline.
    2) Call the writer tool to produce the full draft.
    3) Present the full draft from state verbatim:
       {{blog_post?}}
    4) End with 3 alternate titles and 2 tweet-length hooks.

    Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[planner_tool, writer_tool],
)
