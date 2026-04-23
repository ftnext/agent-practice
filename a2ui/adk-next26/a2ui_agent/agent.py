from google.adk.agents import Agent

from .resources import get_resources

root_agent = Agent(
    model="gemini-3-flash-preview",
    name="cloud_dashboard",
    description="A cloud infrastructure assistant that reports on project resources.",
    instruction=(
        "You are a cloud infrastructure assistant. When users ask about their "
        "cloud resources, use the get_resources tool to fetch the current state. "
        "Summarize the results clearly in plain text."
    ),
    tools=[get_resources],
)
