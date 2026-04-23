from a2ui.basic_catalog.provider import BasicCatalog
from a2ui.schema.manager import A2uiSchemaManager
from google.adk.agents import Agent

from .resources import get_resources

schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[BasicCatalog.get_config("0.8")],
)

instruction = schema_manager.generate_system_prompt(
    role_description=(
        "You are a cloud infrastructure assistant. When users ask about their "
        "cloud resources, use the get_resources tool to fetch the current state."
    ),
    workflow_description="Analyze the user's request and return structured UI when appropriate.",
    ui_description=(
        "Use cards for resource summaries, rows and columns for comparisons, "
        "icons for status indicators, and buttons for drill-down actions. "
        "Do NOT use markdown formatting in text values. Use the usageHint "
        "property for heading levels instead. "
        "Respond ONLY with the A2UI JSON array. Do NOT include any text "
        "outside the JSON. Put all explanations into Text components."
    ),
    include_schema=True,
    include_examples=True,
)

root_agent = Agent(
    model="gemini-3-flash-preview",
    name="cloud_dashboard",
    description="A cloud infrastructure assistant that renders rich A2UI interfaces.",
    instruction=instruction,
    tools=[get_resources],
)
