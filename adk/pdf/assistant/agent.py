# ref: https://github.com/google/adk-python/blob/v2.6.2/contributing/samples/multimodal/multimodal/agent.py

from google.adk import Agent

root_agent = Agent(
    model="gemini-3.6-flash",
    name="root_agent",
)
