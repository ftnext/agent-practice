# ref: https://github.com/google/adk-python/blob/v2.6.3/contributing/samples/multimodal/static_non_text_content/agent.py
from google.adk import Agent
from google.genai import types

root_agent = Agent(
    name="static_pdf_content_demo_agent",
    model="gemini-3.6-flash",
    description="Demonstrates static instructions with non-text content (PDF file)",
    static_instruction=types.Content(
        parts=[
            types.Part.from_text(
                text="You are an AI assistant that analyzes images and documents. You have access to the following reference materials:"
            ),
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
                    mime_type="application/pdf",
                    display_name="AI Research Paper",
                )
            ),
            types.Part.from_text(
                text="""When users ask questions, you should:
1. Reference the AI research paper (from GCS) when discussing AI research, model architectures, or technical details
2. Be helpful and informative in your responses
3. Explain how the provided reference materials relate to their questions

Remember: The reference materials above are available to help you provide better answers."""
            ),
        ],
    ),
    instruction="Please analyze the user's question and provide helpful insights. Reference the materials provided in your static instructions when relevant.",
)
