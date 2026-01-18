# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ag-ui-adk>=0.4.1",
#     "google-adk>=1.22.1",
# ]
# ///
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from fastapi import FastAPI
from google.adk.agents import LlmAgent


def set_theme_color(theme_color: str) -> None:
    """Request a UI theme color change.

    Args:
        theme_color: CSS color (hex, name, or rgb) that the UI should apply
    """
    return None


root_agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Be helpful and fun!",
    tools=[set_theme_color],
)

adk_agent = ADKAgent(
    adk_agent=root_agent,
    app_name="demo_app",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True,
)

app = FastAPI()
add_adk_fastapi_endpoint(app, adk_agent, path="/")


if __name__ == "__main__":
    import logging

    import uvicorn

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    )
    console_handler.addFilter(logging.Filter("google_adk"))
    root_logger.addHandler(console_handler)

    uvicorn.run(app, host="localhost", port=8000)
