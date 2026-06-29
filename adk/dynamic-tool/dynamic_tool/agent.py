from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext

ALLOWED_MODES = ("lower", "reverse", "title", "upper")


def transform_text(text: str, tool_context: ToolContext) -> dict:
  """Transform text using the tool mode supplied via request state.

  Args:
    text: Text to transform.
    tool_context: ADK tool context. The tool reads temp:tool_mode from state.

  Returns:
    A dictionary containing the selected mode and transformed text.
  """
  mode = tool_context.state.get("temp:tool_mode")
  if not mode:
    return {
        "status": "error",
        "message": (
            "Missing temp:tool_mode. Provide it through the /run request's "
            "stateDelta field."
        ),
        "allowed_modes": list(ALLOWED_MODES),
    }

  mode = str(mode).lower().strip()
  transforms = {
      "upper": str.upper,
      "lower": str.lower,
      "reverse": lambda value: value[::-1],
      "title": str.title,
  }
  transform = transforms.get(mode)
  if not transform:
    return {
        "status": "error",
        "message": f"Unsupported temp:tool_mode: {mode}",
        "allowed_modes": sorted(transforms),
    }

  return {
      "status": "ok",
      "mode": mode,
      "input": text,
      "output": transform(text),
  }


root_agent = Agent(
    model="gemini-2.5-flash",
    name="dynamic_tool",
    description="A small agent controlled by request stateDelta.",
    instruction=(
        "You transform user-provided text by calling transform_text. "
        "The transform mode must come only from the session state key "
        "temp:tool_mode, which is supplied by the API caller through the "
        "/run request stateDelta field. Do not infer or set the mode from "
        "the user's natural-language message. If the tool reports an error, "
        "explain the error briefly."
    ),
    tools=[transform_text],
)
