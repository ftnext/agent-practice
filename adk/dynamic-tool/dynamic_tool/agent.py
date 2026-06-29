from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

ALLOWED_MODES = ("lower", "reverse", "title", "upper")


def upper_text(text: str) -> dict:
  """Convert text to uppercase.

  Args:
    text: Text to transform.

  Returns:
    A dictionary containing the selected mode and transformed text.
  """
  return {"status": "ok", "mode": "upper", "input": text, "output": text.upper()}


def lower_text(text: str) -> dict:
  """Convert text to lowercase.

  Args:
    text: Text to transform.

  Returns:
    A dictionary containing the selected mode and transformed text.
  """
  return {"status": "ok", "mode": "lower", "input": text, "output": text.lower()}


def reverse_text(text: str) -> dict:
  """Reverse text.

  Args:
    text: Text to transform.

  Returns:
    A dictionary containing the selected mode and transformed text.
  """
  return {
      "status": "ok",
      "mode": "reverse",
      "input": text,
      "output": text[::-1],
  }


def title_text(text: str) -> dict:
  """Convert text to title case.

  Args:
    text: Text to transform.

  Returns:
    A dictionary containing the selected mode and transformed text.
  """
  return {"status": "ok", "mode": "title", "input": text, "output": text.title()}


def missing_tool_mode(text: str) -> dict:
  """Report that no transform mode was supplied by the API caller.

  Args:
    text: Text that could not be transformed.

  Returns:
    A dictionary explaining the missing stateDelta field.
  """
  return {
      "status": "error",
      "input": text,
      "message": (
          "Missing temp:tool_mode. Provide it through the /run request's "
          "stateDelta field."
      ),
      "allowed_modes": list(ALLOWED_MODES),
  }


class DynamicTransformToolset(BaseToolset):
  """Expose one transform tool based on stateDelta-backed session state."""

  def __init__(self):
    super().__init__()
    self._tools_by_mode = {
        "upper": FunctionTool(upper_text),
        "lower": FunctionTool(lower_text),
        "reverse": FunctionTool(reverse_text),
        "title": FunctionTool(title_text),
    }
    self._missing_mode_tool = FunctionTool(missing_tool_mode)

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    mode = None
    if readonly_context:
      mode = readonly_context.state.get("temp:tool_mode")
    if mode is not None:
      mode = str(mode).lower().strip()

    tool = self._tools_by_mode.get(mode)
    if tool:
      return [tool]
    return [self._missing_mode_tool]


root_agent = Agent(
    model="gemini-2.5-flash",
    name="dynamic_tool",
    description="A small agent whose available tool is controlled by stateDelta.",
    instruction=(
        "Use the available tool to transform the user's text. "
        "The API caller controls which transform tool is available by setting "
        "stateDelta.temp:tool_mode to one of upper, lower, reverse, or title. "
        "Do not infer the mode from the user's natural-language message. "
        "If the available tool reports an error, explain it briefly."
    ),
    tools=[DynamicTransformToolset()],
)
