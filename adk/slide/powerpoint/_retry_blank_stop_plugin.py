from __future__ import annotations

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ReflectAndRetryModelPlugin
from google.genai import types

_EMPTY_RESPONSE_ERROR_CODE = "MODEL_RETURNED_EMPTY_TEXT"
_EMPTY_RESPONSE_ERROR_MESSAGE = (
    "The model returned STOP without visible text or a function call. "
    "Continue the unfinished task instead of ending the invocation."
)


def _is_blank_stop(llm_response: LlmResponse) -> bool:
    """Return whether a completed model turn contains no usable action."""
    if llm_response.finish_reason != types.FinishReason.STOP:
        return False
    if llm_response.get_function_calls():
        return False

    parts = llm_response.content.parts if llm_response.content else []
    for part in parts:
        if part.text and part.text.strip() and not part.thought:
            return False
        if (
            part.inline_data
            or part.file_data
            or part.executable_code
            or part.code_execution_result
        ):
            return False
    return True


class RetryBlankStopPlugin(ReflectAndRetryModelPlugin):
    """Retry blank STOP responses from one coordinator agent."""

    def __init__(
        self,
        *,
        agent_name: str,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            name="retry_blank_stop_plugin",
            max_retries=max_retries,
            throw_exception_if_retry_exceeded=True,
            on_model_errors=[
                types.FinishReason.MALFORMED_FUNCTION_CALL,
                types.FinishReason.STOP,
            ],
        )
        self._agent_name = agent_name

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Convert a targeted blank STOP into the plugin's retry flow."""
        invocation_context = callback_context.get_invocation_context()
        agent = invocation_context.agent
        if (
            isinstance(agent, LlmAgent)
            and agent.name == self._agent_name
            and _is_blank_stop(llm_response)
        ):
            llm_response = llm_response.model_copy(
                update={
                    "error_code": _EMPTY_RESPONSE_ERROR_CODE,
                    "error_message": _EMPTY_RESPONSE_ERROR_MESSAGE,
                }
            )

        return await super().after_model_callback(
            callback_context=callback_context,
            llm_response=llm_response,
        )
