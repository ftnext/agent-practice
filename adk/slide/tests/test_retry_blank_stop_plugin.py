from __future__ import annotations

import unittest
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from powerpoint._retry_blank_stop_plugin import RetryBlankStopPlugin


def _callback_context(agent_name: str) -> CallbackContext:
    agent = Mock(spec=LlmAgent)
    agent.name = agent_name
    agent.canonical_model = Mock()
    agent.canonical_model.model = "gemini-3.7-flash"

    invocation_context = Mock()
    invocation_context.agent = agent
    invocation_context.invocation_id = "test-invocation"

    context = Mock(spec=CallbackContext)
    context.get_invocation_context.return_value = invocation_context
    return context


class RetryBlankStopPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_blank_stop_is_replaced_with_retry_tool_call(self) -> None:
        """A blank STOP response triggers an internal reflection retry."""
        plugin = RetryBlankStopPlugin(
            agent_name="powerpoint_agent",
            max_retries=2,
        )
        response = types.Content(
            role="model",
            parts=[types.Part.from_text(text="")],
        )

        result = await plugin.after_model_callback(
            callback_context=_callback_context("powerpoint_agent"),
            llm_response=_llm_response(response),
        )

        self.assertIsNotNone(result)
        function_calls = result.get_function_calls()
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0].name, "adk_handle_model_error")
        self.assertEqual(
            function_calls[0].args["error_type"],
            "MODEL_RETURNED_EMPTY_TEXT",
        )

    async def test_non_blank_stop_passes_through(self) -> None:
        """A visible final answer is not retried."""
        plugin = RetryBlankStopPlugin(agent_name="powerpoint_agent")
        response = types.Content(
            role="model",
            parts=[types.Part.from_text(text="completed")],
        )

        result = await plugin.after_model_callback(
            callback_context=_callback_context("powerpoint_agent"),
            llm_response=_llm_response(response),
        )

        self.assertIsNone(result)

    async def test_blank_stop_from_other_agent_passes_through(self) -> None:
        """A blank response from a non-target agent is not retried."""
        plugin = RetryBlankStopPlugin(agent_name="powerpoint_agent")
        response = types.Content(
            role="model",
            parts=[types.Part.from_text(text="")],
        )

        result = await plugin.after_model_callback(
            callback_context=_callback_context("search_researcher"),
            llm_response=_llm_response(response),
        )

        self.assertIsNone(result)

    async def test_agent_continues_after_blank_stop(self) -> None:
        """The ADK model loop retries and returns the next visible response."""

        class FakeLlm(BaseLlm):
            responses: list[LlmResponse]

            async def generate_content_async(
                self,
                llm_request: LlmRequest,
                stream: bool = False,
            ):
                del llm_request, stream
                yield self.responses.pop(0)

        model = FakeLlm(
            model="fake-model",
            responses=[
                _llm_response(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="")],
                    )
                ),
                _llm_response(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="completed")],
                    )
                ),
            ],
        )
        agent = LlmAgent(
            name="powerpoint_agent",
            model=model,
            instruction="Complete the task.",
        )
        app = App(
            name="retry_blank_stop_test",
            root_agent=agent,
            plugins=[
                RetryBlankStopPlugin(
                    agent_name=agent.name,
                    max_retries=2,
                )
            ],
        )
        runner = InMemoryRunner(app=app)
        session = await runner.session_service.create_session(
            app_name=app.name,
            user_id="test-user",
        )

        events = [
            event
            async for event in runner.run_async(
                user_id="test-user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="make slides")],
                ),
            )
        ]

        retry_calls = [
            function_call
            for event in events
            for function_call in event.get_function_calls()
            if function_call.name == "adk_handle_model_error"
        ]
        self.assertEqual(len(retry_calls), 1)
        self.assertEqual(events[-1].content.parts[0].text, "completed")
        self.assertEqual(model.responses, [])


def _llm_response(content: types.Content) -> LlmResponse:
    return LlmResponse(
        content=content,
        finish_reason=types.FinishReason.STOP,
    )


if __name__ == "__main__":
    unittest.main()
