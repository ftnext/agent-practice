from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

tool_correctness = ToolCorrectnessMetric()


def test_agent_tool_choice():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")],
        expected_tools=[ToolCall(name="WebSearch")],
    )

    evaluate([test_case], metrics=[tool_correctness])
