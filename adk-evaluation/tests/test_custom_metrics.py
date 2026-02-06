from unittest.mock import MagicMock

import pytest
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.genai import types as genai_types

from custom_metrics import args_any_support_tool_trajectory_metric

_USER_CONTENT = genai_types.Content(parts=[genai_types.Part(text="User input here.")])


@pytest.fixture()
def eval_metric():
    return MagicMock(spec=EvalMetric)


def test_evaluate_invocations_equal_tool_calls(eval_metric):
    tool_call = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    intermediate_data = IntermediateData(tool_uses=[tool_call])
    invocation = Invocation(
        user_content=_USER_CONTENT, intermediate_data=intermediate_data
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [invocation], [invocation]
    )
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert len(result.per_invocation_results) == 1
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_different_tool_call_names(eval_metric):
    tool_call1 = genai_types.FunctionCall(name="test_func1", args={"arg1": "val1"})
    tool_call2 = genai_types.FunctionCall(name="test_func2", args={"arg1": "val1"})
    invocation1 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    invocation2 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call2]),
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [invocation1], [invocation2]
    )
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_tool_call_args(eval_metric):
    tool_call1 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    tool_call2 = genai_types.FunctionCall(name="test_func", args={"arg1": "val2"})
    invocation1 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    invocation2 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call2]),
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [invocation1], [invocation2]
    )
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_number_of_tool_calls(eval_metric):
    tool_call1 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    tool_call2 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    invocation1 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    invocation2 = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1, tool_call2]),
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [invocation1], [invocation2]
    )
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_no_tool_calls(eval_metric):
    invocation = Invocation(
        user_content=_USER_CONTENT, intermediate_data=IntermediateData()
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [invocation], [invocation]
    )
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_multiple_invocations(eval_metric):
    tool_call1 = genai_types.FunctionCall(name="test_func1", args={"arg1": "val1"})
    tool_call2 = genai_types.FunctionCall(name="test_func2", args={"arg1": "val1"})
    inv1_actual = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    inv1_expected = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    inv2_actual = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call1]),
    )
    inv2_expected = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[tool_call2]),
    )

    result = args_any_support_tool_trajectory_metric(
        eval_metric, [inv1_actual, inv2_actual], [inv1_expected, inv2_expected]
    )

    assert result.overall_score == 0.5
    assert result.overall_eval_status == EvalStatus.FAILED
    assert len(result.per_invocation_results) == 2
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[1].score == 0.0
    assert result.per_invocation_results[1].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_tool_call_args_any_support(eval_metric):
    actual_tool_call = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    expected_tool_call = genai_types.FunctionCall(
        name="test_func", args={"arg1": "ANY"}
    )
    actual_invocation = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[actual_tool_call]),
    )
    expected_invocation = Invocation(
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_uses=[expected_tool_call]),
    )
    result = args_any_support_tool_trajectory_metric(
        eval_metric, [actual_invocation], [expected_invocation]
    )
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED
