from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types as genai_types

from custom_metrics import any_support_tool_trajectory_metric

_USER_CONTENT = genai_types.Content(parts=[genai_types.Part(text="User input here.")])


def test_evaluate_invocations_equal_tool_calls():
    tool_call = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
    intermediate_data = IntermediateData(tool_uses=[tool_call])
    invocation = Invocation(
        user_content=_USER_CONTENT, intermediate_data=intermediate_data
    )
    result = any_support_tool_trajectory_metric([invocation], [invocation])
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert len(result.per_invocation_results) == 1
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_different_tool_call_names():
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
    result = any_support_tool_trajectory_metric([invocation1], [invocation2])
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_tool_call_args():
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
    result = any_support_tool_trajectory_metric([invocation1], [invocation2])
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_number_of_tool_calls():
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
    result = any_support_tool_trajectory_metric([invocation1], [invocation2])
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_no_tool_calls():
    invocation = Invocation(
        user_content=_USER_CONTENT, intermediate_data=IntermediateData()
    )
    result = any_support_tool_trajectory_metric([invocation], [invocation])
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_multiple_invocations():
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

    result = any_support_tool_trajectory_metric(
        [inv1_actual, inv2_actual], [inv1_expected, inv2_expected]
    )

    assert result.overall_score == 0.5
    assert result.overall_eval_status == EvalStatus.FAILED
    assert len(result.per_invocation_results) == 2
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[1].score == 0.0
    assert result.per_invocation_results[1].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_tool_call_args_any_support():
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
    result = any_support_tool_trajectory_metric(
        [actual_invocation], [expected_invocation]
    )
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED
