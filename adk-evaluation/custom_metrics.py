from google.adk.evaluation.eval_case import ConversationScenario, Invocation
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator


def practice_tool_trajectory_metric(
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None = None,
) -> EvaluationResult:
    trajectory_evaluator = TrajectoryEvaluator(threshold=1.0)
    return trajectory_evaluator.evaluate_invocations(
        actual_invocations, expected_invocations, conversation_scenario
    )
