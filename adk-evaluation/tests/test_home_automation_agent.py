import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator


@pytest.mark.skip(reason="`adk eval` only supports custom metrics")
@pytest.mark.asyncio
async def test_with_single_test_file():
    await AgentEvaluator.evaluate(
        agent_module="home_automation_agent",
        eval_dataset_file_path_or_dir="tests/fixtures/home_automation_agent/simple_test.test.json",
        num_runs=1,
    )
