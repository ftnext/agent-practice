import pytest
from google.adk.evaluation import AgentEvaluator


@pytest.mark.asyncio
async def test_should_use_search_tool():
     await AgentEvaluator.evaluate(
        agent_module="search_agent",
        eval_dataset_file_path_or_dir="tests/integration/fixtures/search_agent/test_should_use_search_tool",
        num_runs=1,
    )
