import pytest
from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)
from google.adk.evaluation.local_eval_sets_manager import load_eval_set_from_file
from google.adk.evaluation.simulation.user_simulator_provider import (
    UserSimulatorProvider,
)


@pytest.mark.asyncio
async def test_should_use_search_tool(tmp_path):
    test_file = "tests/integration/fixtures/search_agent/test_should_use_search_tool/latest_ai_news.test.json"
    eval_config = AgentEvaluator.find_config_for_test_file(test_file)
    eval_set = load_eval_set_from_file(test_file, eval_set_id="search_agent")
    eval_metrics = get_eval_metrics_from_config(eval_config)
    user_simulator_provider = UserSimulatorProvider(
        user_simulator_config=eval_config.user_simulator_config
    )

    agent_for_eval = await AgentEvaluator._get_agent_for_eval(
        module_name="search_agent", agent_name=None
    )

    eval_results_by_eval_id = await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval,
        eval_set,
        eval_metrics,
        num_runs=1,
        user_simulator_provider=user_simulator_provider,
    )

    results_manager = LocalEvalSetResultsManager(agents_dir=str(tmp_path))
    all_eval_results = [r for v in eval_results_by_eval_id.values() for r in v]
    results_manager.save_eval_set_result(
        app_name="test_app",
        eval_set_id=eval_set.eval_set_id,
        eval_case_results=all_eval_results,
    )

    failures = []
    for eval_case_results in eval_results_by_eval_id.values():
        eval_metric_results = AgentEvaluator._get_eval_metric_results_with_invocation(
            eval_case_results
        )
        failures_per_eval_case = AgentEvaluator._process_metrics_and_get_failures(
            eval_metric_results=eval_metric_results,
            print_detailed_results=True,
            agent_module=None,
        )
        failures.extend(failures_per_eval_case)

    failure_message = "Following are all the test failures.\n" + "\n".join(failures)
    assert not failures, failure_message
