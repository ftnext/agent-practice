from __future__ import annotations

import os
import os.path
from typing import Optional

import pytest
from google.adk.cli.cli_eval import get_default_metric_info
from google.adk.evaluation.agent_evaluator import NUM_RUNS, AgentEvaluator
from google.adk.evaluation.base_eval_service import (
    EvalCaseResult,
    EvaluateConfig,
    EvaluateRequest,
    InferenceConfig,
    InferenceRequest,
)
from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.eval_config import EvalConfig, get_eval_metrics_from_config
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.evaluation.metric_evaluator_registry import (
    _get_default_metric_evaluator_registry,
)
from google.adk.evaluation.simulation.user_simulator_provider import (
    UserSimulatorProvider,
)
from google.adk.runners import Aclosing


@pytest.mark.skip(reason="`adk eval` only supports custom metrics")
@pytest.mark.asyncio
async def test_with_single_test_file():
    await AgentEvaluator.evaluate(
        agent_module="home_automation_agent",
        eval_dataset_file_path_or_dir="tests/fixtures/home_automation_agent/simple_test.test.json",
        num_runs=1,
    )


@pytest.mark.asyncio
async def test_with_single_test_file_workaround():
    await CustomMetricsSupportAgentEvaluator.evaluate(
        agent_module="home_automation_agent",
        eval_dataset_file_path_or_dir="tests/fixtures/home_automation_agent/simple_test.test.json",
        num_runs=1,
    )


class CustomMetricsSupportAgentEvaluator(AgentEvaluator):
    @staticmethod
    async def evaluate_eval_set(
        agent_module: str,
        eval_set: EvalSet,
        criteria: Optional[dict[str, float]] = None,
        eval_config: Optional[EvalConfig] = None,
        num_runs: int = NUM_RUNS,
        agent_name: Optional[str] = None,
        print_detailed_results: bool = True,
    ):
        """Evaluates an agent using the given EvalSet with custom metrics."""
        if criteria:
            base_criteria = {k: BaseCriterion(threshold=v) for k, v in criteria.items()}
            eval_config = EvalConfig(criteria=base_criteria)

        if eval_config is None:
            raise ValueError("`eval_config` is required.")

        agent_for_eval = await AgentEvaluator._get_agent_for_eval(
            module_name=agent_module, agent_name=agent_name
        )
        eval_metrics = get_eval_metrics_from_config(eval_config)

        user_simulator_provider = UserSimulatorProvider(
            user_simulator_config=eval_config.user_simulator_config
        )

        metric_evaluator_registry = _get_default_metric_evaluator_registry()
        if eval_config.custom_metrics:
            for metric_name, config in eval_config.custom_metrics.items():
                if config.metric_info:
                    metric_info = config.metric_info.model_copy()
                    metric_info.metric_name = metric_name
                else:
                    metric_info = get_default_metric_info(
                        metric_name=metric_name, description=config.description
                    )
                metric_evaluator_registry.register_evaluator(
                    metric_info, _CustomMetricEvaluator
                )

        # It is okay to pick up this dummy name.
        app_name = "test_app"
        eval_service = LocalEvalService(
            root_agent=agent_for_eval,
            eval_sets_manager=AgentEvaluator._get_eval_sets_manager(
                app_name=app_name, eval_set=eval_set
            ),
            user_simulator_provider=user_simulator_provider,
            metric_evaluator_registry=metric_evaluator_registry,
        )

        inference_requests = [
            InferenceRequest(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                inference_config=InferenceConfig(),
            )
        ] * num_runs  # Repeat inference request num_runs times.

        # Generate inferences
        inference_results = []
        for inference_request in inference_requests:
            async with Aclosing(
                eval_service.perform_inference(inference_request=inference_request)
            ) as agen:
                async for inference_result in agen:
                    inference_results.append(inference_result)

        # Evaluate metrics
        # As we perform more than one run for an eval case, we collect eval results
        # by eval id.
        eval_results_by_eval_id: dict[str, list[EvalCaseResult]] = {}
        evaluate_request = EvaluateRequest(
            inference_results=inference_results,
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )
        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_result in agen:
                eval_id = eval_result.eval_id
                if eval_id not in eval_results_by_eval_id:
                    eval_results_by_eval_id[eval_id] = []

                eval_results_by_eval_id[eval_id].append(eval_result)

        failures: list[str] = []

        for _, eval_results_per_eval_id in eval_results_by_eval_id.items():
            eval_metric_results = (
                AgentEvaluator._get_eval_metric_results_with_invocation(
                    eval_results_per_eval_id
                )
            )
            failures_per_eval_case = AgentEvaluator._process_metrics_and_get_failures(
                eval_metric_results=eval_metric_results,
                print_detailed_results=print_detailed_results,
                agent_module=agent_name,
            )

            failures.extend(failures_per_eval_case)

        failure_message = "Following are all the test failures."
        if not print_detailed_results:
            failure_message += (
                " If you looking to get more details on the failures, then please"
                " re-run this test with `print_detailed_results` set to `True`."
            )
        failure_message += "\n" + "\n".join(failures)
        assert not failures, failure_message

    @staticmethod
    async def evaluate(
        agent_module: str,
        eval_dataset_file_path_or_dir: str,
        num_runs: int = NUM_RUNS,
        agent_name: Optional[str] = None,
        initial_session_file: Optional[str] = None,
        print_detailed_results: bool = True,
    ):
        """Evaluates an Agent given eval data with custom metrics."""
        test_files = []
        if isinstance(eval_dataset_file_path_or_dir, str) and os.path.isdir(
            eval_dataset_file_path_or_dir
        ):
            for root, _, files in os.walk(eval_dataset_file_path_or_dir):
                for file in files:
                    if file.endswith(".test.json"):
                        test_files.append(os.path.join(root, file))
        else:
            test_files = [eval_dataset_file_path_or_dir]

        initial_session = CustomMetricsSupportAgentEvaluator._get_initial_session(
            initial_session_file
        )

        for test_file in test_files:
            eval_config = CustomMetricsSupportAgentEvaluator.find_config_for_test_file(
                test_file
            )
            eval_set = CustomMetricsSupportAgentEvaluator._load_eval_set_from_file(
                test_file, eval_config, initial_session
            )

            await CustomMetricsSupportAgentEvaluator.evaluate_eval_set(
                agent_module=agent_module,
                eval_set=eval_set,
                eval_config=eval_config,
                num_runs=num_runs,
                agent_name=agent_name,
                print_detailed_results=print_detailed_results,
            )
