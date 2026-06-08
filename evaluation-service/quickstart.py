# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-cloud-aiplatform[evaluation]>=1.156.0",
# ]
# ///
import json
from pathlib import Path

import pandas as pd
from vertexai import Client, types
from vertexai._genai import _evals_visualization as evals_visualization


def save_eval_result_html(eval_result, output_path="eval_result.html"):
    result_dump = eval_result.model_dump(
        mode="json", exclude_none=True, exclude={"evaluation_dataset"}
    )
    input_dataset_list = eval_result.evaluation_dataset
    metadata_payload = result_dump.get("metadata", {})
    is_comparison = input_dataset_list and len(input_dataset_list) > 1

    if is_comparison and input_dataset_list:
        if input_dataset_list[0]:
            metadata_payload["dataset"] = evals_visualization._extract_dataset_rows(
                input_dataset_list[0]
            )

        if "eval_case_results" in result_dump:
            for case_result in result_dump["eval_case_results"]:
                for response_index, candidate_result in enumerate(
                    case_result.get("response_candidate_results", [])
                ):
                    if response_index >= len(input_dataset_list):
                        continue

                    rows = evals_visualization._extract_dataset_rows(
                        input_dataset_list[response_index]
                    )
                    case_index = case_result.get("eval_case_index")
                    if case_index is None or case_index >= len(rows):
                        continue

                    original_case = rows[case_index]
                    candidate_result["display_text"] = original_case[
                        "response_display_text"
                    ]
                    candidate_result["raw_json"] = original_case["response_raw_json"]

        win_rates = eval_result.win_rates if eval_result.win_rates else {}
        if "summary_metrics" in result_dump:
            for summary in result_dump["summary_metrics"]:
                if summary.get("metric_name") in win_rates:
                    summary.update(win_rates[summary["metric_name"]])

        result_dump["metadata"] = metadata_payload
        html = evals_visualization.get_comparison_html(json.dumps(result_dump))
    else:
        single_dataset = input_dataset_list[0] if input_dataset_list else None
        if single_dataset is not None:
            rows = evals_visualization._extract_dataset_rows(single_dataset)
            metadata_payload["dataset"] = rows

            if "eval_case_results" in result_dump and rows:
                for case_result in result_dump["eval_case_results"]:
                    case_index = case_result.get("eval_case_index")
                    if (
                        case_index is None
                        or case_index >= len(rows)
                        or not case_result.get("response_candidate_results")
                    ):
                        continue

                    original_case = rows[case_index]
                    candidate_result = case_result["response_candidate_results"][0]
                    candidate_result["display_text"] = original_case[
                        "response_display_text"
                    ]
                    candidate_result["raw_json"] = original_case["response_raw_json"]

        result_dump["metadata"] = metadata_payload
        html = evals_visualization.get_evaluation_html(json.dumps(result_dump))

    output = Path(output_path)
    output.write_text(html, encoding="utf-8")
    return output


def main():
    client = Client(project="adk-practice-480404", location="us-central1")
    eval_dataset_df = pd.DataFrame(
        {
            "prompt": [
                "Write a four-sentence summary of the provided article about renewable energy, maintaining an optimistic tone."
            ],
            "response": [
                "The article highlights significant growth in solar and wind power. "
                "These advancements are making clean energy more affordable. "
                "The future looks bright for renewables. "
                "However, the report also notes challenges with grid infrastructure."
            ],
        }
    )
    eval_result = client.evals.evaluate(
        dataset=eval_dataset_df,
        metrics=[types.RubricMetric.GENERAL_QUALITY],
    )

    if evals_visualization._is_ipython_env():
        eval_result.show()
    else:
        print(eval_result.summary_metrics)
        html_path = save_eval_result_html(eval_result)
        print(f"Saved evaluation report: {html_path.resolve()}")

    return eval_result


if __name__ == "__main__":
    main()
