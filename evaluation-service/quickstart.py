# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-cloud-aiplatform[evaluation]>=1.156.0",
# ]
# ///
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview#get-started
import pandas as pd
from vertexai import Client, types

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
print(eval_result)
