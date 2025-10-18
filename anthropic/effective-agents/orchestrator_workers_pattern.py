# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic",
# ]
# ///

# https://github.com/anthropics/claude-cookbooks/blob/3ddc4e0a0de45a0a255dd7bb54ecc0918cae7547/patterns/agents/util.py
# https://github.com/anthropics/claude-cookbooks/blob/3ddc4e0a0de45a0a255dd7bb54ecc0918cae7547/patterns/agents/orchestrator_workers.ipynb

import re
from typing import TypedDict

from anthropic import Anthropic


def llm_call(prompt: str, model: str, system_prompt: str = "") -> str:
    client = Anthropic()
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        temperature=0.1,
    )
    return response.content[0].text


def extract_xml(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


class TaskInfo(TypedDict):
    type: str
    description: str


def parse_tasks(tasks_xml: str) -> list[TaskInfo]:
    tasks = []

    for line in tasks_xml.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[13:-14].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)

    return tasks


class ExecutedTask(TaskInfo):
    result: str


class Result(TypedDict):
    analysis: str
    tasks: list[ExecutedTask]


class FlexibleOrchestrator:
    def __init__(self, orchestrator_prompt: str, worker_prompt: str):
        self.orchestrator_prompt = orchestrator_prompt
        self.worker_prompt = worker_prompt

    @staticmethod
    def _format_prompt(template: str, **kwargs) -> str:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}")

    def process(self, task: str, context: str | None = None) -> Result:
        context = context or {}

        orchestrator_input = self._format_prompt(
            self.orchestrator_prompt, task=task, **context
        )
        print("=== ORCHESTRATOR INPUT ===")
        print(orchestrator_input)
        print()

        orchestrator_response = llm_call(
            orchestrator_input, "claude-sonnet-4-5-20250929"
        )

        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print("=== ORCHESTRATOR OUTPUT ===")
        print()
        print("ANALYSIS:")
        print(analysis)
        print()
        print("TASKS:")
        print(tasks)
        print()

        worker_results = []
        for task_info in tasks:
            worker_input = self._format_prompt(
                self.worker_prompt,
                original_task=task,
                task_type=task_info["type"],
                task_description=task_info["description"],
                **context,
            )
            print("=== WORKER INPUT ===")
            print(worker_input)
            print()

            worker_response = llm_call(worker_input, "claude-haiku-4-5-20251001")
            result = extract_xml(worker_response, "response")

            worker_results.append(
                {
                    "type": task_info["type"],
                    "description": task_info["description"],
                    "result": result,
                }
            )

            print(f"=== WORKER RESULT ({task_info['type']}) ===")
            print(result)
            print()

        return {"analysis": analysis, "tasks": worker_results}


ORCHESTRATOR_PROMPT = """
Analyze this task and break it down into 2-3 distinct approaches:

Task: {task}

Return your response in this format:

<analysis>
Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.
</analysis>

<tasks>
    <task>
        <type>formal</type>
        <description>Write a precise, technical version that emphasizes specifications</description>
    </task>
    <task>
        <type>conversational</type>
        <description>Write an engaging, friendly version that connects with readers</description>
    </task>
</tasks>
"""

WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return your response in this format:

<response>
Your content here, maintaining the specified style and fully addressing requirements.
</response>
"""

orchestrator = FlexibleOrchestrator(ORCHESTRATOR_PROMPT, WORKER_PROMPT)
results = orchestrator.process(
    task="Write a product description for a new eco-friendly water bottle",
    context={
        "target_audience": "environmentally conscious millennials",
        "key_features": ["plastic-free", "insulated", "lifetime warranty"],
    },
)
