import asyncio

from google.adk import Event, Workflow

from .fizzbuzz import fizzbuzz_workflow


def start_number() -> int:
    return 1


async def next_number(node_input: int) -> Event:
    if node_input < 40:
        await asyncio.sleep(1)
        return Event(output=node_input + 1, route="continue")
    return Event(output=node_input, route="done")


def finish(node_input: int) -> int:
    return node_input


root_agent = Workflow(
    name="FizzBuzz",
    edges=[
        ("START", start_number, fizzbuzz_workflow, next_number),
        (next_number, {"continue": fizzbuzz_workflow, "done": finish}),
    ],
)
