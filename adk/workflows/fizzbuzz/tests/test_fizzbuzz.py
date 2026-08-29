import pytest
from google.adk import Workflow
from google.adk.runners import InMemoryRunner
from google.genai import types

from fizzbuzz.agent import fizzbuzz_workflow


def parse_number(node_input: str) -> int:
    # User input number (str) -> int for workflow
    return int(node_input)


sut = Workflow(
    name="test_fizzbuzz_logic",
    edges=[("START", parse_number, fizzbuzz_workflow)],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "number,expected", [(1, "1"), (3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz")]
)
async def test_fizzbuzz_one_number(number, expected):
    user_id = "test_user"
    runner = InMemoryRunner(agent=sut)
    session = await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id=user_id
    )
    message = types.Content(role="user", parts=[types.Part(text=str(number))])

    events = [
        event
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        )
    ]
    messages = [
        part.text
        for event in events
        if event.content
        for part in event.content.parts
        if part.text
    ]

    assert messages == [expected]
