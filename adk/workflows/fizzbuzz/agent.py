import asyncio

from google.adk import Event, Workflow


def start_number() -> int:
    return 1


def is_multiple_of_15(node_input: int) -> Event:
    return Event(output=node_input, route=node_input % 15 == 0)


def is_multiple_of_3(node_input: int) -> Event:
    return Event(output=node_input, route=node_input % 3 == 0)


def is_multiple_of_5(node_input: int) -> Event:
    return Event(output=node_input, route=node_input % 5 == 0)


def emit_fizz_buzz(node_input: int) -> Event:
    return Event(message="FizzBuzz", output=node_input)


def emit_fizz(node_input: int) -> Event:
    return Event(message="Fizz", output=node_input)


def emit_buzz(node_input: int) -> Event:
    return Event(message="Buzz", output=node_input)


def emit_number(node_input: int) -> Event:
    return Event(message=str(node_input), output=node_input)


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
        ("START", start_number, is_multiple_of_15),
        (is_multiple_of_15, {True: emit_fizz_buzz, False: is_multiple_of_3}),
        (is_multiple_of_3, {True: emit_fizz, False: is_multiple_of_5}),
        (is_multiple_of_5, {True: emit_buzz, False: emit_number}),
        (emit_fizz_buzz, next_number),
        (emit_fizz, next_number),
        (emit_buzz, next_number),
        (emit_number, next_number),
        (next_number, {"continue": is_multiple_of_15, "done": finish}),
    ],
)
