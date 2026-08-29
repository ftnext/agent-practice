from google.adk import Event, Workflow


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


fizzbuzz_workflow = Workflow(
    name="fizzbuzz_logic",
    edges=[
        ("START", is_multiple_of_15),
        (is_multiple_of_15, {True: emit_fizz_buzz, False: is_multiple_of_3}),
        (is_multiple_of_3, {True: emit_fizz, False: is_multiple_of_5}),
        (is_multiple_of_5, {True: emit_buzz, False: emit_number}),
    ],
)
