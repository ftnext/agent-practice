"""A real SDK JSON-RPC server enforcing the profile before event publication."""

from collections.abc import Callable
from typing import Any

import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.request_handler import validate_request_params
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Message, Role, SendMessageRequest, Task
from a2a.utils.errors import (
    ExtensionSupportRequiredError,
    InvalidParamsError,
    UnsupportedOperationError,
)
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from contract import (
    EXTENSION_URI,
    SKILL_ID,
    data_message,
    detail,
    make_card,
    parameters,
    validate_parts,
)


WeatherFunction = Callable[[dict[str, Any]], dict[str, Any]]


def weather(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic fixture in the requested unit."""
    temperature = 24.3
    if payload['unit'] == 'fahrenheit':
        temperature = round(temperature * 9 / 5 + 32, 2)
    return {'temperature': temperature, 'condition': 'cloudy'}


class ContractContextBuilder(DefaultServerCallContextBuilder):
    """Enforce required activation for all parsed RPC operations."""

    def build(self, request: Request) -> ServerCallContext:
        """Parse SDK service parameters and reject absent extension support."""
        context = super().build(request)
        if EXTENSION_URI not in context.requested_extensions:
            raise ExtensionSupportRequiredError(
                data=detail('activation', 'extension-required')
            )
        request.state.typed_contract_active = True
        return context


class WeatherExecutor(AgentExecutor):
    """Validate output before the SDK can persist or publish it."""

    def __init__(self, produce: WeatherFunction = weather) -> None:
        self.produce = produce

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Emit only one complete, validated Message."""
        contract = parameters()['contracts'][SKILL_ID]
        if context.message is None:
            raise InvalidParamsError('Message is required')
        payload = validate_parts(
            context.message, contract['inputSchema'], 'input'
        )
        response = data_message(self.produce(payload), 'ROLE_AGENT')
        validate_parts(response, contract['outputSchema'], 'output')
        await event_queue.enqueue_event(response)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Reject cancellation of this immediate message-only operation."""
        raise UnsupportedOperationError('No cancellable weather task')


class ContractHandler(DefaultRequestHandler):
    """Reject invalid inputs before the SDK starts agent execution."""

    @validate_request_params
    async def on_message_send(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> Message | Task:
        """Check skill routing and preconditions before the default handler."""
        metadata = MessageToDict(params.message.metadata)
        selection = metadata.get(EXTENSION_URI)
        if selection != {'skillId': SKILL_ID}:
            raise InvalidParamsError(
                'Unknown or missing typed skill selection',
                data=detail('input', 'skill-selection'),
            )
        if params.message.role != Role.ROLE_USER:
            raise InvalidParamsError('Expected ROLE_USER')
        if params.message.task_id or params.message.reference_task_ids:
            raise InvalidParamsError('This profile is message-only')
        if params.configuration.task_push_notification_config.ByteSize():
            raise UnsupportedOperationError('Push notifications unsupported')
        modes = params.configuration.accepted_output_modes
        if modes and 'application/json' not in modes:
            raise InvalidParamsError('Client must accept application/json')
        contract = parameters()['contracts'][SKILL_ID]
        validate_parts(params.message, contract['inputSchema'], 'input')
        return await super().on_message_send(params, context)


def create_app(
    base_url: str = 'http://127.0.0.1:8000',
    produce: WeatherFunction = weather,
) -> Starlette:
    """Create an isolated sample application; production SDK code is unchanged."""
    card = make_card(base_url)
    handler = ContractHandler(
        agent_executor=WeatherExecutor(produce),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(
                handler, '/rpc', context_builder=ContractContextBuilder()
            ),
        ]
    )

    async def activated_header(
        request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)
        if getattr(request.state, 'typed_contract_active', False):
            response.headers['A2A-Extensions'] = EXTENSION_URI
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=activated_header)
    return app


if __name__ == '__main__':
    uvicorn.run(create_app(), host='127.0.0.1', port=8000)
