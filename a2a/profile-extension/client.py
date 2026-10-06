"""Discover a typed skill and validate both sides of a real A2A request."""

import argparse
import asyncio
import json

from typing import Any
from uuid import uuid4

import httpx

from a2a.client.card_resolver import A2ACardResolver
from a2a.extensions.common import get_requested_extensions
from a2a.types import Role, SendMessageRequest, SendMessageResponse
from google.protobuf.json_format import MessageToDict, ParseDict

from contract import (
    EXTENSION_URI,
    SKILL_ID,
    data_message,
    load_schema,
    validate_parts,
    validator,
)


def discover_contract(card: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown required extensions and validate profile discovery data."""
    extensions = card.get('capabilities', {}).get('extensions', [])
    for extension in extensions:
        if extension.get('required') and extension['uri'] != EXTENSION_URI:
            raise ValueError('Unsupported required extension')
    extension = next(
        (entry for entry in extensions if entry['uri'] == EXTENSION_URI), None
    )
    if extension is None:
        raise ValueError('Typed contract extension is not advertised')
    params = extension.get('params', {})
    validator(load_schema('params.schema.json')).validate(params)
    skill_ids = {skill['id'] for skill in card.get('skills', [])}
    if set(params['contracts']) != skill_ids:
        raise ValueError('Contracts must cover exactly the advertised skills')
    contract = params['contracts'][SKILL_ID]
    validator(contract['inputSchema'])
    validator(contract['outputSchema'])
    return contract


async def invoke(
    http: httpx.AsyncClient, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Discover, prevalidate, activate, invoke and validate the response."""
    card_proto = await A2ACardResolver(http, base_url).get_agent_card()
    card = MessageToDict(card_proto)
    contract = discover_contract(card)
    interface = next(
        (
            item
            for item in card_proto.supported_interfaces
            if item.protocol_binding == 'JSONRPC'
            and item.protocol_version == '1.0'
        ),
        None,
    )
    if interface is None:
        raise ValueError('A2A 1.0 JSONRPC interface is required')
    message = data_message(payload, 'ROLE_USER')
    validate_parts(message, contract['inputSchema'], 'input')
    request = SendMessageRequest(message=message)
    request.configuration.accepted_output_modes.append('application/json')
    request_id = str(uuid4())
    # Use SDK models and protobuf JSON encoding; explicit HTTP preserves the
    # negotiated response header, which the SDK transport does not expose.
    response = await http.post(
        interface.url,
        headers={'A2A-Version': '1.0', 'A2A-Extensions': EXTENSION_URI},
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'method': 'SendMessage',
            'params': MessageToDict(request),
        },
    )
    response.raise_for_status()
    body = response.json()
    if body.get('jsonrpc') != '2.0' or body.get('id') != request_id:
        raise ValueError('Invalid JSON-RPC response envelope')
    if 'error' in body:
        raise ValueError(f'A2A error: {body["error"]}')
    active = get_requested_extensions(
        response.headers.get_list('A2A-Extensions')
    )
    if EXTENSION_URI not in active:
        raise ValueError('Server did not confirm typed contract activation')
    result = ParseDict(body['result'], SendMessageResponse())
    if result.WhichOneof('payload') != 'message':
        raise ValueError('This profile requires an immediate Message response')
    if result.message.role != Role.ROLE_AGENT:
        raise ValueError('Expected ROLE_AGENT')
    if MessageToDict(result.message.metadata).get(EXTENSION_URI) != {
        'skillId': SKILL_ID
    }:
        raise ValueError('Response skill does not match request')
    return validate_parts(result.message, contract['outputSchema'], 'output')


async def main(base_url: str, unit: str) -> None:
    """Invoke the sample agent and print its validated structured output."""
    async with httpx.AsyncClient(timeout=10) as http:
        result = await invoke(http, base_url, {'city': 'Tokyo', 'unit': unit})
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument(
        '--unit', choices=['celsius', 'fahrenheit'], default='celsius'
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.url, arguments.unit))
