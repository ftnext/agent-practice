"""Exercise the real SDK dispatcher, executor and client over ASGI HTTP."""

import json

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from a2a.utils.errors import InvalidAgentResponseError, InvalidParamsError
from google.protobuf.json_format import MessageToDict
from jsonschema import ValidationError

from client import discover_contract, invoke
from contract import DIALECT, EXTENSION_URI, SKILL_ID, make_card, validator
from server import create_app


BASE = 'http://testserver'
GOOD = {'city': 'Tokyo', 'unit': 'celsius'}
HEADERS = {'A2A-Version': '1.0', 'A2A-Extensions': EXTENSION_URI}


@pytest_asyncio.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(BASE)), base_url=BASE
    ) as session:
        yield session


def request_body(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a protocol 1.0 envelope; only extension metadata is custom."""
    return {
        'jsonrpc': '2.0',
        'id': 'test',
        'method': 'SendMessage',
        'params': {
            'message': {
                'messageId': 'input',
                'role': 'ROLE_USER',
                'parts': parts,
                'metadata': {EXTENSION_URI: {'skillId': SKILL_ID}},
            }
        },
    }


@pytest.mark.asyncio
async def test_success(http: httpx.AsyncClient) -> None:
    result = await invoke(http, BASE, GOOD)
    assert result == {'temperature': 24.3, 'condition': 'cloudy'}
    assert await invoke(http, BASE, {**GOOD, 'unit': 'fahrenheit'}) == {
        'temperature': 75.74,
        'condition': 'cloudy',
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'parts',
    [
        [{'text': "What's the weather in Tokyo?"}],
        [{'data': {'city': 'Tokyo'}}],
        [{'data': {**GOOD, 'unit': 'kelvin'}}],
        [{'data': {**GOOD, 'foo': 'bar'}}],
        [],
        [{'data': GOOD}, {'data': GOOD}],
        [{'data': GOOD}, {'text': 'extra'}],
        [{'url': 'https://example.com/file'}],
        [{'data': []}],
        [{'data': None}],
    ],
)
async def test_server_rejects_contract_violations(
    http: httpx.AsyncClient, parts: list[dict[str, Any]]
) -> None:
    response = await http.post(
        '/rpc', json=request_body(parts), headers=HEADERS
    )
    body = response.json()
    assert body['error']['code'] == -32602
    assert 'result' not in body
    assert response.headers['A2A-Extensions'] == EXTENSION_URI


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'headers',
    [
        {'A2A-Version': '1.0'},
        {
            'A2A-Version': '1.0',
            'A2A-Extensions': EXTENSION_URI.replace('/v1', '/v2'),
        },
        {'A2A-Version': '1.0', 'A2A-Extensions': 'https://example.com/other'},
    ],
)
async def test_required_activation(
    http: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    body = request_body([{'data': GOOD}])
    # Message.extensions annotates content; it is not service-level activation.
    body['params']['message']['extensions'] = [EXTENSION_URI]
    response = await http.post('/rpc', json=body, headers=headers)
    assert response.json()['error']['code'] == -32008
    assert 'A2A-Extensions' not in response.headers


@pytest.mark.asyncio
async def test_comma_separated_activation(http: httpx.AsyncClient) -> None:
    response = await http.post(
        '/rpc',
        json=request_body([{'data': GOOD}]),
        headers={
            **HEADERS,
            'A2A-Extensions': f'https://example.com/other, {EXTENSION_URI}',
        },
    )
    assert 'result' in response.json()
    assert response.headers['A2A-Extensions'] == EXTENSION_URI


@pytest.mark.asyncio
async def test_output_validation_failure() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_app(BASE, produce=lambda _: {'temperature': '24.3'})
        ),
        base_url=BASE,
    ) as http:
        response = await http.post(
            '/rpc', json=request_body([{'data': GOOD}]), headers=HEADERS
        )
        body = response.json()
        assert body['error']['code'] == -32006
        assert 'result' not in body
        assert '24.3' not in response.text
        assert 'output' in response.text


@pytest.mark.asyncio
async def test_precondition_prevents_execution() -> None:
    calls: list[dict[str, Any]] = []

    def produce(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {'temperature': 24.3, 'condition': 'cloudy'}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(BASE, produce=produce)),
        base_url=BASE,
    ) as http:
        await http.post(
            '/rpc',
            json=request_body([{'data': {'city': 'Tokyo'}}]),
            headers=HEADERS,
        )
        await http.post('/rpc', json=request_body([{'data': GOOD}]))
    assert calls == []


@pytest.mark.asyncio
async def test_unknown_skill(http: httpx.AsyncClient) -> None:
    body = request_body([{'data': GOOD}])
    body['params']['message']['metadata'][EXTENSION_URI]['skillId'] = 'other'
    response = await http.post('/rpc', json=body, headers=HEADERS)
    assert response.json()['error']['code'] == -32602


@pytest.mark.asyncio
async def test_client_prevalidates(http: httpx.AsyncClient) -> None:
    requests: list[str] = []

    async def record(request: httpx.Request) -> None:
        requests.append(request.method)

    http.event_hooks['request'].append(record)
    with pytest.raises(InvalidParamsError):
        await invoke(http, BASE, {'city': 'Tokyo', 'unit': 'kelvin'})
    assert requests == ['GET']


@pytest.mark.parametrize(
    'change', ['missing', 'unknown-required', 'dialect', 'skill']
)
def test_client_rejects_incompatible_cards(change: str) -> None:
    card = MessageToDict(make_card(BASE))
    extension = card['capabilities']['extensions'][0]
    if change == 'missing':
        card['capabilities']['extensions'] = []
    elif change == 'unknown-required':
        extension['uri'] = 'https://example.com/unknown'
    elif change == 'dialect':
        extension['params']['schemaLanguage'] = 'unknown'
    else:
        card['skills'] = []
    with pytest.raises((ValueError, ValidationError)):
        discover_contract(card)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_output', [True, False])
async def test_client_distrusts_invalid_server(bad_output: bool) -> None:
    def remote(request: httpx.Request) -> httpx.Response:
        if request.method == 'GET':
            return httpx.Response(200, json=MessageToDict(make_card(BASE)))
        body = json.loads(request.content)
        headers = {'A2A-Extensions': EXTENSION_URI} if bad_output else {}
        return httpx.Response(
            200,
            headers=headers,
            json={
                'jsonrpc': '2.0',
                'id': body['id'],
                'result': {
                    'message': {
                        'messageId': 'output',
                        'role': 'ROLE_AGENT',
                        'parts': [
                            {
                                'data': {'temperature': '24.3'}
                                if bad_output
                                else {
                                    'temperature': 24.3,
                                    'condition': 'cloudy',
                                }
                            }
                        ],
                        'metadata': {EXTENSION_URI: {'skillId': SKILL_ID}},
                    }
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as http:
        with pytest.raises(
            InvalidAgentResponseError if bad_output else ValueError
        ):
            await invoke(http, BASE, GOOD)


@pytest.mark.asyncio
async def test_streaming_disabled(http: httpx.AsyncClient) -> None:
    body = request_body([{'data': GOOD}])
    body['method'] = 'SendStreamingMessage'
    response = await http.post('/rpc', json=body, headers=HEADERS)
    assert response.json()['error']['code'] == -32004


def test_local_schema_reference() -> None:
    schema = {
        '$schema': DIALECT,
        'type': 'object',
        '$defs': {'name': {'type': 'string'}},
        'properties': {'city': {'$ref': '#/$defs/name'}},
    }
    validator(schema).validate({'city': 'Tokyo'})
    with pytest.raises(ValidationError):
        validator(schema).validate({'city': 123})


def test_remote_schema_reference_rejected_at_discovery() -> None:
    schema = {
        '$schema': DIALECT,
        'type': 'object',
        'properties': {'city': {'$ref': 'https://example.com/schema.json'}},
    }
    with pytest.raises(ValueError, match='in-document'):
        validator(schema)
