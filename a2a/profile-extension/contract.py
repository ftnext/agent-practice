"""Application-layer implementation of the typed contract profile."""

import json

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from a2a.types import AgentCard, Message
from a2a.utils.errors import InvalidAgentResponseError, InvalidParamsError
from google.protobuf.json_format import MessageToDict, ParseDict
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


if TYPE_CHECKING:
    from jsonschema.protocols import Validator


EXTENSION_URI = 'https://example.com/a2a/extensions/typed-contract/v1'
DIALECT = 'https://json-schema.org/draft/2020-12/schema'
SKILL_ID = 'weather-skill'
SCHEMA_DIR = Path(__file__).parent / 'schemas'


def load_schema(name: str) -> dict[str, Any]:
    """Read a bundled schema without network access."""
    return json.loads((SCHEMA_DIR / name).read_text())


def no_remote_reference(uri: str) -> Any:
    """Reject unbundled references rather than implicitly fetching URLs."""
    raise ValueError(f'External schema reference is unsupported: {uri}')


def validator(schema: dict[str, Any]) -> 'Validator':
    """Check a schema and create a validator with offline reference resolution."""
    if schema.get('$schema') != DIALECT:
        raise ValueError('Schema dialect must be Draft 2020-12')
    Draft202012Validator.check_schema(schema)
    resources = [
        Resource.from_contents(schema, default_specification=DRAFT202012)
    ]
    while resources:
        resource = resources.pop()
        content = resource.contents
        if isinstance(content, dict):
            if content.get('$schema', DIALECT) != DIALECT:
                raise ValueError('Mixed schema dialects are unsupported')
            for keyword in ('$ref', '$dynamicRef'):
                if keyword in content and not content[keyword].startswith('#'):
                    raise ValueError(
                        'Only in-document schema references are supported'
                    )
        resources.extend(resource.subresources())
    return Draft202012Validator(
        schema, registry=Registry(retrieve=no_remote_reference)
    )


def parameters() -> dict[str, Any]:
    """Return inline, independently versioned per-skill contracts."""
    return {
        'schemaLanguage': DIALECT,
        'contracts': {
            SKILL_ID: {
                'contractVersion': '1.0.0',
                'inputSchema': load_schema('weather-input.schema.json'),
                'outputSchema': load_schema('weather-output.schema.json'),
            }
        },
    }


def make_card(base_url: str) -> AgentCard:
    """Build an A2A 1.0 Agent Card using real SDK protobuf fields."""
    params = parameters()
    validator(load_schema('params.schema.json')).validate(params)
    for contract in params['contracts'].values():
        validator(contract['inputSchema'])
        validator(contract['outputSchema'])
    return ParseDict(
        {
            'name': 'Typed Weather',
            'description': 'Fixed weather data with validated JSON contracts.',
            'version': '1.0.0',
            'supportedInterfaces': [
                {
                    'url': f'{base_url.rstrip("/")}/rpc',
                    'protocolBinding': 'JSONRPC',
                    'protocolVersion': '1.0',
                }
            ],
            'capabilities': {
                'streaming': False,
                'extensions': [
                    {
                        'uri': EXTENSION_URI,
                        'description': 'Exactly one schema-validated data part.',
                        'required': True,
                        'params': params,
                    }
                ],
            },
            'defaultInputModes': ['application/json'],
            'defaultOutputModes': ['application/json'],
            'skills': [
                {
                    'id': SKILL_ID,
                    'name': 'Weather',
                    'description': 'Returns fixture weather, not live weather.',
                    'tags': ['weather', 'typed-contract'],
                    'inputModes': ['application/json'],
                    'outputModes': ['application/json'],
                }
            ],
        },
        AgentCard(),
    )


def detail(phase: str, reason: str, skill_id: str = SKILL_ID) -> dict[str, str]:
    """Create extension-specific string metadata for google.rpc.ErrorInfo."""
    return {
        'extension': EXTENSION_URI,
        'phase': phase,
        'violation': reason,
        'skillId': skill_id,
    }


def validate_parts(
    message: Message, schema: dict[str, Any], phase: str
) -> dict[str, Any]:
    """Validate the profile's part shape and schema, without leaking values."""
    error_type = (
        InvalidParamsError if phase == 'input' else InvalidAgentResponseError
    )
    if (
        len(message.parts) != 1
        or message.parts[0].WhichOneof('content') != 'data'
    ):
        raise error_type(
            'Exactly one data part is required',
            data=detail(phase, 'part-shape'),
        )
    payload = MessageToDict(message.parts[0].data)
    errors = list(validator(schema).iter_errors(payload))
    if errors:
        # JSON Pointer and keyword are safe diagnostic identifiers; values and
        # validator messages can contain secrets and are intentionally omitted.
        violation = errors[0]
        path = (
            '/'
            + '/'.join(
                str(item).replace('~', '~0').replace('/', '~1')
                for item in violation.absolute_path
            )
            if violation.absolute_path
            else ''
        )
        raise error_type(
            'Typed contract schema validation failed',
            data={
                **detail(phase, 'schema'),
                'instancePath': path,
                'keyword': str(violation.validator),
            },
        )
    return payload


def data_message(payload: dict[str, Any], role: str) -> Message:
    """Construct an A2A message containing a single structured data part."""
    return ParseDict(
        {
            'messageId': str(uuid4()),
            'role': role,
            'parts': [{'data': payload, 'mediaType': 'application/json'}],
            'metadata': {EXTENSION_URI: {'skillId': SKILL_ID}},
            'extensions': [EXTENSION_URI],
        },
        Message(),
    )
