# Typed Contract Profile Extension v1

Status: experimental, locally implemented proposal; **not an official A2A extension**.
Identifier: `https://example.com/a2a/extensions/typed-contract/v1`.
Binding baseline: A2A **1.0 JSON-RPC**, immediate Message-only `SendMessage`.
The identifier is a placeholder and is not a schema download endpoint.
MUST, MUST NOT, SHOULD and MAY are normative within **this extension**.

## 1. Declaration and discovery

An implementing Agent MUST advertise this URI in
`AgentCard.capabilities.extensions`. This object uses only core
`AgentExtension` fields (`uri`, `description`, `required`, `params`).
For an Agent that exclusively accepts typed invocations, `required` MUST
be true. Core required-extension semantics remain applicable.

`params` MUST validate against `schemas/params.schema.json`.
`schemaLanguage` MUST be `https://json-schema.org/draft/2020-12/schema`.
`contracts` MUST map exact, case-sensitive `AgentSkill.id` values to
`contractVersion`, `inputSchema`, and `outputSchema`. Every key MUST name
an advertised skill. Every callable skill on this required-profile Agent
MUST have a contract. A v1 implementation MUST support inline schemas.

Each schema MUST declare the dialect above and MUST itself be valid under
that dialect's metaschema. The params schema checks configuration shape;
implementations MUST additionally check each embedded schema. Schemas MUST
be objects in this profile; boolean schemas are intentionally excluded.
Input/output modes SHOULD advertise `application/json`; these MIME declarations
are neither schemas nor validation rules.

This v1 profile permits in-document `$ref` only. Implementations MUST NOT
silently resolve remote references. Embedded `$id` and URI references MUST NOT
cause network downloads. Unsupported schemas MUST cause configuration/discovery
failure, never be accepted with validation skipped. Schema evaluation MUST NOT
coerce types, insert defaults or drop unknown properties. `format` is annotation
only in v1; format assertion would require an explicitly agreed later profile.

## 2. Activation

Clients MUST understand this specification before requesting its URI using
`A2A-Extensions` (comma-separated extension identifiers) and MUST use
`A2A-Version: 1.0`. Servers MUST check service-level activation before invoking
any typed skill. An absent URI, another version URI, or only
`Message.extensions` MUST NOT count as activation.

For a syntactically parsed RPC lacking required support, servers MUST return
`ExtensionSupportRequiredError` (`-32008`). Core parsing/version errors MAY take
precedence. Unrelated unsupported optional URIs MAY be ignored. Servers MUST NOT
substitute another profile version. A successfully activated response MUST
include this URI in `A2A-Extensions`; this profile strengthens the core/topic
SHOULD to MUST. Clients MUST reject a success without that confirmation.
Activation means agreement to the profile, not successful schema validation.

## 3. Skill selection and precondition

A request Message MUST contain the following **extension-defined** metadata:

```json
{
  "https://example.com/a2a/extensions/typed-contract/v1": {
    "skillId": "weather-skill"
  }
}
```

The object at that metadata key MUST have exactly `skillId`. Clients MUST
select an advertised contract. Missing/unknown selections MUST be rejected;
servers MUST NOT infer the skill from text or input shape. Metadata outside this
namespace MAY coexist. Multiple skills use the same routing rule and independent
schemas; the sample implements only `weather-skill`.

The invocation MUST be a new, immediate message-only call: no `taskId` or
`referenceTaskIds`, no push notification configuration. The input role MUST be
`ROLE_USER`. If accepted output modes are provided, they MUST include
`application/json`.

```text
precondition(request, selectedContract):
    request.message.parts contains exactly one Part
    request.message.parts[0].content is data
    validate(request.message.parts[0].data, selectedContract.inputSchema)
```

Text, file/url/raw parts, mixed parts, zero parts and multiple data parts MUST
be rejected. Input validation MUST finish before business execution. Part
`mediaType` MAY be omitted; the data alternative and schema, not the MIME label,
determine acceptance. `Message.extensions` SHOULD list the profile URI to
annotate contributed content; it does not replace activation.

## 4. Postcondition and validation responsibility

A success MUST be `SendMessageResponse.message` with role `ROLE_AGENT`, matching
skill selection metadata, exactly one data Part, and data satisfying the selected
output schema. Successful Task results are outside this version of the profile.

```text
postcondition(response, selectedContract):
    response.message.parts contains exactly one Part
    response.message.parts[0].content is data
    validate(response.message.parts[0].data, selectedContract.outputSchema)
```

Servers MUST validate the completed output **before enqueueing, publishing or
persisting it**. Merely validating at HTTP serialization is insufficient if task
storage, push notification or another consumer already observed it. Clients MUST
validate input before sending and validate output before using it; server
validation MUST NOT trust that client-side validation happened. Schema violations
MUST NOT be encoded as successful business output.

## 5. Errors

| Failure | Core error / JSON-RPC code | Extension phase |
| --- | --- | --- |
| Required activation absent | ExtensionSupportRequiredError / -32008 | activation |
| Skill selection, part shape, input schema invalid | InvalidParamsError / -32602 | input |
| Agent-generated output schema or shape invalid | InvalidAgentResponseError / -32006 | output |
| Streaming / unsupported lifecycle | UnsupportedOperationError / -32004 | n/a |

Core codes MUST retain their standard meanings. Implementations MUST NOT invent
new core fields or enum values. This SDK serializes exception `data` as string
metadata in a `google.rpc.ErrorInfo` entry within JSON-RPC `error.data`.
The extension defines `extension`, `phase`, `violation`, `skillId`, and, for schema
failures, `instancePath` (JSON Pointer) and `keyword` diagnostic metadata.
Implementations SHOULD expose safe paths/keywords rather than raw payloads or
validator exception text. The response MUST contain `error`, never a normal
`result`, on contract failure. HTTP 200 with a JSON-RPC error is not business
success.

No Task is accepted/published in this profile, so an RPC error is appropriate.
A future asynchronous variant MAY use a failed Task after task acceptance, with
extension-specific failure metadata and no invalid Artifact. It MUST explicitly
define the lifecycle and persistence boundaries. It MUST NOT claim the current
v1 profile while returning Task results.

## 6. Streaming and versioning

Agents conforming to v1 MUST advertise `streaming: false` and reject streaming
invocations. In a future profile, servers could buffer a full JSON value,
validate it, then emit one complete data artifact; clients would still validate
the completed value. Emitting partial JSON before final validation cannot satisfy
v1's no-invalid-output guarantee. Patch/chunk streams need their own schema,
assembly rules, ordering, failure handling and final commit marker; a final
validation alone cannot retract previously emitted invalid content.

A breaking change to this specification MUST use a new extension URI. A breaking
change to a skill's input/output contract MUST change `contractVersion` and Agent
Card version; it SHOULD also use a new skill ID or endpoint when old clients must
continue working. Clients MUST rediscover contracts when using a changed Card,
rather than reuse a stale cached schema. Production URI owners SHOULD publish this
specification durably; a URI identifier alone does not provide a schema resolver.
