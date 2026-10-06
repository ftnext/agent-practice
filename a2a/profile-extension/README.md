# A2A Typed Contract PoC

**実現可能。** A2AのProfile Extensionとして、Skill IDごとのJSON Schemaを
発見し、単一の構造化入力だけを受け取り、検証済みの構造化出力だけを返す
Agentを実装する。自然言語の解釈やLLM、実際の天気APIは使わない。

これは**独自ExtensionのPoC**であり、A2A coreにschemaフィールドを追加したものではない。

## 確認した仕様とSDK

調査日: **2026-10-06**。

- [公式Extension説明](https://a2a-protocol.org/latest/topics/extensions/#scope-of-extensions)
  はProfileによって許容Messageを狭める用途を認め、schema付きdata partを例示する。
- [公式AgentExtension定義](https://a2a-protocol.org/latest/specification/#444-agentextension)
  のフィールドは `uri` / `description` / `required` / `params`。
  `required: true` はclientが理解し遵守すべき要件の宣言。
  [標準の能力検証](https://a2a-protocol.org/latest/specification/#334-capability-validation)
  に従い必要なExtensionのsupportがないrequestには
  `ExtensionSupportRequiredError`を返す。
- [公式AgentSkill定義](https://a2a-protocol.org/latest/specification/#445-agentskill)
  に `inputSchema` / `outputSchema` はない。`inputModes` / `outputModes`は
  MIME typeの宣言であり、JSONの形の契約を表さない。
- [公式Part定義](https://a2a-protocol.org/latest/specification/#416-part)
  は `text` / `raw` / `url` / `data` のoneof。MessageとArtifactはいずれも
  `parts`を持つ。現行Python SDKはprotobuf `Part.data`（JSON Value）を使用する。
  旧0.3系の `DataPart(kind="data", data=...)` とwire形式もAPIも異なる。
- [公式activation説明](https://a2a-protocol.org/latest/topics/extensions/#extension-activation)
  に基づきrequestの `A2A-Extensions` で有効化を要求する。
  `Message.extensions`は内容の注釈であり、このヘッダーの代用ではない。
- A2A coreはExtensionのschema languageをJSON Schemaに固定しない。
  [Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core)
  を要求するのは**今回の独自Profile**。
- [公式SDK latest release](https://github.com/a2aproject/a2a-python/releases/tag/v1.2.1)
  は調査時点の公開latest表示ではv1.2.1。
  最初のPoCで検証したSDK checkout:
  `a0160025d6d211b0be1762cb829e7a47c147191a`,
  `a2a-sdk==1.2.2.post4.dev0+a016002`、wire protocol **1.0**。
  SDK版とwire protocol版は別。公開v1.2.1での動作までは未検証。
  この共有版は上記の公式Git commitを依存として固定し、ローカルSDK checkoutを必要としない。

### SDKの支援とapplication layer

[SDK context builder](https://github.com/a2aproject/a2a-python/blob/a0160025d6d211b0be1762cb829e7a47c147191a/src/a2a/server/routes/common.py)
はヘッダーを `ServerCallContext.requested_extensions` に変換する。
`RequestContext.requested_extensions`でも参照できる。
`find_extension_by_uri`、client側の `with_a2a_extensions` /
`ClientCallContext.service_parameters` も利用可能。

[JSON-RPC dispatcher](https://github.com/a2aproject/a2a-python/blob/a0160025d6d211b0be1762cb829e7a47c147191a/src/a2a/server/routes/jsonrpc_dispatcher.py)
はSDK protobufモデルをparseしhandlerを呼び、標準errorをJSON-RPCに変換する。
SDKにはcore必須フィールドの `validate_request_params` とMIME input mode検証があるが、
今回のJSON Schema契約の自動検証・per-skill schema discoveryは提供しない。
そこでcontext builderでactivationを、handlerで入力を、executorで公開前の出力を
検証する。response activation headerはapplication middlewareで追加する。

ClientはSDKの `A2ACardResolver` とprotobuf型を使用する。呼び出し部分はhttpxで
`SendMessage`を送る。SDKのJsonRpcTransportが応答ヘッダーを公開しないため、
activation確認を含めた最小clientではHTTP responseを直接扱う。

## 実行

このリポジトリをcloneした後、`a2a/profile-extension/` ディレクトリから
実行する（Python 3.10以上、[uv](https://docs.astral.sh/uv/)が必要）:

```bash
cd a2a/profile-extension
uv sync --locked
uv run python server.py
```

別terminal（同じディレクトリで）:

```bash
uv run python client.py
# {"temperature": 24.3, "condition": "cloudy"}
uv run python client.py --unit fahrenheit
# {"temperature": 75.74, "condition": "cloudy"}
```

依存と開発ツールは `pyproject.toml`、解決済みのバージョンは `uv.lock` に保存する。
SDKは公式Git repositoryの検証済みcommitに固定する。初回setupにはGitとネット接続が必要。
API keyやローカルSDK checkoutは不要。天気情報は固定値で返す。

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

このディレクトリはリポジトリ直下のMIT Licenseに従う。
依存するA2A Python SDK自体のlicenseはApache-2.0。

## Design by Contract

```text
precondition:
  明示的なExtension activationとSkill選択
  input.parts == [data Part]
  validate(input.parts[0].data, InputSchema)

postcondition:
  output.parts == [data Part]
  validate(output.parts[0].data, OutputSchema)
```

client送信前とserver実行前で入力を検証し、serverのevent queue公開前と
client受信後で出力を検証する。TextPart、必須field不足、enum違反、追加fieldを
拒否する。出力が `{"temperature":"24.3"}` のように壊れていたら、
その値を正常レスポンスとして公開せずerrorを返す。

## Agent Card例

完全なCardは `GET /.well-known/agent-card.json` で取得できる。
保存した例は [agent-card.example.json](agent-card.example.json)。
以下は契約に関係する部分の抜粋（`…`は説明用省略、JSONとして送信しない）:

```json
{
  "supportedInterfaces": [{
    "url": "http://127.0.0.1:8000/rpc",
    "protocolBinding": "JSONRPC", "protocolVersion": "1.0"
  }],
  "capabilities": {
    "extensions": [{
      "uri": "https://example.com/a2a/extensions/typed-contract/v1",
      "required": true,
      "params": {
        "schemaLanguage": "https://json-schema.org/draft/2020-12/schema",
        "contracts": {
          "weather-skill": {
            "contractVersion": "1.0.0",
            "inputSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "…": "…"},
            "outputSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "…": "…"}
          }
        }
      }
    }]
  },
  "skills": [{
    "id": "weather-skill", "name": "Weather",
    "description": "Fixed weather", "tags": ["weather"],
    "inputModes": ["application/json"], "outputModes": ["application/json"]
  }]
}
```

`params.contracts`とmetadataの `skillId` は今回新設したルール。
Agent単位のdeclarationにSkill単位の契約をmapする。
`required: true` はAgent全体の依存なので、自然言語Skillと混在させる際には
endpoint分離等を考える。schemaの発見方式比較は[設計メモ](design-notes.md)を参照。

## Request / response

実際に実行できるrequest:

```bash
curl -s http://127.0.0.1:8000/rpc \
  -H 'Content-Type: application/json' \
  -H 'A2A-Version: 1.0' \
  -H 'A2A-Extensions: https://example.com/a2a/extensions/typed-contract/v1' \
  -d '{"jsonrpc":"2.0","id":"1","method":"SendMessage","params":{"message":{"messageId":"input-1","role":"ROLE_USER","parts":[{"data":{"city":"Tokyo","unit":"celsius"}}],"metadata":{"https://example.com/a2a/extensions/typed-contract/v1":{"skillId":"weather-skill"}}}}}'
```

成功response（messageIdは毎回生成）:

```json
{
  "jsonrpc": "2.0", "id": "1",
  "result": {
    "message": {
      "messageId": "generated-uuid", "role": "ROLE_AGENT",
      "parts": [{"data": {"temperature": 24.3, "condition": "cloudy"}, "mediaType": "application/json"}],
      "metadata": {"https://example.com/a2a/extensions/typed-contract/v1": {"skillId": "weather-skill"}},
      "extensions": ["https://example.com/a2a/extensions/typed-contract/v1"]
    }
  }
}
```

応答にも `A2A-Extensions` が付く。

## Error semantics

[標準error mapping](https://a2a-protocol.org/latest/specification/#54-error-code-mappings)
を利用し、独自数値codeは追加しない:

| ケース | JSON-RPC error |
| --- | --- |
| Text / 部分形状 / Input Schema / Skill選択違反 | InvalidParamsError, -32602 |
| 不正なAgent出力 | InvalidAgentResponseError, -32006 |
| required extension未advertise / 別版のみ / 非対応client | ExtensionSupportRequiredError, -32008 |
| streaming要求 | UnsupportedOperationError, -32004 |

ここで「requestでadvertise」はservice parameterによるsupport/activationの宣言を指す。
HTTP 200でも `error` を含むJSON-RPCは失敗。例えばmissing unitは:

```json
{
  "jsonrpc": "2.0", "id": "1",
  "error": {
    "code": -32602,
    "message": "Typed contract schema validation failed",
    "data": [{
      "@type": "type.googleapis.com/google.rpc.ErrorInfo",
      "reason": "INVALID_PARAMS", "domain": "a2a-protocol.org",
      "metadata": {
        "extension": "https://example.com/a2a/extensions/typed-contract/v1",
        "phase": "input", "violation": "schema", "skillId": "weather-skill",
        "instancePath": "", "keyword": "required"
      }
    }]
  }
}
```

エラーのmetadataはExtension定義。値そのものは返さない。非同期Taskを使うなら
accepted前のrequest errorとaccepted後のfailed Taskを分けるが、このPoCは
即時Messageのみ。[規範仕様](typed-contract-extension.md)に詳しいルールを記載。

## 検証・制約

共有版の独立環境（Python 3.14.6）でも **28テストは成功**。
Ruff lint/format・tyによる型検証も成功。localhostでserver/clientを実行して
摂氏24.3・華氏75.74の応答も確認した。SDKの本物のJSON-RPC dispatcherとexecutorをASGI HTTPで通す。
正常系・華氏、指定の5失敗例、required activation不足、別版、Message.extensionsだけの
activation、混在/複数/空parts、未知Skill、client validation、悪いserver出力、
streaming拒否を検証する。output failureのfixtureはserver構築時のproducer差し替えで
作る（requestのhidden flagで不正出力に切り替えない）。

このv1はinline schema、ローカル参照、Draft 2020-12、即時Message、JSON-RPC 1.0が対象。
外部schema URL、REST/gRPC、非同期Task/Artifact、streaming、複数Skillの実際のAgent、
他言語SDKとの相互運用は未実装。これはlocal HTTP用のPoCで、天気は固定値。
formatはannotation、protobuf数値はdouble。Schema検証は意味の正しさや実際の気象を
保証しない。型契約を理解するclient双方でのみ相互運用できる。

## 最後の問いへの回答

**A2A coreは現在、OpenAPI/function callingのようなSkillごとの
`inputSchema` / `outputSchema` を表現できるか？ → No。**

**Profile Extensionでどこまで型付きAgentを実現できるか？**
このPoCのように、Agent CardのparamsでSkillごとの契約を発見し、metadataで明示選択し、
標準SendMessageで構造化JSONを運び、両側検証と標準errorによって
「schema付きtyped RPC endpoint」として扱える。ただしschemaの言語、対応づけ、
routing、validation、streamingの扱いはExtensionの合意事項。
A2A core準拠だけで全clientが自動対応する保証はない。
