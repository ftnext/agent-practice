# 設計メモ

## Core・Extension・実装の境界

| 層 | このPoCが利用・追加するもの |
| --- | --- |
| A2A core | Agent Card、AgentSkill.id、AgentExtension.params、Message.metadata、Part.data、SendMessage、標準error、HTTP service parameters |
| 今回の独自Profile | contracts map、contractVersion、Draft 2020-12固定、skillId metadata、単一data part限定、両側検証、エラー診断、即時Message限定 |
| Pythonアプリ層 | jsonschema、ContractContextBuilder、ContractHandler、WeatherExecutor、応答ヘッダーmiddleware |

CoreのSkillは能力の説明であり、Skillを直接選択するtyped RPC selectorも
inputSchema/outputSchemaもない。したがってmetadataによる選択は**新設の仕様**。
ExtensionはAgentCapabilities単位で宣言する。contractsのキーをSkill IDに
対応させると複数Skillを表現できるが、個別Skillだけrequiredにするcore機構ではない。
このAgentではすべての呼び出しをtypedにする。自然言語Skillと混在したい場合は
別Agent Card/endpointへの分離が分かりやすい。optional extensionにして
対象Skillだけ強制する場合は、その条件を別途規範化する必要がある。

## Schema発見方式

| 方式 | discovery | versioning / cache | Cardサイズ | offline |
| --- | --- | --- | --- | --- |
| paramsにinline（採用） | Cardだけで完結 | Card版と契約版を同期、CardのETagで再取得 | 大きいschemaに不向き | Cardを保存すれば可能 |
| paramsにschema URL | 明示URLから取得 | 不変URL、digest、ETag、TTL等を設計する必要 | 小さい | schemaを別途保存 |
| Extension specから発見 | specにresolverを定義しない限り不可 | spec版とAgentごとのschema版は別 | 小さい | specとschemaを保存 |
| Extension URI + Skill ID | 共通URL規約を新設する必要 | 同じSkill名でもAgentごとにschemaが違うためAgent identityも必要 | 小さい | 解決結果を保存 |

公式仕様はinlineとURLのいずれかを標準として要求していない。Extension URIを
GETできることもschema取得できることも前提にしない。小さいPoCにはinlineが
実装も再現も簡単。外部URLは将来版で、HTTPS、許可origin、サイズ・時間制限、
不変schema ID、digest検証、参照閉包の保存を定義したうえで追加したい。
この実装は外部参照を自動取得しない。

params自身のschemaはconfigurationの形だけを検証する。embedded schemaが
正しいことはDraft202012Validator.check_schemaで別途確認する。v1はformatを
annotationとして扱う。additionalProperties:false、required、enum、数値型などは
両側でassertionとして検証する。Protobuf Valueのnumberはdoubleなので、巨大整数の
厳密精度が必要な契約には文字列等の表現を別に決める必要がある。

## 検証の責務とエラー

ClientはCard discovery後の入力検証・受信後の出力検証を担当する。
Serverは起動する実装のschemaを管理し、受信時・実行前と、結果をqueueへ渡す前に
検証する。clientだけの検証では悪意のある/非対応clientを止められず、serverだけでは
不正なremote serverをclientが信頼してしまう。テストはclientを迂回したHTTPも使う。

入力違反はInvalidParamsError、出力違反はInvalidAgentResponseError、activation不足は
ExtensionSupportRequiredErrorに対応づけるのがこの即時応答方式には自然。
Extension固有の数値error codeは作らない。SDKの標準ErrorInfoにstring metadataを
追加して原因を区別する。Task化した場合はaccepted後に失敗する意味を区別して
failed Taskも検討するが、このv1と混同しない。

## Streaming・相互運用・標準化への希望

このv1はstreamingを扱わない。完成JSONのbuffer→validation→公開という方式なら
完全値契約を維持できる。tokenやpatchを先に公開すると最終validationで失敗しても
受信済みデータを撤回できない。structured streamingには別のcommit/assembly契約が必要。

A2Aに準拠するだけのclientはこのExtensionを理解しない。相互運用できるのは、
同じURIの仕様、schema dialect、routing、error semanticsを実装する双方の間。
このPoCではPython内の実HTTP dispatcherと両側検証まで確認する。他言語SDKとの
相互運用、REST/gRPC、外部schema、非同期Taskの適合性までは主張しない。

将来coreにあると便利なのはSkill単位のinput/output schema discovery、明示Skill selector、
schema version/digest、SDKの入出力validator middleware、全transport共通のactivation
結果API、structured streamingの組み立て/commitルール。これは提案であり現行APIではない。
