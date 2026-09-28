# TODO

Follow-up work deliberately left out of the `aidial-adapter-anthropic` 0.18.0
bump, so that the bump stays a migration and nothing else.

## 1. Dead code to remove

Nothing in the repository references these:

| Module / symbol | Note |
| --- | --- |
| `aidial_adapter_bedrock/llm/lazy_stage.py` | The whole module. `LazyStage` has no callers. |
| `aidial_adapter_bedrock/llm/chat_model.py` — `TextCompletionAdapter` | No implementations left. |
| `aidial_adapter_bedrock/utils/json.py` — `json_dumps` | Only `json_dumps_short` and `remove_nones` are used. |
| `aidial_adapter_bedrock/llm/message.py` — `parse_dial_message`, `HumanToolResultMessage`, `HumanFunctionResultMessage`, `AIToolCallMessage`, `AIFunctionCallMessage` | Referenced only from inside the module. |

Exercised by the unit tests only — decide whether to keep the tests or drop
both:

- `aidial_adapter_bedrock/llm/chat_model.py` — `trivial_partitioner`,
  `keep_last_and_system_messages`
- `aidial_adapter_bedrock/llm/truncate_prompt.py` —
  `compute_discarded_messages`, `_partition_indexer`

## 2. Local forks of upstream modules

Each of these is a copy of something the anthropic adapter now ships. They
were kept local because upstream made its copies private; if those are ever
exported, the forks can go.

| Local | Upstream |
| --- | --- |
| `llm/message.py` | `aidial_adapter_anthropic.dial._message` |
| `llm/truncate_prompt.py` | `aidial_adapter_anthropic.adapter._truncate_prompt` |
| `llm/tokenize.py` | `aidial_adapter_anthropic.adapter._tokenize` |
| `llm/lazy_stage.py` | `aidial_adapter_anthropic.dial._lazy_stage` |
| `llm/decorator/{base,preprocess_messages,replicator}.py` | `aidial_adapter_anthropic.adapter._decorator.{base,preprocess,replicator}` — the upstream copies are identical. `llm/decorator/caching.py` stays, it is Converse-specific. |
| `llm/chat_model.py` — `default_preprocess_messages` | `aidial_adapter_anthropic.adapter._base.default_preprocess_messages` |
| `utils/list.py`, `utils/list_projection.py` | `aidial_adapter_anthropic._utils.list` |
| `utils/json.py` | `aidial_adapter_anthropic._utils.json` |

## 3. Imports of private upstream modules

`AdapterRequest.messages` is typed with `ListProjection[AdapterMessage]`, and
both classes live in private modules, so there is no public way to name or
build that value. The imports are confined to:

- `aidial_adapter_bedrock/llm/chat_model.py` — the `AdapterMessages` alias
- `tests/utils/messages.py` — `to_adapter_messages`, used to build a request
  in tests

Ask upstream to export `AdapterMessage`, `ListProjection` and
`parse_dial_message`, then drop the private imports.

## 4. Double parsing of the request messages

`AdapterRequest.create` parses the DIAL messages into `AdapterMessage`, and
`to_dial_messages` (`llm/chat_model.py`) unwraps them straight back so that
the Bedrock adapters can keep working with raw DIAL messages. Converting the
Converse pipeline to `AdapterMessage` would remove the round-trip.
