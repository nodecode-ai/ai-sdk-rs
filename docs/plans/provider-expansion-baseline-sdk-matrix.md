# Provider SDK Baseline Matrix (bead `ai-sdk-rs-am3`)

Generated: 2026-02-14T04:42:12Z (UTC)

Catalog source: `/home/mike/.cache/nodecode/providers/catalog.json`

## Snapshot Summary

- Total providers in catalog: **123**
- Unique `sdk_type` labels in catalog: **7**
- Catalog labels unsupported by current `SdkType` enum: **none**
- Catalog provider IDs with direct `ProviderRegistration.id` match: **6 / 123**
- Direct matches with aligned `sdk_type`: **5 / 6**
- Direct matches with mismatched `sdk_type`: **1 / 6**
- Catalog provider IDs without direct registration match: **117 / 123**

Direct ID matches:

`amazon-bedrock,anthropic,azure,google,google-vertex,openai`

Direct ID matches with sdk_type mismatch:

`azure(catalog=openai-compatible,registry=azure)`

| sdk_type | catalog providers | direct registry id matches | unresolved id mappings |
| --- | ---: | ---: | ---: |
| amazon-bedrock | 5 | 1 | 4 |
| anthropic | 3 | 1 | 2 |
| google | 1 | 1 | 0 |
| google-vertex | 1 | 1 | 0 |
| groq | 1 | 0 | 1 |
| openai | 1 | 1 | 0 |
| openai-compatible | 111 | 1 | 110 |

## Integration Point Status

| Integration point | Baseline status | Evidence |
| --- | --- | --- |
| `ip-upstream-catalog-normalization` | no immediate label mismatch in snapshot | Catalog emits 7 canonical labels (all parseable by current `SdkType` enum). |
| `ip-openai-compatible-provider-id-routing` | gap | 110 catalog providers use `sdk_type=openai-compatible` with no direct `ProviderRegistration.id` entry; plus `azure` is a direct match but with mismatched registry sdk_type. |
| `ip-groq-routing` | gap | Catalog contains `groq -> sdk_type=groq`; no current registration entry handles `SdkType::Groq`. |

## Unresolved Dispatch Gaps (file-level targets)

| Gap | Why unresolved in current snapshot | Target files for follow-up bead(s) |
| --- | --- | --- |
| OpenAI-compatible alias ID routing | Registry only contains one canonical openai-compatible ID while catalog has 110 additional alias IDs in that compatibility class. | `crates/provider/src/lib.rs` (`sdk_type_from_id` behavior), `crates/providers/openai-compatible/src/provider.rs` (alias registration/matching), `crates/providers/openai-compatible/tests/provider_registry_tests.rs` (alias coverage tests). |
| Azure classification drift | Catalog marks `azure` as `openai-compatible` while registry maps `azure` to `SdkType::Azure`; this creates mixed semantics in ID-vs-sdk routing paths. | `crates/providers/azure/src/provider.rs`, `crates/provider/src/lib.rs`, and catalog normalization source in gateway appendix tooling (outside this repo). |
| Groq sdk_type routing | `SdkType::Groq` exists in enum and catalog but lacks a matching provider registration/builder path. | `crates/providers/openai-compatible/src/provider.rs` (explicit `Groq` matcher/registration or dedicated Groq registration), `crates/providers/openai-compatible/tests/provider_registry_tests.rs`, optional `crates/provider/src/lib.rs` alias expectations. |

## Provider ID to sdk_type Matrix

| provider_id | catalog_sdk_type | registry_sdk_type_if_direct_match | sdk_type_alignment | direct_registry_id_match |
| --- | --- | --- | --- | --- |
| 302ai | openai-compatible | n/a | n/a | no |
| abacus | openai-compatible | n/a | n/a | no |
| aihubmix | openai-compatible | n/a | n/a | no |
| alibaba | openai-compatible | n/a | n/a | no |
| alibaba-cn | openai-compatible | n/a | n/a | no |
| amazon-bedrock | amazon-bedrock | amazon-bedrock | aligned | yes |
| amazon-nova | amazon-bedrock | n/a | n/a | no |
| anthropic | anthropic | anthropic | aligned | yes |
| anyscale | openai-compatible | n/a | n/a | no |
| azure | openai-compatible | azure | mismatch | yes |
| azure-ai | openai-compatible | n/a | n/a | no |
| azure-cognitive-services | openai-compatible | n/a | n/a | no |
| bailing | openai-compatible | n/a | n/a | no |
| baseten | openai-compatible | n/a | n/a | no |
| bedrock | amazon-bedrock | n/a | n/a | no |
| bedrock-converse | amazon-bedrock | n/a | n/a | no |
| berget | openai-compatible | n/a | n/a | no |
| cerebras | openai-compatible | n/a | n/a | no |
| chutes | openai-compatible | n/a | n/a | no |
| cloudflare-ai-gateway | openai-compatible | n/a | n/a | no |
| cloudflare-workers-ai | openai-compatible | n/a | n/a | no |
| cohere | openai-compatible | n/a | n/a | no |
| cohere-chat | openai-compatible | n/a | n/a | no |
| cortecs | openai-compatible | n/a | n/a | no |
| dashscope | openai-compatible | n/a | n/a | no |
| databricks | openai-compatible | n/a | n/a | no |
| deepinfra | openai-compatible | n/a | n/a | no |
| deepseek | openai-compatible | n/a | n/a | no |
| fastrouter | openai-compatible | n/a | n/a | no |
| fireworks-ai | openai-compatible | n/a | n/a | no |
| firmware | openai-compatible | n/a | n/a | no |
| friendli | openai-compatible | n/a | n/a | no |
| friendliai | openai-compatible | n/a | n/a | no |
| gemini | openai-compatible | n/a | n/a | no |
| gigachat | openai-compatible | n/a | n/a | no |
| github-copilot | openai-compatible | n/a | n/a | no |
| github-models | openai-compatible | n/a | n/a | no |
| gitlab | openai-compatible | n/a | n/a | no |
| gmi | openai-compatible | n/a | n/a | no |
| google | google | google | aligned | yes |
| google-vertex | google-vertex | google-vertex | aligned | yes |
| google-vertex-anthropic | anthropic | n/a | n/a | no |
| groq | groq | n/a | n/a | no |
| helicone | openai-compatible | n/a | n/a | no |
| heroku | openai-compatible | n/a | n/a | no |
| huggingface | openai-compatible | n/a | n/a | no |
| hyperbolic | openai-compatible | n/a | n/a | no |
| iflowcn | openai-compatible | n/a | n/a | no |
| io-net | openai-compatible | n/a | n/a | no |
| jiekou | openai-compatible | n/a | n/a | no |
| kimi-for-coding | openai-compatible | n/a | n/a | no |
| lambda-ai | openai-compatible | n/a | n/a | no |
| lemonade | openai-compatible | n/a | n/a | no |
| llamagate | openai-compatible | n/a | n/a | no |
| lmstudio | openai-compatible | n/a | n/a | no |
| lucidquery | openai-compatible | n/a | n/a | no |
| meta-llama | openai-compatible | n/a | n/a | no |
| minimax | openai-compatible | n/a | n/a | no |
| minimax-cn | openai-compatible | n/a | n/a | no |
| minimax-cn-coding-plan | openai-compatible | n/a | n/a | no |
| minimax-coding-plan | openai-compatible | n/a | n/a | no |
| mistral | openai-compatible | n/a | n/a | no |
| moark | openai-compatible | n/a | n/a | no |
| modelscope | openai-compatible | n/a | n/a | no |
| moonshot | openai-compatible | n/a | n/a | no |
| moonshotai | openai-compatible | n/a | n/a | no |
| moonshotai-cn | openai-compatible | n/a | n/a | no |
| nano-gpt | openai-compatible | n/a | n/a | no |
| nebius | openai-compatible | n/a | n/a | no |
| nova | amazon-bedrock | n/a | n/a | no |
| novita | openai-compatible | n/a | n/a | no |
| novita-ai | openai-compatible | n/a | n/a | no |
| nvidia | openai-compatible | n/a | n/a | no |
| oci | openai-compatible | n/a | n/a | no |
| ollama | openai-compatible | n/a | n/a | no |
| ollama-cloud | openai-compatible | n/a | n/a | no |
| openai | openai | openai | aligned | yes |
| opencode | openai-compatible | n/a | n/a | no |
| openrouter | openai-compatible | n/a | n/a | no |
| ovhcloud | openai-compatible | n/a | n/a | no |
| perplexity | openai-compatible | n/a | n/a | no |
| poe | openai-compatible | n/a | n/a | no |
| privatemode-ai | openai-compatible | n/a | n/a | no |
| publicai | openai-compatible | n/a | n/a | no |
| replicate | openai-compatible | n/a | n/a | no |
| requesty | openai-compatible | n/a | n/a | no |
| sambanova | openai-compatible | n/a | n/a | no |
| sap-ai-core | openai-compatible | n/a | n/a | no |
| scaleway | openai-compatible | n/a | n/a | no |
| siliconflow | openai-compatible | n/a | n/a | no |
| siliconflow-cn | openai-compatible | n/a | n/a | no |
| submodel | openai-compatible | n/a | n/a | no |
| synthetic | openai-compatible | n/a | n/a | no |
| together-ai | openai-compatible | n/a | n/a | no |
| togetherai | openai-compatible | n/a | n/a | no |
| upstage | openai-compatible | n/a | n/a | no |
| v0 | openai-compatible | n/a | n/a | no |
| venice | openai-compatible | n/a | n/a | no |
| vercel | openai-compatible | n/a | n/a | no |
| vercel-ai-gateway | openai-compatible | n/a | n/a | no |
| vertex-ai | openai-compatible | n/a | n/a | no |
| vertex-ai-anthropic-models | anthropic | n/a | n/a | no |
| vertex-ai-deepseek-models | openai-compatible | n/a | n/a | no |
| vertex-ai-language-models | openai-compatible | n/a | n/a | no |
| vertex-ai-llama-models | openai-compatible | n/a | n/a | no |
| vertex-ai-minimax-models | openai-compatible | n/a | n/a | no |
| vertex-ai-mistral-models | openai-compatible | n/a | n/a | no |
| vertex-ai-moonshot-models | openai-compatible | n/a | n/a | no |
| vertex-ai-qwen-models | openai-compatible | n/a | n/a | no |
| vertex-ai-vision-models | openai-compatible | n/a | n/a | no |
| vertex-ai-zai-models | openai-compatible | n/a | n/a | no |
| vivgrid | openai-compatible | n/a | n/a | no |
| volcengine | openai-compatible | n/a | n/a | no |
| vultr | openai-compatible | n/a | n/a | no |
| wandb | openai-compatible | n/a | n/a | no |
| watsonx | openai-compatible | n/a | n/a | no |
| xai | openai-compatible | n/a | n/a | no |
| xiaomi | openai-compatible | n/a | n/a | no |
| zai | openai-compatible | n/a | n/a | no |
| zai-coding-plan | openai-compatible | n/a | n/a | no |
| zenmux | openai-compatible | n/a | n/a | no |
| zhipuai | openai-compatible | n/a | n/a | no |
| zhipuai-coding-plan | openai-compatible | n/a | n/a | no |
