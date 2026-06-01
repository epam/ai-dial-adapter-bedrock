# Documentation Style

This guide covers how to write and maintain `README.md` and other user-facing documentation.

## When to Update the README

Update the README when a change could affect how a user configures or uses the adapter:

- Adding or removing a supported model → update the model table
- Adding, renaming, or removing an environment variable → update the env vars table
- Adding a new request field, configuration key, or endpoint → add an example
- Deprecating a feature → mark it with `> [!IMPORTANT]` and a "Since: X.Y.Z" note
- Changing a default value → update the table
- Changing behavior a user might rely on → add a note or callout

Internal refactors that don't change observable behavior don't require a README update.

## User-Centric Perspective

The README documents the **interface**, not the implementation.

**Don't expose:**
- Python class names (`BedrockChatCompletion`, `ConverseAdapterFactory`)
- Module paths (`llm/model/claude/`, `upstream_config.py`)
- Internal enum values — use the deployment name strings users actually configure
- Framework/library internals unless they surface as user-visible behavior

**Do document:**
- Deployment name strings (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`)
- Environment variable names and their defaults
- Request/response JSON fields
- Endpoint paths (`/openai/deployments/{id}/chat/completions`)
- DIAL Core configuration JSON
- Known limitations and unsupported features

The "Implementation" column in the model table (`Anthropic SDK / Converse API`) is an acceptable exception: it explains which configuration options apply to each model, which is user-visible.

## Model Table

Each supported model gets one row. Columns:
- **Deployment name**: the exact string users put in their DIAL Core config or request
- **Modality**: input→output in the `(type/type)-to-type` format
- **Feature support**: use ✅ / 🟡 / ❌ with a legend below the table

Always maintain the legend that explains ✅ / 🟡 / ❌ below any capability matrix.

When adding cross-region inference prefixes (`us.`, `eu.`, `apac.`) — document them under a dedicated section, not as separate rows.

## Configuration Examples

Show complete, copy-pasteable JSON. Partial snippets that omit required fields mislead users.

Use `${PLACEHOLDER_NAME}` for values the user must supply. Use `<details><summary>...</summary>` to collapse long examples that would clutter the flow:

```markdown
<details><summary>DIAL Core configuration</summary>

```json
{ ... }
```

</details>
```

When a configuration field accepts an enum, list the valid values in a table rather than prose.

## Callouts

| Callout | Use for |
|---|---|
| `> [!NOTE]` | Caveats, non-obvious constraints, pointers to official docs |
| `> [!IMPORTANT]` | Deprecations, breaking changes, required migration steps |

Don't use callouts for information that fits naturally in the surrounding prose.

## Environment Variables Table

Columns: Variable, Default, Description. Always include the default (`NA` if there is none, empty string if the default is an empty string).

Group dangerous/internal-detail variables in a separate table under a warning header rather than mixing them with the main table.

## Versioning and Deprecation

When a feature is introduced after the initial release, add `**Since:** X.Y.Z` on its own line directly below the section heading.

When deprecating, use `> [!IMPORTANT]` with:
1. The version it was deprecated in
2. What to use instead
3. Whether it will be removed

Do not silently remove documentation for deprecated features until the feature itself is removed.

## Links

Prefer linking to the official AWS or Anthropic documentation rather than re-documenting third-party behavior. This keeps the README accurate as those APIs evolve.

Cross-references within the README use anchor links (`[see Compatibility mode](#compatibility-mode)`).

## Limitations

Document limitations explicitly. If a feature has a known gap or edge case that would surprise a user, it belongs in the README — not just in a code comment. Numbered lists work well for related limitations.

## What to Avoid

- Prose that describes code structure instead of user behavior
- Examples that require the reader to fill in unstated required fields
- Re-documenting information already in the official vendor docs (link instead)
- Documenting private/internal-only endpoints or parameters
