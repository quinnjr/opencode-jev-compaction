# opencode-jev-compaction

Jev-powered context compaction for [opencode](https://opencode.ai).

opencode already summarizes old turns when a session nears the context limit,
but that summary is produced by an LLM and is lossy. This plugin instead runs
before **every** LLM request and asks [Jev](https://docs.typesafe.ai) — TypeSafe's
System One decision model — whether each old tool call and its result are still
worth sending.

Nothing is rewritten. Jev only decides what to *drop* or *truncate*; every tool
call and result it keeps is sent verbatim. User and assistant text is never
touched.

## How it works

The plugin hooks `experimental.chat.messages.transform`, which opencode calls
before each request to the model.

1. Estimate the context size. If it is below `JEV_COMPACTION_THRESHOLD`
   (default 15k tokens), do nothing.
2. Pin the first message and the newest `JEV_PRESERVE_RECENT` (default 6)
   messages. Pinned content is never a candidate.
3. Build a skeletal view of the conversation — tool outputs replaced by
   `ok, NNNN chars (omitted)`, long text abridged — and send it to Jev as the
   `state`.
4. For every non-pinned tool call, ask two `noul` questions in one batched
   request: *should the call stay* and *should the result stay verbatim*.
5. Decide against `JEV_KEEP_THRESHOLD` (default 0.5):
   - `keep_result ≥ threshold` → keep the call and result untouched;
   - else `keep_call ≥ threshold` → keep the call, truncate the result to its
     first `JEV_TRUNCATE_HEAD_CHARS` characters plus a one-line note;
   - else → remove the call together with its result.
6. A message left with no parts is dropped.

Any failure — missing key, network error, unparseable response, or a history
that cannot be fitted into the state budget — leaves the messages exactly as
they were and logs a warning. A session is never broken because compaction
failed.

## Requirements

- opencode with TypeScript plugin support (`@opencode-ai/plugin` 1.3.x).
- A TypeSafe API key: <https://console.typesafe.ai/keys>.

## Install

Copy `jev-compaction.ts` into either plugin directory:

```bash
# global (all projects)
cp jev-compaction.ts ~/.config/opencode/plugins/

# or project-local
cp jev-compaction.ts .opencode/plugins/
```

Then export your key (opencode inherits the shell environment):

```bash
export TYPESAFE_API_KEY=...
```

Restart opencode. Set `JEV_COMPACTION_DEBUG=1` to have the plugin log when it
loads and what it keeps, truncates, or removes.

## Configuration

All configuration is environment-based.

| Variable | Default | Description |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | — | Required. TypeSafe API key. |
| `JEV_COMPACTION_DISABLED` | `0` | Set to `1` to disable the plugin. |
| `JEV_COMPACTION_THRESHOLD` | `15000` | Estimated context tokens below which pruning is skipped. |
| `JEV_PRESERVE_RECENT` | `6` | Newest messages never touched (the first is always pinned). |
| `JEV_KEEP_THRESHOLD` | `0.5` | Minimum Jev probability for a call or result to stay. |
| `JEV_TRUNCATE_HEAD_CHARS` | `300` | Characters of a dropped tool result kept as a preview. |
| `JEV_MODEL` | `jev-latest` | Jev model name. |
| `JEV_BASE_URL` | `https://api.typesafe.ai/v1/systemone` | System One endpoint. |
| `JEV_COMPACTION_DEBUG` | `0` | Set to `1` for debug logging. |

## Caveats

- `experimental.chat.messages.transform` is an experimental opencode hook. If a
  future opencode release stops calling it, the plugin silently does nothing.
- Token sizes are estimated from character counts, not a tokenizer. The
  threshold is a gate, not an accounting.
- A probability is not a proof that a result is safe to delete. The assistant
  can always re-run a tool; a truncated or dropped result is not gone from the
  session, only from what is sent to the model.
- The full state is resent with every batch of questions, so a history near the
  state ceiling costs one Jev request per handful of questions.

## Development

```bash
bun install
bun test          # unit tests with a fake Jev transport
bun run typecheck # tsc --noEmit
```

The unit tests never contact TypeSafe.

## License

MIT — see [LICENSE](LICENSE).
