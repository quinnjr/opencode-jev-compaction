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
3. Build a skeletal view of the conversation and send it to Jev as the
   `state`. Tool results are replaced by a one-line summary
   (`call Read input={...} -> ok NNNN chars`) — the result bodies are **not**
   sent. Abridged user, assistant, and reasoning text (head 600 + tail 300
   characters per part) plus up to 200 characters of each tool input *are*
   sent, unless `JEV_STATE_INCLUDE_TEXT=0` restricts the state to tool
   metadata only. If the state still exceeds `maxStateTokens`, long lines are
   cut further (head 120 + tail 60); if it cannot be fitted, compaction is
   skipped for that turn. See [Data sent to Jev](#data-sent-to-jev).
4. For every non-pinned tool call, ask two `noul` questions in one batched
   request: *should the call stay* and *should the result stay verbatim*.
5. Decide against `JEV_KEEP_THRESHOLD` (default 0.5):
   - `keep_result ≥ threshold` → keep the call and result untouched;
   - else `keep_call ≥ threshold` → keep the call, truncate the result to its
     first `JEV_TRUNCATE_HEAD_CHARS` characters plus a one-line note (any
     attachments are dropped);
   - else → remove the call together with its result.
   An in-flight (`pending`/`running`) call is never removed — only completed or
   errored calls are candidates for truncation or removal.
6. A message left with no parts is dropped.

Failures never break a session. A missing key, an unfittable history, or an
error before batching leaves every message untouched; when only some batches
fail, those batches' calls are kept while the other batches' decisions still
apply. Failures and per-batch problems are logged as warnings; routine skips
(under threshold, no candidates, state too large, no request budget) are logged
at debug level, so set `JEV_COMPACTION_DEBUG=1` to see why nothing happened.

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
| `JEV_BASE_URL` | `https://api.typesafe.ai/v1/systemone` | System One endpoint. Must be `https://` (or `http://` on loopback); anything else is refused so the bearer token cannot be redirected. |
| `JEV_TIMEOUT_MS` | `10000` | Abort a Jev request after this many milliseconds so a stalled call cannot block generation. |
| `JEV_STATE_INCLUDE_TEXT` | `1` | Set to `0` to send only tool metadata as the state, omitting abridged conversation text. |
| `JEV_MAX_CONCURRENT` | `4` | Maximum Jev requests in flight at once. |
| `JEV_COMPACTION_DEBUG` | `0` | Set to `1` for debug logging. |

## Data sent to Jev

When compaction runs, the plugin sends the following to the configured
`JEV_BASE_URL` (TypeSafe by default):

- **Sent:** the role and index of every message; abridged user/assistant/
  reasoning text; file names/URLs; every tool's name, a summarised input, and
  its status/length (`ok 4213 chars`). Tool inputs also appear (up to 300
  characters) in the per-call question instructions, regardless of
  `JEV_STATE_INCLUDE_TEXT`.
- **Not sent:** tool result bodies, file contents, or the raw transcript.

Tool inputs can contain commands, paths, or source snippets, and abridged text
can contain anything pasted into the chat. If that is not acceptable for your
codebase, set `JEV_STATE_INCLUDE_TEXT=0` (tool metadata only) or disable the
plugin with `JEV_COMPACTION_DISABLED=1`.

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
