// Jev context compaction for opencode.
//
// opencode already summarizes old turns when a session nears the context limit,
// but that summary is produced by an LLM and is lossy. This plugin instead runs
// before *every* LLM request (via the experimental.chat.messages.transform hook)
// and asks Jev — TypeSafe's decision model — whether each old tool call and its
// result are still worth sending. Everything kept stays verbatim; only calls
// and results Jev says are stale are dropped or truncated.
//
// Jev is not an LLM: it evaluates typed questions against a state and returns
// probabilities. One batched request asks two noul questions per tool call.
//
// Failures are always non-fatal: if Jev is unreachable, unconfigured, slow, or
// the history cannot be fitted, the messages are left exactly as they were.

import type { Plugin, PluginModule } from "@opencode-ai/plugin"
import type { Message, Part, ToolPart } from "@opencode-ai/sdk"

export type Msg = { info: Message; parts: Part[] }

export type JevQuestion = { type: "noul"; instructions: string }
export type JevAnswer = { noul?: number }
/** Only the subset of the System One response this plugin consumes. */
export type JevResponse = { answers?: Record<string, JevAnswer> }

export type JevAsker = (state: string, questions: Record<string, JevQuestion>) => Promise<JevResponse>

export type CompactionOptions = {
  enabled: boolean
  /** Estimated context tokens below which pruning is skipped entirely. */
  threshold: number
  /** Newest messages that are never candidates (the first is always pinned). */
  preserveRecent: number
  /** Minimum Jev probability for a call or result to stay. */
  keepThreshold: number
  /** Characters of a dropped tool result retained as a head preview. */
  truncateHeadChars: number
  /** Estimated ceiling for the state sent to Jev. */
  maxStateTokens: number
  /** Estimated ceiling for state plus one batch of questions (Jev caps at 32k). */
  maxRequestTokens: number
  /** Milliseconds before a Jev request is aborted. Defaults to 10000. */
  timeoutMs?: number
  /** Send abridged conversation text (not just tool metadata) as part of the state. Defaults to true. */
  sendText?: boolean
  /** Maximum Jev requests in flight at once. Defaults to 4. */
  maxConcurrentRequests?: number
  apiKey?: string
  model: string
  baseUrl: string
  debug: boolean
  ask?: JevAsker
  log?: (level: "debug" | "info" | "warn" | "error", message: string, extra?: Record<string, unknown>) => void
}

export const DEFAULT_OPTIONS: Omit<Required<CompactionOptions>, "ask" | "log" | "apiKey"> = {
  enabled: true,
  threshold: 15_000,
  preserveRecent: 6,
  keepThreshold: 0.5,
  truncateHeadChars: 300,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  timeoutMs: 10_000,
  sendText: true,
  maxConcurrentRequests: 4,
  model: "jev-latest",
  baseUrl: "https://api.typesafe.ai/v1/systemone",
  debug: false,
}

// Internal state/prompt sizing. These shape how the state is fitted into the Jev
// request budget; unlike the options above they are not operator-facing knobs.
const STATE_TEXT_HEAD = 600
const STATE_TEXT_TAIL = 300
const TOOL_INPUT_STATE_CHARS = 200
const TOOL_INPUT_QUESTION_CHARS = 300
const TOOL_NAME_MAX_CHARS = 80
const STATE_LINE_MIN_ABRIDGE = 400
const STATE_LINE_HEAD = 120
const STATE_LINE_TAIL = 60
const QUESTION_OVERHEAD_TOKENS = 120

export function optionsFromEnv(
  env: Record<string, string | undefined> = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env ?? {},
): CompactionOptions {
  const num = (v: string | undefined, fallback: number) => {
    if (v === undefined || v.trim() === "") return fallback
    const n = Number(v)
    return Number.isFinite(n) ? n : fallback
  }
  const atLeast = (v: string | undefined, fallback: number, min: number) => Math.max(min, num(v, fallback))
  return {
    ...DEFAULT_OPTIONS,
    apiKey: env.TYPESAFE_API_KEY,
    enabled: env.JEV_COMPACTION_DISABLED !== "1",
    threshold: atLeast(env.JEV_COMPACTION_THRESHOLD, DEFAULT_OPTIONS.threshold, 0),
    preserveRecent: atLeast(env.JEV_PRESERVE_RECENT, DEFAULT_OPTIONS.preserveRecent, 0),
    keepThreshold: Math.min(1, atLeast(env.JEV_KEEP_THRESHOLD, DEFAULT_OPTIONS.keepThreshold, 0)),
    truncateHeadChars: atLeast(env.JEV_TRUNCATE_HEAD_CHARS, DEFAULT_OPTIONS.truncateHeadChars, 0),
    timeoutMs: atLeast(env.JEV_TIMEOUT_MS, DEFAULT_OPTIONS.timeoutMs, 1),
    sendText: env.JEV_STATE_INCLUDE_TEXT !== "0",
    maxConcurrentRequests: atLeast(env.JEV_MAX_CONCURRENT, DEFAULT_OPTIONS.maxConcurrentRequests, 1),
    model: env.JEV_MODEL || DEFAULT_OPTIONS.model,
    baseUrl: env.JEV_BASE_URL || DEFAULT_OPTIONS.baseUrl,
    debug: env.JEV_COMPACTION_DEBUG === "1",
  }
}

const isToolPart = (part: Part): part is ToolPart => part.type === "tool"

function summarize(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max)}…`
}

/** Collapse whitespace and clamp, so a hostile tool name cannot forge state lines. */
function oneLine(text: string, max = TOOL_NAME_MAX_CHARS): string {
  return text.replace(/\s+/g, " ").slice(0, max)
}

function abridge(text: string, head: number, tail: number): string {
  if (text.length <= head + tail) return text
  return `${text.slice(0, head)}\n[… ${text.length - head - tail} chars omitted …]\n${text.slice(-tail)}`
}

function toolPayload(part: ToolPart): string {
  switch (part.state.status) {
    case "completed":
      return part.state.output
    case "error":
      return errorPayload(part)
    default:
      return ""
  }
}

/** The text opencode actually sends for an errored part (an interrupted tool's metadata.output). */
function errorPayload(part: ToolPart): string {
  const state = part.state
  if (state.status !== "error") return ""
  const metadata = state.metadata
  if (metadata && metadata.interrupted === true && typeof metadata.output === "string") return metadata.output
  return state.error
}

/** Full text of a part as it contributes to the context sent to the model. */
function partText(part: Part): string {
  switch (part.type) {
    case "text":
      return part.text
    case "reasoning":
      return part.text
    case "file":
      return `[file ${oneLine(part.filename ?? part.url)}]`
    case "tool":
      return `[tool ${part.tool} ${JSON.stringify(part.state.input ?? {})} ${toolPayload(part)}]`
    default:
      return ""
  }
}

/** Character-based estimate; deliberately not a tokenizer (see README caveats). */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

/** Estimated context tokens for one part, including any tool attachments. */
function estimatePartTokens(part: Part): number {
  if (isToolPart(part) && part.state.status === "completed") {
    const attachmentChars = (part.state.attachments ?? []).reduce((sum, file) => sum + (file.url?.length ?? 0), 0)
    return estimateTokens(partText(part)) + Math.ceil(attachmentChars / 4)
  }
  return estimateTokens(partText(part))
}

function toolStatus(part: ToolPart): string {
  switch (part.state.status) {
    case "completed":
      return `ok ${part.state.output.length} chars`
    case "error":
      return `error ${errorPayload(part).length} chars`
    default:
      return part.state.status
  }
}

export type ToolRef = { id: string; tool: string; input: Record<string, unknown>; pinned: boolean }

const isPinned = (index: number, total: number, preserveRecent: number) =>
  index === 0 || index >= Math.max(1, total - preserveRecent)

/** Collect every tool call in order, marking the pinned (never-touched) ones. */
export function collectToolCalls(messages: Msg[], preserveRecent: number): ToolRef[] {
  const refs: ToolRef[] = []
  messages.forEach((message, index) => {
    const pinned = isPinned(index, messages.length, preserveRecent)
    for (const part of message.parts) {
      if (!isToolPart(part)) continue
      refs.push({ id: part.id, tool: part.tool, input: part.state.input ?? {}, pinned })
    }
  })
  return refs
}

/** A skeletal view of the conversation: tool outputs are omitted, long text abridged. */
export function buildState(messages: Msg[], preserveRecent: number, sendText = true): string {
  const lines: string[] = []
  messages.forEach((message, index) => {
    const pinned = isPinned(index, messages.length, preserveRecent)
    lines.push(`#${index} ${message.info.role}${pinned ? " (pinned)" : ""}`)
    for (const part of message.parts) {
      if (part.type === "tool") {
        const input = summarize(JSON.stringify(part.state.input ?? {}), TOOL_INPUT_STATE_CHARS)
        lines.push(`  call ${oneLine(part.tool)} input=${input} -> ${toolStatus(part)}`)
      } else if (sendText) {
        const text = partText(part).trim()
        if (text) lines.push(`  ${abridge(text, STATE_TEXT_HEAD, STATE_TEXT_TAIL)}`)
      }
    }
  })
  return lines.join("\n")
}

/** Abridge oldest long lines until the state fits its budget, or give up. */
export function fitState(state: string, maxStateTokens: number): string | null {
  if (estimateTokens(state) <= maxStateTokens) return state
  const lines = state.split("\n")
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].length <= STATE_LINE_MIN_ABRIDGE) continue
    lines[i] = abridge(lines[i], STATE_LINE_HEAD, STATE_LINE_TAIL)
    if (estimateTokens(lines.join("\n")) <= maxStateTokens) return lines.join("\n")
  }
  return null
}

export type Decision = { keepCall: boolean; keepResult: boolean }

function questionsFor(refs: ToolRef[], offset: number): { questions: Record<string, JevQuestion> } {
  const questions: Record<string, JevQuestion> = {}
  refs.forEach((ref, i) => {
    const n = offset + i
    const tool = JSON.stringify(oneLine(ref.tool))
    const input = summarize(JSON.stringify(ref.input), TOOL_INPUT_QUESTION_CHARS)
    questions[`c${n}_keep_call`] = {
      type: "noul",
      instructions:
        `The tool call #${n} (${tool}, input ${input}) was made earlier in this session. ` +
        `Remembering that this call happened, with this input, still matters for completing the current task.`,
    }
    questions[`c${n}_keep_result`] = {
      type: "noul",
      instructions:
        `The result of tool call #${n} (${tool}) is still needed verbatim to continue this task, ` +
        `and re-running the tool would not reproduce it.`,
    }
  })
  return { questions }
}

/** Split candidate calls into batches whose questions fit alongside the state. */
export function batchCalls(refs: ToolRef[], stateTokens: number, maxRequestTokens: number): ToolRef[][] {
  const budget = maxRequestTokens - stateTokens
  if (budget <= 0) return []
  const batches: ToolRef[][] = []
  let current: ToolRef[] = []
  let used = 0
  for (const ref of refs) {
    const cost = estimateTokens(ref.tool) + estimateTokens(JSON.stringify(ref.input)) + QUESTION_OVERHEAD_TOKENS
    if (current.length > 0 && used + cost > budget) {
      batches.push(current)
      current = []
      used = 0
    }
    current.push(ref)
    used += cost
  }
  if (current.length > 0) batches.push(current)
  return batches
}

/** Reject plain http and non-loopback hosts so the bearer token cannot be redirected off-host. */
function safeBaseUrl(url: string): string {
  const parsed = new URL(url)
  if (parsed.protocol === "https:") return parsed.toString()
  const loopback = parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1" || parsed.hostname === "[::1]"
  if (parsed.protocol === "http:" && loopback) return parsed.toString()
  throw new Error(`JEV_BASE_URL must use https (or http on loopback); refusing ${url}`)
}

export async function defaultAsk(options: CompactionOptions): Promise<JevAsker> {
  return async (state, questions) => {
    if (!options.apiKey) throw new Error("TYPESAFE_API_KEY is not set")
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_OPTIONS.timeoutMs)
    try {
      const response = await fetch(safeBaseUrl(options.baseUrl), {
        method: "POST",
        redirect: "error",
        signal: controller.signal,
        headers: { Authorization: `Bearer ${options.apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ state, model: options.model, questions }),
      })
      if (!response.ok) {
        throw new Error(`Jev request failed: ${response.status} ${await response.text()}`)
      }
      return (await response.json()) as JevResponse
    } finally {
      clearTimeout(timer)
    }
  }
}

function answerNoul(response: JevResponse | null | undefined, key: string): number | undefined {
  const value = response?.answers?.[key]?.noul
  // Only a real probability counts; NaN/±Infinity and out-of-range values are
  // treated as unanswered, which fails open to "keep".
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : undefined
}

export type ApplyResult = { messages: Msg[]; changed: boolean; kept: number; truncated: number; removed: number }

/** Rebuild the message list from per-part decisions. Missing decisions keep a part. */
export function applyDecisions(messages: Msg[], decisions: Map<string, Decision>, truncateHeadChars: number): ApplyResult {
  let kept = 0
  let truncated = 0
  let removed = 0
  const out: Msg[] = []
  for (const message of messages) {
    const parts: Part[] = []
    let touched = false
    for (const part of message.parts) {
      if (!isToolPart(part)) {
        parts.push(part)
        continue
      }
      const decision = decisions.get(part.id)
      // keepResult is authoritative: a result still needed verbatim must survive,
      // whether or not the call itself is judged worth remembering.
      if (!decision || decision.keepResult) {
        parts.push(part)
        kept++
        continue
      }
      if (decision.keepCall) {
        const shortened = truncateToolResult(part, truncateHeadChars)
        if (shortened) {
          parts.push(shortened)
          truncated++
          touched = true
        } else {
          parts.push(part)
          kept++
        }
        continue
      }
      // Never drop an in-flight tool call: opencode still needs the call/result
      // pairing, and there is no result to reclaim yet.
      if (part.state.status !== "completed" && part.state.status !== "error") {
        parts.push(part)
        kept++
        continue
      }
      removed++
      touched = true
    }
    if (parts.length === 0) continue
    out.push(touched ? { info: message.info, parts } : message)
  }
  if (truncated + removed === 0) return { messages, changed: false, kept, truncated: 0, removed: 0 }
  return { messages: out, changed: true, kept, truncated, removed }
}

function truncateToolResult(part: ToolPart, headChars: number): ToolPart | null {
  const state = part.state
  if (state.status === "completed") {
    const over = state.output.length > headChars
    const hasAttachments = (state.attachments?.length ?? 0) > 0
    if (!over && !hasAttachments) return null
    const note = over ? `\n\n[jev-compaction: ${state.output.length - headChars} chars omitted; re-run ${oneLine(part.tool)} if needed]` : ""
    // Do not set time.compacted: opencode reads that as "content cleared" and
    // replaces the output with a generic marker, discarding this preview. Drop
    // attachments too, since the result was judged not needed verbatim.
    return {
      ...part,
      state: { ...state, output: (over ? state.output.slice(0, headChars) : state.output) + note, attachments: undefined },
    }
  }
  if (state.status === "error") {
    const payload = errorPayload(part)
    if (payload.length <= headChars) return null
    const note = `\n[jev-compaction: ${payload.length - headChars} chars omitted]`
    const metadata = { ...(state.metadata ?? {}) }
    if (typeof metadata.output === "string" && metadata.output.length > headChars) {
      metadata.output = metadata.output.slice(0, headChars) + note
    }
    return { ...part, state: { ...state, error: state.error.slice(0, headChars) + note, metadata } }
  }
  return null
}

export type SkipReason = "disabled" | "empty" | "under threshold" | "state too large" | "no candidates" | "no request budget"
export type CompactResult = ApplyResult & { skipped?: SkipReason }

/** Run tasks with bounded concurrency, settling each so one failure cannot orphan the rest. */
async function mapLimit<T, R>(items: T[], limit: number, run: (item: T) => Promise<R>): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length)
  let next = 0
  const worker = async () => {
    for (;;) {
      const index = next++
      if (index >= items.length) return
      try {
        results[index] = { status: "fulfilled", value: await run(items[index]) }
      } catch (reason) {
        results[index] = { status: "rejected", reason }
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(Math.max(1, limit), items.length) }, worker))
  return results
}

export async function compact(messages: Msg[], options: CompactionOptions): Promise<CompactResult> {
  const log = options.log ?? (() => {})
  const empty: CompactResult = { messages, changed: false, kept: 0, truncated: 0, removed: 0 }
  if (!options.enabled) return { ...empty, skipped: "disabled" }
  if (messages.length === 0) return { ...empty, skipped: "empty" }

  const totalTokens = messages.reduce((sum, m) => sum + m.parts.reduce((s, p) => s + estimatePartTokens(p), 0), 0)
  if (totalTokens < options.threshold) return { ...empty, skipped: "under threshold" }

  const fitted = fitState(buildState(messages, options.preserveRecent, options.sendText ?? DEFAULT_OPTIONS.sendText), options.maxStateTokens)
  if (fitted === null) return { ...empty, skipped: "state too large" }

  const candidates = collectToolCalls(messages, options.preserveRecent).filter((ref) => !ref.pinned)
  if (candidates.length === 0) return { ...empty, skipped: "no candidates" }

  const ask = options.ask ?? (await defaultAsk(options))
  const batches = batchCalls(candidates, estimateTokens(fitted), options.maxRequestTokens)
  if (batches.length === 0) return { ...empty, skipped: "no request budget" }

  // Precompute each batch's starting offset once, so question keys and answer
  // reads cannot drift apart.
  let next = 0
  const jobs = batches.map((batch) => {
    const start = next
    next += batch.length
    return { batch, start }
  })

  // Bounded concurrency with per-task settling: one failed batch cannot discard
  // the others' decisions, and a large candidate set cannot burst all requests
  // at once. Each task is awaited inside mapLimit's try/catch, so a synchronous
  // throw becomes a handled rejection.
  const settled = await mapLimit(jobs, options.maxConcurrentRequests ?? DEFAULT_OPTIONS.maxConcurrentRequests, ({ batch, start }) =>
    ask(fitted, questionsFor(batch, start).questions),
  )

  const decisions = new Map<string, Decision>()
  for (const [jobIndex, { batch, start }] of jobs.entries()) {
    const outcome = settled[jobIndex]
    if (outcome.status === "rejected") {
      // Leave this batch undecided; missing decisions keep the part.
      log("warn", "Jev batch failed; keeping those calls", {
        batchStart: start,
        error: outcome.reason instanceof Error ? outcome.reason.message : String(outcome.reason),
      })
      continue
    }
    const response = outcome.value
    let answered = 0
    batch.forEach((ref, i) => {
      const n = start + i
      const keepResult = answerNoul(response, `c${n}_keep_result`)
      const keepCall = answerNoul(response, `c${n}_keep_call`)
      if (keepResult !== undefined || keepCall !== undefined) answered++
      // An unanswered question means "avoid deleting": keep.
      decisions.set(ref.id, {
        keepCall: keepCall === undefined ? true : keepCall >= options.keepThreshold,
        keepResult: keepResult === undefined ? true : keepResult >= options.keepThreshold,
      })
    })
    if (answered === 0) log("warn", "Jev returned no usable answers for a batch; keeping those calls", { batchStart: start })
  }

  const result = applyDecisions(messages, decisions, options.truncateHeadChars)
  if (options.debug) {
    log("debug", "jev-compaction applied", {
      batches: batches.length,
      candidates: candidates.length,
      kept: result.kept,
      truncated: result.truncated,
      removed: result.removed,
      approxTokensBefore: totalTokens,
    })
  }
  return result
}

export const JevCompactionPlugin: Plugin = async ({ client }) => {
  const options = optionsFromEnv()
  // Logging is best-effort and must never throw into the request path.
  const log: CompactionOptions["log"] = (level, message, extra) => {
    try {
      void client.app?.log?.({ body: { service: "jev-compaction", level, message, extra } })?.catch?.(() => {})
    } catch {
      /* logging is best-effort */
    }
  }

  if (options.debug) {
    log("info", "jev-compaction loaded", {
      enabled: options.enabled,
      threshold: options.threshold,
      hasKey: Boolean(options.apiKey),
      model: options.model,
    })
  } else if (options.enabled && !options.apiKey) {
    log("warn", "TYPESAFE_API_KEY is not set; jev-compaction will do nothing")
  }

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      try {
        const result = await compact(output.messages, { ...options, log })
        // opencode discards the hook's return value and keeps using its own
        // array reference, so the result must be spliced in place, not assigned.
        if (result.changed) output.messages.splice(0, output.messages.length, ...result.messages)
        else if (options.debug && result.skipped) log("debug", `skipped: ${result.skipped}`)
      } catch (error) {
        // Never break a session because compaction failed.
        log("warn", "jev-compaction failed; leaving messages untouched", {
          error: error instanceof Error ? error.message : String(error),
        })
      }
    },
  }
}

// opencode's loader (readV1Plugin) only accepts a default export that is an
// object carrying `id`/`server`. A bare function default falls through to the
// legacy path, which treats *every* export as a plugin factory and throws on
// non-function exports like DEFAULT_OPTIONS — so the module would never load.
const pluginModule: PluginModule = { id: "jev-compaction", server: JevCompactionPlugin }
export default pluginModule
