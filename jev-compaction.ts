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
// Failures are always non-fatal: if Jev is unreachable, unconfigured, or the
// history cannot be fitted, the messages are left exactly as they were.

import type { Plugin } from "@opencode-ai/plugin"
import type { Message, Part, ToolPart } from "@opencode-ai/sdk"

type Msg = { info: Message; parts: Part[] }

type JevQuestion =
  | { type: "noul"; instructions: string }
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  | { type: "score"; instructions: string; criteria: string[] }

type JevAnswer = { noul?: number; choice?: string; score?: number; confidence?: number }
export type JevResponse = {
  answers?: Record<string, JevAnswer>
  usage?: { input_tokens?: number; output_tokens?: number }
}

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
  apiKey?: string
  model: string
  baseUrl: string
  debug: boolean
  ask?: JevAsker
  log?: (level: "debug" | "info" | "warn" | "error", message: string, extra?: Record<string, unknown>) => void
}

export const DEFAULT_OPTIONS: Omit<CompactionOptions, "ask" | "log"> = {
  enabled: true,
  threshold: 15_000,
  preserveRecent: 6,
  keepThreshold: 0.5,
  truncateHeadChars: 300,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  model: "jev-latest",
  baseUrl: "https://api.typesafe.ai/v1/systemone",
  debug: false,
}

export function optionsFromEnv(
  env: Record<string, string | undefined> = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {},
): CompactionOptions {
  const num = (v: string | undefined, fallback: number) => {
    const n = v === undefined ? NaN : Number(v)
    return Number.isFinite(n) ? n : fallback
  }
  return {
    ...DEFAULT_OPTIONS,
    apiKey: env.TYPESAFE_API_KEY,
    enabled: env.JEV_COMPACTION_DISABLED !== "1",
    threshold: num(env.JEV_COMPACTION_THRESHOLD, DEFAULT_OPTIONS.threshold),
    preserveRecent: num(env.JEV_PRESERVE_RECENT, DEFAULT_OPTIONS.preserveRecent),
    keepThreshold: num(env.JEV_KEEP_THRESHOLD, DEFAULT_OPTIONS.keepThreshold),
    truncateHeadChars: num(env.JEV_TRUNCATE_HEAD_CHARS, DEFAULT_OPTIONS.truncateHeadChars),
    model: env.JEV_MODEL ?? DEFAULT_OPTIONS.model,
    baseUrl: env.JEV_BASE_URL ?? DEFAULT_OPTIONS.baseUrl,
    debug: env.JEV_COMPACTION_DEBUG === "1",
  }
}

const isToolPart = (part: Part): part is ToolPart => part.type === "tool"

function partText(part: Part): string {
  switch (part.type) {
    case "text":
      return part.text
    case "reasoning":
      return part.text
    case "file":
      return `[file ${part.filename ?? part.url}]`
    case "tool":
      return `[tool ${part.tool} ${summarize(JSON.stringify(part.state.input ?? {}), 200)}]`
    default:
      return ""
  }
}

function summarize(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max)}…`
}

function abridge(text: string, head: number, tail: number): string {
  if (text.length <= head + tail) return text
  return `${text.slice(0, head)}\n[… ${text.length - head - tail} chars omitted …]\n${text.slice(-tail)}`
}

/** Character-based estimate; deliberately not a tokenizer (see README caveats). */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

function toolStatus(part: ToolPart): string {
  switch (part.state.status) {
    case "completed":
      return `ok ${part.state.output.length} chars`
    case "error":
      return `error ${part.state.error.length} chars`
    default:
      return part.state.status
  }
}

export type ToolRef = { id: string; tool: string; input: Record<string, unknown>; status: string; pinned: boolean }

/** Collect every tool call in order, marking the pinned (never-touched) ones. */
export function collectToolCalls(messages: Msg[], preserveRecent: number): ToolRef[] {
  const refs: ToolRef[] = []
  const pinnedFrom = Math.max(1, messages.length - preserveRecent)
  messages.forEach((message, index) => {
    const pinned = index === 0 || index >= pinnedFrom
    for (const part of message.parts) {
      if (!isToolPart(part)) continue
      refs.push({
        id: part.id,
        tool: part.tool,
        input: part.state.input ?? {},
        status: toolStatus(part),
        pinned,
      })
    }
  })
  return refs
}

/** A skeletal view of the conversation: tool outputs are omitted, long text abridged. */
export function buildState(messages: Msg[], preserveRecent: number): string {
  const pinnedFrom = Math.max(1, messages.length - preserveRecent)
  const lines: string[] = []
  messages.forEach((message, index) => {
    const pinned = index === 0 || index >= pinnedFrom
    lines.push(`#${index} ${message.info.role}${pinned ? " (pinned)" : ""}`)
    for (const part of message.parts) {
      if (part.type === "tool") {
        const input = summarize(JSON.stringify(part.state.input ?? {}), 200)
        lines.push(`  call ${part.tool} input=${input} -> ${toolStatus(part)}`)
      } else {
        const text = partText(part).trim()
        if (text) lines.push(`  ${abridge(text, 600, 300)}`)
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
    if (lines[i].length <= 400) continue
    lines[i] = abridge(lines[i], 120, 60)
    if (estimateTokens(lines.join("\n")) <= maxStateTokens) return lines.join("\n")
  }
  return null
}

export type Decision = { keepCall: boolean; keepResult: boolean }

function questionsFor(refs: ToolRef[], offset: number): { questions: Record<string, JevQuestion> } {
  const questions: Record<string, JevQuestion> = {}
  refs.forEach((ref, i) => {
    const n = offset + i
    const input = summarize(JSON.stringify(ref.input), 300)
    questions[`c${n}_keep_call`] = {
      type: "noul",
      instructions:
        `The tool call #${n} ("${ref.tool}", input ${input}) was made earlier in this session. ` +
        `Remembering that this call happened, with this input, still matters for completing the current task.`,
    }
    questions[`c${n}_keep_result`] = {
      type: "noul",
      instructions:
        `The result of tool call #${n} ("${ref.tool}") is still needed verbatim to continue this task, ` +
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
    const cost = estimateTokens(ref.tool) + estimateTokens(JSON.stringify(ref.input)) + 120
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

async function defaultAsk(options: CompactionOptions): Promise<JevAsker> {
  return async (state, questions) => {
    if (!options.apiKey) throw new Error("TYPESAFE_API_KEY is not set")
    const response = await fetch(options.baseUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${options.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ state, model: options.model, questions }),
    })
    if (!response.ok) {
      throw new Error(`Jev request failed: ${response.status} ${await response.text()}`)
    }
    return (await response.json()) as JevResponse
  }
}

function answerNoul(response: JevResponse, key: string): number | undefined {
  const value = response.answers?.[key]?.noul
  return typeof value === "number" ? value : undefined
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
      if (!decision || (decision.keepCall && decision.keepResult)) {
        parts.push(part)
        kept++
        continue
      }
      if (decision.keepCall) {
        parts.push(truncateToolResult(part, truncateHeadChars))
        truncated++
        touched = true
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

function truncateToolResult(part: ToolPart, headChars: number): ToolPart {
  const state = part.state
  if (state.status === "completed") {
    const dropped = Math.max(0, state.output.length - headChars)
    const note = `\n\n[jev-compaction: ${dropped} chars omitted; re-run ${part.tool} if needed]`
    return {
      ...part,
      state: { ...state, output: state.output.slice(0, headChars) + note, time: { ...state.time, compacted: Date.now() } },
    }
  }
  if (state.status === "error") {
    const dropped = Math.max(0, state.error.length - headChars)
    const note = `\n[jev-compaction: ${dropped} chars omitted]`
    return { ...part, state: { ...state, error: state.error.slice(0, headChars) + note } }
  }
  return part
}

export type CompactResult = ApplyResult & { skipped?: string }

export async function compact(messages: Msg[], options: CompactionOptions): Promise<CompactResult> {
  const log = options.log ?? (() => {})
  const empty: CompactResult = { messages, changed: false, kept: 0, truncated: 0, removed: 0 }
  if (!options.enabled) return { ...empty, skipped: "disabled" }
  if (messages.length === 0) return { ...empty, skipped: "empty" }

  const totalTokens = messages.reduce((sum, m) => sum + estimateTokens(m.parts.map(partText).join("\n")), 0)
  if (totalTokens < options.threshold) return { ...empty, skipped: "under threshold" }

  const fitted = fitState(buildState(messages, options.preserveRecent), options.maxStateTokens)
  if (fitted === null) return { ...empty, skipped: "state too large" }

  const candidates = collectToolCalls(messages, options.preserveRecent).filter((ref) => !ref.pinned)
  if (candidates.length === 0) return { ...empty, skipped: "no candidates" }

  const ask = options.ask ?? (await defaultAsk(options))
  const stateTokens = estimateTokens(fitted)
  const batches = batchCalls(candidates, stateTokens, options.maxRequestTokens)
  if (batches.length === 0) return { ...empty, skipped: "no request budget" }

  let offset = 0
  const responses = await Promise.all(
    batches.map(async (batch) => {
      const request = questionsFor(batch, offset)
      offset += batch.length
      return ask(fitted, request.questions)
    }),
  )

  const decisions = new Map<string, Decision>()
  let cursor = 0
  for (const [batchIndex, batch] of batches.entries()) {
    const response = responses[batchIndex]
    batch.forEach((ref, i) => {
      const n = cursor + i
      const keepResult = answerNoul(response, `c${n}_keep_result`)
      const keepCall = answerNoul(response, `c${n}_keep_call`)
      // An unanswered question means "avoid deleting": keep.
      decisions.set(ref.id, {
        keepCall: keepCall === undefined ? true : keepCall >= options.keepThreshold,
        keepResult: keepResult === undefined ? true : keepResult >= options.keepThreshold,
      })
    })
    cursor += batch.length
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
  const log: CompactionOptions["log"] =
    options.debug || options.apiKey
      ? (level, message, extra) => {
          void client.app
            .log({ body: { service: "jev-compaction", level, message, extra } })
            .catch(() => {})
        }
      : undefined

  if (options.debug) {
    await log?.("info", "jev-compaction loaded", {
      enabled: options.enabled,
      threshold: options.threshold,
      hasKey: Boolean(options.apiKey),
      model: options.model,
    })
  }

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      try {
        const messages = output.messages as Msg[]
        const result = await compact(messages, { ...options, log })
        if (result.changed) output.messages = result.messages as typeof output.messages
        else if (options.debug && result.skipped) await log?.("debug", `skipped: ${result.skipped}`)
      } catch (error) {
        // Never break a session because compaction failed.
        await log?.("warn", "jev-compaction failed; leaving messages untouched", {
          error: error instanceof Error ? error.message : String(error),
        })
      }
    },
  }
}

export default JevCompactionPlugin
