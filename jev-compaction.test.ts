import { describe, expect, test } from "bun:test"
import {
  DEFAULT_OPTIONS,
  JevCompactionPlugin,
  applyDecisions,
  batchCalls,
  buildState,
  collectToolCalls,
  compact,
  type CompactionOptions,
  type JevAsker,
  type JevResponse,
  type ToolRef,
} from "./jev-compaction"

const opts = (over: Partial<CompactionOptions> = {}): CompactionOptions => ({
  ...DEFAULT_OPTIONS,
  threshold: 0,
  apiKey: "test-key",
  ...over,
})

function userText(id: string, text: string) {
  return { info: { role: "user", id } as any, parts: [{ type: "text", id: `${id}-t`, text } as any] }
}

function assistantText(id: string, text: string) {
  return { info: { role: "assistant", id } as any, parts: [{ type: "text", id: `${id}-t`, text } as any] }
}

function tool(
  messageID: string,
  callID: string,
  tool: string,
  input: Record<string, unknown>,
  output: string,
  status: "completed" | "error" = "completed",
) {
  const state =
    status === "completed"
      ? { status, input, output, title: tool, metadata: {}, time: { start: 0, end: 1 } }
      : { status, input, error: output, metadata: {}, time: { start: 0, end: 1 } }
  return {
    info: { role: "assistant", id: messageID } as any,
    parts: [{ type: "tool", id: callID, callID, tool, state, sessionID: "s", messageID } as any],
  }
}

const big = (n: number) => "x".repeat(n)

describe("estimate/build/collect", () => {
  test("collectToolCalls pins the first and newest messages", () => {
    const messages = [
      tool("m0", "c0", "Read", { file: "first.ts" }, big(10)),
      tool("m1", "c1", "Read", { file: "a.ts" }, big(10)),
      tool("m2", "c2", "Read", { file: "b.ts" }, big(10)),
      tool("m3", "c3", "Read", { file: "recent.ts" }, big(10)),
    ]
    const refs = collectToolCalls(messages, 2)
    expect(refs.map((r) => [r.id, r.pinned])).toEqual([
      ["c0", true],
      ["c1", false],
      ["c2", true],
      ["c3", true],
    ])
  })

  test("buildState omits tool outputs but keeps calls and text", () => {
    const state = buildState([userText("u0", "fix the bug"), tool("m1", "c1", "Read", { file: "a.ts" }, big(9999))], 6)
    expect(state).toContain("fix the bug")
    expect(state).toContain("call Read")
    expect(state).toContain("ok 9999 chars")
    expect(state).not.toContain("xxxxxxxx")
  })

  test("batchCalls returns nothing when the state consumes the budget", () => {
    const refs: ToolRef[] = [{ id: "c1", tool: "Read", input: {}, status: "ok", pinned: false }]
    expect(batchCalls(refs, 30_000, 30_000)).toEqual([])
  })
})

describe("applyDecisions", () => {
  const messages = [
    userText("u0", "hello"),
    tool("m1", "c1", "Read", { file: "keep.ts" }, "keep me verbatim"),
    tool("m2", "c2", "Bash", { cmd: "ls" }, big(1000)),
    tool("m3", "c3", "Read", { file: "drop.ts" }, big(1000)),
  ]

  test("keeps, truncates, and removes per decision", () => {
    const decisions = new Map([
      ["c1", { keepCall: true, keepResult: true }],
      ["c2", { keepCall: true, keepResult: false }],
      ["c3", { keepCall: false, keepResult: false }],
    ])
    const result = applyDecisions(messages, decisions, 100)
    expect(result.kept).toBe(1)
    expect(result.truncated).toBe(1)
    expect(result.removed).toBe(1)
    expect(result.messages.length).toBe(3)
    const bashState: any = (result.messages[2].parts[0] as any).state
    expect(bashState.output.length).toBeLessThan(1000)
    expect(bashState.output).toContain("chars omitted")
  })

  test("drops a message that loses its only part", () => {
    const decisions = new Map([["c3", { keepCall: false, keepResult: false }]])
    const result = applyDecisions([tool("m3", "c3", "Read", {}, big(50))], decisions, 100)
    expect(result.messages).toEqual([])
  })

  test("missing decisions keep everything", () => {
    const result = applyDecisions(messages, new Map(), 100)
    expect(result.changed).toBe(false)
    expect(result.messages).toBe(messages)
  })
})

describe("compact", () => {
  const history = () => [
    userText("u0", "investigate the failure"),
    tool("m1", "c1", "Read", { file: "a.ts" }, big(4000)),
    tool("m2", "c2", "Bash", { cmd: "tests" }, big(4000)),
    tool("m3", "c3", "Read", { file: "b.ts" }, big(4000)),
    tool("m4", "c4", "Read", { file: "c.ts" }, big(4000)),
    tool("m5", "c5", "Read", { file: "d.ts" }, big(4000)),
    assistantText("a6", "still working"),
  ]

  const policyAsk: JevAsker = async (_state, questions): Promise<JevResponse> => {
    const answers: Record<string, { noul: number }> = {}
    for (const key of Object.keys(questions)) {
      const n = Number(key.match(/c(\d+)_/)?.[1])
      const isResult = key.endsWith("keep_result")
      // even calls: drop both; odd calls: keep call, drop result
      answers[key] = { noul: n % 2 === 0 ? 0.05 : isResult ? 0.05 : 0.95 }
    }
    return { answers }
  }

  test("skips when under the token threshold", async () => {
    const result = await compact(history(), opts({ threshold: 1_000_000 }))
    expect(result.changed).toBe(false)
    expect(result.skipped).toBe("under threshold")
  })

  test("prunes using Jev decisions and never touches pinned messages", async () => {
    const messages = history()
    const result = await compact(messages, opts({ ask: policyAsk, preserveRecent: 1, threshold: 0 }))
    expect(result.changed).toBe(true)
    // question keys are numbered by candidate order: even -> drop both, odd -> keep call + truncate result
    expect(result.truncated).toBe(2)
    expect(result.removed).toBe(3)
    const ids = result.messages.flatMap((m) => m.parts.map((p) => (p as any).id))
    expect(ids).toContain("c2")
    expect(ids).toContain("c4")
    expect(ids).not.toContain("c1")
    expect(ids).not.toContain("c3")
    expect(ids).not.toContain("c5")
  })

  test("leaves messages untouched when Jev fails", async () => {
    const boom: JevAsker = async () => {
      throw new Error("network down")
    }
    await expect(compact(history(), opts({ ask: boom, preserveRecent: 1 }))).rejects.toThrow("network down")
  })

  test("no-ops without a candidate set", async () => {
    const onlyRecent = [userText("u0", "hi"), assistantText("a1", "hello")]
    const result = await compact(onlyRecent, opts({ ask: policyAsk }))
    expect(result.changed).toBe(false)
    expect(["under threshold", "no candidates"]).toContain(result.skipped ?? "")
  })
})

describe("plugin wrapper", () => {
  test("swallows Jev errors and leaves messages untouched", async () => {
    process.env.JEV_COMPACTION_THRESHOLD = "0"
    delete process.env.TYPESAFE_API_KEY
    delete process.env.JEV_COMPACTION_DEBUG
    const hooks = await JevCompactionPlugin({ client: { app: { log: async () => {} } } } as any)
    const messages = [
      userText("u0", "x"),
      tool("m1", "c1", "Read", { file: "a.ts" }, big(4000)),
      tool("m2", "c2", "Read", { file: "b.ts" }, big(4000)),
      assistantText("a3", "y"),
    ]
    const output = { messages } as any
    await hooks["experimental.chat.messages.transform"]!({}, output)
    expect(output.messages).toBe(messages)
  })
})
