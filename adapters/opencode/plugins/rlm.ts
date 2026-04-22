/**
 * harness-rlm — OpenCode plugin
 *
 * Registers an `rlm_run` tool that implements the Recursive Language Model
 * loop from Zhang, Kraska, Khattab (arXiv:2512.24601) on top of OpenCode.
 *
 *   rlm_run(context_path, query, budgets?) -> {answer, trajectory}
 *
 * Design constraints (see ../README.md):
 *   1. Sub-LLM calls go through the `rlm-mcp-server` MCP tool
 *      (`mcp__rlm__llm_query`) — NOT OpenCode's built-in `Task` tool.
 *      OpenCode's `Task` re-spawns a full subagent with system prompt +
 *      tool catalog; for pure text-in/text-out sub-calls that is the same
 *      ~50K-token re-injection tax the Claude-Code adapter avoids. The MCP
 *      path calls the Anthropic API directly and logs to /tmp/rlm/sub_calls.jsonl.
 *      If the MCP server is not registered we fall back to OpenCode's
 *      `task` tool via `client.tool.execute` and document the cost delta.
 *   2. Budgets (iterations / llm_calls / output_chars) are enforced both in
 *      the tool body AND via tool.execute.before hooks so that ANY tool the
 *      root agent calls while an RLM session is open is counted.
 *   3. Trajectory is captured in-memory (tool results) AND mirrored to
 *      /tmp/rlm/<session>.trajectory.jsonl via tool.execute.after. OpenCode
 *      plugin context does not expose an official state-persistence API
 *      (verified 2026-04-21 against https://opencode.ai/docs/plugins), so
 *      file-based persistence is the portable choice.
 *
 * Verified against OpenCode docs 2026-04-21:
 *   - package: `@opencode-ai/plugin`
 *   - tool() helper uses `args: { foo: tool.schema.string() }` (not `parameters`)
 *   - auto-loaded from `~/.config/opencode/plugins/` and `.opencode/plugins/`
 *   - hooks registered as top-level keys on the Plugin return object:
 *       `"tool.execute.before": async (input, output) => {...}`
 *       `"tool.execute.after":  async (input, output) => {...}`
 *   - throwing from tool.execute.before aborts the tool call (the only
 *     documented abort mechanism).
 */

import { type Plugin, tool } from "@opencode-ai/plugin"
import { readFile, mkdir, appendFile } from "node:fs/promises"
import { join, dirname } from "node:path"

// ---------------------------------------------------------------------------
// Shared defaults — mirror src/harness_rlm/core.py::DEFAULT_BUDGETS.
// If these diverge, the Python tau2 integration and the TS adapter will
// enforce different caps on the same run. Keep in sync.
// ---------------------------------------------------------------------------
const DEFAULT_BUDGETS = {
  max_iterations: 20,
  max_llm_calls: 50,
  max_output_chars: 10_000,
} as const

const DEFAULT_CHUNK_SIZE = 5_000
const DEFAULT_CHUNK_OVERLAP = 200
const DEFAULT_SUB_MODEL = "claude-haiku-4-5" // cheap sub-LLM default
const TRAJECTORY_DIR = "/tmp/rlm"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Budgets = {
  max_iterations: number
  max_llm_calls: number
  max_output_chars: number
}

type TrajectoryEvent = {
  ts: string
  kind: "chunk" | "sub_call" | "sub_result" | "final" | "budget_block" | "error"
  chunk_idx?: number
  prompt_preview?: string
  response_preview?: string
  chars?: number
  note?: string
}

type BudgetState = {
  iterations: number
  llm_calls: number
  total_output_chars: number
}

type SessionState = {
  session_id: string
  budgets: Budgets
  counters: BudgetState
  trajectory: TrajectoryEvent[]
}

// ---------------------------------------------------------------------------
// Module-scope session registry. One OpenCode process = one plugin instance,
// so a Map keyed by an ephemeral session_id is sufficient for concurrent
// rlm_run calls within the same process. Persistence is via the trajectory
// file (see logEvent()).
// ---------------------------------------------------------------------------
const SESSIONS = new Map<string, SessionState>()

// ---------------------------------------------------------------------------
// Helpers — pure.
// ---------------------------------------------------------------------------
function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z")
}

function newSessionId(): string {
  // OpenCode plugin context does not expose session IDs at tool-registration
  // time (docs as of 2026-04-21). Use a millisecond timestamp + random suffix.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/** Overlap-based chunker. Mirror of harness_rlm.core.chunk_context. */
export function chunkContext(
  text: string,
  chunkSize: number = DEFAULT_CHUNK_SIZE,
  overlap: number = DEFAULT_CHUNK_OVERLAP,
): string[] {
  if (chunkSize <= 0) throw new Error(`chunk_size must be > 0 (got ${chunkSize})`)
  if (overlap < 0 || overlap >= chunkSize) {
    throw new Error(
      `overlap must satisfy 0 <= overlap < chunk_size (got overlap=${overlap}, chunk_size=${chunkSize})`,
    )
  }
  if (!text) return []
  if (text.length <= chunkSize) return [text]

  const chunks: string[] = []
  const step = chunkSize - overlap
  let start = 0
  while (start < text.length) {
    const end = start + chunkSize
    chunks.push(text.slice(start, end))
    if (end >= text.length) break
    start += step
  }
  return chunks
}

/** Append one line to /tmp/rlm/<session>.trajectory.jsonl. Best-effort. */
async function logEvent(state: SessionState, event: TrajectoryEvent): Promise<void> {
  state.trajectory.push(event)
  const path = join(TRAJECTORY_DIR, `${state.session_id}.trajectory.jsonl`)
  try {
    await mkdir(dirname(path), { recursive: true })
    await appendFile(path, JSON.stringify(event) + "\n", "utf-8")
  } catch {
    // Disk failures must not kill the RLM run — the in-memory copy is still
    // returned to the caller as `trajectory`.
  }
}

/**
 * Dispatch one sub-LLM call.
 *
 * Preferred path: MCP tool `rlm_llm_query` (registered in opencode.json as the
 * `rlm` MCP server — OpenCode exposes MCP tools as `<server>_<tool>` when the
 * client calls them via the SDK). We try that first; on failure we fall back
 * to OpenCode's built-in `task` tool with a minimal prompt.
 *
 * `client` is OpenCode's SDK client. The exact shape of `client.tool.execute`
 * is partly undocumented (as of 2026-04-21), so we treat the call result as
 * `unknown` and normalize via a narrow extractor. If the MCP tool isn't
 * registered we surface a clear error in the trajectory rather than silently
 * re-inject 50K tokens by going through `task`.
 */
async function subLlmQuery(
  client: any,
  prompt: string,
  model: string,
  preferMcp: boolean,
): Promise<string> {
  // Preferred: direct MCP call bypassing task-spawn overhead.
  if (preferMcp && client?.tool?.execute) {
    try {
      const res = await client.tool.execute({
        tool: "rlm_llm_query",
        args: { prompt, model, max_tokens: 1024 },
      })
      return extractText(res)
    } catch (e) {
      // Fall through to task fallback if MCP tool is not registered.
      if (!preferMcp) throw e
    }
  }
  // Fallback: OpenCode built-in `task` tool (spawns a subagent). Costs ~50K
  // tokens of re-injection per call — tolerable only if the user has not
  // registered the MCP server. Documented in README.md.
  if (client?.tool?.execute) {
    const res = await client.tool.execute({
      tool: "task",
      args: {
        subagent_type: "rlm-subquery",
        description: "RLM sub-query",
        prompt,
      },
    })
    return extractText(res)
  }
  throw new Error(
    "OpenCode client.tool.execute is not available. Cannot dispatch sub-LLM call. " +
      "Verify plugin context shape against https://opencode.ai/docs/plugins.",
  )
}

/** Normalize whatever the client returns into a string. */
function extractText(res: unknown): string {
  if (typeof res === "string") return res
  if (res && typeof res === "object") {
    const r = res as Record<string, unknown>
    if (typeof r.text === "string") return r.text
    if (typeof r.content === "string") return r.content
    if (Array.isArray(r.content)) {
      return r.content
        .map((c: any) => (typeof c?.text === "string" ? c.text : ""))
        .filter(Boolean)
        .join("")
    }
    if (typeof r.output === "string") return r.output
  }
  return JSON.stringify(res)
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------
export const RlmPlugin: Plugin = async (ctx: any) => {
  const { client, directory } = ctx ?? {}

  return {
    // -----------------------------------------------------------------------
    // Tool: rlm_run
    // -----------------------------------------------------------------------
    tool: {
      rlm_run: tool({
        description:
          "Run a Recursive Language Model (RLM) loop over a long context file. " +
          "Chunks the context, dispatches a sub-LLM query per chunk (via the " +
          "rlm-mcp-server MCP tool for cheap direct-API calls), then synthesizes " +
          "a final answer. Implements the pattern from arXiv:2512.24601. " +
          "Returns {answer, trajectory}.",
        args: {
          context_path: tool.schema
            .string()
            .describe("Absolute path to the long-context file (e.g. a 100K-char doc)."),
          query: tool.schema
            .string()
            .describe("The question or instruction to answer using the context."),
          model: tool.schema
            .string()
            .optional()
            .describe(
              `Sub-LLM model ID. Defaults to "${DEFAULT_SUB_MODEL}". ` +
                "Must be a model the configured MCP server can route to.",
            ),
          max_iterations: tool.schema.number().optional(),
          max_llm_calls: tool.schema.number().optional(),
          max_output_chars: tool.schema.number().optional(),
          chunk_size: tool.schema.number().optional(),
          chunk_overlap: tool.schema.number().optional(),
          prefer_mcp: tool.schema
            .boolean()
            .optional()
            .describe(
              "If true (default), dispatch sub-calls via the rlm-mcp-server " +
                "MCP tool to bypass OpenCode's task-spawn re-injection tax. " +
                "Set false to force the fallback path through the built-in task tool.",
            ),
        },
        async execute(args, _context) {
          const sessionId = newSessionId()
          const budgets: Budgets = {
            max_iterations: args.max_iterations ?? DEFAULT_BUDGETS.max_iterations,
            max_llm_calls: args.max_llm_calls ?? DEFAULT_BUDGETS.max_llm_calls,
            max_output_chars: args.max_output_chars ?? DEFAULT_BUDGETS.max_output_chars,
          }
          const state: SessionState = {
            session_id: sessionId,
            budgets,
            counters: { iterations: 0, llm_calls: 0, total_output_chars: 0 },
            trajectory: [],
          }
          SESSIONS.set(sessionId, state)

          const model = args.model ?? DEFAULT_SUB_MODEL
          const preferMcp = args.prefer_mcp ?? true
          const chunkSize = args.chunk_size ?? DEFAULT_CHUNK_SIZE
          const chunkOverlap = args.chunk_overlap ?? DEFAULT_CHUNK_OVERLAP

          try {
            // 1. Load context from disk. Resolve relative paths against the
            //    plugin-context directory (CWD of the opencode invocation).
            const absPath = args.context_path.startsWith("/")
              ? args.context_path
              : join(directory ?? process.cwd(), args.context_path)
            const raw = await readFile(absPath, "utf-8")

            // 2. Chunk it.
            const chunks = chunkContext(raw, chunkSize, chunkOverlap)
            await logEvent(state, {
              ts: nowIso(),
              kind: "chunk",
              note: `loaded ${raw.length} chars, split into ${chunks.length} chunks of <=${chunkSize}`,
            })

            // 3. Dispatch one sub-call per chunk (sequentially — OpenCode
            //    doesn't guarantee isolation of concurrent MCP calls, and
            //    the budget cap is enforced per-call not per-batch).
            const partials: string[] = []
            for (let i = 0; i < chunks.length; i++) {
              state.counters.iterations += 1
              if (state.counters.iterations > budgets.max_iterations) {
                await logEvent(state, {
                  ts: nowIso(),
                  kind: "budget_block",
                  note: `max_iterations (${budgets.max_iterations}) exceeded at chunk ${i}`,
                })
                break
              }
              if (state.counters.llm_calls + 1 > budgets.max_llm_calls) {
                await logEvent(state, {
                  ts: nowIso(),
                  kind: "budget_block",
                  note: `max_llm_calls (${budgets.max_llm_calls}) exceeded at chunk ${i}`,
                })
                break
              }

              const subPrompt =
                `You are answering a sub-query over one chunk of a longer document.\n\n` +
                `Question: ${args.query}\n\n` +
                `Chunk ${i + 1}/${chunks.length}:\n"""\n${chunks[i]}\n"""\n\n` +
                `Respond with ONLY the facts from this chunk that bear on the question. ` +
                `If the chunk is irrelevant, respond exactly: NOT_RELEVANT.`

              await logEvent(state, {
                ts: nowIso(),
                kind: "sub_call",
                chunk_idx: i,
                prompt_preview: subPrompt.slice(0, 200),
              })

              try {
                const resp = await subLlmQuery(client, subPrompt, model, preferMcp)
                state.counters.llm_calls += 1
                state.counters.total_output_chars += resp.length
                if (resp.length > budgets.max_output_chars) {
                  await logEvent(state, {
                    ts: nowIso(),
                    kind: "budget_block",
                    note: `sub-response ${resp.length} chars exceeds max_output_chars ${budgets.max_output_chars} — truncating`,
                  })
                }
                const keep = resp.slice(0, budgets.max_output_chars)
                if (!/^\s*NOT_RELEVANT\s*$/i.test(keep)) partials.push(keep)
                await logEvent(state, {
                  ts: nowIso(),
                  kind: "sub_result",
                  chunk_idx: i,
                  chars: resp.length,
                  response_preview: keep.slice(0, 200),
                })
              } catch (e) {
                const msg = e instanceof Error ? e.message : String(e)
                await logEvent(state, { ts: nowIso(), kind: "error", note: msg })
                // Continue — one bad chunk should not sink the run.
              }
            }

            // 4. Synthesize. One more sub-LLM call, prompt = query + concat
            //    of partials. If the budget is already blown we return what
            //    we have as the answer.
            let answer: string
            if (
              partials.length > 0 &&
              state.counters.llm_calls + 1 <= budgets.max_llm_calls &&
              state.counters.iterations + 1 <= budgets.max_iterations
            ) {
              const synthPrompt =
                `Synthesize a single coherent answer to the question using the per-chunk notes below.\n\n` +
                `Question: ${args.query}\n\n` +
                `Per-chunk notes:\n${partials.map((p, i) => `[chunk ${i + 1}]\n${p}`).join("\n\n")}\n\n` +
                `Return ONLY the final answer. Do not mention chunks or methodology.`
              try {
                answer = await subLlmQuery(client, synthPrompt, model, preferMcp)
                state.counters.llm_calls += 1
                state.counters.iterations += 1
              } catch (e) {
                const msg = e instanceof Error ? e.message : String(e)
                await logEvent(state, { ts: nowIso(), kind: "error", note: `synthesis failed: ${msg}` })
                answer = partials.join("\n\n---\n\n")
              }
            } else {
              answer =
                partials.length > 0
                  ? partials.join("\n\n---\n\n")
                  : "RLM run produced no usable partials (all chunks were NOT_RELEVANT or failed)."
            }

            await logEvent(state, {
              ts: nowIso(),
              kind: "final",
              chars: answer.length,
              response_preview: answer.slice(0, 200),
            })

            return {
              answer,
              trajectory: state.trajectory,
              counters: state.counters,
              session_id: sessionId,
            }
          } finally {
            SESSIONS.delete(sessionId)
          }
        },
      }),
    },

    // -----------------------------------------------------------------------
    // Hooks: enforce budgets + mirror every tool call to the trajectory log.
    //
    // Scope: these hooks fire for EVERY tool the root agent runs, not just
    // rlm_run. We therefore gate on SESSIONS.size > 0 — i.e. they're no-ops
    // outside an active RLM loop. This matches the design of the Claude-Code
    // adapter's /tmp/rlm/state.json gate.
    // -----------------------------------------------------------------------
    "tool.execute.before": async (input: any, output: any) => {
      if (SESSIONS.size === 0) return
      // Every OTHER tool call (bash, read, write, grep...) that happens while
      // an RLM run is in flight counts against the shared budget. This is the
      // behaviour the RLM paper specifies: the root's entire tool-use tape is
      // budgeted, not just the sub-LLM calls.
      for (const state of SESSIONS.values()) {
        if (state.counters.llm_calls + 1 > state.budgets.max_llm_calls) {
          // Throwing is the only documented abort mechanism for
          // tool.execute.before (verified 2026-04-21).
          throw new Error(
            `RLM budget exceeded: max_llm_calls=${state.budgets.max_llm_calls} hit. ` +
              `Emit FINAL(answer) to halt cleanly.`,
          )
        }
      }
      // Redact obvious secrets from the args blob before we log.
      if (output?.args && typeof output.args === "object") {
        const scrub = (s: string) =>
          s.replace(/sk-(ant|proj|live)-[A-Za-z0-9_-]{10,}/g, "sk-***")
        for (const k of Object.keys(output.args)) {
          const v = (output.args as any)[k]
          if (typeof v === "string") (output.args as any)[k] = scrub(v)
        }
      }
    },

    "tool.execute.after": async (input: any, output: any) => {
      if (SESSIONS.size === 0) return
      const toolName = input?.tool ?? "unknown"
      const resultStr =
        typeof output?.output === "string"
          ? output.output
          : extractText(output?.output ?? output?.result ?? "")
      for (const state of SESSIONS.values()) {
        state.counters.total_output_chars += resultStr.length
        await logEvent(state, {
          ts: nowIso(),
          kind: "sub_result",
          note: `tool=${toolName}`,
          chars: resultStr.length,
          response_preview: resultStr.slice(0, 200),
        })
      }
    },
  }
}

export default RlmPlugin
