"""tau2 agent implementing the RLM (root + sub LLM) pattern directly.

No Claude CLI in the loop — uses the Anthropic SDK for the root model and
``harness_rlm.mcp_server.run_llm_query`` for cheap sub-LLM calls against
Haiku. This lets us exercise the RLM pattern end-to-end on tau2 without the
latency or permission overhead of a full ``claude -p`` shell-out.

Decomposition policy
--------------------
Before each root call we measure ``len(system_prompt) + len(policy)
+ len(transcript_serialized)``. If it exceeds ``decomposition_threshold_chars``
(default ~50K chars), we:

1. Chunk ``domain_policy`` into N equal-ish slices.
2. Dispatch one sub-LLM call per chunk, each asking Haiku to extract the
   policy fragments relevant to the *current user question* (the latest
   user message in the transcript).
3. Concatenate the fragments and build a slimmed-down system prompt
   containing only the relevant fragments.
4. Call the root model with the slimmed prompt.

Below the threshold we skip decomposition entirely and go straight to root.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from harness_rlm.mcp_server import run_llm_query
from harness_rlm.models import LLMQueryRequest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_MODEL_DEFAULT = "claude-sonnet-4-6"
SUB_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
DECOMPOSITION_THRESHOLD_CHARS_DEFAULT = 50_000
DECOMPOSITION_NUM_CHUNKS_DEFAULT = 4
ROOT_MAX_TOKENS = 2048
SUB_MAX_TOKENS = 1024

INVOCATIONS_LOG = Path("/tmp/rlm/tau2_invocations.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(entry: dict) -> None:
    try:
        INVOCATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INVOCATIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RLMAgentState:
    """Transcript + stable session id for one tau2 task."""

    def __init__(
        self,
        system_prompt: str,
        transcript: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.transcript: list[dict] = list(transcript) if transcript else []
        self.session_id = session_id or str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RLMAgent(HalfDuplexAgent[RLMAgentState]):
    """Root/sub-LLM tau2 agent (pure Python, no Claude CLI)."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        root_model: str = ROOT_MODEL_DEFAULT,
        sub_model: str = SUB_MODEL_DEFAULT,
        decomposition_threshold_chars: int = DECOMPOSITION_THRESHOLD_CHARS_DEFAULT,
        decomposition_num_chunks: int = DECOMPOSITION_NUM_CHUNKS_DEFAULT,
        timeout: int = 120,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.root_model = root_model
        self.sub_model = sub_model
        self.decomposition_threshold_chars = int(decomposition_threshold_chars)
        self.decomposition_num_chunks = max(1, int(decomposition_num_chunks))
        self.timeout = int(timeout)

        # Lazy-init to avoid failing on import when ANTHROPIC_API_KEY is unset.
        self._client: Optional[anthropic.Anthropic] = None

    # -- state init ---------------------------------------------------------

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> RLMAgentState:
        system_prompt = self._build_base_system_prompt(self.domain_policy)
        transcript: list[dict] = []
        if message_history:
            for msg in message_history:
                entry = self._tau2_to_entry(msg)
                if entry is not None:
                    transcript.append(entry)
        return RLMAgentState(system_prompt=system_prompt, transcript=transcript)

    # -- main turn ----------------------------------------------------------

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: RLMAgentState,
    ) -> tuple[AssistantMessage, RLMAgentState]:
        state.transcript.extend(self._tau2_to_entries(message))

        transcript_str = self._serialize_transcript(state.transcript)
        combined_len = len(state.system_prompt) + len(self.domain_policy) + len(transcript_str)

        decomposed = False
        relevant_fragments: Optional[str] = None
        last_user_q = self._latest_user_question(state.transcript)
        sub_calls_cost = 0.0
        sub_calls = 0

        if combined_len >= self.decomposition_threshold_chars and last_user_q:
            # Decompose: slice policy, extract per-chunk, stitch.
            relevant_fragments, sub_calls, sub_calls_cost = self._decompose_policy(
                question=last_user_q,
                full_policy=self.domain_policy,
                num_chunks=self.decomposition_num_chunks,
            )
            decomposed = True
            effective_system = self._build_slim_system_prompt(relevant_fragments)
        else:
            effective_system = state.system_prompt

        # Call the root model.
        t0 = time.monotonic()
        root_err: Optional[str] = None
        root_text = ""
        try:
            root_text = self._call_root(effective_system, state.transcript)
        except anthropic.APIError as e:
            root_err = f"anthropic.APIError: {e}"
        except RuntimeError as e:
            root_err = str(e)
        except Exception as e:  # noqa: BLE001
            root_err = f"{type(e).__name__}: {e}"
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        text, tool_call = self._parse_response(root_text)
        if tool_call is not None:
            assistant_msg = AssistantMessage.text(content=None, tool_calls=[tool_call])
            transcript_content = f"[tool_call] {tool_call.name}({json.dumps(tool_call.arguments)})"
        else:
            safe_text = (
                text
                if text and text.strip()
                else f"[rlm-agent error: {root_err or 'empty response'}]"
            )
            assistant_msg = AssistantMessage.text(content=safe_text)
            transcript_content = safe_text

        state.transcript.append({"role": "assistant", "content": transcript_content})

        _append_log(
            {
                "timestamp": _now_iso(),
                "agent": "rlm",
                "session_id": state.session_id,
                "combined_len": combined_len,
                "decomposed": decomposed,
                "sub_calls": sub_calls,
                "sub_calls_cost_usd": round(sub_calls_cost, 6),
                "root_model": self.root_model,
                "sub_model": self.sub_model,
                "root_elapsed_ms": elapsed_ms,
                "root_err": root_err,
                "response_preview": (root_text or "")[:400],
                "tool_call": tool_call.model_dump() if tool_call else None,
            }
        )

        return assistant_msg, state

    # ----------------------------------------------------------------------
    # Prompting
    # ----------------------------------------------------------------------

    def _build_base_system_prompt(self, policy: str) -> str:
        return (
            "You are a customer service agent. Follow the domain policy below "
            "strictly.\n\n"
            "## Output contract\n"
            "If you need to call a tool, output a single JSON line of the form:\n"
            '  {"tool_call": {"name": "<tool_name>", "arguments": {...}}}\n'
            "and nothing else — no prose, no markdown fences around it.\n"
            "Otherwise, output plain text as your reply to the user (no JSON, "
            "no tool_call key).\n\n"
            "## Tools available\n"
            f"{self._tools_schema_json()}\n\n"
            "## Domain policy\n"
            f"{policy}\n"
        )

    def _build_slim_system_prompt(self, relevant_fragments: str) -> str:
        return (
            "You are a customer service agent. Follow the policy fragments "
            "below strictly. The fragments were extracted by a sub-agent as "
            "the portions of the full policy most relevant to the user's "
            "current question.\n\n"
            "## Output contract\n"
            "If you need to call a tool, output a single JSON line of the form:\n"
            '  {"tool_call": {"name": "<tool_name>", "arguments": {...}}}\n'
            "and nothing else.\n"
            "Otherwise, output plain text as your reply to the user.\n\n"
            "## Tools available\n"
            f"{self._tools_schema_json()}\n\n"
            "## Relevant policy fragments\n"
            f"{relevant_fragments}\n"
        )

    def _tools_schema_json(self) -> str:
        schemas: list[dict[str, Any]] = []
        for t in self.tools:
            try:
                schemas.append(t.openai_schema)
            except Exception:  # noqa: BLE001
                schemas.append({"name": getattr(t, "name", "unknown")})
        return json.dumps(schemas, indent=2) if schemas else "[]"

    # ----------------------------------------------------------------------
    # Decomposition
    # ----------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, num_chunks: int) -> list[str]:
        """Split text into ``num_chunks`` roughly equal slices on whitespace."""
        if num_chunks <= 1 or not text:
            return [text]
        # Split on paragraph boundaries first; fall back to char slicing.
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < num_chunks:
            target = max(1, len(text) // num_chunks)
            return [text[i : i + target] for i in range(0, len(text), target)]
        per = max(1, len(paragraphs) // num_chunks)
        chunks: list[str] = []
        for i in range(0, len(paragraphs), per):
            chunks.append("\n\n".join(paragraphs[i : i + per]))
            if len(chunks) == num_chunks:
                # Fold any remainder into the last chunk.
                remainder = paragraphs[i + per :]
                if remainder:
                    chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(remainder)
                break
        return chunks

    def _decompose_policy(
        self, question: str, full_policy: str, num_chunks: int
    ) -> tuple[str, int, float]:
        """Map ``full_policy`` -> list of relevant fragments using the sub-LLM.

        Returns (stitched_fragments, sub_call_count, total_sub_cost_usd).
        """
        chunks = self._chunk_text(full_policy, num_chunks)
        fragments: list[str] = []
        total_cost = 0.0
        count = 0
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Can't do sub-calls; degrade gracefully by using chunk summaries.
            return full_policy, 0, 0.0

        for i, chunk in enumerate(chunks, 1):
            sub_prompt = (
                "You are extracting policy fragments relevant to a specific "
                "user question. Read the POLICY CHUNK below and output ONLY "
                "the sentences/clauses directly relevant to the USER QUESTION. "
                "If nothing is relevant, output the single token: NONE.\n\n"
                f"USER QUESTION:\n{question}\n\n"
                f"POLICY CHUNK {i}/{len(chunks)}:\n{chunk}\n"
            )
            try:
                resp = run_llm_query(
                    LLMQueryRequest(
                        prompt=sub_prompt,
                        model=self.sub_model,
                        max_tokens=SUB_MAX_TOKENS,
                    ),
                    api_key=api_key,
                )
            except anthropic.APIError:
                continue
            except Exception:  # noqa: BLE001
                continue
            count += 1
            total_cost += resp.cost_usd
            extracted = (resp.content or "").strip()
            if extracted and extracted.upper() != "NONE":
                fragments.append(f"### From chunk {i}\n{extracted}")

        if not fragments:
            # Nothing extracted — fall back to the first chunk so the root
            # model still has *some* policy context rather than none.
            return chunks[0] if chunks else full_policy, count, total_cost
        return "\n\n".join(fragments), count, total_cost

    # ----------------------------------------------------------------------
    # Root call
    # ----------------------------------------------------------------------

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set — required for RLMAgent root calls."
                )
            self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
        return self._client

    def _call_root(self, system_prompt: str, transcript: list[dict]) -> str:
        """Send the transcript to the root Anthropic model and return text."""
        client = self._ensure_client()
        messages = self._transcript_to_anthropic_messages(transcript)
        if not messages:
            # Anthropic API rejects empty message lists — inject a placeholder.
            messages = [{"role": "user", "content": "(start of conversation)"}]

        msg = client.messages.create(
            model=self.root_model,
            max_tokens=ROOT_MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    # ----------------------------------------------------------------------
    # Transcript helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _tau2_to_entry(msg: Message) -> Optional[dict]:
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content or ""}
        if isinstance(msg, AssistantMessage):
            if msg.is_tool_call() and msg.tool_calls:
                names = ", ".join(tc.name for tc in msg.tool_calls)
                return {"role": "assistant", "content": f"[tool_call] {names}"}
            return {"role": "assistant", "content": msg.content or ""}
        if isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "content": f"[tool_result id={msg.id}] {msg.content or ''}",
            }
        return None

    def _tau2_to_entries(self, msg: UserMessage | ToolMessage | MultiToolMessage) -> list[dict]:
        if isinstance(msg, MultiToolMessage):
            out: list[dict] = []
            for tm in msg.tool_messages:
                entry = self._tau2_to_entry(tm)
                if entry:
                    out.append(entry)
            return out
        entry = self._tau2_to_entry(msg)
        return [entry] if entry else []

    @staticmethod
    def _serialize_transcript(transcript: list[dict]) -> str:
        return "\n".join(
            f"[{t.get('role', 'user').capitalize()}] {t.get('content', '')}" for t in transcript
        )

    @staticmethod
    def _transcript_to_anthropic_messages(transcript: list[dict]) -> list[dict]:
        """Collapse transcript into alternating user/assistant turns.

        Anthropic rejects consecutive same-role messages, and also rejects the
        Claude-Code-only role "tool" that appears in our internal transcript.
        We therefore fold any tool lines *into* the preceding user turn (if
        they came from a tool result) as a readable ``[tool_result]`` block.
        """
        anthropic_msgs: list[dict] = []
        for entry in transcript:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role == "tool":
                if anthropic_msgs and anthropic_msgs[-1]["role"] == "user":
                    anthropic_msgs[-1]["content"] += f"\n{content}"
                else:
                    anthropic_msgs.append({"role": "user", "content": content})
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            if anthropic_msgs and anthropic_msgs[-1]["role"] == role:
                anthropic_msgs[-1]["content"] += f"\n{content}"
            else:
                anthropic_msgs.append({"role": role, "content": content})
        return anthropic_msgs

    @staticmethod
    def _latest_user_question(transcript: list[dict]) -> Optional[str]:
        for entry in reversed(transcript):
            if entry.get("role") == "user":
                c = (entry.get("content") or "").strip()
                if c:
                    return c
        return None

    # ----------------------------------------------------------------------
    # Parsing (shared contract with ClaudeHeadlessAgent)
    # ----------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, Optional[ToolCall]]:
        if not raw or not raw.strip():
            return "", None
        text = raw.strip()
        tc = RLMAgent._extract_tool_call(text)
        if tc is not None:
            return "", tc
        return text, None

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[ToolCall]:
        for line in text.splitlines():
            line = line.strip()
            if not (line.startswith("{") and '"tool_call"' in line):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tc = RLMAgent._tool_call_from_obj(obj)
            if tc is not None:
                return tc
        match = re.search(r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match:
            try:
                return RLMAgent._tool_call_from_obj(json.loads(match.group(0)))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _tool_call_from_obj(obj: Any) -> Optional[ToolCall]:
        if not isinstance(obj, dict):
            return None
        tc = obj.get("tool_call")
        if not isinstance(tc, dict):
            return None
        name = tc.get("name")
        arguments = tc.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def create_rlm_agent(tools, domain_policy, **kwargs) -> RLMAgent:
    """tau2 registry factory for :class:`RLMAgent`.

    Recognised kwargs (passed via ``llm_args``):
        - root_model (str)
        - sub_model (str)
        - decomposition_threshold_chars (int)
        - decomposition_num_chunks (int)
        - timeout (int)

    ``llm`` (if provided at top level) overrides ``root_model`` for CLI
    compatibility (``--agent-llm``).
    """
    llm_args = kwargs.get("llm_args") or {}
    llm = kwargs.get("llm")
    return RLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        root_model=llm or llm_args.get("root_model", ROOT_MODEL_DEFAULT),
        sub_model=llm_args.get("sub_model", SUB_MODEL_DEFAULT),
        decomposition_threshold_chars=int(
            llm_args.get(
                "decomposition_threshold_chars",
                DECOMPOSITION_THRESHOLD_CHARS_DEFAULT,
            )
        ),
        decomposition_num_chunks=int(
            llm_args.get("decomposition_num_chunks", DECOMPOSITION_NUM_CHUNKS_DEFAULT)
        ),
        timeout=int(llm_args.get("timeout", 120)),
    )
