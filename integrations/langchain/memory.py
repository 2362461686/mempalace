"""MemPalace LangChain Chat Memory — persistent cross-session memory for agents.

Implements LangChain's ``BaseChatMemory`` using MemPalace as the persistent
backend. Every conversation turn is stored verbatim in the palace, and the
memory loads relevant history on demand via semantic search.

Usage:
    from integrations.langchain import MemPalaceChatMemory

    memory = MemPalaceChatMemory(
        palace_path="~/my-palace",
        wing="my-agent",
        return_messages=True,
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field


class MemPalaceChatMemory(BaseChatMemory):
    """Chat memory backed by MemPalace for persistent, local-first agent memory.

    Every HumanMessage and AIMessage is saved as a drawer in the palace.
    When loading memory, the most recent N messages are returned, optionally
    enriched with semantically relevant past conversations.

    Parameters
    ----------
    palace_path : str
        Path to the MemPalace directory.
    wing : str
        Wing name for organizing conversations (default ``"agent-chat"``).
    max_token_limit : int
        Maximum approximate tokens to load (default 4000).
    semantic_recall : bool
        If True, also retrieve semantically similar past messages
        in addition to the most recent ones (default False).
    semantic_k : int
        Number of semantic recall messages to fetch (default 5).
    """

    palace_path: str
    wing: str = "agent-chat"
    max_token_limit: int = 4000
    semantic_recall: bool = False
    semantic_k: int = 5

    # Internal index for the current session
    _session_room: str = Field(default="", exclude=True)
    _message_counter: int = Field(default=0, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Create a unique room name for this session
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self._session_room = f"session-{ts}"
        self._message_counter = 0

    @property
    def memory_variables(self) -> list[str]:
        return ["history"]

    def _save_to_palace(self, role: str, content: str) -> None:
        """Store a single message as a drawer in the palace."""
        try:
            from mempalace.palace import get_collection
            from mempalace.embedding import get_embedding_function

            col = get_collection(self.palace_path, create=True)
            ef = get_embedding_function()

            self._message_counter += 1
            doc_id = f"{self._session_room}-msg-{self._message_counter:04d}"
            embedding = ef([content])

            col.add(
                documents=[content],
                ids=[doc_id],
                embeddings=embedding,
                metadatas=[
                    {
                        "wing": self.wing,
                        "room": self._session_room,
                        "role": role,
                        "source_file": f"chat:{self._session_room}",
                        "filed_at": datetime.now().isoformat(),
                        "chunk_index": self._message_counter,
                    }
                ],
            )
        except Exception:
            # Memory save failures should not crash the agent conversation.
            pass

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Save conversation context to MemPalace."""
        # Save user input
        input_str = self._get_input_string(inputs)
        if input_str:
            self._save_to_palace("human", input_str)

        # Save AI output
        output_str = self._get_output_string(outputs)
        if output_str:
            self._save_to_palace("ai", output_str)

        # Also save to in-memory chat history (parent class behavior)
        super().save_context(inputs, outputs)

    def _get_input_string(self, inputs: dict[str, Any]) -> str:
        """Extract input text from the inputs dict."""
        if "input" in inputs:
            return str(inputs["input"])
        # Concatenate all non-memory inputs
        parts = []
        for k, v in inputs.items():
            if k == "history":
                continue
            parts.append(str(v))
        return " ".join(parts)

    def _get_output_string(self, outputs: dict[str, str]) -> str:
        """Extract output text from the outputs dict."""
        if "output" in outputs:
            return outputs["output"]
        if "response" in outputs:
            return outputs["response"]
        return " ".join(str(v) for v in outputs.values())

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory variables, including semantic recall if enabled."""
        # Always include the recent in-memory messages (parent class behavior)
        base = super().load_memory_variables(inputs)

        if not self.semantic_recall:
            return base

        # Add semantically relevant past messages
        query = self._get_input_string(inputs) if inputs else ""
        if not query:
            return base

        try:
            from mempalace.searcher import search_memories

            results = search_memories(
                query=query,
                palace_path=self.palace_path,
                wing=self.wing,
                n_results=self.semantic_k,
            )

            recall_messages: list = []
            for hit in results.get("hits", []):
                role = hit.get("metadata", {}).get("role", "unknown")
                text = hit.get("text", "")
                if role == "human":
                    recall_messages.append(HumanMessage(content=text))
                elif role == "ai":
                    recall_messages.append(AIMessage(content=text))
                else:
                    recall_messages.append(HumanMessage(content=text))

            # Prepend recalled messages before recent history
            existing = base.get("history", [])
            if isinstance(existing, list):
                base["history"] = recall_messages + existing
            elif isinstance(existing, str):
                recall_str = "\n".join(
                    f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in recall_messages
                )
                base["history"] = recall_str + "\n" + existing

        except Exception:
            # Semantic recall failure should not break the conversation.
            pass

        return base

    def clear(self) -> None:
        """Clear in-memory history (palace content is preserved)."""
        super().clear()
