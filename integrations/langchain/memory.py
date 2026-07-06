"""MemPalace LangChain Chat Memory — persistent cross-session memory for agents.

Provides a standalone chat memory class backed by MemPalace that works with
LangChain >= 1.0. Every conversation turn is stored verbatim, and memory
loads relevant history via semantic search.

Usage:
    from integrations.langchain import MemPalaceChatMemory

    memory = MemPalaceChatMemory(
        palace_path="~/my-palace",
        wing="my-agent",
    )

    # Save turns
    memory.save_context({"input": "hello"}, {"output": "hi there!"})

    # Load history
    history = memory.load_memory_variables({"input": "tell me more"})
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class MemPalaceChatMemory:
    """Standalone chat memory backed by MemPalace for LangChain >= 1.0.

    Stores every conversation turn in the palace with role metadata.
    Loads recent history + semantically relevant past conversations.

    Parameters
    ----------
    palace_path : str
        Path to the MemPalace directory.
    wing : str
        Wing name for organizing this agent's conversations.
    return_messages : bool
        If True, returns a list of LangChain Message objects.
        If False, returns a formatted string (default True).
    max_token_limit : int
        Approximate maximum tokens for history (default 4000).
    semantic_recall : bool
        Also fetch semantically similar past messages (default True).
    semantic_k : int
        Number of recalled messages (default 5).
    """

    def __init__(
        self,
        palace_path: str,
        wing: str = "agent-chat",
        return_messages: bool = True,
        max_token_limit: int = 4000,
        semantic_recall: bool = True,
        semantic_k: int = 5,
    ):
        self.palace_path = palace_path
        self.wing = wing
        self.return_messages = return_messages
        self.max_token_limit = max_token_limit
        self.semantic_recall = semantic_recall
        self.semantic_k = semantic_k

        # In-memory buffer for the current session
        self._chat_history: list[BaseMessage] = []

        # Session room identifier
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self._session_room = f"session-{ts}"
        self._message_counter = 0

    @property
    def memory_variables(self) -> list[str]:
        """Return the memory variable names used by LangChain chains."""
        return ["history"]

    def _save_to_palace(self, role: str, content: str) -> None:
        """Store a message as a drawer in MemPalace."""
        try:
            from mempalace.palace import get_collection
            from mempalace.embedding import get_embedding_function

            col = get_collection(self.palace_path, create=True)
            ef = get_embedding_function()

            self._message_counter += 1
            doc_id = f"{self._session_room}-msg-{self._message_counter:04d}"
            embeddings = ef([content])

            col.add(
                documents=[content],
                ids=[doc_id],
                embeddings=embeddings,
                metadatas=[{
                    "wing": self.wing,
                    "room": self._session_room,
                    "role": role,
                    "source_file": f"chat:{self._session_room}",
                    "filed_at": datetime.now().isoformat(),
                    "chunk_index": self._message_counter,
                }],
            )
        except Exception:
            # Memory persistence failures should not crash conversations.
            pass

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        """Save conversation context to MemPalace and in-memory buffer."""
        # Extract input text
        input_str = ""
        if "input" in inputs:
            input_str = str(inputs["input"])
        else:
            parts = [str(v) for k, v in inputs.items() if k != "history"]
            input_str = " ".join(parts)

        # Extract output text
        output_str = ""
        if "output" in outputs:
            output_str = str(outputs["output"])
        elif "response" in outputs:
            output_str = str(outputs["response"])
        else:
            output_str = " ".join(str(v) for v in outputs.values())

        # Persist to palace
        if input_str:
            self._save_to_palace("human", input_str)
        if output_str:
            self._save_to_palace("ai", output_str)

        # Keep in memory buffer
        if input_str:
            self._chat_history.append(HumanMessage(content=input_str))
        if output_str:
            self._chat_history.append(AIMessage(content=output_str))

    def _load_semantic_memory(
        self, query: str
    ) -> list[BaseMessage]:
        """Fetch semantically relevant past messages from the palace."""
        if not query:
            return []
        try:
            from mempalace.searcher import search_memories

            results = search_memories(
                query=query,
                palace_path=self.palace_path,
                wing=self.wing,
                n_results=self.semantic_k,
            )

            messages: list[BaseMessage] = []
            for hit in results.get("results", results.get("hits", [])):
                role = hit.get("metadata", {}).get("role", "unknown")
                text = hit.get("text", "")
                if not text:
                    continue
                if role in ("human", "user"):
                    messages.append(HumanMessage(content=text))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=text))
                else:
                    messages.append(HumanMessage(content=text))
            return messages
        except Exception:
            return []

    def load_memory_variables(
        self, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Load memory variables (history) for the next conversation turn."""
        query = ""
        if isinstance(inputs, dict):
            query = str(inputs.get("input", ""))

        messages: list[BaseMessage] = []

        # Layer 1: Semantic recall of past conversations
        if self.semantic_recall and query:
            recall = self._load_semantic_memory(query)
            messages.extend(recall)

        # Layer 2: Current session history
        messages.extend(self._chat_history)

        if self.return_messages:
            return {"history": messages}
        else:
            # Format as string
            lines = []
            for msg in messages:
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                lines.append(f"{role}: {msg.content}")
            return {"history": "\n".join(lines)}

    def clear(self) -> None:
        """Clear the in-memory chat history."""
        self._chat_history.clear()
