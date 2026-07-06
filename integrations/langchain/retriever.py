"""MemPalace LangChain Retriever — semantic search over your memory palace.

Implements LangChain's ``BaseRetriever`` interface so you can use MemPalace
as a drop-in retrieval backend in any LangChain pipeline (RAG, agents, etc.).

Usage:
    from integrations.langchain import MemPalaceRetriever

    retriever = MemPalaceRetriever(palace_path="~/my-palace", k=10)
    docs = retriever.invoke("What did I learn about async Python?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MemPalaceRetriever(BaseRetriever):
    """LangChain retriever backed by MemPalace.

    Performs hybrid semantic search (BM25 + vector similarity) over the
    palace content, returning LangChain ``Document`` objects with metadata.

    Parameters
    ----------
    palace_path : str
        Path to the MemPalace directory.
    k : int
        Number of documents to retrieve (default 10).
    wing : str or None
        Optional wing filter to scope the search.
    room : str or None
        Optional room filter to scope the search.
    """

    palace_path: str
    k: int = 10
    wing: Optional[str] = None
    room: Optional[str] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Run semantic search and return LangChain Documents."""
        from mempalace.searcher import search_memories

        results = search_memories(
            query=query,
            palace_path=self.palace_path,
            wing=self.wing,
            room=self.room,
            n_results=self.k,
        )

        if "error" in results:
            raise RuntimeError(f"MemPalace search failed: {results['error']}")

        documents: list[Document] = []
        for hit in results.get("results", results.get("hits", [])):
            text = hit.get("text", "")
            similarity = hit.get("similarity", 0)
            source_file = hit.get("source_file", "")
            if not text:
                continue
            metadata = {
                "source": source_file,
                "wing": hit.get("wing", ""),
                "room": hit.get("room", ""),
                "similarity": similarity,
                "created_at": hit.get("created_at", ""),
                "matched_via": hit.get("matched_via", "drawer"),
                "source_path": hit.get("source_path", source_file),
            }
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Async version (runs sync search in a thread)."""
        return self._get_relevant_documents(query, run_manager=run_manager)
