"""LangChain + MemPalace integration examples.

Demonstrates:
1. Retriever — semantic search over project knowledge
2. Chat Memory — persistent cross-session agent memory
3. Combined — RAG agent with MemPalace as knowledge base

Requirements:
    pip install langchain langchain-core streamlit

Run:
    python integrations/langchain/example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Retriever — use MemPalace as a knowledge base
# ─────────────────────────────────────────────────────────────────────────────
def example_retriever(palace_path: str):
    """Demonstrate using MemPalaceRetriever for semantic search."""
    print("\n" + "=" * 60)
    print("Example 1: MemPalace Retriever")
    print("=" * 60)

    from integrations.langchain import MemPalaceRetriever

    retriever = MemPalaceRetriever(
        palace_path=palace_path,
        k=5,
    )

    queries = [
        "What is the main architecture of this project?",
        "Show me error handling patterns",
        "What database is being used?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            docs = retriever.invoke(query)
            if not docs:
                print("   (No results found)")
            for i, doc in enumerate(docs, 1):
                similarity = doc.metadata.get("similarity", 0)
                print(f"   [{i}] 📄 {doc.metadata.get('source', '?')} "
                      f"(similarity: {similarity:.2f})")
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"       {preview}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Chat Memory — persistent agent conversations
# ─────────────────────────────────────────────────────────────────────────────
def example_chat_memory(palace_path: str):
    """Demonstrate MemPalace as a persistent chat memory backend."""
    print("\n" + "=" * 60)
    print("Example 2: MemPalace Chat Memory")
    print("=" * 60)

    from integrations.langchain import MemPalaceChatMemory

    memory = MemPalaceChatMemory(
        palace_path=palace_path,
        wing="demo-agent",
        return_messages=True,
        semantic_recall=True,
        semantic_k=3,
    )

    # Simulate a conversation
    conversation = [
        ("What is MemPalace?", "MemPalace is a local-first AI memory system "
         "that stores conversation history verbatim for semantic search."),
        ("How does it work?", "It uses a three-layer structure: Wings for "
         "categories, Rooms for time-based groups, and Drawers for full "
         "verbatim content. Search combines BM25 keyword matching with "
         "vector semantic similarity."),
        ("Can I use it with LangChain?", "Yes! MemPalace provides a native "
         "LangChain integration with a Retriever and ChatMemory class, "
         "making it a drop-in backend for any LangChain pipeline."),
    ]

    for human_input, ai_response in conversation:
        # Save context
        memory.save_context(
            {"input": human_input}, {"output": ai_response}
        )
        print(f"\n💬 Human: {human_input}")
        print(f"🤖 AI: {ai_response}")

    # Load memory for a new query
    print("\n--- Loading memory for new query ---")
    history = memory.load_memory_variables({"input": "tell me about the architecture"})
    msg_count = len(history["history"]) if isinstance(history["history"], list) else 0
    print(f"Loaded {msg_count} messages from memory")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Stats & Info
# ─────────────────────────────────────────────────────────────────────────────
def example_stats(palace_path: str):
    """Display palace statistics."""
    print("\n" + "=" * 60)
    print("Example 3: Palace Statistics")
    print("=" * 60)

    try:
        from mempalace.palace import get_collection
        from mempalace.embedding import current_model_name, describe_device
        from mempalace.config import MempalaceConfig

        col = get_collection(palace_path)
        cfg = MempalaceConfig()

        print(f"  Palace path      : {palace_path}")
        print(f"  Total drawers    : {col.count()}")
        print(f"  Embedding model  : {current_model_name()}")
        print(f"  Device           : {describe_device()}")
        print(f"  Distance metric  : {col.distance_metric}")
    except Exception as e:
        print(f"  Error loading stats: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    palace_path = os.environ.get(
        "MEMPALACE_PATH",
        os.path.expanduser("~/mempalace-demo"),
    )

    print("🦜 LangChain + MemPalace Integration Demo")
    print(f"Palace path: {palace_path}")
    print()

    if not Path(palace_path).exists():
        print(f"\n⚠️  Palace not found at: {palace_path}")
        print("To create a demo palace:")
        print(f"  1. mempalace init {palace_path}")
        print(f"  2. mempalace mine <some-project-dir> --target {palace_path}")
        print(f"  3. Set MEMPALACE_PATH={palace_path}")
        print(f"  4. Run this script again")
        return

    example_stats(palace_path)
    example_retriever(palace_path)
    example_chat_memory(palace_path)

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
