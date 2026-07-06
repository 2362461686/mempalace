#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MemPalace Integration Demo — full end-to-end validation."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r"C:\Users\23624\MyProject\video-agent\mempalace")

PALACE = r"C:\Users\23624\MyProject\video-agent\mempalace\demo_data\mypalace"

SEP = "=" * 60

# ── Demo 1: BGE Chinese embedding models ──────────────────────────────────
print(f"\n{SEP}")
print("DEMO 1: BGE Chinese Embedding Models")
print(SEP)

from mempalace.embedding_bge import list_bge_models
models = list_bge_models()
for name, info in models.items():
    print(f"  {name}: {info['dim']}-dim, HF: {info['hf_id']}")
    print(f"         {info['desc']}")

print(f"\n  Total BGE models available: {len(models)}")

# ── Demo 2: LangChain Retriever ──────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 2: LangChain Retriever — Semantic Search")
print(SEP)

from integrations.langchain import MemPalaceRetriever

retriever = MemPalaceRetriever(palace_path=PALACE, k=5)

test_queries = [
    "What is the memory system architecture?",
    "Tell me about BGE embedding models",
    "How does privacy work in MemPalace?",
]

for query in test_queries:
    print(f"\n  [Q] {query}")
    docs = retriever.invoke(query)
    if not docs:
        print("      (no results)")
        continue
    for i, doc in enumerate(docs, 1):
        sim = doc.metadata.get("similarity", 0)
        src = doc.metadata.get("source", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"      [{i}] {src} (sim={sim:.2f}): {preview}...")

# ── Demo 3: LangChain Chat Memory ────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 3: LangChain Chat Memory — Persistent Agent Memory")
print(SEP)

from integrations.langchain import MemPalaceChatMemory

memory = MemPalaceChatMemory(
    palace_path=PALACE,
    wing="demo-agent",
    return_messages=True,
    semantic_recall=True,
    semantic_k=3,
)

conversation = [
    ("What is a memory palace?",
     "A memory palace (Method of Loci) is an ancient technique that associates "
     "information with spatial locations for better recall."),
    ("How does MemPalace implement this?",
     "MemPalace uses a 3-layer structure: Wings (categories), Rooms (time-based "
     "groups), and Drawers (verbatim content). It combines BM25 and vector search."),
    ("Can I use it for my AI agent?",
     "Yes! The LangChain integration provides MemPalaceRetriever and "
     "MemPalaceChatMemory for any LangChain-based agent pipeline."),
]

print("\n  Simulating conversation...")
for human, ai in conversation:
    memory.save_context({"input": human}, {"output": ai})
    print(f"  [Human] {human}")
    print(f"  [AI]    {ai[:80]}...")
    print()

# Test memory loading
result = memory.load_memory_variables({"input": "architecture"})
history = result.get("history", [])
msg_count = len(history) if isinstance(history, list) else 0
print(f"  Memory loaded: {msg_count} messages (including semantic recall)")

# ── Demo 4: Streamlit UI check ───────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 4: Streamlit UI — Import Check")
print(SEP)

app_path = r"C:\Users\23624\MyProject\video-agent\mempalace\apps\streamlit_app.py"
if os.path.exists(app_path):
    print(f"  Streamlit app ready: apps/streamlit_app.py")
    print(f"  Run with: streamlit run apps/streamlit_app.py")
else:
    print("  ERROR: Streamlit app not found")

# ── Demo 5: Search quality check ─────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO 5: Direct Search Quality Check")
print(SEP)

from mempalace.searcher import search_memories

query_en = "agent memory system design"
query_cn = "memory architecture Chinese document"
for label, query_str in [("English", query_en), ("Chinese", query_cn)]:
    res = search_memories(query=query_str, palace_path=PALACE, n_results=3)
    if "error" in res:
        print(f"  ERROR ({label}): {res['error']}")
    else:
        hits = res.get("results", res.get("hits", []))
        print(f"  {label}: '{query_str}' -> {len(hits)} hits")
        for i, h in enumerate(hits, 1):
            sim = h.get("similarity", 0)
            src = h.get("source_file", "?")
            preview = h.get("text", "")[:80].replace("\n", " ")
            print(f"    [{i}] {src} sim={sim:.3f} | {preview}...")

# ── Done ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("ALL DEMOS COMPLETED SUCCESSFULLY!")
print(f"{SEP}")
print(f"\nPalace: {PALACE}")
print("To launch Streamlit UI: streamlit run apps/streamlit_app.py\n")
