"""Setup demo palace: init, mine, and verify."""
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r"C:\Users\23624\MyProject\video-agent\mempalace")

PALACE = r"C:\Users\23624\MyProject\video-agent\mempalace\demo_data\mypalace"
DATA = r"C:\Users\23624\MyProject\video-agent\mempalace\demo_data"

from mempalace.palace import get_collection  # noqa: E402
from mempalace.embedding import get_embedding_function  # noqa: E402
from mempalace.config import MempalaceConfig  # noqa: E402

print("1. Initializing palace...")
col = get_collection(PALACE, create=True)
ef = get_embedding_function()
print(f"   Collection: OK (count={col.count()})")
print(f"   Embedder: {ef.name()}")
print(f"   Device: {os.environ.get('MEMPALACE_EMBEDDING_DEVICE', 'cpu')}")

print("\n2. Mining demo data...")
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime

files = list(Path(DATA).glob("*.txt")) + list(Path(DATA).glob("*.md"))
print(f"   Found {len(files)} files to mine")

documents = []
ids_list = []
metadatas = []

for i, fpath in enumerate(files):
    content = fpath.read_text(encoding="utf-8")
    # Split into paragraphs for better granularity
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    for j, para in enumerate(paragraphs):
        if len(para) < 20:
            continue
        doc_id = hashlib.sha256(f"{fpath.name}:{j}".encode()).hexdigest()[:16]
        documents.append(para)
        ids_list.append(doc_id)
        metadatas.append({
            "wing": "demo",
            "room": fpath.stem,
            "source_file": fpath.name,
            "filed_at": datetime.now().isoformat(),
            "chunk_index": j,
        })

print(f"   Total paragraphs: {len(documents)}")

# Generate embeddings
print("3. Generating embeddings...")
embeddings = ef(documents)
print(f"   Embeddings dimension: {len(embeddings[0])}")

# Store in ChromaDB
print("4. Storing in palace...")
col.add(
    documents=documents,
    ids=ids_list,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"\n   Done! Total drawers: {col.count()}")

# Quick search test
print("\n5. Testing search...")
results = col.query(
    query_texts=["memory system architecture"],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

docs = results.documents
dists = results.distances
if docs and docs[0]:
    for k, (doc, dist) in enumerate(zip(docs[0][:3], dists[0][:3]), 1):
        preview = doc[:100].replace("\n", " ")
        print(f"   [{k}] dist={dist:.3f} | {preview}...")
else:
    print("   No results (search may need warmer)")

print("\n✅ Demo palace setup complete!")
