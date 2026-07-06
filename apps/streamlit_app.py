"""Streamlit Memory Palace Browser — visual interface for MemPalace.

Launch:
    streamlit run apps/streamlit_app.py

Dependencies:
    pip install streamlit
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ---- Page config -----------------------------------------------------------
st.set_page_config(
    page_title="MemPalace Browser",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS -------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .search-result {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;
        margin-bottom: 0.75rem; background: #fafafa;
    }
    .result-header { display: flex; justify-content: space-between; align-items: center; }
    .wing-badge {
        background: #e8f0fe; color: #1a73e8; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8rem; font-weight: 500;
    }
    .room-badge {
        background: #fce8e6; color: #d93025; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8rem; font-weight: 500;
    }
    .similarity-badge {
        background: #e6f4ea; color: #137333; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8rem; font-weight: 600;
    }
    .result-text {
        background: white; border: 1px solid #eee; border-radius: 6px;
        padding: 0.75rem; margin-top: 0.5rem; font-family: 'Consolas', monospace;
        font-size: 0.85rem; white-space: pre-wrap; max-height: 300px; overflow-y: auto;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px; padding: 1rem; text-align: center;
    }
    .stat-value { font-size: 2rem; font-weight: 700; }
    .stat-label { font-size: 0.8rem; opacity: 0.9; }
</style>
""",
    unsafe_allow_html=True,
)


# ---- Helpers ----------------------------------------------------------------
def resolve_palace_path(user_input: str) -> Path | None:
    """Resolve a palace path from user input."""
    path = Path(user_input).expanduser().resolve()
    if path.exists() and path.is_dir():
        return path
    return None


def count_drawers(palace_path: Path) -> int:
    """Count total drawers in the palace."""
    try:
        from mempalace.palace import get_collection

        col = get_collection(str(palace_path))
        if col is None:
            return 0
        return col.count()
    except Exception:
        return 0


def search_palace(
    query: str, palace_path: str, n_results: int = 20
) -> dict:
    """Run a programmatic search and return results dict."""
    from mempalace.searcher import search_memories

    return search_memories(
        query=query,
        palace_path=palace_path,
        n_results=n_results,
    )


def format_text_preview(text: str, max_len: int = 500) -> str:
    """Truncate text for preview."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... ({len(text) - max_len} more characters)"


# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏛️ MemPalace Browser")
    st.markdown("---")

    palace_input = st.text_input(
        "Palace directory path",
        value=os.environ.get("MEMPALACE_PATH", ""),
        placeholder="~/my-palace or C:\\Users\\...",
        help="Path to your MemPalace directory (created with `mempalace init`)",
    )

    palace_path = resolve_palace_path(palace_input) if palace_input else None

    if palace_path:
        st.success(f"✅ Palace found: `{palace_path.name}`")
        try:
            total = count_drawers(palace_path)
            st.metric("Total Drawers", total)
        except Exception:
            st.metric("Total Drawers", "N/A")
    else:
        if palace_input:
            st.error("❌ Palace not found at this path")
        else:
            st.info("Enter a palace directory path to begin")

    st.markdown("---")
    n_results = st.slider("Max results", 5, 50, 20, 5)
    st.markdown("---")
    st.caption("Powered by [MemPalace](https://github.com/MemPalace/mempalace)")


# ---- Main Area --------------------------------------------------------------
st.markdown('<div class="main-header">🏛️ Memory Palace Browser</div>', unsafe_allow_html=True)
st.caption("Search your AI conversation history and project knowledge — verbatim, local, private.")

if not palace_path:
    # Welcome / getting-started view
    st.info(
        """
        ### Welcome to MemPalace Browser!

        This is a visual interface for your **MemPalace** — a local-first AI memory system
        that stores your conversation history and project knowledge as verbatim text.

        **To get started:**

        1. Initialize a palace: `mempalace init ~/my-palace`
        2. Mine some content: `mempalace mine ~/my-project`
        3. Paste your palace path in the sidebar
        4. Start searching!

        No data ever leaves your machine. No API keys required.
        """
    )
else:
    # ---- Search bar ---------------------------------------------------------
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search your memory palace…",
            placeholder="e.g. what did I work on last week?",
            key="search_query",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

    if query or search_clicked:
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner(f"Searching for: *{query}*"):
                try:
                    results = search_palace(
                        query=query,
                        palace_path=str(palace_path),
                        n_results=n_results,
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = None

            if results is None:
                pass  # Error already shown
            elif "error" in results:
                st.error(f"Search error: {results['error']}")
            elif "hits" not in results or not results["hits"]:
                st.info(f'No results found for "{query}". Try different keywords.')
            else:
                hits = results["hits"]
                st.success(f"Found {len(hits)} results for *{query}*")

                for i, hit in enumerate(hits, 1):
                    similarity = hit.get("similarity", 0)
                    sim_pct = f"{similarity * 100:.1f}%"
                    sim_color = (
                        "green" if similarity > 0.7 else "orange" if similarity > 0.4 else "red"
                    )

                    with st.container():
                        st.markdown(
                            f"""
                        <div class="search-result">
                            <div class="result-header">
                                <div>
                                    <span class="wing-badge">🪽 {hit.get("wing", "?")}</span>
                                    <span class="room-badge">🏠 {hit.get("room", "?")}</span>
                                    <small style="color:#888">📄 {hit.get("source_file", "?")}</small>
                                </div>
                                <span class="similarity-badge" style="color:{sim_color}">
                                    🎯 {sim_pct}
                                </span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        text = hit.get("text", "")
                        with st.expander(f"📝 Result #{i} — View Content", expanded=(i <= 3)):
                            st.code(text, language=None)
                            if "created_at" in hit:
                                st.caption(f"Created: {hit['created_at']}")
                            if hit.get("matched_via") == "drawer+closet":
                                st.caption("🔗 Matched via drawer + closet (higher confidence)")

                        if i < len(hits):
                            st.markdown("---")

    elif not query:
        # Show a prompt to search
        st.markdown("### 👆 Enter a query above to search your memory palace")
        st.markdown(
            """
        **Example queries:**
        - What bug did I fix last week?
        - Summarize the architecture decisions
        - Who is working on the auth module?
        """
        )

# ---- Footer ----------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data stays local. No API calls. No telemetry. "
    "MemPalace Browser is a community contribution — "
    "star us on [GitHub](https://github.com/MemPalace/mempalace)!"
)
