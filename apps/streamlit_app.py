"""记忆宫殿浏览器 — MemPalace 可视化搜索界面

启动:
    streamlit run apps/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import streamlit as st

# ---- 页面配置 ---------------------------------------------------------------
st.set_page_config(
    page_title="记忆宫殿浏览器",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS 样式 ---------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
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
        border-radius: 4px; font-size: 0.9rem; font-weight: 600;
    }
    .result-text {
        background: white; border: 1px solid #eee; border-radius: 6px;
        padding: 0.75rem; margin-top: 0.5rem; font-family: 'Consolas', 'Microsoft YaHei', monospace;
        font-size: 0.85rem; white-space: pre-wrap; max-height: 300px; overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---- 工具函数 ----------------------------------------------------------------
@st.cache_resource
def get_collection_cached(palace_path_str: str):
    """缓存 collection 连接，避免每次查询都重新连接."""
    from mempalace.palace import get_collection
    return get_collection(palace_path_str)


def resolve_palace_path(user_input: str) -> Path | None:
    """解析 palace 路径."""
    if not user_input:
        return None
    path = Path(user_input).expanduser().resolve()
    if path.exists() and path.is_dir():
        return path
    return None


def count_drawers(palace_path_str: str) -> int:
    """统计抽屉总数."""
    try:
        col = get_collection_cached(palace_path_str)
        if col is None:
            return 0
        return col.count()
    except Exception:
        return 0


def do_search(query: str, palace_path_str: str, n_results: int) -> dict:
    """执行语义搜索，返回结果字典."""
    from mempalace.searcher import search_memories

    return search_memories(
        query=query,
        palace_path=palace_path_str,
        n_results=n_results,
    )


# ---- 侧边栏 ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏛️ 记忆宫殿浏览器")
    st.markdown("---")

    palace_input = st.text_input(
        "记忆宫殿路径",
        value=os.environ.get("MEMPALACE_PATH", ""),
        placeholder="例如: C:\\Users\\...\\mypalace",
        help="mempalace init 创建的目录路径",
    )

    palace_path = resolve_palace_path(palace_input) if palace_input else None

    if palace_path:
        palace_str = str(palace_path)
        st.success(f"✅ 已找到宫殿: `{palace_path.name}`")
        try:
            total = count_drawers(palace_str)
            st.metric("抽屉总数", total)
        except Exception as e:
            st.metric("抽屉总数", f"加载失败: {e}")
    else:
        if palace_input:
            st.error("❌ 未找到此路径的记忆宫殿")
        else:
            st.info("在左侧输入宫殿路径以开始使用")

    st.markdown("---")
    n_results = st.slider("最大搜索结果数", 5, 50, 20, 5)
    st.markdown("---")
    st.caption("基于 [MemPalace](https://github.com/2362461686/mempalace) 构建 | 本地运行 · 零 API 调用")


# ---- 主页面 ----------------------------------------------------------------
st.markdown('<div class="main-header">🏛️ 记忆宫殿浏览器</div>', unsafe_allow_html=True)
st.caption("搜索你的 AI 对话历史和项目知识 — 逐字存储、本地运行、完全私密。")

# 初始化 session state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_query_display" not in st.session_state:
    st.session_state.search_query_display = ""

if not palace_path:
    # 欢迎页
    st.info(
        """
        ### 欢迎使用记忆宫殿浏览器！

        这是一个 **MemPalace** 的可视化界面 — 一个本地优先的 AI 记忆系统，
        能够将你的对话历史和项目文档逐字存储，并支持语义搜索。

        **快速开始：**

        1. 初始化宫殿：`python demo_data/setup_demo.py`
        2. 在左侧输入宫殿路径（例如 `demo_data/mypalace`）
        3. 输入关键词开始搜索！
        """
    )
else:
    palace_str = str(palace_path)

    # 搜索栏
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "搜索你的记忆…",
            placeholder="例如: 什么是记忆系统架构？",
            key="search_input",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("🔍 搜索", use_container_width=True, type="primary")

    # 执行搜索
    if search_clicked or (query and st.session_state.get("_last_query") != query):
        if not query:
            st.warning("请输入搜索关键词。")
        else:
            st.session_state._last_query = query
            with st.spinner(f"正在搜索: *{query}*"):
                try:
                    raw = do_search(query=query, palace_path_str=palace_str, n_results=n_results)

                    # search_memories 返回 dict，结果在 "results" 键中
                    hits = raw.get("results", []) if isinstance(raw, dict) else []

                    # 调试模式 (按 D 键查看原始返回结构)
                    if st.session_state.get("debug", False):
                        with st.expander("🔧 调试信息"):
                            st.json(raw)

                    if not hits:
                        st.info(f'未找到与 "{query}" 相关的结果，请尝试其他关键词。')
                        st.session_state.search_results = []
                        st.session_state.search_query_display = query
                    else:
                        st.success(f'找到 {len(hits)} 条与 "*{query}*" 相关的结果')
                        st.session_state.search_results = hits
                        st.session_state.search_query_display = query

                except Exception as e:
                    st.error(f"搜索失败: {e}")
                    with st.expander("错误详情"):
                        st.code(traceback.format_exc())
                    st.session_state.search_results = None

    # 显示搜索结果
    hits = st.session_state.search_results
    if hits is not None and len(hits) > 0:
        for i, hit in enumerate(hits, 1):
            similarity = hit.get("similarity", 0)
            sim_pct = f"{similarity * 100:.1f}%"

            if similarity > 0.7:
                sim_color, sim_bg = "#137333", "#e6f4ea"
            elif similarity > 0.4:
                sim_color, sim_bg = "#e37400", "#fef7e0"
            else:
                sim_color, sim_bg = "#c5221f", "#fce8e6"

            wing = hit.get("wing", "未知")
            room = hit.get("room", "未知")
            source = hit.get("source_file", "未知")

            with st.container():
                st.markdown(
                    f"""
                    <div class="search-result">
                        <div class="result-header">
                            <div>
                                <span class="wing-badge">翼楼: {wing}</span>
                                <span class="room-badge">房间: {room}</span>
                                <small style="color:#888; margin-left:8px;">源文件: {source}</small>
                            </div>
                            <span class="similarity-badge" style="color:{sim_color}; background:{sim_bg}; padding:4px 10px; border-radius:6px;">
                                相关度 {sim_pct}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                text = hit.get("text", "")
                matched_via = hit.get("matched_via", "")
                matched_label = "（混合匹配，置信度更高）" if "closet" in matched_via else ""
                with st.expander(f"结果 #{i} — 查看内容 {matched_label}", expanded=(i <= 3)):
                    st.code(text, language=None)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if hit.get("created_at"):
                            st.caption(f"创建时间: {hit['created_at']}")
                        if hit.get("bm25_score"):
                            st.caption(f"BM25 得分: {hit['bm25_score']:.3f}")
                    with col_b:
                        st.caption(f"相似度距离: {hit.get('distance', 'N/A')}")
                        st.caption(f"匹配方式: {matched_via}")

                st.markdown("---")

    elif st.session_state.search_results == []:
        pass  # 已显示"未找到"

    elif not query:
        st.markdown("### 👆 在上方输入关键词搜索你的记忆宫殿")
        st.markdown(
            """
            **示例查询:**
            - 什么是记忆系统架构？
            - 这个项目用了哪些技术栈？
            - 隐私保护是如何实现的？
            """
        )


# ---- 底部 ----------------------------------------------------------------
st.markdown("---")
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption(
        "数据完全存储在本地。零 API 调用。零遥测。"
        "记忆宫殿浏览器为社区贡献 — "
        "在 [GitHub](https://github.com/2362461686/mempalace) 上给我们 Star！"
    )
with col_f2:
    debug = st.checkbox("调试模式", value=False, key="debug")
