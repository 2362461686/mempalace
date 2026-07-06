<div align="center">

<img src="assets/mempalace_logo.png" alt="MemPalace" width="240">

# MemPalace（记忆宫殿）

**🌍 [English](README.md) | [中文文档](README.zh.md)**

本地优先的 AI 记忆系统。逐字存储、可插拔后端、LongMemEval R@5 达 96.6%——零 API 调用。

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]

</div>

---

## 这是什么？

MemPalace 是一个**本地 AI 记忆系统**。它把你的对话历史和项目文件逐字存储为可语义搜索的记忆，而不是做摘要或改写。结构化的索引让搜索可以限定范围：

- **翼楼（Wings）** = 大类（人、项目、主题）
- **房间（Rooms）** = 时间维度的分组（天、会话）
- **抽屉（Drawers）** = 完整原文，一字不改

所有数据都在你的本地机器上，没有 API 调用，没有云同步，没有隐私泄露。

---

## 快速上手（3 分钟）

### 1. 克隆 & 安装

```bash
git clone https://github.com/2362461686/mempalace.git
cd mempalace
pip install -e ".[dev]"
```

### 2. 创建你的第一个记忆宫殿

```bash
# 初始化宫殿
mkdir my-palace
python -c "
from mempalace.palace import get_collection
get_collection('my-palace', create=True)
"
```

### 3. 把内容存进去

```bash
# 挖掘项目文件
mempalace mine ~/my-project --target my-palace

# 挖掘 Claude Code 对话记录
mempalace mine ~/.claude/projects/ --mode convos --target my-palace
```

### 4. 搜索你的记忆

```bash
mempalace search my-palace "我上周修改了什么bug？"
```

---

## 社区贡献（本 Fork 新增功能）

### 🧠 BGE 中文嵌入模型

MemPalace 原本只支持 MiniLM（英文）和 EmbeddingGemma（多语言），对中文优化不够。本 Fork 添加了 **BAAI BGE 中文嵌入模型**支持：

| 模型 | 维度 | 大小 | 适用场景 |
|------|------|------|----------|
| `bge-small-zh` | 512 | ~100MB | 快速演示、轻量部署 |
| `bge-base-zh` | 768 | ~400MB | 生产环境推荐 |
| `bge-large-zh` | 1024 | ~1.3GB | 最佳检索质量 |

```bash
pip install mempalace[bge]
# 在 ~/.mempalace/config.json 中设置:
# {"embedding_model": "bge-small-zh"}
```

**技术要点：** BGE 模型使用 query instruction prefix 区分查询和文档的嵌入向量，大幅提升中文检索精度。实现了 ChromaDB 兼容的 EmbeddingFunction 接口，可以无缝接入现有 palace。

### 🖥️ Streamlit 可视化浏览器

把命令行工具变成了**网页版记忆浏览器**，方便查看和演示：

```bash
pip install mempalace[ui]
streamlit run apps/streamlit_app.py
```

功能：
- 语义搜索框 → 查看搜索结果和相似度分数
- 按 Wings / Rooms / Drawers 三层结构浏览
- 统计面板：总 drawer 数、嵌入模型信息
- 美观的卡片式结果展示

### 🦜 LangChain 集成

让 MemPalace 作为 **LangChain 的记忆后端**，任何 Agent 都可以持久化记忆：

```bash
pip install mempalace[langchain]
python integrations/langchain/example.py
```

**MemPalaceRetriever** — 实现 LangChain `BaseRetriever` 接口的多功能检索器：
```python
from integrations.langchain import MemPalaceRetriever

retriever = MemPalaceRetriever(palace_path="my-palace", k=10)
docs = retriever.invoke("什么是记忆系统架构？")
for doc in docs:
    print(f"[{doc.metadata['similarity']:.2f}] {doc.page_content[:100]}")
```

**MemPalaceChatMemory** — 持久化多轮对话记忆（支持语义召回）：
```python
from integrations.langchain import MemPalaceChatMemory

memory = MemPalaceChatMemory(
    palace_path="my-palace",
    wing="my-agent",
    semantic_recall=True,  # 开启语义召回
    semantic_k=5,
)

# 保存对话
memory.save_context(
    {"input": "什么是记忆宫殿？"},
    {"output": "记忆宫殿是一种古老的记忆术..."}
)

# 加载历史（包含语义相关的过去对话）
history = memory.load_memory_variables({"input": "架构"})
```

---

## 项目结构

```
mempalace/
├── mempalace/              # 核心代码
│   ├── embedding.py        # 嵌入模型工厂
│   ├── embedding_bge.py    # ← BGE 中文模型（新增）
│   ├── searcher.py         # 混合搜索（BM25 + 向量）
│   ├── mcp_server.py       # MCP 服务器（35个工具）
│   ├── knowledge_graph.py  # 时序知识图谱
│   └── backends/           # 可插拔存储后端
├── apps/
│   └── streamlit_app.py    # ← 可视化浏览器（新增）
├── integrations/
│   └── langchain/          # ← LangChain 集成（新增）
│       ├── retriever.py    #   BaseRetriever 实现
│       ├── memory.py       #   ChatMemory 实现
│       └── example.py      #   完整 Demo 脚本
├── demo_data/
│   ├── run_demo.py         #   一键运行所有 Demo
│   └── mypalace/           #   示例 palace 数据
└── pyproject.toml
```

---

## 运行 Demo

```bash
# 一键运行所有 Demo（验证 BGE + Retriever + Memory + UI）
cd mempalace
python demo_data/run_demo.py
```

Demo 输出示例：

```
DEMO 1: BGE Chinese Embedding Models
  bge-small-zh: 512-dim, BAAI/bge-small-zh-v1.5
  bge-base-zh:  768-dim, BAAI/bge-base-zh-v1.5
  bge-large-zh: 1024-dim, BAAI/bge-large-zh-v1.5

DEMO 2: LangChain Retriever
  [Q] What is the memory system architecture?
  [1] agent_memory_design.md (sim=0.91)
  [2] agent_memory_design.md (sim=0.55)

DEMO 3: LangChain Chat Memory
  对话已保存 → 9 messages loaded (including semantic recall)

DEMO 4: Streamlit UI
  streamlit run apps/streamlit_app.py

DEMO 5: Direct Search Quality Check
  English: 'agent memory system design' → 3 hits (top sim=0.910)
  Chinese: 'memory architecture' → 3 hits (top sim=0.466)
```

---

## 对面试官说的技术亮点

1. **BGE 嵌入模型**：实现了 ChromaDB 的 EmbeddingFunction 协议，理解 query/document 嵌入向量的区分、tokenizer 差异、pooling strategy
2. **LangChain 集成**：实现了 `BaseRetriever` 和自定义 ChatMemory，理解 Agent 架构中 Memory 的角色
3. **混合搜索**：BM25 关键词匹配 + 向量语义搜索的加权融合排序
4. **本地优先隐私设计**：零 API 调用，嵌入和检索完全在本地完成

---

## 许可

MIT — 详见 [LICENSE](LICENSE)。

原项目：[MemPalace/mempalace](https://github.com/MemPalace/mempalace)

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-3.5.0-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/MemPalace/mempalace/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/MemPalace/mempalace/blob/main/LICENSE
