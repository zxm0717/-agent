# 用户查询全链路优化方案

> 基于索引构建（语义分块 + BM25 + 结构化编码）的查询端配套优化

## 现状诊断

```
用户查询 → Supervisor → 路由到单个Agent ──→ 合规审查 → 汇总
                │                    │
                ▼                    ▼
         意图路由器          knowledge_rag (FAISS only)
                           graph_rag     (FAISS + KG)
```

| # | 问题 | 位置 |
|---|------|------|
| 1 | **BM25 建了不用** — `SparseIndex` 启动时建好，查询端从未调用 | `api/main.py:205` vs `knowledge_rag.py:58` |
| 2 | **无混合检索融合** — FAISS(稠密) 和 BM25(稀疏) 各跑各的，没有 RRF | 缺少 `Retriever` 抽象 |
| 3 | **Chunk 层级浪费** — SemanticChunker 的 parent/child 关系在检索时完全不用 | `long_term.py:242-248` |
| 4 | **单路由瓶颈** — Supervisor 只走一条路，错过互补信息 | `supervisor.py:196-206` |
| 5 | **元数据不参与检索** — `chunk_type`、`entity_ids` 存了但不过滤/加权 | `Chunk.metadata` 闲置 |
| 6 | **Query Rewrite 太弱** — 仅 LLM 转写，无实体提取和术语扩展 | `knowledge_rag.py:33-38` |

---

## 总架构（优化后）

```
用户 Query
    │
    ▼
Supervisor (多路并行)
    │
    ├──→ knowledge_rag ──→ HybridRetriever.retrieve(query) ──→ generate
    │         │                       │
    │         │         ┌─────────────┼─────────────┐
    │         │         ▼             ▼             ▼
    │         │    FAISS(dense)  BM25(sparse)  元数据加权
    │         │         │             │             │
    │         │         └───── RRF 融合 ────────────┘
    │         │                    │
    │         │             层级上下文扩展
    │         │
    ├──→ graph_rag   ──→ 实体提取 → HybridRetriever(entity_hints=[]) → KG遍历 → generate
    │
    └──→ ticket/vision (不变)
```

---

## Phase 1：统一混合检索引擎

**新增文件：`memory/retriever.py`**

### 数据结构

```python
@dataclass
class QueryAnalysis:
    original: str              # 原始 query
    rewritten: str             # LLM 改写后的检索词
    entities: list[str]        # 提取的实体名 ["X1", "65W充电器"]
    entity_ids: list[str]      # KG 匹配到的 ID ["x1_phone", "charger_65w"]
    chunk_type_hint: str|None  # "sku_fact" | "policy_rule" | None
    is_comparison: bool
    is_multi_hop: bool

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float | None
    sparse_score: float | None
    expanded_context: str
    matched_entities: list[str]
```

### 管线

```
HybridRetriever.retrieve(query, top_k, entity_hints?)

  ① _analyze_query(query)
       一次 LLM 调用完成：
       - query rewrite（补充品牌名、型号）
       - 实体提取 + KG 匹配 entity_ids
       - chunk_type 推断

  ② 并行检索
       FAISS.search(rewritten, 2×top_k)     → dense_results
       SparseIndex.search(rewritten, 2×top_k) → sparse_results

  ③ RRF 融合
       score(d) = Σ 1/(k + rank_i)    k=60

  ④ 元数据加权
       chunk_type 匹配 hint   → ×1.3
       entity_id 命中         → ×1.5
       section_title 匹配     → ×1.2

  ⑤ 层级上下文扩展
       对每个 top-K chunk:
         if parent_id: 附带 parent 摘要
         子 chunk 附带前后 sibling

  ⑥ 返回 list[RetrievalResult]
```

---

## Phase 2：改造 knowledge_rag

**修改文件：`agents/knowledge_rag.py`**

```
现:  QueryRewrite → FAISS检索 → LLM Rerank → Generate
改:  HybridRetriever.retrieve(query) → Generate
```

- 删掉 `rewrite_query`、`retrieve_documents`、`rerank_documents`
- `process()` 直接调用 `retriever.retrieve(query)`
- `generate_answer` 注入 expanded_context

---

## Phase 3：改造 graph_rag

**修改文件：`agents/graph_rag.py`**

```
现:  EntityExtraction → 实体名搜FAISS → KG遍历 → Generate
改:  EntityExtraction → HybridRetriever(entity_hints=实体名) → KG遍历 → Generate
```

- `hybrid_retrieve` 调用 `retriever.retrieve(query, entity_hints=matched_ids)`
- 实体匹配触发元数据加权

---

## Phase 4：Supervisor 多路并行

**修改文件：`agents/supervisor.py`**

```
现:  单路由 → knowledge_rag 或 graph_rag（二选一）
改:  并行 → knowledge_rag ∥ graph_rag → 合规审查 → 结果融合
```

- `synthesize_response` 做去重/互补融合
- 同一事实双路命中则合并，互补信息则拼接

---

## 文件变更清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `memory/retriever.py` | **新增** | HybridRetriever + QueryAnalysis + RetrievalResult + RRF |
| `agents/knowledge_rag.py` | **重写** | 精简为 retriever.retrieve → generate |
| `agents/graph_rag.py` | **修改** | hybrid_retrieve 改用 HybridRetriever |
| `agents/supervisor.py` | **修改** | 单路由 → 多路并行 |
| `api/main.py` | **修改** | 初始化 HybridRetriever 注入 Agent |
| `memory/__init__.py` | **修改** | 导出新增类 |

**不改的文件：** `ticket_handler.py`、`compliance_checker.py`、`vision.py`、`intent_router.py`

---

## 效果对比

| 维度 | 现状 | 优化后 |
|------|------|--------|
| 检索通路 | FAISS only | FAISS + BM25 RRF 融合 |
| LLM 调用/查询 | 3次（rewrite + rerank + generate）| 2次（analyze + generate）|
| chunk_type 感知 | 无 | metadata 加权 |
| 上下文窗口 | 单 chunk text | chunk + expanded_context |
| KG 联动 | 单向 | 双向（entity_id → FAISS+BM25）|
| Agent 并发 | 串行单路由 | knowledge_rag ∥ graph_rag |
