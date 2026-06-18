"""
混合检索引擎 — FAISS(稠密) + BM25(稀疏) → RRF 融合 → 元数据加权 → 上下文扩展

索引构建阶段产出了三套索引：
  FAISS      — 稠密语义向量 (LongTermMemory)
  SparseIndex — BM25 关键词倒排
  Chunk 元数据 — chunk_type / entity_ids / section_title / parent_id

本模块在查询时统一调用这三套索引，做 RRF 融合 + 元数据加权 + 层级上下文扩展，
替代原先各 Agent 各自调 FAISS 的分散检索逻辑。

Usage:
    retriever = HybridRetriever(llm, long_term_memory, sparse_index, knowledge_graph)
    results = retriever.retrieve("X1充电器多少钱", top_k=5)

     # graph_rag 侧带实体提示
    results = retriever.retrieve(query, entity_hints=["x1_phone", "charger_65w"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from memory.long_term import Chunk, LongTermMemory
from memory.sparse_index import SparseIndex
from memory.knowledge_graph import KnowledgeGraph


# ─── 数据结构 ───

@dataclass
class QueryAnalysis:
    """一次 LLM 调用产出的查询分析结果"""
    original: str
    rewritten: str = ""
    entities: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    chunk_type_hint: str | None = None
    is_comparison: bool = False
    is_multi_hop: bool = False

    @classmethod
    def empty(cls, query: str) -> QueryAnalysis:
        return cls(original=query, rewritten=query)


@dataclass
class RetrievalResult:
    """单条检索结果，附带融合分数和扩展上下文"""
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    expanded_context: str = ""
    matched_entities: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_context_prompt(self, max_chars: int = 800) -> str:
        """生成 LLM 可用的上下文字符串"""
        parts = [f"[来源: {self.chunk.source}] {self.chunk.content[:max_chars]}"]
        if self.expanded_context:
            parts.append(f"\n(关联上下文) {self.expanded_context[:300]}")
        return "\n".join(parts)


# ─── Query 分析 Prompt ───

QUERY_ANALYSIS_PROMPT = """你是一个电商查询分析专家。分析用户的客服问题，提取关键信息以辅助检索引擎。

## 任务

1. **改写查询**：将口语化问题改写为更适合向量/B关键词检索的短句。补充品牌名、正式型号名、同义词。中英文专业术语保留原名。
2. **提取实体**：列出问题中涉及的商品名/型号/配件名/政策类型/品牌名
3. **推断检索类型**：判断最适合检索的 chunk 类型
   - "sku_fact": 商品参数、价格、规格
   - "policy_rule": 政策规则、退换货条件、流程步骤
   - 如果无法判断，返回 null
4. **判断问题类型**：是否为对比类（含"对比/vs/哪个好"）、是否为多跳推理（需要多个实体关联才能回答）

## 输出格式（严格 JSON）

{
  "rewritten": "改写后的检索短句",
  "entities": ["实体1", "实体2"],
  "chunk_type_hint": "sku_fact|policy_rule|null",
  "is_comparison": true/false,
  "is_multi_hop": true/false
}

## 用户问题
{query}

只返回 JSON，不要其他内容。"""


# ─── HybridRetriever ───

class HybridRetriever:
    """
    统一混合检索引擎。

    管线：
    1. _analyze_query    — 一次 LLM 分析（实体 + chunk_type + rewrite）
    2. _dense_search     — FAISS 稠密向量检索
    3. _sparse_search    — BM25 关键词检索
    4. _rrf_fuse         — RRF 分数融合
    5. _apply_boosting   — chunk_type + entity_id 加权
    6. _expand_contexts  — parent/child 层级扩展
    """

    # chunk_type → boost 倍率
    CHUNK_TYPE_BOOST: dict[str, float] = {
        "sku_fact": 1.3,
        "policy_rule": 1.3,
        "heading": 1.1,
    }
    ENTITY_MATCH_BOOST = 1.5
    SECTION_MATCH_BOOST = 1.2
    RRF_K = 60

    def __init__(
        self,
        llm: Any = None,
        long_term_memory: LongTermMemory | None = None,
        sparse_index: SparseIndex | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.llm = llm
        self.ltm = long_term_memory or LongTermMemory()
        self.sparse = sparse_index
        self.kg = knowledge_graph

    # ─── 主入口 ───

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_hints: list[str] | None = None,
        expand_context: bool = True,
    ) -> list[RetrievalResult]:
        """
        混合检索主入口。

        Args:
            query: 用户原始查询
            top_k: 返回结果数
            entity_hints: 外部传入的实体 ID 列表（graph_rag 提取后用）
            expand_context: 是否做层级上下文扩展

        Returns:
            RetrievalResult 列表，按融合分数降序
        """
        # Step 1: Query 分析
        analysis = self._analyze_query(query)

        # 合并外部实体提示
        if entity_hints:
            analysis.entity_ids = list(set(analysis.entity_ids + entity_hints))

        # 检查索引可用性
        has_dense = self.ltm._index is not None and len(self.ltm.chunks) > 0
        has_sparse = self.sparse is not None and self.sparse._doc_count > 0

        if not has_dense and not has_sparse:
            return []

        # Step 2-3: 并行检索
        dense_results: dict[str, tuple[Chunk, float]] = {}
        sparse_results: dict[str, tuple[Chunk, float]] = {}

        if has_dense:
            dense_results = self._dense_search(analysis.rewritten, top_k * 2)

        if has_sparse:
            sparse_results = self._sparse_search(analysis.rewritten, top_k * 2)

        # Step 4: RRF 融合
        fused = self._rrf_fuse(dense_results, sparse_results, analysis.rewritten)

        # Step 5: 元数据加权
        fused = self._apply_boosting(fused, analysis)

        # Step 6: 层级上下文扩展
        if expand_context:
            self._expand_contexts(fused)

        return fused[:top_k]

    def retrieve_with_analysis(
        self,
        analysis: QueryAnalysis,
        top_k: int = 5,
        expand_context: bool = True,
    ) -> list[RetrievalResult]:
        """
        跳过 _analyze_query 的检索入口（调用方已有 QueryAnalysis 时使用）。
        graph_rag 可以先做实体提取再传入，避免重复 LLM 调用。
        """
        has_dense = self.ltm._index is not None and len(self.ltm.chunks) > 0
        has_sparse = self.sparse is not None and self.sparse._doc_count > 0

        if not has_dense and not has_sparse:
            return []

        dense_results: dict[str, tuple[Chunk, float]] = {}
        sparse_results: dict[str, tuple[Chunk, float]] = {}

        if has_dense:
            dense_results = self._dense_search(analysis.rewritten or analysis.original, top_k * 2)
        if has_sparse:
            sparse_results = self._sparse_search(analysis.rewritten or analysis.original, top_k * 2)

        fused = self._rrf_fuse(dense_results, sparse_results, analysis.rewritten or analysis.original)
        fused = self._apply_boosting(fused, analysis)

        if expand_context:
            self._expand_contexts(fused)

        return fused[:top_k]

    # ─── Query 分析 ───

    def _analyze_query(self, query: str) -> QueryAnalysis:
        """一次 LLM 调用：实体提取 + chunk_type 推断 + query rewrite"""
        if self.llm is None:
            return QueryAnalysis.empty(query)

        try:
            from langchain_core.messages import HumanMessage

            response = self.llm.invoke([
                HumanMessage(content=QUERY_ANALYSIS_PROMPT.format(query=query)),
            ])
            content = response.content.strip()

            # 清理 markdown 代码块
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            data = json.loads(content)
        except (json.JSONDecodeError, Exception):
            return QueryAnalysis.empty(query)

        analysis = QueryAnalysis(
            original=query,
            rewritten=data.get("rewritten", query),
            entities=data.get("entities", []),
            chunk_type_hint=data.get("chunk_type_hint"),
            is_comparison=data.get("is_comparison", False),
            is_multi_hop=data.get("is_multi_hop", False),
        )

        # 用 KG 匹配实体名 → entity_ids
        if self.kg is not None and analysis.entities:
            for ent_name in analysis.entities:
                matches = self.kg.search_entities(ent_name)
                for m in matches[:2]:  # 每种实体最多取 2 个匹配
                    eid = m.get("id", "")
                    if eid and eid not in analysis.entity_ids:
                        analysis.entity_ids.append(eid)

        return analysis

    # ─── 稠密检索 ───

    def _dense_search(self, query: str, top_k: int) -> dict[str, tuple[Chunk, float]]:
        """FAISS 向量检索"""
        results: dict[str, tuple[Chunk, float]] = {}
        raw = self.ltm.search(query, top_k=top_k)
        for r in raw:
            chunk_id = r.get("chunk_id", "")
            if chunk_id:
                # 从 ltm.chunks 找到原始 Chunk 对象
                for c in self.ltm.chunks:
                    if c.chunk_id == chunk_id:
                        results[chunk_id] = (c, r.get("score", 0.0))
                        break
        return results

    # ─── 稀疏检索 ───

    def _sparse_search(self, query: str, top_k: int) -> dict[str, tuple[Chunk, float]]:
        """BM25 关键词检索"""
        results: dict[str, tuple[Chunk, float]] = {}
        if self.sparse is None:
            return results

        ranked = self.sparse.search(query, top_k=top_k)
        for chunk_id, score in ranked:
            for c in self.ltm.chunks:
                if c.chunk_id == chunk_id:
                    results[chunk_id] = (c, score)
                    break
        return results

    # ─── RRF 融合 ───

    def _rrf_fuse(
        self,
        dense: dict[str, tuple[Chunk, float]],
        sparse: dict[str, tuple[Chunk, float]],
        query: str,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion。

        score(d) = Σ 1 / (k + rank_i)
        k=60 是常用默认值，对排名差异不敏感。
        """
        # 按分数排序得到排名
        dense_ranked = sorted(dense.items(), key=lambda x: x[1][1], reverse=True)
        sparse_ranked = sorted(sparse.items(), key=lambda x: x[1][1], reverse=True)

        # RRF 分数
        rrf_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk_id, (chunk, score)) in enumerate(dense_ranked, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)
            dense_scores[chunk_id] = score
            chunk_map[chunk_id] = chunk

        for rank, (chunk_id, (chunk, score)) in enumerate(sparse_ranked, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)
            sparse_scores[chunk_id] = score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

        results = [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=rrf_scores[cid],
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
            )
            for cid in rrf_scores
        ]

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ─── 元数据加权 ───

    def _apply_boosting(
        self, results: list[RetrievalResult], analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """在 RRF 分数上叠加 chunk_type + entity_id 加权"""
        for r in results:
            chunk = r.chunk

            # 1. chunk_type 匹配
            if analysis.chunk_type_hint:
                boost = self.CHUNK_TYPE_BOOST.get(chunk.chunk_type, 1.0)
                r.score *= boost

            # 2. entity_id 匹配
            if analysis.entity_ids:
                chunk_entities = chunk.metadata.get("entity_ids", [])
                if any(eid in chunk_entities for eid in analysis.entity_ids):
                    r.score *= self.ENTITY_MATCH_BOOST
                    r.matched_entities = [
                        eid for eid in analysis.entity_ids if eid in chunk_entities
                    ]

            # 3. section_title 匹配（部分匹配）
            if chunk.section_title and analysis.rewritten:
                title_words = set(chunk.section_title)
                query_words = set(analysis.rewritten)
                overlap = title_words & query_words
                if len(overlap) >= 2:
                    r.score *= self.SECTION_MATCH_BOOST

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ─── 上下文扩展 ───

    def _expand_contexts(self, results: list[RetrievalResult]):
        """对每个结果注入 parent/child 层级上下文"""
        # 构建 chunk_id → Chunk 映射
        chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in self.ltm.chunks}

        for r in results:
            parts = []

            # parent 上下文
            if r.chunk.parent_id and r.chunk.parent_id in chunk_map:
                parent = chunk_map[r.chunk.parent_id]
                parts.append(f"[章节] {parent.content[:300]}")

            # 如果本身是 section，附带子 chunk 摘要
            if r.chunk.chunk_type == "section":
                child_count = r.chunk.metadata.get("child_count", 0)
                if child_count > 0:
                    children = [
                        c for c in self.ltm.chunks
                        if c.parent_id == r.chunk.chunk_id
                    ][:3]
                    if children:
                        child_summary = " | ".join(
                            c.content[:120] for c in children
                        )
                        parts.append(f"[要点] {child_summary}")

            r.expanded_context = "\n".join(parts)

    # ─── 便捷方法 ───

    def retrieve_for_graph_rag(
        self,
        query: str,
        entities: list[str],
        entity_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        graph_rag 专用入口：携带已提取的实体信息。
        跳过 _analyze_query，直接用已有 entity_ids 做加权。
        """
        analysis = QueryAnalysis(
            original=query,
            rewritten=query,
            entities=entities,
            entity_ids=entity_ids or [],
        )
        return self.retrieve_with_analysis(analysis, top_k=top_k)

    def search_raw(
        self, query: str, top_k: int = 5,
    ) -> list[RetrievalResult]:
        """零 LLM 的纯检索（不做 query 分析，不做加权，适合内部调试）"""
        has_dense = self.ltm._index is not None and len(self.ltm.chunks) > 0
        has_sparse = self.sparse is not None and self.sparse._doc_count > 0

        dense_results: dict[str, tuple[Chunk, float]] = {}
        sparse_results: dict[str, tuple[Chunk, float]] = {}

        if has_dense:
            dense_results = self._dense_search(query, top_k * 2)
        if has_sparse:
            sparse_results = self._sparse_search(query, top_k * 2)

        fused = self._rrf_fuse(dense_results, sparse_results, query)
        return fused[:top_k]

    def get_statistics(self) -> dict[str, Any]:
        """检索器状态"""
        return {
            "dense_available": self.ltm._index is not None and len(self.ltm.chunks) > 0,
            "sparse_available": self.sparse is not None and self.sparse._doc_count > 0,
            "chunk_count": len(self.ltm.chunks),
            "sparse_term_count": len(self.sparse._inverted) if self.sparse else 0,
            "kg_available": self.kg is not None,
            "kg_node_count": self.kg._graph.number_of_nodes() if self.kg else 0,
        }
