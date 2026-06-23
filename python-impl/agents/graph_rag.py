"""
GraphRAG Agent — 混合检索（向量 + BM25 + 图谱）

结合混合检索引擎（FAISS+BM25）和知识图谱结构推理，回答多跳关系问题。
典型场景：
- "这款手机搭配哪种充电器最好？" → 商品 → 兼容配件 → 规格匹配
- "订单取消后退款要走什么流程？" → 取消政策 → 退款流程 → 前置条件
- "X产品下架了，有什么替代品？" → 商品 → alternative_to → 同类商品

管线（优化后）：
  实体提取 → HybridRetriever(entity_hints=实体名+entity_ids) → 图谱遍历 → 生成回答
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.retriever import HybridRetriever
from memory.knowledge_graph import KnowledgeGraph
from tracing.otel_config import trace_agent_call


# ─── Prompt 模板 ───

ENTITY_EXTRACTION_PROMPT = """你是一个查询分析专家。从用户的电商客服问题中提取关键实体。

识别以下类型的实体：
- 商品名/型号（Product）
- 品牌名（Brand）
- 品类名（Category）
- 规格参数（Spec）
- 政策/服务类型（Policy）
- 业务流程（Process）

只返回 JSON 数组，每个元素包含 name 和 type：
[{"name": "实体名", "type": "Product|Brand|Category|..."}]

用户问题: {query}
"""


GRAPH_RAG_SYSTEM_PROMPT = """你是一个电商智能客服专家，擅长结合产品知识图谱回答用户问题。

回答规则：
1. 基于提供的图结构信息和检索文档回答，不要编造
2. 如果知识库中没有相关信息，明确告知用户
3. 涉及产品推荐时，说明推荐理由（兼容性/性价比依据）
4. 涉及流程时，清晰列出步骤和前置条件
5. 在回答末尾标注信息来源

回答格式：
- 先直接回答核心问题
- 如有多个关联信息，用编号列出
- 如有风险或注意事项，加粗提示
"""


# ─── GraphRAG Agent ───

class GraphRAGAgent:
    """
    混合检索 Agent：HybridRetriever（FAISS+BM25+元数据加权） + 图谱遍历。

    Usage:
        agent = GraphRAGAgent(llm, retriever, knowledge_graph)
        result = await agent.process(state)
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: HybridRetriever | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

    @trace_agent_call("graphrag_extract_entities")
    async def extract_query_entities(self, query: str) -> list[dict[str, str]]:
        """从用户问题中提取关键实体"""
        messages = [
            HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(query=query)),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            entities = json.loads(content)
            return entities if isinstance(entities, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    @trace_agent_call("graphrag_graph_traverse")
    async def graph_traverse(
        self, entities: list[dict[str, str]], max_hops: int = 2
    ) -> dict[str, Any]:
        """
        在图谱中匹配实体并遍历邻居。
        返回匹配到的实体 IDs 和子图上下文文本。
        """
        matched_entity_ids: list[str] = []

        for ent in entities:
            ent_name = ent.get("name", "")
            ent_type = ent.get("type", "")

            candidates = self.knowledge_graph.search_entities(ent_name)

            if not candidates:
                # 放宽条件：用名称片段搜索
                if len(ent_name) >= 2:
                    candidates = self.knowledge_graph.search_entities(ent_name[:2])

            # 优先匹配同类型
            for cand in candidates:
                if ent_type and cand.get("type") == ent_type:
                    matched_entity_ids.append(cand["id"])
                    break
            else:
                if candidates:
                    matched_entity_ids.append(candidates[0]["id"])

        if not matched_entity_ids:
            return {"entity_ids": [], "graph_context": ""}

        # 获取子图上下文
        graph_context = self.knowledge_graph.get_subgraph_context(
            matched_entity_ids, max_hops=max_hops,
        )

        # 获取邻居详情
        neighbors: list[dict[str, Any]] = []
        for eid in matched_entity_ids[:3]:
            nbrs = self.knowledge_graph.get_neighbors(eid, hops=max_hops)
            for n in nbrs:
                if n["id"] not in [x["id"] for x in neighbors] + matched_entity_ids:
                    neighbors.append(n)

        return {
            "entity_ids": matched_entity_ids,
            "graph_context": graph_context,
            "neighbors": neighbors[:20],
        }

    @trace_agent_call("graphrag_generate")
    async def generate_answer(
        self,
        query: str,
        graph_context: str,
        retrieval_results: list,
        history_messages: list | None = None,
    ) -> str:
        """结合图谱上下文 + 混合检索结果 + 对话历史生成最终回答"""
        doc_text = ""
        if retrieval_results:
            parts = []
            for r in retrieval_results[:5]:
                source = r.chunk.source or "未知"
                part = f"来源: {source}\n内容: {r.chunk.content[:800]}"
                if r.expanded_context:
                    part += f"\n关联: {r.expanded_context[:200]}"
                parts.append(part)
            doc_text = "\n\n---\n\n".join(parts)

        context_parts = []
        if graph_context:
            context_parts.append(graph_context)
        if doc_text:
            context_parts.append(f"## 检索相关文档\n\n{doc_text}")

        combined_context = "\n\n".join(context_parts) if context_parts else "暂无相关知识库信息。"

        # 构建对话历史（最近 3 轮，用于多轮理解）
        history_text = ""
        if history_messages:
            from langchain_core.messages import HumanMessage
            recent = history_messages[-6:]  # 最近 3 轮
            if len(recent) > 1:
                lines = []
                for m in recent:
                    role = "用户" if isinstance(m, HumanMessage) else "客服"
                    lines.append(f"{role}: {m.content[:200]}")
                history_text = "## 对话历史\n" + "\n".join(lines) + "\n\n"

        messages = [
            SystemMessage(content=GRAPH_RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"{history_text}"
                f"用户问题: {query}\n\n"
                f"参考信息:\n{combined_context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("graphrag_hybrid_retrieve")
    async def hybrid_retrieve(self, query: str) -> dict[str, Any]:
        """
        完整混合检索管线（优化后）：
        1. 从 query 提取实体
        2. 通过 HybridRetriever 做 FAISS+BM25 混合检索（带实体加权）
        3. 图谱遍历获取结构化关系
        """
        entities = await self.extract_query_entities(query)

        # 将实体名匹配为 KG entity_ids
        entity_ids: list[str] = []
        for ent in entities:
            ent_name = ent.get("name", "")
            if self.knowledge_graph:
                matches = self.knowledge_graph.search_entities(ent_name)
                for m in matches[:2]:
                    eid = m.get("id", "")
                    if eid and eid not in entity_ids:
                        entity_ids.append(eid)

        # 用 HybridRetriever 做混合检索（带实体加权）
        if self.retriever is not None:
            entity_names = [e.get("name", "") for e in entities if e.get("name")]
            retrieval_results = self.retriever.retrieve_for_graph_rag(
                query=query,
                entities=entity_names,
                entity_ids=entity_ids,
                top_k=5,
            )
        else:
            retrieval_results = []

        # 图谱遍历
        graph_result = await self.graph_traverse(entities)

        return {
            "entities": entities,
            "retrieval_results": retrieval_results,
            "graph_context": graph_result.get("graph_context", ""),
            "neighbors": graph_result.get("neighbors", []),
            "matched_entity_ids": graph_result.get("entity_ids", []),
        }

    @trace_agent_call("graph_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        作为 LangGraph 节点处理状态（带对话历史上下文）。
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        user_query = messages[-1].content if messages else ""

        # 对话历史（不含当前消息，供生成阶段理解多轮上下文）
        history = messages[:-1] if len(messages) > 1 else []

        # 运行混合检索
        retrieval_result = await self.hybrid_retrieve(user_query)

        # 生成回答
        answer = await self.generate_answer(
            query=user_query,
            graph_context=retrieval_result.get("graph_context", ""),
            retrieval_results=retrieval_result.get("retrieval_results", []),
            history_messages=history,
        )

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "graph_rag": answer,
            },
            "graph_context": retrieval_result,
        }
