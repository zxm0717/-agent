"""
GraphRAG Agent — 混合检索（向量 + 图谱）

结合向量语义搜索和知识图谱结构推理，回答多跳关系问题。
典型场景：
- "这款手机搭配哪种充电器最好？" → 商品 → 兼容配件 → 规格匹配
- "订单取消后退款要走什么流程？" → 取消政策 → 退款流程 → 前置条件
- "X产品下架了，有什么替代品？" → 商品 → alternative_to → 同类商品

管线：实体提取 → 向量检索 → 图谱遍历 → 上下文合并 → 生成回答
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.long_term import LongTermMemory
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
3. 涉及产品推荐时，说明推荐理由（兼容性/性价比/用户评价依据）
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
    混合检索 Agent：向量搜索 + 图谱遍历。

    Usage:
        agent = GraphRAGAgent(llm, long_term_memory, knowledge_graph)
        result = await agent.process(state)
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        long_term_memory: LongTermMemory | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

    @trace_agent_call("graphrag_extract_entities")
    async def extract_query_entities(self, query: str) -> list[dict[str, str]]:
        """从用户问题中提取关键实体"""
        messages = [
            HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(query=query)),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            # 清理可能的 markdown 代码块包装
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            entities = json.loads(content)
            return entities if isinstance(entities, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    @trace_agent_call("graphrag_vector_search")
    async def vector_search(
        self, entities: list[dict[str, str]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """对每个实体做向量检索，合并去重"""
        seen_ids: set[str] = set()
        all_docs: list[dict[str, Any]] = []

        # 用实体 name 拼接成查询语句
        queries = [e.get("name", "") for e in entities if e.get("name")]
        if not queries:
            queries = [""]

        for query in queries:
            docs = self.long_term_memory.search(query, top_k=top_k)
            for doc in docs:
                doc_id = doc.get("id", doc.get("content", "")[:60])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        return all_docs

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

            # 先按名称搜索图谱中的实体
            candidates = self.knowledge_graph.search_entities(ent_name)

            if not candidates:
                # 放宽条件，只用名称片段搜索（取前两个字）
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
        for eid in matched_entity_ids[:3]:  # 限制起点，避免爆炸
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
        vector_docs: list[dict[str, Any]],
    ) -> str:
        """结合图谱上下文和向量检索文档生成最终回答"""
        doc_text = ""
        if vector_docs:
            doc_text = "\n\n---\n\n".join(
                f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')[:800]}"
                for doc in vector_docs[:5]
            )

        context_parts = []
        if graph_context:
            context_parts.append(graph_context)
        if doc_text:
            context_parts.append(f"## 检索相关文档\n\n{doc_text}")

        combined_context = "\n\n".join(context_parts) if context_parts else "暂无相关知识库信息。"

        messages = [
            SystemMessage(content=GRAPH_RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题: {query}\n\n"
                f"参考信息:\n{combined_context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("graphrag_hybrid_retrieve")
    async def hybrid_retrieve(self, query: str) -> dict[str, Any]:
        """
        完整混合检索管线：
        1. 从 query 提取实体
        2. 向量检索获取语义相关文档
        3. 图谱遍历获取结构化关系
        """
        entities = await self.extract_query_entities(query)

        vector_docs = await self.vector_search(entities)

        graph_result = await self.graph_traverse(entities)

        return {
            "entities": entities,
            "vector_docs": vector_docs,
            "graph_context": graph_result.get("graph_context", ""),
            "neighbors": graph_result.get("neighbors", []),
            "matched_entity_ids": graph_result.get("entity_ids", []),
        }

    @trace_agent_call("graph_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        作为 LangGraph 节点处理状态。
        从 state 中读取用户最后一条消息，执行混合检索并生成回答。
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        # 取最后一条用户消息
        user_query = messages[-1].content if messages else ""

        # 运行混合检索
        retrieval_result = await self.hybrid_retrieve(user_query)

        # 生成回答
        answer = await self.generate_answer(
            query=user_query,
            graph_context=retrieval_result.get("graph_context", ""),
            vector_docs=retrieval_result.get("vector_docs", []),
        )

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "graph_rag": answer,
            },
            "graph_context": retrieval_result,
        }
