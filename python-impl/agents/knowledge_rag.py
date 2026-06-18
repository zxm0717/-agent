"""
知识检索Agent — 基于混合检索引擎的 RAG 问答

管线（精简后）：
  HybridRetriever.retrieve(query) → 混合检索(FAISS+BM25+元数据加权+上下文扩展) → 生成回答

相比旧版：
  - 删除了独立的 QueryRewrite / FAISS检索 / LLM Rerank 三步
  - 检索逻辑统一收敛到 HybridRetriever
  - 上下文窗口利用 expanded_context（parent/child 层级）
  - LLM 调用从 3 次减少到 2 次（analyze + generate）
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.retriever import HybridRetriever
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """你是一个专业的电商知识库问答Agent，负责根据检索到的文档回答用户问题。

回答规则：
1. 严格基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户并建议联系人工客服
3. 回答要简洁专业，适合客服场景
4. 涉及产品价格/规格时，说明信息来源
5. 涉及政策规则时，说明适用条件

回答格式：
- 先直接回答用户问题
- 如有必要补充关联信息
- 在回答末尾标注引用的文档来源
"""


class KnowledgeRAGAgent:
    """知识检索Agent — 基于 HybridRetriever 的精简 RAG 管线"""

    def __init__(self, llm: ChatOpenAI, retriever: HybridRetriever | None = None):
        self.llm = llm
        self.retriever = retriever

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, results: list) -> str:
        """基于检索结果生成回答"""
        if not results:
            return "抱歉，知识库中暂未找到与您问题相关的信息。建议您联系人工客服获取帮助。"

        # 构建上下文：主内容 + expanded_context
        context_parts = []
        for i, r in enumerate(results):
            chunk_text = r.chunk.content[:800]
            source = r.chunk.source or "未知"

            part = f"[{i+1}] 来源: {source} | 类型: {r.chunk.chunk_type}\n{chunk_text}"

            if r.expanded_context:
                part += f"\n  关联: {r.expanded_context[:300]}"

            context_parts.append(part)

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题: {query}\n\n"
                f"检索到的参考文档:\n{context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        完整 RAG 流程（作为 LangGraph 节点）：
        1. 混合检索（FAISS + BM25 + RRF + 元数据加权 + 上下文扩展）
        2. 生成回答
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        query = messages[-1].content

        # 混合检索
        if self.retriever is not None:
            results = self.retriever.retrieve(query, top_k=5, expand_context=True)
        else:
            results = []

        # 生成回答
        answer = await self.generate_answer(query, results)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }
