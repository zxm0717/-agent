"""
Supervisor编排Agent — 中央协调者（v3: 多路并行）

负责接收用户请求，根据意图并行调度多个子Agent，汇总结果返回。
采用LangGraph StateGraph + Send API 实现多路并行 fan-out。

v3 变更：
  - 单路由 → 多路并行（knowledge_rag ∥ graph_rag）
  - sub_results 增加 merge reducer 支持并行结果合并
  - synthesize_response 做去重/互补融合
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from agents.knowledge_rag import KnowledgeRAGAgent
from agents.ticket_handler import TicketHandlerAgent
from agents.compliance_checker import ComplianceCheckerAgent
from agents.graph_rag import GraphRAGAgent
from agents.vision import VisionAgent
from memory.short_term import ShortTermMemory
from memory.knowledge_graph import KnowledgeGraph
from memory.retriever import HybridRetriever
from tracing.otel_config import trace_agent_call


# ─── Reducer: 并行节点 sub_results 合并 ───

def _merge_sub_results(left: dict | None, right: dict | None) -> dict[str, Any]:
    """合并两个并行节点的 sub_results，后到的覆盖同 key"""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    sub_results: Annotated[dict[str, Any], _merge_sub_results]
    compliance_passed: bool
    final_response: str
    current_agent: str
    retry_count: int
    has_images: bool
    images: list[str]
    graph_context: dict[str, Any]
    vision_results: dict[str, Any]


# ─── Preprocess 节点 ───

async def preprocess_input(state: AgentState) -> AgentState:
    """预处理节点（入口）：检测多模态内容"""
    images = state.get("images", [])
    return {
        **state,
        "has_images": len(images) > 0,
    }


# ─── Supervisor Prompt ───

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能客服系统的Supervisor（主管编排Agent）。

可用的子Agent：
- knowledge_rag: 知识库检索和回答（适用：产品参数、价格、政策查询、通用咨询）
- graph_rag: 知识图谱关系推理（适用：配件兼容、替代推荐、多跳关联、生态搭配）
- ticket_handler: 工单创建和查询（适用：退款、投诉、业务办理）
- vision: 多模态图片分析（适用：截图、票据、商品图片）
- compliance_checker: 合规审查和敏感词检测（所有回复必经此节点）

路由决策：
- 图片/截图请求 → vision
- 退款/投诉/工单 → ticket_handler
- 其他全部 → knowledge_rag + graph_rag 并行（双路互补）

根据用户消息，返回应路由到的Agent名称列表。
只返回 JSON 数组，如 ["knowledge_rag", "graph_rag"] 或 ["ticket_handler"] 或 ["vision"]。
"""


# ─── Supervisor节点 ───

class SupervisorNode:
    """Supervisor决策节点"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """分析用户意图，决定路由（基于对话历史上下文）"""
        messages = state["messages"]
        has_images = state.get("has_images", False)

        # 从对话历史中提取最近几轮作为路由上下文
        recent_history = messages[-10:] if len(messages) > 10 else messages
        history_context = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '客服'}: {m.content[:200]}"
            for m in recent_history[:-1]  # 排除最后一条（当前请求）
        )

        routing_prompt = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            *messages,
            HumanMessage(content="请返回应路由到的Agent名称列表（JSON数组）。"),
        ]

        if history_context:
            routing_prompt.insert(
                1,
                SystemMessage(content=f"最近的对话历史: {history_context}"),
            )

        response = await self.llm.ainvoke(routing_prompt)

        # 解析路由决策
        import json
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
            agent_list = json.loads(content)
            if not isinstance(agent_list, list):
                agent_list = ["knowledge_rag", "graph_rag"]
        except (json.JSONDecodeError, Exception):
            # 默认：并行 knowledge_rag + graph_rag
            agent_list = ["knowledge_rag", "graph_rag"]

        # 确保有图片时包含 vision
        if has_images and "vision" not in agent_list:
            agent_list.insert(0, "vision")

        # 过滤无效值
        valid = {"knowledge_rag", "graph_rag", "ticket_handler", "vision", "compliance_check"}
        agent_list = [a for a in agent_list if a in valid]

        if not agent_list:
            agent_list = ["knowledge_rag", "graph_rag"]

        # 意图标记：取第一个作为主意图
        primary_intent = agent_list[0] if agent_list else "knowledge_rag"

        return {
            **state,
            "intent": primary_intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> AgentState:
        """汇总并行子Agent结果，生成最终回复"""
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)

        if not compliance_passed:
            final_response = (
                "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
                "工单编号已自动生成，请留意后续通知。"
            )
        else:
            result_parts = []

            # vision 结果优先展示
            vision_result = sub_results.get("vision")
            if vision_result and isinstance(vision_result, str) and vision_result.strip():
                result_parts.append(f"📷 图片分析结果：\n{vision_result}")

            # knowledge_rag 和 graph_rag 结果融合
            kg_result = sub_results.get("knowledge_rag")
            graph_result = sub_results.get("graph_rag")

            if kg_result and graph_result:
                # 双路都有结果——用 LLM 做一次轻量融合去重
                fused = await self._fuse_parallel_results(
                    state.get("messages", [])[-1].content if state.get("messages") else "",
                    kg_result,
                    graph_result,
                )
                result_parts.append(fused)
            elif kg_result:
                result_parts.append(kg_result)
            elif graph_result:
                result_parts.append(graph_result)

            # 工单结果
            ticket_result = sub_results.get("ticket_handler")
            if ticket_result and isinstance(ticket_result, str) and ticket_result.strip():
                result_parts.append(ticket_result)

            final_response = "\n\n".join(result_parts) if result_parts else (
                "抱歉，暂时无法处理您的请求，请稍后重试。如需人工协助，请拨打客服热线。"
            )

        return {
            **state,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }

    async def _fuse_parallel_results(
        self, query: str, kg_answer: str, graph_answer: str,
    ) -> str:
        """融合两个并行Agent的回答：去重 + 互补拼接"""
        # 如果两个回答高度重合（开头相同），只保留更长的
        kg_start = kg_answer[:80].strip()
        graph_start = graph_answer[:80].strip()

        if kg_start == graph_start:
            # 完全相同，保留更详细的
            return kg_answer if len(kg_answer) >= len(graph_answer) else graph_answer

        # 互补：合并两个回答
        return (
            f"{kg_answer}\n\n"
            f"---\n"
            f"📎 补充信息（知识图谱）：\n{graph_answer}"
        )


# ─── 路由函数（Send API 多路并行） ───

def continue_to_agents(state: AgentState) -> list[Send]:
    """
    使用 LangGraph Send API 实现多路并行 fan-out。

    根据 supervisor_route 的分析结果，向多个 Agent 同时发送状态。
    LangGraph 会在所有分支完成后自动合并状态（通过 _merge_sub_results reducer）。
    """
    # 从 working memory 上下文读取路由决策
    # 默认行为：电商查询同时走 knowledge_rag + graph_rag
    intent = state.get("intent", "")
    has_images = state.get("has_images", False)
    sends: list[Send] = []

    if has_images:
        sends.append(Send("vision", state))

    if intent == "ticket_handler":
        sends.append(Send("ticket_handler", state))
    else:
        # 电商查询默认双路并行
        sends.append(Send("knowledge_rag", state))
        sends.append(Send("graph_rag", state))

    return sends


# ─── 构建Graph ───

def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    short_term_memory: ShortTermMemory | None = None,
    retriever: HybridRetriever | None = None,
    knowledge_graph: KnowledgeGraph | None = None,
    enable_checkpointing: bool = True,
) -> StateGraph:
    """
    构建Supervisor编排的多Agent StateGraph（v3: 多路并行）。

    Args:
        llm: 语言模型实例
        short_term_memory: 短期记忆（对话历史）
        retriever: 混合检索引擎（替代 long_term_memory + sparse_index）
        knowledge_graph: 知识图谱
        enable_checkpointing: 是否启用检查点
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

    supervisor = SupervisorNode(llm)

    # 实例化所有子 Agent
    knowledge_agent = KnowledgeRAGAgent(llm, retriever)
    ticket_agent = TicketHandlerAgent(llm)
    compliance_agent = ComplianceCheckerAgent(llm)
    graph_rag_agent = GraphRAGAgent(llm, retriever, knowledge_graph)
    vision_agent = VisionAgent(llm)

    # 构建 StateGraph
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("preprocess", preprocess_input)
    graph.add_node("supervisor_route", supervisor.route_decision)
    graph.add_node("knowledge_rag", knowledge_agent.process)
    graph.add_node("ticket_handler", ticket_agent.process)
    graph.add_node("compliance_check", compliance_agent.process)
    graph.add_node("graph_rag", graph_rag_agent.process)
    graph.add_node("vision", vision_agent.process)
    graph.add_node("synthesize", supervisor.synthesize_response)

    # 入口: preprocess → supervisor_route
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "supervisor_route")

    # 条件路由: supervisor_route → 多路并行 fan-out
    graph.add_conditional_edges(
        "supervisor_route",
        continue_to_agents,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "graph_rag": "graph_rag",
            "vision": "vision",
        },
    )

    # 所有子 Agent 完成后 → 合规审查
    graph.add_edge("knowledge_rag", "compliance_check")
    graph.add_edge("ticket_handler", "compliance_check")
    graph.add_edge("graph_rag", "compliance_check")
    graph.add_edge("vision", "compliance_check")

    # 合规审查 → 汇总 → 结束
    graph.add_edge("compliance_check", "synthesize")
    graph.add_edge("synthesize", END)

    # 可选检查点
    checkpointer = MemorySaver() if enable_checkpointing else None
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
