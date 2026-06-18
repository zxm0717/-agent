"""
Supervisor编排Agent — 中央协调者
负责接收用户请求，根据意图路由到对应子Agent，汇总结果返回。
采用LangGraph StateGraph实现，支持并行调度和Human-in-the-Loop断点。

v2: 新增 GraphRAG Agent + Vision Agent 支持。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agents.intent_router import IntentRouterAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from agents.ticket_handler import TicketHandlerAgent
from agents.compliance_checker import ComplianceCheckerAgent
from agents.graph_rag import GraphRAGAgent
from agents.vision import VisionAgent
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.knowledge_graph import KnowledgeGraph
from tracing.otel_config import trace_agent_call


# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    sub_results: dict[str, Any]
    compliance_passed: bool
    final_response: str
    current_agent: str
    retry_count: int
    # v2 新增字段
    has_images: bool
    images: list[str]
    graph_context: dict[str, Any]
    vision_results: dict[str, Any]


# ─── Preprocess 节点 ───

async def preprocess_input(state: AgentState) -> AgentState:
    """
    预处理节点（入口）：检测多模态内容。
    如果用户请求包含图片，设置 has_images 标志供后续路由决策使用。
    """
    images = state.get("images", [])
    return {
        **state,
        "has_images": len(images) > 0,
    }


# ─── Supervisor节点 ───

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能客服系统的Supervisor（主管编排Agent）。
你的职责是：
1. 分析用户意图，决定分发给哪个子Agent处理
2. 汇总子Agent的处理结果，生成最终回复
3. 确保所有回复都经过合规审查

可用的子Agent：
- intent_router: 意图识别和分类
- knowledge_rag: 知识库检索和回答
- ticket_handler: 工单创建和查询
- compliance_checker: 合规审查和敏感词检测
- graph_rag: 知识图谱关系推理（用于产品关联、兼容搭配、替代推荐等多跳问题）
- vision: 多模态图片分析（用于截图、票据、商品图片等视觉输入）

路由决策规则：
- 如果用户上传了图片/截图 → 优先路由到 vision
- 如果问题涉及产品关联关系、兼容性、替代品推荐 → 路由到 graph_rag
- 如果问题需要知识库搜索 → 路由到 knowledge_rag
- 如果问题需要创建/查询工单 → 路由到 ticket_handler
- 所有回复最终都要经过合规审查

根据用户消息，决定下一步路由到哪个Agent。
"""


class SupervisorNode:
    """Supervisor决策节点"""

    def __init__(self, llm: ChatOpenAI, working_memory: WorkingMemory):
        self.llm = llm
        self.working_memory = working_memory

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """分析用户意图，决定路由"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        has_images = state.get("has_images", False)

        context = self.working_memory.get_context(session_id)

        routing_prompt = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            SystemMessage(content=f"当前工作记忆上下文: {context}"),
            *messages,
        ]

        # 如果有图片，在 prompt 中提示优先考虑 vision
        if has_images:
            routing_prompt.append(HumanMessage(content=(
                "注意：此请求包含图片/附件。请优先考虑是否需要 vision Agent 处理图片内容。"
                "请根据用户消息内容和图片特征，返回最合适的Agent名称。"
            )))
        else:
            routing_prompt.append(HumanMessage(content=(
                "请分析用户的最新消息，返回应该路由到的Agent名称。"
                "只返回以下之一: knowledge_rag, ticket_handler, compliance_checker, graph_rag, vision"
            )))

        response = await self.llm.ainvoke(routing_prompt)
        intent = response.content.strip().lower()

        valid_intents = {"knowledge_rag", "ticket_handler", "compliance_checker", "graph_rag", "vision"}
        if intent not in valid_intents:
            # 如果用户上传了图片但 LLM 没选 vision，强制走 vision
            if has_images:
                intent = "vision"
            else:
                intent = "knowledge_rag"

        self.working_memory.update(session_id, {"last_intent": intent})

        return {
            **state,
            "intent": intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> AgentState:
        """汇总子Agent结果，生成最终回复"""
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)
        graph_context = state.get("graph_context", {})

        if not compliance_passed:
            final_response = (
                "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
                "工单编号已自动生成，请留意后续通知。"
            )
        else:
            result_parts = []

            # 优先展示 vision 结果（如果有图片）
            vision_result = sub_results.get("vision")
            if vision_result and isinstance(vision_result, str) and vision_result.strip():
                result_parts.append(f"📷 图片分析结果：\n{vision_result}")

            # 然后展示 graph_rag 结果
            graph_rag_result = sub_results.get("graph_rag")
            if graph_rag_result and isinstance(graph_rag_result, str) and graph_rag_result.strip():
                result_parts.append(graph_rag_result)

            # 知识检索结果
            knowledge_result = sub_results.get("knowledge_rag")
            if knowledge_result and isinstance(knowledge_result, str) and knowledge_result.strip():
                if "knowledge_rag" not in graph_context:
                    result_parts.append(knowledge_result)

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


# ─── 路由函数 ───

def route_to_agent(state: AgentState) -> str:
    """根据意图路由到对应Agent节点"""
    intent = state.get("intent", "knowledge_rag")
    route_map = {
        "knowledge_rag": "knowledge_rag",
        "ticket_handler": "ticket_handler",
        "compliance_checker": "compliance_check",
        "graph_rag": "graph_rag",
        "vision": "vision",
    }
    return route_map.get(intent, "knowledge_rag")


def should_check_compliance(state: AgentState) -> str:
    """所有回复都需经过合规审查"""
    return "compliance_check"


# ─── 构建Graph ───

def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    working_memory: WorkingMemory | None = None,
    short_term_memory: ShortTermMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    knowledge_graph: KnowledgeGraph | None = None,
    enable_checkpointing: bool = True,
) -> StateGraph:
    """
    构建Supervisor编排的多Agent StateGraph。

    这是整个系统的核心入口，将多个子Agent通过有向图连接起来，
    由Supervisor节点负责路由决策和结果汇总。

    v2: 新增 preprocess、graph_rag、vision 节点。

    Args:
        llm: 语言模型实例
        working_memory: 工作记忆
        short_term_memory: 短期记忆
        long_term_memory: 长期记忆
        knowledge_graph: 知识图谱（新增）
        enable_checkpointing: 是否启用检查点（支持断点恢复）
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    if working_memory is None:
        working_memory = WorkingMemory()

    supervisor = SupervisorNode(llm, working_memory)

    # 实例化所有子 Agent
    intent_router = IntentRouterAgent(llm)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    ticket_agent = TicketHandlerAgent(llm)
    compliance_agent = ComplianceCheckerAgent(llm)
    graph_rag_agent = GraphRAGAgent(llm, long_term_memory, knowledge_graph)
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

    # preprocess 完成后进入路由
    graph.add_edge("preprocess", "supervisor_route")

    # 条件路由: supervisor → 各子 Agent
    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "compliance_check": "compliance_check",
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
