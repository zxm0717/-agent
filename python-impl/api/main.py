"""
FastAPI入口 — 提供REST API + SSE流式响应

v2: 新增多模态聊天端点 + 知识图谱管理端点。
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.supervisor import create_supervisor_graph
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.knowledge_graph import KnowledgeGraph
from memory.sparse_index import SparseIndex
from memory.structured_encoder import StructuredEncoder
from memory.retriever import HybridRetriever
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import init_tracer, AgentMetrics

# v2: 新增模型
from api.models import (
    ChatRequest,
    ChatMultiModalRequest,
    ChatResponse,
    EntityQuery,
    GraphBuildRequest,
    GraphStatsResponse,
)

load_dotenv()


# ─── 全局服务实例 ───

working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
long_term_memory = LongTermMemory(
    index_path=os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index"),
    embedding_mode=os.getenv("EMBEDDING_MODE", "hash"),  # "hash" 或 "openai"
)
sparse_index = None  # 在 lifespan 中初始化
retriever = None     # 在 lifespan 中初始化
knowledge_graph = KnowledgeGraph(
    persist_path=os.getenv("GRAPH_STORE_PATH", "./graph_store/graph.json"),
)
mcp_server = create_default_tools(MCPToolServer(), knowledge_graph)
metrics = AgentMetrics()
graph = None


# ─── 电商示例数据 ───

# 知识库文档目录（txt文件按 chunk_size=512, overlap=128 分块索引）
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")

ECOMMERCE_GRAPH_DATA = {
    "entities": [
        # 商品
        {"id": "x1_phone", "type": "Product", "name": "X1智能手机",
         "properties": {"brand": "星耀", "price": "3999-5999元", "category": "手机"}},
        {"id": "pad_air", "type": "Product", "name": "星耀Pad Air",
         "properties": {"brand": "星耀", "price": "2499元起", "category": "平板"}},
        {"id": "watch_gt", "type": "Product", "name": "星耀Watch GT",
         "properties": {"brand": "星耀", "price": "1299元", "category": "智能手表"}},
        {"id": "buds_pro3", "type": "Product", "name": "星耀Buds Pro 3",
         "properties": {"brand": "星耀", "price": "499元", "category": "耳机"}},
        {"id": "charger_65w", "type": "Product", "name": "65W氮化镓充电器",
         "properties": {"brand": "星耀", "price": "129元", "category": "配件"}},
        {"id": "wireless_charger", "type": "Product", "name": "30W无线充电板",
         "properties": {"brand": "星耀", "price": "199元", "category": "配件"}},
        {"id": "magnetic_case", "type": "Product", "name": "磁吸保护壳",
         "properties": {"brand": "星耀", "price": "79元", "category": "配件"}},
        {"id": "stylus_pen", "type": "Product", "name": "智能触控笔",
         "properties": {"brand": "星耀", "price": "349元", "category": "配件"}},
        {"id": "magic_keyboard", "type": "Product", "name": "磁吸键盘",
         "properties": {"brand": "星耀", "price": "599元", "category": "配件"}},
        {"id": "leather_strap", "type": "Product", "name": "真皮表带",
         "properties": {"brand": "星耀", "price": "199元", "category": "配件"}},
        {"id": "steel_strap", "type": "Product", "name": "不锈钢表带",
         "properties": {"brand": "星耀", "price": "299元", "category": "配件"}},
        # 竞品
        {"id": "mate70", "type": "Product", "name": "Mate 70",
         "properties": {"brand": "某为", "price": "4299元起", "category": "手机"}},
        {"id": "mi15pro", "type": "Product", "name": "15 Pro",
         "properties": {"brand": "某米", "price": "3699元起", "category": "手机"}},
        # 品牌
        {"id": "brand_xingyao", "type": "Brand", "name": "星耀",
         "properties": {"country": "中国", "positioning": "中高端"}},
        # 品类
        {"id": "cat_phone", "type": "Category", "name": "智能手机"},
        {"id": "cat_tablet", "type": "Category", "name": "平板电脑"},
        {"id": "cat_wearable", "type": "Category", "name": "智能穿戴"},
        {"id": "cat_audio", "type": "Category", "name": "音频设备"},
        {"id": "cat_accessory", "type": "Category", "name": "配件"},
        # 政策
        {"id": "policy_return", "type": "Policy", "name": "7天无理由退换"},
        {"id": "policy_warranty", "type": "Policy", "name": "1年整机保修"},
        {"id": "policy_shipping", "type": "Policy", "name": "全国包邮"},
        {"id": "policy_screen_insurance", "type": "Policy", "name": "碎屏保"},
        {"id": "policy_extended_warranty", "type": "Policy", "name": "延保服务"},
        {"id": "policy_trade_in", "type": "Policy", "name": "以旧换新"},
        # 流程
        {"id": "process_return", "type": "Process", "name": "退款流程"},
        {"id": "process_order", "type": "Process", "name": "下单流程"},
        {"id": "process_exchange", "type": "Process", "name": "换货流程"},
        # 服务
        {"id": "service_hotline", "type": "Service", "name": "客服热线"},
        {"id": "service_repair", "type": "Service", "name": "官方维修"},
        {"id": "service_enterprise", "type": "Service", "name": "企业团购"},
    ],
    "relationships": [
        # 品牌归属
        {"source": "x1_phone", "target": "brand_xingyao", "type": "has_brand"},
        {"source": "pad_air", "target": "brand_xingyao", "type": "has_brand"},
        {"source": "watch_gt", "target": "brand_xingyao", "type": "has_brand"},
        {"source": "buds_pro3", "target": "brand_xingyao", "type": "has_brand"},
        # 品类归属
        {"source": "x1_phone", "target": "cat_phone", "type": "belongs_to"},
        {"source": "mate70", "target": "cat_phone", "type": "belongs_to"},
        {"source": "mi15pro", "target": "cat_phone", "type": "belongs_to"},
        {"source": "pad_air", "target": "cat_tablet", "type": "belongs_to"},
        {"source": "watch_gt", "target": "cat_wearable", "type": "belongs_to"},
        {"source": "buds_pro3", "target": "cat_audio", "type": "belongs_to"},
        # 配件兼容关系
        {"source": "charger_65w", "target": "x1_phone", "type": "compatible_with"},
        {"source": "charger_65w", "target": "pad_air", "type": "compatible_with"},
        {"source": "wireless_charger", "target": "x1_phone", "type": "compatible_with"},
        {"source": "wireless_charger", "target": "buds_pro3", "type": "compatible_with"},
        {"source": "magnetic_case", "target": "x1_phone", "type": "exclusively_for"},
        {"source": "stylus_pen", "target": "pad_air", "type": "compatible_with"},
        {"source": "magic_keyboard", "target": "pad_air", "type": "compatible_with"},
        {"source": "leather_strap", "target": "watch_gt", "type": "compatible_with"},
        {"source": "steel_strap", "target": "watch_gt", "type": "compatible_with"},
        # 生态互联
        {"source": "x1_phone", "target": "pad_air", "type": "ecosystem_sync",
         "properties": {"feature": "跨设备剪贴板、文件拖拽、应用流转"}},
        {"source": "x1_phone", "target": "watch_gt", "type": "ecosystem_sync",
         "properties": {"feature": "消息通知、音乐控制、远程拍照"}},
        {"source": "x1_phone", "target": "buds_pro3", "type": "ecosystem_sync",
         "properties": {"feature": "一键弹窗配对、查找耳机"}},
        {"source": "watch_gt", "target": "buds_pro3", "type": "ecosystem_sync",
         "properties": {"feature": "手表端控制音乐和降噪"}},
        # 竞品关系
        {"source": "x1_phone", "target": "mate70", "type": "competitor"},
        {"source": "x1_phone", "target": "mi15pro", "type": "competitor"},
        # 替代推荐
        {"source": "pad_air", "target": "x1_phone", "type": "bundle_recommend",
         "properties": {"note": "手机与平板搭配使用体验最佳"}},
        # 政策适用范围
        {"source": "policy_return", "target": "x1_phone", "type": "applies_to"},
        {"source": "policy_return", "target": "pad_air", "type": "applies_to"},
        {"source": "policy_return", "target": "watch_gt", "type": "applies_to"},
        {"source": "policy_return", "target": "buds_pro3", "type": "applies_to"},
        {"source": "policy_return", "target": "charger_65w", "type": "applies_to",
         "properties": {"condition": "需不影响二次销售"}},
        {"source": "policy_warranty", "target": "x1_phone", "type": "applies_to"},
        {"source": "policy_warranty", "target": "pad_air", "type": "applies_to"},
        {"source": "policy_screen_insurance", "target": "x1_phone", "type": "applies_to"},
        {"source": "policy_extended_warranty", "target": "x1_phone", "type": "applies_to"},
        {"source": "policy_trade_in", "target": "x1_phone", "type": "applies_to"},
        {"source": "policy_shipping", "target": "x1_phone", "type": "applies_to"},
        # 流程依赖
        {"source": "process_return", "target": "policy_return", "type": "depends_on",
         "properties": {"note": "退货需满足7天无理由条件"}},
        {"source": "process_order", "target": "process_return", "type": "related_to"},
        # 服务覆盖
        {"source": "service_repair", "target": "x1_phone", "type": "applies_to"},
        {"source": "service_repair", "target": "pad_air", "type": "applies_to"},
        {"source": "service_hotline", "target": "process_return", "type": "supports"},
        {"source": "service_hotline", "target": "process_order", "type": "supports"},
        {"source": "service_enterprise", "target": "x1_phone", "type": "applies_to",
         "properties": {"condition": "10台起购"}},
    ],
}


# ─── 应用生命周期 ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph

    # 初始化 OpenTelemetry
    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    # 从 data/knowledge_base/ 加载文档，自动解析 + 语义分块 + 向量索引
    chunk_count = long_term_memory.load_knowledge_base(KNOWLEDGE_BASE_DIR)
    print(f"[init] 向量索引: {chunk_count} chunks (embedding={long_term_memory.embedding_mode})")

    # 构建 BM25 稀疏关键词索引
    global sparse_index
    sparse_index = SparseIndex()
    sparse_index.index_chunks(long_term_memory.chunks)
    print(f"[init] BM25 索引: {sparse_index._doc_count} docs, {len(sparse_index._inverted)} terms")

    # 结构化数据编码后追加入索引
    enc = StructuredEncoder()
    sku_chunk = enc.encode_sku(
        "星耀X1", {"storage": "256GB", "price": 4299, "color": "曜石黑"}, "标准版"
    )
    long_term_memory.add_chunk(sku_chunk)
    sparse_index.index_chunk(sku_chunk)
    encoded_count = 1

    # 编码价格梯度
    price_chunk = enc.encode_price_tier("星耀X1", [
        {"label": "256GB", "price": 4299, "entity_id": "x1_phone"},
        {"label": "512GB", "price": 5299, "entity_id": "x1_phone"},
        {"label": "1TB", "price": 6299, "entity_id": "x1_phone"},
    ])
    long_term_memory.add_chunk(price_chunk)
    sparse_index.index_chunk(price_chunk)
    encoded_count += 1

    print(f"[init] 结构化编码: {encoded_count} chunks")

    # 构建知识图谱（结构化数据）
    knowledge_graph.build_from_structured(
        entities=ECOMMERCE_GRAPH_DATA["entities"],
        relationships=ECOMMERCE_GRAPH_DATA["relationships"],
    )
    knowledge_graph.save()
    print(f"[init] 知识图谱: {knowledge_graph._graph.number_of_nodes()} nodes, "
          f"{knowledge_graph._graph.number_of_edges()} edges")

    # 初始化 LLM（供 HybridRetriever + Supervisor Graph 共用）
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        temperature=0,
    )

    # 构建混合检索引擎（FAISS + BM25 + KG 元数据加权）
    global retriever
    retriever = HybridRetriever(
        llm=llm,
        long_term_memory=long_term_memory,
        sparse_index=sparse_index,
        knowledge_graph=knowledge_graph,
    )
    print(f"[init] 混合检索引擎: dense={retriever.get_statistics()['dense_available']}, "
          f"sparse={retriever.get_statistics()['sparse_available']}")

    # 构建 Supervisor Graph（v3: 多路并行）
    graph = create_supervisor_graph(
        llm=llm,
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        retriever=retriever,
        knowledge_graph=knowledge_graph,
    )

    yield


# ─── FastAPI 应用 ───

app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统（电商版）— 支持知识图谱 + 多模态",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 原有端点（向后兼容）───


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口（纯文本）"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
        "has_images": False,
        "images": [],
        "graph_context": {},
        "vision_results": {},
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


# ─── v2 新增端点 ───


@app.post("/api/chat/multimodal", response_model=ChatResponse)
async def chat_multimodal(request: ChatMultiModalRequest):
    """
    多模态聊天接口。
    支持文本 + 图片（base64）+ 附件（PDF等）。
    """
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    # 校验图片
    from agents.vision import VisionAgent
    images = request.images or []
    valid_images = []
    for img in images:
        validation = VisionAgent.validate_image_base64(img)
        if validation["valid"]:
            valid_images.append(img)

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
        "has_images": len(valid_images) > 0,
        "images": valid_images,
        "graph_context": {},
        "vision_results": {},
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


@app.get("/api/knowledge-graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """获取知识图谱统计信息"""
    stats = knowledge_graph.get_statistics()
    return GraphStatsResponse(**stats)


@app.get("/api/knowledge-graph/query")
async def query_graph(
    entity: str = Query(..., description="实体名称，支持模糊匹配"),
    entity_type: str | None = Query(None, description="实体类型过滤"),
):
    """查询知识图谱中的实体"""
    results = knowledge_graph.search_entities(entity, entity_type=entity_type)

    # 为每个结果附加关系和邻居
    enriched = []
    for r in results[:10]:
        eid = r["id"]
        relationships = knowledge_graph.get_relationships(eid)
        neighbors = knowledge_graph.get_neighbors(eid, hops=1)[:10]
        enriched.append({
            "entity": r,
            "relationships": relationships,
            "neighbors": neighbors,
        })

    return {"query": entity, "count": len(enriched), "results": enriched}


@app.post("/api/knowledge-graph/build")
async def build_graph(request: GraphBuildRequest):
    """从文档构建/扩展知识图谱（需要 LLM）"""
    if knowledge_graph.llm is None:
        from langchain_openai import ChatOpenAI
        knowledge_graph.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
        )

    count = await knowledge_graph.build_from_documents(request.documents)
    return {"status": "ok", "entities_extracted": count, "total_nodes": knowledge_graph._graph.number_of_nodes()}


@app.get("/api/knowledge-graph/search")
async def search_graph_relations(
    source: str = Query(..., description="起点实体名称"),
    target: str | None = Query(None, description="目标实体名称（可选，用于查找路径）"),
):
    """
    图谱关系搜索。
    - 只提供 source: 返回该实体的所有关系和邻居
    - 同时提供 source 和 target: 查找两个实体间的所有路径
    """
    source_entities = knowledge_graph.search_entities(source)
    if not source_entities:
        return {"error": f"未找到实体: {source}", "results": []}

    source_id = source_entities[0]["id"]

    if target:
        target_entities = knowledge_graph.search_entities(target)
        if not target_entities:
            return {"error": f"未找到实体: {target}", "results": []}
        target_id = target_entities[0]["id"]

        paths = knowledge_graph.find_paths(source_id, target_id, max_length=4)
        return {
            "source": source_entities[0],
            "target": target_entities[0],
            "path_count": len(paths),
            "paths": [[" -> ".join(p)] for p in paths[:10]],
        }
    else:
        relationships = knowledge_graph.get_relationships(source_id)
        neighbors = knowledge_graph.get_neighbors(source_id, hops=2)
        return {
            "entity": source_entities[0],
            "relationships": relationships,
            "neighbors": neighbors,
        }


# ─── 通用端点 ───


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口"""
    return {"tools": mcp_server.list_tools()}


@app.post("/api/tools/call")
async def call_tool(request: dict):
    """MCP工具调用接口"""
    result = await mcp_server.call_tool(
        name=request.get("name", ""),
        arguments=request.get("arguments", {}),
    )
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
