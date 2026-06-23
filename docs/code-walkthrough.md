# 代码讲解文档 — 核心模块解析

> 本文档逐层讲解 Python 实现的核心模块，帮助快速理解代码设计。

---

## 1. 入口：FastAPI 应用 (`api/main.py`)

### 1.1 应用生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 初始化 OpenTelemetry 追踪
    init_tracer(...)

    # 2. 播种向量库文档（12篇电商知识文档）
    for doc in ECOMMERCE_DOCUMENTS:
        long_term_memory.add_document(content=doc["content"], source=doc["source"])

    # 3. 构建知识图谱（60+实体、70+关系）
    knowledge_graph.build_from_structured(
        entities=ECOMMERCE_GRAPH_DATA["entities"],
        relationships=ECOMMERCE_GRAPH_DATA["relationships"],
    )

    # 4. 构建 LangGraph 编排图
    graph = create_supervisor_graph(
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        knowledge_graph=knowledge_graph,
    )
    yield
```

启动时自动完成所有初始化，无需外部数据库。

### 1.2 路由端点

- `/api/chat` — 纯文本聊天。`ChatRequest` 入参 → `graph.ainvoke()` → `ChatResponse`
- `/api/chat/multimodal` — 多模态聊天。`ChatMultiModalRequest` 包含 `images: list[str]`，预处理后调用同一 Graph
- `/api/knowledge-graph/*` — 图谱查询、统计、构建

---

## 2. 编排核心 (`agents/supervisor.py`)

### 2.1 StateGraph 拓扑

```
preprocess → supervisor_route ──→ knowledge_rag ──┐
                    │          → ticket_handler ──┤
                    │          → graph_rag ────────┤
                    │          → vision ───────────┤
                    │          → compliance_check ─┤
                    │                              │
                    └──────────  conditional_edges ┘
                                    │
                              compliance_check
                                    │
                                synthesize → END
```

### 2.2 preprocess 节点

最简单但关键的节点：检测 `state.images` 是否为空，设 `has_images` 标志，供后续路由参考。

```python
async def preprocess_input(state: AgentState) -> AgentState:
    images = state.get("images", [])
    return {**state, "has_images": len(images) > 0}
```

### 2.3 路由决策

`SupervisorNode.route_decision` 调用 LLM 分析用户意图，从 5 个可选 Agent 中选出最优目标。有图片时自动提升 vision 优先级。

### 2.4 结果汇总

`synthesize_response` 按照 vision → graph_rag → knowledge_rag → ticket 顺序合并输出，合规不通过则返回转人工话术。

---

## 3. 知识检索 (`agents/knowledge_rag.py`)

四步标准 RAG 管线：

```
用户原始问题: "X1电池多大"
    ↓
[Query改写] → "X1智能手机 电池容量 mAh"
    ↓
[向量检索] FAISS.search → Top-5 相关文档
    ↓
[重排序]   LLM 评估相关性 → Top-3
    ↓
[生成回答] System Prompt + 文档 + 问题 → 回答
```

Query 改写将口语化问题转为检索友好格式，重排序利用 LLM 判断文档与问题的实际相关性。

---

## 4. 图谱推理 (`agents/graph_rag.py`)

混合检索 = 向量语义匹配 + 图谱结构遍历。

### 4.1 实体提取

`extract_query_entities` 让 LLM 从用户问题中提取关键实体（商品名、品牌、品类等），返回 `[{"name": "X1", "type": "Product"}]`。

### 4.2 图遍历

`graph_traverse` 先调用 `KnowledgeGraph.search_entities()` 模糊匹配实体，再取 2-hop 邻居，生成 `get_subgraph_context()` 文本。

### 4.3 上下文拼装

```
graph_context: "实体[X1手机] --[compatible_with]--> [65W充电器] ..."
vector_docs:   ["X1手机配备5000mAh电池，支持65W快充..."]

→ LLM 生成：结合结构化关系 + 文档细节的回答
```

---

## 5. 视觉分析 (`agents/vision.py`)

### 5.1 图片校验

`validate_image_base64` 检查数据格式（magic bytes）、解码正确性、文件大小（<10MB）。

### 5.2 多类型分析

| analysis_type | 适用场景 | 提取内容 |
|---------------|----------|----------|
| screenshot | 报错截图、订单截图 | 错误信息、订单号、金额 |
| invoice | 发票、收据 | 发票号、日期、金额、商家 |
| product | 商品照片 | 品牌、型号、外观状态 |
| delivery | 物流截图 | 快递单号、状态 |

### 5.3 批量处理

`process(state)` 遍历 `state.images`，每张图片调用 GPT-4o vision，汇总所有结果写入 `sub_results.vision`。

---

## 6. 知识图谱 (`memory/knowledge_graph.py`)

### 6.1 数据模型

```
Node: {id, type(Product|Brand|Policy|...), name, properties}
Edge: {source → target, type(compatible_with|applies_to|...), properties}
```

### 6.2 核心操作

- `search_entities(query, type)` — 按名称模糊匹配，按匹配度排序
- `get_neighbors(entity_id, hops)` — BFS 邻居遍历
- `get_subgraph_context(entity_ids)` — 子图序列化为 LLM 可读文本
- `find_paths(source, target)` — 两实体间路径查找
- `build_from_documents(docs)` — LLM 自动提取实体关系

### 6.3 持久化

`save()` → JSON，`_load()` 从 JSON 恢复。预留 `GRAPH_BACKEND=neo4j` 升级路径。

---

## 7. 记忆系统

| 记忆层 | 文件 | 存储 | 生命周期 | 用途 |
|--------|------|------|----------|------|
| 短期记忆 | `short_term.py` | Redis (内存 fallback) | TTL 30min | 每次请求加载历史注入 Agent 管线 |
| 长期记忆 | `long_term.py` | FAISS 磁盘 | 永久 | 文档向量检索 + BM25 关键词索引 |
| 知识图谱 | `knowledge_graph.py` | NetworkX/JSON | 永久 | 实体关系推理 + 多跳路径 |

---

## 8. MCP 工具协议 (`mcp/mcp_server.py`)

遵循 JSON-RPC 2.0 标准，提供 6 个工具：

| 工具 | 分类 | 用途 |
|------|------|------|
| order_query | order | 查询订单信息 |
| knowledge_search | knowledge | 搜索知识库 |
| ticket_create | ticket | 创建工单 |
| risk_check | compliance | 风控检查 |
| graph_search | knowledge_graph | 图谱实体检索 |
| graph_find_path | knowledge_graph | 实体路径查找 |
| image_analysis | vision | 图片分析 |

工具注册使用装饰器模式：

```python
@server.register(name="graph_search", ..., category="knowledge_graph")
async def graph_search(entity_name, ...): ...
```
