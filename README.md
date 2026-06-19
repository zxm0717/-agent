# 电商多 Agent 智能客服系统

基于 **LangGraph** Supervisor 编排的企业级多 Agent 智能客服系统，面向电商场景。覆盖**文档智能入库 → 混合检索 → 知识图谱推理 → 多模态视觉分析 → 合规审查 → MCP 工具协议 → 全链路追踪**的完整闭环。

---

## 架构总览

```
                        POST /api/chat
                              │
                     ┌────────▼────────┐
                     │  FastAPI Gateway │
                     └────────┬────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Supervisor (LangGraph)     │
                │  Send API 并行路由           │
                └──┬────────┬────────┬──────┘
                   │        │        │
          ┌────────▼──┐ ┌──▼──────┐ ┌▼──────────┐
          │knowledge   │ │graph   │ │vision /    │
          │_rag       │ │_rag    │ │ticket      │
          └─────┬─────┘ └──┬─────┘ └─────┬──────┘
                │          │              │
          ┌─────▼──────────▼──────┬───────▼──────┐
          │   HybridRetriever     │  KG Traverse │
          │  FAISS + BM25 + RRF   │  BFS 2-hop   │
          └──────────┬────────────┴──────┬───────┘
                     │                   │
                ┌────▼───────────────────▼────┐
                │     Compliance 合规审查       │
                │  规则引擎 + LLM 双阶段         │
                └──────────────┬──────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  Synthesize 融合   │
                     │  去重 + 互补拼接    │
                     └───────────────────┘
```

### 两条核心管线

**离线索引构建**（启动时自动执行，支持增量更新）：

```
data/knowledge_base/ (txt / md / pdf / 图片)
        │
        ├── DocParser (多格式解析 + 结构识别 + 噪音清洗)
        ├── OCRBackend (可插拔: PaddleOCR 离线 / GPT-4o Vision 在线)
        │
        ▼
    ParsedDocument (统一文本格式)
        │
        ▼
    SemanticChunker (tiktoken 分块, 512token 窗口, 128token 重叠,
                     parent/child 层级, chunk_type 标注)
        │
        ├──→ FAISS 稠密向量索引 (1536维, hash / OpenAI 双模式)
        ├──→ BM25 稀疏倒排索引 (jieba 分词, BM25 评分)
        ├──→ KnowledgeGraph (NetworkX, 30+实体, 40+关系)
        └──→ StructuredEncoder (SKU/价格/政策 → descriptive chunk)

    IndexManifest (文件 Hash 跟踪)
        └──→ 增量索引: 仅处理新增/修改/删除的文件
```

**在线查询**（每次请求执行）：

```
Query
  → Supervisor (并行路由)
      ├─→ knowledge_rag → HybridRetriever (①LLM分析 → ②FAISS → ③BM25 → ④RRF
      │                     → ⑤元数据加权 → ⑥上下文扩展) → Generate
      └─→ graph_rag → 实体提取 → HybridRetriever(entity_hints) → KG Traverse → Generate
  → Compliance Check (规则 + LLM 双阶段)
  → Synthesize (去重融合)
```

---

## 快速开始

### 环境要求

- Python ≥ 3.11
- Redis（可选，会话历史。无 Redis 仍可启动，历史仅存内存）
- OpenAI API Key（LLM 必须；embedding 可选——hash 模式零依赖）

### 安装运行

```bash
cd python-impl
cp .env.example .env              # 编辑 .env，填入 OPENAI_API_KEY
pip install -r requirements.txt
python api/main.py                # → http://localhost:8000
```

启动日志示例：
```
[init] 向量索引 (全量): 22 chunks | embedding=hash
[init] BM25 索引: 22 docs, 408 terms
[init] 结构化编码: 2 chunks 更新
[init] 知识图谱: 30 nodes, 40 edges
[init] 混合检索引擎: dense=True, sparse=True
INFO:     Uvicorn running on http://0.0.0.0:8000
```

第二次启动（增量模式）：
```
[init] 向量索引 (增量): 22 chunks (+0/~0/-0) | embedding=hash
[init] BM25 索引: 22 docs, 408 terms
[init] 结构化编码: 无变更（跳过）
```

### Docker

```bash
docker compose up -d               # Redis + Python App + Jaeger
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 主聊天接口（纯文本） |
| POST | `/api/chat/multimodal` | 多模态聊天（文本 + 图片 base64） |
| GET | `/api/history/{session_id}` | 对话历史 |
| GET | `/api/knowledge-graph/stats` | 知识图谱统计（节点/边/类型分布） |
| GET | `/api/knowledge-graph/query?entity=xxx` | 实体搜索 + 关系 + 邻居 |
| GET | `/api/knowledge-graph/search?source=xxx&target=yyy` | 多跳路径查找 |
| POST | `/api/knowledge-graph/build` | LLM 从文档自动构建图谱 |
| GET | `/api/tools` | MCP 工具发现列表 (JSON-RPC 2.0) |
| POST | `/api/tools/call` | MCP 工具调用 |
| GET | `/api/metrics` | Agent 调用指标（延迟/错误率/工具日志） |
| GET | `/health` | 健康检查 |

### 调用示例

```bash
# 纯文本咨询
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "X1充电器多少钱？", "user_id": "u1"}'

# 多模态（带图片）
curl -X POST http://localhost:8000/api/chat/multimodal \
  -H "Content-Type: application/json" \
  -d '{"message": "这张发票能报销吗", "images": ["<base64>"]}'

# 知识图谱查询
curl "http://localhost:8000/api/knowledge-graph/query?entity=X1"

# 多跳路径查找
curl "http://localhost:8000/api/knowledge-graph/search?source=X1&target=充电器"

# MCP 工具发现
curl "http://localhost:8000/api/tools"

# MCP 工具调用
curl -X POST http://localhost:8000/api/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "knowledge_search", "arguments": {"query": "退换货", "top_k": 3}}'
```

---

## 项目结构

```
python-impl/                          # ~6200 行 Python
├── agents/                           # 7 个 Agent，LangGraph 节点
│   ├── supervisor.py                 # 中央编排: 并行路由(Send API) + 结果融合
│   ├── knowledge_rag.py              # 知识检索: HybridRetriever → LLM 生成回答
│   ├── graph_rag.py                  # 图谱检索: 实体提取 → 混合检索 + KG 遍历 → 回答
│   ├── vision.py                     # 视觉分析: GPT-4o 多模态识别(商品/截图/票据)
│   ├── intent_router.py              # 意图路由: 7 类意图分类 + 实体提取
│   ├── ticket_handler.py             # 工单处理: 创建/查询/流转
│   └── compliance_checker.py         # 合规审查: 规则引擎 + LLM 双阶段
│
├── memory/                           # 记忆系统 + 索引引擎 + 检索器
│   ├── working_memory.py             # 工作记忆: 会话级临时上下文 (dict)
│   ├── short_term.py                 # 短期记忆: 对话历史 (Redis)
│   ├── long_term.py                  # 长期记忆: FAISS 向量库 + SemanticChunker
│   ├── knowledge_graph.py            # 知识图谱: NetworkX 有向图 + BFS + JSON 持久化
│   ├── doc_parser.py                 # 文档解析: txt/md/pdf/图片 → 统一文本
│   ├── ocr.py                        # OCR 可插拔后端: PaddleOCR | GPT-4o Vision
│   ├── sparse_index.py               # BM25 倒排索引: jieba 分词 + 停用词过滤
│   ├── structured_encoder.py         # 结构化编码: SKU/价格/配件/政策 → chunk
│   ├── retriever.py                  # 混合检索引擎: FAISS+BM25 → RRF + 加权 + 上下文扩展
│   └── index_manifest.py             # 索引清单: 文件 Hash 跟踪 → 增量索引
│
├── api/                              # Web 层
│   ├── main.py                       # FastAPI 应用、生命周期、增量索引、11 个端点
│   └── models.py                     # Pydantic 请求/响应模型
│
├── mcp/                              # MCP 工具协议 (手写实现)
│   ├── mcp_server.py                 # JSON-RPC 2.0: 工具注册/发现/调用 + 内置 7 个工具
│   └── tools/
│
├── tracing/                          # 可观测性
│   └── otel_config.py                # OpenTelemetry Span 装饰器 + AgentMetrics
│
├── data/knowledge_base/              # 电商知识文档 (11 个文件)
│   ├── products/                     # 产品参数、配件兼容、竞品对比
│   └── policies/                     # 退换货、物流、售后、促销政策
│
├── vector_store/                     # FAISS + chunks pickle + index_manifest (自动生成)
├── graph_store/                      # 知识图谱 JSON (自动生成)
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## 核心设计

### 1. 多 Agent 并行编排

基于 LangGraph `StateGraph` + `Send` API 实现多路并行 fan-out：

```python
def continue_to_agents(state):
    return [
        Send("knowledge_rag", state),   # FAISS + BM25 混合检索
        Send("graph_rag", state),       # KG 遍历 + 实体加权检索
    ]
```

- 两个 Agent 并行跑，墙钟 = max(单路)，而非 sum
- 自定义 Reducer 自动合并 `sub_results`
- `synthesize_response` 做去重/互补融合：同一事实双路命中取详细版，互补信息拼接

### 2. 混合检索引擎 `HybridRetriever`

每次查询走 6 步管线：

```
Query → ① LLM 分析 (实体提取 + chunk_type 推断 + rewrite，1 次调用)
        ② FAISS 稠密检索 (语义向量，top_k×2)
        ③ BM25 稀疏检索 (关键词倒排，top_k×2)
        ④ RRF 融合 (score = Σ 1/(60+rank_i))
        ⑤ 元数据加权 (chunk_type ×1.3, entity_id ×1.5, section_title ×1.2)
        ⑥ 上下文扩展 (parent 章节 + 子 chunk 要点)
```

| 机制 | 效果 |
|------|------|
| RRF 融合 | FAISS(语义相近) + BM25(精确关键词) 互补，单路缺的另路补上 |
| chunk_type 加权 | 问"X1 价格"时 sku_fact 类自动 ×1.3，排到退货政策前面 |
| entity_id 加权 | KG 识别到 x1_phone 实体后，含该 ID 的 chunk ×1.5 |
| 上下文扩展 | 命中 child chunk 时附带 parent section 概述，避免 LLM 看到碎片 |

### 3. 语义分块引擎 `SemanticChunker`

- **tiktoken 驱动**：512 token 自然边界窗口，128 token 重叠
- **结构感知**：标题触发 section 边界，段落/列表/标题自动标注 chunk_type
- **层级索引**：每 4 个 child chunk 生成 1 个 parent（section 级粗粒度 summary）
- **超长句处理**：按中文标点边界（。！？）回退切分；单句超限则硬切

### 4. Embedding 双模式

| 模式 | 向量维度 | 依赖 | 质量 | 适用 |
|------|---------|------|------|------|
| `hash` | 1536 | 无 | 确定性随机（无语义） | 开发调试 / 零配置跑通全链路 |
| `openai` | 1536 (text-embedding-3-small) | API key + 网络 | 语义匹配 | 生产环境 |

`.env` 中设置 `EMBEDDING_MODE=openai` 一键切换。

### 5. 知识图谱 `KnowledgeGraph`

- 基于 NetworkX 有向图，零外部服务依赖
- 30+ 实体（商品/品牌/品类/政策/流程/服务）、40+ 关系边（归属/兼容/竞品/生态/推荐）
- BFS 多跳邻居遍历、`find_paths()` 路径查找
- `get_subgraph_context()` 将子图序列化为 LLM 可读文本
- JSON 持久化，LLM 辅助从文档自动提取实体关系（`POST /api/knowledge-graph/build`）

### 6. 增量索引 `IndexManifest`

- 每个文件记录 MD5 Hash → `index_manifest.json`
- 启动时扫描目录 + diff，仅处理新增/修改/删除的文件
- 修改文件：先移旧 chunk → 重新解析分块 → 追加新 chunk
- 删除文件：移除关联 chunk，自动解绑子 chunk 的 parent_id 引用
- 结构化编码（SKU/价格）同样纳入增量感知，按 content_hash 比对
- 首次启动自动退化为全量模式
- `INCREMENTAL_INDEX_ENABLED=0` 可强制全量重建

### 7. 三层记忆

| 层 | 存活周期 | 存储 | 容量 |
|----|---------|------|------|
| 工作记忆 | 单会话 | dict | 意图/路由/实体 |
| 短期记忆 | 多轮对话 | Redis | 可配置 LRU 窗口 |
| 长期记忆 | 持久 | FAISS + BM25 + NetworkX | 知识库全量 |

### 8. MCP 工具协议（手写实现）

从头实现 Model Context Protocol，零外部 MCP SDK 依赖：

- **协议层**：JSON-RPC 2.0 消息格式，`tools/list` / `tools/call` / `ping`
- **工具注册**：装饰器风格 `@server.register()`，支持 inputSchema 声明和分类
- **内置 7 个工具**：order_query / knowledge_search / ticket_create / risk_check / graph_search / graph_find_path / image_analysis
- **REST 暴露**：`GET /api/tools`（发现）、`POST /api/tools/call`（调用）

### 9. 合规审查（规则 + LLM 双阶段）

```
回复内容
  ├─→ 规则引擎快速检查（毫秒级）
  │     敏感词检测: "保证收益"、"零风险" 等禁语
  │     PII 正则匹配: 手机号/身份证号/银行卡号/邮箱
  │     高风险 → 直接阻断，不进入 LLM 阶段
  │
  └─→ LLM 深度语义审查
        越权承诺检查 / 歧视性内容 / 监管合规用语
```

### 10. 全链路追踪

- OpenTelemetry SDK 统一埋点，每个 Agent 节点独立 Span
- `AgentMetrics` 收集延迟/错误率/调用次数
- OTLP 协议对接 Jaeger，Docker 内一键启动
- `GET /api/metrics` 暴露 Agent + 工具调用日志

---

## 配置参考

```bash
# .env 关键配置
OPENAI_API_KEY=sk-xxx              # 必填（LLM 需要）
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o                  # LLM 模型
EMBEDDING_MODE=hash                # hash | openai
INCREMENTAL_INDEX_ENABLED=1        # 1=增量(默认) | 0=全量重建
FAISS_INDEX_PATH=./vector_store/faiss_index
INDEX_MANIFEST_PATH=./vector_store/index_manifest.json
GRAPH_STORE_PATH=./graph_store/graph.json
REDIS_URL=redis://localhost:6379/0
OTEL_SERVICE_NAME=smart-cs-multi-agent
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
VISION_MODEL_NAME=gpt-4o           # 必选多模态模型
```

---

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 编排框架 | LangGraph (StateGraph + Send API) | ≥ 0.3 |
| LLM | OpenAI GPT-4o (langchain-openai) | ≥ 0.3 |
| API | FastAPI + Uvicorn | ≥ 0.115 |
| 稠密向量 | FAISS-cpu (IndexFlatIP) | ≥ 1.9 |
| 稀疏索引 | jieba + BM25 | ≥ 0.42 |
| 图存储 | NetworkX (DiGraph) | ≥ 3.4 |
| Token 计数 | tiktoken (cl100k_base) | ≥ 0.8 |
| PDF 解析 | PyPDF2 | ≥ 3.0 |
| OCR（可选） | PaddleOCR / GPT-4o Vision | — |
| 对话历史 | Redis | ≥ 5.2 |
| 链路追踪 | OpenTelemetry → Jaeger | ≥ 1.29 |
| 工具协议 | MCP (JSON-RPC 2.0，手写) | — |
| 容器化 | Docker Compose | 3.8 |

---

## License

MIT
