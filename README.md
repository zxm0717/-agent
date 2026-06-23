# 电商多 Agent 智能客服系统

基于 **LangGraph** + **Supervisor** 模式编排的电商智能客服系统，覆盖**文档入库 → 混合检索 → 知识图谱推理 → 多模态视觉 → 合规审查 → MCP 工具协议 → 全链路追踪**。

---

## 架构

```
                    POST /api/chat
                          │
                 ┌────────▼────────┐
                 │  FastAPI Gateway │
                 └────────┬────────┘
                          │
            ┌─────────────▼─────────────┐
            │  Supervisor (LangGraph)    │
            │  Send API 并行路由          │
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

**离线索引构建**（启动时自动执行，支持增量）：

```
data/knowledge_base/ (txt / md / pdf / 图片)
        │
        ├── DocParser → SemanticChunker (tiktoken, 512 token 窗口)
        ├── FAISS 稠密索引 (1536维, hash / openai / bge 三模式)
        ├── BM25 稀疏倒排索引 (jieba 分词)
        ├── KnowledgeGraph (NetworkX, 30+ 实体, 40+ 关系)
        └── StructuredEncoder (SKU/价格/政策 → chunk)

  IndexManifest → 文件 Hash 跟踪 → 增量索引
```

**在线查询**（每次请求执行）：

```
Query
  → 短期记忆加载 (Redis 对话历史注入 state)
  → Supervisor 并行路由（基于历史上下文做意图判断）
      ├─→ knowledge_rag → LLM分析 → FAISS → BM25 → RRF融合 → 加权 → 生成（带历史）
      └─→ graph_rag → 实体提取 → KG遍历 → 生成（带历史）
  → Compliance Check (规则 + LLM 双阶段)
  → Synthesize (去重融合)
  → 短期记忆写入 (本轮 user + assistant 追加)
```

---

## 快速开始

### 环境

- Python ≥ 3.11
- Redis（可选，无 Redis 仍可启动，历史仅存内存）
- OpenAI API Key（LLM 必须；embedding 选 openai 模式时需要）

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

### Docker

```bash
docker compose up -d               # Redis + Python App + Jaeger
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 主聊天接口 |
| POST | `/api/chat/multimodal` | 多模态聊天（文本 + 图片 base64）|
| GET | `/api/history/{session_id}` | 对话历史 |
| GET | `/api/knowledge-graph/stats` | 图谱统计（节点/边/类型分布）|
| GET | `/api/knowledge-graph/query?entity=xxx` | 实体搜索 + 关系 + 邻居 |
| GET | `/api/knowledge-graph/search?source=xxx&target=yyy` | 多跳路径查找 |
| POST | `/api/knowledge-graph/build` | LLM 从文档构建图谱 |
| GET | `/api/tools` | MCP 工具发现 (JSON-RPC 2.0) |
| POST | `/api/tools/call` | MCP 工具调用 |
| GET | `/api/metrics` | Agent 指标（延迟/错误率/调用日志）|
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
python-impl/
├── agents/                           # 5 个 Agent，LangGraph 节点
│   ├── supervisor.py                 # 中央编排: 并行路由(Send API) + 结果融合
│   ├── knowledge_rag.py              # 知识检索: HybridRetriever → LLM 生成
│   ├── graph_rag.py                  # 图谱检索: 实体提取 → KG 遍历 → 生成
│   ├── vision.py                     # 视觉分析: GPT-4o 多模态识别
│   ├── ticket_handler.py             # 工单处理: 创建/查询/流转
│   └── compliance_checker.py         # 合规审查: 规则引擎 + LLM 双阶段
│
├── memory/                           # 记忆系统 + 索引引擎 + 检索器
│   ├── short_term.py                 # 短期记忆: 对话历史 (Redis)，每次请求注入 Agent 管线
│   ├── long_term.py                  # 长期记忆: FAISS + SemanticChunker + 三模式 Embedding
│   ├── knowledge_graph.py            # 知识图谱: NetworkX 有向图 + BFS + JSON 持久化
│   ├── doc_parser.py                 # 文档解析: txt/md/pdf/图片 → 统一文本
│   ├── ocr.py                        # OCR 可插拔后端: PaddleOCR / GPT-4o Vision
│   ├── sparse_index.py               # BM25 倒排索引: jieba 分词 + 停用词过滤
│   ├── structured_encoder.py         # 结构化编码: SKU/价格/配件/政策 → chunk
│   ├── retriever.py                  # 混合检索引擎: FAISS+BM25 → RRF + 加权 + 扩展
│   └── index_manifest.py             # 索引清单: 文件 Hash 跟踪 → 增量索引
│
├── api/                              # Web 层
│   ├── main.py                       # FastAPI 应用、生命周期、增量索引、11 个端点
│   └── models.py                     # Pydantic 请求/响应模型
│
├── mcp/                              # MCP 工具协议
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
├── vector_store/                     # FAISS + chunks + index_manifest (自动生成)
├── graph_store/                      # 知识图谱 JSON (自动生成)
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## 核心设计

### 1. 多 Agent 并行编排

基于 LangGraph `StateGraph` + `Send` API 实现多路 fan-out，两个 Agent 并行跑，墙钟 = max(单路)：

```python
def continue_to_agents(state):
    return [
        Send("knowledge_rag", state),   # FAISS + BM25 混合检索
        Send("graph_rag", state),       # KG 遍历 + 实体加权检索
    ]
```

### 2. 混合检索引擎 HybridRetriever

6 步管线，每条查询执行：

```
Query → ① LLM 分析 (实体提取 + chunk_type 推断 + rewrite)
        ② FAISS 稠密检索 (语义向量)
        ③ BM25 稀疏检索 (关键词倒排)
        ④ RRF 融合 (score = Σ 1/(60+rank_i))
        ⑤ 元数据加权 (chunk_type ×1.3, entity_id ×1.5, section_title ×1.2)
        ⑥ 上下文扩展 (parent 章节 + 子 chunk 要点)
```

### 3. Embedding 三模式

| 模式 | 向量维度 | 模型 | 依赖 | 适用场景 |
|------|---------|------|------|---------|
| `hash` | 1536 | SHA256 确定性随机 | **零依赖** | 开发调试、全链路验证 |
| `openai` | 1536 | text-embedding-3-small | API key + 网络 | 生产环境 |
| `bge` | 512 | BAAI/bge-small-zh-v1.5 | sentence-transformers | **免费离线，中文最优** |

`.env` 中 `EMBEDDING_MODE=hash|openai|bge` 一键切换。BGE 模式首次运行自动下载模型（~100MB），CPU 可用，中文语义匹配效果优于 OpenAI。

### 4. 语义分块引擎 SemanticChunker

- **tiktoken 驱动**：512 token 自然边界窗口，128 token 重叠
- **结构感知**：标题触发 section 边界，段落/列表/标题自动标注 chunk_type
- **层级索引**：每 4 个 child chunk 生成 1 个 parent（section 级粗粒度 summary）
- **超长句处理**：按中文标点边界（。！？）回退切分

### 5. 知识图谱 KnowledgeGraph

- NetworkX 有向图，零外部依赖，30+ 实体、40+ 关系边
- 实体类型：Product / Brand / Category / Policy / Process / Service
- 关系类型：has_brand / belongs_to / compatible_with / ecosystem_sync / competitor / applies_to / depends_on
- BFS 多跳遍历 + 路径查找 + 子图上下文序列化
- LLM 辅助自动构建（`POST /api/knowledge-graph/build`）

### 6. 增量索引 IndexManifest

- 每个文件记录 MD5 Hash → `index_manifest.json`
- 启动时扫描目录 diff，仅处理新增/修改/删除
- 结构化编码（SKU/价格）同样纳入增量感知
- `INCREMENTAL_INDEX_ENABLED=0` 可强制全量重建

### 7. 双层记忆

| 层 | 存活周期 | 存储 | 用途 |
|----|---------|------|------|
| 短期记忆 | 多轮对话 (TTL 30min) | Redis (内存 fallback) | 每次请求加载历史注入 Agent 管线，Supervisor + 所有子 Agent 均感知多轮上下文 |
| 长期记忆 | 持久 | FAISS + BM25 + NetworkX | 知识库全量：文档检索、图谱推理、结构化编码 |

### 8. MCP 工具协议

手写实现 JSON-RPC 2.0，内置 7 个工具：

| 工具名 | 分类 | 功能 |
|--------|------|------|
| `order_query` | order | 订单查询 |
| `knowledge_search` | knowledge | 知识库搜索 |
| `ticket_create` | ticket | 创建工单 |
| `risk_check` | risk | 合规风险检查 |
| `graph_search` | knowledge | 图谱实体搜索 |
| `graph_find_path` | knowledge | 图谱路径查找 |
| `image_analysis` | vision | 图片分析 |

### 9. 合规审查（规则 + LLM 双阶段）

- **规则引擎**（毫秒级）：敏感词检测、PII 正则匹配，高风险直接阻断
- **LLM 深度审查**：越权承诺检查、歧视性内容、监管用语

### 10. 全链路追踪

- OpenTelemetry SDK 统一埋点，Agent 节点独立 Span
- OTLP 协议对接 Jaeger，Docker 内一键启动
- `GET /api/metrics` 暴露 Agent + 工具调用日志

---

## 配置参考

```bash
# LLM
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o

# Embedding — hash | openai | bge
EMBEDDING_MODE=hash

# Redis（可选）
REDIS_URL=redis://localhost:6379/0

# 增量索引
INCREMENTAL_INDEX_ENABLED=1
INDEX_MANIFEST_PATH=./vector_store/index_manifest.json

# 知识图谱
GRAPH_STORE_PATH=./graph_store/graph.json

# 多模态
VISION_MODEL_NAME=gpt-4o

# 链路追踪
OTEL_SERVICE_NAME=smart-cs-multi-agent
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Server
HOST=0.0.0.0
PORT=8000
```

---

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 编排框架 | LangGraph (StateGraph + Send API) | ≥ 0.3 |
| LLM | OpenAI GPT-4o (langchain-openai) | ≥ 0.3 |
| API | FastAPI + Uvicorn | ≥ 0.115 |
| 稠密向量 | FAISS-cpu (IndexFlatIP) | ≥ 1.9 |
| Embedding | OpenAI text-embedding-3-small / BGE-small-zh-v1.5 / Hash | 三模式 |
| 稀疏索引 | jieba + BM25 | ≥ 0.42 |
| 图存储 | NetworkX (DiGraph) | ≥ 3.4 |
| Token 计数 | tiktoken (cl100k_base) | ≥ 0.8 |
| PDF 解析 | PyPDF2 | ≥ 3.0 |
| 对话历史 | Redis | ≥ 5.2 |
| 链路追踪 | OpenTelemetry → Jaeger | ≥ 1.29 |
| 工具协议 | MCP (JSON-RPC 2.0，手写) | — |
| 容器化 | Docker Compose | 3.8 |

---

## License

MIT
