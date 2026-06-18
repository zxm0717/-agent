# 电商多 Agent 智能客服系统

基于 **LangGraph** Supervisor 编排的企业级多 Agent 智能客服系统，面向电商场景。支持多格式知识文档自动入库、混合检索（FAISS + BM25 + RRF）、知识图谱推理、多模态图片分析、工具调用协议（MCP）和全链路追踪。

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
                    │  路由决策 → 并行调度 → 融合  │
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
                    │       Compliance 合规审查     │
                    │    规则引擎 + LLM 深度审查     │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │  Synthesize 融合   │
                         │  最终回复          │
                         └───────────────────┘
```

### 两条核心管线

**索引构建**（启动时执行一次）：

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
    SemanticChunker (tiktoken 分块, 512token窗口, 128token重叠,
                     parent/child层级, chunk_type标注)
        │
        ├──→ FAISS 稠密向量索引 (1536维, OpenAI/hash)
        ├──→ SparseIndex BM25 倒排索引 (jieba分词)
        ├──→ KnowledgeGraph (NetworkX, 30+实体, 40+关系)
        └──→ StructuredEncoder (SKU/价格/政策 → descriptive chunk)
```

**用户查询**（每次请求执行）：

```
Query → Supervisor (并行路由) → HybridRetriever (FAISS+BM25+RRF+元数据加权+上下文扩展)
                               → Graph Traverse (知识图谱遍历)
                               → Compliance Check (规则+LLM双阶段)
                               → Synthesize (去重融合)
```

## 快速开始

### 环境要求

- Python ≥ 3.11
- Redis（可选，会话历史）
- OpenAI API Key（可选，hash embedding 模式可零依赖运行）

### 安装运行

```bash
cd python-impl
cp .env.example .env              # 编辑 .env，填入 OPENAI_API_KEY
pip install -r requirements.txt
python api/main.py                # → http://localhost:8000
```

启动日志示例：
```
[init] 向量索引: 22 chunks (embedding=hash)
[init] BM25 索引: 22 docs, 408 terms
[init] 结构化编码: 3 chunks
[init] 知识图谱: 30 nodes, 40 edges
[init] 混合检索引擎: dense=True, sparse=True
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Docker

```bash
docker compose up -d               # Redis + Python App + Jaeger
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 主聊天接口（纯文本） |
| POST | `/api/chat/multimodal` | 多模态聊天（文本 + 图片 base64） |
| GET | `/api/history/{session_id}` | 对话历史 |
| GET | `/api/knowledge-graph/stats` | 知识图谱统计（节点/边/类型分布） |
| GET | `/api/knowledge-graph/query?entity=xxx` | 实体搜索 + 关系 + 邻居 |
| GET | `/api/knowledge-graph/search?source=xxx&target=yyy` | 关系搜索 / 多跳路径查找 |
| POST | `/api/knowledge-graph/build` | 从文档 LLM 自动构建图谱 |
| GET | `/api/tools` | MCP 工具发现列表 |
| POST | `/api/tools/call` | MCP 工具调用 (JSON-RPC 2.0) |
| GET | `/api/metrics` | Agent 调用指标（延迟/错误率） |
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

# 路径查找
curl "http://localhost:8000/api/knowledge-graph/search?source=X1&target=充电器"
```

## 项目结构

```
python-impl/
├── agents/                       # 7个 Agent，LangGraph 节点
│   ├── supervisor.py             # 中央编排：路由 + 并行调度(Send API) + 结果融合
│   ├── knowledge_rag.py          # 知识检索：HybridRetriever → LLM 生成回答
│   ├── graph_rag.py              # 图谱检索：实体提取 → 图谱遍历 + 混合检索 → 回答
│   ├── vision.py                 # 视觉分析：GPT-4o 多模态识别（商品/截图/票据）
│   ├── intent_router.py          # 意图路由：咨询/投诉/交易/多跳推理/多模态
│   ├── ticket_handler.py         # 工单处理：创建/查询/流转
│   └── compliance_checker.py     # 合规审查：规则引擎 + LLM 双阶段
│
├── memory/                       # 记忆系统 + 索引引擎
│   ├── working_memory.py         # 工作记忆：会话级临时上下文 (dict)
│   ├── short_term.py             # 短期记忆：对话历史 (Redis)
│   ├── long_term.py              # 长期记忆：FAISS 向量库 + SemanticChunker
│   ├── knowledge_graph.py        # 知识图谱：NetworkX 有向图 + BFS 遍历 + JSON 持久化
│   ├── doc_parser.py             # 文档解析：txt/md/pdf/图片 → 统一文本 + 结构识别
│   ├── ocr.py                    # OCR 可插拔后端：PaddleOCR(离线) | GPT-4o Vision(在线)
│   ├── sparse_index.py           # BM25 关键词索引：jieba 分词 + 倒排表 + pickle 持久化
│   ├── structured_encoder.py     # 结构化编码：SKU/价格/配件/政策 → descriptive chunk
│   └── retriever.py              # 统一混合检索引擎：FAISS+BM25 → RRF + 元数据加权 + 上下文扩展
│
├── api/                          # Web 层
│   ├── main.py                   # FastAPI 应用、生命周期、11 个端点
│   └── models.py                 # Pydantic 请求/响应模型（ChatRequest/ChatMultiModal/etc.）
│
├── mcp/                          # MCP 工具协议 (Model Context Protocol)
│   ├── mcp_server.py             # JSON-RPC 2.0：工具注册/发现/调用 + 内置 7 个工具
│   └── tools/
│
├── tracing/                      # 可观测性
│   └── otel_config.py            # OpenTelemetry Span 装饰器 + AgentMetrics 收集器
│
├── data/knowledge_base/          # 电商知识文档
│   ├── products/                 # smartphone_x1.txt / buds_pro3.txt / accessories / ...
│   └── policies/                 # return_policy.txt / shipping_policy.txt / after_sales / ...
│
├── vector_store/                 # FAISS 索引持久化（自动生成）
├── graph_store/                  # 知识图谱 JSON 持久化（自动生成）
├── requirements.txt
├── .env.example
└── Dockerfile
```

## 核心设计

### 混合检索引擎 `HybridRetriever`

每个查询走 6 步管线：

```
Query → ① LLM分析(实体提取+chunk_type推断+rewrite，1次调用)
        ② FAISS稠密检索 (语义向量，top_k×2)
        ③ BM25稀疏检索 (关键词倒排，top_k×2)
        ④ RRF融合 (score = Σ 1/(60+rank_i))
        ⑤ 元数据加权 (chunk_type ×1.3, entity_id ×1.5)
        ⑥ 上下文扩展 (parent章节 + 子chunk要点)
```

| 机制 | 效果 |
|------|------|
| RRF 融合 | FAISS(查语义相近) + BM25(查精确关键词) 互补，单路缺的另路补上 |
| chunk_type 加权 | 问"X1价格"时 sku_fact 类 chunk 自动 ×1.3，排到退货政策前面 |
| entity_id 加权 | 识别到 x1_phone 实体后，含该 ID 的 chunk ×1.5 |
| 上下文扩展 | 命中 child chunk 时附带 parent section 概述，避免 LLM 只看到碎片 |

### 多路由并行

v1/v2 版本中 Supervisor 每次只路由到一个 Agent。v3 改为 **LangGraph Send API 多路并行**：

```python
def continue_to_agents(state):
    return [
        Send("knowledge_rag", state),   # FAISS+BM25 检索
        Send("graph_rag", state),       # KG 遍历 + 检索
    ]
```

两个 Agent 并行跑完后，`synthesize_response` 做去重/互补融合——同一事实双路命中则取详细版，互补信息则拼接。

### 三层记忆

| 层 | 存活周期 | 存储 | 容量 |
|----|---------|------|------|
| 工作记忆 | 单会话 | dict | 意图/路由/实体 |
| 短期记忆 | 多轮对话 | Redis | 可配置 LRU 窗口 |
| 长期记忆 | 持久 | FAISS + NetworkX + pickle | 知识库全量 |

### 文档智能入库

| 输入 | 处理方式 |
|------|---------|
| `.txt` | 直接读取 (UTF-8/GBK fallback) → 结构检测 → 语义分块 |
| `.md` | 同上，标题语法增强结构检测 |
| `.pdf` | PyPDF2 提取文本页，空白页标记 OCR 需求 |
| `.png/.jpg` | OCR 提取文字后走分块管线（无 OCR 后端则跳过 + warning） |

分块策略：512 token 自然边界窗口、128 token 重叠、parent/child 层级（每 4 个 child 生成 1 个 section 级 parent）。

### Embedding 模式

| 模式 | 向量维度 | 依赖 | 质量 | 默认 |
|------|---------|------|------|------|
| `hash` | 1536 | 无 | 无语义（确定性随机） | ✅ |
| `openai` | 1536 (text-embedding-3-small) | API key + 网络 | 语义匹配 | |

`hash` 模式保证零配置可跑通全链路；配好 API key 后在 `.env` 中设置 `EMBEDDING_MODE=openai` 即可切换。

## 配置

```bash
# .env 关键配置
OPENAI_API_KEY=sk-xxx          # 必填（LLM 需要）
MODEL_NAME=gpt-4o              # LLM 模型
EMBEDDING_MODE=hash            # hash | openai
FAISS_INDEX_PATH=./vector_store/faiss_index
GRAPH_STORE_PATH=./graph_store/graph.json
REDIS_URL=redis://localhost:6379/0
```

## 技术栈

| 层 | 技术 | 版本 |
|----|------|------|
| 编排框架 | LangGraph (StateGraph + Send API) | ≥ 0.3 |
| LLM | OpenAI GPT-4o (langchain-openai) | ≥ 0.3 |
| API | FastAPI + Uvicorn | ≥ 0.115 |
| 稠密向量 | FAISS-cpu (IndexFlatIP) | ≥ 1.9 |
| 稀疏索引 | jieba + BM25 | ≥ 0.42 |
| 图存储 | NetworkX | ≥ 3.4 |
| Token 计数 | tiktoken (cl100k_base) | ≥ 0.8 |
| PDF 解析 | PyPDF2 | ≥ 3.0 |
| OCR（可选） | PaddleOCR / GPT-4o Vision | — |
| 对话历史 | Redis | ≥ 5.2 |
| 链路追踪 | OpenTelemetry → Jaeger | ≥ 1.29 |
| 工具协议 | MCP (JSON-RPC 2.0) | ≥ 1.0 |
| 容器化 | Docker Compose | 3.8 |

## License

MIT
