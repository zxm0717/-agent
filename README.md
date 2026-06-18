# 电商智能客服多Agent系统

基于 **LangGraph** Supervisor 编排的企业级多Agent智能客服系统，面向电商场景，支持知识图谱推理和多模态交互。

## 架构

```
用户请求（文本 / 图片）
       │
  ┌────▼─────┐
  │ FastAPI   │  API Gateway
  └────┬─────┘
       │
  ┌────▼──────────────────────────┐
  │  Supervisor 编排 Agent         │
  │  LangGraph StateGraph          │
  └────┬───┬───┬───┬───┬──────────┘
       │   │   │   │   │
  ┌────▼┐ ┌▼──┐┌▼──┐┌▼──┐┌─────▼──┐
  │意图 │ │知识││工单││合规││图谱    │┌─────────┐
  │路由 │ │检索││处理││审查││推理    ││视觉分析  │
  │Agent│ │RAG ││CRUD││规则││GraphRAG││Vision   │
  └─────┘ └───┘└───┘└───┘└────────┘└─────────┘
       │           │
  ┌────▼───────────▼────┐
  │  记忆系统            │
  │  工作记忆 → 短期(Redis) → 长期(FAISS)  │
  │  → 知识图谱(NetworkX)                │
  └─────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排引擎 | LangGraph StateGraph |
| LLM | LangChain + OpenAI (GPT-4o) |
| API | FastAPI + Uvicorn |
| 短期记忆 | Redis |
| 向量检索 | FAISS |
| 知识图谱 | NetworkX |
| 工具协议 | MCP (Model Context Protocol) |
| 可观测性 | OpenTelemetry → Jaeger |

## 快速开始

```bash
cd python-impl
cp .env.example .env          # 填入 OPENAI_API_KEY
pip install -r requirements.txt
python api/main.py            # http://localhost:8000
```

Docker 方式：
```bash
docker compose up -d
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 纯文本聊天 |
| POST | `/api/chat/multimodal` | 多模态聊天（文本+图片） |
| GET | `/api/history/{session_id}` | 对话历史 |
| GET | `/api/knowledge-graph/stats` | 知识图谱统计 |
| GET | `/api/knowledge-graph/query?entity=xxx` | 实体查询+关系遍历 |
| GET | `/api/knowledge-graph/search?source=xxx&target=yyy` | 路径查找 |
| GET | `/api/tools` | MCP工具发现 |
| GET | `/api/metrics` | 系统指标 |
| GET | `/health` | 健康检查 |

## 项目结构

```
python-impl/
├── agents/          # 6个Agent：supervisor/intent/knowledge_rag/graph_rag/vision/ticket/compliance
├── api/             # FastAPI 入口 + Pydantic 模型
├── memory/          # 四层记忆：工作/短期(Redis)/长期(FAISS)/知识图谱(NetworkX)
├── mcp/             # MCP 工具协议服务端
└── tracing/         # OpenTelemetry 全链路追踪
```
