# 架构设计文档

## 1. 系统总体架构

### 1.1 架构概览

采用 **Supervisor 编排模式**，由一个 Supervisor Agent 统一调度 6 个专业子 Agent。

```
                      用户 (Web / App / API)
                              │  HTTP / SSE
                              ▼
                    ┌─────────────────┐
                    │  FastAPI Gateway │
                    │  认证 | 限流 | 日志 │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                   │
              ▼              ▼                   ▼
     ┌────────────┐  ┌──────────────┐  ┌─────────────────┐
     │ 短期记忆    │  │  Supervisor  │  │ 全链路追踪        │
     │ (Redis)    │◄►│  编排Agent   │─►│ (OpenTelemetry)  │
     └────────────┘  └──────┬───────┘  └─────────────────┘
                            │
       ┌────────────────────┼────────────────────────┐
       │          │         │         │              │
       ▼          ▼         ▼         ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ ┌──────────┐
│ 意图路由  │ │ 知识检索  │ │ 工单处理  │ │   合规审查        │ │ 视觉分析  │
│ Agent    │ │ RAG      │ │ CRUD     │ │ 规则+LLM         │ │ Vision   │
│ (分类)   │ │ Agent    │ │ Agent    │ │ Agent            │ │ Agent    │
└──────────┘ └────┬─────┘ └──────────┘ └──────────────────┘ └──────────┘
                  │    │
          ┌───────┘    └──────────┐
          ▼                       ▼
   ┌──────────────┐     ┌────────────────┐
   │ 长期记忆      │     │  知识图谱       │
   │ FAISS 向量库  │     │  NetworkX 图存储│
   └──────────────┘     └────────────────┘
                  │    │
                  ▼    ▼
           ┌──────────────────┐
           │  GraphRAG Agent  │
           │  混合检索         │
           │  (向量 + 图遍历)  │
           └──────────────────┘
```

### 1.2 编排流程

```
用户请求
    │
    ▼
[preprocess] ──── 检测是否含图片 ──→ has_images
    │
    ▼
[Supervisor] ── 分析意图 ──→ 路由决策
    │
    ├── knowledge_rag ──→ 知识检索Agent ──→ 合规审查 ──┐
    ├── ticket_handler ──→ 工单处理Agent ──→ 合规审查 ──┤
    ├── graph_rag ──→ 图谱推理Agent ──→ 合规审查 ──────┤
    ├── vision ──→ 视觉分析Agent ──→ 合规审查 ─────────┤
    └── compliance_checker ──→ 合规审查Agent ──────────┤
                                                        │
                             ┌──────────────────────────┘
                             ▼
                      [Supervisor 汇总] ──→ 响应用户
```

---

## 2. 核心组件设计

### 2.1 Agent 清单

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| Supervisor | 编排调度、结果汇总 | 用户消息 + 全局State | 路由决策 + 最终回复 |
| 意图路由 | 意图分类 | 用户消息 | intent 标签 + 置信度 |
| 知识检索 (RAG) | 向量语义搜索 + 生成 | 用户问题 | 基于文档的回答 |
| 工单处理 | 工单 CRUD | 用户需求 | 工单号 + 状态 |
| 合规审查 | 内容审查 | Agent 回复 | 通过/不通过 + 违规项 |
| GraphRAG | 图谱关系推理 | 用户查询 | 多跳推理回答 |
| Vision | 图片/文档分析 | base64 图片 | 结构化描述 |

### 2.2 State 设计

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]      # 对话消息列表
    user_id: str                      # 用户标识
    session_id: str                   # 会话标识
    intent: str                       # 路由意图
    sub_results: dict[str, Any]       # 各Agent输出
    compliance_passed: bool           # 合规审查结果
    final_response: str               # 最终回复
    current_agent: str                # 当前执行Agent
    retry_count: int                  # 重试次数
    # v2 新增
    has_images: bool                  # 是否含图片
    images: list[str]                 # base64 图片
    graph_context: dict[str, Any]     # 图谱检索结果
    vision_results: dict[str, Any]    # 视觉分析结果
```

### 2.3 分层记忆架构

```
┌──────────────────────────────────────────────┐
│  短期记忆 (Short-term Memory)                │
│  ├── 存储: Redis (内存 fallback)             │
│  ├── 生命周期: TTL 30分钟                     │
│  └── 用途: 每次请求加载历史注入 state          │
│            Supervisor + 所有子 Agent 感知多轮  │
├──────────────────────────────────────────────┤
│  长期记忆 (Long-term Memory)                 │
│  ├── 存储: FAISS 向量索引                    │
│  ├── 生命周期: 持久化磁盘                     │
│  └── 用途: 语义检索、RAG 文档库               │
├──────────────────────────────────────────────┤
│  知识图谱 (Knowledge Graph)                  │
│  ├── 存储: NetworkX 有向图                    │
│  ├── 生命周期: 持久化 JSON                    │
│  └── 用途: 结构化关系推理、多跳查询            │
└──────────────────────────────────────────────┘
```

---

## 3. GraphRAG 混合检索流程

```
用户问题: "X1手机配什么充电器最好？"
        │
        ▼
[实体提取] ── LLM ──→ [{name: "X1手机", type: "Product"}]
        │
        ├──────────────────┐
        ▼                  ▼
[向量检索]              [图谱遍历]
FAISS.search("X1手机")  KG.search("X1手机") → neighbors(hops=2)
→ 相关文档               → compatible_with: [65W充电器, 30W无线充]
        │                  │
        └────────┬─────────┘
                 ▼
        [上下文合并]
        文档片段 + 图谱子图
                 │
                 ▼
        [LLM 生成回答]
        "X1支持65W氮化镓充电器(129元)和30W无线充电板(199元)，
         推荐65W版本，充电更快且性价比高。"
```

---

## 4. 知识图谱数据模型

### 4.1 实体类型

| 类型 | 示例 | 说明 |
|------|------|------|
| Product | X1智能手机, Pad Air, Watch GT | 商品 |
| Brand | 星耀 | 品牌 |
| Category | 智能手机, 平板电脑 | 品类 |
| Policy | 7天无理由退换, 碎屏保 | 政策规则 |
| Process | 退款流程, 下单流程 | 业务流程 |
| Service | 客服热线, 官方维修 | 售后服务 |

### 4.2 关系类型

| 类型 | 示例 | 含义 |
|------|------|------|
| compatible_with | 充电器 → X1手机 | 配件兼容 |
| ecosystem_sync | X1手机 → Pad Air | 生态互联 |
| competitor | X1手机 → Mate 70 | 竞品关系 |
| bundle_recommend | Pad Air → X1手机 | 搭配推荐 |
| applies_to | 退货政策 → X1手机 | 政策覆盖 |
| depends_on | 退款流程 → 退货政策 | 前置依赖 |

---

## 5. MCP 工具协议

```
┌─────────────┐    JSON-RPC 2.0    ┌─────────────────────────┐
│   Agent      │ ◄────────────────► │  MCP Tool Server        │
│              │                    │                         │
│ tools/list   │ ──── 发现 ─────►  │  ┌─order_query         │
│ tools/call   │ ──── 调用 ─────►  │  ├─knowledge_search    │
│              │ ◄─── 结果 ──────  │  ├─ticket_create       │
│              │                    │  ├─risk_check          │
│              │                    │  ├─graph_search (NEW)  │
│              │                    │  └─image_analysis (NEW)│
└─────────────┘                    └─────────────────────────┘
```

---

## 6. 可观测性

### Span 层级

```
[Root] user_request
  ├── [Span] preprocess
  ├── [Span] supervisor.route_decision
  ├── [Span] graphrag.extract_entities
  ├── [Span] graphrag.vector_search
  ├── [Span] graphrag.graph_traverse
  ├── [Span] graphrag.generate_answer
  ├── [Span] vision.analyze_image
  ├── [Span] compliance.rule_check
  ├── [Span] compliance.llm_check
  └── [Span] supervisor.synthesize
```
