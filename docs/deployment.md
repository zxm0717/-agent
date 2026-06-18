# 部署指南

## 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 运行环境 |
| Redis | 7.x | 短期记忆存储（可选，不装则用内存回退） |
| Docker | 24+ | 容器化部署（可选） |

---

## 2. 本地开发

### 2.1 安装依赖

```bash
cd python-impl
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```ini
OPENAI_API_KEY=sk-your-key       # 必填
MODEL_NAME=gpt-4o                # 默认即可
REDIS_URL=redis://localhost:6379/0  # 可选，不填则使用内存回退
GRAPH_STORE_PATH=./graph_store/graph.json
```

### 2.3 启动

```bash
python api/main.py
# 或
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/health` 验证启动成功。

---

## 3. Docker 部署

```bash
# 在项目根目录
docker compose up -d
```

启动 3 个服务：

| 服务 | 端口 | 说明 |
|------|------|------|
| python-agent | 8000 | FastAPI 应用 |
| redis | 6379 | 短期记忆 |
| jaeger | 16686 | 追踪可视化 |

查看日志：

```bash
docker compose logs -f python-agent
```

停止：

```bash
docker compose down
```

---

## 4. 验证

### 4.1 健康检查

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "2.0.0"}
```

### 4.2 文本聊天

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "X1手机支持哪些配件？", "user_id": "test"}'
```

### 4.3 知识图谱查询

```bash
curl "http://localhost:8000/api/knowledge-graph/query?entity=X1"
curl "http://localhost:8000/api/knowledge-graph/stats"
```

### 4.4 多模态聊天

```bash
curl -X POST http://localhost:8000/api/chat/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "message": "这张截图里有什么错误？",
    "user_id": "test",
    "images": ["<base64图片数据>"]
  }'
```

---

## 5. 常见问题

### Redis 连接失败

不装 Redis 时系统自动降级为内存存储（`ShortTermMemory` 使用 `_fallback_store`），不影响功能但重启后对话历史丢失。

### FAISS 未安装

部分 Windows 环境下 `faiss-cpu` 安装可能失败。`LongTermMemory` 会自动回退到关键词匹配搜索。

### 图片分析无响应

Vision 功能依赖 GPT-4o 多模态能力，确保 `MODEL_NAME` 配置为支持视觉的模型。
