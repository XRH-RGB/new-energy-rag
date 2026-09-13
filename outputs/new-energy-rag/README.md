# 新能源知识库智能问答系统

沿用 `business-flow.html` 的“用户提问 → 意图识别 → 知识检索 → 答案生成 → 引用溯源”流程，使用 Python、FastAPI、LangChain、RAG、FAISS 和 Docker 构建。

## 快速启动

```bash
cp .env.example .env
docker compose up --build
```

打开 http://localhost:8000/docs。没有配置 OpenAI Key 时，系统仍可使用本地检索结果作为可验证的回答；配置 `OPENAI_API_KEY` 后启用 LLM 生成。

## API

- `POST /api/ingest`：上传 `.txt/.md/.pdf` 知识文件
- `POST /api/query`：`{"question":"光伏组件的衰减率是多少？"}`
- `GET /api/health`：服务与索引状态

默认内置新能源示例知识，数据和 FAISS 索引保存在 `data/`。
