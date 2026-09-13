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

## 解释与分析

`POST /api/query` 除了返回答案和引用，还会返回 `analysis` 字段：

- `intent`：根据问题关键词识别的新能源主题
- `retrieved_chunks`：参与回答的知识片段数量
- `confidence`：最高相似度归一化后的参考置信度（仅用于提示，不替代人工审核）
- `explanation`：本次 RAG 推理过程的可读说明

这样可以在演示中清楚展示“为什么这样回答”：答案来自哪些文档、检索到多少证据，以及检索和生成分别做了什么。

默认内置新能源示例知识，数据和 FAISS 索引保存在 `data/`。
