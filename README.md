# 新能源知识库智能问答系统

沿用 `business-flow.html` 的“用户提问 → 意图识别 → 知识检索 → 答案生成 → 引用溯源”流程，使用 Python、FastAPI、LangChain、RAG、FAISS 和 Docker 构建。

本版本参考公开 GitHub 同类项目的工程做法：文档库与问答 API 分离、PDF 保留页码来源、FAISS 持久化、混合检索、低置信度拒答、会话记忆、SSE 流式接口和检索评估。系统不依赖外部数据库即可运行，适合课程展示、毕业设计和小型内网部署。

## 系统架构

```text
浏览器工作台
   ↓ REST / SSE
FastAPI API
   ├─ 文档服务：PDF/TXT/Markdown 解析、上传、删除
   ├─ LangChain：段落切分（600 字，重叠 80 字）
   ├─ 检索服务：FAISS 语义相似度 + 中文词法覆盖率
   ├─ 证据门控：低于 MIN_RELEVANCE 时拒答
   ├─ 生成服务：OpenAI 兼容模型 / 离线证据摘录
   └─ SQLite：文档、会话、分析结果、评估记录
```

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
- `GET /api/documents`：知识库文档清单
- `GET /api/history/{session_id}`：多轮会话记录
- `POST /api/query/stream`：SSE 流式回答
- `POST /api/evaluate`：运行内置检索评估集

## 解释与分析

`POST /api/query` 除了返回答案和引用，还会返回 `analysis` 字段：

- `intent`：根据问题关键词识别的新能源主题
- `retrieved_chunks`：参与回答的知识片段数量
- `confidence`：最高相似度归一化后的参考置信度（仅用于提示，不替代人工审核）
- `explanation`：本次 RAG 推理过程的可读说明

这样可以在演示中清楚展示“为什么这样回答”：答案来自哪些文档、检索到多少证据，以及检索和生成分别做了什么。

## 目录结构

```text
app/main.py          # FastAPI、RAG 流程和 API
app/static/          # 问答、知识库、分析三个页面
app/flow.html        # business-flow 流程说明
data/knowledge.md    # 内置新能源示例知识
data/index/          # 持久化 FAISS 索引（运行时生成）
data/uploads/        # 用户上传文档
```

默认内置新能源示例知识，数据和 FAISS 索引保存在 `data/`。
