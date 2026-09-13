"""新能源知识库 RAG 服务。

流程：文档解析 -> LangChain 切分 -> FAISS + 词法混合检索 -> 证据门控 -> LLM/原文回答。
所有回答都带来源与可解释指标，便于工程核验和项目展示。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPLOADS, INDEX_DIR = DATA / "uploads", DATA / "index"
UPLOADS.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "rag.db"
DIM = int(os.getenv("EMBEDDING_DIM", "384"))
INDEX_FILE, META_FILE = INDEX_DIR / "vectors.faiss", INDEX_DIR / "metadata.json"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.16"))

app = FastAPI(title="新能源知识库智能问答系统", version="2.0.0", description="可解释、可追溯的新能源 RAG 知识服务")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
          id TEXT PRIMARY KEY, filename TEXT NOT NULL, source TEXT NOT NULL,
          file_type TEXT NOT NULL, size INTEGER NOT NULL DEFAULT 0,
          chunks INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'indexed', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
          content TEXT NOT NULL, analysis TEXT, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        CREATE TABLE IF NOT EXISTS evaluations(
          id TEXT PRIMARY KEY, question TEXT NOT NULL, expected_source TEXT,
          top_source TEXT, score REAL, passed INTEGER NOT NULL, created_at TEXT NOT NULL);
        """)
        built_in = DATA / "knowledge.md"
        if built_in.exists():
            conn.execute("INSERT OR IGNORE INTO documents(id,filename,source,file_type,size,status,created_at) VALUES(?,?,?,?,?,?,?)", ("builtin-knowledge", built_in.name, built_in.name, "md", built_in.stat().st_size, "indexed", now()))


def embed(text: str) -> np.ndarray:
    """可复现的离线向量；配置 OPENAI_API_KEY 后可替换为生产 embedding 服务。"""
    vector = np.zeros(DIM, dtype="float32")
    for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower()):
        digest = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        vector[digest % DIM] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def read_sources() -> list[Document]:
    paths = [DATA / "knowledge.md", *sorted(UPLOADS.glob("*.md")), *sorted(UPLOADS.glob("*.txt"))]
    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append(Document(page_content=text, metadata={"source": path.name, "page": None}))
    for path in sorted(UPLOADS.glob("*.pdf")):
        try:
            from pypdf import PdfReader
            for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": path.name, "page": page_number}))
        except Exception:
            continue
    return documents


def rebuild_index() -> int:
    chunks: list[Document] = []
    for source in read_sources():
        chunks.extend(splitter.split_documents([source]))
    index = faiss.IndexFlatIP(DIM)
    if chunks:
        index.add(np.vstack([embed(chunk.page_content) for chunk in chunks]))
    faiss.write_index(index, str(INDEX_FILE))
    metadata = [{"text": c.page_content, "source": c.metadata.get("source", "unknown"), "page": c.metadata.get("page")} for c in chunks]
    META_FILE.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    with db() as conn:
        conn.execute("UPDATE documents SET chunks=0")
        counts: dict[str, int] = {}
        for item in metadata:
            counts[item["source"]] = counts.get(item["source"], 0) + 1
        for source, count in counts.items():
            conn.execute("UPDATE documents SET chunks=?, status='indexed' WHERE filename=?", (count, source))
    return len(chunks)


def tokenize(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.add(run)
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def classify_intent(question: str) -> str:
    rules = {"光伏": "光伏发电", "组件": "光伏发电", "风机": "风电运维", "齿轮箱": "风电运维", "储能": "储能安全", "电池": "储能安全", "氢": "氢能制备", "电解": "氢能制备"}
    for key, label in rules.items():
        if key in question.lower():
            return label
    return "新能源综合咨询"


def retrieve(question: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not INDEX_FILE.exists():
        rebuild_index()
    metadata = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else []
    if not metadata:
        return [], {"method": "FAISS + 词法混合检索", "candidate_count": 0, "selected_count": 0}
    index = faiss.read_index(str(INDEX_FILE))
    dense_scores, ids = index.search(np.array([embed(question)]), len(metadata))
    query_terms = tokenize(question)
    ranked = []
    for pos, idx in enumerate(ids[0]):
        if idx < 0:
            continue
        item = metadata[int(idx)]
        lexical = len(query_terms & tokenize(item["text"])) / max(1, len(query_terms))
        dense = max(0.0, float(dense_scores[0][pos]))
        ranked.append({**item, "dense_score": round(dense, 4), "lexical_score": round(lexical, 4), "relevance": round(0.65 * dense + 0.35 * lexical, 4)})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    selected = [item for item in ranked[:top_k] if item["relevance"] >= MIN_RELEVANCE]
    metrics = {"method": "FAISS + 词法混合检索", "candidate_count": len(ranked), "selected_count": len(selected), "threshold": MIN_RELEVANCE, "max_relevance": ranked[0]["relevance"] if ranked else 0}
    return selected, metrics


def model_answer(question: str, context: str, history: list[dict[str, str]]) -> tuple[str, str]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            messages = [("system", "你是新能源工程知识助手。只根据给定资料回答，用中文，资料不足要明确说不知道，并在结论后标注来源名称。"), *[(m["role"], m["content"]) for m in history[-6:]], ("user", f"问题：{question}\n资料：\n{context}")]
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke(messages).content, "LLM 基于证据生成"
        except Exception:
            pass
    return "基于知识库检索结果：\n" + context, "离线证据摘录（未配置 LLM）"


class Query(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)
    session_id: str = Field(default="default", min_length=1, max_length=80)


@app.on_event("startup")
def startup() -> None:
    init_db()
    if not INDEX_FILE.exists():
        rebuild_index()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/flow", response_class=HTMLResponse)
def flow() -> str:
    return (ROOT / "app" / "flow.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    with db() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    return {"status": "ok", "service": "new-energy-rag", "version": app.version, "documents": docs, "index_ready": INDEX_FILE.exists()}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    with db() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    chunks = len(json.loads(META_FILE.read_text(encoding="utf-8"))) if META_FILE.exists() else 0
    return {"documents": docs, "chunks": chunks, "messages": messages, "retrieval": "FAISS + 词法混合检索", "llm": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    with db() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM documents ORDER BY created_at DESC")]
    return {"items": rows}


@app.post("/api/reindex")
def reindex() -> dict[str, Any]:
    return {"message": "索引重建完成", "chunks": rebuild_index()}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or Path(file.filename).suffix.lower() not in {".md", ".txt", ".pdf"}:
        raise HTTPException(400, "目前支持 .md、.txt 和 .pdf 文件")
    safe_name = Path(file.filename).name
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "单个文件不能超过 20 MB")
    target = UPLOADS / safe_name
    target.write_bytes(raw)
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO documents(id,filename,source,file_type,size,status,created_at) VALUES(?,?,?,?,?,?,?)", (uuid.uuid4().hex, safe_name, safe_name, target.suffix.lower()[1:], len(raw), "indexing", now()))
    count = rebuild_index()
    return {"message": "知识已入库并完成索引", "file": safe_name, "chunks": count}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str) -> dict[str, str]:
    safe = Path(filename).name
    if safe == "knowledge.md":
        raise HTTPException(400, "内置示例知识不能删除")
    target = UPLOADS / safe
    if target.exists():
        target.unlink()
    with db() as conn:
        conn.execute("DELETE FROM documents WHERE filename=?", (safe,))
    rebuild_index()
    return {"message": "文档已删除"}


def save_message(session_id: str, role: str, content: str, analysis: dict[str, Any] | None = None) -> None:
    with db() as conn:
        conn.execute("INSERT INTO messages VALUES(?,?,?,?,?)", (uuid.uuid4().hex, session_id, role, content, json.dumps(analysis or {}, ensure_ascii=False), now()))


@app.get("/api/history/{session_id}")
def history(session_id: str) -> dict[str, Any]:
    with db() as conn:
        rows = [dict(row) for row in conn.execute("SELECT role,content,analysis,created_at FROM messages WHERE session_id=? ORDER BY created_at", (session_id,))]
    return {"items": rows}


def answer_payload(body: Query) -> dict[str, Any]:
    started = time.perf_counter()
    hits, retrieval = retrieve(body.question, body.top_k)
    intent = classify_intent(body.question)
    with db() as conn:
        history_rows = [dict(row) for row in conn.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT 6", (body.session_id,))]
    history_rows.reverse()
    if not hits:
        answer = "当前知识库没有找到足够依据，建议补充设备型号、时间范围或上传相关技术资料。"
        mode, generation = "低置信度拒答", "证据门控"
    else:
        context = "\n\n".join(f"[{i + 1}] {hit['text']}" for i, hit in enumerate(hits))
        answer, generation = model_answer(body.question, context, history_rows)
        mode = "混合检索 + " + generation
    analysis = {"intent": intent, "retrieval": retrieval, "generation": generation, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "explanation": "先识别主题，再融合 FAISS 语义相似度与词法覆盖率排序；低于证据门槛时拒答，达到门槛后依据片段生成并返回引用。"}
    save_message(body.session_id, "user", body.question)
    save_message(body.session_id, "assistant", answer, analysis)
    return {"answer": answer, "mode": mode, "session_id": body.session_id, "citations": [{"source": h["source"], "page": h.get("page"), "snippet": h["text"][:240], "dense_score": h["dense_score"], "lexical_score": h["lexical_score"], "relevance": h["relevance"]} for h in hits], "analysis": analysis, "flow": ["intent", "retrieve", "gate", "generate", "cite"]}


@app.post("/api/query")
def query(body: Query) -> dict[str, Any]:
    return answer_payload(body)


@app.post("/api/query/stream")
def query_stream(body: Query) -> StreamingResponse:
    payload = answer_payload(body)
    def events():
        yield f"event: analysis\ndata: {json.dumps(payload['analysis'], ensure_ascii=False)}\n\n"
        for word in payload["answer"].split(" "):
            yield f"data: {json.dumps({'delta': word + ' '}, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/evaluate")
def evaluate() -> dict[str, Any]:
    cases = [("光伏组件的衰减率是多少？", "knowledge.md"), ("储能系统需要哪些安全配置？", "knowledge.md"), ("PEM 电解适合什么场景？", "knowledge.md")]
    results = []
    for question, expected in cases:
        hits, _ = retrieve(question, 3)
        top = hits[0]["source"] if hits else ""
        passed = bool(hits) and top == expected
        results.append({"question": question, "expected_source": expected, "top_source": top, "score": hits[0]["relevance"] if hits else 0, "passed": passed})
        with db() as conn:
            conn.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?)", (uuid.uuid4().hex, question, expected, top, hits[0]["relevance"] if hits else 0, int(passed), now()))
    return {"total": len(results), "passed": sum(int(item["passed"]) for item in results), "results": results}
