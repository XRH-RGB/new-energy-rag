import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPLOADS, INDEX_DIR = DATA / "uploads", DATA / "index"
UPLOADS.mkdir(parents=True, exist_ok=True); INDEX_DIR.mkdir(parents=True, exist_ok=True)
DIM = int(os.getenv("EMBEDDING_DIM", "384"))
INDEX_FILE, META_FILE = INDEX_DIR / "vectors.faiss", INDEX_DIR / "metadata.json"

app = FastAPI(title="新能源知识库智能问答系统", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

def embed(text: str) -> np.ndarray:
    # 无外部模型时的可复现本地向量；生产环境可替换为 HuggingFace/OpenAI Embeddings。
    v = np.zeros(DIM, dtype="float32")
    for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower()):
        h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v

def load_docs() -> List[Document]:
    docs = []
    for path in [DATA / "knowledge.md", *UPLOADS.glob("*.md"), *UPLOADS.glob("*.txt")]:
        docs.extend(splitter.split_documents([Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})]))
    return docs

def rebuild() -> int:
    docs = load_docs(); index = faiss.IndexFlatIP(DIM)
    if docs: index.add(np.vstack([embed(d.page_content) for d in docs]))
    faiss.write_index(index, str(INDEX_FILE)); META_FILE.write_text(json.dumps([{"text": d.page_content, "source": d.metadata.get("source", "unknown")} for d in docs], ensure_ascii=False), encoding="utf-8")
    return len(docs)

def classify_intent(question: str) -> str:
    q = question.lower()
    rules = {
        "光伏": "光伏发电",
        "组件": "光伏发电",
        "风机": "风电运维",
        "齿轮箱": "风电运维",
        "储能": "储能安全",
        "电池": "储能安全",
        "氢": "氢能制备",
        "电解": "氢能制备",
    }
    for word, intent in rules.items():
        if word in q:
            return intent
    return "新能源综合咨询"

def retrieve(question: str, k: int = 4):
    if not INDEX_FILE.exists(): rebuild()
    index = faiss.read_index(str(INDEX_FILE)); meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else []
    if not meta: return []
    scores, ids = index.search(np.array([embed(question)]), min(k, len(meta)))
    return [{**meta[i], "score": round(float(scores[0][pos]), 4)} for pos, i in enumerate(ids[0]) if i >= 0]

class Query(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)

@app.on_event("startup")
def startup():
    if not INDEX_FILE.exists(): rebuild()

@app.get("/api/health")
def health():
    count = len(json.loads(META_FILE.read_text(encoding="utf-8"))) if META_FILE.exists() else 0
    return {"status": "ok", "service": "new-energy-rag", "chunks": count, "flow": ["intent", "retrieve", "generate", "cite"]}

@app.get("/flow", response_class=HTMLResponse)
def flow():
    return (ROOT / "app" / "flow.html").read_text(encoding="utf-8")

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in {".md", ".txt", ".pdf"}:
        raise HTTPException(400, "目前支持 .md、.txt 和 .pdf 文件")
    target = UPLOADS / Path(file.filename).name
    raw = await file.read(); target.write_bytes(raw)
    if target.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(target)).pages)
        target.with_suffix(".txt").write_text(text, encoding="utf-8")
    count = rebuild()
    return {"message": "知识已入库", "file": target.name, "chunks": count}

@app.post("/api/query")
def query(body: Query):
    hits = retrieve(body.question, body.top_k)
    intent = classify_intent(body.question)
    if not hits:
        return {"answer": "知识库暂无相关内容。", "citations": [], "flow": [], "analysis": {"intent": intent, "retrieved_chunks": 0, "confidence": 0, "explanation": "没有找到可作为依据的知识片段。"}}
    context = "\n\n".join(h["text"] for h in hits)
    answer = "基于知识库检索结果：\n" + context
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            answer = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).invoke(f"请仅根据资料回答问题并注明依据。\n资料：{context}\n问题：{body.question}").content
        except Exception:
            pass
    confidence = round(max(0.0, min(1.0, (hits[0]["score"] + 1) / 2)), 2)
    return {"answer": answer, "citations": [{"source": h["source"], "snippet": h["text"][:160], "score": h["score"]} for h in hits], "flow": ["intent", "retrieve", "generate", "cite"], "analysis": {"intent": intent, "retrieved_chunks": len(hits), "confidence": confidence, "explanation": "先按问题关键词识别主题，再用 FAISS 相似度召回片段，最后基于召回内容生成答案并保留引用。"}}
