import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.search import RAGSearch, RetrievalResult

import uvicorn

load_dotenv()

# Internal API key for Express → FastAPI bridge security
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

# Global variable for RAG system
rag_search: Optional[RAGSearch] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_search
    try:
        persist_dir = os.getenv("PERSIST_DIR", "faiss_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        rag_search = RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )
        print("[INFO] RAG system loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load RAG system: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    print("[INFO] Shutting down RAG system")

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="RAG Question Answering API",
    description="FAISS + SentenceTransformers + Groq LLM",
    version="2.1.0",
    lifespan=lifespan,
    root_path=os.getenv("ROOT_PATH", ""),
)

# -------------------------
# Internal API Key Middleware
# Rejects any non-health request lacking the correct X-Internal-API-Key header
# -------------------------
@app.middleware("http")
async def verify_internal_api_key(request: Request, call_next):
    # Allow health and root endpoints without auth
    if request.url.path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    
    if not INTERNAL_API_KEY:
        # If no key is configured, allow all (dev mode)
        return await call_next(request)
    
    provided_key = request.headers.get("X-Internal-API-Key", "")
    if provided_key != INTERNAL_API_KEY:
        return JSONResponse(
            status_code=403,
            content={"detail": "Forbidden: Invalid or missing internal API key"}
        )
    
    return await call_next(request)

# CORS for React/Node clients
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(     
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins] if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Models
# -------------------------
class SourceItem(BaseModel):
    index: int
    distance: float
    text: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    subject: Optional[str] = None   # Metadata filter: subject name
    chapter: Optional[str] = None   # Metadata filter: chapter name

class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[SourceItem]


# -------------------------
# Query Rewriting Helper
# -------------------------
def rewrite_query_for_search(raw_query: str) -> str:
    """
    Uses a preliminary Groq LLM call to rewrite messy student queries 
    into highly specific, academic keyword queries for better embedding search.
    """
    if not rag_search or not rag_search.llm:
        return raw_query
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_msg = SystemMessage(content=(
            "You are a query rewriter for an academic search engine. "
            "Rewrite the following student question into a concise, keyword-rich academic query "
            "that will maximize retrieval from a textbook vector database. "
            "Output ONLY the rewritten query, nothing else. "
            "Keep it under 50 words. Use formal academic terminology."
        ))
        human_msg = HumanMessage(content=f"Student question: {raw_query}")
        
        response = rag_search.llm.invoke([system_msg, human_msg])
        rewritten = response.content.strip()
        
        # Sanity check — if rewriter returns garbage or is too long, use original
        if len(rewritten) > 0 and len(rewritten) < 500:
            print(f"[INFO] Query rewritten: '{raw_query[:50]}...' → '{rewritten[:50]}...'")
            return rewritten
    except Exception as e:
        print(f"[WARN] Query rewriting failed, using original: {e}")
    
    return raw_query


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running. Go to /docs"}

@app.get("/health")
def health():
    if not rag_search:
        return {"ready": False}
    meta_count = len(rag_search.vectorstore.metadata) if rag_search.vectorstore else 0
    return {
        "ready": True,
        "persist_dir": rag_search.vectorstore.persist_dir if rag_search.vectorstore else None,
        "documents_indexed": meta_count,
        "embedding_model": rag_search.embedding_model,
        "llm_model": rag_search.llm_model,
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not rag_search:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    try:
        # Step 1: Rewrite the query for better academic retrieval
        rewritten_query = rewrite_query_for_search(payload.query)
        
        # Step 2: Build metadata filters from optional subject/chapter
        metadata_filter = {}
        if payload.subject:
            metadata_filter["subject"] = payload.subject
        if payload.chapter:
            metadata_filter["chapter"] = payload.chapter
        
        # Step 3: Retrieve with optional metadata filtering
        sources: List[RetrievalResult] = rag_search.retrieve(
            rewritten_query, 
            top_k=payload.top_k,
            metadata_filter=metadata_filter if metadata_filter else None
        )
        
        # Step 4: Summarize with original query for natural answer
        answer: str = rag_search.summarize(payload.query, sources)

        # Map sources for response
        resp_sources = [
            SourceItem(index=s.index, distance=float(s.distance), text=s.text)
            for s in sources
        ]
        return QueryResponse(
            query=payload.query, 
            rewritten_query=rewritten_query if rewritten_query != payload.query else None,
            answer=answer, 
            sources=resp_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=True
    )