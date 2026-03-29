"""
Offline Ingestion Script — Run Locally Only
============================================
This script chunks textbooks, embeds them using all-MiniLM-L6-v2,
and batch-upserts to a Pinecone cloud vector database.

Usage:
    python scripts/ingest_to_pinecone.py --data-dir ./Data --index-name blackitab-books

Prerequisites:
    1. pip install pinecone-client sentence-transformers langchain-text-splitters
    2. Set PINECONE_API_KEY in your .env or environment
    3. Create a Pinecone index with dimension=384, metric=dotproduct

This script is NOT meant for deployment — it runs on your local machine
where you have enough RAM to load and process textbooks.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 100  # Pinecone upsert batch size
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Markdown-aware separators for better academic text splitting
MARKDOWN_SEPARATORS = [
    "\n## ",       # H2 headers
    "\n### ",      # H3 headers
    "\n#### ",     # H4 headers
    "\n\n",        # Paragraph breaks
    "\n",          # Line breaks
    ". ",          # Sentence breaks
    " ",           # Word breaks
]


def load_documents(data_dir: str) -> List[Any]:
    """Load documents using LangChain loaders."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        sys.exit(1)
    
    documents = []
    
    # Load PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            # Enrich metadata with source info
            for doc in loaded:
                doc.metadata["source_file"] = pdf_file.stem
                # Try to extract subject from directory structure
                relative = pdf_file.relative_to(data_path)
                if len(relative.parts) > 1:
                    doc.metadata["subject"] = relative.parts[0]
            documents.extend(loaded)
            print(f"  → Loaded {len(loaded)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {pdf_file.name}: {e}")
    
    # Load text/markdown files
    text_files = list(data_path.glob("**/*.txt")) + list(data_path.glob("**/*.md"))
    print(f"[INFO] Found {len(text_files)} text/markdown files")
    for txt_file in text_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = txt_file.stem
                relative = txt_file.relative_to(data_path)
                if len(relative.parts) > 1:
                    doc.metadata["subject"] = relative.parts[0]
            documents.extend(loaded)
            print(f"  → Loaded {len(loaded)} docs from {txt_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {txt_file.name}: {e}")
    
    # Filter invalid documents
    valid = [d for d in documents if getattr(d, "page_content", None) and isinstance(d.page_content, str) and d.page_content.strip()]
    print(f"[INFO] Total valid documents: {len(valid)} / {len(documents)}")
    return valid


def chunk_documents(documents: List[Any]) -> List[Any]:
    """Split documents into chunks using markdown-aware splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=MARKDOWN_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_and_upsert(chunks: List[Any], index_name: str):
    """Embed chunks and batch-upsert to Pinecone."""
    from pinecone import Pinecone
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ERROR] PINECONE_API_KEY not set in environment!")
        sys.exit(1)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print(f"[INFO] Connected to Pinecone index: {index_name}")
    
    # Initialize embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[INFO] Loaded embedding model: {EMBEDDING_MODEL}")
    
    # Process in batches of BATCH_SIZE
    total_upserted = 0
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_idx:batch_idx + BATCH_SIZE]
        current_batch = batch_idx // BATCH_SIZE + 1
        
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        # Build upsert vectors with metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            vector_id = f"chunk_{batch_idx + i}"
            metadata = {
                "texts": chunk.page_content[:2000],  # Pinecone metadata limit
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "subject": chunk.metadata.get("subject", ""),
                "chapter": chunk.metadata.get("chapter", ""),
                "page": chunk.metadata.get("page", 0),
            }
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            })
        
        # Upsert batch
        try:
            index.upsert(vectors=vectors)
            total_upserted += len(vectors)
            print(f"  [Batch {current_batch}/{total_batches}] Upserted {len(vectors)} vectors (total: {total_upserted})")
        except Exception as e:
            print(f"  ✗ [Batch {current_batch}] Upsert failed: {e}")
    
    print(f"\n[SUCCESS] Total vectors upserted: {total_upserted}")
    
    # Verify
    stats = index.describe_index_stats()
    print(f"[INFO] Index stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Ingest textbooks into Pinecone vector database")
    parser.add_argument("--data-dir", default="./Data", help="Path to data directory containing textbooks")
    parser.add_argument("--index-name", default="blackitab-books", help="Pinecone index name")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Blackitab — Offline Textbook Ingestion Pipeline")
    print("=" * 60)
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Pinecone Index: {args.index_name}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("=" * 60)
    
    # Step 1: Load
    documents = load_documents(args.data_dir)
    if not documents:
        print("[ERROR] No documents loaded. Exiting.")
        sys.exit(1)
    
    # Step 2: Chunk
    chunks = chunk_documents(documents)
    if not chunks:
        print("[ERROR] No chunks generated. Exiting.")
        sys.exit(1)
    
    # Step 3: Embed & Upsert
    embed_and_upsert(chunks, args.index_name)
    
    print("\n[DONE] Ingestion complete!")


if __name__ == "__main__":
    main()
