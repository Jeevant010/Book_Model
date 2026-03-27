import os
from unidecode import unidecode
from dotenv import load_dotenv
from typing import List, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_text_splitters import MarkdownTextSplitter

from src.data_loader import load_all_documents

# Try to load existing local env vars
load_dotenv()

def chunk_markdown(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Markdown splitter produced {len(chunks)} chunks.")
    return chunks

def batch_upsert_pinecone(
    chunks: List[Any], 
    index_name: str, 
    embedding_model_name: str = "all-MiniLM-L6-v2", 
    batch_size: int = 100
):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ERROR] PINECONE_API_KEY not found in environment. Please set it in .env")
        return

    print(f"[INFO] Connecting to Pinecone Index: {index_name}")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print(f"[INFO] Loading SentenceTransformer: {embedding_model_name}")
    dense_model = SentenceTransformer(embedding_model_name)
    
    print(f"[INFO] Loading BM25Encoder with default vocabulary for sparse vectors...")
    bm25 = BM25Encoder().default()

    total_chunks = len(chunks)
    print(f"[INFO] Commencing batch upsert for {total_chunks} total chunks (batch_size={batch_size})...")

    # Clean text helper
    def clean_text(text: str) -> str:
        # Convert smart quotes to regular quotes and strip out rare unicodes that might cause issues.
        text = unidecode(text)
        return text

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        upsert_payload = []
        texts_to_embed = []
        
        # Prepare valid texts
        for doc in batch:
            text = clean_text(getattr(doc, 'page_content', ''))
            texts_to_embed.append(text)
            
        # 1. Generate dense vectors
        try:
            dense_vectors = dense_model.encode(texts_to_embed, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"[ERROR] Failed to generate dense vectors for batch {i}: {e}")
            continue

        # 2. Generate sparse vectors
        try:
            sparse_vectors = bm25.encode_documents(texts_to_embed)
        except Exception as e:
            print(f"[ERROR] Failed to generate sparse vectors for batch {i}: {e}")
            continue

        for j, doc in enumerate(batch):
            text = texts_to_embed[j]
            if not text.strip():
                continue
                
            # Create a unique ID for each chunk based on index, or hash
            chunk_id = f"chunk_{i + j}"
            
            # Prepare metadata
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") and doc.metadata else {}
            metadata["texts"] = text
            
            # Extract subject / chapter if present in the data folder structure (optional)
            # This allows metadata pre-filtering to work well! Ensure only strings/booleans/numbers
            for key, val in metadata.items():
                if val is None:
                    metadata[key] = ""
                    
            upsert_payload.append({
                "id": chunk_id,
                "values": dense_vectors[j],
                "sparse_values": sparse_vectors[j],
                "metadata": metadata
            })
            
        # 3. Batch Upsert 
        if upsert_payload:
            try:
                index.upsert(vectors=upsert_payload)
                print(f"[INFO] Upserted batch {i} to {i + len(upsert_payload)} / {total_chunks} successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to upsert batch {i}: {e}")
                
    print("[INFO] Offline ingestion complete.")

if __name__ == "__main__":
    print("-" * 50)
    print("Welcome to the Offline Ingestion Pipeline for RAG.")
    print("-" * 50)
    
    # Setup Paths
    data_dirs = ["Research/data", "data", "Data"]
    docs = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"[INFO] Loading documents from: {data_dir}")
            docs = load_all_documents(data_dir)
            if docs:
                break
                
    if not docs:
        print("[WARNING] No documents found to process. Please ensure textbooks are in the data directory.")
        exit(1)
        
    print(f"\n[INFO] Starting markdown-aware chunking on {len(docs)} documents...")
    chunks = chunk_markdown(docs)
    
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "book-model-index")
    batch_upsert_pinecone(chunks, pinecone_index)
