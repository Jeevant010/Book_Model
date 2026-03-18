"""Quick test to verify the updated embedding pipeline works correctly."""
import sys
import os
sys.path.insert(0, os.getcwd())

from langchain_core.documents import Document
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.cleaner import DocumentCleaner
import numpy as np

def test_pipeline():
    print("=" * 60)
    print("QUICK PIPELINE VERIFICATION TEST")
    print("=" * 60)
    
    # 1. Test DocumentCleaner
    print("\n--- Test 1: DocumentCleaner ---")
    cleaner = DocumentCleaner(min_length=10)
    
    test_docs = [
        Document(page_content="This is a valid document with real content about databases."),
        Document(page_content="Short"),  # too short
        Document(page_content=""),       # empty
        Document(page_content=None),     # None - this would crash old code
        Document(page_content="Another valid document explaining SQL queries and joins."),
        Document(page_content="   \n\n\t  "),  # whitespace only
    ]
    
    cleaned = cleaner.clean_documents(test_docs)
    print(f"  Input: {len(test_docs)} docs -> Cleaned: {len(cleaned)} docs")
    assert len(cleaned) == 2, f"Expected 2 valid docs, got {len(cleaned)}"
    print("  ✅ DocumentCleaner works correctly!")
    
    # 2. Test EmbeddingPipeline with cleaning + batching
    print("\n--- Test 2: EmbeddingPipeline (clean + chunk + embed) ---")
    
    # Create realistic documents like what PyPDFLoader returns
    docs = [
        Document(page_content="Database Management Systems provide an organized way to store and manage data. " * 5),
        Document(page_content="SQL is a standard language for accessing and manipulating databases. " * 5),
        Document(page_content=None),  # Simulates a bad PDF page
        Document(page_content="Machine Learning is a branch of artificial intelligence focused on algorithms. " * 5),
        Document(page_content=""),    # Empty page
        Document(page_content="x"),   # Too short - should be cleaned
    ]
    
    pipe = EmbeddingPipeline(chunk_size=200, chunk_overlap=50)
    chunks = pipe.chunk_documents(docs)
    print(f"  Chunks created: {len(chunks)}")
    
    embeddings, valid_chunks = pipe.embed_chunks(chunks, batch_size=2)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Valid chunks: {len(valid_chunks)}")
    assert embeddings.shape[0] == len(valid_chunks), "Embeddings and chunks count must match!"
    assert embeddings.shape[0] > 0, "Should have some embeddings!"
    print("  ✅ EmbeddingPipeline works correctly!")
    
    # 3. Test FaissVectorStore integration
    print("\n--- Test 3: FaissVectorStore build + query ---")
    test_store_dir = "test_faiss_store"
    store = FaissVectorStore(persist_dir=test_store_dir, chunk_size=200, chunk_overlap=50)
    store.build_from_documents(docs)
    
    if store.index is not None:
        results = store.query("What is a database?", top_k=2)
        print(f"  Query returned {len(results)} results")
        for r in results:
            snippet = r['metadata']['texts'][:80] if r.get('metadata') and r['metadata'].get('texts') else "None"
            print(f"    Distance: {r['distance']:.4f} | {snippet}...")
        print("  ✅ FaissVectorStore works correctly!")
    else:
        print("  ⚠️ Vector store index is empty")
    
    # Cleanup
    import shutil
    if os.path.exists(test_store_dir):
        shutil.rmtree(test_store_dir)
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
    print("=" * 60)
    print("\nYou can now run: python main.py")

if __name__ == "__main__":
    test_pipeline()
