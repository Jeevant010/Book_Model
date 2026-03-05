"""
Simple CLI test runner for RAG retrieval + summarization.
Run: python main.py
"""

from src.search import RAGSearch
import os

if __name__ == "__main__":
    # Check for GROQ API key
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment variables.")
        print("Please set it in a .env file or environment variable to use the LLM features.")
        print("You can still test document loading and embeddings without it.")
    
    try:
        rag_search = RAGSearch(
            persist_dir="faiss_store",
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama-3.1-8b-instant"
        )
        
        query = "What is Database Management System?"
        answer = rag_search.search_and_summarize(query=query, top_k=3)
        print("Query:", query)
        print("Answer:\n", answer)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have documents in 'Research/data/', 'data/', or 'Data/' directory")
        print("2. Set GROQ_API_KEY in your .env file")
        print("3. Run: pip install -r requirements.txt")