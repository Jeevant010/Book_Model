import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.vectorstore import FaissVectorStore

load_dotenv()

@dataclass
class RetrievalResult:
    index: int
    distance: float
    text: Optional[str]

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        # Vector store setup
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            # Build from local 'Research/data' directory if index doesn't exist  
            from src.data_loader import load_all_documents
            # Try multiple possible data directories
            data_dirs = ["Research/data", "data", "Data"]
            docs = []
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    print(f"[INFO] Checking for documents in: {data_dir}")
                    docs = load_all_documents(data_dir)
                    if docs:
                        print(f"[INFO] Found {len(docs)} documents in {data_dir}")
                        break
                        
            if not docs:
                print("[WARNING] No documents found in any data directory. Vector store will be empty.")
                print("[INFO] Please add documents to 'Research/data/', 'data/', or 'Data/' directory.")
                # Create empty index for now
                self.vectorstore.index = None
                self.vectorstore.metadata = []
            else:
                self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # LLM setup
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing in environment")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm = ChatGroq(api_key=groq_api_key, model=llm_model, temperature=0.1)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        # Check if vector store is empty
        if self.vectorstore.index is None or len(self.vectorstore.metadata) == 0:
            print("[WARNING] Vector store is empty. No documents to search.")
            return []
            
        results = self.vectorstore.query(query_text=query, top_k=top_k)
        out: List[RetrievalResult] = []
        for r in results:
            text = r["metadata"]["texts"] if r.get("metadata") and r["metadata"].get("texts") else None
            out.append(RetrievalResult(index=int(r["index"]), distance=float(r["distance"]), text=text))
        return out

    def summarize(self, query: str, retrieved: List[RetrievalResult]) -> str:
        texts = [r.text for r in retrieved if r.text]
        context = "\n\n".join(texts)
        
        # Using proper message formatting for better LLM interaction
        system_message = SystemMessage(content="You are a helpful assistant. Use the provided context to answer the user's question if the information is present. If the answer is not in the context, or if the context is empty, answer the question using your own knowledge.")
        human_message = HumanMessage(content=f"""
Context:
{context}

Query: {query}

Answer:
""")
        
        try:
            response = self.llm.invoke([system_message, human_message])
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your GROQ_API_KEY is set correctly."

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve(query, top_k=top_k)
        return self.summarize(query, retrieved)

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is Database Management System?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)