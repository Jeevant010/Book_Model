import os
from dataclasses import dataclass
from typing import List, Optional, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# LangChain's document compressor module changed across versions.
# Avoid hard failure if unavailable (often absent in langchain 1.x minimal installs).
LLMChainExtractor = None
try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ModuleNotFoundError:
    try:
        from langchain_experimental.compression import LLMChainExtractor
    except ModuleNotFoundError:
        LLMChainExtractor = None

from src.vectorstore import FaissVectorStore, PineconeVectorStore

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
        vector_store_type = os.getenv("VECTOR_STORE", "faiss").lower()
        self.vector_store_type = "faiss"

        if vector_store_type == "pinecone" and os.getenv("PINECONE_API_KEY"):
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "book-model-index")
            print(f"[INFO] Using Pinecone Vector Store (Index: {pinecone_index})")
            self.vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding_model=embedding_model)
            self.vector_store_type = "pinecone"
        else:
            if vector_store_type == "pinecone" and not os.getenv("PINECONE_API_KEY"):
                print("[WARN] VECTOR_STORE is pinecone but PINECONE_API_KEY is missing. Falling back to local FAISS.")
            print("[INFO] Using Local FAISS Vector Store")
            self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
            faiss_path = os.path.join(persist_dir, "faiss.index")
            meta_path = os.path.join(persist_dir, "metadata.pkl")

            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                # Build from local data directory if index doesn't exist  
                from src.data_loader import load_all_documents
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

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents. If Pinecone is used, it applies metadata_filter natively.
        If FAISS is used, it falls back to post-filtering.
        """
        if getattr(self.vectorstore, "index", False) is None:
            print("[WARNING] Vector store is empty. No documents to search.")
            return []
        
        is_pinecone = self.vector_store_type == "pinecone"
        
        if is_pinecone:
            # Native Pre-Filtering for Pinecone
            results = self.vectorstore.query(
                query_text=query, 
                top_k=top_k, 
                metadata_filter=metadata_filter
            )
        else:
            # Post-Filtering for FAISS
            fetch_k = top_k * 3 if metadata_filter else top_k
            results = self.vectorstore.query(query_text=query, top_k=fetch_k)
            
        out: List[RetrievalResult] = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            text = meta.get("texts")
            
            # Apply post filter only if it's not pinecone
            if not is_pinecone and metadata_filter:
                matches = True
                for key, value in metadata_filter.items():
                    meta_value = meta.get(key, "")
                    if meta_value and value.lower() not in str(meta_value).lower():
                        matches = False
                        break
                if not matches:
                    continue
            
            out.append(RetrievalResult(
                index=int(r.get("index", 0) if "index" in r else 0), 
                distance=float(r.get("distance", 0.0)), 
                text=text
            ))
            
            if len(out) >= top_k:
                break
        
        return out

    def _compress_context(self, query: str, texts: List[str]) -> str:
        """
        Contextual Compression: Uses LangChain LLMChainExtractor to dynamically
        strip irrelevant sentences from the fetched chunks.
        """
        if not texts:
            return ""
        
        raw_context = "\n\n---\n\n".join(texts)
        if len(raw_context) < 500:
            return raw_context

        if LLMChainExtractor is None:
            print("[INFO] LLMChainExtractor not available, skipping context compression")
            return raw_context

        try:
            compressor = LLMChainExtractor.from_llm(self.llm)
            docs = [Document(page_content=t) for t in texts]

            # This uses Groq to extract relevant phrases natively
            compressed_docs = compressor.compress_documents(docs, query)

            if compressed_docs:
                compressed_text = "\n\n---\n\n".join([d.page_content for d in compressed_docs])
                saved_pct = round((1 - len(compressed_text) / len(raw_context)) * 100)
                if saved_pct > 0:
                    print(f"[INFO] Context compressed using LangChain: saved {saved_pct}% tokens")
                return compressed_text
        except Exception as e:
            print(f"[WARN] LangChain context compression failed, using raw context: {e}")

        return raw_context

    def summarize(self, query: str, retrieved: List[RetrievalResult]) -> str:
        texts = [r.text for r in retrieved if r.text]
        
        # Apply contextual compression to reduce irrelevant content
        context = self._compress_context(query, texts)
        
        system_message = SystemMessage(content=(
            "You are a helpful assistant. Use the provided context to answer the user's question "
            "if the information is present. If the answer is not in the context, or if the context "
            "is empty, answer the question using your own knowledge."
        ))
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