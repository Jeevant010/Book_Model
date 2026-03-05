#!/usr/bin/env python3
"""
Test script to verify the updated Langchain RAG system works correctly.
"""
import sys
import os
import traceback

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        # Core imports
        from src.search import RAGSearch, RetrievalResult
        from src.embedding import EmbeddingPipeline
        from src.vectorstore import FaissVectorStore
        from src.data_loader import load_all_documents
        
        # External dependencies
        import langchain
        import langchain_core
        import langchain_community
        import langchain_text_splitters
        import sentence_transformers
        import faiss
        import pydantic
        import fastapi
        import uvicorn
        
        print("✅ All imports successful!")
        
        # Print versions
        print(f"\nPackage versions:")
        print(f"- LangChain: {langchain.__version__}")
        print(f"- LangChain Core: {langchain_core.__version__}")
        print(f"- Sentence Transformers: {sentence_transformers.__version__}")
        print(f"- Pydantic: {pydantic.__version__}")
        print(f"- FastAPI: {fastapi.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    print("\nTesting basic functionality...")
    try:
        from src.embedding import EmbeddingPipeline
        from src.vectorstore import FaissVectorStore
        
        # Test embedding pipeline initialization
        embedding_pipeline = EmbeddingPipeline()
        print("✅ Embedding pipeline initialized")
        
        # Test vector store initialization
        vector_store = FaissVectorStore(persist_dir="test_store", embedding_model="all-MiniLM-L6-v2")
        print("✅ Vector store initialized")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    try:
        import yaml
        
        # Test params.yaml
        if os.path.exists("params.yaml"):
            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
            print("✅ params.yaml loaded successfully")
            
        # Test config.yaml
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml", "r") as f:
                config = yaml.safe_load(f)
            print("✅ config/config.yaml loaded successfully")
            
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Updated Langchain RAG System Test ===\n")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality  
    if not test_basic_functionality():
        all_passed = False
    
    # Test config loading
    if not test_config_loading():
        all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! Your project is successfully updated!")
        print("\nNext steps:")
        print("1. Set your GROQ_API_KEY in a .env file")
        print("2. Place your documents in the 'data' directory")
        print("3. Run: python main.py")
        print("4. Or start the API: python app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())