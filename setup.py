import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
__version__ = "2.0.0"

REPO_NAME = "Langchain_RAG_System"
AUTHOR_USER_NAME = "jeevant"
SRC_REPO = "langchain_rag"
AUTHOR_EMAIL = "jeevantmudgil10@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modern RAG system using LangChain, FAISS, and Groq LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Documentation": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}#readme",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "langchain-groq>=0.2.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu>=1.8.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)