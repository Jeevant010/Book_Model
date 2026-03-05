import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:(message)s:')

project_name = "testSummarizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/data_loader.py",
    f"src/embedding.py",
    f"src/__init__.py",
    f"src/search.py",
    f"src/vectorstore.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "setup.py",
    "research/trails.ipynb"
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directry:{filepath} for the file {filepath}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
            
    else :
        logging.info(f"{filename} is already exists.")