from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

PROJECT_ROOT=Path(__file__).resolve().parents[2]
pdf_path=PROJECT_ROOT/"mlcs229.pdf"

loader = PyPDFLoader(str(pdf_path))
pages=loader.load()
print(f"loaded{len(pages)} pages")
