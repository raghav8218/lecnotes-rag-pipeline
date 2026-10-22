from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT=Path(__file__).resolve().parents[2]
pdf_path=PROJECT_ROOT/"mlcs229.pdf"

loader = PyPDFLoader(str(pdf_path))
pages=loader.load()
print(f"loaded{len(pages)} pages")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
)

print(f"Stored {len(chunks)} chunks in ChromaDB")
# print(f"total chunks: {len(chunks)}")
# print("\nfirst chunk:")
# print(chunks[0].page_content)
# print(f"first chunk metadata:\n{chunks[0].metadata}")