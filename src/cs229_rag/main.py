from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


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

print(f" {len(chunks)} chunks in chroma")
# print(f"total chunks: {len(chunks)}")
# print("\nfirst chunk:")
# print(chunks[0].page_content)
# print(f"first chunk metadata:\n{chunks[0].metadata}")

retriever =vector_store.as_retriever(search_kwargs={"k":5})
query = "linear regression model, hypothesis, cost function, and fitting parameters"


results = vector_store.similarity_search_with_score(query, k=5)

# for i, (doc, score) in enumerate(results, 1):
#     print(f"\n--- Chunk {i} | Distance: {score:.4f} ---")
#     print(f"Page: {doc.metadata.get('page')}")
#     print(doc.page_content[:800])
llm=ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite'
)
context='\n\n'.join(
    doc.page_content for doc,score in results
    
)
prompt = ChatPromptTemplate.from_template("""
You are a helpful tutor for Stanford's CS229 machine learning course.
Answer the question using the provided lecture notes. answer using sentences instead of math equations
cuz the latex would look messy in terminal

Lecture notes:
{context}

Question:
{question}
""")


messages= prompt.format_messages(context=context,question=query)
response=llm.invoke(messages)
print("\n rag final ans:\n")
print(response.content[0]["text"])#final thing by gemini
# boom bam pow
# for i, doc in enumerate(results,1):
#     print(f"--chunk {i}--")
#     print(doc.page_content[:300])
#     print()