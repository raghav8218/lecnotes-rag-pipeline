# CS229 Notes RAG

A simple Retrieval Augmented Generation (RAG) project built around Andrew Ng's CS229 machine learning notes. The idea is to ask questions about the notes and get answers based on the relevant sections of the PDF instead of passing the entire document to an LLM every time.

The project uses LangChain to load and split the PDF into overlapping chunks, Hugging Face to generate embeddings locally, and ChromaDB to store and retrieve relevant chunks. Gemini is used to generate answers from the retrieved context.

## How it works

1. Load the CS229 PDF and extract its text.
2. Split the text into smaller chunks with overlap so useful context is not lost at chunk boundaries.
3. Generate embeddings for the chunks using a Hugging Face model.
4. Store the embeddings and corresponding text in ChromaDB.
5. Retrieve the most relevant chunks for a question.
6. Pass the retrieved context and question to Gemini to generate an answer.

## Tech stack

- Python
- uv for dependency and environment management
- LangChain for document loading, splitting and pipeline construction
- Hugging Face Sentence Transformers for embeddings
- ChromaDB for vector storage and similarity search
- Gemini API for answer generation

## Setup

Clone the repository and move into the project directory.

Install the dependencies using uv.

```bash
uv sync
```

Place the CS229 PDF in the project directory and create a `.env` file containing your Gemini API key.

```env
GOOGLE_API_KEY=your_api_key_here
```

Run the project with:

```bash
uv run main.py
```

## Notes

The Hugging Face embedding model runs locally, while Gemini is used for answer generation. The vector database can be persisted locally so document embeddings do not need to be regenerated every time the application starts.
