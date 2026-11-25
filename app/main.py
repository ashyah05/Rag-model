import os
from pathlib import Path

import faiss
import numpy as np

from fastapi import FastAPI, HTTPException, Header, Depends
import secrets
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# Load environment variables from .env
load_dotenv()
RAG_ACCESS_CODE = os.getenv("RAG_ACCESS_CODE")


def verify_access_code(x_access_code: str = Header(...)):
    if not RAG_ACCESS_CODE:
        raise HTTPException(
            status_code=503,
            detail="Access code is not configured"
        )

    if not secrets.compare_digest(x_access_code, RAG_ACCESS_CODE):
        raise HTTPException(
            status_code=401,
            detail="Invalid access code"
        )

    return True

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

DOCUMENT_FOLDER = Path("/app/documents")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3


# Initialize FastAPI
app = FastAPI(
    title="RAG Model API",
    description="RAG Model - Automated CI/CD Testing",
    version="1.0.0"
)


# Global variables
embedding_model = None
gemini_client = None
faiss_index = None
document_chunks = []


# Request model
class QueryRequest(BaseModel):
    query: str


# Load PDF and TXT documents
def load_documents():

    documents = []

    if not DOCUMENT_FOLDER.exists():
        raise RuntimeError("Documents folder does not exist")

    for file_path in DOCUMENT_FOLDER.iterdir():

        if file_path.suffix.lower() == ".pdf":

            reader = PdfReader(str(file_path))

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text and text.strip():

                    documents.append({
                        "text": text,
                        "source": file_path.name,
                        "page": page_number
                    })

        elif file_path.suffix.lower() == ".txt":

            text = file_path.read_text(encoding="utf-8")

            if text.strip():

                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": None
                })

    return documents


# Split documents into chunks
def split_documents(documents):

    chunks = []

    for document in documents:

        text = document["text"]
        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"]
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# Initialize Gemini and embedding model
def initialize_models():

    global embedding_model
    global gemini_client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")

    gemini_client = genai.Client(api_key=api_key)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# Create FAISS vector database
def build_vector_database():

    global faiss_index
    global document_chunks

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No PDF or TXT documents found in the documents folder"
        )

    document_chunks = split_documents(documents)

    if not document_chunks:
        raise RuntimeError("No document chunks generated")

    texts = [
        chunk["text"]
        for chunk in document_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)

    print("FAISS vector database created successfully")
    print("Documents:", len(documents))
    print("Chunks:", len(document_chunks))


# Retrieve relevant document chunks
def retrieve_documents(query):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = faiss_index.search(
        query_embedding,
        min(TOP_K, len(document_chunks))
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index == -1:
            continue

        chunk = document_chunks[index].copy()

        chunk["similarity_score"] = float(score)

        results.append(chunk)

    return results


# Generate response using Gemini
def generate_answer(query, retrieved_chunks):

    context = ""

    for number, chunk in enumerate(retrieved_chunks, start=1):

        context += (
            f"\nDocument {number}: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using only the
information available in the retrieved documents.

If the answer is not available in the documents,
clearly state that the information was not found.

Do not invent information or use outside knowledge.

Retrieved Context:
{context}

User Question:
{query}

Provide a clear and understandable answer.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text


# Home GET API
@app.get("/")
def home():

    return {
        "status": "success",
        "message": "RAG Model API is running"
    }


# RAG POST API
@app.post("/ask")
def ask_question(
    request: QueryRequest,
    authorized: bool = Depends(verify_access_code)
):

    global faiss_index

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        # Initialize models on first request
        if embedding_model is None or gemini_client is None:
            initialize_models()

        # Build vector database on first request
        if faiss_index is None:
            build_vector_database()

        # Retrieve relevant information
        retrieved_chunks = retrieve_documents(query)

        # Generate answer
        answer = generate_answer(
            query,
            retrieved_chunks
        )

        return {
            "status": "success",
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "document": chunk["source"],
                    "page": chunk["page"]
                }
                for chunk in retrieved_chunks
            ]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
