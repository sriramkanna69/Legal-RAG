# Legal-RAG

This is a simple Retrieval-Augmented Generation (RAG) application made for lawyers.  
It can retrieve relevant legal documents and answer questions using cloud AI models.

## Features
- Upload text files of case laws or contracts
- Ask legal questions
- Uses cloud models from Hugging Face
- Uses Pinecone vector database for storing embeddings

## Tech Stack
- Python
- FastAPI
- Hugging Face API
- Pinecone Cloud DB

## How to Run
1. pip install -r requirements.txt
2. Create .env and fill in your API keys
3. Run: uvicorn main:app --reload
4. Use curl or Postman to test endpoints


