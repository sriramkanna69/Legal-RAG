
import os
import requests
import pinecone
from dotenv import load_dotenv

load_dotenv()


HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-small")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-rag-index")


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def get_embedding(text):
    """Get embedding from Hugging Face cloud model."""
    url = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": text}
    res = requests.post(url, headers=headers, json=data)
    return res.json()[0]

def get_answer(prompt):
    """Generate answer using Hugging Face text generation model."""
    url = f"https://api-inference.huggingface.co/models/{HF_GEN_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": prompt}
    res = requests.post(url, headers=headers, json=data)
    try:
        return res.json()[0]["generated_text"]
    except:
        return "Sorry, I couldn’t generate an answer."

def ingest_docs(doc_id, text):
    """Split text, get embeddings, and upload to Pinecone."""
    index = pinecone.Index(PINECONE_INDEX_NAME)
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vectors.append((f"{doc_id}_{i}", emb, {"text": chunk}))
    index.upsert(vectors=vectors)
    return f"Document '{doc_id}' uploaded with {len(vectors)} chunks."

def ask_question(question):
    """Find similar passages and answer."""
    index = pinecone.Index(PINECONE_INDEX_NAME)
    q_emb = get_embedding(question)
    results = index.query(vector=q_emb, top_k=2, include_metadata=True)
    context = ""
    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n"

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return get_answer(prompt)
