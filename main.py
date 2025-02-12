import os
import logging
from typing import List, Dict, Any

import numpy as np
import faiss
import uvicorn
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# AI libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Additional libraries for PDF handling and Google Drive download
import gdown
from PyPDF2 import PdfReader

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI App and CORS Middleware Setup
# ----------------------------
app = FastAPI(title="Market Research RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Additional Endpoints
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}


@app.get("/favicon.ico")
def favicon():
    """Return the favicon.ico file."""
    favicon_path = "/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/x-icon", filename="favicon.ico")
    else:
        raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/tf_loss")
def compute_tf_loss():
    """Compute a dummy TensorFlow loss."""
    labels = [1]
    logits = [[0.1, 0.9]]
    loss_value = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    try:
        loss_float = float(loss_value.numpy())
    except Exception:
        loss_float = str(loss_value)
    return {"loss": loss_float}


# ----------------------------
# Data Models
# ----------------------------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    sentiment: Dict[str, Any] = None
    topics: List[str] = None


# ----------------------------
# Google Drive PDF Download and Report Loading
# ----------------------------
def download_drive_folder(folder_url: str, output_dir: str):
    """Downloads files from a Google Drive folder."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Downloading files from Google Drive folder: {folder_url} into {output_dir}")
    gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)


def ensure_reports_downloaded():
    """Ensure PDF reports are downloaded from Google Drive."""
    folder_url = "https://drive.google.com/drive/folders/1amtFlrA4qCZwdLe_6h2ZAELZtWwrMb1w"
    output_dir = "data/reports_pdf"
    if not os.path.exists(output_dir) or not any(fname.lower().endswith(".pdf") for fname in os.listdir(output_dir)):
        download_drive_folder(folder_url, output_dir)
    else:
        logger.info("PDF reports already downloaded.")


def load_reports_from_pdf(data_dir="data/reports_pdf"):
    """Extract text from PDFs in a directory."""
    reports = {}
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            report_id = os.path.splitext(filename)[0]
            file_path = os.path.join(data_dir, filename)
            logger.info(f"Extracting text from {file_path}")
            try:
                reader = PdfReader(file_path)
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                continue

            paragraphs = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    ps = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                    paragraphs.extend(ps)
            reports[report_id] = paragraphs
    return reports


ensure_reports_downloaded()
reports_data = load_reports_from_pdf()
documents = [
    {
        "id": f"{report_id}_p{idx+1}",
        "text": para,
        "source": f"{report_id} - Paragraph {idx+1}",
        "report": report_id,
    }
    for report_id, paragraphs in reports_data.items()
    for idx, para in enumerate(paragraphs)
]
logger.info(f"Loaded {len(documents)} document segments from PDF reports.")


# ----------------------------
# Embedding and FAISS Index Setup
# ----------------------------
embedding_model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)
doc_texts = [doc["text"] for doc in documents]
logger.info("Encoding document texts...")
doc_embeddings = embedder.encode(doc_texts, show_progress_bar=True).astype("float32")

embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)
logger.info("FAISS index built.")


# ----------------------------
# Language Model Setup
# ----------------------------
generation_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
lm_model.resize_token_embeddings(len(tokenizer))
generator = pipeline("text-generation", model=lm_model, tokenizer=tokenizer)

sentiment_analyzer = pipeline("sentiment-analysis")


def extract_topics(text: str, top_n: int = 3) -> List[str]:
    """
    Naively extracts topics by returning the top-n most frequent words (ignoring short/common words).
    """
    words = text.split()
    freq = {}
    for word in words:
        if len(word) > 4:
            word_lower = word.lower().strip(".,!?")
            freq[word_lower] = freq.get(word_lower, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    topics = [word for word, count in sorted_words[:top_n]]
    return topics


def construct_prompt(query: str) -> str:
    """Retrieve relevant document segments and construct a query prompt."""
    query_embedding = embedder.encode([query]).astype("float32")
    k = 3
    distances, indices = index.search(query_embedding, k)
    relevant_docs = [documents[i] for i in indices[0]]
    context_text = "\n\n".join([doc["text"] for doc in relevant_docs])

    return (
        f"Context from market research reports:\n{context_text}\n\n"
        f"Answer the following query: {query}\n"
        "Include references to the original sources."
    )


def process_query(query: str) -> str:
    """Generate a response from the language model with improved tokenization and error handling."""
    prompt = construct_prompt(query)
    try:
        # Tokenize the prompt with a maximum length of 512 tokens.
        inputs = tokenizer(
            prompt,
            max_length=512,     # set maximum sequence length
            truncation=True,    # enable truncation if the input exceeds max_length
            return_tensors="pt",
            padding="max_length"  # pads sequences shorter than max_length
        )
        generation_ids = generator.model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=400, detail="Query processing error.")

    generated_text = tokenizer.decode(generation_ids[0], skip_special_tokens=True)

    if not generated_text.strip():
        logger.warning("Generated text is empty. Returning fallback message.")
        generated_text = "I'm sorry, I couldn't generate an answer for your query."

    return generated_text


# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/query")
def query_rag(payload: QueryRequest):
    """Process a query using the RAG system."""
    query = payload.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        answer = process_query(query)
        sources = ["Source 1", "Source 2"]
        sentiment_result = sentiment_analyzer(answer)
        sentiment = sentiment_result[0]["label"] if sentiment_result else "neutral"
        topics = extract_topics(answer, top_n=3)

        response_data = {
            "answer": answer,
            "sources": sources,
            "sentiment": sentiment,
            "topics": topics
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")


@app.get("/download/{report_id}")
def download_report(report_id: str):
    """Download a PDF report."""
    pdf_dir = "data/reports_pdf"
    file_path = os.path.join(pdf_dir, f"{report_id}.pdf")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=f"{report_id}.pdf")
    else:
        raise HTTPException(status_code=404, detail="Report not found")


#if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000)

