from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def process_text(text):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chunks = text.split(". ")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def summarizer(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file in 3-5 sentences."

        results = knowledgeBase.similarity_search(query, k=5)

        summary = "\n".join([res.page_content for res in results])
        return summary

    return "❌ No PDF file provided."