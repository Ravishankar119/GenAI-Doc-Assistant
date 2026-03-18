from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# ✅ Task-specific classes use karke pipeline load karein (Zero Error Method)
def load_qa_pipeline():
    model_name = "distilbert-base-uncased-distilled-squad"
    # Manual loading to avoid KeyError in Transformers registry
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

def load_summarizer():
    # Summarization ke liye bhi explicit loading better hai
    model_name = "sshleifer/distilbart-cnn-6-6"
    return pipeline("summarization", model=model_name, tokenizer=model_name)

# Globally initialize
qa_pipeline = load_qa_pipeline()
summarizer = load_summarizer()
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
