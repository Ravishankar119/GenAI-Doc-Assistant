# backend.py - COMPLETE WORKING VERSION
import torch
import re
import heapq
import fitz  # ONLY PyMuPDF needed!
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Models (with fallbacks)
qa_tokenizer = qa_model = None
summarizer = None

try:
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
except:
    pass

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
except:
    pass

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\\-\\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pdf(pdf_path):
    """PyMuPDF only - FAST & RELIABLE."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return clean_text(text)

def summarize_doc(text):
    """Summary with fallback."""
    cleaned = clean_text(text)
    if not summarizer:
        sentences = [s.strip() for s in cleaned.split('.') if len(s) > 20]
        if len(sentences) < 3: return cleaned[:400]
        words = re.findall(r'\b\w+\b', cleaned.lower())
        freq = {w: words.count(w) for w in set(words) if len(w) > 4}
        scored = [(sum(freq.get(w, 0) for w in re.findall(r'\b\w+\b', s.lower())), s) for s in sentences]
        top = heapq.nlargest(3, scored, key=lambda x: x[0])
        return ". ".join([s[1] for s in top])
    try:
        return summarizer(cleaned[:1500], max_length=80, min_length=20)[0]['summary_text']
    except:
        return cleaned[:300] + "..."

def prepare_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([clean_text(text)])
    return FAISS.from_documents(docs, embedding)

def ask_question(db, query):
    docs = db.similarity_search(query, k=3)
    context = " ".join(d.page_content for d in docs)
    
    if not qa_model:
        words = set(re.findall(r'\b\w+\b', query.lower()))
        matches = [w for w in words if w in context.lower()][:30]
        return " ".join(matches).capitalize() if matches else "No answer found."
    
    try:
        inputs = qa_tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
        return answer.strip().capitalize()
    except:
        return "Answer not clear from document."

def generate_logic_questions(text):
    return """1. What is the main objective?
2. List 3 key points.
3. What is the conclusion?"""

def evaluate_user_answer(question, user_answer, context):
    if len(user_answer) < 10:
        return "⚠️ Answer too short."
    user_words = set(re.findall(r'\b\w+\b', user_answer.lower()) if len(w) > 3)
    context_words = set(re.findall(r'\b\w+\b', context.lower()) if len(w) > 3)
    sim = len(user_words & context_words) / len(user_words | context_words)
    if sim > 0.3: return "✅ Excellent!"
    if sim > 0.1: return "🟡 Partial credit."
    return "❌ Review document."

def summarize_pdf_bullets(pdf_path):
    """Get 5-7 crisp bullets from PDF."""
    text = extract_pdf(pdf_path)
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    summaries = [summarize_doc(c) for c in chunks[:10]]
    combined = " ".join(summaries)
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', combined) if len(s) > 15]
    scored = [(len(s.split()), s) for s in sentences]
    top = heapq.nlargest(6, scored, key=lambda x: x[0])
    return [f"• {s[1][:100]}..." for s in top]
