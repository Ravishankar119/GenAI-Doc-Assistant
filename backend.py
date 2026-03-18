# backend.py - NO PyPDF2, PyMuPDF ONLY
import torch
import re
import heapq
import fitz  # PyMuPDF - ONLY dependency needed
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global models (lazy load)
qa_tokenizer = qa_model = None
summarizer = None
embedding = None

def clean_text(text):
    """Clean text."""
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\\-\\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pdf(pdf_path):
    """PyMuPDF ONLY - NO PyPDF2 fallback."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return clean_text(text)
    except Exception as e:
        return f"PDF Error: {str(e)}"

def get_embedding():
    """Lazy load embedding."""
    global embedding
    if embedding is None:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding

def get_qa_models():
    """Lazy load QA models."""
    global qa_tokenizer, qa_model
    if qa_tokenizer is None:
        try:
            qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
            qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        except:
            qa_tokenizer = qa_model = None
    return qa_tokenizer, qa_model

def rule_based_summary(text):
    """Fallback summary."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if len(sentences) < 3:
        return text[:300] + "..."
    
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {w: words.count(w) for w in set(words) if len(w) > 4}
    
    scored = []
    for i, sent in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in re.findall(r'\b\w+\b', sent.lower()) if len(w) > 4)
        scored.append((score * (1.2 - i*0.05), sent))
    
    top = heapq.nlargest(3, scored, key=lambda x: x[0])
    return ". ".join([s[1] for s in top])

def summarize_doc(text):
    """Summary with AI fallback."""
    cleaned = clean_text(text)
    if len(cleaned) < 50:
        return "Text too short."
    
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
        except:
            pass
    
    if summarizer:
        try:
            return summarizer(cleaned[:1500], max_length=80, min_length=20, do_sample=False)[0]['summary_text']
        except:
            pass
    
    return rule_based_summary(cleaned)

def prepare_vector_db(text):
    """Create FAISS vector store."""
    embedding = get_embedding()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([clean_text(text)])
    return FAISS.from_documents(docs, embedding)

def ask_question(db, query):
    """QA over document."""
    docs = db.similarity_search(query, k=3)
    context = " ".join(d.page_content for d in docs)
    
    tokenizer, model = get_qa_models()
    if model is None:
        # Keyword fallback
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        matches = [w for w in query_words if w in context.lower()][:30]
        return " ".join(matches).capitalize() if matches else "No clear answer."
    
    try:
        inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
        return answer.strip().capitalize()
    except:
        return "Answer not clear from document."

def generate_logic_questions(text):
    """Generate analysis questions."""
    return """1. What is the main objective?
2. List 3 key points.
3. What is the conclusion?"""

def evaluate_user_answer(question, user_answer, context):
    """Evaluate answer quality."""
    if len(user_answer.strip()) < 10:
        return "⚠️ Answer too short."
    
    user_words = set(w for w in re.findall(r'\b\w+\b', user_answer.lower()) if len(w) > 3)
    context_words = set(w for w in re.findall(r'\b\w+\b', context.lower()) if len(w) > 3)
    
    if not user_words:
        return "❌ No meaningful words."
    
    sim = len(user_words & context_words) / len(user_words | context_words)
    if sim > 0.3:
        return "✅ Excellent!"
    elif sim > 0.1:
        return "🟡 Partial credit."
    else:
        return "❌ Review document."
