import torch
import re
import heapq
import fitz  # pip install PyMuPDF
from PyPDF2 import PdfReader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========== MODELS (Lightweight + Fallbacks) ==========
print("🔄 Loading models...")
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = qa_model = None

try:
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    print("✅ QA model loaded")
except:
    print("⚠️ QA fallback enabled")

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
    print("✅ Summarizer loaded")
except:
    summarizer = None
    print("⚠️ Rule-based summarizer")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embeddings ready")

# ========== UTILITIES ==========
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\\-\\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def rule_based_summary(text):
    sentences = [s.strip() for s in text.split('.') if len(s) > 20]
    if len(sentences) < 3: return text[:400] + "..."
    
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {w: words.count(w) for w in set(words) if len(w) > 4}
    
    scored = []
    for i, sent in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in re.findall(r'\b\w+\b', sent.lower()) if len(w) > 4)
        scored.append((score * (1.2 - i*0.05), sent))
    
    top = heapq.nlargest(4, scored, key=lambda x: x[0])
    return ". ".join([s[1] for s in top])

def summarize_chunk(chunk):
    if not summarizer:
        return rule_based_summary(chunk)
    try:
        return summarizer(chunk[:1500], max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    except:
        return rule_based_summary(chunk)

# ========== PDF HANDLING ==========
def extract_pdf(pdf_path):
    """Fast PDF extraction."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return clean_text(text)
    except:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages)
        return clean_text(text)

def chunk_text(text, chunk_words=300):
    """Smart chunking."""
    words = text.split()
    return [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words//2)]

# ========== CORE FUNCTIONS ==========
def summarize_pdf(pdf_path):
    """🔥 MAIN FUNCTION: PDF → 7 crisp bullets."""
    print(f"📄 Processing {pdf_path}...")
    
    text = extract_pdf(pdf_path)
    if len(text) < 100:
        return ["⚠️ Empty or tiny PDF."]
    
    chunks = chunk_text(text)
    print(f"✂️ {len(chunks)} chunks")
    
    chunk_summaries = [summarize_chunk(c) for c in chunks[:20]]  # Limit for speed
    combined = " ".join(chunk_summaries)
    
    # Extract TOP bullets
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if len(s) > 15]
    
    scored = [(len(s.split()) * (1.5 - i*0.08), s) for i, s in enumerate(sentences)]
    top_bullets = heapq.nlargest(7, scored, key=lambda x: x[0])
    
    return [f"• {b[1].capitalize()[:110]}..." for b in top_bullets]

def create_qa_system(pdf_path):
    """Build Q&A vector DB."""
    text = extract_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

def ask_question(db, question):
    """Answer questions about PDF."""
    docs = db.similarity_search(question, k=3)
    context = " ".join(d.page_content for d in docs)
    
    if not qa_model:
        words = set(re.findall(r'\b\w+\b', question.lower()))
        matches = [w for w in words if w in context.lower()][:40]
        return f"💡 {''.join(matches).capitalize()}" if matches else "❓ Check PDF context."
    
    try:
        inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
        return f"💡 {answer.strip().capitalize()}"
    except:
        return "❓ Answer not found in PDF."

# ========== 🚀 USAGE ==========
if __name__ == "__main__":
    PDF_PATH = "your_file.pdf"  # ← CHANGE THIS
    
    # 1. Get bullet summary
    print("\n📋 SUMMARY:")
    summary = summarize_pdf(PDF_PATH)
    for line in summary:
        print(line)
    
    # 2. Setup Q&A
    print("\n🔍 Q&A READY! (db created)")
    db = create_qa_system(PDF_PATH)
    
    # 3. Ask questions
    while True:
        q = input("\n❓ Ask about PDF (or 'quit'): ")
        if q.lower() == 'quit': break
        print(ask_question(db, q))
