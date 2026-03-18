import torch
import re
import heapq
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ 1. MANUAL MODEL LOADING (Bypassing KeyError)
qa_model_name = "distilbert-base-uncased-distilled-squad"
try:
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
except Exception as e:
    print(f"QA Model Loading Error: {e}")

# ✅ 2. SUMMARIZER WITH FAIL-SAFE
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
except Exception:
    summarizer = None # Fallback to Rule-based if memory is low

# ✅ 3. EMBEDDINGS
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- UTILITY FUNCTIONS ---

def clean_text(text):
    """Clean garbage characters and normalize spacing"""
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rule_based_summary(text):
    """Extractive summary fallback if AI model fails"""
    sentences = [s.strip() for s in text.split('.') if len(s) > 20]
    if len(sentences) < 3: return text[:500] + "..."
    
    words = text.lower().split()
    freq = {}
    for w in words:
        if len(w) > 4: freq[w] = freq.get(w, 0) + 1
            
    ranking = {}
    for i, sent in enumerate(sentences):
        for word in sent.lower().split():
            if word in freq:
                ranking[i] = ranking.get(i, 0) + freq[word]
    
    top_indices = heapq.nlargest(3, ranking, key=ranking.get)
    return "💡 (Summary): " + ". ".join([sentences[j] for j in sorted(top_indices)]) + "."

# --- CORE FUNCTIONS ---

def summarize_doc(text):
    """Generate summary using AI or Rule-based fallback"""
    cleaned = clean_text(text)
    if not summarizer:
        return rule_based_summary(cleaned)
    
    try:
        input_text = cleaned[:1500] # Model limit
        res = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
        return res[0]['summary_text']
    except:
        return rule_based_summary(cleaned)

def prepare_vector_db(text):
    """Create FAISS Vector Store"""
    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

def ask_question(retriever, query):
    """Manual Logits-based QA to bypass Pipeline KeyError"""
    try:
        docs = retriever.invoke(query)
        if not docs: return "⚠️ No relevant context found."
        
        context = " ".join(doc.page_content for doc in docs[:2])
        inputs = qa_tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = qa_model(**inputs)
            
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        
        answer = qa_tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)
        
        if not answer.strip() or "[CLS]" in answer:
            return "💡 I couldn't find a specific answer in the document."
        return f"💡 {answer.strip().capitalize()}"
    except Exception as e:
        return f"⚠️ QA Error: {str(e)}"

def generate_logic_questions(text):
    """Generate generic analysis questions"""
    return (
        "1. What is the core objective of this document?\n"
        "2. Identify the top 3 critical points mentioned.\n"
        "3. How would you summarize the conclusion in one sentence?"
    )

def evaluate_user_answer(question, user_answer, context):
    """Simple semantic check for user answers"""
    if not user_answer or len(user_answer) < 5:
        return "⚠️ Please provide a detailed answer."
    
    context_low = context.lower()
    user_low = user_answer.lower()
    
    # Keyword overlap check
    keywords = [w for w in user_low.split() if len(w) > 4]
    matches = sum(1 for w in keywords if w in context_low)
    
    if user_low in context_low or matches >= 3:
        return "✅ Excellent! Your answer is well-aligned with the document."
    elif matches >= 1:
        return "🟡 Partially correct. You captured some points, but check the doc for more details."
    else:
        return "❌ Not quite. Your answer doesn't seem to match the document's content."
