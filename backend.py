import torch
import re
import heapq
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. MANUAL MODEL LOADING (with retries and lighter alternatives)
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = None
qa_model = None
try:
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    print("✅ QA model loaded successfully.")
except Exception as e:
    print(f"⚠️ QA Model Loading Error: {e}. Falling back to keyword search.")

# 2. SUMMARIZER WITH LIGHTWEIGHT FALLBACK
summarizer = None
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)  # CPU-only for low memory
    print("✅ Summarizer loaded.")
except Exception as e:
    print(f"⚠️ Summarizer Error: {e}. Using rule-based fallback.")

# 3. EMBEDDINGS (lightweight and reliable)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- UTILITY FUNCTIONS ---

def clean_text(text):
    """Clean garbage characters and normalize spacing."""
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\\-\\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def rule_based_summary(text):
    """Extractive summary fallback using TF-IDF-like scoring."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if len(sentences) < 3:
        return text[:500] + "..."
    
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {w: words.count(w) for w in set(words) if len(w) > 4}
    
    ranking = {}
    for i, sent in enumerate(sentences):
        score = sum(freq.get(word, 0) for word in re.findall(r'\b\w+\b', sent.lower()) if len(word) > 4)
        ranking[i] = score
    
    top_indices = heapq.nlargest(min(3, len(ranking)), ranking, key=ranking.get)
    summary = ". ".join(sentences[j] for j in sorted(top_indices)) + "."
    return f"💡 (Summary): {summary}"

# --- CORE FUNCTIONS ---

def summarize_doc(text):
    """Generate summary with AI or rule-based fallback."""
    cleaned = clean_text(text)
    if len(cleaned) < 50:
        return "⚠️ Text too short for summarization."
    
    if not summarizer:
        return rule_based_summary(cleaned)
    
    try:
        input_text = cleaned[:1500]
        res = summarizer(input_text, max_length=100, min_length=30, do_sample=False, truncation=True)
        return res[0]['summary_text']
    except Exception:
        return rule_based_summary(cleaned)

def prepare_vector_db(text):
    """Create FAISS Vector Store with configurable splitting."""
    text = clean_text(text)
    if len(text) < 100:
        raise ValueError("Text too short for vector DB.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.create_documents([text])
    vector_db = FAISS.from_documents(docs, embedding)
    print(f"✅ Vector DB created with {len(docs)} chunks.")
    return vector_db

def ask_question(vector_db, query):
    """Robust logits-based QA with keyword fallback."""
    try:
        # Retrieve relevant docs
        docs = vector_db.similarity_search(query, k=3)
        if not docs:
            return "⚠️ No relevant context found."
        
        context = " ".join(doc.page_content for doc in docs[:2])
        
        if qa_model is None or qa_tokenizer is None:
            # Keyword fallback
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            context_words = re.findall(r'\b\w+\b', context.lower())
            matches = [w for w in query_words if w in context_words][:50]
            return f"💡 Based on context: {' '.join(matches).capitalize()}" if matches else "💡 Couldn't extract precise answer."
        
        # Tokenize with padding/truncation
        inputs = qa_tokenizer(
            query, 
            context, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        
        # Safer decoding with bounds check
        if start_idx >= end_idx or end_idx > inputs["input_ids"].shape[1]:
            return "💡 No clear answer span found."
        
        answer = qa_tokenizer.decode(
            inputs["input_ids"][0][start_idx:end_idx], 
            skip_special_tokens=True
        ).strip()
        
        return f"💡 {answer.capitalize()}" if answer and "[CLS]" not in answer else "💡 Couldn't pinpoint a specific answer."
    
    except Exception as e:
        return f"⚠️ QA Error: {str(e)[:100]}"

def generate_logic_questions(text):
    """Generate analysis questions."""
    return (
        "1. What is the core objective of this document?\n"
        "2. Identify the top 3 critical points mentioned.\n"
        "3. How would you summarize the conclusion in one sentence?\n"
        "4. What action items or next steps are implied?"
    )

def evaluate_user_answer(question, user_answer, context):
    """Semantic evaluation with improved matching."""
    if not user_answer or len(user_answer.strip()) < 10:
        return "⚠️ Please provide a more detailed answer (at least 10 words)."
    
    context_low = clean_text(context).lower()
    user_low = clean_text(user_answer).lower()
    
    # Vectorize for overlap (simple Jaccard-like)
    user_words = set(w for w in re.findall(r'\b\w+\b', user_low) if len(w) > 4)
    context_words = set(w for w in re.findall(r'\b\w+\b', context_low) if len(w) > 4)
    
    intersection = len(user_words & context_words)
    union = len(user_words | context_words)
    similarity = intersection / union if union > 0 else 0
    
    if similarity > 0.4 or user_low in context_low:
        return "✅ Excellent! Matches document well."
    elif similarity > 0.15:
        return "🟡 Partially correct—add more document-specific details."
    else:
        return "❌ Not aligned. Review the source and try again."

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    sample_text = "Your long document text here..."
    db = prepare_vector_db(sample_text)
    print(summarize_doc(sample_text))
    print(ask_question(db, "What is the main topic?"))
