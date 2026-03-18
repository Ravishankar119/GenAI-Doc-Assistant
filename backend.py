from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# ✅ Robust Pipeline Initialization
def load_qa_pipeline():
    model_name = "distilbert-base-uncased-distilled-squad"
    return pipeline("question-answering", model=model_name, tokenizer=model_name)

def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

qa_pipeline = load_qa_pipeline()
summarizer = load_summarizer()
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ CLEANING FUNCTION
def clean_text(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'(\b\w+\b(?:\s+\b\w+\b){0,10})\s+\1+', r'\1', text)
    text = re.sub(r'(stairs\s*)+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ✅ SUMMARY
def summarize_doc(text):
    text = clean_text(text)
    words = text.split()
    if len(words) < 30: return "⚠️ PDF not readable"
    
    # Text trimming for model limits
    text_input = text[:1500] 
    summary = summarizer(text_input, max_length=120, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# ✅ VECTOR DB
def prepare_vector_db(text):
    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

# ✅ Q&A
def ask_question(retriever, query):
    # LangChain v0.1+ search method
    docs = retriever.invoke(query) 
    if not docs: return "⚠️ No relevant information found."
    
    context = " ".join(doc.page_content for doc in docs[:3])
    try:
        result = qa_pipeline(question=query, context=context)
        return f"💡 {result.get('answer', 'No answer found')}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ✅ LOGIC QUESTIONS
def generate_logic_questions(text):
    return (
        "1. What is the main idea of the document?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain the content in your own words."
    )

# ✅ EVALUATION
def evaluate_user_answer(question, user_answer, context):
    context = clean_text(context).lower()
    user_answer = user_answer.lower().strip()
    
    if not user_answer: return "⚠️ Please provide an answer."
    if user_answer in context: return "✅ Good answer! It matches the document."
    
    # Simple keyword matching logic
    keywords = [w for w in user_answer.split() if len(w) > 3]
    match_count = sum(1 for w in keywords if w in context)
    
    if match_count > 2: return "🟡 Partially correct answer."
    return "⚠️ Answer is not aligned with the document."
