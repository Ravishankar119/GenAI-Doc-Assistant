from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import re

# ✅ Manual Model Loading (KeyError Bypass)
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Summarizer fallback
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
except:
    summarizer = None

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_doc(text):
    if not summarizer:
        return "⚠️ Summarizer model loading failed."
    text = clean_text(text)[:1500]
    if len(text.split()) < 20: return "⚠️ Text too short."
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def prepare_vector_db(text):
    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

# ✅ Manual Q&A Logic (Isse KeyError kabhi nahi aayega)
def ask_question(retriever, query):
    docs = retriever.invoke(query)
    if not docs: return "⚠️ No relevant information found."
    
    context = " ".join(doc.page_content for doc in docs[:2])
    
    # Manually Tokenize
    inputs = qa_tokenizer(query, context, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    # Get Answer logic
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    if not answer.strip() or len(answer) < 1:
        return "💡 I couldn't find a clear answer in the text."
    return f"💡 {answer.strip()}"

def generate_logic_questions(text):
    return "1. What is the primary focus of this document?\n2. What are the key points discussed?"

def evaluate_user_answer(question, user_answer, context):
    if not user_answer: return "⚠️ Please enter an answer."
    if user_answer.lower() in context.lower(): return "✅ Correct! Well done."
    return "🟡 Your answer might be missing some details from the text."
