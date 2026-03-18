from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import re

# ✅ Manual Initialization (Avoiding pipeline task registry)
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Summarizer ko pipeline se hi rakhte hain, agar ye bhi error de toh batana
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,!?%()\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_doc(text):
    text = clean_text(text)[:2000]
    if len(text.split()) < 30: return "⚠️ Text too short to summarize."
    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def prepare_vector_db(text):
    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

# ✅ New Manual Q&A Logic (Bypassing the KeyError)
def ask_question(retriever, query):
    docs = retriever.invoke(query)
    if not docs: return "⚠️ No relevant information found."
    
    context = " ".join(doc.page_content for doc in docs[:3])
    
    # Manually tokenize and get model output
    inputs = qa_tokenizer(query, context, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    # Get the best answer span
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    if not answer or "[CLS]" in answer:
        return "💡 I couldn't find a specific answer in the document."
    return f"💡 {answer}"

def generate_logic_questions(text):
    return "1. What is the core message?\n2. What are the key takeaways?"

def evaluate_user_answer(question, user_answer, context):
    if user_answer.lower() in context.lower(): return "✅ Correct!"
    return "🟡 Please review the document again."
