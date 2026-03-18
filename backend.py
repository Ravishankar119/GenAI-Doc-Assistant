from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re


# ✅ SUMMARY (CLEAN + FIXED)
def summarize_doc(text):
    # 🔥 remove junk repeated words like "stairs stairs..."
    text = re.sub(r'(stairs\s*)+', '', text, flags=re.IGNORECASE)

    # 🔥 remove unwanted section like comments
    if "Comments" in text:
        text = text.split("Comments")[0]

    # 🔥 clean extra spaces
    text = re.sub(r'\s+', ' ', text)

    # 🔥 limit size
    text = text[:2000]

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)

    return summary[0]['summary_text']


# ✅ VECTOR DB
def prepare_vector_db(text):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, embedding)


# ✅ Q&A
def ask_question(retriever, query):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    docs = retriever.get_relevant_documents(query)
    context = " ".join(doc.page_content for doc in docs[:2])

    result = qa_pipeline(question=query, context=context)

    return result["answer"]


# ✅ LOGIC QUESTIONS
def generate_logic_questions(text):
    return (
        "1. What is the main idea of the document?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain the content in your own words."
    )


# ✅ ANSWER EVALUATION
def evaluate_user_answer(question, user_answer, context):
    if user_answer.lower() in context.lower():
        return "✅ Good answer! It matches the context."
    else:
        return "⚠️ Answer is not fully aligned with the document."
