from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re


# ✅ load models once (fast + stable)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ✅ CLEAN TEXT FUNCTION
def clean_text(text):
    text = re.sub(r'(stairs\s*)+', '', text, flags=re.IGNORECASE)
    if "Comments" in text:
        text = text.split("Comments")[0]
    text = re.sub(r'\s+', ' ', text)
    return text


# ✅ SUMMARY
def summarize_doc(text):
    text = clean_text(text)
    text = text[:2000]

    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)
    return summary[0]['summary_text']


# ✅ VECTOR DB
def prepare_vector_db(text):
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, embedding)


# ✅ CHATBOT Q&A (SMART ANSWER)
def ask_question(retriever, query):
    docs = retriever.get_relevant_documents(query)
    context = " ".join(doc.page_content for doc in docs[:3])

    result = qa_pipeline(question=query, context=context)

    answer = result["answer"]

    return f"💡 {answer}"


# ✅ LOGIC QUESTIONS
def generate_logic_questions(text):
    return (
        "1. What is the main idea of the document?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain the content in your own words."
    )


# ✅ EVALUATION
def evaluate_user_answer(question, user_answer, context):
    context = clean_text(context)

    if user_answer.lower() in context.lower():
        return "✅ Good answer! It matches the context."
    else:
        return "⚠️ Answer is not fully aligned with the document."
