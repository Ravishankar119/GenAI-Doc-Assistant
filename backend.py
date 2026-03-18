from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# ✅ load models once (fast + stable)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ✅ IMPROVED CLEAN TEXT FUNCTION (MAIN FIX 🔥)
def clean_text(text):
    # remove repeated words like stairs stairs stairs
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # remove specific garbage words
    text = re.sub(r'(stairs\s*)+', '', text, flags=re.IGNORECASE)

    # remove very long repeated characters
    text = re.sub(r'(.)\1{10,}', '', text)

    # remove "Comments" section if exists
    if "Comments" in text:
        text = text.split("Comments")[0]

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ✅ SUMMARY (MORE SAFE)
def summarize_doc(text):
    text = clean_text(text)
    text = text[:2000]

    # safety check
    if len(text.split()) < 30:
        return "⚠️ PDF content not readable or too short."

    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )

    return summary[0]['summary_text']


# ✅ VECTOR DB
def prepare_vector_db(text):
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, embedding)


# ✅ CHATBOT Q&A (MORE ROBUST)
def ask_question(retriever, query):
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "⚠️ No relevant information found."

    context = " ".join(doc.page_content for doc in docs[:3])

    try:
        result = qa_pipeline(question=query, context=context)
        answer = result.get("answer", "No answer found")
    except:
        answer = "⚠️ Error generating answer"

    return f"💡 {answer}"


# ✅ LOGIC QUESTIONS (SAME)
def generate_logic_questions(text):
    return (
        "1. What is the main idea of the document?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain the content in your own words."
    )


# ✅ EVALUATION (BETTER MATCHING)
def evaluate_user_answer(question, user_answer, context):
    context = clean_text(context)

    user_answer = user_answer.lower()
    context = context.lower()

    if user_answer in context:
        return "✅ Good answer! It matches the document."
    elif any(word in context for word in user_answer.split()):
        return "🟡 Partially correct answer."
    else:
        return "⚠️ Answer is not aligned with the document."
