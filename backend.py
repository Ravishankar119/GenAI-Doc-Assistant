from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def summarize_doc(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    text = text[:2000]

    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)

    return summary[0]['summary_text']


def prepare_vector_db(text):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text])

    return FAISS.from_documents(docs, embedding)


def ask_question(retriever, query):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    docs = retriever.get_relevant_documents(query)
    context = " ".join(doc.page_content for doc in docs[:2])

    result = qa_pipeline(question=query, context=context)

    return result["answer"]


def generate_logic_questions(text):
    return (
        "1. What is the main idea of the document?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain the content in your own words."
    )


def evaluate_user_answer(question, user_answer, context):
    if user_answer.lower() in context.lower():
        return "✅ Good answer! It matches the context."
    else:
        return "⚠️ Answer is not fully aligned with the document."
