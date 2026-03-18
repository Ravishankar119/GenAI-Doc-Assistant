from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# move everything inside functions (IMPORTANT)

def summarize_doc(text):
    summarizer = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    result = summarizer("Summarize:\n" + text[:500], max_length=100)
    return result[0]['generated_text']


def prepare_vector_db(text):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # lighter
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
        "1. What is the main idea?\n"
        "2. What conclusions can be drawn?\n"
        "3. Explain briefly."
    )


def evaluate_user_answer(question, user_answer, context):
    if user_answer.lower() in context.lower():
        return "✅ Good answer!"
    else:
        return "⚠️ Not fully correct."
