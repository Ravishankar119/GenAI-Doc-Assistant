
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def summarize_doc(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = summarizer(chunks[0], max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def prepare_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)

def ask_question(retriever, query):
    docs = retriever.get_relevant_documents(query)
    context = " ".join(doc.page_content for doc in docs[:3])
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

def generate_logic_questions(text):
    return (
        "1. What is the main idea presented in the document?\n"
        "2. Can you infer any assumptions or conclusions from the content?\n"
        "3. Summarize the document in your own words."
    )

def evaluate_user_answer(question, user_answer, context):
    feedback = (
        f"Question: {question}\n"
        f"User Answer: {user_answer}\n"
        f"Context: {context[:500]}...\n\n"
        f"Feedback: Based on the given context, the user's answer is "
    )
    return feedback + ("relevant." if user_answer.lower() in context.lower() else "not fully aligned with the context.")
