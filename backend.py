from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# lightweight models (important)
def get_summarizer():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

def get_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def summarize_doc(text):
    summarizer = get_summarizer()
    prompt = "Summarize this:\n" + text[:1000]
    result = summarizer(prompt, max_length=120, do_sample=False)
    return result[0]['generated_text']


def prepare_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding)


def ask_question(retriever, query):
    qa_pipeline = get_qa_pipeline()
    docs = retriever.get_relevant_documents(query)
    context = " ".join(doc.page_content for doc in docs[:3])
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
