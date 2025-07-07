import streamlit as st
import fitz  
from backend import summarize_doc, prepare_vector_db, ask_question, generate_logic_questions, evaluate_user_answer

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.set_page_config(page_title="📄 GenAI Doc Assistant", layout="wide")

st.markdown("## 📄 GenAI Document Assistant")
st.markdown("Interact with your documents using AI: Summary, Q&A, Logic Questions, and Answer Evaluation.")

with st.sidebar:
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    st.markdown("---")
    st.markdown("### 💡 Smart Insight")
    st.success("“You're not just uploading files — you're unlocking knowledge.🌟”")
    

    st.markdown("---")
    st.markdown("Built by **Ravishankar Kumar**")


if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""



if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        file_content = extract_text_from_pdf(uploaded_file)
    else:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")

    MAX_CHARS = 10000
    if len(file_content) > MAX_CHARS:
        file_content = file_content[:MAX_CHARS]
        st.info(f"Only first {MAX_CHARS} characters processed for speed.")

    st.session_state.doc_text = file_content
    st.success("✅ File uploaded successfully!")

if st.session_state.doc_text:
    text = st.session_state.doc_text
    with st.spinner("🔍 Processing document..."):
        vectorstore = prepare_vector_db(text)

   
    tabs = st.tabs(["📑 Summary", "❓ Q&A", "🧠 Logic Questions", "✅ Evaluation"])

    with tabs[0]:
        st.subheader("📑 Document Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_doc(text)
                st.write(summary)

    with tabs[1]:
        st.subheader("❓ Ask Questions")
        user_query = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                result = ask_question(vectorstore.as_retriever(), user_query)
                st.write(result)

    with tabs[2]:
        st.subheader("🧠 Logic-Based Questions")
        if st.button("Generate Questions"):
            with st.spinner("Generating logic questions..."):
                questions = generate_logic_questions(text)
                st.write(questions)

    with tabs[3]:
        st.subheader("✅ Evaluate User Answer")
        q = st.text_input("Question:")
        user_ans = st.text_area("Your Answer:")
        context = text[:1000]  
        if st.button("Evaluate Answer"):
            with st.spinner("Evaluating..."):
                result = evaluate_user_answer(q, user_ans, context)
                st.write(result)
else:
    st.warning("📂 Please upload a file to begin.")