import streamlit as st
import fitz  
from backend import summarize_doc, prepare_vector_db, ask_question, generate_logic_questions, evaluate_user_answer

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        t = page.get_text("text")
        if len(t.split()) < 20:
            blocks = page.get_text("blocks")
            t = " ".join([b[4] for b in blocks if isinstance(b[4], str)])
        text += t + "\n"
    return text

st.set_page_config(page_title="📄 GenAI Doc Assistant", layout="wide")
st.markdown("## 📄 GenAI Document Assistant")

with st.sidebar:
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    st.markdown("---")
    st.markdown("### 💡 Smart Insight")
    st.success("“You're not just uploading files — you're unlocking knowledge.🌟”")
    st.markdown("Built by **Ravishankar Kumar & Rekha Kumari**")

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

if uploaded_file is not None and not st.session_state.doc_text:
    with st.status("Processing file..."):
        if uploaded_file.type == "application/pdf":
            file_content = extract_text_from_pdf(uploaded_file)
        else:
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        
        st.session_state.doc_text = file_content[:10000] # Limit for performance
        st.success("✅ File uploaded successfully!")

if st.session_state.doc_text:
    # Memoize vector db so it doesn't rebuild every click
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = prepare_vector_db(st.session_state.doc_text)

    tabs = st.tabs(["📑 Summary", "❓ Q&A", "🧠 Logic Questions", "✅ Evaluation"])

    with tabs[0]:
        st.subheader("📑 Document Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                st.write(summarize_doc(st.session_state.doc_text))

    with tabs[1]:
        st.subheader("❓ Ask Questions")
        user_query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                res = ask_question(st.session_state.vectorstore.as_retriever(), user_query)
                st.write(res)

    with tabs[2]:
        st.subheader("🧠 Logic-Based Questions")
        if st.button("Generate Questions"):
            st.write(generate_logic_questions(st.session_state.doc_text))

    with tabs[3]:
        st.subheader("✅ Evaluate User Answer")
        q = st.text_input("Question Context:")
        user_ans = st.text_area("Your Answer:")
        if st.button("Evaluate"):
            res = evaluate_user_answer(q, user_ans, st.session_state.doc_text[:1000])
            st.write(res)
else:
    st.warning("📂 Please upload a file to begin.")
