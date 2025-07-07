
# 📄 GenAI Document Assistant

A powerful Streamlit-based GenAI application that lets you **upload PDF or TXT documents** and interact with them using artificial intelligence. Get instant summaries, ask questions, generate logic-based questions, and even evaluate your own answers — all powered by OpenAI and LangChain.

## Features

-  **Document Summarization**
-  **Question & Answering**
-  **Logic-Based Question Generation**
-  **Answer Evaluation**

## Installation

```bash
git clone https://github.com/Ravishankar119/genai-doc-assistant.git
cd genai-doc-assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup API Key

Create a `.env` file and add your OpenAI key:
```env
```

## Run the App

```bash
streamlit run app.py
```

## File Structure

```
genai-doc-assistant/
├── app.py
├── backend.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

## 👤 Author

**Ravishankar Kumar**

## 🛡️ License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
