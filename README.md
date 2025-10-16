.MY Policy Chatbot

A local AI-powered chatbot that answers frequently asked questions about Malaysia’s .MY domain policies.
It uses Retrieval-Augmented Generation (RAG) with FAISS, SentenceTransformers, and Ollama (local LLMs such as Llama 3, Mistral, or Gemma) all integrated into a Streamlit web interface.

Overview

The chatbot retrieves relevant policy texts from locally stored .txt files, ranks them using semantic similarity, and generates context-aware answers in English or Malay.
It is designed to work fully offline using lightweight open-source components.

Features

Retrieval-Augmented Generation (RAG): Combines semantic search and LLM responses.

Bilingual (English + Malay): Answers in the same language as the question.

Local Inference: Runs entirely offline using Ollama and local models.

FAISS Vector Store: Efficient semantic retrieval from pre-embedded domain policy documents.

Streamlit Interface: Clean, lightweight web UI for querying and inspecting retrieved sources.

Tech Stack
Component	Technology Used
Programming Language	Python 3.10+
Framework	Streamlit
Model Hosting	Ollama (local inference)
Embedding Model	sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Vector Database	FAISS
File Management	NumPy, Pathlib
Data Source	Local .txt policy files
Environment	Virtual environment (venv)

Installation and Setup
1. Clone the Repository
git clone https://github.com/niainsyrh/policy-chatbot.git
cd policy-chatbot

2. Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Prepare the Vector Database
Ensure your .txt policy files are placed in data/sources, then run:
python -m src.prep.make_chunks
python -m src.embed.build_faiss

5. Pull a Local LLM Model via Ollama
ollama pull llama3
(You may also use mistral:7b or gemma:2b for smaller models.)

6. Launch the Chatbot
streamlit run app.py
Access it in your browser at http://localhost:8501.
