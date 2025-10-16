import streamlit as st
from pathlib import Path
from langdetect import detect
import re


# ==========================================================
# INLINE RETRIEVER CLASS
# ==========================================================
EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class Retriever:
    """FAISS retriever for English + Malay .MY domain FAQs"""
    def __init__(self, index_path="vectordb/index.faiss", meta_path="vectordb/meta.npy"):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        self._np = np
        self._faiss = faiss
        self._model = SentenceTransformer(EMB)

        ip, mp = Path(index_path), Path(meta_path)
        if not ip.exists():
            raise FileNotFoundError(f"❌ FAISS index missing: {ip}\nRun: python -m src.embed.build_faiss")
        if not mp.exists():
            raise FileNotFoundError(f"❌ Metadata missing: {mp}\nRun: python -m src.embed.build_faiss")

        self._index = self._faiss.read_index(str(ip))
        self._meta = self._np.load(str(mp), allow_pickle=True)

    def search(self, query: str, k: int = 3):
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self._index.search(q, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            meta_obj = self._meta[idx]
            # robustly unwrap whether it's already a dict or a 0-D numpy object
            if isinstance(meta_obj, dict):
                meta = meta_obj
            else:
                try:
                    meta = meta_obj.item()
                except Exception:
                    meta = meta_obj
            text = Path(meta["path"]).read_text(encoding="utf-8")
            results.append({"score": float(score), "text": text, "source": meta["path"]})
        return results


# ==========================================================
# GENERATOR (Ollama local model)
# ==========================================================
import ollama

SYSTEM_MS = (
    "Anda pembantu yang hanya menjawab berdasarkan KONTEXT yang diberi. "
    "Jika maklumat tiada dalam konteks, jawab: 'Saya tidak pasti berdasarkan dokumen rujukan.' "
    "Jawab **DALAM BAHASA MELAYU SAHAJA**, ringkas (2–4 ayat)"
)
SYSTEM_EN = (
    "You are an assistant that ONLY answers using the provided CONTEXT. "
    "If the info is not in context, say: 'I’m not sure based on the provided documents.' "
    "Answer **IN ENGLISH ONLY**, concise (2–4 sentences)"
)

def build_messages(question: str, contexts: list[dict]):
    # detect language (langdetect returns 'ms' or often 'id' for Malay)
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    is_malay = lang in ("ms", "id")

    system = SYSTEM_MS if is_malay else SYSTEM_EN

    # stitch sources
    ctx_txt = ""
    for i, c in enumerate(contexts, 1):
        ctx_txt += f"[{i}] Source: {c['source']}\n{c['text']}\n\n"

    # few-shot to **anchor** the language
    if is_malay:
        shots = [
            {"role": "user", "content": "Soalan: Siapa layak untuk .edu.my?"},
            {"role": "assistant", "content": "Institusi pendidikan yang diiktiraf oleh KPM atau KPT layak memohon .edu.my."},
        ]
    else:
        shots = [
            {"role": "user", "content": "Question: Who can register .edu.my?"},
            {"role": "assistant", "content": "Educational institutions recognized by MOE or MOHE are eligible for .edu.my."},
        ]

    user = (
        f"CONTEXT:\n{ctx_txt}\n"
        f"QUESTION:\n{question}\n\n"
        "Only use the CONTEXT. If missing, say you don’t know. Keep it short."
        if not is_malay else
        f"KONTEKS:\n{ctx_txt}\n"
        f"SOALAN:\n{question}\n\n"
        "Hanya guna KONTEKS. Jika tiada maklumat, nyatakan tidak pasti. Jawab ringkas."
    )

    messages = [{"role": "system", "content": system}, *shots, {"role": "user", "content": user}]
    return messages, ("ms" if is_malay else "en")

def ensure_language(text: str, target_lang: str, model: str):
    # if model still replied in the wrong language, translate once
    try:
        lang = detect(text)
    except Exception:
        lang = ""
    if target_lang == "ms" and lang not in ("ms", "id"):
        tr = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a translator. Output Malay only."},
                {"role": "user", "content": f"Terjemah jawapan berikut ke Bahasa Melayu yang ringkas dan tepat:\n\n{text}"}
            ],
            options={"num_predict": 220},
            keep_alive="30m",
        )
        return tr["message"]["content"]
    if target_lang == "en" and lang != "en":
        tr = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a translator. Output English only."},
                {"role": "user", "content": f"Translate to concise English:\n\n{text}"}
            ],
            options={"num_predict": 220},
            keep_alive="30m",
        )
        return tr["message"]["content"]
    return text

def generate_answer(question: str, contexts: list[dict], model="llama3"):
    messages, target = build_messages(question, contexts)
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"num_predict": 220},
        keep_alive="30m",
    )
    out = resp["message"]["content"]
    return ensure_language(out, target, model)


# ==========================================================
# STREAMLIT APP
# ==========================================================
st.set_page_config(page_title=".MY Policy Chatbot", page_icon="🇲🇾", layout="centered")
st.title(" .MY Chatbot")

# lazy init retriever
if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = Retriever()
    except Exception as e:
        st.error(str(e))
        st.stop()

# maintain chat history
if "history" not in st.session_state:
    st.session_state.history = []

# chat history
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# chat input
question = st.chat_input("Ask in English or Malay (e.g., 'Siapa layak untuk .edu.my?')")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    ctx = st.session_state.retriever.search(question, k=3)

    with st.expander("See retrieved sources"):
        for i, c in enumerate(ctx, 1):
            st.markdown(f"**[{i}]** {c['source']} — score: {c['score']:.3f}")
            st.code(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # use the model you actually pulled (e.g., "llama3" or "mistral:7b" or "gemma:2b")
            answer = generate_answer(question, ctx, model="mistral:7b")
        st.markdown(answer)

    # ➜ save to history (these two lines were missing)
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))
