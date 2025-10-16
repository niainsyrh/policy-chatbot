import ollama

SYSTEM_PROMPT = """You are a helpful assistant that ONLY answers using the provided context.
If the answer is not in the context, say you don't know.
Reply in the user's language (Malay or English).
Always include a short citation like (Source: {source_file}). Be concise and precise.
"""

def build_prompt(question: str, contexts: list[dict]):
    joined = ""
    for i, c in enumerate(contexts, 1):
        joined += f"[{i}] Source: {c['source']}\n{c['text']}\n\n"
    user = f"Question: {question}\n\nUse only the sources above. If not in sources, say you don't know."
    return SYSTEM_PROMPT + "\n\n" + joined, user

def generate_answer(question: str, contexts: list[dict], model="llama3:8b") -> str:
    system, user = build_prompt(question, contexts)
    prompt = system + "\n\n" + user
    r = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
    return r["message"]["content"]
