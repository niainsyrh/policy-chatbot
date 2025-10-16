import re
from src.utils.io import read_all_texts, write_chunks

def split_into_chunks(text, max_chars=800, overlap=120):
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w); cur_len += len(w) + 1
        if cur_len >= max_chars:
            s = " ".join(cur)
            chunks.append(s)
            back = s[-overlap:]
            cur = back.split()
            cur_len = len(back)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.strip()) >= 50]

def build_chunks(src_folder="data/sources", out_folder="data/chunks"):
    docs = read_all_texts(src_folder)
    chunks = []
    for fname, text in docs:
        text = re.sub(r"\n{2,}", "\n", text)
        for part in split_into_chunks(text):
            chunks.append({"source": fname, "text": part})
    write_chunks(chunks, out_folder)
    print(f"wrote {len(chunks)} chunks to {out_folder}")  # ✅ important

if __name__ == "__main__":
    build_chunks()  # ✅ this line must exist
