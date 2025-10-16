from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from pathlib import Path

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # supports Malay + English

def load_chunks(folder="data/chunks"):
    texts, meta = [], []
    for p in sorted(Path(folder).glob("*.txt")):
        t = p.read_text(encoding="utf-8")
        texts.append(t)
        meta.append({"path": str(p)})
    return texts, meta

def build_index():
    model = SentenceTransformer(EMB)
    texts, meta = load_chunks()
    if not texts:
        raise SystemExit("no chunks found; run src/prep/make_chunks.py first")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if normalized
    index.add(emb)
    Path("vectordb").mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, "vectordb/index.faiss")
    np.save("vectordb/meta.npy", np.array(meta, dtype=object))
    print("saved FAISS index with", index.ntotal, "vectors")

if __name__ == "__main__":
    build_index()
