from pathlib import Path

def read_all_texts(folder: str):
    folder = Path(folder)
    docs = []
    print("📂 Reading from:", folder.resolve())   # debug
    for p in folder.glob("*.txt"):
        print("✅ Found:", p.name)                 # debug
        docs.append((p.name, p.read_text(encoding="utf-8")))
    if not docs:
        print("⚠️ No text files found in", folder)
    return docs

def write_chunks(chunks, folder: str):
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks):
        path = Path(folder) / f"chunk_{i:05d}.txt"
        path.write_text(c["text"], encoding="utf-8")
        print("📝 Wrote:", path)                   # debug
