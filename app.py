import os
import glob
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def ollama_is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False

def embed_texts(texts: List[str], model: str) -> np.ndarray:
    # Calls Ollama embeddings API once per text (simple and robust)
    vecs = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=60,
        )
        resp.raise_for_status()
        vecs.append(resp.json()["embedding"])
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def call_ollama_chat(model: str, messages: List[Dict]) -> str:
    # Request a single JSON response instead of streaming NDJSON
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        headers={"Accept": "application/json"},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    # Typical non-stream response
    if isinstance(data, dict):
        # Prefer chat message content
        msg = data.get("message", {})
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        # Some models use 'response'
        if "response" in data:
            return data["response"]
        if "content" in data:
            return data["content"]
    return str(data)

def find_data_files(data_dir: str) -> List[str]:
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    paths = []
    for p in Path(data_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)

def load_csvs(data_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    files = find_data_files(data_dir)
    out = []
    for fp in files:
        df = None
        try:
            if fp.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(fp)
            elif fp.lower().endswith(".tsv"):
                df = pd.read_csv(fp, sep="\t", engine="python")
            else:
                # Let pandas infer the delimiter
                df = pd.read_csv(fp, sep=None, engine="python")
        except Exception:
            # One more permissive attempt
            try:
                df = pd.read_csv(fp)
            except Exception:
                df = None

        # Fallback: treat file as a list of lines (useful for non-tabular text)
        if df is None or df.shape[0] == 0:
            try:
                with open(fp, "r", errors="ignore") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    df = pd.DataFrame({"text": lines})
            except Exception:
                pass

        if df is not None and df.shape[0] > 0:
            out.append((fp, df))
    return out

def rows_to_chunks(df: pd.DataFrame, source: str, max_rows: int = None) -> List[str]:
    chunks = []
    cols = list(df.columns)
    # If it's a single-column text dataframe, simplify chunking
    if len(cols) == 1:
        col = cols[0]
        for i, v in enumerate(df[col].astype(str).tolist()):
            if max_rows is not None and i >= max_rows:
                break
            chunks.append(f"file: {os.path.basename(source)} | row_index: {i}\n{col}: {v}")
        return chunks

    iterable = df.itertuples(index=False, name=None)
    for i, row in enumerate(iterable):
        if max_rows is not None and i >= max_rows:
            break
        pairs = [f"{c}: {v}" for c, v in zip(cols, row)]
        text = f"file: {os.path.basename(source)} | row_index: {i}\n" + " | ".join(pairs)
        chunks.append(text)
    return chunks

def build_index(data_dir: str, embed_model: str, row_limit_per_file: int = None):
    csvs = load_csvs(data_dir)
    chunks, meta = [], []
    for fp, df in csvs:
        # Guard against mistaken CSVs (like 'pip install ...' single-line)
        if df.shape[0] == 0 and df.shape[1] <= 1:
            continue
        per_file_chunks = rows_to_chunks(df, fp, max_rows=row_limit_per_file)
        chunks.extend(per_file_chunks)
        meta.extend([{"source": fp}] * len(per_file_chunks))
    if not chunks:
        return None, None, None
    embeddings = embed_texts(chunks, embed_model)
    return chunks, embeddings, meta

def retrieve(query: str, chunks: List[str], embeddings: np.ndarray, embed_model: str, top_k: int = 8) -> List[Tuple[int, str, float]]:
    q_vec = embed_texts([query], embed_model)
    sims = cosine_sim(q_vec, embeddings)[0]
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), chunks[int(i)], float(sims[int(i)])) for i in idxs]

def format_context(hits: List[Tuple[int, str, float]]) -> str:
    lines = []
    for _, text, score in hits:
        lines.append(f"- {text}  (score={score:.3f})")
    return "\n".join(lines)

st.set_page_config(page_title="CSV RAG (Ollama)", page_icon="📊")
st.title("CSV Chatbot (Ollama, no LangChain)")

with st.sidebar:
    st.markdown("Settings")
    data_dir = st.text_input("Data folder", value=str(Path(__file__).resolve().parent / "data"))
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Top K rows", min_value=3, max_value=25, value=8, step=1)
    row_limit = st.number_input("Row limit per file (optional)", min_value=0, value=0, step=100)
    if row_limit == 0:
        row_limit = None
    rebuild = st.button("Rebuild index")

    with st.expander("Discovered files", expanded=False):
        discovered = find_data_files(data_dir)
        st.caption(f"{len(discovered)} files")
        if not discovered:
            st.write("No data files found. Point 'Data folder' to where your files are.")
        else:
            for fp in discovered[:100]:
                st.write(os.path.relpath(fp, data_dir))

if not ollama_is_running():
    st.error("Ollama is not running. Start it with: `ollama serve` and pull models:\n- ollama pull llama3.1:8b\n- ollama pull nomic-embed-text")
    st.stop()

if "index" not in st.session_state or rebuild:
    with st.status("Building index...", expanded=False):
        chunks, embeddings, meta = build_index(data_dir, embed_model, row_limit_per_file=row_limit)
        st.session_state["index"] = {"chunks": chunks, "embeddings": embeddings, "meta": meta, "built_at": time.time()}
    if st.session_state["index"]["chunks"] is None:
        st.warning("No rows indexed. Check the 'Discovered files' list and preview your files.")
    else:
        st.success(f"Indexed {len(st.session_state['index']['chunks'])} rows.")

msgs = st.session_state.get("messages", [])
for role, content in msgs:
    with st.chat_message(role):
        st.write(content)

user_q = st.chat_input("Ask about your CSVs...")
if user_q:
    if not st.session_state["index"] or st.session_state["index"]["chunks"] is None:
        st.warning("Index is empty. Add CSVs to the data folder and click Rebuild index.")
        st.stop()

    chunks = st.session_state["index"]["chunks"]
    embeddings = st.session_state["index"]["embeddings"]

    with st.chat_message("user"):
        st.write(user_q)
    st.session_state.setdefault("messages", []).append(("user", user_q))

    hits = retrieve(user_q, chunks, embeddings, embed_model, top_k=top_k)
    context = format_context(hits)

    system_prompt = (
        "You are a helpful data assistant. Use ONLY the provided CSV context to answer. "
        "If the answer is not in the context, say you don't know. Be concise and cite file names and row indices when relevant."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_q}\n\nContext rows:\n{context}"},
    ]

    with st.chat_message("assistant"):
        try:
            answer = call_ollama_chat(chat_model, messages)
            st.write(answer)
            st.session_state["messages"].append(("assistant", answer))
        except Exception as e:
            st.error(f"Ollama error: {e}")