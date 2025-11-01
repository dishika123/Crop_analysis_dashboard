import os
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# RAG imports from app.py
import numpy as np
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ...existing code from app.py...
def embed_texts(texts: List[str], model: str) -> np.ndarray:
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

def retrieve(query: str, chunks: List[str], embeddings: np.ndarray, embed_model: str, top_k: int = 8):
    q_vec = embed_texts([query], embed_model)
    sims = cosine_sim(q_vec, embeddings)[0]
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), chunks[int(i)], float(sims[int(i)])) for i in idxs]

# Prompts from aap2.py
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

# Load data
@st.cache_data
def load_data():
    merge_df = pd.read_csv("./data/merge.csv").fillna(value=0)
    crop_pattern_df = pd.read_csv("./data/crop_pattern.csv")
    return merge_df, crop_pattern_df

st.set_page_config(page_title="Crop Intelligence System", page_icon="🌾")
st.title("🌾 Crop Intelligence System")
st.markdown("Combine LangChain Agents + RAG for comprehensive crop analysis")

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    inference_mode = st.radio(
        "Select Inference Mode:",
        ["LangChain Agent", "RAG (Retrieval)", "Hybrid (Both)"],
        index=2
    )
    
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    
    st.markdown("---")
    st.markdown("### Data Files")
    st.info("Using:\n- merge.csv\n- crop_pattern.csv")

# Load data
merge_df, crop_pattern_df = load_data()

# Show data preview
with st.expander("📊 Dataset Preview", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Merge Dataset**")
        st.dataframe(merge_df.head())
    with col2:
        st.markdown("**Crop Pattern Dataset**")
        st.dataframe(crop_pattern_df.head())

# Initialize LangChain agent
@st.cache_resource
def get_langchain_agent(_df, model_name):
    model = ChatOllama(
        model=model_name,
        base_url=OLLAMA_URL,
        temperature=0,
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=_df,
        agent_type="zero-shot-react-description",
        verbose=True,
        allow_dangerous_code=True,
    )
    return agent

# Initialize RAG index
@st.cache_resource
def build_rag_index(_df, embed_model):
    chunks = []
    for idx, row in _df.iterrows():
        chunk = f"State: {row['State']}, District: {row['District']}, Crop: {row['Crop']}, "
        chunk += f"Year: {row['Year']}, Season: {row['Season']}, Production: {row['Production']}, "
        chunk += f"Area: {row['Area']}, Yield: {row['Yield']}"
        chunks.append(chunk)
    
    embeddings = embed_texts(chunks[:1000], embed_model)  # Limit for demo
    return chunks[:1000], embeddings

# Question input
st.markdown("### 🔍 Ask a Question")
question = st.text_area(
    "Enter your question about crops, production, rainfall, etc.:",
    value="What is the total rice production in Andhra Pradesh in 2007?",
    height=100
)

# Run inference
if st.button("🚀 Generate Answer", type="primary"):
    with st.spinner("Processing..."):
        
        # LangChain Agent Response
        if inference_mode in ["LangChain Agent", "Hybrid (Both)"]:
            st.markdown("### 🤖 LangChain Agent Response")
            try:
                agent = get_langchain_agent(merge_df, chat_model)
                query = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
                result = agent.invoke(query)
                
                with st.container():
                    st.markdown("**Agent Analysis:**")
                    st.markdown(result["output"])
            except Exception as e:
                st.error(f"LangChain error: {e}")
        
        # RAG Response
        if inference_mode in ["RAG (Retrieval)", "Hybrid (Both)"]:
            st.markdown("### 📚 RAG (Retrieval) Response")
            try:
                chunks, embeddings = build_rag_index(merge_df, embed_model)
                hits = retrieve(question, chunks, embeddings, embed_model, top_k=5)
                
                # Format context
                context = "\n".join([f"- {text} (relevance: {score:.2f})" 
                                   for _, text, score in hits])
                
                # Call Ollama chat
                messages = [
                    {"role": "system", "content": "You are a helpful agricultural data assistant. Use ONLY the provided context to answer questions. Be concise and cite specific data points."},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
                ]
                
                resp = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": chat_model, "messages": messages, "stream": False},
                    timeout=180
                )
                resp.raise_for_status()
                answer = resp.json().get("message", {}).get("content", "No response")
                
                with st.container():
                    st.markdown("**RAG Answer:**")
                    st.markdown(answer)
                    
                    with st.expander("📌 Retrieved Context", expanded=False):
                        for i, (_, text, score) in enumerate(hits, 1):
                            st.markdown(f"{i}. `{score:.3f}` - {text}")
                            
            except Exception as e:
                st.error(f"RAG error: {e}")

# Additional context from crop_pattern.csv
st.markdown("---")
with st.expander("🌱 Crop Pattern Information", expanded=False):
    if st.text_input("Search crop pattern (e.g., Rice, Wheat):"):
        search_crop = st.session_state.get("search_crop", "Rice")
        pattern_info = crop_pattern_df[
            crop_pattern_df['Crop'].str.contains(search_crop, case=False, na=False)
        ]
        st.dataframe(pattern_info)