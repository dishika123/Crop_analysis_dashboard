# 🌾 Crop Analysis & CSV Chat Assistant

A lightweight project combining **Streamlit dashboards** and a powerful **CSV-based Chat Assistant** for exploring Indian crop production and climate data.

## 🚀 What’s Inside  
### 🤖 CSV Chat Assistant (Main Highlight)
- Ask **natural-language questions** about any CSV in the `data/` folder  
- Powered by **Ollama embeddings + local LLM chat models**  
- Supports querying patterns, summaries, correlations, and row-level insights  
- Automatically indexes CSV files on launch  
- Customizable model & server settings via environment variables

### 📊 Crop Dashboard  
- Explore production & rainfall trends by **state, crop, year, season**  
- Visualize seasonal patterns, correlations & time-series trends  
- Built using **Streamlit + Plotly**  
👉 **Live Demo:** https://dishika123-crop-analysis-dashboard-crops-dashboard-n3hxgz.streamlit.app/

## 📁 Data  
Includes merged rainfall–crop datasets, season pattern metadata, and helper CSVs.  
You can also plug in your **own CSV/XLSX files**.

## 🛠 Requirements  
- Python 3.8+  
- Install dependencies:
pip install -r requirements.txt

## ▶️ Run Locally
Run the Dashboard
streamlit run crops_dashboard.py

## Run the CSV Chat Assistant
Requires an Ollama server running locally:
streamlit run app.py

## Recommended Ollama setup:
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

## 📂 Project Contents
crops_dashboard.py — main dashboard
app.py — CSV chat assistant (Ollama-powered)
data/*.csv — datasets
*.ipynb — analysis notebooks
requirements.txt

```bash
