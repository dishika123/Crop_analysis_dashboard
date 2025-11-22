# Crop Analysis Dashboard

A small collection of Streamlit dashboards, scripts and notebooks for exploring and visualising Indian crop production and rainfall/climate data. The project contains interactive dashboards, data-cleaning utilities and example notebooks used during analysis.

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [Data included](#data-included)
- [Requirements](#requirements)
- [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
- [Run the dashboards](#run-the-dashboards)
- [CSV chatbot (app.py)](#csv-chatbot)
- [Notebooks and scripts](#notebooks-and-scripts)
- [Project structure](#project-structure)
- [Notes & tips](#notes--tips)
- [License](#license)

## Project overview

This repository hosts dashboards and supporting scripts for analysing relationships between climate (rainfall) and crop production in India. The interactive dashboards (built with Streamlit) let you filter by state, crop, year and season, inspect trends, and visualise seasonal and annual patterns.

Two main Streamlit apps are present:

- `crops_dashboard.py` — the primary dashboard for climate and production analysis (Project Samarth UI, uses `merge.csv` and `season_pattern.csv`).
- `app.py` — a CSV-based RAG/chat assistant that indexes CSVs and queries them via an Ollama chat/embedding backend (requires Ollama running locally and model pulls).

## Features

- Interactive filtering by State, Crop, Year, and Season
- Time series and decadal visualisations of rainfall and production
- Correlation views between climate metrics and crop production
- A CSV-chat assistant for asking natural language questions about any CSV files in the `data/` folder (powered by Ollama)

## Data included

This repo includes several CSVs and notebooks used to prepare and explore data, for example:

- `merge.csv` — merged dataset used by the dashboard (crop, state, year, rainfall columns)
- `season_pattern.csv` — season/sowing/harvest pattern metadata
- `rain.csv`, `crop.csv`, and other helper CSVs/notebooks used during analysis

If you prefer to use your own data, point the apps to a folder containing CSV/XLSX files with appropriate columns.

## Requirements

- Python 3.8 or newer
- See `requirements.txt` for the Python packages used. Key libraries include:
	- streamlit, pandas, plotly, numpy, matplotlib, seaborn, duckdb, openpyxl, scipy, statsmodels

## Quick start (Windows PowerShell)

Open PowerShell and run:

```powershell
# create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

## Run the dashboards

Run the main crop dashboard (`crops_dashboard.py`) with Streamlit:

```powershell
streamlit run crops_dashboard.py
```

Run the CSV-chat assistant (`app.py`) with Streamlit. Note: `app.py` expects an Ollama server for embeddings/chat. If you're not using the assistant, you can skip these steps.

```powershell
# Start Streamlit UI
streamlit run app.py
```

<a name="csv-chatbot"></a>
## CSV chatbot (app.py) — detailed setup & CSV instructions

Yes — `app.py` implements a CSV-powered RAG/chat assistant. The following step-by-step instructions will get it running and show how to prepare your first CSV so you can test the chatbot quickly.

1) Put your first CSV in the `data/` folder (highlighted — do this first)

- Create a folder named `data` in the project root if it doesn't exist:

```powershell
md data
```

- Add a small sample CSV file to that folder so the app has something to index. Example path: `data/sample.csv`.

Example minimal CSV content (save as `data/sample.csv`):

```csv
State,Crop,Year,ANNUAL,JUN,JUL,AUG,SEP,Production
Karnataka,Rice,2019,800,50,200,300,250,1500000
```

Notes about CSVs the app accepts:
- The app will discover files under the Data folder with extensions: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`.
- There is no strict schema required. `app.py` will index every row and convert each row to a text chunk for retrieval. Single-column text files are also supported.
- If your files have different encodings or non-standard separators, Streamlit's sidebar allows pointing to another folder or you can pre-convert files to CSV/Excel.

2) Install dependencies and start Streamlit (PowerShell)

```powershell
# activate or create virtual environment (if not already done)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the app
streamlit run app.py
```

3) Ollama setup (required for chat + embeddings)

- `app.py` uses Ollama for both embeddings and chat. Ollama must be running and have the embedding and chat models pulled locally. By default the app looks for `OLLAMA_URL=http://localhost:11434`.
- Example Ollama commands (run in a separate terminal following Ollama's install instructions):

```powershell
# start Ollama server (follow Ollama installation docs first)
# ollama serve

# pull models used by the app
# ollama pull llama3.1:8b
# ollama pull nomic-embed-text
```

Environment variables you can set (optional):

- `OLLAMA_URL` — Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL` — chat model name (default in `app.py`)
- `OLLAMA_EMBED_MODEL` — embedding model name (default in `app.py`)

4) Using the app UI (sidebar controls)

- Data folder: the sidebar contains a `Data folder` input (default: `./data`). If your CSVs are elsewhere, set the path there.
- Rebuild index: click `Rebuild index` after adding or changing CSVs. The app will show the discovered files and the number of indexed rows.
- Top K / Row limit: the sidebar lets you tune how many matching rows to retrieve and how many rows to index per file (useful for large files).

5) Performance & tips

- Indexing: `app.py` currently creates one embedding request per chunk (row). For many rows this can be slow and use a lot of memory — for production you may want to batch embeddings or persist the index to disk.
- Use small sample files first to verify the pipeline (CSV discovery, indexing, Ollama responses) before indexing large datasets.
- If Ollama is not available, I can help adapt the code to use another provider (OpenAI, local embedding server, etc.).

6) What the assistant does when you ask a question

- The app retrieves top-K similar rows (based on embeddings) and builds a short context that is passed to the chat model. The system prompt instructs the assistant to answer using only the CSV context and to cite file names / row indices when relevant.

If you want, I can add a tiny `data/sample.csv` file to the repo as a working example and/or modify `app.py` to batch embeddings and persist the index — tell me which you'd prefer.

## Notebooks and scripts

- `crop_data.ipynb`, `crop_pattern.ipynb`, `rain_data.ipynb`, `data_fetch.ipynb` — analysis notebooks used to prepare and explore the datasets.
- `remove_dup.py`, `fix_duplicate.py`, `merge.ipynb` — data cleaning and merging helpers.
- `aap2.py`, `app3.py`, `crops_dashboard.py` — alternative apps or experiment scripts.

## Project structure

Top-level files and purpose (non-exhaustive):

- `crops_dashboard.py` — main Streamlit dashboard (interactive visualisations)
- `app.py` — CSV RAG/chat assistant using Ollama
- `merge.csv`, `season_pattern.csv`, `rain.csv`, `crop.csv`, `crop_pattern.csv` — datasets
- `*.ipynb` — Jupyter notebooks with exploratory analysis and data prep
- `requirements.txt` — Python dependencies

## Notes & tips

- If the Streamlit app fails to find data files, confirm your current working directory is the project root or set absolute paths for CSVs inside the scripts.
- Large CSVs may increase app start time; consider sampling or using DuckDB for faster queries.
- The CSV-chat assistant (`app.py`) builds an index of rows and creates embeddings for retrieval. This requires a running Ollama instance and the embedding model specified in the app.


