# Crop Analysis Dashboard

A small collection of Streamlit dashboards, scripts and notebooks for exploring and visualising Indian crop production and rainfall/climate data. The project contains interactive dashboards, data-cleaning utilities and example notebooks used during analysis.

## Table of contents

- [Project overview](#project-overview)
- [Features](#features)
- [Data included](#data-included)
- [Requirements](#requirements)
- [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
- [Run the dashboards](#run-the-dashboards)
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

app.py notes (Ollama):

- The app expects an Ollama server running locally and models pulled. By default it looks for `OLLAMA_URL` at `http://localhost:11434`.
- To use the assistant, install and run Ollama and pull recommended models (example commands shown by the app):

	- Start Ollama (see Ollama docs): `ollama serve`
	- Pull models used in the app, for example:
		- `ollama pull llama3.1:8b`
		- `ollama pull nomic-embed-text`

You can override the model and server via environment variables:

- `OLLAMA_URL` — Ollama server URL
- `OLLAMA_CHAT_MODEL` — chat model name (default used in `app.py`)
- `OLLAMA_EMBED_MODEL` — embedding model name

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


