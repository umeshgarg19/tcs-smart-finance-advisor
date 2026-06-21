# Smart Finance Advisor - RAG System

A Retrieval-Augmented Generation (RAG) system for financial document analysis using Google Gemini API and LangChain.

## Setup

### 1. Install Python

Download and install Python 3.14 or later from [python.org](https://www.python.org/downloads/) or use a package manager:

**Windows (using winget):**
```powershell
winget install Python.Python.3.14
```

**macOS (using brew):**
```bash
brew install python@3.14
```

**Linux (using apt):**
```bash
sudo apt-get install python3.14 python3.14-venv
```

### 2. Create Virtual Environment

Navigate to the project directory and create a local virtual environment:

```powershell
cd tcs-smart-finance-advisor
python -m venv .venv
```

### 3. Set Execution Policy (Windows Only)

On Windows, you may need to set the execution policy to allow script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Activate Virtual Environment

Activate the virtual environment:

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 5. Get Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 6. Configure Environment

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 7. Install Dependencies

With the virtual environment activated, install all required packages:

```powershell
python -m pip install -r requirements.txt
```

### 8. Add PDF Documents

Place PDF files in the `data/` directory:

```
data/
  annual_report.pdf
  financial_policy.pdf
  ...
```

## Usage

### Ingest PDFs and Create Vector Store

```powershell
python -m src.ingest_data
```

This will:
- Load all PDFs from `data/`
- Split into chunks (1000 chars, 200 char overlap)
- Create embeddings using Google's `models/embedding-001`
- Store vectors in `chroma_db/`

### Query the Knowledge Base

```powershell
python -m src.main
```

Edit the query in `src/main.py` to ask different questions.

### Run Interactive Retrieval

```powershell
python -m src.retrieve
```

## Project Structure

```
tcs-smart-finance-advisor/
├── .venv/                  # Virtual environment
├── data/                   # PDF documents
├── chroma_db/              # Vector database
├── src/
│   ├── config.py          # Configuration
│   ├── ingest_data.py     # PDF ingestion & embedding
│   ├── retrieve.py        # Vector search
│   └── main.py            # QA pipeline
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## Configuration

Edit `.env` to customize:

```
CHUNK_SIZE=1000           # Document chunk size
CHUNK_OVERLAP=200         # Overlap between chunks
EMBEDDING_MODEL=models/embedding-001
CHAT_MODEL=gemini-1.5-flash
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=smart-finance-docs
```

## Models

- **Embedding**: `models/embedding-001` (Google)
- **Chat**: `gemini-1.5-flash` (fast, low-cost)
  - Alternative: `gemini-1.5-pro` (more capable)

## Troubleshooting

**Error: "GOOGLE_API_KEY not set"**
- Ensure `.env` file exists with your API key
- Check key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)

**Error: "No PDF files found"**
- Add PDFs to `data/` directory
- Ensure filenames end with `.pdf`

**Vector store not found**
- Run `python -m src.ingest_data` first to create embeddings
