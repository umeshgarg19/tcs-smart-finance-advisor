import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of how the script is run
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME
)


def _normalize_name(value: str) -> str:
    """Normalize header names for tolerant column matching."""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a DataFrame column by trying normalized candidate names."""
    normalized_map = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None


def _to_float(value, default: float = 0.0) -> float:
    """Best-effort numeric conversion for mixed Excel text/number cells."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_excels(data_dir: str = "./data") -> list:
    """
    Load all Excel files from the data directory.

    Args:
        data_dir: Path to directory containing Excel files

    Returns:
        List of documents built from sheet rows
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    excel_files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls"))
    if not excel_files:
        return documents

    print(f"Found {len(excel_files)} Excel file(s)")

    for excel_file in excel_files:
        print(f"Loading Excel: {excel_file.name}")
        # Read raw sheet rows to detect the actual header row.
        sheets = pd.read_excel(excel_file, sheet_name=None, header=None)

        for sheet_name, raw_df in sheets.items():
            df = raw_df.copy()

            # Many financial sheets include a title row before real headers.
            header_row = None
            required_markers = {
                "tradeid",
                "date",
                "symbol",
                "sector",
                "action",
                "quantity",
            }

            scan_limit = min(len(df), 30)
            for i in range(scan_limit):
                row_values = [
                    _normalize_name(v)
                    for v in df.iloc[i].tolist()
                    if pd.notna(v)
                ]
                if len(required_markers.intersection(set(row_values))) >= 4:
                    header_row = i
                    break

            if header_row is not None:
                header_values = df.iloc[header_row].tolist()
                columns = []
                for idx, val in enumerate(header_values):
                    name = str(val).strip() if pd.notna(val) else ""
                    columns.append(name if name else f"col_{idx}")

                df = df.iloc[header_row + 1 :].copy()
                df.columns = columns

            #print(df.head(10)) #debug

            # Keep only rows with at least one non-null value.
            cleaned_df = df.dropna(how="all")
            for row_index, row in cleaned_df.iterrows():
                row_text = " | ".join(
                    f"{col}: {row[col]}"
                    for col in cleaned_df.columns
                    if pd.notna(row[col])
                )                
                if row_text.strip():
                    documents.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "source": excel_file.name,
                                "sheet": sheet_name,
                                "row": int(row_index),
                            },
                        )
                    )

    print(f"Total Excel rows loaded: {len(documents)}")
    return documents


def _chunk_dicts_to_documents(chunks: list[dict]) -> list[Document]:
    """Convert hybrid chunk dictionaries into LangChain Document objects."""
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=str(chunk.get("text", "")),
                metadata={
                    "level": chunk.get("level", "unknown"),
                    **(chunk.get("metadata") or {}),
                },
            )
        )
    return documents


def create_embeddings_and_vector_store(chunks: list[dict]) -> Chroma:
    """
    Create embeddings using Google Gemini and store in Chroma vector database.
    Processes chunks in small batches to respect free-tier rate limits (100 req/min).

    Args:
        chunks: List of document chunks

    Returns:
        Chroma vector store instance
    """
    import time

    print(f"\nCreating embeddings using {EMBEDDING_MODEL}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    print(f"Storing vectors in Chroma database: {CHROMA_DB_PATH}")
    chunk_documents = _chunk_dicts_to_documents(chunks)

    # Free tier allows 100 requests/min; batch by 50 with a 65s pause between batches
    BATCH_SIZE = 50
    PAUSE_SECONDS = 65
    vector_store = None

    for i in range(0, len(chunk_documents), BATCH_SIZE):
        batch = chunk_documents[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunk_documents) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DB_PATH,
            )
        else:
            vector_store.add_documents(batch)

        if i + BATCH_SIZE < len(chunk_documents):
            print(f"  Rate-limit pause: waiting {PAUSE_SECONDS}s before next batch...")
            time.sleep(PAUSE_SECONDS)

    print("Vector store created and persisted successfully")
    return vector_store

def convert_documents_to_df(documents: list) -> pd.DataFrame:
    """
    Convert a list of LangChain Document objects into a pandas DataFrame.

    For Excel-derived documents, this also parses the `key: value | ...` text into columns.

    Args:
        documents: List of LangChain Document objects

    Returns:
        pandas DataFrame with text, metadata, and parsed key/value columns
    """
    rows = []
    for doc_id, doc in enumerate(documents):
        row = {
            "document_id": doc_id,
            "page_content": doc.page_content,
        }

        # Flatten metadata into top-level columns.
        for key, value in (doc.metadata or {}).items():
            row[key] = value

        # Parse rows generated from Excel: "col1: val1 | col2: val2".
        if isinstance(doc.page_content, str) and " | " in doc.page_content and ": " in doc.page_content:
            for part in doc.page_content.split(" | "):
                if ": " in part:
                    k, v = part.split(": ", 1)
                    if k and k not in row:
                        row[k.strip()] = v.strip()

        rows.append(row)

    return pd.DataFrame(rows)



def chunk_data(df: pd.DataFrame) -> list[dict]:
    chunks = []

    trade_id_col = _find_column(df, ["TradeID", "Trade ID"])
    action_col = _find_column(df, ["Action"])
    quantity_col = _find_column(df, ["Quantity"])
    symbol_col = _find_column(df, ["Symbol"])
    date_col = _find_column(df, ["Date"])
    gain_col = _find_column(df, ["GainLoss", "Gain/Loss ($)", "Gain Loss"])
    value_col = _find_column(df, ["Value", "Value ($)"])
    year_col = _find_column(df, ["Year"])
    sector_col = _find_column(df, ["Sector"])

    # Layer 1: Row-level (granular detail)
    for idx, r in df.iterrows():
        trade_id = r.get(trade_id_col) if trade_id_col else idx
        action = r.get(action_col, "") if action_col else ""
        quantity = r.get(quantity_col, "") if quantity_col else ""
        symbol = r.get(symbol_col, "") if symbol_col else ""
        date = r.get(date_col, "") if date_col else ""
        gain_loss = _to_float(r.get(gain_col, 0.0)) if gain_col else 0.0
        value = _to_float(r.get(value_col, 0.0)) if value_col else 0.0
        year = r.get(year_col, "") if year_col else ""
        sector = r.get(sector_col, "") if sector_col else ""

        text = (
            f"Trade {trade_id}: {action} {quantity} {symbol} on {date} for value {value} in year {year} sector is {sector} - "
            f"Gain/Loss ${gain_loss:.2f}"
        )
        chunks.append({"text": text, "level": "row", "metadata": r.to_dict()})    

    return chunks


def ingest_pipeline(data_dir: str = "./data") -> None:
    """
    Complete ingestion pipeline: load PDFs -> chunk -> embed -> store.
    
    Args:
        data_dir: Path to directory containing PDF files
    """
    try:
        print("=" * 60)
        print("Starting RAG Ingestion Pipeline")
        print("=" * 60)
        
        # Step 1: Load supported documents (PDF + Excel)
        documents = load_excels(data_dir)
        
        documents_df = convert_documents_to_df(documents)
        print(f"DataFrame rows: {len(documents_df)}")
        
        #print(documents_df.head(10)) #debug
              
        # Step 2: Chunk Excel DataFrame
        chunks = chunk_data(documents_df)
        
        #Step 3: Create embeddings and store
        vector_store = create_embeddings_and_vector_store(chunks)        
        
        print("\n" + "=" * 60)
        print("[OK] Ingestion Pipeline Complete")
        print(f"  - Documents: {len(documents)} pages")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Vector DB: {CHROMA_DB_PATH}")
        print(f"  - Collection: {COLLECTION_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Error during ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    ingest_pipeline()
