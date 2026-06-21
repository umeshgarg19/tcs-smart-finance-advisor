import re
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (  # noqa: E402
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)


@dataclass(frozen=True)
class QuerySpec:
    """Normalized query interpretation used by the retriever."""

    intent: str
    sector_intent: str
    period: str
    group_by_sector: bool = False
    group_by_stock: bool = False
    stock_rank_mode: str = "best"
    year: str | None = None


class FinanceRetriever:
    """Intent-aware retriever for finance data stored in Chroma."""

    AGGREGATE_KEYWORDS = [
        "total",
        "average",
        "summary",
        "overview",
        "overall",
        "how much",
        "how many",
        "aggregate",
        "best",
        "worst",
        "highest",
        "lowest",
        "top",
        "perform",
        "performance",
    ]
    SECTOR_KEYWORDS = ["technology", "healthcare", "finance", "energy", "consumer"]
    NUMERIC_COLUMNS = [
        "Year",
        "Quantity",
        "Buy Price ($)",
        "Current Price ($)",
        "Value ($)",
        "Gain/Loss ($)",
        "row",
    ]

    def __init__(
        self,
        google_api_key: str = GOOGLE_API_KEY,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.google_api_key = google_api_key
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vector_store: Chroma | None = None

    def get_vector_store(self) -> Chroma:
        """Initialize the Chroma vector store lazily and reuse it."""
        if self._vector_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=self.google_api_key,
            )
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def _get_all_documents(self) -> list[Document]:
        """Return every stored document from the current Chroma collection."""
        vector_store = self.get_vector_store()
        payload = vector_store._collection.get(include=["documents", "metadatas"])
        docs = payload.get("documents", [])
        metas = payload.get("metadatas", [])
        return [Document(page_content=text, metadata=meta or {}) for text, meta in zip(docs, metas)]

    def detect_query_intent(self, query: str) -> str:
        """Return aggregate or detail based on query keywords."""
        q = query.lower()
        if any(token in q for token in self.AGGREGATE_KEYWORDS):
            return "aggregate"
        return "detail"

    def detect_sector_intent(self, query: str) -> str:
        """Return detected sector keyword, otherwise general."""
        q = query.lower()
        for sector in self.SECTOR_KEYWORDS:
            if sector in q:
                return sector
        return "general"

    @staticmethod
    def _documents_to_dataframe(documents: list[Document]) -> pd.DataFrame:
        """Convert LangChain documents into a normalized pandas DataFrame."""
        rows = []
        for doc in documents:
            parsed = {}
            for part in doc.page_content.split("|"):
                part = part.strip()
                if ": " in part:
                    key, value = part.split(": ", 1)
                    parsed[key] = value
            rows.append({**doc.metadata, **parsed})

        df = pd.DataFrame(rows)
        for col in FinanceRetriever.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        return df

    @staticmethod
    def _filter_dataframe_by_sector_intent(df: pd.DataFrame, sector_intent: str) -> pd.DataFrame:
        """Filter rows by sector intent; general/all means no filtering."""
        if df.empty or "Sector" not in df.columns:
            return df

        normalized_sector = (sector_intent or "general").strip().lower()
        if normalized_sector in {"general", "all", "*"}:
            return df

        return df[
            df["Sector"].astype(str).str.lower().str.contains(normalized_sector, na=False)
        ].copy()

    @staticmethod
    def _filter_documents_by_sector_intent(documents: list[Document], sector_intent: str) -> list[Document]:
        """Filter documents by sector from metadata or page text."""
        normalized_sector = (sector_intent or "general").strip().lower()
        if normalized_sector in {"general", "all", "*"}:
            return documents

        filtered = []
        for doc in documents:
            meta_sector = str(doc.metadata.get("Sector", "")).lower()
            text = doc.page_content.lower()
            if normalized_sector in meta_sector or f"sector: {normalized_sector}" in text:
                filtered.append(doc)
        return filtered

    @staticmethod
    def _infer_period_from_query(query: str) -> str:
        q = query.lower()
        if "quarter" in q or "qtr" in q:
            return "qtr"
        if "month" in q:
            return "month"
        return "year"

    def parse_query(self, query: str) -> QuerySpec:
        """Parse query into a normalized intent specification."""
        intent = self.detect_query_intent(query)
        sector_intent = self.detect_sector_intent(query)
        period = self._infer_period_from_query(query)
        query_l = query.lower()
        group_by_sector = "sector" in query_l and sector_intent == "general"
        group_by_stock = ("stock" in query_l or "stocks" in query_l) and any(
            token in query_l for token in ["best", "worst", "least", "bad", "top", "perform"]
        )
        stock_rank_mode = "best"
        if any(token in query_l for token in ["worst", "least", "bad", "poor"]):
            stock_rank_mode = "worst"
        if "best" in query_l and any(token in query_l for token in ["worst", "least", "bad"]):
            stock_rank_mode = "both"
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        year = year_match.group(0) if year_match else None
        return QuerySpec(
            intent=intent,
            sector_intent=sector_intent,
            period=period,
            group_by_sector=group_by_sector,
            group_by_stock=group_by_stock,
            stock_rank_mode=stock_rank_mode,
            year=year,
        )

    @staticmethod
    def _filter_aggregate_by_year(df: pd.DataFrame, year: str | None, period: str) -> pd.DataFrame:
        if not year or "period" not in df.columns:
            return df

        if period == "year":
            return df[df["period"] == year]
        return df[df["period"].astype(str).str.startswith(year)]

    @staticmethod
    def _aggregate_to_document(df: pd.DataFrame, period: str) -> list[Document]:
        if df.empty:
            return [
                Document(
                    page_content="No aggregate data found for the requested filters.",
                    metadata={"level": "aggregate_summary", "source": "computed_aggregate"},
                )
            ]

        table_text = df.to_string(index=False)
        return [
            Document(
                page_content=f"Aggregate results by {period}:\n{table_text}",
                metadata={"level": f"{period}_summary", "source": "computed_aggregate"},
            )
        ]

    @staticmethod
    def _sector_aggregate_to_document(df: pd.DataFrame, year: str | None = None) -> list[Document]:
        """Render sector aggregate table plus explicit top/bottom summary for LLM reliability."""
        if df.empty:
            scope = f" for year {year}" if year else ""
            return [
                Document(
                    page_content=f"No sector aggregate data found{scope}.",
                    metadata={"level": "sector_summary", "source": "computed_aggregate"},
                )
            ]

        scope_line = f"Filtered Year: {year}\n" if year else ""
        sorted_df = df.sort_values("total_gain_loss", ascending=False).reset_index(drop=True)
        best = sorted_df.iloc[0]
        worst = sorted_df.iloc[-1]
        table_text = sorted_df.to_string(index=False)
        summary = (
            f"{scope_line}Best sector by total gain/loss: {best['Sector']} ({best['total_gain_loss']:.2f}).\n"
            f"Worst sector by total gain/loss: {worst['Sector']} ({worst['total_gain_loss']:.2f}).\n"
            f"Aggregate results by sector:\n{table_text}"
        )
        return [
            Document(
                page_content=summary,
                metadata={"level": "sector_summary", "source": "computed_aggregate"},
            )
        ]

    @staticmethod
    def _stock_aggregate_to_document(
        best_df: pd.DataFrame,
        worst_df: pd.DataFrame,
        mode: str,
        year: str | None = None,
    ) -> list[Document]:
        """Render stock aggregate details with explicit best/worst summary lines."""
        if best_df.empty and worst_df.empty:
            scope = f" for year {year}" if year else ""
            return [
                Document(
                    page_content=f"No stock performance data found{scope}.",
                    metadata={"level": "stock_summary", "source": "computed_aggregate"},
                )
            ]

        scope_line = f"Filtered Year: {year}\n" if year else ""
        lines = [scope_line.rstrip()] if scope_line else []

        if mode in {"best", "both"} and not best_df.empty:
            best = best_df.iloc[0]
            lines.append(
                f"Best stock by total gain/loss: {best['Symbol']} ({best['Company']}) with {best['total_gain_loss']:.2f}."
            )
            lines.append("Top performing stocks:")
            lines.append(best_df.to_string(index=False))

        if mode in {"worst", "both"} and not worst_df.empty:
            worst = worst_df.iloc[0]
            lines.append(
                f"Worst stock by total gain/loss: {worst['Symbol']} ({worst['Company']}) with {worst['total_gain_loss']:.2f}."
            )
            lines.append("Least performing stocks:")
            lines.append(worst_df.to_string(index=False))

        summary = "\n".join(line for line in lines if line)
        return [
            Document(
                page_content=summary,
                metadata={"level": "stock_summary", "source": "computed_aggregate"},
            )
        ]

    def get_stock_performance_details(
        self,
        query: str = "Show all stock performance",
        sector_intent: str = "general",
        year: str | None = None,
        mode: str = "best",
        top_n: int = 5,
        use_full_dataset: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return best/worst performing stocks with optional year and sector filters."""
        if use_full_dataset:
            results = self._get_all_documents()
        else:
            results = self.get_vector_store().similarity_search(query, k=max(50, top_n * 10))

        df = self._documents_to_dataframe(results)
        df = self._filter_dataframe_by_sector_intent(df, sector_intent)

        if year is not None:
            if "Year" in df.columns:
                df = df[df["Year"].astype("Int64").astype(str) == year]
            elif "Date" in df.columns:
                df = df[df["Date"].dt.year.astype("Int64").astype(str) == year]

        required_cols = ["Symbol", "Company", "Gain/Loss ($)", "Value ($)"]
        if any(col not in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Required columns missing for stock performance: {', '.join(missing)}")

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        grouped = (
            df.dropna(subset=["Symbol"]) 
            .groupby(["Symbol", "Company"], as_index=False)
            .agg(
                trades=("Symbol", "count"),
                total_value=("Value ($)", "sum"),
                total_gain_loss=("Gain/Loss ($)", "sum"),
                avg_gain_loss=("Gain/Loss ($)", "mean"),
            )
        )

        top_n = max(1, top_n)
        best_df = grouped.sort_values("total_gain_loss", ascending=False).head(top_n).reset_index(drop=True)
        worst_df = grouped.sort_values("total_gain_loss", ascending=True).head(top_n).reset_index(drop=True)

        if mode == "best":
            return best_df, pd.DataFrame()
        if mode == "worst":
            return pd.DataFrame(), worst_df
        return best_df, worst_df

    def get_aggregate_results(
        self,
        period: str = "year",
        query: str = "Show all stock performance",
        sector_intent: str = "general",
        k: int = 100,
        use_full_dataset: bool = True,
    ) -> pd.DataFrame:
        """Return aggregate trade performance by year, quarter, or month."""
        period_map = {
            "year": "year",
            "y": "year",
            "qtr": "qtr",
            "quarter": "qtr",
            "q": "qtr",
            "month": "month",
            "m": "month",
        }
        normalized_period = period_map.get(period.lower())
        if not normalized_period:
            raise ValueError("period must be one of: year, qtr, month")

        if use_full_dataset:
            results = self._get_all_documents()
        else:
            results = self.get_vector_store().similarity_search(query, k=k)
        df = self._documents_to_dataframe(results)
        df = self._filter_dataframe_by_sector_intent(df, sector_intent)

        if df.empty:
            return df

        if normalized_period == "year":
            if "Year" in df.columns and df["Year"].notna().any():
                df["period"] = df["Year"].astype("Int64").astype(str)
            elif "Date" in df.columns:
                df["period"] = df["Date"].dt.year.astype("Int64").astype(str)
            else:
                raise ValueError("No Year or Date column available for yearly aggregation")
        elif normalized_period == "qtr":
            if "Date" not in df.columns:
                raise ValueError("Date column is required for quarterly aggregation")
            df["period"] = df["Date"].dt.to_period("Q").astype(str)
        else:
            if "Date" not in df.columns:
                raise ValueError("Date column is required for monthly aggregation")
            df["period"] = df["Date"].dt.to_period("M").astype(str)

        if "Gain/Loss ($)" not in df.columns or "Value ($)" not in df.columns:
            raise ValueError("Required columns missing: Gain/Loss ($), Value ($)")

        return (
            df.dropna(subset=["period"])
            .groupby("period", as_index=False)
            .agg(
                trades=("period", "count"),
                total_value=("Value ($)", "sum"),
                total_gain_loss=("Gain/Loss ($)", "sum"),
                avg_gain_loss=("Gain/Loss ($)", "mean"),
            )
            .sort_values("period")
            .reset_index(drop=True)
        )

    def get_sector_aggregate_results(
        self,
        query: str = "Show all stock performance",
        sector_intent: str = "general",
        year: str | None = None,
        k: int = 200,
        use_full_dataset: bool = True,
    ) -> pd.DataFrame:
        """Return aggregate trade performance grouped by stock sector."""
        if use_full_dataset:
            results = self._get_all_documents()
        else:
            results = self.get_vector_store().similarity_search(query, k=k)
        df = self._documents_to_dataframe(results)
        df = self._filter_dataframe_by_sector_intent(df, sector_intent)

        if year is not None:
            if "Year" in df.columns:
                df = df[df["Year"].astype("Int64").astype(str) == year]
            elif "Date" in df.columns:
                df = df[df["Date"].dt.year.astype("Int64").astype(str) == year]

        if df.empty:
            return df

        if "Sector" not in df.columns:
            raise ValueError("Sector column is required for sector aggregation")

        if "Gain/Loss ($)" not in df.columns or "Value ($)" not in df.columns:
            raise ValueError("Required columns missing: Gain/Loss ($), Value ($)")

        return (
            df.dropna(subset=["Sector"])
            .groupby("Sector", as_index=False)
            .agg(
                trades=("Sector", "count"),
                total_value=("Value ($)", "sum"),
                total_gain_loss=("Gain/Loss ($)", "sum"),
                avg_gain_loss=("Gain/Loss ($)", "mean"),
            )
            .sort_values("total_gain_loss", ascending=False)
            .reset_index(drop=True)
        )

    def smart_retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Intent-aware retrieval with optional sector filtering."""
        docs = self.get_vector_store().similarity_search(query, k=max(k * 5, 20))
        sector_intent = self.detect_sector_intent(query)
        docs = self._filter_documents_by_sector_intent(docs, sector_intent)

        # For detail questions, prioritize row-level docs.
        if self.detect_query_intent(query) == "detail":
            row_docs = [d for d in docs if d.metadata.get("level") == "row"]
            if row_docs:
                return row_docs[:k]

        return docs[:k]

    def retrieve_context(self, query: str, k: int = 4) -> list[Document]:
        """Main retrieval entrypoint used by the RAG pipeline."""
        spec = self.parse_query(query)

        if spec.intent != "aggregate":
            return self.smart_retrieve(query, k=k)

        if spec.group_by_stock:
            best_df, worst_df = self.get_stock_performance_details(
                query="Show all stock performance",
                sector_intent=spec.sector_intent,
                year=spec.year,
                mode=spec.stock_rank_mode,
                top_n=max(1, k),
                use_full_dataset=True,
            )
            return self._stock_aggregate_to_document(
                best_df=best_df,
                worst_df=worst_df,
                mode=spec.stock_rank_mode,
                year=spec.year,
            )

        if spec.group_by_sector:
            sector_df = self.get_sector_aggregate_results(
                query="Show all stock performance",
                sector_intent="general",
                year=spec.year,
                k=300,
            )
            return self._sector_aggregate_to_document(sector_df, year=spec.year)

        agg_df = self.get_aggregate_results(
            period=spec.period,
            query="Show all stock performance",
            sector_intent=spec.sector_intent,
            k=300,
        )
        agg_df = self._filter_aggregate_by_year(agg_df, year=spec.year, period=spec.period)
        return self._aggregate_to_document(agg_df, spec.period)

# Backward-compatible module-level API
_DEFAULT_RETRIEVER = FinanceRetriever()


def get_aggregate_results(
    period: str = "year",
    query: str = "Show all stock performance",
    sector_intent: str = "general",
    k: int = 100,
    use_full_dataset: bool = True,
) -> pd.DataFrame:
    return _DEFAULT_RETRIEVER.get_aggregate_results(
        period=period,
        query=query,
        sector_intent=sector_intent,
        k=k,
        use_full_dataset=use_full_dataset,
    )


def get_sector_aggregate_results(
    query: str = "Show all stock performance",
    sector_intent: str = "general",
    year: str | None = None,
    k: int = 200,
    use_full_dataset: bool = True,
) -> pd.DataFrame:
    return _DEFAULT_RETRIEVER.get_sector_aggregate_results(
        query=query,
        sector_intent=sector_intent,
        year=year,
        k=k,
        use_full_dataset=use_full_dataset,
    )


def retrieve_context(query: str, k: int = 4) -> list[Document]:
    return _DEFAULT_RETRIEVER.retrieve_context(query, k=k)


def get_stock_performance_details(
    query: str = "Show all stock performance",
    sector_intent: str = "general",
    year: str | None = None,
    mode: str = "best",
    top_n: int = 5,
    use_full_dataset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _DEFAULT_RETRIEVER.get_stock_performance_details(
        query=query,
        sector_intent=sector_intent,
        year=year,
        mode=mode,
        top_n=top_n,
        use_full_dataset=use_full_dataset,
    )