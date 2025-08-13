"""Base classes for evaluation system abstraction."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from loguru import logger
from pydantic import BaseModel, SecretStr
from tqdm import tqdm

T = TypeVar("T", bound=BaseModel)


class DateTimeEncoder(json.JSONEncoder):
    """datetime → ISO8601 で JSON 化"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ProgressTracker:
    """Manages database operations for tracking evaluation progress."""

    def __init__(self, db_path: Path, table_suffix: str = ""):
        self.db_path = db_path
        self.table_name = f"progress{table_suffix}"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        ) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    session_id    TEXT PRIMARY KEY,
                    processed     INTEGER DEFAULT 0,
                    bad_count     INTEGER DEFAULT 0,
                    cost          REAL    DEFAULT 0,
                    last_updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def get_progress(self, session_id: str) -> tuple[bool, int, float] | None:
        """Get progress for a session. Returns (processed, bad_count, cost) or None."""
        with sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        ) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute(
                f"SELECT processed, bad_count, cost FROM {self.table_name} WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return bool(row[0]), row[1], row[2]
            return None

    def update_progress(self, session_id: str, bad_count: int, cost: float) -> None:
        """Update progress for a session."""
        with sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        ) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name} (session_id, processed, bad_count, cost, last_updated)
                VALUES (?, 1, ?, ?, CURRENT_TIMESTAMP)
                """,
                (session_id, bad_count, cost),
            )
            conn.commit()

    def get_unprocessed_sessions(
        self, dataset: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get list of unprocessed sessions from dataset."""
        entries_todo = []
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.cursor()
            for entry in dataset:
                session_id = entry["id"]
                cur.execute(
                    f"SELECT processed FROM {self.table_name} WHERE session_id = ?",
                    (session_id,),
                )
                if not cur.fetchone():
                    entry_copy = dict(entry)
                    entry_copy["id"] = session_id
                    entries_todo.append(entry_copy)
        return entries_todo


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(schema: type[BaseModel], use_azure: bool = False) -> Any:
        """Create LLM instance with structured output."""
        if use_azure:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY environment variable is required"
                )

            return AzureChatOpenAI(
                azure_deployment="gpt-4.1",
                azure_endpoint="https://experimental.openai.azure.com",
                api_version="2025-01-01-preview",
                api_key=SecretStr(secret_value=api_key),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            ).with_structured_output(schema, include_raw=True)
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")

            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                api_key=api_key,
            ).with_structured_output(schema, include_raw=True)


class EvaluationConfig(Generic[T]):
    """Configuration for evaluation."""

    def __init__(
        self,
        schema: type[T],
        prompt_template: str,
        output_dir: str,
        table_suffix: str = "",
    ):
        self.schema = schema
        self.prompt_template = prompt_template
        self.output_dir = output_dir
        self.table_suffix = table_suffix


class BaseEvaluator(Generic[T], ABC):
    """Base class for conversation evaluators."""

    def __init__(self, config: EvaluationConfig[T], run_name: str):
        self.config = config
        self.run_name = run_name
        self.log_dir = Path(f"{os.environ['RESULTS_DIR']}/{run_name}")
        self.output_dir = self.log_dir / config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_tracker = ProgressTracker(
            self.log_dir / f"{config.output_dir}/progress.db", config.table_suffix
        )

    @abstractmethod
    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format conversation messages for evaluation."""
        pass

    @abstractmethod
    def process_response(
        self, response: T, entry: dict[str, Any], character: str
    ) -> tuple[int, list[dict[str, Any]]]:
        """Process evaluation response and return (bad_count, corrections)."""
        pass

    @abstractmethod
    def get_characters_from_entry(self, entry: dict[str, Any]) -> list[str]:
        """Extract character names from conversation entry."""
        pass

    def calculate_cost(self, usage_metadata: dict[str, Any]) -> float:
        """Calculate cost from usage metadata."""
        cache_tokens = usage_metadata.get("input_token_details", {}).get(
            "cache_read", 0
        )
        input_tokens = usage_metadata.get("input_tokens", 0) - cache_tokens
        output_tokens = usage_metadata.get("output_tokens", 0)
        return ((input_tokens * 2 + output_tokens * 8 + cache_tokens * 0.5) / 1e6) * 150

    def process_entry(
        self,
        entry: dict[str, Any],
        chain: Any,
        agg_lock: threading.Lock,
        num_bad_total: list[int],
        cost_total: list[float],
    ) -> None:
        """Process a single conversation entry."""
        session_id = entry["id"]

        # Check if already processed
        progress = self.progress_tracker.get_progress(session_id)
        if progress:
            with agg_lock:
                num_bad_total[0] += progress[1]
                cost_total[0] += progress[2]
            return

        messages_str = self.format_messages(entry["conversations"])
        characters = self.get_characters_from_entry(entry)

        local_bad, local_cost = 0, 0.0

        for character in characters:
            try:
                data = {
                    "messages": messages_str,
                    "role": character,
                    "role_instruction": self.get_character_profile(character),
                    "scene_instruction": self.get_scene_instruction(entry),
                }
                logger.info(f"PROMPT: [{session_id}] {data}")
                result = chain.invoke(data)

                response = result["parsed"]
                usage = result["raw"].usage_metadata or {}
                local_cost += self.calculate_cost(usage)

                bad_count, corrections = self.process_response(
                    response, entry, character
                )
                local_bad += bad_count

                # Save corrections to files
                self.save_corrections(corrections, entry, session_id)

            except Exception as e:
                logger.error(f"[{session_id}] invoke error: {e}")
                continue

        logger.info(f"Session {session_id}: issues={local_bad}, cost={local_cost:.2f}")

        try:
            self.progress_tracker.update_progress(session_id, local_bad, local_cost)
        except sqlite3.Error as e:
            logger.error(f"Database error while saving progress: {e}")
            raise

        with agg_lock:
            num_bad_total[0] += local_bad
            cost_total[0] += local_cost

    @abstractmethod
    def get_character_profile(self, character: str) -> str:
        """Get character profile for evaluation."""
        pass

    @abstractmethod
    def get_scene_instruction(self, entry: dict[str, Any]) -> str:
        """Get scene instruction for evaluation."""
        pass

    @abstractmethod
    def save_corrections(
        self, corrections: list[dict[str, Any]], entry: dict[str, Any], session_id: str
    ) -> None:
        """Save corrections to output files."""
        pass

    def run_evaluation(
        self,
        dataset: Any,
        max_items: int | None = None,
        max_workers: int = 8,
        use_azure: bool = False,
    ) -> None:
        """Run parallel evaluation on dataset."""
        logger.info(f"Logging to {self.output_dir}")

        # Setup parser and prompt
        prompt_tpl = ChatPromptTemplate.from_template(self.config.prompt_template)

        # Thread-safe aggregation variables
        agg_lock = threading.Lock()
        num_bad_total = [0]
        cost_total = [0.0]

        def process_entry_wrapper(entry: dict[str, Any]) -> None:
            """Wrapper to create per-thread LLM instance."""
            llm = LLMFactory.create_llm(self.config.schema, use_azure)
            chain = prompt_tpl | llm
            self.process_entry(entry, chain, agg_lock, num_bad_total, cost_total)

        # Get unprocessed entries
        entries_todo = self.progress_tracker.get_unprocessed_sessions(list(dataset))

        # Run parallel processing
        futures = []
        cancelled = False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry in entries_todo:
                futures.append(executor.submit(process_entry_wrapper, entry))

            logger.info(f"Processing {len(futures)} entries")
            for fut in tqdm(as_completed(futures), total=len(futures)):
                if (
                    max_items is not None
                    and num_bad_total[0] > max_items
                    and not cancelled
                ):
                    logger.info(
                        f"サンプル数が上限 {max_items} を超えたため残ジョブをキャンセルします…"
                    )
                    cancelled = True
                    for f in futures:
                        f.cancel()
                    break

        logger.success(
            f"Finished! Issues found: {num_bad_total[0]}, Total cost: {cost_total[0]:.2f} yen"
        )
