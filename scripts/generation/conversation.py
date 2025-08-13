#!/usr/bin/env python3
"""
Async conversation synthesizer
==============================

CLI script for generating conversations using the spalign library.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from spalign.conversation import ConversationGenerator
from spalign.database import (
    get_completed_results,
    get_failed_scenarios,
    get_pending_scenarios,
    get_progress_stats,
    init_db,
    insert_pending_scenarios,
    reset_failed_to_pending,
)
from spalign.projects.happyrat.happyrat import CHARACTERS


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Run identifier – used in path")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", ""),
        help="Model name",
    )
    parser.add_argument(
        "-p",
        "--persona_type",
        type=str,
        choices=["original", "normal", "dataset"],
        default="original",
        help="Which persona set to sample from",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=40,
        help="Max concurrent conversation tasks (to limit RAM)",
    )
    parser.add_argument(
        "--gpt_concurrency",
        type=int,
        default=20,
        help="Simultaneous GPT‑4 requests allowed",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=40,
        help="Number of conversation turns to generate per scenario",
    )
    parser.add_argument(
        "-r",
        "--retry-failed",
        action="store_true",
        help="Retry only failed scenarios from previous runs",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Spiral-AI/Synthesized-Scenario-20250812",
        help="Scenario repo to use",
    )
    parser.add_argument(
        "--scenario_subset",
        type=str,
        default="default",
        help="Scenario subset to use",
    )
    return parser.parse_args()


def setup_directories(run_name: str) -> tuple[Path, Path, Path, str]:
    """Setup output directories and database file."""
    output_dir = Path(f"{os.environ['RESULTS_DIR']}/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations_dir = output_dir / "conversations"
    conversations_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    db_file = str(output_dir / "progress.db")

    return output_dir, conversations_dir, backup_dir, db_file


def create_backup_summary(backup_dir: Path, db_file: str) -> None:
    """Create a summary backup with all completed conversations."""
    all_results = get_completed_results(db_file)
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.json"
        with backup_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"バックアップ作成: {backup_file} ({len(all_results)} 件)")


def count_saved_conversations(conversations_dir: Path) -> int:
    """Count the number of saved conversation files."""
    return len(list(conversations_dir.glob("*.json")))


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup directories
    output_dir, conversations_dir, backup_dir, db_file = setup_directories(
        args.run_name
    )

    # Initialize database
    init_db(db_file)

    # Load dataset
    dataset = load_dataset(args.scenario, args.scenario_subset, split="train").filter(
        lambda x: set(CHARACTERS.keys()) == set(x["character_list"])
    )  # 使われていないキャラクターがあるので、ここでフィルタリング

    if args.retry_failed:
        # Retry failed scenarios mode
        print("失敗したシナリオの再試行モードです。")

        # Get failed scenarios before resetting
        failed_scenarios = get_failed_scenarios(dataset, db_file)
        print(f"失敗したシナリオ: {len(failed_scenarios)} 件")

        if len(failed_scenarios) == 0:
            print("失敗したシナリオはありません。")
            return

        # Reset failed scenarios to pending
        reset_count = reset_failed_to_pending(db_file)
        print(f"{reset_count} 件の失敗したシナリオを再試行対象にしました。")

        # Use failed scenarios as the target
        target_scenarios = failed_scenarios
    else:
        # Normal mode: insert all scenarios as pending (if not already exists)
        insert_pending_scenarios(dataset, db_file)

        # Get only pending scenarios
        target_scenarios = get_pending_scenarios(dataset, db_file)

    # Get progress statistics
    stats = get_progress_stats(db_file)
    print(f"進行状況: {stats}")
    print(f"作業対象: {len(target_scenarios)} 件")

    if len(target_scenarios) == 0:
        if args.retry_failed:
            print("再試行対象のシナリオがありません。")
        else:
            print("全ての作業が完了しています。")
            # Show final results
            saved_count = count_saved_conversations(conversations_dir)
            print(
                f"完了: {saved_count} 件の会話が {conversations_dir} に保存されています"
            )
        return

    # Initialize conversation generator
    print("エンジンを初期化中...")
    generator = ConversationGenerator(args.model)
    generator.start_batcher()

    gpt_sem = asyncio.Semaphore(args.gpt_concurrency)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def limited_task(d):
        async with semaphore:
            return await generator.generate_conversation(
                d, gpt_sem, args.turns, args.persona_type, db_file, conversations_dir
            )

    print(f"会話生成を開始: {len(target_scenarios)} 件")
    tasks = [limited_task(d) for d in target_scenarios]

    completed_count = 0
    for coro in asyncio.as_completed(tasks):
        try:
            await coro
            completed_count += 1

            # Progress update every 10 completions
            if completed_count % 10 == 0:
                current_stats = get_progress_stats(db_file)
                saved_count = count_saved_conversations(conversations_dir)
                print(
                    f"[進行状況] 完了: {current_stats.get('completed', 0)} 件, "
                    f"失敗: {current_stats.get('failed', 0)} 件, "
                    f"残り: {current_stats.get('pending', 0)} 件, "
                    f"保存済み: {saved_count} 件"
                )

                # Backup every 50 completions
                if completed_count % 50 == 0:
                    create_backup_summary(backup_dir, db_file)
                    print("バックアップ作成完了")

        except Exception as e:
            print(f"エラー: {e}")
            continue

    # Final export
    print("最終結果を確認中...")
    saved_count = count_saved_conversations(conversations_dir)
    create_backup_summary(backup_dir, db_file)

    final_stats = get_progress_stats(db_file)
    print(f"完了: {saved_count} 件の会話が {conversations_dir} に保存されました")
    print(f"最終統計: {final_stats}")


if __name__ == "__main__":
    asyncio.run(main())
