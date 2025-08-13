"""Conversation quality evaluation using the new evaluation framework."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, Field

from spalign.evaluation.base import BaseEvaluator, EvaluationConfig
from spalign.projects.happyrat.happyrat import CHARACTERS
from spalign.utils import extract_next_speaker


class CharacterUtteranceCorrection(BaseModel):
    """Detects character utterances that have conversation quality issues and suggests improved alternatives."""

    reason: str = Field(
        description="Specific conversation quality issue identified and detailed explanation of the correction direction. Include what makes the utterance problematic and how the suggested improvement addresses those issues. Up to ~200 characters."
    )
    index: int | None = Field(
        description="The index of the specific character utterance with quality issues. If unclear, set to None."
    )
    issue_type: Literal[
        "repetition", "inconsistency", "unnatural_grammar", "vague_response"
    ] = Field(
        description="The type of quality issue identified. If unclear, set to None."
    )
    correction: str | None = Field(
        description="If the utterance has quality issues, rewrite it to improve conversation quality while maintaining character consistency. Avoid parentheses or meta-comments."
    )
    emotion: Literal[
        "neutral",
        "happy",
        "sad",
        "angry",
        "scared",
        "disgusted",
        "surprised",
        "friendly",
        "interested",
        "confused",
        "laughing",
        "crying",
    ] = Field(description="the emotion of the corrected utterance")


class ConversationEvaluation(BaseModel):
    score: int = Field(
        description="""Score indicating the conversation quality based on LLM performance.
[Scoring Criteria]
1: Poor conversation quality with multiple severe issues (repetitive, inconsistent, unnatural grammar, lack of specificity).
2: Below average quality with noticeable issues that detract from conversation flow.
3: Average quality with some minor issues but generally acceptable conversation.
4: Good quality conversation with natural flow and appropriate character consistency.
5: Excellent conversation quality with natural dialogue, character consistency, and engaging specificity."""
    )
    corrections: list[CharacterUtteranceCorrection] = Field(
        description="List of character utterances that have quality issues requiring correction. Only required if score is 3 or below; otherwise, leave empty."
    )


QUALITY_PROMPT_TEMPLATE = """\
### Instructions
- Below is a conversation between a user and in-app AI characters.
- Evaluate the conversation quality based on LLM performance issues.
- Identify character utterances with the following quality problems:
  1. Unnatural repetition of similar content
    - The character repeats the same phrases or ideas in a way that feels robotic and adds no new information to the conversation.
    - 
  2. Inconsistency with character's past context or personality
    Examples:
        - 
        -
    - 会話に登場している他のキャラクターを忘れる。
    - 一人称を間違える、またはキャラクターに合わない不適切な口調（例：「みなさまにおかれましては」のような過度に丁寧な表現）を使う。
    - 直前の文脈を理解できていない応答（例：「あなたの話をしていたのよ」→「え、何の話？」）。
    - 発言や状況の矛盾（例：お店で寝てしまう）。
    - 会話の中で不自然に時間が進んでしまう。
  3. Grammatically unnatural or awkward expressions
  4. Lack of specificity or vague responses
- Propose improved replacements that maintain character consistency and improve conversation flow.

### Output Format
{format_instructions}

### Notes
- **Focus on conversation quality, not user satisfaction.**
- **Identify specific issues: repetition, inconsistency, unnatural grammar, lack of specificity.**
- **If suggesting corrections, ensure the revised line improves conversation quality while staying true to the character.**
- **Consider the entire conversation context when evaluating consistency.**

### Character Profile
{role_instruction}

### Conversation Log
{messages}
"""


class QualityEvaluator(BaseEvaluator[ConversationEvaluation]):
    """Evaluator for conversation quality assessment."""

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        try:
            for i, m in enumerate(messages):
                try:
                    if not isinstance(m, dict):
                        logger.error(f"Message {i} is not a dict: {type(m)} - {m}")
                        lines.append(f"(index={i}) ERROR: Invalid message format")
                        continue

                    if "name" not in m:
                        logger.error(f"Message {i} missing 'name' field: {m.keys()}")
                        lines.append(f"(index={i}) ERROR: Missing name field")
                        continue

                    if "utterance" not in m:
                        logger.error(
                            f"Message {i} missing 'utterance' field: {m.keys()}"
                        )
                        lines.append(f"(index={i}) ERROR: Missing utterance field")
                        continue

                    if m["name"] in CHARACTERS.keys():
                        speaker = m["name"]
                    else:
                        speaker = "ユーザー"
                    lines.append(f"(index={i}) {speaker}: {m['utterance']}")

                except Exception as e:
                    logger.error(f"Error processing message {i}: {e}")
                    import traceback

                    logger.error(
                        f"Message processing traceback: {traceback.format_exc()}"
                    )
                    lines.append(f"(index={i}) ERROR: {str(e)}")

        except Exception as e:
            logger.error(f"Fatal error in format_messages: {e}")
            import traceback

            logger.error(f"Format messages traceback: {traceback.format_exc()}")
            return f"ERROR: Could not format messages - {str(e)}"

        return "\n".join(lines)

    def get_characters_from_entry(self, entry: dict[str, Any]) -> list[str]:
        characters_set: set[str] = set()
        try:
            conversations = entry.get("conversations", [])
            if not conversations:
                logger.warning("Entry has no conversations")
                return []

            for i, c in enumerate(conversations):
                try:
                    if not isinstance(c, dict):
                        logger.error(f"Conversation {i} is not a dict: {type(c)}")
                        continue

                    if "name" not in c:
                        logger.error(
                            f"Conversation {i} missing 'name' field: {c.keys()}"
                        )
                        continue

                    if c["name"] in CHARACTERS.keys():
                        characters_set.add(c["name"])
                        logger.debug(f"Added character: {c['name']}")

                except Exception as e:
                    logger.error(
                        f"Error processing conversation {i} in get_characters_from_entry: {e}"
                    )
                    import traceback

                    logger.error(
                        f"Character extraction traceback: {traceback.format_exc()}"
                    )

        except Exception as e:
            logger.error(f"Fatal error in get_characters_from_entry: {e}")
            import traceback

            logger.error(
                f"Character extraction fatal traceback: {traceback.format_exc()}"
            )

        result = list(characters_set)
        logger.debug(f"Extracted characters: {result}")
        return result

    def get_character_profile(self, character: str) -> str:
        return CHARACTERS[character]["profile"]

    def process_response(
        self, response: ConversationEvaluation, entry: dict[str, Any], character: str
    ) -> tuple[int, list[dict[str, Any]]]:
        corrections = []
        conversations_length = len(entry["conversations"])

        for c in response.corrections:
            if c.index is None:
                continue

            # Validate index is within valid range
            if c.index >= conversations_length:
                logger.warning(
                    f"Skipping correction with invalid index {c.index} "
                    f"(conversations length: {conversations_length})"
                )
                continue

            corrections.append(
                {
                    "index": c.index,
                    "reason": c.reason,
                    "chosen": c.correction,
                    "emotion": c.emotion,
                    "issue_type": c.issue_type,
                    "score": response.score,
                }
            )
        return len(corrections), corrections

    def save_corrections(
        self, corrections: list[dict[str, Any]], entry: dict[str, Any], session_id: str
    ) -> None:
        try:
            logger.debug(
                f"[{session_id}] Starting save_corrections with {len(corrections)} corrections"
            )
            msgs: list[dict[str, Any]] = []
            conversations = entry.get("conversations", [])

            if not conversations:
                logger.error(f"[{session_id}] No conversations found in entry")
                return

            logger.debug(f"[{session_id}] Conversations length: {len(conversations)}")

            # Validate correction indices before processing
            for i, correction in enumerate(corrections):
                logger.debug(
                    f"[{session_id}] Processing correction {i}: index={correction.get('index')}"
                )
                if correction.get("index") is None:
                    logger.warning(f"[{session_id}] Correction {i} has None index")
                    continue
                if correction["index"] >= len(conversations):
                    logger.warning(
                        f"[{session_id}] Invalid correction index {correction['index']} "
                        f"(conversations length: {len(conversations)})"
                    )

            for i, m in enumerate(conversations):
                try:
                    logger.debug(
                        f"[{session_id}] Processing conversation {i}/{len(conversations)}"
                    )

                    # Validate conversation message structure
                    if not isinstance(m, dict):
                        logger.error(
                            f"[{session_id}] Conversation {i} is not a dict: {type(m)}"
                        )
                        msgs.append(m)
                        continue

                    if "name" not in m:
                        logger.error(
                            f"[{session_id}] Conversation {i} missing 'name' field: {m.keys()}"
                        )
                        msgs.append(m)
                        continue

                    if "utterance" not in m:
                        logger.error(
                            f"[{session_id}] Conversation {i} missing 'utterance' field: {m.keys()}"
                        )
                        msgs.append(m)
                        continue

                    for correction in corrections:
                        try:
                            # Check if correction index is valid
                            if correction.get("index") == i and correction[
                                "index"
                            ] < len(conversations):
                                logger.debug(
                                    f"[{session_id}] Applying correction to message {i}"
                                )

                                # Check if character exists in CHARACTERS
                                if m["name"] not in CHARACTERS:
                                    logger.error(
                                        f"[{session_id}] Unknown character: {m['name']}"
                                    )
                                    continue

                                next_speaker = extract_next_speaker(m["utterance"])
                                chosen = (
                                    CHARACTERS[m["name"]]["tag"]
                                    + f"[emotion:{correction['emotion']}]"
                                    + correction["chosen"]
                                    + f"[next:{next_speaker}]"
                                )
                                data = {
                                    "scene": entry.get("scenario", ""),
                                    "messages": msgs.copy(),
                                    "reason": correction["reason"],
                                    "chosen": chosen,
                                    "rejected": m["utterance"],
                                    "issue_type": correction["issue_type"],
                                    "score": correction["score"],
                                    "speaker": m["name"],
                                }
                                sample_path = (
                                    self.output_dir
                                    / f"{session_id}={correction['index']:02d}.json"
                                )

                                try:
                                    with open(sample_path, "w") as f_json:
                                        json.dump(
                                            data,
                                            f_json,
                                            indent=2,
                                            ensure_ascii=False,
                                            cls=self.DateTimeEncoder,
                                        )
                                    logger.debug(
                                        f"[{session_id}] Saved correction to {sample_path}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"[{session_id}] Failed to save correction file {sample_path}: {e}"
                                    )

                        except Exception as e:
                            logger.error(
                                f"[{session_id}] Error processing correction {correction}: {e}"
                            )
                            import traceback

                            logger.error(
                                f"[{session_id}] Correction error traceback: {traceback.format_exc()}"
                            )

                except Exception as e:
                    logger.error(
                        f"[{session_id}] Error processing conversation {i}: {e}"
                    )
                    import traceback

                    logger.error(
                        f"[{session_id}] Conversation error traceback: {traceback.format_exc()}"
                    )
                finally:
                    msgs.append(m)

        except Exception as e:
            logger.error(f"[{session_id}] Fatal error in save_corrections: {e}")
            import traceback

            logger.error(
                f"[{session_id}] Fatal error traceback: {traceback.format_exc()}"
            )
            raise

    @property
    def DateTimeEncoder(self) -> type:
        """Access to DateTimeEncoder for compatibility."""
        from spalign.evaluation.base import DateTimeEncoder

        return DateTimeEncoder


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("run_name", type=str)
        parser.add_argument("-n", "--max_items", type=int, default=1000)
        parser.add_argument("-w", "--max_workers", type=int, default=8)
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable debug logging"
        )
        args = parser.parse_args()

        # Configure logging level
        if args.verbose:
            logger.remove()
            logger.add(lambda msg: print(msg, end=""), level="DEBUG")

        logger.info(f"Starting evaluation for run: {args.run_name}")
        logger.info(f"Max items: {args.max_items}, Max workers: {args.max_workers}")
        logger.info(f"Debug logging: {'enabled' if args.verbose else 'disabled'}")

        # Setup configuration
        config = EvaluationConfig(
            schema=ConversationEvaluation,
            prompt_template=QUALITY_PROMPT_TEMPLATE,
            output_dir="evaluation/illogical/jsons",
            table_suffix="_quality",
        )

        # Create evaluator
        evaluator = QualityEvaluator(config, args.run_name)

        # Load dataset
        log_dir = Path(f"{os.environ['RESULTS_DIR']}/{args.run_name}")
        logger.info(f"Looking for data in: {log_dir}")

        try:
            if (log_dir / "conversations").exists():
                logger.info("Loading from conversations directory")
                dataset = load_dataset(
                    "json",
                    data_files=str(log_dir / "conversations" / "*.json"),
                    split=f"train[:{args.max_items}]",
                )
            else:
                logger.info("Loading from conversations.bak.jsonl")
                dataset = load_dataset(
                    "json",
                    data_files=str(log_dir / "conversations.bak.jsonl"),
                    split=f"train[:{args.max_items}]",
                )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            import traceback

            logger.error(f"Dataset loading traceback: {traceback.format_exc()}")
            raise

        logger.info(f"Dataset loaded with {len(dataset)} entries")

        try:
            dataset = dataset.filter(lambda x: len(x.get("conversations", [])) >= 5)
            logger.info(
                f"Filtered dataset to {len(dataset)} entries (conversations >= 5)"
            )
        except Exception as e:
            logger.error(f"Failed to filter dataset: {e}")
            import traceback

            logger.error(f"Dataset filtering traceback: {traceback.format_exc()}")
            raise

        # Run evaluation
        try:
            evaluator.run_evaluation(
                dataset=dataset,
                max_items=args.max_items,
                max_workers=args.max_workers,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback

            logger.error(f"Evaluation traceback: {traceback.format_exc()}")
            raise

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback

        logger.error(f"Main function traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
