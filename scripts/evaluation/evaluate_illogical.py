"""Conversation quality evaluation using the new evaluation framework."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, Field

from spalign.evaluation.base import BaseEvaluator, EvaluationConfig, LLMConfig
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
You are an expert conversation quality evaluator. Based on the provided "Character Profile," "Scene Setting," and "Conversation Log," your task is to analyze the AI character's utterances. Strictly adhere to the "Conversation Quality Evaluation Criteria" below to identify issues stemming from the LLM's performance and propose concrete improvements to enhance the conversation quality.

### Character Profile
{role_instruction}

### Scene Setting
{scene_instruction}

### Conversation Log
{messages}

---

### Conversation Quality Evaluation Criteria

#### 1. Adherence to Character Profile
- **Top Priority:** This is the foundation for all other evaluations.
- **Issue:** Does the character's utterance align with the personality, speaking style, background, and first-person pronouns defined in the "Character Profile"?
- **Bad Example:** (Profile: A cool, blunt, lone-wolf character who isn't very interested in others.)
    - AI: "Wow! Being with everyone is so much fun! Let's make the best memories, tee-hee!"
- **Critique:** This cheerful and collaborative tone completely contradicts the character profile, constituting a critical failure in consistency.

#### 2. Depth and Specificity
- **Issue:** Does the conversation remain superficial, avoiding concrete topics and consisting only of pleasantries?
- **Bad Example:**
    - User: "I've been really into watching movies lately."
    - AI: "Oh, movies are nice, aren't they?"
- **Critique:** The AI only affirms the user's statement without taking any action to deepen the conversation. This is a classic example of a "shallow conversation."
- **Direction:** The AI should actively delve deeper into the user's statements by **1) asking specific questions** or **2) sharing its own related experiences or thoughts.**
    - **Good Example (Question):** "What genre of movies do you like? Have you seen anything interesting recently?"
    - **Good Example (Sharing):** "Movies are great. I'm into sci-fi myself, and the spaceship design in a film I saw the other day was amazing."

#### 3. Factuality and Appropriate Future Expressions
- **Issue:** Does the AI state unconfirmed future events as facts (hallucination) or confuse facts with desires?
- **Bad Example:**
    - User: "I hope we can come to this festival again next year."
    - AI: "Yeah. Next year's festival will be even better because they're adding 100 new types of fireworks."
- **Critique:** The claim that "100 new types of fireworks will be added" is an unconfirmed piece of information and is likely a hallucination.
- **Exception:** If the future event is described in the provided "Scene Setting," then it is acceptable for the AI to state it as a fact.
- **Direction:** For future topics not covered by the scene setting, the AI should express them as feelings or hopes, not as facts. (e.g., "That would be great! If they did that, it would be so exciting!")

#### 4. Fundamental Flaws in LLM's Conversational Performance
- **Issue:** Are there clear errors attributable to the LLM's basic conversational capabilities?
    1.  **Repetition:** Unnaturally repeating the same or very similar phrases/ideas in a short span.
    2.  **Forgetting Context:** Completely ignoring the immediate preceding context. (e.g., Responding "Who are you talking about?" right after a character was mentioned.)
    3.  **Forgetting Characters:** Forgetting the existence of a character who was part of the conversation.
    4.  **Unnatural Language:** Grammatical errors or using expressions that are overly stiff and unnatural for a friendly conversation. (e.g., "It is my turn to speak. Please proceed.")
    5.  **Contradicting the Situation:** Making statements that contradict the established setting (weather, location, time). (e.g., Suggesting "Let's go for a walk" right after a "The rain is so heavy" exchange.)

#### 5. Social and Situational Appropriateness
- **Issue:** Does the AI propose or attempt actions that are bizarre or grossly inappropriate for the situation or social norms?
- **Bad Example:**
    - User: "This museum is so nice. It has a quiet, solemn atmosphere."
    - AI: "I know, right? Let's play tag in here!"
- **Critique:** Proposing to "play tag" in a museum, a public space that requires quiet, is completely inappropriate. Similar examples include sleeping in a restaurant or singing loudly in a movie theater.
"""


class QualityEvaluator(BaseEvaluator[ConversationEvaluation]):
    """Evaluator for conversation quality assessment."""

    # We need to add scene_instruction to the format call in the base class.
    # For now, we add a placeholder method here.

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
        # logger.debug(f"Extracted characters: {result}")
        if not result:
            logger.warning(
                f"No characters found in entry: {entry.get('session_id', 'N/A')}"
            )
            return ["dummy_character"]  # Return a dummy to prevent index errors
        return result

    def get_character_profile(self, character: str) -> str:
        if character not in CHARACTERS:
            logger.warning(
                f"Character '{character}' not found in CHARACTERS dict. Returning empty profile."
            )
            return "No profile available for this character."
        return CHARACTERS[character]["profile"]

    def get_scene_instruction(self, entry: dict[str, Any]) -> str:
        return entry.get("scenario", "No specific scene setting provided.")

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
            msgs: list[dict[str, Any]] = []
            conversations = entry.get("conversations", [])

            if not conversations:
                logger.error(f"[{session_id}] No conversations found in entry")
                return

            # Validate correction indices before processing
            for i, correction in enumerate(corrections):
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
        parser.add_argument(
            "--temperature", type=float, default=0.3, help="LLM temperature parameter"
        )
        parser.add_argument(
            "--max-tokens", type=int, default=None, help="Maximum tokens for LLM output"
        )
        parser.add_argument(
            "--max-retries", type=int, default=2, help="Maximum retries for LLM calls"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gemini-2.5-pro",
            help="Model name for LLM (e.g., gemini-2.5-pro)",
        )
        parser.add_argument(
            "--use-azure",
            action="store_true",
            help="Use Azure OpenAI instead of Gemini",
        )
        args = parser.parse_args()

        # Configure logging level
        if args.verbose:
            logger.remove()
            logger.add(lambda msg: print(msg, end=""), level="DEBUG")

        logger.info(f"Starting evaluation for run: {args.run_name}")
        logger.info(f"Max items: {args.max_items}, Max workers: {args.max_workers}")
        logger.info(f"Debug logging: {'enabled' if args.verbose else 'disabled'}")

        # Setup LLM configuration
        llm_config = LLMConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            model=args.model,
        )

        # Setup configuration
        config = EvaluationConfig(
            schema=ConversationEvaluation,
            prompt_template=QUALITY_PROMPT_TEMPLATE,
            output_dir="evaluation/illogical",
            table_suffix="_quality",
            llm_config=llm_config,
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
                model=args.model,
                dataset=dataset,
                max_items=args.max_items,
                max_workers=args.max_workers,
                use_azure=args.use_azure,
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
