"""User satisfaction evaluation using the new evaluation framework."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from pydantic import BaseModel, Field

from spalign.evaluation.base import BaseEvaluator, EvaluationConfig
from spalign.projects.happyrat.happyrat import CHARACTERS
from spalign.utils import extract_next_speaker


class CharacterUtteranceCorrection(BaseModel):
    """Detects character utterances that may have caused dissatisfaction and suggests alternative lines that better match the character."""

    reason: str = Field(
        description="The user's utterance or reaction indicating dissatisfaction, along with the reason. Up to ~150 characters."
    )
    index: int | None = Field(
        description="The index of the specific character utterance that likely caused dissatisfaction. If unclear, set to None."
    )
    correction: str | None = Field(
        description="If the utterance is problematic, rewrite it based on the character's profile and previous context—independently from the original line. Avoid parentheses or meta-comments."
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


class ConversationSatisfactionEvaluation(BaseModel):
    score: int = Field(
        description="""Score indicating the user's satisfaction at the end of the conversation.
[Scoring Criteria]
1: The user is highly dissatisfied and unlikely to use the app again.
2: The user found the conversation boring or unpleasant and may abandon the app.
3: The user felt neutral and may occasionally return.
4: The user felt some enjoyment and is likely to keep using the app.
5: The user greatly enjoyed the conversation and may even consider paying for the app."""
    )
    corrections: list[CharacterUtteranceCorrection] = Field(
        description="List of character utterances revised to better match their personality. Only required if score is 3 or below; otherwise, leave empty."
    )


SATISFACTION_PROMPT_TEMPLATE = """\
### Instructions
- Below is a conversation between a user and in-app AI characters.
- Your task is to estimate how satisfied the user was with the conversation.
- If the user appears dissatisfied:
- Identify any specific character utterance that seems to have caused the issue, and propose a replacement that aligns with the character's personality and prior context.
- If the dissatisfaction is more general (e.g., the conversation felt flat), you may skip specific corrections and provide only the score.

### Output Format
{format_instructions}

### Notes
- **Do not evaluate the character's personality or identity directly.**
- **If suggesting corrections, ensure the revised line is non-repetitive and true to the character's style, but you may ignore consistency with later utterances.**
- **Base your judgment solely on the user's utterances and reactions.**

### Character Profile
{role_instruction}

### Conversation Log
{messages}
"""


class SatisfactionEvaluator(BaseEvaluator[ConversationSatisfactionEvaluation]):
    """Evaluator for user satisfaction assessment."""

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for i, m in enumerate(messages):
            if m["name"] in CHARACTERS.keys():
                speaker = m["name"]
            else:
                speaker = "ユーザー"
            lines.append(f"(index={i}) {speaker}: {m['utterance']}")
        return "\n".join(lines)

    def get_characters_from_entry(self, entry: dict[str, Any]) -> list[str]:
        characters: set[str] = set()
        for c in entry["conversations"]:
            if c["name"] in CHARACTERS.keys():
                characters.add(c["name"])
        return list(characters)

    def get_character_profile(self, character: str) -> str:
        return CHARACTERS[character]["profile"]

    def process_response(
        self,
        response: ConversationSatisfactionEvaluation,
        entry: dict[str, Any],
        character: str,
    ) -> tuple[int, list[dict[str, Any]]]:
        corrections = []
        for c in response.corrections:
            if c.index is None:
                continue
            corrections.append(
                {
                    "index": c.index,
                    "reason": c.reason,
                    "chosen": c.correction,
                    "emotion": c.emotion,
                    "score": response.score,
                }
            )
        return len(corrections), corrections

    def save_corrections(
        self, corrections: list[dict[str, Any]], entry: dict[str, Any], session_id: str
    ) -> None:
        msgs: list[dict[str, Any]] = []
        for i, m in enumerate(entry["conversations"]):
            for correction in corrections:
                if correction["index"] == i:
                    next_speaker = extract_next_speaker(m["utterance"])
                    chosen = (
                        CHARACTERS[m["name"]]["tag"]
                        + f"[emotion:{correction['emotion']}]"
                        + correction["chosen"]
                        + f"[next:{next_speaker}]"
                    )
                    data = {
                        "scene": entry["scenario"],
                        "messages": msgs,
                        "reason": correction["reason"],
                        "chosen": chosen,
                        "rejected": m["utterance"],
                        "score": correction["score"],
                        "speaker": m["name"],
                    }
                    sample_path = (
                        self.output_dir / f"{session_id}={correction['index']:02d}.json"
                    )
                    with open(sample_path, "w") as f_json:
                        json.dump(
                            data,
                            f_json,
                            indent=2,
                            ensure_ascii=False,
                            cls=self.DateTimeEncoder,
                        )
            msgs.append(m)

    @property
    def DateTimeEncoder(self) -> type:
        """Access to DateTimeEncoder for compatibility."""
        from spalign.evaluation.base import DateTimeEncoder

        return DateTimeEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("-m", "--max", type=int, default=1000)
    parser.add_argument("-w", "--max_workers", type=int, default=8)
    args = parser.parse_args()

    # Setup configuration
    config = EvaluationConfig(
        schema=ConversationSatisfactionEvaluation,
        prompt_template=SATISFACTION_PROMPT_TEMPLATE,
        output_dir="evaluation/satisfaction/jsons",
        table_suffix="_satisfaction",
    )

    # Create evaluator
    evaluator = SatisfactionEvaluator(config, args.run_name)

    # Load dataset
    log_dir = Path(f"{os.environ['RESULTS_DIR']}/{args.run_name}")
    if (log_dir / "conversations").exists():
        dataset = load_dataset(
            "json",
            data_files=str(log_dir / "conversations" / "*.json"),
            split=f"train[:{args.max}]",
        )
    else:
        dataset = load_dataset(
            "json",
            data_files=str(log_dir / "conversations.bak.jsonl"),
            split=f"train[:{args.max}]",
        )
    dataset = dataset.filter(lambda x: len(x["conversations"]) >= 5)

    # Run evaluation
    evaluator.run_evaluation(
        dataset=dataset,
        max_items=args.max,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
