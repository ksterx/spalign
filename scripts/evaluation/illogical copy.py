"""Conversation quality evaluation using the new evaluation framework."""

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
### 指示
- 以下は、ユーザーとアプリ内のAIキャラクターとの会話です。
- LLMのパフォーマンス問題に基づき、会話の品質を評価してください。
- 以下の品質問題を特定し、キャラクターの発言を評価してください：
  1. 不自然な内容の繰り返し（例：「星影」「紅蓮華」のような特定の単語の繰り返し）。
  2. これまでの文脈やキャラクター設定との矛盾。
     - 会話に登場している他のキャラクターを忘れる。
     - 一人称を間違える、またはキャラクターに合わない不適切な口調（例：「みなさまにおかれましては」のような過度に丁寧な表現）を使う。
     - 直前の文脈を理解できていない応答（例：「あなたの話をしていたのよ」→「え、何の話？」）。
     - 発言や状況の矛盾（例：お店で寝てしまう）。
     - 会話の中で不自然に時間が進んでしまう。
  3. 文法的に不自然、またはぎこちない表現。
  4. 具体性に欠ける、または曖昧な応答。
  5. 会話が停滞する、または新しい話題が出ても前の話に引き戻してしまうような応答（浅瀬チャプチャプ問題）。
- キャラクター設定を維持し、会話の流れを改善するような修正案を提案してください。

### 出力フォーマット
{format_instructions}

### 注意事項
- **ユーザーの満足度ではなく、会話の品質に焦点を当ててください。**
- **具体的な問題（繰り返し、矛盾、不自然な文法、具体性の欠如など）を特定してください。**
- **修正を提案する場合、修正案がキャラクター性を維持しつつ、会話の品質を向上させるものであることを確認してください。**
- **矛盾を評価する際は、会話全体の文脈を考慮してください。**

### キャラクター設定
{role_instruction}

### 会話ログ
{messages}
"""


class QualityEvaluator(BaseEvaluator[ConversationEvaluation]):
    """Evaluator for conversation quality assessment."""

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
        characters_set: set[str] = set()
        for c in entry["conversations"]:
            if c["name"] in CHARACTERS.keys():
                characters_set.add(c["name"])
        return list(characters_set)

    def get_character_profile(self, character: str) -> str:
        return CHARACTERS[character]["profile"]

    def process_response(
        self, response: ConversationEvaluation, entry: dict[str, Any], character: str
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
                    "issue_type": c.issue_type,
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
                        "issue_type": correction["issue_type"],
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
    parser.add_argument("-n", "--max_items", type=int, default=1000)
    parser.add_argument("-w", "--max_workers", type=int, default=8)
    args = parser.parse_args()

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
    if (log_dir / "conversations").exists():
        dataset = load_dataset(
            "json",
            data_files=str(log_dir / "conversations" / "*.json"),
            split=f"train[:{args.max_items}]",
        )
    else:
        dataset = load_dataset(
            "json",
            data_files=str(log_dir / "conversations.bak.jsonl"),
            split=f"train[:{args.max_items}]",
        )
    dataset = dataset.filter(lambda x: len(x["conversations"]) >= 5)

    # Run evaluation
    evaluator.run_evaluation(
        dataset=dataset,
        max_items=args.max_items,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
