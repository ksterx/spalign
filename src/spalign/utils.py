"""Utility functions for conversation generation."""

from __future__ import annotations

import re
from typing import Any


def strip_tags(text: str) -> str:
    """Remove [tag] patterns used for emotion etc."""
    return re.sub(r"\[.*?\]", "", text)


def parse_role(characters: list[str], persona_name: str) -> dict[int, str]:
    """Map role index → name (characters + persona speaker)."""
    roles = {i: n for i, n in enumerate(characters)}
    roles[len(characters)] = persona_name
    return roles


def history_to_msgs(hist: list[dict[str, Any]], speaker: str, idx_map: dict[str, int]):
    """Convert history into Llama‑3 chat‑template friendly list."""
    msgs: list[dict[str, str]] = []
    for h in hist:
        idx = idx_map[h["name"]]
        if h["name"] != speaker:
            msgs.append(
                {"role": f"user_{idx:02d}", "content": strip_tags(h["utterance"])}
            )
        else:
            msgs.append({"role": "assistant", "content": h["utterance"]})
    return msgs


def parse_utterance(text: str):
    speaker = None
    emotion = None
    next_speaker = "[next:user_00]"  # デフォルト

    # 全ての [xxx] タグを抽出
    tags = list(re.finditer(r"\[([^\]]+)\]", text))

    for tag in tags:
        full_tag = tag.group(0)  # 例: "[cammy]"
        inner = tag.group(1)  # 例: "cammy"

        if inner.startswith("emotion:") and emotion is None:
            emotion = full_tag
        elif inner.startswith("next:") and next_speaker == "[next:user_00]":
            next_speaker = full_tag
        elif speaker is None:
            speaker = full_tag

    # remove tags
    cleaned_text = re.sub(r"\[[^]]+\]", "", text)

    return speaker, emotion, cleaned_text, next_speaker


def extract_next_speaker(text: str) -> str:
    match = re.search(r"\[next:(.*?)\]", text)
    if match:
        return match.group(1).strip()
    return "user_00"
