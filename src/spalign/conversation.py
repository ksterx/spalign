"""Core conversation generation logic."""

from __future__ import annotations

import asyncio
import random
import uuid
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer
from vllm import LLM

from spalign.personas import NORMAL_PERSONAS, PERSONAS
from spalign.projects.happyrat.happyrat import CHARACTERS
from spalign.utils import parse_utterance

from .batcher import VLLMBatcher
from .database import get_scenario_hash, mark_completed, mark_failed
from .models import PersonaParams
from .persona import PersonaGenerator
from .utils import history_to_msgs, parse_role


class ConversationGenerator:
    """Generate conversations using vLLM and persona models."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 8,
        max_num_seqs: int = 500,
        max_num_batched_tokens: int = 8192,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.95,
    ):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.batcher = VLLMBatcher(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.persona_generator = PersonaGenerator()

    def start_batcher(self):
        """Start the vLLM batcher."""
        self.batcher.start()

    async def generate_conversation(
        self,
        data: dict[str, Any],
        gpt_semaphore: asyncio.Semaphore,
        n_turns: int,
        persona_type: str,
        db_file: str,
        conversations_dir: Path,
    ) -> dict[str, Any]:
        """Generate a single conversation asynchronously."""
        scenario_hash = get_scenario_hash(data)

        try:
            # Validate input data structure
            if not isinstance(data, dict):
                raise ValueError(f"Data must be a dict, got {type(data)}")

            if "scenario" not in data:
                raise ValueError("Data missing required 'scenario' field")

            if "character_list" not in data:
                raise ValueError("Data missing required 'character_list' field")

            scenario = data["scenario"]
            characters = data["character_list"]

            # Validate characters list
            if not isinstance(characters, list):
                raise ValueError(
                    f"character_list must be a list, got {type(characters)}"
                )

            if len(characters) == 0:
                raise ValueError("character_list cannot be empty")

            logger.debug(
                f"Starting conversation generation with {len(characters)} characters: {characters}"
            )

            # Persona selection
            if persona_type == "original":
                p_name = random.choice(list(PERSONAS.keys()))
                info = PERSONAS[p_name]
            elif persona_type == "normal":
                p_name = random.choice(list(NORMAL_PERSONAS.keys()))
                info = NORMAL_PERSONAS[p_name]
            elif persona_type == "dataset":  # dataset mode (randomized)
                ds_row = random.choice(
                    load_dataset(
                        "Spiral-AI/Synthesized-Persona-20250103", split="train"
                    )
                )
                p_name = ds_row["new_persona_name"]
                base = max(
                    0.01, random.gauss(mu=0.5, sigma=0.2)
                )  # ユーザーの介入度合いが少なめのため、muを小さめに設定
                info = {
                    "profile": ds_row["new_persona"],
                    "base_prob": base,
                    "max_prob": base * 1.5,
                    "decay": random.uniform(0.3, 0.8),
                    "recovery_step": random.uniform(0.05, 0.3),
                }
            elif persona_type == "metadata":
                persona = data["metadata"]["users"][0]
                p_name = persona["name"]
                base = max(
                    0.01, random.gauss(mu=0.5, sigma=0.2)
                )  # ユーザーの介入度合いが少なめのため、muを小さめに設定
                info = {
                    "profile": persona["profile"],
                    "base_prob": base,
                    "max_prob": base * 1.5,
                    "decay": random.uniform(0.3, 0.8),
                    "recovery_step": random.uniform(0.05, 0.3),
                }
            params = PersonaParams.model_validate(info)
            p_prob = params.base_prob

            roles = parse_role(characters, p_name)
            idx_map = {v: k for k, v in roles.items()}

            # Validate roles dictionary
            if not roles:
                raise ValueError("Failed to parse roles from characters")

            logger.debug(f"Parsed roles: {roles}")
            logger.debug(f"Index mapping: {idx_map}")

            histories: list[dict[str, Any]] = []

            def format_hist(msgs, speaker, scenario):
                new_msgs = msgs.copy()
                new_msgs.insert(0, {"role": "assistant_name", "content": speaker})
                new_msgs.insert(1, {"role": "system", "content": scenario})
                if not new_msgs:
                    return "<s><|start_header_id|>assistant<|end_header_id|>\n\n"
                prompt = self.tokenizer.apply_chat_template(
                    new_msgs, tokenize=False, add_generation_prompt=True
                )
                prompt += CHARACTERS.get("speaker", {}).get("tag", "")
                return prompt

            for t in range(n_turns):
                try:
                    logger.debug(f"Turn {t}/{n_turns}: Starting speaker selection")

                    # Speaker selection with bounds checking
                    if t == 0:
                        # Ensure roles[0] exists
                        if 0 not in roles:
                            raise ValueError(
                                f"Role index 0 not found in roles: {roles}"
                            )
                        speaker = roles[0]
                        logger.debug(f"Turn {t}: Initial speaker selected: {speaker}")
                    else:
                        others = [n for n in roles.values() if n != p_name]
                        logger.debug(f"Turn {t}: Available other speakers: {others}")

                        if random.random() < p_prob * (
                            len(characters) + 1
                        ):  # キャラクター数に応じて確率を調整。
                            speaker = p_name
                            logger.debug(
                                f"Turn {t}: Persona speaker selected: {speaker}"
                            )
                        else:
                            # Safe access to histories
                            if len(histories) > 0:
                                last_history = histories[-1]
                                if (
                                    isinstance(last_history, dict)
                                    and "next_speaker" in last_history
                                    and last_history["next_speaker"] == "myself"
                                ):
                                    if "name" in last_history:
                                        speaker = last_history["name"]
                                        logger.debug(
                                            f"Turn {t}: 'myself' speaker selected: {speaker}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Turn {t}: Last history missing 'name' field"
                                        )
                                        speaker = (
                                            random.choice(others) if others else p_name
                                        )
                                else:
                                    if others:
                                        if random.random() < 0.9:
                                            speaker = random.choice(others)
                                        else:
                                            if "name" in last_history:
                                                speaker = last_history["name"]
                                            else:
                                                speaker = random.choice(others)
                                        logger.debug(
                                            f"Turn {t}: Random/repeat speaker selected: {speaker}"
                                        )
                                    else:
                                        speaker = p_name
                                        logger.warning(
                                            f"Turn {t}: No other speakers available, using persona"
                                        )
                            else:
                                # Fallback when histories is empty
                                speaker = random.choice(others) if others else p_name
                                logger.debug(
                                    f"Turn {t}: Fallback speaker selected: {speaker}"
                                )

                    logger.debug(f"Turn {t}: Final speaker: {speaker}")

                except Exception as e:
                    logger.error(f"Turn {t}: Error in speaker selection: {e}")
                    import traceback

                    logger.error(
                        f"Turn {t}: Speaker selection traceback: {traceback.format_exc()}"
                    )
                    # Fallback to first role or persona
                    speaker = roles.get(0, p_name)
                    logger.warning(f"Turn {t}: Using fallback speaker: {speaker}")

                # Build prompt
                hist_msgs = history_to_msgs(histories, speaker, idx_map)
                prompt = format_hist(hist_msgs, speaker, scenario)

                # Update probabilities (before await!)
                if speaker == p_name:
                    p_prob = max(params.base_prob, p_prob * params.decay)
                else:
                    p_prob = min(params.max_prob, p_prob + params.recovery_step)

                # Generation
                if speaker == p_name:
                    # GPT persona
                    try:
                        text = await self.persona_generator.generate(
                            prompt, params.profile, p_name, gpt_semaphore
                        )
                    except Exception as e:
                        # fallback to empty utterance on error
                        print(f"Error generating persona: {e}")
                        return {}
                    emo = speaker_role = nxt = None
                else:
                    # vLLM path
                    fut = await self.batcher.put(prompt)
                    text = await fut
                    # text = (
                    #     CHARACTERS.get(speaker, {}).get("tag", "") + text
                    # )  # Add tag if not present

                    # Skip if generation failed (empty result)
                    if not text:
                        print(
                            f"Skipping turn {t} for {speaker} due to generation failure"
                        )
                        continue

                    # Parse metadata
                    try:
                        speaker_role, emo, content, nxt = parse_utterance(text)
                    except Exception:
                        speaker_role = emo = nxt = None

                # Record with validation
                try:
                    history_entry = {
                        "index": t,
                        "name": speaker,
                        "utterance": text if text else "",
                        "emotion": emo,
                        "speaker": speaker_role,
                        "next_speaker": nxt,
                    }
                    histories.append(history_entry)
                    logger.debug(f"Turn {t}: Added history entry for {speaker}")
                except Exception as e:
                    logger.error(f"Turn {t}: Failed to record history: {e}")
                    import traceback

                    logger.error(
                        f"Turn {t}: History recording traceback: {traceback.format_exc()}"
                    )
                    # Continue with next turn even if recording failed

            # Pack result with validation
            if not histories:
                logger.warning("No conversation history generated")
                raise ValueError("Failed to generate any conversation turns")

            logger.info(f"Generated {len(histories)} conversation turns")

            result = data.copy()
            result.update(
                {
                    "conversations": histories,
                    "conversation_gen_model": self.model_name,
                    "id": str(uuid.uuid4()),
                }
            )

            # Mark as completed in database
            mark_completed(scenario_hash, result, db_file)

            # Save individual conversation file immediately
            self._save_individual_conversation(result, conversations_dir)

            logger.info(
                f"Successfully completed conversation generation with {len(histories)} turns"
            )
            return result

        except Exception as e:
            logger.error(f"Fatal error in conversation generation: {e}")
            import traceback

            logger.error(f"Conversation generation traceback: {traceback.format_exc()}")
            # Mark as failed in database
            mark_failed(scenario_hash, str(e), db_file)
            raise

    def _save_individual_conversation(
        self, result: dict[str, Any], conversations_dir: Path
    ):
        """Save individual conversation as JSON file."""
        import json

        conversation_id = result["id"]
        file_path = conversations_dir / f"{conversation_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
