"""Persona generation using OpenAI GPT models."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import PersonaResponse


class PersonaGenerator:
    """Generate persona responses using OpenAI GPT models."""

    def __init__(
        self,
        language: Literal["Japanese", "English"],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, api_key=os.getenv("GEMINI_API_KEY")
        ).with_structured_output(PersonaResponse)
        self.prompt = ChatPromptTemplate.from_template(
            template="""
You embody the persona '{persona_name}'. You will generate a response as the persona in the messaging app.

### Profile
{profile}

### Current conversation context
{text}

### Behavioral Guidelines

**Response Style:**
- Return ONLY the direct utterance/speech of the character
- Do NOT include speaker names, labels like "発言者:", or any narrative descriptions
- The character may speak multiple times in succession
- Focus purely on what the character says, nothing else
- Your utterance should be in the same language as the input text
- **Your utterance must be in {language}**
- If any character profile or description is written in a language other than {language}, reinterpret it in {language} before using it.

**Conversation Approach:**
- You may follow the conversation context, ignore it, or redirect it based on your persona
- You can introduce completely unrelated topics if that fits your character
- You can misunderstand the context if your persona tends to do so
- You can focus on your own interests regardless of what others are discussing
- You can experiment with the app's features or test its limits if that's your character
- Feel free to be unpredictable, eccentric, or unconventional in your responses

**Communication Style:**
- Keep responses BRIEF and NATURAL like real conversation (1-2 sentences max)
- Avoid long explanations or AI-like verbose responses
- Make it sound like how people actually talk in casual conversation
- For Japanese: Use natural, colloquial expressions typical of everyday conversation
- Match the formality level to your persona (casual, polite, eccentric, etc.)
- Use emojis, typos, slang, or repetitive phrases if that fits your character

**Important Reminders:**
- Prioritize authenticity to your persona over logical conversation flow
- Real users don't always respond appropriately or stay on topic
- It's okay to be confusing, off-topic, or seemingly random if that's who you are
- Your goal is to behave like a real person with quirks, not a perfect conversationalist
""",
            partial_variables={"language": language},
        )

    async def generate(
        self,
        text: str,
        profile: str,
        persona_name: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Call GPT asynchronously to get persona utterance."""

        messages = self.prompt.format_messages(
            profile=profile,
            persona_name=persona_name,
            text=text,
        )
        # Use semaphore to rate‑limit.
        async with semaphore:
            resp = await self.llm.ainvoke(messages)
        if hasattr(resp, "utterance"):
            return resp.utterance
        else:
            # Fallback if response structure is unexpected
            return str(resp)
