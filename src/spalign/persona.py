"""Persona generation using OpenAI GPT models."""

from __future__ import annotations

import asyncio
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import PersonaResponse


class PersonaGenerator:
    """Generate persona responses using OpenAI GPT models."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, api_key=os.getenv("GEMINI_API_KEY")
        ).with_structured_output(PersonaResponse)
        self.prompt = ChatPromptTemplate.from_template(
            template="""
You embody the persona '{persona_name}'. You will generate a response as the persona in the messaging app.

### Profile
{profile}

### Input
{text}
"""
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
