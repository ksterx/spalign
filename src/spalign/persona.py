"""Persona generation using OpenAI GPT models."""

from __future__ import annotations

import asyncio
import os
import random

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

### How to play
{how_to_play}

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

        how_to_play = random.choice(
            [
                # Standard roleplay - following the context
                "You will play the role of the persona in the messaging app, responding naturally to the conversation flow.",
                "You will embody this persona and engage in authentic conversation based on the given context.",
                "Act as this persona, responding appropriately to what others are saying in the chat.",
                # Slightly off-topic but still engaged
                "You will play this persona who may occasionally bring up tangential topics while still participating in the conversation.",
                "Embody this persona who sometimes shifts the conversation slightly but remains engaged with others.",
                "Act as this persona who may introduce related but unexpected topics during the chat.",
                # More independent conversation style
                "You will play this persona who doesn't always follow the main conversation thread and may start their own topics.",
                "Embody this persona who often has their own agenda and may steer conversations in unexpected directions.",
                "Act as this persona who frequently brings up personal interests or concerns regardless of the current topic.",
                # Completely independent/disruptive style
                "You will play this persona who often ignores the current conversation and talks about whatever is on their mind.",
                "Embody this persona who rarely follows the group conversation and instead shares random thoughts or experiences.",
                "Act as this persona who treats the chat like their personal diary, often posting unrelated content.",
                # Experimental/testing behaviors
                "You will play this persona who is testing the limits of the app by trying unusual conversation patterns.",
                "Embody this persona who enjoys experimenting with different ways of communicating in the chat.",
                "Act as this persona who deliberately tries unconventional conversation approaches to see how others respond.",
                # Misunderstanding/confusion based
                "You will play this persona who often misunderstands the context and responds to what they think is happening.",
                "Embody this persona who frequently misinterprets messages and responds based on their misunderstanding.",
                "Act as this persona who seems to be having a different conversation than everyone else due to confusion.",
            ]
        )
        messages = self.prompt.format_messages(
            how_to_play=how_to_play,
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
