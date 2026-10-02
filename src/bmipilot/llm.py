"""LLM factory."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from bmipilot.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
