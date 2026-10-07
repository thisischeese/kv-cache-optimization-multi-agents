"""Shared chat model factory. Agents call this instead of building their own
client, so the model name and temperature stay in one place (config.py)."""

from langchain_openai import ChatOpenAI

from kv_eval.config import llm_model


def chat_model(temperature: float = 0.0, model: str | None = None, **kwargs) -> ChatOpenAI:
    """`model` overrides the generator model, e.g. config.judge_model() for evaluation."""
    return ChatOpenAI(model=model or llm_model(), temperature=temperature, **kwargs)
