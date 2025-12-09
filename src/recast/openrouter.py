"""OpenRouter API client for ReCast benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 600  # 10 minutes for long CoT


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    answer: str | None
    reasoning: str | None
    finish_reason: str


async def call_openrouter(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    include_reasoning: bool = True,
) -> LLMResponse:
    """Call the OpenRouter API.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier (e.g., "deepseek/deepseek-r1")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        include_reasoning: Whether to request reasoning output

    Returns:
        LLMResponse with answer, reasoning, and finish_reason

    Raises:
        ValueError: If API key not found
        RuntimeError: If API request fails
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if include_reasoning:
            payload["include_reasoning"] = True

        async with session.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"API request failed ({response.status}): {text}")

            data = await response.json()
            message = data["choices"][0]["message"]

            return LLMResponse(
                answer=message.get("content"),
                reasoning=message.get("reasoning"),
                finish_reason=data["choices"][0]["finish_reason"],
            )


async def call_openrouter_with_prefill(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    max_continuations: int = 3,
) -> LLMResponse:
    """Call OpenRouter with automatic prefill continuation for long responses.

    If the model hits the length limit, this function will automatically
    continue the generation by prefilling the assistant message.

    Args:
        messages: List of message dicts (must have assistant prefill as last message)
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_continuations: Maximum number of continuation attempts

    Returns:
        LLMResponse with combined answer and reasoning
    """
    continuations = 0

    while True:
        result = await call_openrouter(
            messages, model, temperature, max_tokens, include_reasoning=True
        )

        if result.finish_reason != "length" or continuations >= max_continuations:
            break

        # Prefill continuation
        continuations += 1
        print(f"Prefilled response for {model} with {continuations} responses")

        if result.reasoning and result.answer:
            messages[-1]["content"] += f"{result.reasoning}</think>{result.answer}"
        elif result.reasoning:
            messages[-1]["content"] += result.reasoning
        elif result.answer:
            messages[-1]["content"] += result.answer
        else:
            raise RuntimeError("No response from model to continue")

    # Combine prefilled content with final response
    if continuations > 0 and result.reasoning:
        result = LLMResponse(
            answer=result.answer,
            reasoning=messages[-1]["content"] + result.reasoning,
            finish_reason=result.finish_reason,
        )

    # Clean up response
    answer = result.answer.strip() if result.answer else None
    reasoning = result.reasoning.strip() if result.reasoning else None

    # Ensure reasoning starts with <think>
    if reasoning and not reasoning.startswith("<think>"):
        reasoning = "<think>" + reasoning

    # Extract reasoning from answer if embedded and not already extracted
    if not reasoning and answer and "</think>" in answer:
        think_start = answer.rfind("<think>") + len("<think>")
        think_end = answer.rfind("</think>")
        if think_start > 0 and think_end > think_start:
            reasoning = answer[think_start:think_end].strip()
            answer = (answer[: think_start - len("<think>")] + answer[think_end + len("</think>") :]).strip()
    elif not answer and reasoning and reasoning.endswith("</think>"):
        think_end = reasoning.rfind("</think>")
        if think_end > 0:
            answer = reasoning[think_end + len("</think>"):].strip()
            reasoning = reasoning[:think_end].strip() + "</think>"

    return LLMResponse(answer=answer, reasoning=reasoning, finish_reason=result.finish_reason)
