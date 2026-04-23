"""
LLM Client - Core interface for NVIDIA API with streaming support
"""

from openai import OpenAI
import json
import time
from typing import Generator, Optional


NVIDIA_API_KEY = "your-nvidia-api-key-here"
BEST_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"  # Best reasoning model on NVIDIA API

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


def stream_completion(
    messages: list,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream a completion from the LLM, yielding text chunks."""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    completion = client.chat.completions.create(
        model=BEST_MODEL,
        messages=full_messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield f"[THINKING] {reasoning}"
        if delta.content:
            yield delta.content


def get_completion(
    messages: list,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> str:
    """Get a full completion (non-streaming) and return as string."""
    result = ""
    for chunk in stream_completion(messages, temperature, max_tokens, system_prompt):
        if not chunk.startswith("[THINKING]"):
            result += chunk
    return result.strip()


def get_json_completion(
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
) -> dict:
    """Get a JSON-structured completion."""
    sys = (system_prompt or "") + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no backticks, no extra text."
    raw = get_completion(messages, temperature, max_tokens, sys)
    # Strip markdown fences if present
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": "Failed to parse JSON", "raw": raw}


MODEL_INFO = {
    "name": BEST_MODEL,
    "provider": "NVIDIA API",
    "description": "Llama 3.3 Nemotron Super 49B - Best reasoning model with 131k context",
}
