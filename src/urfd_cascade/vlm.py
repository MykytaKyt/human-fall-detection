"""Semantic verification stage (second cascade): queries a vision-language
model for a scene-level judgement, following the paper's prompt design
("is this a controlled activity, or an accident?" rather than asking the
VLM to reason about physics/acceleration directly).

Talks to any OpenAI-compatible chat-completions server -- by default
LM Studio (https://lmstudio.ai) running locally, e.g. serving
google/gemma-3-4b (a small vision-capable model). Point base_url at any
other OpenAI-compatible endpoint (vLLM, Ollama, a hosted API, ...) to swap
providers without touching the rest of the cascade.

ponytail: uses the `openai` package purely as an HTTP client for LM
Studio's OpenAI-compatible API -- no OpenAI account/key involved. A raw
`requests.post` would work identically; the SDK just saves us writing the
request/response schema by hand.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass

PROMPT = (
    "Is the action performed by the person in the centre of the frame a "
    "controlled sporting or everyday activity, or is it an accident? "
    'Respond with strict JSON only: {"p_anomaly": <0..1>, "entropy": <0..1>, '
    '"reason": "<one short sentence>"}. p_anomaly is your probability that '
    "this is a genuine accident/fall requiring help. entropy is your own "
    "uncertainty about that judgement (0 = certain, 1 = a coin flip)."
)


@dataclass
class SemanticJudgement:
    p_anomaly: float
    entropy: float
    reason: str


class VLMClient:
    """Thin client for the semantic cascade. Default settings match a
    local LM Studio server (see README: `lms load google/gemma-3-4b`).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "google/gemma-3-4b",
        api_key: str = "lm-studio",  # ponytail: LM Studio ignores the key; SDK requires a non-empty string
    ):
        from openai import OpenAI  # lazy import: only needed if you actually use the VLM

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def judge_image(self, image_path: str, timeout: float = 60.0) -> SemanticJudgement:
        """Send one frame to the VLM and parse its semantic judgement."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.0,
            timeout=timeout,
        )
        raw = response.choices[0].message.content
        return _parse_judgement(raw)

    def ping(self) -> bool:
        """Health check: True if the server is reachable and serving `model`."""
        try:
            names = {m.id for m in self._client.models.list().data}
            return self.model in names or not names  # empty list = server up, model list unknown
        except Exception:
            return False


def _parse_judgement(raw: str) -> SemanticJudgement:
    """Parse the model's JSON reply, tolerating markdown code fences that
    small local models commonly wrap JSON in."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    return SemanticJudgement(
        p_anomaly=float(data["p_anomaly"]),
        entropy=float(data["entropy"]),
        reason=str(data.get("reason", "")),
    )
