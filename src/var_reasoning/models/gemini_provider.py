"""Gemini API provider with structured output support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from var_reasoning.models.schemas import (
    CodeFix,
    InferenceRevision,
    InferenceStep,
    StepOutput,
)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GeminiProvider:
    model_name: str = "gemini-2.5-flash"
    _client: genai.Client = field(init=False, repr=False)
    _cumulative_usage: TokenUsage = field(default_factory=TokenUsage, init=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key)

    def _track_usage(self, response: types.GenerateContentResponse) -> TokenUsage:
        usage = TokenUsage()
        if response.usage_metadata:
            usage.input_tokens = response.usage_metadata.prompt_token_count or 0
            usage.output_tokens = response.usage_metadata.candidates_token_count or 0
        self._cumulative_usage.input_tokens += usage.input_tokens
        self._cumulative_usage.output_tokens += usage.output_tokens
        return usage

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._cumulative_usage

    def reset_usage(self) -> None:
        self._cumulative_usage = TokenUsage()

    def generate_reasoning_step(
        self, system_prompt: str, conversation: list[str]
    ) -> tuple[StepOutput, TokenUsage]:
        contents = "\n\n".join(conversation)
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=StepOutput,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = self._track_usage(response)
        return response.parsed, usage

    def generate_inference(
        self, system_prompt: str, observation_context: str
    ) -> tuple[InferenceStep, TokenUsage]:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=observation_context,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=InferenceStep,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = self._track_usage(response)
        return response.parsed, usage

    def generate_code_fix(self, prompt: str) -> tuple[CodeFix, TokenUsage]:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=CodeFix,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = self._track_usage(response)
        return response.parsed, usage

    def generate_inference_revision(
        self, prompt: str
    ) -> tuple[InferenceRevision, TokenUsage]:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=InferenceRevision,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = self._track_usage(response)
        return response.parsed, usage

    def generate_one_shot(
        self, problem: str, system_prompt: str, model_override: str | None = None
    ) -> tuple[str, TokenUsage]:
        model = model_override or self.model_name
        response = self._client.models.generate_content(
            model=model,
            contents=problem,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = self._track_usage(response)
        text = response.text or ""
        return text, usage
