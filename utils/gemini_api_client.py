"""
Utilities for interacting with the Google Gemini API.

This file was extracted from `gemini_batch_query.py` to make the
Gemini‐specific logic reusable across different scripts and easier to
maintain.  The public surface is the `GeminiAPIClient` class which wraps
both the google-genai SDK and the raw REST endpoint behind a single
interface.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Optional dependency: google-genai SDK.  We fall back to REST calls if the
# library is unavailable or the caller explicitly disables it.
# ---------------------------------------------------------------------------
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    GENAI_AVAILABLE: bool = True
except ImportError:  # pragma: no cover – library is optional
    GENAI_AVAILABLE = False

__all__ = ["GeminiAPIClient", "GENAI_AVAILABLE"]


class GeminiAPIClient:  # pylint: disable=too-many-public-methods
    """Light-weight wrapper around the Gemini API (REST & python-sdk).

    The client automatically falls back to the REST endpoint when the
    python SDK is not installed, or when running in environments where the
    SDK cannot be used (for example because the *thinking* feature is not
    required).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        *,
        enable_thinking: bool = False,
        use_genai_client: Optional[bool] = None,
    ) -> None:
        """Create a new Gemini client.

        Args:
            api_key:   Google AI API key.
            model:     Model identifier, e.g. ``"gemini-2.0-flash"``.
            enable_thinking: Whether to request *thinking* traces from the
                model.  Only supported when using the ``google-genai`` SDK.
            use_genai_client: Force usage of the SDK (``True``) or the REST
                API (``False``).  When *None* (default) the choice is made
                automatically: the SDK is used only if *thinking* is enabled
                *and* the library is available.
        """
        self.api_key = api_key
        self.model = model
        self.enable_thinking = enable_thinking
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

        # Decide which transport to use --------------------------------------------------
        if use_genai_client is None:
            use_genai_client = enable_thinking and GENAI_AVAILABLE

        self._use_genai = bool(use_genai_client)
        self._genai_client = None
        self._session: Optional[requests.Session] = None

        print(
            f"Initializing GeminiAPIClient with use_genai={self._use_genai} and enable_thinking={self.enable_thinking}"
        )
        if self._use_genai:
            print("Using genai client")
            if not GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai library not available but use_genai_client=True. "
                    "Install it with: pip install google-genai"
                )
            self._genai_client = genai.Client(api_key=api_key)  # type: ignore[arg-type]
        else:
            print("Using rest api")
            self._session = requests.Session()
            self._session.headers.update(  # type: ignore[assignment]
                {
                    "Content-Type": "application/json",
                    "X-goog-api-key": api_key,
                }
            )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def generate_content(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Send *prompt* to Gemini and return the raw JSON response."""
        print(f"Generating content with {self._use_genai} and {self.enable_thinking}")
        if self._use_genai:
            return self._generate_with_genai(prompt, **kwargs)
        return self._generate_with_rest(prompt, **kwargs)

    # Convenience wrappers -------------------------------------------------

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Return the *visible* textual answer from a Gemini response."""
        if "error" in response:
            return f"Error: {response['error']}"

        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return "No valid response found"

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return "No valid response found"

            thinking_enabled = response.get("thinking_enabled", False)
            thinking_supported = response.get("thinking_supported", False)

            if thinking_enabled and thinking_supported:
                return self._merge_thought_and_text(parts)
            return parts[0].get("text", "No text in response")
        except Exception as exc:  # pragma: no cover
            return f"Error parsing response: {exc}"

    # ------------------------------------------------------------------
    # Backwards-compatibility shims (for gemini_batch_query.py) ----------
    # ------------------------------------------------------------------

    def extract_text_response(self, response: Dict[str, Any]) -> str:  # noqa: D401
        """Alias for :py:meth:`extract_text` to keep old callers working."""
        return self.extract_text(response)

    def extract_thinking_and_text_separately(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return separate *thinking* and *text* portions from a response.

        Matches the behaviour of the legacy implementation so that existing
        scripts do not need to change when switching to the new module.
        """
        if "error" in response:
            return {"error": response["error"], "thinking": "", "text": ""}

        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return {"thinking": "", "text": "", "has_thinking": False}

            parts = candidates[0].get("content", {}).get("parts", [])

            thoughts: List[str] = []
            texts: List[str] = []

            for part in parts:
                if part.get("thought") and "text" in part:
                    thoughts.append(part["text"])
                elif "text" in part and not part.get("thought"):
                    texts.append(part["text"])

            return {
                "thinking": "\n\n".join(thoughts),
                "text": "\n\n".join(texts),
                "has_thinking": bool(thoughts),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc), "thinking": "", "text": ""}

    # ---------------------------------------------------------------------
    # Internal SDK implementation -----------------------------------------
    # ---------------------------------------------------------------------

    def _generate_with_genai(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        assert self._genai_client is not None  # for mypy

        import google.genai as genai_mod  # type: ignore  # Local import to avoid hard dep

        print(f"Prompt: {prompt}")
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        max_tokens = kwargs.get("max_tokens", 4096)

        generation_config = types.GenerateContentConfig(  # type: ignore
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        if self.enable_thinking:
            generation_config.thinking_config = types.ThinkingConfig(  # type: ignore[attr-defined]
                thinking_budget=2048,
                include_thoughts=True,
            )

        try:
            response = self._genai_client.models.generate_content(  # type: ignore
                model=self.model,
                contents=prompt,
                config=generation_config,
            )
        except genai_mod.ApiError as api_err:  # type: ignore[attr-defined]
            return {"error": str(api_err), "api_method": "genai_library"}

        transformed: Dict[str, Any] = {
            "candidates": [],
            "thinking_enabled": self.enable_thinking,
            "thinking_supported": False,
            "api_method": "genai_library",
        }

        print(f"Response: {response}")
        for cand in response.candidates:  # type: ignore[attr-defined]
            cand_dict: Dict[str, Any] = {"content": {"parts": []}}
            for part in cand.content.parts:  # type: ignore[attr-defined]
                part_info: Dict[str, Any] = {}
                if getattr(part, "text", None):
                    part_info["text"] = part.text
                if getattr(part, "thought", None):
                    part_info["thought"] = part.thought
                    transformed["thinking_supported"] = True
                if part_info:
                    cand_dict["content"]["parts"].append(part_info)
            transformed["candidates"].append(cand_dict)

        return transformed

    # ---------------------------------------------------------------------
    # Internal REST implementation ----------------------------------------
    # ---------------------------------------------------------------------

    def _generate_with_rest(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        assert self._session is not None  # for mypy

        url = f"{self.base_url}/{self.model}:generateContent"

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
            },
        }

        safety_settings = kwargs.get("safety_settings")
        if safety_settings:
            payload["safetySettings"] = safety_settings

        try:
            response = self._session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            result.update(
                {
                    "api_method": "rest_api",
                    "thinking_enabled": False,
                }
            )
            return result
        except requests.exceptions.RequestException as exc:  # pragma: no cover
            return {
                "error": str(exc),
                "status_code": getattr(exc.response, "status_code", None),
                "api_method": "rest_api",
            }

    # ---------------------------------------------------------------------
    # Helper utilities -----------------------------------------------------
    # ---------------------------------------------------------------------

    @staticmethod
    def _merge_thought_and_text(parts: List[Dict[str, Any]]) -> str:
        """Format *thought* and regular text parts into one string output."""
        thoughts: List[str] = []
        texts: List[str] = []

        for part in parts:
            if part.get("thought") and "text" in part:
                thoughts.append(part["text"])
            elif "text" in part:
                texts.append(part["text"])

        blended: List[str] = []
        if thoughts:
            thinking_block = "\n\n".join(thoughts)
            blended.append(f"<thinking>\n{thinking_block}\n</thinking>")
        if texts:
            blended.append("\n\n".join(texts))
        return "\n\n".join(blended)
