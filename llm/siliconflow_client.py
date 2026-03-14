import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib import error, request


class SiliconFlowClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _resolve_api_key(raw_value: Optional[str]) -> str:
        if not raw_value:
            return ""

        # Prefer environment variable value when raw_value is an env key.
        env_val = os.getenv(raw_value)
        if env_val:
            return env_val

        # Also support directly configured key.
        return raw_value

    @classmethod
    def from_config(cls, llm_config: Dict[str, Any]) -> "SiliconFlowClient":
        api_key = cls._resolve_api_key(llm_config.get("api_key_env"))
        if not api_key:
            raise ValueError("SiliconFlow API key is missing. Set env var or api_key_env value in cfg/config.yaml")

        base_url = llm_config.get("base_url")
        if not base_url:
            raise ValueError("llm.base_url is required in cfg/config.yaml")

        return cls(
            base_url=base_url,
            api_key=api_key,
            timeout=int(llm_config.get("request_timeout", 120)),
        )

    def chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"SiliconFlow HTTP error: {e.code}, details: {details}") from e
        except error.URLError as e:
            raise RuntimeError(f"SiliconFlow request failed: {e}") from e

        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"SiliconFlow returned no choices: {response_data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            return ""
        return content.strip()


def load_generation_system_prompt(path: Optional[str]) -> str:
    default_prompt = (
        "你是 SMT-LIB v2 代码生成器。你必须严格输出 SMT-LIB v2 代码，"
        "不要输出额外解释。"
    )
    if not path:
        return default_prompt

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or default_prompt
    except OSError:
        return default_prompt


def extract_smtlib_code(raw_output: str) -> str:
    if not raw_output:
        return ""

    # Prefer fenced code block content when present.
    match = re.search(r"```(?:smt2|smt|smt-lib)?\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Remove a possible single leading or trailing fence when output is malformed.
    cleaned = raw_output.replace("```smt2", "").replace("```smt", "").replace("```", "")
    return cleaned.strip()


def generate_smtlib_code(prompt_text: str, llm_config: Dict[str, Any], llm_client: SiliconFlowClient) -> str:
    model = llm_config.get("generation_model")
    if not model:
        raise ValueError("generation_model is required in cfg/config.yaml")

    system_prompt = load_generation_system_prompt(llm_config.get("generation_system_prompt_path"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    raw = llm_client.chat_complete(
        model=model,
        messages=messages,
        temperature=float(llm_config.get("generation_temperature", 0.1)),
        max_tokens=int(llm_config.get("generation_max_tokens", 4096)),
    )
    return extract_smtlib_code(raw)
