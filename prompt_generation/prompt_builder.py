import json
from typing import Any, Dict


def _to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def build_generation_prompt(account_data: Dict, instruct_data: Dict, prompt_config: Dict) -> str:
    """Build the base prompt for Python z3 generation from a template file."""
    instruct_text = (instruct_data or {}).get("instruct", "")
    template_path = prompt_config.get("base_prompt_template_path")
    if not template_path:
        raise ValueError("prompt.base_prompt_template_path is required in cfg/config.yaml")

    template = load_prompt_file(template_path)
    return template.format(
        account_json=_to_pretty_json(account_data),
        instruct_text=instruct_text,
    )


def load_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def rewrite_prompt_with_llm(base_prompt: str, prompt_config: Dict, llm_config: Dict, llm_client: Any) -> str:
    """Optionally rewrite the prompt with an LLM to improve Python z3 generation quality."""
    if not prompt_config.get("rewrite_enabled", False):
        return base_prompt

    rewrite_model = llm_config.get("rewrite_model")
    if not rewrite_model:
        return base_prompt

    system_prompt = "你是 prompt 优化专家。请在不改变语义的前提下，重写用户输入，使其更适合让模型生成高质量的 Python z3 代码。"
    system_prompt_path = prompt_config.get("rewrite_system_prompt_path")
    if system_prompt_path:
        try:
            system_prompt = load_prompt_file(system_prompt_path)
        except OSError:
            # Fallback to default system prompt when file is unavailable.
            pass

    rewrite_user_instruction = prompt_config.get(
        "rewrite_user_instruction",
        "请重写以下 prompt，并保留其核心语义。输出仅包含重写后的 prompt 文本，且目标是生成 Python z3 代码。",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{rewrite_user_instruction}\n\n"
                f"{base_prompt}"
            ),
        },
    ]

    rewritten = llm_client.chat_complete(
        model=rewrite_model,
        messages=messages,
        temperature=float(llm_config.get("rewrite_temperature", 0.1)),
        max_tokens=int(llm_config.get("rewrite_max_tokens", 4096)),
    )
    return rewritten.strip() or base_prompt
