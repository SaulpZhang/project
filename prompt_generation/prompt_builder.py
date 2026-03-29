import json
from pathlib import Path
from typing import Any, Dict, Optional


def _to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def load_prompt_file(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()


def get_generation_system_prompt_by_mode(generate_mode: int, prompt_config: Dict) -> str:
    mode_mapping = prompt_config.get("generation_mode_mapping", {})
    if generate_mode not in mode_mapping:
        raise ValueError(f"Invalid generate_mode: {generate_mode}. Must be 0, 1, 2, or 3.")

    prompt_key = mode_mapping[generate_mode]
    prompt_path = prompt_config.get(prompt_key)
    if not prompt_path:
        raise ValueError(f"prompt.{prompt_key} is required in cfg/config.yaml")
    return load_prompt_file(prompt_path)


def build_generation_prompt(
    account_data: Dict,
    instruct_data: Dict,
    prompt_config: Dict,
    generate_mode: int = 0,
) -> str:
    """Build generation prompt by combining mode-specific system prompt and base user template."""
    instruct_text = (instruct_data or {}).get("instruct", "")
    template_path = prompt_config.get("base_prompt_template_path")
    if not template_path:
        raise ValueError("prompt.base_prompt_template_path is required in cfg/config.yaml")

    template = load_prompt_file(template_path)
    user_prompt = template.format(
        account_json=_to_pretty_json(account_data),
        instruct_text=instruct_text,
    )
    system_prompt = get_generation_system_prompt_by_mode(generate_mode, prompt_config)
    system_prompt = (
        system_prompt
        .replace("{account_json}", _to_pretty_json(account_data))
        .replace("{instruct_text}", instruct_text)
    )

    return f"{system_prompt}\n\n{user_prompt}"


def build_regenerate_prompt(
    code: str,
    error: str,
    prompt_config: Dict,
    generate_mode: int = 0,
    account_data: Optional[Dict] = None,
    instruct_data: Optional[Dict] = None,
) -> str:
    """Build regenerate prompt with failed code and execution error context."""
    repair_prompt_path = prompt_config.get("regenerate_error_repair_path")
    if not repair_prompt_path:
        raise ValueError("prompt.regenerate_error_repair_path is required in cfg/config.yaml")

    repair_template = load_prompt_file(repair_prompt_path)
    user_prompt = repair_template.format(code=code, error=error)

    instruct_text = (instruct_data or {}).get("instruct", "")
    system_prompt = get_generation_system_prompt_by_mode(generate_mode, prompt_config)
    system_prompt = (
        system_prompt
        .replace("{account_json}", _to_pretty_json(account_data or {}))
        .replace("{instruct_text}", instruct_text)
    )

    context = ""
    if account_data:
        context += f"\n\n【账户信息】\n{_to_pretty_json(account_data)}"
    if instruct_text:
        context += f"\n\n【验证目标】\n{instruct_text}"

    return f"{system_prompt}\n\n{user_prompt}{context}"
