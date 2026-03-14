import yaml
import data_processing.match_pairs as match_pairs
import data_processing.extract_data as extract_data
from prompt_generation.prompt_builder import (
    build_generation_prompt,
    rewrite_prompt_with_llm,
)
from llm.siliconflow_client import SiliconFlowClient, generate_smtlib_code
import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    argsParser = argparse.ArgumentParser(description="Natural language to SMT-LIB V2 code")
    argsParser.add_argument("--account_dir", type=str, default=None)
    argsParser.add_argument("--instruct_dir", type=str, default=None)
    argsParser.add_argument("--label_file", type=str, default=None)
    argsParser.add_argument("--output_dir", type=str, default=None)
    argsParser.add_argument("--limit", type=int, default=None, help="Only process first N samples when > 0")
    args = argsParser.parse_args()

    with open("cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_process_config = config.get("data_process", {})
    runtime_config = config.get("runtime", {})
    prompt_config = config.get("prompt", {})
    llm_config = config.get("llm", {})


    data_base_dir = data_process_config.get("data_base_dir")

    accounts_dir = args.account_dir if args.account_dir is not None else os.path.join(
        data_base_dir, data_process_config.get("accounts_dir", "accounts")
    )
    instructs_dir = args.instruct_dir if args.instruct_dir is not None else os.path.join(
        data_base_dir, data_process_config.get("instructs_dir", "instructs")
    )
    label_file = args.label_file if args.label_file is not None else os.path.join(
        data_base_dir,
        data_process_config.get("label_file", "answer_valid_permission.json"),
    )

    data_pairs = match_pairs.match_account_instructs(
        accounts_dir, instructs_dir, label_file
    )

    llm_client = SiliconFlowClient.from_config(llm_config)

    output_dir_value = args.output_dir if args.output_dir is not None else runtime_config.get("output_dir", "./outputs/smtlib")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit is not None else runtime_config.get("limit", 0)
    if limit and limit > 0:
        data_pairs = data_pairs[: limit]

    for account_path, instruct_path, _label in data_pairs:
        account_data, instruct_data = extract_data.extract_data(account_path, instruct_path)

        base_prompt = build_generation_prompt(account_data, instruct_data, prompt_config)
        final_prompt = rewrite_prompt_with_llm(base_prompt, prompt_config, llm_config, llm_client)
        smtlib_code = generate_smtlib_code(final_prompt, llm_config, llm_client)

        out_name = f"{Path(account_path).stem}.smt2"
        out_path = output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(smtlib_code)

        print(f"generated: {out_path}")




