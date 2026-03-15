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
from utils import script_runner
import json
from tqdm import tqdm
import log.get_log

logger_console = log.get_log.get_console_logger(__file__)
logger_file = log.get_log.get_file_logger(__file__ + "file")

def generate_code(prompt_config, llm_config, llm_client, output_dir, account_path, instruct_path):
    account_data, instruct_data = extract_data.extract_data(account_path, instruct_path)

    base_prompt = build_generation_prompt(account_data, instruct_data, prompt_config)
    final_prompt = rewrite_prompt_with_llm(base_prompt, prompt_config, llm_config, llm_client)
    smtlib_code = generate_smtlib_code(final_prompt, llm_config, llm_client)

    out_name = f"{Path(account_path).stem}.py"
    out_path = output_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(smtlib_code)
    
    logger_console.info(f"Generated code is {out_path}")

    return out_path


if __name__ == "__main__":
    argsParser = argparse.ArgumentParser(description="Natural language to SMT-LIB V2 code")
    argsParser.add_argument("--account_dir", type=str, default=None)
    argsParser.add_argument("--instruct_dir", type=str, default=None)
    argsParser.add_argument("--label_file", type=str, default=None)
    argsParser.add_argument("--output_dir", type=str, default=None)
    argsParser.add_argument("--limit", type=int, default=None, help="Only process first N samples when > 0")
    argsParser.add_argument("--mode", nargs='+', type=int, default=[2], help="1: generation 2: run")
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

    output_dir_value = args.output_dir if args.output_dir is not None else runtime_config.get("output_dir", "./outputs")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit is not None else runtime_config.get("limit", 0)
    if limit and limit > 0:
        data_pairs = data_pairs[: limit]

    if 1 in args.mode:
        logger_console.info(f"Starting code generation for {len(data_pairs)} data pairs...")
        for account_path, instruct_path, _label in tqdm(data_pairs, desc="Processing data pairs"):
            out_path = generate_code(prompt_config, llm_config, llm_client, output_dir, account_path, instruct_path)


            logger_console.info(f"Run results for {out_path}:")
            run_results = script_runner.run_py_files_in_dir(out_path.as_posix())

            for res in run_results:
                logger_file.info(f"File: {res['file']}, Stdout: {res['stdout']}, Result: {res['result']}, Label: {_label}")
    
    elif 2 in args.mode:
        logger_console.info(f"Starting to run generated code in {output_dir}...")
        run_results = script_runner.run_py_files_in_dir(output_dir.as_posix())

        for i, res in enumerate(run_results):
            logger_file.info(f"File: {res['file']}, Stdout: {res['stdout']}, Result: {res['result']}, Label: {data_pairs[i][2]}")
