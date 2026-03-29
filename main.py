import yaml
import data_processing.match_pairs as match_pairs
import data_processing.extract_data as extract_data
from llm.siliconflow_client import SiliconFlowClient
from llm.code_generator import CodeGenerator
import os
import argparse
from pathlib import Path
from utils import script_runner
from eva.statistics import compute_metrics, save_generation_records, save_metrics
import log.get_log
import time

logger_console = log.get_log.get_console_logger(__file__)
logger_file = log.get_log.get_file_logger(__file__ + "file")

def generate_code(
    code_gen: CodeGenerator,
    output_dir: Path,
    account_path: str,
    instruct_path: str,
    sample_idx: int,
):
    account_data, instruct_data = extract_data.extract_data(account_path, instruct_path)

    start_time = time.perf_counter()
    generated_code, success, error_msg, retries_used = code_gen.generate_code(account_data, instruct_data)
    generation_time_sec = time.perf_counter() - start_time

    out_name = f"{Path(account_path).stem}_s{sample_idx}.py"
    out_path = output_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    status = "success" if success else "failed"
    logger_console.info(
        f"Generated code {status}: {out_path}, retries_used={retries_used}, "
        f"generation_time_sec={generation_time_sec:.3f}"
    )

    return out_path, success, error_msg, retries_used, generation_time_sec


def get_args():
    argsParser = argparse.ArgumentParser(description="Natural language to SMT-LIB V2 code")
    argsParser.add_argument("--account_dir", type=str, default=None)
    argsParser.add_argument("--instruct_dir", type=str, default=None)
    argsParser.add_argument("--label_file", type=str, default=None)
    argsParser.add_argument("--output_dir", type=str, default=None)
    argsParser.add_argument("--limit", type=int, default=None, help="Only process first N samples when > 0")
    argsParser.add_argument("--generate_mode", type=int, default=0, help="0: zero-shot, 1: one-shot, 2: few-shot, 3: chain-of-thought")

    args = argsParser.parse_args()
    return args

def update_config(args, base_config):
    if args.generate_mode is not None:
        base_config["llm"]["generate_mode"] = args.generate_mode

if __name__ == "__main__":
    args = get_args()

    with open("cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        update_config(args, config)
    
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

    samples_per_case = int(runtime_config.get("samples_per_case", 1))
    if samples_per_case < 1:
        logger_console.warning("runtime.samples_per_case must be >= 1, fallback to 1")
        samples_per_case = 1

    pass_k = int(runtime_config.get("pass_k", samples_per_case))
    if pass_k < 1:
        logger_console.warning("runtime.pass_k must be >= 1, fallback to 1")
        pass_k = 1

    code_gen = CodeGenerator(llm_client, llm_config, prompt_config)
    generation_records = []

    logger_console.info(
        f"Starting code generation for {len(data_pairs)} data pairs, "
        f"samples_per_case={samples_per_case}, pass_k={pass_k}"
    )

    for case_idx, (account_path, instruct_path, label) in enumerate(data_pairs, start=1):
        case_id = Path(account_path).stem
        logger_console.info(f"Processing case {case_idx}/{len(data_pairs)}: {case_id}")

        for sample_idx in range(1, samples_per_case + 1):
            out_path, success, error_msg, retries_used, generation_time_sec = generate_code(
                code_gen=code_gen,
                output_dir=output_dir,
                account_path=account_path,
                instruct_path=instruct_path,
                sample_idx=sample_idx,
            )

            run_result = None
            run_stdout = ""
            if success:
                run_results = script_runner.run_py_files_in_dir(out_path.as_posix())
                if run_results:
                    run_result = run_results[0].get("result")
                    run_stdout = run_results[0].get("stdout", "")

            record = {
                "case_id": case_id,
                "account_path": account_path,
                "instruct_path": instruct_path,
                "label": bool(label),
                "sample_idx": sample_idx,
                "output_file": out_path.name,
                "generation_success": bool(success),
                "run_result": run_result,
                "retries_used": retries_used,
                "generation_time_sec": generation_time_sec,
                "error_message": error_msg,
            }
            generation_records.append(record)

            logger_file.info(
                f"Case={case_id}, Sample=s{sample_idx}, Output={out_path.name}, "
                f"Label={bool(label)}, GenSuccess={success}, RunResult={run_result}, "
                f"Retries={retries_used}, TimeSec={generation_time_sec:.3f}, "
                f"Stdout={run_stdout}, Error={error_msg}"
            )

    metrics = compute_metrics(generation_records, pass_k=pass_k)
    save_generation_records(generation_records, output_dir)
    save_metrics(metrics, output_dir)

    logger_console.info(
        f"Metrics summary: pass@{pass_k}={metrics.get('pass_at_k'):.4f}, "
        f"generation_success_rate={metrics.get('generation_success_rate'):.4f}, "
        f"avg_retries={metrics.get('avg_retries'):.4f}, "
        f"avg_generation_time_sec={metrics.get('avg_generation_time_sec'):.4f}"
    )

