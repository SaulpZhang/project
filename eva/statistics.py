import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _normalize_run_result(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return value


def compute_metrics(records: List[Dict[str, Any]], pass_k: int) -> Dict[str, Any]:
    total_generations = len(records)
    if total_generations == 0:
        return {
            "total_cases": 0,
            "total_generations": 0,
            "pass_k": pass_k,
            "pass_at_k": 0.0,
            "avg_accuracy": 0.0,
            "generation_success_rate": 0.0,
            "avg_retries": 0.0,
            "avg_generation_time_sec": 0.0,
        }

    success_count = sum(1 for r in records if bool(r.get("generation_success")))
    avg_retries = sum(int(r.get("retries_used", 0)) for r in records) / total_generations
    avg_generation_time_sec = (
        sum(float(r.get("generation_time_sec", 0.0)) for r in records) / total_generations
    )

    grouped = defaultdict(list)
    for r in records:
        grouped[r.get("case_id")].append(r)

    total_cases = len(grouped)
    passed_cases = 0
    pass_case_num = 0

    for _, case_records in grouped.items():
        label = bool(case_records[0].get("label"))
        ordered = sorted(case_records, key=lambda x: int(x.get("sample_idx", 0)))
        top_k = ordered[: max(pass_k, 1)]

        case_pass = False
        for rec in top_k:
            if not bool(rec.get("generation_success")):
                continue
            run_result = _normalize_run_result(rec.get("run_result"))
            if isinstance(run_result, bool) and run_result == label:
                case_pass = True
                pass_case_num += 1

        if case_pass:
            passed_cases += 1

    pass_at_k = passed_cases / total_cases if total_cases else 0.0

    return {
        "total_cases": total_cases,
        "total_generations": total_generations,
        "pass_k": pass_k,
        "pass_at_k": pass_at_k,
        "avg_accuracy": pass_case_num / total_generations if total_generations else 0.0,
        "generation_success_rate": success_count / total_generations,
        "avg_retries": avg_retries,
        "avg_generation_time_sec": avg_generation_time_sec,
    }


def save_generation_records(records: List[Dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "generation_records.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "metrics_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path
