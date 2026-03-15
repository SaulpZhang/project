from pathlib import Path
import runpy
import contextlib
from io import StringIO
from typing import List, Dict, Any
from data_processing.match_pairs import get_key
import log.get_log

logger_console = log.get_log.get_console_logger(__file__)


def run_py_files_in_dir(directory: str) -> List[Dict[str, Any]]:
    """Run all .py files in `directory` (non-recursive), capture stdout and a
    `result` variable if present, and return a list of dicts with the outputs.

    Each dict contains: `file`, `stdout`, `result`.
    Exceptions are captured and returned in the `result` field as strings.
    """
    py_files = []
    results: List[Dict[str, Any]] = []
    base = Path(directory)
    if not base.exists():
        logger_console.info(f"Directory {directory} does not exist. No files to run.")
        return results
    if base.is_dir():
        py_files = sorted([p for p in base.iterdir() if p.suffix == ".py"], key=get_key)
    elif base.is_file() and base.suffix == ".py":
        py_files = [base]
    
    logger_console.info(f"Found {len(py_files)} .py files in {directory} to run.")

    for p in py_files:
        buf = StringIO()
        globals_dict: Dict[str, Any] = {}
        try:
            with contextlib.redirect_stdout(buf):
                globals_dict = runpy.run_path(str(p))
            out = buf.getvalue()
            res = globals_dict.get("result")
            if str(res) == "sat":
                res = True
            elif str(res) == "unsat":
                res = False
            results.append({"file": p.name, "stdout": out, "result": res})
        except ModuleNotFoundError as e:
            results.append({"file": p.name, "stdout": buf.getvalue(), "result": f"error: missing module {e.name}"})
        except Exception as e:
            results.append({"file": p.name, "stdout": buf.getvalue(), "result": f"error: {e!r}"})

    return results