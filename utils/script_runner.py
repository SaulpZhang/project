from pathlib import Path
import runpy
import contextlib
from io import StringIO
from typing import List, Dict, Any
from data_processing.match_pairs import get_key
import log.get_log
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger_console = log.get_log.get_console_logger(__file__)

# Global timeout setting (20 seconds)
EXECUTION_TIMEOUT = 20


def _run_code_with_timeout(func, *args, **kwargs) -> Any:
    """
    Execute a function with a timeout of EXECUTION_TIMEOUT seconds.
    
    Args:
        func: The function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        The return value of func if it completes within timeout
        
    Raises:
        TimeoutError if execution exceeds EXECUTION_TIMEOUT
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=EXECUTION_TIMEOUT)
        except FuturesTimeoutError:
            raise TimeoutError(f"Code execution exceeded {EXECUTION_TIMEOUT} seconds timeout")


def _normalize_value_for_json(value: Any) -> Any:
    """
    Convert z3 objects and other non-JSON-serializable types to JSON-serializable Python types.
    
    Args:
        value: The value to normalize
        
    Returns:
        A JSON-serializable Python value
    """
    if value is None:
        return None
    
    # Handle z3 objects
    try:
        import z3
        
        # Check if it's a z3 BoolRef or other z3 expression
        if isinstance(value, (z3.BoolRef, z3.ExprRef, z3.FuncDeclRef)):
            # Try to convert to string and then to Python type
            str_val = str(value)
            if str_val == "True":
                return True
            elif str_val == "False":
                return False
            elif str_val == "sat":
                return True
            elif str_val == "unsat":
                return False
            else:
                # Return string representation
                return str_val
        
        # Check for z3 CheckSatResult
        if hasattr(z3, 'CheckSatResult') and isinstance(value, z3.CheckSatResult):
            result_str = str(value)
            if result_str == "sat":
                return True
            elif result_str == "unsat":
                return False
            else:
                return result_str
                
    except ImportError:
        pass
    
    # If it's already a basic type, return as is
    if isinstance(value, (bool, int, float, str, list, dict, type(None))):
        return value
    
    # For other types, try string conversion
    try:
        return str(value)
    except:
        return repr(value)


def run_py_files_in_dir(directory: str) -> List[Dict[str, Any]]:
    """Run all .py files in `directory` (non-recursive), capture stdout and a
    `result` variable if present, and return a list of dicts with the outputs.

    Each dict contains: `file`, `stdout`, `result`.
    Exceptions are captured and returned in the `result` field as strings.
    Code execution has a maximum timeout of 20 seconds.
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
        
        def execute_file():
            """Wrapper function for executing the file with timeout"""
            with contextlib.redirect_stdout(buf):
                return runpy.run_path(str(p))
        
        try:
            globals_dict = _run_code_with_timeout(execute_file)
            out = buf.getvalue()
            res = globals_dict.get("result")
            # Normalize z3 and other non-JSON-serializable objects
            res = _normalize_value_for_json(res)
            results.append({"file": p.name, "stdout": out, "result": res})
        except TimeoutError as e:
            results.append({"file": p.name, "stdout": buf.getvalue(), "result": f"error: {str(e)}"})
            logger_console.warning(f"File {p.name} execution timed out: {e}")
        except ModuleNotFoundError as e:
            results.append({"file": p.name, "stdout": buf.getvalue(), "result": f"error: missing module {e.name}"})
        except Exception as e:
            results.append({"file": p.name, "stdout": buf.getvalue(), "result": f"error: {e!r}"})

    return results


def run_smt_files_in_dir(directory: str) -> List[Dict[str, Any]]:
    """Run all .smt2 files in `directory` (non-recursive) using z3 solver.

    Each dict contains: `file`, `stdout`, `result` (True for sat, False for unsat).
    Exceptions are captured and returned in the `result` field as strings.
    """
    smt_files = []
    results: List[Dict[str, Any]] = []
    base = Path(directory)
    if not base.exists():
        logger_console.info(f"Directory {directory} does not exist. No files to run.")
        return results
    if base.is_dir():
        smt_files = sorted([p for p in base.iterdir() if p.suffix == ".smt2"], key=get_key)
    elif base.is_file() and base.suffix == ".smt2":
        smt_files = [base]
    
    logger_console.info(f"Found {len(smt_files)} .smt2 files in {directory} to run.")

    for p in smt_files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                smt_code = f.read()
            
            # Try to execute SMT code using z3
            result = execute_smt_code(smt_code)
            results.append({"file": p.name, "stdout": "", "result": result})
            
        except Exception as e:
            results.append({"file": p.name, "stdout": "", "result": f"error: {e!r}"})

    return results


def execute_smt_code(code: str) -> Any:
    """Execute SMT-LIB V2 code and return the satisfiability result.
    
    Has a maximum timeout of 20 seconds.
    
    Args:
        code: SMT-LIB V2 code to execute
        
    Returns:
        True for sat, False for unsat, or error string if execution failed
    """
    try:
        import z3
        import tempfile
        import subprocess
        
        # Try to use z3 command line tool first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Try z3 CLI first with 20 second timeout
            result = subprocess.run(
                ['z3', '-smt2', temp_file],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT
            )
            
            output = result.stdout.strip()
            
            # Check if the solver returned 'sat', 'unsat', or 'unknown'
            output_lower = output.lower()
            if output_lower == 'unsat':
                return False
            elif output_lower == 'sat':
                return True
            elif 'unsat' in output_lower:
                # More lenient check - if unsat appears anywhere
                return False
            elif 'sat' in output_lower:
                # More lenient check - if sat appears anywhere (but not unsat)
                return True
            else:
                return f"Unknown result: {output}"
                
        except subprocess.TimeoutExpired:
            error_msg = f"SMT execution exceeded {EXECUTION_TIMEOUT} seconds timeout"
            logger_console.warning(error_msg)
            return f"error: {error_msg}"
        except FileNotFoundError:
            # z3 CLI not available, try z3 Python API
            logger_console.debug("z3 CLI not found, using z3 Python API")
            # Use z3 Python API to parse and check SMT code
            z3.parse_smt2_string(code)
            # If we can parse it without error, consider it valid
            # Note: This doesn't actually run the solver, just parses the code
            return None  # Return None to indicate parse-only success
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except ImportError:
        return "error: z3 library not available"
    except Exception as e:
        return f"error: {e!r}"