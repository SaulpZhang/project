import json
import re
import traceback
from typing import Any, Dict, Optional, Tuple
from log.get_log import get_console_logger
from prompt_generation.prompt_builder import build_generation_prompt, build_regenerate_prompt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = get_console_logger(__file__)

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


class CodeGenerator:
    """Handles code generation with optional error recovery and regeneration."""
    
    def __init__(self, llm_client: Any, llm_config: Dict[str, Any], prompt_config: Dict[str, Any]):
        """
        Initialize the code generator.
        
        Args:
            llm_client: The LLM client for making API calls
            llm_config: LLM configuration dict
            prompt_config: Prompt configuration dict
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.prompt_config = prompt_config
        
        self.generate_mode = int(llm_config.get("generate_mode", 0))
        self.generate_code_language = int(llm_config.get("generate_code_language", 0))
        self.regenerate_enabled = llm_config.get("regenerate_enabled", True)
        self.regenerate_max_retries = int(llm_config.get("regenerate_max_retries", 3))
        
        logger.info(f"CodeGenerator initialized - generate_mode: {self.generate_mode}, generate_code_language: {self.generate_code_language}, regenerate_enabled: {self.regenerate_enabled}, max_retries: {self.regenerate_max_retries}")
    
    def generate_code(
        self,
        account_data: Dict,
        instruct_data: Dict,
    ) -> Tuple[str, bool, Optional[str], int]:
        """
        Generate code with optional regeneration on error.
        
        Args:
            account_data: Account information
            instruct_data: Instruction data containing 'instruct' field
            
        Returns:
            Tuple of (generated_code, success, error_message, retries_used)
            - generated_code: The Python code generated
            - success: Whether the code executed successfully
            - error_message: Error message if failed, None if successful
            - retries_used: Number of regenerate attempts used
        """
        
        # Build initial generation prompt
        generation_prompt = build_generation_prompt(
            account_data=account_data,
            instruct_data=instruct_data,
            prompt_config=self.prompt_config,
            generate_mode=self.generate_mode,
            generate_code_language=self.generate_code_language,
        )
        
        # Generate code
        generated_code = self._call_llm_for_code_generation(generation_prompt)
        
        if not generated_code:
            logger.error("Failed to generate code from LLM")
            return "", False, "Failed to generate code from LLM", 0
        
        if not self.regenerate_enabled:
            return generated_code, True, None, 0


        # Try to execute the code

        success, error_msg = self._execute_code(generated_code)
        
        if success:

            return generated_code, success, error_msg, 0
        

        logger.warning(f"code execution failed: {error_msg}")
        if not self.regenerate_enabled:
            logger.warning("Regeneration is disabled, returning initial failure")
            return generated_code, success, error_msg, 0
        
        # Regenerate with error handling if enabled
        return self._regenerate_with_error_handling(
            generated_code=generated_code,
            error_message=error_msg or "Unknown error",
            account_data=account_data,
            instruct_data=instruct_data,
        )
    
    def _call_llm_for_code_generation(self, prompt: str) -> Optional[str]:
        """
        Call LLM to generate code and extract from JSON response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The generated Python code or None if failed
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            logger.debug(f"Calling LLM with model: {self.llm_config.get('generation_model')}")
            response = self.llm_client.chat_complete(
                model=self.llm_config.get("generation_model"),
                messages=messages,
                temperature=float(self.llm_config.get("generation_temperature", 0.1)),
                max_tokens=int(self.llm_config.get("generation_max_tokens", 4096)),
            )
            
            # Try to parse as JSON and extract 'result' field
            result = self._extract_result_from_response(response)
            if result:
                logger.debug(f"LLM response parsed successfully (result length: {len(result)} chars)")
            else:
                logger.warning("LLM response is empty or invalid")
            return result
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return None
    
    def _extract_result_from_response(self, response: str) -> Optional[str]:
        """
        Extract the 'result' field from LLM response (JSON format).
        
        Args:
            response: The LLM response
            
        Returns:
            The code string from the 'result' field, or the raw response if not JSON
        """
        try:
            # Try parsing as JSON wrapper first.
            data = json.loads(response)
            if isinstance(data, dict):
                for key in ("result", "code", "python_code", "python"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return self._normalize_generated_code(value)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        # Fallback to raw content; still normalize common wrappers.
        return self._normalize_generated_code(response)

    def _normalize_generated_code(self, text: str) -> str:
        """Normalize model outputs into executable Python code."""
        if not isinstance(text, str):
            return ""

        code = text.strip()

        # Prefer fenced python block content when present.
        fenced_match = re.search(r"```(?:python|py)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            code = fenced_match.group(1).strip()
        else:
            # Remove malformed leftover fences.
            code = (
                code.replace("```python", "")
                .replace("```py", "")
                .replace("```", "")
                .strip()
            )

        # Some prompts/examples use top-level "return ...", which is invalid in scripts.
        normalized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if line.startswith("return "):
                expr = stripped[len("return "):].strip()
                if expr:
                    normalized_lines.append(f"result = {expr}")
                else:
                    normalized_lines.append("result = None")
                continue
            normalized_lines.append(line)

        return "\n".join(normalized_lines).strip()
    
    def _execute_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Execute generated code (Python or SMT-LIB V2) and check for errors.
        
        Args:
            code: The code to execute (Python or SMT-LIB V2)
            
        Returns:
            Tuple of (success, error_message)
            - success: True if code executed successfully
            - error_message: Error message if failed, None if successful
        """
        if self.generate_code_language == 1:
            # Execute SMT-LIB V2 code
            return self._execute_smt_code(code)
        else:
            # Execute Python (z3) code
            return self._execute_python_code(code)
    
    def _execute_python_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Execute generated Python code and check for errors.
        Has a maximum timeout of 20 seconds.
        
        Args:
            code: The Python code to execute
            
        Returns:
            Tuple of (success, error_message)
        """
        def run_code():
            """Wrapper function for executing code with timeout"""
            logger.debug("Executing Python code in isolated namespace")
            # Create a local namespace for code execution with common libraries
            local_namespace = {}
            global_namespace = {
                "__builtins__": __builtins__,
                "z3": __import__("z3"),
                "json": __import__("json"),
            }
            
            exec(code, global_namespace, local_namespace)
            return local_namespace
        
        try:
            local_namespace = _run_code_with_timeout(run_code)
            
            # Check if result variable was set
            if "result" in local_namespace:
                logger.info("Code execution successful, 'result' variable is set")
                return True, None
            else:
                error_msg = "Code executed but 'result' variable not set"
                logger.warning(error_msg)
                return False, error_msg
        except TimeoutError as e:
            error_msg = f"Timeout: {str(e)}"
            logger.error(f"Code execution timed out: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Code execution failed: {error_msg}")
            return False, error_msg
    
    def _execute_smt_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Execute generated SMT-LIB V2 code and check for errors.
        Has a maximum timeout of 20 seconds.
        
        Args:
            code: The SMT-LIB V2 code to execute
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            import tempfile
            import subprocess
            
            logger.debug("Executing SMT-LIB V2 code")
            
            # Write SMT code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Try to use z3 command line tool if available with 20 second timeout
                result = subprocess.run(
                    ['z3', '-smt2', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=EXECUTION_TIMEOUT
                )
                
                output = result.stdout.strip()
                logger.debug(f"SMT solver output: {output}")
                
                # Check if the solver returned 'sat', 'unsat', or 'unknown'
                if 'sat' in output or 'unsat' in output or 'unknown' in output:
                    logger.info("SMT-LIB V2 code executed successfully")
                    return True, None
                else:
                    error_msg = f"SMT solver returned unexpected output: {output}"
                    logger.warning(error_msg)
                    return False, error_msg
                    
            except subprocess.TimeoutExpired:
                error_msg = f"SMT execution exceeded {EXECUTION_TIMEOUT} seconds timeout"
                logger.error(error_msg)
                return False, error_msg
            except FileNotFoundError:
                # z3 command line tool not available, try using z3 Python API
                logger.debug("z3 command-line tool not found, using z3 Python API as fallback")
                return self._execute_smt_with_python_z3(code)
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"SMT execution failed: {error_msg}")
            return False, error_msg
    
    def _execute_smt_with_python_z3(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Execute SMT-LIB V2 code using z3 Python API.
        Has a maximum timeout of 20 seconds.
        
        Args:
            code: The SMT-LIB V2 code to execute
            
        Returns:
            Tuple of (success, error_message)
        """
        def parse_and_execute():
            """Wrapper function for parsing SMT code with timeout"""
            import z3
            logger.debug("Parsing SMT code with z3 Python API")
            z3.parse_smt2_string(code)
            return True
        
        try:
            _run_code_with_timeout(parse_and_execute)
            logger.info("SMT-LIB V2 code parsed and executed successfully")
            return True, None
        except TimeoutError as e:
            error_msg = f"Timeout: {str(e)}"
            logger.error(f"SMT execution with z3 API timed out: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"SMT execution with z3 API failed: {error_msg}")
            return False, error_msg
    
    def _regenerate_with_error_handling(
        self,
        generated_code: str,
        error_message: str,
        account_data: Dict,
        instruct_data: Dict,
    ) -> Tuple[str, bool, Optional[str], int]:
        """
        Attempt to regenerate code when initial generation fails.
        
        Args:
            generated_code: The initially generated code that failed
            error_message: The error message from execution
            account_data: Account information
            instruct_data: Instruction data
            
        Returns:
            Tuple of (generated_code, success, error_message, retries_used)
        """
        current_code = generated_code
        logger.warning(f"Starting code regeneration - Initial error: {error_message}")
        retries_used = 0
        
        for retry in range(self.regenerate_max_retries):
            retries_used += 1
            logger.info(f"Regeneration attempt {retry + 1}/{self.regenerate_max_retries}")
            
            # Build regenerate prompt
            regenerate_prompt = build_regenerate_prompt(
                code=current_code,
                error=error_message,
                prompt_config=self.prompt_config,
                generate_mode=self.generate_mode,
                generate_code_language=self.generate_code_language,
                account_data=account_data,
                instruct_data=instruct_data,
            )
            
            # Call LLM to generate fixed code
            logger.info(f"Calling LLM for code regeneration (attempt {retry + 1})")
            fixed_code = self._call_llm_for_code_generation(regenerate_prompt)
            
            if not fixed_code:
                logger.warning(f"Failed to get regenerated code from LLM on attempt {retry + 1}")
                continue
            
            logger.info(f"Regenerated code received (length: {len(fixed_code)} chars), testing execution")
            # Try to execute the fixed code
            success, error_msg = self._execute_code(fixed_code)
            
            if success:
                logger.info(f"Regenerated code executed successfully on attempt {retry + 1}")
                return fixed_code, True, None, retries_used
            
            logger.warning(f"Regenerated code failed on attempt {retry + 1}: {error_msg}")
            # Update for next iteration
            current_code = fixed_code
            error_message = error_msg or "Unknown error"
        
        # All retries exhausted
        logger.error(f"Regeneration failed after {self.regenerate_max_retries} retries. Last error: {error_message}")
        return (
            current_code,
            False,
            f"Regeneration failed after {self.regenerate_max_retries} retries. Last error: {error_message}",
            retries_used,
        )
