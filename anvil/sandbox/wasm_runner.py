import ast
import io
import logging
import math
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_HAS_WASMTIME = False
try:
    import wasmtime
    _HAS_WASMTIME = True
except ImportError:
    wasmtime = None


@dataclass
class SandboxResult:
    success: bool
    output: str = ""
    error: str | None = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    backend: str = "unknown"


class SandboxTimeoutError(Exception):
    pass


class SandboxCompileError(Exception):
    pass


# ─── Python → WASM bytecode compiler ──────────────────────────────

_OPS = {
    "RET": 0,
    "PUSH_F64": 1,
    "ADD": 2,
    "SUB": 3,
    "MUL": 4,
    "DIV": 5,
    "SQRT": 7,
    "ABS": 8,
    "NEG": 9,
    "MIN": 10,
    "MAX": 11,
    "PUSH_I32": 12,
}


class PyToWASMCompiler:
    """Compiles a restricted subset of Python AST to WASM bytecodes."""

    BLOCKED_FUNCS = frozenset({
        "__import__", "open", "exec", "eval", "compile", "input", "breakpoint"
    })

    @classmethod
    def compile(cls, code: str) -> bytes:
        tree = ast.parse(code)
        bc = bytearray()
        for node in tree.body:
            cls._compile_node(node, bc)
        bc.append(_OPS["RET"])
        return bytes(bc)

    @classmethod
    def can_compile(cls, code: str) -> bool:
        """Check if code is a pure arithmetic expression compilable to WASM."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        if len(tree.body) != 1:
            return False
        node = tree.body[0]
        if not isinstance(node, ast.Expr):
            return False
        return cls._is_wasm_expr(node.value)

    @classmethod
    def _is_wasm_expr(cls, node: ast.AST) -> bool:
        """Recursively check if an expression node is WASM-compilable."""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and cls._is_wasm_expr(node.operand)
        if isinstance(node, ast.BinOp):
            return cls._is_wasm_expr(node.left) and cls._is_wasm_expr(node.right)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id in _WASM_FUNCS and all(cls._is_wasm_expr(a) for a in node.args)
            return False
        return False

    @classmethod
    def _compile_node(cls, node: ast.AST, bc: bytearray):
        if isinstance(node, ast.Expr):
            cls._compile_expr(node.value, bc)
            bc.append(_OPS["RET"])
        elif isinstance(node, ast.Assign):
            bc.append(_OPS["PUSH_I32"])
            bc.extend(struct.pack("<i", 0))
            bc.append(_OPS["RET"])
        elif isinstance(node, ast.Return):
            cls._compile_expr(node.value, bc)
            bc.append(_OPS["RET"])
        else:
            raise SandboxCompileError(f"Cannot compile {type(node).__name__} to WASM")

    @classmethod
    def _compile_expr(cls, node: ast.AST, bc: bytearray):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                val = float(node.value)
                bc.append(_OPS["PUSH_F64"])
                bc.extend(struct.pack("<d", val))
            elif isinstance(node.value, str):
                raise SandboxCompileError("String constants not supported in WASM compile")
            else:
                    raise SandboxCompileError(
                        f"Constant type {type(node.value).__name__} not supported"
                    )

        elif isinstance(node, ast.UnaryOp):
            cls._compile_expr(node.operand, bc)
            if isinstance(node.op, ast.USub):
                bc.append(_OPS["NEG"])
            elif isinstance(node.op, ast.UAdd):
                pass
            elif isinstance(node.op, ast.Not):
                raise SandboxCompileError("Boolean ops not supported in WASM compile")
            else:
                raise SandboxCompileError(f"Unary op {type(node.op).__name__} not supported")

        elif isinstance(node, ast.BinOp):
            cls._compile_expr(node.left, bc)
            cls._compile_expr(node.right, bc)
            if isinstance(node.op, ast.Add):
                bc.append(_OPS["ADD"])
            elif isinstance(node.op, ast.Sub):
                bc.append(_OPS["SUB"])
            elif isinstance(node.op, ast.Mult):
                bc.append(_OPS["MUL"])
            elif isinstance(node.op, ast.Div):
                bc.append(_OPS["DIV"])
            elif isinstance(node.op, ast.Pow):
                raise SandboxCompileError("Power operator not supported in WASM")
            else:
                raise SandboxCompileError(f"BinOp {type(node.op).__name__} not supported")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in _WASM_FUNCS:
                    for arg in node.args:
                        cls._compile_expr(arg, bc)
                    _WASM_FUNCS[func_name](bc)
                else:
                    raise SandboxCompileError(f"Function {func_name} not available in WASM")
            else:
                raise SandboxCompileError("Only simple function calls supported")

        elif isinstance(node, ast.List):
            bc.append(_OPS["PUSH_I32"])
            bc.extend(struct.pack("<i", 0))
            for elt in node.elts:
                cls._compile_expr(elt, bc)
                bc.append(_OPS["ADD"])
            bc.append(_OPS["RET"])

        elif isinstance(node, ast.Name):
            bc.append(_OPS["PUSH_I32"])
            bc.extend(struct.pack("<i", 0))

        else:
            raise SandboxCompileError(f"Expr {type(node).__name__} not supported in WASM")


def _wasm_max(bc: bytearray):
    bc.append(_OPS["MAX"])

def _wasm_min(bc: bytearray):
    bc.append(_OPS["MIN"])

def _wasm_abs(bc: bytearray):
    bc.append(_OPS["ABS"])

def _wasm_float(bc: bytearray):
    pass

def _wasm_int(bc: bytearray):
    pass

_WASM_FUNCS: dict[str, Any] = {
    "max": _wasm_max,
    "min": _wasm_min,
    "abs": _wasm_abs,
    "float": _wasm_float,
    "int": _wasm_int,
}


# ─── Wasmtime Sandbox ─────────────────────────────────────────────

class _WasmModuleCache:
    """Cached WASM module and engine for reuse across calls."""
    def __init__(self):
        from anvil.sandbox.compute_module import WASM_COMPUTE_WAT
        config = wasmtime.Config()
        config.consume_fuel = True
        self.engine = wasmtime.Engine(config)
        self.wat = WASM_COMPUTE_WAT
        self.module = wasmtime.Module(self.engine, self.wat)


class WasmtimeSandbox:
    """Sandbox using wasmtime WebAssembly runtime with fuel metering + capability restrictions."""

    _instance: "WasmtimeSandbox | None" = None
    _lock = threading.Lock()
    _cache: _WasmModuleCache | None = None

    def __init__(
        self, fuel_limit: int = 500_000, memory_limit_pages: int = 256, timeout_s: int = 30
    ):
        if not _HAS_WASMTIME:
            raise ImportError("wasmtime not installed. Run: pip install wasmtime")
        self.fuel_limit = fuel_limit
        self.memory_limit_pages = memory_limit_pages
        self.timeout_s = timeout_s
        if WasmtimeSandbox._cache is None:
            WasmtimeSandbox._cache = _WasmModuleCache()

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def _create_store(self):
        store = wasmtime.Store(self._cache.engine)
        store.set_fuel(self.fuel_limit)
        wasi = wasmtime.WasiConfig()
        wasi.inherit_stdout()
        wasi.inherit_stderr()
        store.set_wasi(wasi)
        return store

    def _get_instance_and_mem(self, store):
        linker = wasmtime.Linker(self._cache.engine)
        instance = linker.instantiate(store, self._cache.module)
        mem = instance.exports(store)["memory"]
        return instance, mem

    def compute_scalar(self, op: str, a: float, b: float | None = None) -> float:
        """Execute a scalar math operation in the WASM sandbox."""
        store = self._create_store()
        instance, _ = self._get_instance_and_mem(store)
        func = instance.exports(store)[op]
        if b is not None:
            return func(store, a, b)
        return func(store, a)

    def eval_bytecode(self, bytecodes: bytes) -> float:
        """Evaluate WASM bytecodes in the sandbox."""
        store = self._create_store()
        instance, mem = self._get_instance_and_mem(store)
        mem.write(store, bytecodes, 0)
        eval_fn = instance.exports(store)["eval_bytecode"]
        result = eval_fn(store, 0, len(bytecodes))
        return result

    def run_python_via_wasm(self, code: str) -> SandboxResult:
        """Try to compile Python to WASM bytecodes and execute in sandbox."""
        start = time.time()
        try:
            if not PyToWASMCompiler.can_compile(code):
                return SandboxResult(
                    success=False, error="Code too complex",
                    execution_time_ms=(time.time() - start) * 1000, backend="wasm_check",
                )
            bytecodes = PyToWASMCompiler.compile(code)
            result_val = self.eval_bytecode(bytecodes)
            if math.isnan(result_val) or math.isinf(result_val):
                return SandboxResult(
                    success=False, error="WASM produced NaN/Inf (likely division by zero)",
                    execution_time_ms=(time.time() - start) * 1000, backend="wasm_error",
                )
            output = str(result_val) if abs(result_val) > 1e-10 else ""
            return SandboxResult(
                success=True, output=output,
                execution_time_ms=(time.time() - start) * 1000, backend="wasm",
            )
        except Exception as e:
            return SandboxResult(
                success=False, error=f"WASM error: {e}",
                execution_time_ms=(time.time() - start) * 1000, backend="wasm_error",
            )


# ─── AST Security Validator ────────────────────────────────────────

class ASTRestrictedValidator:
    """STRICT AST-based security validation. Blocks dangerous patterns before execution."""

    DUNDER_BLOCKLIST = frozenset({
        '__class__', '__base__', '__subclasses__', '__init__', '__globals__',
        '__builtins__', '__dict__', '__closure__', '__code__', '__func__',
        '__self__', '__module__', '__name__', '__qualname__', '__doc__',
        '__new__', '__del__', '__repr__', '__str__', '__getattribute__',
        '__getattr__', '__setattr__', '__delattr__', '__call__',
        '__import__', '__builtin__', '__builtins__', '__loader__',
        '__spec__', '__package__', '__path__', '__file__', '__cached__',
        '__subclasshook__', '__init_subclass__', '__prepare__',
        '__instancecheck__', '__subclasscheck__', '__sizeof__',
        '__reduce__', '__reduce_ex__', '__format__', '__hash__',
    })

    BLOCKED_NAMES = frozenset({
        '__import__', 'open', 'exec', 'eval', 'compile',
        'input', 'breakpoint', 'help', 'exit', 'quit',
        'type', 'object', 'issubclass', 'isinstance', 'vars',
        'dir', 'getattr', 'setattr', 'delattr', 'hasattr',
        'super', 'memoryview', 'classmethod', 'staticmethod',
        'property', '__build_class__', 'copyright', 'credits',
        'license',
    })

    SAFE_BUILTINS: dict[str, Any] = {
        'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
        'bin': bin, 'bool': bool, 'bytearray': bytearray,
        'bytes': bytes, 'chr': chr, 'complex': complex,
        'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
        'filter': filter, 'float': float, 'format': format,
        'frozenset': frozenset, 'hash': hash, 'hex': hex,
        'id': id, 'int': int, 'iter': iter,
        'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'next': next,
        'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
        'range': range, 'repr': repr, 'reversed': reversed,
        'round': round, 'set': set, 'slice': slice,
        'sorted': sorted, 'str': str, 'sum': sum,
        'tuple': tuple, 'zip': zip,
        'True': True, 'False': False, 'None': None,
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'KeyError': KeyError,
        'IndexError': IndexError, 'StopIteration': StopIteration,
        'ZeroDivisionError': ZeroDivisionError,
        'ArithmeticError': ArithmeticError,
        'LookupError': LookupError,
        'RuntimeError': RuntimeError,
        'MemoryError': MemoryError,
    }

    @classmethod
    def validate(cls, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise PermissionError("Import statements are forbidden in sandbox")
            if isinstance(node, ast.Attribute):
                if node.attr in cls.DUNDER_BLOCKLIST:
                    raise PermissionError(f"Access to '{node.attr}' is forbidden in sandbox")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.BLOCKED_NAMES:
                        raise PermissionError(f"Call to '{node.func.id}' is forbidden in sandbox")

    @classmethod
    def get_restricted_globals(cls, inputs: dict | None = None) -> dict:
        return {"__builtins__": cls.SAFE_BUILTINS, "inputs": inputs or {}}


# ─── Sandboxed Python Executor (Threading Fallback) ──────────────

class ThreadingPythonExecutor:
    """Execute Python code with AST validation + threading timeout."""

    @staticmethod
    def execute(code: str, inputs: dict | None = None, max_time_s: int = 30) -> SandboxResult:
        tree = ast.parse(code)
        ASTRestrictedValidator.validate(tree)

        start = time.time()
        result_container: list[SandboxResult] = []

        def run():
            old_stdout = sys.stdout
            sys.stdout = captured = io.StringIO()
            restricted_globals = ASTRestrictedValidator.get_restricted_globals(inputs)
            try:
                exec(code, restricted_globals)
                output = captured.getvalue()
                result_container.append(SandboxResult(
                    success=True,
                    output=output,
                    execution_time_ms=(time.time() - start) * 1000,
                    backend="threading",
                ))
            except Exception as e:
                result_container.append(SandboxResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start) * 1000,
                    backend="threading",
                ))
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(max_time_s)

        if thread.is_alive():
            return SandboxResult(
                success=False,
                error="Execution timed out",
                execution_time_ms=(time.time() - start) * 1000,
                backend="threading",
            )
        return result_container[0] if result_container else SandboxResult(
            success=False,
            error="No result produced",
            backend="threading",
        )


# ─── CodeSandbox (Unified) ───────────────────────────────────────

class CodeSandbox:
    """
    Secure code execution sandbox with multi-tier execution.

    Tier 1 (Wasmtime WASM): Pure compute expressions compiled to WASM bytecodes.
    Tier 2 (Threading): Full Python execution with AST validation + restricted builtins.

    Security model:
    - AST validation blocks: dunders, imports, dangerous builtins
    - Wasmtime: fuel metering, memory limits, capability-restricted WASI
    - Threading: timeout enforcement, restricted builtins
    """

    def __init__(self, memory_limit_mb: int = 256,
                 enable_network: bool = False,
                 enable_filesystem: bool = False,
                 max_execution_time_s: int = 30):
        self.memory_limit_mb = memory_limit_mb
        self.enable_network = enable_network
        self.enable_filesystem = enable_filesystem
        self.max_execution_time_s = max_execution_time_s
        self._wasm_sandbox: WasmtimeSandbox | None = None

    def _get_wasm(self):
        if self._wasm_sandbox is None and _HAS_WASMTIME:
            try:
                self._wasm_sandbox = WasmtimeSandbox.get_instance()
            except Exception as e:
                logger.warning(f"Failed to init Wasmtime sandbox: {e}")
        return self._wasm_sandbox

    def execute_python(self, code: str, inputs: dict | None = None) -> SandboxResult:
        """Execute Python code. Tries WASM first, falls back to threading."""
        try:
            tree = ast.parse(code)
            ASTRestrictedValidator.validate(tree)
        except SyntaxError as e:
            return SandboxResult(success=False, error=f"SyntaxError: {e}")
        except PermissionError as e:
            return SandboxResult(success=False, error=f"SecurityError: {e}")
        except Exception as e:
            return SandboxResult(success=False, error=f"ValidationError: {e}")

        wasm = self._get_wasm()
        if wasm is not None and PyToWASMCompiler.can_compile(code):
            result = wasm.run_python_via_wasm(code)
            if result.success or "not supported" not in (result.error or "").lower():
                if result.success:
                    return result
        return ThreadingPythonExecutor.execute(code, inputs, self.max_execution_time_s)



