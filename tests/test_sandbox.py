"""Tests for sandboxed code execution security."""

import struct
import time

import pytest
from anvil.sandbox.wasm_runner import CodeSandbox, SandboxPolicy


class TestCodeSandbox:
    def setup_method(self):
        self.sandbox = CodeSandbox(max_execution_time_s=5)

    def test_execute_safe_print(self):
        result = self.sandbox.execute_python('print("hello world")')
        assert result.success is True
        assert result.output.strip() == "hello world"

    def test_execute_safe_math(self):
        code = "result = sum([1, 2, 3, 4, 5])\nprint(result)"
        result = self.sandbox.execute_python(code)
        assert result.success is True
        assert result.output.strip() == "15"

    def test_execute_safe_sum_range(self):
        code = "print(sum(range(10)))"
        result = self.sandbox.execute_python(code)
        assert result.success is True
        assert result.output.strip() == "45"

    def test_execute_safe_list_operations(self):
        code = (
            "items = [x * 2 for x in range(5)]\n"
            "print(items)"
        )
        result = self.sandbox.execute_python(code)
        assert result.success is True
        assert result.output.strip() == "[0, 2, 4, 6, 8]"

    def test_execute_with_inputs(self):
        code = "print(inputs['name'])"
        result = self.sandbox.execute_python(code, inputs={"name": "Anvil"})
        assert result.success is True
        assert result.output.strip() == "Anvil"

    def test_blocks_import_statement(self):
        code = "import os\nprint('should not run')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_from_import(self):
        code = "from sys import path\nprint('should not run')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_open_call(self):
        code = "open('/etc/passwd')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_eval_call(self):
        code = "eval('2 + 2')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_exec_call(self):
        code = "exec('x = 1')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_compile_call(self):
        code = "compile('x = 1', '<string>', 'exec')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_importlib(self):
        code = "__import__('os')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_input_call(self):
        code = "input('prompt')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_subscript_dunder_chain(self):
        code = "().__class__.__base__.__subclasses__()"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_simple_dunder_class(self):
        code = "().__class__"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_dunder_base(self):
        code = "''.__class__.__base__"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_dunder_subclasses(self):
        code = "''.__class__.__base__.__subclasses__"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_dunder_init(self):
        code = "().__class__.__init__"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_blocks_dunder_globals(self):
        code = "().__class__.__init__.__globals__"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_type_not_available(self):
        code = "type('', (), {})"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "not defined" in result.error.lower() or "type" in result.error.lower()

    def test_object_not_available(self):
        code = "object"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "not defined" in result.error.lower()

    def test_issubclass_not_available(self):
        code = "issubclass(str, object)"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_vars_not_available(self):
        code = "vars()"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_dir_not_available(self):
        code = "dir(())"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_getattr_not_available(self):
        code = "getattr((), '__class__')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_hasattr_not_available(self):
        code = "hasattr((), '__class__')"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_super_not_available(self):
        code = "super()"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "forbidden" in result.error.lower() or "not defined" in result.error.lower()

    def test_timeout_infinite_loop(self):
        code = "while True:\n    pass"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_blocks_ast_syntax_error(self):
        code = "print("
        result = self.sandbox.execute_python(code)
        assert result.success is False

    def test_error_execution_returns_error_message(self):
        code = "1 / 0"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "division by zero" in result.error

    def test_error_undefined_variable(self):
        code = "print(undefined_var)"
        result = self.sandbox.execute_python(code)
        assert result.success is False
        assert "undefined_var" in result.error or "not defined" in result.error

    def test_execution_time_is_measured(self):
        code = "print('timing')"
        result = self.sandbox.execute_python(code)
        assert result.execution_time_ms > 0

    def test_output_contains_multiple_lines(self):
        code = "for i in range(3):\n    print(f'line {i}')"
        result = self.sandbox.execute_python(code)
        assert result.success is True
        lines = result.output.strip().split('\n')
        assert len(lines) == 3
        assert lines[0] == "line 0"
        assert lines[2] == "line 2"


class TestSandboxPolicy:
    def test_default_policy_is_restrictive(self):
        policy = SandboxPolicy()
        assert policy.filesystem == SandboxPolicy.ALLOW_NONE
        assert policy.network == SandboxPolicy.ALLOW_NONE
        assert policy.process_spawn is False
        assert policy.memory_limit_mb == 256
        assert policy.time_limit_s == 30

    def test_policy_allows_read_filesystem(self):
        policy = SandboxPolicy(filesystem=SandboxPolicy.ALLOW_READ)
        assert policy.filesystem == SandboxPolicy.ALLOW_READ

    def test_policy_allows_network(self):
        policy = SandboxPolicy(network=SandboxPolicy.ALLOW_ALL)
        assert policy.network == SandboxPolicy.ALLOW_ALL

    def test_custom_memory_limit(self):
        policy = SandboxPolicy(memory_limit_mb=512)
        assert policy.memory_limit_mb == 512

    def test_custom_time_limit(self):
        policy = SandboxPolicy(time_limit_s=60)
        assert policy.time_limit_s == 60


# ─── Wasmtime Sandbox Tests ──────────────────────────────────────

class TestWasmtimeSandbox:
    def test_compute_scalar_add(self):
        try:
            from anvil.sandbox.wasm_runner import WasmtimeSandbox
        except ImportError:
            pytest.skip("wasmtime not available")
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("add", 3.0, 4.0) == 7.0

    def test_compute_scalar_mul(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("mul", 3.0, 4.0) == 12.0

    def test_compute_scalar_sub(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("sub", 10.0, 3.0) == 7.0

    def test_compute_scalar_div(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("div", 10.0, 2.0) == 5.0

    def test_compute_scalar_sqrt(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("sqrt", 9.0) == 3.0

    def test_compute_scalar_abs(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        ws = WasmtimeSandbox.get_instance()
        assert ws.compute_scalar("abs", -5.0) == 5.0

    def test_bytecode_simple_add(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        import struct
        ws = WasmtimeSandbox.get_instance()
        bc = bytearray()
        bc.append(1)  # PUSH_F64
        bc.extend(struct.pack("<d", 3.0))
        bc.append(1)  # PUSH_F64
        bc.extend(struct.pack("<d", 4.0))
        bc.append(2)  # ADD
        bc.append(0)  # RET
        result = ws.eval_bytecode(bytes(bc))
        assert result == 7.0

    def test_bytecode_complex_expression(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        import struct
        ws = WasmtimeSandbox.get_instance()
        bc = bytearray()
        bc.append(1)
        bc.extend(struct.pack("<d", 2.0))
        bc.append(1)
        bc.extend(struct.pack("<d", 3.0))
        bc.append(4)  # MUL → 6.0
        bc.append(1)
        bc.extend(struct.pack("<d", 4.0))
        bc.append(2)  # ADD → 10.0
        bc.append(0)  # RET
        result = ws.eval_bytecode(bytes(bc))
        assert result == 10.0

    def test_bytecode_sqrt(self):
        from anvil.sandbox.wasm_runner import WasmtimeSandbox
        import struct
        ws = WasmtimeSandbox.get_instance()
        bc = bytearray()
        bc.append(1)
        bc.extend(struct.pack("<d", 25.0))
        bc.append(7)  # SQRT
        bc.append(0)
        result = ws.eval_bytecode(bytes(bc))
        assert result == 5.0

    def test_py_to_wasm_simple_math(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler, WasmtimeSandbox
        assert not PyToWASMCompiler.can_compile("print(sum([1,2,3]))")
        assert PyToWASMCompiler.can_compile("3 + 4")
        bc = PyToWASMCompiler.compile("3 + 4")
        ws = WasmtimeSandbox.get_instance()
        result = ws.eval_bytecode(bc)
        assert result == 7.0

    def test_py_to_wasm_can_compile_rejects_import(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert not PyToWASMCompiler.can_compile("import os")

    def test_py_to_wasm_can_compile_rejects_open(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert not PyToWASMCompiler.can_compile("open('/etc/passwd')")

    def test_py_to_wasm_can_compile_rejects_exec_eval(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert not PyToWASMCompiler.can_compile("exec('x=1')")
        assert not PyToWASMCompiler.can_compile("eval('2+2')")

    def test_py_to_wasm_can_compile_rejects_class_def(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert not PyToWASMCompiler.can_compile("class Foo: pass")

    def test_py_to_wasm_can_compile_rejects_func_def(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert not PyToWASMCompiler.can_compile("def f(): pass")

    def test_py_to_wasm_can_compile_simple_math(self):
        from anvil.sandbox.wasm_runner import PyToWASMCompiler
        assert PyToWASMCompiler.can_compile("1 + 2 * 3")
        assert PyToWASMCompiler.can_compile("-5 + 3")
        assert PyToWASMCompiler.can_compile("max(1, 2)")
        assert not PyToWASMCompiler.can_compile("sum([1,2,3])")  # needs iterable
        assert not PyToWASMCompiler.can_compile("print('hi')")  # print not supported

    def test_code_sandbox_wasm_path_or_fallback(self):
        """WASM path or threading fallback should both produce correct results."""
        sandbox = CodeSandbox(max_execution_time_s=5)
        result = sandbox.execute_python("print(sum([1, 2, 3]))")
        assert result.success is True
        assert result.output.strip() in ("6", "6.0")
        assert result.backend in ("wasm", "threading")
