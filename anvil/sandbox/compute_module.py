"""Embedded WebAssembly compute module (WAT) for sandboxed execution."""

WASM_COMPUTE_WAT = r"""(module
  (memory (export "memory") 1 256)

  (func (export "add") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.add)

  (func (export "sub") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.sub)

  (func (export "mul") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.mul)

  (func (export "div") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.div)

  (func (export "sqrt") (param f64) (result f64)
    local.get 0 f64.sqrt)

  (func (export "abs") (param f64) (result f64)
    local.get 0 f64.abs)

  (func (export "neg") (param f64) (result f64)
    local.get 0 f64.neg)

  (func (export "min") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.min)

  (func (export "max") (param f64 f64) (result f64)
    local.get 0 local.get 1 f64.max)

  ;; Bytecode evaluator
  (func (export "eval_bytecode") (param i32 i32) (result f64)
    (local f64 f64 f64 f64 i32 i32)
    block $done
      loop $main
        ;; if pc >= len: done
        local.get 6
        local.get 1
        i32.ge_s
        br_if $done

        ;; read opcode
        local.get 0
        local.get 6
        i32.add
        i32.load8_u
        local.set 7

        ;; PUSH_F64 (opcode 1)
        block $skip1
          local.get 7
          i32.const 1
          i32.ne
          br_if $skip1
          local.get 0
          local.get 6
          i32.const 1
          i32.add
          i32.add
          f64.load
          local.set 5
          local.get 3
          local.set 4
          local.get 2
          local.set 3
          local.get 5
          local.set 2
          local.get 6
          i32.const 9
          i32.add
          local.set 6
          br $main
        end

        ;; ADD (opcode 2)
        block $skip2
          local.get 7
          i32.const 2
          i32.ne
          br_if $skip2
          local.get 3
          local.get 2
          f64.add
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; SUB (opcode 3)
        block $skip3
          local.get 7
          i32.const 3
          i32.ne
          br_if $skip3
          local.get 3
          local.get 2
          f64.sub
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; MUL (opcode 4)
        block $skip4
          local.get 7
          i32.const 4
          i32.ne
          br_if $skip4
          local.get 3
          local.get 2
          f64.mul
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; DIV (opcode 5)
        block $skip5
          local.get 7
          i32.const 5
          i32.ne
          br_if $skip5
          local.get 3
          local.get 2
          f64.div
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; SQRT (opcode 7)
        block $skip7
          local.get 7
          i32.const 7
          i32.ne
          br_if $skip7
          local.get 2
          f64.sqrt
          local.set 2
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; ABS (opcode 8)
        block $skip8
          local.get 7
          i32.const 8
          i32.ne
          br_if $skip8
          local.get 2
          f64.abs
          local.set 2
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; NEG (opcode 9)
        block $skip9
          local.get 7
          i32.const 9
          i32.ne
          br_if $skip9
          local.get 2
          f64.neg
          local.set 2
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; MIN (opcode 10)
        block $skip10
          local.get 7
          i32.const 10
          i32.ne
          br_if $skip10
          local.get 3
          local.get 2
          f64.min
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; MAX (opcode 11)
        block $skip11
          local.get 7
          i32.const 11
          i32.ne
          br_if $skip11
          local.get 3
          local.get 2
          f64.max
          local.set 2
          local.get 4
          local.set 3
          local.get 6
          i32.const 1
          i32.add
          local.set 6
          br $main
        end

        ;; unknown opcode: advance pc
        local.get 6
        i32.const 1
        i32.add
        local.set 6
        br $main
      end
    end
    local.get 2)
)
"""

__all__ = ["WASM_COMPUTE_WAT"]
