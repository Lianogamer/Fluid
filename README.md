# Fluid

Fluid is a new, small, statically typed programming language experiment. This
starter compiler translates Fluid into C and uses an installed C compiler to
produce a native executable. That gives the prototype access to native
optimization; it does not yet establish that Fluid is faster than Rust or C++.

## Requirements

- Python 3.10 or newer
- A C compiler available on `PATH`: `cc`, `clang`, `gcc`, or Microsoft `cl`

On Windows, install LLVM/Clang with:

```powershell
winget install --id LLVM.LLVM --exact
```

Open a new terminal after installation so the updated `PATH` is available.

## Try it

```text
python fluid.py check examples/hello.fl
python fluid.py emit-c examples/hello.fl
python fluid.py build examples/hello.fl
python fluid.py run examples/hello.fl
python fluid.py run examples/sum.fl --native-tune
```

Use `-o program` with `build` to choose an output path, and `--cc "compiler
command"` to select a compiler explicitly.

## Language starter

```fluid
fn twice(value: i64) -> i64 {
    return value * 2;
}

fn main() -> i64 {
    let answer = twice(21);
    let scores: [i64; 3] = [98, 87, 100];
    print(scores[0]);
    if answer == 42 {
        print("the answer is");
        print(answer);
    }
    return 0;
}
```

The current subset includes `i64`, `f64`, `bool`, `str`, and `unit`; immutable
`let` and mutable `var` bindings; assignment to `var`; functions; arithmetic
and comparisons; `if`/`else if`/`else`; `while`; exclusive integer ranges
with `for index in start..end`; `break` and `continue`; `print`; and `return`.
Range bounds must be `i64` and are evaluated once, before the loop. The loop
variable is immutable. Fixed-size arrays use `[type; length]` annotations or
inferred array literals, for example `let scores: [i64; 3] = [98, 87, 100];`.
Array reads and mutable-array writes are bounds-checked at runtime. Array
parameters, return values, nested arrays, and whole-array assignments are not
supported yet. `int`, `float`, and `string` are aliases for `i64`, `f64`, and
`str`. Strings support content-based `==` and `!=` comparisons. Built-ins
include `len`, `abs`, `min`, `max`, `clamp`, `sign`, `sqrt`, `pow`, `floor`,
`ceil`, `round`, `contains`, `starts_with`, `ends_with`, and `is_empty`.
`assert(condition);` exits with a source-positioned diagnostic when the
condition is false. `len(string)` returns its UTF-8 byte length;
`len(array_variable)` returns the fixed element count at compile time.
`abs(i64_min)` exits with an overflow diagnostic. Function arguments and return
types are explicit, while local types can be inferred or written after the
variable name. Blocks use braces, statements end in semicolons, and `//` starts
a comment.

The native build uses optimization level 3 by default. `build` and `run`
accept `--native-tune` to add GCC/Clang's `-march=native`; this may improve
performance on the build machine but creates binaries that may not run on older
or different CPUs. The compiler automatically links the C math library where
needed for `sqrt`.

`examples/sum.fl` is a dependency-heavy CPU-loop sample for trying native
builds. It is not a cross-language benchmark; compare optimized binaries using
the same workload and your target machines before drawing performance
conclusions.

This is the foundation, not a finished replacement for mature systems
languages: ownership, memory safety, modules, generics, a package manager,
diagnostics, and benchmarking still need to be designed and implemented.

## Tests

```text
python -m unittest discover -s tests
```
