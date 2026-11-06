# Fluid

Fluid is a new, small, statically typed programming language experiment. This
starter compiler translates Fluid into C and uses an installed C compiler to
produce a native executable. That gives the prototype access to native
optimization; it does not yet establish that Fluid is faster than Rust or C++.

## Requirements

- Python 3.10 or newer
- A C compiler available on `PATH`: `cc`, `clang`, `gcc`, or Microsoft `cl`

## Try it

```text
python fluid.py check examples/hello.fl
python fluid.py emit-c examples/hello.fl
python fluid.py build examples/hello.fl
python fluid.py run examples/hello.fl
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
    if answer == 42 {
        print("the answer is");
        print(answer);
    }
    return 0;
}
```

The current subset includes `i64`, `f64`, `bool`, `str`, and `unit`; immutable
`let` and mutable `var` bindings; assignment to `var`; functions; arithmetic
and comparisons; `if`/`else if`/`else`; `while` with `break` and `continue`;
`print`; and `return`. `int`, `float`, and `string` are aliases for `i64`,
`f64`, and `str`. Function arguments and return types are explicit, while
local types can be inferred or written after the variable name. Blocks use
braces, statements end in semicolons, and `//` starts a comment.

This is the foundation, not a finished replacement for mature systems
languages: ownership, memory safety, modules, generics, a package manager,
diagnostics, and benchmarking still need to be designed and implemented.

## Tests

```text
python -m unittest discover -s tests
```
