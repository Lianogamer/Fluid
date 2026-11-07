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
Win32 window programs also need a Windows SDK and are supported only by native
Windows builds.

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

## Projects and dependencies

`fluid init my-app` creates a project with `fluid.json` and `src/main.fl`.
Commands that compile or run accept either a source file or a project directory.
The manifest is JSON and records the entry point and direct dependencies:

```json
{
  "name": "my-app",
  "version": "0.1.0",
  "entry": "src/main.fl",
  "dependencies": {
    "math": { "path": "../math" }
  }
}
```

Add a local package with `fluid add math --path ../math`, or a Git package with
`fluid add math --git https://example.com/math.git --rev <full-commit-hash>`.
Git dependencies must pin a complete commit hash; checkouts are cached under
`.fluid/deps/`. `fluid install` resolves dependencies and writes `fluid.lock`.
Packages export public functions with `pub fn`, imported as
`use math::function;`. Local path and pinned Git dependencies are supported;
transitive dependencies and dependency-defined types are not supported yet.
The `fluid.lock` file and `fluid.json` should be committed to keep project
inputs reproducible.

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
include `len`, `push`, `pop`, `insert`, `remove`, `reserve`, `shrink_to_fit`,
`capacity`, `clear`, `slice`, `arena`, `arena_i64`, `arena_f64`, `arena_bool`,
`arena_str`, `abs`, `min`, `max`, `clamp`, `sign`, `sqrt`, `pow`, `floor`,
`ceil`, `round`, `contains`, `starts_with`, `ends_with`, and `is_empty`.
`assert(condition);` exits with a source-positioned diagnostic when the
condition is false. `len(string)` returns its UTF-8 byte length;
`len(array_variable)` returns the fixed element count at compile time, while
`len(vector_variable)` reads its current element count.
`abs(i64_min)` exits with an overflow diagnostic. Function arguments and return
types are explicit, while local types can be inferred or written after the
variable name. Blocks use braces, statements end in semicolons, and `//` starts
a comment. Payload-free enums and exhaustive `match` statements are also
supported. `String` is an owned, mutable UTF-8 byte string; `str` remains a
borrowed string literal/view.

## Strings and ownership

Create owned strings with `string_from`, and make explicit copies with
`string_clone`. Mutable `var` bindings can be extended with `string_append` or
emptied with `string_clear`; `string_concat` and `string_slice` return new owned
values. `len(String)` reports UTF-8 bytes, while `string_slice` uses byte
offsets and rejects offsets inside a multi-byte UTF-8 character. `String`
values are freed at scope exit, cannot be implicitly copied, and can be passed
to functions by shared reference (`&String`). Owned strings cannot yet be
returned from functions or stored inside structs/arrays.

```fluid
fn byte_length(value: &String) -> i64 {
    return len(value);
}

fn main() -> i64 {
    var greeting = string_from("Hello");
    string_append(greeting, ", Fluid!");
    let copy = string_clone(greeting);
    assert(byte_length(copy) == 13);
    print(copy);
    return 0;
}
```

## Generic functions

Generic functions use type parameters in their signature. Fluid infers each
parameter from the call arguments and generates a separate native function for
each concrete type combination:

```fluid
fn identity<T>(value: T) -> T {
    return value;
}

fn main() -> i64 {
    let answer = identity(42);
    return answer - 42;
}
```

All occurrences of the same type parameter in a call must resolve to the same
type. Explicit type arguments, generic structs, and trait bounds are not
supported yet.

## Traits and dynamic dispatch

Traits declare shared-reference methods, and `impl Trait for Type` provides
their implementations. A `dyn Trait` parameter is a non-owning value containing
a data pointer and a vtable; concrete struct arguments are coerced to it
automatically. Calls through the trait object use dynamic dispatch:

```fluid
struct Circle { radius: i64 }

trait Measure {
    fn area(self: &Self) -> i64;
}

impl Measure for Circle {
    fn area(self: &Circle) -> i64 {
        return self.radius * self.radius;
    }
}

fn measure(shape: dyn Measure) -> i64 {
    return shape.area();
}
```

Trait methods currently require `self: &Self`, primitive non-receiver
arguments, and primitive or `unit` return types. Generic methods, trait bounds,
trait objects in return values or struct fields, and traits in imported modules
or dependencies are not supported yet.

## Enums and match

Declare an enum before functions and use `Type::Variant` to construct a value.
`match` requires exactly one arm for every variant; missing, duplicate, or
foreign variants are compile-time errors. Enum types stay distinct from `i64`,
even though the current native backend represents their payload-free variants
as integer discriminants:

```fluid
enum Status { Queued, Running, Complete }

fn next(status: Status) -> Status {
    match status {
        Status::Queued => { return Status::Running; }
        Status::Running => { return Status::Complete; }
        Status::Complete => { return Status::Complete; }
    }
}

fn main() -> i64 {
    let status = next(Status::Queued);
    match status {
        Status::Queued => { return 1; }
        Status::Running => { return 2; }
        Status::Complete => { return 0; }
    }
}
```

Enums may be used in function parameters, return values, and struct fields.
Payload-bearing variants and enum imports from modules are not supported yet.

## Modules and imports

Each `mod name;` declaration loads `name.fl` beside the declaring file; nested
modules use matching subdirectories. Imported functions are private by default
and must be marked `pub` to be used outside their module. Use
`use math::twice;` to bring a public function into the current file's scope:

```text
main.fl
math.fl
```

```fluid
// math.fl
fn double(value: i64) -> i64 { return value * 2; }
pub fn twice(value: i64) -> i64 { return double(value); }
```

```fluid
// main.fl
mod math;
use math::twice;

fn main() -> i64 { return twice(21); }
```

`use` imports functions, not modules or structs yet. Calls within a module can
use its private helpers directly. Module declarations and imports are resolved
relative to the source file, and imported function names may not shadow a
function defined locally.

## Parallel loops

`parallel for index in start..end { ... }` divides the exclusive range into
static chunks and runs them on up to 64 available CPU cores. The call waits for
all workers before continuing. Each worker owns a temporary arena, available
inside the body through `worker_arena()`:

```fluid
fn main() -> i64 {
    let factor = 3;
    var results: [i64] = [0, 0, 0, 0];
    parallel for index in 0..len(results) {
        var scratch = arena_i64(worker_arena(), 1);
        scratch[0] = index * factor;
        results[index] = scratch[0];
    }
    return results[3];
}
```

Outer values captured by the body must be immutable. The only permitted write
to outer state is an indexed write to a mutable array or vector using the
parallel loop index directly, such as `results[index] = value`; the output
cannot also be read in the body. `return`, `break`, and `continue` are not
supported inside a parallel body. Worker arenas are reclaimed when the entire
parallel loop finishes, not after each iteration. `print` calls from workers
may appear in any order. Spawning workers has overhead, so ordinary `for`
loops remain preferable for small workloads. Native builds use platform
threads (Windows threads or POSIX threads). `examples/parallel.fl` shows a
complete workload with a checksum and per-worker arena storage.

## Task pools and channels

Create a fixed-size pool with `task_pool(worker_count)`, submit a top-level
function with a typed task operation, and retrieve its result by task ID:

```fluid
fn square(value: i64) -> i64 {
    return value * value;
}

fn main() -> i64 {
    var pool = task_pool(4);
    let task = task_submit_i64(pool, square, 12);
    let result = task_result_i64(pool, task);
    task_pool_wait(pool);
    assert(result == 144);
    return 0;
}
```

`task_submit_i64`, `task_submit_f64`, and `task_submit_bool` require matching
argument and return types. Each accepts a function with one argument; an
optional channel argument lets a worker function receive `&channel_type` as
its first parameter. `task_result_i64/f64/bool` waits for that task. Use
`task_pool_wait` to wait for all submitted work and `task_pool_close` to wait
and reject later submissions. Pools accept 1-64 workers.

`channel_i64`, `channel_f64`, and `channel_bool` create bounded FIFO channels
with positive capacities. `channel_send_*` and `channel_receive_*` block when
the buffer is full or empty; `channel_try_receive_*` returns immediately.
Send returns false after close, while receive drains buffered items and then
returns false. Channels and pools are non-copyable local resources. On scope
exit, Fluid closes channels and waits for active tasks before releasing
channels and pools.

## Native Windows

The Win32 window API is available in native Windows builds:

```fluid
fn main() -> i64 {
    var app = window_create("Fluid window", 800, 600);
    window_set_title(app, "Move or resize me");
    window_move(app, 120, 100);
    window_resize(app, 800, 600);
    window_show(app);

    while window_wait(app) {
    }
    return 0;
}
```

`window_create(title, width, height)` creates a hidden window with the requested
client-area dimensions. Use `window_show` or `window_hide` to control visibility.
`window_move(window, x, y)` sets the outer window's screen position, while
`window_resize(window, width, height)` sets its client-area dimensions.
`window_set_title` updates the UTF-8 title. `window_x` and `window_y` query the
outer position; `window_width` and `window_height` query the client area.
`window_poll` processes queued messages without blocking; `window_wait` waits
for a message and then processes pending events, so it is suitable for a
non-busy event loop. They return `false` once the window is closed and queued
events have been drained. The X button closes the native handle and queues a
close event. `window_close` closes it explicitly, and `window_is_open` checks
whether it is still open.

`window_next_event(window)` returns a `window_event` with `kind`, `key`, `x`,
`y`, `button`, `delta`, `text`, `control_id`, `checked`, `width`, and `height`
fields. Compare `kind`
with `event_key_down()`, `event_key_up()`, `event_mouse_move()`,
`event_mouse_down()`, `event_mouse_up()`, `event_mouse_wheel()`, `event_text()`,
`event_resize()`, `event_close()`, `event_control()`, or `event_none()`. For a
control click, `control_id` contains the ID supplied when creating that control,
and `checked` reports its checkbox state (false for buttons). `key` is a Win32 virtual
key code; mouse buttons are 1 (left), 2 (right), and 3 (middle); wheel `delta`
uses Win32 wheel units (120 per notch); and `text` is a UTF-16 code unit. Mouse coordinates are
relative to the client area. The queue holds up to 256 events; consecutive
mouse-move and resize events are coalesced, and overflow is reported rather
than silently dropping input.

The drawing API uses a resizable off-screen GDI backbuffer:
`window_clear(window, rgb)`, `window_fill_rect(window, x, y, width, height, rgb)`,
`window_draw_rect(...)`, `window_fill_ellipse(...)`, `window_draw_ellipse(...)`,
`window_draw_line(window, x1, y1, x2, y2, rgb)`, `window_draw_text(window, x, y, text, rgb)`,
and `window_present(window)`. Use `window_present_rect(window, x, y, width, height)`
to copy just a changed area from the backbuffer to the window. Ellipse bounds use
the same x, y, width, and height
arguments as rectangles; equal width and height produce a circle.
Colors are packed `0xRRGGBB` values (decimal literals also work). Draw after
processing events, then call `window_present`; Windows redraw requests reuse
the backbuffer, which is recreated after a resize.

Native controls can be added with `window_add_button(window, id, text, x, y,
width, height)`, `window_add_checkbox(...)`, `window_add_label(...)`, and
`window_add_textbox(...)`. Text-box edits generate `event_control()` events;
read or replace their contents with `window_textbox_text(window, id)` and
`window_set_textbox_text(window, id, text)`.
Control IDs must be unique within a window and in the range 1–65535. Button
and checkbox clicks arrive as `event_control()` events; use `event.control_id`
to identify them. Read or set a checkbox with
`window_checkbox_checked(window, id)` and
`window_set_checkbox_checked(window, id, checked)`. Toggle any control with
`window_set_control_enabled(window, id, enabled)`, and update a button, label,
or text box caption with `window_set_control_text(window, id, text)`.

Run the interactive mouse, wheel, native-controls, and drawing demo on a Windows
desktop with:

```powershell
python fluid.py run examples/ui_demo.fl
```

Windows are owned local resources: Fluid destroys them automatically at scope
exit, including return and loop control. Bind them with `var` when using
operations that can change the native window. Window handles cannot be copied,
returned, or passed to functions. These APIs require an interactive Windows
desktop and are not available in parallel worker bodies. See `examples/windows.fl`,
the interactive [`ui_demo.fl`](./examples/ui_demo.fl), and the native integration
test in `tests/programs/windows.fl`.

## Memory and dynamic vectors

`[type; length]` remains a fixed-size local array. `[type]` declares a
heap-backed dynamic vector of `i64`, `f64`, `bool`, or `str`; initialize it from
a literal or an empty list:

```fluid
fn main() -> i64 {
    var values: [i64] = [];
    reserve(values, 16);
    push(values, 42);
    insert(values, 0, 7);
    let first = remove(values, 0);
    print(values[0]);
    print(len(values));
    print(capacity(values));
    let last = pop(values);
    shrink_to_fit(values);
    clear(values);
    return 0;
}
```

Vectors grow automatically when `push` exceeds their capacity. `reserve`
requests a minimum capacity, `insert` adds an item at an index (including the
end), and `remove` deletes and returns the item at an existing index. `pop`
removes and returns the last item. `shrink_to_fit` releases unused capacity;
`clear` removes all items while retaining allocated capacity. Index reads,
writes, insertion positions, and removals are bounds-checked. A vector is
locally owned: its heap allocation is freed automatically when its lexical
scope exits, including `return`, `break`, and `continue`. Use `var` for vectors
that are mutated. Vector copies and vector return values are unsupported.
Empty literals require an explicit vector type, such as
`var values: [i64] = [];`.
`examples/memory.fl` demonstrates vector growth and cleanup.

Functions can borrow a vector with `values: &[i64]`, borrow it mutably with
`values: &mut [i64]`, or take ownership with `values: move [i64]`. Callers pass
vectors normally to borrowed parameters and write `move values` for an owning
parameter. Shared borrows cannot mutate; mutable borrows require a `var`
binding. Moving a vector clears the caller's handle, and later use of that
binding is rejected. Borrowed vectors and slices cannot be returned. Slices use
`slice(values, start, end)` and are non-owning views; `slice[i64]` is their
annotation type. The compiler prevents mutation or reallocation of a vector
while one of its slices is live. Bind read-only views with `let`; use `var` for
a view over mutable storage that should permit element writes.

`arena()` creates a local arena, and `arena_i64(arena, count)` (also
`arena_f64`, `arena_bool`, and `arena_str`) allocates a zero-initialized typed
slice from it; arena string elements start as empty strings. Arena chunks are
released together at scope exit. Arena handles are local-only; slices must not
be returned. Use `var` for the arena and its typed slices when writing
elements.

Structs with primitive fields use declarations such as
`struct Point { x: i64, y: i64 }` and literals such as
`Point { x: 3, y: 4 }`. Fields are type-checked and mutable through `var`
bindings. Declare structs before functions. Struct values can be passed to and
returned from functions by value. Struct fields currently cannot contain
vectors, slices, or arenas, and struct values are not yet vector elements.

The native build uses optimization level 3 by default. `build` and `run`
accept `--native-tune` to add GCC/Clang's `-march=native`; this may improve
performance on the build machine but creates binaries that may not run on older
or different CPUs. The compiler automatically links the C math library where
needed for `sqrt`.

`examples/sum.fl` is a dependency-heavy CPU-loop sample.
`examples/memory_bench.fl` measures vector reservation, one million pushes,
and a borrowed checksum; build it once with
`python fluid.py build examples/memory_bench.fl -o memory_bench.exe` and time
repeated runs of the resulting executable separately from compilation. Its
checksum should be `499999500000`; the printed capacity is an implementation
detail of the vector's growth policy.
Neither example is a cross-language benchmark; use equivalent implementations,
compiler optimization settings, and target machines before drawing performance
conclusions.

This is the foundation, not a finished replacement for mature systems
languages: nested/escaping lifetimes, stronger memory safety guarantees,
generic trait bounds, richer package management, and broader benchmarking
still need to be designed and implemented.

## Tests

```text
python -m unittest discover -s tests
```

`tests/programs/` contains end-to-end programs written in Fluid. The test
runner executes them through the CLI and native compiler, checks their
assertions and output, and also verifies runtime assertion failures and
parallel-loop compile errors. Native execution tests are skipped when no C
compiler is available.
