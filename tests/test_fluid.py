import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fluid import (
    Compiler, FluidError, Parser, add_project_dependency, build_binary, compiler_command,
    initialize_project, install_project, native_compile_args, parse_project, parse_source,
)


class FluidCompilerTests(unittest.TestCase):
    def compile_source(self, source: str) -> str:
        return Compiler(Parser(source).parse()).compile()

    def test_emits_native_c_for_functions_and_control_flow(self):
        result = self.compile_source(
            "fn double(value: i64) -> i64 { return value * 2; }"
            "fn main() -> i64 { let result = double(21); "
            "if result == 42 { print(result); } return 0; }"
        )
        self.assertIn("long long fluid_double(long long v_value);", result)
        self.assertIn("if (v_result == 42LL)", result)
        self.assertNotIn("if ((v_result == 42LL))", result)
        self.assertIn('printf("%lld\\n", v_result);', result)
        self.assertIn("return (int)fluid_main();", result)

    def test_type_errors_are_reported(self):
        with self.assertRaisesRegex(FluidError, "condition must have type bool"):
            self.compile_source("fn main() -> i64 { if 1 { return 0; } return 0; }")

    def test_unknown_variables_are_reported(self):
        with self.assertRaisesRegex(FluidError, "unknown variable"):
            self.compile_source("fn main() -> i64 { print(missing); return 0; }")

    def test_unit_main_gets_a_c_entrypoint_wrapper(self):
        result = self.compile_source("fn main() -> unit { print(true); }")
        self.assertIn("void fluid_main(void)", result)
        self.assertIn("fluid_main();", result)
        self.assertIn("return 0;", result)

    def test_var_can_be_reassigned(self):
        result = self.compile_source(
            "fn main() -> i64 { var count = 2; count = count - 1; return count; }"
        )
        self.assertIn("long long v_count = 2LL;", result)
        self.assertIn("v_count = (v_count - 1LL);", result)

    def test_let_cannot_be_reassigned(self):
        with self.assertRaisesRegex(FluidError, "immutable let binding"):
            self.compile_source("fn main() -> i64 { let count = 2; count = 1; return 0; }")

    def test_comment_at_end_of_file_is_accepted(self):
        result = self.compile_source("fn main() -> i64 { return 0; } // final comment")
        self.assertIn("fluid_main", result)

    def test_source_file_with_utf8_bom_is_accepted(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bom.fl"
            path.write_bytes(b"\xef\xbb\xbffn main() -> i64 { return 0; }")
            source = parse_source(path)
        self.assertIn("fn main()", source)
        self.assertIn("fluid_main", self.compile_source(source))

    def test_module_imports_public_functions_and_private_helpers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.fl").write_text(
                "mod math; use math::twice; "
                "fn main() -> i64 { return twice(21); }",
                encoding="utf-8",
            )
            (root / "math.fl").write_text(
                "fn helper(value: i64) -> i64 { return value * 2; } "
                "pub fn twice(value: i64) -> i64 { return helper(value); }",
                encoding="utf-8",
            )
            source = parse_project(root / "main.fl")
        generated = Compiler(source).compile()
        self.assertIn("fluid_math__twice(long long v_value)", generated)
        self.assertIn("return fluid_math__twice(21LL);", generated)

    def test_private_module_function_cannot_be_imported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.fl").write_text(
                "mod math; use math::hidden; "
                "fn main() -> i64 { return hidden(); }",
                encoding="utf-8",
            )
            (root / "math.fl").write_text(
                "fn hidden() -> i64 { return 7; }",
                encoding="utf-8",
            )
            source = parse_project(root / "main.fl")
        with self.assertRaisesRegex(FluidError, "is private; mark it pub"):
            Compiler(source)

    def test_unimported_module_function_is_not_in_scope(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.fl").write_text(
                "mod math; fn main() -> i64 { return twice(21); }",
                encoding="utf-8",
            )
            (root / "math.fl").write_text(
                "pub fn twice(value: i64) -> i64 { return value * 2; }",
                encoding="utf-8",
            )
            source = parse_project(root / "main.fl")
        with self.assertRaisesRegex(FluidError, "unknown function 'twice'"):
            Compiler(source).compile()

    def test_task_submission_checks_worker_signature(self):
        source = (
            "fn worker(value: f64) -> f64 { return value; } "
            "fn main() -> i64 { var pool = task_pool(1); "
            "let id = task_submit_i64(pool, worker, 1); return 0; }"
        )
        with self.assertRaisesRegex(FluidError, "must have signature fn\\(i64\\) -> i64"):
            self.compile_source(source)

    def test_channels_cannot_be_copied(self):
        source = (
            "fn main() -> i64 { let channel = channel_i64(1); "
            "let copied = channel; return 0; }"
        )
        with self.assertRaisesRegex(FluidError, "channels cannot be copied"):
            self.compile_source(source)

    def test_enum_match_is_exhaustive_and_emits_all_branches(self):
        source = (
            "enum Direction { North, East, South } "
            "struct Route { direction: Direction, distance: i64 } "
            "fn next(direction: Direction) -> Direction { "
            "match direction { "
            "Direction::North => { return Direction::East; } "
            "Direction::East => { return Direction::South; } "
            "Direction::South => { return Direction::North; } } } "
            "fn main() -> i64 { "
            "let route = Route { direction: next(Direction::North), distance: 4 }; "
            "match route.direction { "
            "Direction::North => { return 1; } "
            "Direction::East => { return route.distance; } "
            "Direction::South => { return 3; } } }"
        )
        generated = self.compile_source(source)
        self.assertIn("fluid_next(long long v_direction)", generated)
        self.assertIn("struct fluid_Route {", generated)
        self.assertIn("if (fluid_match_value_0 == 0LL)", generated)
        self.assertIn("fluid_match_value_1", generated)

    def test_enum_match_rejects_missing_variants(self):
        source = (
            "enum State { Ready, Running, Done } "
            "fn main() -> i64 { match State::Ready { "
            "State::Ready => { return 0; } } }"
        )
        with self.assertRaisesRegex(FluidError, "non-exhaustive match.*State::Running"):
            self.compile_source(source)

    def test_enum_match_rejects_duplicate_variants(self):
        source = (
            "enum State { Ready, Done } "
            "fn main() -> i64 { match State::Ready { "
            "State::Ready => { return 0; } "
            "State::Ready => { return 1; } "
            "State::Done => { return 2; } } }"
        )
        with self.assertRaisesRegex(FluidError, "duplicate match arm"):
            self.compile_source(source)

    def test_distinct_enum_types_cannot_be_compared(self):
        source = (
            "enum First { One } enum Second { One } "
            "fn main() -> i64 { "
            "if First::One == Second::One { return 1; } return 0; }"
        )
        with self.assertRaisesRegex(FluidError, "cannot be used with enum:First and enum:Second"):
            self.compile_source(source)

    def test_owned_strings_emit_mutations_borrowing_and_cleanup(self):
        source = (
            "fn length(value: &String) -> i64 { return len(value); } "
            "fn main() -> i64 { "
            'var text: String = string_from("h\\u00e9"); '
            'string_append(text, "llo"); '
            "let copy = string_clone(text); "
            "let tail = string_slice(copy, 3, 5); "
            'let expected = string_from("lo"); '
            "assert(length(text) == 5); "
            "assert(tail == expected); "
            "return 0; }"
        )
        generated = self.compile_source(source)
        self.assertIn("typedef struct { size_t len; size_t capacity; char *data; } fluid_string;",
                      generated)
        self.assertIn("fluid_string_free(&v_text);", generated)
        self.assertIn("fluid_string_free(&v_copy);", generated)
        self.assertIn("fluid_string_free(&v_tail);", generated)
        self.assertIn("fluid_string_free(&v_expected);", generated)

    def test_owned_string_binding_cannot_be_copied(self):
        source = (
            'fn main() -> i64 { let first = string_from("text"); '
            "let second = first; return 0; }"
        )
        with self.assertRaisesRegex(FluidError, "String values cannot be copied"):
            self.compile_source(source)

    def test_string_append_requires_mutable_binding(self):
        source = (
            'fn main() -> i64 { let text = string_from("text"); '
            'string_append(text, "!"); return 0; }'
        )
        with self.assertRaisesRegex(FluidError, "requires a mutable String variable"):
            self.compile_source(source)

    def test_generic_functions_are_specialized_per_concrete_type(self):
        source = (
            "fn identity<T>(value: T) -> T { return value; } "
            "fn add<T>(left: T, right: T) -> T { return left + right; } "
            "fn main() -> i64 { "
            "let integer = identity(add(20, 22)); "
            "let decimal = identity(add(1.5, 2.5)); "
            "assert(decimal == 4.0); return integer - 42; }"
        )
        generated = self.compile_source(source)
        self.assertIn("fluid_identity__generic_i64(long long v_value)", generated)
        self.assertIn("fluid_identity__generic_f64(double v_value)", generated)
        self.assertIn("fluid_add__generic_i64(long long v_left, long long v_right)", generated)
        self.assertIn("fluid_add__generic_f64(double v_left, double v_right)", generated)

    def test_generic_function_requires_consistent_type_arguments(self):
        source = (
            "fn same<T>(left: T, right: T) -> T { return left; } "
            "fn main() -> i64 { return same(1, 2.0); }"
        )
        with self.assertRaisesRegex(FluidError, "inferred as both i64 and f64"):
            self.compile_source(source)

    def test_trait_objects_use_dynamic_vtable_dispatch(self):
        source = (
            "struct Circle { radius: i64 } "
            "trait Measure { fn area(self: &Self, scale: i64) -> i64; } "
            "impl Measure for Circle { "
            "fn area(self: &Circle, factor: i64) -> i64 { "
            "return self.radius * self.radius * factor; } } "
            "fn measure(shape: dyn Measure) -> i64 { return shape.area(1); } "
            "fn main() -> i64 { "
            "let circle = Circle { radius: 7 }; "
            "let shape: dyn Measure = circle; "
            "return measure(shape) - 49; }"
        )
        generated = self.compile_source(source)
        self.assertIn("fluid_dyn_Measure", generated)
        self.assertIn(".vtable->area(v_shape.data, 1LL)", generated)
        self.assertIn(
            ".area = fluid_trait_dispatch_Measure_Circle_area",
            generated,
        )

    def test_trait_implementation_must_define_every_method(self):
        source = (
            "struct Circle { radius: i64 } "
            "trait Measure { fn area(self: &Self, scale: i64) -> i64; } "
            "impl Measure for Circle { } "
            "fn main() -> i64 { return 0; }"
        )
        with self.assertRaisesRegex(FluidError, "missing: area"):
            self.compile_source(source)

    def test_project_init_add_and_install_local_dependency(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "app"
            dependency = root / "math"
            initialize_project(app)
            (dependency / "src").mkdir(parents=True)
            (dependency / "fluid.json").write_text(
                '{"name":"math","entry":"src/lib.fl","dependencies":{}}',
                encoding="utf-8",
            )
            (dependency / "src" / "lib.fl").write_text(
                "pub fn twice(value: i64) -> i64 { return value * 2; }",
                encoding="utf-8",
            )
            add_project_dependency(app, "math", "../math", None, None)
            lock_path = install_project(app)
            self.assertTrue(lock_path.is_file())
            manifest = json.loads((app / "fluid.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dependencies"]["math"]["path"], "../math")
            source = (app / "src" / "main.fl")
            source.write_text(
                "use math::twice; fn main() -> i64 { return twice(21); }",
                encoding="utf-8",
            )
            generated = Compiler(parse_project(app)).compile()
            self.assertIn("fluid_math__twice(21LL)", generated)

    def test_non_unit_function_must_return_on_all_paths(self):
        with self.assertRaisesRegex(FluidError, "may finish without returning"):
            self.compile_source("fn main() -> i64 { if true { return 0; } }")

    def test_else_if_is_supported_and_counts_for_return_analysis(self):
        result = self.compile_source(
            "fn main() -> i64 { if true { return 1; } "
            "else if false { return 2; } else { return 3; } }"
        )
        self.assertIn("else\n", result)
        self.assertIn("if (false)", result)

    def test_break_and_continue_work_inside_while(self):
        result = self.compile_source(
            "fn main() -> i64 { while true { continue; } "
            "while true { break; } return 0; }"
        )
        self.assertIn("        continue;", result)
        self.assertIn("        break;", result)

    def test_break_and_continue_are_rejected_outside_while(self):
        for keyword in ("break", "continue"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(FluidError, "only valid inside a loop"):
                    self.compile_source(f"fn main() -> i64 {{ {keyword}; return 0; }}")

    def test_for_range_uses_i64_and_an_exclusive_end(self):
        result = self.compile_source(
            "fn main() -> i64 { for index in 2..5 { print(index); } return 0; }"
        )
        self.assertIn("for (long long v_index = fluid_range_start_0; "
                      "v_index < fluid_range_end_0; ++v_index)", result)
        self.assertIn('printf("%lld\\n", v_index);', result)

    def test_float_literals_still_work_alongside_range_operator(self):
        result = self.compile_source(
            "fn main() -> i64 { print(.5); print(1.); "
            "for index in 1..2 { print(index); } return 0; }"
        )
        self.assertIn('printf("%g\\n", .5);', result)
        self.assertIn('printf("%g\\n", 1.);', result)

    def test_for_range_evaluates_bounds_once(self):
        result = self.compile_source(
            "fn start() -> i64 { return 1; }"
            "fn end() -> i64 { return 4; }"
            "fn main() -> i64 { for index in start()..end() { print(index); } return 0; }"
        )
        self.assertEqual(result.count("= fluid_start();"), 1)
        self.assertEqual(result.count("= fluid_end();"), 1)
        self.assertLess(result.index("long long fluid_range_start_0 = fluid_start();"),
                        result.index("long long fluid_range_end_0 = fluid_end();"))

    def test_for_range_requires_integer_bounds(self):
        with self.assertRaisesRegex(FluidError, "for loop start must have type i64"):
            self.compile_source(
                "fn main() -> i64 { for index in true..3 { print(index); } return 0; }"
            )

    def test_parallel_for_emits_workers_and_disjoint_vector_output(self):
        result = self.compile_source(
            "fn main() -> i64 { let factor = 3; "
            "var values: [i64] = [0, 0, 0, 0]; "
            "parallel for index in 0..len(values) { "
            "values[index] = index * factor; } return values[3]; }"
        )
        self.assertIn("fluid_parallel_for(fluid_range_start_0, fluid_range_end_0", result)
        self.assertIn("fluid_parallel_body_0(long long v_index, void *raw", result)
        self.assertIn("fluid_vec_i64 *out_values;", result)
        self.assertIn("(*fluid_context->out_values).data", result)
        self.assertIn("fluid_context->capture_factor", result)
        self.assertIn("pthread_create", result)
        self.assertIn("fluid_arena_free(&arena);", result)

    def test_native_parallel_for_computes_disjoint_results_and_worker_memory(self):
        source = (
            "fn copy(input: &[i64]) -> i64 { var output: [i64; 2] = [0, 0]; "
            "parallel for index in 0..2 { output[index] = input[index]; } "
            "return output[1]; }"
            "fn main() -> i64 { let input: [i64; 2] = [4, 9]; assert(copy(input) == 9); "
            "let factor = 3; "
            "var values: [i64] = [0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0]; "
            "parallel for index in 0..len(values) { "
            "var scratch = arena_i64(worker_arena(), 1); "
            "scratch[0] = index * factor; values[index] = scratch[0]; } "
            "var total = 0; for index in 0..len(values) { total = total + values[index]; } "
            "assert(total == 360); print(total); return 0; }"
        )
        try:
            compiler_command(None)
        except FluidError as error:
            self.skipTest(str(error))
        with TemporaryDirectory() as directory:
            executable = Path(directory) / "parallel-program.exe"
            build_binary(self.compile_source(source), executable, None)
            result = subprocess.run(
                [str(executable)], capture_output=True, check=False, text=True
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "360\n")

    def test_parallel_for_can_allocate_from_worker_local_arena(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [0, 0]; "
            "parallel for index in 0..2 { "
            "var scratch = arena_i64(worker_arena(), 1); "
            "scratch[0] = index; values[index] = scratch[0]; } "
            "return values[1]; }"
        )
        self.assertIn("fluid_arena_i64(fluid_worker_arena, 1LL", result)

    def test_parallel_for_can_read_borrowed_inputs_and_write_fixed_arrays(self):
        result = self.compile_source(
            "fn fill(input: &[i64]) -> i64 { var output: [i64; 2] = [0, 0]; "
            "parallel for index in 0..2 { output[index] = input[index]; } "
            "return output[1]; }"
            "fn main() -> i64 { let values: [i64; 2] = [4, 9]; return fill(values); }"
        )
        self.assertIn("long long (*capture_output)[2];", result)
        self.assertIn("fluid_slice_i64 capture_input;", result)
        self.assertIn("(*fluid_context->capture_output)[fluid_checked_index", result)

    def test_worker_arena_is_only_valid_inside_parallel_for(self):
        with self.assertRaisesRegex(FluidError, "only valid inside a parallel for"):
            self.compile_source(
                "fn main() -> i64 { let memory = worker_arena(); return 0; }"
            )

    def test_parallel_for_rejects_mutable_captures(self):
        with self.assertRaisesRegex(FluidError, "cannot capture mutable variable"):
            self.compile_source(
                "fn main() -> i64 { var factor = 3; "
                "var values: [i64] = [0, 0]; "
                "parallel for index in 0..2 { values[index] = index * factor; } "
                "return 0; }"
            )

    def test_parallel_for_rejects_output_reads_and_non_disjoint_writes(self):
        with self.assertRaisesRegex(FluidError, "cannot be read"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [1, 2]; "
                "parallel for index in 0..2 { values[index] = values[index] + 1; } "
                "return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "use the loop index directly"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [0, 0]; "
                "parallel for index in 0..2 { values[0] = index; } return 0; }"
            )

    def test_parallel_for_tracks_nested_loop_bounds_and_rejects_index_shadowing(self):
        result = self.compile_source(
            "fn main() -> i64 { let limit = 2; var values: [i64] = [0, 0]; "
            "parallel for index in 0..2 { "
            "for inner in 0..limit { values[index] = inner; } } return 0; }"
        )
        self.assertIn("fluid_context->capture_limit", result)
        with self.assertRaisesRegex(FluidError, "cannot shadow an existing variable"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [0, 0]; "
                "parallel for index in 0..2 { "
                "for index in 0..2 { values[index] = index; } } return 0; }"
            )

    def test_parallel_for_rejects_control_flow_that_changes_worker_ownership(self):
        for keyword in ("return", "break", "continue"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(FluidError, "not supported inside a parallel for"):
                    self.compile_source(
                        f"fn main() -> i64 {{ parallel for index in 0..2 {{ "
                        f"{keyword}; }} return 0; }}"
                    )

    def test_windows_emit_native_runtime_and_movement_operations(self):
        result = self.compile_source(
            'fn main() -> i64 { var app = window_create("Fluid", 320, 200); '
            'window_add_button(app, 1, "OK", 10, 10, 80, 24); '
            'window_add_checkbox(app, 2, "Check", 100, 10, 100, 24); '
            'window_add_label(app, 3, "Label", 210, 10, 80, 24); '
            'window_add_textbox(app, 4, "Text", 10, 50, 180, 28); '
            'window_set_textbox_text(app, 4, "Changed"); '
            'let textbox_text = window_textbox_text(app, 4); '
            'window_set_control_text(app, 3, "Changed label"); '
            'window_set_control_enabled(app, 1, false); '
            'window_set_checkbox_checked(app, 2, true); '
            'assert(window_checkbox_checked(app, 2)); '
            'window_set_title(app, "Moved"); window_show(app); window_hide(app); '
            'window_move(app, 40, 50); window_resize(app, 640, 480); '
            'assert(window_is_open(app)); assert(window_width(app) == 640); '
            'assert(window_height(app) == 480); assert(window_poll(app)); '
            'let event = window_next_event(app); '
            'assert(event.kind == event_resize()); '
            'assert(event.width == 640); assert(event.height == 480); '
            'assert(event_control() == 10); let control_id = event.control_id; '
            'let control_checked = event.checked; '
            'assert(event_mouse_wheel() == 6); assert(event.delta == 120); '
            'window_clear(app, 16777215); '
            'window_fill_rect(app, 10, 20, 40, 50, 16711680); '
            'window_fill_ellipse(app, 90, 100, 30, 30, 16711680); '
            'window_draw_rect(app, 5, 6, 70, 80, 255); '
            'window_draw_ellipse(app, 130, 100, 30, 30, 255); '
            'window_draw_line(app, 0, 0, 100, 100, 65280); '
            'window_draw_text(app, 12, 14, "Fluid", 0); window_present(app); '
            'window_present_rect(app, 10, 20, 40, 50); '
            'window_close(app); return 0; }'
        )
        self.assertIn("#include <windows.h>", result)
        self.assertIn("fluid_window_create(\"Fluid\", 320LL, 200LL", result)
        self.assertIn("fluid_window_move(&v_app, 40LL, 50LL", result)
        self.assertIn("fluid_window_resize(&v_app, 640LL, 480LL", result)
        self.assertIn("fluid_window_destroy(&v_app);", result)
        self.assertIn("SetWindowPos", result)
        self.assertIn("fluid_window_next_event(&v_app", result)
        self.assertIn("fluid_window_add_button(&v_app, 1LL", result)
        self.assertIn("fluid_window_add_checkbox(&v_app, 2LL", result)
        self.assertIn("fluid_window_add_label(&v_app, 3LL", result)
        self.assertIn("fluid_window_add_textbox(&v_app, 4LL", result)
        self.assertIn("fluid_window_textbox_text(&v_app, 4LL", result)
        self.assertIn("fluid_window_set_textbox_text(&v_app, 4LL", result)
        self.assertIn("fluid_window_set_control_text(&v_app, 3LL", result)
        self.assertIn("fluid_window_checkbox_checked(&v_app, 2LL", result)
        self.assertIn("FLUID_EVENT_CONTROL", result)
        self.assertIn("WS_CLIPCHILDREN", result)
        self.assertIn("EN_CHANGE", result)
        self.assertIn("RDW_NOCHILDREN", result)
        self.assertIn("RDW_NOERASE", result)
        self.assertIn("controls_ready = true", result)
        self.assertIn("FLUID_EVENT_RESIZE", result)
        self.assertIn("FLUID_EVENT_MOUSE_WHEEL", result)
        self.assertIn("TextOutW", result)
        self.assertIn("fluid_window_fill_ellipse(&v_app", result)
        self.assertIn("fluid_window_draw_ellipse(&v_app", result)
        self.assertIn("Ellipse(state->back_dc", result)
        self.assertIn("fluid_window_present_rect(&v_app, 10LL, 20LL, 40LL, 50LL", result)
        self.assertIn("paint.rcPaint.left", result)
        self.assertIn("BitBlt", result)
        self.assertIn("PatBlt", result)

    def test_windows_require_mutable_owned_handles(self):
        cases = (
            (
                'let app = window_create("Fluid", 320, 200); window_move(app, 1, 2);',
                "requires a mutable window variable",
            ),
            (
                'var app = window_create("Fluid", 320, 200); let copy = app;',
                "windows cannot be copied",
            ),
            (
                'var app = window_create("Fluid", true, 200);',
                "window_create expects i64, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_fill_rect(app, 0, 0, 20, 20, true);',
                "window_fill_rect expects i64, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_fill_ellipse(app, 0, 0, 20, 20, true);',
                "window_fill_ellipse expects i64, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_add_button(app, 1, "OK", 0, 0, 80, true);',
                "window_add_button expects i64, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_present_rect(app, 0, 0, 20, true);',
                "window_present_rect expects i64, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_add_textbox(app, 1, "Text", 0, 0, 80, 24); '
                'window_set_textbox_text(app, 1, true);',
                "window_set_textbox_text expects str, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'window_add_label(app, 1, "Text", 0, 0, 80, 24); '
                'window_set_control_text(app, 1, true);',
                "window_set_control_text expects str, got bool",
            ),
            (
                'var app = window_create("Fluid", 320, 200); '
                'let event = window_next_event(app); print(event.invalid);',
                "window_event has no field 'invalid'",
            ),
        )
        for body, message in cases:
            with self.subTest(body=body):
                with self.assertRaisesRegex(FluidError, message):
                    self.compile_source(f"fn main() -> i64 {{ {body} return 0; }}")

    def test_windows_cannot_escape_through_function_signatures(self):
        with self.assertRaisesRegex(FluidError, "cannot return arrays, vectors, arenas, windows"):
            self.compile_source(
                'fn create() -> window { return window_create("Fluid", 1, 1); }'
                "fn main() -> i64 { return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "cannot be passed to functions"):
            self.compile_source(
                "fn move_window(app: window) -> unit {} "
                "fn main() -> i64 { return 0; }"
            )

    def test_windows_are_not_available_in_parallel_workers(self):
        with self.assertRaisesRegex(FluidError, "not allowed in parallel loops"):
            self.compile_source(
                'fn main() -> i64 { parallel for index in 0..1 { '
                'var app = window_create("Fluid", 1, 1); } return 0; }'
            )

    def test_for_loop_variable_is_immutable(self):
        with self.assertRaisesRegex(FluidError, "immutable let binding"):
            self.compile_source(
                "fn main() -> i64 { for index in 0..2 { index = 1; } return 0; }"
            )

    def test_break_and_continue_work_inside_for(self):
        result = self.compile_source(
            "fn main() -> i64 { for index in 0..3 { "
            "if index == 1 { continue; } break; } return 0; }"
        )
        self.assertIn("continue;", result)
        self.assertIn("break;", result)

    def test_array_literal_and_index_read_are_emitted(self):
        result = self.compile_source(
            "fn main() -> i64 { let values: [i64; 3] = [10, 20, 30]; "
            "print(values[1]); return 0; }"
        )
        self.assertIn("long long v_values[3] = {10LL, 20LL, 30LL};", result)
        self.assertRegex(result, r"fluid_checked_index\(\(long long\)\(1LL\), 3, 1, \d+\)")
        self.assertIn("array index out of bounds", result)

    def test_array_literal_infers_element_type_and_length(self):
        result = self.compile_source(
            "fn main() -> i64 { let flags = [true, false]; "
            "print(flags[0]); return 0; }"
        )
        self.assertIn("bool v_flags[2] = {true, false};", result)

    def test_var_array_elements_can_be_assigned(self):
        result = self.compile_source(
            "fn main() -> i64 { var values = [1, 2]; "
            "values[0] = 9; print(values[0]); return 0; }"
        )
        self.assertIn("v_values[fluid_checked_index", result)
        self.assertIn(" = 9LL;", result)

    def test_let_array_elements_cannot_be_assigned(self):
        with self.assertRaisesRegex(FluidError, "immutable let binding"):
            self.compile_source(
                "fn main() -> i64 { let values = [1, 2]; "
                "values[0] = 9; return 0; }"
            )

    def test_array_literal_rejects_mixed_element_types(self):
        with self.assertRaisesRegex(FluidError, "array elements must have the same type"):
            self.compile_source("fn main() -> i64 { let values = [1, true]; return 0; }")

    def test_array_annotation_requires_matching_length(self):
        with self.assertRaisesRegex(FluidError, "cannot assign array:i64:2 to array:i64:3"):
            self.compile_source(
                "fn main() -> i64 { let values: [i64; 3] = [1, 2]; return 0; }"
            )

    def test_array_index_must_be_an_integer(self):
        with self.assertRaisesRegex(FluidError, "array index must have type i64"):
            self.compile_source(
                "fn main() -> i64 { let values = [1, 2]; "
                "print(values[false]); return 0; }"
            )

    def test_non_array_values_cannot_be_indexed(self):
        with self.assertRaisesRegex(FluidError, "cannot index value of type i64"):
            self.compile_source("fn main() -> i64 { let value = 1; print(value[0]); return 0; }")

    def test_arrays_are_rejected_in_function_signatures(self):
        with self.assertRaisesRegex(FluidError, "arrays cannot be function parameters"):
            self.compile_source(
                "fn first(values: [i64; 2]) -> i64 { return values[0]; }"
                "fn main() -> i64 { return 0; }"
            )

    def test_dynamic_vector_grows_and_checks_indexes_at_runtime(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [10, 20]; "
            "push(values, 30); print(len(values)); print(values[2]); return 0; }"
        )
        self.assertIn("long long fluid_items_values[2] = {10LL, 20LL};", result)
        self.assertIn("fluid_vec_i64 fluid_vector_0 = "
                      "fluid_vec_i64_make(fluid_items_values, 2,", result)
        self.assertIn("fluid_vec_i64_push(&fluid_vector_0, 30LL", result)
        self.assertIn("((long long)fluid_vector_0.len)", result)
        self.assertRegex(result, r"fluid_checked_index\(\(long long\)\(2LL\), fluid_vector_0\.len,")
        self.assertIn("fluid_vec_i64_free(&fluid_vector_0);", result)

    def test_empty_vector_can_be_initialized_and_reserved(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = []; reserve(values, 32); "
            "print(capacity(values)); return 0; }"
        )
        self.assertIn("fluid_vec_i64_make(NULL, 0,", result)
        self.assertIn("fluid_vec_i64_reserve(&fluid_vector_0, 32LL", result)
        self.assertIn("((long long)fluid_vector_0.capacity)", result)

    def test_vector_pop_and_clear_mutate_the_vector(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [4]; "
            "let item = pop(values); clear(values); print(item); return 0; }"
        )
        self.assertIn("fluid_vec_i64_pop(&fluid_vector_0", result)
        self.assertIn("fluid_vec_i64_clear(&fluid_vector_0);", result)

    def test_vector_insert_and_remove_shift_elements(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [1, 3]; "
            "insert(values, 1, 2); let removed = remove(values, 0); "
            "print(removed); print(values[0]); return 0; }"
        )
        self.assertIn("fluid_vec_i64_insert(&fluid_vector_0, 1LL, 2LL", result)
        self.assertIn("long long v_removed = fluid_vec_i64_remove(&fluid_vector_0, 0LL", result)
        self.assertIn("fluid_checked_insert_index", result)
        self.assertIn("memmove(vec->data + at + 1", result)
        self.assertIn("memmove(vec->data + at, vec->data + at + 1", result)

    def test_vector_shrink_to_fit_updates_capacity(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = []; reserve(values, 20); "
            "push(values, 3); shrink_to_fit(values); print(capacity(values)); return 0; }"
        )
        self.assertIn("fluid_vec_i64_shrink_to_fit(&fluid_vector_0", result)
        self.assertIn("vec->capacity = vec->len;", result)

    def test_vector_insert_and_remove_require_valid_argument_types(self):
        with self.assertRaisesRegex(FluidError, "insert index expects i64"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = []; insert(values, false, 1); return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "insert expects i64, got bool"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = []; insert(values, 0, true); return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "remove expects exactly 2 arguments"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = []; remove(values); return 0; }"
            )

    def test_vector_insert_and_remove_require_mutable_bindings(self):
        for operation in ("insert(values, 0, 1)", "remove(values, 0)", "shrink_to_fit(values)"):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(FluidError, "requires a mutable vector"):
                    self.compile_source(
                        f"fn main() -> i64 {{ let values: [i64] = [1]; "
                        f"let removed = {operation}; return 0; }}"
                        if operation.startswith("remove")
                        else f"fn main() -> i64 {{ let values: [i64] = [1]; "
                             f"{operation}; return 0; }}"
                    )

    def test_vector_mutation_requires_a_mutable_binding(self):
        with self.assertRaisesRegex(FluidError, "push requires a mutable vector"):
            self.compile_source(
                "fn main() -> i64 { let values: [i64] = []; push(values, 1); return 0; }"
            )

    def test_vector_values_cannot_be_copied(self):
        with self.assertRaisesRegex(FluidError, "vectors cannot be copied"):
            self.compile_source(
                "fn main() -> i64 { let values: [i64] = []; let other = values; return 0; }"
            )

    def test_vectors_are_rejected_in_function_signatures(self):
        with self.assertRaisesRegex(FluidError, "explicit borrow"):
            self.compile_source(
                "fn use(values: [i64]) -> i64 { return len(values); }"
                "fn main() -> i64 { return 0; }"
            )

    def test_vectors_can_be_borrowed_by_functions(self):
        result = self.compile_source(
            "fn total(values: &[i64]) -> i64 { return values[0] + len(values); }"
            "fn main() -> i64 { var values: [i64] = [8]; return total(values); }"
        )
        self.assertIn("const fluid_slice_i64 * v_values", result)
        self.assertIn("fluid_total(&((fluid_slice_i64){fluid_vector_0.len, fluid_vector_0.data}))",
                      result)
        self.assertIn("fluid_vec_i64_free(&fluid_vector_0);", result)
        self.assertNotIn("fluid_vec_i64_free(&v_values)", result)

    def test_mutable_vector_borrow_can_modify_caller(self):
        result = self.compile_source(
            "fn add(values: &mut [i64]) -> unit { push(values, 5); }"
            "fn main() -> i64 { var values: [i64] = []; add(values); return values[0]; }"
        )
        self.assertIn("fluid_vec_i64 * v_values", result)
        self.assertIn("fluid_vec_i64_push(v_values", result)
        self.assertIn("fluid_add(&fluid_vector_0)", result)

    def test_mutable_borrow_can_be_forwarded_to_another_function(self):
        result = self.compile_source(
            "fn add(values: &mut [i64]) -> unit { push(values, 5); }"
            "fn forward(values: &mut [i64]) -> unit { add(values); }"
            "fn main() -> i64 { var values: [i64] = []; "
            "forward(values); return values[0]; }"
        )
        self.assertIn("fluid_add(&(*v_values))", result)

    def test_shared_vector_borrow_cannot_mutate(self):
        with self.assertRaisesRegex(FluidError, "push requires a mutable vector"):
            self.compile_source(
                "fn add(values: &[i64]) -> unit { push(values, 5); }"
                "fn main() -> i64 { var values: [i64] = []; add(values); return 0; }"
            )

    def test_mutable_vector_borrow_requires_mutable_owner(self):
        with self.assertRaisesRegex(FluidError, "mutable borrow requires a mutable vector"):
            self.compile_source(
                "fn add(values: &mut [i64]) -> unit { push(values, 5); }"
                "fn main() -> i64 { let values: [i64] = []; add(values); return 0; }"
            )

    def test_vector_ownership_transfer_requires_explicit_move(self):
        with self.assertRaisesRegex(FluidError, "requires an explicit move"):
            self.compile_source(
                "fn take(values: move [i64]) -> unit {}"
                "fn main() -> i64 { var values: [i64] = []; take(values); return 0; }"
            )
        result = self.compile_source(
            "fn take(values: move [i64]) -> i64 { return len(values); }"
            "fn main() -> i64 { var values: [i64] = [4, 5]; return take(move values); }"
        )
        self.assertIn("fluid_vec_i64_take(&fluid_vector_0)", result)
        self.assertIn("fluid_vec_i64_free(&v_values);", result)

    def test_move_requires_owned_mutable_vector(self):
        with self.assertRaisesRegex(FluidError, "move requires a mutable vector binding"):
            self.compile_source(
                "fn take(values: move [i64]) -> unit {}"
                "fn main() -> i64 { let values: [i64] = []; take(move values); return 0; }"
            )

    def test_moved_vectors_cannot_be_used_again(self):
        with self.assertRaisesRegex(FluidError, "use of moved vector 'values'"):
            self.compile_source(
                "fn size(values: move [i64]) -> i64 { return len(values); }"
                "fn main() -> i64 { var values: [i64] = [1]; "
                "let count = size(move values); return len(values); }"
            )

    def test_borrowing_prevents_conflicting_aliases_in_one_call(self):
        with self.assertRaisesRegex(FluidError, "cannot mix shared and mutable borrows"):
            self.compile_source(
                "fn update(read: &[i64], write: &mut [i64]) -> unit {}"
                "fn main() -> i64 { var values: [i64] = []; "
                "update(values, values); return 0; }"
            )

    def test_move_cannot_alias_another_argument(self):
        with self.assertRaisesRegex(FluidError, "moved vector cannot be used by another argument"):
            self.compile_source(
                "fn inspect(values: move [i64], count: i64) -> i64 { return count; }"
                "fn main() -> i64 { var values: [i64] = [1]; "
                "return inspect(move values, len(values)); }"
            )

    def test_slices_view_vector_ranges_without_copying(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [2, 4, 6, 8]; "
            "let middle = slice(values, 1, 3); "
            "print(len(middle)); print(middle[0]); return 0; }"
        )
        self.assertIn("fluid_slice_i64_make(fluid_vector_0.data, fluid_vector_0.len, 1LL, 3LL", result)
        self.assertIn("fluid_slice_i64 v_middle", result)
        self.assertIn("v_middle.data[fluid_checked_index", result)
        with self.assertRaisesRegex(FluidError, "while a slice into it is live"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [2]; "
                "let view = slice(values, 0, 1); push(values, 3); return 0; }"
            )

    def test_slice_can_be_borrowed_as_a_function_parameter(self):
        result = self.compile_source(
            "fn first(values: &slice[i64]) -> i64 { return values[0]; }"
            "fn main() -> i64 { var values: [i64] = [7]; "
            "let view = slice(values, 0, 1); return first(view); }"
        )
        self.assertIn("const fluid_slice_i64 * v_values", result)
        self.assertIn("fluid_first(&v_view)", result)

    def test_fixed_array_can_be_borrowed_as_a_slice_parameter(self):
        result = self.compile_source(
            "fn first(values: &[i64]) -> i64 { return values[0]; }"
            "fn main() -> i64 { let values: [i64; 2] = [7, 9]; return first(values); }"
        )
        self.assertIn("&((fluid_slice_i64){2, v_values})", result)

    def test_slice_bounds_and_element_mutability_are_checked(self):
        with self.assertRaisesRegex(FluidError, "slice start"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [1]; "
                "let view = slice(values, false, 1); return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "immutable let binding"):
            self.compile_source(
                "fn main() -> i64 { let values: [i64; 1] = [1]; "
                "let view = slice(values, 0, 1); view[0] = 2; return 0; }"
            )

    def test_arena_allocates_zeroed_typed_slices_and_frees_on_scope_exit(self):
        result = self.compile_source(
            "fn main() -> i64 { var memory = arena(); "
            "var values = arena_i64(memory, 4); values[2] = 9; "
            "print(values[2]); return 0; }"
        )
        self.assertIn("fluid_arena_make()", result)
        self.assertIn("fluid_arena_i64(&v_memory, 4LL", result)
        self.assertIn("fluid_arena_free(&v_memory);", result)
        self.assertIn("fluid_slice_i64 v_values", result)

    def test_arena_slice_uses_are_scoped_and_mutable(self):
        with self.assertRaisesRegex(FluidError, "requires a mutable arena"):
            self.compile_source(
                "fn main() -> i64 { let memory = arena(); "
                "let values = arena_i64(memory, 1); return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "arenas cannot be copied"):
            self.compile_source(
                "fn main() -> i64 { var memory = arena(); "
                "let other = memory; return 0; }"
            )

    def test_mutable_slice_writes_through_to_its_owner(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [1]; "
            "var view = slice(values, 0, 1); view[0] = 9; return values[0]; }"
        )
        self.assertIn("fluid_vector_0.data[fluid_checked_index", result)
        self.assertIn("v_view.data[fluid_checked_index", result)

    def test_slice_rejects_out_of_range_at_runtime(self):
        result = self.compile_source(
            "fn main() -> i64 { let values: [i64; 2] = [1, 2]; "
            "let view = slice(values, 0, 3); return len(view); }"
        )
        self.assertIn("slice bounds out of range", result)

    def test_structs_support_typed_literals_fields_and_mutation(self):
        result = self.compile_source(
            "struct Point { x: i64, y: i64 }"
            "fn main() -> i64 { var point = Point { x: 3, y: 4 }; "
            "point.x = 8; return point.x + point.y; }"
        )
        self.assertIn("struct fluid_Point {", result)
        self.assertIn("struct fluid_Point v_point = (struct fluid_Point){.x = 3LL, .y = 4LL};", result)
        self.assertIn("v_point.x = 8LL;", result)

    def test_struct_values_can_cross_function_boundaries(self):
        result = self.compile_source(
            "struct Pair { left: i64, right: i64 }"
            "fn sum(pair: Pair) -> i64 { return pair.left + pair.right; }"
            "fn main() -> i64 { let pair = Pair { left: 10, right: 20 }; return sum(pair); }"
        )
        self.assertIn("long long fluid_sum(struct fluid_Pair v_pair);", result)
        self.assertIn("fluid_sum(v_pair)", result)

    def test_struct_literal_rejects_missing_extra_and_mistyped_fields(self):
        cases = (
            ("Point { x: 1 }", "missing fields: y"),
            ("Point { x: 1, y: 2, z: 3 }", "unknown fields: z"),
            ("Point { x: true, y: 2 }", "field 'x' expects i64"),
        )
        for expression, error in cases:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(FluidError, error):
                    self.compile_source(
                        f"struct Point {{ x: i64, y: i64 }} "
                        f"fn main() -> i64 {{ let point = {expression}; return 0; }}"
                    )

    def test_struct_fields_reject_unknown_and_immutable_writes(self):
        with self.assertRaisesRegex(FluidError, "has no field"):
            self.compile_source(
                "struct Point { x: i64 }"
                "fn main() -> i64 { let point = Point { x: 1 }; return point.y; }"
            )
        with self.assertRaisesRegex(FluidError, "immutable let binding"):
            self.compile_source(
                "struct Point { x: i64 }"
                "fn main() -> i64 { let point = Point { x: 1 }; point.x = 2; return 0; }"
            )

    def test_vector_pop_requires_matching_type_and_a_value_target(self):
        with self.assertRaisesRegex(FluidError, "push expects i64, got bool"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = []; push(values, true); return 0; }"
            )
        with self.assertRaisesRegex(FluidError, "only unit-returning calls"):
            self.compile_source(
                "fn main() -> i64 { var values: [i64] = [1]; pop(values); return 0; }"
            )

    def test_vector_cleanup_is_emitted_for_scope_exit_and_loop_control(self):
        result = self.compile_source(
            "fn main() -> i64 { while true { var values: [i64] = []; "
            "push(values, 1); continue; } return 0; }"
        )
        self.assertIn("fluid_vec_i64_free(&fluid_vector_0);", result)
        self.assertLess(result.index("fluid_vec_i64_free(&fluid_vector_0);"),
                        result.index("continue;"))

    def test_vector_return_expression_is_evaluated_before_cleanup(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [7]; return values[0]; }"
        )
        self.assertLess(result.index("fluid_return_value_0 = fluid_vector_0.data["),
                        result.index("fluid_vec_i64_free(&fluid_vector_0);"))
        self.assertLess(result.index("fluid_vec_i64_free(&fluid_vector_0);"),
                        result.index("return fluid_return_value_0;"))

    def test_return_cleanup_survives_shadowing_by_a_scalar(self):
        result = self.compile_source(
            "fn main() -> i64 { var values: [i64] = [9]; "
            "if true { let values = 1; return values; } return 0; }"
        )
        self.assertIn("fluid_return_value_0 = v_values;", result)
        self.assertIn("fluid_vec_i64_free(&fluid_vector_0);", result)

    def test_string_equality_compares_contents(self):
        result = self.compile_source(
            'fn main() -> i64 { if "fluid" == "fluid" { return 1; } return 0; }'
        )
        self.assertIn("#include <string.h>", result)
        self.assertIn('(strcmp("fluid", "fluid") == 0)', result)

    def test_string_inequality_compares_contents(self):
        result = self.compile_source(
            'fn main() -> i64 { if "fluid" != "python" { return 1; } return 0; }'
        )
        self.assertIn('(strcmp("fluid", "python") != 0)', result)

    def test_string_ordering_is_not_supported(self):
        with self.assertRaisesRegex(FluidError, "cannot be used with str and str"):
            self.compile_source(
                'fn main() -> i64 { if "a" < "b" { return 1; } return 0; }'
            )

    def test_len_string_uses_strlen(self):
        result = self.compile_source(
            'fn main() -> i64 { print(len("fluid")); return 0; }'
        )
        self.assertIn("#include <string.h>", result)
        self.assertIn('printf("%lld\\n", ((long long)strlen("fluid")));', result)

    def test_len_array_is_a_compile_time_constant(self):
        result = self.compile_source(
            "fn main() -> i64 { let values = [10, 20, 30]; "
            "print(len(values)); return 0; }"
        )
        self.assertIn('printf("%lld\\n", 3LL);', result)

    def test_len_requires_one_string_or_array_argument(self):
        with self.assertRaisesRegex(FluidError, "len expects exactly one argument"):
            self.compile_source('fn main() -> i64 { print(len()); return 0; }')
        with self.assertRaisesRegex(FluidError, "len does not accept values of type i64"):
            self.compile_source("fn main() -> i64 { print(len(1)); return 0; }")

    def test_len_of_array_literal_is_rejected(self):
        with self.assertRaisesRegex(FluidError, "len of an array requires an array variable"):
            self.compile_source(
                "fn value() -> i64 { return 1; }"
                "fn main() -> i64 { print(len([value(), value()])); return 0; }"
            )

    def test_builtin_names_are_reserved(self):
        with self.assertRaisesRegex(FluidError, "len.*reserved"):
            self.compile_source("fn len(value: str) -> i64 { return 0; } fn main() -> i64 { return 0; }")

    def test_window_builtin_names_are_reserved(self):
        with self.assertRaisesRegex(FluidError, "window_add_button.*reserved"):
            self.compile_source(
                "fn window_add_button() -> i64 { return 0; } "
                "fn main() -> i64 { return 0; }"
            )

    def test_numeric_builtins_emit_native_helpers(self):
        result = self.compile_source(
            "fn main() -> i64 { print(abs(-5)); print(min(3, 7)); "
            "print(max(3, 7)); print(sqrt(9.0)); return 0; }"
        )
        self.assertIn("fluid_abs_i64((-5LL))", result)
        self.assertIn("fluid_min_i64(3LL, 7LL)", result)
        self.assertIn("fluid_max_i64(3LL, 7LL)", result)
        self.assertIn("sqrt(9.0)", result)
        self.assertIn("value == LLONG_MIN", result)

    def test_string_builtins_emit_native_helpers(self):
        result = self.compile_source(
            'fn main() -> i64 { print(contains("fluid", "lu")); '
            'print(starts_with("fluid", "flu")); '
            'print(ends_with("fluid", "id")); return 0; }'
        )
        self.assertIn("fluid_contains(\"fluid\", \"lu\")", result)
        self.assertIn("fluid_starts_with(\"fluid\", \"flu\")", result)
        self.assertIn("fluid_ends_with(\"fluid\", \"id\")", result)

    def test_numeric_builtins_emit_clamp_sign_and_rounding_helpers(self):
        result = self.compile_source(
            "fn main() -> i64 { print(clamp(12, 0, 10)); print(sign(-7)); "
            "print(pow(2.0, 3.0)); print(floor(1.9)); print(ceil(1.1)); "
            "print(round(1.6)); return 0; }"
        )
        self.assertIn("fluid_clamp_i64(12LL, 0LL, 10LL)", result)
        self.assertIn("fluid_sign_i64((-7LL))", result)
        self.assertIn("pow(2.0, 3.0)", result)
        self.assertIn("floor(1.9)", result)
        self.assertIn("ceil(1.1)", result)
        self.assertIn("round(1.6)", result)

    def test_is_empty_string_builtin(self):
        result = self.compile_source(
            'fn main() -> i64 { print(is_empty("")); return 0; }'
        )
        self.assertIn("fluid_is_empty(\"\")", result)

    def test_builtins_check_arity_and_argument_types(self):
        cases = (
            ("abs()", "abs expects exactly 1 argument"),
            ("min(1)", "min expects exactly 2 argument"),
            ("sqrt(1)", "sqrt expects f64, got i64"),
            ("clamp(1, 2)", "clamp expects exactly 3 argument"),
            ("round(1)", "round expects f64, got i64"),
            ('contains("a", 1)', "contains expects str, got i64"),
            ("is_empty(false)", "is_empty expects str, got bool"),
        )
        for expression, message in cases:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(FluidError, message):
                    self.compile_source(f"fn main() -> i64 {{ print({expression}); return 0; }}")

    def test_gcc_native_tuning_is_opt_in(self):
        source = Path("program.c")
        output = Path("program")
        standard = native_compile_args(["gcc"], source, output)
        tuned = native_compile_args(["gcc"], source, output, native_tune=True)
        self.assertNotIn("-march=native", standard)
        self.assertIn("-O3", standard)
        self.assertIn("-lm", standard)
        self.assertGreater(standard.index("-lm"), standard.index(str(source)))
        self.assertIn("-march=native", tuned)

    def test_posix_parallel_build_enables_thread_linking(self):
        with (
            patch("fluid.os.name", "posix"),
            patch.object(Path, "resolve", lambda path: path),
        ):
            arguments = native_compile_args(
                ["gcc"], Path("program.c"), Path("program"), parallel=True
            )
        self.assertIn("-pthread", arguments)

    def test_windows_api_links_user32_and_gdi32(self):
        source = Path("program.c")
        output = Path("program.exe")
        with (
            patch("fluid.os.name", "nt"),
            patch.object(Path, "resolve", lambda path: path),
        ):
            gcc_args = native_compile_args(
                ["clang"], source, output, windows_api=True
            )
            msvc_args = native_compile_args(
                ["cl.exe"], source, output, windows_api=True
            )
        self.assertIn("-luser32", gcc_args)
        self.assertIn("-lgdi32", gcc_args)
        self.assertIn("/utf-8", msvc_args)
        self.assertEqual(msvc_args[-3:], ["/link", "user32.lib", "gdi32.lib"])

    def test_msvc_native_tuning_reports_unsupported_mode(self):
        with self.assertRaisesRegex(FluidError, "only with GCC or Clang"):
            native_compile_args(["cl.exe"], Path("program.c"), Path("program.exe"), native_tune=True)
        self.assertNotIn(
            "-lm",
            native_compile_args(["clang-cl.exe"], Path("program.c"), Path("program.exe")),
        )

    def test_native_tune_flag_is_available_for_build_and_run(self):
        from fluid import create_parser

        build = create_parser().parse_args(["build", "main.fl", "--native-tune"])
        run = create_parser().parse_args(["run", "main.fl", "--native-tune"])
        self.assertTrue(build.native_tune)
        self.assertTrue(run.native_tune)

    def test_assert_statement_emits_runtime_check(self):
        result = self.compile_source(
            "fn main() -> i64 { assert(1 + 1 == 2); return 0; }"
        )
        self.assertIn("fluid_assert(((1LL + 1LL) == 2LL), 1, 20);", result)
        self.assertIn("assertion failed at %d:%d", result)

    def test_assert_condition_must_be_boolean(self):
        with self.assertRaisesRegex(FluidError, "assert condition must have type bool"):
            self.compile_source("fn main() -> i64 { assert(1); return 0; }")

    def test_installed_llvm_is_discovered_from_default_windows_path(self):
        llvm = Path("C:/Program Files/LLVM/bin/clang.exe")
        with (
            patch("fluid.os.name", "nt"),
            patch("fluid.shutil.which", return_value=None),
            patch.object(
                Path, "is_file", autospec=True,
                side_effect=lambda path: path.name == "clang.exe",
            ),
        ):
            command = compiler_command(None)
        self.assertEqual(command[0], str(llvm))

    def test_statement_after_return_is_rejected(self):
        with self.assertRaisesRegex(FluidError, "unreachable statement"):
            self.compile_source("fn main() -> i64 { return 0; print(1); }")

    def test_statement_after_fully_returning_if_else_is_rejected(self):
        with self.assertRaisesRegex(FluidError, "unreachable statement"):
            self.compile_source(
                "fn main() -> i64 { if true { return 1; } "
                "else { return 2; } return 3; }"
            )

    def test_statement_after_break_or_continue_is_rejected(self):
        for keyword in ("break", "continue"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(FluidError, "unreachable statement"):
                    self.compile_source(
                        f"fn main() -> i64 {{ while true {{ {keyword}; print(1); }} return 0; }}"
                    )

    def test_statement_after_loop_with_terminating_if_else_is_rejected(self):
        with self.assertRaisesRegex(FluidError, "unreachable statement"):
            self.compile_source(
                "fn main() -> i64 { while true { "
                "if true { break; } else { continue; } print(1); } return 0; }"
            )


if __name__ == "__main__":
    unittest.main()
