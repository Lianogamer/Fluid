import unittest

from fluid import Compiler, FluidError, Parser


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
        self.assertIn("if ((v_result == 42LL))", result)
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
            "fn main() -> i64 { while true { continue; break; } return 0; }"
        )
        self.assertIn("        continue;", result)
        self.assertIn("        break;", result)

    def test_break_and_continue_are_rejected_outside_while(self):
        for keyword in ("break", "continue"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(FluidError, "only valid inside a while loop"):
                    self.compile_source(f"fn main() -> i64 {{ {keyword}; return 0; }}")


if __name__ == "__main__":
    unittest.main()
