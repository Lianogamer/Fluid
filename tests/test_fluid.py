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

    def test_len_is_reserved_for_builtin(self):
        with self.assertRaisesRegex(FluidError, "len.*reserved"):
            self.compile_source("fn len(value: str) -> i64 { return 0; } fn main() -> i64 { return 0; }")

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
