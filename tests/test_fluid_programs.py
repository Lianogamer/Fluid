import subprocess
import sys
import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fluid import FluidError, compiler_command


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "tests" / "programs"
CLI = ROOT / "fluid.py"


class FluidProgramTests(unittest.TestCase):
    def run_fluid(self, command: str, program: Path) -> subprocess.CompletedProcess[str]:
        if command == "run":
            try:
                compiler_command(None)
            except FluidError as error:
                self.skipTest(f"native Fluid program tests need a C compiler: {error}")
        return subprocess.run(
            [sys.executable, str(CLI), command, str(program)],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )

    def test_core_language_program(self):
        result = self.run_fluid("run", PROGRAMS / "core.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: core\n")

    def test_memory_language_program(self):
        result = self.run_fluid("run", PROGRAMS / "memory.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: memory\n")

    def test_parallel_language_program(self):
        result = self.run_fluid("run", PROGRAMS / "parallel.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: parallel\n")

    def test_task_pool_and_typed_channels(self):
        result = self.run_fluid("run", PROGRAMS / "concurrency.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: concurrency\n")

    def test_enum_and_exhaustive_match_program(self):
        result = self.run_fluid("run", PROGRAMS / "enum_match.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: enum match\n")

    def test_owned_string_program(self):
        result = self.run_fluid("run", PROGRAMS / "strings.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "lo\n")

    def test_generic_functions_run_for_multiple_types(self):
        with TemporaryDirectory() as directory:
            program = Path(directory) / "generics.fl"
            program.write_text(
                "fn identity<T>(value: T) -> T { return value; } "
                "fn add<T>(left: T, right: T) -> T { return left + right; } "
                'fn main() -> i64 { if identity(add(20, 22)) == 42 { '
                'print("PASS: generics"); } '
                "if identity(add(1.5, 2.5)) == 4.0 { "
                'print("PASS: generic f64"); } return 0; }',
                encoding="utf-8",
            )
            result = self.run_fluid("run", program)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: generics\nPASS: generic f64\n")

    def test_dynamic_trait_dispatch_runs_native_program(self):
        with TemporaryDirectory() as directory:
            program = Path(directory) / "traits.fl"
            program.write_text(
                "struct Circle { radius: i64 } "
                "struct Square { side: i64 } "
                "trait Measure { fn area(self: &Self, scale: i64) -> i64; } "
                "impl Measure for Circle { "
                "fn area(self: &Circle, factor: i64) -> i64 { "
                "return self.radius * self.radius * factor; } } "
                "impl Measure for Square { "
                "fn area(self: &Square, factor: i64) -> i64 { "
                "return self.side * self.side * factor; } } "
                "fn measure(shape: dyn Measure) -> i64 { return shape.area(1); } "
                "fn main() -> i64 { "
                "let circle = Circle { radius: 7 }; "
                "let square = Square { side: 5 }; "
                "if circle.area(1) == 49 && measure(circle) == 49 "
                "&& measure(square) == 25 { "
                'print("PASS: dynamic traits"); } return 0; }',
                encoding="utf-8",
            )
            result = self.run_fluid("run", program)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: dynamic traits\n")

    def test_project_with_local_dependency(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "app"
            dependency = root / "math"
            app_source = app / "src"
            dep_source = dependency / "src"
            app_source.mkdir(parents=True)
            dep_source.mkdir(parents=True)
            (app / "fluid.json").write_text(
                '{"name":"demo","entry":"src/main.fl",'
                '"dependencies":{"math":{"path":"../math"}}}',
                encoding="utf-8",
            )
            (app_source / "main.fl").write_text(
                "use math::twice; "
                'fn main() -> i64 { if twice(21) == 42 { print("PASS: package"); } return 0; }',
                encoding="utf-8",
            )
            (dependency / "fluid.json").write_text(
                '{"name":"math","entry":"src/lib.fl","dependencies":{}}',
                encoding="utf-8",
            )
            (dep_source / "lib.fl").write_text(
                "pub fn twice(value: i64) -> i64 { return value * 2; }",
                encoding="utf-8",
            )
            result = self.run_fluid("run", app)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: package\n")

    def test_module_import_program(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.fl").write_text(
                "mod math; use math::twice; "
                'fn main() -> i64 { let result = twice(21); '
                'if result == 42 { print("PASS: module"); } return 0; }',
                encoding="utf-8",
            )
            (root / "math.fl").write_text(
                "fn double(value: i64) -> i64 { return value * 2; } "
                "pub fn twice(value: i64) -> i64 { return double(value); }",
                encoding="utf-8",
            )
            result = self.run_fluid("run", root / "main.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: module\n")

    @unittest.skipUnless(os.name == "nt", "Win32 window integration test")
    def test_windows_language_program(self):
        result = self.run_fluid("run", PROGRAMS / "windows.fl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "PASS: windows\n")

    @unittest.skipUnless(os.name == "nt", "Win32 UI demo build")
    def test_windows_ui_demo_builds(self):
        try:
            compiler_command(None)
        except FluidError as error:
            self.skipTest(f"native Fluid program tests need a C compiler: {error}")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "ui_demo.exe"
            result = subprocess.run(
                [sys.executable, str(CLI), "build", str(ROOT / "examples" / "ui_demo.fl"),
                 "-o", str(output)],
                cwd=ROOT,
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(
            "warning: equality comparison with extraneous parentheses", result.stderr
        )

    def test_fluid_assertion_fails_at_runtime(self):
        result = self.run_fluid("run", PROGRAMS / "fail_assert.fl")
        self.assertEqual(result.returncode, 1)
        self.assertIn("fluid: assertion failed at 2:5", result.stderr)

    def test_parallel_mutable_capture_is_rejected(self):
        program = PROGRAMS / "compile_errors" / "parallel_mutable_capture.fl"
        result = self.run_fluid("check", program)
        self.assertEqual(result.returncode, 1)
        self.assertIn("parallel for cannot capture mutable variable 'counter'", result.stderr)

    def test_parallel_output_read_is_rejected(self):
        program = PROGRAMS / "compile_errors" / "parallel_output_read.fl"
        result = self.run_fluid("check", program)
        self.assertEqual(result.returncode, 1)
        self.assertIn("parallel output 'values' cannot be read", result.stderr)


if __name__ == "__main__":
    unittest.main()
