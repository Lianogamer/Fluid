#!/usr/bin/env python3
"""Fluid's starter compiler: a small typed language with a native C backend."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class FluidError(Exception):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    column: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    column = 1
    while i < len(source):
        char = source[i]
        if char in " \t\r":
            i += 1
            column += 1
            continue
        if char == "\n":
            i += 1
            line += 1
            column = 1
            continue
        if source.startswith("//", i):
            end = source.find("\n", i)
            if end == -1:
                column += len(source) - i
                i = len(source)
                continue
            column += end - i
            i = end
            continue

        start_line, start_column = line, column
        if source.startswith("..", i):
            tokens.append(Token("symbol", "..", start_line, start_column))
            i += 2
            column += 2
            continue
        if char == '"':
            start = i
            i += 1
            column += 1
            escaped = False
            while i < len(source):
                current = source[i]
                if current == "\n":
                    raise FluidError(f"{line}:{column}: newline in string literal")
                i += 1
                column += 1
                if current == '"' and not escaped:
                    break
                if current == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
            else:
                raise FluidError(f"{start_line}:{start_column}: unterminated string literal")
            raw = source[start:i]
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise FluidError(f"{start_line}:{start_column}: invalid string escape: {error.msg}") from error
            tokens.append(Token("string", value, start_line, start_column))
            continue

        number = re.match(r"(?:[0-9]+(?:\.(?!\.)[0-9]*)?|\.(?!\.)[0-9]+)", source[i:])
        if number:
            value = number.group(0)
            tokens.append(Token("number", value, start_line, start_column))
            i += len(value)
            column += len(value)
            continue

        identifier = re.match(r"[A-Za-z_][A-Za-z_0-9]*", source[i:])
        if identifier:
            value = identifier.group(0)
            tokens.append(Token("identifier", value, start_line, start_column))
            i += len(value)
            column += len(value)
            continue

        operator = next((op for op in ("->", "==", "!=", "<=", ">=", "&&", "||")
                         if source.startswith(op, i)), None)
        if operator:
            tokens.append(Token("symbol", operator, start_line, start_column))
            i += len(operator)
            column += len(operator)
            continue
        if char in "(){}:;,=+-*/%!<>":
            tokens.append(Token("symbol", char, start_line, start_column))
            i += 1
            column += 1
            continue
        raise FluidError(f"{line}:{column}: unexpected character {char!r}")

    tokens.append(Token("eof", "", line, column))
    return tokens


@dataclass
class Expr:
    kind: str
    value: object
    children: list[Expr]
    token: Token


@dataclass
class Statement:
    kind: str
    value: object
    expression: Expr | None
    body: list[Statement] | None
    alternative: list[Statement] | None
    token: Token


@dataclass
class Function:
    name: str
    parameters: list[tuple[str, str]]
    return_type: str
    body: list[Statement]
    token: Token


class Parser:
    PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
        "+": 4, "-": 4,
        "*": 5, "/": 5, "%": 5,
    }

    def __init__(self, source: str):
        self.tokens = tokenize(source)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def fail(self, message: str, token: Token | None = None) -> None:
        token = token or self.current
        raise FluidError(f"{token.line}:{token.column}: {message}")

    def accept(self, value: str) -> Token | None:
        if self.current.value == value:
            token = self.current
            self.index += 1
            return token
        return None

    def expect(self, value: str) -> Token:
        token = self.accept(value)
        if token is None:
            self.fail(f"expected {value!r}, found {self.current.value or 'end of file'!r}")
        return token

    def expect_identifier(self) -> Token:
        if self.current.kind != "identifier":
            self.fail("expected an identifier")
        token = self.current
        self.index += 1
        return token

    def parse(self) -> list[Function]:
        functions = []
        while self.current.kind != "eof":
            functions.append(self.parse_function())
        if not functions:
            self.fail("a program must define at least one function")
        return functions

    def parse_function(self) -> Function:
        start = self.expect("fn")
        name = self.expect_identifier().value
        self.expect("(")
        parameters = []
        if self.current.value != ")":
            while True:
                parameter = self.expect_identifier().value
                self.expect(":")
                parameters.append((parameter, self.parse_type()))
                if not self.accept(","):
                    break
        self.expect(")")
        self.expect("->")
        return_type = self.parse_type()
        return Function(name, parameters, return_type, self.parse_block(), start)

    def parse_type(self) -> str:
        token = self.expect_identifier()
        aliases = {"int": "i64", "float": "f64", "string": "str"}
        value = aliases.get(token.value, token.value)
        if value not in {"i64", "f64", "bool", "str", "unit"}:
            self.fail(f"unknown type {token.value!r}", token)
        return value

    def parse_block(self) -> list[Statement]:
        self.expect("{")
        statements = []
        while self.current.value != "}":
            if self.current.kind == "eof":
                self.fail("expected '}' before end of file")
            statements.append(self.parse_statement())
        self.expect("}")
        return statements

    def parse_statement(self) -> Statement:
        token = self.current
        if self.accept("let"):
            return self.parse_binding(token, mutable=False)
        if self.accept("var"):
            return self.parse_binding(token, mutable=True)
        if self.accept("print"):
            self.expect("(")
            expression = self.parse_expression()
            self.expect(")")
            self.expect(";")
            return Statement("print", None, expression, None, None, token)
        if self.accept("return"):
            expression = None if self.current.value == ";" else self.parse_expression()
            self.expect(";")
            return Statement("return", None, expression, None, None, token)
        if self.accept("if"):
            condition = self.parse_expression()
            body = self.parse_block()
            alternative = None
            if self.accept("else"):
                if self.current.value == "if":
                    alternative = [self.parse_statement()]
                else:
                    alternative = self.parse_block()
            return Statement("if", None, condition, body, alternative, token)
        if self.accept("while"):
            condition = self.parse_expression()
            return Statement("while", None, condition, self.parse_block(), None, token)
        if self.accept("for"):
            name = self.expect_identifier().value
            self.expect("in")
            start = self.parse_expression()
            self.expect("..")
            end = self.parse_expression()
            return Statement("for", (name, start, end), None, self.parse_block(), None, token)
        if self.accept("break"):
            self.expect(";")
            return Statement("break", None, None, None, None, token)
        if self.accept("continue"):
            self.expect(";")
            return Statement("continue", None, None, None, None, token)
        if self.current.kind == "identifier":
            name = self.current.value
            self.index += 1
            self.expect("=")
            expression = self.parse_expression()
            self.expect(";")
            return Statement("assign", name, expression, None, None, token)
        self.fail(f"unexpected token {self.current.value!r} in statement")
        raise AssertionError("unreachable")

    def parse_binding(self, token: Token, mutable: bool) -> Statement:
        name = self.expect_identifier().value
        annotation = self.parse_type() if self.accept(":") else None
        self.expect("=")
        expression = self.parse_expression()
        self.expect(";")
        kind = "var" if mutable else "let"
        return Statement(kind, (name, annotation), expression, None, None, token)

    def parse_expression(self, minimum_precedence: int = 0) -> Expr:
        token = self.current
        if self.accept("-") or self.accept("!"):
            operator = token.value
            left = Expr("unary", operator, [self.parse_expression(6)], token)
        elif self.accept("("):
            left = self.parse_expression()
            self.expect(")")
        elif token.kind == "number":
            self.index += 1
            kind = "float" if "." in token.value else "integer"
            left = Expr(kind, token.value, [], token)
        elif token.kind == "string":
            self.index += 1
            left = Expr("string", token.value, [], token)
        elif token.kind == "identifier" and token.value in {"true", "false"}:
            self.index += 1
            left = Expr("boolean", token.value, [], token)
        elif token.kind == "identifier":
            self.index += 1
            if self.accept("("):
                arguments = []
                if self.current.value != ")":
                    while True:
                        arguments.append(self.parse_expression())
                        if not self.accept(","):
                            break
                self.expect(")")
                left = Expr("call", token.value, arguments, token)
            else:
                left = Expr("name", token.value, [], token)
        else:
            self.fail("expected an expression")

        while self.current.value in self.PRECEDENCE:
            precedence = self.PRECEDENCE[self.current.value]
            if precedence < minimum_precedence:
                break
            operator = self.current
            self.index += 1
            right = self.parse_expression(precedence + 1)
            left = Expr("binary", operator.value, [left, right], operator)
        return left


class Compiler:
    C_TYPES = {
        "i64": "long long",
        "f64": "double",
        "bool": "bool",
        "str": "const char *",
        "unit": "void",
    }

    def __init__(self, functions: list[Function]):
        self.functions = functions
        self.function_map: dict[str, Function] = {}
        self.scopes: list[dict[str, tuple[str, bool]]] = []
        self.return_type = "unit"
        self.indent = 0
        self.loop_depth = 0
        self.range_counter = 0
        for function in functions:
            if function.name in self.function_map:
                self.fail(f"function {function.name!r} is already defined", function.token)
            parameter_names = [name for name, _ in function.parameters]
            if len(parameter_names) != len(set(parameter_names)):
                self.fail("function parameter names must be unique", function.token)
            self.function_map[function.name] = function
        if "main" not in self.function_map:
            self.fail("program must define fn main()")
        main = self.function_map["main"]
        if main.parameters or main.return_type not in {"i64", "unit"}:
            self.fail("main must have no parameters and return i64 or unit", main.token)

    @staticmethod
    def fail(message: str, token: Token) -> None:
        raise FluidError(f"{token.line}:{token.column}: {message}")

    @staticmethod
    def c_type(type_name: str) -> str:
        return Compiler.C_TYPES[type_name]

    def lookup(self, name: str, token: Token) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        self.fail(f"unknown variable {name!r}", token)
        raise AssertionError("unreachable")

    def infer(self, expression: Expr) -> str:
        if expression.kind == "integer":
            value = int(str(expression.value))
            if value > 9_223_372_036_854_775_807:
                self.fail("integer literal is outside the i64 range", expression.token)
            return "i64"
        if expression.kind == "float":
            return "f64"
        if expression.kind == "string":
            return "str"
        if expression.kind == "boolean":
            return "bool"
        if expression.kind == "name":
            return self.lookup(str(expression.value), expression.token)
        if expression.kind == "call":
            name = str(expression.value)
            function = self.function_map.get(name)
            if function is None:
                self.fail(f"unknown function {name!r}", expression.token)
            if len(expression.children) != len(function.parameters):
                self.fail(f"{name} expects {len(function.parameters)} argument(s)", expression.token)
            for argument, (_, expected) in zip(expression.children, function.parameters):
                actual = self.infer(argument)
                if actual != expected:
                    self.fail(f"{name} expects {expected}, got {actual}", argument.token)
            return function.return_type
        if expression.kind == "unary":
            actual = self.infer(expression.children[0])
            if expression.value == "-" and actual in {"i64", "f64"}:
                return actual
            if expression.value == "!" and actual == "bool":
                return "bool"
            self.fail(f"operator {expression.value!r} does not accept {actual}", expression.token)
        if expression.kind == "binary":
            left, right = (self.infer(child) for child in expression.children)
            operator = str(expression.value)
            if operator in {"&&", "||"}:
                if left == right == "bool":
                    return "bool"
            elif operator in {"==", "!="}:
                if left == right and left in {"i64", "f64", "bool"}:
                    return "bool"
            elif operator in {"<", "<=", ">", ">="}:
                if left == right and left in {"i64", "f64"}:
                    return "bool"
            elif operator in {"+", "-", "*", "/", "%"}:
                allowed = {"i64", "f64"} if operator != "%" else {"i64"}
                if left == right and left in allowed:
                    return left
            self.fail(f"operator {operator!r} cannot be used with {left} and {right}", expression.token)
        self.fail("invalid expression", expression.token)
        raise AssertionError("unreachable")

    def emit_expression(self, expression: Expr) -> str:
        kind = expression.kind
        if kind == "integer":
            return f"{expression.value}LL"
        if kind == "float":
            value = str(expression.value)
            return value if "." in value else value + ".0"
        if kind == "string":
            return json.dumps(expression.value, ensure_ascii=False)
        if kind == "boolean":
            return "true" if expression.value == "true" else "false"
        if kind == "name":
            return f"v_{expression.value}"
        if kind == "call":
            arguments = ", ".join(self.emit_expression(arg) for arg in expression.children)
            return f"fluid_{expression.value}({arguments})"
        if kind == "unary":
            return f"({expression.value}{self.emit_expression(expression.children[0])})"
        if kind == "binary":
            left, right = (self.emit_expression(child) for child in expression.children)
            return f"({left} {expression.value} {right})"
        raise AssertionError(f"unhandled expression: {kind}")

    def line(self, text: str) -> str:
        return "    " * self.indent + text + "\n"

    def emit_block(self, statements: list[Statement]) -> str:
        self.scopes.append({})
        result = self.line("{")
        self.indent += 1
        for statement in statements:
            result += self.emit_statement(statement)
        self.indent -= 1
        result += self.line("}")
        self.scopes.pop()
        return result

    def emit_control_flow(self, statement: Statement) -> str:
        if self.loop_depth == 0:
            self.fail(f"{statement.kind} is only valid inside a loop", statement.token)
        return self.line(f"{statement.kind};")

    @staticmethod
    def always_returns(statements: list[Statement]) -> bool:
        for statement in statements:
            if statement.kind == "return":
                return True
            if (statement.kind == "if" and statement.alternative is not None
                    and Compiler.always_returns(statement.body)
                    and Compiler.always_returns(statement.alternative)):
                return True
        return False

    def emit_statement(self, statement: Statement) -> str:
        if statement.kind in {"let", "var"}:
            name, annotation = statement.value
            actual = self.infer(statement.expression)
            declared = annotation or actual
            if declared != actual:
                self.fail(f"cannot assign {actual} to {declared}", statement.token)
            if declared == "unit":
                self.fail("a variable cannot have type unit", statement.token)
            scope = self.scopes[-1]
            if name in scope:
                self.fail(f"variable {name!r} is already defined in this scope", statement.token)
            scope[name] = (declared, statement.kind == "var")
            value = self.emit_expression(statement.expression)
            return self.line(f"{self.c_type(declared)} v_{name} = {value};")
        if statement.kind == "assign":
            name = str(statement.value)
            declared = None
            mutable = False
            for scope in reversed(self.scopes):
                if name in scope:
                    declared, mutable = scope[name]
                    break
            if declared is None:
                self.fail(f"unknown variable {name!r}", statement.token)
            if not mutable:
                self.fail(f"cannot assign to immutable let binding {name!r}; use var", statement.token)
            actual = self.infer(statement.expression)
            if actual != declared:
                self.fail(f"cannot assign {actual} to {declared}", statement.token)
            return self.line(f"v_{name} = {self.emit_expression(statement.expression)};")
        if statement.kind == "print":
            value_type = self.infer(statement.expression)
            value = self.emit_expression(statement.expression)
            if value_type == "i64":
                return self.line(f'printf("%lld\\n", {value});')
            if value_type == "f64":
                return self.line(f'printf("%g\\n", {value});')
            if value_type == "bool":
                return self.line(f'printf("%s\\n", {value} ? "true" : "false");')
            if value_type == "str":
                return self.line(f'printf("%s\\n", {value});')
            self.fail("print cannot display a unit value", statement.token)
        if statement.kind == "return":
            if statement.expression is None:
                if self.return_type != "unit":
                    self.fail(f"expected a {self.return_type} return value", statement.token)
                return self.line("return;")
            actual = self.infer(statement.expression)
            if actual != self.return_type:
                self.fail(f"expected {self.return_type} return value, got {actual}", statement.token)
            return self.line(f"return {self.emit_expression(statement.expression)};")
        if statement.kind in {"break", "continue"}:
            return self.emit_control_flow(statement)
        if statement.kind == "for":
            name, start, end = statement.value
            for label, expression in (("start", start), ("end", end)):
                if self.infer(expression) != "i64":
                    self.fail(f"for loop {label} must have type i64", expression.token)

            range_id = self.range_counter
            self.range_counter += 1
            start_name = f"fluid_range_start_{range_id}"
            end_name = f"fluid_range_end_{range_id}"

            result = self.line("{")
            self.indent += 1
            result += self.line(f"long long {start_name} = {self.emit_expression(start)};")
            result += self.line(f"long long {end_name} = {self.emit_expression(end)};")
            self.scopes.append({name: ("i64", False)})
            self.loop_depth += 1
            try:
                loop = self.line(
                    f"for (long long v_{name} = {start_name}; "
                    f"v_{name} < {end_name}; ++v_{name})"
                )
                result += loop + self.emit_block(statement.body)
            finally:
                self.loop_depth -= 1
                self.scopes.pop()
                self.indent -= 1
            return result + self.line("}")
        if statement.kind in {"if", "while"}:
            if self.infer(statement.expression) != "bool":
                self.fail(f"{statement.kind} condition must have type bool", statement.token)
            condition = self.emit_expression(statement.expression)
            if statement.kind == "while":
                self.loop_depth += 1
                try:
                    body = self.emit_block(statement.body)
                finally:
                    self.loop_depth -= 1
                return self.line(f"while ({condition})") + body
            result = self.line(f"if ({condition})") + self.emit_block(statement.body)
            if statement.alternative is not None:
                result += self.line("else") + self.emit_block(statement.alternative)
            return result
        raise AssertionError(f"unhandled statement: {statement.kind}")

    def compile(self) -> str:
        output = [
            "#include <stdbool.h>",
            "#include <stdio.h>",
            "",
        ]
        for function in self.functions:
            parameters = ", ".join(
                f"{self.c_type(type_name)} v_{name}" for name, type_name in function.parameters
            ) or "void"
            output.append(f"{self.c_type(function.return_type)} fluid_{function.name}({parameters});")
        output.append("")
        for function in self.functions:
            if function.return_type != "unit" and not self.always_returns(function.body):
                self.fail(
                    f"function {function.name!r} may finish without returning {function.return_type}",
                    function.token,
                )
            self.return_type = function.return_type
            self.scopes = [{}]
            self.loop_depth = 0
            self.range_counter = 0
            parameters = ", ".join(
                f"{self.c_type(type_name)} v_{name}" for name, type_name in function.parameters
            ) or "void"
            output.append(f"{self.c_type(function.return_type)} fluid_{function.name}({parameters})")
            output.append("{")
            self.indent = 1
            for name, type_name in function.parameters:
                self.scopes[0][name] = (type_name, False)
            for statement in function.body:
                output.append(self.emit_statement(statement).rstrip("\n"))
            if function.return_type == "unit":
                output.append(self.line("return;").rstrip("\n"))
            output.append("}")
            output.append("")
        if self.function_map["main"].return_type == "unit":
            output.extend(["int main(void)", "{", "    fluid_main();", "    return 0;", "}"])
        else:
            output.extend(["int main(void)", "{", "    return (int)fluid_main();", "}"])
        return "\n".join(output) + "\n"


def parse_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise FluidError(f"cannot read {path}: {error}") from error


def compiler_command(override: str | None) -> list[str]:
    candidates = [override] if override else ["cc", "clang", "gcc", "cl"]
    for candidate in candidates:
        if not candidate:
            continue
        command = shlex.split(candidate, posix=os.name != "nt")
        executable = command[0]
        resolved = shutil.which(executable)
        if resolved:
            command[0] = resolved
            return command
    requested = override or "cc, clang, gcc, or cl"
    raise FluidError(f"native C compiler not found ({requested}); install one or pass --cc")


def build_binary(c_source: str, output: Path, override: str | None) -> None:
    command = compiler_command(override)
    is_msvc = Path(command[0]).name.lower() in {"cl", "cl.exe"}
    with tempfile.TemporaryDirectory(prefix="fluid-") as directory:
        work = Path(directory)
        source_file = work / "program.c"
        source_file.write_text(c_source, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        if is_msvc:
            args = command + ["/nologo", "/O2", "/std:c11", f"/Fe:{output.resolve()}", str(source_file)]
        else:
            args = command + ["-O3", "-std=c11", str(source_file), "-o", str(output.resolve())]
        try:
            subprocess.run(args, check=True, cwd=work)
        except subprocess.CalledProcessError as error:
            raise FluidError(f"native compiler exited with status {error.returncode}") from error
        except OSError as error:
            raise FluidError(f"could not start native compiler: {error}") from error


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluid", description="Fluid starter compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "emit-c", "build", "run"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("source", type=Path, help="Fluid source file")
        if name in {"emit-c", "build"}:
            subparser.add_argument("-o", "--output", type=Path)
        if name in {"build", "run"}:
            subparser.add_argument("--cc", help="C compiler command (default: cc, clang, gcc, or cl)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        functions = Parser(parse_source(args.source)).parse()
        c_source = Compiler(functions).compile()
        if args.command == "check":
            print(f"{args.source}: OK")
            return 0
        if args.command == "emit-c":
            if args.output:
                args.output.write_text(c_source, encoding="utf-8")
            else:
                sys.stdout.write(c_source)
            return 0
        if args.command == "build":
            output = args.output or args.source.with_suffix(".exe" if os.name == "nt" else "")
            build_binary(c_source, output, args.cc)
            print(f"Built {output}")
            return 0
        if args.command == "run":
            with tempfile.TemporaryDirectory(prefix="fluid-run-") as directory:
                output = Path(directory) / ("program.exe" if os.name == "nt" else "program")
                build_binary(c_source, output, args.cc)
                return subprocess.run([str(output)], check=False).returncode
    except FluidError as error:
        print(f"fluid: error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"fluid: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
