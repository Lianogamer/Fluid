#!/usr/bin/env python3
"""Fluid's starter compiler: a small typed language with a native C backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

        operator = next((op for op in ("->", "=>", "::", "==", "!=", "<=", ">=", "&&", "||")
                         if source.startswith(op, i)), None)
        if operator:
            tokens.append(Token("symbol", operator, start_line, start_column))
            i += len(operator)
            column += len(operator)
            continue
        if char in "(){}[]:;,=+-*/%!<>.&":
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
    module: tuple[str, ...] = ()
    is_public: bool = False
    type_parameters: tuple[str, ...] = ()
    specialized_name: str | None = None

    @property
    def qualified_name(self) -> str:
        if self.specialized_name is not None:
            return self.specialized_name
        prefix = "__".join(self.module)
        return f"{prefix}__{self.name}" if prefix else self.name


@dataclass(frozen=True)
class Import:
    module: tuple[str, ...]
    name: str
    token: Token


@dataclass
class StructDefinition:
    name: str
    fields: list[tuple[str, str]]
    token: Token


@dataclass
class EnumDefinition:
    name: str
    variants: list[str]
    token: Token


@dataclass
class TraitDefinition:
    name: str
    methods: list[Function]
    token: Token


@dataclass
class TraitImplementation:
    trait_name: str
    type_name: str
    methods: list[Function]
    token: Token


@dataclass
class MatchArm:
    enum_name: str
    variant_name: str
    token: Token
    body: list[Statement]


class Program(list[Function]):
    def __init__(self):
        super().__init__()
        self.structs: list[StructDefinition] = []
        self.enums: list[EnumDefinition] = []
        self.traits: list[TraitDefinition] = []
        self.trait_implementations: list[TraitImplementation] = []
        self.module_path: tuple[str, ...] = ()
        self.modules: list[tuple[str, Token]] = []
        self.imports: list[Import] = []
        self.project_imports: dict[tuple[str, ...], list[Import]] = {}


class Parser:
    PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
        "+": 4, "-": 4,
        "*": 5, "/": 5, "%": 5,
    }

    def __init__(self, source: str, module_path: tuple[str, ...] = ()):
        self.tokens = tokenize(source)
        self.index = 0
        self.structs: list[StructDefinition] = []
        self.struct_names: set[str] = set()
        self.enums: list[EnumDefinition] = []
        self.enum_names: set[str] = set()
        self.traits: list[TraitDefinition] = []
        self.trait_implementations: list[TraitImplementation] = []
        self.trait_names: set[str] = set()
        self.active_type_parameters: set[str] = set()
        self.module_path = module_path
        self.modules: list[tuple[str, Token]] = []
        self.imports: list[Import] = []

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def fail(self, message: str, token: Token | None = None) -> None:
        token = token or self.current
        raise FluidError(f"{token.line}:{token.column}: {message}")

    def accept(self, value: str) -> Token | None:
        if self.current.kind != "string" and self.current.value == value:
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
        program = Program()
        while self.current.kind != "eof":
            if self.current.value == "mod":
                token = self.expect("mod")
                name = self.expect_identifier().value
                self.expect(";")
                if any(existing == name for existing, _ in self.modules):
                    self.fail(f"module {name!r} is already declared", token)
                self.modules.append((name, token))
            elif self.current.value == "use":
                token = self.expect("use")
                path = [self.expect_identifier().value]
                while self.accept("::"):
                    path.append(self.expect_identifier().value)
                if len(path) < 2:
                    self.fail("use paths must include a module and a function", token)
                self.expect(";")
                self.imports.append(Import(tuple(path[:-1]), path[-1], token))
            elif self.current.value == "enum":
                if program:
                    self.fail("enum declarations must appear before function definitions")
                definition = self.parse_enum()
                self.enums.append(definition)
                self.enum_names.add(definition.name)
            elif self.current.value == "struct":
                if program:
                    self.fail("struct declarations must appear before function definitions")
                definition = self.parse_struct()
                self.structs.append(definition)
                self.struct_names.add(definition.name)
            elif self.current.value == "trait":
                if program:
                    self.fail("trait declarations must appear before function definitions")
                definition = self.parse_trait()
                self.traits.append(definition)
                self.trait_names.add(definition.name)
            elif self.current.value == "impl":
                if program:
                    self.fail("trait implementations must appear before function definitions")
                self.trait_implementations.append(self.parse_trait_implementation())
            elif self.current.value in {"fn", "pub"}:
                is_public = bool(self.accept("pub"))
                if self.current.value != "fn":
                    self.fail("'pub' must precede a function definition")
                program.append(self.parse_function(is_public))
            else:
                self.fail(
                    "expected a module, import, type, trait, implementation, or function definition"
                )
        if not program and not self.module_path:
            self.fail("a program must define at least one function")
        program.structs = self.structs
        program.enums = self.enums
        program.traits = self.traits
        program.trait_implementations = self.trait_implementations
        program.module_path = self.module_path
        program.modules = self.modules
        program.imports = self.imports
        return program

    def parse_enum(self) -> EnumDefinition:
        token = self.expect("enum")
        name = self.expect_identifier().value
        if (name in self.enum_names or name in self.struct_names
                or name in self.trait_names):
            self.fail(f"type {name!r} is already defined", token)
        self.expect("{")
        variants: list[str] = []
        while self.current.value != "}":
            variant = self.expect_identifier()
            if variant.value in variants:
                self.fail(f"enum variant {variant.value!r} is already defined", variant)
            variants.append(variant.value)
            if self.current.value != "}":
                if not self.accept(",") and not self.accept(";"):
                    self.fail("expected ',' or ';' between enum variants")
        self.expect("}")
        if not variants:
            self.fail("enums must define at least one variant", token)
        return EnumDefinition(name, variants, token)

    def parse_struct(self) -> StructDefinition:
        token = self.expect("struct")
        name = self.expect_identifier().value
        if name in self.enum_names or name in self.trait_names:
            self.fail(f"type {name!r} is already defined", token)
        if name in self.struct_names:
            self.fail(f"struct {name!r} is already defined", token)
        self.expect("{")
        fields = []
        names: set[str] = set()
        while self.current.value != "}":
            field_token = self.current
            field_name = self.expect_identifier().value
            if field_name in names:
                self.fail(f"field {field_name!r} is already defined", field_token)
            names.add(field_name)
            self.expect(":")
            fields.append((field_name, self.parse_type()))
            if self.current.value != "}":
                if not self.accept(",") and not self.accept(";"):
                    self.fail("expected ',' or ';' between struct fields")
        self.expect("}")
        if not fields:
            self.fail("structs must define at least one field", token)
        return StructDefinition(name, fields, token)

    def parse_trait(self) -> TraitDefinition:
        token = self.expect("trait")
        name_token = self.expect_identifier()
        if name_token.value in self.trait_names:
            self.fail(f"trait {name_token.value!r} is already defined", name_token)
        if (name_token.value in self.struct_names or name_token.value in self.enum_names
                or name_token.value in self.trait_names):
            self.fail(f"type {name_token.value!r} is already defined", name_token)
        self.expect("{")
        methods = []
        names: set[str] = set()
        previous_parameters = set(self.active_type_parameters)
        self.active_type_parameters.add("Self")
        while self.current.value != "}":
            method = self.parse_function_signature()
            if method.name in names:
                self.fail(f"trait method {method.name!r} is already defined", method.token)
            names.add(method.name)
            self.expect(";")
            methods.append(method)
        self.expect("}")
        self.active_type_parameters = previous_parameters
        if not methods:
            self.fail("traits must declare at least one method", token)
        return TraitDefinition(name_token.value, methods, token)

    def parse_trait_implementation(self) -> TraitImplementation:
        token = self.expect("impl")
        trait_name = self.expect_identifier().value
        self.expect("for")
        type_token = self.expect_identifier()
        self.expect("{")
        methods = []
        names: set[str] = set()
        while self.current.value != "}":
            method = self.parse_function()
            if method.name in names:
                self.fail(f"trait method {method.name!r} is implemented more than once",
                          method.token)
            names.add(method.name)
            methods.append(method)
        self.expect("}")
        return TraitImplementation(trait_name, type_token.value, methods, token)

    def parse_function_signature(self) -> Function:
        start = self.expect("fn")
        name = self.expect_identifier().value
        if self.accept("<"):
            self.fail("generic trait methods are not supported", start)
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
        return Function(name, parameters, return_type, [], start, self.module_path)

    def parse_function(self, is_public: bool = False) -> Function:
        start = self.expect("fn")
        name = self.expect_identifier().value
        type_parameters: list[str] = []
        if self.accept("<"):
            while True:
                parameter = self.expect_identifier()
                if parameter.value in type_parameters:
                    self.fail(f"type parameter {parameter.value!r} is already defined",
                              parameter)
                type_parameters.append(parameter.value)
                if not self.accept(","):
                    break
            self.expect(">")
        previous_parameters = set(self.active_type_parameters)
        self.active_type_parameters.update(type_parameters)
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
        body = self.parse_block()
        self.active_type_parameters = previous_parameters
        return Function(name, parameters, return_type, body, start,
                        self.module_path, is_public, tuple(type_parameters))

    def parse_type(self) -> str:
        if self.accept("&"):
            mutable = self.accept("mut") is not None
            borrowed_type = self.parse_type()
            return f"borrow_mut:{borrowed_type}" if mutable else f"borrow:{borrowed_type}"
        if self.accept("move"):
            return f"move:{self.parse_type()}"
        if self.accept("dyn"):
            trait = self.expect_identifier()
            return f"dyn:{trait.value}"
        if self.accept("slice"):
            self.expect("[")
            token = self.expect_identifier()
            aliases = {"int": "i64", "float": "f64", "string": "str"}
            element_type = aliases.get(token.value, token.value)
            if element_type in self.active_type_parameters:
                element_type = f"typevar:{element_type}"
            elif element_type not in {"i64", "f64", "bool", "str"}:
                self.fail(f"invalid slice element type {token.value!r}", token)
            self.expect("]")
            return f"slice:{element_type}"
        if self.accept("["):
            token = self.expect_identifier()
            aliases = {"int": "i64", "float": "f64", "string": "str"}
            element_type = aliases.get(token.value, token.value)
            if element_type in self.active_type_parameters:
                element_type = f"typevar:{element_type}"
            elif element_type in self.enum_names:
                element_type = f"enum:{element_type}"
            elif element_type not in {"i64", "f64", "bool", "str"}:
                self.fail(f"invalid array element type {token.value!r}", token)
            if self.accept("]"):
                return f"vec:{element_type}"
            self.expect(";")
            length = self.current
            if length.kind != "number" or "." in length.value:
                self.fail("expected a positive integer array length", length)
            self.index += 1
            if int(length.value) < 1:
                self.fail("array length must be at least 1", length)
            self.expect("]")
            return f"array:{element_type}:{length.value}"

        token = self.expect_identifier()
        aliases = {"int": "i64", "float": "f64", "string": "str"}
        value = aliases.get(token.value, token.value)
        if value in self.active_type_parameters:
            return f"typevar:{value}"
        if value == "String":
            return "String"
        if value in self.enum_names:
            return f"enum:{value}"
        if value in {"i64", "f64", "bool", "str", "unit", "arena", "window", "task_pool"}:
            return value
        if value in {"channel_i64", "channel_f64", "channel_bool"}:
            return f"channel:{value.removeprefix('channel_')}"
        return f"struct:{value}"

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
        if self.accept("assert"):
            self.expect("(")
            expression = self.parse_expression()
            self.expect(")")
            self.expect(";")
            return Statement("assert", None, expression, None, None, token)
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
        if self.accept("match"):
            expression = self.parse_expression()
            self.expect("{")
            arms: list[MatchArm] = []
            while self.current.value != "}":
                enum_name = self.expect_identifier()
                self.expect("::")
                variant = self.expect_identifier()
                arm_token = enum_name
                self.expect("=>")
                arms.append(MatchArm(enum_name.value, variant.value, arm_token,
                                     self.parse_block()))
            self.expect("}")
            if not arms:
                self.fail("match must contain at least one arm", token)
            return Statement("match", arms, expression, None, None, token)
        if self.accept("parallel"):
            self.expect("for")
            name = self.expect_identifier().value
            self.expect("in")
            start = self.parse_expression()
            self.expect("..")
            end = self.parse_expression()
            return Statement("parallel_for", (name, start, end), None, self.parse_block(), None, token)
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
            target = self.parse_expression()
            if target.kind == "call" and self.accept(";"):
                return Statement("expression", None, target, None, None, token)
            if target.kind not in {"name", "index", "field"}:
                self.fail("assignment target must be a variable or array element", token)
            self.expect("=")
            expression = self.parse_expression()
            self.expect(";")
            return Statement("assign", target, expression, None, None, token)
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
        if self.accept("-") or self.accept("!") or self.accept("move"):
            operator = token.value
            precedence = 6 if operator in {"-", "!"} else 6
            left = Expr("unary", operator, [self.parse_expression(precedence)], token)
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
            if self.accept("::"):
                variant = self.expect_identifier()
                left = Expr("enum_variant", (token.value, variant.value), [], token)
            elif self.accept("("):
                arguments = []
                if self.current.value != ")":
                    while True:
                        arguments.append(self.parse_expression())
                        if not self.accept(","):
                            break
                self.expect(")")
                left = Expr("call", token.value, arguments, token)
            elif token.value in self.struct_names and self.accept("{"):
                field_names = []
                values = []
                while self.current.value != "}":
                    field_name = self.expect_identifier().value
                    self.expect(":")
                    field_names.append(field_name)
                    values.append(self.parse_expression())
                    if self.current.value != "}":
                        self.expect(",")
                self.expect("}")
                left = Expr("struct", (token.value, field_names), values, token)
            else:
                left = Expr("name", token.value, [], token)
        elif self.accept("["):
            elements = []
            if self.current.value != "]":
                while True:
                    elements.append(self.parse_expression())
                    if not self.accept(","):
                        break
            self.expect("]")
            left = Expr("array", None, elements, token)
        else:
            self.fail("expected an expression")

        while True:
            if self.accept("["):
                index = self.parse_expression()
                self.expect("]")
                left = Expr("index", None, [left, index], token)
                continue
            if self.accept("."):
                field = self.expect_identifier()
                if self.accept("("):
                    arguments = []
                    if self.current.value != ")":
                        while True:
                            arguments.append(self.parse_expression())
                            if not self.accept(","):
                                break
                    self.expect(")")
                    left = Expr("method_call", field.value, [left, *arguments], field)
                else:
                    left = Expr("field", field.value, [left], field)
                continue
            if self.current.value not in self.PRECEDENCE:
                break
            precedence = self.PRECEDENCE[self.current.value]
            if precedence < minimum_precedence:
                break
            operator = self.current
            self.index += 1
            right = self.parse_expression(precedence + 1)
            left = Expr("binary", operator.value, [left, right], operator)
        return left


class Compiler:
    CHANNEL_TYPES = {
        "channel_i64": "i64",
        "channel_f64": "f64",
        "channel_bool": "bool",
    }
    TASK_RESULT_TYPES = {
        "task_submit_i64": "i64",
        "task_submit_f64": "f64",
        "task_submit_bool": "bool",
    }
    CONCURRENCY_BUILTINS = {
        "task_pool",
        "task_pool_wait",
        "task_pool_close",
        *TASK_RESULT_TYPES,
        "task_result_i64",
        "task_result_f64",
        "task_result_bool",
        "channel_send_i64",
        "channel_send_f64",
        "channel_send_bool",
        "channel_receive_i64",
        "channel_receive_f64",
        "channel_receive_bool",
        "channel_try_receive_i64",
        "channel_try_receive_f64",
        "channel_try_receive_bool",
        "channel_close_i64",
        "channel_close_f64",
        "channel_close_bool",
    }
    WINDOW_BUILTINS = {
        "window_create": (("str", "i64", "i64"), "window"),
        "window_show": (("window",), "unit"),
        "window_hide": (("window",), "unit"),
        "window_close": (("window",), "unit"),
        "window_poll": (("window",), "bool"),
        "window_wait": (("window",), "bool"),
        "window_is_open": (("window",), "bool"),
        "window_set_title": (("window", "str"), "unit"),
        "window_move": (("window", "i64", "i64"), "unit"),
        "window_resize": (("window", "i64", "i64"), "unit"),
        "window_x": (("window",), "i64"),
        "window_y": (("window",), "i64"),
        "window_width": (("window",), "i64"),
        "window_height": (("window",), "i64"),
        "window_add_button": (("window", "i64", "str", "i64", "i64", "i64", "i64"), "unit"),
        "window_add_checkbox": (("window", "i64", "str", "i64", "i64", "i64", "i64"), "unit"),
        "window_add_label": (("window", "i64", "str", "i64", "i64", "i64", "i64"), "unit"),
        "window_add_textbox": (("window", "i64", "str", "i64", "i64", "i64", "i64"), "unit"),
        "window_textbox_text": (("window", "i64"), "str"),
        "window_set_textbox_text": (("window", "i64", "str"), "unit"),
        "window_set_control_text": (("window", "i64", "str"), "unit"),
        "window_set_control_enabled": (("window", "i64", "bool"), "unit"),
        "window_checkbox_checked": (("window", "i64"), "bool"),
        "window_set_checkbox_checked": (("window", "i64", "bool"), "unit"),
        "window_next_event": (("window",), "window_event"),
        "window_clear": (("window", "i64"), "unit"),
        "window_fill_rect": (("window", "i64", "i64", "i64", "i64", "i64"), "unit"),
        "window_fill_ellipse": (("window", "i64", "i64", "i64", "i64", "i64"), "unit"),
        "window_draw_rect": (("window", "i64", "i64", "i64", "i64", "i64"), "unit"),
        "window_draw_ellipse": (("window", "i64", "i64", "i64", "i64", "i64"), "unit"),
        "window_draw_line": (("window", "i64", "i64", "i64", "i64", "i64"), "unit"),
        "window_draw_text": (("window", "i64", "i64", "str", "i64"), "unit"),
        "window_present": (("window",), "unit"),
        "window_present_rect": (("window", "i64", "i64", "i64", "i64"), "unit"),
        "event_none": ((), "i64"),
        "event_key_down": ((), "i64"),
        "event_key_up": ((), "i64"),
        "event_mouse_move": ((), "i64"),
        "event_mouse_down": ((), "i64"),
        "event_mouse_up": ((), "i64"),
        "event_mouse_wheel": ((), "i64"),
        "event_text": ((), "i64"),
        "event_resize": ((), "i64"),
        "event_close": ((), "i64"),
        "event_control": ((), "i64"),
    }

    BUILTINS = {
        "len": (None, "i64", None),
        "slice": (None, "unit", None),
        "arena": ((), "arena", None),
        "worker_arena": ((), "arena_ref", None),
        "arena_i64": (None, "slice:i64", None),
        "arena_f64": (None, "slice:f64", None),
        "arena_bool": (None, "slice:bool", None),
        "arena_str": (None, "slice:str", None),
        "window_create": (None, "window", None),
        "window_show": (None, "unit", None),
        "window_hide": (None, "unit", None),
        "window_close": (None, "unit", None),
        "window_poll": (None, "bool", None),
        "window_wait": (None, "bool", None),
        "window_is_open": (None, "bool", None),
        "window_set_title": (None, "unit", None),
        "window_move": (None, "unit", None),
        "window_resize": (None, "unit", None),
        "window_x": (None, "i64", None),
        "window_y": (None, "i64", None),
        "window_width": (None, "i64", None),
        "window_height": (None, "i64", None),
        "window_next_event": (None, "window_event", None),
        "window_clear": (None, "unit", None),
        "window_fill_rect": (None, "unit", None),
        "window_fill_ellipse": (None, "unit", None),
        "window_draw_rect": (None, "unit", None),
        "window_draw_ellipse": (None, "unit", None),
        "window_draw_line": (None, "unit", None),
        "window_draw_text": (None, "unit", None),
        "window_present": (None, "unit", None),
        "window_present_rect": (None, "unit", None),
        "event_none": ((), "i64", None),
        "event_key_down": ((), "i64", None),
        "event_key_up": ((), "i64", None),
        "event_mouse_move": ((), "i64", None),
        "event_mouse_down": ((), "i64", None),
        "event_mouse_up": ((), "i64", None),
        "event_mouse_wheel": ((), "i64", None),
        "event_text": ((), "i64", None),
        "event_resize": ((), "i64", None),
        "event_close": ((), "i64", None),
        "window_next_event": (None, "window_event", None),
        "window_clear": (None, "unit", None),
        "window_fill_rect": (None, "unit", None),
        "window_fill_ellipse": (None, "unit", None),
        "window_draw_rect": (None, "unit", None),
        "window_draw_ellipse": (None, "unit", None),
        "window_draw_line": (None, "unit", None),
        "window_draw_text": (None, "unit", None),
        "window_present": (None, "unit", None),
        "window_present_rect": (None, "unit", None),
        "event_none": ((), "i64", None),
        "event_key_down": ((), "i64", None),
        "event_key_up": ((), "i64", None),
        "event_mouse_move": ((), "i64", None),
        "event_mouse_down": ((), "i64", None),
        "event_mouse_up": ((), "i64", None),
        "event_text": ((), "i64", None),
        "event_resize": ((), "i64", None),
        "event_close": ((), "i64", None),
        "push": (None, "unit", None),
        "pop": (None, "unit", None),
        "insert": (None, "unit", None),
        "remove": (None, "unit", None),
        "reserve": (None, "unit", None),
        "shrink_to_fit": (None, "unit", None),
        "capacity": (None, "i64", None),
        "clear": (None, "unit", None),
        "abs": (("i64",), "i64", "fluid_abs_i64"),
        "min": (("i64", "i64"), "i64", "fluid_min_i64"),
        "max": (("i64", "i64"), "i64", "fluid_max_i64"),
        "clamp": (("i64", "i64", "i64"), "i64", "fluid_clamp_i64"),
        "sign": (("i64",), "i64", "fluid_sign_i64"),
        "sqrt": (("f64",), "f64", "sqrt"),
        "pow": (("f64", "f64"), "f64", "pow"),
        "floor": (("f64",), "f64", "floor"),
        "ceil": (("f64",), "f64", "ceil"),
        "round": (("f64",), "f64", "round"),
        "contains": (("str", "str"), "bool", "fluid_contains"),
        "starts_with": (("str", "str"), "bool", "fluid_starts_with"),
        "ends_with": (("str", "str"), "bool", "fluid_ends_with"),
        "is_empty": (("str",), "bool", "fluid_is_empty"),
        "string_from": (("str",), "String", "fluid_string_from"),
        "string_clone": (("String",), "String", "fluid_string_clone"),
        "string_concat": (("String", "String"), "String", "fluid_string_concat"),
        "string_length": (("String",), "i64", "fluid_string_length"),
        "string_equal": (("String", "String"), "bool", "fluid_string_equal"),
        "string_slice": (("String", "i64", "i64"), "String", "fluid_string_slice"),
        "string_append": (None, "unit", None),
        "string_clear": (None, "unit", None),
    }

    C_TYPES = {
        "i64": "long long",
        "f64": "double",
        "bool": "bool",
        "str": "const char *",
        "unit": "void",
        "window": "fluid_window",
        "window_event": "fluid_window_event",
        "task_pool": "fluid_task_pool *",
        "String": "fluid_string",
    }

    def __init__(self, functions: list[Function]):
        all_functions = list(functions)
        self.generic_templates: dict[tuple[tuple[str, ...], str], Function] = {}
        self.generic_specializations: dict[
            tuple[tuple[str, ...], str, tuple[str, ...]], Function
        ] = {}
        self.functions = [function for function in all_functions if not function.type_parameters]
        self.structs = list(getattr(functions, "structs", []))
        self.enums = list(getattr(functions, "enums", []))
        self.traits = list(getattr(functions, "traits", []))
        self.trait_implementations = list(
            getattr(functions, "trait_implementations", [])
        )
        self.project_imports = getattr(functions, "project_imports", {})
        self.struct_map: dict[str, StructDefinition] = {}
        self.struct_fields: dict[str, dict[str, str]] = {}
        self.enum_map: dict[str, EnumDefinition] = {}
        self.trait_map: dict[str, TraitDefinition] = {}
        self.trait_impl_map: dict[tuple[str, str], dict[str, Function]] = {}
        self.trait_methods: dict[tuple[str, str], Function] = {}
        self.enum_match_checked: set[int] = set()
        self.function_map: dict[str, Function] = {}
        self.module_functions: dict[tuple[tuple[str, ...], str], Function] = {}
        self.function_imports: dict[tuple[str, ...], dict[str, Function]] = {}
        self.current_module: tuple[str, ...] = ()
        self.scopes: list[dict[str, tuple[str, bool]]] = []
        self.vector_c_names: list[dict[str, str]] = []
        self.slice_sources: list[dict[str, str | None]] = []
        self.moved_vectors: set[tuple[int, str]] = set()
        self.return_type = "unit"
        self.indent = 0
        self.loop_depth = 0
        self.loop_scope_depths: list[int] = []
        self.range_counter = 0
        self.return_counter = 0
        self.vector_counter = 0
        self.parallel_counter = 0
        self.parallel_workers: list[str] = []
        self.task_workers: dict[str, Function] = {}
        self.has_strings = False
        self.match_counter = 0
        self.c_name_overrides: dict[tuple[int, str], str] = {}
        self.parallel_worker_depth = 0
        for function in all_functions:
            local_key = (function.module, function.name)
            if local_key in self.module_functions:
                self.fail(f"function {function.name!r} is already defined", function.token)
            if (function.name in self.BUILTINS or function.name in self.WINDOW_BUILTINS
                    or function.name in self.CONCURRENCY_BUILTINS
                    or function.name in self.CHANNEL_TYPES):
                self.fail(f"{function.name!r} is reserved for a built-in function", function.token)
            parameter_names = [name for name, _ in function.parameters]
            if len(parameter_names) != len(set(parameter_names)):
                self.fail("function parameter names must be unique", function.token)
            self.module_functions[local_key] = function
            self.function_map[function.qualified_name] = function
            if function.type_parameters:
                self.generic_templates[local_key] = function
        loaded_modules = {module for module, _ in self.module_functions}
        loaded_modules.update(self.project_imports)
        for module, imports in self.project_imports.items():
            aliases: dict[str, Function] = {}
            local_names = {name for function_module, name in self.module_functions
                           if function_module == module}
            for imported in imports:
                target = self.module_functions.get((imported.module, imported.name))
                if target is None or imported.module not in loaded_modules:
                    self.fail(
                        f"cannot import {imported.name!r} from unknown module "
                        f"{'::'.join(imported.module)!r}",
                        imported.token,
                    )
                if not target.is_public:
                    self.fail(
                        f"function {'::'.join(imported.module + (imported.name,))!r} "
                        "is private; mark it pub to import it",
                        imported.token,
                    )
                if imported.name in local_names:
                    self.fail(
                        f"imported name {imported.name!r} conflicts with a local function",
                        imported.token,
                    )
                if imported.name in aliases:
                    self.fail(f"function {imported.name!r} is imported more than once",
                              imported.token)
                aliases[imported.name] = target
            self.function_imports[module] = aliases
        for definition in self.enums:
            if definition.name in self.enum_map or definition.name in self.struct_map:
                self.fail(f"type {definition.name!r} is already defined", definition.token)
            if (definition.name in self.BUILTINS or definition.name in self.WINDOW_BUILTINS
                    or definition.name in self.CONCURRENCY_BUILTINS
                    or definition.name in self.CHANNEL_TYPES
                    or definition.name in self.function_map):
                self.fail(f"enum name {definition.name!r} is already reserved", definition.token)
            self.enum_map[definition.name] = definition
        for definition in self.structs:
            if definition.name in self.struct_map or definition.name in self.enum_map:
                self.fail(f"struct {definition.name!r} is already defined", definition.token)
            if (definition.name in self.BUILTINS or definition.name in self.WINDOW_BUILTINS
                    or definition.name in self.CONCURRENCY_BUILTINS
                    or definition.name in self.CHANNEL_TYPES
                    or definition.name in self.function_map):
                self.fail(f"struct name {definition.name!r} is already reserved", definition.token)
            fields = dict(definition.fields)
            for _, field_type in definition.fields:
                if (field_type not in {"i64", "f64", "bool", "str"}
                        and not (field_type.startswith("enum:")
                                 and field_type.split(":", 1)[1] in self.enum_map)):
                    self.fail("struct fields must use primitive or enum types for now",
                              definition.token)
            self.struct_map[definition.name] = definition
            self.struct_fields[definition.name] = fields
        for definition in self.traits:
            if (definition.name in self.trait_map or definition.name in self.struct_map
                    or definition.name in self.enum_map):
                self.fail(f"trait {definition.name!r} is already defined", definition.token)
            if (definition.name in self.BUILTINS or definition.name in self.WINDOW_BUILTINS
                    or definition.name in self.CONCURRENCY_BUILTINS
                    or definition.name in self.CHANNEL_TYPES
                    or definition.name in self.function_map):
                self.fail(f"trait name {definition.name!r} is already reserved",
                          definition.token)
            self.trait_map[definition.name] = definition
            for method in definition.methods:
                if method.type_parameters:
                    self.fail("generic trait methods are not supported", method.token)
                if (not method.parameters or method.parameters[0] !=
                        ("self", "borrow:typevar:Self")):
                    self.fail(
                        "trait methods must start with self: &Self",
                        method.token,
                    )
                parameter_names = [name for name, _ in method.parameters]
                if len(parameter_names) != len(set(parameter_names)):
                    self.fail("trait method parameter names must be unique", method.token)
                if method.return_type not in {"i64", "f64", "bool", "unit"}:
                    self.fail(
                        "trait methods must return i64, f64, bool, or unit",
                        method.token,
                    )
                for _, parameter_type in method.parameters[1:]:
                    if parameter_type not in {"i64", "f64", "bool", "str"}:
                        self.fail(
                            "trait method arguments must use i64, f64, bool, or str",
                            method.token,
                        )
                self.trait_methods[(definition.name, method.name)] = method
        for implementation in self.trait_implementations:
            trait = self.trait_map.get(implementation.trait_name)
            if trait is None:
                self.fail(
                    f"unknown trait {implementation.trait_name!r}",
                    implementation.token,
                )
            if implementation.type_name not in self.struct_map:
                self.fail(
                    f"trait implementations require a known struct, got "
                    f"{implementation.type_name!r}",
                    implementation.token,
                )
            key = (implementation.trait_name, implementation.type_name)
            if key in self.trait_impl_map:
                self.fail(
                    f"trait {implementation.trait_name!r} is already implemented for "
                    f"{implementation.type_name!r}",
                    implementation.token,
                )
            methods = {method.name: method for method in implementation.methods}
            expected_methods = {method.name: method for method in trait.methods}
            missing = sorted(expected_methods.keys() - methods.keys())
            extra = sorted(methods.keys() - expected_methods.keys())
            if missing or extra:
                detail = []
                if missing:
                    detail.append(f"missing: {', '.join(missing)}")
                if extra:
                    detail.append(f"unknown: {', '.join(extra)}")
                self.fail(
                    f"invalid implementation of {implementation.trait_name} for "
                    f"{implementation.type_name} ({'; '.join(detail)})",
                    implementation.token,
                )
            for name, method in methods.items():
                signature = expected_methods[name]
                if method.type_parameters:
                    self.fail("generic trait methods are not supported", method.token)
                expected_types = [
                    f"borrow:struct:{implementation.type_name}",
                    *(type_name for _, type_name in signature.parameters[1:]),
                ]
                actual_types = [type_name for _, type_name in method.parameters]
                if (not method.parameters or method.parameters[0][0] != "self"
                        or actual_types != expected_types):
                    self.fail(
                        f"method {name!r} has an incompatible parameter list",
                        method.token,
                    )
                parameter_names = [parameter for parameter, _ in method.parameters]
                if len(parameter_names) != len(set(parameter_names)):
                    self.fail("implementation method parameter names must be unique",
                              method.token)
                if method.return_type != signature.return_type:
                    self.fail(
                        f"method {name!r} must return {signature.return_type}",
                        method.token,
                    )
                method.specialized_name = (
                    f"traitimpl__{implementation.trait_name}__"
                    f"{implementation.type_name}__{name}"
                )
            self.trait_impl_map[key] = methods
        self.functions.extend(
            method
            for methods in self.trait_impl_map.values()
            for method in methods.values()
        )
        main = self.module_functions.get(((), "main"))
        if main is None:
            self.fail("program must define fn main()")
        if main.parameters or main.return_type not in {"i64", "unit"}:
            self.fail("main must have no parameters and return i64 or unit", main.token)
        for function in all_functions:
            if function.return_type == "String":
                self.fail("functions cannot return owned String values yet", function.token)
            if function.type_parameters and (
                function.module == () and function.name == "main"
            ):
                self.fail("main cannot be generic", function.token)
            if function.return_type.startswith(("array:", "vec:", "slice:", "arena:", "arena",
                                                "window", "task_pool", "channel:",
                                                "move:", "borrow:", "borrow_mut:", "dyn:")):
                self.fail(
                    "functions cannot return arrays, vectors, arenas, windows, "
                    "or borrowed references",
                    function.token,
                )
            for _, type_name in function.parameters:
                if type_name.startswith("move:") and not type_name[5:].startswith("vec:"):
                    self.fail("move parameters must own a vector", function.token)
                if type_name.startswith(("borrow:", "borrow_mut:")) and not type_name.split(":", 1)[1].startswith(
                    ("vec:", "slice:", "channel:", "struct:")
                ):
                    if type_name.split(":", 1)[1] != "String":
                        self.fail(
                            "borrow parameters must reference a vector, slice, channel, "
                            "struct, or String",
                              function.token)
                if type_name.startswith("vec:"):
                    self.fail("vector parameters must use move or an explicit borrow", function.token)
                if type_name.startswith("array:"):
                    self.fail("arrays cannot be function parameters or return values", function.token)
                if type_name == "arena":
                    self.fail("arenas are local-only; pass slices instead", function.token)
                if type_name == "window":
                    self.fail("windows are local-only and cannot be passed to functions", function.token)
                if type_name == "task_pool":
                    self.fail("task pools are local-only and cannot be passed to functions",
                              function.token)
                if type_name == "String":
                    self.fail("String parameters must be borrowed with &String", function.token)
                if type_name.startswith("channel:"):
                    self.fail("channels must be passed by reference", function.token)
                if type_name.startswith("struct:") and type_name.split(":", 1)[1] not in self.struct_map:
                    self.fail(f"unknown struct type {type_name.split(':', 1)[1]!r}", function.token)
                if type_name.startswith("enum:") and type_name.split(":", 1)[1] not in self.enum_map:
                    self.fail(f"unknown enum type {type_name.split(':', 1)[1]!r}", function.token)
                if type_name.startswith("borrow:struct:") or type_name.startswith(
                    "borrow_mut:struct:"
                ):
                    if type_name.split(":", 2)[2] not in self.struct_map:
                        self.fail(
                            f"unknown struct type {type_name.split(':', 2)[2]!r}",
                            function.token,
                        )
                if type_name.startswith("dyn:") and type_name.split(":", 1)[1] not in self.trait_map:
                    self.fail(f"unknown trait type {type_name.split(':', 1)[1]!r}",
                              function.token)
            if function.return_type.startswith("struct:") and function.return_type.split(":", 1)[1] not in self.struct_map:
                self.fail(f"unknown struct return type {function.return_type.split(':', 1)[1]!r}",
                          function.token)
            if function.return_type.startswith("enum:") and function.return_type.split(":", 1)[1] not in self.enum_map:
                self.fail(f"unknown enum return type {function.return_type.split(':', 1)[1]!r}",
                          function.token)

    @staticmethod
    def fail(message: str, token: Token) -> None:
        raise FluidError(f"{token.line}:{token.column}: {message}")

    @staticmethod
    def c_type(type_name: str) -> str:
        if type_name.startswith("enum:"):
            return "long long"
        if type_name.startswith("channel:"):
            return f"fluid_channel_{type_name.split(':', 1)[1]} *"
        if type_name.startswith("borrow:channel:"):
            return f"fluid_channel_{type_name.split(':', 2)[2]} *"
        if type_name.startswith("struct:"):
            return f"struct fluid_{type_name.split(':', 1)[1]}"
        if type_name.startswith("dyn:"):
            return f"fluid_dyn_{type_name.split(':', 1)[1]}"
        if type_name.startswith(("borrow:struct:", "borrow_mut:struct:")):
            qualifier = "const " if type_name.startswith("borrow:") else ""
            struct_name = type_name.split(":", 2)[2]
            return f"{qualifier}struct fluid_{struct_name} *"
        if type_name in {"borrow:String", "borrow_mut:String"}:
            qualifier = "const " if type_name == "borrow:String" else ""
            return f"{qualifier}fluid_string *"
        if type_name == "arena":
            return "fluid_arena"
        if type_name == "arena_ref":
            return "fluid_arena *"
        if type_name == "window":
            return "fluid_window"
        if type_name.startswith(("slice:", "slice_mut:")):
            return f"fluid_slice_{type_name.split(':', 1)[1]}"
        if type_name.startswith("borrow:slice:"):
            return f"const fluid_slice_{type_name.split(':', 2)[2]} *"
        if type_name.startswith("borrow_mut:slice:"):
            return f"fluid_slice_{type_name.split(':', 2)[2]} *"
        if type_name.startswith("borrow:") and type_name.split(":", 1)[1].startswith("vec:"):
            element_type = Compiler.vector_type_parts(type_name)
            return f"const fluid_slice_{element_type} *"
        if type_name.startswith("borrow:"):
            element_type = Compiler.vector_type_parts(type_name)
            return f"const fluid_vec_{element_type} *"
        if type_name.startswith("borrow_mut:"):
            element_type = Compiler.vector_type_parts(type_name)
            return f"fluid_vec_{element_type} *"
        if type_name.startswith("move:"):
            type_name = type_name[5:]
        array = Compiler.array_type_parts(type_name)
        if array is not None:
            type_name, _ = array
        vector = Compiler.vector_type_parts(type_name)
        if vector is not None:
            return f"fluid_vec_{vector}"
        if type_name.startswith("enum:"):
            return "long long"
        return Compiler.C_TYPES[type_name]

    @staticmethod
    def array_type_parts(type_name: str) -> tuple[str, int] | None:
        if not type_name.startswith("array:"):
            return None
        element_type, length = type_name[len("array:"):].rsplit(":", 1)
        return element_type, int(length)

    @staticmethod
    def vector_type_parts(type_name: str) -> str | None:
        if type_name.startswith(("borrow:", "borrow_mut:", "move:")):
            type_name = type_name.split(":", 1)[1]
        if type_name.startswith("vec:"):
            return type_name.split(":", 1)[1]
        return None

    @staticmethod
    def is_vector_owner(type_name: str) -> bool:
        return type_name.startswith(("vec:", "move:vec:"))

    @staticmethod
    def container_element_type(type_name: str) -> str | None:
        array = Compiler.array_type_parts(type_name)
        if array is not None:
            return array[0]
        vector = Compiler.vector_type_parts(type_name)
        if vector is not None:
            return vector
        if type_name.startswith(("slice:", "slice_mut:")):
            return type_name.split(":", 1)[1]
        if type_name.startswith("borrow:slice:"):
            return type_name.split(":", 2)[2]
        return None

    def lookup(self, name: str, token: Token) -> str:
        for index in range(len(self.scopes) - 1, -1, -1):
            scope = self.scopes[index]
            if name in scope:
                if (index, name) in self.moved_vectors:
                    self.fail(f"use of moved vector {name!r}", token)
                return scope[name][0]
        self.fail(f"unknown variable {name!r}", token)
        raise AssertionError("unreachable")

    def resolve_function(self, name: str, token: Token) -> Function:
        local = self.module_functions.get((self.current_module, name))
        if local is not None:
            return local
        imported = self.function_imports.get(self.current_module, {}).get(name)
        if imported is not None:
            return imported
        self.fail(f"unknown function {name!r}", token)
        raise AssertionError("unreachable")

    @staticmethod
    def substitute_type(type_name: str, substitutions: dict[str, str]) -> str:
        for parameter, concrete in substitutions.items():
            type_name = type_name.replace(f"typevar:{parameter}", concrete)
        return type_name

    def unify_generic_type(
        self, pattern: str, actual: str, substitutions: dict[str, str], token: Token
    ) -> None:
        if pattern.startswith("typevar:"):
            parameter = pattern.split(":", 1)[1]
            previous = substitutions.get(parameter)
            if previous is not None and previous != actual:
                self.fail(
                    f"generic type {parameter} was inferred as both {previous} and {actual}",
                    token,
                )
            substitutions[parameter] = actual
            return
        if pattern.startswith(("borrow:", "borrow_mut:")):
            _, pattern_inner = pattern.split(":", 1)
            actual_inner = actual
            if actual.startswith(("borrow:", "borrow_mut:")):
                actual_inner = actual.split(":", 1)[1]
            self.unify_generic_type(pattern_inner, actual_inner, substitutions, token)
            return
        if pattern.startswith(("vec:", "slice:", "slice_mut:")):
            prefix, pattern_inner = pattern.split(":", 1)
            if not actual.startswith(prefix + ":"):
                self.fail(f"generic argument expects {pattern}, got {actual}", token)
            self.unify_generic_type(
                pattern_inner, actual.split(":", 1)[1], substitutions, token
            )
            return
        if pattern.startswith("array:"):
            pattern_element, pattern_length = pattern[len("array:"):].rsplit(":", 1)
            if not actual.startswith("array:"):
                self.fail(f"generic argument expects {pattern}, got {actual}", token)
            actual_element, actual_length = actual[len("array:"):].rsplit(":", 1)
            if pattern_length != actual_length:
                self.fail(f"generic argument expects {pattern}, got {actual}", token)
            self.unify_generic_type(pattern_element, actual_element, substitutions, token)
            return
        if pattern != actual:
            self.fail(f"generic argument expects {pattern}, got {actual}", token)

    def specialize_generic(
        self, function: Function, arguments: list[Expr], token: Token
    ) -> Function:
        key_base = (function.module, function.name)
        substitutions: dict[str, str] = {}
        if len(arguments) != len(function.parameters):
            self.fail(f"{function.name} expects {len(function.parameters)} argument(s)", token)
        for argument, (_, parameter_type) in zip(arguments, function.parameters):
            self.unify_generic_type(
                parameter_type, self.infer(argument), substitutions, argument.token
            )
        missing = [parameter for parameter in function.type_parameters
                   if parameter not in substitutions]
        if missing:
            self.fail(
                f"cannot infer generic type parameter(s): {', '.join(missing)}",
                token,
            )
        specialization_types = tuple(substitutions[name] for name in function.type_parameters)
        key = (*key_base, specialization_types)
        existing = self.generic_specializations.get(key)
        if existing is not None:
            return existing
        type_codes = [
            re.sub(r"[^A-Za-z0-9_]", "_", type_name).strip("_")
            for type_name in specialization_types
        ]
        prefix = "__".join(function.module)
        stem = f"{prefix}__{function.name}" if prefix else function.name
        specialized_name = f"{stem}__generic_{'_'.join(type_codes)}"

        def specialize_statements(statements: list[Statement]) -> list[Statement]:
            cloned = copy.deepcopy(statements)
            for statement in cloned:
                if statement.kind in {"let", "var"}:
                    name, annotation = statement.value
                    if annotation is not None:
                        statement.value = (
                            name, self.substitute_type(annotation, substitutions)
                        )
                if statement.body is not None:
                    statement.body = specialize_statements(statement.body)
                if statement.alternative is not None:
                    statement.alternative = specialize_statements(statement.alternative)
                if statement.kind == "match":
                    for arm in statement.value:
                        arm.body = specialize_statements(arm.body)
            return cloned

        specialized = Function(
            function.name,
            [(name, self.substitute_type(type_name, substitutions))
             for name, type_name in function.parameters],
            self.substitute_type(function.return_type, substitutions),
            specialize_statements(function.body),
            function.token,
            function.module,
            function.is_public,
            (),
            specialized_name,
        )
        self.generic_specializations[key] = specialized
        self.functions.append(specialized)
        return specialized

    def resolve_call_function(self, name: str, arguments: list[Expr], token: Token) -> Function:
        function = self.resolve_function(name, token)
        if function.type_parameters:
            return self.specialize_generic(function, arguments, token)
        return function

    def require_trait_implementation(
        self, trait_name: str, type_name: str, token: Token
    ) -> None:
        methods = self.trait_impl_map.get((trait_name, type_name))
        if methods is None:
            self.fail(
                f"{type_name!r} does not implement trait {trait_name!r}",
                token,
            )

    def resolve_method(
        self, receiver_type: str, method_name: str, token: Token
    ) -> tuple[str, Function, Function | None]:
        if receiver_type.startswith("dyn:"):
            trait_name = receiver_type.split(":", 1)[1]
            method = self.trait_methods.get((trait_name, method_name))
            if method is None:
                self.fail(
                    f"trait {trait_name!r} has no method {method_name!r}",
                    token,
                )
            return trait_name, method, None

        if receiver_type.startswith("borrow:struct:"):
            type_name = receiver_type.split(":", 2)[2]
        elif receiver_type.startswith("struct:"):
            type_name = receiver_type.split(":", 1)[1]
        else:
            self.fail(
                f"type {receiver_type} has no method {method_name!r}",
                token,
            )
        candidates = [
            (trait_name, methods[method_name])
            for (trait_name, implemented_type), methods in self.trait_impl_map.items()
            if implemented_type == type_name and method_name in methods
        ]
        if not candidates:
            self.fail(
                f"no implementation of method {method_name!r} for {type_name!r}",
                token,
            )
        if len(candidates) != 1:
            self.fail(
                f"method {method_name!r} is ambiguous for {type_name!r}",
                token,
            )
        trait_name, implementation = candidates[0]
        signature = self.trait_methods[(trait_name, method_name)]
        return trait_name, signature, implementation

    def trait_object_value(self, expression: Expr, trait_name: str) -> str:
        actual = self.infer(expression)
        if actual == f"dyn:{trait_name}":
            return self.emit_expression(expression)
        if not actual.startswith("struct:"):
            self.fail(
                f"expected a value implementing {trait_name}, got {actual}",
                expression.token,
            )
        type_name = actual.split(":", 1)[1]
        self.require_trait_implementation(trait_name, type_name, expression.token)
        return (
            f"(fluid_dyn_{trait_name}){{.data = (void *)&("
            f"{self.emit_expression(expression)}), .vtable = "
            f"&fluid_trait_vtable_{trait_name}_{type_name}}}"
        )

    @staticmethod
    def trait_dispatch_name(trait_name: str, type_name: str, method_name: str) -> str:
        return f"fluid_trait_dispatch_{trait_name}_{type_name}_{method_name}"

    @staticmethod
    def trait_vtable_name(trait_name: str, type_name: str) -> str:
        return f"fluid_trait_vtable_{trait_name}_{type_name}"

    def binding_scope(self, name: str, token: Token) -> int:
        for index in range(len(self.scopes) - 1, -1, -1):
            if name in self.scopes[index]:
                return index
        self.fail(f"unknown variable {name!r}", token)
        raise AssertionError("unreachable")

    def c_variable(self, name: str, type_name: str) -> str:
        scope_index = self.binding_scope(name, Token("identifier", name, 1, 1))
        override = self.c_name_overrides.get((scope_index, name))
        if override is not None:
            return override
        if type_name.startswith("borrow:channel:") or type_name == "borrow:String":
            return f"v_{name}"
        if type_name.startswith(("borrow:", "borrow_mut:")):
            return f"(*v_{name})"
        if self.vector_type_parts(type_name) is not None:
            for index in range(len(self.scopes) - 1, -1, -1):
                if name in self.scopes[index]:
                    return self.vector_c_names[index][name]
        return f"v_{name}"

    def require_mutable_vector(self, expression: Expr, operation: str) -> None:
        name = str(expression.value)
        for scope in reversed(self.scopes):
            if name in scope:
                type_name, mutable = scope[name]
                if self.vector_type_parts(type_name) is None:
                    self.fail(f"{operation} expects a vector variable", expression.token)
                if not mutable:
                    self.fail(f"{operation} requires a mutable vector; use var", expression.token)
                return
        self.fail(f"unknown variable {name!r}", expression.token)

    def variable_mutability(self, expression: Expr) -> bool:
        if expression.kind != "name":
            return False
        name = str(expression.value)
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        self.fail(f"unknown variable {name!r}", expression.token)
        raise AssertionError("unreachable")

    def ensure_no_live_slice(self, name: str, token: Token) -> None:
        if any(name in sources.values() for sources in self.slice_sources):
            self.fail(f"cannot mutate {name!r} while a slice into it is live", token)

    def lookup_slice_source(self, name: str) -> str | None:
        for scope in reversed(self.slice_sources):
            if name in scope:
                return scope[name]
        return None

    @staticmethod
    def expression_names(expression: Expr) -> set[str]:
        names = {str(expression.value)} if expression.kind == "name" else set()
        for child in expression.children:
            names.update(Compiler.expression_names(child))
        return names

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
        if expression.kind == "enum_variant":
            enum_name, variant_name = expression.value
            definition = self.enum_map.get(str(enum_name))
            if definition is None:
                self.fail(f"unknown enum {enum_name!r}", expression.token)
            if variant_name not in definition.variants:
                self.fail(
                    f"enum {enum_name!r} has no variant {variant_name!r}",
                    expression.token,
                )
            return f"enum:{enum_name}"
        if expression.kind == "struct":
            struct_name, field_names = expression.value
            definition = self.struct_map.get(struct_name)
            if definition is None:
                self.fail(f"unknown struct {struct_name!r}", expression.token)
            declared_fields = self.struct_fields[struct_name]
            if len(field_names) != len(set(field_names)):
                self.fail("struct literal fields must be unique", expression.token)
            if set(field_names) != set(declared_fields):
                missing = sorted(set(declared_fields) - set(field_names))
                extra = sorted(set(field_names) - set(declared_fields))
                detail = f"missing fields: {', '.join(missing)}" if missing else f"unknown fields: {', '.join(extra)}"
                self.fail(f"invalid {struct_name} literal ({detail})", expression.token)
            for field_name, value in zip(field_names, expression.children):
                actual = self.infer(value)
                expected = declared_fields[field_name]
                if actual != expected:
                    self.fail(f"field {field_name!r} expects {expected}, got {actual}", value.token)
            return f"struct:{struct_name}"
        if expression.kind == "field":
            object_type = self.infer(expression.children[0])
            if object_type == "window_event":
                event_fields = {
                    "kind": "i64",
                    "key": "i64",
                    "x": "i64",
                    "y": "i64",
                    "button": "i64",
                    "delta": "i64",
                    "text": "i64",
                    "control_id": "i64",
                    "checked": "bool",
                    "width": "i64",
                    "height": "i64",
                }
                field_type = event_fields.get(str(expression.value))
                if field_type is None:
                    self.fail(f"window_event has no field {expression.value!r}",
                              expression.token)
                return field_type
            if object_type.startswith("borrow:struct:"):
                struct_name = object_type.split(":", 2)[2]
            elif object_type.startswith("struct:"):
                struct_name = object_type.split(":", 1)[1]
            else:
                self.fail(f"cannot access field on value of type {object_type}", expression.token)
            field_type = self.struct_fields[struct_name].get(str(expression.value))
            if field_type is None:
                self.fail(f"struct {struct_name!r} has no field {expression.value!r}",
                          expression.token)
            return field_type
        if expression.kind == "array":
            if not expression.children:
                self.fail("empty array literal needs an explicit [type] annotation", expression.token)
            element_type = self.infer(expression.children[0])
            if element_type in {"unit", "String"} or self.container_element_type(element_type) is not None:
                self.fail("array elements must have a primitive non-unit type", expression.token)
            for element in expression.children[1:]:
                actual = self.infer(element)
                if actual != element_type:
                    self.fail(f"array elements must have the same type; got {element_type} and {actual}",
                              element.token)
            return f"array:{element_type}:{len(expression.children)}"
        if expression.kind == "index":
            array_type = self.infer(expression.children[0])
            element_type = self.container_element_type(array_type)
            if element_type is None:
                self.fail(f"cannot index value of type {array_type}", expression.token)
            if self.infer(expression.children[1]) != "i64":
                self.fail("array index must have type i64", expression.children[1].token)
            return element_type
        if expression.kind == "name":
            return self.lookup(str(expression.value), expression.token)
        if expression.kind == "method_call":
            receiver, *arguments = expression.children
            receiver_type = self.infer(receiver)
            _, signature, _ = self.resolve_method(
                receiver_type, str(expression.value), expression.token
            )
            expected_parameters = signature.parameters[1:]
            if len(arguments) != len(expected_parameters):
                self.fail(
                    f"method {expression.value!r} expects "
                    f"{len(expected_parameters)} argument(s)",
                    expression.token,
                )
            for argument, (_, expected) in zip(arguments, expected_parameters):
                actual = self.infer(argument)
                if actual != expected:
                    self.fail(
                        f"method {expression.value!r} expects {expected}, got {actual}",
                        argument.token,
                    )
            return signature.return_type
        if expression.kind == "call":
            name = str(expression.value)
            string_type = self.infer_string_call(expression)
            if string_type is not None:
                return string_type
            concurrency_type = self.infer_concurrency_call(expression)
            if concurrency_type is not None:
                return concurrency_type
            if name in self.WINDOW_BUILTINS:
                if self.parallel_worker_depth:
                    self.fail("Win32 window operations are not allowed in parallel loops",
                              expression.token)
                expected_types, result_type = self.WINDOW_BUILTINS[name]
                if len(expression.children) != len(expected_types):
                    self.fail(
                        f"{name} expects exactly {len(expected_types)} argument(s)",
                        expression.token,
                    )
                for index, (argument, expected) in enumerate(
                    zip(expression.children, expected_types)
                ):
                    actual = self.infer(argument)
                    if actual != expected:
                        self.fail(f"{name} expects {expected}, got {actual}", argument.token)
                    if expected == "window":
                        if argument.kind != "name":
                            self.fail(f"{name} expects a window variable", argument.token)
                        if name in {
                            "window_show", "window_hide", "window_close", "window_poll",
                            "window_wait",
                            "window_set_title", "window_move", "window_resize",
                            "window_add_button", "window_add_checkbox", "window_add_label",
                            "window_add_textbox", "window_textbox_text",
                            "window_set_textbox_text", "window_set_control_text",
                            "window_set_control_enabled",
                            "window_set_checkbox_checked",
                            "window_next_event", "window_clear", "window_fill_rect",
                            "window_fill_ellipse", "window_draw_rect", "window_draw_ellipse",
                            "window_draw_line", "window_draw_text",
                            "window_present", "window_present_rect",
                        } and not self.variable_mutability(argument):
                            self.fail(
                                f"{name} requires a mutable window variable; use var",
                                argument.token,
                            )
                    elif name == "window_create" and index in {1, 2}:
                        continue
                return result_type
            builtin = self.BUILTINS.get(name)
            if builtin is not None:
                expected_types, result_type, _ = builtin
                if name == "len":
                    if len(expression.children) != 1:
                        self.fail("len expects exactly one argument", expression.token)
                    argument = expression.children[0]
                    argument_type = self.infer(argument)
                    if argument_type in {"str", "String", "borrow:String"}:
                        return "i64"
                    if (self.container_element_type(argument_type) is not None
                            and argument.kind == "name"):
                        return "i64"
                    if self.array_type_parts(argument_type) is not None:
                        self.fail("len of an array requires an array variable", argument.token)
                    if self.vector_type_parts(argument_type) is not None:
                        self.fail("len of a vector requires a vector variable", argument.token)
                    if argument_type.startswith(("slice:", "slice_mut:")):
                        self.fail("len of a slice requires a slice variable", argument.token)
                    self.fail(f"len does not accept values of type {argument_type}", argument.token)
                if name == "arena":
                    if expression.children:
                        self.fail("arena expects no arguments", expression.token)
                    return "arena"
                if name == "worker_arena":
                    if expression.children:
                        self.fail("worker_arena expects no arguments", expression.token)
                    if not self.parallel_worker_depth:
                        self.fail("worker_arena is only valid inside a parallel for loop",
                                  expression.token)
                    return "arena_ref"
                if name == "slice":
                    if len(expression.children) != 3:
                        self.fail("slice expects a container, start, and end", expression.token)
                    container, start, end = expression.children
                    container_type = self.infer(container)
                    element_type = self.container_element_type(container_type)
                    if element_type is None or container.kind != "name":
                        self.fail("slice expects an array, vector, or slice variable", container.token)
                    for label, bound in (("start", start), ("end", end)):
                        if self.infer(bound) != "i64":
                            self.fail(f"slice {label} must have type i64", bound.token)
                    return f"slice_mut:{element_type}" if self.variable_mutability(container) else f"slice:{element_type}"
                if name.startswith("arena_"):
                    element_type = name[6:]
                    if element_type not in {"i64", "f64", "bool", "str"}:
                        self.fail(f"unknown arena element type in {name!r}", expression.token)
                    if len(expression.children) != 2:
                        self.fail(f"{name} expects an arena and an element count", expression.token)
                    arena_expr, count_expr = expression.children
                    arena_type = self.infer(arena_expr)
                    if arena_type not in {"arena", "arena_ref"}:
                        self.fail(f"{name} expects an arena variable", arena_expr.token)
                    if arena_type == "arena" and not self.variable_mutability(arena_expr):
                        self.fail(f"{name} requires a mutable arena; use var", arena_expr.token)
                    if self.infer(count_expr) != "i64":
                        self.fail(f"{name} count must have type i64", count_expr.token)
                    return f"slice_mut:{element_type}"
                if name in {
                    "push", "pop", "insert", "remove", "reserve",
                    "shrink_to_fit", "capacity", "clear",
                }:
                    if not expression.children:
                        self.fail(f"{name} expects a vector variable", expression.token)
                    vector_type = self.infer(expression.children[0])
                    element_type = self.vector_type_parts(vector_type)
                    if element_type is None or expression.children[0].kind != "name":
                        self.fail(f"{name} expects a vector variable", expression.children[0].token)
                    if name == "capacity":
                        if len(expression.children) != 1:
                            self.fail("capacity expects exactly 1 argument", expression.token)
                        return "i64"
                    if name in {"push", "pop", "insert", "remove", "reserve",
                                "shrink_to_fit", "clear"}:
                        self.require_mutable_vector(expression.children[0], name)
                        self.ensure_no_live_slice(str(expression.children[0].value),
                                                  expression.children[0].token)
                    if name == "push":
                        if len(expression.children) != 2:
                            self.fail("push expects exactly 2 arguments", expression.token)
                        actual = self.infer(expression.children[1])
                        if actual != element_type:
                            self.fail(f"push expects {element_type}, got {actual}",
                                      expression.children[1].token)
                        return "unit"
                    if name == "reserve":
                        if len(expression.children) != 2:
                            self.fail("reserve expects exactly 2 arguments", expression.token)
                        actual = self.infer(expression.children[1])
                        if actual != "i64":
                            self.fail(f"reserve expects i64, got {actual}", expression.children[1].token)
                        return "unit"
                    if name in {"insert", "remove"}:
                        expected_count = 3 if name == "insert" else 2
                        if len(expression.children) != expected_count:
                            self.fail(f"{name} expects exactly {expected_count} arguments",
                                      expression.token)
                        index_type = self.infer(expression.children[1])
                        if index_type != "i64":
                            self.fail(f"{name} index expects i64, got {index_type}",
                                      expression.children[1].token)
                        if name == "insert":
                            actual = self.infer(expression.children[2])
                            if actual != element_type:
                                self.fail(f"insert expects {element_type}, got {actual}",
                                          expression.children[2].token)
                            return "unit"
                        return element_type
                    if name == "shrink_to_fit":
                        if len(expression.children) != 1:
                            self.fail("shrink_to_fit expects exactly 1 argument", expression.token)
                        return "unit"
                    if len(expression.children) != 1:
                        self.fail(f"{name} expects exactly 1 argument", expression.token)
                    return "unit" if name == "clear" else element_type
                if len(expression.children) != len(expected_types):
                    self.fail(f"{name} expects exactly {len(expected_types)} argument(s)", expression.token)
                for argument, expected in zip(expression.children, expected_types):
                    actual = self.infer(argument)
                    if actual != expected:
                        self.fail(f"{name} expects {expected}, got {actual}", argument.token)
                return result_type
            function = self.resolve_call_function(name, expression.children, expression.token)
            if len(expression.children) != len(function.parameters):
                self.fail(f"{name} expects {len(function.parameters)} argument(s)", expression.token)
            mutable_borrows: set[str] = set()
            borrows: set[str] = set()
            moves: set[str] = set()
            for argument, (_, expected) in zip(expression.children, function.parameters):
                actual = self.infer(argument)
                if expected.startswith("dyn:"):
                    if actual != expected:
                        if not actual.startswith("struct:"):
                            self.fail(f"{name} expects {expected}, got {actual}", argument.token)
                        trait_name = expected.split(":", 1)[1]
                        type_name = actual.split(":", 1)[1]
                        self.require_trait_implementation(
                            trait_name, type_name, argument.token
                        )
                    continue
                if expected.startswith(("borrow:", "borrow_mut:")):
                    if expected in {"borrow:String", "borrow_mut:String"}:
                        if argument.kind != "name" or actual != "String":
                            self.fail(f"{name} expects a borrowed String", argument.token)
                        if expected == "borrow_mut:String" and not self.variable_mutability(argument):
                            self.fail("mutable String borrow requires a var binding", argument.token)
                        borrows.add(str(argument.value))
                        continue
                    if expected.startswith("borrow:channel:"):
                        if (argument.kind != "name"
                                or actual not in {
                                    expected.split(":", 1)[1],
                                    f"borrow:{expected.split(':', 1)[1]}",
                                }):
                            self.fail(f"{name} expects a borrow of type {expected}",
                                      argument.token)
                        borrows.add(str(argument.value))
                        continue
                    borrowed_element = self.container_element_type(expected.split(":", 1)[1])
                    actual_element = self.container_element_type(actual)
                    if argument.kind != "name" or actual_element != borrowed_element:
                        self.fail(f"{name} expects a borrow of type {expected}", argument.token)
                    if (expected.startswith("borrow_mut:")
                            and not (actual.startswith("vec:")
                                     or actual.startswith("borrow_mut:vec:"))):
                        self.fail("mutable borrows require a mutable vector or mutable borrow",
                                  argument.token)
                    variable = str(argument.value)
                    if variable in moves:
                        self.fail("cannot move and borrow the same vector in one call", argument.token)
                    if expected.startswith("borrow_mut:"):
                        self.ensure_no_live_slice(variable, argument.token)
                        if not self.variable_mutability(argument):
                            self.fail("mutable borrow requires a mutable vector binding", argument.token)
                        if variable in borrows:
                            self.fail("cannot mix shared and mutable borrows in one call",
                                      argument.token)
                        if variable in mutable_borrows:
                            self.fail("cannot mutably borrow a vector more than once in one call",
                                      argument.token)
                        mutable_borrows.add(variable)
                    else:
                        if variable in mutable_borrows:
                            self.fail("cannot mix shared and mutable borrows in one call", argument.token)
                        borrows.add(variable)
                    continue
                if expected.startswith("move:"):
                    if not actual.startswith("move:") or actual[5:] != expected[5:]:
                        self.fail(f"{name} requires an explicit move of {expected[5:]}", argument.token)
                    variable = str(argument.children[0].value)
                    if variable in moves:
                        self.fail("cannot move a vector more than once in one call", argument.token)
                    if variable in borrows or variable in mutable_borrows:
                        self.fail("cannot move and borrow the same vector in one call", argument.token)
                    self.ensure_no_live_slice(variable, argument.token)
                    moves.add(variable)
                    continue
                if actual.startswith("move:"):
                    self.fail("move expressions can only be passed to owning parameters", argument.token)
                if actual != expected:
                    self.fail(f"{name} expects {expected}, got {actual}", argument.token)
            for index, variable in (
                (index, variable)
                for index, (argument, (_, parameter_type)) in enumerate(
                    zip(expression.children, function.parameters)
                )
                if parameter_type.startswith("move:")
                for variable in (str(argument.children[0].value),)
            ):
                if any(
                    variable in self.expression_names(other)
                    for other_index, other in enumerate(expression.children)
                    if other_index != index
                ):
                    self.fail("a moved vector cannot be used by another argument in the same call",
                              expression.children[index].token)
            return function.return_type
        if expression.kind == "unary":
            actual = self.infer(expression.children[0])
            if expression.value == "-" and actual in {"i64", "f64"}:
                return actual
            if expression.value == "!" and actual == "bool":
                return "bool"
            if expression.value == "move":
                source = expression.children[0]
                if source.kind != "name" or not actual.startswith("vec:"):
                    self.fail("move requires an owned vector variable", expression.token)
                if not self.variable_mutability(source):
                    self.fail("move requires a mutable vector binding", source.token)
                return f"move:{actual}"
            self.fail(f"operator {expression.value!r} does not accept {actual}", expression.token)
        if expression.kind == "binary":
            left, right = (self.infer(child) for child in expression.children)
            operator = str(expression.value)
            if operator in {"&&", "||"}:
                if left == right == "bool":
                    return "bool"
            elif operator in {"==", "!="}:
                if left == right and (
                    left in {"i64", "f64", "bool", "str", "String", "borrow:String"}
                    or left.startswith("enum:")
                ):
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

    def infer_string_call(self, expression: Expr) -> str | None:
        name = str(expression.value)
        if name in {"string_from", "string_clone", "string_concat", "string_length",
                    "string_equal", "string_slice"}:
            expected: dict[str, tuple[tuple[str, ...], str]] = {
                "string_from": (("str",), "String"),
                "string_clone": (("String",), "String"),
                "string_concat": (("String", "String"), "String"),
                "string_length": (("String",), "i64"),
                "string_equal": (("String", "String"), "bool"),
                "string_slice": (("String", "i64", "i64"), "String"),
            }
            types, result = expected[name]
            if len(expression.children) != len(types):
                self.fail(f"{name} expects {len(types)} argument(s)", expression.token)
            for argument, expected_type in zip(expression.children, types):
                actual = self.infer(argument)
                if actual != expected_type and actual != f"borrow:{expected_type}":
                    self.fail(f"{name} expects {expected_type}, got {actual}", argument.token)
            return result
        if name == "string_append":
            if len(expression.children) != 2:
                self.fail("string_append expects a mutable String variable and str text",
                          expression.token)
            target, value = expression.children
            if (target.kind != "name" or self.infer(target) != "String"
                    or not self.variable_mutability(target)):
                self.fail("string_append requires a mutable String variable", target.token)
            if self.infer(value) != "str":
                self.fail("string_append text must have type str", value.token)
            return "unit"
        if name == "string_clear":
            if (len(expression.children) != 1 or expression.children[0].kind != "name"
                    or self.infer(expression.children[0]) != "String"
                    or not self.variable_mutability(expression.children[0])):
                self.fail("string_clear requires a mutable String variable", expression.token)
            return "unit"
        return None

    def infer_concurrency_call(self, expression: Expr) -> str | None:
        name = str(expression.value)
        if name == "task_pool":
            if len(expression.children) != 1 or self.infer(expression.children[0]) != "i64":
                self.fail("task_pool expects one i64 worker count", expression.token)
            return "task_pool"
        if name == "task_pool_wait":
            if (len(expression.children) != 1 or expression.children[0].kind != "name"
                    or self.infer(expression.children[0]) != "task_pool"):
                self.fail("task_pool_wait expects a task pool", expression.token)
            return "unit"
        if name == "task_pool_close":
            if (len(expression.children) != 1 or expression.children[0].kind != "name"
                    or self.infer(expression.children[0]) != "task_pool"):
                self.fail("task_pool_close expects a task pool", expression.token)
            return "unit"
        for suffix, value_type in self.TASK_RESULT_TYPES.items():
            if name == suffix:
                if len(expression.children) not in {3, 4}:
                    self.fail(f"{name} expects a pool, function, value, and optional channel",
                              expression.token)
                pool, function_expr = expression.children[:2]
                if self.infer(pool) != "task_pool" or pool.kind != "name":
                    self.fail(f"{name} expects a task pool variable", pool.token)
                if function_expr.kind != "name":
                    self.fail(f"{name} expects a function name as its second argument",
                              function_expr.token)
                function = self.resolve_function(str(function_expr.value), function_expr.token)
                if len(expression.children) == 3:
                    if (len(function.parameters) != 1
                            or function.parameters[0][1] != value_type
                            or function.return_type != value_type):
                        self.fail(
                            f"{name} function must have signature "
                            f"fn({value_type}) -> {value_type}",
                            function_expr.token,
                        )
                    argument = expression.children[2]
                else:
                    channel_expr, argument = expression.children[2:]
                    expected_channel = f"channel:{value_type}"
                    actual_channel = self.infer(channel_expr)
                    if (channel_expr.kind != "name"
                            or actual_channel not in {
                                expected_channel, f"borrow:{expected_channel}"
                            }):
                        self.fail(f"{name} expects a {expected_channel} variable",
                                  channel_expr.token)
                    if (len(function.parameters) != 2
                            or function.parameters[0][1] != f"borrow:{expected_channel}"
                            or function.parameters[1][1] != value_type
                            or function.return_type != value_type):
                        self.fail(
                            f"{name} function must have signature "
                            f"fn(&channel_{value_type}, {value_type}) -> {value_type}",
                            function_expr.token,
                        )
                if self.infer(argument) != value_type:
                    self.fail(f"{name} expects a {value_type} task argument", argument.token)
                return "i64"
        if name.startswith("task_result_"):
            result_type = name.removeprefix("task_result_")
            if result_type not in {"i64", "f64", "bool"}:
                self.fail(f"unknown task result type in {name!r}", expression.token)
            if (len(expression.children) != 2
                    or self.infer(expression.children[0]) != "task_pool"
                    or self.infer(expression.children[1]) != "i64"):
                self.fail(f"{name} expects a pool and task id", expression.token)
            return result_type
        if name.startswith("channel_") and name in self.CHANNEL_TYPES:
            if len(expression.children) != 1 or self.infer(expression.children[0]) != "i64":
                self.fail(f"{name} expects one i64 capacity", expression.token)
            return f"channel:{self.CHANNEL_TYPES[name]}"
        for operation in ("channel_send_", "channel_receive_", "channel_try_receive_",
                          "channel_close_"):
            if name.startswith(operation):
                element_type = name.removeprefix(operation)
                if element_type not in {"i64", "f64", "bool"}:
                    self.fail(f"unknown channel element type in {name!r}", expression.token)
                if operation == "channel_close_":
                    if len(expression.children) != 1:
                        self.fail(f"{name} expects a channel", expression.token)
                    self.require_channel_argument(expression.children[0], element_type, name)
                    return "unit"
                if len(expression.children) != 2:
                    self.fail(f"{name} expects a channel and a value variable", expression.token)
                channel, value = expression.children
                self.require_channel_argument(channel, element_type, name)
                if operation == "channel_send_":
                    if self.infer(value) != element_type:
                        self.fail(f"{name} expects a {element_type} value", value.token)
                else:
                    if (value.kind != "name" or self.infer(value) != element_type
                            or not self.variable_mutability(value)):
                        self.fail(f"{name} expects a mutable {element_type} output variable",
                                  value.token)
                return "bool" if operation != "channel_close_" else "unit"
        return None

    def require_channel_argument(self, expression: Expr, element_type: str, name: str) -> None:
        expected = f"channel:{element_type}"
        actual = self.infer(expression)
        if (expression.kind != "name"
                or actual not in {expected, f"borrow:{expected}"}):
            self.fail(f"{name} expects a {expected} variable", expression.token)

    def emit_expression(self, expression: Expr) -> str:
        kind = expression.kind
        if kind == "integer":
            return f"{expression.value}LL"
        if kind == "float":
            value = str(expression.value)
            return value if "." in value else value + ".0"
        if kind == "string":
            return json.dumps(expression.value, ensure_ascii=False)
        if kind == "struct":
            struct_name, field_names = expression.value
            fields = ", ".join(
                f".{field_name} = {self.emit_expression(value)}"
                for field_name, value in zip(field_names, expression.children)
            )
            return f"(struct fluid_{struct_name}){{{fields}}}"
        if kind == "array":
            return "{" + ", ".join(self.emit_expression(child) for child in expression.children) + "}"
        if kind == "boolean":
            return "true" if expression.value == "true" else "false"
        if kind == "enum_variant":
            enum_name, variant_name = expression.value
            value = self.enum_map[str(enum_name)].variants.index(str(variant_name))
            return f"{value}LL"
        if kind == "name":
            name = str(expression.value)
            return self.c_variable(name, self.lookup(name, expression.token))
        if kind == "field":
            return f"{self.emit_expression(expression.children[0])}.{expression.value}"
        if kind == "method_call":
            receiver, *arguments = expression.children
            receiver_type = self.infer(receiver)
            _, _, implementation = self.resolve_method(
                receiver_type, str(expression.value), expression.token
            )
            emitted_arguments = ", ".join(
                self.emit_expression(argument) for argument in arguments
            )
            if receiver_type.startswith("dyn:"):
                if receiver.kind != "name":
                    self.fail(
                        "dynamic trait method receivers must be variables",
                        receiver.token,
                    )
                value = self.emit_expression(receiver)
                arguments_text = f"{value}.data"
                if emitted_arguments:
                    arguments_text += f", {emitted_arguments}"
                return (
                    f"{value}.vtable->{expression.value}({arguments_text})"
                )
            if implementation is None:
                self.fail("missing static trait implementation", expression.token)
            value = f"&({self.emit_expression(receiver)})"
            arguments_text = value
            if emitted_arguments:
                arguments_text += f", {emitted_arguments}"
            return f"fluid_{implementation.qualified_name}({arguments_text})"
        if kind == "call":
            name = str(expression.value)
            if name in {"string_from", "string_clone", "string_concat",
                        "string_length", "string_equal", "string_slice",
                        "string_append", "string_clear"}:
                arguments: list[str] = []
                for index, argument in enumerate(expression.children):
                    if name in {"string_clone", "string_concat", "string_length",
                                "string_equal", "string_slice"} and (
                        self.infer(argument) in {"String", "borrow:String"}
                    ):
                        value = self.emit_expression(argument)
                        arguments.append(
                            value if self.infer(argument) == "borrow:String" else f"&{value}"
                        )
                    elif name in {"string_append", "string_clear"} and index == 0:
                        arguments.append(f"&{self.emit_expression(argument)}")
                    else:
                        arguments.append(self.emit_expression(argument))
                if name in {"string_concat", "string_slice"}:
                    arguments.extend((str(expression.token.line), str(expression.token.column)))
                elif name in {"string_from", "string_append"}:
                    arguments.extend((str(expression.token.line), str(expression.token.column)))
                return f"fluid_{name}({', '.join(arguments)})"
            if name == "task_pool":
                return (
                    f"fluid_task_pool_create({self.emit_expression(expression.children[0])}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name in {"task_pool_wait", "task_pool_close"}:
                pool = self.emit_expression(expression.children[0])
                operation = "wait" if name == "task_pool_wait" else "close"
                return (
                    f"fluid_task_pool_{operation}({pool}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name in self.TASK_RESULT_TYPES:
                pool, function_expr = expression.children[:2]
                function = self.resolve_function(str(function_expr.value), function_expr.token)
                self.task_workers[function.qualified_name] = function
                arguments = expression.children[2:]
                if len(arguments) == 1:
                    value = self.emit_expression(arguments[0])
                    channel = "NULL"
                else:
                    channel = self.emit_expression(arguments[0])
                    value = self.emit_expression(arguments[1])
                value_type = self.TASK_RESULT_TYPES[name]
                member = "boolean" if value_type == "bool" else value_type
                return (
                    f"fluid_task_pool_submit({self.emit_expression(pool)}, "
                    f"fluid_task_run_{function.qualified_name}, "
                    f"(fluid_task_value){{.{member} = {value}}}, {channel}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name.startswith("task_result_"):
                result_type = name.removeprefix("task_result_")
                pool, task_id = expression.children
                return (
                    f"fluid_task_pool_result_{result_type}("
                    f"{self.emit_expression(pool)}, {self.emit_expression(task_id)}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name.startswith("channel_") and name in self.CHANNEL_TYPES:
                element_type = self.CHANNEL_TYPES[name]
                return (
                    f"fluid_channel_{element_type}_create("
                    f"{self.emit_expression(expression.children[0])}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            for operation in ("channel_send_", "channel_receive_",
                              "channel_try_receive_", "channel_close_"):
                if name.startswith(operation):
                    element_type = name.removeprefix(operation)
                    channel = self.emit_expression(expression.children[0])
                    if operation == "channel_close_":
                        return f"fluid_channel_{element_type}_close({channel})"
                    value = expression.children[1]
                    if operation == "channel_send_":
                        return (
                            f"fluid_channel_{element_type}_send("
                            f"{channel}, {self.emit_expression(value)}, "
                            f"{expression.token.line}, {expression.token.column})"
                        )
                    function = "receive" if operation == "channel_receive_" else "try_receive"
                    output = self.emit_expression(value)
                    return (
                        f"fluid_channel_{element_type}_{function}("
                        f"{channel}, &{output}, {expression.token.line}, "
                        f"{expression.token.column})"
                    )
            event_kinds = {
                "event_none": "FLUID_EVENT_NONE",
                "event_key_down": "FLUID_EVENT_KEY_DOWN",
                "event_key_up": "FLUID_EVENT_KEY_UP",
                "event_mouse_move": "FLUID_EVENT_MOUSE_MOVE",
                "event_mouse_down": "FLUID_EVENT_MOUSE_DOWN",
                "event_mouse_up": "FLUID_EVENT_MOUSE_UP",
                "event_mouse_wheel": "FLUID_EVENT_MOUSE_WHEEL",
                "event_text": "FLUID_EVENT_TEXT",
                "event_resize": "FLUID_EVENT_RESIZE",
                "event_close": "FLUID_EVENT_CLOSE",
                "event_control": "FLUID_EVENT_CONTROL",
            }
            if name in event_kinds:
                return event_kinds[name]
            if name == "window_create":
                title, width, height = expression.children
                return (
                    f"fluid_window_create({self.emit_expression(title)}, "
                    f"{self.emit_expression(width)}, {self.emit_expression(height)}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            window_operations = {
                "window_show": "fluid_window_show",
                "window_hide": "fluid_window_hide",
                "window_close": "fluid_window_close",
                "window_poll": "fluid_window_poll",
                "window_wait": "fluid_window_wait",
                "window_is_open": "fluid_window_is_open",
                "window_set_title": "fluid_window_set_title",
                "window_move": "fluid_window_move",
                "window_resize": "fluid_window_resize",
                "window_x": "fluid_window_x",
                "window_y": "fluid_window_y",
                "window_width": "fluid_window_width",
                "window_height": "fluid_window_height",
                "window_add_button": "fluid_window_add_button",
                "window_add_checkbox": "fluid_window_add_checkbox",
                "window_add_label": "fluid_window_add_label",
                "window_add_textbox": "fluid_window_add_textbox",
                "window_textbox_text": "fluid_window_textbox_text",
                "window_set_textbox_text": "fluid_window_set_textbox_text",
                "window_set_control_text": "fluid_window_set_control_text",
                "window_set_control_enabled": "fluid_window_set_control_enabled",
                "window_checkbox_checked": "fluid_window_checkbox_checked",
                "window_set_checkbox_checked": "fluid_window_set_checkbox_checked",
                "window_next_event": "fluid_window_next_event",
                "window_clear": "fluid_window_clear",
                "window_fill_rect": "fluid_window_fill_rect",
                "window_fill_ellipse": "fluid_window_fill_ellipse",
                "window_draw_rect": "fluid_window_draw_rect",
                "window_draw_ellipse": "fluid_window_draw_ellipse",
                "window_draw_line": "fluid_window_draw_line",
                "window_draw_text": "fluid_window_draw_text",
                "window_present": "fluid_window_present",
                "window_present_rect": "fluid_window_present_rect",
            }
            if name in window_operations:
                arguments = [
                    f"&{self.emit_expression(expression.children[0])}",
                    *(self.emit_expression(argument) for argument in expression.children[1:]),
                    str(expression.token.line),
                    str(expression.token.column),
                ]
                return f"{window_operations[name]}({', '.join(arguments)})"
            if name == "len":
                argument = expression.children[0]
                argument_type = self.infer(argument)
                if argument_type in {"String", "borrow:String"}:
                    value = self.emit_expression(argument)
                    pointer = value if argument_type == "borrow:String" else f"&{value}"
                    return f"fluid_string_length({pointer})"
                array = self.array_type_parts(argument_type)
                if array is not None:
                    return f"{array[1]}LL"
                if self.vector_type_parts(argument_type) is not None:
                    variable = self.c_variable(str(argument.value), argument_type)
                    return f"((long long){variable}.len)"
                if self.container_element_type(argument_type) is not None:
                    variable = self.c_variable(str(argument.value), argument_type)
                    return f"((long long){variable}.len)"
                return f"((long long)strlen({self.emit_expression(argument)}))"
            if name == "slice":
                container, start, end = expression.children
                container_type = self.infer(container)
                element_type = self.container_element_type(container_type)
                variable = self.c_variable(str(container.value), container_type)
                if self.array_type_parts(container_type) is not None:
                    data = variable
                    length = str(self.array_type_parts(container_type)[1])
                elif self.vector_type_parts(container_type) is not None:
                    data = f"{variable}.data"
                    length = f"{variable}.len"
                else:
                    data = f"{variable}.data"
                    length = f"{variable}.len"
                return (
                    f"fluid_slice_{element_type}_make({data}, {length}, "
                    f"{self.emit_expression(start)}, {self.emit_expression(end)}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name == "arena":
                return "fluid_arena_make()"
            if name == "worker_arena":
                if not self.parallel_worker_depth:
                    self.fail("worker_arena is only valid inside a parallel for loop",
                              expression.token)
                return "fluid_worker_arena"
            if name.startswith("arena_"):
                arena_expr, count_expr = expression.children
                element_type = name[6:]
                arena_type = self.infer(arena_expr)
                arena_name = self.emit_expression(arena_expr)
                arena_pointer = arena_name if arena_type == "arena_ref" else f"&{arena_name}"
                return (
                    f"fluid_arena_{element_type}({arena_pointer}, {self.emit_expression(count_expr)}, "
                    f"{expression.token.line}, {expression.token.column})"
                )
            if name in {
                "push", "pop", "insert", "remove", "reserve",
                "shrink_to_fit", "capacity", "clear",
            }:
                argument = expression.children[0]
                argument_type = self.infer(argument)
                element_type = self.vector_type_parts(argument_type)
                if element_type is None or argument.kind != "name":
                    self.fail(f"{name} expects a vector variable", argument.token)
                variable = self.c_variable(str(argument.value), argument_type)
                pointer = (f"v_{argument.value}" if argument_type.startswith(
                    ("borrow:", "borrow_mut:")
                ) else f"&{variable}")
                helper = f"fluid_vec_{element_type}"
                if name == "capacity":
                    return f"((long long){variable}.capacity)"
                if name == "push":
                    value = self.emit_expression(expression.children[1])
                    return (f"{helper}_push({pointer}, {value}, "
                            f"{expression.token.line}, {expression.token.column})")
                if name == "pop":
                    return (f"{helper}_pop({pointer}, "
                            f"{expression.token.line}, {expression.token.column})")
                if name == "insert":
                    index = self.emit_expression(expression.children[1])
                    value = self.emit_expression(expression.children[2])
                    return (f"{helper}_insert({pointer}, {index}, {value}, "
                            f"{expression.token.line}, {expression.token.column})")
                if name == "remove":
                    index = self.emit_expression(expression.children[1])
                    return (f"{helper}_remove({pointer}, {index}, "
                            f"{expression.token.line}, {expression.token.column})")
                if name == "reserve":
                    value = self.emit_expression(expression.children[1])
                    return (f"{helper}_reserve({pointer}, {value}, "
                            f"{expression.token.line}, {expression.token.column})")
                if name == "shrink_to_fit":
                    return (f"{helper}_shrink_to_fit({pointer}, "
                            f"{expression.token.line}, {expression.token.column})")
                return f"{helper}_clear({pointer})"
            builtin = self.BUILTINS.get(name)
            if builtin is not None:
                _, _, c_name = builtin
                arguments = ", ".join(self.emit_expression(arg) for arg in expression.children)
                return f"{c_name}({arguments})"
            function = self.resolve_call_function(
                str(expression.value), expression.children, expression.token
            )
            emitted_arguments = []
            for argument, (_, parameter_type) in zip(expression.children, function.parameters):
                if parameter_type.startswith("dyn:"):
                    emitted_arguments.append(
                        self.trait_object_value(
                            argument, parameter_type.split(":", 1)[1]
                        )
                    )
                elif parameter_type.startswith("borrow:channel:"):
                    emitted_arguments.append(self.emit_expression(argument))
                elif parameter_type == "borrow:String":
                    argument_type = self.infer(argument)
                    variable = self.c_variable(str(argument.value), argument_type)
                    emitted_arguments.append(
                        variable if argument_type == "borrow:String" else f"&{variable}"
                    )
                elif parameter_type.startswith("borrow:"):
                    argument_type = self.infer(argument)
                    variable = self.c_variable(str(argument.value), argument_type)
                    vector_element = self.vector_type_parts(argument_type)
                    array = self.array_type_parts(argument_type)
                    if vector_element is not None:
                        emitted_arguments.append(
                            f"&((fluid_slice_{vector_element}){{{variable}.len, {variable}.data}})"
                        )
                    elif array is not None:
                        emitted_arguments.append(
                            f"&((fluid_slice_{array[0]}){{{array[1]}, {variable}}})"
                        )
                    else:
                        emitted_arguments.append(f"&{variable}")
                elif parameter_type.startswith("borrow_mut:"):
                    variable = self.c_variable(str(argument.value), self.infer(argument))
                    emitted_arguments.append(f"&{variable}")
                elif parameter_type.startswith("move:"):
                    source = argument.children[0]
                    variable = self.c_variable(str(source.value), self.infer(source))
                    element_type = self.vector_type_parts(parameter_type)
                    emitted_arguments.append(f"fluid_vec_{element_type}_take(&{variable})")
                    self.moved_vectors.add((self.binding_scope(str(source.value), source.token),
                                            str(source.value)))
                else:
                    emitted_arguments.append(self.emit_expression(argument))
            arguments = ", ".join(emitted_arguments)
            return f"fluid_{function.qualified_name}({arguments})"
        if kind == "unary":
            return f"({expression.value}{self.emit_expression(expression.children[0])})"
        if kind == "binary":
            left_type = self.infer(expression.children[0])
            left, right = (self.emit_expression(child) for child in expression.children)
            if left_type in {"String", "borrow:String"} and expression.value in {"==", "!="}:
                left_ptr = left if left_type == "borrow:String" else f"&{left}"
                right_type = self.infer(expression.children[1])
                right_ptr = right if right_type == "borrow:String" else f"&{right}"
                comparison = f"fluid_string_equal({left_ptr}, {right_ptr})"
                return f"({comparison})" if expression.value == "==" else f"(!{comparison})"
            if left_type == "str" and expression.value in {"==", "!="}:
                comparison = f"strcmp({left}, {right})"
                return f"({comparison} {'==' if expression.value == '==' else '!='} 0)"
            return f"({left} {expression.value} {right})"
        if kind == "index":
            array_type = self.infer(expression.children[0])
            array = self.array_type_parts(array_type)
            vector_element = self.vector_type_parts(array_type)
            slice_element = self.container_element_type(array_type) if array_type.startswith(
                ("slice:", "slice_mut:", "borrow:slice:")
            ) else None
            if array is None and vector_element is None and slice_element is None:
                self.fail(f"cannot index value of type {array_type}", expression.token)
            value = self.emit_expression(expression.children[0])
            index = self.emit_expression(expression.children[1])
            length = (str(array[1]) if array is not None
                      else f"{value}.len")
            checked = (
                f"fluid_checked_index((long long)({index}), {length}, "
                f"{expression.token.line}, {expression.token.column})"
            )
            return (f"{value}.data[{checked}]"
                    if vector_element is not None or slice_element is not None else f"{value}[{checked}]")
        raise AssertionError(f"unhandled expression: {kind}")

    def line(self, text: str) -> str:
        return "    " * self.indent + text + "\n"

    def vector_cleanup(self, start_scope: int) -> str:
        result = ""
        has_channels = any(
            type_name.startswith("channel:")
            for scope in self.scopes[start_scope:]
            for type_name, _ in scope.values()
        )
        for index in range(len(self.scopes) - 1, start_scope - 1, -1):
            for name, (type_name, _) in reversed(list(self.scopes[index].items())):
                if type_name.startswith("channel:"):
                    element_type = type_name.split(":", 1)[1]
                    result += self.line(f"fluid_channel_{element_type}_close(v_{name});")
        if has_channels:
            for scope in reversed(self.scopes):
                for name, (type_name, _) in reversed(list(scope.items())):
                    if type_name == "task_pool":
                        result += self.line(f"fluid_task_pool_wait(v_{name}, 0, 0);")
        for index in range(len(self.scopes) - 1, start_scope - 1, -1):
            for name, (type_name, _) in reversed(list(self.scopes[index].items())):
                if type_name == "task_pool":
                    result += self.line(f"fluid_task_pool_destroy(v_{name});")
        for index in range(len(self.scopes) - 1, start_scope - 1, -1):
            scope = self.scopes[index]
            for name, (type_name, _) in reversed(list(scope.items())):
                element_type = self.vector_type_parts(type_name)
                if element_type is not None and self.is_vector_owner(type_name):
                    c_name = self.vector_c_names[index][name]
                    result += self.line(f"fluid_vec_{element_type}_free(&{c_name});")
                elif type_name == "arena":
                    result += self.line(f"fluid_arena_free(&v_{name});")
                elif type_name == "window":
                    result += self.line(f"fluid_window_destroy(&v_{name});")
                elif type_name == "String":
                    result += self.line(f"fluid_string_free(&v_{name});")
                elif type_name.startswith("channel:"):
                    element_type = type_name.split(":", 1)[1]
                    result += self.line(f"fluid_channel_{element_type}_destroy(v_{name});")
        return result

    def emit_block(self, statements: list[Statement]) -> str:
        scope_index = len(self.scopes)
        self.moved_vectors = {
            entry for entry in self.moved_vectors if entry[0] < scope_index
        }
        self.scopes.append({})
        self.vector_c_names.append({})
        self.slice_sources.append({})
        result = self.line("{")
        self.indent += 1
        for statement in statements:
            result += self.emit_statement(statement)
        result += self.vector_cleanup(len(self.scopes) - 1)
        self.indent -= 1
        result += self.line("}")
        self.scopes.pop()
        self.vector_c_names.pop()
        self.slice_sources.pop()
        self.moved_vectors = {
            entry for entry in self.moved_vectors if entry[0] < scope_index
        }
        return result

    def emit_control_flow(self, statement: Statement) -> str:
        if self.loop_depth == 0:
            self.fail(f"{statement.kind} is only valid inside a loop", statement.token)
        cleanup = self.vector_cleanup(self.loop_scope_depths[-1])
        return cleanup + self.line(f"{statement.kind};")

    def match_info(self, statement: Statement) -> list[tuple[int, list[Statement]]]:
        actual = self.infer(statement.expression)
        if not actual.startswith("enum:"):
            self.fail(f"match expects an enum value, got {actual}", statement.token)
        enum_name = actual.split(":", 1)[1]
        definition = self.enum_map[enum_name]
        seen: set[str] = set()
        resolved: list[tuple[int, list[Statement]]] = []
        for arm in statement.value:
            if arm.enum_name != enum_name:
                self.fail(f"match arm must use {enum_name} variants", arm.token)
            if arm.variant_name not in definition.variants:
                self.fail(
                    f"enum {enum_name!r} has no variant {arm.variant_name!r}",
                    arm.token,
                )
            if arm.variant_name in seen:
                self.fail(f"duplicate match arm for {enum_name}::{arm.variant_name}",
                          arm.token)
            seen.add(arm.variant_name)
            resolved.append((definition.variants.index(arm.variant_name), arm.body))
        missing = [variant for variant in definition.variants if variant not in seen]
        if missing:
            self.fail(
                f"non-exhaustive match for {enum_name}; missing "
                + ", ".join(f"{enum_name}::{variant}" for variant in missing),
                statement.token,
            )
        self.enum_match_checked.add(id(statement))
        return resolved

    def always_returns(self, statements: list[Statement]) -> bool:
        for statement in statements:
            if statement.kind == "return":
                return True
            if (statement.kind == "if" and statement.alternative is not None
                    and self.always_returns(statement.body)
                    and self.always_returns(statement.alternative)):
                return True
            if (statement.kind == "match" and id(statement) in self.enum_match_checked
                    and all(self.always_returns(arm.body) for arm in statement.value)):
                return True
        return False

    def always_terminates(self, statements: list[Statement]) -> bool:
        for statement in statements:
            if statement.kind in {"return", "break", "continue"}:
                return True
            if (statement.kind == "if" and statement.alternative is not None
                    and self.always_terminates(statement.body)
                    and self.always_terminates(statement.alternative)):
                return True
            if (statement.kind == "match" and id(statement) in self.enum_match_checked
                    and all(self.always_terminates(arm.body) for arm in statement.value)):
                return True
        return False

    def validate_reachable(self, statements: list[Statement]) -> None:
        terminated = False
        for statement in statements:
            if terminated:
                self.fail("unreachable statement", statement.token)
            if statement.kind == "if":
                self.validate_reachable(statement.body)
                if statement.alternative is not None:
                    self.validate_reachable(statement.alternative)
                    terminated = (
                        self.always_terminates(statement.body)
                        and self.always_terminates(statement.alternative)
                    )
            elif statement.kind == "match":
                for arm in statement.value:
                    self.validate_reachable(arm.body)
                terminated = (
                    id(statement) in self.enum_match_checked
                    and all(self.always_terminates(arm.body) for arm in statement.value)
                )
            elif statement.kind in {"while", "for", "parallel_for"}:
                self.validate_reachable(statement.body)
            elif statement.kind in {"return", "break", "continue"}:
                terminated = True

    def parallel_body_info(
        self, statement: Statement, loop_name: str
    ) -> tuple[dict[str, tuple[str, bool]], set[str]]:
        outer: dict[str, tuple[str, bool]] = {}
        for scope in self.scopes:
            outer.update(scope)
        local_names: set[str] = {loop_name}
        references: set[str] = set()
        outputs: set[str] = set()

        def expression_refs(expression: Expr | None) -> None:
            if expression is not None:
                references.update(self.expression_names(expression))

        def inspect(statements: list[Statement]) -> None:
            for child in statements:
                if child.kind in {"return", "break", "continue", "parallel_for"}:
                    self.fail(
                        f"{child.kind} is not supported inside a parallel for body",
                        child.token,
                    )
                if child.kind in {"let", "var"}:
                    name, _ = child.value
                    if name in outer or name in local_names:
                        self.fail(
                            f"parallel for local {name!r} cannot shadow an existing variable",
                            child.token,
                        )
                    local_names.add(name)
                    expression_refs(child.expression)
                elif child.kind == "for":
                    nested_name, nested_start, nested_end = child.value
                    if nested_name in outer or nested_name in local_names:
                        self.fail(
                            f"parallel for local {nested_name!r} cannot shadow an existing variable",
                            child.token,
                        )
                    local_names.add(nested_name)
                    expression_refs(nested_start)
                    expression_refs(nested_end)
                elif child.kind == "assign":
                    target = child.value
                    if (target.kind == "index" and target.children[0].kind == "name"
                            and str(target.children[0].value) in outer):
                        output = str(target.children[0].value)
                        index = target.children[1]
                        if index.kind != "name" or str(index.value) != loop_name:
                            self.fail(
                                "parallel writes must use the loop index directly",
                                target.token,
                            )
                        output_type, mutable = outer[output]
                        if (not mutable or not (
                            self.array_type_parts(output_type)
                            or self.vector_type_parts(output_type)
                        )):
                            self.fail(
                                "parallel writes require a mutable array or vector",
                                target.token,
                            )
                        outputs.add(output)
                        expression_refs(index)
                    else:
                        expression_refs(target)
                    expression_refs(child.expression)
                else:
                    expression_refs(child.expression)

                if child.body is not None:
                    inspect(child.body)
                if child.alternative is not None:
                    inspect(child.alternative)
                if child.kind == "match":
                    for arm in child.value:
                        inspect(arm.body)

        inspect(statement.body)
        if loop_name in outer:
            self.fail(f"parallel loop variable {loop_name!r} shadows an existing variable",
                      statement.token)
        if references.intersection(outputs):
            name = sorted(references.intersection(outputs))[0]
            self.fail(
                f"parallel output {name!r} cannot be read in the parallel body",
                statement.token,
            )
        captures = references.intersection(outer)
        for name in captures:
            type_name, mutable = outer[name]
            if name in outputs:
                continue
            if mutable:
                self.fail(
                    f"parallel for cannot capture mutable variable {name!r}; "
                    "use a disjoint indexed output",
                    statement.token,
                )
            if type_name.startswith(("arena", "borrow_mut:")):
                self.fail(f"parallel for cannot capture {type_name} variable {name!r}",
                          statement.token)
        for name in outputs:
            if any(name in source.values() for source in self.slice_sources):
                self.fail(
                    f"parallel output {name!r} has a live slice and cannot be written safely",
                    statement.token,
                )
        return {name: outer[name] for name in captures | outputs}, outputs

    def emit_parallel_for(self, statement: Statement) -> str:
        name, start, end = statement.value
        for label, expression in (("start", start), ("end", end)):
            if self.infer(expression) != "i64":
                self.fail(f"parallel for loop {label} must have type i64", expression.token)
        captures, outputs = self.parallel_body_info(statement, name)
        capture_names = sorted(captures)
        worker_id = self.parallel_counter
        self.parallel_counter += 1
        context_name = f"fluid_parallel_context_{worker_id}"
        body_name = f"fluid_parallel_body_{worker_id}"
        context_fields: list[tuple[str, str]] = []
        context_values: list[str] = []
        aliases: dict[tuple[int, str], str] = {}
        capture_scope: dict[str, tuple[str, bool]] = {}
        capture_vectors: dict[str, str] = {}

        for capture in capture_names:
            type_name, _ = captures[capture]
            storage_type = type_name
            if type_name.startswith("borrow:"):
                borrowed_type = type_name.split(":", 1)[1]
                if borrowed_type.startswith("vec:"):
                    storage_type = f"slice:{borrowed_type.split(':', 1)[1]}"
                else:
                    storage_type = borrowed_type
            array = self.array_type_parts(storage_type)
            vector = self.vector_type_parts(storage_type)
            if capture in outputs and vector is not None:
                field_name = f"out_{capture}"
                field = f"fluid_vec_{vector} *{field_name}"
                value = f"&{self.c_variable(capture, type_name)}"
                alias = f"(*fluid_context->{field_name})"
                mutable = True
            elif array is not None:
                element, length = array
                field_name = f"capture_{capture}"
                field = f"{self.c_type(element)} (*{field_name})[{length}]"
                value = f"&{self.c_variable(capture, type_name)}"
                alias = f"(*fluid_context->{field_name})"
                mutable = capture in outputs
            else:
                field_name = f"capture_{capture}"
                field = f"{self.c_type(storage_type)} {field_name}"
                value = self.c_variable(capture, type_name)
                alias = f"fluid_context->{field_name}"
                mutable = capture in outputs
            context_fields.append((field, field_name))
            context_values.append(f".{field_name} = {value}")
            aliases[(0, capture)] = alias
            capture_scope[capture] = (type_name, mutable)
            if vector is not None and capture not in outputs:
                capture_vectors[capture] = alias

        context_definition = (
            f"typedef struct {{ {' '.join(field + ';' for field, _ in context_fields)} }}"
            f" {context_name};"
        )
        context_initialization = (
            f"{context_name} fluid_parallel_values_{worker_id} = "
            f"{{{', '.join(context_values)}}};"
        )

        saved_scopes = self.scopes
        saved_vector_names = self.vector_c_names
        saved_slice_sources = self.slice_sources
        saved_overrides = self.c_name_overrides
        saved_return_type = self.return_type
        saved_indent = self.indent
        self.scopes = [capture_scope, {name: ("i64", False)}]
        self.vector_c_names = [capture_vectors, {}]
        self.slice_sources = [{}, {}]
        self.c_name_overrides = aliases
        self.return_type = "unit"
        self.indent = 1
        self.parallel_worker_depth += 1
        try:
            body = self.emit_block(statement.body)
        finally:
            self.parallel_worker_depth -= 1
            self.scopes = saved_scopes
            self.vector_c_names = saved_vector_names
            self.slice_sources = saved_slice_sources
            self.c_name_overrides = saved_overrides
            self.return_type = saved_return_type
            self.indent = saved_indent

        self.parallel_workers.append(
            "\n".join([
                context_definition,
                f"static void {body_name}(long long v_{name}, void *raw, "
                "fluid_arena *fluid_worker_arena)",
                "{",
                f"    {context_name} *fluid_context = ({context_name} *)raw;",
                body.rstrip("\n"),
                "}",
            ])
        )

        range_id = self.range_counter
        self.range_counter += 1
        start_name = f"fluid_range_start_{range_id}"
        end_name = f"fluid_range_end_{range_id}"
        result = self.line("{")
        self.indent += 1
        result += self.line(f"long long {start_name} = {self.emit_expression(start)};")
        result += self.line(f"long long {end_name} = {self.emit_expression(end)};")
        result += self.line(context_initialization)
        result += self.line(
            f"fluid_parallel_for({start_name}, {end_name}, {body_name}, "
            f"&fluid_parallel_values_{worker_id}, {statement.token.line}, "
            f"{statement.token.column});"
        )
        self.indent -= 1
        return result + self.line("}")

    def emit_statement(self, statement: Statement) -> str:
        if statement.kind in {"let", "var"}:
            name, annotation = statement.value
            if statement.expression.kind == "array" and not statement.expression.children:
                if annotation is None or self.vector_type_parts(annotation) is None:
                    self.fail("empty array literal needs an explicit [type] annotation", statement.token)
                actual = annotation
            else:
                actual = self.infer(statement.expression)
            declared = annotation or actual
            if declared.startswith("dyn:") and actual.startswith("struct:"):
                trait_name = declared.split(":", 1)[1]
                type_name = actual.split(":", 1)[1]
                self.require_trait_implementation(
                    trait_name, type_name, statement.expression.token
                )
                actual = declared
            fixed_array = self.array_type_parts(actual)
            declared_vector = self.vector_type_parts(declared)
            if declared_vector is not None and fixed_array is not None:
                if fixed_array[0] != declared_vector:
                    self.fail(f"cannot assign {actual} to {declared}", statement.token)
                actual = declared
            elif declared_vector is not None and actual == declared:
                if statement.expression.kind != "array":
                    self.fail("vectors cannot be copied; initialize from an array literal", statement.token)
            if declared != actual:
                self.fail(f"cannot assign {actual} to {declared}", statement.token)
            if declared == "unit":
                self.fail("a variable cannot have type unit", statement.token)
            if declared == "arena" and statement.expression.kind != "call":
                self.fail("arenas cannot be copied", statement.token)
            if declared == "window" and not (
                statement.expression.kind == "call"
                and statement.expression.value == "window_create"
            ):
                self.fail("windows cannot be copied; initialize with window_create", statement.token)
            if declared == "task_pool" and not (
                statement.expression.kind == "call"
                and statement.expression.value == "task_pool"
            ):
                self.fail("task pools cannot be copied; initialize with task_pool", statement.token)
            if declared == "String" and not (
                statement.expression.kind == "call"
                and statement.expression.value in {
                    "string_from", "string_clone", "string_concat", "string_slice"
                }
            ):
                self.fail("String values cannot be copied; initialize with string_from or string_clone",
                          statement.token)
            if declared.startswith("channel:") and not (
                statement.expression.kind == "call"
                and statement.expression.value == f"channel_{declared.split(':', 1)[1]}"
            ):
                self.fail("channels cannot be copied; initialize with channel_<type>",
                          statement.token)
            if declared.startswith("slice:") and statement.kind == "var":
                self.fail("slices are immutable views; bind them with let", statement.token)
            scope = self.scopes[-1]
            if name in scope:
                self.fail(f"variable {name!r} is already defined in this scope", statement.token)
            scope[name] = (declared, statement.kind == "var")
            value = (
                self.trait_object_value(
                    statement.expression, declared.split(":", 1)[1]
                )
                if declared.startswith("dyn:") and
                self.infer(statement.expression) != declared
                else self.emit_expression(statement.expression)
            )
            array = self.array_type_parts(declared)
            if array is not None:
                return self.line(f"{self.c_type(declared)} v_{name}[{array[1]}] = {value};")
            vector_element = self.vector_type_parts(declared)
            if vector_element is not None:
                c_name = f"fluid_vector_{self.vector_counter}"
                self.vector_counter += 1
                self.vector_c_names[-1][name] = c_name
                c_element = self.c_type(vector_element)
                count = len(statement.expression.children)
                result = ""
                if count:
                    result += self.line(f"{c_element} fluid_items_{name}[{count}] = {value};")
                items = f"fluid_items_{name}" if count else "NULL"
                return result + self.line(
                    f"{self.c_type(declared)} {c_name} = fluid_vec_{vector_element}_make("
                    f"{items}, {count}, {statement.token.line}, {statement.token.column});"
                )
            if declared.startswith(("slice:", "slice_mut:")):
                source: str | None = None
                if statement.expression.kind == "call":
                    if statement.expression.value == "slice":
                        source = str(statement.expression.children[0].value)
                    elif str(statement.expression.value).startswith("arena_"):
                        source = None
                elif statement.expression.kind == "name":
                    source = self.lookup_slice_source(str(statement.expression.value))
                self.slice_sources[-1][name] = source
            return self.line(f"{self.c_type(declared)} v_{name} = {value};")
        if statement.kind == "assign":
            target = statement.value
            if target.kind == "name":
                name = str(target.value)
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
                target_type = declared
            else:
                base = target.children[0]
                if base.kind != "name":
                    self.fail("array assignment requires a mutable array variable", statement.token)
                name = str(base.value)
                declared = None
                mutable = False
                for scope in reversed(self.scopes):
                    if name in scope:
                        declared, mutable = scope[name]
                        break
                if declared is None:
                    self.fail(f"unknown variable {name!r}", statement.token)
                if not mutable:
                    self.fail(f"cannot assign to an element of immutable let binding {name!r}", statement.token)
                target_type = self.infer(target)
            actual = self.infer(statement.expression)
            if target_type.startswith("array:"):
                self.fail("whole-array assignment is not supported", statement.token)
            if target_type.startswith("vec:"):
                self.fail("vectors cannot be replaced; mutate the vector with its built-ins",
                          statement.token)
            if target_type.startswith(("slice:", "slice_mut:")):
                self.fail("slice bindings cannot be reassigned", statement.token)
            if target_type == "arena":
                self.fail("arenas cannot be replaced", statement.token)
            if target_type == "window":
                self.fail("windows cannot be replaced", statement.token)
            if target_type in {"task_pool"} or target_type.startswith("channel:"):
                self.fail(f"{target_type} handles cannot be replaced", statement.token)
            if target_type == "String":
                self.fail("String values cannot be replaced; mutate with string_append or string_clear",
                          statement.token)
            if actual != target_type:
                self.fail(f"cannot assign {actual} to {target_type}", statement.token)
            return self.line(
                f"{self.emit_expression(target)} = {self.emit_expression(statement.expression)};"
            )
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
            if value_type in {"String", "borrow:String"}:
                return self.line(f'printf("%s\\n", {value}.data);')
            self.fail("print cannot display a unit value", statement.token)
        if statement.kind == "assert":
            if self.infer(statement.expression) != "bool":
                self.fail("assert condition must have type bool", statement.token)
            expression = self.emit_expression(statement.expression)
            return self.line(f"fluid_assert({expression}, {statement.token.line}, {statement.token.column});")
        if statement.kind == "return":
            if statement.expression is None:
                if self.return_type != "unit":
                    self.fail(f"expected a {self.return_type} return value", statement.token)
                return self.vector_cleanup(0) + self.line("return;")
            actual = self.infer(statement.expression)
            if actual != self.return_type:
                self.fail(f"expected {self.return_type} return value, got {actual}", statement.token)
            cleanup = self.vector_cleanup(0)
            if cleanup:
                temporary = f"fluid_return_value_{self.return_counter}"
                self.return_counter += 1
                result = self.line(
                    f"{self.c_type(self.return_type)} {temporary} = "
                    f"{self.emit_expression(statement.expression)};"
                )
                result += cleanup
                return result + self.line(f"return {temporary};")
            return self.line(f"return {self.emit_expression(statement.expression)};")
        if statement.kind == "expression":
            expression_type = self.infer(statement.expression)
            if expression_type != "unit":
                self.fail("only unit-returning calls can be used as statements", statement.token)
            return self.line(f"{self.emit_expression(statement.expression)};")
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
            self.vector_c_names.append({})
            self.slice_sources.append({})
            before_loop = set(self.moved_vectors)
            self.loop_depth += 1
            self.loop_scope_depths.append(len(self.scopes))
            try:
                loop = self.line(
                    f"for (long long v_{name} = {start_name}; "
                    f"v_{name} < {end_name}; ++v_{name})"
                )
                result += loop + self.emit_block(statement.body)
            finally:
                self.loop_scope_depths.pop()
                self.loop_depth -= 1
                self.scopes.pop()
                self.vector_c_names.pop()
                self.slice_sources.pop()
                self.moved_vectors |= before_loop
                self.indent -= 1
            return result + self.line("}")
        if statement.kind == "parallel_for":
            return self.emit_parallel_for(statement)
        if statement.kind == "match":
            arms = self.match_info(statement)
            match_id = self.match_counter
            self.match_counter += 1
            value_name = f"fluid_match_value_{match_id}"
            enum_name = self.infer(statement.expression).split(":", 1)[1]
            result = self.line(
                f"long long {value_name} = (long long)({self.emit_expression(statement.expression)});"
            )
            previous_moves = set(self.moved_vectors)
            branch_moves: list[set[tuple[int, str]]] = []
            for index, (variant_value, body) in enumerate(arms):
                keyword = "if" if index == 0 else "else if"
                result += self.line(
                    f"{keyword} ({value_name} == {variant_value}LL)"
                )
                self.moved_vectors = set(previous_moves)
                result += self.emit_block(body)
                branch_moves.append(set(self.moved_vectors))
            result += self.line("else {")
            self.indent += 1
            result += self.line(
                f'fluid_memory_error("invalid {enum_name} enum value", '
                f"{statement.token.line}, {statement.token.column});"
            )
            result += self.line("exit(1);")
            self.indent -= 1
            result += self.line("}")
            self.moved_vectors = set.union(previous_moves, *branch_moves)
            return result
        if statement.kind in {"if", "while"}:
            if self.infer(statement.expression) != "bool":
                self.fail(f"{statement.kind} condition must have type bool", statement.token)
            condition = self.emit_expression(statement.expression)
            if statement.expression.kind == "binary":
                condition = condition[1:-1]
            if statement.kind == "while":
                before_loop = set(self.moved_vectors)
                self.loop_depth += 1
                self.loop_scope_depths.append(len(self.scopes))
                try:
                    body = self.emit_block(statement.body)
                finally:
                    self.loop_scope_depths.pop()
                    self.loop_depth -= 1
                self.moved_vectors |= before_loop
                return self.line(f"while ({condition})") + body
            before_if = set(self.moved_vectors)
            result = self.line(f"if ({condition})") + self.emit_block(statement.body)
            after_body = set(self.moved_vectors)
            if statement.alternative is not None:
                self.moved_vectors = before_if
                result += self.line("else") + self.emit_block(statement.alternative)
                after_alternative = set(self.moved_vectors)
                self.moved_vectors = after_body | after_alternative
            else:
                self.moved_vectors = after_body | before_if
            return result
        raise AssertionError(f"unhandled statement: {statement.kind}")

    @staticmethod
    def contains_parallel_for(statements: list[Statement]) -> bool:
        return any(
            statement.kind == "parallel_for"
            or (statement.body is not None
                and Compiler.contains_parallel_for(statement.body))
            or (statement.alternative is not None
                and Compiler.contains_parallel_for(statement.alternative))
            or (statement.kind == "match"
                and any(Compiler.contains_parallel_for(arm.body) for arm in statement.value))
            for statement in statements
        )

    @staticmethod
    def contains_window_call(expression: Expr | None) -> bool:
        return bool(
            expression is not None
            and (
                (expression.kind == "call" and str(expression.value).startswith("window_"))
                or any(Compiler.contains_window_call(child) for child in expression.children)
            )
        )

    @staticmethod
    def contains_window_statement(statements: list[Statement]) -> bool:
        return any(
            Compiler.contains_window_call(statement.expression)
            or (statement.body is not None
                and Compiler.contains_window_statement(statement.body))
            or (statement.alternative is not None
                and Compiler.contains_window_statement(statement.alternative))
            or (statement.kind == "match"
                and any(Compiler.contains_window_statement(arm.body) for arm in statement.value))
            for statement in statements
        )

    @staticmethod
    def contains_concurrency_call(expression: Expr | None) -> bool:
        return bool(
            expression is not None
            and (
                (expression.kind == "call"
                 and str(expression.value).startswith(("task_", "channel_")))
                or any(Compiler.contains_concurrency_call(child) for child in expression.children)
            )
        )

    @staticmethod
    def contains_concurrency_statement(statements: list[Statement]) -> bool:
        return any(
            Compiler.contains_concurrency_call(statement.expression)
            or (statement.body is not None
                and Compiler.contains_concurrency_statement(statement.body))
            or (statement.alternative is not None
                and Compiler.contains_concurrency_statement(statement.alternative))
            or (statement.kind == "match"
                and any(Compiler.contains_concurrency_statement(arm.body)
                        for arm in statement.value))
            for statement in statements
        )

    def emit_task_worker(self, function: Function) -> list[str]:
        value_type = function.return_type
        member = "boolean" if value_type == "bool" else value_type
        arguments = [f"job->argument.{member}"]
        if len(function.parameters) == 2:
            channel_type = function.parameters[0][1].split(":", 2)[2]
            arguments.insert(0, f"(fluid_channel_{channel_type} *)job->channel")
        call = f"fluid_{function.qualified_name}({', '.join(arguments)})"
        return [
            f"static void fluid_task_run_{function.qualified_name}(fluid_task_job *job)",
            "{",
            f"    job->result.{member} = {call};",
            "}",
        ]

    @staticmethod
    def concurrency_runtime() -> list[str]:
        runtime = [
            "#if defined(_WIN32)",
            "typedef CRITICAL_SECTION fluid_mutex;",
            "typedef CONDITION_VARIABLE fluid_condition;",
            "typedef HANDLE fluid_thread;",
            "static void fluid_mutex_init(fluid_mutex *mutex) { InitializeCriticalSection(mutex); }",
            "static void fluid_mutex_destroy(fluid_mutex *mutex) { DeleteCriticalSection(mutex); }",
            "static void fluid_mutex_lock(fluid_mutex *mutex) { EnterCriticalSection(mutex); }",
            "static void fluid_mutex_unlock(fluid_mutex *mutex) { LeaveCriticalSection(mutex); }",
            "static void fluid_condition_init(fluid_condition *condition) { InitializeConditionVariable(condition); }",
            "static void fluid_condition_destroy(fluid_condition *condition) { (void)condition; }",
            "static void fluid_condition_wait(fluid_condition *condition, fluid_mutex *mutex)",
            "{",
            "    if (!SleepConditionVariableCS(condition, mutex, INFINITE))",
            '        fluid_memory_error("condition wait failed", 0, 0);',
            "}",
            "static void fluid_condition_signal(fluid_condition *condition) { WakeConditionVariable(condition); }",
            "static void fluid_condition_broadcast(fluid_condition *condition) { WakeAllConditionVariable(condition); }",
            "static fluid_thread fluid_thread_start(LPTHREAD_START_ROUTINE entry, void *data)",
            "{",
            "    HANDLE thread = CreateThread(NULL, 0, entry, data, 0, NULL);",
            "    if (!thread) fluid_memory_error(\"worker creation failed\", 0, 0);",
            "    return thread;",
            "}",
            "static void fluid_thread_join(fluid_thread thread)",
            "{",
            "    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0)",
            '        fluid_memory_error("worker join failed", 0, 0);',
            "    if (!CloseHandle(thread)) fluid_memory_error(\"worker handle release failed\", 0, 0);",
            "}",
            "#else",
            "typedef pthread_mutex_t fluid_mutex;",
            "typedef pthread_cond_t fluid_condition;",
            "typedef pthread_t fluid_thread;",
            "static void fluid_mutex_init(fluid_mutex *mutex)",
            "{ if (pthread_mutex_init(mutex, NULL) != 0) fluid_memory_error(\"mutex initialization failed\", 0, 0); }",
            "static void fluid_mutex_destroy(fluid_mutex *mutex)",
            "{ if (pthread_mutex_destroy(mutex) != 0) fluid_memory_error(\"mutex destruction failed\", 0, 0); }",
            "static void fluid_mutex_lock(fluid_mutex *mutex)",
            "{ if (pthread_mutex_lock(mutex) != 0) fluid_memory_error(\"mutex lock failed\", 0, 0); }",
            "static void fluid_mutex_unlock(fluid_mutex *mutex)",
            "{ if (pthread_mutex_unlock(mutex) != 0) fluid_memory_error(\"mutex unlock failed\", 0, 0); }",
            "static void fluid_condition_init(fluid_condition *condition)",
            "{ if (pthread_cond_init(condition, NULL) != 0) fluid_memory_error(\"condition initialization failed\", 0, 0); }",
            "static void fluid_condition_destroy(fluid_condition *condition)",
            "{ if (pthread_cond_destroy(condition) != 0) fluid_memory_error(\"condition destruction failed\", 0, 0); }",
            "static void fluid_condition_wait(fluid_condition *condition, fluid_mutex *mutex)",
            "{ if (pthread_cond_wait(condition, mutex) != 0) fluid_memory_error(\"condition wait failed\", 0, 0); }",
            "static void fluid_condition_signal(fluid_condition *condition)",
            "{ if (pthread_cond_signal(condition) != 0) fluid_memory_error(\"condition signal failed\", 0, 0); }",
            "static void fluid_condition_broadcast(fluid_condition *condition)",
            "{ if (pthread_cond_broadcast(condition) != 0) fluid_memory_error(\"condition broadcast failed\", 0, 0); }",
            "static fluid_thread fluid_thread_start(void *(*entry)(void *), void *data)",
            "{",
            "    fluid_thread thread;",
            "    if (pthread_create(&thread, NULL, entry, data) != 0)",
            '        fluid_memory_error("worker creation failed", 0, 0);',
            "    return thread;",
            "}",
            "static void fluid_thread_join(fluid_thread thread)",
            "{ if (pthread_join(thread, NULL) != 0) fluid_memory_error(\"worker join failed\", 0, 0); }",
            "#endif",
            "typedef union { long long i64; double f64; bool boolean; } fluid_task_value;",
            "typedef struct fluid_task_job fluid_task_job;",
            "struct fluid_task_job {",
            "    void (*run)(fluid_task_job *);",
            "    fluid_task_value argument;",
            "    fluid_task_value result;",
            "    void *channel;",
            "    bool done;",
            "    fluid_task_job *next;",
            "};",
            "typedef struct fluid_task_pool {",
            "    fluid_mutex mutex;",
            "    fluid_condition changed;",
            "    fluid_thread *threads;",
            "    size_t thread_count;",
            "    fluid_task_job **jobs;",
            "    size_t job_count;",
            "    size_t job_capacity;",
            "    size_t completed;",
            "    fluid_task_job *first;",
            "    fluid_task_job *last;",
            "    bool closed;",
            "    bool stopping;",
            "} fluid_task_pool;",
            "static void fluid_task_pool_finish(fluid_task_pool *pool)",
            "{",
            "    fluid_mutex_lock(&pool->mutex);",
            "    while (pool->completed < pool->job_count)",
            "        fluid_condition_wait(&pool->changed, &pool->mutex);",
            "    fluid_mutex_unlock(&pool->mutex);",
            "}",
            "#if defined(_WIN32)",
            "static DWORD WINAPI fluid_task_worker(LPVOID raw)",
            "{",
            "    fluid_task_pool *pool = (fluid_task_pool *)raw;",
            "#else",
            "static void *fluid_task_worker(void *raw)",
            "{",
            "    fluid_task_pool *pool = (fluid_task_pool *)raw;",
            "#endif",
            "    for (;;) {",
            "        fluid_mutex_lock(&pool->mutex);",
            "        while (!pool->first && !pool->stopping)",
            "            fluid_condition_wait(&pool->changed, &pool->mutex);",
            "        if (pool->stopping && !pool->first) {",
            "            fluid_mutex_unlock(&pool->mutex);",
            "#if defined(_WIN32)",
            "            return 0;",
            "#else",
            "            return NULL;",
            "#endif",
            "        }",
            "        fluid_task_job *job = pool->first;",
            "        pool->first = job->next;",
            "        if (!pool->first) pool->last = NULL;",
            "        fluid_mutex_unlock(&pool->mutex);",
            "        job->run(job);",
            "        fluid_mutex_lock(&pool->mutex);",
            "        job->done = true;",
            "        ++pool->completed;",
            "        fluid_condition_broadcast(&pool->changed);",
            "        fluid_mutex_unlock(&pool->mutex);",
            "    }",
            "}",
            "static fluid_task_pool *fluid_task_pool_create(long long worker_count, int line, int column)",
            "{",
            "    if (worker_count < 1 || worker_count > 64)",
            '        fluid_memory_error("task pool worker count must be between 1 and 64", line, column);',
            "    fluid_task_pool *pool = (fluid_task_pool *)calloc(1, sizeof(*pool));",
            "    if (!pool) fluid_memory_error(\"out of memory\", line, column);",
            "    pool->thread_count = (size_t)worker_count;",
            "    pool->threads = (fluid_thread *)calloc(pool->thread_count, sizeof(*pool->threads));",
            "    if (!pool->threads) fluid_memory_error(\"out of memory\", line, column);",
            "    fluid_mutex_init(&pool->mutex);",
            "    fluid_condition_init(&pool->changed);",
            "    for (size_t i = 0; i < pool->thread_count; ++i)",
            "        pool->threads[i] = fluid_thread_start(fluid_task_worker, pool);",
            "    return pool;",
            "}",
            "static void fluid_task_pool_wait(fluid_task_pool *pool, int line, int column)",
            "{",
            "    if (!pool) fluid_memory_error(\"invalid task pool\", line, column);",
            "    fluid_task_pool_finish(pool);",
            "}",
            "static void fluid_task_pool_close(fluid_task_pool *pool, int line, int column)",
            "{",
            "    if (!pool) fluid_memory_error(\"invalid task pool\", line, column);",
            "    fluid_mutex_lock(&pool->mutex);",
            "    pool->closed = true;",
            "    fluid_condition_broadcast(&pool->changed);",
            "    fluid_mutex_unlock(&pool->mutex);",
            "    fluid_task_pool_finish(pool);",
            "}",
            "static long long fluid_task_pool_submit(fluid_task_pool *pool, void (*run)(fluid_task_job *),",
            "                                        fluid_task_value argument, void *channel,",
            "                                        int line, int column)",
            "{",
            "    if (!pool || !run) fluid_memory_error(\"invalid task submission\", line, column);",
            "    fluid_task_job *job = (fluid_task_job *)calloc(1, sizeof(*job));",
            "    if (!job) fluid_memory_error(\"out of memory\", line, column);",
            "    job->run = run;",
            "    job->argument = argument;",
            "    job->channel = channel;",
            "    fluid_mutex_lock(&pool->mutex);",
            "    if (pool->closed) {",
            "        fluid_mutex_unlock(&pool->mutex);",
            "        free(job);",
            '        fluid_memory_error("cannot submit to a closed task pool", line, column);',
            "    }",
            "    if (pool->job_count == (size_t)LLONG_MAX) {",
            "        fluid_mutex_unlock(&pool->mutex);",
            "        free(job);",
            '        fluid_memory_error("task id space exhausted", line, column);',
            "    }",
            "    if (pool->job_count == pool->job_capacity) {",
            "        size_t next = pool->job_capacity ? pool->job_capacity * 2 : 16;",
            "        if (next < pool->job_capacity || next > (size_t)-1 / sizeof(*pool->jobs)) {",
            "            fluid_mutex_unlock(&pool->mutex);",
            "            free(job);",
            '            fluid_memory_error("task list capacity overflow", line, column);',
            "        }",
            "        fluid_task_job **jobs = (fluid_task_job **)realloc(pool->jobs, next * sizeof(*jobs));",
            "        if (!jobs) {",
            "            fluid_mutex_unlock(&pool->mutex);",
            "            free(job);",
            '            fluid_memory_error("out of memory", line, column);',
            "        }",
            "        pool->jobs = jobs;",
            "        pool->job_capacity = next;",
            "    }",
            "    size_t id = pool->job_count++;",
            "    pool->jobs[id] = job;",
            "    if (pool->last) pool->last->next = job;",
            "    else pool->first = job;",
            "    pool->last = job;",
            "    fluid_condition_signal(&pool->changed);",
            "    fluid_mutex_unlock(&pool->mutex);",
            "    return (long long)id;",
            "}",
            "static fluid_task_job *fluid_task_pool_get(fluid_task_pool *pool, long long id, int line, int column)",
            "{",
            "    if (!pool || id < 0)",
            '        fluid_memory_error("task id is out of range", line, column);',
            "    fluid_mutex_lock(&pool->mutex);",
            "    if ((unsigned long long)id >= pool->job_count) {",
            "        fluid_mutex_unlock(&pool->mutex);",
            '        fluid_memory_error("task id is out of range", line, column);',
            "    }",
            "    fluid_task_job *job = pool->jobs[(size_t)id];",
            "    fluid_mutex_unlock(&pool->mutex);",
            "    return job;",
            "}",
            "static void fluid_task_pool_wait_job(fluid_task_pool *pool, fluid_task_job *job)",
            "{",
            "    fluid_mutex_lock(&pool->mutex);",
            "    while (!job->done) fluid_condition_wait(&pool->changed, &pool->mutex);",
            "    fluid_mutex_unlock(&pool->mutex);",
            "}",
        ]
        for element_type, c_type in (("i64", "long long"), ("f64", "double"), ("bool", "bool")):
            member = "boolean" if element_type == "bool" else element_type
            runtime.extend([
                f"static {c_type} fluid_task_pool_result_{element_type}(fluid_task_pool *pool, long long id, int line, int column)",
                "{",
                "    fluid_task_job *job = fluid_task_pool_get(pool, id, line, column);",
                "    fluid_task_pool_wait_job(pool, job);",
                f"    return job->result.{member};",
                "}",
            ])
        runtime.extend([
            "static void fluid_task_pool_destroy(fluid_task_pool *pool)",
            "{",
            "    if (!pool) return;",
            "    fluid_task_pool_close(pool, 0, 0);",
            "    fluid_mutex_lock(&pool->mutex);",
            "    pool->stopping = true;",
            "    fluid_condition_broadcast(&pool->changed);",
            "    fluid_mutex_unlock(&pool->mutex);",
            "    for (size_t i = 0; i < pool->thread_count; ++i)",
            "        fluid_thread_join(pool->threads[i]);",
            "    for (size_t i = 0; i < pool->job_count; ++i) free(pool->jobs[i]);",
            "    free(pool->jobs);",
            "    free(pool->threads);",
            "    fluid_condition_destroy(&pool->changed);",
            "    fluid_mutex_destroy(&pool->mutex);",
            "    free(pool);",
            "}",
        ])
        for element_type, c_type in (("i64", "long long"), ("f64", "double"), ("bool", "bool")):
            runtime.extend([
                f"typedef struct fluid_channel_{element_type} {{",
                "    fluid_mutex mutex;",
                "    fluid_condition can_send;",
                "    fluid_condition can_receive;",
                f"    {c_type} *data;",
                "    size_t capacity;",
                "    size_t count;",
                "    size_t head;",
                "    bool closed;",
                f"}} fluid_channel_{element_type};",
                f"static fluid_channel_{element_type} *fluid_channel_{element_type}_create(long long capacity, int line, int column)",
                "{",
                "    if (capacity < 1 || (unsigned long long)capacity > (size_t)-1 / sizeof(*((fluid_channel_"
                f"{element_type} *)0)->data))",
                '        fluid_memory_error("channel capacity must be positive and fit in memory", line, column);',
                f"    fluid_channel_{element_type} *channel = (fluid_channel_{element_type} *)calloc(1, sizeof(*channel));",
                "    if (!channel) fluid_memory_error(\"out of memory\", line, column);",
                "    channel->data = (" + c_type + " *)malloc((size_t)capacity * sizeof(*channel->data));",
                "    if (!channel->data) fluid_memory_error(\"out of memory\", line, column);",
                "    channel->capacity = (size_t)capacity;",
                "    fluid_mutex_init(&channel->mutex);",
                "    fluid_condition_init(&channel->can_send);",
                "    fluid_condition_init(&channel->can_receive);",
                "    return channel;",
                "}",
                f"static bool fluid_channel_{element_type}_send(fluid_channel_{element_type} *channel, {c_type} value, int line, int column)",
                "{",
                "    if (!channel) fluid_memory_error(\"invalid channel\", line, column);",
                "    fluid_mutex_lock(&channel->mutex);",
                "    while (channel->count == channel->capacity && !channel->closed)",
                "        fluid_condition_wait(&channel->can_send, &channel->mutex);",
                "    if (channel->closed) {",
                "        fluid_mutex_unlock(&channel->mutex);",
                "        return false;",
                "    }",
                "    size_t tail = (channel->head + channel->count) % channel->capacity;",
                "    channel->data[tail] = value;",
                "    ++channel->count;",
                "    fluid_condition_signal(&channel->can_receive);",
                "    fluid_mutex_unlock(&channel->mutex);",
                "    return true;",
                "}",
                f"static bool fluid_channel_{element_type}_receive(fluid_channel_{element_type} *channel, {c_type} *value, int line, int column)",
                "{",
                "    if (!channel || !value) fluid_memory_error(\"invalid channel receive\", line, column);",
                "    fluid_mutex_lock(&channel->mutex);",
                "    while (!channel->count && !channel->closed)",
                "        fluid_condition_wait(&channel->can_receive, &channel->mutex);",
                "    if (!channel->count) {",
                "        fluid_mutex_unlock(&channel->mutex);",
                "        return false;",
                "    }",
                "    *value = channel->data[channel->head];",
                "    channel->head = (channel->head + 1) % channel->capacity;",
                "    --channel->count;",
                "    fluid_condition_signal(&channel->can_send);",
                "    fluid_mutex_unlock(&channel->mutex);",
                "    return true;",
                "}",
                f"static bool fluid_channel_{element_type}_try_receive(fluid_channel_{element_type} *channel, {c_type} *value, int line, int column)",
                "{",
                "    if (!channel || !value) fluid_memory_error(\"invalid channel receive\", line, column);",
                "    fluid_mutex_lock(&channel->mutex);",
                "    if (!channel->count) {",
                "        fluid_mutex_unlock(&channel->mutex);",
                "        return false;",
                "    }",
                "    *value = channel->data[channel->head];",
                "    channel->head = (channel->head + 1) % channel->capacity;",
                "    --channel->count;",
                "    fluid_condition_signal(&channel->can_send);",
                "    fluid_mutex_unlock(&channel->mutex);",
                "    return true;",
                "}",
                f"static void fluid_channel_{element_type}_close(fluid_channel_{element_type} *channel)",
                "{",
                "    if (!channel) return;",
                "    fluid_mutex_lock(&channel->mutex);",
                "    channel->closed = true;",
                "    fluid_condition_broadcast(&channel->can_send);",
                "    fluid_condition_broadcast(&channel->can_receive);",
                "    fluid_mutex_unlock(&channel->mutex);",
                "}",
                f"static void fluid_channel_{element_type}_destroy(fluid_channel_{element_type} *channel)",
                "{",
                "    if (!channel) return;",
                f"    fluid_channel_{element_type}_close(channel);",
                "    fluid_condition_destroy(&channel->can_receive);",
                "    fluid_condition_destroy(&channel->can_send);",
                "    fluid_mutex_destroy(&channel->mutex);",
                "    free(channel->data);",
                "    free(channel);",
                "}",
            ])
        return runtime

    def compile(self) -> str:
        self.parallel_counter = 0
        self.parallel_workers = []
        self.task_workers = {}
        self.match_counter = 0
        self.enum_match_checked = set()
        self.c_name_overrides = {}
        callable_functions = self.functions + list(self.generic_templates.values())
        has_parallel = any(
            self.contains_parallel_for(function.body) for function in callable_functions
        )
        has_windows = any(
            self.contains_window_statement(function.body) for function in callable_functions
        )
        has_concurrency = any(
            self.contains_concurrency_statement(function.body) for function in callable_functions
        )
        output = [
            "#include <stdbool.h>",
            "#include <stdio.h>",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <wchar.h>",
            "#include <limits.h>",
            "#include <math.h>",
        ]
        if has_windows:
            output.extend([
                "#if defined(_WIN32)",
                "#define WIN32_LEAN_AND_MEAN",
                "#define NOMINMAX",
                "#include <windows.h>",
                "#else",
                '#error "Fluid Win32 windows require a Windows target"',
                "#endif",
            ])
        if has_parallel or has_concurrency:
            output.extend([
                "#if defined(_WIN32)",
                "#include <windows.h>",
                "#else",
                "#include <pthread.h>",
                *(["#include <unistd.h>"] if has_parallel else []),
                "#endif",
            ])
        output.extend([
            "",
            "static void fluid_memory_error(const char *message, int line, int column)",
            "{",
            '    fprintf(stderr, "fluid: %s at %d:%d\\n", message, line, column);',
            "    exit(1);",
            "}",
            "",
            "typedef struct { size_t len; size_t capacity; char *data; } fluid_string;",
            "static void fluid_string_reserve(fluid_string *value, size_t needed, int line, int column)",
            "{",
            "    if (needed <= value->capacity && value->data) return;",
            "    if (needed == (size_t)-1) fluid_memory_error(\"string length overflow\", line, column);",
            "    size_t capacity = value->capacity ? value->capacity : 16;",
            "    while (capacity < needed) {",
            "        if (capacity > ((size_t)-1 - 1) / 2) { capacity = needed; break; }",
            "        capacity *= 2;",
            "    }",
            "    char *data = (char *)realloc(value->data, capacity + 1);",
            "    if (!data) fluid_memory_error(\"out of memory\", line, column);",
            "    value->data = data;",
            "    value->capacity = capacity;",
            "}",
            "static fluid_string fluid_string_from(const char *text, int line, int column)",
            "{",
            "    if (!text) fluid_memory_error(\"cannot create String from null text\", line, column);",
            "    size_t length = strlen(text);",
            "    fluid_string value = {0, 0, NULL};",
            "    fluid_string_reserve(&value, length, line, column);",
            "    if (length) memcpy(value.data, text, length);",
            "    value.data[length] = '\\0';",
            "    value.len = length;",
            "    return value;",
            "}",
            "static fluid_string fluid_string_clone(const fluid_string *source)",
            "{",
            "    fluid_string value = {0, 0, NULL};",
            "    fluid_string_reserve(&value, source->len, 0, 0);",
            "    if (source->len) memcpy(value.data, source->data, source->len);",
            "    value.data[source->len] = '\\0';",
            "    value.len = source->len;",
            "    return value;",
            "}",
            "static fluid_string fluid_string_concat(const fluid_string *left, const fluid_string *right, int line, int column)",
            "{",
            "    if (right->len > (size_t)-1 - left->len)",
            '        fluid_memory_error("string length overflow", line, column);',
            "    size_t length = left->len + right->len;",
            "    fluid_string value = {0, 0, NULL};",
            "    fluid_string_reserve(&value, length, line, column);",
            "    if (left->len) memcpy(value.data, left->data, left->len);",
            "    if (right->len) memcpy(value.data + left->len, right->data, right->len);",
            "    value.data[length] = '\\0';",
            "    value.len = length;",
            "    return value;",
            "}",
            "static long long fluid_string_length(const fluid_string *value)",
            "{",
            "    if (value->len > (size_t)LLONG_MAX)",
            '        fluid_memory_error("String is too large for i64 length", 0, 0);',
            "    return (long long)value->len;",
            "}",
            "static bool fluid_string_equal(const fluid_string *left, const fluid_string *right)",
            "{",
            "    return left->len == right->len",
            "        && (!left->len || memcmp(left->data, right->data, left->len) == 0);",
            "}",
            "static fluid_string fluid_string_slice(const fluid_string *source, long long start, long long end, int line, int column)",
            "{",
            "    if (start < 0 || end < start || (unsigned long long)end > source->len)",
            '        fluid_memory_error("String slice bounds out of range", line, column);',
            "    size_t first = (size_t)start;",
            "    size_t last = (size_t)end;",
            "    if ((first < source->len && ((unsigned char)source->data[first] & 0xC0) == 0x80)",
            "            || (last < source->len && ((unsigned char)source->data[last] & 0xC0) == 0x80))",
            '        fluid_memory_error("String slice must use UTF-8 character boundaries", line, column);',
            "    fluid_string value = {0, 0, NULL};",
            "    fluid_string_reserve(&value, last - first, line, column);",
            "    if (last > first) memcpy(value.data, source->data + first, last - first);",
            "    value.len = last - first;",
            "    value.data[value.len] = '\\0';",
            "    return value;",
            "}",
            "static void fluid_string_append(fluid_string *value, const char *text, int line, int column)",
            "{",
            "    if (!text) fluid_memory_error(\"cannot append null text\", line, column);",
            "    size_t length = strlen(text);",
            "    if (length > (size_t)-1 - value->len)",
            '        fluid_memory_error("string length overflow", line, column);',
            "    size_t old_length = value->len;",
            "    fluid_string_reserve(value, old_length + length, line, column);",
            "    if (length) memcpy(value->data + old_length, text, length);",
            "    value->len = old_length + length;",
            "    value->data[value->len] = '\\0';",
            "}",
            "static void fluid_string_clear(fluid_string *value)",
            "{",
            "    value->len = 0;",
            "    if (value->data) value->data[0] = '\\0';",
            "}",
            "static void fluid_string_free(fluid_string *value)",
            "{",
            "    free(value->data);",
            "    value->data = NULL;",
            "    value->len = 0;",
            "    value->capacity = 0;",
            "}",
            "",
        ])
        if has_parallel:
            parallel_runtime = [
                "typedef void (*fluid_parallel_body)(long long, void *, fluid_arena *);",
                "typedef struct {",
                "    fluid_parallel_body body;",
                "    void *context;",
                "    long long start;",
                "    unsigned long long count;",
                "    unsigned int worker_count;",
                "    unsigned int worker_index;",
                "} fluid_parallel_task;",
                "static void fluid_parallel_execute(fluid_parallel_task *task)",
                "{",
                "    fluid_arena arena = fluid_arena_make();",
                "    unsigned long long base = task->count / task->worker_count;",
                "    unsigned long long remainder = task->count % task->worker_count;",
                "    unsigned long long offset = task->worker_index * base",
                "        + (task->worker_index < remainder ? task->worker_index : remainder);",
                "    unsigned long long count = base + (task->worker_index < remainder);",
                "    for (unsigned long long item = 0; item < count; ++item)",
                "        task->body(task->start + (long long)(offset + item),",
                "                   task->context, &arena);",
                "    fluid_arena_free(&arena);",
                "}",
                "#if defined(_WIN32)",
                "static DWORD WINAPI fluid_parallel_entry(LPVOID raw)",
                "{",
                "    fluid_parallel_execute((fluid_parallel_task *)raw);",
                "    return 0;",
                "}",
                "#else",
                "static void *fluid_parallel_entry(void *raw)",
                "{",
                "    fluid_parallel_execute((fluid_parallel_task *)raw);",
                "    return NULL;",
                "}",
                "#endif",
                "static void fluid_parallel_for(long long start, long long end,",
                "                               fluid_parallel_body body, void *context,",
                "                               int line, int column)",
                "{",
                "    if (end <= start) return;",
                "    unsigned long long total = (unsigned long long)end - (unsigned long long)start;",
                "    if (total > (unsigned long long)LLONG_MAX)",
                '        fluid_memory_error("parallel range is too large", line, column);',
                "    unsigned int cores = 1;",
                "#if defined(_WIN32)",
                "    SYSTEM_INFO system_info;",
                "    GetSystemInfo(&system_info);",
                "    if (!system_info.dwNumberOfProcessors)",
                '        fluid_memory_error("could not determine available CPU cores", line, column);',
                "    cores = system_info.dwNumberOfProcessors;",
                "#else",
                "    long online_cores = sysconf(_SC_NPROCESSORS_ONLN);",
                "    if (online_cores <= 0)",
                '        fluid_memory_error("could not determine available CPU cores", line, column);',
                "    cores = (unsigned int)online_cores;",
                "#endif",
                "    if (cores > 64) cores = 64;",
                "    if ((unsigned long long)cores > total) cores = (unsigned int)total;",
                "    fluid_parallel_task tasks[64];",
                "    for (unsigned int index = 0; index < cores; ++index)",
                "        tasks[index] = (fluid_parallel_task){",
                "            body, context, start, total, cores, index",
                "        };",
                "#if defined(_WIN32)",
                "    HANDLE threads[64] = {0};",
                "    for (unsigned int index = 1; index < cores; ++index) {",
                "        threads[index] = CreateThread(NULL, 0, fluid_parallel_entry,",
                "                                       &tasks[index], 0, NULL);",
                "        if (!threads[index]) {",
                '            fprintf(stderr, "fluid: worker creation failed (%lu) at %d:%d; '
                'running this chunk on the current thread\\n", (unsigned long)GetLastError(), '
                "line, column);",
                "            fluid_parallel_execute(&tasks[index]);",
                "        }",
                "    }",
                "    fluid_parallel_execute(&tasks[0]);",
                "    for (unsigned int index = 1; index < cores; ++index) {",
                "        if (threads[index]) {",
                "            if (WaitForSingleObject(threads[index], INFINITE) != WAIT_OBJECT_0)",
                '                fluid_memory_error("could not join parallel worker", line, column);',
                "            if (!CloseHandle(threads[index]))",
                '                fluid_memory_error("could not release parallel worker handle", '
                "line, column);",
                "        }",
                "    }",
                "#else",
                "    pthread_t threads[64];",
                "    bool started[64] = {false};",
                "    for (unsigned int index = 1; index < cores; ++index) {",
                "        int thread_error = pthread_create(&threads[index], NULL, "
                "fluid_parallel_entry, &tasks[index]);",
                "        if (thread_error == 0)",
                "            started[index] = true;",
                "        else {",
                '            fprintf(stderr, "fluid: worker creation failed (%d) at %d:%d; '
                'running this chunk on the current thread\\n", thread_error, line, column);',
                "            fluid_parallel_execute(&tasks[index]);",
                "        }",
                "    }",
                "    fluid_parallel_execute(&tasks[0]);",
                "    for (unsigned int index = 1; index < cores; ++index)",
                "        if (started[index] && pthread_join(threads[index], NULL) != 0)",
                '            fluid_memory_error("could not join parallel worker", line, column);',
                "#endif",
                "}",
                "",
            ]
        output.extend([
            "typedef struct fluid_arena_chunk {",
            "    struct fluid_arena_chunk *next;",
            "    size_t capacity;",
            "    size_t used;",
            "    max_align_t data[];",
            "} fluid_arena_chunk;",
            "typedef struct {",
            "    fluid_arena_chunk *first;",
            "    fluid_arena_chunk *current;",
            "    size_t next_capacity;",
            "} fluid_arena;",
            "static fluid_arena fluid_arena_make(void)",
            "{",
            "    fluid_arena result = {NULL, NULL, 4096};",
            "    return result;",
            "}",
            "static void fluid_arena_free(fluid_arena *arena)",
            "{",
            "    fluid_arena_chunk *chunk = arena->first;",
            "    while (chunk) {",
            "        fluid_arena_chunk *next = chunk->next;",
            "        free(chunk);",
            "        chunk = next;",
            "    }",
            "    arena->first = NULL;",
            "    arena->current = NULL;",
            "    arena->next_capacity = 4096;",
            "}",
            "static void *fluid_arena_alloc(fluid_arena *arena, size_t bytes, int line, int column)",
            "{",
            "    if (!bytes) return NULL;",
            "    size_t alignment = sizeof(max_align_t);",
            "    if (bytes > (size_t)-1 - alignment) fluid_memory_error(\"arena allocation overflow\", line, column);",
            "    fluid_arena_chunk *chunk = arena->current;",
            "    size_t offset = chunk ? (chunk->used + alignment - 1) / alignment * alignment : 0;",
            "    if (!chunk || offset > chunk->capacity || bytes > chunk->capacity - offset) {",
            "        size_t capacity = arena->next_capacity;",
            "        if (capacity < bytes) capacity = bytes;",
            "        if (capacity > (size_t)-1 - sizeof(*chunk)) fluid_memory_error(\"arena allocation overflow\", line, column);",
            "        chunk = (fluid_arena_chunk *)malloc(sizeof(*chunk) + capacity);",
            "        if (!chunk) fluid_memory_error(\"out of memory\", line, column);",
            "        chunk->next = NULL;",
            "        chunk->capacity = capacity;",
            "        chunk->used = 0;",
            "        if (arena->current) arena->current->next = chunk;",
            "        else arena->first = chunk;",
            "        arena->current = chunk;",
            "        arena->next_capacity = capacity <= (size_t)-1 / 2 ? capacity * 2 : capacity;",
            "        offset = 0;",
            "    }",
            "    unsigned char *data = (unsigned char *)chunk->data;",
            "    void *result = data + offset;",
            "    chunk->used = offset + bytes;",
            "    return result;",
            "}",
            "",
            "static size_t fluid_checked_index(long long index, size_t length, int line, int column);",
            "static size_t fluid_checked_insert_index(long long index, size_t length, int line, int column);",
            "static void fluid_checked_slice(long long start, long long end, size_t length, int line, int column);",
            "",
            "#define FLUID_DEFINE_VECTOR(NAME, TYPE) \\",
            "typedef struct { size_t len; size_t capacity; TYPE *data; } fluid_vec_##NAME; \\",
            "static void fluid_vec_##NAME##_free(fluid_vec_##NAME *vec) \\",
            "{ \\",
            "    free(vec->data); \\",
            "    vec->data = NULL; \\",
            "    vec->len = 0; \\",
            "    vec->capacity = 0; \\",
            "} \\",
            "static fluid_vec_##NAME fluid_vec_##NAME##_take(fluid_vec_##NAME *vec) \\",
            "{ \\",
            "    fluid_vec_##NAME result = *vec; \\",
            "    *vec = (fluid_vec_##NAME){0, 0, NULL}; \\",
            "    return result; \\",
            "} \\",
            "static void fluid_vec_##NAME##_ensure(fluid_vec_##NAME *vec, size_t needed, int line, int column) \\",
            "{ \\",
            "    if (needed <= vec->capacity) return; \\",
            "    size_t max_items = (size_t)-1 / sizeof(TYPE); \\",
            "    if (needed > max_items) fluid_memory_error(\"vector capacity overflow\", line, column); \\",
            "    size_t next = vec->capacity ? vec->capacity : 4; \\",
            "    while (next < needed) { \\",
            "        if (next > max_items / 2) { next = needed; break; } \\",
            "        next *= 2; \\",
            "    } \\",
            "    TYPE *data = (TYPE *)realloc(vec->data, next * sizeof(TYPE)); \\",
            "    if (!data) fluid_memory_error(\"out of memory\", line, column); \\",
            "    vec->data = data; \\",
            "    vec->capacity = next; \\",
            "} \\",
            "static fluid_vec_##NAME fluid_vec_##NAME##_make(TYPE *items, size_t count, int line, int column) \\",
            "{ \\",
            "    fluid_vec_##NAME result = {0, 0, NULL}; \\",
            "    if (count) { \\",
            "        fluid_vec_##NAME##_ensure(&result, count, line, column); \\",
            "        memcpy(result.data, items, count * sizeof(TYPE)); \\",
            "        result.len = count; \\",
            "    } \\",
            "    return result; \\",
            "} \\",
            "static void fluid_vec_##NAME##_push(fluid_vec_##NAME *vec, TYPE value, int line, int column) \\",
            "{ \\",
            "    if (vec->len == (size_t)-1) fluid_memory_error(\"vector length overflow\", line, column); \\",
            "    fluid_vec_##NAME##_ensure(vec, vec->len + 1, line, column); \\",
            "    vec->data[vec->len++] = value; \\",
            "} \\",
            "static void fluid_vec_##NAME##_insert(fluid_vec_##NAME *vec, long long index, TYPE value, int line, int column) \\",
            "{ \\",
            "    size_t at = fluid_checked_insert_index(index, vec->len, line, column); \\",
            "    if (vec->len == (size_t)-1) fluid_memory_error(\"vector length overflow\", line, column); \\",
            "    fluid_vec_##NAME##_ensure(vec, vec->len + 1, line, column); \\",
            "    memmove(vec->data + at + 1, vec->data + at, (vec->len - at) * sizeof(TYPE)); \\",
            "    vec->data[at] = value; \\",
            "    ++vec->len; \\",
            "} \\",
            "static TYPE fluid_vec_##NAME##_pop(fluid_vec_##NAME *vec, int line, int column) \\",
            "{ \\",
            "    if (!vec->len) fluid_memory_error(\"pop from empty vector\", line, column); \\",
            "    return vec->data[--vec->len]; \\",
            "} \\",
            "static TYPE fluid_vec_##NAME##_remove(fluid_vec_##NAME *vec, long long index, int line, int column) \\",
            "{ \\",
            "    size_t at = fluid_checked_index(index, vec->len, line, column); \\",
            "    TYPE value = vec->data[at]; \\",
            "    memmove(vec->data + at, vec->data + at + 1, (vec->len - at - 1) * sizeof(TYPE)); \\",
            "    --vec->len; \\",
            "    return value; \\",
            "} \\",
            "static void fluid_vec_##NAME##_reserve(fluid_vec_##NAME *vec, long long requested, int line, int column) \\",
            "{ \\",
            "    if (requested < 0) fluid_memory_error(\"negative vector capacity\", line, column); \\",
            "    fluid_vec_##NAME##_ensure(vec, (size_t)requested, line, column); \\",
            "} \\",
            "static void fluid_vec_##NAME##_shrink_to_fit(fluid_vec_##NAME *vec, int line, int column) \\",
            "{ \\",
            "    if (vec->len == vec->capacity) return; \\",
            "    if (!vec->len) { free(vec->data); vec->data = NULL; vec->capacity = 0; return; } \\",
            "    TYPE *data = (TYPE *)realloc(vec->data, vec->len * sizeof(TYPE)); \\",
            "    if (!data) fluid_memory_error(\"out of memory\", line, column); \\",
            "    vec->data = data; \\",
            "    vec->capacity = vec->len; \\",
            "} \\",
            "static void fluid_vec_##NAME##_clear(fluid_vec_##NAME *vec) \\",
            "{ vec->len = 0; }",
            "FLUID_DEFINE_VECTOR(i64, long long)",
            "FLUID_DEFINE_VECTOR(f64, double)",
            "FLUID_DEFINE_VECTOR(bool, bool)",
            "FLUID_DEFINE_VECTOR(str, const char *)",
            "#undef FLUID_DEFINE_VECTOR",
            "",
            "#define FLUID_DEFINE_SLICE(NAME, TYPE, INITIAL) \\",
            "typedef struct { size_t len; TYPE *data; } fluid_slice_##NAME; \\",
            "static fluid_slice_##NAME fluid_slice_##NAME##_make(TYPE *data, size_t length, long long start, long long end, int line, int column) \\",
            "{ \\",
            "    fluid_checked_slice(start, end, length, line, column); \\",
            "    fluid_slice_##NAME result = {(size_t)(end - start), data ? data + start : NULL}; \\",
            "    return result; \\",
            "} \\",
            "static fluid_slice_##NAME fluid_arena_##NAME(fluid_arena *arena, long long count, int line, int column) \\",
            "{ \\",
            "    if (count < 0 || (unsigned long long)count > (size_t)-1 / sizeof(TYPE)) fluid_memory_error(\"invalid arena element count\", line, column); \\",
            "    size_t length = (size_t)count; \\",
            "    TYPE *data = (TYPE *)fluid_arena_alloc(arena, length * sizeof(TYPE), line, column); \\",
            "    for (size_t i = 0; i < length; ++i) data[i] = (TYPE)(INITIAL); \\",
            "    fluid_slice_##NAME result = {length, data}; \\",
            "    return result; \\",
            "}",
            "FLUID_DEFINE_SLICE(i64, long long, 0)",
            "FLUID_DEFINE_SLICE(f64, double, 0)",
            "FLUID_DEFINE_SLICE(bool, bool, 0)",
            'FLUID_DEFINE_SLICE(str, const char *, "")',
            "#undef FLUID_DEFINE_SLICE",
            "",
            "static long long fluid_abs_i64(long long value)",
            "{",
            "    if (value == LLONG_MIN) {",
            '        fprintf(stderr, "fluid: abs overflow for i64 minimum\\n");',
            "        exit(1);",
            "    }",
            "    return value < 0 ? -value : value;",
            "}",
            "static inline long long fluid_min_i64(long long left, long long right)",
            "{",
            "    return left < right ? left : right;",
            "}",
            "static inline long long fluid_max_i64(long long left, long long right)",
            "{",
            "    return left > right ? left : right;",
            "}",
            "static long long fluid_clamp_i64(long long value, long long lower, long long upper)",
            "{",
            "    if (lower > upper) {",
            '        fprintf(stderr, "fluid: clamp lower bound exceeds upper bound\\n");',
            "        exit(1);",
            "    }",
            "    return value < lower ? lower : (value > upper ? upper : value);",
            "}",
            "static inline long long fluid_sign_i64(long long value)",
            "{",
            "    return (value > 0) - (value < 0);",
            "}",
            "static inline bool fluid_contains(const char *text, const char *part)",
            "{",
            "    return strstr(text, part) != NULL;",
            "}",
            "static inline bool fluid_starts_with(const char *text, const char *prefix)",
            "{",
            "    return strncmp(text, prefix, strlen(prefix)) == 0;",
            "}",
            "static inline bool fluid_ends_with(const char *text, const char *suffix)",
            "{",
            "    size_t text_length = strlen(text);",
            "    size_t suffix_length = strlen(suffix);",
            "    return suffix_length <= text_length",
            "        && memcmp(text + text_length - suffix_length, suffix, suffix_length) == 0;",
            "}",
            "static inline bool fluid_is_empty(const char *text)",
            "{",
            "    return text[0] == '\\0';",
            "}",
            "static void fluid_assert(bool condition, int line, int column)",
            "{",
            "    if (!condition) {",
            '        fprintf(stderr, "fluid: assertion failed at %d:%d\\n", line, column);',
            "        exit(1);",
            "    }",
            "}",
            "",
            "static size_t fluid_checked_index(long long index, size_t length, int line, int column)",
            "{",
            "    if (index < 0 || (unsigned long long)index >= length) {",
            '        fprintf(stderr, "fluid: array index out of bounds at %d:%d\\n", line, column);',
            "        exit(1);",
            "    }",
            "    return (size_t)index;",
            "}",
            "",
            "static void fluid_checked_slice(long long start, long long end, size_t length, int line, int column)",
            "{",
            "    if (start < 0 || end < start || (unsigned long long)end > length) {",
            '        fprintf(stderr, "fluid: slice bounds out of range at %d:%d\\n", line, column);',
            "        exit(1);",
            "    }",
            "}",
            "",
            "static size_t fluid_checked_insert_index(long long index, size_t length, int line, int column)",
            "{",
            "    if (index < 0 || (unsigned long long)index > length) {",
            '        fprintf(stderr, "fluid: vector insert index out of bounds at %d:%d\\n", line, column);',
            "        exit(1);",
            "    }",
            "    return (size_t)index;",
            "}",
            "",
        ])
        if has_parallel:
            output.extend(parallel_runtime)
        if has_concurrency:
            output.extend(self.concurrency_runtime())
        if has_windows:
            output.extend([
                "#define FLUID_EVENT_NONE 0LL",
                "#define FLUID_EVENT_KEY_DOWN 1LL",
                "#define FLUID_EVENT_KEY_UP 2LL",
                "#define FLUID_EVENT_MOUSE_MOVE 3LL",
                "#define FLUID_EVENT_MOUSE_DOWN 4LL",
                "#define FLUID_EVENT_MOUSE_UP 5LL",
                "#define FLUID_EVENT_MOUSE_WHEEL 6LL",
                "#define FLUID_EVENT_TEXT 7LL",
                "#define FLUID_EVENT_RESIZE 8LL",
                "#define FLUID_EVENT_CLOSE 9LL",
                "#define FLUID_EVENT_CONTROL 10LL",
                "#define FLUID_WINDOW_EVENT_CAPACITY 256",
                "typedef struct {",
                "    long long kind;",
                "    long long key;",
                "    long long x;",
                "    long long y;",
                "    long long button;",
                "    long long delta;",
                "    long long text;",
                "    long long control_id;",
                "    bool checked;",
                "    long long width;",
                "    long long height;",
                "} fluid_window_event;",
                "typedef struct fluid_window_state {",
                "    HWND handle;",
                "    bool open;",
                "    bool controls_ready;",
                "    fluid_window_event events[FLUID_WINDOW_EVENT_CAPACITY];",
                "    size_t event_head;",
                "    size_t event_count;",
                "    bool event_overflow;",
                "    HDC back_dc;",
                "    HBITMAP back_bitmap;",
                "    HGDIOBJ original_bitmap;",
                "    int back_width;",
                "    int back_height;",
                "    char *text_buffer;",
                "    size_t text_capacity;",
                "} fluid_window_state;",
                "typedef struct { fluid_window_state *state; } fluid_window;",
                "static const wchar_t *fluid_window_class_name = L\"FluidWindowClass\";",
                "static void fluid_window_error(const char *, int, int);",
                "static void fluid_window_queue(fluid_window_state *state,",
                "                               fluid_window_event event)",
                "{",
                "    if (!state) return;",
                "    if (state->event_count && (event.kind == FLUID_EVENT_MOUSE_MOVE",
                "            || event.kind == FLUID_EVENT_RESIZE)) {",
                "        size_t last = (state->event_head + state->event_count - 1)",
                "            % FLUID_WINDOW_EVENT_CAPACITY;",
                "        if (state->events[last].kind == event.kind) {",
                "            state->events[last] = event;",
                "            return;",
                "        }",
                "    }",
                "    if (state->event_count == FLUID_WINDOW_EVENT_CAPACITY) {",
                "        if (state) state->event_overflow = true;",
                "        return;",
                "    }",
                "    size_t index = (state->event_head + state->event_count)",
                "        % FLUID_WINDOW_EVENT_CAPACITY;",
                "    state->events[index] = event;",
                "    ++state->event_count;",
                "}",
                "static void fluid_window_release_backbuffer(fluid_window_state *state)",
                "{",
                "    if (state->back_dc && state->original_bitmap)",
                "        SelectObject(state->back_dc, state->original_bitmap);",
                "    if (state->back_bitmap && !DeleteObject(state->back_bitmap))",
                '        fluid_window_error("DeleteObject(backbuffer)", 0, 0);',
                "    if (state->back_dc && !DeleteDC(state->back_dc))",
                '        fluid_window_error("DeleteDC(backbuffer)", 0, 0);',
                "    state->back_dc = NULL;",
                "    state->back_bitmap = NULL;",
                "    state->original_bitmap = NULL;",
                "    state->back_width = 0;",
                "    state->back_height = 0;",
                "}",
                "static bool fluid_window_ensure_backbuffer(fluid_window_state *state,",
                "                                            int line, int column)",
                "{",
                "    RECT client;",
                "    if (!GetClientRect(state->handle, &client))",
                '        fluid_window_error("GetClientRect", line, column);',
                "    int width = client.right - client.left;",
                "    int height = client.bottom - client.top;",
                "    if (width <= 0 || height <= 0) return false;",
                "    if (state->back_dc && state->back_width == width",
                "            && state->back_height == height) return true;",
                "    fluid_window_release_backbuffer(state);",
                "    HDC screen = GetDC(state->handle);",
                "    if (!screen) fluid_window_error(\"GetDC\", line, column);",
                "    state->back_dc = CreateCompatibleDC(screen);",
                "    state->back_bitmap = CreateCompatibleBitmap(screen, width, height);",
                "    if (!state->back_dc || !state->back_bitmap) {",
                "        DWORD error = GetLastError();",
                "        if (state->back_bitmap) DeleteObject(state->back_bitmap);",
                "        if (state->back_dc) DeleteDC(state->back_dc);",
                "        state->back_bitmap = NULL;",
                "        state->back_dc = NULL;",
                "        ReleaseDC(state->handle, screen);",
                "        SetLastError(error);",
                '        fluid_window_error("CreateCompatibleBitmap", line, column);',
                "    }",
                "    state->original_bitmap = SelectObject(state->back_dc, state->back_bitmap);",
                "    if (!state->original_bitmap || state->original_bitmap == HGDI_ERROR) {",
                "        DWORD error = GetLastError();",
                "        DeleteObject(state->back_bitmap);",
                "        DeleteDC(state->back_dc);",
                "        state->back_bitmap = NULL;",
                "        state->back_dc = NULL;",
                "        ReleaseDC(state->handle, screen);",
                "        SetLastError(error);",
                '        fluid_window_error("SelectObject(backbuffer)", line, column);',
                "    }",
                "    state->back_width = width;",
                "    state->back_height = height;",
                "    if (!PatBlt(state->back_dc, 0, 0, width, height, WHITENESS))",
                '        fluid_window_error("PatBlt(backbuffer initialization)", line, column);',
                "    ReleaseDC(state->handle, screen);",
                "    return true;",
                "}",
                "static COLORREF fluid_window_color(long long color, int line, int column)",
                "{",
                "    if (color < 0 || color > 0xFFFFFF)",
                '        fluid_memory_error("color must be an RGB value from 0 to 0xFFFFFF", '
                "line, column);",
                "    return RGB((BYTE)((color >> 16) & 255),",
                "               (BYTE)((color >> 8) & 255), (BYTE)(color & 255));",
                "}",
                "static int fluid_window_coordinate(long long value, int line, int column)",
                "{",
                "    if (value < INT_MIN || value > INT_MAX)",
                '        fluid_memory_error("drawing coordinate is outside the Win32 range", '
                "line, column);",
                "    return (int)value;",
                "}",
                "static LRESULT CALLBACK fluid_window_proc(HWND handle, UINT message,",
                "                                           WPARAM wparam, LPARAM lparam)",
                "{",
                "    fluid_window_state *state = (fluid_window_state *)GetWindowLongPtrW(",
                "        handle, GWLP_USERDATA);",
                "    if (message == WM_NCCREATE) {",
                "        CREATESTRUCTW *creation = (CREATESTRUCTW *)lparam;",
                "        state = (fluid_window_state *)creation->lpCreateParams;",
                "        state->handle = handle;",
                "        state->open = true;",
                "        SetWindowLongPtrW(handle, GWLP_USERDATA, (LONG_PTR)state);",
                "        return TRUE;",
                "    }",
                "    if (message == WM_PAINT && state) {",
                "        PAINTSTRUCT paint;",
                "        HDC target = BeginPaint(handle, &paint);",
                "        if (!target) fluid_window_error(\"BeginPaint\", 0, 0);",
                "        if (state->back_dc) {",
                "            int left = paint.rcPaint.left < 0 ? 0 : paint.rcPaint.left;",
                "            int top = paint.rcPaint.top < 0 ? 0 : paint.rcPaint.top;",
                "            int right = paint.rcPaint.right > state->back_width",
                "                ? state->back_width : paint.rcPaint.right;",
                "            int bottom = paint.rcPaint.bottom > state->back_height",
                "                ? state->back_height : paint.rcPaint.bottom;",
                "            if (right > left && bottom > top",
                "                    && !BitBlt(target, left, top, right - left,",
                "                              bottom - top, state->back_dc, left, top, SRCCOPY))",
                '                fluid_window_error("BitBlt", 0, 0);',
                "        } else {",
                "            HBRUSH background = (HBRUSH)(COLOR_WINDOW + 1);",
                "            if (!FillRect(target, &paint.rcPaint, background))",
                '                fluid_window_error("FillRect", 0, 0);',
                "        }",
                "        if (!EndPaint(handle, &paint)) fluid_window_error(\"EndPaint\", 0, 0);",
                "        return 0;",
                "    }",
                "    if (message == WM_ERASEBKGND && state && state->back_dc) return 1;",
                "    if (message == WM_CLOSE && state) {",
                "        fluid_window_queue(state, (fluid_window_event){.kind = FLUID_EVENT_CLOSE});",
                "        if (!DestroyWindow(handle)) fluid_window_error(\"DestroyWindow\", 0, 0);",
                "        return 0;",
                "    }",
                "    if (message == WM_DESTROY && state) {",
                "        state->open = false;",
                "        return 0;",
                "    }",
                "    if (message == WM_NCDESTROY && state) {",
                "        state->handle = NULL;",
                "        state->open = false;",
                "        SetWindowLongPtrW(handle, GWLP_USERDATA, 0);",
                "    }",
                "    if (state) {",
                "        switch (message) {",
                "        case WM_COMMAND:",
                "            if (lparam && (HIWORD(wparam) == BN_CLICKED",
                "                    || (state->controls_ready && HIWORD(wparam) == EN_CHANGE))) {",
                "                UINT id = LOWORD(wparam);",
                "                fluid_window_queue(state, (fluid_window_event){",
                "                    .kind = FLUID_EVENT_CONTROL, .control_id = (long long)id,",
                "                    .checked = IsDlgButtonChecked(handle, (int)id) == BST_CHECKED});",
                "                return 0;",
                "            }",
                "            break;",
                "        case WM_KEYDOWN:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_KEY_DOWN, .key = (long long)wparam});",
                "            break;",
                "        case WM_SYSKEYDOWN:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_KEY_DOWN, .key = (long long)wparam});",
                "            break;",
                "        case WM_KEYUP:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_KEY_UP, .key = (long long)wparam});",
                "            break;",
                "        case WM_SYSKEYUP:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_KEY_UP, .key = (long long)wparam});",
                "            break;",
                "        case WM_CHAR:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_TEXT, .text = (long long)wparam});",
                "            break;",
                "        case WM_MOUSEMOVE:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_MOUSE_MOVE,",
                "                .x = (short)LOWORD(lparam), .y = (short)HIWORD(lparam)});",
                "            break;",
                "        case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_MOUSE_DOWN,",
                "                .button = message == WM_LBUTTONDOWN ? 1 :",
                "                          (message == WM_RBUTTONDOWN ? 2 : 3),",
                "                .x = (short)LOWORD(lparam), .y = (short)HIWORD(lparam)});",
                "            break;",
                "        case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP:",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_MOUSE_UP,",
                "                .button = message == WM_LBUTTONUP ? 1 :",
                "                          (message == WM_RBUTTONUP ? 2 : 3),",
                "                .x = (short)LOWORD(lparam), .y = (short)HIWORD(lparam)});",
                "            break;",
                "        case WM_MOUSEWHEEL: {",
                "            POINT point = {(short)LOWORD(lparam), (short)HIWORD(lparam)};",
                "            if (!ScreenToClient(handle, &point))",
                '                fluid_window_error("ScreenToClient", 0, 0);',
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_MOUSE_WHEEL,",
                "                .delta = (short)HIWORD(wparam), .x = point.x, .y = point.y});",
                "            break;",
                "        }",
                "        case WM_SIZE:",
                "            fluid_window_release_backbuffer(state);",
                "            fluid_window_queue(state, (fluid_window_event){",
                "                .kind = FLUID_EVENT_RESIZE,",
                "                .width = (long long)LOWORD(lparam),",
                "                .height = (long long)HIWORD(lparam)});",
                "            break;",
                "        default:",
                "            break;",
                "        }",
                "    }",
                "    return DefWindowProcW(handle, message, wparam, lparam);",
                "}",
                "static void fluid_window_error(const char *operation, int line, int column)",
                "{",
                '    fprintf(stderr, "fluid: Win32 %s failed (error %lu) at %d:%d\\n",',
                "            operation, (unsigned long)GetLastError(), line, column);",
                "    exit(1);",
                "}",
                "static wchar_t *fluid_window_wide(const char *text, int line, int column)",
                "{",
                "    if (!text) fluid_memory_error(\"window title cannot be null\", line, column);",
                "    size_t byte_count = strlen(text);",
                "    if (byte_count > INT_MAX) fluid_memory_error(\"window title is too long\", line, column);",
                "    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text,",
                "                                   -1, NULL, 0);",
                "    if (!count) fluid_window_error(\"UTF-8 title conversion\", line, column);",
                "    wchar_t *wide = (wchar_t *)malloc((size_t)count * sizeof(wchar_t));",
                "    if (!wide) fluid_memory_error(\"out of memory\", line, column);",
                "    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text,",
                "                             -1, wide, count)) {",
                "        free(wide);",
                "        fluid_window_error(\"UTF-8 title conversion\", line, column);",
                "    }",
                "    return wide;",
                "}",
                "static void fluid_window_register_class(int line, int column)",
                "{",
                "    static ATOM registered;",
                "    if (registered) return;",
                "    HINSTANCE instance = GetModuleHandleW(NULL);",
                "    if (!instance) fluid_window_error(\"GetModuleHandleW\", line, column);",
                "    WNDCLASSEXW definition = {0};",
                "    definition.cbSize = sizeof(definition);",
                "    definition.style = CS_HREDRAW | CS_VREDRAW;",
                "    definition.lpfnWndProc = fluid_window_proc;",
                "    definition.hInstance = instance;",
                "    definition.hCursor = LoadCursorW(NULL, MAKEINTRESOURCEW(32512));",
                "    if (!definition.hCursor) fluid_window_error(\"LoadCursorW\", line, column);",
                "    definition.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);",
                "    definition.lpszClassName = fluid_window_class_name;",
                "    registered = RegisterClassExW(&definition);",
                "    if (!registered && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)",
                '        fluid_window_error("RegisterClassExW", line, column);',
                "    if (!registered) registered = 1;",
                "}",
                "static HWND fluid_window_require(fluid_window *window, int line, int column)",
                "{",
                "    if (!window || !window->state || !window->state->open",
                "            || !window->state->handle || !IsWindow(window->state->handle))",
                "        fluid_memory_error(\"window is closed\", line, column);",
                "    return window->state->handle;",
                "}",
                "static fluid_window fluid_window_create(const char *title, long long width,",
                "                                       long long height, int line, int column)",
                "{",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("window dimensions must be positive and fit in an int", '
                "line, column);",
                "    wchar_t *wide_title = fluid_window_wide(title, line, column);",
                "    fluid_window_register_class(line, column);",
                "    HINSTANCE instance = GetModuleHandleW(NULL);",
                "    fluid_window_state *state = (fluid_window_state *)calloc(1, sizeof(*state));",
                "    if (!state) {",
                "        free(wide_title);",
                '        fluid_memory_error("out of memory", line, column);',
                "    }",
                "    RECT frame = {0, 0, (LONG)width, (LONG)height};",
                "    DWORD style = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN;",
                "    if (!AdjustWindowRectEx(&frame, style, FALSE, 0)) {",
                "        free(state);",
                "        free(wide_title);",
                '        fluid_window_error("AdjustWindowRectEx", line, column);',
                "    }",
                "    if (frame.right <= frame.left || frame.bottom <= frame.top) {",
                "        free(state);",
                "        free(wide_title);",
                '        fluid_memory_error("adjusted window dimensions overflow", line, column);',
                "    }",
                "    HWND handle = CreateWindowExW(0, fluid_window_class_name, wide_title,",
                "        style, CW_USEDEFAULT, CW_USEDEFAULT, frame.right - frame.left,",
                "        frame.bottom - frame.top, NULL, NULL, instance, state);",
                "    free(wide_title);",
                "    if (!handle) {",
                "        free(state);",
                '        fluid_window_error("CreateWindowExW", line, column);',
                "    }",
                "    fluid_window result = {state};",
                "    return result;",
                "}",
                "static void fluid_window_show(fluid_window *window, int line, int column)",
                "{",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    ShowWindow(handle, SW_SHOW);",
                "    UpdateWindow(handle);",
                "    window->state->controls_ready = true;",
                "}",
                "static void fluid_window_hide(fluid_window *window, int line, int column)",
                "{",
                "    ShowWindow(fluid_window_require(window, line, column), SW_HIDE);",
                "}",
                "static void fluid_window_close(fluid_window *window, int line, int column)",
                "{",
                "    if (window && window->state && window->state->handle",
                "            && IsWindow(window->state->handle))",
                "        if (!DestroyWindow(window->state->handle))",
                '            fluid_window_error("DestroyWindow", line, column);',
                "    (void)line;",
                "    (void)column;",
                "}",
                "static void fluid_window_destroy(fluid_window *window)",
                "{",
                "    if (!window || !window->state) return;",
                "    if (window->state->handle && IsWindow(window->state->handle))",
                "        if (!DestroyWindow(window->state->handle))",
                '            fluid_window_error("DestroyWindow during cleanup", 0, 0);',
                "    fluid_window_release_backbuffer(window->state);",
                "    free(window->state->text_buffer);",
                "    free(window->state);",
                "    window->state = NULL;",
                "}",
                "static bool fluid_window_poll(fluid_window *window, int line, int column)",
                "{",
                "    (void)line;",
                "    (void)column;",
                "    if (!window || !window->state) return false;",
                "    MSG message;",
                "    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {",
                "        TranslateMessage(&message);",
                "        DispatchMessageW(&message);",
                "    }",
                "    return window->state->open || window->state->event_count != 0;",
                "}",
                "static bool fluid_window_is_open(fluid_window *window, int line, int column)",
                "{",
                "    (void)line;",
                "    (void)column;",
                "    return window && window->state && window->state->open;",
                "}",
                "static bool fluid_window_wait(fluid_window *window, int line, int column)",
                "{",
                "    if (!window || !window->state) return false;",
                "    if (window->state->event_count) return true;",
                "    if (!fluid_window_is_open(window, line, column)) return false;",
                "    if (!WaitMessage()) fluid_window_error(\"WaitMessage\", line, column);",
                "    return fluid_window_poll(window, line, column);",
                "}",
                "static fluid_window_event fluid_window_next_event(fluid_window *window,",
                "                                                  int line, int column)",
                "{",
                "    fluid_window_event empty = {0};",
                "    if (!window || !window->state) return empty;",
                "    if (window->state->event_overflow)",
                '        fluid_memory_error("window event queue overflow", line, column);',
                "    if (!window->state->event_count) return empty;",
                "    fluid_window_event event = window->state->events[window->state->event_head];",
                "    window->state->events[window->state->event_head] = empty;",
                "    window->state->event_head = (window->state->event_head + 1)",
                "        % FLUID_WINDOW_EVENT_CAPACITY;",
                "    --window->state->event_count;",
                "    return event;",
                "}",
                "static HWND fluid_window_control(fluid_window *window, long long id,",
                "                                 int line, int column)",
                "{",
                "    if (id <= 0 || id > 65535)",
                '        fluid_memory_error("control id must be from 1 to 65535", line, column);',
                "    HWND handle = fluid_window_require(window, line, column);",
                "    HWND control = GetDlgItem(handle, (int)id);",
                "    if (!control)",
                '        fluid_memory_error("window control id was not found", line, column);',
                "    return control;",
                "}",
                "static void fluid_window_add_control(fluid_window *window, long long id,",
                "                                    const char *text, long long x, long long y,",
                "                                    long long width, long long height,",
                "                                    const wchar_t *class_name, DWORD style,",
                "                                    int line, int column)",
                "{",
                "    HWND parent = fluid_window_require(window, line, column);",
                "    if (id <= 0 || id > 65535)",
                '        fluid_memory_error("control id must be from 1 to 65535", line, column);',
                "    if (GetDlgItem(parent, (int)id))",
                '        fluid_memory_error("window control id is already in use", line, column);',
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("control dimensions must be positive", line, column);',
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("control bounds exceed the Win32 range", line, column);',
                "    wchar_t *wide_text = fluid_window_wide(text, line, column);",
                "    HINSTANCE instance = GetModuleHandleW(NULL);",
                "    if (!instance) { free(wide_text); fluid_window_error(\"GetModuleHandleW\", line, column); }",
                "    HWND control = CreateWindowExW(0, class_name, wide_text,",
                "        WS_CHILD | WS_VISIBLE | style, left, top, (int)width, (int)height,",
                "        parent, (HMENU)(INT_PTR)id, instance, NULL);",
                "    free(wide_text);",
                "    if (!control) fluid_window_error(\"CreateWindowExW(control)\", line, column);",
                "    HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);",
                "    if (!font) fluid_window_error(\"GetStockObject(DEFAULT_GUI_FONT)\", line, column);",
                "    SendMessageW(control, WM_SETFONT, (WPARAM)font, TRUE);",
                "}",
                "static void fluid_window_add_button(fluid_window *window, long long id,",
                "                                   const char *text, long long x, long long y,",
                "                                   long long width, long long height,",
                "                                   int line, int column)",
                "{",
                "    fluid_window_add_control(window, id, text, x, y, width, height,",
                "        L\"BUTTON\", BS_PUSHBUTTON | WS_TABSTOP, line, column);",
                "}",
                "static void fluid_window_add_checkbox(fluid_window *window, long long id,",
                "                                      const char *text, long long x, long long y,",
                "                                      long long width, long long height,",
                "                                      int line, int column)",
                "{",
                "    fluid_window_add_control(window, id, text, x, y, width, height,",
                "        L\"BUTTON\", BS_AUTOCHECKBOX | WS_TABSTOP, line, column);",
                "}",
                "static void fluid_window_add_label(fluid_window *window, long long id,",
                "                                   const char *text, long long x, long long y,",
                "                                   long long width, long long height,",
                "                                   int line, int column)",
                "{",
                "    fluid_window_add_control(window, id, text, x, y, width, height,",
                "        L\"STATIC\", SS_LEFT, line, column);",
                "}",
                "static void fluid_window_add_textbox(fluid_window *window, long long id,",
                "                                     const char *text, long long x, long long y,",
                "                                     long long width, long long height,",
                "                                     int line, int column)",
                "{",
                "    fluid_window_add_control(window, id, text, x, y, width, height,",
                "        L\"EDIT\", ES_LEFT | ES_AUTOHSCROLL | WS_BORDER | WS_TABSTOP,",
                "        line, column);",
                "}",
                "static HWND fluid_window_textbox_control(fluid_window *window, long long id,",
                "                                         int line, int column)",
                "{",
                "    HWND control = fluid_window_control(window, id, line, column);",
                "    wchar_t class_name[16];",
                "    if (!GetClassNameW(control, class_name, 16))",
                '        fluid_window_error("GetClassNameW", line, column);',
                "    if (wcscmp(class_name, L\"Edit\") != 0)",
                '        fluid_memory_error("control is not a text box", line, column);',
                "    return control;",
                "}",
                "static const char *fluid_window_textbox_text(fluid_window *window, long long id,",
                "                                             int line, int column)",
                "{",
                "    HWND control = fluid_window_textbox_control(window, id, line, column);",
                "    int length = GetWindowTextLengthW(control);",
                "    if (length < 0 || length == INT_MAX)",
                '        fluid_memory_error("text box content is too large", line, column);',
                "    size_t wide_capacity = (size_t)length + 1;",
                "    wchar_t *wide = (wchar_t *)malloc(wide_capacity * sizeof(wchar_t));",
                "    if (!wide) fluid_memory_error(\"out of memory\", line, column);",
                "    int copied = GetWindowTextW(control, wide, length + 1);",
                "    if (copied < 0 || copied > length) { free(wide);",
                '        fluid_window_error("GetWindowTextW", line, column); }',
                "    int needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,",
                "        wide, -1, NULL, 0, NULL, NULL);",
                "    if (!needed) { free(wide);",
                '        fluid_window_error("text box UTF-8 conversion", line, column); }',
                "    if ((size_t)needed > window->state->text_capacity) {",
                "        char *replacement = (char *)realloc(window->state->text_buffer,",
                "                                            (size_t)needed);",
                "        if (!replacement) { free(wide);",
                '            fluid_memory_error("out of memory", line, column); }',
                "        window->state->text_buffer = replacement;",
                "        window->state->text_capacity = (size_t)needed;",
                "    }",
                "    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide, -1,",
                "            window->state->text_buffer, needed, NULL, NULL)) {",
                "        free(wide);",
                '        fluid_window_error("text box UTF-8 conversion", line, column);',
                "    }",
                "    free(wide);",
                "    return window->state->text_buffer;",
                "}",
                "static void fluid_window_set_textbox_text(fluid_window *window, long long id,",
                "                                          const char *text, int line, int column)",
                "{",
                "    HWND control = fluid_window_textbox_control(window, id, line, column);",
                "    wchar_t *wide = fluid_window_wide(text, line, column);",
                "    if (!SetWindowTextW(control, wide)) { free(wide);",
                '        fluid_window_error("SetWindowTextW(text box)", line, column); }',
                "    free(wide);",
                "}",
                "static void fluid_window_set_control_text(fluid_window *window, long long id,",
                "                                          const char *text, int line, int column)",
                "{",
                "    HWND control = fluid_window_control(window, id, line, column);",
                "    wchar_t *wide = fluid_window_wide(text, line, column);",
                "    if (!SetWindowTextW(control, wide)) { free(wide);",
                '        fluid_window_error("SetWindowTextW(control)", line, column); }',
                "    free(wide);",
                "}",
                "static void fluid_window_set_control_enabled(fluid_window *window, long long id,",
                "                                             bool enabled, int line, int column)",
                "{",
                "    EnableWindow(fluid_window_control(window, id, line, column), enabled);",
                "}",
                "static bool fluid_window_checkbox_checked(fluid_window *window, long long id,",
                "                                         int line, int column)",
                "{",
                "    HWND control = fluid_window_control(window, id, line, column);",
                "    if (((DWORD)GetWindowLongPtrW(control, GWL_STYLE) & BS_TYPEMASK)",
                "            != BS_AUTOCHECKBOX)",
                '        fluid_memory_error("control is not a checkbox", line, column);',
                "    return IsDlgButtonChecked(window->state->handle, (int)id) == BST_CHECKED;",
                "}",
                "static void fluid_window_set_checkbox_checked(fluid_window *window, long long id,",
                "                                              bool checked, int line, int column)",
                "{",
                "    fluid_window_checkbox_checked(window, id, line, column);",
                "    if (!CheckDlgButton(window->state->handle, (int)id,",
                "                        checked ? BST_CHECKED : BST_UNCHECKED))",
                '        fluid_window_error("CheckDlgButton", line, column);',
                "}",
                "static void fluid_window_set_title(fluid_window *window, const char *title,",
                "                                  int line, int column)",
                "{",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    wchar_t *wide_title = fluid_window_wide(title, line, column);",
                "    BOOL changed = SetWindowTextW(handle, wide_title);",
                "    free(wide_title);",
                "    if (!changed) fluid_window_error(\"SetWindowTextW\", line, column);",
                "}",
                "static void fluid_window_move(fluid_window *window, long long x, long long y,",
                "                             int line, int column)",
                "{",
                "    if (x < LONG_MIN || x > LONG_MAX || y < LONG_MIN || y > LONG_MAX)",
                '        fluid_memory_error("window position is outside the Win32 range", '
                "line, column);",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    if (!SetWindowPos(handle, NULL, (int)x, (int)y, 0, 0,",
                "                     SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE))",
                '        fluid_window_error("SetWindowPos", line, column);',
                "}",
                "static void fluid_window_resize(fluid_window *window, long long width,",
                "                               long long height, int line, int column)",
                "{",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("window dimensions must be positive and fit in an int", '
                "line, column);",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    DWORD style = (DWORD)GetWindowLongPtrW(handle, GWL_STYLE);",
                "    DWORD extended_style = (DWORD)GetWindowLongPtrW(handle, GWL_EXSTYLE);",
                "    RECT frame = {0, 0, (LONG)width, (LONG)height};",
                "    if (!AdjustWindowRectEx(&frame, style, FALSE, extended_style))",
                '        fluid_window_error("AdjustWindowRectEx", line, column);',
                "    if (frame.right <= frame.left || frame.bottom <= frame.top)",
                '        fluid_memory_error("adjusted window dimensions overflow", line, column);',
                "    if (!SetWindowPos(handle, NULL, 0, 0, frame.right - frame.left,",
                "                      frame.bottom - frame.top,",
                "                      SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE))",
                '        fluid_window_error("SetWindowPos", line, column);',
                "}",
                "static long long fluid_window_x(fluid_window *window, int line, int column)",
                "{",
                "    RECT bounds;",
                "    if (!GetWindowRect(fluid_window_require(window, line, column), &bounds))",
                '        fluid_window_error("GetWindowRect", line, column);',
                "    return bounds.left;",
                "}",
                "static long long fluid_window_y(fluid_window *window, int line, int column)",
                "{",
                "    RECT bounds;",
                "    if (!GetWindowRect(fluid_window_require(window, line, column), &bounds))",
                '        fluid_window_error("GetWindowRect", line, column);',
                "    return bounds.top;",
                "}",
                "static long long fluid_window_width(fluid_window *window, int line, int column)",
                "{",
                "    RECT client;",
                "    if (!GetClientRect(fluid_window_require(window, line, column), &client))",
                '        fluid_window_error("GetClientRect", line, column);',
                "    return (long long)(client.right - client.left);",
                "}",
                "static long long fluid_window_height(fluid_window *window, int line, int column)",
                "{",
                "    RECT client;",
                "    if (!GetClientRect(fluid_window_require(window, line, column), &client))",
                '        fluid_window_error("GetClientRect", line, column);',
                "    return (long long)(client.bottom - client.top);",
                "}",
                "static void fluid_window_clear(fluid_window *window, long long color,",
                "                              int line, int column)",
                "{",
                "    fluid_window_state *state = window->state;",
                "    fluid_window_require(window, line, column);",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    RECT client = {0, 0, state->back_width, state->back_height};",
                "    HBRUSH brush = CreateSolidBrush(fluid_window_color(color, line, column));",
                "    if (!brush) fluid_window_error(\"CreateSolidBrush\", line, column);",
                "    int filled = FillRect(state->back_dc, &client, brush);",
                "    BOOL deleted = DeleteObject(brush);",
                "    if (!filled) fluid_window_error(\"FillRect\", line, column);",
                "    if (!deleted) fluid_window_error(\"DeleteObject(brush)\", line, column);",
                "}",
                "static void fluid_window_fill_rect(fluid_window *window, long long x,",
                "                                  long long y, long long width,",
                "                                  long long height, long long color,",
                "                                  int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("rectangle dimensions must be positive", line, column);',
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("rectangle dimensions exceed the Win32 range", '
                "line, column);",
                "    int right = (int)(x + width);",
                "    int bottom = (int)(y + height);",
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    RECT bounds = {left, top, right, bottom};",
                "    HBRUSH brush = CreateSolidBrush(fluid_window_color(color, line, column));",
                "    if (!brush) fluid_window_error(\"CreateSolidBrush\", line, column);",
                "    int filled = FillRect(state->back_dc, &bounds, brush);",
                "    BOOL deleted = DeleteObject(brush);",
                "    if (!filled) fluid_window_error(\"FillRect\", line, column);",
                "    if (!deleted) fluid_window_error(\"DeleteObject(brush)\", line, column);",
                "}",
                "static void fluid_window_fill_ellipse(fluid_window *window, long long x,",
                "                                      long long y, long long width,",
                "                                      long long height, long long color,",
                "                                      int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("ellipse dimensions must be positive", line, column);',
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("ellipse dimensions exceed the Win32 range", '
                "line, column);",
                "    int right = (int)(x + width);",
                "    int bottom = (int)(y + height);",
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    HBRUSH brush = CreateSolidBrush(fluid_window_color(color, line, column));",
                "    if (!brush) fluid_window_error(\"CreateSolidBrush\", line, column);",
                "    int saved = SaveDC(state->back_dc);",
                "    if (!saved) { DeleteObject(brush); fluid_window_error(\"SaveDC\", line, column); }",
                "    HGDIOBJ old_brush = SelectObject(state->back_dc, brush);",
                "    HGDIOBJ old_pen = SelectObject(state->back_dc, GetStockObject(NULL_PEN));",
                "    if (!old_brush || !old_pen || old_brush == HGDI_ERROR || old_pen == HGDI_ERROR)",
                "        fluid_window_error(\"SelectObject\", line, column);",
                "    BOOL drawn = Ellipse(state->back_dc, left, top, right, bottom);",
                "    if (!RestoreDC(state->back_dc, saved)) fluid_window_error(\"RestoreDC\", line, column);",
                "    if (!DeleteObject(brush)) fluid_window_error(\"DeleteObject(brush)\", line, column);",
                "    if (!drawn) fluid_window_error(\"Ellipse\", line, column);",
                "}",
                "static void fluid_window_draw_rect(fluid_window *window, long long x,",
                "                                  long long y, long long width,",
                "                                  long long height, long long color,",
                "                                  int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("invalid rectangle dimensions", line, column);',
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("rectangle dimensions exceed the Win32 range", '
                "line, column);",
                "    int right = (int)(x + width);",
                "    int bottom = (int)(y + height);",
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    HPEN pen = CreatePen(PS_SOLID, 1, fluid_window_color(color, line, column));",
                "    if (!pen) fluid_window_error(\"CreatePen\", line, column);",
                "    int saved = SaveDC(state->back_dc);",
                "    if (!saved) { DeleteObject(pen); fluid_window_error(\"SaveDC\", line, column); }",
                "    HGDIOBJ old_pen = SelectObject(state->back_dc, pen);",
                "    HGDIOBJ old_brush = SelectObject(state->back_dc, GetStockObject(NULL_BRUSH));",
                "    if (!old_pen || !old_brush || old_pen == HGDI_ERROR || old_brush == HGDI_ERROR)",
                "        fluid_window_error(\"SelectObject\", line, column);",
                "    BOOL drawn = Rectangle(state->back_dc, left, top, right, bottom);",
                "    if (!RestoreDC(state->back_dc, saved)) fluid_window_error(\"RestoreDC\", line, column);",
                "    if (!DeleteObject(pen)) fluid_window_error(\"DeleteObject(pen)\", line, column);",
                "    if (!drawn) fluid_window_error(\"Rectangle\", line, column);",
                "}",
                "static void fluid_window_draw_ellipse(fluid_window *window, long long x,",
                "                                      long long y, long long width,",
                "                                      long long height, long long color,",
                "                                      int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("invalid ellipse dimensions", line, column);',
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("ellipse dimensions exceed the Win32 range", '
                "line, column);",
                "    int right = (int)(x + width);",
                "    int bottom = (int)(y + height);",
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    HPEN pen = CreatePen(PS_SOLID, 1, fluid_window_color(color, line, column));",
                "    if (!pen) fluid_window_error(\"CreatePen\", line, column);",
                "    int saved = SaveDC(state->back_dc);",
                "    if (!saved) { DeleteObject(pen); fluid_window_error(\"SaveDC\", line, column); }",
                "    HGDIOBJ old_pen = SelectObject(state->back_dc, pen);",
                "    HGDIOBJ old_brush = SelectObject(state->back_dc, GetStockObject(NULL_BRUSH));",
                "    if (!old_pen || !old_brush || old_pen == HGDI_ERROR || old_brush == HGDI_ERROR)",
                "        fluid_window_error(\"SelectObject\", line, column);",
                "    BOOL drawn = Ellipse(state->back_dc, left, top, right, bottom);",
                "    if (!RestoreDC(state->back_dc, saved)) fluid_window_error(\"RestoreDC\", line, column);",
                "    if (!DeleteObject(pen)) fluid_window_error(\"DeleteObject(pen)\", line, column);",
                "    if (!drawn) fluid_window_error(\"Ellipse\", line, column);",
                "}",
                "static void fluid_window_draw_line(fluid_window *window, long long x1,",
                "                                  long long y1, long long x2, long long y2,",
                "                                  long long color, int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    int start_x = fluid_window_coordinate(x1, line, column);",
                "    int start_y = fluid_window_coordinate(y1, line, column);",
                "    int end_x = fluid_window_coordinate(x2, line, column);",
                "    int end_y = fluid_window_coordinate(y2, line, column);",
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    HPEN pen = CreatePen(PS_SOLID, 1, fluid_window_color(color, line, column));",
                "    if (!pen) fluid_window_error(\"CreatePen\", line, column);",
                "    int saved = SaveDC(state->back_dc);",
                "    if (!saved) { DeleteObject(pen); fluid_window_error(\"SaveDC\", line, column); }",
                "    HGDIOBJ old_pen = SelectObject(state->back_dc, pen);",
                "    if (!old_pen || old_pen == HGDI_ERROR)",
                "        fluid_window_error(\"SelectObject(pen)\", line, column);",
                "    BOOL moved = MoveToEx(state->back_dc, start_x, start_y, NULL);",
                "    BOOL drawn = moved && LineTo(state->back_dc, end_x, end_y);",
                "    if (!RestoreDC(state->back_dc, saved)) fluid_window_error(\"RestoreDC\", line, column);",
                "    if (!DeleteObject(pen)) fluid_window_error(\"DeleteObject(pen)\", line, column);",
                "    if (!drawn) fluid_window_error(\"LineTo\", line, column);",
                "}",
                "static void fluid_window_draw_text(fluid_window *window, long long x,",
                "                                  long long y, const char *text,",
                "                                  long long color, int line, int column)",
                "{",
                "    fluid_window_require(window, line, column);",
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    wchar_t *wide = fluid_window_wide(text, line, column);",
                "    size_t length = wcslen(wide);",
                "    if (length > INT_MAX) { free(wide);",
                '        fluid_memory_error("text is too long to draw", line, column); }',
                "    fluid_window_state *state = window->state;",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) { free(wide); return; }",
                "    int saved = SaveDC(state->back_dc);",
                "    if (!saved) { free(wide); fluid_window_error(\"SaveDC\", line, column); }",
                "    if (SetBkMode(state->back_dc, TRANSPARENT) == 0)",
                '        fluid_window_error("SetBkMode", line, column);',
                "    SetTextColor(state->back_dc, fluid_window_color(color, line, column));",
                "    BOOL drawn = TextOutW(state->back_dc, left, top, wide, (int)length);",
                "    if (!RestoreDC(state->back_dc, saved)) fluid_window_error(\"RestoreDC\", line, column);",
                "    free(wide);",
                "    if (!drawn) fluid_window_error(\"TextOutW\", line, column);",
                "}",
                "static void fluid_window_present(fluid_window *window, int line, int column)",
                "{",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    if (!RedrawWindow(handle, NULL, NULL, RDW_INVALIDATE | RDW_NOERASE",
                "            | RDW_UPDATENOW | RDW_NOCHILDREN))",
                '        fluid_window_error("RedrawWindow", line, column);',
                "}",
                "static void fluid_window_present_rect(fluid_window *window, long long x,",
                "                                      long long y, long long width,",
                "                                      long long height, int line, int column)",
                "{",
                "    HWND handle = fluid_window_require(window, line, column);",
                "    fluid_window_state *state = window->state;",
                "    if (width <= 0 || height <= 0 || width > INT_MAX || height > INT_MAX)",
                '        fluid_memory_error("present rectangle dimensions must be positive", '
                "line, column);",
                "    int left = fluid_window_coordinate(x, line, column);",
                "    int top = fluid_window_coordinate(y, line, column);",
                "    if (x > INT_MAX - width || y > INT_MAX - height)",
                '        fluid_memory_error("present rectangle exceeds the Win32 range", '
                "line, column);",
                "    int right = (int)(x + width);",
                "    int bottom = (int)(y + height);",
                "    if (!fluid_window_ensure_backbuffer(state, line, column)) return;",
                "    if (left < 0) left = 0;",
                "    if (top < 0) top = 0;",
                "    if (right > state->back_width) right = state->back_width;",
                "    if (bottom > state->back_height) bottom = state->back_height;",
                "    if (right <= left || bottom <= top) return;",
                "    HDC target = GetDC(handle);",
                "    if (!target) fluid_window_error(\"GetDC\", line, column);",
                "    BOOL copied = BitBlt(target, left, top, right - left, bottom - top,",
                "                        state->back_dc, left, top, SRCCOPY);",
                "    DWORD error = copied ? ERROR_SUCCESS : GetLastError();",
                "    if (!ReleaseDC(handle, target))",
                '        fluid_window_error("ReleaseDC", line, column);',
                "    if (!copied) {",
                "        SetLastError(error);",
                '        fluid_window_error("BitBlt(region)", line, column);',
                "    }",
                "}",
                "",
            ])
        if self.structs:
            for definition in self.structs:
                output.append(f"struct fluid_{definition.name} {{")
                output.extend(
                    f"    {self.c_type(field_type)} {field_name};"
                    for field_name, field_type in definition.fields
                )
                output.extend(["};", ""])
        for trait in self.traits:
            vtable_type = f"struct fluid_trait_vtable_{trait.name}"
            output.extend([
                f"typedef struct {{ void *data; const {vtable_type} *vtable; }} "
                f"fluid_dyn_{trait.name};",
                f"{vtable_type} {{",
            ])
            for method in trait.methods:
                parameters = [
                    "void *data",
                    *(
                        f"{self.c_type(type_name)} v_{name}"
                        for name, type_name in method.parameters[1:]
                    ),
                ]
                output.append(
                    f"    {self.c_type(method.return_type)} "
                    f"(*{method.name})({', '.join(parameters)});"
                )
            output.extend(["};", ""])
        for (trait_name, type_name), methods in self.trait_impl_map.items():
            trait = self.trait_map[trait_name]
            vtable_type = f"struct fluid_trait_vtable_{trait_name}"
            for method in trait.methods:
                parameters = [
                    "void *data",
                    *(
                        f"{self.c_type(parameter_type)} v_{parameter_name}"
                        for parameter_name, parameter_type in method.parameters[1:]
                    ),
                ]
                dispatch_name = self.trait_dispatch_name(
                    trait_name, type_name, method.name
                )
                output.append(
                    f"static {self.c_type(method.return_type)} {dispatch_name}"
                    f"({', '.join(parameters)});"
                )
            output.append(
                f"static const {vtable_type} "
                f"{self.trait_vtable_name(trait_name, type_name)} = {{"
            )
            output.extend(
                f"    .{method.name} = "
                f"{self.trait_dispatch_name(trait_name, type_name, method.name)},"
                for method in trait.methods
            )
            output.extend(["};", ""])
        prototype_start = len(output)
        for function in self.functions:
            parameters = ", ".join(
                f"{self.c_type(type_name)} v_{name}" for name, type_name in function.parameters
            ) or "void"
            output.append(
                f"{self.c_type(function.return_type)} "
                f"fluid_{function.qualified_name}({parameters});"
            )
        output.append("")
        function_definitions_start = len(output)
        for function in self.functions:
            self.current_module = function.module
            self.return_type = function.return_type
            self.scopes = [{}]
            self.vector_c_names = [{}]
            self.slice_sources = [{}]
            self.moved_vectors = set()
            self.loop_depth = 0
            self.loop_scope_depths = []
            self.range_counter = 0
            self.return_counter = 0
            self.vector_counter = 0
            self.c_name_overrides = {}
            self.parallel_worker_depth = 0
            parameters = ", ".join(
                f"{self.c_type(type_name)} v_{name}" for name, type_name in function.parameters
            ) or "void"
            output.append(
                f"{self.c_type(function.return_type)} "
                f"fluid_{function.qualified_name}({parameters})"
            )
            output.append("{")
            self.indent = 1
            for name, type_name in function.parameters:
                self.scopes[0][name] = (type_name, type_name.startswith("borrow_mut:"))
                element_type = self.vector_type_parts(type_name)
                if element_type is not None and (
                    type_name.startswith("move:") or type_name.startswith("vec:")
                ):
                    owned_type = type_name[5:] if type_name.startswith("move:") else type_name
                    self.scopes[0][name] = (owned_type, True)
                    self.vector_c_names[0][name] = f"v_{name}"
            for statement in function.body:
                output.append(self.emit_statement(statement).rstrip("\n"))
            self.validate_reachable(function.body)
            if function.return_type != "unit" and not self.always_returns(function.body):
                self.fail(
                    f"function {function.name!r} may finish without returning {function.return_type}",
                    function.token,
                )
            if function.return_type == "unit":
                output.append(self.vector_cleanup(0).rstrip("\n"))
                output.append(self.line("return;").rstrip("\n"))
            output.append("}")
            output.append("")
        for (trait_name, type_name), methods in self.trait_impl_map.items():
            trait = self.trait_map[trait_name]
            for method_signature in trait.methods:
                implementation = methods[method_signature.name]
                dispatch_name = self.trait_dispatch_name(
                    trait_name, type_name, method_signature.name
                )
                arguments = [
                    f"(const struct fluid_{type_name} *)data",
                    *(f"v_{name}" for name, _ in method_signature.parameters[1:]),
                ]
                call = f"fluid_{implementation.qualified_name}({', '.join(arguments)})"
                output.append(
                    f"static {self.c_type(method_signature.return_type)} "
                    f"{dispatch_name}("
                    f"void *data"
                    + "".join(
                        f", {self.c_type(parameter_type)} v_{name}"
                        for name, parameter_type in method_signature.parameters[1:]
                    )
                    + ")"
                )
                output.append("{")
                if method_signature.return_type == "unit":
                    output.append(f"    {call};")
                else:
                    output.append(f"    return {call};")
                output.extend(["}", ""])
        prototypes = []
        for function in self.functions:
            parameters = ", ".join(
                f"{self.c_type(type_name)} v_{name}"
                for name, type_name in function.parameters
            ) or "void"
            prototypes.append(
                f"{self.c_type(function.return_type)} "
                f"fluid_{function.qualified_name}({parameters});"
            )
        output[prototype_start:function_definitions_start] = prototypes + [""]
        function_definitions_start = prototype_start + len(prototypes) + 1
        if self.parallel_workers or self.task_workers:
            function_definitions = output[function_definitions_start:]
            output = output[:function_definitions_start]
            for worker in self.parallel_workers:
                output.extend(worker.splitlines())
                output.append("")
            for worker in self.task_workers.values():
                output.extend(self.emit_task_worker(worker))
                output.append("")
            output.extend(function_definitions)
        main_function = self.module_functions[((), "main")]
        if main_function.return_type == "unit":
            output.extend([
                "int main(void)", "{", f"    fluid_{main_function.qualified_name}();",
                "    return 0;", "}",
            ])
        else:
            output.extend([
                "int main(void)", "{",
                f"    return (int)fluid_{main_function.qualified_name}();", "}",
            ])
        return "\n".join(output) + "\n"


def parse_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise FluidError(f"cannot read {path}: {error}") from error


def read_manifest(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "fluid.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FluidError(f"cannot read project manifest {manifest_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise FluidError(
            f"invalid project manifest {manifest_path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error
    if not isinstance(manifest, dict):
        raise FluidError(f"{manifest_path}: the project manifest must be a JSON object")
    project_name = manifest.get("name")
    entry = manifest.get("entry", "src/main.fl")
    dependencies = manifest.get("dependencies", {})
    if not isinstance(project_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]*",
                                                               project_name):
        raise FluidError(f"{manifest_path}: 'name' must be a valid package name")
    if not isinstance(entry, str) or not entry:
        raise FluidError(f"{manifest_path}: 'entry' must be a non-empty source path")
    if not isinstance(dependencies, dict):
        raise FluidError(f"{manifest_path}: 'dependencies' must be an object")
    return manifest


def project_root_for(path: Path) -> tuple[Path, Path] | None:
    resolved = path.resolve()
    if resolved.is_dir():
        root = resolved
    elif resolved.name == "fluid.json":
        root = resolved.parent
    else:
        return None
    manifest = read_manifest(root)
    entry = manifest.get("entry", "src/main.fl")
    entry_path = (root / str(entry)).resolve()
    if not entry_path.is_file():
        raise FluidError(f"project entry source does not exist: {entry_path}")
    return root, entry_path


def resolve_git_dependency(project_root: Path, name: str, spec: dict[str, object]) -> tuple[Path, str]:
    url = spec.get("git")
    revision = spec.get("rev")
    if not isinstance(url, str) or not re.match(r"^(https://|ssh://|git@|file://)", url):
        raise FluidError(f"dependency {name!r}: git URL must use https, ssh, git@, or file:// syntax")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise FluidError(
            f"dependency {name!r}: git dependencies must pin a full commit hash in 'rev'"
        )
    revision = revision.lower()
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    cache_root = project_root / ".fluid" / "deps"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"{name}-{cache_key}-{revision[:12]}"
    if not (destination / ".git").is_dir():
        with tempfile.TemporaryDirectory(prefix="git-dep-", dir=cache_root) as temporary:
            checkout = Path(temporary) / "checkout"
            clone = subprocess.run(
                ["git", "clone", "--quiet", "--no-checkout", "--", url, str(checkout)],
                capture_output=True,
                check=False,
                text=True,
            )
            if clone.returncode:
                raise FluidError(
                    f"could not clone git dependency {name!r} "
                    f"(git exit status {clone.returncode})"
                )
            fetch = subprocess.run(
                ["git", "-C", str(checkout), "fetch", "--quiet", "origin", revision],
                capture_output=True,
                check=False,
                text=True,
            )
            if fetch.returncode:
                raise FluidError(
                    f"could not fetch pinned revision for dependency {name!r} "
                    f"(git exit status {fetch.returncode})"
                )
            checkout_result = subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", revision],
                capture_output=True,
                check=False,
                text=True,
            )
            if checkout_result.returncode:
                raise FluidError(
                    f"could not check out pinned revision for dependency {name!r} "
                    f"(git exit status {checkout_result.returncode})"
                )
            resolved = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True,
                check=False,
                text=True,
            )
            if resolved.returncode or resolved.stdout.strip().lower() != revision:
                raise FluidError(f"git dependency {name!r} did not resolve to its pinned commit")
            shutil.move(str(checkout), str(destination))
    return destination, revision


def dependency_root(project_root: Path, name: str, spec: object) -> tuple[Path, str]:
    if not isinstance(spec, dict):
        raise FluidError(f"dependency {name!r} must be an object with 'path' or 'git'")
    if "path" in spec and len(spec) == 1:
        path = spec["path"]
        if not isinstance(path, str) or not path:
            raise FluidError(f"dependency {name!r}: 'path' must be a non-empty string")
        root = (project_root / path).resolve()
        if not root.is_dir():
            raise FluidError(f"local dependency {name!r} does not exist: {root}")
        return root, path
    if set(spec) == {"git", "rev"}:
        return resolve_git_dependency(project_root, name, spec)
    raise FluidError(f"dependency {name!r} must specify either 'path' or pinned 'git' and 'rev'")


def install_project(path: Path) -> Path:
    project = project_root_for(path)
    if project is None:
        raise FluidError(f"{path} is not a Fluid project directory")
    root, _ = project
    manifest = read_manifest(root)
    dependencies = manifest.get("dependencies", {})
    lock_dependencies: dict[str, dict[str, str]] = {}
    for name, spec in dependencies.items():
        dependency, resolved = dependency_root(root, str(name), spec)
        if isinstance(spec, dict) and "path" in spec:
            lock_dependencies[str(name)] = {"path": str(spec["path"])}
        else:
            lock_dependencies[str(name)] = {
                "git": str(spec["git"]),
                "rev": resolved,
            }
        if not (dependency / "fluid.json").is_file():
            raise FluidError(f"dependency {name!r} is missing fluid.json: {dependency}")
    manifest_bytes = (root / "fluid.json").read_bytes()
    lock = {
        "lockVersion": 1,
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dependencies": lock_dependencies,
    }
    lock_path = root / "fluid.lock"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock_path


def initialize_project(path: Path) -> Path:
    root = path.resolve()
    manifest_path = root / "fluid.json"
    if manifest_path.exists():
        raise FluidError(f"project already exists: {manifest_path}")
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / "src"
    source_dir.mkdir(exist_ok=True)
    package_name = re.sub(r"[^A-Za-z_0-9-]", "-", root.name).strip("-")
    if not package_name or not re.match(r"[A-Za-z_]", package_name):
        package_name = f"fluid-{package_name}" if package_name else "fluid-project"
    manifest = {
        "name": package_name,
        "version": "0.1.0",
        "entry": "src/main.fl",
        "dependencies": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (source_dir / "main.fl").write_text(
        "fn main() -> i64 {\n    return 0;\n}\n", encoding="utf-8"
    )
    return manifest_path


def add_project_dependency(
    project_path: Path, name: str, path: str | None, git_url: str | None, revision: str | None
) -> Path:
    project = project_root_for(project_path)
    if project is None:
        raise FluidError(f"{project_path} is not a Fluid project directory")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise FluidError(f"invalid dependency name {name!r}")
    root, _ = project
    manifest = read_manifest(root)
    dependencies = manifest.get("dependencies", {})
    if name in dependencies:
        raise FluidError(f"dependency {name!r} is already declared")
    if path is not None:
        dependency_path = Path(path)
        resolved = (root / dependency_path).resolve()
        if not resolved.is_dir():
            raise FluidError(f"local dependency path does not exist: {resolved}")
        relative = os.path.relpath(resolved, root).replace(os.sep, "/")
        dependencies[name] = {"path": relative}
    else:
        if git_url is None or revision is None:
            raise FluidError("git dependencies require both --git and --rev")
        # Validate and pin the source before it enters the manifest.
        resolve_git_dependency(root, name, {"git": git_url, "rev": revision})
        dependencies[name] = {"git": git_url, "rev": revision.lower()}
    manifest["dependencies"] = dependencies
    manifest_path = root / "fluid.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_project(path: Path) -> Program:
    project = project_root_for(path)
    if project is None:
        root_path = path.resolve()
        root = Parser(parse_source(root_path)).parse()
        root_dir = root_path.parent
    else:
        root_dir, root_path = project
        root = Parser(parse_source(root_path)).parse()

    modules: dict[tuple[str, ...], tuple[Program, Path, tuple[str, ...]]] = {
        (): (root, root_path.parent, ())
    }
    manifest = read_manifest(root_dir) if project is not None else None
    dependency_specs = manifest.get("dependencies", {}) if manifest else {}
    if dependency_specs:
        for name, spec in dependency_specs.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                raise FluidError(f"invalid dependency name {name!r}")
            dep_root, _ = dependency_root(root_dir, name, spec)
            dep_manifest = read_manifest(dep_root)
            if dep_manifest.get("dependencies"):
                raise FluidError(
                    f"dependency {name!r} declares dependencies; transitive dependencies "
                    "are not supported yet"
                )
            entry = dep_manifest.get("entry", "src/main.fl")
            dep_entry = (dep_root / str(entry)).resolve()
            if not dep_entry.is_file():
                raise FluidError(f"dependency {name!r} entry source does not exist: {dep_entry}")
            module_prefix = (name,)
            dep_program = Parser(parse_source(dep_entry), module_prefix).parse()
            if (dep_program.structs or dep_program.enums or dep_program.traits
                    or dep_program.trait_implementations):
                raise FluidError(
                    f"dependency {name!r} declares types or traits; these are not supported yet"
                )
            modules[module_prefix] = (dep_program, dep_entry.parent, module_prefix)

    pending = list(modules)
    while pending:
        parent_module = pending.pop()
        parent_program, parent_dir, prefix = modules[parent_module]
        for module_name, token in parent_program.modules:
            module_path = parent_module + (module_name,)
            if module_path in modules:
                raise FluidError(
                    f"{token.line}:{token.column}: module {'::'.join(module_path)!r} "
                    "is already loaded"
                )
            relative_module = module_path[len(prefix):]
            module_file = parent_dir.joinpath(*relative_module[:-1], f"{module_name}.fl")
            child = Parser(parse_source(module_file), module_path).parse()
            if (child.structs or child.enums or child.traits
                    or child.trait_implementations):
                raise FluidError(
                    f"{module_file}: module types or traits are not supported yet; "
                    "declare them in the root source file"
                )
            modules[module_path] = (child, parent_dir, prefix)
            pending.append(module_path)

    program = Program()
    program.module_path = ()
    program.project_imports = {}
    for module_path, (module_program, _, prefix) in modules.items():
        imports = module_program.imports
        if prefix:
            imports = [
                Import(
                    imported.module if imported.module[:len(prefix)] == prefix
                    else prefix + imported.module,
                    imported.name,
                    imported.token,
                )
                for imported in imports
            ]
        program.project_imports[module_path] = imports
        program.extend(module_program)
        if not module_path:
            program.structs = module_program.structs
            program.enums = module_program.enums
            program.traits = module_program.traits
            program.trait_implementations = module_program.trait_implementations
    return program


def compiler_command(override: str | None) -> list[str]:
    candidates = [override] if override else ["cc", "clang", "gcc", "cl"]
    for candidate in candidates:
        if not candidate:
            continue
        command = shlex.split(candidate, posix=os.name != "nt")
        executable = command[0]
        resolved = shutil.which(executable)
        if not resolved and os.name == "nt" and not override:
            executable_names = (executable, f"{executable}.exe") if not Path(executable).suffix else (executable,)
            conventional_paths = (
                base / "LLVM" / "bin" / name
                for base in (
                    Path(os.environ.get("ProgramFiles", "C:\\Program Files")),
                    Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")),
                )
                for name in executable_names
            )
            resolved = next((str(path) for path in conventional_paths if path.is_file()), None)
        if resolved:
            command[0] = resolved
            return command
    requested = override or "cc, clang, gcc, or cl"
    raise FluidError(f"native C compiler not found ({requested}); install one or pass --cc")


def native_compile_args(
    command: list[str], source_file: Path, output: Path, native_tune: bool = False,
    parallel: bool = False, windows_api: bool = False,
) -> list[str]:
    compiler_name = Path(command[0]).name.lower()
    is_msvc = compiler_name in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}
    if is_msvc:
        if native_tune:
            raise FluidError("--native-tune is currently supported only with GCC or Clang")
        args = command + [
            "/nologo", "/O2", "/std:c11", "/utf-8",
            f"/Fe:{output.resolve()}", str(source_file),
        ]
        if windows_api:
            args.extend(["/link", "user32.lib", "gdi32.lib"])
        return args
    args = command + ["-O3", "-std=c11"]
    if parallel and os.name != "nt":
        args.append("-pthread")
    if native_tune:
        args.append("-march=native")
    args.extend([str(source_file), "-o", str(output.resolve())])
    if os.name != "nt" or "gcc" in compiler_name:
        args.append("-lm")
    if windows_api:
        if os.name != "nt":
            raise FluidError("Win32 window programs can only be built on Windows")
        args.extend(["-luser32", "-lgdi32"])
    return args


def build_binary(
    c_source: str, output: Path, override: str | None, native_tune: bool = False
) -> None:
    command = compiler_command(override)
    with tempfile.TemporaryDirectory(prefix="fluid-") as directory:
        work = Path(directory)
        source_file = work / "program.c"
        source_file.write_text(c_source, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        parallel = (
            "static void fluid_parallel_for(" in c_source
            or "typedef struct fluid_task_pool {" in c_source
        )
        windows_api = "static fluid_window fluid_window_create(" in c_source
        if windows_api and os.name != "nt":
            raise FluidError("Win32 window programs can only be built on Windows")
        args = native_compile_args(
            command, source_file, output, native_tune, parallel, windows_api
        )
        try:
            subprocess.run(args, check=True, cwd=work)
        except subprocess.CalledProcessError as error:
            raise FluidError(f"native compiler exited with status {error.returncode}") from error
        except OSError as error:
            raise FluidError(f"could not start native compiler: {error}") from error


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluid", description="Fluid starter compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="create a Fluid project")
    init_parser.add_argument("path", type=Path, help="directory for the new project")
    add_parser = subparsers.add_parser("add", help="add a project dependency")
    add_parser.add_argument("name", help="dependency name used in imports")
    source_group = add_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--path", help="local dependency project path")
    source_group.add_argument("--git", help="Git repository URL")
    add_parser.add_argument("--rev", help="full Git commit hash (required with --git)")
    add_parser.add_argument("--project", type=Path, default=Path("."),
                            help="project directory (default: current directory)")
    install_parser = subparsers.add_parser("install", help="resolve dependencies and write fluid.lock")
    install_parser.add_argument("path", type=Path, nargs="?", default=Path("."),
                                help="project directory (default: current directory)")
    for name in ("check", "emit-c", "build", "run"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("source", type=Path,
                               help="Fluid source file or project directory")
        if name in {"emit-c", "build"}:
            subparser.add_argument("-o", "--output", type=Path)
        if name in {"build", "run"}:
            subparser.add_argument("--cc", help="C compiler command (default: cc, clang, gcc, or cl)")
            subparser.add_argument(
                "--native-tune",
                action="store_true",
                help="optimize for the build machine CPU (GCC/Clang; less portable binaries)",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "init":
            manifest_path = initialize_project(args.path)
            print(f"Created {manifest_path.parent}")
            return 0
        if args.command == "add":
            if args.git and not args.rev:
                raise FluidError("git dependencies require --rev with a full commit hash")
            if args.path and args.rev:
                raise FluidError("--rev can only be used with --git")
            manifest_path = add_project_dependency(
                args.project, args.name, args.path, args.git, args.rev
            )
            print(f"Updated {manifest_path}")
            return 0
        if args.command == "install":
            lock_path = install_project(args.path)
            print(f"Resolved dependencies in {lock_path}")
            return 0
        functions = parse_project(args.source)
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
            build_binary(c_source, output, args.cc, args.native_tune)
            print(f"Built {output}")
            return 0
        if args.command == "run":
            with tempfile.TemporaryDirectory(prefix="fluid-run-") as directory:
                output = Path(directory) / ("program.exe" if os.name == "nt" else "program")
                build_binary(c_source, output, args.cc, args.native_tune)
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
