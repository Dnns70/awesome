"""Tests for the Flux parser."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from lexer import Lexer
from parser import Parser, ParseError
from ast_nodes import *


def parse(src: str):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def parse_expr(src: str):
    prog = parse(src.rstrip("\n") + "\n")
    assert len(prog.stmts) == 1
    stmt = prog.stmts[0]
    assert isinstance(stmt, ExprStmt)
    return stmt.expr


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

def test_number_literal():
    node = parse_expr("42\n")
    assert isinstance(node, NumberLiteral)
    assert node.value == 42.0

def test_string_literal():
    node = parse_expr('"hello"\n')
    assert isinstance(node, StringLiteral)
    assert node.value == "hello"

def test_bool_true():
    node = parse_expr("true\n")
    assert isinstance(node, BoolLiteral) and node.value is True

def test_bool_false():
    node = parse_expr("false\n")
    assert isinstance(node, BoolLiteral) and node.value is False

def test_error_literal():
    node = parse_expr("error\n")
    assert isinstance(node, ErrorLiteral)

# ---------------------------------------------------------------------------
# Binary expressions
# ---------------------------------------------------------------------------

def test_addition():
    node = parse_expr("1 + 2\n")
    assert isinstance(node, BinaryExpr)
    assert node.op == "+"

def test_precedence_mul_over_add():
    node = parse_expr("1 + 2 * 3\n")
    assert isinstance(node, BinaryExpr) and node.op == "+"
    assert isinstance(node.right, BinaryExpr) and node.right.op == "*"

def test_comparison():
    node = parse_expr("x == y\n")
    assert isinstance(node, BinaryExpr) and node.op == "=="

def test_logical_and():
    node = parse_expr("a and b\n")
    assert isinstance(node, BinaryExpr) and node.op == "and"

def test_logical_or():
    node = parse_expr("a or b\n")
    assert isinstance(node, BinaryExpr) and node.op == "or"

def test_unary_not():
    node = parse_expr("not x\n")
    assert isinstance(node, UnaryExpr) and node.op == "not"

def test_unary_neg():
    node = parse_expr("-5\n")
    assert isinstance(node, UnaryExpr) and node.op == "-"

# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def test_let_declaration():
    prog = parse("let x = 10\n")
    stmt = prog.stmts[0]
    assert isinstance(stmt, VarDeclaration)
    assert stmt.name == "x"
    assert isinstance(stmt.value, NumberLiteral)

def test_assignment():
    prog = parse("let x = 1\nx = 2\n")
    assert isinstance(prog.stmts[1], Assignment)

def test_fn_declaration():
    src = "fn add(a, b)\n    return a + b\n"
    prog = parse(src)
    fn = prog.stmts[0]
    assert isinstance(fn, FnDeclaration)
    assert fn.name == "add"
    assert fn.params == ["a", "b"]
    assert isinstance(fn.body, Block)

def test_fn_no_params():
    src = "fn greet()\n    return 1\n"
    prog = parse(src)
    fn = prog.stmts[0]
    assert fn.params == []

def test_if_stmt():
    src = "if x\n    y\n"
    prog = parse(src)
    stmt = prog.stmts[0]
    assert isinstance(stmt, IfStmt)
    assert isinstance(stmt.then_block, Block)
    assert stmt.else_block is None

def test_if_else():
    src = "if x\n    a\nelse\n    b\n"
    prog = parse(src)
    stmt = prog.stmts[0]
    assert isinstance(stmt, IfStmt)
    assert stmt.else_block is not None

def test_if_elif_else():
    src = "if x\n    a\nelif y\n    b\nelse\n    c\n"
    prog = parse(src)
    stmt = prog.stmts[0]
    assert len(stmt.elif_clauses) == 1

def test_while_stmt():
    src = "while x\n    y\n"
    prog = parse(src)
    assert isinstance(prog.stmts[0], WhileStmt)

def test_for_stmt():
    src = "for item in items\n    item\n"
    prog = parse(src)
    stmt = prog.stmts[0]
    assert isinstance(stmt, ForStmt)
    assert stmt.var == "item"
    assert isinstance(stmt.iterable, Identifier)

def test_return_with_value():
    src = "fn f()\n    return 42\n"
    prog = parse(src)
    ret = prog.stmts[0].body.stmts[0]
    assert isinstance(ret, ReturnStmt)
    assert isinstance(ret.value, NumberLiteral)

def test_break_continue():
    src = "while true\n    break\n"
    prog = parse(src)
    body = prog.stmts[0].body.stmts[0]
    assert isinstance(body, BreakStmt)

def test_import_stmt():
    prog = parse("import utils.math\n")
    stmt = prog.stmts[0]
    assert isinstance(stmt, ImportStmt)
    assert stmt.path == "utils.math"

# ---------------------------------------------------------------------------
# Call expressions & postfix
# ---------------------------------------------------------------------------

def test_call_no_args():
    node = parse_expr("foo()\n")
    assert isinstance(node, CallExpr)
    assert len(node.args) == 0

def test_call_with_args():
    node = parse_expr("add(1, 2)\n")
    assert isinstance(node, CallExpr)
    assert len(node.args) == 2

def test_index_expr():
    node = parse_expr("arr[0]\n")
    assert isinstance(node, IndexExpr)

def test_member_expr():
    node = parse_expr("obj.name\n")
    assert isinstance(node, MemberExpr)
    assert node.attr == "name"

def test_chained_calls():
    node = parse_expr("f()()\n")
    assert isinstance(node, CallExpr)
    assert isinstance(node.callee, CallExpr)

# ---------------------------------------------------------------------------
# Collection literals
# ---------------------------------------------------------------------------

def test_list_literal():
    node = parse_expr("[1, 2, 3]\n")
    assert isinstance(node, ListLiteral)
    assert len(node.elements) == 3

def test_empty_list():
    node = parse_expr("[]\n")
    assert isinstance(node, ListLiteral) and len(node.elements) == 0

def test_dict_literal():
    node = parse_expr('{"a": 1, "b": 2}\n')
    assert isinstance(node, DictLiteral)
    assert len(node.pairs) == 2

def test_empty_dict():
    node = parse_expr("{}\n")
    assert isinstance(node, DictLiteral) and len(node.pairs) == 0

# ---------------------------------------------------------------------------
# Span tracking
# ---------------------------------------------------------------------------

def test_span_preserved_on_literal():
    node = parse_expr("42\n")
    assert node.span.line == 1
    assert node.span.col == 1

def test_span_second_line():
    prog = parse("a\nb\n")
    second = prog.stmts[1].expr
    assert second.span.line == 2

# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_fn_name_raises():
    with pytest.raises(ParseError):
        parse("fn ()\n    x\n")

def test_unclosed_paren_raises():
    with pytest.raises((ParseError, Exception)):
        parse("foo(1\n")

def test_invalid_assignment_target_raises():
    with pytest.raises(ParseError):
        parse("1 + 2 = 3\n")
