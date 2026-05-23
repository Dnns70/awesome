"""Flux DSL AST node definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any
from lexer import Span


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Node:
    span: Span


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class NumberLiteral(Node):
    value: float


@dataclass
class StringLiteral(Node):
    value: str


@dataclass
class BoolLiteral(Node):
    value: bool


@dataclass
class ErrorLiteral(Node):
    """The bare `error` keyword used as a value (e.g. `return error`)."""


@dataclass
class Identifier(Node):
    name: str


@dataclass
class BinaryExpr(Node):
    op: str          # "+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">=", "and", "or"
    left: Any        # Node
    right: Any       # Node


@dataclass
class UnaryExpr(Node):
    op: str          # "not", "-"
    operand: Any     # Node


@dataclass
class CallExpr(Node):
    callee: Any           # Node (Identifier or member access)
    args: List[Any]


@dataclass
class IndexExpr(Node):
    obj: Any
    index: Any


@dataclass
class MemberExpr(Node):
    obj: Any
    attr: str


@dataclass
class ListLiteral(Node):
    elements: List[Any]


@dataclass
class DictLiteral(Node):
    pairs: List[tuple]   # list of (key_node, value_node)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class VarDeclaration(Node):
    name: str
    value: Any           # expression node


@dataclass
class Assignment(Node):
    target: Any          # Identifier or MemberExpr or IndexExpr
    value: Any


@dataclass
class ReturnStmt(Node):
    value: Optional[Any]


@dataclass
class BreakStmt(Node):
    pass


@dataclass
class ContinueStmt(Node):
    pass


@dataclass
class ExprStmt(Node):
    expr: Any


@dataclass
class Block(Node):
    stmts: List[Any]


@dataclass
class IfStmt(Node):
    condition: Any
    then_block: Block
    elif_clauses: List[tuple]   # list of (condition, Block)
    else_block: Optional[Block]


@dataclass
class WhileStmt(Node):
    condition: Any
    body: Block


@dataclass
class ForStmt(Node):
    var: str
    iterable: Any
    body: Block


@dataclass
class FnDeclaration(Node):
    name: str
    params: List[str]
    body: Block


@dataclass
class ImportStmt(Node):
    path: str            # dotted module path, e.g. "utils.math"


@dataclass
class Program(Node):
    stmts: List[Any]
