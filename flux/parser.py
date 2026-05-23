"""Flux DSL Parser — converts a token stream into an AST."""

from __future__ import annotations
from typing import List, Optional, Any
from lexer import Token, TT, Span
from ast_nodes import *


class ParseError(Exception):
    def __init__(self, msg: str, span: Span):
        super().__init__(f"[{span}] ParseError: {msg}")
        self.span = span


class Parser:
    """Recursive-descent parser for Flux.

    Grammar overview (simplified):
        program     := stmt* EOF
        stmt        := fn_decl | let_decl | if_stmt | while_stmt
                     | for_stmt | return_stmt | break_stmt
                     | continue_stmt | import_stmt | assign_or_expr
        block       := NEWLINE INDENT stmt+ DEDENT
        expr        := or_expr
        or_expr     := and_expr ("or" and_expr)*
        and_expr    := not_expr ("and" not_expr)*
        not_expr    := "not" not_expr | cmp_expr
        cmp_expr    := add_expr (("==" | "!=" | "<" | ">" | "<=" | ">=") add_expr)?
        add_expr    := mul_expr (("+"|"-") mul_expr)*
        mul_expr    := unary (("*"|"/") unary)*
        unary       := "-" unary | postfix
        postfix     := primary (call_args | "[" expr "]" | "." IDENT)*
        primary     := NUMBER | STRING | "true" | "false" | "error"
                     | IDENT | "(" expr ")" | list_literal | dict_literal
    """

    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        span = self._peek().span
        stmts = []
        while not self._check(TT.EOF):
            self._skip_newlines()
            if self._check(TT.EOF):
                break
            stmts.append(self._stmt())
        return Program(span, stmts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]  # EOF sentinel
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _check(self, *types: TT) -> bool:
        return self._peek().type in types

    def _match(self, *types: TT) -> Optional[Token]:
        if self._peek().type in types:
            return self._advance()
        return None

    def _expect(self, tt: TT, msg: str = "") -> Token:
        if self._peek().type == tt:
            return self._advance()
        tok = self._peek()
        raise ParseError(msg or f"expected {tt.name}, got {tok.type.name}", tok.span)

    def _skip_newlines(self):
        while self._check(TT.NEWLINE):
            self._advance()

    def _error(self, msg: str) -> ParseError:
        return ParseError(msg, self._peek().span)

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _block(self) -> Block:
        span = self._peek().span
        self._expect(TT.NEWLINE, "expected newline before block")
        self._expect(TT.INDENT, "expected indented block")
        stmts = []
        while not self._check(TT.DEDENT, TT.EOF):
            self._skip_newlines()
            if self._check(TT.DEDENT, TT.EOF):
                break
            stmts.append(self._stmt())
        self._expect(TT.DEDENT, "expected dedent after block")
        return Block(span, stmts)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _stmt(self) -> Any:
        tok = self._peek()

        if tok.type == TT.FN:
            return self._fn_decl()
        if tok.type == TT.LET:
            return self._let_decl()
        if tok.type == TT.IF:
            return self._if_stmt()
        if tok.type == TT.WHILE:
            return self._while_stmt()
        if tok.type == TT.FOR:
            return self._for_stmt()
        if tok.type == TT.RETURN:
            return self._return_stmt()
        if tok.type == TT.BREAK:
            self._advance()
            self._expect(TT.NEWLINE, "expected newline after break")
            return BreakStmt(tok.span)
        if tok.type == TT.CONTINUE:
            self._advance()
            self._expect(TT.NEWLINE, "expected newline after continue")
            return ContinueStmt(tok.span)
        if tok.type == TT.IMPORT:
            return self._import_stmt()

        return self._assign_or_expr_stmt()

    def _fn_decl(self) -> FnDeclaration:
        span = self._peek().span
        self._expect(TT.FN)
        name_tok = self._expect(TT.IDENT, "expected function name")
        self._expect(TT.LPAREN, "expected '(' after function name")
        params: List[str] = []
        if not self._check(TT.RPAREN):
            params.append(self._expect(TT.IDENT, "expected parameter name").value)
            while self._match(TT.COMMA):
                params.append(self._expect(TT.IDENT, "expected parameter name").value)
        self._expect(TT.RPAREN, "expected ')'")
        body = self._block()
        return FnDeclaration(span, name_tok.value, params, body)

    def _let_decl(self) -> VarDeclaration:
        span = self._peek().span
        self._expect(TT.LET)
        name_tok = self._expect(TT.IDENT, "expected variable name")
        self._expect(TT.EQ, "expected '=' in let declaration")
        value = self._expr()
        self._expect(TT.NEWLINE, "expected newline after declaration")
        return VarDeclaration(span, name_tok.value, value)

    def _if_stmt(self) -> IfStmt:
        span = self._peek().span
        self._expect(TT.IF)
        cond = self._expr()
        then_block = self._block()

        elif_clauses = []
        while self._check(TT.ELIF):
            self._advance()
            elif_cond = self._expr()
            elif_block = self._block()
            elif_clauses.append((elif_cond, elif_block))

        else_block = None
        if self._check(TT.ELSE):
            self._advance()
            else_block = self._block()

        return IfStmt(span, cond, then_block, elif_clauses, else_block)

    def _while_stmt(self) -> WhileStmt:
        span = self._peek().span
        self._expect(TT.WHILE)
        cond = self._expr()
        body = self._block()
        return WhileStmt(span, cond, body)

    def _for_stmt(self) -> ForStmt:
        span = self._peek().span
        self._expect(TT.FOR)
        var_tok = self._expect(TT.IDENT, "expected loop variable")
        self._expect(TT.IN, "expected 'in' in for loop")
        iterable = self._expr()
        body = self._block()
        return ForStmt(span, var_tok.value, iterable, body)

    def _return_stmt(self) -> ReturnStmt:
        span = self._peek().span
        self._expect(TT.RETURN)
        value = None
        if not self._check(TT.NEWLINE, TT.EOF):
            value = self._expr()
        self._expect(TT.NEWLINE, "expected newline after return")
        return ReturnStmt(span, value)

    def _import_stmt(self) -> ImportStmt:
        span = self._peek().span
        self._expect(TT.IMPORT)
        parts = [self._expect(TT.IDENT, "expected module name").value]
        while self._match(TT.DOT):
            parts.append(self._expect(TT.IDENT, "expected module name").value)
        self._expect(TT.NEWLINE, "expected newline after import")
        return ImportStmt(span, ".".join(parts))

    def _assign_or_expr_stmt(self) -> Any:
        span = self._peek().span
        expr = self._expr()

        if self._match(TT.EQ):
            # Assignment: target = value
            if not isinstance(expr, (Identifier, MemberExpr, IndexExpr)):
                raise ParseError("invalid assignment target", span)
            value = self._expr()
            self._expect(TT.NEWLINE, "expected newline after assignment")
            return Assignment(span, expr, value)

        self._expect(TT.NEWLINE, "expected newline after expression")
        return ExprStmt(span, expr)

    # ------------------------------------------------------------------
    # Expressions  (precedence climbing via recursive descent)
    # ------------------------------------------------------------------

    def _expr(self) -> Any:
        return self._or_expr()

    def _or_expr(self) -> Any:
        left = self._and_expr()
        while self._check(TT.OR):
            span = self._peek().span
            self._advance()
            right = self._and_expr()
            left = BinaryExpr(span, "or", left, right)
        return left

    def _and_expr(self) -> Any:
        left = self._not_expr()
        while self._check(TT.AND):
            span = self._peek().span
            self._advance()
            right = self._not_expr()
            left = BinaryExpr(span, "and", left, right)
        return left

    def _not_expr(self) -> Any:
        if self._check(TT.NOT):
            span = self._peek().span
            self._advance()
            return UnaryExpr(span, "not", self._not_expr())
        return self._cmp_expr()

    _CMP_OPS = {TT.EQEQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE}
    _CMP_STR = {TT.EQEQ: "==", TT.NEQ: "!=", TT.LT: "<",
                TT.GT: ">", TT.LTE: "<=", TT.GTE: ">="}

    def _cmp_expr(self) -> Any:
        left = self._add_expr()
        if self._peek().type in self._CMP_OPS:
            op_tok = self._advance()
            right = self._add_expr()
            return BinaryExpr(op_tok.span, self._CMP_STR[op_tok.type], left, right)
        return left

    def _add_expr(self) -> Any:
        left = self._mul_expr()
        while self._check(TT.PLUS, TT.MINUS):
            op_tok = self._advance()
            op = "+" if op_tok.type == TT.PLUS else "-"
            right = self._mul_expr()
            left = BinaryExpr(op_tok.span, op, left, right)
        return left

    def _mul_expr(self) -> Any:
        left = self._unary()
        while self._check(TT.STAR, TT.SLASH):
            op_tok = self._advance()
            op = "*" if op_tok.type == TT.STAR else "/"
            right = self._unary()
            left = BinaryExpr(op_tok.span, op, left, right)
        return left

    def _unary(self) -> Any:
        if self._check(TT.MINUS):
            span = self._peek().span
            self._advance()
            return UnaryExpr(span, "-", self._unary())
        return self._postfix()

    def _postfix(self) -> Any:
        node = self._primary()
        while True:
            if self._check(TT.LPAREN):
                span = self._peek().span
                self._advance()
                args = []
                if not self._check(TT.RPAREN):
                    args.append(self._expr())
                    while self._match(TT.COMMA):
                        args.append(self._expr())
                self._expect(TT.RPAREN, "expected ')' after arguments")
                node = CallExpr(span, node, args)
            elif self._check(TT.LBRACKET):
                span = self._peek().span
                self._advance()
                index = self._expr()
                self._expect(TT.RBRACKET, "expected ']'")
                node = IndexExpr(span, node, index)
            elif self._check(TT.DOT):
                span = self._peek().span
                self._advance()
                attr_tok = self._expect(TT.IDENT, "expected attribute name")
                node = MemberExpr(span, node, attr_tok.value)
            else:
                break
        return node

    def _primary(self) -> Any:
        tok = self._peek()

        if tok.type == TT.NUMBER:
            self._advance()
            return NumberLiteral(tok.span, tok.value)

        if tok.type == TT.STRING:
            self._advance()
            return StringLiteral(tok.span, tok.value)

        if tok.type == TT.TRUE:
            self._advance()
            return BoolLiteral(tok.span, True)

        if tok.type == TT.FALSE:
            self._advance()
            return BoolLiteral(tok.span, False)

        if tok.type == TT.ERROR:
            self._advance()
            return ErrorLiteral(tok.span)

        if tok.type == TT.IDENT:
            self._advance()
            return Identifier(tok.span, tok.value)

        if tok.type == TT.LPAREN:
            self._advance()
            expr = self._expr()
            self._expect(TT.RPAREN, "expected ')'")
            return expr

        if tok.type == TT.LBRACKET:
            return self._list_literal()

        if tok.type == TT.LBRACE:
            return self._dict_literal()

        raise ParseError(f"unexpected token {tok.type.name!r}", tok.span)

    def _list_literal(self) -> ListLiteral:
        span = self._peek().span
        self._expect(TT.LBRACKET)
        elements = []
        if not self._check(TT.RBRACKET):
            elements.append(self._expr())
            while self._match(TT.COMMA):
                if self._check(TT.RBRACKET):
                    break
                elements.append(self._expr())
        self._expect(TT.RBRACKET, "expected ']'")
        return ListLiteral(span, elements)

    def _dict_literal(self) -> DictLiteral:
        span = self._peek().span
        self._expect(TT.LBRACE)
        pairs = []
        if not self._check(TT.RBRACE):
            key = self._expr()
            self._expect(TT.COLON, "expected ':' in dict literal")
            val = self._expr()
            pairs.append((key, val))
            while self._match(TT.COMMA):
                if self._check(TT.RBRACE):
                    break
                key = self._expr()
                self._expect(TT.COLON, "expected ':' in dict literal")
                val = self._expr()
                pairs.append((key, val))
        self._expect(TT.RBRACE, "expected '}'")
        return DictLiteral(span, pairs)
