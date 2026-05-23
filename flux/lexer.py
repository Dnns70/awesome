"""Flux DSL Lexer — indentation-based tokenizer with line/column tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class TT(Enum):
    # Literals
    NUMBER   = auto()
    STRING   = auto()
    TRUE     = auto()
    FALSE    = auto()
    ERROR    = auto()   # error literal keyword

    # Identifiers & keywords
    IDENT    = auto()
    LET      = auto()
    IF       = auto()
    ELIF     = auto()
    ELSE     = auto()
    FOR      = auto()
    WHILE    = auto()
    BREAK    = auto()
    CONTINUE = auto()
    RETURN   = auto()
    FN       = auto()
    AND      = auto()
    OR       = auto()
    NOT      = auto()
    IN       = auto()
    IMPORT   = auto()

    # Operators
    PLUS     = auto()
    MINUS    = auto()
    STAR     = auto()
    SLASH    = auto()
    EQ       = auto()   # =
    EQEQ     = auto()   # ==
    NEQ      = auto()   # !=
    LT       = auto()
    GT       = auto()
    LTE      = auto()
    GTE      = auto()

    # Delimiters
    LPAREN   = auto()
    RPAREN   = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE   = auto()
    RBRACE   = auto()
    COMMA    = auto()
    COLON    = auto()
    DOT      = auto()
    ARROW    = auto()   # ->

    # Indentation
    NEWLINE  = auto()
    INDENT   = auto()
    DEDENT   = auto()

    EOF      = auto()


KEYWORDS: dict[str, TT] = {
    "let":      TT.LET,
    "if":       TT.IF,
    "elif":     TT.ELIF,
    "else":     TT.ELSE,
    "for":      TT.FOR,
    "while":    TT.WHILE,
    "break":    TT.BREAK,
    "continue": TT.CONTINUE,
    "return":   TT.RETURN,
    "fn":       TT.FN,
    "and":      TT.AND,
    "or":       TT.OR,
    "not":      TT.NOT,
    "in":       TT.IN,
    "import":   TT.IMPORT,
    "true":     TT.TRUE,
    "false":    TT.FALSE,
    "error":    TT.ERROR,
}


@dataclass(frozen=True)
class Span:
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


@dataclass(frozen=True)
class Token:
    type: TT
    value: object          # raw lexeme value (str, float, or None)
    span: Span

    def __repr__(self) -> str:
        v = f"({self.value!r})" if self.value is not None else ""
        return f"Token({self.type.name}{v} @{self.span})"


class LexError(Exception):
    def __init__(self, msg: str, span: Span):
        super().__init__(f"[{span}] LexError: {msg}")
        self.span = span


class Lexer:
    """Converts Flux source text into a flat token stream.

    Indentation rules:
    - Only spaces are allowed for indentation (tabs raise LexError).
    - An increase in indentation emits INDENT; a decrease emits one or
      more DEDENT tokens (one per level closed).
    - NEWLINE is emitted for logical line endings (blank lines and
      comment-only lines are suppressed).
    """

    def __init__(self, source: str):
        self._src: str = source
        self._pos: int = 0
        self._line: int = 1
        self._line_start: int = 0          # char offset of the current line's first char
        self._indent_stack: List[int] = [0]
        self._tokens: List[Token] = []
        self._pending: List[Token] = []    # INDENT/DEDENT to emit before next real token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> List[Token]:
        self._scan_all()
        return self._tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _col(self) -> int:
        return self._pos - self._line_start + 1

    @property
    def _span(self) -> Span:
        return Span(self._line, self._col)

    def _peek(self, offset: int = 0) -> str:
        idx = self._pos + offset
        return self._src[idx] if idx < len(self._src) else ""

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._line_start = self._pos
        return ch

    def _match(self, expected: str) -> bool:
        if self._peek() == expected:
            self._advance()
            return True
        return False

    def _add(self, tt: TT, value: object = None, span: Optional[Span] = None):
        self._tokens.append(Token(tt, value, span or self._span))

    # ------------------------------------------------------------------
    # Core scan loop
    # ------------------------------------------------------------------

    def _scan_all(self):
        while self._pos < len(self._src):
            self._scan_line()

        # Close any open indentation blocks at EOF
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._add(TT.DEDENT)

        self._add(TT.EOF)

    def _scan_line(self):
        """Scan one logical line, handling indentation at the start."""
        indent_span = self._span
        indent = self._measure_indent()

        # Skip blank / comment-only lines entirely
        if self._at_line_end():
            self._consume_line_end()
            return

        # Emit INDENT / DEDENT before the first token on this line
        current = self._indent_stack[-1]
        if indent > current:
            self._indent_stack.append(indent)
            self._tokens.append(Token(TT.INDENT, None, indent_span))
        elif indent < current:
            while self._indent_stack[-1] > indent:
                self._indent_stack.pop()
                self._tokens.append(Token(TT.DEDENT, None, indent_span))
            if self._indent_stack[-1] != indent:
                raise LexError("inconsistent dedent", indent_span)

        # Scan tokens until end of line
        while not self._at_line_end():
            self._scan_token()

        # Emit NEWLINE then consume the actual newline character
        if self._tokens and self._tokens[-1].type not in (TT.NEWLINE, TT.INDENT):
            self._add(TT.NEWLINE)
        self._consume_line_end()

    def _measure_indent(self) -> int:
        count = 0
        while self._peek() == " ":
            self._advance()
            count += 1
        if self._peek() == "\t":
            raise LexError("tabs are not allowed for indentation; use spaces", self._span)
        return count

    def _at_line_end(self) -> bool:
        ch = self._peek()
        return ch in ("", "\n", "#")

    def _consume_line_end(self):
        # Skip comment
        if self._peek() == "#":
            while self._peek() not in ("", "\n"):
                self._advance()
        # Consume newline
        if self._peek() == "\n":
            self._advance()

    # ------------------------------------------------------------------
    # Token scanners
    # ------------------------------------------------------------------

    def _scan_token(self):
        # Skip inline whitespace
        while self._peek() in (" ", "\t") and not self._at_line_end():
            self._advance()
        if self._at_line_end():
            return

        span = self._span
        ch = self._advance()

        if ch.isdigit() or (ch == "." and self._peek().isdigit()):
            self._scan_number(ch, span)
        elif ch == '"' or ch == "'":
            self._scan_string(ch, span)
        elif ch.isalpha() or ch == "_":
            self._scan_ident(ch, span)
        elif ch == "+":
            self._tokens.append(Token(TT.PLUS, None, span))
        elif ch == "-":
            if self._match(">"):
                self._tokens.append(Token(TT.ARROW, None, span))
            else:
                self._tokens.append(Token(TT.MINUS, None, span))
        elif ch == "*":
            self._tokens.append(Token(TT.STAR, None, span))
        elif ch == "/":
            self._tokens.append(Token(TT.SLASH, None, span))
        elif ch == "=":
            if self._match("="):
                self._tokens.append(Token(TT.EQEQ, None, span))
            else:
                self._tokens.append(Token(TT.EQ, None, span))
        elif ch == "!":
            if self._match("="):
                self._tokens.append(Token(TT.NEQ, None, span))
            else:
                raise LexError(f"unexpected character '!'", span)
        elif ch == "<":
            if self._match("="):
                self._tokens.append(Token(TT.LTE, None, span))
            else:
                self._tokens.append(Token(TT.LT, None, span))
        elif ch == ">":
            if self._match("="):
                self._tokens.append(Token(TT.GTE, None, span))
            else:
                self._tokens.append(Token(TT.GT, None, span))
        elif ch == "(":
            self._tokens.append(Token(TT.LPAREN, None, span))
        elif ch == ")":
            self._tokens.append(Token(TT.RPAREN, None, span))
        elif ch == "[":
            self._tokens.append(Token(TT.LBRACKET, None, span))
        elif ch == "]":
            self._tokens.append(Token(TT.RBRACKET, None, span))
        elif ch == "{":
            self._tokens.append(Token(TT.LBRACE, None, span))
        elif ch == "}":
            self._tokens.append(Token(TT.RBRACE, None, span))
        elif ch == ",":
            self._tokens.append(Token(TT.COMMA, None, span))
        elif ch == ":":
            self._tokens.append(Token(TT.COLON, None, span))
        elif ch == ".":
            self._tokens.append(Token(TT.DOT, None, span))
        else:
            raise LexError(f"unexpected character {ch!r}", span)

    def _scan_number(self, first: str, span: Span):
        buf = [first]
        has_dot = first == "."
        while self._peek().isdigit() or (self._peek() == "." and not has_dot):
            ch = self._advance()
            if ch == ".":
                has_dot = True
            buf.append(ch)
        self._tokens.append(Token(TT.NUMBER, float("".join(buf)), span))

    def _scan_string(self, quote: str, span: Span):
        buf: List[str] = []
        while True:
            ch = self._peek()
            if ch == "" or ch == "\n":
                raise LexError("unterminated string literal", span)
            self._advance()
            if ch == quote:
                break
            if ch == "\\" :
                esc = self._advance()
                buf.append({"n": "\n", "t": "\t", "\\": "\\", quote: quote}.get(esc, esc))
            else:
                buf.append(ch)
        self._tokens.append(Token(TT.STRING, "".join(buf), span))

    def _scan_ident(self, first: str, span: Span):
        buf = [first]
        while self._peek().isalnum() or self._peek() == "_":
            buf.append(self._advance())
        name = "".join(buf)
        tt = KEYWORDS.get(name, TT.IDENT)
        value = name if tt == TT.IDENT else None
        self._tokens.append(Token(tt, value, span))
