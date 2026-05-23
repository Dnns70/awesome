"""Tests for the Flux lexer."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from lexer import Lexer, TT, LexError


def lex(src: str):
    return Lexer(src).tokenize()


def types(src: str):
    return [t.type for t in lex(src)]


def values(src: str):
    return [(t.type, t.value) for t in lex(src)]


# ---------------------------------------------------------------------------
# Basic tokens
# ---------------------------------------------------------------------------

def test_empty_source():
    toks = lex("")
    assert toks[-1].type == TT.EOF

def test_number_integer():
    toks = lex("42\n")
    assert toks[0].type == TT.NUMBER
    assert toks[0].value == 42.0

def test_number_float():
    toks = lex("3.14\n")
    assert toks[0].value == pytest.approx(3.14)

def test_string_double_quote():
    toks = lex('"hello"\n')
    assert toks[0].type == TT.STRING
    assert toks[0].value == "hello"

def test_string_single_quote():
    toks = lex("'world'\n")
    assert toks[0].type == TT.STRING
    assert toks[0].value == "world"

def test_string_escape_sequences():
    toks = lex('"line1\\nline2"\n')
    assert toks[0].value == "line1\nline2"

def test_boolean_tokens():
    ts = types("true\n")
    assert TT.TRUE in ts
    ts = types("false\n")
    assert TT.FALSE in ts

def test_keywords():
    for kw in ("let", "if", "elif", "else", "for", "while", "fn", "return",
               "break", "continue", "and", "or", "not", "in", "import"):
        toks = lex(f"{kw}\n")
        # should NOT be IDENT
        assert toks[0].type != TT.IDENT, f"{kw!r} was lexed as IDENT"

def test_identifier():
    toks = lex("myVar\n")
    assert toks[0].type == TT.IDENT
    assert toks[0].value == "myVar"

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def test_operators():
    src = "+ - * / = == != < > <= >= ->\n"
    ts = types(src)
    expected = [TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH,
                TT.EQ, TT.EQEQ, TT.NEQ,
                TT.LT, TT.GT, TT.LTE, TT.GTE, TT.ARROW,
                TT.NEWLINE, TT.EOF]
    assert ts == expected

# ---------------------------------------------------------------------------
# Line and column tracking
# ---------------------------------------------------------------------------

def test_span_line_col():
    toks = lex("x\ny\n")
    x_tok = toks[0]
    y_tok = toks[2]   # after NEWLINE
    assert x_tok.span.line == 1 and x_tok.span.col == 1
    assert y_tok.span.line == 2 and y_tok.span.col == 1

def test_column_within_line():
    toks = lex("a b\n")
    assert toks[0].span.col == 1
    assert toks[1].span.col == 3

# ---------------------------------------------------------------------------
# Indentation
# ---------------------------------------------------------------------------

def test_indent_dedent():
    src = "if x\n    y\n"
    ts = types(src)
    assert TT.INDENT in ts
    assert TT.DEDENT in ts

def test_nested_indent():
    src = (
        "if x\n"
        "    if y\n"
        "        z\n"
    )
    ts = types(src)
    assert ts.count(TT.INDENT) == 2
    assert ts.count(TT.DEDENT) == 2

def test_multiple_dedents():
    src = (
        "if x\n"
        "    if y\n"
        "        z\n"
        "a\n"
    )
    ts = types(src)
    assert ts.count(TT.DEDENT) == 2

def test_blank_lines_suppressed():
    src = "a\n\n\nb\n"
    ts = types(src)
    # blank lines should not produce extra NEWLINEs or INDENT/DEDENT
    assert ts.count(TT.INDENT) == 0
    assert ts.count(TT.DEDENT) == 0

def test_comment_suppressed():
    src = "a\n# this is a comment\nb\n"
    ts = types(src)
    assert ts.count(TT.IDENT) == 2   # only a and b

def test_tab_indentation_raises():
    with pytest.raises(LexError):
        lex("if x\n\ty\n")

def test_inconsistent_dedent_raises():
    with pytest.raises(LexError):
        lex("if x\n    y\n  z\n")   # dedent to a level never pushed

# ---------------------------------------------------------------------------
# Unterminated string
# ---------------------------------------------------------------------------

def test_unterminated_string_raises():
    with pytest.raises(LexError):
        lex('"oops\n')

# ---------------------------------------------------------------------------
# Delimiters & punctuation
# ---------------------------------------------------------------------------

def test_delimiters():
    src = "( ) [ ] { } , : .\n"
    ts = types(src)
    for tt in (TT.LPAREN, TT.RPAREN, TT.LBRACKET, TT.RBRACKET,
               TT.LBRACE, TT.RBRACE, TT.COMMA, TT.COLON, TT.DOT):
        assert tt in ts
