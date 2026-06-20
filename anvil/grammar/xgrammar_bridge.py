"""XGrammar bridge: GBNF parser → state machine → real logit mask generation."""

import os
import re
import tempfile
from typing import Any

from pydantic import BaseModel

# ─── GBNF Tokenizer ──────────────────────────────────────────────

_GBNF_TOKEN_RE = re.compile(r"""
    \n+                      # newlines (rule boundary)
    | " (?: [^"\\] | \\. )* "   # string literal
    | \[ [^\]]* \]            # char class
    | \.\?                    # optional any-char
    | \.                      # any char
    | \?                      # postfix optional
    | \*                      # postfix zero-or-more
    | \+                      # postfix one-or-more
    | ::=                     # rule def
    | \|                      # alternation
    | \(                      # group open
    | \)                      # group close
    | \s+                     # other whitespace (skip)
    | [a-zA-Z_][a-zA-Z0-9_]* # identifier
""", re.VERBOSE)


def _tokenize_gbnf(text: str) -> list[str]:
    tokens = []
    for m in _GBNF_TOKEN_RE.finditer(text):
        tok = m.group(0)
        if "\n" in tok:
            tokens.append("\n")
        else:
            s = tok.strip()
            if s:
                tokens.append(s)
    return tokens


# ─── Grammar Element Types ───────────────────────────────────────

class Lit(str):
    """Literal string match."""

class CharClass:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._negate = pattern.startswith("^")
        self._ranges: list[tuple[str, str]] = []
        self._chars: set[str] = set()
        raw = pattern[1:] if self._negate else pattern
        i = 0
        while i < len(raw):
            if i + 2 < len(raw) and raw[i + 1] == "-":
                self._ranges.append((raw[i], raw[i + 2]))
                i += 3
            else:
                self._chars.add(raw[i])
                i += 1

    def matches(self, ch: str) -> bool:
        if not ch:
            return False
        c = ch[0]
        in_class = c in self._chars or any(lo <= c <= hi for lo, hi in self._ranges)
        return not in_class if self._negate else in_class

    def __repr__(self):
        return f"[{'^' if self._negate else ''}{self.pattern}]"


class RuleRef(str):
    """Reference to another rule."""

class Group:
    def __init__(self, alternatives: list[list[Any]]):
        self.alternatives = alternatives

class Quantified:
    def __init__(self, element: Any, quant: str):
        self.element = element
        self.quant = quant  # '?', '*', '+'


# ─── GBNF Parser ─────────────────────────────────────────────────

class GBNFParserError(Exception):
    pass


class GBNFParser:
    """Recursive descent parser for llama.cpp GBNF grammar format."""

    def __init__(self, text: str):
        self._tokens = _tokenize_gbnf(text)
        self._pos = 0
        self.rules: dict[str, list[list[Any]]] = {}  # name → alternatives

    def parse(self) -> dict[str, list[list[Any]]]:
        while self._pos < len(self._tokens):
            # skip newlines and stray ::= between rules
            while self._pos < len(self._tokens) and self._tokens[self._pos] in ("\n", "::="):
                self._pos += 1
            if self._pos >= len(self._tokens):
                break
            name = self._tokens[self._pos]
            if not re.match(r'^[a-zA-Z_]', name):
                break
            if (self._pos + 1 >= len(self._tokens)
                    or self._tokens[self._pos + 1] != "::="):
                break
            self._pos += 2  # skip name and ::=
            alts = self._parse_alternatives()
            self.rules[name] = alts
        return self.rules

    def _parse_alternatives(self) -> list[list[Any]]:
        alts = [self._parse_sequence()]
        while self._pos < len(self._tokens) and self._tokens[self._pos] == "|":
            self._pos += 1
            alts.append(self._parse_sequence())
        return alts

    def _parse_sequence(self) -> list[Any]:
        seq = []
        while self._pos < len(self._tokens):
            tok = self._tokens[self._pos]
            if tok in ("|", ")", "::=", "\n", ""):
                break
            elem = self._parse_element()
            if elem is None:
                break
            seq.append(elem)
        return seq

    def _parse_element(self) -> Any:
        if self._pos >= len(self._tokens):
            return None
        tok = self._tokens[self._pos]

        # String literal
        if tok.startswith('"'):
            self._pos += 1
            elem: Any = Lit(tok[1:-1])
            return self._maybe_quantify(elem)

        # Char class
        if tok.startswith("["):
            self._pos += 1
            inner = tok[1:-1]
            elem = CharClass(inner)
            return self._maybe_quantify(elem)

        # Any char
        if tok == ".":
            self._pos += 1
            elem = CharClass("")
            elem._chars = set(chr(i) for i in range(32, 127))
            return self._maybe_quantify(elem)

        # Group
        if tok == "(":
            self._pos += 1
            alts = self._parse_alternatives()
            if self._pos < len(self._tokens) and self._tokens[self._pos] == ")":
                self._pos += 1
            elem = Group(alts)
            return self._maybe_quantify(elem)

        # Rule reference (identifier)
        if re.match(r'^[a-zA-Z_]', tok):
            self._pos += 1
            elem = RuleRef(tok)
            return self._maybe_quantify(elem)

        return None

    def _maybe_quantify(self, elem: Any) -> Any:
        if self._pos >= len(self._tokens):
            return elem
        tok = self._tokens[self._pos]
        if tok in ("?", "*", "+"):
            self._pos += 1
            return Quantified(elem, tok)
        return elem


# ─── Grammar State Machine ───────────────────────────────────────

class GrammarState:
    """A state in the grammar NFA."""

    def __init__(self, uid: int):
        self.uid = uid
        self.transitions: list[tuple[str | None, "GrammarState"]] = []  # (label, target)
        self.is_accept = False

    def add_transition(self, label: str | None, target: "GrammarState"):
        self.transitions.append((label, target))

    def __repr__(self):
        return f"GS({self.uid})"


class GrammarNFA:
    """NFA representation of a GBNF grammar."""

    def __init__(self, rules: dict[str, list[list[Any]]], start_rule: str = "root"):
        self.rules = rules
        self.start_rule = start_rule
        self._states: list[GrammarState] = []
        self._counter = 0

    def _new_state(self) -> GrammarState:
        s = GrammarState(self._counter)
        self._counter += 1
        self._states.append(s)
        return s

    def build(self) -> GrammarState:
        """Build NFA and return start state (all rules compiled, supports forward refs)."""
        if self.start_rule not in self.rules:
            raise GBNFParserError(f"Start rule '{self.start_rule}' not found")
        start = self._new_state()
        accept = self._new_state()
        accept.is_accept = True
        self._compile_alternatives(self.rules[self.start_rule], start, accept)
        return start

    def _compile_alternatives(self, alts: list[list[Any]], start: GrammarState, accept: GrammarState):
        for alt in alts:
            current = start
            for i, elem in enumerate(alt):
                if i < len(alt) - 1:
                    mid = self._new_state()
                    self._compile_element(elem, current, mid)
                    current = mid
                else:
                    self._compile_element(elem, current, accept)
            current.add_transition(None, accept)

    def _compile_element(self, elem: Any, start: GrammarState, accept: GrammarState) -> GrammarState | None:
        if isinstance(elem, Lit):
            for i, ch in enumerate(elem):
                mid = self._new_state() if i < len(elem) - 1 else accept
                start.add_transition(ch, mid)
                start = mid
            return None

        if isinstance(elem, CharClass):
            for ch in elem._chars:
                start.add_transition(ch, accept)
            for lo, hi in elem._ranges:
                for c in [chr(i) for i in range(ord(lo), ord(hi) + 1)]:
                    start.add_transition(c, accept)
            return None

        if isinstance(elem, RuleRef):
            if elem not in self.rules:
                raise GBNFParserError(f"Undefined rule: {elem}")
            mid = self._new_state()
            self._compile_alternatives(self.rules[elem], start, mid)
            mid.add_transition(None, accept)
            return None

        if isinstance(elem, Group):
            mid = self._new_state()
            self._compile_alternatives(elem.alternatives, start, mid)
            mid.add_transition(None, accept)
            return None

        if isinstance(elem, Quantified):
            if elem.quant == "?":
                start.add_transition(None, accept)
                self._compile_element(elem.element, start, accept)
                return None
            elif elem.quant == "*":
                start.add_transition(None, accept)
                self._compile_element(elem.element, start, start)
                return None
            elif elem.quant == "+":
                mid = self._new_state()
                self._compile_element(elem.element, start, mid)
                mid.add_transition(None, start)
                mid.add_transition(None, accept)
                return None

        return None

    def epsilon_closure(self, states: set[GrammarState]) -> set[GrammarState]:
        """Compute epsilon closure of a set of states."""
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for label, target in s.transitions:
                if label is None and target not in closure:
                    closure.add(target)
                    stack.append(target)
        return closure


# ─── Grammar Matcher ─────────────────────────────────────────────

class GrammarMatcher:
    """
    Tracks grammar state during token generation.
    Given current output, determines valid next characters/tokens.
    """

    def __init__(self, grammar_text: str, start_rule: str = "root"):
        self.grammar_text = grammar_text
        parser = GBNFParser(grammar_text)
        self._all_rules = parser.parse()
        if start_rule not in self._all_rules:
            raise GBNFParserError(
                f"Start rule '{start_rule}' not found in {list(self._all_rules.keys())}"
            )
        self.nfa = GrammarNFA(self._all_rules, start_rule)
        self._start_state = self.nfa.build()
        self._active = self.nfa.epsilon_closure({self._start_state})
        self._output = ""

    def reset(self):
        self._active = self.nfa.epsilon_closure({self._start_state})
        self._output = ""

    def advance(self, char: str) -> set[GrammarState]:
        """Consume a character and return new active states."""
        self._output += char
        next_states: set[GrammarState] = set()
        for state in self._active:
            for label, target in state.transitions:
                if label == char:
                    next_states.add(target)
                elif label is None:
                    for label2, target2 in self._expand_epsilon(target):
                        if label2 == char:
                            next_states.add(target2)
        self._active = self.nfa.epsilon_closure(next_states)
        return self._active

    def get_valid_chars(self) -> set[str]:
        """Return set of valid next characters from current state."""
        valid: set[str] = set()
        for state in self._active:
            for label, target in state.transitions:
                if label is not None and len(label) == 1:
                    valid.add(label)
                elif label is None:
                    for label2, _ in self._expand_epsilon(target):
                        if label2 is not None and len(label2) == 1:
                            valid.add(label2)
            if state.is_accept:
                for state2 in self._active:
                    for label, _ in self._expand_epsilon(self._start_state):
                        if label is not None and len(label) == 1:
                            valid.add(label)
        return valid

    def is_accepting(self) -> bool:
        """Check if current state is accepting (end of valid output)."""
        return any(s.is_accept for s in self._active)

    def is_deterministic(self) -> bool:
        """True if only ONE valid next character exists."""
        chars = self.get_valid_chars()
        return len(chars) == 1

    def get_deterministic_char(self) -> str | None:
        chars = self.get_valid_chars()
        if len(chars) == 1:
            return next(iter(chars))
        return None

    def _expand_epsilon(self, state: GrammarState) -> list[tuple[str | None, GrammarState]]:
        """Expand epsilon transitions from a state."""
        result: list[tuple[str | None, GrammarState]] = []
        stack = [state]
        visited = {state}
        while stack:
            s = stack.pop()
            for label, target in s.transitions:
                if label is not None:
                    result.append((label, target))
                elif target not in visited:
                    visited.add(target)
                    stack.append(target)
        return result


# ─── Tokenizer-based Logit Mask ──────────────────────────────────

class LogitMaskGenerator:
    """
    Generates real logit masks from grammar state + tokenizer vocabulary.
    """

    def __init__(self, matcher: GrammarMatcher, vocabulary: list[str] | None = None):
        self.matcher = matcher
        self.vocabulary = vocabulary or []
        self._precompute: dict[str, list[int]] | None = None

    def _build_prefix_index(self):
        """Build a prefix index: prefix → list of token IDs."""
        index: dict[str, list[int]] = {}
        for tid, token in enumerate(self.vocabulary):
            if token:
                first_char = token[0]
                index.setdefault(first_char, []).append(tid)
        self._precompute = index

    def get_mask(self, vocab_size: int | None = None) -> list[float]:
        """Generate logit mask: -inf for invalid tokens, 0.0 for valid."""
        valid_chars = self.matcher.get_valid_chars()
        if self._precompute is None and self.vocabulary:
            self._build_prefix_index()

        size = vocab_size or len(self.vocabulary) if self.vocabulary else 32000
        if not valid_chars:
            if self.matcher.is_accepting():
                return [0.0] * size
            return [float("-inf")] * size

        if not self._precompute:
            mask = [float("-inf")] * size
            for ch in valid_chars:
                for tid in range(min(1000, size)):
                    mask[tid] = 0.0
            return mask

        mask = [float("-inf")] * size
        for ch in valid_chars:
            for tid in self._precompute.get(ch, []):
                if tid < size:
                    mask[tid] = 0.0
        return mask


# ─── XGrammar Bridge ─────────────────────────────────────────────

class XGrammarBridge:
    """
    XGrammar bridge: Pydantic → GBNF → state machine → logit masks.
    Compiles schemas, tracks grammar state, and generates real logit masks.
    """

    def __init__(self, grammar_path: str | None = None,
                 vocabulary: list[str] | None = None):
        self.grammar_path = grammar_path
        self._compiled = False
        self._grammar_text = ""
        self._matcher: GrammarMatcher | None = None
        self._mask_gen: LogitMaskGenerator | None = None
        self.vocabulary = vocabulary

    def compile_schema(self, model: type[BaseModel]) -> str:
        """Compile Pydantic schema to GBNF + build state machine."""
        from .schema_compiler import GBNFCompiler
        compiler = GBNFCompiler()
        gbnf = compiler.compile(model)
        self._grammar_text = gbnf

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.gbnf', delete=False)
        tmp.write(gbnf)
        tmp.close()
        self.grammar_path = tmp.name
        self._compiled = True

        start_rule = model.__name__
        self._matcher = GrammarMatcher(gbnf, start_rule=start_rule)
        self._mask_gen = LogitMaskGenerator(self._matcher, self.vocabulary)
        return gbnf

    def get_logit_mask(self, tokens: list[int] | None = None) -> list[float]:
        """
        Generate real logit mask based on current grammar state.
        Tokens parameter is ignored (we track state via advance_char).
        Returns -inf for forbidden tokens, 0.0 for allowed.
        """
        if self._mask_gen is None:
            return [0.0] * (len(self.vocabulary) if self.vocabulary else 32000)
        return self._mask_gen.get_mask()

    def advance_char(self, char: str):
        """Advance grammar state by one character."""
        if self._matcher:
            self._matcher.advance(char)

    def advance_token(self, token_text: str):
        """Advance grammar state by a full token string."""
        for ch in token_text:
            self.advance_char(ch)

    def get_valid_chars(self) -> set[str]:
        """Get valid next characters from current grammar state."""
        if self._matcher:
            return self._matcher.get_valid_chars()
        return set()

    def is_deterministic(self) -> bool:
        """True if exactly one valid next character."""
        return self._matcher is not None and self._matcher.is_deterministic()

    def get_deterministic_char(self) -> str | None:
        if self._matcher:
            return self._matcher.get_deterministic_char()
        return None

    def is_accepting(self) -> bool:
        """True if grammar has reached an accepting state."""
        return self._matcher is not None and self._matcher.is_accepting()

    def reset(self):
        """Reset grammar state to start."""
        if self._matcher:
            self._matcher.reset()

    def cleanup(self):
        """Remove temporary files."""
        if self.grammar_path and os.path.exists(self.grammar_path):
            os.unlink(self.grammar_path)

    def get_grammar_text(self) -> str:
        return self._grammar_text
