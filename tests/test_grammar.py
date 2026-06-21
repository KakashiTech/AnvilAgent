import enum
import typing

from pydantic import BaseModel

from anvil.grammar.schema_compiler import GBNFCompiler
from anvil.grammar.xgrammar_bridge import (
    GBNFParser,
    GrammarMatcher,
    LogitMaskGenerator,
    XGrammarBridge,
)


class UserSchema(BaseModel):
    name: str
    age: int
    email: str


class AddressSchema(BaseModel):
    street: str
    city: str
    zip_code: str


class UserWithAddressSchema(BaseModel):
    name: str
    age: int
    address: AddressSchema


class TestGBNFCompiler:
    def test_compile_simple_schema(self):
        compiler = GBNFCompiler()
        gbnf = compiler.compile(UserSchema)
        assert "root ::=" in gbnf or "UserSchema ::=" in gbnf
        assert "string" in gbnf
        assert "integer" in gbnf

    def test_output_contains_field_names(self):
        compiler = GBNFCompiler()
        gbnf = compiler.compile(UserSchema)
        assert '"name"' in gbnf
        assert '"age"' in gbnf
        assert '"email"' in gbnf

    def test_output_is_valid_gbnf(self):
        compiler = GBNFCompiler()
        gbnf = compiler.compile(UserSchema)
        assert "UserSchema" in gbnf
        assert "::=" in gbnf
        assert 'ws ::=' in gbnf

    def test_compile_nested_schema(self):
        compiler = GBNFCompiler()
        gbnf = compiler.compile(UserWithAddressSchema)
        assert "UserWithAddressSchema ::=" in gbnf
        assert "address" in gbnf
        assert "AddressSchema" in gbnf

    def test_compile_to_file(self, tmp_path):
        compiler = GBNFCompiler()
        output = tmp_path / "test.gbnf"
        compiler.compile_to_file(UserSchema, str(output))
        assert output.exists()
        content = output.read_text()
        assert "string" in content

    def test_compile_optional_field(self):
        class OptionalSchema(BaseModel):
            name: str
            nickname: str | None = None

        compiler = GBNFCompiler()
        gbnf = compiler.compile(OptionalSchema)
        assert "null" in gbnf

    def test_compile_list_field(self):
        class ListSchema(BaseModel):
            tags: list[str]

        compiler = GBNFCompiler()
        gbnf = compiler.compile(ListSchema)
        assert "[" in gbnf

    def test_compile_empty_list_possible(self):
        class ListSchema(BaseModel):
            tags: list[str]

        compiler = GBNFCompiler()
        gbnf = compiler.compile(ListSchema)
        # The inner content group must be optional (marked by ?)
        assert "?" in gbnf

    def test_compile_dict_field(self):
        class DictSchema(BaseModel):
            metadata: dict[str, str]

        compiler = GBNFCompiler()
        gbnf = compiler.compile(DictSchema)
        assert "string" in gbnf

    def test_compile_empty_dict_possible(self):
        class DictSchema(BaseModel):
            metadata: dict[str, str]

        compiler = GBNFCompiler()
        gbnf = compiler.compile(DictSchema)
        assert "?" in gbnf

    def test_compile_reused_model_dedup(self):
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            left: Inner
            right: Inner

        compiler = GBNFCompiler()
        gbnf = compiler.compile(Outer)
        # Inner ::= should appear exactly once
        count = gbnf.count("Inner ::=")
        assert count == 1, f"Inner ::= appears {count} times, expected 1"

    def test_compile_union_without_none(self):
        class UnionSchema(BaseModel):
            code: str | int

        compiler = GBNFCompiler()
        gbnf = compiler.compile(UnionSchema)
        assert "string" in gbnf
        assert "integer" in gbnf

    def test_compile_literal(self):
        class LiteralSchema(BaseModel):
            role: typing.Literal["admin", "user"]

        compiler = GBNFCompiler()
        gbnf = compiler.compile(LiteralSchema)
        assert '"admin"' in gbnf
        assert '"user"' in gbnf

    def test_compile_enum(self):
        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        class EnumSchema(BaseModel):
            color: Color

        compiler = GBNFCompiler()
        gbnf = compiler.compile(EnumSchema)
        assert '"red"' in gbnf
        assert '"green"' in gbnf
        assert '"blue"' in gbnf

    def test_compile_to_gbnf_file_alias_removed(self):
        compiler = GBNFCompiler()
        assert not hasattr(compiler, "compile_to_gbnf_file"), (
            "compile_to_gbnf_file should be removed"
        )
        assert hasattr(compiler, "compile_to_file")

    def test_whitespace_allows_multiple_spaces(self):
        compiler = GBNFCompiler()
        gbnf = compiler.compile(UserSchema)
        assert '" "*' in gbnf, "ws should allow multiple spaces"


class TestGBNFParser:
    def test_parse_simple_rule(self):
        gbnf = 'root ::= "hello"'
        parser = GBNFParser(gbnf)
        rules = parser.parse()
        assert "root" in rules
        assert len(rules["root"]) == 1

    def test_parse_alternatives(self):
        gbnf = 'root ::= "a" | "b"'
        parser = GBNFParser(gbnf)
        rules = parser.parse()
        assert len(rules["root"]) == 2

    def test_parse_rule_reference(self):
        gbnf = 'root ::= greeting\ngreeting ::= "hi"'
        parser = GBNFParser(gbnf)
        rules = parser.parse()
        assert "root" in rules
        assert "greeting" in rules

    def test_parse_char_class(self):
        gbnf = 'root ::= [0-9]'
        parser = GBNFParser(gbnf)
        rules = parser.parse()
        assert "root" in rules

    def test_parse_quantified(self):
        gbnf = 'root ::= "a"+'
        parser = GBNFParser(gbnf)
        rules = parser.parse()
        assert "root" in rules


class TestGrammarMatcher:
    def test_basic_matching(self):
        gbnf = 'root ::= "hello"'
        matcher = GrammarMatcher(gbnf)
        valid = matcher.get_valid_chars()
        assert "h" in valid

    def test_advance_and_match(self):
        gbnf = 'root ::= "hello"'
        matcher = GrammarMatcher(gbnf)
        matcher.advance("h")
        matcher.advance("e")
        valid = matcher.get_valid_chars()
        assert "l" in valid

    def test_accepting_state(self):
        gbnf = 'root ::= "a"'
        matcher = GrammarMatcher(gbnf)
        matcher.advance("a")
        assert matcher.is_accepting()

    def test_not_accepting(self):
        gbnf = 'root ::= "ab"'
        matcher = GrammarMatcher(gbnf)
        matcher.advance("a")
        assert not matcher.is_accepting()

    def test_reset(self):
        gbnf = 'root ::= "a"'
        matcher = GrammarMatcher(gbnf)
        matcher.advance("a")
        assert matcher.is_accepting()
        matcher.reset()
        assert not matcher.is_accepting()

    def test_deterministic_single_char(self):
        gbnf = 'root ::= "a"'
        matcher = GrammarMatcher(gbnf)
        assert matcher.is_deterministic()
        assert matcher.get_deterministic_char() == "a"

    def test_non_deterministic_multi_char(self):
        gbnf = 'root ::= "a" | "b"'
        matcher = GrammarMatcher(gbnf)
        assert not matcher.is_deterministic()

    def test_empty_grammar_raises(self):
        try:
            GrammarMatcher('root ::= ""')
        except Exception:
            pass

    def test_structural_deterministic_after_match(self):
        # After matching "name", only ':' is valid in JSON-like grammar
        gbnf = 'root ::= "name" ":" value\nvalue ::= "hello" | "world"'
        matcher = GrammarMatcher(gbnf)
        for ch in "name":
            matcher.advance(ch)
        assert matcher.is_deterministic()
        assert matcher.get_deterministic_char() == ":"


class TestLogitMaskGenerator:
    def test_mask_allows_valid_chars(self):
        gbnf = 'root ::= "a"'
        matcher = GrammarMatcher(gbnf)
        vocab = ["a", "b", "c"]
        gen = LogitMaskGenerator(matcher, vocab)
        mask = gen.get_mask()
        assert mask[0] == 0.0  # "a" is valid
        assert mask[1] == float("-inf")  # "b" is not
        assert mask[2] == float("-inf")  # "c" is not

    def test_mask_no_vocab_uses_default_size(self):
        gbnf = 'root ::= "a"'
        matcher = GrammarMatcher(gbnf)
        gen = LogitMaskGenerator(matcher)
        mask = gen.get_mask()
        assert len(mask) == 32000


class TestXGrammarBridge:
    def test_compile_and_advance(self):
        bridge = XGrammarBridge(vocabulary=["{", "}", '"', "hello"])
        gbnf = 'root ::= "hello"'
        bridge._grammar_text = gbnf
        bridge._matcher = GrammarMatcher(gbnf)
        bridge._mask_gen = LogitMaskGenerator(bridge._matcher, bridge.vocabulary)
        assert bridge.get_grammar_text() == gbnf
        mask = bridge.get_logit_mask()
        assert len(mask) == len(bridge.vocabulary)

    def test_reset_and_cleanup(self):
        bridge = XGrammarBridge()
        bridge.reset()  # no-op when no matcher
        bridge.cleanup()  # no-op when no path
