import enum
import typing

from pydantic import BaseModel

from anvil.grammar.constrained_decoder import GrammarMatcher
from anvil.grammar.schema_compiler import GBNFCompiler


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


class TestGrammarMatcher:
    def test_deterministic_detection(self):
        matcher = GrammarMatcher("")
        assert matcher.is_deterministic('{"name":')
        assert not matcher.is_deterministic('{"name": "john')

    def test_deterministic_colon(self):
        matcher = GrammarMatcher("")
        token = matcher.get_deterministic_token('{"name"')
        assert token == ':'

    def test_deterministic_space_after_key(self):
        matcher = GrammarMatcher("")
        token = matcher.get_deterministic_token('{"name":')
        assert token == ' '

    def test_deterministic_newline_after_comma(self):
        matcher = GrammarMatcher("")
        token = matcher.get_deterministic_token('"age": 30,')
        assert token == '\n'

    def test_deterministic_structural_markers(self):
        matcher = GrammarMatcher("")
        assert matcher.is_deterministic('{')
        assert matcher.is_deterministic('}')
        assert matcher.is_deterministic('[')
        assert matcher.is_deterministic(']')
        assert matcher.is_deterministic(':')
        assert matcher.is_deterministic(',')
        assert matcher.is_deterministic(' ')

    def test_non_deterministic_values(self):
        matcher = GrammarMatcher("")
        assert not matcher.is_deterministic('"hello')
        assert not matcher.is_deterministic('42')

    def test_allowed_tokens(self):
        matcher = GrammarMatcher("")
        tokens = matcher.get_allowed_tokens("")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
