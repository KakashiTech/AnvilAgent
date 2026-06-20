import enum
import typing
from typing import Union, get_args, get_origin

from pydantic import BaseModel


class GBNFCompiler:
    def __init__(self):
        self._compiled: set[str] = set()

    def compile(self, model: type[BaseModel]) -> str:
        self._compiled = set()
        rules = []
        root_name = model.__name__
        rules.append(self._compile_model(model, root_name))
        rules.append(self._compile_whitespace())
        rules.append(self._compile_primitives())
        return "\n".join(rules)

    def compile_from_dict(self, schema: dict, name: str = "root") -> str:
        self._compiled = set()
        rules = []
        rules.append(self._dict_to_rule(schema, name))
        rules.append(self._compile_whitespace())
        rules.append(self._compile_primitives())
        return "\n".join(rules)

    def _dict_to_rule(self, schema: dict, name: str) -> str:
        if name in self._compiled:
            return ""
        self._compiled.add(name)

        schema_type = schema.get("type", "object")
        enum_vals = schema.get("enum")

        if enum_vals is not None:
            alts = " | ".join(f'"{v}"' for v in enum_vals)
            return f'{name} ::= ({alts})'

        if schema_type == "object":
            properties = schema.get("properties", {})
            if not properties:
                return f'{name} ::= "{"{"} ws "{"}"}'

            field_rules = []
            sub_rules = []
            for i, (prop_name, prop_schema) in enumerate(properties.items()):
                separator = "" if i == 0 else ' "," '
                prop_type = prop_schema.get("type", "object")
                enum_vals = prop_schema.get("enum")

                if prop_type == "object" or prop_type == "array" or enum_vals is not None:
                    sub_name = f"{name}_{prop_name}"
                    field_rule = f'{separator} ws "\\"" "{prop_name}" "\\"" ws ":" ws {sub_name}'
                    sub_rules.extend(self._compile_dict_field_type(prop_schema, sub_name))
                else:
                    prim = self._dict_type_to_prim(prop_type, prop_schema)
                    field_rule = f'{separator} ws "\\"" "{prop_name}" "\\"" ws ":" ws {prim}'

                field_rules.append(field_rule)

            sub_rules_str = ""
            if sub_rules:
                sub_rules_str = "\n" + "\n".join(sub_rules)

            body = " ".join(field_rules) if field_rules else "ws"
            return f'{name} ::= "{{" {body} "}}" {sub_rules_str}'

        if schema_type == "array":
            item_name = f"{name}_item"
            return f'{name} ::= "[" ws ({item_name} ("," ws {item_name})*)? ws "]"'

        return f'{name} ::= {self._dict_type_to_prim(schema_type, schema)}'

    def _compile_dict_field_type(self, schema: dict, name: str) -> list[str]:
        schema_type = schema.get("type", "object")
        enum_vals = schema.get("enum")
        if schema_type == "object":
            return [self._dict_to_rule(schema, name)]
        if schema_type == "array":
            items = schema.get("items", {})
            item_name = f"{name}_item"
            return self._compile_dict_field_type(items, item_name)
        if enum_vals is not None:
            return [self._dict_to_rule(schema, name)]
        return []

    @staticmethod
    def _dict_type_to_prim(schema_type: str, schema: dict) -> str:
        mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "null": "null",
        }
        return mapping.get(schema_type, "string")

    def _compile_model(self, model: type[BaseModel], name: str) -> str:
        fields = {}
        for field_name, field_info in model.model_fields.items():
            fields[field_name] = field_info.annotation

        self._compiled.add(name)
        rule = f'{name} ::= "{{" {self._compile_fields(fields)} "}}"'
        sub_rules = []
        for field_name, field_type in fields.items():
            sub_rules.extend(self._compile_field_type(field_type, f"{name}_{field_name}"))
        if sub_rules:
            rule += "\n" + "\n".join(sub_rules)
        return rule

    def _compile_fields(self, fields: dict) -> str:
        if not fields:
            return "ws"
        field_rules = []
        for i, (fname, ftype) in enumerate(fields.items()):
            separator = "" if i == 0 else ' "," '
            rule_ref = self._type_to_rule(ftype, fname)
            field_rule = f'{separator} ws "\\"" "{fname}" "\\"" ws ":" ws {rule_ref}'
            field_rules.append(field_rule)
        return " ".join(field_rules)

    def _compile_field_type(self, t: type, name: str) -> list[str]:
        origin = get_origin(t)
        args = get_args(t)

        if origin is Union and type(None) in args:
            actual = [a for a in args if a is not type(None)][0]
            return self._compile_field_type(actual, name)

        if origin is Union:
            rules = []
            for i, arg in enumerate(args):
                rules.extend(self._compile_field_type(arg, f"{name}_alt{i}"))
            return rules

        if origin is list and args:
            return self._compile_field_type(args[0], f"{name}_item")

        if origin is dict and args:
            return self._compile_field_type(args[1], f"{name}_val")

        if origin is typing.Literal:
            return []

        if hasattr(t, 'model_fields'):
            if t.__name__ in self._compiled:
                return []
            return [self._compile_model(t, t.__name__)]

        if isinstance(t, type) and issubclass(t, enum.Enum):
            return []

        return []

    def _type_to_rule(self, t: type, name: str) -> str:
        origin = get_origin(t)
        args = get_args(t)

        if origin is Union and type(None) in args:
            actual = [a for a in args if a is not type(None)][0]
            inner = self._type_to_rule(actual, name)
            return f"({inner} | null)"

        if origin is Union:
            alts = " | ".join(
                self._type_to_rule(arg, f"{name}_alt{i}") for i, arg in enumerate(args)
            )
            return f"({alts})"

        if origin is list and args:
            inner = args[0]
            inner_rule = self._type_to_rule(inner, f"{name}_item")
            return f'"[" ws ({inner_rule} ("," ws {inner_rule})*)? ws "]"'

        if origin is dict and args:
            _, val_type = args
            val_rule = self._type_to_rule(val_type, f"{name}_val")
            return (
                f'"{{" ws (string ws ":" ws {val_rule}'
                f' ("," ws string ws ":" ws {val_rule})*)? ws "}}"'
            )

        if origin is typing.Literal:
            alts = " | ".join(self._literal_value(v) for v in args)
            return f"({alts})"

        if t is str:
            return "string"
        if t is int:
            return "integer"
        if t is float:
            return "number"
        if t is bool:
            return "boolean"

        if isinstance(t, type) and issubclass(t, enum.Enum):
            alts = " | ".join(f'"{e.value}"' for e in t)
            return f"({alts})"

        if hasattr(t, 'model_fields'):
            return t.__name__

        return "string"

    @staticmethod
    def _literal_value(v) -> str:
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    @staticmethod
    def _compile_whitespace() -> str:
        return 'ws ::= " "*'

    @staticmethod
    def _compile_primitives() -> str:
        return (
            'string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]'
            ' [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""\n'
            'integer ::= ("-"? [0-9]+)\n'
            'number ::= ("-"? ([0-9] | [1-9] [0-9]*))'
            ' ("." [0-9]+)? ([eE] [-+]? [0-9]+)?\n'
            'boolean ::= "true" | "false"\n'
            'null ::= "null"\n'
        )

    def compile_to_file(self, model: type[BaseModel], output_path: str):
        gbnf = self.compile(model)
        with open(output_path, 'w') as f:
            f.write(gbnf)
        return gbnf
