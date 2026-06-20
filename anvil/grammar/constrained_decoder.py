

class GrammarMatcher:
    """
    Matcher de gramática GBNF para constrained decoding.
    En producción usaríamos XGrammar o llguidance como backend.
    Esta es una implementación didáctica del concepto.
    """

    def __init__(self, grammar: str):
        self.grammar = grammar
        self._state: list[str] = []

    def get_allowed_tokens(self, current_output: str) -> list[int]:
        """
        Retorna lista de IDs de tokens permitidos en el paso actual.
        En producción esto llama a XGrammar/llguidance.
        """
        return list(range(128))

    def is_deterministic(self, current_output: str) -> bool:
        """
        Determina si el próximo token es determinista
        (ej: cerrar JSON, coma estructural).
        Si es True, podemos saltar el sampling y emitir directamente.
        """
        structural_markers = ['"', '{', '}', '[', ']', ':', ',', ' ']
        return any(current_output.endswith(m) for m in structural_markers)

    def get_deterministic_token(self, current_output: str) -> str | None:
        """Retorna el token determinista si aplica."""
        if current_output.endswith('":'):
            return ' '
        if current_output.endswith('"'):
            return ':'
        if current_output.endswith(','):
            return '\n'
        return None
