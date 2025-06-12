"""Wrapper around the Ollama chat API with SQL-aware routing."""

from __future__ import annotations

import ollama
from sqlglot import parse_one, transpile
from sqlglot import expressions as exp


STORED_PROCEDURE_PROMPT = (
    "You are an expert in SingleStore Helios stored-procedure syntax and "
    "semantics.  Your job is to take the INPUT_PROCEDURE, identify any "
    "language constructs or SQL features that are not supported by SingleStore, "
    "and emit a fully valid SingleStore stored procedure as OUTPUT_PROCEDURE.\n\n"
    "Make these changes automatically:\n"
    "\u2022 Rewrite the header as CREATE OR REPLACE PROCEDURE, include any RETURNS "
    "clause and the appropriate AUTHORIZE AS DEFINER or AUTHORIZE AS CURRENT_USER "
    "clause.\n"
    "\u2022 Declare all local variables with DECLARE and default-value assignments, "
    "remove unsupported modifiers on query-type parameters.\n"
    "\u2022 Replace any multi-row returns with ECHO SELECT to emit a rowset, or use "
    "RETURNS QUERY(...) where appropriate.\n"
    "\u2022 Wrap the procedure body between DELIMITER // and DELIMITER ; so that "
    "semicolons within the body are handled correctly by MySQL clients.\n"
    "\u2022 Strip or refactor any DDL statements that SingleStore does not allow "
    "inside procedures invoked from pipelines.\n"
    "\u2022 Ensure that any unsupported statements (for example EXPLAIN BALANCE or "
    "DDL inside a pipeline) are either removed or rewritten to SingleStore "
    "equivalents.\n"
    "\u2022 Preserve nested DECLARE blocks, %ROWTYPE and %TYPE uses, and loop or "
    "control-flow constructs, translating them into SingleStore’s procedural "
    "extensions.\n"
    "At the end, output the converted procedure code only, formatted with "
    "appropriate DELIMITER commands and SQL fences.\n\n"
    "Here is the procedure to convert.  Replace INPUT_PROCEDURE with the original "
    "routine:\n\nINPUT_PROCEDURE:\n{procedure}\n\nOUTPUT_PROCEDURE:"
)

BREAKDOWN_PROMPT = (
    "List each data manipulation statement (SELECT, INSERT, UPDATE, DELETE, "
    "MERGE) from the following stored procedure. Return one statement per "
    "line in the same order and do not include any additional text.\n\n"
    "STORED_PROCEDURE:\n{procedure}\n\nSTATEMENTS:"
)

REASSEMBLE_PROMPT = (
    STORED_PROCEDURE_PROMPT
    + "\nUse these SingleStore statements for the procedure body:\n{converted}\n"
)


class LLMWrapper:
    """Generate SingleStore SQL using Ollama."""

    def __init__(self, model: str = "llama3") -> None:
        self.model = model

    def _query_type(self, expression: exp.Expression) -> str:
        if isinstance(expression, exp.Create) and ((expression.kind or "").upper() == "PROCEDURE"):
            return "stored procedure"
        if isinstance(expression, exp.Insert):
            return "insert query"
        if isinstance(expression, exp.DDL):
            return "DDL query"
        return "query"

    def _prompt(self, kind: str, sql: str, dialect: str | None) -> str:
        if kind == "stored procedure":
            return STORED_PROCEDURE_PROMPT.format(procedure=sql)
        return f"Double check that the following {kind} is valid SingleStore SQL:\n{sql}"

    def _decompose_procedure(self, sql: str) -> list[str]:
        """Use the LLM to extract DML statements from a stored procedure."""
        prompt = BREAKDOWN_PROMPT.format(procedure=sql)
        messages = [
            {"role": "system", "content": "You extract SQL statements."},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=self.model, messages=messages)
        content = response["message"]["content"]
        return [line.strip() for line in content.splitlines() if line.strip()]

    def _reassemble_procedure(self, original: str, converted: str) -> str:
        """Use the LLM to create a SingleStore procedure from converted DML."""
        prompt = REASSEMBLE_PROMPT.format(procedure=original, converted=converted)
        messages = [
            {"role": "system", "content": "You translate SQL between dialects."},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"].strip()

    def _transpile_procedure(self, sql: str, dialect: str) -> str:
        """Break down, transpile, and rebuild a stored procedure."""
        statements = self._decompose_procedure(sql)
        converted = []
        for stmt in statements:
            text = stmt.rstrip(";")
            try:
                text = transpile(text, read=dialect, write="singlestore")[0]
            except Exception:
                pass  # best effort
            converted.append(text + ";")
        joined = "\n".join(converted)
        return self._reassemble_procedure(sql, joined)

    def to_singlestore(self, sql: str, dialect: str = "tsql") -> str:
        """Translate SQL from the given dialect to SingleStore SQL."""
        try:
            expression = parse_one(sql, read=dialect)
        except Exception as e:  # pragma: no cover - best effort parsing
            raise ValueError("Please provide sql.") from e

        kind = self._query_type(expression)

        if kind == "stored procedure":
            return self._transpile_procedure(sql, dialect)

        transpiled = transpile(sql, read=dialect, write="singlestore")[0]
        prompt = self._prompt(kind, transpiled, None)

        messages = [
            {"role": "system", "content": "You translate SQL between dialects."},
            {"role": "user", "content": prompt},
        ]

        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"].strip()
