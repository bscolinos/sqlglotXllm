"""Helpers for transpiling stored procedures via a local LLM."""

from __future__ import annotations

from typing import Iterable, List

import requests


LLM_URL = "http://localhost:11434/api/generate"
"""Default endpoint for a local LLM (Ollama compatible)."""


def _call_llm(prompt: str, model: str = "llama2", url: str = LLM_URL) -> str:
    """Call a local LLM to process a prompt and return the raw response."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def extract_transpilable_sql(
    procedure: str, model: str = "llama2", url: str = LLM_URL
) -> List[str]:
    """Use a local LLM to extract transpilable SQL statements from a stored procedure."""
    prompt = (
        "Extract the SQL statements from the following stored procedure. "
        "Return the statements separated by a semicolon.\n" + procedure
    )
    raw = _call_llm(prompt, model=model, url=url)
    return [s.strip() for s in raw.split(";") if s.strip()]


def transpile_procedure(
    procedure: str,
    read: str = "tsql",
    write: str = "singlestore",
    model: str = "llama2",
    url: str = LLM_URL,
) -> str:
    """Transpile SQL inside a stored procedure into SingleStore syntax."""
    from sqlglot import transpile

    statements = extract_transpilable_sql(procedure, model=model, url=url)
    converted: Iterable[str] = (
        transpile(stmt, read=read, write=write)[0] for stmt in statements
    )
    body = ";\n".join(converted) + ";"
    return f"DELIMITER //\n{body}\n//\nDELIMITER ;"
