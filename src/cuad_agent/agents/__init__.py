"""Framework-specific CUAD question agents.

This package holds the per-category agent layer — the only framework-specific
piece of the evaluation pipeline. Everything around it (sampling, RAG context,
caching, dashboards, CLI) is framework-neutral and lives elsewhere.

- ``langchain_agent`` — LangChain chain + ``CuadAnswer`` Pydantic schema.
- ``dspy_agent`` — DSPy signature/module factories.

The ``evaluators.langchain_runner`` and ``evaluators.dspy_runner`` modules
re-import these names so historical import sites keep working.
"""
