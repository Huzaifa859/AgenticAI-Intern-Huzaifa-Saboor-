# demo1 — logic bugs for LLM analysis

Small Python package with **intentional semantic bugs** that static tools
(pyflakes / AST checks) typically miss: wrong comparisons, off-by-one,
auth logic holes, stale cache keys, float money, path traversal, etc.

Use this with the Code Analysis agent, e.g. question:

> Find bugs and potential issues

Do not use this as a template for production code.
