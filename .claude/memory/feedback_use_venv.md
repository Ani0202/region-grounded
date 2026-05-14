---
name: feedback-use-venv
description: Always use the .venv in the project root for Python commands, not system Python
metadata:
  type: feedback
---

Use `.venv/bin/python` (or `.venv/bin/pip`, `.venv/bin/jupyter`, etc.) for all Python commands in this project.

**Why:** The user created a project-local `.venv` and expects all tooling to run inside it.

**How to apply:** Prefix every Python/pip/jupyter invocation with `.venv/bin/` rather than calling `python3`/`python`/`pip` directly.
