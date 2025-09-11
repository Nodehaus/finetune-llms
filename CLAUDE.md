# Development Guidelines

This document contains critical information about working with this codebase.
Follow these guidelines precisely.

## Python Package Management with uv

Use uv exclusively for Python package management in this project.

### Package Management Commands

-   All Python dependencies **must be installed, synchronized, and locked** using
    uv
-   Always use the latest version of a package
-   Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

-   Install dependencies: `uv add <package>`
-   Remove dependencies: `uv remove <package>`
-   Sync dependencies: `uv sync`

### Running Python Code

-   Run a Python script with `uv run <script-name>.py`
-   Run Python tools like ruff with `uv run ruff`
-   Launch a Python repl with `uv run python`

## Linting and code formatting

-   Use ruff to format files and for linting

## Code Quality

-   Type hints required for all code
-   Public APIs must have docstrings
-   Functions must be focused and small
-   Follow existing patterns exactly
-   Catch exception only on few lines of code where it makes sense, do not wrap
    whole blocks of code in `try`/`except`

## Code Style

-   PEP 8 naming (snake_case for functions/variables)
-   Class names in PascalCase
-   Constants in UPPER_SNAKE_CASE
-   Document with docstrings
-   Use f-strings for formatting

## Additional rules

-   Do not create empty **init**.py files
-   Always use absolute imports on the application level
