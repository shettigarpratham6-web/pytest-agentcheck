# Contributing to AgentCheck

Thank you for contributing to AgentCheck!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shettigarpratham6-web/pytest-agentcheck.git
   cd pytest-agentcheck
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in editable mode with all development dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

## Development Guidelines

- **Code Style**:
  - Source files inside `src/pytest_agentcheck/` follow a strict **zero-docstring and zero-comment** rule. Write clean, self-documenting code with descriptive function, variable, and class names.
  - Line lengths must strictly fit within 88 characters.
  - Run `ruff check src tests` and `ruff format --check src tests`.

- **Type Safety**:
  - All source files must be fully typed and pass `mypy --strict src`.

- **Testing**:
  - Core modules must maintain at least 85% unit test coverage.
  - Run the full test suite:
    ```bash
    pytest -v --cov=pytest_agentcheck --cov-report=term-missing
    ```
  - Use `pytester` for any new plugin hooks or CLI features.
  - All tests must run offline with mock objects; no live network or paid API keys required.

## Pull Request Process

1. Ensure all linters, type checks, and tests pass locally:
   ```bash
   ruff check src tests
   mypy --strict src
   pytest
   ```
2. Open a Pull Request describing your changes and link any related issues.
