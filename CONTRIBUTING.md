# Contributing — Agentic Honeypot

## Branching
- `main`: Protected. Release-quality only.
- `develop`: Integration branch.
- `feature/<short-name>`: New features.
- `bugfix/<short-name>`: Bug fixes.

## PR Rules
- No direct commits to `main`.
- Every PR must include:
  - What changed
  - How to test it
  - Screenshots/plots if relevant

## Naming Conventions
- Python: `snake_case` for functions/variables, `PascalCase` for classes
- Modules: lowercase with underscores
- Data files: `data/raw/*.jsonl`, `data/processed/*.csv`
- Figures: `report/figures/<descriptive_name>.png`

## Code Standards
- Max line length: 120 characters
- Run `.\scripts\make.ps1 lint` before committing
- Run `.\scripts\make.ps1 test` before pushing