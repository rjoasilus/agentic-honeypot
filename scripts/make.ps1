#Project Task Runner (PowerShell replacement for Makefile)
#Usage: .\scripts\make.ps1
#Commands: run, test, lint, init-db, clean

param([string]$Target = "help")

switch ($Target) {
    "run" {
        # Boot the FastAPI server with hot-reload
        uvicorn api.app.main:app --reload
    }
    "test" {
        # Run all pytest tests with verbose output
        pytest api/tests/ -v
    }
    "lint" {
        # Check code style across all source directories
        flake8 api/ agents/ features/ models/ --max-line-length=120
    }
    "init-db" {
        # Create SQLite tables (safe to run multiple times)
        python -m api.app.db
    }
    "clean" {
        # Delete generated data files (database + telemetry log)
        Remove-Item -Force -ErrorAction SilentlyContinue data\agentic_honeypot.sqlite3
        Remove-Item -Force -ErrorAction SilentlyContinue data\agentic_honeypot.sqlite3-shm
        Remove-Item -Force -ErrorAction SilentlyContinue data\agentic_honeypot.sqlite3-wal
        Remove-Item -Force -ErrorAction SilentlyContinue data\raw\telemetry.jsonl
        Write-Host "Cleaned database and log files."
    }
    default {
        Write-Host "Usage: .\scripts\make.ps1 [run|test|lint|init-db|clean]"
        Write-Host ""
        Write-Host "  run       Start the FastAPI server with hot-reload"
        Write-Host "  test      Run all tests"
        Write-Host "  lint      Check code style with flake8"
        Write-Host "  init-db   Create SQLite tables"
        Write-Host "  clean     Delete database and log files"
    }
}