import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_open_source_files_exist() -> None:
    assert (ROOT / "README.md").exists()
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / ".env.example").exists()
    assert (ROOT / "pyproject.toml").exists()


def test_env_file_is_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "!.env.example" in gitignore
    assert ".crewai_storage/" in gitignore


def test_ci_and_dependency_automation_are_configured() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert all(version in ci for version in ('"3.10"', '"3.11"', '"3.12"', '"3.13"'))
    assert "ruff format --check" in ci
    assert "run: mypy" in ci, "the type check must stay wired into CI"
    assert "cache: pip" in ci, "dependency downloads should be cached"
    assert "pip-audit" in ci
    assert "--ignore-vuln PYSEC-2026-311" in ci, "the no-fix advisory exception must stay narrow"
    assert "dependency-review-action" in ci
    actions = re.findall(r"^\s*uses:\s+([^\s#]+)", ci, flags=re.MULTILINE)
    assert actions and all(re.search(r"@[0-9a-f]{40}$", action) for action in actions), (
        "third-party actions must be pinned to immutable commits"
    )

    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert 'versions: [">=3.0.0"]' in dependabot


def test_streamlit_upload_ceiling_is_configured() -> None:
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "maxUploadSize = 50" in config


def test_package_ships_its_type_marker() -> None:
    """Without py.typed the annotations are invisible to consumers."""
    assert (ROOT / "src" / "auto_report_agent" / "py.typed").is_file()


def test_mypy_is_configured_with_its_exclusion_documented() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.mypy]" in pyproject
    assert "crew" in pyproject.split("[tool.mypy]")[1], "the exclusion must stay explicit"
    exclude_line = next(line for line in pyproject.splitlines() if line.startswith("exclude = "))
    pattern = ast.literal_eval(exclude_line.split("=", 1)[1].strip())
    assert re.search(pattern, "src/auto_report_agent/crew.py")
    assert re.search(pattern, r"src\auto_report_agent\crew.py")
