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
    assert "pip-audit" in ci
    assert "dependency-review-action" in ci
    assert (ROOT / ".github" / "dependabot.yml").is_file()


def test_streamlit_upload_ceiling_is_configured() -> None:
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "maxUploadSize = 50" in config
