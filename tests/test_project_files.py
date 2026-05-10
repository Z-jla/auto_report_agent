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
