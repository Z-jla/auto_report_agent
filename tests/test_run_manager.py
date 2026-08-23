import json
from pathlib import Path

import pytest

from auto_report_agent.run_manager import (
    create_run_paths,
    discard_intermediate_draft,
    handle_failed_report_files,
    write_run_metadata,
    write_text_atomic,
)


def test_each_run_gets_an_isolated_directory(tmp_path):
    first = create_run_paths(runs_dir=tmp_path)
    second = create_run_paths(runs_dir=tmp_path)

    assert first.run_id != second.run_id
    assert first.directory != second.directory
    assert first.report_file.parent == first.directory
    assert second.report_file.parent == second.directory
    assert first.draft_file == first.directory / "draft_report.md"


def test_run_id_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        create_run_paths(run_id="../../escape", runs_dir=tmp_path)


def test_atomic_report_and_metadata_writes(tmp_path):
    paths = create_run_paths(run_id="safe_run_123", runs_dir=tmp_path)
    write_text_atomic(paths.report_file, "report")
    write_run_metadata(paths, {"topic": "topic"})

    assert paths.report_file.read_text(encoding="utf-8") == "report"
    assert json.loads(paths.metadata_file.read_text(encoding="utf-8")) == {
        "run_id": "safe_run_123",
        "topic": "topic",
    }


def test_failed_run_keeps_nonempty_final_and_draft_untouched(tmp_path):
    paths = create_run_paths(run_id="failed_run_123", runs_dir=tmp_path)
    paths.draft_file.write_text("complete writer draft", encoding="utf-8")
    paths.report_file.write_text("complete reviewed final", encoding="utf-8")

    preserved = handle_failed_report_files(paths, keep_draft=True)

    assert preserved == paths.report_file
    assert paths.report_file.read_text(encoding="utf-8") == "complete reviewed final"
    assert paths.draft_file.read_text(encoding="utf-8") == "complete writer draft"


def test_failed_run_never_overwrites_an_unreadable_existing_final(monkeypatch, tmp_path):
    paths = create_run_paths(run_id="locked_final_123", runs_dir=tmp_path)
    paths.draft_file.write_text("complete writer draft", encoding="utf-8")
    paths.report_file.write_text("complete reviewed final", encoding="utf-8")
    real_read_text = Path.read_text

    def fail_for_final(path, *args, **kwargs):
        if path == paths.report_file:
            raise PermissionError("final is temporarily locked")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_for_final)

    preserved = handle_failed_report_files(paths, keep_draft=True)

    assert preserved == paths.draft_file
    assert paths.report_file.read_bytes() == b"complete reviewed final"
    assert paths.draft_file.read_bytes() == b"complete writer draft"


@pytest.mark.parametrize("existing_final", [None, "   \n"])
def test_failed_writer_draft_is_promoted_when_final_is_unusable(tmp_path, existing_final):
    paths = create_run_paths(run_id="promote_run_123", runs_dir=tmp_path)
    paths.draft_file.write_text("complete writer draft", encoding="utf-8")
    if existing_final is not None:
        paths.report_file.write_text(existing_final, encoding="utf-8")

    preserved = handle_failed_report_files(paths, keep_draft=True)

    assert preserved == paths.report_file
    assert paths.report_file.read_text(encoding="utf-8") == "complete writer draft"
    assert not paths.draft_file.exists()


def test_disabled_failure_retention_removes_draft_and_partial_final(tmp_path):
    paths = create_run_paths(run_id="private_run_123", runs_dir=tmp_path)
    paths.draft_file.write_text("draft", encoding="utf-8")
    paths.report_file.write_text("partial", encoding="utf-8")

    assert handle_failed_report_files(paths, keep_draft=False) is None
    assert not paths.draft_file.exists()
    assert not paths.report_file.exists()


def test_success_cleanup_removes_only_the_intermediate_draft(tmp_path):
    paths = create_run_paths(run_id="success_run_123", runs_dir=tmp_path)
    paths.draft_file.write_text("draft", encoding="utf-8")
    paths.report_file.write_text("reviewed", encoding="utf-8")

    discard_intermediate_draft(paths)

    assert not paths.draft_file.exists()
    assert paths.report_file.read_text(encoding="utf-8") == "reviewed"
