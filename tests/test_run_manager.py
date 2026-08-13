import json

import pytest

from auto_report_agent.run_manager import create_run_paths, write_run_metadata, write_text_atomic


def test_each_run_gets_an_isolated_directory(tmp_path):
    first = create_run_paths(runs_dir=tmp_path)
    second = create_run_paths(runs_dir=tmp_path)

    assert first.run_id != second.run_id
    assert first.directory != second.directory
    assert first.report_file.parent == first.directory
    assert second.report_file.parent == second.directory


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
