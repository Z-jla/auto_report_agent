"""Tests for the CLI argument handling.

Only the argument/prompt layer is exercised here: generate_report drives CrewAI
and would need a live model. The point of the split is that everything deciding
*what* to run is reachable without one.
"""

from types import SimpleNamespace

import pytest

from auto_report_agent import main as main_module
from auto_report_agent.main import (
    DEFAULT_PAPER_TOPIC,
    CliOptions,
    build_parser,
    generate_report,
    resolve_options,
)
from auto_report_agent.run_manager import create_run_paths
from auto_report_agent.settings import MAX_INSTRUCTION_CHARS, MAX_TOPIC_CHARS


def _parse(*argv: str):
    return build_parser().parse_args(list(argv))


def _resolve(*argv: str, answers: list[str] | None = None):
    """Resolve options, feeding queued answers to any interactive prompt."""
    queue = list(answers or [])

    def prompt(_question: str) -> str:
        if not queue:
            raise AssertionError("the CLI asked more questions than the test queued")
        return queue.pop(0)

    options = resolve_options(_parse(*argv), prompt=prompt)
    assert not queue, f"unused answers: {queue}"
    return options


# --- flags only (no prompting) -----------------------------------------------


def test_topic_mode_from_flags():
    options = _resolve("--mode", "topic", "--topic", "AI 研究进展")

    assert options.mode == "topic"
    assert options.topic == "AI 研究进展"
    assert options.file_paths == []


def test_paper_mode_from_flags():
    options = _resolve(
        "--mode", "paper", "--file", "a.pdf", "--file", "b.pdf", "--instruction", "对比方法"
    )

    assert options.mode == "paper"
    assert options.file_paths == ["a.pdf", "b.pdf"]
    assert options.instruction == "对比方法"


def test_files_imply_paper_mode():
    """Passing --file is unambiguous, so --mode should not be required."""
    options = _resolve("--file", "paper.pdf")

    assert options.mode == "paper"
    assert options.file_paths == ["paper.pdf"]


def test_paper_mode_without_a_topic_uses_the_default():
    options = _resolve("--file", "a.pdf", "--no-input")

    assert options.topic == DEFAULT_PAPER_TOPIC


def test_flags_never_trigger_optional_prompts():
    """`--file a.pdf` should just run, not stop to ask about an optional topic."""
    options = _resolve("--file", "a.pdf")

    assert options.mode == "paper"
    assert options.topic == DEFAULT_PAPER_TOPIC
    assert options.instruction == ""


def test_blank_and_whitespace_paths_are_dropped():
    options = _resolve("--file", "  a.pdf  ", "--file", "   ", "--no-input")

    assert options.file_paths == ["a.pdf"]


# --- prompting fallback ------------------------------------------------------


def test_mode_and_topic_are_prompted_when_absent():
    options = _resolve(answers=["1", "提示里输入的主题"])

    assert options.mode == "topic"
    assert options.topic == "提示里输入的主题"


def test_paper_flow_prompts_for_topic_instruction_and_paths():
    options = _resolve(answers=["2", "综述方向", "分析要求", "a.pdf ; b.pdf"])

    assert options.mode == "paper"
    assert options.topic == "综述方向"
    assert options.instruction == "分析要求"
    assert options.file_paths == ["a.pdf", "b.pdf"]


def test_any_answer_other_than_2_means_topic_mode():
    assert _resolve(answers=["1", "t"]).mode == "topic"
    assert _resolve(answers=["", "t"]).mode == "topic"
    assert _resolve(answers=["nonsense", "t"]).mode == "topic"


def test_flags_suppress_the_prompts_they_supply():
    """Only the missing paths should be asked for."""
    options = _resolve("--mode", "paper", "--topic", "t", "--instruction", "i", answers=["x.pdf"])

    assert options.file_paths == ["x.pdf"]


# --- --no-input --------------------------------------------------------------


def test_no_input_refuses_to_prompt_for_mode():
    with pytest.raises(ValueError, match="--mode"):
        _resolve("--no-input")


def test_no_input_refuses_to_prompt_for_topic():
    with pytest.raises(ValueError, match="--topic"):
        _resolve("--mode", "topic", "--no-input")


def test_no_input_refuses_to_prompt_for_files():
    with pytest.raises(ValueError, match="--file"):
        _resolve("--mode", "paper", "--no-input")


def test_empty_topic_answer_is_rejected():
    with pytest.raises(ValueError, match="报告主题不能为空"):
        _resolve("--mode", "topic", answers=[""])


def test_empty_path_answer_is_rejected():
    with pytest.raises(ValueError, match="至少一个文献文件路径"):
        _resolve("--mode", "paper", "--topic", "t", answers=["", " ; ; "])


def test_cli_rejects_unbounded_topic_and_instruction_inputs():
    with pytest.raises(ValueError, match="报告主题不能超过"):
        _resolve("--mode", "topic", "--topic", "x" * (MAX_TOPIC_CHARS + 1))

    with pytest.raises(ValueError, match="文献分析要求不能超过"):
        _resolve(
            "--mode",
            "paper",
            "--file",
            "paper.pdf",
            "--instruction",
            "x" * (MAX_INSTRUCTION_CHARS + 1),
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            CliOptions(mode="topic", topic="x" * (MAX_TOPIC_CHARS + 1)),
            "报告主题不能超过",
        ),
        (
            CliOptions(
                mode="paper",
                topic="topic",
                instruction="x" * (MAX_INSTRUCTION_CHARS + 1),
                file_paths=["paper.pdf"],
            ),
            "文献分析要求不能超过",
        ),
    ],
)
def test_generate_report_revalidates_direct_cli_options(monkeypatch, options, message):
    def unexpected_call(*args, **kwargs):
        pytest.fail("document parsing and CrewAI must not run before input validation")

    monkeypatch.setattr(main_module, "parse_document_paths", unexpected_call)
    monkeypatch.setattr(main_module, "AutoReportCrew", unexpected_call)

    with pytest.raises(ValueError, match=message):
        generate_report(options)


# --- parser surface ----------------------------------------------------------


def test_parser_defaults():
    args = _parse()

    assert args.mode is None
    assert args.topic == ""
    assert args.instruction == ""
    assert args.files is None
    assert args.no_input is False


def test_parser_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        _parse("--mode", "sideways")


def test_help_mentions_every_flag(capsys):
    with pytest.raises(SystemExit):
        _parse("--help")

    help_text = capsys.readouterr().out
    for flag in ("--mode", "--topic", "--instruction", "--file", "--no-input"):
        assert flag in help_text


def test_direct_cli_options_are_normalized_before_staged_analysis_and_crew(monkeypatch, tmp_path):
    paths = create_run_paths(run_id="cli_validated_123", runs_dir=tmp_path)
    captured = {}

    def summarize(documents, *, topic, instruction, **kwargs):
        captured["staged"] = (topic, instruction)
        return SimpleNamespace(paper_context="bounded context")

    class CapturingCrew:
        def kickoff(self, *, inputs):
            captured["crew"] = inputs
            return "# report"

    class Builder:
        def build_crew(self, *, mode):
            return CapturingCrew()

    monkeypatch.setattr(main_module, "parse_document_paths", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(main_module, "summarize_documents_staged", summarize)
    monkeypatch.setattr(main_module, "create_run_paths", lambda: paths)
    monkeypatch.setattr(main_module, "AutoReportCrew", Builder)

    generate_report(
        CliOptions(
            mode="paper",
            topic="  topic  ",
            instruction="  instruction  ",
            file_paths=["paper.pdf"],
        )
    )

    assert captured["staged"] == ("topic", "instruction")
    assert captured["crew"]["topic"] == "topic"
    assert captured["crew"]["paper_instruction"] == "instruction"


@pytest.mark.parametrize("keep_draft", [True, False])
def test_cli_applies_failed_paper_artifact_policy(monkeypatch, tmp_path, keep_draft):
    paths = create_run_paths(run_id="cli_failed_123", runs_dir=tmp_path)

    class FailingCrew:
        def kickoff(self, *, inputs):
            paths.draft_file.write_text("complete writer draft", encoding="utf-8")
            paths.report_file.write_text("complete reviewed final", encoding="utf-8")
            raise RuntimeError("review failed")

    class Builder:
        def build_crew(self, *, mode):
            return FailingCrew()

    monkeypatch.setattr(main_module, "parse_document_paths", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        main_module,
        "summarize_documents_staged",
        lambda *args, **kwargs: SimpleNamespace(paper_context="context"),
    )
    monkeypatch.setattr(main_module, "create_run_paths", lambda: paths)
    monkeypatch.setattr(main_module, "AutoReportCrew", Builder)
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", str(keep_draft).lower())

    with pytest.raises(RuntimeError, match="review failed"):
        generate_report(CliOptions(mode="paper", file_paths=["paper.pdf"]))

    if keep_draft:
        assert paths.report_file.read_text(encoding="utf-8") == "complete reviewed final"
        assert paths.draft_file.read_text(encoding="utf-8") == "complete writer draft"
    else:
        assert not paths.draft_file.exists()
        assert not paths.report_file.exists()


def test_cli_does_not_discard_draft_when_successful_kickoff_writes_blank_final(
    monkeypatch, tmp_path
):
    paths = create_run_paths(run_id="cli_blank_123", runs_dir=tmp_path)

    class BlankFinalCrew:
        def kickoff(self, *, inputs):
            paths.draft_file.write_text("complete writer draft", encoding="utf-8")
            paths.report_file.write_text("   \n", encoding="utf-8")
            return ""

    class Builder:
        def build_crew(self, *, mode):
            return BlankFinalCrew()

    monkeypatch.setattr(main_module, "parse_document_paths", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        main_module,
        "summarize_documents_staged",
        lambda *args, **kwargs: SimpleNamespace(paper_context="context"),
    )
    monkeypatch.setattr(main_module, "create_run_paths", lambda: paths)
    monkeypatch.setattr(main_module, "AutoReportCrew", Builder)
    monkeypatch.setenv("PAPER_KEEP_DRAFT_ON_FAILURE", "true")

    with pytest.raises(RuntimeError, match="最终审核返回了空内容"):
        generate_report(CliOptions(mode="paper", file_paths=["paper.pdf"]))

    assert not paths.draft_file.exists()
    assert paths.report_file.read_text(encoding="utf-8") == "complete writer draft"
