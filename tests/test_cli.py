"""Tests for the CLI argument handling.

Only the argument/prompt layer is exercised here: generate_report drives CrewAI
and would need a live model. The point of the split is that everything deciding
*what* to run is reachable without one.
"""

import pytest

from auto_report_agent.main import DEFAULT_PAPER_TOPIC, build_parser, resolve_options


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
