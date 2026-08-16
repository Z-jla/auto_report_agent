"""Tests for parsing CrewAI / LangChain callback payloads into timeline events.

These payloads are whatever object the agent loop happened to produce, so the
parser works by class name with attribute fallbacks. The property that matters
most is that it never raises: it runs inside step_callback, and an exception
there would abort the crew run it is only meant to report on.
"""

import pytest

from auto_report_agent.agent_events import (
    AgentEvent,
    agent_icon,
    parse_step_output,
    parse_task_output,
    safe_short_text,
)


def _payload(class_name: str, **attributes):
    """Build a stand-in for a callback object of the given class name."""
    return type(class_name, (), attributes)()


# --- safe_short_text ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ("  padded  ", "padded"),
        ("multi\n\nline\ttext", "multi line text"),
        (None, ""),
        ("", ""),
        (123, "123"),
    ],
)
def test_safe_short_text_normalises_whitespace(value, expected):
    assert safe_short_text(value) == expected


def test_safe_short_text_truncates_with_ellipsis():
    assert safe_short_text("x" * 100, 10) == "xxxxxxxxxx..."


def test_safe_short_text_keeps_text_at_the_limit():
    assert safe_short_text("x" * 10, 10) == "x" * 10


# --- agent_icon --------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("researcher", "🔍"),
        ("Researcher", "🔍"),
        ("专业研究员", "🔍"),
        ("writer", "✍️"),
        ("专业报告写作者", "✍️"),
        ("reviewer", "🧐"),
        ("paper_analyst", "📚"),
        ("学术文献分析专家", "📚"),
    ],
)
def test_agent_icon_maps_known_roles(role, expected):
    assert agent_icon(role) == expected


@pytest.mark.parametrize("role", ["", "   ", "unknown role", None])
def test_agent_icon_falls_back_for_unknown_roles(role):
    assert agent_icon(role) == "🤖"


def test_agent_icon_matches_a_role_embedded_in_a_longer_string():
    assert agent_icon("Senior Researcher Agent") == "🔍"


# --- parse_step_output: tool calls -------------------------------------------


@pytest.mark.parametrize(
    "class_name", ["AgentAction", "ToolUsage", "ToolUsageStarted", "ToolUsageEvent"]
)
def test_tool_call_shapes_are_recognised(class_name):
    event = parse_step_output(
        _payload(class_name, tool="openai_web_search", tool_input="latest agents")
    )

    assert event.kind == "tool_call"
    assert event.icon == "🛠️"
    assert "openai_web_search" in event.title
    assert "latest agents" in event.title


def test_tool_call_records_thought_and_arguments_in_detail():
    event = parse_step_output(
        _payload("AgentAction", tool="search", tool_input="query text", log="  I should search  ")
    )

    assert "**思考过程**" in event.detail
    assert "I should search" in event.detail
    assert "**调用参数**" in event.detail
    assert "query text" in event.detail


def test_tool_call_falls_back_across_attribute_names():
    """Different CrewAI versions use tool/tool_name and tool_input/input."""
    event = parse_step_output(_payload("ToolUsage", tool_name="alt_tool", input="alt input"))

    assert "alt_tool" in event.title
    assert "alt input" in event.title


def test_tool_call_without_a_tool_name_is_still_reported():
    event = parse_step_output(_payload("AgentAction"))

    assert event.kind == "tool_call"
    assert "未知工具" in event.title


def test_long_tool_arguments_are_capped():
    event = parse_step_output(_payload("AgentAction", tool="t", tool_input="y" * 5000))

    assert len(event.detail) < 2000


# --- parse_step_output: tool results -----------------------------------------


@pytest.mark.parametrize("class_name", ["ToolResult", "ToolUsageFinished", "ToolUsageEnded"])
def test_tool_result_shapes_are_recognised(class_name):
    event = parse_step_output(_payload(class_name, result="found 3 papers"))

    assert event.kind == "tool_result"
    assert event.icon == "📥"
    assert "14" in event.title, "the title reports the result length"
    assert event.detail == "found 3 papers"


def test_tool_result_reads_output_and_raw_fallbacks():
    assert parse_step_output(_payload("ToolResult", output="from output")).detail == "from output"
    assert parse_step_output(_payload("ToolResult", raw="from raw")).detail == "from raw"


def test_tool_result_detail_is_capped():
    event = parse_step_output(_payload("ToolResult", result="z" * 5000))

    assert len(event.detail) == 2000


# --- parse_step_output: conclusions and fallbacks ----------------------------


def test_agent_finish_prefers_the_return_value():
    event = parse_step_output(
        _payload("AgentFinish", return_values={"output": "  the answer  "}, log="the log")
    )

    assert event.kind == "thought"
    assert event.icon == "💭"
    assert event.detail == "the answer"


def test_agent_finish_falls_back_to_the_log():
    event = parse_step_output(_payload("AgentFinish", return_values={}, log="only a log"))

    assert event.detail == "only a log"


def test_unknown_class_with_a_tool_name_is_treated_as_a_tool_call():
    event = parse_step_output(_payload("SomethingNew", tool_name="mystery"))

    assert event.kind == "tool_call"
    assert "mystery" in event.title


def test_unknown_class_with_an_action_is_reported_as_a_system_step():
    event = parse_step_output(_payload("SomethingNew", action="doing a thing"))

    assert event.kind == "system"
    assert "doing a thing" in event.title


def test_completely_unknown_payload_still_yields_an_event():
    """A callback that raised would abort the crew run it only reports on."""
    event = parse_step_output(_payload("TotallyUnexpected"))

    assert event.kind == "system"
    assert "TotallyUnexpected" in event.title


@pytest.mark.parametrize("payload", [None, object(), "a bare string", 42])
def test_odd_payloads_do_not_raise(payload):
    assert isinstance(parse_step_output(payload), AgentEvent)


def test_agent_role_is_read_from_either_attribute():
    assert parse_step_output(_payload("AgentAction", agent_role="researcher")).agent == "researcher"
    assert parse_step_output(_payload("AgentAction", agent="writer")).agent == "writer"


# --- parse_task_output -------------------------------------------------------


def test_task_output_summarises_name_agent_and_length():
    event = parse_task_output(
        _payload("TaskOutput", name="research_task", agent="researcher", raw="x" * 42)
    )

    assert event.kind == "task_done"
    assert event.icon == "✅"
    assert "任务完成" in event.title
    assert "research_task" in event.title
    assert "42" in event.title
    assert event.agent == "researcher"
    assert event.detail == "x" * 42


def test_task_output_without_a_name_or_body():
    event = parse_task_output(_payload("TaskOutput"))

    assert event.title == "任务完成"
    assert event.detail == ""


def test_task_output_detail_is_capped():
    assert len(parse_task_output(_payload("TaskOutput", raw="q" * 9000)).detail) == 3000
