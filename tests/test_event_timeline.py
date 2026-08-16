"""Tests for the Streamlit run-progress timeline in app.py.

The timeline is driven by CrewAI callbacks during a long run, so the property
that matters is that adding an event touches only that event: the previous
implementation redrew the whole list each time, which is O(n^2) in both element
creations and websocket deltas.

Most tests drive EventTimeline through a stub container to assert the exact call
pattern; the last one runs a real Streamlit script to confirm what actually
renders.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import EventTimeline
from auto_report_agent.agent_events import AgentEvent, agent_icon

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeExpander:
    def __init__(self, label: str):
        self.label = label
        self.markdowns: list[str] = []

    def markdown(self, body: str) -> None:
        self.markdowns.append(body)


class _FakeStatus:
    def __init__(self):
        self.markdowns: list[str] = []
        self.expanders: list[_FakeExpander] = []
        self.updates: list[dict] = []

    def markdown(self, body: str) -> None:
        self.markdowns.append(body)

    def expander(self, label: str, expanded: bool = False) -> _FakeExpander:
        expander = _FakeExpander(label)
        self.expanders.append(expander)
        return expander

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class _FakeContainer:
    """Minimal stand-in for the DeltaGenerator surface EventTimeline uses."""

    def __init__(self):
        self.status_calls: list[dict] = []
        self.status_container = _FakeStatus()

    def status(self, label: str, *, state: str, expanded: bool) -> _FakeStatus:
        self.status_calls.append({"label": label, "state": state, "expanded": expanded})
        return self.status_container


def _timeline(**kwargs):
    container = _FakeContainer()
    return container, EventTimeline(container, **kwargs)


def test_events_are_appended_not_redrawn():
    container, timeline = _timeline()

    for index in range(8):
        timeline.log(f"步骤 {index}")

    status = container.status_container
    assert len(container.status_calls) == 1, "the status container must be created only once"
    assert len(status.markdowns) == 8, "each event should render exactly one element"
    assert status.markdowns[0].endswith("步骤 0")
    assert status.markdowns[-1].endswith("步骤 7")


def test_event_detail_goes_into_its_own_expander():
    container, timeline = _timeline()

    timeline.log("有细节", detail="**思考过程**")
    timeline.log("无细节")

    status = container.status_container
    assert len(status.expanders) == 1
    assert status.expanders[0].markdowns == ["**思考过程**"]
    assert status.expanders[0].label.endswith("有细节")
    assert len(status.markdowns) == 1
    assert status.markdowns[0].endswith("无细节")


def test_detail_is_collapsed_to_one_line_when_disabled():
    container, timeline = _timeline(show_detail=False)

    timeline.log("有细节", detail="不应展示")

    status = container.status_container
    assert status.expanders == []
    assert len(status.markdowns) == 1


def test_header_counts_steps_and_reports_success():
    container, timeline = _timeline()

    timeline.log("第一步")
    timeline.log("第二步")
    timeline.finalize(success=True)

    updates = container.status_container.updates
    assert updates[-1]["state"] == "complete"
    assert updates[-1]["label"] == "🎉 完成 · 共 2 个步骤"
    assert updates[0]["state"] == "running"
    assert "已记录 1 个步骤" in updates[0]["label"]


def test_header_reports_failure():
    container, timeline = _timeline()

    timeline.log("出问题了")
    timeline.finalize(success=False)

    last = container.status_container.updates[-1]
    assert last["state"] == "error"
    assert last["label"] == "❌ 出错 · 已记录 1 个步骤"


def test_finalize_without_events_does_not_crash():
    container, timeline = _timeline()

    timeline.finalize(success=True)

    assert len(container.status_calls) == 1
    assert container.status_container.updates[-1]["label"] == "🎉 完成 · 共 0 个步骤"


def test_label_includes_timestamp_icon_and_agent():
    container, timeline = _timeline()

    timeline.add(AgentEvent(kind="tool_call", title="调用工具", icon="🛠️", agent="researcher"))

    label = container.status_container.markdowns[0]
    assert label.startswith("🛠️ ")
    assert "**researcher**" in label
    assert agent_icon("researcher") in label
    assert timeline.events[0].timestamp, "a timestamp should be filled in automatically"


def test_supplied_timestamp_is_preserved():
    container, timeline = _timeline()

    timeline.add(AgentEvent(kind="system", title="固定时间", timestamp="01:02:03"))

    assert "`01:02:03`" in container.status_container.markdowns[0]


def test_renders_under_a_real_streamlit_runtime():
    """End-to-end check that the appended elements actually reach the page."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import streamlit as st
from app import EventTimeline, AgentEvent

timeline = EventTimeline(st.container(), show_detail=True)
timeline.log("收到任务")
timeline.add(AgentEvent(kind="tool_call", title="调用工具", icon="TOOL",
                        agent="researcher", detail="思考过程详情"))
timeline.log("完成")
timeline.finalize(success=True)
"""

    at = AppTest.from_string(script).run(timeout=60)

    assert not list(at.exception)
    assert len(at.status) == 1, "the whole run should live in one status container"
    assert at.status[0].label == "🎉 完成 · 共 3 个步骤"
    assert at.status[0].state == "complete"

    assert len(at.expander) == 1, "only the event carrying detail gets an expander"
    assert "**researcher**" in at.expander[0].label
    assert [child.value for child in at.expander[0].markdown] == ["思考过程详情"]

    labels = [element.value for element in at.markdown]
    assert any(label.endswith("收到任务") for label in labels)
    assert any(label.endswith("完成") for label in labels)
