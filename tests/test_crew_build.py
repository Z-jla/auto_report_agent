import pytest

from auto_report_agent.crew import AutoReportCrew


@pytest.fixture
def llm_env(monkeypatch):
    """CrewAI validates agent config on construction and requires model/key env vars."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "demo-model")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "demo-model")


def test_topic_mode_uses_research_writing_review(llm_env):
    crew = AutoReportCrew().build_crew(mode="topic")

    roles = [getattr(agent, "role", "").strip() for agent in crew.agents]
    assert any("研究员" in role for role in roles)
    assert any("写作" in role for role in roles)
    assert any("审核" in role for role in roles)
    assert len(crew.tasks) == 3


def test_paper_mode_fast_uses_direct_writing(llm_env, monkeypatch):
    monkeypatch.setenv("PAPER_CREW_MODE", "fast")
    crew = AutoReportCrew().build_crew(mode="paper")

    assert len(crew.agents) == 2
    assert len(crew.tasks) == 2


def test_paper_mode_full_runs_analysis_pipeline(llm_env, monkeypatch):
    monkeypatch.setenv("PAPER_CREW_MODE", "full")
    crew = AutoReportCrew().build_crew(mode="paper")

    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3


def test_unknown_mode_falls_back_to_topic(llm_env):
    crew = AutoReportCrew().build_crew(mode="nonsense")
    assert len(crew.tasks) == 3


@pytest.mark.parametrize("mode", ["topic", "paper"])
def test_final_task_interpolates_isolated_output_path(llm_env, monkeypatch, mode):
    monkeypatch.setenv("PAPER_CREW_MODE", "fast")
    crew = AutoReportCrew().build_crew(mode=mode)
    output_file = "output/runs/safe_run_123/final_report.md"

    crew._interpolate_inputs(
        {
            "topic": "topic",
            "paper_context": "context",
            "paper_instruction": "instruction",
            "mode": mode,
            "output_file": output_file,
        }
    )

    assert crew.tasks[-1].output_file == output_file
