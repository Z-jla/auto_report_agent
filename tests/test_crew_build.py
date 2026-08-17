import pytest

from auto_report_agent.api_config import ResolvedLLMConfig
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


# --- explicit LLM config -----------------------------------------------------


EXPLICIT = ResolvedLLMConfig(
    api_key="explicit-key",
    base_url="https://explicit.example/v1",
    model="explicit-model",
    vision_model="",
    api_mode="chat",
    enable_web_search=False,
)


@pytest.fixture
def no_llm_env(monkeypatch):
    """Strip every variable CrewAI would otherwise fall back to."""
    for key in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_VISION_MODEL",
        "LLM_API_MODE",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME",
        "OPENAI_VISION_MODEL_NAME",
        "MODEL",
        "MODEL_NAME",
        "BASE_URL",
        "API_BASE",
    ):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_crew_runs_without_any_llm_environment(no_llm_env):
    """The point of passing config explicitly: no process-global state needed."""
    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    for agent in crew.agents:
        assert agent.llm.model == "explicit-model"
        assert agent.llm.api_key == "explicit-key"
        assert agent.llm.base_url == "https://explicit.example/v1"


def test_every_agent_shares_one_llm(no_llm_env):
    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    assert len({id(agent.llm) for agent in crew.agents}) == 1


def test_search_tool_receives_the_same_config(no_llm_env):
    """Otherwise the tool would fall back to reading the environment itself."""
    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)
    researcher = next(a for a in crew.agents if "研究员" in a.role)

    tool = researcher.tools[0]
    assert tool.llm_config is not None
    assert tool.llm_config.api_key == "explicit-key"


def test_two_crews_do_not_share_configuration(no_llm_env):
    """Concurrent sessions must not see each other's credentials."""
    other = ResolvedLLMConfig(
        api_key="other-key",
        base_url="https://other.example/v1",
        model="other-model",
        vision_model="",
        api_mode="chat",
        enable_web_search=False,
    )

    first = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)
    second = AutoReportCrew().build_crew(mode="topic", llm_config=other)

    assert first.agents[0].llm.model == "explicit-model"
    assert second.agents[0].llm.model == "other-model"
    assert first.agents[0].llm.api_key == "explicit-key"


def test_missing_model_is_reported_clearly(no_llm_env):
    blank = ResolvedLLMConfig("k", "https://x.example/v1", "", "", "chat", False)

    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        AutoReportCrew().build_crew(mode="topic", llm_config=blank)


def test_without_explicit_config_it_still_reads_the_environment(llm_env):
    """The CLI has no session config and relies on .env."""
    crew = AutoReportCrew().build_crew(mode="topic")

    assert crew.agents[0].llm.model == "demo-model"


def test_concurrent_builders_get_distinct_agents(no_llm_env):
    """Two sessions building at once must not be handed the same agent objects."""
    other = ResolvedLLMConfig(
        api_key="other-key",
        base_url="https://other.example/v1",
        model="other-model",
        vision_model="",
        api_mode="chat",
        enable_web_search=False,
    )

    first_builder = AutoReportCrew()
    second_builder = AutoReportCrew()
    first = first_builder.build_crew(mode="topic", llm_config=EXPLICIT)
    second = second_builder.build_crew(mode="topic", llm_config=other)

    assert {id(a) for a in first.agents}.isdisjoint({id(a) for a in second.agents})
    assert first.agents[0].llm.model == "explicit-model"
    assert second.agents[0].llm.model == "other-model"


# --- streaming ---------------------------------------------------------------
# Report writing runs long enough that a buffered response gets dropped by
# proxies that cut idle connections (~120s on Cloudflare-fronted relays), so
# streaming is the default rather than an opt-in.


def test_streaming_is_on_by_default(no_llm_env):
    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    for agent in crew.agents:
        assert agent.llm.stream is True


def test_streaming_can_be_disabled(no_llm_env):
    """Escape hatch for a provider whose SSE implementation is broken."""
    no_llm_env.setenv("LLM_STREAM", "false")

    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    for agent in crew.agents:
        assert agent.llm.stream is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("on", True), ("0", False), ("no", False), ("off", False)],
)
def test_stream_flag_reads_the_usual_boolean_spellings(no_llm_env, value, expected):
    no_llm_env.setenv("LLM_STREAM", value)

    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    assert crew.agents[0].llm.stream is expected


def test_unparseable_stream_flag_keeps_the_default(no_llm_env):
    """A typo must not silently turn streaming off and reintroduce the timeout."""
    no_llm_env.setenv("LLM_STREAM", "ture")

    crew = AutoReportCrew().build_crew(mode="topic", llm_config=EXPLICIT)

    assert crew.agents[0].llm.stream is True
