import os
from copy import copy as shallow_copy
from typing import Any

from auto_report_agent.settings import (
    crewai_task_output_persistence_enabled,
    crewai_verbose_enabled,
    env_bool,
    initialize_runtime,
)

initialize_runtime()

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.agents.agent_builder.base_agent import BaseAgent  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402
from crewai.utilities.task_output_storage_handler import (  # noqa: E402
    TaskOutputStorageHandler,
)
from pydantic import PrivateAttr  # noqa: E402

from auto_report_agent.api_config import (  # noqa: E402
    ResolvedLLMConfig,
    resolve_llm_env,
    validate_base_url,
)
from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool  # noqa: E402

# Appended to research_task depending on whether the search tool was attached.
# Neither string may contain braces: CrewAI interpolates the description at
# kickoff and would read them as missing input fields.
_SEARCH_AVAILABLE_DIRECTIVE = (
    "检索方式：使用 openai_web_search 工具联网检索，优先覆盖最近三年的公开资料。"
)
_SEARCH_UNAVAILABLE_DIRECTIVE = (
    "检索方式：本次运行没有联网检索能力，不要尝试调用任何搜索工具。"
    "请完全基于你已掌握的知识作答，并在「来源清单」中如实说明本次无法提供实时链接、"
    "信息可能存在时效性局限。"
)


class _EphemeralTaskOutputHandler:
    """CrewAI task-output handler that deliberately retains nothing.

    CrewAI currently has no public constructor flag for disabling its replay
    database.  This implements the small handler interface the 1.x Crew class
    calls, keeping public requests out of the process-global SQLite file.  The
    methods accept future-compatible extra arguments because this is a privacy
    boundary: an added optional argument should not silently restore disk
    persistence after a dependency upgrade.
    """

    def reset(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def load(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _new_task_output_handler() -> TaskOutputStorageHandler | _EphemeralTaskOutputHandler:
    if crewai_task_output_persistence_enabled():
        return TaskOutputStorageHandler()
    return _EphemeralTaskOutputHandler()


class _PrivacyAwareCrew(Crew):
    """Crew variant whose replay storage follows this app's deployment policy.

    Overriding the Pydantic private-attribute factory is important: replacing
    the handler *after* constructing a normal ``Crew`` would already have
    created/opened ``latest_kickoff_task_outputs.db``.
    """

    _task_output_handler: Any = PrivateAttr(default_factory=_new_task_output_handler)

    def copy(self) -> Crew:  # type: ignore[override]
        """Clone without falling back to CrewAI's persistent base ``Crew``.

        CrewAI 1.x hard-codes ``Crew(...)`` in its copy implementation. Batch,
        async, train, and test helpers call that method internally, which would
        silently restore the shared replay SQLite handler in public mode.
        This mirrors the upstream component-aware copy while retaining this
        privacy-aware subclass.
        """
        exclude = {
            "id",
            "_rpm_controller",
            "_logger",
            "_execution_span",
            "_file_handler",
            "_cache_handler",
            "_memory",
            "agents",
            "tasks",
            "knowledge_sources",
            "knowledge",
            "manager_agent",
            "manager_llm",
        }
        cloned_agents = [agent.copy() for agent in self.agents]
        manager_agent = self.manager_agent.copy() if self.manager_agent else None
        manager_llm = shallow_copy(self.manager_llm) if self.manager_llm else None
        task_mapping: dict[str, Any] = {}
        cloned_tasks = []

        for original_task in self.tasks:
            cloned_task = original_task.copy(cloned_agents, task_mapping)
            cloned_tasks.append(cloned_task)
            task_mapping[original_task.key] = cloned_task

        for cloned_task, original_task in zip(cloned_tasks, self.tasks, strict=False):
            if isinstance(original_task.context, list):
                cloned_task.context = [
                    task_mapping[context_task.key] for context_task in original_task.context
                ]

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {key: value for key, value in copied_data.items() if value is not None}
        if getattr(self, "_memory", None):
            copied_data["memory"] = self._memory
        copied_data.pop("agents", None)
        copied_data.pop("tasks", None)

        return _PrivacyAwareCrew(
            **copied_data,
            agents=cloned_agents,
            tasks=cloned_tasks,
            knowledge_sources=shallow_copy(self.knowledge_sources),
            knowledge=shallow_copy(self.knowledge),
            manager_agent=manager_agent,
            manager_llm=manager_llm,
        )


@CrewBase
class AutoReportCrew:
    """主题研究 / 文献分析 的自动报告生成 Crew。"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    _llm_config: ResolvedLLMConfig | None = None

    def _llm(self, config: ResolvedLLMConfig) -> LLM:
        """Build the LLM every agent in this crew shares.

        Passing it explicitly is what keeps a run off ``os.environ``. Left to
        itself CrewAI reads MODEL/OPENAI_* and silently substitutes its own
        default model when they are unset.

        Streaming is on by default because report writing is exactly the shape of
        request that reverse proxies cut off: many relays (Cloudflare-fronted ones
        in particular) drop a connection that produces no bytes for ~120s, and a
        buffered call returns nothing until the whole report is generated. Capping
        output tokens does not help — the model is still silent while it works, so
        the timeout fires anyway. A streamed response keeps bytes flowing and
        survives. Set ``LLM_STREAM=false`` for a provider that mishandles SSE.
        """
        config.require("model")
        base_url = self._checked_base_url(config)
        # base_url and api_base are both set because CrewAI's own environment
        # path sets both and downstream code reads one or the other.
        return LLM(
            model=config.model,
            api_key=config.api_key or None,
            base_url=base_url or None,
            api_base=base_url or None,
            stream=env_bool("LLM_STREAM", True),
        )

    @staticmethod
    def _checked_base_url(config: ResolvedLLMConfig) -> str:
        """Apply the same SSRF checks every other model caller applies.

        staged_literature and the search tool validate whatever configuration
        they are handed. Without this the crew was the one path that would talk
        to any host the environment named, so a single ``.env`` could be refused
        by literature analysis and accepted by topic mode — and a CLI user only
        found out halfway through a run.
        """
        if not config.base_url:
            return ""
        try:
            return validate_base_url(config.base_url)
        except ValueError as exc:
            raise RuntimeError(f"Base URL 安全校验未通过：{exc}") from exc

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[OpenAIWebSearchTool()],
            verbose=crewai_verbose_enabled(),
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            verbose=crewai_verbose_enabled(),
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],
            verbose=crewai_verbose_enabled(),
        )

    @agent
    def paper_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["paper_analyst"],
            verbose=crewai_verbose_enabled(),
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config["writing_task"])

    @task
    def review_task(self) -> Task:
        return Task(config=self.tasks_config["review_task"])

    @task
    def paper_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["paper_analysis_task"])

    @task
    def paper_writing_task(self) -> Task:
        return Task(config=self.tasks_config["paper_writing_task"])

    @task
    def paper_review_task(self) -> Task:
        return Task(config=self.tasks_config["paper_review_task"])

    @task
    def paper_direct_writing_task(self) -> Task:
        return Task(config=self.tasks_config["paper_direct_writing_task"])

    @task
    def paper_direct_review_task(self) -> Task:
        return Task(config=self.tasks_config["paper_direct_review_task"])

    def build_crew(
        self, mode: str = "topic", *, llm_config: ResolvedLLMConfig | None = None
    ) -> Crew:
        """Assemble the crew for one run using ``llm_config``.

        Keep this builder referenced for as long as the returned crew is in use.
        CrewAI memoizes the agent methods on ``id(self)``, so if the builder is
        collected mid-run its address can be reused by another builder, which
        would then be handed these same agent objects and overwrite their
        configuration.
        """
        mode = (mode or "topic").strip().lower()
        paper_crew_mode = os.getenv("PAPER_CREW_MODE", "fast").strip().lower()
        self._llm_config = llm_config

        config = llm_config or resolve_llm_env()
        # The tool refuses every other combination, so anything else means the
        # researcher would be handed a tool that can only ever say "skipped".
        search_enabled = config.enable_web_search and config.api_mode == "responses"

        research_task: Task | None = None
        if mode == "paper":
            if paper_crew_mode in {"full", "standard", "classic"}:
                agents = [
                    self.paper_analyst(),
                    self.writer(),
                    self.reviewer(),
                ]
                tasks = [
                    self.paper_analysis_task(),
                    self.paper_writing_task(),
                    self.paper_review_task(),
                ]
            else:
                agents = [
                    self.writer(),
                    self.reviewer(),
                ]
                tasks = [
                    self.paper_direct_writing_task(),
                    self.paper_direct_review_task(),
                ]
        else:
            agents = [
                self.researcher(),
                self.writer(),
                self.reviewer(),
            ]
            research_task = self.research_task()
            tasks = [
                research_task,
                self.writing_task(),
                self.review_task(),
            ]

        # Injected after construction rather than passed to Agent(): CrewBase
        # builds the agents eagerly when this object is created, before any
        # configuration could be handed to it.
        llm = self._llm(config)
        for built in agents:
            built.llm = llm
            self._apply_search_tool(built, enabled=search_enabled)

        if research_task is not None:
            self._apply_search_directive(research_task, enabled=search_enabled)

        return _PrivacyAwareCrew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=crewai_verbose_enabled(),
        )

    def _apply_search_tool(self, built: BaseAgent, *, enabled: bool) -> None:
        """配置或摘掉联网搜索工具。

        Leaving an unusable tool attached is not harmless: the researcher keeps
        calling it, gets the same "skipped" notice back every time, and burns
        iterations and tokens on a loop that cannot make progress (observed at
        8-10 calls in a single run). An agent with no search tool simply answers
        from what it knows.
        """
        tools = list(getattr(built, "tools", None) or [])
        if enabled:
            for tool in tools:
                if isinstance(tool, OpenAIWebSearchTool):
                    tool.llm_config = self._llm_config
            return

        remaining = [tool for tool in tools if not isinstance(tool, OpenAIWebSearchTool)]
        if len(remaining) != len(tools):
            built.tools = remaining

    def _apply_search_directive(self, research_task: Task, *, enabled: bool) -> None:
        """Tell the researcher how it is expected to gather material.

        Rebuilt from the YAML source rather than appended to the live
        description, because ``@task`` memoizes on the builder: appending would
        stack a second directive if the same builder assembled a crew twice.
        """
        base = str(self.tasks_config["research_task"]["description"]).rstrip()
        directive = _SEARCH_AVAILABLE_DIRECTIVE if enabled else _SEARCH_UNAVAILABLE_DIRECTIVE
        research_task.description = f"{base}\n{directive}"

    @crew
    def crew(self) -> Crew:
        return self.build_crew(mode="topic")
