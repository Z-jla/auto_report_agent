import os

from auto_report_agent.settings import initialize_runtime

initialize_runtime()

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.agents.agent_builder.base_agent import BaseAgent  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402

from auto_report_agent.api_config import ResolvedLLMConfig, resolve_llm_env  # noqa: E402
from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool  # noqa: E402


@CrewBase
class AutoReportCrew:
    """主题研究 / 文献分析 的自动报告生成 Crew。"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    _llm_config: ResolvedLLMConfig | None = None

    def _llm(self) -> LLM:
        """Build the LLM every agent in this crew shares.

        Passing it explicitly is what keeps a run off ``os.environ``. Left to
        itself CrewAI reads MODEL/OPENAI_* and silently substitutes its own
        default model when they are unset.
        """
        config = self._llm_config or resolve_llm_env()
        config.require("model")
        # base_url and api_base are both set because CrewAI's own environment
        # path sets both and downstream code reads one or the other.
        return LLM(
            model=config.model,
            api_key=config.api_key or None,
            base_url=config.base_url or None,
            api_base=config.base_url or None,
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[OpenAIWebSearchTool()],
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],
            verbose=True,
        )

    @agent
    def paper_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["paper_analyst"],
            verbose=True,
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
            tasks = [
                self.research_task(),
                self.writing_task(),
                self.review_task(),
            ]

        # Injected after construction rather than passed to Agent(): CrewBase
        # builds the agents eagerly when this object is created, before any
        # configuration could be handed to it.
        llm = self._llm()
        for built in agents:
            built.llm = llm
            for tool in getattr(built, "tools", None) or []:
                if isinstance(tool, OpenAIWebSearchTool):
                    tool.llm_config = self._llm_config

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        return self.build_crew(mode="topic")
