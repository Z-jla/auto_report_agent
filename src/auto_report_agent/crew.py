import os
from typing import List

from auto_report_agent.settings import initialize_runtime


initialize_runtime()

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402
from crewai.agents.agent_builder.base_agent import BaseAgent  # noqa: E402

from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool  # noqa: E402


@CrewBase
class AutoReportCrew:
    """主题研究 / 文献分析 的自动报告生成 Crew。"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

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

    def build_crew(self, mode: str = "topic") -> Crew:
        mode = (mode or "topic").strip().lower()
        paper_crew_mode = os.getenv("PAPER_CREW_MODE", "fast").strip().lower()

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

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        return self.build_crew(mode="topic")
