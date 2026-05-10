from auto_report_agent.settings import initialize_runtime

initialize_runtime()

from auto_report_agent.crew import AutoReportCrew  # noqa: E402
from auto_report_agent.document_ingest import (  # noqa: E402
    build_paper_context,
    parse_document_paths,
)


def run() -> None:
    """运行自动报告生成流程。"""
    mode_input = input("请选择模式（1=主题研究报告，2=文献分析总结）：").strip()
    mode = "paper" if mode_input == "2" else "topic"

    if mode == "paper":
        topic = input("请输入文献主题/综述方向（可留空）：").strip() or "上传文献分析与总结"
        instruction = input("请输入补充分析要求（可留空）：").strip()
        raw_paths = input("请输入文献文件路径，多个文件用英文分号 ; 分隔：").strip()

        if not raw_paths:
            raise ValueError("文献分析模式下，必须提供至少一个文献文件路径")

        file_paths = [part.strip() for part in raw_paths.split(";") if part.strip()]
        documents = parse_document_paths(file_paths)
        paper_context = build_paper_context(documents)

        result = AutoReportCrew().build_crew(mode="paper").kickoff(
            inputs={
                "topic": topic,
                "paper_instruction": instruction,
                "paper_context": paper_context,
                "mode": "paper",
            }
        )
    else:
        topic = input("请输入报告主题：").strip()

        if not topic:
            raise ValueError("报告主题不能为空")

        result = AutoReportCrew().build_crew(mode="topic").kickoff(
            inputs={
                "topic": topic,
                "paper_instruction": "",
                "paper_context": "",
                "mode": "topic",
            }
        )

    print("\n========== 最终结果 ==========")
    print(result)


if __name__ == "__main__":
    run()