import os
from datetime import datetime

from auto_report_agent.settings import initialize_runtime

initialize_runtime()

from auto_report_agent.cache_manager import enforce_retention  # noqa: E402
from auto_report_agent.crew import AutoReportCrew  # noqa: E402
from auto_report_agent.document_ingest import parse_document_paths  # noqa: E402
from auto_report_agent.run_manager import (  # noqa: E402
    create_run_paths,
    write_run_metadata,
    write_text_atomic,
)
from auto_report_agent.staged_literature import summarize_documents_staged  # noqa: E402


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def run() -> None:
    """运行自动报告生成流程。"""
    sweep = enforce_retention()
    if sweep.removed_dirs:
        print(f"[清理] 已删除 {sweep.removed_dirs} 个过期运行/缓存目录。")

    mode_input = input("请选择模式（1=主题研究报告，2=文献分析总结）：").strip()
    mode = "paper" if mode_input == "2" else "topic"

    if mode == "paper":
        topic = input("请输入文献主题/综述方向（可留空）：").strip() or "上传文献分析与总结"
        instruction = input("请输入补充分析要求（可留空）：").strip()
        raw_paths = input("请输入文献文件路径，多个文件用英文分号 ; 分隔：").strip()

        if not raw_paths:
            raise ValueError("文献分析模式下，必须提供至少一个文献文件路径")

        file_paths = [part.strip() for part in raw_paths.split(";") if part.strip()]
        documents = parse_document_paths(
            file_paths,
            max_chars_per_doc=_env_int("PAPER_MAX_CHARS_PER_DOC", 120000, 20000, 200000),
        )
        staged_result = summarize_documents_staged(
            documents,
            topic=topic,
            instruction=instruction,
            progress=lambda message: print(f"[文献分析] {message}"),
            cache_namespace="cli",
        )
        paper_context = staged_result.paper_context
        run_paths = create_run_paths()

        result = (
            AutoReportCrew()
            .build_crew(mode="paper")
            .kickoff(
                inputs={
                    "topic": topic,
                    "paper_instruction": instruction,
                    "paper_context": paper_context,
                    "mode": "paper",
                    "output_file": run_paths.report_file.as_posix(),
                }
            )
        )
    else:
        topic = input("请输入报告主题：").strip()

        if not topic:
            raise ValueError("报告主题不能为空")

        run_paths = create_run_paths()
        result = (
            AutoReportCrew()
            .build_crew(mode="topic")
            .kickoff(
                inputs={
                    "topic": topic,
                    "paper_instruction": "",
                    "paper_context": "",
                    "mode": "topic",
                    "output_file": run_paths.report_file.as_posix(),
                }
            )
        )

    if run_paths.report_file.exists():
        report_text = run_paths.report_file.read_text(encoding="utf-8")
    else:
        report_text = str(result)
        write_text_atomic(run_paths.report_file, report_text)

    print("\n========== 最终结果 ==========")
    print(report_text)
    write_run_metadata(
        run_paths,
        {
            "topic": topic,
            "mode": mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "char_count": len(report_text),
        },
    )
    print(f"\n报告文件：{run_paths.report_file}")


if __name__ == "__main__":
    run()
