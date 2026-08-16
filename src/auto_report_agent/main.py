"""Command-line entrypoint.

Flags make the generator scriptable; anything not passed is still prompted for,
so the previous fully interactive flow keeps working. ``--no-input`` turns a
missing value into an error instead of a prompt, which is what an automated
caller wants.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from auto_report_agent.settings import env_int, initialize_runtime

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

DEFAULT_PAPER_TOPIC = "上传文献分析与总结"
Prompt = Callable[[str], str]


@dataclass(frozen=True)
class CliOptions:
    mode: str
    topic: str = ""
    instruction: str = ""
    file_paths: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-report-agent",
        description="生成主题研究报告或文献分析报告。",
        epilog=(
            "示例：\n"
            "  auto-report-agent --mode topic --topic 'AI Agent 最新进展'\n"
            "  auto-report-agent --file a.pdf --file b.pdf --instruction '对比方法与结论'\n"
            "\n未提供的参数会转为交互式询问；--no-input 时改为直接报错。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("topic", "paper"),
        help="任务模式。省略时：给了 --file 就按 paper，否则询问。",
    )
    parser.add_argument("--topic", default="", help="报告主题或文献综述方向。")
    parser.add_argument(
        "--instruction",
        default="",
        help="文献分析的补充要求（仅 paper 模式使用）。",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        metavar="PATH",
        help="要分析的文献路径，可重复传入。",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="不进行交互式询问；缺少必需参数时直接报错。",
    )
    return parser


def resolve_options(args: argparse.Namespace, *, prompt: Prompt = input) -> CliOptions:
    """Fill in whatever the flags did not supply.

    A bare invocation reproduces the old fully interactive flow. As soon as any
    flag is passed the call is treated as scripted: only genuinely required
    values are asked for, so ``--file a.pdf`` runs instead of stopping to ask
    about an optional topic. ``--no-input`` turns even those into errors.
    """
    files = [path.strip() for path in (args.files or []) if path.strip()]
    topic = args.topic.strip()
    instruction = args.instruction.strip()
    scripted = bool(args.mode or files or topic or instruction or args.no_input)

    def ask(question: str, *, required_for: str) -> str:
        if args.no_input:
            raise ValueError(f"--no-input 模式下缺少必需参数：{required_for}")
        return prompt(question).strip()

    def ask_optional(question: str) -> str:
        return "" if scripted else prompt(question).strip()

    mode = args.mode
    if mode is None and files:
        # Files were given, so the intent is unambiguous without asking.
        mode = "paper"
    if mode is None:
        answer = ask("请选择模式（1=主题研究报告，2=文献分析总结）：", required_for="--mode")
        mode = "paper" if answer == "2" else "topic"

    if mode == "paper":
        topic = topic or ask_optional("请输入文献主题/综述方向（可留空）：")
        instruction = instruction or ask_optional("请输入补充分析要求（可留空）：")
        if not files:
            raw = ask("请输入文献文件路径，多个文件用英文分号 ; 分隔：", required_for="--file")
            files = [part.strip() for part in raw.split(";") if part.strip()]
        if not files:
            raise ValueError("文献分析模式下，必须提供至少一个文献文件路径")
        return CliOptions(
            mode="paper",
            topic=topic or DEFAULT_PAPER_TOPIC,
            instruction=instruction,
            file_paths=files,
        )

    if not topic:
        topic = ask("请输入报告主题：", required_for="--topic")
    if not topic:
        raise ValueError("报告主题不能为空")
    return CliOptions(mode="topic", topic=topic)


def generate_report(options: CliOptions) -> tuple[str, object]:
    """Run the configured pipeline and return the report text and its run paths."""
    paper_context = ""
    if options.mode == "paper":
        documents = parse_document_paths(
            options.file_paths,
            max_chars_per_doc=env_int("PAPER_MAX_CHARS_PER_DOC", 120000, 20000, 200000),
        )
        staged_result = summarize_documents_staged(
            documents,
            topic=options.topic,
            instruction=options.instruction,
            progress=lambda message: print(f"[文献分析] {message}"),
            cache_namespace="cli",
        )
        paper_context = staged_result.paper_context

    run_paths = create_run_paths()
    result = (
        AutoReportCrew()
        .build_crew(mode=options.mode)
        .kickoff(
            inputs={
                "topic": options.topic,
                "paper_instruction": options.instruction,
                "paper_context": paper_context,
                "mode": options.mode,
                "output_file": run_paths.report_file.as_posix(),
            }
        )
    )

    if run_paths.report_file.exists():
        report_text = run_paths.report_file.read_text(encoding="utf-8")
    else:
        report_text = str(result)
        write_text_atomic(run_paths.report_file, report_text)

    write_run_metadata(
        run_paths,
        {
            "topic": options.topic,
            "mode": options.mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "char_count": len(report_text),
        },
    )
    return report_text, run_paths


def run(argv: Sequence[str] | None = None) -> int:
    """运行自动报告生成流程。"""
    args = build_parser().parse_args(argv)

    sweep = enforce_retention()
    if sweep.removed_dirs:
        print(f"[清理] 已删除 {sweep.removed_dirs} 个过期运行/缓存目录。")

    try:
        options = resolve_options(args)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    report_text, run_paths = generate_report(options)

    print("\n========== 最终结果 ==========")
    print(report_text)
    print(f"\n报告文件：{run_paths.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
