# Auto Report Agent

**Language / 语言**: [中文](./README.md) | [English](./README.en.md)

[![CI](https://github.com/Z-jla/auto_report_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Z-jla/auto_report_agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Project status: experimental / portfolio project.** APIs and behavior may change without notice; not recommended for production use.

An automated report-generation project built on **CrewAI + Streamlit + OpenAI-compatible APIs**. Given a topic, an uploaded image, or uploaded literature, the system runs a "research → writing → review" pipeline and produces a high-quality Markdown report, with PDF / Word export.

The project is designed as a portfolio piece for AI Agent / multi-agent collaboration / automated research-writing work.

## Features

- **Multi-agent workflow**: researcher, writer, and reviewer agents collaborate.
- **Web search**: supports OpenAI-compatible providers exposing the Responses API `web_search_preview` tool.
- **Topic report**: given a topic, it searches, organizes material, drafts a report, and reviews it.
- **Image understanding**: upload screenshots, charts, paper pages, or exam figures; the image is parsed before the report is drafted.
- **Literature analysis**: upload PDFs / documents; content is extracted in stages and turned into a literature review or analytical report.
- **Web UI**: a Streamlit frontend provides inputs, API config, progress view, and report preview.
- **Progress visibility**: the page surfaces task stages, tool calls, and result summaries so the agent workflow is easy to follow.
- **Multiple exports**: Markdown, PDF, and Word.
- **Caching**: staged literature results are cached to cut redundant model calls.

## Project layout

```text
auto_report_agent/
├── app.py
├── src/
│   └── auto_report_agent/
│       ├── api_config.py
│       ├── cache_manager.py
│       ├── crew.py
│       ├── document_ingest.py
│       ├── docx_export.py
│       ├── main.py
│       ├── openai_web_search_tool.py
│       ├── pdf_export.py
│       ├── settings.py
│       ├── staged_literature.py
│       ├── vision.py
│       └── config/
│           ├── agents.yaml
│           └── tasks.yaml
├── output/
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Requirements

- Python `>=3.10,<3.14`
- Any OpenAI-compatible API service: OpenAI, Azure OpenAI, DeepSeek, Qwen, Kimi, Zhipu GLM, Ollama, vLLM, and third-party aggregators.
- Web search requires a provider that exposes the Responses API `web_search_preview` tool.
- Image understanding requires a model with vision input.

## Install

```bash
git clone https://github.com/Z-jla/auto_report_agent.git
cd auto_report_agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e .
```

Dev install:

```bash
pip install -e ".[dev]"
```

## Configuration

Copy the env template:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
# Recommended: LLM_* names (aligned with CrewAI / LiteLLM conventions)
LLM_API_KEY=your API key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name

# Optional: separate vision model; leave empty to reuse LLM_MODEL
LLM_VISION_MODEL=

# chat: plain Chat Completions
# responses: Responses API (required for web_search_preview)
LLM_API_MODE=chat

# Enable provider web search
ENABLE_WEB_SEARCH=false
```

> Compatibility: the legacy names `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL_NAME`, `OPENAI_VISION_MODEL_NAME`, and `OPENAI_API_MODE` still work as fallback. If both are set, `LLM_*` wins.

> Note: do not commit `.env` to GitHub. `.env` is already listed in `.gitignore`.

## Run the frontend

```bash
streamlit run app.py
```

In the browser you can:

1. Fill in API key, base URL, model name, and whether to enable web search in the sidebar.
2. Pick a task mode:
   - Topic research report
   - Image understanding + report
   - Literature analysis / review
3. Click "Generate report".
4. Inspect the run and the final report on the page.
5. Download Markdown / PDF / Word.

## Run from the command line

```bash
python -m auto_report_agent.main
```

Or after install:

```bash
auto-report-agent
```

## Output

Default output:

```text
output/final_report.md
```

The frontend also offers downloads for:

- `final_report.md`
- `final_report.pdf`
- `final_report.docx`

Generated files are not committed by default.

## Development and testing

```bash
pip install -e ".[dev]"
ruff check .
python -m compileall -q .
pytest -q
```

## Security notes

- Do not commit `.env`, API keys, logs, caches, or generated reports.
- If an API key is accidentally pushed, revoke it immediately and rotate.
- API config entered in the Streamlit page is only applied to the current process. `api_environment()` restores environment variables after each run so keys are not leaked across sessions.

## FAQ

### 1. Why do some compatible providers not support web search?

Plain Chat Completions only generates text; it does not imply web search. Web search in this project relies on the provider exposing the Responses API `web_search_preview` tool.

### 2. Why do the PDF and the web page not look exactly the same?

The web page is rendered by the browser's Markdown renderer; the PDF is rendered by `reportlab`. Headings, paragraphs, lists, tables, bold text, and CJK line-wrapping are handled, but complex Markdown may need more polish.

### 3. Why is literature analysis slow with many papers?

Literature analysis reads, summarizes, and merges in stages; long documents cause multiple model calls. Tune `PAPER_CHUNK_SIZE`, `PAPER_MAX_CHUNKS_PER_DOC`, etc. in `.env` to trade off cost and speed.

## License

MIT License