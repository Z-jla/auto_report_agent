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
- **Web search**: prefers the Responses API `web_search` tool and falls back to the legacy `web_search_preview` for older compatible providers.
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
│       ├── agent_events.py
│       ├── api_config.py
│       ├── cache_manager.py
│       ├── crew.py
│       ├── document_ingest.py
│       ├── docx_export.py
│       ├── main.py
│       ├── openai_web_search_tool.py
│       ├── pdf_export.py
│       ├── run_manager.py
│       ├── settings.py
│       ├── staged_literature.py
│       ├── vision.py
│       ├── py.typed
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
- Web search requires a provider that exposes the Responses API `web_search` tool or the compatible legacy `web_search_preview` tool.
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
# responses: Responses API (used for web search)
LLM_API_MODE=chat

# Stream model output; on by default, since buffered calls get cut off by proxies
LLM_STREAM=true

# Enable provider web search
ENABLE_WEB_SEARCH=false

# Safe public mode; use local for Ollama/private endpoints on a trusted machine
APP_DEPLOYMENT_MODE=public
```

> Compatibility: the legacy names `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL_NAME`, `OPENAI_VISION_MODEL_NAME`, and `OPENAI_API_MODE` still work as fallback. If both are set, `LLM_*` wins.

> Note: do not commit `.env` to GitHub. `.env` is already listed in `.gitignore`.

### Deployment mode and upload limits

- `APP_DEPLOYMENT_MODE=public` (the safe default): never pre-fills the browser with a server-side key, accepts only public HTTPS API hosts, and disables global cache cleanup.
- `APP_DEPLOYMENT_MODE=local`: for a trusted single-user machine; permits `http://localhost`, Ollama, vLLM, and private API hosts.
- Public mode permits the built-in provider hosts plus custom hosts in `APP_ALLOWED_API_HOSTS` (wildcards such as `*.example.com` are supported). Private endpoints require both an allowlist entry and `APP_ALLOW_PRIVATE_API_HOSTS=true`; enable this only in a trusted environment.
- Defaults: 10 MB per image, 5 documents, 25 MB per document, 50 MB total, and 200 PDF pages. The `MAX_*` variables in `.env.example` tune these limits; Streamlit also enforces a 50 MB pre-application upload ceiling.

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

With no arguments it prompts for each value. Flags make it scriptable:

```bash
# Topic research report
auto-report-agent --mode topic --topic "Recent progress in AI agents"

# Literature analysis (--file implies paper mode)
auto-report-agent --file a.pdf --file b.pdf --instruction "Compare methods and conclusions"

# Automation: fail instead of blocking on stdin
auto-report-agent --mode topic --topic "..." --no-input
```

Passing any flag suppresses the optional prompts; with `--no-input` a missing required value exits with code 2.

## Output

Each run gets an isolated directory:

```text
output/runs/<run_id>/final_report.md
output/runs/<run_id>/run.json
```

The frontend also offers downloads for:

- `final_report.md`
- `final_report.pdf`
- `final_report.docx`

Generated files are not committed by default. Streamlit only displays report history from the current browser session and does not load another session's run directory.

### Automatic cleanup

Run directories and per-session literature caches accumulate indefinitely, and public mode disables the manual cleanup sidebar, so on startup the app deletes `output/runs/<run_id>/` entries and literature-cache namespaces older than `OUTPUT_RETENTION_DAYS` (default 7). Set it to `0` to disable. This is a disk-usage backstop rather than a precise cache policy: an evicted entry is simply recomputed on the next request.

## Development and testing

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m compileall -q app.py src tests
mypy
pytest -q
```

## Security notes

- Do not commit `.env`, API keys, logs, caches, or generated reports.
- If an API key is accidentally pushed, revoke it immediately and rotate.
- Public mode never sends a server key to the browser. Session settings are passed into each model call as an argument rather than written to process environment variables, so concurrent browser sessions neither share credentials nor wait on each other.
- Base URLs are checked for scheme, host, DNS result, and private addresses; shared deployments should also configure a host allowlist.
- Every report uses an isolated run directory. Literature caches include the browser session, model, base URL, API mode, prompt version, and generation settings.

## FAQ

### 1. Why do some compatible providers not support web search?

Plain Chat Completions only generates text; it does not imply web search. The project prefers the Responses API `web_search` tool and falls back to `web_search_preview` for older compatible providers.

### 2. Why do the PDF and the web page not look exactly the same?

The web page is rendered by the browser's Markdown renderer; the PDF is rendered by `reportlab`. Headings (levels 1-6), paragraphs, lists, tables, bold, italic, and CJK line-wrapping are handled, and unbalanced inline markers degrade to plain text instead of failing the whole export. Complex Markdown may still need more polish.

### 3. Why is literature analysis slow with many papers?

Literature analysis reads, summarizes, and merges in stages. When a document exceeds the chunk cap, chunks are sampled evenly from the beginning, middle, and end, and the coverage ratio is included in the merged material. Tune `PAPER_CHUNK_SIZE`, `PAPER_MAX_CHUNKS_PER_DOC`, etc. in `.env` to trade off cost and speed.

Chunk summaries within one document are independent, so four of them run concurrently by default (`PAPER_STAGE_CONCURRENCY`, 1-16), measured at roughly 3x faster on a ten-chunk document. Set it to `1` for the previous serial behaviour if your provider quota is low and returns 429s. Cached chunks do not consume a concurrency slot.

### 4. Generation fails partway through with "Connection error"

Usually a relay or reverse proxy dropped a long request. Many relays — Cloudflare-fronted ones especially — close a connection that has produced no bytes for about 120 seconds, and writing a report regularly takes two or three minutes. A buffered request sends nothing until generation finishes, so the connection is closed first and the client only sees `Connection error`. The SDK retries twice by default, which is why this typically looks like "hung for six minutes, then failed".

Streaming is on by default (`LLM_STREAM=true`) precisely to avoid this: bytes keep flowing, so the idle timeout never fires. If you turned it off and hit this error, set it back to `true`.

Note that **lowering an output cap such as `max_tokens` does not work around it** — the model is equally silent while it works, so the timeout still fires.

If short requests fail too, it is not this timeout: check that `LLM_BASE_URL` is reachable and the key is valid.

## License

MIT License
