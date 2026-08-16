# Auto Report Agent

**Language / 语言**: [中文](./README.md) | [English](./README.en.md)

[![CI](https://github.com/Z-jla/auto_report_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Z-jla/auto_report_agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **项目状态：实验性 / 作品展示（portfolio project）**。接口与行为可能随时调整，不建议直接用于生产环境。

一个基于 **CrewAI + Streamlit + OpenAI-compatible API** 的自动报告生成项目。用户输入主题、上传图片或上传文献后，系统会按「研究 → 写作 → 审核」流程生成高质量 Markdown 报告，并支持导出 PDF / Word。

这个项目适合作为 AI Agent / 多 Agent 协作 / 自动化研究写作方向的实习作品展示。

## 功能特性

- **多 Agent 工作流**：研究员、写作者、审核员分工协作。
- **联网搜索**：支持遵循 OpenAI 协议的服务商的 Responses API `web_search_preview` 工具。
- **主题报告生成**：输入一个主题，自动搜索资料、整理观点、撰写报告并审核。
- **图片识别**：上传截图、图表、论文页面或题目图片，先识别图片内容，再生成报告。
- **文献分析**：上传 PDF / 文档，分阶段提取内容并生成文献综述或分析报告。
- **前端页面**：使用 Streamlit 提供可视化输入、API 配置、任务运行过程和报告预览。
- **运行过程展示**：页面展示任务阶段、工具调用和结果摘要，便于理解 Agent 工作流。
- **多格式导出**：支持 Markdown、PDF、Word 下载。
- **缓存管理**：对文献分阶段处理结果进行缓存，减少重复调用。

## 项目结构

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

## 环境要求

- Python `>=3.10,<3.14`
- 支持任何遵循 OpenAI 协议的 API 服务，包括 OpenAI、Azure OpenAI、DeepSeek、Qwen、Kimi、智谱 GLM、Ollama、vLLM 等官方或自建服务，以及各类第三方聚合商
- 如需联网搜索，服务商需要支持 Responses API 的 `web_search` 或兼容的 `web_search_preview` 工具
- 如需图片识别，模型需要支持视觉输入

## 安装

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

开发模式：

```bash
pip install -e ".[dev]"
```

## 配置

复制环境变量示例文件：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```env
# 推荐：使用 LLM_* 命名（与 CrewAI / LiteLLM 保持一致）
LLM_API_KEY=你的 API Key
LLM_BASE_URL=https://你的兼容商地址/v1
LLM_MODEL=你的模型名

# 如果视觉模型和文本模型不同，可以单独配置
LLM_VISION_MODEL=

# chat：普通 Chat Completions
# responses：Responses API，适合 Web Search
LLM_API_MODE=chat

# 是否启用兼容商 Web Search
ENABLE_WEB_SEARCH=false

# 公共部署安全模式；本机 Ollama/私网服务请改为 local
APP_DEPLOYMENT_MODE=public
```

> 兼容性：旧的 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_MODEL_NAME`、`OPENAI_VISION_MODEL_NAME`、`OPENAI_API_MODE` 仍然可用（作为 fallback）。同时设置时，优先读取 `LLM_*`。

> 注意：不要把 `.env` 提交到 GitHub。项目已经在 `.gitignore` 中忽略 `.env`。

### 部署模式与上传限制

- `APP_DEPLOYMENT_MODE=public`（默认安全策略）：页面不会预填服务器 `.env` 中的 Key，只允许 HTTPS 公网 API 地址，并禁用全局缓存清理。
- `APP_DEPLOYMENT_MODE=local`：适合本机单用户使用，允许 `http://localhost`、Ollama、vLLM 等本地或私网接口。
- 公共模式允许页面内置服务商域名，并叠加 `APP_ALLOWED_API_HOSTS` 中的自定义主机（支持 `*.example.com`）。访问私网接口需要同时设置主机白名单和 `APP_ALLOW_PRIVATE_API_HOSTS=true`，且只应在受信环境中开启。
- 默认限制为：图片 10 MB、最多 5 篇文献、单篇 25 MB、合计 50 MB、PDF 200 页。可通过 `.env.example` 中的 `MAX_*` 变量调整，Streamlit 服务器还有 50 MB 的请求前置上限。

## 运行前端

```bash
streamlit run app.py
```

打开浏览器后可以：

1. 在侧边栏填写 API Key、Base URL、模型名和是否启用联网搜索。
2. 选择任务模式：
   - 主题研究报告
   - 图片识别 + 报告生成
   - 文献分析 / 文献综述
3. 点击「生成报告」。
4. 在页面查看运行过程和最终报告。
5. 下载 Markdown / PDF / Word。

## 命令行运行

```bash
python -m auto_report_agent.main
```

或安装后运行：

```bash
auto-report-agent
```

## 输出文件

每次运行使用独立目录：

```text
output/runs/<run_id>/final_report.md
output/runs/<run_id>/run.json
```

前端支持下载：

- `final_report.md`
- `final_report.pdf`
- `final_report.docx`

这些生成文件默认不会被 Git 提交。Streamlit 只展示当前浏览器会话生成的历史报告，不读取其他会话的运行目录。

### 自动清理

运行目录和按会话隔离的文献缓存都会持续累积，而公共模式禁用了侧边栏的手动清理入口，因此应用启动时会自动删除超过 `OUTPUT_RETENTION_DAYS`（默认 7 天）的 `output/runs/<run_id>/` 和文献缓存命名空间。设为 `0` 可关闭。这是磁盘占用的兜底，不是精确的缓存策略：被删掉的缓存会在下次请求时重新生成。

## 开发与测试

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m compileall -q app.py src tests
pytest -q
```

## 安全说明

- 不要提交 `.env`、API Key、日志、缓存和生成报告。
- 如果误把 API Key 提交到远程仓库，请立刻撤销并重新生成 Key。
- 公共模式不会把服务器 Key 下发到浏览器；每次调用会先清空受管环境变量、串行应用当前会话配置，并在结束后恢复。
- Base URL 会执行协议、主机、DNS 和私网地址校验；公共部署建议再设置主机白名单。
- 每次报告写入独立运行目录；文献缓存按浏览器会话、模型、Base URL、接口模式、提示词版本和生成参数隔离。

## 常见问题

### 1. 为什么有的兼容服务不能联网搜索？

普通 Chat Completions 只负责文本生成，不等于具备联网搜索。项目优先使用 OpenAI Responses API 的 `web_search` 工具；对旧兼容商会回退到 `web_search_preview`。

### 2. PDF 和网页显示不完全一样怎么办？

网页由浏览器渲染 Markdown，PDF 由 `reportlab` 渲染。项目已经对标题（1-6 级）、段落、列表、表格、加粗、斜体和中文换行做了适配；行内标记不成对时会自动降级为纯文本，不会导致整份导出失败。复杂 Markdown 仍可能需要继续优化。

### 3. 文献很多时为什么慢？

文献分析会分阶段读取、摘要和合并，长文档会产生多次模型调用。超过片段上限时会均匀选择首部、中部和尾部，并把覆盖比例写入综合材料；可以通过 `.env` 中的 `PAPER_CHUNK_SIZE`、`PAPER_MAX_CHUNKS_PER_DOC` 等参数控制成本和速度。

同一篇文献内的片段摘要互相独立，默认会并发 4 个同时分析（`PAPER_STAGE_CONCURRENCY`，可设 1-16）。10 个片段的文档实测约有 3 倍加速。如果你的兼容商配额较低、容易返回 429，把它调回 `1` 就是原来的串行行为。命中缓存的片段不占用并发额度。

## License

MIT License

