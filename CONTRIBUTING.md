# Contributing

感谢你愿意参与这个项目。欢迎提 issue、提 PR，也欢迎只是留言讨论。

## 准备开发环境

```bash
git clone https://github.com/Z-jla/auto_report_agent.git
cd auto_report_agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

复制并填好环境变量：

```bash
cp .env.example .env
# 编辑 .env，至少填上 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
# （旧名 OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL_NAME 仍然兼容）
```

## 本地验证

提交前请至少跑通这三项：

```bash
ruff check .
python -m compileall -q .
pytest -q
```

跑 Streamlit 前端：

```bash
streamlit run app.py
```

跑命令行版：

```bash
python -m auto_report_agent.main
```

## 代码风格

- 使用 `ruff` 管理 lint 和 import 排序，规则见 [pyproject.toml](pyproject.toml) 的 `[tool.ruff]` 部分。
- 行宽限制为 100，实际被 `E501` ignore 掉，所以不会强制截断，但尽量保持可读。
- 优先改动最小。不要在无关的文件上做风格重排。

## 提交与 PR

- 不要在提交中包含 `.env`、API Key、日志（`*.log`）、缓存（`.crewai_storage/`、`.ruff_cache/` 等）、或生成的报告（`output/*.md|pdf|docx|json`）。如果不小心误传了 Key，**立刻到服务商后台吊销**并参考 [SECURITY.md](SECURITY.md)。
- commit 信息写清楚"做了什么 / 为什么"，一行标题 + 可选正文即可。
- PR 尽量保持单一主题；大改动欢迎先开 issue 讨论。
- PR 模板中的检查清单请逐条确认。

## 新增功能时

- 涉及新 API 调用或新模块：尽量补一条 happy-path 测试（参考 [tests/](tests/)）。
- 影响命令行 / Streamlit UI 使用方式：在 [README.md](README.md) 对应章节同步更新。
- 引入新的环境变量：同步写入 [.env.example](.env.example) 并在 README 的"配置"章节说明含义。

## 报告问题

- Bug 请用 [issue 模板](.github/ISSUE_TEMPLATE/bug_report.yml) 描述，提供复现步骤、Python 版本和 API 兼容商信息。
- **安全问题请走 Security Advisory**，不要在公开 issue 中披露，详见 [SECURITY.md](SECURITY.md)。
