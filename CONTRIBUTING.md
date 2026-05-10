# Contributing

欢迎提交 issue 和 pull request。

## 本地开发

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
ruff check .
```

## 提交建议

- 不要提交 `.env`、API Key、缓存文件和生成报告。
- 新增功能请尽量补充测试或在 README 中说明使用方式。
- PR 描述中说明改动动机、主要文件和验证命令。
