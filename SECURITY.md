# Security Policy

## 支持的版本

本项目仍处于早期阶段，仅对 `main` 分支的最新提交提供安全修复。

## 报告漏洞

如果你发现了安全问题，请**不要**直接在 Issues 中公开。请通过以下方式私下联系：

- 在 GitHub 上打开一个 [Security Advisory](https://github.com/Z-jla/auto_report_agent/security/advisories/new)（推荐）
- 或发送邮件到仓库所有者的 GitHub 公开邮箱

请在报告中尽量说明：

- 漏洞影响范围（信息泄露、RCE、拒绝服务等）
- 复现步骤或 PoC
- 受影响的文件、函数或版本
- 你认为的修复思路（可选）

我会在 72 小时内确认收到，并在修复后再公开披露。

## API Key 与敏感信息

- **永远不要**把 `.env`、API Key、token、账号密码提交到公开仓库。
- 项目默认在 `.gitignore` 中忽略 `.env`、日志、缓存和生成的报告。
- 如果误提交了 API Key：
  1. **立即到服务商后台吊销并重新生成 Key。**
  2. 从 Git 历史中移除（`git filter-repo` 或 BFG），仅从最新提交删除不够。
  3. 强推（`git push --force`）或联系维护者处理。
- 公共模式不会把服务器 `.env` 中的 API Key 下发到 Streamlit 页面。前端填写的 Key 只保存在当前会话状态和受保护的调用上下文中；调用结束后会恢复环境变量。
- 公共部署会拒绝 HTTP、本机、私网、链路本地和保留地址，并可通过 `APP_ALLOWED_API_HOSTS` 进一步限制允许的 API 主机。
- 每次报告写入独立运行目录，页面不会读取其他浏览器会话的报告；公共模式下禁用全局缓存清理。
- 上传文件有数量、压缩包、体积和 PDF 页数限制。这些限制可缓解资源耗尽，但公共服务仍应在反向代理/容器层配置请求体、CPU、内存和执行时间上限。

## 第三方依赖

本项目依赖 `crewai`、`openai`、`streamlit`、`reportlab`、`python-docx`、`pypdf` 等第三方库。这些库的漏洞请直接向其上游仓库报告。
