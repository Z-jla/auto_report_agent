from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class ModePresentation:
    key: str
    label: str
    eyebrow: str
    title: str
    description: str
    input_title: str
    asset_title: str
    cta: str
    workflow: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReadinessItem:
    label: str
    detail: str
    ready: bool


MODE_PRESENTATIONS: dict[str, ModePresentation] = {
    "topic": ModePresentation(
        key="topic",
        label="主题研究",
        eyebrow="主题研究工作流",
        title="从一个问题出发，得到一份可信报告",
        description="输入主题，也可以附上截图或图表。Agent 会检索资料、组织论据、完成写作并做最终审核。",
        input_title="定义研究任务",
        asset_title="添加视觉素材",
        cta="开始研究并生成报告  →",
        workflow=(
            ("01", "检索与取证"),
            ("02", "结构化写作"),
            ("03", "事实与表达审核"),
        ),
    ),
    "paper": ModePresentation(
        key="paper",
        label="文献综述",
        eyebrow="文献综合工作流",
        title="让多篇文献共同回答一个问题",
        description="上传 PDF、DOCX、TXT 或 Markdown。系统会分块精读、跨文献综合，再输出可交付的综述或分析报告。",
        input_title="定义分析方向",
        asset_title="上传文献材料",
        cta="开始精读并生成综述  →",
        workflow=(
            ("01", "解析与分块"),
            ("02", "跨文献综合"),
            ("03", "写作与审核"),
        ),
    ),
}


APP_STYLES = """
<style>
:root {
  --ar-bg: #f5f7fb;
  --ar-surface: #ffffff;
  --ar-surface-muted: #f8fafc;
  --ar-text: #172033;
  --ar-text-secondary: #526075;
  --ar-text-muted: #5f6f84;
  --ar-border: #e1e7ef;
  --ar-brand: #335cdb;
  --ar-brand-hover: #2949b8;
  --ar-brand-soft: #e9eeff;
  --ar-accent: #0f8f83;
  --ar-success: #15803d;
  --ar-success-bg: #ecfdf3;
  --ar-warning: #b45309;
  --ar-warning-bg: #fff7ed;
  --ar-danger: #b42318;
  --ar-danger-bg: #fef3f2;
  --ar-radius-sm: 8px;
  --ar-radius-md: 12px;
  --ar-radius-lg: 16px;
  --ar-shadow-card: 0 1px 2px rgba(15, 23, 42, .04), 0 8px 24px rgba(15, 23, 42, .05);
}

html, body {
  font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 84% 2%, rgba(51, 92, 219, .08), transparent 25rem),
    var(--ar-bg);
}

[data-testid="stHeader"] {
  background: rgba(245, 247, 251, .82);
  backdrop-filter: blur(12px);
}

[data-testid="stMainBlockContainer"], .main .block-container {
  max-width: 1180px;
  padding-top: 2rem;
  padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
  background: #fbfcfe;
  border-right: 1px solid var(--ar-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: 1.2rem;
}

.ar-sidebar-brand {
  padding: .2rem .1rem .9rem;
}

.ar-sidebar-brand__mark {
  display: inline-grid;
  place-items: center;
  width: 2.15rem;
  height: 2.15rem;
  margin-right: .55rem;
  border-radius: .7rem;
  color: #fff;
  background: var(--ar-brand);
  font-size: 1rem;
  font-weight: 800;
  box-shadow: 0 8px 18px rgba(51, 92, 219, .2);
}

.ar-sidebar-brand__name {
  color: var(--ar-text);
  font-weight: 760;
  letter-spacing: -.01em;
}

.ar-sidebar-brand__caption {
  margin: .75rem 0 0;
  color: var(--ar-text-muted);
  font-size: .8rem;
  line-height: 1.55;
}

.ar-hero {
  position: relative;
  overflow: hidden;
  padding: 2.35rem 2.5rem;
  margin-bottom: 1.25rem;
  border: 1px solid rgba(255, 255, 255, .11);
  border-radius: 1.25rem;
  color: #fff;
  background: linear-gradient(120deg, #15204a 0%, #233a8b 62%, #156f76 118%);
  box-shadow: 0 18px 50px rgba(20, 32, 74, .16);
}

.ar-hero::after {
  content: "";
  position: absolute;
  right: -4rem;
  top: -7rem;
  width: 19rem;
  height: 19rem;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, .16);
  box-shadow: 0 0 0 3.5rem rgba(255, 255, 255, .035), 0 0 0 7rem rgba(255, 255, 255, .025);
}

.ar-hero__eyebrow {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: .55rem;
  margin-bottom: .9rem;
  color: rgba(255, 255, 255, .74);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .12em;
}

.ar-hero__dot {
  width: .48rem;
  height: .48rem;
  border-radius: 50%;
  background: #72e4cc;
  box-shadow: 0 0 0 .28rem rgba(114, 228, 204, .14);
}

.ar-hero h1 {
  position: relative;
  z-index: 1;
  max-width: 760px;
  margin: 0;
  color: #fff;
  font-size: clamp(2rem, 4vw, 3.25rem);
  line-height: 1.12;
  letter-spacing: -.045em;
}

.ar-hero p {
  position: relative;
  z-index: 1;
  max-width: 680px;
  margin: 1rem 0 1.25rem;
  color: rgba(255, 255, 255, .78);
  font-size: 1rem;
  line-height: 1.75;
}

.ar-hero__meta {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  gap: .55rem;
}

.ar-chip {
  display: inline-flex;
  align-items: center;
  gap: .38rem;
  padding: .38rem .68rem;
  border: 1px solid rgba(255, 255, 255, .14);
  border-radius: 999px;
  color: rgba(255, 255, 255, .88);
  background: rgba(255, 255, 255, .08);
  font-size: .76rem;
  font-weight: 620;
}

.ar-chip--status {
  background: rgba(114, 228, 204, .13);
}

.ar-section-heading {
  display: flex;
  align-items: flex-start;
  gap: .8rem;
  margin: 1.8rem 0 .85rem;
}

.ar-section-heading__index {
  display: grid;
  place-items: center;
  flex: 0 0 auto;
  width: 2rem;
  height: 2rem;
  border-radius: .65rem;
  color: var(--ar-brand);
  background: var(--ar-brand-soft);
  font-size: .74rem;
  font-weight: 800;
}

.ar-section-heading h2 {
  margin: 0;
  color: var(--ar-text);
  font-size: 1.18rem;
  letter-spacing: -.02em;
}

.ar-section-heading p {
  margin: .22rem 0 0;
  color: var(--ar-text-muted);
  font-size: .84rem;
}

ol.ar-page-progress {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .7rem;
  width: 100%;
  padding: 0 !important;
  margin: .25rem 0 1rem;
  list-style: none;
}

.ar-page-progress__step {
  display: flex;
  align-items: center;
  gap: .62rem;
  padding: .72rem .82rem;
  border: 1px solid var(--ar-border);
  border-radius: .8rem;
  color: var(--ar-text-muted);
  background: rgba(255, 255, 255, .66);
  font-size: .8rem;
  font-weight: 650;
  margin: 0 !important;
  list-style: none !important;
}

.ar-page-progress__step b {
  display: grid;
  place-items: center;
  width: 1.45rem;
  height: 1.45rem;
  border-radius: .48rem;
  background: #edf1f6;
  font-size: .67rem;
}

.ar-page-progress__step.is-active {
  color: #2448b5;
  border-color: #b9c7f6;
  background: #f0f3ff;
}

.ar-page-progress__step.is-active b {
  color: #fff;
  background: var(--ar-brand);
}

.ar-page-progress__step.is-done {
  color: var(--ar-success);
  border-color: #b9e6c9;
  background: var(--ar-success-bg);
}

.ar-page-progress__step.is-done b {
  color: #fff;
  background: var(--ar-success);
}

.ar-page-progress__step.is-error {
  color: var(--ar-danger);
  border-color: #fecaca;
  background: var(--ar-danger-bg);
}

.ar-mode-intro {
  padding: 1.05rem 1.15rem;
  margin: .65rem 0 1rem;
  border-left: 3px solid var(--ar-brand);
  border-radius: .15rem .85rem .85rem .15rem;
  background: linear-gradient(90deg, #f0f3ff, rgba(255, 255, 255, .82));
}

.ar-mode-intro__eyebrow {
  color: var(--ar-brand);
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .12em;
}

.ar-mode-intro h3 {
  margin: .35rem 0 .28rem;
  color: var(--ar-text);
  font-size: 1rem;
}

.ar-mode-intro p {
  margin: 0;
  color: var(--ar-text-secondary);
  font-size: .83rem;
  line-height: 1.65;
}

.ar-workflow {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .55rem;
  margin-top: .85rem;
}

.ar-workflow__item {
  padding: .72rem .75rem;
  border: 1px solid var(--ar-border);
  border-radius: .72rem;
  background: var(--ar-surface-muted);
}

.ar-workflow__item b {
  display: block;
  color: var(--ar-brand);
  font-size: .68rem;
  letter-spacing: .06em;
}

.ar-workflow__item span {
  display: block;
  margin-top: .22rem;
  color: var(--ar-text-secondary);
  font-size: .76rem;
  line-height: 1.35;
}

.ar-panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .8rem;
  margin-bottom: .2rem;
}

.ar-panel-title h3 {
  margin: 0;
  color: var(--ar-text);
  font-size: .95rem;
}

.ar-panel-title span {
  color: var(--ar-text-muted);
  font-size: .72rem;
}

.ar-readiness {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .65rem;
  margin: .8rem 0 .2rem;
}

.ar-readiness__item {
  min-height: 4.5rem;
  padding: .75rem .8rem;
  border: 1px solid var(--ar-border);
  border-radius: .78rem;
  background: var(--ar-surface-muted);
}

.ar-readiness__item.is-ready {
  border-color: #b9e6c9;
  background: var(--ar-success-bg);
}

.ar-readiness__label {
  color: var(--ar-text);
  font-size: .78rem;
  font-weight: 720;
}

.ar-readiness__detail {
  margin-top: .28rem;
  color: var(--ar-text-muted);
  font-size: .72rem;
  line-height: 1.45;
}

.ar-delivery-meta {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem .8rem;
  padding: .78rem .9rem;
  margin: .25rem 0 .8rem;
  border: 1px solid var(--ar-border);
  border-radius: .75rem;
  background: var(--ar-surface-muted);
  color: var(--ar-text-muted);
  font-size: .76rem;
}

.ar-delivery-meta strong {
  color: var(--ar-text-secondary);
}

.ar-mobile-config-hint {
  display: none;
}

div[role="radiogroup"] {
  gap: .5rem;
}

div[role="radiogroup"] > label {
  padding: .52rem .86rem;
  border: 1px solid var(--ar-border);
  border-radius: .72rem;
  background: #fff;
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
  min-height: 8rem;
  border: 1.5px dashed #b9c5d5;
  border-radius: .9rem;
  background: #f8faff;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] {
  display: none;
}

.ar-sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.stButton > button, .stDownloadButton > button {
  min-height: 2.75rem;
  border-radius: .72rem;
  font-weight: 680;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, .08);
}

.stButton > button[kind="primary"] {
  border-color: var(--ar-brand);
  background: var(--ar-brand);
}

.stButton > button[kind="primary"]:hover {
  border-color: var(--ar-brand-hover);
  background: var(--ar-brand-hover);
}

[data-testid="stStatusWidget"] {
  border-radius: .9rem;
}

code {
  overflow-wrap: anywhere;
}

@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"], .main .block-container {
    padding: 1.15rem 1rem 3rem;
  }

  .ar-hero {
    padding: 1.35rem 1.15rem;
    border-radius: 1rem;
  }

  .ar-hero h1 {
    font-size: 1.8rem;
  }

  .ar-hero p {
    margin: .7rem 0 .85rem;
    font-size: .9rem;
    line-height: 1.6;
  }

  .ar-hero__meta .ar-chip:nth-child(n + 3) {
    display: none;
  }

  .ar-mobile-config-hint {
    position: relative;
    z-index: 1;
    display: block;
    padding: .58rem .68rem;
    margin-top: .75rem;
    border: 1px solid rgba(255, 255, 255, .16);
    border-radius: .65rem;
    color: rgba(255, 255, 255, .9);
    background: rgba(255, 255, 255, .08);
    font-size: .75rem;
    line-height: 1.5;
  }

  ol.ar-page-progress, .ar-workflow {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .35rem;
  }

  .ar-page-progress__step {
    flex-direction: column;
    gap: .32rem;
    padding: .5rem .25rem;
    text-align: center;
    font-size: .7rem;
  }

  .ar-workflow__item {
    padding: .55rem .45rem;
  }

  .ar-workflow__item span {
    font-size: .68rem;
  }

  .ar-readiness {
    grid-template-columns: 1fr;
  }

  .ar-readiness__item {
    min-height: 0;
  }

  .ar-mode-intro {
    padding: .85rem .9rem;
  }

  [data-testid="stHorizontalBlock"] {
    flex-direction: column;
    gap: .75rem;
  }

  [data-testid="column"], [data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }

  .stButton button, .stDownloadButton button {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    animation-duration: .01ms !important;
    transition-duration: .01ms !important;
  }
}
</style>
"""


def mode_presentation(mode: str) -> ModePresentation:
    try:
        return MODE_PRESENTATIONS[mode]
    except KeyError as exc:
        raise ValueError(f"未知任务模式：{mode}") from exc


def api_fields_ready(config: Mapping[str, object]) -> bool:
    return all(str(config.get(key) or "").strip() for key in ("api_key", "base_url", "model"))


def build_readiness_items(
    *,
    mode: str,
    api_ready: bool,
    has_topic: bool,
    has_image: bool = False,
    image_valid: bool = True,
    document_count: int = 0,
    documents_valid: bool = True,
    document_error: str = "",
) -> tuple[ReadinessItem, ...]:
    api_item = ReadinessItem(
        label="模型连接",
        detail="API 配置可用于本次调用" if api_ready else "请先在左侧完成 API 配置",
        ready=api_ready,
    )
    if mode == "topic":
        goal_ready = has_topic or (has_image and image_valid)
        goal_item = ReadinessItem(
            label="任务目标",
            detail=("已获得文字或图片任务目标" if goal_ready else "输入主题，或上传一张可识别图片"),
            ready=goal_ready,
        )
        asset_item = ReadinessItem(
            label="辅助素材",
            detail=(
                "图片已通过格式与大小校验"
                if has_image and image_valid
                else (
                    "图片未通过格式或大小校验，请更换后重试"
                    if has_image
                    else "未添加图片，将按纯文本主题研究"
                )
            ),
            ready=not has_image or image_valid,
        )
        return (api_item, goal_item, asset_item)
    if mode == "paper":
        document_item = ReadinessItem(
            label="文献材料",
            detail=(
                document_error
                if document_error
                else (f"已选择 {document_count} 篇文献" if document_count else "至少上传一篇文献")
            ),
            ready=document_count > 0 and documents_valid,
        )
        direction_item = ReadinessItem(
            label="分析方向",
            detail="已填写综述方向" if has_topic else "可留空，系统将使用默认分析主题",
            ready=True,
        )
        return (api_item, document_item, direction_item)
    raise ValueError(f"未知任务模式：{mode}")


def can_generate(items: Sequence[ReadinessItem]) -> bool:
    return bool(items) and all(item.ready for item in items)


def hero_html(*, api_ready: bool) -> str:
    connection = "API 配置已就绪" if api_ready else "等待 API 配置"
    mobile_hint = (
        ""
        if api_ready
        else '<div class="ar-mobile-config-hint">从左上角打开侧栏，先完成模型连接配置。</div>'
    )
    return f"""
<section class="ar-hero">
  <div class="ar-hero__eyebrow"><span class="ar-hero__dot"></span>AI 研究工作台</div>
  <h1>把复杂研究，变成可交付的报告</h1>
  <p>从主题、图片或文献出发，让多个 Agent 完成资料研究、结构化写作、事实审核与多格式导出。</p>
  <div class="ar-hero__meta">
    <span class="ar-chip ar-chip--status">● {connection}</span>
    <span class="ar-chip">多 Agent 协作</span>
    <span class="ar-chip">可观察执行过程</span>
    <span class="ar-chip">Markdown · PDF · Word</span>
  </div>
  {mobile_hint}
</section>
""".strip()


def section_heading_html(index: str, title: str, description: str) -> str:
    return f"""
<div class="ar-section-heading">
  <span class="ar-section-heading__index">{escape(index)}</span>
  <div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>
</div>
""".strip()


def page_progress_html(active_step: int, *, error: bool = False) -> str:
    labels = ("配置任务", "Agent 执行", "预览交付")
    rendered: list[str] = []
    for index, label in enumerate(labels):
        if index < active_step:
            state = "is-done"
            marker = "✓"
            status = "已完成"
        elif index == active_step:
            state = "is-error" if error else "is-active"
            marker = "!" if error else f"0{index + 1}"
            status = "失败" if error else "当前步骤"
        else:
            state = ""
            marker = f"0{index + 1}"
            status = "未开始"
        current = ' aria-current="step"' if index == active_step else ""
        rendered.append(
            f'<li class="ar-page-progress__step {state}"{current}><b>{marker}</b>'
            f'<span>{label}</span><span class="ar-sr-only">，{status}</span></li>'
        )
    return (
        '<ol class="ar-page-progress" aria-label="报告生成进度" aria-live="polite">'
        f"{''.join(rendered)}</ol>"
    )


def mode_intro_html(presentation: ModePresentation) -> str:
    workflow = "".join(
        f'<div class="ar-workflow__item"><b>{escape(index)}</b><span>{escape(label)}</span></div>'
        for index, label in presentation.workflow
    )
    return f"""
<div class="ar-mode-intro">
  <div class="ar-mode-intro__eyebrow">{escape(presentation.eyebrow)}</div>
  <h3>{escape(presentation.title)}</h3>
  <p>{escape(presentation.description)}</p>
  <div class="ar-workflow">{workflow}</div>
</div>
""".strip()


def panel_title_html(title: str, hint: str) -> str:
    return f'<div class="ar-panel-title"><h3>{escape(title)}</h3><span>{escape(hint)}</span></div>'


def readiness_html(items: Sequence[ReadinessItem]) -> str:
    blocks = "".join(
        (
            f'<div class="ar-readiness__item {"is-ready" if item.ready else ""}">'
            f'<div class="ar-readiness__label">{"✓" if item.ready else "○"} '
            f"{escape(item.label)}</div>"
            f'<div class="ar-readiness__detail">{escape(item.detail)}</div></div>'
        )
        for item in items
    )
    return f'<div class="ar-readiness">{blocks}</div>'


def report_meta_html(*, mode: str, generated_at: str, run_id: str, char_count: int) -> str:
    mode_label = "文献综述" if mode == "paper" else "主题研究"
    return f"""
<div class="ar-delivery-meta">
  <span><strong>{escape(mode_label)}</strong></span>
  <span>生成时间：{escape(generated_at or "未知")}</span>
  <span>正文：{char_count:,} 字符</span>
  <span>Run ID：<code>{escape(run_id or "未知")}</code></span>
</div>
""".strip()


SIDEBAR_BRAND_HTML = """
<div class="ar-sidebar-brand">
  <div><span class="ar-sidebar-brand__mark">AR</span><span class="ar-sidebar-brand__name">Auto Report Agent</span></div>
  <p class="ar-sidebar-brand__caption">模型凭证仅用于当前会话。公共部署不会向浏览器下发服务器 API Key。</p>
</div>
""".strip()
