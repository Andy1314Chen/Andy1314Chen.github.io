# AGENTS.md

AI 编码助手操作手册。本项目是 Hugo 静态博客（hello-friend-ng 主题），无测试框架，构建成功即视为验证通过。
中文回复，言简意赅。

## 构建与验证命令

```bash
hugo server -D              # 本地预览（含草稿），默认 http://localhost:1313
hugo server                 # 仅已发布文章
hugo --minify               # 生产构建

# 验证构建（无 ERROR 即通过）
hugo --minify 2>&1 | grep -i error
hugo --minify 2>&1 | grep "Pages"     # 当前约 40 页

# 新建文章
hugo new content posts/<slug>.md       # 基于 archetypes/posts.md 生成
```

本项目无测试框架、无 package.json、无 Makefile。**唯一验证标准：`hugo --minify` 无 ERROR 输出。**

## CI/CD

- 触发：push 到 `master` 分支（或手动 workflow_dispatch）
- 构建：`hugo --minify --baseURL "${{ steps.pages.outputs.base_url }}/"`
- Hugo 版本：`0.159.1`（extended），部署至 GitHub Pages
- 查看：https://github.com/Andy1314Chen/Andy1314Chen.github.io/actions

**部署**：push 到 master 后 CI 自动构建部署，无需手动生成 `public/`。

## 项目结构

```
content/posts/              # 博客文章（Markdown），约 14 篇
content/about.md            # 关于页
layouts/
  index.html               # 首页模板覆盖（按年分组文章列表）
  posts/single.html        # 文章详情页覆盖（TOC 侧边栏布局）
  partials/favicons.html   # Favicon partial 覆盖（SVG 支持）
  partials/footer.html     # 页脚 partial 覆盖（社交图标）
static/
  css/toc.css              # TOC 侧边栏 + blockquote 修复样式
  favicon.svg              # SVG favicon
scripts/
  publish.sh               # Obsidian 一键发布脚本
archetypes/
  default.md               # 通用模板
  posts.md                 # posts 目录模板
themes/hello-friend-ng/    # 主题 submodule，**禁止直接修改**
hugo.toml                  # 站点配置
.github/workflows/hugo.yml # CI/CD
backup/                    # 历史备份，不参与构建
```

## Front Matter 规范（YAML）

```yaml
---
title: "文章标题"
date: 2026-01-01
tags:
  - "cuda"
  - "triton"
description: "单行纯文本摘要，用于 SEO，不含代码/图片/换行"
toc: false        # 长文（80+ 行）建议设为 true
draft: true       # 发布时改为 false
---
```

**硬性规则**：
- `title` 必填，双引号包裹
- `date` 格式 `YYYY-MM-DD`
- `description` 必须单行纯文本
- `draft: false` 才发布
- **不要加 `categories` 字段**（项目已移除该分类，空 categories 会导致首页渲染异常）

## Markdown 正文规范

- 图片：使用 GitHub 图床完整 URL `![描述](https://raw.githubusercontent.com/Andy1314Chen/obsidian-pic/main/image/xxx.png)`
- 代码块：必须标注语言 ` ```python `
- 数学：行内 `$公式$`，块级 `$$公式$$`（独占一行）
- 图表：` ```mermaid ` 代码块

## Go Template 规范

- 4 空格缩进
- `{{- ` 和 ` -}}` 去除渲染空白
- 条件：`{{- with .Params.xxx }}...{{- end }}`
- 循环：`{{ range .Pages }}...{{ end }}`
- **覆盖主题**：在 `layouts/` 创建同名文件（禁止修改 `themes/` 目录）

常用变量：`.Site.Title`、`.Site.RegularPages`、`.Permalink`、`.Params.tags`、`.Params.toc`、`.Date`、`.ReadingTime`

## 自定义 CSS 规范

- 自定义样式放 `static/css/` 目录，通过 `hugo.toml` 中 `params.customCSS` 数组加载
- 当前已加载：`/css/toc.css`
- 主题 SCSS 变量（只读参考）：`$max-width: 860px`、`.post` max-width `800px`
- 响应式断点：tablet `max-width: 900px`、phone `max-width: 684px`、TOC 隐藏 `max-width: 1200px`

## hugo.toml 修改规范

- 标题需同步修改三处：`title`、`params.homeSubtitle`、`languages.zh-cn.params.subtitle`
- Hugo 版本固定为 `0.159.1`（extended），新增功能先查阅 `themes/hello-friend-ng/README.md`
- 添加新 CSS 文件时追加到 `params.customCSS` 数组

## 主题定制说明

- 覆盖方式：在 `layouts/partials/` 创建同名文件即可
- 当前已覆盖 4 个文件（见项目结构）
- 可覆盖的 partial：`header.html`、`footer.html`、`head.html`、`javascript.html`、`menu.html`、`logo.html`、`social-icons.html`、`sharing-buttons.html`、`tags.html`、`categories.html`、`pagination-list.html`、`pagination-single.html`、`subtitle.html`
- 未启用的主题功能：Disqus 评论、utterances 评论、Plausible/GA 分析、Git commit 信息、Series 分类、Cover 封面图、Audio 音频

## 发布流程

### 方式一：Obsidian 一键发布

在 Obsidian Shell Commands 插件中配置：
```bash
bash /Users/chenzq/git/Andy1314Chen.github.io/scripts/publish.sh "{{file_path:absolute}}"
```
脚本会自动去除中文路径转义、复制文件到 `content/posts/`、git commit + push。

### 方式二：命令行发布

```bash
hugo new content posts/my-new-post.md   # 1. 创建文章
# 2. 编辑 front matter + 正文
# 3. draft: false
hugo --minify                            # 4. 验证构建
git add . && git commit -m "publish: my-new-post" && git push  # 5. 推送
```

## 常见问题排查

| 问题 | 排查方向 |
|------|----------|
| 文章未出现在首页 | `draft` 是否为 `false`；`date` 格式是否正确 |
| 页面数量异常 | 是否有空 categories；是否有缺失 front matter 的文件 |
| 数学公式不渲染 | 确认 `params.math = true`；块级公式 `$$...$$` 独占一行 |
| TOC 不显示 | 确认 `toc: true`；屏幕宽度需 > 1200px（小屏自动隐藏） |
| blockquote 引号溢出 | `toc.css` 中已有 `blockquote { position: relative }` 修复 |
| Obsidian 发布失败 | 检查中文路径转义问题，`publish.sh` 已用 `sed` 处理 |

## 注意事项

- `public/` 在 .gitignore 中，由 CI 自动生成
- `.obsidian/` 在 .gitignore 中，防止编辑器配置被提交
- `backup/` 不参与构建
- 主题为 submodule，clone 后需 `git submodule update --init --recursive`
- 无测试框架，构建无 ERROR 即视为通过
