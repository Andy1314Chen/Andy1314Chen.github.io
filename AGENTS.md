# AGENTS.md

AI 编码助手操作手册。本项目是 Hugo 静态博客（hello-friend-ng 主题），无测试框架，构建成功即视为验证通过。

## 常用命令

```bash
hugo server -D          # 本地预览（含草稿）
hugo server             # 仅已发布文章
hugo --minify           # 生产构建

# 验证构建
hugo --minify 2>& | grep -i error   # 无 ERROR 即通过
hugo --minify 2>& | grep "Pages"    # 当前约 33 页

# 新建文章
hugo new content posts/<slug>.md      # 基于 archetypes/posts.md 生成
```

**部署**：push 到 master 分支后 CI 自动构建部署，无需手动生成 public/。

## 项目结构

```
content/posts/              # 博客文章（Markdown）
content/about.md            # 关于页
layouts/
  index.html               # 首页模板覆盖
  partials/favicons.html   # Favicon partial 覆盖
static/favicon.svg         # SVG favicon
archetypes/
  default.md               # 通用模板
  posts.md                # posts 目录模板
themes/hello-friend-ng/    # 主题 submodule，勿直接修改
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

**规则**：`title` 必填双引号，`date` 格式 `YYYY-MM-DD`，`description` 必须单行纯文本，`draft: false` 才发布。

## Markdown 规范

- 图片：使用 GitHub 图床完整 URL `![描述](https://raw.githubusercontent.com/Andy1314Chen/obsidian-pic/main/image/xxx.png)`
- 代码块：必须标注语言 ` ```python `
- 数学：行内 `$公式$`，块级 `$$公式$$`
- 图表：` ```mermaid ` 代码块

## Go Template 规范

- 4 空格缩进，`{{- ` 和 ` -}}` 去除渲染空白
- 条件：`{{- with .Params.xxx }}...{{- end }}`
- 循环：`{{ range .Pages }}...{{ end }}`
- 覆盖主题：在 `layouts/` 创建同名文件（禁止修改 `themes/`）

常用变量：`.Site.Title`、`.Site.RegularPages`、`.Permalink`、`.Params.tags`、`.Params.toc`、`.Date`、`.ReadingTime`

## hugo.toml 规范

- 标题同步修改 `title`、`params.homeSubtitle`、`languages.zh-cn.params.subtitle` 三处
- Hugo 版本固定为 `0.159.1`（extended），新增功能查阅 `themes/hello-friend-ng/README.md`

## CI/CD

- 触发：push 到 master
- 构建：`hugo --minify --baseURL "${{ steps.pages.outputs.base_url }}/"`
- Hugo：0.159.1（extended），部署至 GitHub Pages
- 查看：https://github.com/Andy1314Chen/Andy1314Chen.github.io/actions

## 主题定制说明

- 主题 partial 覆盖：在 `layouts/partials/` 创建同名文件即可
- 当前已覆盖：`layouts/partials/favicons.html`（SVG favicon 支持）
- 主题 partial 列表：`header.html`、`footer.html`、`head.html`、`javascript.html`、`menu.html`、`logo.html`、`social-icons.html`、`sharing-buttons.html`、`tags.html`、`categories.html`、`pagination-list.html`、`pagination-single.html`、`subtitle.html`
- 主题 SCSS 变量：`$light-background`、`$dark-background`、`$light-color`、`$dark-color`、`$max-width: 860px`
- 主题支持的功能（未启用）：Disqus 评论、utterances 评论、Plausible/GA 分析、Git commit 信息、自定义 CSS/JS、Series 分类、Cover 封面图、Audio 音频

## 常见构建问题排查

**文章没有出现在首页**
- 检查 front matter 中 `draft` 是否为 `false`
- 检查 date 日期格式是否为 `YYYY-MM-DD`

**页面数量异常**
- 检查是否有空 categories 分类（当前已移除 categories 分类）
- 检查是否有缺失 front matter 的文件

**数学公式不渲染**
- 确认 `params.math = true` 已启用
- 块级公式使用 `$$...$$`（注意独占一行）

**TOC 目录不显示**
- 确认 front matter 中 `toc: true`
- 仅在 80+ 行长文中推荐启用

## 文章发布流程

1. `hugo new content posts/my-new-post.md` 创建文章
2. 填写 front matter（title、date、tags、description）
3. 编写正文内容
4. 设置 `draft: false`
5. 长文（80+ 行）建议设置 `toc: true`
6. `hugo --minify` 本地验证构建
7. `git add . && git commit -m "描述" && git push`
8. CI 自动构建部署，约 1-2 分钟生效

## 注意事项

- `public/` 在 .gitignore 中，由 CI 自动生成
- `backup/` 不参与构建
- 主题为 submodule，clone 后需 `git submodule update --init --recursive`
- 无测试框架，构建无 ERROR 即视为通过
- 中文回复，言简意赅
