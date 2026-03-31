# John Chen's Blog

个人技术博客，基于 Hugo + hello-friend-ng 主题，部署于 GitHub Pages。

**博客地址**：https://Andy1314Chen.github.io

## 技术栈

- [Hugo](https://gohugo.io/)（静态网站生成器，extended 版本）
- [hello-friend-ng](https://github.com/rhazdon/hugo-theme-hello-friend-ng)（主题）

## 本地开发

```bash
# 启动预览服务器（含草稿）
hugo server -D

# 生产构建
hugo --minify
```

## 部署

推送至 `master` 分支后，GitHub Actions 自动构建部署。

## 文章规范

- Front matter 使用 YAML 格式，`draft: false` 才发布
- 图片使用 GitHub 图床完整 URL
- 新建文章：`hugo new content posts/<slug>.md`

详见 [AGENTS.md](AGENTS.md)。
