博客采用 Gmeek 框架部署在 GitHub 上，首先本地在 Obsidian 中利用 Markdown 写，再复制粘贴到 GitHub 中 Issue 中，最后自动发布。

因此，期望的工作流程:，图片在本地 Obsidian 写的时候，自动上传图床并生成链接，嵌入在 Markdown 文档中。

参考链接: [使用 PicGo + GitHub 搭建 Obsidian 图床 | Leehow 的小站 (haoyep.com)](https://www.haoyep.com/posts/github-graph-beds/) 配置，大概包括 3 个步骤：

第一步: 创建 GitHub 公开仓库作为图床仓库。

第二步: 下载并安装 PicGo，其支持 Windows、MacOS、Linux，配置 GitHub 图床。

第三步: Obsidian 中安装第三方插件: `Image auto upload Plugin`.

下图为配置后自动上传图床并生成链接的例子，链接中仓库即是第一步中创建的图床仓库。

![测试图床功能是否正常](https://raw.githubusercontent.com/Andy1314Chen/obsidian-pic/main/image/20241014164959.png)
