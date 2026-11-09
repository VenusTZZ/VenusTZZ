# PR 学习笔记 · 交互页目录

把独立的 HTML 交互页放进这个目录，博客的 `/notes/` 索引会在**构建时自动收录**，不需要改代码。

## 卡片元信息（可选）

在页面的 `<head>` 里加 `note:*` meta，索引页会优先使用它们；没写也能自动推断：

| meta | 说明 |
| --- | --- |
| `note:title` | 卡片标题，缺省取页面 `<h1>` |
| `note:description` | 一两句话的摘要，缺省取 `<h1>` 后的第一段 |
| `note:pr` | PR 号（纯数字），缺省从标题或正文里的 `PR #xxxx` 推断 |
| `note:series` | 卡片右上角的来源，例如 `DeepSpeed · ZeRO++` |
| `note:tags` | 标签，用竖线或逗号分隔 |
| `note:accent` | 卡片强调色，形如 `#4fd1a5`；缺省按文件名稳定取色 |

示例：

```html
<meta name="note:pr" content="8627">
<meta name="note:series" content="DeepSpeed · ZeRO++">
<meta name="note:accent" content="#4fd1a5">
<meta name="note:tags" content="hpZ int8|qgZ int4|payload −50%">
<meta name="note:description" content="一两句话的摘要。">
```

## 排序与命名

- 有 `note:pr` 的按 PR 号从大到小排；没有的排在后面，按标题排序。
- 文件名建议用 ASCII slug（如 `npu-quant-comm.html`）；中文文件名也能用，链接会自动做 URL 编码。
