# PR 交互页英文化规范

目标：为 `docs/public/notes/` 下的每个中文交互页生成一个英文副本，结构、样式和交互逻辑与中文版完全一致，只翻译**用户能看到或由 JS 输出的文字**。

## 文件约定

- 源文件：`docs/public/notes/<slug>.html`（中文，**不要修改**）
- 英文副本：`docs/public/notes/<slug>.en.html`
- 英文副本不会被笔记索引单独收录；索引会根据当前语言链接到对应版本。
- 不要手动添加语言切换按钮，仓库脚本会统一注入。

## 硬性要求

1. **结构不变**：不增删 DOM 节点、不改 `id` / `class` / `data-*` 属性、不改缩进风格。
2. **逻辑不变**：不改变 JavaScript 变量名、条件、数值、函数调用、事件绑定；只替换会展示给用户的字符串。
3. **不要翻译**：代码标识符、文件路径、API 名称、PR/issue 编号、公式、命令、模型名、产品名、`code` 里的内容。
4. **翻译所有可见文案**：
   - HTML 文本节点：标题、导航、段落、按钮、表格、图注、footer、SVG `<text>`
   - 属性：`title`、`alt`、`aria-label`、`placeholder`、`data-tip` 等
   - JS 中会进入 DOM 的字符串：`textContent`、`innerHTML`、模板字符串、步骤/状态数组、图表 legend 与轴标签、提示语、确认框
   - `<html lang="zh-CN">` 改成 `lang="en"`；`<title>` 改成英文；`<meta name="description">` 如有则翻译
5. **保留中文注释**可以接受（不影响显示）；若顺手翻译注释也可以，但不要为此改代码。
6. 英文要通顺、技术化、简洁，不要逐字直译；统一术语。

## 术语表（优先使用）

| 中文 | English |
| --- | --- |
| 昇腾 NPU | Ascend NPU |
| 量化通信 | quantized communication |
| 权重 / 梯度 | weights / gradients |
| 分区 / 分片 | partition / shard |
| 合并桶 | coalesced bucket |
| 反量化 | dequantize |
| 冻结权重 | frozen weights |
| 图捕获 / 重放 | graph capture / replay |
| 空图重放 | empty graph replay |
| 张量并行 | tensor parallelism (TP) |
| 非连续梯度 | non-contiguous gradients |
| 设备守卫 | device guard |
| 加速器无关 | accelerator-agnostic |
| 旋转位置编码 | rotary position embedding (RoPE) |
| KV 缓存 | KV cache |
| 缓存重建 | cache rebuild |
| 生成 token | generated tokens |
| 融合优化器 | fused optimizer |
| 契约 | contract |
| 逐比特对齐 | bit-for-bit aligned |
| 静默失效 | silent failure |
| 回写 | write-back |
| 梯度裁剪 / 范数 | gradient clipping / norm |
| 前向 / 反向 | forward / backward |
| 吞吐 / 延迟 | throughput / latency |
| 显存 | device memory |
| 训练 / 推理 | training / inference |
| 端到端 | end-to-end |
| 基线 | baseline |
| 开销 | overhead |

## 工作流程

1. 完整读一遍中文源文件，理解页面结构和交互。
2. 复制为 `<slug>.en.html`，然后逐段替换可见文案。
3. 用 `rg -n "[\u4e00-\u9fff]" <slug>.en.html` 自查剩余中文，区分“注释/代码/必须保留”与“漏翻的可见文案”。
4. 确认没有 JS 报错：用无头 Chrome 或至少检查标签闭合。
5. 汇报：翻译了什么、哪些中文是有意保留的、有没有不确定的术语。

## 验收参考

- 英文页面在浏览器中不应出现中文正文，除了即将注入的「中文」切换按钮。
- 所有交互控件点击后仍正常工作。
- 与中文版本的 diff 应主要是文本变化。
