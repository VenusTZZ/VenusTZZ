---
layout: false
title: PR 学习笔记
description: 几个 DeepSpeed PR 的交互式原理拆解
---

<div class="notes-page">
  <header class="notes-top">
    <a class="notes-brand" href="../"><span class="notes-dot" aria-hidden="true"></span>Venus</a>
    <a class="notes-back" href="../">← 返回首页</a>
  </header>
  <main class="notes-main">
    <p class="notes-eyebrow">DeepSpeed · Ascend NPU</p>
    <h1>PR 学习笔记</h1>
    <p class="notes-intro">把几个合并进 DeepSpeed 的 PR 拆成了可以点的交互页：量化通信、推理正确性、Triton 守卫和图捕获。每页的结论与数据都来自 PR / issue 原文，能算的地方尽量在浏览器里实时算。</p>
    <div class="notes-grid">
      <a class="note-card" style="--note-accent:#4fd1a5" href="./npu-quant-comm.html" target="_blank" rel="noopener">
        <div class="note-meta"><span class="note-pr">#8627</span><span>DeepSpeed · ZeRO++</span></div>
        <h2>在昇腾上补齐 ZeRO++ 的量化通信三件套</h2>
        <p>为 NPU 后端实现 FusedLion、FusedLamb 与 Quantizer，让 Lion/LAMB 训练和 hpZ（int8 权重）、qgZ（int4 梯度）量化通信从初始化崩溃变为可用。纯 PyTorch 实现，契约逐比特对齐 CUDA。</p>
        <div class="note-tags"><span>hpZ int8</span><span>qgZ int4</span><span>4× 910B4</span><span>payload −50%</span></div>
        <span class="note-open">打开交互页 <span aria-hidden="true">↗</span></span>
      </a>
      <a class="note-card" style="--note-accent:#7fb3ff" href="./npu-inference-fix.html" target="_blank" rel="noopener">
        <div class="note-meta"><span class="note-pr">#8776</span><span>DeepSpeed · NPU 推理</span></div>
        <h2>昇腾 NPU 推理路径上的 RoPE 与 KV Cache 修复</h2>
        <p>两个静默正确性 bug 叠加一个 O(n²) 的缓存重建性能问题。修复后 Qwen3-4B 上 32 个生成 token 与 Hugging Face 参考完全一致，32k 上下文端到端 decode 提速 2.4×。</p>
        <div class="note-tags"><span>7 处缺陷</span><span>32/32 tokens</span><span>2.4× decode</span><span>GQA</span></div>
        <span class="note-open">打开交互页 <span aria-hidden="true">↗</span></span>
      </a>
      <a class="note-card" style="--note-accent:#e8a44a" href="./triton-gate.html" target="_blank" rel="noopener">
        <div class="note-meta"><span class="note-pr">#8868</span><span>DeepSpeed · Triton</span></div>
        <h2>让 Triton 的设备守卫做到加速器无关</h2>
        <p>重点不是「让 Triton 支持 NPU」，而是设计一套机制，让每个硬件平台自己回答：后端能不能跑、这个 kernel 能不能跑、跑不了该用什么实现。三层 gate，CUDA 默认路径零变化。</p>
        <div class="note-tags"><span>3 层守卫</span><span>35 files</span><span>591 tests</span><span>0 新增失败</span></div>
        <span class="note-open">打开交互页 <span aria-hidden="true">↗</span></span>
      </a>
      <a class="note-card" style="--note-accent:#b494ff" href="./graph-capture.html" target="_blank" rel="noopener">
        <div class="note-meta"><span class="note-pr">#8933</span><span>DeepSpeed · HybridEngine</span></div>
        <h2>把图捕获从 CUDA 专属变成加速器通用能力</h2>
        <p>HybridEngine 把上千次 kernel launch 压成一次 replay，但两条快路径被硬编码成 CUDA 专属。这页拆解问题根因、捕获工作流，以及让昇腾 NPU 真正跑通的两个关键修复。</p>
        <div class="note-tags"><span>Graph capture</span><span>issue #8932</span><span>+178 / −66</span><span>已合入</span></div>
        <span class="note-open">打开交互页 <span aria-hidden="true">↗</span></span>
      </a>
    </div>
    <p class="notes-foot">交互页是独立的静态文档，会在新标签页打开；关掉标签页即可回到这里。</p>
  </main>
</div>
