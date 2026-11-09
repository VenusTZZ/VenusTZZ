<script setup>
import { computed, onMounted, watch } from 'vue'
import { useData } from 'vitepress'
import notes from 'virtual:notes'
import { useLang } from '../composables/useLang.js'

const copy = {
  zh: {
    eyebrow: 'DeepSpeed · Ascend NPU',
    title: 'PR 学习笔记',
    intro:
      '把合并进 DeepSpeed 的 PR 拆成了可以点的交互页。每页的结论与数据都来自 PR / issue 原文，能算的地方尽量在浏览器里实时算。',
    back: '← 返回首页',
    open: '打开交互页',
    emptyPrefix: '还没有笔记。把 HTML 放进 ',
    emptySuffix: '，重新构建即可自动收录。',
    foot: '交互页是独立的静态文档，会在新标签页打开；关掉标签页即可回到这里。',
    switchLabel: 'EN',
    switchAria: 'Switch to English',
  },
  en: {
    eyebrow: 'DeepSpeed · Ascend NPU',
    title: 'PR Study Notes',
    intro:
      'Interactive breakdowns of DeepSpeed PRs. Every claim and number comes from the PR / issue itself; wherever possible, the math runs live in your browser.',
    back: '← Home',
    open: 'Open page',
    emptyPrefix: 'No notes yet. Drop HTML files into ',
    emptySuffix: ' and they will be picked up on the next build.',
    foot: 'Each explainer is a standalone static document that opens in a new tab; close it to return here.',
    switchLabel: '中文',
    switchAria: '切换到中文',
  },
}

const { lang, toggleLang } = useLang()
const { site } = useData()

const t = computed(() => copy[lang.value])
const isEn = computed(() => lang.value === 'en')

function noteTitle(note) {
  return isEn.value && note.titleEn ? note.titleEn : note.title
}

function noteDescription(note) {
  return isEn.value && note.descriptionEn ? note.descriptionEn : note.description
}

function noteSeries(note) {
  return isEn.value && note.seriesEn ? note.seriesEn : note.series
}

function noteTags(note) {
  return isEn.value && note.tagsEn.length ? note.tagsEn : note.tags
}

const docTitle = computed(() => `${t.value.title} | ${site.value.title}`)

onMounted(() => {
  document.title = docTitle.value
})

watch(docTitle, (title) => {
  document.title = title
})
</script>

<template>
  <div class="notes-page">
    <header class="notes-top">
      <a class="notes-brand" href="../"><span class="notes-dot" aria-hidden="true"></span>Venus</a>
      <div class="notes-tools">
        <button
          type="button"
          class="notes-lang"
          :aria-label="t.switchAria"
          @click="toggleLang"
        >{{ t.switchLabel }}</button>
        <a class="notes-back" href="../">{{ t.back }}</a>
      </div>
    </header>

    <main class="notes-main">
      <p class="notes-eyebrow">{{ t.eyebrow }}</p>
      <h1>{{ t.title }}</h1>
      <p class="notes-intro">{{ t.intro }}</p>

      <div v-if="notes.length" class="notes-grid">
        <a
          v-for="note in notes"
          :key="note.slug"
          class="note-card"
          :style="{ '--note-accent': note.accent }"
          :href="note.href"
          target="_blank"
          rel="noopener"
        >
          <div class="note-meta">
            <span class="note-pr">{{ note.prLabel }}</span>
            <span v-if="noteSeries(note)">{{ noteSeries(note) }}</span>
          </div>
          <h2 :lang="isEn && note.titleEn ? 'en' : 'zh-CN'">{{ noteTitle(note) }}</h2>
          <p
            v-if="noteDescription(note)"
            :lang="isEn && note.descriptionEn ? 'en' : 'zh-CN'"
          >{{ noteDescription(note) }}</p>
          <div v-if="noteTags(note).length" class="note-tags">
            <span v-for="tag in noteTags(note)" :key="tag">{{ tag }}</span>
          </div>
          <span class="note-open">{{ t.open }} <span aria-hidden="true">↗</span></span>
        </a>
      </div>

      <p v-else class="notes-empty">{{ t.emptyPrefix }}<code>docs/public/notes/</code>{{ t.emptySuffix }}</p>

      <p class="notes-foot">{{ t.foot }}</p>
    </main>
  </div>
</template>

<style scoped>
.notes-page {
  min-height: 100vh;
  padding: 0 24px 80px;
  color: var(--vp-c-text-1);
  background:
    radial-gradient(ellipse 90% 55% at 50% -12%, rgba(127, 236, 173, 0.09), transparent 70%),
    var(--vp-c-bg);
}

.notes-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1040px;
  height: 72px;
  margin: 0 auto;
  border-bottom: 1px solid var(--vp-c-divider);
}

.notes-brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.notes-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--vp-c-brand);
  box-shadow: 0 0 12px var(--vp-c-brand);
}

.notes-back {
  font-size: 13px;
  color: var(--vp-c-text-2);
  text-decoration: none;
  transition: color 0.25s;
}

.notes-back:hover {
  color: var(--vp-c-brand-light);
}

.notes-tools {
  display: flex;
  align-items: center;
  gap: 14px;
}

.notes-lang {
  padding: 0 12px;
  line-height: 28px;
  border: 1px solid rgba(192, 235, 215, 0.45);
  border-radius: 999px;
  background: transparent;
  color: var(--vp-c-brand-light);
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 0.25s, background-color 0.25s, color 0.25s;
}

.notes-lang:hover {
  border-color: var(--vp-c-brand-light);
  background: rgba(192, 235, 215, 0.08);
  color: #eafff4;
}

.notes-lang:focus-visible {
  outline: 2px solid var(--vp-c-brand);
  outline-offset: 2px;
}

.notes-main {
  max-width: 1040px;
  margin: 0 auto;
  padding-top: 76px;
}

.notes-eyebrow {
  margin: 0 0 18px;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12.5px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--vp-c-brand);
}

.notes-main h1 {
  margin: 0 0 22px;
  font-size: clamp(36px, 6vw, 58px);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.02em;
}

.notes-intro {
  max-width: 620px;
  margin: 0;
  font-size: 16px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
}

.notes-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 52px;
}

.note-card {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 26px 26px 22px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(192, 235, 215, 0.045), rgba(192, 235, 215, 0.012));
  color: inherit;
  text-decoration: none;
  transition: transform 0.3s ease, border-color 0.3s, background-color 0.3s;
}

.note-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 2px;
  background: var(--note-accent, var(--vp-c-brand));
  opacity: 0.55;
  transition: width 0.3s, opacity 0.3s;
}

.note-card:hover {
  transform: translateY(-3px);
  border-color: var(--note-accent, var(--vp-c-brand));
}

.note-card:hover::before {
  width: 3px;
  opacity: 1;
}

.note-card:focus-visible {
  outline: 2px solid var(--note-accent, var(--vp-c-brand));
  outline-offset: 3px;
}

.note-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 11.5px;
  letter-spacing: 0.02em;
  color: var(--vp-c-text-3);
}

.note-pr {
  padding: 2px 9px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  color: var(--note-accent, var(--vp-c-brand));
}

.note-card h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  line-height: 1.45;
  letter-spacing: -0.01em;
  color: var(--vp-c-text-1);
}

.note-card p {
  flex: 1;
  margin: 0;
  font-size: 14px;
  line-height: 1.75;
  color: var(--vp-c-text-2);
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.note-tags span {
  padding: 3px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--vp-c-text-3);
}

.note-open {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12.5px;
  color: var(--note-accent, var(--vp-c-brand));
  transition: gap 0.25s;
}

.note-card:hover .note-open {
  gap: 10px;
}

.notes-empty {
  margin: 52px 0 0;
  font-size: 14px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
}

.notes-empty code {
  padding: 1px 6px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.9em;
}

.notes-foot {
  margin: 36px 0 0;
  font-size: 12.5px;
  color: var(--vp-c-text-3);
}

@media (max-width: 760px) {
  .notes-page {
    padding-bottom: 56px;
  }

  .notes-tools {
    gap: 10px;
  }

  .notes-main {
    padding-top: 52px;
  }

  .notes-grid {
    grid-template-columns: 1fr;
    margin-top: 38px;
  }

  .note-card {
    padding: 22px 22px 20px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .note-card,
  .note-open {
    transition: none;
  }

  .note-card:hover {
    transform: none;
  }
}
</style>
