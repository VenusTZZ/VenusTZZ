<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import BackGround from './BackGround.vue'

const copy = {
  zh: { text: 'Hi 🖐️', tagline: '在代码里种点东西' },
  en: { text: 'Hi 🖐️', tagline: 'Planting seeds in code' },
}

const lang = ref('zh')
const t = computed(() => copy[lang.value])

onMounted(() => {
  lang.value = localStorage.getItem('lang') === 'en' ? 'en' : 'zh'
  document.documentElement.lang = lang.value === 'zh' ? 'zh-CN' : 'en'
})

watch(lang, (val) => {
  document.documentElement.lang = val === 'zh' ? 'zh-CN' : 'en'
  localStorage.setItem('lang', val)
})

function toggleLang() {
  lang.value = lang.value === 'zh' ? 'en' : 'zh'
}
</script>

<template>
  <div class="home-hero">
    <BackGround />
    <button class="lang-toggle" data-avoid @click="toggleLang">
      {{ lang === 'zh' ? 'EN' : '中文' }}
    </button>
    <main class="hero-main" data-avoid>
      <img class="avatar" :src="withBase('/Saly-19.svg')" alt="Venus" />
      <h1 class="name"><span class="clip">Venus</span></h1>
      <p class="text">{{ t.text }}</p>
      <p class="tagline">{{ t.tagline }}</p>
      <div class="actions">
        <a class="btn" href="https://github.com/VenusTZZ">GitHub</a>
      </div>
    </main>
    <footer class="footer" data-avoid>© 2026 Venus</footer>
  </div>
</template>

<style scoped>
.home-hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.hero-main {
  margin: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px;
  text-align: center;
  animation: hero-in 0.7s ease-out both;
}

.avatar {
  width: 150px;
  border-radius: 50%;
  margin-bottom: 24px;
  transition: filter 0.3s;
  will-change: filter;
}

.avatar:hover {
  filter: drop-shadow(0 0 2em #bd34fe);
}

.name {
  font-size: 72px;
  line-height: 1.1;
  letter-spacing: 0.06em;
}

.clip {
  background: -webkit-linear-gradient(120deg, #bd34fe, #41d1ff, #7fecad);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.text {
  margin-top: 8px;
  font-size: 22px;
  color: var(--vp-c-text-2);
}

.tagline {
  margin-top: 14px;
  font-size: 16px;
  color: var(--vp-c-text-2);
}

.actions {
  margin-top: 36px;
}

.btn {
  display: inline-block;
  border-radius: 999px;
  padding: 0 26px;
  line-height: 42px;
  font-size: 14px;
  background: #c0ebd7;
  color: #0d1512;
  transition: background-color 0.25s;
}

.btn:hover {
  background: #d7f4e6;
}

.lang-toggle {
  position: fixed;
  top: 20px;
  right: 24px;
  z-index: 10;
  border: 1px solid rgba(192, 235, 215, 0.45);
  background: transparent;
  color: #c0ebd7;
  border-radius: 999px;
  padding: 0 14px;
  line-height: 30px;
  font-size: 13px;
  cursor: pointer;
  transition: background-color 0.25s, border-color 0.25s, color 0.25s;
}

.lang-toggle:hover {
  border-color: #c0ebd7;
  background: rgba(192, 235, 215, 0.08);
  color: #eafff4;
}

.footer {
  padding: 20px 24px;
  text-align: center;
  font-size: 12px;
  color: var(--vp-c-text-3);
}

@keyframes hero-in {
  from {
    opacity: 0;
    transform: translateY(14px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .hero-main {
    animation: none;
  }
}

@media (max-width: 640px) {
  .name {
    font-size: 52px;
  }
}
</style>
