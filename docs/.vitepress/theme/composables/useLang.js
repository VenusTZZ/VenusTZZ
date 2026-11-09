import { onMounted, ref, watch } from 'vue'

/**
 * 全站共享的中英文状态。
 * 存在 localStorage 的 `lang` 键里，首页和笔记页共用同一个选择。
 */
export function useLang() {
  const lang = ref('en')

  function applyDocumentLang(value) {
    document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en-US'
  }

  onMounted(() => {
    lang.value = localStorage.getItem('lang') === 'zh' ? 'zh' : 'en'
    applyDocumentLang(lang.value)
  })

  watch(lang, (value) => {
    localStorage.setItem('lang', value)
    applyDocumentLang(value)
  })

  function toggleLang() {
    lang.value = lang.value === 'zh' ? 'en' : 'zh'
  }

  return { lang, toggleLang }
}
