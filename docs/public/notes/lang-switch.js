/* 记录语言选择；索引页与首页共用 localStorage.lang */
(function () {
  var link = document.querySelector('.lang-switch')
  if (!link) return
  link.addEventListener('click', function () {
    try {
      localStorage.setItem('lang', link.getAttribute('data-lang') || '')
    } catch (e) {
      /* 隐私模式下忽略 */
    }
  })
})()
