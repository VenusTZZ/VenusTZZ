/**
 * 给一对中英文交互页注入语言切换按钮。
 *
 * 用法：
 *   node scripts/inject-lang-switch.mjs <zh.html> <en.html>
 *
 * 两个文件必须位于同一目录。脚本会在 <head> 注入共享样式，
 * 在 </body> 前注入固定右下角的切换按钮与共享脚本；重复执行不会叠加。
 */
import fs from 'node:fs'
import path from 'node:path'

const [zhFile, enFile] = process.argv.slice(2)
if (!zhFile || !enFile) {
  console.error('usage: node scripts/inject-lang-switch.mjs <zh.html> <en.html>')
  process.exit(1)
}

const START = '<!-- lang-switch:start -->'
const END = '<!-- lang-switch:end -->'
const CSS_TAG = '<link rel="stylesheet" href="./lang-switch.css">'
const JS_TAG = '<script src="./lang-switch.js"></script>'

function escapeAttr(value) {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
}

function inject(file, otherFile, targetLang, label, ariaLabel) {
  let html = fs.readFileSync(file, 'utf8')

  // 幂等：先移除旧块
  html = html.replace(new RegExp(`${START}[\\s\\S]*?${END}\\s*`, 'g'), '')

  if (!html.includes('lang-switch.css')) {
    html = html.replace(/<\/head>/i, `${CSS_TAG}\n</head>`)
  }

  const href = './' + encodeURIComponent(path.basename(otherFile))
  const hreflang = targetLang === 'en' ? 'en' : 'zh-CN'
  const block = [
    START,
    `<a class="lang-switch" href="${escapeAttr(href)}" hreflang="${hreflang}" data-lang="${targetLang}" aria-label="${escapeAttr(ariaLabel)}">${label}</a>`,
    JS_TAG,
    END,
    '',
  ].join('\n')

  html = html.replace(/<\/body>/i, `${block}</body>`)
  fs.writeFileSync(file, html)
  console.log(`injected ${path.basename(file)} -> ${href}`)
}

inject(zhFile, enFile, 'en', 'EN', 'Switch to English')
inject(enFile, zhFile, 'zh', '中文', '切换到中文')
