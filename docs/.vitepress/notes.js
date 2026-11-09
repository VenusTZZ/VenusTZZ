/**
 * PR 笔记索引的构建期数据源。
 *
 * 约定：把独立的 HTML 交互页放进 docs/public/notes/，索引页会自动收录，
 * 不需要改任何代码。元信息优先读取 <head> 里的 <meta name="note:*">，
 * 缺失时从 <h1>、正文首段、<title> 中尽力推断：
 *
 *   <meta name="note:title"       content="卡片标题">
 *   <meta name="note:description" content="一两句话的摘要">
 *   <meta name="note:pr"          content="8627">
 *   <meta name="note:series"      content="DeepSpeed · ZeRO++">
 *   <meta name="note:tags"        content="int8|int4|0 新增失败">  (用 | 或 , 分隔)
 *   <meta name="note:accent"      content="#4fd1a5">
 *
 * 英文版可选，缺失时回退到中文：
 *
 *   <meta name="note:title:en"       content="English title">
 *   <meta name="note:description:en" content="English summary">
 *   <meta name="note:series:en"      content="DeepSpeed · Triton">
 *   <meta name="note:tags:en"        content="3 gate layers|591 tests">
 *
 * 页面按 PR 号从大到小排序；没有 PR 号的排在后面，按标题排序。
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const NOTES_DIR = fileURLToPath(new URL('../public/notes', import.meta.url))
const VIRTUAL_ID = 'virtual:notes'
const RESOLVED_ID = '\0' + VIRTUAL_ID

const ACCENTS = ['#4fd1a5', '#7fb3ff', '#e8a44a', '#b494ff', '#5cc8e0', '#e5685c']

const ENTITIES = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  mdash: '—',
  ndash: '–',
  hellip: '…',
  times: '×',
  middot: '·',
  rarr: '→',
  larr: '←',
}

function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (raw, body) => {
    if (body[0] === '#') {
      const hex = body[1] === 'x' || body[1] === 'X'
      const code = parseInt(hex ? body.slice(2) : body.slice(1), hex ? 16 : 10)
      return Number.isFinite(code) && code >= 0 && code <= 0x10ffff ? String.fromCodePoint(code) : raw
    }
    return ENTITIES[body.toLowerCase()] ?? raw
  })
}

function cleanText(html) {
  const spaced = decodeEntities(
    html.replace(/<br\s*\/?>/gi, ' ').replace(/<[^>]*>/g, ' ')
  )
  return spaced
    .replace(/\s+/g, ' ')
    // `<br>` 换行常常夹在中文中间，别因此多出一个空格
    .replace(/([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])/g, '$1')
    .replace(/([，。、；：！？])\s+/g, '$1')
    .replace(/\s+([，。、；：！？）」』])/g, '$1')
    .trim()
}

function readMeta(html, name) {
  for (const tag of html.match(/<meta\b(?:[^>"']|"[^"]*"|'[^']*')*>/gi) || []) {
    const nameMatch = tag.match(/\bname\s*=\s*("([^"]*)"|'([^']*)')/i)
    if (!nameMatch) continue
    const metaName = (nameMatch[2] ?? nameMatch[3]).trim().toLowerCase()
    if (metaName !== name) continue
    const contentMatch = tag.match(/\bcontent\s*=\s*("([^"]*)"|'([^']*)')/i)
    if (contentMatch) return decodeEntities(contentMatch[2] ?? contentMatch[3]).trim()
  }
  return ''
}

function firstParagraph(html, fromIndex) {
  const match = html.slice(fromIndex).match(/<p\b[^>]*>([\s\S]*?)<\/p>/i)
  return match ? cleanText(match[1]) : ''
}

function findPr(html, title) {
  for (const source of [title, html]) {
    for (const re of [/\bPR\s*#\s*(\d{2,6})/i, /#\s*(\d{3,6})/]) {
      const match = source.match(re)
      if (match) return match[1]
    }
  }
  return ''
}

function truncate(text, max = 180) {
  const chars = Array.from(text)
  return chars.length > max ? chars.slice(0, max).join('') + '…' : text
}

function truncateWords(text, max = 400) {
  if (text.length <= max) return text
  const cut = text.slice(0, max)
  const lastSpace = cut.lastIndexOf(' ')
  return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut) + '…'
}

function splitTags(raw) {
  return raw ? raw.split(/[|,，]/).map((tag) => tag.trim()).filter(Boolean) : []
}

function hashAccent(name) {
  let hash = 0
  for (const char of name) hash = (hash * 31 + char.codePointAt(0)) >>> 0
  return ACCENTS[hash % ACCENTS.length]
}

function parseNote(file, html) {
  const h1 = html.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i)
  const titleTag = html.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i)

  const title =
    readMeta(html, 'note:title') ||
    (h1 ? cleanText(h1[1]) : '') ||
    (titleTag ? cleanText(titleTag[1]) : '') ||
    file.replace(/\.html?$/i, '')

  const description =
    readMeta(html, 'note:description') ||
    (h1 ? firstParagraph(html, h1.index + h1[0].length) : '')

  const prMeta = readMeta(html, 'note:pr')
  const pr = (prMeta.match(/\d+/) || [])[0] || findPr(html, title)

  const tagsRaw = readMeta(html, 'note:tags') || readMeta(html, 'keywords')
  const tags = splitTags(tagsRaw)

  const tagsEnRaw = readMeta(html, 'note:tags:en')
  const tagsEn = splitTags(tagsEnRaw)

  const accentMeta = readMeta(html, 'note:accent')
  const accent = /^#[0-9a-f]{3,8}$/i.test(accentMeta) ? accentMeta : hashAccent(file)

  return {
    slug: file.replace(/\.html?$/i, ''),
    file,
    href: './' + encodeURIComponent(file),
    title,
    titleEn: readMeta(html, 'note:title:en'),
    description: truncate(description),
    descriptionEn: truncateWords(readMeta(html, 'note:description:en'), 400),
    pr,
    prLabel: pr ? `#${pr}` : 'NOTE',
    series: readMeta(html, 'note:series') || (/deepspeed/i.test(html) ? 'DeepSpeed' : ''),
    seriesEn: readMeta(html, 'note:series:en'),
    tags,
    tagsEn,
    accent,
  }
}

export function scanNotes(dir = NOTES_DIR) {
  if (!fs.existsSync(dir)) return []

  const notes = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.html?$/i.test(entry.name))
    .filter((entry) => entry.name.toLowerCase() !== 'index.html')
    .map((entry) => parseNote(entry.name, fs.readFileSync(path.join(dir, entry.name), 'utf8')))

  notes.sort((a, b) => {
    if (a.pr && b.pr) return Number(b.pr) - Number(a.pr)
    if (a.pr) return -1
    if (b.pr) return 1
    return a.title.localeCompare(b.title, 'zh')
  })

  return notes
}

export function notesPlugin() {
  return {
    name: 'venus-notes-manifest',
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_ID
    },
    load(id) {
      if (id !== RESOLVED_ID) return
      const json = JSON.stringify(scanNotes())
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029')
      return `export default ${json}`
    },
    configureServer(server) {
      const isNote = (file) =>
        path.resolve(file).startsWith(NOTES_DIR + path.sep) && /\.html?$/i.test(file)

      const refresh = (file) => {
        if (!isNote(file)) return
        server.moduleGraph.invalidateAll()
        server.ws.send({ type: 'full-reload' })
      }

      server.watcher.add(NOTES_DIR)
      server.watcher.on('add', refresh)
      server.watcher.on('unlink', refresh)
      server.watcher.on('change', refresh)
    },
  }
}
