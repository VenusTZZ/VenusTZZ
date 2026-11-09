import { defineConfig } from 'vitepress'
import { notesPlugin } from './notes.js'

export default defineConfig({
  base: '/VenusTZZ/',
  title: 'Venus',
  description: 'Venus 的个人主页',
  vite: {
    plugins: [notesPlugin()],
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: './favicon.svg' }],
  ],
  themeConfig: {
    siteTitle: 'Venus',
    logo: '/favicon.svg',
    nav: [],
    footer: {
      copyright: '© 2026 Venus',
    },
  },
})
