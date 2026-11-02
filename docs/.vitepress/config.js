import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/',
  title: 'Venus',
  description: 'Venus 的个人主页',
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
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
