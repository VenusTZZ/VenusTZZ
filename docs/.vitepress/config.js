import { defineConfig } from 'vitepress'
export default defineConfig({
  title: 'Venus', //标题
  titleTemplate: 'Blog', //标题模板
  description: 'XXG-BLOG', //描述
  lastUpdated: true, //开启上次更新时间
  vue: {
    reactivityTransform: true,
  },
  markdown: {
    theme: 'nord',
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
  ],
  themeConfig: {
    siteTitle: 'VenusBlog',
    logo: '/favicon.svg',
    description: 'Venus的博客',
    // themeConfig: {},
    nav: [
      { text: '主页', link: '/' },
      { text: '计划', link: '/Myplan/' },
      { text: '习作', link: '/learn/' },
      { text: '课题', link: '/classwork/' },
      { text: 'LeetCode', link: '/leetcode/' },
      { text: 'Entertainment', link: '/entertainment/' },
      // {
      //   text: '分类',
      //   items: [
      //     { text: '习作', link: '/learn/', activeMatch: '/learn/' },
      //     { text: '课题', link: '/classwork/', activeMatch: '/classwork/' },
      //     { text: 'LeetCode', link: '/leetcode/', activeMatch: '/leetcode/' },
      //     { text: 'Entertainment', link: '/entertainment/', activeMatch: '/entertainment/' },
      //   ],
      // },
    ],
    sidebar: {
      '/learn/': [
        {
          text: '我的习作',
          collapsible: true,
          collapsed: false,
          items: [
            // This shows `/config/index.md` page.
            { text: 'Fluent emoji Maker', link: '/learn/' },
            { text: 'Vue2-Vuetify', link: '/learn/02.md' },
            { text: 'MineSweeper', link: '/learn/03.md' },
            { text: 'nodejs', link: '/learn/04.md' },
            { text: 'Vue3 实用工具', link: '/learn/05MyVuePlugins.md' },
            { text: '素材使用', link: '/learn/99Element.md' },
          ],
        },
      ],
      '/classwork/': [
        {
          text: '课题相关',
          collapsible: true,
          collapsed: false,
          items: [
            { text: '已经做的事', link: '/classwork/' },
            { text: 'Bert-BiLSTM-CRF', link: '/classwork/01.md' },
            { text: 'doccano', link: '/classwork/02.md' },
            { text: 'Others', link: '/classwork/99.md' },
          ],
        },
      ],
      '/leetcode/': [
        {
          text: 'LeetCode',
          collapsible: true,
          collapsed: false,
          items: [
            { text: 'page1', link: '/leetcode/' },
            { text: 'page2', link: '/leetcode/page2.md' },
          ],
        },
      ],
      // '/entertainment/': [
      //   {
      //     text: '首页',
      //     collapsible: true,
      //     collapsed: false,
      //     items: [
      //       { text: '介绍', link: '/entertainment/' },
      //     ],
      //   },
      //   {
      //     text: 'section1',
      //     collapsible: true,
      //     collapsed: false,
      //     items: [
      //       { text: '文弱书生', link: '/entertainment/t2-p1.md' },
      //       { text: 'Lost Paradise', link: '/entertainment/t2-p2.md' },
      //     ],
      //   },
      // ],
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/VenusTZZ' }],
    base: './',
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2022-present Venus',
    },
    // carbonAds: {
    //   code: 'your-carbon-code',
    //   placement: 'your-carbon-placement',
    // },
  },
})
