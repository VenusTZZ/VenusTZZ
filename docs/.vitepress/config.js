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
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }]],
  themeConfig: {
    siteTitle: 'VenusBlog',
    logo: '/favicon.svg',
    description: 'Venus的博客',
    themeConfig: {},
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
      // This sidebar gets displayed when user is
      // under `guide` directory.
      // '/Myplan/': [
      //   {
      //     text: '计划',
      //     collapsible: true,
      //     collapsed: false,
      //     items: [
      //       // This shows `/config/index.md` page.
      //       { text: 'Fluent emoji Maker', link: '/learn/' },
      //       { text: 'PatientDashboard', link: '/learn/page2.md' }, // /config/index.md
      //       { text: '素材使用', link: '/learn/Element.md' },
      //     ],
      //   },
      // ],
      '/learn/': [
        {
          text: '我的习作',
          collapsible: true,
          collapsed: false,
          items: [
            // This shows `/config/index.md` page.
            { text: 'Fluent emoji Maker', link: '/learn/' },
            { text: 'PatientDashboard', link: '/learn/page2.md' }, // /config/index.md
            { text: '素材使用', link: '/learn/Element.md' },
          ],
        },
      ],

      // This sidebar gets displayed when user is
      // under `config` directory.
      '/classwork/': [
        {
          text: '课题相关',
          collapsible: true,
          collapsed: false,
          items: [
            // This shows `/config/index.md` page.
            { text: '已经做的事', link: '/classwork/' },
            { text: '故障推荐', link: '/classwork/page2.md' }, // /config/index.md
            { text: '知识图谱故障推荐', link: '/classwork/page3.md' },
            { text: 'Bert实体识别', link: '/classwork/page4.md' },
            { text: 'Bert-BiLSTM-crf实体识别', link: '/classwork/page5.md' },
          ],
        },
      ],
      '/leetcode/': [
        {
          text: 'LeetCode',
          collapsible: true,
          collapsed: false,
          items: [
            // This shows `/config/index.md` page.
            { text: 'page1', link: '/leetcode/' },
            { text: 'page2', link: '/leetcode/page2.md' }, // /config/index.md
          ],
        },
      ],
      // '/entertainment/': [
      //   {
      //     text: '首页',
      //     collapsible: true,
      //     collapsed: false,
      //     items: [
      //       // This shows `/config/index.md` page.
      //       { text: '介绍', link: '/entertainment/' },
      //     ],
      //   },
      //   {
      //     text: 'section1',
      //     collapsible: true,
      //     collapsed: false,
      //     items: [
      //       // This shows `/config/index.md` page.
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
    carbonAds: {
      code: 'your-carbon-code',
      placement: 'your-carbon-placement',
    },
  },
})
