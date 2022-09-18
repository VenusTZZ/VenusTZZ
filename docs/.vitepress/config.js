export default {
  title: 'Venus', //标题
  titleTemplate: 'Blog', //标题模板
  description: 'XXG-BLOG', //描述
  lastUpdated: true, //开启上次更新时间
  markdown: {
    theme: 'nord',
  },
  themeConfig: {
    siteTitle: 'VenusBlog',
    logo: '/favicon.svg',
    head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
    description: 'Venus的博客',
    themeConfig: {},
    nav: [
      { text: '主页', link: '/' },
      {
        text: '分类',
        items: [
          { text: '习作', link: '/learn/', activeMatch: '/learn/' },
          { text: '课题', link: '/classwork/', activeMatch: '/classwork/' },
          { text: 'LeetCode', link: '/leetcode/', activeMatch: '/leetcode/' },
        ],
      },
    ],
    sidebar: {
      // This sidebar gets displayed when user is
      // under `guide` directory.
      '/learn/': [
        {
          text: '我的习作',
          collapsible: true,
          collapsed: false,
          items: [
            // This shows `/config/index.md` page.
            { text: 'Fluent emoji Maker', link: '/learn/' },
            { text: 'patientdashboard', link: '/learn/page2.md' }, // /config/index.md
            { text: '素材使用', link: '/learn/Element.md' },
          ],
        },
      ],

      // This sidebar gets displayed when user is
      // under `config` directory.
      '/classwork/': [
        {
          text: '课题相关',
          items: [
            // This shows `/config/index.md` page.
            { text: '故障推荐', link: '/classwork/' },
            { text: '已经做的事', link: '/classwork/page2.md' }, // /config/index.md
          ],
        },
      ],
      '/leetcode/': [
        {
          text: 'LeetCode',
          items: [
            // This shows `/config/index.md` page.
            { text: 'page1', link: '/leetcode/' },
            { text: 'page2', link: '/leetcode/page2.md' }, // /config/index.md
          ],
        },
      ],
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
}
