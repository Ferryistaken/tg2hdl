import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'tg2hdl',
  description: 'Tinygrad to HDL experiments for INT8 GEMV acceleration.',
  cleanUrls: true,
  themeConfig: {
    nav: [
      { text: 'Introduction', link: '/guide/introduction' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'GitHub', link: 'https://github.com/' }
    ],
    sidebar: [
      {
        text: 'Overview',
        items: [
          { text: 'Introduction', link: '/guide/introduction' },
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Architecture', link: '/guide/architecture' },
          { text: 'Verification', link: '/guide/verification' }
        ]
      }
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/' }],
    search: {
      provider: 'local'
    }
  }
})
