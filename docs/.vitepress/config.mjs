import { defineConfig } from 'vitepress'

const base = process.env.DOCS_BASE || '/'

export default defineConfig({
  title: 'tg2hdl',
  description: 'Tinygrad to HDL experiments for INT8 GEMV acceleration.',
  cleanUrls: true,
  base,
  themeConfig: {
    nav: [
      { text: 'Introduction', link: '/guide/introduction' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'Deployment', link: '/guide/deployment' },
      { text: 'GitHub', link: 'https://github.com/' }
    ],
    sidebar: [
      {
        text: 'Overview',
        items: [
          { text: 'Introduction', link: '/guide/introduction' },
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Architecture', link: '/guide/architecture' },
          { text: 'Verification', link: '/guide/verification' },
          { text: 'Deployment', link: '/guide/deployment' }
        ]
      }
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/' }],
    search: {
      provider: 'local'
    }
  }
})
