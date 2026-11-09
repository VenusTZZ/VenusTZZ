import Layout from './components/Layout.vue'
import NotesIndex from './components/NotesIndex.vue'
import NotFound from './components/NotFound.vue'
import DefaultTheme from 'vitepress/theme'
import './styles/index.css'

export default {
  ...DefaultTheme,
  Layout,
  NotFound,
  enhanceApp({ app }) {
    app.component('NotesIndex', NotesIndex)
  }
}
