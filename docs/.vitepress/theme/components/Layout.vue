<script setup>
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import BackGround from './BackGround.vue'
import HomeHero from './HomeHero.vue'

const { frontmatter } = useData()
</script>

<template>
  <HomeHero v-if="frontmatter.layout === 'home'" />
  <DefaultTheme.Layout v-else>
    <template #layout-top>
      <BackGround />
    </template>
  </DefaultTheme.Layout>
</template>
