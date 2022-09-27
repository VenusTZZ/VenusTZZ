<script setup>
import BackGround from './BackGround.vue'
import DefaultTheme from 'vitepress/theme'
import ToBlog from './ToBlog.vue'
const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-top>
      <BackGround />
    </template>
  </Layout>
</template>
