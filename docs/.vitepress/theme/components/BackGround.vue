<script setup>
import { onMounted, ref } from 'vue'

const r15 = Math.PI / 12
const color = '#c0ebd7'
const canvas = ref(null)

let ctx
let width
let height
let steps = []
let prevSteps = []
let iterations = 0
let lastTime = 0
let zones = []
const interval = 1000 / 40
const AVOID_MARGIN = 16

function polar2cart(x, y, r, theta) {
  return [x + r * Math.cos(theta), y + r * Math.sin(theta)]
}

// 收集页面上带 data-avoid 标记的元素（文字/组件），树枝不画进这些区域
function collectZones() {
  zones = Array.from(document.querySelectorAll('[data-avoid]')).map((el) => {
    const r = el.getBoundingClientRect()
    return {
      left: r.left - AVOID_MARGIN,
      right: r.right + AVOID_MARGIN,
      top: r.top - AVOID_MARGIN,
      bottom: r.bottom + AVOID_MARGIN,
    }
  })
}

function inZone(x, y) {
  return zones.some((z) => x > z.left && x < z.right && y > z.top && y < z.bottom)
}

function step(x, y, rad) {
  const len = Math.random() * 6
  const [nx, ny] = polar2cart(x, y, len, rad)

  if (inZone(nx, ny)) return

  ctx.beginPath()
  ctx.moveTo(x, y)
  ctx.lineTo(nx, ny)
  ctx.stroke()

  if (nx < -200 || nx > width + 200 || ny < -200 || ny > height + 200) return

  const rad1 = rad + Math.random() * r15
  const rad2 = rad - Math.random() * r15

  if (iterations <= 4 || Math.random() > 0.5) steps.push(() => step(nx, ny, rad1))
  if (iterations <= 4 || Math.random() > 0.5) steps.push(() => step(nx, ny, rad2))
}

function frame() {
  if (performance.now() - lastTime < interval) {
    requestAnimationFrame(frame)
    return
  }

  iterations++
  prevSteps = steps
  steps = []
  lastTime = performance.now()
  prevSteps.forEach((fn) => fn())

  // 周期性刷新避让区域（元素位置/尺寸变化后生效）
  if (iterations % 30 === 0) collectZones()

  if (steps.length) requestAnimationFrame(frame)
}

onMounted(() => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

  collectZones()

  const dpr = window.devicePixelRatio || 1
  width = window.innerWidth
  height = window.innerHeight
  canvas.value.style.width = `${width}px`
  canvas.value.style.height = `${height}px`
  canvas.value.width = width * dpr
  canvas.value.height = height * dpr
  ctx = canvas.value.getContext('2d')
  ctx.scale(dpr, dpr)

  ctx.lineWidth = 1
  ctx.strokeStyle = color

  steps = [
    () => step(Math.random() * width, 0, Math.PI / 2),
    () => step(Math.random() * width, height, -Math.PI / 2),
    () => step(0, Math.random() * height, 0),
    () => step(width, Math.random() * height, Math.PI),
  ]
  if (width < 500) steps = steps.slice(0, 2)

  lastTime = performance.now()
  requestAnimationFrame(frame)
})
</script>

<template>
  <div class="background" style="z-index: -1; position: fixed; top: 0; left: 0">
    <canvas ref="canvas"></canvas>
  </div>
</template>
