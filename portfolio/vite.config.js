import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// Base path: set to '/' for username.github.io root site,
// or '/repo-name/' for a project site. Update before deployment.
export default defineConfig({
  plugins: [react()],
  base: '/image-generator/',
})
