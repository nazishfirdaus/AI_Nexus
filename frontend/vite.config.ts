import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  // Absolute base: the built bundle is served from the API root, and nested
  // SPA paths would break relative asset URLs.
  base: "/",
  server: {
    port: 5173,
    proxy: {
      // The API owns /api (uvicorn on 8000); Vite proxies it in dev so the
      // browser only ever talks to one origin.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
})