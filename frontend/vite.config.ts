/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // The sample flows live in ../examples (shared with the API docs), so the
  // dev server may read one level up.
  server: { port: 5173, fs: { allow: [".."] } },
  test: { environment: "jsdom", setupFiles: ["./src/setupTests.ts"] },
});
