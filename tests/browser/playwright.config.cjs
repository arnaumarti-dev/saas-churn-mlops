const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: "*.spec.cjs",
  timeout: 90_000,
  expect: { timeout: 25_000 },
  retries: 1,
  workers: 1,
  reporter: [["line"], ["html", { outputFolder: "test-results/browser-report", open: "never" }]],
  outputDir: "test-results/browser-output",
  use: {
    baseURL: "http://127.0.0.1:8000",
    browserName: "chromium",
    actionTimeout: 20_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
});
