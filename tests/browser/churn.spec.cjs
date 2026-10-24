"use strict";

const path = require("node:path");
const { test, expect } = require("@playwright/test");

const widths = [
  { name: "mobile-375", width: 375, height: 812 },
  { name: "tablet-768", width: 768, height: 1024 },
  { name: "desktop-1440", width: 1440, height: 900 },
];

test.describe("Responsive real-browser visual and workflow QA", () => {
  for (const viewport of widths) {
    test(viewport.name + ": single mode, CSV dashboard and filters", async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      const crashes = [];
      page.on("pageerror", err => crashes.push(err.message));
      await page.goto("/", { waitUntil: "networkidle" });
      await expect(page.getByRole("heading", { name: /Anticipa el abandono/ })).toBeVisible();
      await expect(page.locator("#tab-single")).toHaveAttribute("aria-selected", "true");
      if (viewport.width <= 375) {
        const actionSize = await page.locator("#submit-button").evaluate(element => ({
          height: element.getBoundingClientRect().height,
          overflow: element.scrollWidth - element.clientWidth
        }));
        expect(actionSize.height, "Individual prediction CTA should fit on one line").toBeLessThan(65);
        expect(actionSize.overflow, "Individual prediction CTA text should not be clipped").toBeLessThanOrEqual(1);
      }
      await page.screenshot({ path: path.join("test-results", "visual-qa", viewport.name + "-initial.png"), fullPage: true });

      const overflowBefore = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflowBefore, "Homepage must not scroll horizontally").toBeLessThanOrEqual(1);

      await page.locator("#tab-batch").click();
      await expect(page.locator("#batch-view")).toBeVisible();
      await expect(page.locator(".workflow-steps li")).toHaveCount(3);
      await page.locator("#batch-example").click();
      await expect(page.locator("#batch-submit")).toBeEnabled();
      await page.locator("#batch-submit").click();
      await expect(page.locator("#batch-metrics")).toBeVisible({ timeout: 45_000 });
      await expect(page.locator("#batch-table tr")).toHaveCount(12);
      await expect(page.locator("#batch-jump-results")).toBeVisible();

      if (viewport.width <= 375) {
        await expect(page.locator(".mobile-table-hint")).toBeVisible();
        await expect(page.locator(".mobile-table-hint")).toContainText("Desliza");
      }
      await page.screenshot({ path: path.join("test-results", "visual-qa", viewport.name + "-dashboard.png"), fullPage: true });

      await page.locator("#batch-risk-filter").selectOption("high");
      await expect(page.locator("#batch-match-count")).toContainText("de 12 predicciones");
      await page.locator("#batch-contract-filter").selectOption("Otros");
      await expect(page.locator("#batch-no-matches")).toBeVisible();
      await expect(page.locator("#batch-export")).toBeDisabled();
      await page.locator("#batch-clear-filters").click();
      await expect(page.locator("#batch-table tr")).toHaveCount(12);

      const overflowAfter = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflowAfter, "Dashboard must not scroll horizontally").toBeLessThanOrEqual(1);
      expect(crashes).toEqual([]);
    });
  }
});

test("Accessible tab keyboard operation, invalid CSV and safe export", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/", { waitUntil: "networkidle" });
  await page.locator("#tab-single").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#tab-batch")).toBeFocused();
  await expect(page.locator("#batch-view")).toBeVisible();
  await page.keyboard.press("Home");
  await expect(page.locator("#tab-single")).toBeFocused();
  await page.keyboard.press("End");
  await expect(page.locator("#tab-batch")).toBeFocused();

  await page.locator("#batch-file").setInputFiles({
    name: "invalid-demo.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("bad_column,customerID\n1,private\n", "utf8"),
  });
  await expect(page.locator("#batch-error")).toBeVisible();
  await expect(page.locator("#batch-submit")).toBeDisabled();
  await page.locator("#batch-example").click();
  await expect(page.locator("#batch-error")).toBeHidden();
  await page.locator("#batch-submit").click();
  await expect(page.locator("#batch-results")).toBeVisible({ timeout: 45_000 });

  await page.locator("#batch-risk-filter").selectOption("high");
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#batch-export").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("churn-resultados-demo.csv");
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const csv = Buffer.concat(chunks).toString("utf8");
  expect(csv).toContain("fila_csv,contrato,probabilidad_pct,clasificacion");
  expect(csv).not.toContain("customerID");
  expect(csv).not.toContain("MonthlyCharges");
  expect(csv).not.toContain("HYPERLINK");
});
