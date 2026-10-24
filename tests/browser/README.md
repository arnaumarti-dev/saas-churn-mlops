# Visual QA — Churn Lab

Run the existing Docker demo locally on port 8000, then:

```bash
npm install --no-save @playwright/test@1.56.1
npx playwright install chromium
mkdir -p test-results/visual-qa
npx playwright test --config tests/browser/playwright.config.cjs
```

The dedicated GitHub Action builds and starts the self-contained Docker image
automatically. It exercises real Chromium rendering at **375 px**, **768 px**
and **1440 px**. Each viewport produces a full-page screenshot of the
landing state and of the analyzed CSV dashboard. Tests also detect horizontal
page overflow, browser JS errors, keyboard tab navigation, CSV import errors,
real batch inference, filtering, clearing filters, and results CSV download.

Results are published as the workflow artifact **churn-browser-visual-qa**,
including screenshots, report, and failure traces. This suite is intentionally
separate from the regular Python/Docker CI, so that adding browser tests does
not increase normal API test costs.

This verifies rendering and basic interactions in Chromium; the screenshot
artifact still needs human visual review. Browser checks are not complete WCAG
conformance testing, accessibility auditing or multi-browser cross-device QA.
Never upload real customer records; all test inputs are synthetic.
