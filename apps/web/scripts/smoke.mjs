import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "playwright-core";

const edgePath = process.env.PITALPHA_EDGE_PATH
  ?? "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const baseUrl = process.env.PITALPHA_WEB_URL ?? "http://127.0.0.1:3000";
const screenshotDir = resolve("../../artifacts/ui");
await mkdir(screenshotDir, { recursive: true });

const browser = await chromium.launch({ executablePath: edgePath, headless: true });
const failures = [];

function observe(page, name, allowExpectedNetworkErrors = false) {
  page.on("pageerror", (error) => failures.push(`${name} pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (
      message.type() === "error"
      && !(allowExpectedNetworkErrors && message.text().startsWith("Failed to load resource"))
    ) {
      failures.push(`${name} console: ${message.text()}`);
    }
  });
}

async function expectText(locator, expected, label) {
  await locator.filter({ hasText: expected }).waitFor({ state: "visible", timeout: 15_000 });
  const actual = (await locator.textContent())?.trim();
  if (actual !== expected) throw new Error(`${label}: expected ${expected}, received ${actual}`);
}

try {
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  observe(desktop, "desktop");
  await desktop.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await expectText(desktop.locator(".connection-status"), "Live artifacts", "API state");
  await expectText(desktop.locator(".chart-summary > strong"), "1.48×", "Ridge 10 bps wealth");
  await expectText(desktop.locator(".advanced-stat-row > div").last().locator("strong"), "0 / 5", "Five-seed decision");
  await expectText(desktop.locator(".rejected-pill"), "Not promoted", "Neural promotion status");

  await desktop.getByRole("button", { name: "LightGBM" }).click();
  await expectText(desktop.locator(".chart-summary > strong"), "1.29×", "LightGBM 10 bps wealth");
  await desktop.getByRole("button", { name: "0 bps", exact: true }).click();
  await expectText(desktop.locator(".chart-summary > strong"), "1.87×", "LightGBM 0 bps wealth");

  await desktop.getByRole("button", { name: "MLP" }).click();
  await desktop.getByRole("button", { name: "10 bps", exact: true }).click();
  await expectText(desktop.locator(".chart-summary > strong"), "0.56×", "MLP 10 bps wealth");
  await desktop.locator("label.buffer-toggle").click();
  if (!(await desktop.getByRole("checkbox", { name: "Turnover buffer" }).isChecked())) {
    throw new Error("turnover buffer did not activate from its visible control");
  }
  await expectText(desktop.locator(".chart-summary > strong"), "0.62×", "Buffered MLP wealth");

  await desktop.getByRole("button", { name: "Ridge" }).click();
  await desktop.locator("label.buffer-toggle").click();
  if (await desktop.getByRole("checkbox", { name: "Turnover buffer" }).isChecked()) {
    throw new Error("turnover buffer did not deactivate from its visible control");
  }
  await expectText(desktop.locator(".chart-summary > strong"), "1.48×", "Restored Ridge wealth");
  await desktop.locator(".recharts-line-curve").last().waitFor({ state: "visible" });
  await desktop.screenshot({ path: resolve(screenshotDir, "dashboard-smoke-desktop.png"), fullPage: true });

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true });
  observe(mobile, "mobile");
  await mobile.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await expectText(mobile.locator(".connection-status"), "Live artifacts", "Mobile API state");
  await expectText(mobile.locator(".chart-summary > strong"), "1.48×", "Mobile Ridge wealth");
  await mobile.locator(".recharts-line-curve").last().waitFor({ state: "visible" });
  const dimensions = await mobile.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  if (dimensions.content > dimensions.viewport + 1) {
    throw new Error(`mobile horizontal overflow: ${dimensions.content}px > ${dimensions.viewport}px`);
  }
  await mobile.screenshot({ path: resolve(screenshotDir, "dashboard-smoke-mobile.png"), fullPage: true });

  const snapshot = await browser.newPage({ viewport: { width: 1100, height: 800 } });
  observe(snapshot, "snapshot", true);
  await snapshot.route("**/api/**", (route) => route.abort("connectionfailed"));
  await snapshot.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await expectText(snapshot.locator(".connection-status"), "Audited snapshot", "Snapshot state");
  if (!(await snapshot.getByRole("button", { name: "0 bps", exact: true }).isDisabled())) {
    throw new Error("cost controls must be disabled when the API is offline");
  }
  await expectText(snapshot.locator(".metric-card").first().locator(".metric-note"), "+6.9% vs equal-weight", "Snapshot benchmark delta");

  if (failures.length) throw new Error(failures.join("\n"));
  console.log("browser_smoke_cases 3");
  console.log("desktop_interactions model cost buffer restore: pass");
  console.log(`mobile_overflow ${dimensions.content}/${dimensions.viewport}: pass`);
  console.log("offline_snapshot controls and benchmark: pass");
} finally {
  await browser.close();
}
