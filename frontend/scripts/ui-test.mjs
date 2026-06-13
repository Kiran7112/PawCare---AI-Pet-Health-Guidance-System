/**
 * Headless UI test: drives the running app, submits one assessment, and
 * screenshots every screen where data is shown — the assessment form and all
 * four result tabs — into public/screenshots/.
 *
 * Prereqs (started separately): backend on :8000, Vite dev server on :5173.
 *   node scripts/ui-test.mjs
 */
import puppeteer from "puppeteer-core";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync, mkdirSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "..", "public", "screenshots");
mkdirSync(OUT_DIR, { recursive: true });
const URL = process.env.UI_URL || "http://localhost:5173";

const BROWSERS = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
];
const executablePath = BROWSERS.find((p) => existsSync(p));
if (!executablePath) throw new Error("No Chrome/Edge executable found");

const SAMPLE = {
  about_pet:
    "Max is a 7-year-old neutered male Labrador Retriever, slightly overweight. No major past illnesses, up to date on vaccines.",
  daily_routine:
    "Lives indoors with a fenced yard. Two 20-minute walks daily, eats premium kibble twice a day. Experienced owner, vet within 10 minutes.",
  health_concerns:
    "Drinking noticeably more water for the last 10 days, occasional lethargy and reduced appetite in the evenings.",
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath,
  headless: "new",
  args: ["--no-sandbox", "--window-size=1366,1000"],
  defaultViewport: { width: 1366, height: 1000, deviceScaleFactor: 2 },
});

async function shot(page, name) {
  const path = resolve(OUT_DIR, name);
  await page.screenshot({ path, fullPage: true });
  console.log("  ✓", name);
}

async function clickTab(page, label) {
  await page.evaluate((lbl) => {
    const tab = [...document.querySelectorAll('[role="tab"]')].find((b) =>
      new RegExp(lbl, "i").test(b.textContent || ""),
    );
    if (!tab) throw new Error("tab not found: " + lbl);
    tab.click();
  }, label);
  await sleep(800); // let the fade-in animation settle
}

try {
  const page = await browser.newPage();
  console.log("→ opening", URL);
  await page.goto(URL, { waitUntil: "networkidle2", timeout: 60000 });

  // 1. Assessment input page
  await page.waitForSelector("textarea", { timeout: 30000 });
  await sleep(400);
  await shot(page, "01-assessment-form.png");

  // Fill + submit
  const areas = await page.$$("textarea");
  if (areas.length < 3) throw new Error(`expected 3 textareas, found ${areas.length}`);
  const values = [SAMPLE.about_pet, SAMPLE.daily_routine, SAMPLE.health_concerns];
  for (let i = 0; i < 3; i++) {
    await areas[i].click();
    await areas[i].type(values[i], { delay: 1 });
  }
  console.log("→ form filled, submitting");
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("button")].find((b) =>
      /generate assessment/i.test(b.textContent || ""),
    );
    if (!btn) throw new Error("submit button not found");
    btn.click();
  });

  console.log("→ waiting for results (LLM call, up to 120s)…");
  await page.waitForFunction(
    () => /Assessment Results/i.test(document.body.innerText),
    { timeout: 120000 },
  );
  await sleep(1500); // gauges/animations

  // 2–5. Each result tab
  console.log("→ capturing result tabs");
  await clickTab(page, "Overview");
  await shot(page, "02-overview.png");

  await clickTab(page, "Health Guidance");
  await shot(page, "03-health-guidance.png");

  await clickTab(page, "Detailed Analysis");
  await shot(page, "04-detailed-analysis.png");

  await clickTab(page, "Summary");
  await shot(page, "05-summary.png");

  console.log("✓ all screenshots saved to", OUT_DIR);
} finally {
  await browser.close();
}
