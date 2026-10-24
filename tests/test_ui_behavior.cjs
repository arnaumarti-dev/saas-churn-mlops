"use strict";
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const helpers = require("../src/churn_mlops/api/static/dashboard-utils.js");

function browserHarness(fetchImpl) {
  const nodes = new Map();
  const doc = { activeElement: null };
  class Node {
    constructor(id = "") {
      this.id = id;
      this.hidden = false;
      this.disabled = false;
      this.value = "all";
      this.tabIndex = -1;
      this.textContent = "";
      this.style = {};
      this.attributes = {};
      this.handlers = {};
      this.children = [];
      this.classes = new Set();
      this.classList = {
        add: name => this.classes.add(name),
        remove: name => this.classes.delete(name),
        toggle: (name, force) => {
          if (force === true || (force === undefined && !this.classes.has(name))) {
            this.classes.add(name);
          } else {
            this.classes.delete(name);
          }
        },
        contains: name => this.classes.has(name),
      };
    }
    addEventListener(type, handler) { this.handlers[type] = handler; }
    fire(type, overrides = {}) {
      const event = {
        preventDefault() { this.defaultPrevented = true; },
        key: undefined,
        ...overrides,
      };
      return this.handlers[type]?.(event);
    }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key]; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    focus() { doc.activeElement = this; }
    scrollIntoView() { this.wasScrolledTo = true; }
    reportValidity() { return true; }
  }
  doc.getElementById = id => {
    if (!nodes.has(id)) nodes.set(id, new Node(id));
    return nodes.get(id);
  };
  doc.createElement = () => new Node();
  doc.getElementById("batch-view").hidden = true;
  doc.getElementById("batch-empty").hidden = false;
  doc.getElementById("batch-results").hidden = true;
  doc.getElementById("batch-cancel").hidden = true;
  doc.getElementById("batch-submit").disabled = true;
  doc.getElementById("prediction-form").elements = { namedItem: () => null };
  class FakeFormData { entries() { return [][Symbol.iterator](); } }
  const requests = [];
  const fetch = (url, options) => {
    requests.push({ url, options });
    return fetchImpl(url, options);
  };
  const downloads = [];
  const URLMock = {
    createObjectURL(blob) { downloads.push(blob); return "blob:fake"; },
    revokeObjectURL() {},
  };
  const context = {
    document: doc,
    window: { ChurnDashboardUtils: helpers },
    fetch, URL: URLMock,
    Blob, AbortController, FormData: FakeFormData,
    setTimeout: callback => callback(),
    Intl, console,
  };
  const js = fs.readFileSync(path.join(__dirname, "../src/churn_mlops/api/static/demo.js"), "utf8");
  vm.runInNewContext(js, context, { filename: "demo.js" });
  return { doc, get: doc.getElementById, requests, downloads };
}

test("tablist supports arrows, Home/End and keyboard focus", () => {
  const ui = browserHarness(async () => { throw Error("Unexpected request"); });
  const single = ui.get("tab-single"), batch = ui.get("tab-batch");
  assert.equal(batch.tabIndex, -1);
  assert.equal(ui.get("batch-view").hidden, true);
  single.fire("keydown", { key: "ArrowRight" });
  assert.equal(ui.doc.activeElement, batch);
  assert.equal(batch.getAttribute("aria-selected"), "true");
  assert.equal(single.getAttribute("aria-selected"), "false");
  assert.equal(batch.tabIndex, 0);
  assert.equal(single.tabIndex, -1);
  assert.equal(ui.get("batch-view").hidden, false);
  assert.equal(ui.get("single-view").hidden, true);
  batch.fire("keydown", { key: "Home" });
  assert.equal(ui.doc.activeElement, single);
  single.fire("keydown", { key: "End" });
  assert.equal(ui.doc.activeElement, batch);
  batch.fire("keydown", { key: "ArrowLeft" });
  assert.equal(ui.doc.activeElement, single);
});

test("batch example, loading, filters, no-matches and reset remain usable", async () => {
  const ui = browserHarness(async () => ({
    ok: true,
    json: async () => ({
      predictions: Array.from({ length: 12 }, (_, i) =>
        ({ churn_probability: i < 6 ? 0.8 : 0.2, churn_prediction: i < 6 ? 1 : 0 })
      ),
      summary: {
        total_customers: 12, high_risk_customers: 6, low_risk_customers: 6,
        average_churn_probability: 0.5, decision_threshold: 0.32, high_risk_monthly_charges: 600,
      },
    }),
  }));
  const get = ui.get;
  get("batch-example").fire("click");
  assert.equal(get("batch-submit").disabled, false);
  assert.match(get("batch-status").textContent, /12 perfiles listos/);
  await get("batch-submit").fire("click");
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.requests[0].url, "/predict/batch");
  assert.equal(get("batch-cancel").hidden, true);
  assert.equal(get("batch-submit").disabled, false);
  assert.equal(get("batch-results").hidden, false);
  assert.equal(get("batch-jump-results").hidden, false);
  assert.equal(get("batch-export").disabled, false);
  assert.equal(get("batch-table").children.length, 12);
  assert.match(get("batch-status").textContent, /Análisis completado/);

  get("batch-risk-filter").value = "high";
  get("batch-risk-filter").fire("change");
  assert.equal(get("batch-table").children.length, 6);
  get("batch-contract-filter").value = "Two year";
  get("batch-contract-filter").fire("change");
  assert.equal(get("batch-table").children.length, 0);
  assert.equal(get("batch-no-matches").hidden, false);
  assert.equal(get("batch-table-scroll").hidden, true);
  assert.equal(get("batch-export").disabled, true);

  get("batch-clear-filters").fire("click");
  assert.equal(get("batch-table").children.length, 12);
  assert.equal(get("batch-table-scroll").hidden, false);
  assert.equal(get("batch-no-matches").hidden, true);
  assert.equal(get("batch-clear-filters").disabled, true);
  get("batch-jump-results").fire("click");
  assert.equal(ui.doc.activeElement, get("batch-results-heading"));
});

test("user can cancel slow analysis without losing loaded profiles", async () => {
  const ui = browserHarness((_, options) => new Promise((_, reject) => {
    options.signal.addEventListener("abort", () => reject(new Error("aborted")));
  }));
  const get = ui.get;
  get("batch-example").fire("click");
  const work = get("batch-submit").fire("click");
  assert.equal(get("batch-cancel").hidden, false);
  assert.equal(get("batch-submit").disabled, true);
  get("batch-cancel").fire("click");
  await work;
  assert.match(get("batch-status").textContent, /cancelado/);
  assert.equal(get("batch-cancel").hidden, true);
  assert.equal(get("batch-submit").disabled, false);
  assert.equal(get("batch-error").hidden, true);
});
