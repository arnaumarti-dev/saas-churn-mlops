"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const utils = require("../src/churn_mlops/api/static/dashboard-utils.js");

function sample() {
  const customers = [
    { Contract: "Month-to-month", tenure: 3 },
    { Contract: "Two year", tenure: 48 },
    { Contract: "Month-to-month", tenure: 5 },
    { Contract: "One year", tenure: 20 },
  ];
  const predictions = [
    { churn_probability: 0.9, churn_prediction: 1 },
    { churn_probability: 0.1, churn_prediction: 0 },
    { churn_probability: 0.65, churn_prediction: 1 },
    { churn_probability: 0.3, churn_prediction: 0 },
  ];
  return utils.validateRows(customers, predictions);
}

test("segment contract groups and compute actual classification proportions", () => {
  const segments = utils.contractSegments(sample());
  assert.deepEqual(segments.map(s => [s.contract, s.total, s.high, s.low]), [
    ["Month-to-month", 2, 2, 0],
    ["One year", 1, 0, 1],
    ["Two year", 1, 0, 1],
  ]);
  assert.equal(segments[0].highPercent, 100);
  assert.equal(segments[1].highPercent, 0);
});

test("filter risk and contract, sort by probability, preserve original CSV rows", () => {
  const rows = sample();
  assert.deepEqual(utils.selectRows(rows, "high").map(r => r.csvRow), [2, 4]);
  assert.deepEqual(utils.selectRows(rows, "all", "Two year").map(r => r.csvRow), [3]);
  assert.deepEqual(utils.selectRows(rows, "all", "all", "highest").map(r => r.csvRow), [2, 4, 5, 3]);
  assert.deepEqual(utils.selectRows(rows, "all", "all", "lowest").map(r => r.csvRow), [3, 5, 4, 2]);
  assert.deepEqual(utils.selectRows(rows, "high", "Two year"), []);
});

test("never export arbitrary customer identifiers or spreadsheet formulas", () => {
  const customers = [{
    Contract: '=HYPERLINK("https://attacker.invalid","click")',
    customerID: "secret-example",
    MonthlyCharges: 15,
  }];
  const predictions = [{ churn_probability: 0.75, churn_prediction: 1 }];
  const rows = utils.validateRows(customers, predictions);
  assert.equal(rows[0].contract, "Otros");
  const csv = utils.csvExport(rows);
  assert.ok(csv.startsWith("\uFEFFfila_csv,contrato,probabilidad_pct,clasificacion"));
  assert.ok(csv.includes("\r\n2,Otros,75.00,sobre_umbral\r\n"));
  assert.ok(!csv.includes("HYPERLINK"));
  assert.ok(!csv.includes("secret-example"));
  assert.ok(!csv.includes("MonthlyCharges"));
});

test("reject invalid predictions and unexpected controls", () => {
  assert.throws(() => utils.validateRows([{ Contract: "One year" }], [
    { churn_probability: NaN, churn_prediction: 0 },
  ]), /inválida/);
  assert.throws(() => utils.validateRows([{ Contract: "One year" }], [
    { churn_probability: 0.2, churn_prediction: 2 },
  ]), /inválida/);
  assert.throws(() => utils.validateRows([{}], []), /incompatible/);
  assert.throws(() => utils.selectRows([], "unsafe"), /inválido/);
  assert.throws(() => utils.selectRows([], "all", "unsafe"), /inválido/);
  assert.throws(() => utils.selectRows([], "all", "all", "unsafe"), /inválido/);
});

test("empty cohorts and exports remain valid", () => {
  assert.deepEqual(utils.contractSegments([]), []);
  assert.deepEqual(utils.selectRows([]), []);
  assert.equal(utils.csvExport([]).split("\r\n")[0], "\uFEFFfila_csv,contrato,probabilidad_pct,clasificacion");
});
