/* In-browser analytical helpers. No tracking, storage, or network requests.
 * CommonJS export enables dependency-free Node unit tests. */
(function (root) {
  "use strict";

  const ALLOWED_CONTRACTS = ["Month-to-month", "One year", "Two year"];

  function contractLabel(raw) {
    return ALLOWED_CONTRACTS.includes(raw) ? raw : "Otros";
  }

  function validateRows(customers, predictions) {
    if (!Array.isArray(customers) || !Array.isArray(predictions) ||
        customers.length !== predictions.length || customers.length > 200) {
      throw new Error("El lote de predicciones es incompatible con los datos.");
    }
    return customers.map((customer, index) => {
      const prediction = predictions[index];
      const probability = Number(prediction.churn_probability);
      const label = prediction.churn_prediction;
      if (!Number.isFinite(probability) || probability < 0 || probability > 1 ||
          (label !== 0 && label !== 1)) {
        throw new Error("Probabilidad o clasificación inválida.");
      }
      return {
        csvRow: index + 2,
        probability,
        label,
        contract: contractLabel(customer.Contract),
      };
    });
  }

  function contractSegments(rows) {
    return [...ALLOWED_CONTRACTS, "Otros"].map(contract => {
      const matching = rows.filter(row => row.contract === contract);
      const high = matching.filter(row => row.label === 1).length;
      return {
        contract,
        total: matching.length,
        high,
        low: matching.length - high,
        highPercent: matching.length ? (100 * high) / matching.length : 0,
      };
    }).filter(segment => segment.total > 0);
  }

  function selectRows(rows, risk = "all", contract = "all", sort = "file") {
    if (!["all", "high", "low"].includes(risk)) throw new Error("Filtro de riesgo inválido.");
    if (!["all", ...ALLOWED_CONTRACTS, "Otros"].includes(contract)) {
      throw new Error("Filtro de contrato inválido.");
    }
    if (!["file", "highest", "lowest"].includes(sort)) throw new Error("Orden inválido.");

    const selected = rows.filter(row => (
      (risk === "all" || row.label === (risk === "high" ? 1 : 0)) &&
      (contract === "all" || row.contract === contract)
    ));

    if (sort === "highest") {
      selected.sort((a, b) => b.probability - a.probability || a.csvRow - b.csvRow);
    } else if (sort === "lowest") {
      selected.sort((a, b) => a.probability - b.probability || a.csvRow - b.csvRow);
    } else {
      selected.sort((a, b) => a.csvRow - b.csvRow);
    }
    return selected;
  }

  function csvExport(rows) {
    // Export only trusted enum text, validated numbers and CSV row indexes.
    // Never echo arbitrary fields or personal identifiers from input CSVs.
    const lines = ["fila_csv,contrato,probabilidad_pct,clasificacion"];
    for (const row of rows) {
      const contract = contractLabel(row.contract);
      const probability = Number(row.probability);
      if (!Number.isInteger(row.csvRow) || row.csvRow < 2 ||
          !Number.isFinite(probability) || probability < 0 || probability > 1 ||
          ![0, 1].includes(row.label)) {
        throw new Error("No se pueden exportar resultados inválidos.");
      }
      lines.push([
        row.csvRow,
        contract,
        (probability * 100).toFixed(2),
        row.label === 1 ? "sobre_umbral" : "bajo_umbral",
      ].join(","));
    }
    return "\uFEFF" + lines.join("\r\n") + "\r\n";
  }

  const api = { contractLabel, validateRows, contractSegments, selectRows, csvExport };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ChurnDashboardUtils = api;
})(typeof window !== "undefined" ? window : globalThis);
