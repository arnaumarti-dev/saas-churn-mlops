"use strict";
    const fields = [
      ["gender", "Género (categoría del dataset)", ["Female", "Male"], "Female"],
      ["SeniorCitizen", "Persona mayor (indicador del dataset)", ["0", "1"], "0"],
      ["Partner", "Pareja", ["Yes", "No"], "No"],
      ["Dependents", "Personas dependientes", ["Yes", "No"], "No"],
      ["PhoneService", "Servicio telefónico", ["Yes", "No"], "Yes"],
      ["MultipleLines", "Varias líneas", ["No", "Yes", "No phone service"], "No"],
      ["OnlineSecurity", "Seguridad online", ["No", "Yes", "No internet service"], "No"],
      ["OnlineBackup", "Copia de seguridad", ["No", "Yes", "No internet service"], "No"],
      ["DeviceProtection", "Protección de dispositivos", ["No", "Yes", "No internet service"], "No"],
      ["TechSupport", "Soporte técnico", ["No", "Yes", "No internet service"], "No"],
      ["StreamingTV", "TV en streaming", ["No", "Yes", "No internet service"], "No"],
      ["StreamingMovies", "Películas en streaming", ["No", "Yes", "No internet service"], "No"],
      ["PaperlessBilling", "Factura electrónica", ["Yes", "No"], "Yes"]
    ];
    const container = document.getElementById("advanced-fields");
    for (const [name, title, options, selected] of fields) {
      const label = document.createElement("label");
      const caption = document.createElement("span");
      caption.textContent = title;
      const select = document.createElement("select");
      select.name = name;
      select.required = true;
      for (const value of options) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === selected;
        select.append(option);
      }
      label.append(caption, select);
      container.append(label);
    }

    const form = document.getElementById("prediction-form");
    const error = document.getElementById("error-message");
    const submit = document.getElementById("submit-button");
    const samples = {
      risky: { Contract: "Month-to-month", tenure: 2, MonthlyCharges: 89.50, TotalCharges: 179,
        InternetService: "Fiber optic", PaymentMethod: "Electronic check", OnlineSecurity: "No",
        TechSupport: "No", PaperlessBilling: "Yes", Partner: "No", Dependents: "No" },
      stable: { Contract: "Two year", tenure: 62, MonthlyCharges: 79, TotalCharges: 4898,
        InternetService: "DSL", PaymentMethod: "Bank transfer (automatic)", OnlineSecurity: "Yes",
        TechSupport: "Yes", PaperlessBilling: "No", Partner: "Yes", Dependents: "Yes" }
    };
    function setSample(key) {
      for (const [name, value] of Object.entries(samples[key])) {
        const field = form.elements.namedItem(name);
        if (field) field.value = String(value);
      }
      error.hidden = true;
      document.getElementById("result-data").hidden = true;
      document.getElementById("result-empty").hidden = false;
    }
    document.getElementById("example-risk").addEventListener("click", () => setSample("risky"));
    document.getElementById("example-stable").addEventListener("click", () => setSample("stable"));

    form.addEventListener("submit", async event => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      error.hidden = true;
      submit.disabled = true;
      submit.textContent = "Calculando predicción…";
      const data = Object.fromEntries(new FormData(form).entries());
      for (const key of ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]) {
        data[key] = Number(data[key]);
      }
      try {
        const response = await fetch("/predict", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data)
        });
        if (!response.ok) {
          if (response.status === 503) throw new Error("El modelo todavía no está cargado. Ejecuta el entrenamiento y reinicia la API.");
          throw new Error("No se pudo generar la predicción (HTTP " + response.status + "). Comprueba los datos.");
        }
        const result = await response.json();
        const probability = Math.max(0, Math.min(1, Number(result.churn_probability)));
        const threshold = Number(result.decision_threshold);
        if (!Number.isFinite(probability) || !(threshold > 0 && threshold < 1)) throw new Error("La API devolvió una respuesta no válida.");
        const high = result.churn_prediction === 1;
        const percent = (probability * 100).toFixed(1).replace(".", ",") + " %";
        document.getElementById("result-empty").hidden = true;
        document.getElementById("result-data").hidden = false;
        document.getElementById("probability-value").textContent = percent;
        document.getElementById("progress-fill").style.width = (probability * 100) + "%";
        document.getElementById("probability-bar").setAttribute("aria-valuenow", String(Math.round(probability * 100)));
        document.getElementById("threshold-value").textContent = (threshold * 100).toFixed(0) + " %";
        document.getElementById("classification-value").textContent = high ? "Positivo" : "Negativo";
        const pill = document.getElementById("prediction-label");
        pill.textContent = high ? "Riesgo sobre el umbral" : "Riesgo bajo el umbral";
        pill.classList.toggle("low", !high);
      } catch (exception) {
        error.textContent = exception instanceof Error ? exception.message : "Error inesperado";
        error.hidden = false;
      } finally {
        submit.disabled = false;
        submit.textContent = "Calcular riesgo de abandono →";
      }
    });


    // Batch dashboard: all examples are synthetic and nothing is persisted.
    const csvColumns = [
      "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
      "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
      "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
      "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
      "MonthlyCharges", "TotalCharges"
    ];
    const numericColumns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"];
    const batchMaxRows = 200;
    const batchMaxFileBytes = 256 * 1024;
    const batchFile = document.getElementById("batch-file");
    const batchError = document.getElementById("batch-error");
    const batchSubmit = document.getElementById("batch-submit");
    let batchCustomers = [];
    let analyzedRows = null;
    let batchGeneration = 0;
    const analytics = window.ChurnDashboardUtils;

    function viewTab(which) {
      const isBatch = which === "batch";
      document.getElementById("single-view").hidden = isBatch;
      document.getElementById("batch-view").hidden = !isBatch;
      for (const name of ["single", "batch"]) {
        const tab = document.getElementById("tab-" + name);
        const active = name === which;
        tab.classList.toggle("active", active);
        tab.setAttribute("aria-selected", String(active));
      }
    }
    document.getElementById("tab-single").addEventListener("click", () => viewTab("single"));
    document.getElementById("tab-batch").addEventListener("click", () => viewTab("batch"));

    function fakeCustomers() {
      const baseline = Object.fromEntries(new FormData(form).entries());
      const data = [];
      for (let i = 0; i < 12; i += 1) {
        const risky = i < 6;
        const override = risky ? samples.risky : samples.stable;
        const customer = { ...baseline, ...override };
        customer.tenure = risky ? 1 + i * 2 : 36 + i * 3;
        customer.MonthlyCharges = risky ? 89.5 + i : 61 + i * 2;
        customer.TotalCharges = Number((customer.MonthlyCharges * customer.tenure).toFixed(2));
        customer.SeniorCitizen = i % 3 === 0 ? 1 : 0;
        customer.gender = i % 2 === 0 ? "Female" : "Male";
        data.push(customer);
      }
      return data;
    }

    function setBatchCustomers(customers, description) {
      batchGeneration += 1;
      batchCustomers = customers;
      analyzedRows = null;
      document.getElementById("batch-risk-filter").value = "all";
      document.getElementById("batch-contract-filter").value = "all";
      document.getElementById("batch-sort").value = "file";
      document.getElementById("batch-export").disabled = true;
      document.getElementById("batch-segments-panel").hidden = true;
      batchSubmit.disabled = customers.length === 0;
      batchError.hidden = true;
      document.getElementById("batch-preview").textContent = description;
      document.getElementById("batch-empty").hidden = false;
      document.getElementById("batch-metrics").hidden = true;
      document.getElementById("batch-results").hidden = true;
    }

    function parseCSV(text) {
      // RFC-4180-style quoted cells and escaped double quotes, including newlines.
      const rows = [];
      let cells = [];
      let cell = "";
      let quoted = false;
      let closedQuote = false;
      for (let i = 0; i < text.length; i += 1) {
        const char = text[i];
        if (quoted) {
          if (char === '"') {
            if (text[i + 1] === '"') { cell += '"'; i += 1; }
            else { quoted = false; closedQuote = true; }
          } else {
            cell += char;
          }
        } else if (char === '"') {
          if (cell.length || closedQuote) throw new Error("Las comillas del CSV no son válidas.");
          quoted = true;
        } else if (char === "," || char === "\n") {
          cells.push(cell);
          cell = "";
          closedQuote = false;
          if (char === "\n") {
            if (cells.some(value => value.trim() !== "")) rows.push(cells);
            cells = [];
          }
        } else if (char === "\r") {
          continue;
        } else {
          if (closedQuote && char !== " ") throw new Error("Hay caracteres tras una celda entrecomillada.");
          cell += char;
        }
      }
      if (quoted) throw new Error("El CSV contiene comillas sin cerrar.");
      cells.push(cell);
      if (cells.some(value => value.trim() !== "")) rows.push(cells);
      return rows;
    }

    function customersFromCSV(text) {
      const rows = parseCSV(text.replace(/^\uFEFF/, ""));
      if (rows.length < 2) throw new Error("El archivo debe incluir cabecera y al menos un cliente.");
      const headers = rows[0].map(value => value.trim());
      if (new Set(headers).size !== headers.length ||
          headers.length !== csvColumns.length ||
          csvColumns.some(column => !headers.includes(column))) {
        throw new Error("Las columnas no coinciden con el CSV de ejemplo (19 columnas, sin identificadores ni datos personales).");
      }
      if (rows.length - 1 > batchMaxRows) throw new Error("Límite de 200 clientes por análisis.");
      return rows.slice(1).map((cells, index) => {
        if (cells.length !== headers.length) throw new Error("La fila " + (index + 2) + " tiene un número incorrecto de columnas.");
        const customer = {};
        headers.forEach((name, columnIndex) => {
          const raw = cells[columnIndex].trim();
          if (raw === "") throw new Error("Campo vacío en la fila " + (index + 2) + ": " + name);
          if (numericColumns.includes(name)) {
            const parsed = Number(raw);
            if (!Number.isFinite(parsed)) throw new Error("Número no válido en fila " + (index + 2) + ": " + name);
            customer[name] = parsed;
          } else {
            customer[name] = raw;
          }
        });
        return customer;
      });
    }

    function csvEscape(value) {
      const text = String(value);
      return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
    }

    document.getElementById("batch-example").addEventListener("click", () => {
      batchFile.value = "";
      setBatchCustomers(fakeCustomers(), "12 clientes de prueba ficticios cargados. Ninguno corresponde a una persona real.");
    });

    document.getElementById("batch-download").addEventListener("click", () => {
      const rows = fakeCustomers();
      const lines = [csvColumns.join(",")];
      for (const row of rows) lines.push(csvColumns.map(name => csvEscape(row[name])).join(","));
      const file = new Blob(["\uFEFF", lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
      const href = URL.createObjectURL(file);
      const link = document.createElement("a");
      link.href = href;
      link.download = "churn-clientes-ficticios.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    });

    batchFile.addEventListener("change", async () => {
      const file = batchFile.files && batchFile.files[0];
      if (!file) return;
      try {
        if (file.size > batchMaxFileBytes) throw new Error("El CSV no puede superar 256 KiB.");
        const customers = customersFromCSV(await file.text());
        setBatchCustomers(customers, customers.length + " filas importadas de CSV, listas para analizar.");
      } catch (problem) {
        setBatchCustomers([], "Archivo no cargado.");
        batchError.textContent = problem instanceof Error ? problem.message : "No se pudo leer el CSV.";
        batchError.hidden = false;
      }
    });

    function formatPercent(number) {
      return (Number(number) * 100).toFixed(1).replace(".", ",") + " %";
    }

    function showContractSegments(rows) {
      const container = document.getElementById("contract-segments");
      container.replaceChildren();
      const contracts = {
        "Month-to-month": "Mensual",
        "One year": "Anual",
        "Two year": "Dos años",
        "Otros": "Otros"
      };
      for (const segment of analytics.contractSegments(rows)) {
        const item = document.createElement("div");
        item.className = "contract-segment";
        const header = document.createElement("div");
        header.className = "contract-segment-header";
        const title = document.createElement("strong");
        title.textContent = contracts[segment.contract];
        const stat = document.createElement("span");
        stat.textContent = segment.high + " / " + segment.total + " sobre el umbral";
        header.append(title, stat);
        const track = document.createElement("div");
        track.className = "contract-segment-track";
        track.setAttribute("role", "img");
        track.setAttribute("aria-label", contracts[segment.contract] +
          ": " + segment.high + " de " + segment.total + " perfiles sobre el umbral.");
        const filled = document.createElement("div");
        filled.className = "contract-segment-fill";
        filled.style.width = segment.highPercent + "%";
        track.append(filled);
        item.append(header, track);
        container.append(item);
      }
      document.getElementById("batch-segments-panel").hidden = false;
    }

    function currentFilteredRows() {
      if (!analyzedRows) return [];
      return analytics.selectRows(
        analyzedRows,
        document.getElementById("batch-risk-filter").value,
        document.getElementById("batch-contract-filter").value,
        document.getElementById("batch-sort").value
      );
    }

    function renderFilteredRows() {
      const selected = currentFilteredRows();
      const table = document.getElementById("batch-table");
      table.replaceChildren();
      for (const prediction of selected) {
        const row = document.createElement("tr");
        const number = document.createElement("td");
        const probability = document.createElement("td");
        const status = document.createElement("td");
        number.textContent = String(prediction.csvRow);
        probability.textContent = formatPercent(prediction.probability);
        status.textContent = prediction.label === 1 ? "Sobre el umbral" : "Bajo el umbral";
        status.className = prediction.label === 1 ? "risk-high" : "risk-low";
        row.append(number, probability, status);
        table.append(row);
      }
      document.getElementById("batch-match-count").textContent =
        selected.length + " de " + (analyzedRows ? analyzedRows.length : 0) + " predicciones mostradas.";
      document.getElementById("batch-export").disabled = selected.length === 0;
    }

    for (const id of ["batch-risk-filter", "batch-contract-filter", "batch-sort"]) {
      document.getElementById(id).addEventListener("change", renderFilteredRows);
    }

    document.getElementById("batch-export").addEventListener("click", () => {
      const rows = currentFilteredRows();
      if (!rows.length) return;
      const csv = analytics.csvExport(rows);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = "churn-resultados-demo.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    });

    batchSubmit.addEventListener("click", async () => {
      if (!batchCustomers.length) return;
      const generation = batchGeneration;
      const submittedCustomers = batchCustomers.slice();
      batchError.hidden = true;
      batchSubmit.disabled = true;
      batchSubmit.textContent = "Analizando clientes…";
      try {
        const response = await fetch("/predict/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ customers: submittedCustomers })
        });
        if (!response.ok) {
          if (response.status === 429) throw new Error("Límite de solicitudes alcanzado. Prueba dentro de un minuto.");
          if (response.status === 413) throw new Error("El lote es demasiado grande. Reduce el tamaño.");
          throw new Error("Error de la API al analizar clientes (HTTP " + response.status + "). Revisa el archivo.");
        }
        const result = await response.json();
        if (generation !== batchGeneration) return;
        if (!Array.isArray(result.predictions) || result.predictions.length !== submittedCustomers.length ||
            result.summary.total_customers !== submittedCustomers.length) {
          throw new Error("Respuesta inesperada del modelo.");
        }
        const summary = result.summary;
        document.getElementById("batch-total").textContent = String(summary.total_customers);
        document.getElementById("batch-high").textContent = String(summary.high_risk_customers);
        document.getElementById("batch-average").textContent = formatPercent(summary.average_churn_probability);
        document.getElementById("batch-threshold").textContent = formatPercent(summary.decision_threshold);
        document.getElementById("batch-revenue").textContent =
          new Intl.NumberFormat("es-ES", { style: "currency", currency: "USD" }).format(summary.high_risk_monthly_charges);
        const highFraction = 100 * summary.high_risk_customers / summary.total_customers;
        document.getElementById("chart-high").style.width = highFraction + "%";
        document.getElementById("chart-low").style.width = (100 - highFraction) + "%";
        document.getElementById("batch-chart").setAttribute("aria-label",
          summary.high_risk_customers + " por encima del umbral y " + summary.low_risk_customers + " por debajo");
        document.getElementById("legend-high").textContent = String(summary.high_risk_customers);
        document.getElementById("legend-low").textContent = String(summary.low_risk_customers);

        analyzedRows = analytics.validateRows(submittedCustomers, result.predictions);
        showContractSegments(analyzedRows);
        renderFilteredRows();
        document.getElementById("batch-empty").hidden = true;
        document.getElementById("batch-metrics").hidden = false;
        document.getElementById("batch-results").hidden = false;
      } catch (problem) {
        if (generation === batchGeneration) {
          batchError.textContent = problem instanceof Error ? problem.message : "Error inesperado.";
          batchError.hidden = false;
        }
      } finally {
        batchSubmit.disabled = batchCustomers.length === 0;
        batchSubmit.textContent = "Analizar clientes →";
      }
    });
