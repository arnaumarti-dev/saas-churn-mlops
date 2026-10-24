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
