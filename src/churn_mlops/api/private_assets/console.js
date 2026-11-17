/* Churn Lab staging console: no frameworks, analytics, storage or third-party calls.
 * ALL server data is inserted via textContent; never via innerHTML.
 * Invitation codes are deliberately ephemeral in the DOM only.
 */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const feedback = $("feedback");
  const picker = $("org-picker");
  let organizations = [];
  let currentId = "";
  let runsCache = [];
  let runsRequest = 0;

  function message(text, error = false) {
    feedback.textContent = text;
    feedback.classList.toggle("error", error);
  }

  function clearCode() {
    $("issued-code").textContent = "";
    $("one-time-code").hidden = true;
  }

  function apiError(status) {
    if (status === 401) return "Tu sesión ha caducado. Vuelve a iniciar sesión.";
    if (status === 403) return "No tienes permisos para realizar esta acción.";
    if (status === 404) return "No se ha encontrado el recurso solicitado.";
    if (status === 415 || status === 422) return "Revisa los datos introducidos.";
    if (status === 429) return "Demasiadas solicitudes; prueba de nuevo más tarde.";
    if (status >= 500) return "El servicio privado de pruebas no está disponible.";
    return "No se ha podido completar la operación.";
  }

  async function api(method, url, payload = undefined) {
    const options = {
      method, credentials: "same-origin", cache: "no-store",
      redirect: "error",
      headers: { Accept: "application/json" }
    };
    if (payload !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(payload);
    }
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("No se ha podido conectar al servidor.");
    }
    if (response.status === 401) {
      message(apiError(401), true);
    }
    if (!response.ok) throw new Error(apiError(response.status));
    return response.json();
  }

  function option(value, label) {
    const el = document.createElement("option");
    el.value = value;
    el.textContent = label;
    return el;
  }

  function scenarioLabel(key) {
    return ({
      baseline: "Mixto", at_risk: "Riesgo potencial", loyal: "Fidelización"
    })[key] || "Mixto";
  }

  function scenarioHelp(key) {
    return ({
      baseline: "Tres ejemplos originales con distintas condiciones.",
      at_risk: "Tres casos ficticios de contratos mensuales y poca antigüedad.",
      loyal: "Tres casos ficticios de contratos largos y mayor antigüedad."
    })[key] || "Selecciona un escenario ficticio.";
  }

  $("run-scenario").addEventListener("change", () => {
    $("scenario-explanation").textContent = scenarioHelp($("run-scenario").value);
  });

  function roleLabel(role) {
    return ({ owner: "Propietario", analyst: "Analista", viewer: "Lector" })[role] || "Sin permisos";
  }

  function currentOrg() {
    return organizations.find((org) => org.tenant_id === currentId);
  }

  async function reload(preferredId = currentId) {
    const result = await api("GET", "/saas/v3/organizations/list");
    organizations = Array.isArray(result.organizations) ? result.organizations : [];
    picker.replaceChildren(option("", "Selecciona una empresa"));
    for (const org of organizations) {
      picker.append(option(org.tenant_id, org.name + " · " + roleLabel(org.role)));
    }
    picker.disabled = organizations.length === 0;
    currentId = organizations.some((o) => o.tenant_id === preferredId)
      ? preferredId : (organizations[0]?.tenant_id || "");
    picker.value = currentId;
    await selectOrg();
  }

  async function selectOrg() {
    clearCode();
    const org = currentOrg();
    const owner = Boolean(org && org.role === "owner");
    $("owner-tools").hidden = !owner;
    $("invite-tools").hidden = !owner;
    $("member-list").replaceChildren();
    $("overview-tools").hidden = !org;
    $("alias-tools").hidden = !org;
    $("alias-input").value = org?.alias || "";
    $("overview-org-name").textContent = org ? org.name : "";
    runsCache = [];
    runsRequest += 1; // Invalidate in-flight requests for previously selected companies.
    renderHistory();
    $("analysis-tools").hidden = !org;
    $("analysis-result").replaceChildren();
    $("analysis-result").hidden = true;
    $("comparison-results").replaceChildren();
    $("comparison-results").hidden = true;
    $("run-scenario").value = "baseline";
    $("scenario-explanation").textContent = scenarioHelp("baseline");
    $("run-sample").disabled = !org || !["owner", "analyst"].includes(org.role);
    $("compare-scenarios").disabled = $("run-sample").disabled;
    $("run-permission-hint").textContent = !org ? "" : org.role === "viewer"
      ? "Tu rol de lector permite consultar el historial, pero no ejecutar análisis."
      : "Los análisis usan el modelo de IA y tres perfiles ficticios preparados.";
    $("org-role").textContent = org ? "Tu rol: " + roleLabel(org.role) : "";
    $("selected-hint").textContent = !org
      ? "Todavía no tienes empresas. Crea una ficticia o acepta una invitación."
      : owner ? "Puedes administrar las personas de esta empresa."
        : "Tu acceso es de " + roleLabel(org.role).toLowerCase()
          + ". Solo un propietario puede administrar miembros e invitaciones.";
    if (owner) await loadMembers();
    if (org) await loadRuns();
  }

  function memberRow(member, tenantId) {
    const item = document.createElement("div");
    item.className = "member";
    const name = document.createElement("span");
    name.className = "subject";
    const displayName = member.alias || "Miembro sin alias · #" + member.subject.slice(0, 8);
    name.textContent = displayName;
    const actions = document.createElement("div");
    actions.className = "member-actions";
    const label = document.createElement("label");
    label.className = "sr-only";
    label.textContent = "Rol de " + displayName;
    const roles = document.createElement("select");
    roles.setAttribute("aria-label", "Rol de " + displayName);
    for (const value of ["viewer", "analyst", "owner"]) {
      roles.append(option(value, roleLabel(value)));
    }
    roles.value = member.role;
    roles.addEventListener("change", async () => {
      const previous = member.role;
      roles.disabled = true;
      try {
        await api("PATCH", "/saas/v3/organizations/" + encodeURIComponent(tenantId)
          + "/members/" + encodeURIComponent(member.subject), { role: roles.value });
        message("Permisos actualizados.");
        await reload(currentId);
      } catch (error) {
        roles.value = previous;
        message(error.message, true);
      } finally {
        roles.disabled = false;
      }
    });
    const revoke = document.createElement("button");
    revoke.type = "button";
    revoke.className = "danger";
    revoke.textContent = "Revocar";
    revoke.setAttribute("aria-label", "Revocar acceso de " + displayName);
    revoke.addEventListener("click", async () => {
      if (!window.confirm("¿Revocar el acceso de este miembro? Esta acción es inmediata.")) return;
      revoke.disabled = true;
      try {
        await api("DELETE", "/saas/v3/organizations/" + encodeURIComponent(tenantId)
          + "/members/" + encodeURIComponent(member.subject));
        message("Acceso revocado.");
        await reload(currentId);
      } catch (error) {
        message(error.message, true);
      } finally {
        revoke.disabled = false;
      }
    });
    actions.append(roles, revoke);
    item.append(name, actions);
    return item;
  }

  async function loadMembers() {
    const org = currentOrg();
    if (!org || org.role !== "owner") return;
    const tenantId = org.tenant_id;
    const data = await api(
      "GET", "/saas/v3/organizations/" + encodeURIComponent(tenantId) + "/members"
    );
    // A slower response for company A must never paint over company B.
    if (currentId !== tenantId || currentOrg()?.role !== "owner") return;
    const members = $("member-list");
    members.replaceChildren();
    if (!Array.isArray(data.members) || data.members.length === 0) {
      const empty = document.createElement("p");
      empty.textContent = "No hay miembros activos.";
      members.append(empty);
      return;
    }
    for (const member of data.members) members.append(memberRow(member, tenantId));
  }

  function validRun(run) {
    const total = Number(run.total_profiles);
    const high = Number(run.high_risk_customers);
    const when = Date.parse(run.created_at);
    return Number.isInteger(total) && total >= 1 && total <= 200
      && Number.isInteger(high) && high >= 0 && high <= total
      && Number.isFinite(when);
  }

  function visibleRuns() {
    // Server returns no more than 20, newest first. Never claim a full-period search.
    const period = $("run-period").value;
    const cutoff = period === "7" || period === "30"
      ? Date.now() - Number(period) * 86400000 : -Infinity;
    return runsCache.filter((run) => validRun(run) && Date.parse(run.created_at) >= cutoff);
  }

  function renderOverview(runs) {
    const profiles = runs.reduce((sum, run) => sum + Number(run.total_profiles), 0);
    const highRisk = runs.reduce((sum, run) => sum + Number(run.high_risk_customers), 0);
    $("overview-runs").textContent = String(runs.length);
    $("overview-profiles").textContent = String(profiles);
    $("overview-highrisk").textContent = String(highRisk);
    $("overview-rate").textContent = profiles
      ? (highRisk * 100 / profiles).toFixed(1) + "%" : "—";
    $("overview-state").textContent = runs.length
      ? "Indicadores del filtro actual: ejecuciones sintéticas repetidas, no clientes distintos."
      : "No hay análisis en el periodo seleccionado.";
  }

  function riskPercentage(run) {
    return 100 * Number(run.high_risk_customers) / Number(run.total_profiles);
  }

  function renderTrend(runs) {
    const plot = $("trend-plot");
    plot.replaceChildren();
    if (runs.length < 2) {
      $("trend-delta").textContent = "Se necesitan dos ejecuciones para comparar.";
    } else if ((runs[0].scenario || "baseline") !== (runs[1].scenario || "baseline")) {
      $("trend-delta").textContent = "Escenarios distintos: la diferencia no mide una evolución temporal.";
    } else {
      const difference = riskPercentage(runs[0]) - riskPercentage(runs[1]);
      $("trend-delta").textContent = Math.abs(difference) < 0.05
        ? "Sin cambio respecto al análisis anterior."
        : (difference > 0 ? "Sube " : "Baja ") +
          Math.abs(difference).toFixed(1) + " puntos porcentuales frente al anterior.";
    }
    if (!runs.length) {
      const empty = document.createElement("p");
      empty.className = "muted small";
      empty.textContent = "Todavía no hay datos para dibujar la evolución.";
      plot.append(empty);
      plot.setAttribute("aria-label", "Sin análisis en el periodo seleccionado");
      return;
    }
    const bars = document.createElement("div");
    bars.className = "risk-bars";
    const descriptions = [];
    for (const [index, run] of runs.slice().reverse().entries()) {
      const rate = riskPercentage(run);
      const date = new Date(run.created_at).toLocaleDateString("es-ES");
      const label = "Ejecución " + (index + 1) + " · " + date + " · " +
        scenarioLabel(run.scenario) + " · " + rate.toFixed(1) + "%";
      descriptions.push(label);
      const item = document.createElement("div");
      item.className = "risk-bar";
      item.title = label;
      const rateText = document.createElement("strong");
      rateText.className = "risk-bar-value";
      rateText.textContent = rate.toFixed(1) + "%";
      const column = document.createElement("div");
      column.className = "risk-bar-column";
      const fill = document.createElement("span");
      fill.className = "risk-bar-fill";
      // Percentages come from integer DB constraints; no untrusted style values.
      fill.style.height = rate.toFixed(1) + "%";
      column.append(fill);
      const sequence = document.createElement("span");
      sequence.className = "risk-bar-label";
      sequence.textContent = "Nº " + (index + 1);
      item.append(rateText, column, sequence);
      bars.append(item);
    }
    plot.append(bars);
    plot.setAttribute("aria-label", "Evolución cronológica del riesgo alto. " + descriptions.join(". "));
  }

  function renderHistory() {
    const runs = visibleRuns();
    renderOverview(runs);
    renderTrend(runs);
    const list = $("run-list");
    list.replaceChildren();
    if (!runs.length) {
      const empty = document.createElement("p");
      empty.className = "muted small";
      empty.textContent = runsCache.length
        ? "No hay análisis en el periodo seleccionado."
        : "Todavía no hay análisis guardados para esta empresa.";
      list.append(empty);
      return;
    }
    for (const run of runs) {
      const row = document.createElement("div");
      row.className = "member";
      const when = new Date(run.created_at).toLocaleString("es-ES");
      row.textContent = when + " · " + scenarioLabel(run.scenario) + " · " +
        run.total_profiles + " perfiles ficticios · " + run.high_risk_customers +
        " riesgo alto · " + riskPercentage(run).toFixed(1) + "% alto";
      list.append(row);
    }
  }

  async function loadRuns() {
    const org = currentOrg();
    if (!org) return;
    const tenantId = org.tenant_id;
    const requestId = ++runsRequest;
    const result = await api(
      "GET", "/saas/v3/organizations/" + encodeURIComponent(tenantId) + "/runs"
    );
    if (currentId !== tenantId || requestId !== runsRequest) return;
    runsCache = Array.isArray(result.runs) ? result.runs.slice(0, 20) : [];
    renderHistory();
  }

  $("run-period").addEventListener("change", renderHistory);

  $("refresh-runs").addEventListener("click", () => loadRuns().then(
    () => message("Historial actualizado."),
    (e) => message(e.message, true)
  ));

  function renderComparison(results) {
    const expected = ["baseline", "at_risk", "loyal"];
    if (!Array.isArray(results) || results.length !== 3 ||
        !expected.every((key) => results.some((r) => r.scenario === key))) {
      throw new Error("La comparación no ha devuelto los tres escenarios.");
    }
    const area = $("comparison-results");
    const cards = [];
    for (const key of expected) {
      const run = results.find((r) => r.scenario === key);
      const total = Number(run.total_profiles);
      const high = Number(run.high_risk_customers);
      const average = Number(run.average_churn_probability);
      if (!Number.isInteger(total) || total < 1 || total > 200 ||
          !Number.isInteger(high) || high < 0 || high > total ||
          !Number.isFinite(average) || average < 0 || average > 1) {
        throw new Error("Resultados de comparación no válidos.");
      }
      const card = document.createElement("div");
      card.className = "comparison-card";
      const title = document.createElement("h4");
      title.textContent = scenarioLabel(key);
      const percentage = document.createElement("strong");
      percentage.className = "comparison-rate";
      percentage.textContent = (high * 100 / total).toFixed(1) + "%";
      const counts = document.createElement("p");
      counts.textContent = high + " de " + total + " perfiles ficticios en riesgo alto.";
      const confidence = document.createElement("p");
      confidence.className = "muted small";
      confidence.textContent = "Probabilidad media del modelo: " +
        (average * 100).toFixed(1) + "%";
      card.append(title, percentage, counts, confidence);
      cards.push(card);
    }
    area.replaceChildren(...cards);
    area.hidden = false;
  }

  $("compare-scenarios").addEventListener("click", async () => {
    const org = currentOrg();
    if (!org || !["owner", "analyst"].includes(org.role)) return;
    const tenantId = org.tenant_id;
    $("compare-scenarios").disabled = true;
    $("run-sample").disabled = true;
    $("run-scenario").disabled = true;
    try {
      const response = await api(
        "POST", "/saas/v3/organizations/" + encodeURIComponent(tenantId) +
        "/runs/compare", {}
      );
      if (currentId !== tenantId) return;
      renderComparison(response.results);
      await loadRuns();
      if (currentId === tenantId) {
        message("Tres escenarios ficticios comparados y guardados en el historial.");
      }
    } catch (error) {
      if (currentId === tenantId) message(error.message, true);
    } finally {
      const canAnalyze = currentOrg() &&
        ["owner", "analyst"].includes(currentOrg().role);
      $("compare-scenarios").disabled = !canAnalyze;
      $("run-sample").disabled = !canAnalyze;
      $("run-scenario").disabled = false;
    }
  });

  $("run-sample").addEventListener("click", async () => {
    const org = currentOrg();
    if (!org || !["owner", "analyst"].includes(org.role)) return;
    const tenantId = org.tenant_id;
    const scenario = $("run-scenario").value;
    $("run-sample").disabled = true;
    $("compare-scenarios").disabled = true;
    $("run-scenario").disabled = true;
    try {
      const result = await api(
        "POST", "/saas/v3/organizations/" + encodeURIComponent(tenantId) +
        "/runs/sample", { scenario }
      );
      if (currentId !== tenantId) return;
      const summary = result.summary;
      const output = $("analysis-result");
      output.textContent = scenarioLabel(result.scenario) + " · " +
        summary.total_customers + " perfiles analizados · " +
        summary.high_risk_customers + " riesgo alto · " +
        summary.low_risk_customers + " riesgo bajo · Probabilidad media: " +
        (summary.average_churn_probability * 100).toFixed(1) + "%";
      output.hidden = false;
      await loadRuns();
      message("Análisis sintético guardado en el historial.");
    } catch (error) {
      message(error.message, true);
    } finally {
      $("run-scenario").disabled = false;
      $("run-sample").disabled = !currentOrg() ||
        !["owner", "analyst"].includes(currentOrg().role);
      $("compare-scenarios").disabled = $("run-sample").disabled;
    }
  });

  async function submit(form, fn) {
    const button = form.querySelector('button[type="submit"]');
    if (button) button.disabled = true;
    try {
      await fn();
    } catch (error) {
      message(error.message, true);
    } finally {
      if (button) button.disabled = false;
    }
  }

  $("save-alias").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(event.currentTarget, async () => {
      const tenantId = currentId;
      if (!currentOrg()) throw new Error("Selecciona una empresa.");
      const alias = $("alias-input").value.trim();
      if (alias.length < 2 || alias.length > 40
          || /[<>@\x00-\x1f\x7f]/.test(alias)) {
        throw new Error("Usa un alias ficticio válido de 2 a 40 caracteres.");
      }
      await api("PUT", "/saas/v3/organizations/" + encodeURIComponent(tenantId)
        + "/me/alias", { alias });
      await reload(currentId);
      message("Alias actualizado para tu empresa de pruebas.");
    });
  });

  $("create-org").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(event.currentTarget, async () => {
      const name = $("org-name").value.trim();
      if (!name || name.length > 120) throw new Error("Indica un nombre de 1 a 120 caracteres.");
      const org = await api("POST", "/saas/v3/organizations", { name });
      $("org-name").value = "";
      await reload(org.tenant_id);
      message("Empresa de pruebas creada.");
    });
  });

  $("redeem-invite").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(event.currentTarget, async () => {
      const code = $("invite-input").value.trim();
      if (!/^[A-Za-z0-9_-]{43}$/.test(code)) throw new Error("El código debe tener 43 caracteres.");
      const response = await api("POST", "/saas/v3/invitations/accept", { code });
      $("invite-input").value = "";
      await reload(response.tenant_id);
      message("Invitación aceptada.");
    });
  });

  $("issue-invite").addEventListener("submit", (event) => {
    event.preventDefault();
    submit(event.currentTarget, async () => {
      if (currentOrg()?.role !== "owner") throw new Error("Solo los propietarios pueden invitar.");
      clearCode();
      const role = $("invite-role").value;
      const result = await api(
        "POST", "/saas/v3/organizations/" + encodeURIComponent(currentId) + "/invitations",
        { role }
      );
      $("issued-code").textContent = result.code;
      $("one-time-code").hidden = false;
      message("Código generado. Compártelo de forma privada.");
    });
  });

  $("hide-code").addEventListener("click", clearCode);
  $("refresh").addEventListener("click", () => reload().then(
    () => message("Lista actualizada."),
    (e) => message(e.message, true)
  ));
  $("refresh-members").addEventListener("click", () => loadMembers().then(
    () => message("Miembros actualizados."),
    (e) => message(e.message, true)
  ));
  picker.addEventListener("change", () => {
    currentId = picker.value;
    selectOrg().catch((e) => message(e.message, true));
  });

  // Fixed, generated SaaS-only reference. Never uses company history, uploads,
  // browser persistence or customer predictions. Only aggregated numbers.
  let demoResults = null;
  let demoSelectedMetric = "roc_auc";
  const demoTabNames = ["performance", "quality", "robustness"];
  const demoMetricLabels = {
    roc_auc: "ROC-AUC", pr_auc: "PR-AUC"
  };
  const demoFeatureLabels = {
    tenure_months: "Antigüedad (meses)",
    mrr_eur: "MRR (€)",
    licensed_seats: "Licencias",
    active_days_30d: "Días activos (30 días)",
    support_tickets_30d: "Tickets (30 días)"
  };
  const demoStressLabels = {
    activity_5_days_less: "5 días menos de actividad",
    support_2_more_tickets: "2 tickets adicionales",
    mrr_20_percent_lower: "MRR un 20 % menor"
  };

  function demoDecimal(value, digits = 3) {
    return Number(value).toLocaleString("es-ES", {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    });
  }

  function demoPct(value) {
    return demoDecimal(value * 100, 1) + " %";
  }

  function selectDemoTab(name, shouldFocus = false) {
    if (!demoTabNames.includes(name)) return;
    for (const key of demoTabNames) {
      const selected = key === name;
      const tab = $("demo-tab-" + key);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $("demo-panel-" + key).hidden = !selected;
    }
    if (shouldFocus) $("demo-tab-" + name).focus();
  }

  demoTabNames.forEach((name, index) => {
    const tab = $("demo-tab-" + name);
    tab.addEventListener("click", () => selectDemoTab(name));
    tab.addEventListener("keydown", (event) => {
      const next = ({
        ArrowRight: (index + 1) % demoTabNames.length,
        ArrowLeft: (index + demoTabNames.length - 1) % demoTabNames.length,
        Home: 0, End: demoTabNames.length - 1
      })[event.key];
      if (next === undefined) return;
      event.preventDefault();
      selectDemoTab(demoTabNames[next], true);
    });
  });

  function renderDemoMetric() {
    if (!demoResults) return;
    const key = demoSelectedMetric;
    const model = demoResults.model[key], baseline = demoResults.baseline[key];
    for (const metric of Object.keys(demoMetricLabels)) {
      $("demo-metric-" + (metric === "roc_auc" ? "roc" : "pr"))
        .setAttribute("aria-pressed", String(metric === key));
    }
    $("demo-bar-model").style.width = (model * 100) + "%";
    $("demo-bar-baseline").style.width = (baseline * 100) + "%";
    $("demo-bar-model-value").textContent = demoDecimal(model);
    $("demo-bar-baseline-value").textContent = demoDecimal(baseline);
    $("demo-comparison-chart").setAttribute(
      "aria-label", demoMetricLabels[key] + " en cuentas ficticias: modelo "
      + demoDecimal(model) + "; línea base " + demoDecimal(baseline)
    );
    $("demo-metric-note").textContent = key === "roc_auc"
      ? "ROC-AUC mide ordenación de bajas ficticias y no-bajas. 0,5 equivale al azar."
      : "PR-AUC (average precision) refleja la precisión-recall de un problema con pocas bajas.";
  }

  for (const metric of Object.keys(demoMetricLabels)) {
    $("demo-metric-" + (metric === "roc_auc" ? "roc" : "pr"))
      .addEventListener("click", () => {
        demoSelectedMetric = metric;
        renderDemoMetric();
      });
  }

  function renderDemoQuality(report) {
    const rows = $("demo-quality-rows");
    rows.replaceChildren();
    for (const feature of report.quality.features) {
      const tr = document.createElement("tr");
      const columns = [
        demoFeatureLabels[feature.name] || "Variable del modelo",
        demoDecimal(feature.training_mean, 2),
        demoDecimal(feature.test_mean, 2),
        feature.psi_test === null ? "—" : demoDecimal(feature.psi_test)
      ];
      columns.forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.setAttribute("scope", "row");
        cell.textContent = value;
        tr.append(cell);
      });
      rows.append(tr);
    }
  }

  function renderDemoRobustness(report) {
    const list = $("demo-stress-cards");
    list.replaceChildren();
    for (const item of report.robustness.scenarios) {
      const card = document.createElement("article");
      card.className = "demo-eval-stress-card";
      const title = document.createElement("strong");
      title.textContent = demoStressLabels[item.scenario] || "Escenario ficticio";
      const change = document.createElement("p");
      const delta = item.mean_score_change_pp;
      change.textContent = "Cambio medio de puntuación: "
        + (delta > 0 ? "+" : "") + demoDecimal(delta, 2) + " puntos porcentuales";
      const flags = document.createElement("p");
      flags.textContent = "Perfiles sobre el umbral: "
        + item.risk_flags_above_validation_threshold
        + " · Coincidencia top 20 %: " + item.same_top_20pct_count
        + " de " + item.top_20pct_count;
      card.append(title, change, flags);
      list.append(card);
    }
  }

  function validDemoReport(data) {
    if (!data || data.demo_only !== true
        || data.source !== "fixed_seed_28_fictional_accounts"
        || data.not_real_customer_data !== true
        || data.not_validated_for_production !== true
        || data.accounts !== 1200
        || !data.splits || data.splits.training !== 700
        || data.splits.validation !== 200 || data.splits.test !== 300
        || !data.model || !data.baseline || !data.quality || !data.robustness
        || !Array.isArray(data.quality.features)
        || data.quality.features.length !== 5
        || !Array.isArray(data.robustness.scenarios)
        || data.robustness.scenarios.length !== 3
        || data.robustness.evaluated_split !== "validation_only"
        || !Number.isInteger(data.quality.distribution_flag_count)
        || data.quality.distribution_flag_count < 0) {
      return false;
    }
    for (const key of ["roc_auc", "pr_auc", "precision", "recall"]) {
      if (!Number.isFinite(data.model[key]) || data.model[key] < 0
          || data.model[key] > 1) return false;
    }
    for (const key of ["roc_auc", "pr_auc"]) {
      if (!Number.isFinite(data.baseline[key]) || data.baseline[key] < 0
          || data.baseline[key] > 1) return false;
    }
    for (const feature of data.quality.features) {
      if (typeof feature.name !== "string"
          || !Number.isFinite(feature.training_mean)
          || !Number.isFinite(feature.test_mean)
          || (feature.psi_test !== null && (!Number.isFinite(feature.psi_test)
          || feature.psi_test < 0))) return false;
    }
    for (const scenario of data.robustness.scenarios) {
      if (!demoStressLabels[scenario.scenario]
          || !Number.isFinite(scenario.mean_score_change_pp)
          || !Number.isInteger(scenario.risk_flags_above_validation_threshold)
          || !Number.isInteger(scenario.same_top_20pct_count)
          || !Number.isInteger(scenario.top_20pct_count)) return false;
    }
    return true;
  }

  function renderDemoEvaluation(report) {
    demoResults = report;
    $("demo-kpi-accounts").textContent = report.accounts.toLocaleString("es-ES", { useGrouping: "always" });
    $("demo-kpi-auc").textContent = demoDecimal(report.model.roc_auc);
    $("demo-kpi-recall").textContent = demoPct(report.model.recall);
    $("demo-kpi-flags").textContent = String(report.quality.distribution_flag_count);
    $("demo-splits").textContent = report.splits.training + " entreno · "
      + report.splits.validation + " validación · " + report.splits.test + " prueba";
    $("demo-test-precision").textContent = demoPct(report.model.precision);
    renderDemoMetric();
    renderDemoQuality(report);
    renderDemoRobustness(report);
    $("demo-eval-content").hidden = false;
    $("demo-eval-status").textContent = "Ejemplo ficticio cargado (semilla 28).";
  }

  async function loadDemoEvaluation() {
    $("demo-eval-retry").hidden = true;
    $("demo-eval-status").textContent = "Calculando resultados ficticios…";
    try {
      const report = await api("GET", "/saas/v3/demo-evaluation");
      if (!validDemoReport(report)) {
        throw new Error("No se ha podido validar el informe ficticio.");
      }
      renderDemoEvaluation(report);
    } catch {
      $("demo-eval-status").textContent =
        "No se pudo cargar el ejemplo ficticio. Puedes reintentarlo sin afectar a tus empresas.";
      $("demo-eval-retry").hidden = false;
    }
  }
  $("demo-eval-retry").addEventListener("click", loadDemoEvaluation);
  loadDemoEvaluation();

  reload("").then(() => message("Panel listo."), (e) => message(e.message, true));
})();
