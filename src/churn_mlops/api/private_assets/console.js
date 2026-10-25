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
    $("run-scenario").value = "baseline";
    $("scenario-explanation").textContent = scenarioHelp("baseline");
    $("run-sample").disabled = !org || !["owner", "analyst"].includes(org.role);
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

  $("run-sample").addEventListener("click", async () => {
    const org = currentOrg();
    if (!org || !["owner", "analyst"].includes(org.role)) return;
    const tenantId = org.tenant_id;
    const scenario = $("run-scenario").value;
    $("run-sample").disabled = true;
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

  reload("").then(() => message("Panel listo."), (e) => message(e.message, true));
})();
