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
    $("analysis-tools").hidden = !org;
    $("run-list").replaceChildren();
    $("analysis-result").replaceChildren();
    $("analysis-result").hidden = true;
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

  function memberRow(member) {
    const item = document.createElement("div");
    item.className = "member";
    const name = document.createElement("span");
    name.className = "subject";
    name.textContent = member.subject;
    const actions = document.createElement("div");
    actions.className = "member-actions";
    const label = document.createElement("label");
    label.className = "sr-only";
    label.textContent = "Rol de " + member.subject;
    const roles = document.createElement("select");
    roles.setAttribute("aria-label", "Rol de " + member.subject);
    for (const value of ["viewer", "analyst", "owner"]) {
      roles.append(option(value, roleLabel(value)));
    }
    roles.value = member.role;
    roles.addEventListener("change", async () => {
      const previous = member.role;
      roles.disabled = true;
      try {
        await api("PATCH", "/saas/v3/organizations/" + encodeURIComponent(currentId)
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
    revoke.setAttribute("aria-label", "Revocar acceso de " + member.subject);
    revoke.addEventListener("click", async () => {
      if (!window.confirm("¿Revocar el acceso de este miembro? Esta acción es inmediata.")) return;
      revoke.disabled = true;
      try {
        await api("DELETE", "/saas/v3/organizations/" + encodeURIComponent(currentId)
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
    const data = await api(
      "GET", "/saas/v3/organizations/" + encodeURIComponent(org.tenant_id) + "/members"
    );
    const members = $("member-list");
    members.replaceChildren();
    if (!Array.isArray(data.members) || data.members.length === 0) {
      const empty = document.createElement("p");
      empty.textContent = "No hay miembros activos.";
      members.append(empty);
      return;
    }
    for (const member of data.members) members.append(memberRow(member));
  }

  async function loadRuns() {
    const org = currentOrg();
    if (!org) return;
    const tenantId = org.tenant_id;
    const result = await api(
      "GET", "/saas/v3/organizations/" + encodeURIComponent(tenantId) + "/runs"
    );
    if (currentId !== tenantId) return;
    const list = $("run-list");
    list.replaceChildren();
    if (!Array.isArray(result.runs) || result.runs.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted small";
      empty.textContent = "Todavía no hay análisis guardados para esta empresa.";
      list.append(empty);
      return;
    }
    for (const run of result.runs) {
      const row = document.createElement("div");
      row.className = "member";
      const timestamp = new Date(run.created_at);
      const when = Number.isNaN(timestamp.getTime())
        ? "Fecha no disponible" : timestamp.toLocaleString("es-ES");
      row.textContent = when + " · " + run.total_profiles +
        " perfiles ficticios · " + run.high_risk_customers + " riesgo alto";
      list.append(row);
    }
  }

  $("refresh-runs").addEventListener("click", () => loadRuns().then(
    () => message("Historial actualizado."),
    (e) => message(e.message, true)
  ));

  $("run-sample").addEventListener("click", async () => {
    const org = currentOrg();
    if (!org || !["owner", "analyst"].includes(org.role)) return;
    const tenantId = org.tenant_id;
    $("run-sample").disabled = true;
    try {
      const result = await api(
        "POST", "/saas/v3/organizations/" + encodeURIComponent(tenantId) +
        "/runs/sample", {}
      );
      if (currentId !== tenantId) return;
      const summary = result.summary;
      const output = $("analysis-result");
      output.textContent = summary.total_customers + " perfiles analizados · " +
        summary.high_risk_customers + " riesgo alto · " +
        summary.low_risk_customers + " riesgo bajo · Probabilidad media: " +
        (summary.average_churn_probability * 100).toFixed(1) + "%";
      output.hidden = false;
      await loadRuns();
      message("Análisis sintético guardado en el historial.");
    } catch (error) {
      message(error.message, true);
    } finally {
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
