(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  let payload = null;

  const percent = (value, digits = 1) => `${(Number(value || 0) * 100).toFixed(digits)}%`;
  const number = (value, digits = 2) => Number(value || 0).toFixed(digits);

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function syncControlsFromPayload() {
    if (!payload) return;
    byId("simulationCount").value = String(payload.simulations || 1000);
    byId("simulationSeed").value = String(payload.seed || 20261003);
    byId("parkFactor").value = String(payload.park_factor || 1);
    byId("weatherFactor").value = String(payload.weather_run_factor || 1);
    byId("awayTeamName").textContent = payload.away?.name || "Away";
    byId("homeTeamName").textContent = payload.home?.name || "Home";
    byId("awayStarterName").textContent = `Starter: ${payload.away?.starter?.name || "—"}`;
    byId("homeStarterName").textContent = `Starter: ${payload.home?.starter?.name || "—"}`;
    byId("matchupTitle").textContent = `${payload.away?.name || "Away"} at ${payload.home?.name || "Home"}`;
    byId("dataStatus").textContent = payload.data_status || "";
    byId("payloadEditor").value = JSON.stringify(payload, null, 2);
  }

  async function loadExample() {
    payload = await fetchJson("/api/game-simulator/example");
    syncControlsFromPayload();
    byId("results").classList.add("hidden");
    byId("runMessage").textContent = "Demo ratings loaded. The live MLB/Statcast adapter is the next data layer.";
  }

  async function loadHealth() {
    try {
      const health = await fetchJson("/api/game-simulator/health");
      byId("engineStatus").textContent = `${health.engine} · ${health.provider}`;
      byId("statusDetail").textContent = `PA provider: ${health.provider_validation}. Game layer: ${health.game_engine_validation}.`;
      document.querySelector(".status-dot").style.background = "#ffd27a";
    } catch (error) {
      byId("engineStatus").textContent = "Engine unavailable";
      byId("statusDetail").textContent = error.message;
      byId("statusDetail").classList.add("error");
    }
  }

  function currentPayload() {
    const edited = JSON.parse(byId("payloadEditor").value);
    edited.simulations = Number(byId("simulationCount").value);
    edited.seed = Number(byId("simulationSeed").value);
    edited.park_factor = Number(byId("parkFactor").value);
    edited.weather_run_factor = Number(byId("weatherFactor").value);
    return edited;
  }

  function renderScorelines(result) {
    const rows = result.top_scorelines.map((item) => `
      <tr>
        <td>${item.label}</td>
        <td>${percent(item.probability)}</td>
        <td>${item.count.toLocaleString()}</td>
      </tr>`).join("");
    byId("scorelineRows").innerHTML = rows;
  }

  function renderStarters(result) {
    const cards = [result.starters.away, result.starters.home].map((starter) => `
      <div class="stat-card">
        <small>${starter.name}</small>
        <strong>${number(starter.innings.mean, 1)} IP</strong>
        <small>Median ${number(starter.innings.median, 1)} · 80% range ${number(starter.innings.p10, 1)}–${number(starter.innings.p90, 1)}</small>
      </div>`).join("");
    byId("starterCards").innerHTML = cards;
  }

  function renderInnings(result) {
    const away = new Map(result.inning_run_outlook.away.map((row) => [row.inning, row]));
    const home = new Map(result.inning_run_outlook.home.map((row) => [row.inning, row]));
    const innings = [...new Set([...away.keys(), ...home.keys()])].sort((a, b) => a - b);
    const maxMean = Math.max(.25, ...innings.flatMap((inning) => [away.get(inning)?.mean_runs || 0, home.get(inning)?.mean_runs || 0]));
    byId("inningChart").innerHTML = innings.map((inning) => {
      const a = away.get(inning) || { mean_runs: 0, scoreless_probability: 1 };
      const h = home.get(inning) || { mean_runs: 0, scoreless_probability: 1 };
      return `
        <div class="inning-row">
          <strong>${inning}</strong>
          <div class="inning-bars">
            <small>${result.matchup.away}: ${number(a.mean_runs)} runs · ${percent(a.scoreless_probability)} scoreless</small>
            <div class="run-bar"><span style="width:${Math.max(1, a.mean_runs / maxMean * 100)}%"></span></div>
          </div>
          <div class="inning-bars">
            <small>${result.matchup.home}: ${number(h.mean_runs)} runs · ${percent(h.scoreless_probability)} scoreless</small>
            <div class="run-bar home"><span style="width:${Math.max(1, h.mean_runs / maxMean * 100)}%"></span></div>
          </div>
        </div>`;
    }).join("");
  }

  function pitcherList(title, rows) {
    return `<div class="bullpen-side"><h3>${title}</h3>${rows
      .filter((row) => row.role !== "starter")
      .map((row) => `<div class="pitcher-probability"><span>${row.name}<br><small>${row.role}</small></span><strong>${percent(row.probability)}</strong></div>`)
      .join("") || "<p class='small-note'>No reliever appearances in the selected simulations.</p>"}</div>`;
  }

  function renderBullpen(result) {
    byId("bullpenProbabilities").innerHTML = [
      pitcherList(result.matchup.away, result.pitcher_appearance_probabilities.away),
      pitcherList(result.matchup.home, result.pitcher_appearance_probabilities.home),
    ].join("");
  }

  function renderRepresentative(result) {
    const game = result.representative_game;
    byId("representativeScore").textContent = `${game.away_team} ${game.away_score}, ${game.home_team} ${game.home_score}`;
    const innings = [...new Set([
      ...Object.keys(game.inning_runs.away || {}).map(Number),
      ...Object.keys(game.inning_runs.home || {}).map(Number),
    ])].sort((a, b) => a - b);
    byId("representativeInnings").innerHTML = innings.map((inning) =>
      `<span>${inning}: ${game.inning_runs.away?.[inning] || 0}–${game.inning_runs.home?.[inning] || 0}</span>`
    ).join("");
    byId("representativeEvents").innerHTML = game.events.map((event) =>
      `<li><strong>${event.half === "top" ? "Top" : "Bottom"} ${event.inning}</strong> · ${event.description} ${event.runs_scored ? `(${event.runs_scored} run${event.runs_scored === 1 ? "" : "s"})` : ""} · ${event.away_score}–${event.home_score}</li>`
    ).join("");
  }

  function renderResult(result) {
    const away = result.matchup.away;
    const home = result.matchup.home;
    const awayWin = result.win_probabilities.away;
    const homeWin = result.win_probabilities.home;
    byId("awayWinTeam").textContent = away;
    byId("homeWinTeam").textContent = home;
    byId("awayWinProbability").textContent = percent(awayWin);
    byId("homeWinProbability").textContent = percent(homeWin);
    byId("awayWinBar").style.width = `${awayWin * 100}%`;
    byId("winIntervalNote").textContent = `Monte Carlo sampling interval: ${away} ${percent(result.win_probabilities.away_monte_carlo_95_interval[0])}–${percent(result.win_probabilities.away_monte_carlo_95_interval[1])}; ${home} ${percent(result.win_probabilities.home_monte_carlo_95_interval[0])}–${percent(result.win_probabilities.home_monte_carlo_95_interval[1])}.`;
    byId("awayRunTeam").textContent = away;
    byId("homeRunTeam").textContent = home;
    byId("awayExpectedRuns").textContent = number(result.runs.away.mean, 1);
    byId("homeExpectedRuns").textContent = number(result.runs.home.mean, 1);
    byId("awayRunRange").textContent = `80% range ${number(result.runs.away.p10, 0)}–${number(result.runs.away.p90, 0)}`;
    byId("homeRunRange").textContent = `80% range ${number(result.runs.home.p10, 0)}–${number(result.runs.home.p90, 0)}`;
    renderScorelines(result);
    renderStarters(result);
    renderInnings(result);
    renderBullpen(result);
    renderRepresentative(result);
    byId("modelInterpretation").textContent = result.model_status.interpretation;
    byId("providerTag").textContent = `PA provider: ${result.model_status.pa_provider} (${result.model_status.pa_provider_validation})`;
    byId("gameValidationTag").textContent = `Game layer: ${result.model_status.game_engine_validation}`;
    byId("results").classList.remove("hidden");
  }

  async function runSimulation() {
    const button = byId("runButton");
    button.disabled = true;
    byId("runMessage").classList.remove("error");
    byId("runMessage").textContent = "Running sequential game paths…";
    try {
      payload = currentPayload();
      const result = await fetchJson("/api/game-simulator/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderResult(result);
      byId("runMessage").textContent = `${result.simulation_count.toLocaleString()} games completed with a reproducible seed.`;
    } catch (error) {
      byId("runMessage").textContent = error.message;
      byId("runMessage").classList.add("error");
    } finally {
      button.disabled = false;
    }
  }

  byId("loadExampleButton").addEventListener("click", () => loadExample().catch((error) => {
    byId("runMessage").textContent = error.message;
    byId("runMessage").classList.add("error");
  }));
  byId("runButton").addEventListener("click", runSimulation);

  Promise.all([loadHealth(), loadExample()]).catch((error) => {
    byId("runMessage").textContent = error.message;
    byId("runMessage").classList.add("error");
  });
})();
