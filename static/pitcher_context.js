(() => {
    "use strict";

    const STORAGE_KEY = "pitcherResearchLab.selectedPitcherId";
    const PROFILE_KEY = "pitcherResearchLab.selectedPitcherProfile";
    const SEASON_STORAGE_PREFIX = "pitcherResearchLab.researchSeason.";
    const WINDOW_STORAGE_PREFIX = "pitcherResearchLab.researchWindow.";
    const originalFetch = window.fetch.bind(window);

    let currentMeta = null;
    let resolveReady;
    let rejectReady;

    const ready = new Promise((resolve, reject) => {
        resolveReady = resolve;
        rejectReady = reject;
    });

    function selectedPitcherId() {
        const value = Number(localStorage.getItem(STORAGE_KEY));
        return Number.isFinite(value) && value > 0 ? value : null;
    }

    function selectedResearchSeason() {
        const stored = Number(
            localStorage.getItem(`${SEASON_STORAGE_PREFIX}${selectedPitcherId()}`)
        );
        if (Number.isFinite(stored) && stored >= 2015) return stored;

        const current = Number(currentMeta?.research_defaults?.target_season);
        return Number.isFinite(current) ? current : null;
    }

    function cachedProfile() {
        try {
            const profile = JSON.parse(localStorage.getItem(PROFILE_KEY) || "null");
            if (!profile || Number(profile.mlbam_id) !== selectedPitcherId()) return null;
            return profile;
        } catch (_) {
            return null;
        }
    }

    function apiPath(resource = "") {
        const pitcherId = selectedPitcherId();
        if (!pitcherId) return null;
        const clean = String(resource).replace(/^\/+|\/+$/g, "");
        return `/api/pitchers/${pitcherId}${clean ? `/${clean}` : ""}`;
    }

    function apiUrl(resource = "", params = {}) {
        const path = apiPath(resource);
        if (!path) return null;
        const url = new URL(path, window.location.origin);
        const season = selectedResearchSeason();
        if (season) url.searchParams.set("season", String(season));

        Object.entries(params || {}).forEach(([key, value]) => {
            if (value !== null && value !== undefined && value !== "") {
                url.searchParams.set(key, String(value));
            }
        });

        return `${url.pathname}${url.search}`;
    }

    function customResearchWindow() {
        const season = selectedResearchSeason();
        if (!season) return null;
        const key = `${WINDOW_STORAGE_PREFIX}${selectedPitcherId()}.${season}`;
        try {
            const saved = JSON.parse(localStorage.getItem(key) || "null");
            if (!saved?.start || !saved?.end || saved.start > saved.end) return null;
            if (!saved.start.startsWith(`${season}-`) || !saved.end.startsWith(`${season}-`)) return null;
            return { start: saved.start, end: saved.end, source: "custom" };
        } catch (_) {
            return null;
        }
    }

    function researchWindow(changes = []) {
        const custom = customResearchWindow();
        if (custom) return custom;

        const dates = (Array.isArray(changes) ? changes : [])
            .filter(row => row?.first_sustained_change)
            .map(row => row.first_sustained_change)
            .sort();

        if (dates.length) {
            return { start: dates[0], end: dates[dates.length - 1], source: "detected" };
        }

        const defaults = currentMeta?.research_defaults || {};
        return {
            start: defaults.transition_start || null,
            end: defaults.transition_end || null,
            source: "comparison",
        };
    }

    window.pitcherResearchLab = {
        get pitcherId() { return selectedPitcherId(); },
        get meta() { return currentMeta; },
        get season() { return selectedResearchSeason(); },
        apiPath,
        apiUrl,
        researchWindow,
        customResearchWindow,
        ready,
    };

    function setStatus(text, state = "") {
        const el = document.getElementById("database-status");
        if (!el) return;
        el.textContent = text;
        el.dataset.state = state;
    }

    function formatHand(code) {
        if (code === "R") return "RHP";
        if (code === "L") return "LHP";
        return code ? `${code}HP` : "MLB Pitcher";
    }

    function ensureLoadingOverlay() {
        let overlay = document.getElementById("pitcher-loading-overlay");
        if (overlay) return overlay;

        overlay = document.createElement("div");
        overlay.id = "pitcher-loading-overlay";
        overlay.className = "pitcher-loading-overlay";
        overlay.innerHTML = `
            <div class="pitcher-loading-card">
                <div class="pitcher-loading-kicker">PITCHER RESEARCH LAB</div>
                <div id="pitcher-loading-title" class="pitcher-loading-title">Preparing pitcher…</div>
                <div id="pitcher-loading-copy" class="pitcher-loading-copy">Checking the local research database.</div>
                <div class="pitcher-loading-bar"><span></span></div>
            </div>
        `;
        document.body.appendChild(overlay);
        return overlay;
    }

    function showLoading(profile = null, copy = null) {
        const overlay = ensureLoadingOverlay();
        overlay.classList.remove("pitcher-loading-error");
        const title = overlay.querySelector("#pitcher-loading-title");
        const description = overlay.querySelector("#pitcher-loading-copy");
        const name = profile?.name || (selectedPitcherId() ? `MLB Pitcher ${selectedPitcherId()}` : "pitcher");
        if (title) title.textContent = `Preparing ${name}`;
        if (description) description.textContent = copy || "Checking the local research database.";
        overlay.hidden = false;
        document.body.classList.add("pitcher-data-loading");
    }

    function hideLoading() {
        const overlay = document.getElementById("pitcher-loading-overlay");
        if (overlay) {
            overlay.hidden = true;
            overlay.classList.remove("pitcher-loading-error");
        }
        document.body.classList.remove("pitcher-data-loading");
    }

    function showLoadingError(message) {
        const overlay = ensureLoadingOverlay();
        const title = overlay.querySelector("#pitcher-loading-title");
        const description = overlay.querySelector("#pitcher-loading-copy");
        if (title) title.textContent = "Pitcher data could not load";
        if (description) description.textContent = message || "Check the Flask window for the exact error, then refresh the page.";
        overlay.classList.add("pitcher-loading-error");
        overlay.hidden = false;
        document.body.classList.add("pitcher-data-loading");
    }

    function renderProfile(profile, database = null) {
        if (!profile) return;

        const name = document.getElementById("pitcher-name");
        const details = document.getElementById("pitcher-details");
        const caseStudy = document.getElementById("sidebar-case-study");
        if (name) name.textContent = profile.name || `MLB Pitcher ${selectedPitcherId()}`;

        if (details) {
            const parts = [];
            if (profile.team_abbreviation) parts.push(profile.team_abbreviation);
            else if (profile.team_name) parts.push(profile.team_name);
            parts.push(formatHand(profile.pitch_hand));
            if (database?.seasons?.length) {
                const first = database.seasons[0];
                const last = database.seasons[database.seasons.length - 1];
                parts.push(first === last ? String(first) : `${first}–${last}`);
            }
            details.textContent = parts.filter(Boolean).join("  •  ");
        }

        if (caseStudy) {
            caseStudy.textContent = "Multi-Pitcher Research";
        }
    }

    function renderNeutralState() {
        currentMeta = null;
        document.body.classList.add("pitcher-neutral-state");

        const name = document.getElementById("pitcher-name");
        const details = document.getElementById("pitcher-details");
        const caseStudy = document.getElementById("sidebar-case-study");
        if (name) name.textContent = "Select a pitcher to begin";
        if (details) details.textContent = "Search for any MLB pitcher to build or open a research profile.";
        if (caseStudy) caseStudy.textContent = "Multi-Pitcher Research";

        ["research-season-select", "pitch-select", "investigation-pitch", "location-pitch", "career-pitch"]
            .forEach(id => {
                const select = document.getElementById(id);
                if (!select) return;
                select.innerHTML = '<option value="">Select a pitcher</option>';
                select.disabled = true;
            });

        setStatus("Waiting for pitcher selection", "neutral");
        hideLoading();
    }

    function renderResearchDefaults(meta) {
        const badge = document.getElementById("change-baseline-badge");
        if (!badge) return;
        const baseline = meta?.research_defaults?.baseline_seasons || [];
        const target = meta?.research_defaults?.target_season;
        if (baseline.length) {
            badge.textContent = `Baseline: ${baseline.join("–")}`;
        } else if (target) {
            badge.textContent = `Baseline: earlier ${target} outings`;
        } else {
            badge.textContent = "Baseline unavailable";
        }
    }

    function renderSeasonSelector(meta) {
        const select = document.getElementById("research-season-select");
        if (!select) return;

        const seasons = [...(meta?.database?.seasons || [])]
            .map(Number)
            .filter(Number.isFinite)
            .sort((a, b) => b - a);
        const target = Number(meta?.research_defaults?.target_season);

        select.innerHTML = "";
        if (!seasons.length) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No seasons";
            select.appendChild(option);
            select.disabled = true;
            return;
        }

        seasons.forEach(season => {
            const option = document.createElement("option");
            option.value = String(season);
            option.textContent = String(season);
            option.selected = season === target;
            select.appendChild(option);
        });
        select.disabled = seasons.length === 1;

        if (Number.isFinite(target)) {
            localStorage.setItem(`${SEASON_STORAGE_PREFIX}${selectedPitcherId()}`, String(target));
        }

        select.onchange = () => {
            const season = Number(select.value);
            if (!Number.isFinite(season)) return;
            localStorage.setItem(`${SEASON_STORAGE_PREFIX}${selectedPitcherId()}`, String(season));
            showLoading(currentMeta?.pitcher, `Switching the research season to ${season}.`);
            window.location.reload();
        };
    }

    function setupResearchWindowControls(meta) {
        const startInput = document.getElementById("research-window-start");
        const endInput = document.getElementById("research-window-end");
        const applyButton = document.getElementById("research-window-apply");
        const resetButton = document.getElementById("research-window-reset");
        const note = document.getElementById("research-window-note");
        if (!startInput || !endInput || !applyButton || !resetButton) return;

        const season = Number(meta?.research_defaults?.target_season);
        const key = `${WINDOW_STORAGE_PREFIX}${selectedPitcherId()}.${season}`;
        const custom = customResearchWindow();
        startInput.value = custom?.start || "";
        endInput.value = custom?.end || "";
        startInput.min = Number.isFinite(season) ? `${season}-01-01` : "";
        startInput.max = Number.isFinite(season) ? `${season}-12-31` : "";
        endInput.min = startInput.min;
        endInput.max = startInput.max;

        if (note) {
            note.textContent = custom
                ? "Custom mode is active for this pitcher and season."
                : "Optional. Auto mode uses sustained flags when available.";
        }

        applyButton.onclick = () => {
            const start = startInput.value;
            const end = endInput.value;
            if (!start || !end) {
                if (note) note.textContent = "Choose both a start and end date.";
                return;
            }
            if (start > end) {
                if (note) note.textContent = "The end date must be on or after the start date.";
                return;
            }
            localStorage.setItem(key, JSON.stringify({ start, end }));
            showLoading(meta?.pitcher, "Applying the custom research window.");
            window.location.reload();
        };

        resetButton.onclick = () => {
            localStorage.removeItem(key);
            showLoading(meta?.pitcher, "Returning to the automatic research window.");
            window.location.reload();
        };
    }

    function renderArsenal(arsenal) {
        const selects = ["pitch-select", "investigation-pitch", "location-pitch", "career-pitch"];
        selects.forEach(id => {
            const select = document.getElementById(id);
            if (!select) return;
            const previous = select.value;
            select.innerHTML = "";

            if (!Array.isArray(arsenal) || !arsenal.length) {
                const option = document.createElement("option");
                option.value = "";
                option.textContent = "No pitch data available";
                select.appendChild(option);
                return;
            }

            arsenal.forEach((pitch, index) => {
                const option = document.createElement("option");
                option.value = pitch.pitch_type;
                option.textContent = `${pitch.pitch_name || pitch.pitch_type} (${pitch.usage_pct ?? 0}%)`;
                if (pitch.pitch_type === previous || (!previous && index === 0)) option.selected = true;
                select.appendChild(option);
            });

            if (!select.value && select.options.length) select.options[0].selected = true;
        });
    }

    function needsAutomaticSync(profile, database) {
        if (!database?.pitch_rows) return true;
        if (Number(database?.official_outing_count || 0) === 0) return true;
        const stamp = profile?.last_statcast_sync;
        if (!stamp) return true;
        const age = Date.now() - new Date(stamp).getTime();
        return !Number.isFinite(age) || age > 6 * 60 * 60 * 1000;
    }

    async function fetchMeta(pitcherId) {
        const storedSeason = Number(localStorage.getItem(`${SEASON_STORAGE_PREFIX}${pitcherId}`));
        const query = Number.isFinite(storedSeason) && storedSeason >= 2015
            ? `?season=${encodeURIComponent(storedSeason)}`
            : "";
        const response = await originalFetch(`/api/pitchers/${pitcherId}/meta${query}`);
        const meta = await response.json();
        if (!response.ok) throw new Error(meta.error || "Could not load pitcher metadata.");
        return meta;
    }

    async function syncSelectedPitcher() {
        const pitcherId = selectedPitcherId();
        const profile = cachedProfile();
        showLoading(
            profile,
            "Downloading or updating Statcast data. First-time pitcher searches take longer because the local history is built once."
        );
        setStatus("Updating pitcher data…", "syncing");

        const response = await originalFetch(`/api/pitchers/${pitcherId}/sync`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                force_full: false,
                season: currentMeta?.research_defaults?.target_season || selectedResearchSeason(),
            }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Pitcher update failed.");
        return result;
    }

    function renderDatabaseStatus(meta) {
        const database = meta?.database || {};
        const parts = [];
        const season = currentMeta?.research_defaults?.target_season;
        if (season) parts.push(`Research ${season}`);
        if (database.last_game_date) parts.push(`Data through ${database.last_game_date}`);
        if (database.pitch_rows) parts.push(`${Number(database.pitch_rows).toLocaleString()} pitches`);
        if (database.outing_count) parts.push(`${Number(database.outing_count).toLocaleString()} outings`);
        setStatus(parts.length ? parts.join(" • ") : "No cached pitches", parts.length ? "ready" : "error");
    }

    async function initializeSelectedPitcher() {
        const pitcherId = selectedPitcherId();
        if (!pitcherId) {
            renderNeutralState();
            resolveReady(null);
            window.dispatchEvent(new CustomEvent("pitcherResearchLab:neutral"));
            return null;
        }

        document.body.classList.remove("pitcher-neutral-state");
        const cached = cachedProfile();
        renderProfile(cached);
        showLoading(cached);

        try {
            let meta = await fetchMeta(pitcherId);
            currentMeta = meta;
            renderProfile(meta.pitcher, meta.database);
            renderResearchDefaults(meta);
            renderSeasonSelector(meta);
            setupResearchWindowControls(meta);
            localStorage.setItem(PROFILE_KEY, JSON.stringify(meta.pitcher));

            if (needsAutomaticSync(meta.pitcher, meta.database)) {
                await syncSelectedPitcher();
                meta = await fetchMeta(pitcherId);
                currentMeta = meta;
                renderProfile(meta.pitcher, meta.database);
                renderResearchDefaults(meta);
                renderSeasonSelector(meta);
                setupResearchWindowControls(meta);
                localStorage.setItem(PROFILE_KEY, JSON.stringify(meta.pitcher));
            }

            renderArsenal(meta.database?.arsenal);

            if (!meta.database?.pitch_rows) {
                throw new Error(`No regular-season Statcast pitches were found for ${meta.pitcher?.name || "this pitcher"}.`);
            }

            renderDatabaseStatus(meta);
            hideLoading();
            resolveReady(meta);
            window.dispatchEvent(new CustomEvent("pitcherResearchLab:ready", { detail: meta }));
            return meta;
        } catch (error) {
            console.error(error);
            setStatus("Pitcher data unavailable", "error");
            showLoadingError(error.message);
            rejectReady(error);
            throw error;
        }
    }

    let searchTimer = null;
    function setupSearch() {
        const input = document.getElementById("pitcher-search-input");
        const results = document.getElementById("pitcher-search-results");
        if (!input || !results) return;

        function closeResults() {
            results.hidden = true;
            results.innerHTML = "";
        }

        input.addEventListener("input", () => {
            clearTimeout(searchTimer);
            const query = input.value.trim();
            if (query.length < 2) {
                closeResults();
                return;
            }

            searchTimer = setTimeout(async () => {
                results.hidden = false;
                results.innerHTML = '<div class="pitcher-search-message">Searching MLB pitchers…</div>';
                try {
                    const response = await originalFetch(`/api/pitchers/search?q=${encodeURIComponent(query)}`);
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.error || "Search failed.");
                    results.innerHTML = "";

                    if (!payload.length) {
                        results.innerHTML = '<div class="pitcher-search-message">No pitchers found.</div>';
                        return;
                    }

                    payload.forEach(player => {
                        const button = document.createElement("button");
                        button.type = "button";
                        button.className = "pitcher-search-result";
                        const detail = [player.team_name, player.pitch_hand ? `${player.pitch_hand}HP` : null, player.position]
                            .filter(Boolean).join(" • ");
                        const strong = document.createElement("strong");
                        strong.textContent = player.name || `MLB Player ${player.mlbam_id}`;
                        const span = document.createElement("span");
                        span.textContent = detail || `MLBAM ${player.mlbam_id}`;
                        button.append(strong, span);
                        button.addEventListener("click", () => {
                            localStorage.removeItem(PROFILE_KEY);
                            localStorage.setItem(STORAGE_KEY, String(player.mlbam_id));
                            localStorage.setItem(PROFILE_KEY, JSON.stringify(player));
                            closeResults();
                            input.value = "";
                            showLoading(player, "Switching pitcher and preparing the research data.");
                            window.location.reload();
                        });
                        results.appendChild(button);
                    });
                } catch (error) {
                    results.innerHTML = `<div class="pitcher-search-message">${error.message}</div>`;
                }
            }, 250);
        });

        document.addEventListener("click", event => {
            if (!results.contains(event.target) && event.target !== input) closeResults();
        });
    }

    setupSearch();
    initializeSelectedPitcher().catch(() => {});
})();
