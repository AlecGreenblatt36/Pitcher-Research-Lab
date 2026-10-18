(() => {
    "use strict";

    // Requests cut off by a page navigation (season or research-window reloads) end as
    // "Failed to fetch". Those are not application errors; report them as warnings.
    let navigating = false;
    window.addEventListener("pagehide", () => { navigating = true; });
    window.addEventListener("beforeunload", () => { navigating = true; });
    window.prlReportError = function (...args) {
        const text = args.map((a) => (a && a.message) ? a.message : String(a)).join(" ");
        const aborted = /Failed to fetch|NetworkError|The user aborted|AbortError|Load failed/i.test(text);
        if (navigating || aborted) {
            console.warn(...args);
            return;
        }
        console.error(...args);
    };

    const STORAGE_KEY = "pitcherResearchLab.selectedPitcherId";
    const PROFILE_KEY = "pitcherResearchLab.selectedPitcherProfile";
    const SEASON_STORAGE_PREFIX = "pitcherResearchLab.researchSeason.";
    const WINDOW_STORAGE_PREFIX = "pitcherResearchLab.comparisonPeriods.";
    const originalFetch = window.fetch.bind(window);

    let currentMeta = null;
    let syncWarning = null;
    let resolveReady;

    const ready = new Promise(resolve => {
        resolveReady = resolve;
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

        // Every comparison endpoint receives the same explicit, inclusive dates.
        const shared = ["changes", "research", "release", "location", "performance", "career"].includes(resource);
        const queryParams = {...params};
        if (shared) {
            delete queryParams.start;
            delete queryParams.end;
            Object.assign(queryParams, customResearchWindow() || {});
        }
        Object.entries(queryParams).forEach(([key, value]) => {
            if (value !== null && value !== undefined && value !== "") {
                url.searchParams.set(key, String(value));
            }
        });

        return `${url.pathname}${url.search}`;
    }

    const PERIOD_KEYS = ["baseline_start", "baseline_end", "comparison_start", "comparison_end"];
    function periodError(periods) {
        if (!PERIOD_KEYS.every(key => /^\d{4}-\d{2}-\d{2}$/.test(periods?.[key] || ""))) {
            return "Choose all four dates.";
        }
        const {baseline_start: bs, baseline_end: be, comparison_start: cs, comparison_end: ce} = periods;
        if (bs > be || cs > ce || be >= cs) return "Periods must be ordered and must not overlap.";
        const db = currentMeta?.database;
        if (db && (bs < db.first_game_date || ce > db.last_game_date)) {
            return `Choose dates within cached coverage: ${db.first_game_date} to ${db.last_game_date}.`;
        }
        return null;
    }

    function customResearchWindow() {
        const key = `${WINDOW_STORAGE_PREFIX}${selectedPitcherId()}.${selectedResearchSeason()}`;
        try {
            const saved = JSON.parse(localStorage.getItem(key) || "null");
            return saved && !periodError(saved) ? saved : null;
        } catch (_) { return null; }
    }

    function comparisonParams() {
        const periods = customResearchWindow() || currentMeta?.research_defaults?.comparison_periods || {};
        return Object.fromEntries(PERIOD_KEYS.filter(key => periods[key]).map(key => [key, periods[key]]));
    }

    function researchWindow() {
        const periods = comparisonParams();
        return {...periods, start: periods.baseline_end, end: periods.comparison_start,
            source: customResearchWindow() ? "custom" : "automatic"};
    }

    function periodText(payload = null) {
        const p = payload || (() => {
            const dates = comparisonParams();
            return {baseline: {start: dates.baseline_start, end: dates.baseline_end},
                comparison: {start: dates.comparison_start, end: dates.comparison_end}};
        })();
        if (!p.baseline?.start || !p.comparison?.end) return "Comparison periods unavailable";
        return `Baseline: ${p.baseline.start} to ${p.baseline.end} • Comparison: ${p.comparison.start} to ${p.comparison.end}`;
    }

    window.pitcherResearchLab = {
        get pitcherId() { return selectedPitcherId(); },
        get meta() { return currentMeta; },
        get season() { return selectedResearchSeason(); },
        apiPath,
        apiUrl,
        researchWindow,
        comparisonParams,
        periodText,
        get syncWarning() { return syncWarning; },
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
        let actions = overlay.querySelector(".pitcher-loading-actions");
        if (!actions) {
            actions = document.createElement("div");
            actions.className = "pitcher-loading-actions";
            const retry = document.createElement("button");
            retry.type = "button";
            retry.textContent = "Retry";
            retry.onclick = () => window.location.reload();
            const choose = document.createElement("button");
            choose.type = "button";
            choose.textContent = "Choose another pitcher";
            choose.onclick = () => {
                localStorage.removeItem(STORAGE_KEY);
                localStorage.removeItem(PROFILE_KEY);
                window.location.reload();
            };
            actions.append(retry, choose);
            overlay.querySelector(".pitcher-loading-card").append(actions);
        }
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
        const dates = comparisonParams();
        const badge = document.getElementById("change-baseline-badge");
        if (badge) badge.textContent = dates.baseline_start
            ? `Baseline: ${dates.baseline_start} to ${dates.baseline_end}`
            : "Baseline unavailable";
        let context = document.getElementById("active-comparison-periods");
        if (!context) {
            context = document.createElement("p");
            context.id = "active-comparison-periods";
            context.className = "active-comparison-periods";
            document.querySelector(".pitcher-header").after(context);
        }
        context.textContent = periodText();
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
        const inputs = Object.fromEntries(PERIOD_KEYS.map(key => [key, document.getElementById(`research-${key.replaceAll("_", "-")}`)]));
        const apply = document.getElementById("research-window-apply");
        const reset = document.getElementById("research-window-reset");
        const note = document.getElementById("research-window-note");
        if (!apply || !reset || Object.values(inputs).some(input => !input)) return;
        const key = `${WINDOW_STORAGE_PREFIX}${selectedPitcherId()}.${selectedResearchSeason()}`;
        const dates = comparisonParams();
        Object.entries(inputs).forEach(([name, input]) => {
            input.value = dates[name] || "";
            input.min = meta.database.first_game_date || "";
            input.max = meta.database.last_game_date || "";
        });
        note.textContent = customResearchWindow()
            ? "Custom mode is active. These dates apply across the comparison views."
            : "Optional. Automatic periods use prior seasons, or earlier and later outing dates for a rookie.";
        apply.onclick = async () => {
            const periods = Object.fromEntries(Object.entries(inputs).map(([key, input]) => [key, input.value]));
            const error = periodError(periods);
            if (error) { note.textContent = error; return; }
            apply.disabled = true;
            try {
                const query = new URLSearchParams({...periods, season: selectedResearchSeason()});
                const response = await originalFetch(`${apiPath("research")}?${query}`);
                const payload = await response.json();
                if (!response.ok) throw new Error(payload.error || "These periods have no usable pitches.");
                localStorage.setItem(key, JSON.stringify(periods));
                showLoading(meta.pitcher, "Applying the baseline and comparison periods.");
                window.location.reload();
            } catch (error) { note.textContent = error.message; }
            finally { apply.disabled = false; }
        };
        reset.onclick = () => {
            localStorage.removeItem(key);
            showLoading(meta.pitcher, "Restoring automatic comparison periods.");
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

    function renderSyncWarning(meta) {
        setStatus(`Cached data through ${meta.database.last_game_date}`, "warning");
        let banner = document.getElementById("pitcher-sync-warning");
        if (!banner) {
            banner = document.createElement("div");
            banner.id = "pitcher-sync-warning";
            banner.className = "pitcher-sync-warning";
            banner.setAttribute("role", "status");
            document.querySelector(".topbar").after(banner);
        }
        banner.replaceChildren();
        const message = document.createElement("span");
        message.textContent = `Using cached data through ${meta.database.last_game_date}. ${syncWarning}`;
        const retry = document.createElement("button");
        retry.type = "button";
        retry.textContent = "Retry update";
        retry.onclick = async () => {
            retry.disabled = true;
            try {
                const result = await syncSelectedPitcher();
                if (["error", "partial"].includes(result.official_outings?.status)) {
                    throw new Error("Official game lines could not be fully refreshed.");
                }
                window.location.reload();
            } catch (error) {
                syncWarning = error.message;
                hideLoading();
                renderSyncWarning(meta);
            }
        };
        banner.append(message, retry);
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
                try {
                    const result = await syncSelectedPitcher();
                    if (["error", "partial"].includes(result.official_outings?.status)) {
                        syncWarning = "Official game lines could not be fully refreshed.";
                    }
                    meta = await fetchMeta(pitcherId);
                } catch (error) {
                    if (!meta.database?.pitch_rows) throw error;
                    syncWarning = error.message;
                }
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
            if (syncWarning) renderSyncWarning(meta);
            hideLoading();
            resolveReady(meta);
            window.dispatchEvent(new CustomEvent("pitcherResearchLab:ready", { detail: meta }));
            return meta;
        } catch (error) {
            window.prlReportError(error);
            setStatus("Pitcher data unavailable", "error");
            showLoadingError(error.message);
            resolveReady(null);
            return null;
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
                    results.textContent = error.message;
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
