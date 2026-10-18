(() => {
    "use strict";
    const METRICS = {
        release_pos_x: {value: "release-x-value", detail: "release-x-detail"},
        release_pos_z: {value: "release-z-value", detail: "release-z-detail"},
        release_extension: {value: "release-extension-value", detail: "release-extension-detail"},
        arm_angle: {value: "release-arm-angle-value", detail: "release-arm-angle-detail"},
    };
    let requestVersion = 0;

    function formatNumber(value, unit) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return null;
        return `${Number(value).toFixed(2)}${unit === "deg" ? "°" : unit ? ` ${unit}` : ""}`;
    }

    function renderMetric(key, row) {
        const card = document.querySelector(`[data-release-metric='${key}']`);
        const value = document.getElementById(METRICS[key].value);
        const detail = document.getElementById(METRICS[key].detail);
        if (!card || !value || !detail) return false;
        const baseline = formatNumber(row?.baseline_mean, row?.unit);
        const comparison = formatNumber(row?.current_mean, row?.unit);
        const delta = formatNumber(row?.change, row?.unit);
        if (!baseline && !comparison) {
            card.hidden = true;
            value.textContent = "--";
            detail.textContent = "No measured values for this pitch in either period.";
            return false;
        }
        card.hidden = false;
        value.textContent = comparison
            ? `${comparison}${delta ? ` (${row.change > 0 ? "+" : ""}${delta})` : ""}`
            : "Comparison unavailable";
        const detection = !row.screen_eligible
            ? "Sustained-change screen unavailable: too few eligible outings or no baseline variation."
            : row.first_sustained_change
                ? `Sustained deviation detected ${row.first_sustained_change}.`
                : "No sustained deviation detected.";
        detail.textContent = `${row.pitch_type}: ${baseline || "unavailable"} baseline (${row.baseline_outings} outings, ${row.baseline_pitches} measured pitches) to ${comparison || "unavailable"} comparison (${row.current_outings} outings, ${row.current_pitches} measured pitches). ${detection}`;
        return true;
    }

    async function loadReleaseProfile() {
        const version = ++requestVersion;
        const pitch = document.getElementById("pitch-select")?.value;
        const title = document.getElementById("release-context-title");
        const copy = document.getElementById("release-context-copy");
        Object.keys(METRICS).forEach(key => renderMetric(key, null));
        if (!pitch) return;
        title.textContent = `Loading ${pitch} release measurements…`;
        copy.textContent = window.pitcherResearchLab.periodText();
        try {
            const response = await fetch(window.pitcherResearchLab.apiUrl("release", {pitch}));
            const payload = await response.json();
            if (version !== requestVersion) return;
            if (!response.ok) throw new Error(payload.error || "Release comparison could not load.");
            const rows = payload.measurements || [];
            let shown = 0;
            Object.keys(METRICS).forEach(key => {
                const row = rows.find(row => row.metric_key === key && row.pitch_type === pitch);
                if (renderMetric(key, row)) shown += 1;
            });
            title.textContent = shown ? `${pitch}: measured release comparison` : `${pitch}: release measurements unavailable`;
            copy.textContent = `${shown} release metrics available. ${window.pitcherResearchLab.periodText(payload.comparison_periods)}. Values are means of measured outing averages; higher or lower does not by itself indicate improvement.`;
        } catch (error) {
            if (version !== requestVersion) return;
            title.textContent = "Release measurements unavailable";
            copy.textContent = error.message;
        }
    }

    async function initializeReleaseProfile() {
        const meta = await window.pitcherResearchLab.ready;
        if (!meta) return;
        document.getElementById("pitch-select")?.addEventListener("change", loadReleaseProfile);
        await loadReleaseProfile();
    }
    initializeReleaseProfile();
})();
