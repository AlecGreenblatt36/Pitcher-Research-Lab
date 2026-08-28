(() => {
    "use strict";

    const METRICS = {
        release_pos_x: {
            card: "[data-release-metric='release_pos_x']",
            value: "release-x-value",
            detail: "release-x-detail",
            label: "Release X",
        },
        release_pos_z: {
            card: "[data-release-metric='release_pos_z']",
            value: "release-z-value",
            detail: "release-z-detail",
            label: "Release Z",
        },
        release_extension: {
            card: "[data-release-metric='release_extension']",
            value: "release-extension-value",
            detail: "release-extension-detail",
            label: "Extension",
        },
        arm_angle: {
            card: "[data-release-metric='arm_angle']",
            value: "release-arm-angle-value",
            detail: "release-arm-angle-detail",
            label: "Arm angle",
        },
    };

    function formatNumber(value, unit) {
        const number = Number(value);
        if (!Number.isFinite(number)) return null;
        const suffix = unit === "deg" ? "°" : unit ? ` ${unit}` : "";
        return `${number.toFixed(2)}${suffix}`;
    }

    function preferredPitch(meta) {
        const selected = document.getElementById("pitch-select")?.value;
        if (selected) return selected;
        return meta?.database?.arsenal?.[0]?.pitch_type || null;
    }

    function selectMetricRow(rows, metricKey, pitchType) {
        const candidates = rows.filter(row => row?.metric_key === metricKey);
        return candidates.find(row => row.pitch_type === pitchType) || candidates[0] || null;
    }

    function renderMetric(metricKey, row) {
        const config = METRICS[metricKey];
        const card = document.querySelector(config.card);
        const value = document.getElementById(config.value);
        const detail = document.getElementById(config.detail);
        if (!card || !value || !detail) return false;

        const baseline = formatNumber(row?.baseline_mean, row?.unit);
        const comparison = formatNumber(row?.current_mean, row?.unit);
        const delta = formatNumber(row?.change, row?.unit);
        if (!baseline || !comparison || !delta) {
            card.hidden = true;
            return false;
        }

        card.hidden = false;
        const sign = Number(row.change) > 0 ? "+" : "";
        value.textContent = `${comparison} (${sign}${delta})`;
        const detection = row.first_sustained_change
            ? `Sustained deviation detected ${row.first_sustained_change}.`
            : "Comparison-window difference; no sustained deviation detected.";
        const baselineSeasons = Array.isArray(row.baseline_seasons)
            ? row.baseline_seasons.join("–")
            : "prior";
        detail.textContent = `${row.pitch_type || "Pitch"}: ${baseline} baseline (${baselineSeasons}) to ${comparison}. ${detection}`;
        return true;
    }

    async function initializeReleaseProfile() {
        if (window.pitcherResearchLab?.ready) {
            await window.pitcherResearchLab.ready;
        }
        if (!window.pitcherResearchLab?.pitcherId) return;

        const title = document.getElementById("release-context-title");
        const copy = document.getElementById("release-context-copy");

        try {
            const response = await fetch(window.pitcherResearchLab.apiUrl("changes"));
            if (!response.ok) throw new Error(`Release measurements API ${response.status}`);
            const rows = await response.json();
            const measurements = Array.isArray(rows) ? rows : [];
            const pitchType = preferredPitch(window.pitcherResearchLab.meta);
            let shown = 0;
            let detected = 0;

            Object.keys(METRICS).forEach(metricKey => {
                const row = selectMetricRow(measurements, metricKey, pitchType);
                if (renderMetric(metricKey, row)) {
                    shown += 1;
                    if (row.first_sustained_change) detected += 1;
                }
            });

            if (!shown) {
                if (title) title.textContent = "Release measurements unavailable for this sample";
                if (copy) copy.textContent = "The selected pitcher and season do not contain enough eligible outings with measured release fields for the current comparison.";
                return;
            }

            if (title) {
                title.textContent = detected
                    ? `${detected} sustained release deviation${detected === 1 ? "" : "s"} detected`
                    : "Measured release comparison; no sustained deviation detected";
            }
            if (copy) {
                copy.textContent = `${shown} release metric${shown === 1 ? " is" : "s are"} available for ${pitchType || "the selected pitch"}. Values compare the documented baseline with the selected target-season window; higher or lower measurements do not by themselves imply improvement or deterioration.`;
            }
        } catch (error) {
            if (title) title.textContent = "Release measurements unavailable";
            if (copy) copy.textContent = "The release comparison could not be loaded from the current pitcher sample.";
        }
    }

    initializeReleaseProfile();
})();
