// ==================================================
// Pitcher Research Lab
// Research Command Center
// Research analysis view
// ==================================================

(() => {
    "use strict";

    let researchData = null;
    let researchChanges = [];

    const pitchNames = {
        FF: "Four-Seam",
        SI: "Sinker",
        SL: "Slider",
        ST: "Sweeper",
        FS: "Splitter",
        CH: "Changeup",
        CU: "Curveball",
        FC: "Cutter",
        KC: "Knuckle Curve",
        SV: "Slurve",
        FO: "Forkball",
        SC: "Screwball",
        KN: "Knuckleball",
        EP: "Eephus",
        CS: "Slow Curve"
    };

    // ==================================================
    // Helpers
    // ==================================================

    function numberOrNull(value) {
        if (value === null || value === undefined || value === "") return null;
        const n = Number(value);
        return Number.isNaN(n) ? null : n;
    }

    function valueOrDash(value, decimals = 1) {
        const n = numberOrNull(value);
        return n === null ? "--" : n.toFixed(decimals);
    }

    function signed(value, decimals = 1) {
        const n = numberOrNull(value);
        if (n === null) return "--";
        return `${n > 0 ? "+" : ""}${n.toFixed(decimals)}`;
    }

    function formatDate(dateString) {
        if (!dateString) return "--";

        const date =
            new Date(
                `${dateString}T00:00:00`
            );

        if (Number.isNaN(date.getTime())) {
            return dateString;
        }

        return date.toLocaleDateString(
            "en-US",
            {
                month: "short",
                day: "numeric"
            }
        );
    }

    function percentagePointChange(value) {
        const n = numberOrNull(value);

        if (n === null) {
            return "--";
        }

        if (Math.abs(n) < 0.05) {
            return "No meaningful change";
        }

        return (
            `${n > 0 ? "↑" : "↓"} ` +
            `${Math.abs(n).toFixed(1)} percentage points`
        );
    }

    function runValueChange(value) {
        const n = numberOrNull(value);

        if (n === null) {
            return "--";
        }

        if (Math.abs(n) < 0.005) {
            return "No meaningful change";
        }

        return (
            `${n > 0 ? "↑" : "↓"} ` +
            `${Math.abs(n).toFixed(2)} runs / 100`
        );
    }

    function describeZScore(value) {
        const z =
            numberOrNull(
                value
            );

        if (z === null) {
            return "No z-score available";
        }

        const magnitude =
            Math.abs(
                z
            );

        const direction =
            z < 0
                ? "below"
                : "above";

        let meaning =
            "within normal historical variation";

        if (magnitude >= 3) {
            meaning =
                "very unusual relative to his history";
        }

        else if (magnitude >= 2) {
            meaning =
                "unusually large relative to his history";
        }

        else if (magnitude >= 1) {
            meaning =
                "noticeably different from his history";
        }

        return (
            `${magnitude.toFixed(2)} SD ` +
            `${direction} baseline — ${meaning}`
        );
    }

    function getPitchRows() {
        return Array.isArray(
            researchData?.pitches
        )
            ? researchData.pitches
            : [];
    }

    function getOverallRows() {
        return Array.isArray(
            researchData?.overall
        )
            ? researchData.overall
            : [];
    }

    function getPitchPeriod(
        pitchType,
        period
    ) {
        return (
            getPitchRows().find(
                row =>
                    String(
                        row.pitch_type
                        ??
                        ""
                    ).toUpperCase()
                    ===
                    pitchType

                    &&

                    String(
                        row.period
                        ??
                        ""
                    ).toLowerCase()
                    ===
                    period
            )
            ??
            null
        );
    }

    function getOverallPeriod(
        period
    ) {
        return (
            getOverallRows().find(
                row =>
                    String(
                        row.period
                        ??
                        ""
                    ).toLowerCase()
                    ===
                    period
            )
            ??
            null
        );
    }

    function getPitchCount(
        row
    ) {
        if (!row) {
            return 0;
        }

        return Number(
            row.pitch_count
            ??
            row.count
            ??
            row.pitches
            ??
            0
        );
    }

    function guideKey(
        row
    ) {
        const metric =
            String(
                row?.metric
                ??
                ""
            ).toLowerCase();

        if (
            metric.includes(
                "velocity"
            )
        ) {
            return "velocity";
        }

        if (
            metric.includes(
                "spin"
            )
        ) {
            return "spin";
        }

        if (
            metric.includes(
                "extension"
            )
        ) {
            return "extension";
        }

        if (
            metric.includes(
                "horizontal movement"
            )
        ) {
            return "horizontal_movement";
        }

        if (
            metric.includes(
                "vertical movement"
            )
        ) {
            return "vertical_movement";
        }

        if (
            metric.includes(
                "release x"
            )
        ) {
            return "release_x";
        }

        if (
            metric.includes(
                "release z"
            )
        ) {
            return "release_z";
        }

        return "z_score";
    }

    function infoButton(
        metricKey,
        label
    ) {
        return `
            <button
                type="button"
                class="metric-inline-info"
                title="Explain ${label}"
                aria-label="Explain ${label}"
                onclick="window.openMetricGuide && window.openMetricGuide('${metricKey}')"
            >
                i
            </button>
        `;
    }

    function createSignalRow(
        title,
        value,
        detail,
        metricKey = null
    ) {
        const row =
            document.createElement(
                "div"
            );

        row.className =
            "research-signal-row";

        row.innerHTML = `

            <div class="research-signal-main">

                <div class="research-signal-name">

                    ${title}

                    ${
                        metricKey
                            ? infoButton(
                                metricKey,
                                title
                            )
                            : ""
                    }

                </div>

                <div class="research-signal-detail">
                    ${detail}
                </div>

            </div>

            <div class="research-signal-value">
                ${value}
            </div>

        `;

        return row;
    }

    function showEmptyState(
        container,
        title,
        detail
    ) {
        if (!container) {
            return;
        }

        container.innerHTML = `

            <div class="research-signal-row">

                <div class="research-signal-main">

                    <div class="research-signal-name">
                        ${title}
                    </div>

                    <div class="research-signal-detail">
                        ${detail}
                    </div>

                </div>

                <div class="research-signal-value">
                    --
                </div>

            </div>

        `;
    }


    // ==================================================
    // Stuff / Physical Changes
    // ==================================================

    function rankedStuffChanges() {
        return researchChanges

            .filter(
                row =>
                    numberOrNull(
                        row.z_score
                    )
                    !==
                    null
            )

            .sort(
                (
                    a,
                    b
                ) =>
                    Math.abs(
                        Number(
                            b.z_score
                        )
                    )
                    -
                    Math.abs(
                        Number(
                            a.z_score
                        )
                    )
            );
    }

    function renderStuffSignals() {
        const container =
            document.getElementById(
                "stuff-signals"
            );

        if (!container) {
            return;
        }

        container.innerHTML =
            "";

        const rows =
            rankedStuffChanges()
                .slice(
                    0,
                    4
                );

        if (
            !rows.length
        ) {
            showEmptyState(
                container,
                "No physical-change flags available",
                "The API loaded, but no valid z-score rows were returned."
            );

            return;
        }

        rows.forEach(
            row => {

                const dateText =
                    row.first_sustained_change

                        ?

                        `First sustained flag: ${formatDate(
                            row.first_sustained_change
                        )}`

                        :

                        "Season-level deviation; no sustained date flag";

                container.appendChild(

                    createSignalRow(

                        row.metric
                        ??
                        "Detected change",

                        describeZScore(
                            row.z_score
                        ),

                        `${valueOrDash(
                            row.baseline_mean,
                            2
                        )} → ${valueOrDash(
                            row.current_mean,
                            2
                        )} ${row.unit ?? ""} • ${dateText}`,

                        guideKey(
                            row
                        )

                    )

                );

            }
        );
    }


    // ==================================================
    // Strategy / Usage
    // ==================================================

    function getUsageChanges() {
        const rows =
            [];

        Object.keys(
            pitchNames
        ).forEach(
            pitchType => {

                const early =
                    getPitchPeriod(
                        pitchType,
                        "early"
                    );

                const post =
                    getPitchPeriod(
                        pitchType,
                        "post"
                    );

                if (
                    !early
                    ||
                    !post
                ) {
                    return;
                }

                const earlyUsage =
                    numberOrNull(
                        early.usage_pct
                    );

                const postUsage =
                    numberOrNull(
                        post.usage_pct
                    );

                if (
                    earlyUsage === null
                    ||
                    postUsage === null
                ) {
                    return;
                }

                rows.push(
                    {
                        pitch_type:
                            pitchType,

                        early:
                            earlyUsage,

                        post:
                            postUsage,

                        delta:
                            postUsage
                            -
                            earlyUsage,

                        early_count:
                            getPitchCount(
                                early
                            ),

                        post_count:
                            getPitchCount(
                                post
                            )
                    }
                );

            }
        );

        return rows.sort(
            (
                a,
                b
            ) =>
                Math.abs(
                    b.delta
                )
                -
                Math.abs(
                    a.delta
                )
        );
    }

    function renderStrategySignals() {
        const container =
            document.getElementById(
                "strategy-signals"
            );

        if (!container) {
            return;
        }

        container.innerHTML =
            "";

        const rows =
            getUsageChanges()
                .slice(
                    0,
                    4
                );

        if (
            !rows.length
        ) {
            showEmptyState(
                container,
                "Usage comparison unavailable",
                "No pitch had both an early-period and later-period usage value. This is a data-shape issue, not evidence that nothing changed."
            );

            return;
        }

        rows.forEach(
            row => {

                container.appendChild(

                    createSignalRow(

                        pitchNames[
                            row.pitch_type
                        ]
                        ??
                        row.pitch_type,

                        percentagePointChange(
                            row.delta
                        ),

                        `${row.early.toFixed(
                            1
                        )}% → ${row.post.toFixed(
                            1
                        )}% usage • ${row.early_count} early pitches / ${row.post_count} later pitches`,

                        "usage"

                    )

                );

            }
        );
    }


    // ==================================================
    // Results
    // ==================================================

    function getResultChanges() {
        const rows =
            [];

        Object.keys(
            pitchNames
        ).forEach(
            pitchType => {

                const early =
                    getPitchPeriod(
                        pitchType,
                        "early"
                    );

                const post =
                    getPitchPeriod(
                        pitchType,
                        "post"
                    );

                if (
                    !early
                    ||
                    !post
                ) {
                    return;
                }

                const earlyWhiff =
                    numberOrNull(
                        early.whiff_pct
                    );

                const postWhiff =
                    numberOrNull(
                        post.whiff_pct
                    );

                const earlyRV =
                    numberOrNull(
                        early.run_value_per_100
                    );

                const postRV =
                    numberOrNull(
                        post.run_value_per_100
                    );

                const earlySwings =
                    Number(
                        early.swings
                        ??
                        0
                    );

                const postSwings =
                    Number(
                        post.swings
                        ??
                        0
                    );

                if (
                    earlyWhiff === null
                    ||
                    postWhiff === null
                    ||
                    earlyRV === null
                    ||
                    postRV === null
                    ||
                    earlySwings < 10
                    ||
                    postSwings < 10
                ) {
                    return;
                }

                rows.push(
                    {
                        pitch_type:
                            pitchType,

                        early_whiff:
                            earlyWhiff,

                        post_whiff:
                            postWhiff,

                        whiff_delta:
                            postWhiff
                            -
                            earlyWhiff,

                        early_rv:
                            earlyRV,

                        post_rv:
                            postRV,

                        rv_delta:
                            postRV
                            -
                            earlyRV,

                        early_swings:
                            earlySwings,

                        post_swings:
                            postSwings,

                        early_count:
                            getPitchCount(
                                early
                            ),

                        post_count:
                            getPitchCount(
                                post
                            )
                    }
                );

            }
        );

        return rows.sort(
            (
                a,
                b
            ) =>

                (
                    Math.abs(
                        b.rv_delta
                    )
                    +
                    Math.abs(
                        b.whiff_delta
                    )
                    /
                    10
                )

                -

                (
                    Math.abs(
                        a.rv_delta
                    )
                    +
                    Math.abs(
                        a.whiff_delta
                    )
                    /
                    10
                )
        );
    }

    function renderResultSignals() {
        const container =
            document.getElementById(
                "result-signals"
            );

        if (!container) {
            return;
        }

        container.innerHTML =
            "";

        const rows =
            getResultChanges()
                .slice(
                    0,
                    4
                );

        if (
            !rows.length
        ) {
            showEmptyState(
                container,
                "Performance comparison limited",
                "No pitch met the current minimum of 10 swings in both the early and later comparison periods."
            );

            return;
        }

        rows.forEach(
            row => {

                container.appendChild(

                    createSignalRow(

                        pitchNames[
                            row.pitch_type
                        ]
                        ??
                        row.pitch_type,

                        `Pitch value ${runValueChange(
                            row.rv_delta
                        )}`,

                        `Whiff Rate: ${row.early_whiff.toFixed(
                            1
                        )}% → ${row.post_whiff.toFixed(
                            1
                        )}% (${percentagePointChange(
                            row.whiff_delta
                        )}) • Pitch Value: ${signed(
                            row.early_rv,
                            2
                        )} → ${signed(
                            row.post_rv,
                            2
                        )} runs / 100`,

                        "pitch_value"

                    )

                );

            }
        );
    }


    // ==================================================
    // Research Summary
    // ==================================================

    function renderResearchStatement() {
        const element =
            document.getElementById(
                "research-signal-text"
            );

        if (!element) {
            return;
        }

        const earlyOverall =
            getOverallPeriod(
                "early"
            );

        const postOverall =
            getOverallPeriod(
                "post"
            );

        const strongestStuff =
            rankedStuffChanges()[0]
            ??
            null;

        const largestUsage =
            getUsageChanges()[0]
            ??
            null;

        const largestResult =
            getResultChanges()[0]
            ??
            null;

        const hasAnything =
            earlyOverall
            ||
            postOverall
            ||
            strongestStuff
            ||
            largestUsage
            ||
            largestResult;

        if (
            !hasAnything
        ) {
            element.innerHTML = `

                <strong>
                    Research summary unavailable.
                </strong>

                The API returned no usable overall,
                pitch-period, or change-detection rows.
                This indicates a data-loading problem
                rather than a baseball conclusion.

            `;

            return;
        }

        const sentences =
            [];

        if (
            earlyOverall
            &&
            postOverall
        ) {

            const earlyWhiff =
                numberOrNull(
                    earlyOverall.whiff_pct
                );

            const postWhiff =
                numberOrNull(
                    postOverall.whiff_pct
                );

            const earlyHardHit =
                numberOrNull(
                    earlyOverall.hard_hit_pct
                );

            const postHardHit =
                numberOrNull(
                    postOverall.hard_hit_pct
                );

            const earlyRV =
                numberOrNull(
                    earlyOverall.run_value_per_100
                );

            const postRV =
                numberOrNull(
                    postOverall.run_value_per_100
                );

            if (
                earlyWhiff !== null
                &&
                postWhiff !== null
                &&
                earlyRV !== null
                &&
                postRV !== null
            ) {

                const whiffDelta =
                    postWhiff
                    -
                    earlyWhiff;

                const rvDelta =
                    postRV
                    -
                    earlyRV;

                sentences.push(

                    `Overall Whiff Rate moved from ${earlyWhiff.toFixed(
                        1
                    )}% to ${postWhiff.toFixed(
                        1
                    )}%, while Pitch Value (RV/100) moved from ${signed(
                        earlyRV,
                        2
                    )} to ${signed(
                        postRV,
                        2
                    )}.`

                );

            }

            if (
                earlyHardHit !== null
                &&
                postHardHit !== null
            ) {

                const delta =
                    postHardHit
                    -
                    earlyHardHit;

                sentences.push(

                    `Hard-Hit Rate moved from ${earlyHardHit.toFixed(
                        1
                    )}% to ${postHardHit.toFixed(
                        1
                    )}% (${percentagePointChange(
                        delta
                    ).toLowerCase()}).`

                );

            }

        }

        if (
            strongestStuff
        ) {

            sentences.push(

                `The strongest current physical flag is ${String(
                    strongestStuff.metric
                    ??
                    "a pitch characteristic"
                ).toLowerCase()}, at ${describeZScore(
                    strongestStuff.z_score
                )}.`

            );

        }

        if (
            largestUsage
        ) {

            const name =
                pitchNames[
                    largestUsage.pitch_type
                ]
                ??
                largestUsage.pitch_type;

            sentences.push(

                `${name} usage changed from ${largestUsage.early.toFixed(
                    1
                )}% to ${largestUsage.post.toFixed(
                    1
                )}% (${percentagePointChange(
                    largestUsage.delta
                ).toLowerCase()}), showing that the arsenal itself was redistributed.`

            );

        }

        if (
            largestResult
        ) {

            const name =
                pitchNames[
                    largestResult.pitch_type
                ]
                ??
                largestResult.pitch_type;

            sentences.push(

                `${name} currently shows one of the largest qualifying pitch-level performance shifts, with Pitch Value changing ${runValueChange(
                    largestResult.rv_delta
                ).toLowerCase()}.`

            );

        }

        sentences.push(
            "Taken together, the public data support a multi-factor investigation of pitch characteristics, usage, location and contact quality. They do not establish injury, fatigue, mechanical intent or any other private cause."
        );

        element.textContent =
            sentences.join(
                " "
            );
    }


    // ==================================================
    // Early vs Later Comparison Table
    // ==================================================

    function renderComparisonTable() {
        const body =
            document.getElementById(
                "research-comparison-body"
            );

        if (!body) {
            return;
        }

        body.innerHTML =
            "";

        Object.keys(
            pitchNames
        ).forEach(
            pitchType => {

                const early =
                    getPitchPeriod(
                        pitchType,
                        "early"
                    );

                const post =
                    getPitchPeriod(
                        pitchType,
                        "post"
                    );

                if (
                    !early
                    &&
                    !post
                ) {
                    return;
                }

                const row =
                    document.createElement(
                        "tr"
                    );

                row.innerHTML = `

                    <td class="research-pitch-name">
                        ${pitchNames[pitchType] ?? pitchType}
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.usage_pct
                        )}%
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.usage_pct
                        )}%
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.avg_velocity,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.avg_velocity,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.avg_vmov,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.avg_vmov,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.whiff_pct
                        )}%
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.whiff_pct
                        )}%
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.run_value_per_100,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.run_value_per_100,
                            2
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            early?.avg_ev,
                            1
                        )}
                    </td>

                    <td>
                        ${valueOrDash(
                            post?.avg_ev,
                            1
                        )}
                    </td>

                `;

                body.appendChild(
                    row
                );

            }
        );
    }


    // ==================================================
    // Research Window
    // ==================================================

    function renderTransitionWindow() {
        const element =
            document.getElementById(
                "transition-window-value"
            );

        if (!element) {
            return;
        }

        const start =
            researchData
                ?.transition_window
                ?.start;

        const end =
            researchData
                ?.transition_window
                ?.end;

        element.textContent =

            start
            &&
            end

                ?

                `${formatDate(
                    start
                )} — ${formatDate(
                    end
                )}`

                :

                "Window unavailable";
    }


    // ==================================================
    // Initialize
    // ==================================================

    async function initializeResearchCenter() {
        try {
            if (window.pitcherResearchLab?.ready) {
                await window.pitcherResearchLab.ready;
            }
            if (!window.pitcherResearchLab?.pitcherId) return;

            const changesResponse =
                await fetch(
                    window.pitcherResearchLab.apiUrl("changes")
                );

            if (
                !changesResponse.ok
            ) {
                throw new Error(
                    "Change API failed."
                );
            }

            const changesPayload =
                await changesResponse.json();

            researchChanges =
                Array.isArray(
                    changesPayload
                )
                    ?
                    changesPayload
                    :
                    [];

            const windowRange =
                window.pitcherResearchLab.researchWindow(
                    researchChanges
                );

            const start =
                windowRange.start;

            const end =
                windowRange.end;

            if (!start || !end) {
                throw new Error(
                    "Not enough outings to define a research window."
                );
            }

            const researchResponse =
                await fetch(

                    window.pitcherResearchLab.apiUrl("research", {
                        start,
                        end,
                    })

                );

            if (
                !researchResponse.ok
            ) {
                throw new Error(
                    "Research API failed."
                );
            }

            researchData =
                await researchResponse.json();

            renderTransitionWindow();

            renderStuffSignals();

            renderStrategySignals();

            renderResultSignals();

            renderResearchStatement();

            renderComparisonTable();

        }

        catch (
            error
        ) {

            console.error(
                "Research Command Center error:",
                error
            );

            const element =
                document.getElementById(
                    "research-signal-text"
                );

            if (
                element
            ) {

                element.textContent =
                    "Research summary could not be loaded because the application encountered a data or API error. Check the browser console and Flask terminal for the underlying error.";

            }

        }
    }


    initializeResearchCenter();

})();
