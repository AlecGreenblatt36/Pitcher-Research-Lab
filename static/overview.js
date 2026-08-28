// ==================================================
// Pitcher Research Lab
// Executive Overview
// ==================================================

let overviewResearchData = null;
let overviewChangeData = [];

const overviewPitchNames = {
    FF: "Four-Seam",
    SI: "Sinker",
    SL: "Slider",
    ST: "Sweeper",
    FS: "Splitter",
    CH: "Changeup",
    CU: "Curveball",
    FC: "Cutter", KC: "Knuckle Curve", SV: "Slurve", FO: "Forkball", SC: "Screwball", KN: "Knuckleball", EP: "Eephus", CS: "Slow Curve"
};


// ==================================================
// Formatting Helpers
// ==================================================

function numberOrNull(value) {

    if (
        value === null
        ||
        value === undefined
        ||
        Number.isNaN(Number(value))
    ) {
        return null;
    }

    return Number(value);
}


function difference(earlyValue, laterValue) {
    const early = numberOrNull(earlyValue);
    const later = numberOrNull(laterValue);
    if (early === null || later === null) return null;
    return later - early;
}

function changeClass(value, higherIsBetter = true, threshold = 0.05) {
    const number = numberOrNull(value);
    if (number === null || Math.abs(number) < threshold) return "overview-change-neutral";
    const favorable = higherIsBetter ? number > 0 : number < 0;
    return favorable ? "overview-change-good" : "overview-change-bad";
}

function formatNumber(
    value,
    decimals = 1
) {

    const number = numberOrNull(value);

    if (number === null) {
        return "--";
    }

    return number.toFixed(decimals);
}


function formatSignedNumber(
    value,
    decimals = 1
) {

    const number = numberOrNull(value);

    if (number === null) {
        return "--";
    }

    const sign =
        number > 0
            ? "+"
            : "";

    return `${sign}${number.toFixed(decimals)}`;
}


function formatPercentagePointChange(
    value
) {

    const number = numberOrNull(value);

    if (number === null) {
        return "--";
    }

    if (Math.abs(number) < 0.05) {
        return "No meaningful change";
    }

    const arrow =
        number > 0
            ? "↑"
            : "↓";

    return (
        `${arrow} ${Math.abs(number).toFixed(1)} percentage points`
    );
}


function formatUnitChange(
    value,
    unit,
    decimals = 1
) {

    const number = numberOrNull(value);

    if (number === null) {
        return "--";
    }

    if (Math.abs(number) < 0.005) {
        return "No meaningful change";
    }

    const arrow =
        number > 0
            ? "↑"
            : "↓";

    return (
        `${arrow} ${Math.abs(number).toFixed(decimals)} ${unit}`
    );
}


function formatDate(
    dateString
) {

    if (!dateString) {
        return "--";
    }

    const date =
        new Date(
            `${dateString}T00:00:00`
        );

    return date.toLocaleDateString(
        "en-US",
        {
            month: "short",
            day: "numeric"
        }
    );
}


// ==================================================
// Data Helpers
// ==================================================

function getOverallPeriod(
    period
) {

    return (
        overviewResearchData
        ?.overall
        ?.find(
            row =>
                row.period === period
        )
        ??
        null
    );
}


function getPitchPeriod(
    pitchType,
    period
) {

    return (
        overviewResearchData
        ?.pitches
        ?.find(
            row =>
                row.pitch_type === pitchType
                &&
                row.period === period
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
        row.count
        ??
        row.pitches
        ??
        row.pitch_count
        ??
        0
    );
}


// ==================================================
// KPI Styling
// ==================================================

function setKpi(
    elementId,
    changeText,
    rangeText,
    resultClass
) {

    const valueElement =
        document.getElementById(
            elementId
        );

    if (!valueElement) {
        return;
    }

    valueElement.textContent =
        changeText;

    valueElement.classList.remove(
        "overview-change-good",
        "overview-change-bad",
        "overview-change-neutral"
    );

    valueElement.classList.add(
        resultClass
    );

    const card =
        valueElement.closest(
            ".overview-kpi"
        );

    if (!card) {
        return;
    }

    const small =
        card.querySelector(
            "small"
        );

    if (small) {
        small.textContent =
            rangeText;
    }

}


// ==================================================
// Executive Summary
// ==================================================

function renderExecutiveSummary() {

    const early =
        getOverallPeriod(
            "early"
        );

    const post =
        getOverallPeriod(
            "post"
        );

    if (
        !early
        ||
        !post
    ) {
        const findingElement = document.getElementById("overview-primary-finding");
        const detailElement = document.getElementById("overview-primary-detail");
        if (findingElement) {
            findingElement.textContent = "Overview limited by the selected sample";
        }
        if (detailElement) {
            detailElement.textContent = "Both baseline and comparison periods need usable pitches before an overall result direction can be summarized. The deeper views remain available for the measurements supported by this sample.";
        }
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

    const earlyHardHit =
        numberOrNull(
            early.hard_hit_pct
        );

    const postHardHit =
        numberOrNull(
            post.hard_hit_pct
        );

    const earlyEV =
        numberOrNull(
            early.avg_ev
        );

    const postEV =
        numberOrNull(
            post.avg_ev
        );

    const earlyRV =
        numberOrNull(
            early.run_value_per_100
        );

    const postRV =
        numberOrNull(
            post.run_value_per_100
        );


    const whiffDelta =
        difference(earlyWhiff, postWhiff);

    const hardHitDelta =
        difference(earlyHardHit, postHardHit);

    const evDelta =
        difference(earlyEV, postEV);

    const rvDelta =
        difference(earlyRV, postRV);


    // ==================================================
    // KPI Cards
    // ==================================================

    setKpi(

        "overview-whiff-delta",

        formatPercentagePointChange(
            whiffDelta
        ),

        `${formatNumber(
            earlyWhiff
        )}% → ${formatNumber(
            postWhiff
        )}%`,

        changeClass(whiffDelta, true, 0.5)

    );


    setKpi(

        "overview-hard-hit-delta",

        formatPercentagePointChange(
            hardHitDelta
        ),

        `${formatNumber(
            earlyHardHit
        )}% → ${formatNumber(
            postHardHit
        )}%`,

        changeClass(hardHitDelta, false, 0.5)

    );


    setKpi(

        "overview-ev-delta",

        formatUnitChange(
            evDelta,
            "mph"
        ),

        `${formatNumber(
            earlyEV
        )} → ${formatNumber(
            postEV
        )} mph`,

        changeClass(evDelta, false, 0.2)

    );


    setKpi(

        "overview-rv-delta",

        formatUnitChange(
            rvDelta,
            "runs / 100",
            2
        ),

        `${formatSignedNumber(
            earlyRV,
            2
        )} → ${formatSignedNumber(
            postRV,
            2
        )}`,

        changeClass(rvDelta, true, 0.1)

    );


    // ==================================================
    // Primary Finding
    // ==================================================

    const notableProfileChanges =
        overviewChangeData.filter(
            row => Math.abs(Number(row.z_score ?? 0)) >= 1.5
        );

    const sustainedProfileChanges =
        notableProfileChanges.filter(
            row => row.first_sustained_change
        );

    const resultDirection =
        rvDelta === null || Math.abs(rvDelta) < 0.20
            ? "stable"
            : rvDelta > 0
                ? "better"
                : "worse";

    let headline;

    if (!notableProfileChanges.length && resultDirection === "stable") {
        headline = "The selected pitcher is relatively stable in the current comparison.";
    } else if (!notableProfileChanges.length && resultDirection === "better") {
        headline = "Results improved without a large pitch-characteristic departure from baseline.";
    } else if (!notableProfileChanges.length && resultDirection === "worse") {
        headline = "Results moved backward without a large pitch-characteristic departure from baseline.";
    } else if (resultDirection === "better") {
        headline = "Parts of the pitch profile changed alongside better results.";
    } else if (resultDirection === "worse") {
        headline = "Parts of the pitch profile changed alongside weaker results.";
    } else {
        headline = "The pitch profile changed, while overall results stayed relatively steady.";
    }

    const detailParts = [];

    if (earlyWhiff !== null && postWhiff !== null) {
        detailParts.push(`Whiff rate: ${formatNumber(earlyWhiff)}% → ${formatNumber(postWhiff)}%.`);
    }

    if (earlyHardHit !== null && postHardHit !== null) {
        detailParts.push(`Hard-hit rate: ${formatNumber(earlyHardHit)}% → ${formatNumber(postHardHit)}%.`);
    }

    if (earlyEV !== null && postEV !== null) {
        detailParts.push(`Average exit velocity: ${formatNumber(earlyEV)} → ${formatNumber(postEV)} mph.`);
    }

    if (earlyRV !== null && postRV !== null) {
        detailParts.push(`Pitcher RV/100: ${formatSignedNumber(earlyRV, 2)} → ${formatSignedNumber(postRV, 2)}.`);
    }

    if (notableProfileChanges.length) {
        const flagText = sustainedProfileChanges.length
            ? `${sustainedProfileChanges.length} sustained flag${sustainedProfileChanges.length === 1 ? "" : "s"}`
            : `${notableProfileChanges.length} notable baseline departure${notableProfileChanges.length === 1 ? "" : "s"}`;
        detailParts.push(`The pitch-characteristic screen found ${flagText}.`);
    } else {
        detailParts.push("No pitch characteristic is currently at least 1.5 standard deviations from its comparison baseline.");
    }

    const detail = detailParts.join(" ");


    const findingElement =
        document.getElementById(
            "overview-primary-finding"
        );

    const detailElement =
        document.getElementById(
            "overview-primary-detail"
        );


    if (findingElement) {
        findingElement.textContent =
            headline;
    }


    if (detailElement) {
        detailElement.textContent =
            detail;
    }

}


// ==================================================
// Research Window
// ==================================================

function renderTransitionWindow() {

    const windowData =
        overviewResearchData
        ?.transition_window;

    if (!windowData) {
        return;
    }


    const element =
        document.getElementById(
            "overview-transition-window"
        );


    if (element) {

        element.textContent =

            `${formatDate(
                windowData.start
            )} — ${formatDate(
                windowData.end
            )}`;

    }

}


// ==================================================
// Stuff Signal
// ==================================================

function getFallbackStuffChange() {

    const saferRows =
        overviewChangeData.filter(
            row => {

                const pitch =
                    String(
                        row.pitch_type
                        ??
                        ""
                    )
                    .toUpperCase();


                const metric =
                    String(
                        row.metric
                        ??
                        ""
                    )
                    .toLowerCase();


                // Slider vertical-movement signal is treated
                // cautiously because classification/design
                // changes may influence this field.

                const classificationSensitive =

                    pitch === "SL"
                    &&
                    metric.includes(
                        "vertical"
                    );


                return !classificationSensitive;

            }
        );


    return (
        [...saferRows]
        .sort(
            (
                a,
                b
            ) =>

                Math.abs(
                    Number(
                        b.z_score
                        ??
                        0
                    )
                )

                -

                Math.abs(
                    Number(
                        a.z_score
                        ??
                        0
                    )
                )
        )[0]
        ??
        null
    );

}


function renderStuffSignal() {

    const change =
        overviewChangeData
            .filter(row => Math.abs(Number(row.z_score ?? 0)) >= 1.5)
            .sort((a, b) => Math.abs(Number(b.z_score)) - Math.abs(Number(a.z_score)))[0]
        ?? null;


    if (!change) {
        const title = document.getElementById("overview-stuff-title");
        const copy = document.getElementById("overview-stuff-copy");
        if (title) title.textContent = "No large physical departure detected";
        if (copy) copy.textContent = "Tracked pitch characteristics remain within 1.5 standard deviations of the current baseline screen.";
        return;
    }


    const title =
        document.getElementById(
            "overview-stuff-title"
        );


    const copy =
        document.getElementById(
            "overview-stuff-copy"
        );


    if (title) {

        title.textContent =
            change.metric
            ??
            "Physical pitch change";

    }


    if (copy) {

        const z =
            numberOrNull(
                change.z_score
            );


        const zText =

            z === null

                ? ""

                :

                `${Math.abs(
                    z
                ).toFixed(2)} standard deviations ${z < 0 ? "below" : "above"} the pitcher's historical baseline.`;


        const dateText =

            change.first_sustained_change

                ?

                ` The first sustained flag occurred on ${formatDate(
                    change.first_sustained_change
                )}.`

                :

                "";


        copy.textContent =

            `${formatNumber(
                change.baseline_mean,
                2
            )} ${change.unit ?? ""} historical baseline → ${formatNumber(
                change.current_mean,
                2
            )} ${change.unit ?? ""} in the target season. ${zText}${dateText} This tells us the change is unusual relative to the pitcher's own history; it does not tell us why it happened.`;

    }

}


// ==================================================
// Pitch Comparisons
// ==================================================

function buildPitchComparisons() {

    const comparisons =
        [];


    Object.keys(
        overviewPitchNames
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


            comparisons.push(
                {

                    pitch_type:
                        pitchType,

                    early:
                        early,

                    post:
                        post,

                    usage_delta:

                        Number(
                            post.usage_pct
                            ??
                            0
                        )

                        -

                        Number(
                            early.usage_pct
                            ??
                            0
                        ),

                    whiff_delta:

                        Number(
                            post.whiff_pct
                            ??
                            0
                        )

                        -

                        Number(
                            early.whiff_pct
                            ??
                            0
                        ),

                    rv_delta:

                        Number(
                            post.run_value_per_100
                            ??
                            0
                        )

                        -

                        Number(
                            early.run_value_per_100
                            ??
                            0
                        )

                }
            );

        }
    );


    return comparisons;

}


// ==================================================
// Arsenal Signal
// ==================================================

function renderArsenalSignal() {

    const comparisons =
        buildPitchComparisons();


    if (
        comparisons.length === 0
    ) {
        const title = document.getElementById("overview-arsenal-title");
        const copy = document.getElementById("overview-arsenal-copy");
        if (title) title.textContent = "Arsenal comparison unavailable";
        if (copy) copy.textContent = "The selected periods do not have enough overlapping pitch types for a usage comparison.";
        return;
    }


    const largest =
        [...comparisons]
        .sort(
            (
                a,
                b
            ) =>

                Math.abs(
                    b.usage_delta
                )

                -

                Math.abs(
                    a.usage_delta
                )
        )[0];


    const pitchName =
        overviewPitchNames[
            largest.pitch_type
        ]
        ??
        largest.pitch_type;


    const direction =
        largest.usage_delta > 0
            ? "increased"
            : "decreased";


    const title =
        document.getElementById(
            "overview-arsenal-title"
        );


    const copy =
        document.getElementById(
            "overview-arsenal-copy"
        );


    if (title) {

        title.textContent =
            `${pitchName} usage ${direction}`;

    }


    if (copy) {

        copy.textContent =

            `${pitchName} usage moved from ${formatNumber(
                largest.early.usage_pct
            )}% in the early period to ${formatNumber(
                largest.post.usage_pct
            )}% in the later period — a change of ${Math.abs(
                largest.usage_delta
            ).toFixed(1)} percentage points. This is a change in arsenal deployment, not automatically evidence that the pitch itself became better or worse.`;

    }

}


// ==================================================
// Performance Signal
// ==================================================

function renderPerformanceSignal() {

    const comparisons =
        buildPitchComparisons();


    const qualified =
        comparisons.filter(
            row =>

                getPitchCount(
                    row.early
                )
                >=
                60

                &&

                getPitchCount(
                    row.post
                )
                >=
                60
        );


    if (
        qualified.length === 0
    ) {
        const title = document.getElementById("overview-performance-title");
        const copy = document.getElementById("overview-performance-copy");
        if (title) title.textContent = "Performance sample is still limited";
        if (copy) copy.textContent = "No pitch currently has the minimum sample in both comparison periods for a pitch-level result summary.";
        return;
    }


    const largestResultShift =
        [...qualified]
        .sort(
            (a, b) =>
                Math.abs(b.rv_delta)
                -
                Math.abs(a.rv_delta)
        )[0];


    const pitchName =
        overviewPitchNames[
            largestResultShift.pitch_type
        ]
        ??
        largestResultShift.pitch_type;


    const title =
        document.getElementById(
            "overview-performance-title"
        );


    const copy =
        document.getElementById(
            "overview-performance-copy"
        );


    if (title) {
        const direction = Math.abs(largestResultShift.rv_delta) < 0.10
            ? "was stable"
            : largestResultShift.rv_delta > 0
                ? "improved"
                : "declined";
        title.textContent = `${pitchName} pitch value ${direction}`;
    }


    if (copy) {

        copy.textContent =

            `${pitchName} pitcher run value moved from ${formatSignedNumber(
                largestResultShift.early.run_value_per_100,
                2
            )} to ${formatSignedNumber(
                largestResultShift.post.run_value_per_100,
                2
            )} runs per 100 pitches. Whiff rate changed by ${Math.abs(
                largestResultShift.whiff_delta
            ).toFixed(1)} percentage points. Run value describes what happened on the field; it does not by itself identify the cause.`;

    }

}


// ==================================================
// Load Metric Guide
// ==================================================

function loadMetricGuide() {

    if (
        document.querySelector(
            'script[data-metric-guide="true"]'
        )
    ) {
        return;
    }


    const script =
        document.createElement(
            "script"
        );


    script.src =
        "/static/metric_guide.js";


    script.dataset.metricGuide =
        "true";


    document.body.appendChild(
        script
    );

}


// ==================================================
// Initialize
// ==================================================

async function initializeExecutiveOverview() {

    try {
        if (window.pitcherResearchLab?.ready) {
            await window.pitcherResearchLab.ready;
        }
        if (!window.pitcherResearchLab?.pitcherId) return;

        const changeResponse =
            await fetch(
                window.pitcherResearchLab.apiUrl("changes")
            );

        if (!changeResponse.ok) {
            throw new Error("Overview change request failed.");
        }

        overviewChangeData =
            await changeResponse.json();

        const windowRange =
            window.pitcherResearchLab.researchWindow(
                overviewChangeData
            );

        if (!windowRange.start || !windowRange.end) {
            throw new Error("Not enough outings to define an overview comparison window.");
        }

        const researchResponse =
            await fetch(
                window.pitcherResearchLab.apiUrl("research", {
                    start: windowRange.start,
                    end: windowRange.end,
                })
            );

        if (!researchResponse.ok) {
            throw new Error("Overview research request failed.");
        }

        overviewResearchData =
            await researchResponse.json();


        renderExecutiveSummary();

        renderTransitionWindow();

        renderStuffSignal();

        renderArsenalSignal();

        renderPerformanceSignal();

    }

    catch (error) {

        console.error(
            "Executive Overview error:",
            error
        );

        const headline = document.getElementById("overview-primary-finding");
        const detail = document.getElementById("overview-primary-detail");
        if (headline) headline.textContent = "Overview unavailable for this sample";
        if (detail) detail.textContent = "The selected pitcher does not currently have enough usable data for the full overview comparison.";

        [
            ["overview-stuff-title", "Pitch-characteristic comparison unavailable"],
            ["overview-arsenal-title", "Arsenal comparison unavailable"],
            ["overview-performance-title", "Performance comparison unavailable"],
        ].forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });

    }

}


// ==================================================
// Start
// ==================================================

loadMetricGuide();

initializeExecutiveOverview();
