// ==================================================
// Pitcher Research Lab
// Command & Location Lab V2
// ==================================================


let locationData = null;


let locationTransitionStart = null;


let locationTransitionEnd = null;


// ==================================================
// Pitch names
// ==================================================

const locationPitchNames = {

    FF: "Four-Seam",

    SI: "Sinker",

    CH: "Changeup",

    FS: "Splitter",

    SL: "Slider",

    ST: "Sweeper"

};


// ==================================================
// Research periods
// ==================================================

const locationPeriods = [

    "early",

    "transition",

    "post"

];


// ==================================================
// Metrics
// ==================================================

const locationMetrics = [

    {

        key:
            "zone_pct",

        label:
            "Zone %",

        meaning:
            "Pitches inside the normalized strike zone",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "neutral"

    },


    {

        key:
            "heart_pct",

        label:
            "Heart %",

        meaning:
            "Pitches in the central portion of the strike zone",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "neutral"

    },


    {

        key:
            "edge_pct",

        label:
            "Edge %",

        meaning:
            "Pitches in the outer third of the strike zone",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "neutral"

    },


    {

        key:
            "chase_pct",

        label:
            "Chase %",

        meaning:
            "Out-of-zone pitches that hitters swung at",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "higher"

    },


    {

        key:
            "whiff_pct",

        label:
            "Whiff %",

        meaning:
            "Swinging misses divided by swings",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "higher"

    },


    {

        key:
            "zone_whiff_pct",

        label:
            "Zone Whiff %",

        meaning:
            "Whiffs on swings at pitches inside the zone",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "higher"

    },


    {

        key:
            "avg_ev",

        label:
            "Avg EV",

        meaning:
            "Average exit velocity on tracked batted balls",

        suffix:
            " mph",

        decimals:
            1,

        direction:
            "lower"

    },


    {

        key:
            "hard_hit_pct",

        label:
            "Hard-Hit %",

        meaning:
            "Batted balls hit at least 95 mph",

        suffix:
            "%",

        decimals:
            1,

        direction:
            "lower"

    },


    {

        key:
            "run_value_per_100",

        label:
            "Pitcher RV / 100",

        meaning:
            "Pitcher run value per 100 pitches; positive is good",

        suffix:
            "",

        decimals:
            2,

        direction:
            "higher"

    }

];


// ==================================================
// Formatting helpers
// ==================================================

function formatLocationValue(
    value,
    suffix = "",
    decimals = 1
) {

    if (
        value === null
        ||
        value === undefined
        ||
        Number.isNaN(
            Number(value)
        )
    ) {

        return "--";

    }


    return (

        Number(
            value
        )
        .toFixed(
            decimals
        )

        +

        suffix

    );

}


function signedDelta(
    value,
    suffix = "",
    decimals = 1
) {

    if (
        value === null
        ||
        value === undefined
        ||
        Number.isNaN(
            Number(value)
        )
    ) {

        return "--";

    }


    const number =
        Number(
            value
        );


    const sign =

        number > 0

            ? "+"

            : "";


    return (

        `${sign}${number.toFixed(
            decimals
        )}${suffix}`

    );

}


// ==================================================
// Get period summary
// ==================================================

function getSummary(
    period
) {

    return (

        locationData
            ?.periods
            ?.[period]
            ?.summary

        ??

        null

    );

}


// ==================================================
// Get period heatmap
// ==================================================

function getHeatmap(
    period
) {

    return (

        locationData
            ?.periods
            ?.[period]
            ?.heatmap

        ??

        []

    );

}


// ==================================================
// Color utilities
// ==================================================

function hexToRgb(
    hex
) {

    const cleaned =
        hex.replace(
            "#",
            ""
        );


    return {

        r:
            parseInt(
                cleaned.substring(
                    0,
                    2
                ),
                16
            ),

        g:
            parseInt(
                cleaned.substring(
                    2,
                    4
                ),
                16
            ),

        b:
            parseInt(
                cleaned.substring(
                    4,
                    6
                ),
                16
            )

    };

}


function mixColor(
    colorA,
    colorB,
    amount
) {

    const a =
        hexToRgb(
            colorA
        );


    const b =
        hexToRgb(
            colorB
        );


    const t =
        Math.max(
            0,
            Math.min(
                1,
                amount
            )
        );


    const red =
        Math.round(

            a.r

            +

            (
                b.r
                -
                a.r
            )

            *
            t

        );


    const green =
        Math.round(

            a.g

            +

            (
                b.g
                -
                a.g
            )

            *
            t

        );


    const blue =
        Math.round(

            a.b

            +

            (
                b.b
                -
                a.b
            )

            *
            t

        );


    return (

        `rgb(${red}, ${green}, ${blue})`

    );

}


// ==================================================
// Shared scale
//
// IMPORTANT:
// All three periods use the same scale.
// This makes the maps directly comparable.
// ==================================================

function getLocationScale(
    mode
) {

    const bins =

        locationPeriods.flatMap(
            period =>
                getHeatmap(
                    period
                )
        );


    // --------------------------------------------------
    // Pitch density
    // --------------------------------------------------

    if (
        mode ===
        "density"
    ) {

        return Math.max(

            ...bins.map(
                bin =>
                    Number(
                        bin.count
                        ??
                        0
                    )
            ),

            1

        );

    }


    // --------------------------------------------------
    // Run value
    // --------------------------------------------------

    if (
        mode ===
        "run_value"
    ) {

        const reliableBins =

            bins.filter(
                bin =>
                    Number(
                        bin.count
                    )
                    >=
                    5
            );


        if (
            reliableBins.length
            ===
            0
        ) {

            return 1;

        }


        return Math.max(

            ...reliableBins.map(
                bin =>
                    Math.abs(
                        Number(
                            bin.run_value_per_100
                            ??
                            0
                        )
                    )
            ),

            1

        );

    }


    // --------------------------------------------------
    // Hard-hit rate
    // --------------------------------------------------

    if (
        mode ===
        "hard_hit"
    ) {

        const reliableBins =

            bins.filter(
                bin =>
                    Number(
                        bin.batted_balls
                    )
                    >=
                    3
            );


        if (
            reliableBins.length
            ===
            0
        ) {

            return 100;

        }


        return Math.max(

            ...reliableBins.map(
                bin =>
                    Number(
                        bin.hard_hit_pct
                        ??
                        0
                    )
            ),

            1

        );

    }


    return 1;

}


// ==================================================
// Heatmap color system
// ==================================================

function getBinColor(
    bin,
    mode,
    scale
) {

    const empty =
        "#171d24";


    const neutral =
        "#29313b";


    // --------------------------------------------------
    // PITCH LOCATIONS
    //
    // Gray -> light blue -> dark blue
    //
    // Blue means frequency ONLY.
    // --------------------------------------------------

    if (
        mode ===
        "density"
    ) {

        const count =
            Number(
                bin.count
            );


        if (
            count === 0
        ) {

            return empty;

        }


        const amount =
            Math.min(
                count
                /
                scale,
                1
            );


        if (
            amount <= 0.5
        ) {

            return mixColor(

                "#343b45",

                "#74aaf5",

                amount
                *
                2

            );

        }


        return mixColor(

            "#74aaf5",

            "#174f9e",

            (
                amount
                -
                0.5
            )

            *
            2

        );

    }


    // --------------------------------------------------
    // PITCH RESULTS
    //
    // Red = worse for pitcher
    // Gray = neutral
    // Teal = better for pitcher
    //
    // Fewer than five pitches = muted.
    // --------------------------------------------------

    if (
        mode ===
        "run_value"
    ) {

        if (
            Number(
                bin.count
            )
            <
            5
        ) {

            return neutral;

        }


        const value =
            Number(
                bin.run_value_per_100
                ??
                0
            );


        const amount =
            Math.min(

                Math.abs(
                    value
                )
                /
                scale,

                1

            );


        if (
            value >= 0
        ) {

            return mixColor(

                neutral,

                "#2aa88f",

                amount

            );

        }


        return mixColor(

            neutral,

            "#d85d63",

            amount

        );

    }


    // --------------------------------------------------
    // CONTACT DAMAGE
    //
    // Gray -> orange -> red
    //
    // Fewer than three tracked balls = muted.
    // --------------------------------------------------

    if (
        mode ===
        "hard_hit"
    ) {

        if (
            Number(
                bin.batted_balls
            )
            <
            3
        ) {

            return neutral;

        }


        const amount =
            Math.min(

                Number(
                    bin.hard_hit_pct
                    ??
                    0
                )

                /

                scale,

                1

            );


        if (
            amount <= 0.60
        ) {

            return mixColor(

                neutral,

                "#d8893b",

                amount
                /
                0.60

            );

        }


        return mixColor(

            "#d8893b",

            "#d54c4c",

            (
                amount
                -
                0.60
            )
            /
            0.40

        );

    }


    return neutral;

}


// ==================================================
// Load location data
// ==================================================

async function loadLocationData() {

    const pitchElement =
        document.getElementById(
            "location-pitch"
        );


    const handElement =
        document.getElementById(
            "location-hand"
        );


    if (
        !pitchElement
        ||
        !handElement
    ) {

        return;

    }


    const pitch =
        pitchElement.value;


    const hand =
        handElement.value;


    const url =

        window.pitcherResearchLab.apiUrl("location", {
            pitch,
            hand,
            start: locationTransitionStart,
            end: locationTransitionEnd,
        });


    try {

        const response =
            await fetch(
                url
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "Location API request failed."
            );

        }


        locationData =
            await response.json();


        renderLocationLab();

    }

    catch (
        error
    ) {

        console.error(

            "Location API error:",

            error

        );

        const title = document.getElementById("location-finding-title");
        const copy = document.getElementById("location-finding-text");
        const sample = document.getElementById("location-sample-note");
        const body = document.getElementById("location-comparison-body");

        if (title) title.textContent = "Location comparison unavailable";
        if (copy) copy.textContent = "The selected pitch does not have enough usable location data for this research window.";
        if (sample) sample.textContent = "No qualifying location sample";
        if (body) body.innerHTML = '<tr><td colspan="6">No qualifying location data are available for this selection.</td></tr>';

    }

}


// ==================================================
// Render entire lab
// ==================================================

function renderLocationLab() {

    if (
        !locationData
    ) {

        return;

    }


    renderLocationFinding();

    renderLocationMaps();

    renderLocationLegend();

    renderLocationComparisonTable();

}


// ==================================================
// Summary finding
// ==================================================

function renderLocationFinding() {

    const early =
        getSummary(
            "early"
        );


    const post =
        getSummary(
            "post"
        );


    if (
        !early
        ||
        !post
    ) {

        return;

    }


    const pitchType =
        document.getElementById(
            "location-pitch"
        ).value;


    const pitchName =
        locationPitchNames[
            pitchType
        ]
        ??
        pitchType;


    const zoneDelta =

        Number(
            post.zone_pct
        )

        -

        Number(
            early.zone_pct
        );


    const whiffDelta =

        Number(
            post.whiff_pct
        )

        -

        Number(
            early.whiff_pct
        );


    const hardHitDelta =

        Number(
            post.hard_hit_pct
        )

        -

        Number(
            early.hard_hit_pct
        );


    const rvDelta =

        Number(
            post.run_value_per_100
        )

        -

        Number(
            early.run_value_per_100
        );


    // --------------------------------------------------
    // Delta cards
    // --------------------------------------------------

    document.getElementById(
        "location-delta-zone"
    ).textContent =

        signedDelta(
            zoneDelta,
            " pts"
        );


    document.getElementById(
        "location-delta-whiff"
    ).textContent =

        signedDelta(
            whiffDelta,
            " pts"
        );


    document.getElementById(
        "location-delta-hard-hit"
    ).textContent =

        signedDelta(
            hardHitDelta,
            " pts"
        );


    document.getElementById(
        "location-delta-rv"
    ).textContent =

        signedDelta(
            rvDelta,
            "",
            2
        );


    // --------------------------------------------------
    // Automatic finding headline
    // --------------------------------------------------

    let title =
        "Mixed later-period profile";


    if (
        whiffDelta > 0
        &&
        rvDelta < 0
    ) {

        title =
            "More swing-and-miss, less overall pitch value";

    }


    else if (
        whiffDelta < 0
        &&
        rvDelta < 0
    ) {

        title =
            "Swing-and-miss and pitch value both moved lower";

    }


    else if (
        whiffDelta > 0
        &&
        rvDelta > 0
    ) {

        title =
            "Swing-and-miss and pitch value both improved";

    }


    document.getElementById(
        "location-finding-title"
    ).textContent =

        `${pitchName}: ${title}`;


    // --------------------------------------------------
    // Automatic explanatory paragraph
    // --------------------------------------------------

    const zoneDirection =

        zoneDelta >= 0

            ? "increased"

            : "decreased";


    const whiffDirection =

        whiffDelta >= 0

            ? "increased"

            : "decreased";


    const hardHitDirection =

        hardHitDelta >= 0

            ? "increased"

            : "decreased";


    const rvDirection =

        rvDelta >= 0

            ? "improved"

            : "declined";


    const text =

        `${pitchName} zone rate ${zoneDirection} by ${Math.abs(
            zoneDelta
        ).toFixed(1)} percentage points from the early period to the later comparison period. `

        +

        `Whiff rate ${whiffDirection} by ${Math.abs(
            whiffDelta
        ).toFixed(1)} points, while hard-hit rate ${hardHitDirection} by ${Math.abs(
            hardHitDelta
        ).toFixed(1)} points. `

        +

        `Pitcher RV/100 ${rvDirection} by ${Math.abs(
            rvDelta
        ).toFixed(2)}. `

        +

        `This combination identifies what deserves further investigation, but it does not by itself establish a mechanical or strategic cause.`;


    document.getElementById(
        "location-finding-text"
    ).textContent =
        text;

}


// ==================================================
// Render three maps
// ==================================================

function renderLocationMaps() {

    const mode =
        document.getElementById(
            "location-mode"
        ).value;


    const scale =
        getLocationScale(
            mode
        );


    locationPeriods.forEach(
        period => {

            drawSingleLocationMap(

                period,

                mode,

                scale

            );

        }
    );

}


// ==================================================
// Draw one location map
// ==================================================

function drawSingleLocationMap(
    period,
    mode,
    scale
) {

    const svg =
        document.getElementById(
            `location-heatmap-${period}`
        );


    const periodData =
        locationData
            .periods[
                period
            ];


    if (
        !svg
        ||
        !periodData
    ) {

        return;

    }


    svg.innerHTML =
        "";


    const summary =
        periodData.summary;


    const meta =
        document.getElementById(
            `location-meta-${period}`
        );


    if (
        meta
    ) {

        meta.textContent =

            `${summary.pitches} pitches`;

    }


    const bins =
        periodData.heatmap;


    const chartX =
        35;


    const chartY =
        35;


    const chartSize =
        430;


    const gridSize =
        7;


    const cellSize =

        chartSize

        /

        gridSize;


    // --------------------------------------------------
    // Draw cells
    // --------------------------------------------------

    bins.forEach(
        bin => {

            const x =

                chartX

                +

                Number(
                    bin.x_bin
                )

                *

                cellSize;


            const visualZBin =

                gridSize

                -

                1

                -

                Number(
                    bin.z_bin
                );


            const y =

                chartY

                +

                visualZBin

                *

                cellSize;


            const rect =
                document.createElementNS(

                    "http://www.w3.org/2000/svg",

                    "rect"

                );


            rect.setAttribute(
                "x",
                x
            );


            rect.setAttribute(
                "y",
                y
            );


            rect.setAttribute(
                "width",
                cellSize
            );


            rect.setAttribute(
                "height",
                cellSize
            );


            rect.setAttribute(

                "fill",

                getBinColor(
                    bin,
                    mode,
                    scale
                )

            );


            rect.setAttribute(

                "class",

                "location-v2-bin"

            );


            // --------------------------------------------------
            // Tooltip numbers
            // --------------------------------------------------

            const pitchShare =

                Number(
                    summary.pitches
                ) > 0

                    ?

                    (
                        Number(
                            bin.count
                        )

                        /

                        Number(
                            summary.pitches
                        )

                        *

                        100
                    )

                    :

                    0;


            const whiffRate =

                Number(
                    bin.swings
                ) > 0

                    ?

                    (
                        Number(
                            bin.whiffs
                        )

                        /

                        Number(
                            bin.swings
                        )

                        *

                        100
                    )

                    :

                    null;


            const tooltip =
                document.createElementNS(

                    "http://www.w3.org/2000/svg",

                    "title"

                );


            tooltip.textContent =

                `Pitches: ${bin.count}
Pitch share: ${pitchShare.toFixed(1)}%
Swings: ${bin.swings}
Whiff%: ${whiffRate === null ? "--" : whiffRate.toFixed(1) + "%"}
Batted balls: ${bin.batted_balls}
Avg EV: ${bin.avg_ev ?? "--"}
Hard-Hit%: ${bin.hard_hit_pct ?? "--"}
Pitcher RV/100: ${bin.run_value_per_100 ?? "--"}`;


            rect.appendChild(
                tooltip
            );


            svg.appendChild(
                rect
            );

        }
    );


    // ==================================================
    // Strike Zone
    // ==================================================

    const fullRange =
        3.5;


    const zoneStart =

        (
            0.75

            /

            fullRange
        )

        *

        chartSize;


    const zoneSize =

        (
            2

            /

            fullRange
        )

        *

        chartSize;


    const strikeZone =
        document.createElementNS(

            "http://www.w3.org/2000/svg",

            "rect"

        );


    strikeZone.setAttribute(

        "x",

        chartX
        +
        zoneStart

    );


    strikeZone.setAttribute(

        "y",

        chartY
        +
        zoneStart

    );


    strikeZone.setAttribute(
        "width",
        zoneSize
    );


    strikeZone.setAttribute(
        "height",
        zoneSize
    );


    strikeZone.setAttribute(

        "class",

        "location-v2-zone"

    );


    svg.appendChild(
        strikeZone
    );


    // ==================================================
    // 3 x 3 strike-zone grid
    // ==================================================

    for (
        let i = 1;
        i < 3;
        i++
    ) {

        const offset =

            zoneSize

            *

            i

            /

            3;


        // Vertical subdivision

        const vertical =
            document.createElementNS(

                "http://www.w3.org/2000/svg",

                "line"

            );


        vertical.setAttribute(

            "x1",

            chartX
            +
            zoneStart
            +
            offset

        );


        vertical.setAttribute(

            "x2",

            chartX
            +
            zoneStart
            +
            offset

        );


        vertical.setAttribute(

            "y1",

            chartY
            +
            zoneStart

        );


        vertical.setAttribute(

            "y2",

            chartY
            +
            zoneStart
            +
            zoneSize

        );


        vertical.setAttribute(

            "class",

            "location-v2-zone-grid"

        );


        svg.appendChild(
            vertical
        );


        // Horizontal subdivision

        const horizontal =
            document.createElementNS(

                "http://www.w3.org/2000/svg",

                "line"

            );


        horizontal.setAttribute(

            "x1",

            chartX
            +
            zoneStart

        );


        horizontal.setAttribute(

            "x2",

            chartX
            +
            zoneStart
            +
            zoneSize

        );


        horizontal.setAttribute(

            "y1",

            chartY
            +
            zoneStart
            +
            offset

        );


        horizontal.setAttribute(

            "y2",

            chartY
            +
            zoneStart
            +
            offset

        );


        horizontal.setAttribute(

            "class",

            "location-v2-zone-grid"

        );


        svg.appendChild(
            horizontal
        );

    }

}


// ==================================================
// Legend
// ==================================================

function renderLocationLegend() {

    const mode =
        document.getElementById(
            "location-mode"
        ).value;


    const title =
        document.getElementById(
            "location-legend-title"
        );


    const left =
        document.getElementById(
            "location-legend-left"
        );


    const right =
        document.getElementById(
            "location-legend-right"
        );


    const bar =
        document.getElementById(
            "location-legend-bar"
        );


    const note =
        document.getElementById(
            "location-sample-note"
        );


    bar.className =
        "location-legend-bar";


    // --------------------------------------------------
    // Pitch frequency
    // --------------------------------------------------

    if (
        mode ===
        "density"
    ) {

        title.textContent =
            "Pitch Frequency";


        left.textContent =
            "Fewer pitches";


        right.textContent =
            "More pitches";


        bar.classList.add(
            "legend-density"
        );


        note.textContent =
            "Blue represents frequency only. Darker blue means more pitches were thrown in that area — not that the location performed better.";

    }


    // --------------------------------------------------
    // Pitch results
    // --------------------------------------------------

    if (
        mode ===
        "run_value"
    ) {

        title.textContent =
            "Pitch Results";


        left.textContent =
            "Worse for pitcher";


        right.textContent =
            "Better for pitcher";


        bar.classList.add(
            "legend-results"
        );


        note.textContent =
            "Red means worse pitcher run value. Teal means better pitcher run value. Cells with fewer than 5 pitches are muted gray.";

    }


    // --------------------------------------------------
    // Contact damage
    // --------------------------------------------------

    if (
        mode ===
        "hard_hit"
    ) {

        title.textContent =
            "Contact Damage";


        left.textContent =
            "Lower damage";


        right.textContent =
            "Higher damage";


        bar.classList.add(
            "legend-contact"
        );


        note.textContent =
            "Orange and red indicate higher hard-hit rates. Cells with fewer than 3 tracked batted balls are muted gray.";

    }

}


// ==================================================
// Comparison Table
// ==================================================

function renderLocationComparisonTable() {

    const body =
        document.getElementById(
            "location-comparison-body"
        );


    if (
        !body
    ) {

        return;

    }


    body.innerHTML =
        "";


    const early =
        getSummary(
            "early"
        );


    const transition =
        getSummary(
            "transition"
        );


    const post =
        getSummary(
            "post"
        );


    if (
        !early
        ||
        !transition
        ||
        !post
    ) {

        return;

    }


    locationMetrics.forEach(
        metric => {

            const earlyValue =
                early[
                    metric.key
                ];


            const transitionValue =
                transition[
                    metric.key
                ];


            const postValue =
                post[
                    metric.key
                ];


            let delta =
                null;


            if (
                earlyValue !== null
                &&
                earlyValue !== undefined
                &&
                postValue !== null
                &&
                postValue !== undefined
            ) {

                delta =

                    Number(
                        postValue
                    )

                    -

                    Number(
                        earlyValue
                    );

            }


            // --------------------------------------------------
            // Determine whether change is favorable
            // --------------------------------------------------

            let deltaClass =
                "location-delta-neutral";


            if (
                delta !== null
                &&
                metric.direction
                !==
                "neutral"
            ) {

                const improved =

                    metric.direction
                    ===
                    "higher"

                        ?

                        delta > 0

                        :

                        delta < 0;


                if (
                    delta === 0
                ) {

                    deltaClass =
                        "location-delta-neutral";

                }

                else {

                    deltaClass =

                        improved

                            ?

                            "location-delta-good"

                            :

                            "location-delta-bad";

                }

            }


            // --------------------------------------------------
            // Delta suffix
            // --------------------------------------------------

            let deltaSuffix =
                metric.suffix;


            if (
                metric.suffix
                ===
                "%"
            ) {

                deltaSuffix =
                    " pts";

            }


            if (
                metric.key
                ===
                "run_value_per_100"
            ) {

                deltaSuffix =
                    "";

            }


            // --------------------------------------------------
            // Row
            // --------------------------------------------------

            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML = `

                <td class="location-metric-name">

                    ${metric.label}

                </td>


                <td class="location-metric-meaning">

                    ${metric.meaning}

                </td>


                <td>

                    ${formatLocationValue(
                        earlyValue,
                        metric.suffix,
                        metric.decimals
                    )}

                </td>


                <td>

                    ${formatLocationValue(
                        transitionValue,
                        metric.suffix,
                        metric.decimals
                    )}

                </td>


                <td>

                    ${formatLocationValue(
                        postValue,
                        metric.suffix,
                        metric.decimals
                    )}

                </td>


                <td class="${deltaClass}">

                    ${signedDelta(
                        delta,
                        deltaSuffix,
                        metric.decimals
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
// Controls
// ==================================================

const locationPitchControl =
    document.getElementById(
        "location-pitch"
    );


const locationHandControl =
    document.getElementById(
        "location-hand"
    );


const locationModeControl =
    document.getElementById(
        "location-mode"
    );


if (
    locationPitchControl
) {

    locationPitchControl.addEventListener(

        "change",

        loadLocationData

    );

}


if (
    locationHandControl
) {

    locationHandControl.addEventListener(

        "change",

        loadLocationData

    );

}


if (
    locationModeControl
) {

    locationModeControl.addEventListener(

        "change",

        function () {

            renderLocationMaps();

            renderLocationLegend();

        }

    );

}


// ==================================================
// Initialize Location Lab
// ==================================================

async function initializeLocationLab() {
    if (window.pitcherResearchLab?.ready) {
        await window.pitcherResearchLab.ready;
    }
    if (!window.pitcherResearchLab?.pitcherId) return;

    try {

        // --------------------------------------------------
        // Use the shared research window
        // as the rest of the application.
        // --------------------------------------------------

        const response =
            await fetch(
                window.pitcherResearchLab.apiUrl("changes")
            );


        if (
            response.ok
        ) {

            const changes =
                await response.json();


            const windowRange =
                window.pitcherResearchLab.researchWindow(
                    changes
                );

            locationTransitionStart =
                windowRange.start;

            locationTransitionEnd =
                windowRange.end;

        }


        if (!locationTransitionStart || !locationTransitionEnd) {
            const windowRange =
                window.pitcherResearchLab.researchWindow([]);

            locationTransitionStart = windowRange.start;
            locationTransitionEnd = windowRange.end;
        }

        if (!locationTransitionStart || !locationTransitionEnd) {
            throw new Error("Not enough outings to define a location comparison window.");
        }


        await loadLocationData();

    }

    catch (
        error
    ) {

        console.error(

            "Location Lab initialization error:",

            error

        );

        const title = document.getElementById("location-finding-title");
        const copy = document.getElementById("location-finding-text");
        if (title) title.textContent = "Location comparison unavailable";
        if (copy) copy.textContent = "More usable regular-season pitch-location data are needed for this view.";

    }

}


// ==================================================
// Start
// ==================================================

initializeLocationLab();
