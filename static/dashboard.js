// ==================================================
// Pitcher Research Lab
// Dashboard JavaScript
// Pitch profile dashboard
// ==================================================

(() => {
    "use strict";

    const pitchNames = {
        FF: "Four-Seam Fastball",
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

    let changeData = [];
    let timelineData = [];

    const pitchSelect =
        document.getElementById(
            "pitch-select"
        );

    const pitchName =
        document.getElementById(
            "pitch-name"
        );


    // ==================================================
    // Helpers
    // ==================================================

    function numberOrNull(
        value
    ) {
        if (
            value === null
            ||
            value === undefined
            ||
            value === ""
        ) {
            return null;
        }

        const n =
            Number(
                value
            );

        return Number.isNaN(
            n
        )
            ?
            null
            :
            n;
    }


    function targetSeason(data) {
        const seasons = data
            .map(row => Number(row.season))
            .filter(Number.isFinite);
        return seasons.length ? Math.max(...seasons) : null;
    }

    function baselineSeasons(data) {
        const target = targetSeason(data);
        if (target === null) return [];
        const seasons = [...new Set(
            data.map(row => Number(row.season)).filter(Number.isFinite)
        )]
            .filter(season => season < target)
            .sort((a, b) => a - b);
        return seasons.slice(-2);
    }

    function calculateBaseline(
        data,
        metricKey
    ) {
        const rows =
            data.filter(
                row =>

                    baselineSeasons(data).includes(
                        Number(row.season)
                    )

                    &&

                    numberOrNull(
                        row[
                            metricKey
                        ]
                    )
                    !==
                    null

                    &&

                    Number(
                        row.pitches
                        ??
                        0
                    )
                    >
                    0
            );

        const totalPitches =
            rows.reduce(

                (
                    sum,
                    row
                ) =>

                    sum
                    +
                    Number(
                        row.pitches
                    ),

                0

            );

        if (
            !totalPitches
        ) {
            return null;
        }

        return (

            rows.reduce(

                (
                    sum,
                    row
                ) =>

                    sum
                    +
                    Number(
                        row[
                            metricKey
                        ]
                    )
                    *
                    Number(
                        row.pitches
                    ),

                0

            )

            /

            totalPitches

        );
    }


    function formatChange(
        value,
        decimals,
        unit
    ) {
        const n =
            numberOrNull(
                value
            );

        if (
            n === null
        ) {
            return "--";
        }

        const threshold =
            Math.pow(
                10,
                -decimals
            )
            /
            2;

        if (
            Math.abs(
                n
            )
            <
            threshold
        ) {
            return "No meaningful change";
        }

        return (
            `${n > 0 ? "↑" : "↓"} ` +
            `${Math.abs(n).toFixed(decimals)} ${unit}`
        );
    }


    function describeZScore(
        value
    ) {
        const z =
            numberOrNull(
                value
            );

        if (
            z === null
        ) {
            return null;
        }

        const magnitude =
            Math.abs(
                z
            );

        const direction =
            z < 0
                ?
                "below"
                :
                "above";

        let description =
            "within normal historical variation";

        if (
            magnitude >= 3
        ) {
            description =
                "very unusual departure";
        }

        else if (
            magnitude >= 2
        ) {
            description =
                "unusually large departure";
        }

        else if (
            magnitude >= 1
        ) {
            description =
                "noticeable departure";
        }

        return (
            `${magnitude.toFixed(2)} SD ` +
            `${direction} baseline — ${description}`
        );
    }


    function setMetricValue(
        id,
        value
    ) {
        const element =
            document.getElementById(
                id
            );

        if (
            element
        ) {
            element.textContent =
                value;
        }
    }


    function updateMetricDetail(
        valueElementId,
        text
    ) {
        const valueElement =
            document.getElementById(
                valueElementId
            );

        if (
            !valueElement
        ) {
            return;
        }

        const card =
            valueElement.closest(
                ".metric"
            );

        if (
            !card
        ) {
            return;
        }

        let detail =
            card.querySelector(
                ".metric-change"
            );

        if (
            !detail
        ) {
            detail =
                document.createElement(
                    "div"
                );

            detail.className =
                "metric-change";

            card.appendChild(
                detail
            );
        }

        detail.textContent =
            text;
    }


    function findChangeResult(
        pitchType,
        metricName
    ) {
        return (

            changeData.find(
                row =>

                    row.pitch_type
                    ===
                    pitchType

                    &&

                    String(
                        row.metric
                        ??
                        ""
                    ).includes(
                        metricName
                    )
            )

            ??

            null

        );
    }


    function formatDate(
        dateString
    ) {
        if (
            !dateString
        ) {
            return "--";
        }

        const date =
            new Date(
                `${dateString}T00:00:00`
            );

        return date.toLocaleDateString(
            "en-US",
            {
                month:
                    "short",

                day:
                    "numeric"
            }
        );
    }


    // ==================================================
    // Pitch Profile / Arsenal
    // ==================================================

    async function loadPitchData(
        pitchType
    ) {
        try {

            const response =
                await fetch(
                    `${window.pitcherResearchLab.apiUrl(`pitch/${pitchType}`)}`
                );

            if (
                !response.ok
            ) {
                throw new Error(
                    "Pitch API request failed."
                );
            }

            const payload =
                await response.json();

            const data =
                Array.isArray(
                    payload
                )
                    ?
                    payload
                    :
                    [];

            if (
                pitchName
            ) {
                pitchName.textContent =
                    pitchNames[
                        pitchType
                    ]
                    ??
                    pitchType;
            }

            updateCurrentMetrics(
                data,
                pitchType
            );

            updateSeasonTable(
                data
            );

            drawVelocityChart(
                data
            );

        }

        catch (
            error
        ) {

            console.error(
                "Error loading pitch data:",
                error
            );

            [
                "metric-velocity",
                "metric-spin",
                "metric-extension",
                "metric-vmov",
                "metric-hmov"
            ].forEach(
                id => {

                    setMetricValue(
                        id,
                        "--"
                    );

                    updateMetricDetail(
                        id,
                        "Pitch profile could not be loaded. Check the API and browser console."
                    );

                }
            );

        }
    }


    function updateCurrentMetrics(
        data,
        pitchType
    ) {
        const currentSeason = targetSeason(data);
        const current =
            data.find(
                row => Number(row.season) === currentSeason
            );

        if (
            !current
        ) {

            [
                "metric-velocity",
                "metric-spin",
                "metric-extension",
                "metric-vmov",
                "metric-hmov"
            ].forEach(
                id => {

                    setMetricValue(
                        id,
                        "--"
                    );

                    updateMetricDetail(
                        id,
                        currentSeason ? `No ${currentSeason} data available for this pitch.` : "No current-season data available for this pitch."
                    );

                }
            );

            return;
        }


        function detail(
            metricKey,
            changeMetricName,
            decimals,
            unit
        ) {
            const baseline =
                calculateBaseline(
                    data,
                    metricKey
                );

            const currentValue =
                numberOrNull(
                    current[
                        metricKey
                    ]
                );

            if (
                baseline === null
                ||
                currentValue === null
            ) {
                return "Historical baseline unavailable.";
            }

            const difference =
                currentValue
                -
                baseline;

            let text =

                `${formatChange(
                    difference,
                    decimals,
                    unit
                )} vs weighted ${baselineSeasons(data).join("–") || "historical"} baseline`;

            if (
                changeMetricName
            ) {

                const detected =
                    findChangeResult(
                        pitchType,
                        changeMetricName
                    );

                const zText =
                    detected

                        ?

                        describeZScore(
                            detected.z_score
                        )

                        :

                        null;

                if (
                    zText
                ) {
                    text +=
                        ` • ${zText}`;
                }
            }

            return text;
        }


        const velocity =
            numberOrNull(
                current.avg_velocity
            );

        const spin =
            numberOrNull(
                current.avg_spin_rate
            );

        const extension =
            numberOrNull(
                current.avg_extension
            );

        const vmov =
            numberOrNull(
                current.avg_vertical_movement
            );

        const hmov =
            numberOrNull(
                current.avg_horizontal_movement
            );


        setMetricValue(
            "metric-velocity",

            velocity === null
                ?
                "--"
                :
                `${velocity.toFixed(2)} mph`
        );


        setMetricValue(
            "metric-spin",

            spin === null
                ?
                "--"
                :
                `${Math.round(spin)} rpm`
        );


        setMetricValue(
            "metric-extension",

            extension === null
                ?
                "--"
                :
                `${extension.toFixed(2)} ft`
        );


        setMetricValue(
            "metric-vmov",

            vmov === null
                ?
                "--"
                :
                `${vmov.toFixed(2)} in`
        );


        setMetricValue(
            "metric-hmov",

            hmov === null
                ?
                "--"
                :
                `${hmov.toFixed(2)} in`
        );


        updateMetricDetail(

            "metric-velocity",

            detail(
                "avg_velocity",
                "Velocity",
                2,
                "mph"
            )

        );


        updateMetricDetail(

            "metric-spin",

            detail(
                "avg_spin_rate",
                "Spin Rate",
                0,
                "rpm"
            )

        );


        updateMetricDetail(

            "metric-extension",

            detail(
                "avg_extension",
                "Extension",
                2,
                "ft"
            )

        );


        updateMetricDetail(

            "metric-vmov",

            detail(
                "avg_vertical_movement",
                "Vertical Movement",
                2,
                "in"
            )

        );


        updateMetricDetail(

            "metric-hmov",

            detail(
                "avg_horizontal_movement",
                "Horizontal Movement",
                2,
                "in"
            )

        );
    }


    // ==================================================
    // Season Comparison Table
    // ==================================================

    function updateSeasonTable(
        data
    ) {
        const body =
            document.getElementById(
                "season-table-body"
            );

        if (
            !body
        ) {
            return;
        }

        body.innerHTML =
            "";

        if (
            !data.length
        ) {
            const row =
                document.createElement(
                    "tr"
                );

            row.innerHTML =
                `<td colspan="9">No season data available for this pitch.</td>`;

            body.appendChild(
                row
            );

            return;
        }

        data.forEach(
            row => {

                const tr =
                    document.createElement(
                        "tr"
                    );

                tr.innerHTML = `

                    <td>
                        ${row.season}
                    </td>

                    <td>
                        ${row.pitches ?? "--"}
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_velocity
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_velocity
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_spin_rate
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Math.round(
                                    Number(
                                        row.avg_spin_rate
                                    )
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_extension
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_extension
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_horizontal_movement
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_horizontal_movement
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_vertical_movement
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_vertical_movement
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_release_x
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_release_x
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                    <td>
                        ${
                            numberOrNull(
                                row.avg_release_z
                            )
                            ===
                            null

                                ?

                                "--"

                                :

                                Number(
                                    row.avg_release_z
                                ).toFixed(
                                    2
                                )
                        }
                    </td>

                `;

                body.appendChild(
                    tr
                );

            }
        );
    }


    // ==================================================
    // Velocity Evolution Chart
    // ==================================================

    function drawVelocityChart(
        data
    ) {
        const svg =
            document.getElementById(
                "velocity-chart"
            );

        if (
            !svg
        ) {
            return;
        }

        svg.innerHTML =
            "";

        const valid =
            data.filter(
                row =>
                    numberOrNull(
                        row.avg_velocity
                    )
                    !==
                    null
            );

        if (
            !valid.length
        ) {

            const message =
                document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "text"
                );

            message.setAttribute(
                "x",
                "380"
            );

            message.setAttribute(
                "y",
                "140"
            );

            message.setAttribute(
                "text-anchor",
                "middle"
            );

            message.setAttribute(
                "class",
                "chart-axis-text"
            );

            message.textContent =
                "No velocity data available.";

            svg.appendChild(
                message
            );

            return;
        }

        const width =
            760;

        const height =
            280;

        const left =
            55;

        const right =
            35;

        const top =
            30;

        const bottom =
            45;

        const values =
            valid.map(
                row =>
                    Number(
                        row.avg_velocity
                    )
            );

        let minimum =
            Math.floor(
                Math.min(
                    ...values
                )
                -
                1
            );

        let maximum =
            Math.ceil(
                Math.max(
                    ...values
                )
                +
                1
            );

        if (
            minimum === maximum
        ) {
            maximum =
                minimum
                +
                1;
        }

        function xPosition(
            index
        ) {
            if (
                valid.length === 1
            ) {
                return (
                    width
                    /
                    2
                );
            }

            return (
                left
                +
                index
                *
                (
                    (
                        width
                        -
                        left
                        -
                        right
                    )
                    /
                    (
                        valid.length
                        -
                        1
                    )
                )
            );
        }

        function yPosition(
            value
        ) {
            return (
                top
                +
                (
                    (
                        maximum
                        -
                        value
                    )
                    /
                    (
                        maximum
                        -
                        minimum
                    )
                )
                *
                (
                    height
                    -
                    top
                    -
                    bottom
                )
            );
        }

        for (
            let value = minimum;
            value <= maximum;
            value++
        ) {

            const y =
                yPosition(
                    value
                );

            const line =
                document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "line"
                );

            line.setAttribute(
                "x1",
                left
            );

            line.setAttribute(
                "x2",
                width
                -
                right
            );

            line.setAttribute(
                "y1",
                y
            );

            line.setAttribute(
                "y2",
                y
            );

            line.setAttribute(
                "class",
                "chart-grid-line"
            );

            svg.appendChild(
                line
            );


            const label =
                document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "text"
                );

            label.setAttribute(
                "x",
                left
                -
                10
            );

            label.setAttribute(
                "y",
                y
                +
                4
            );

            label.setAttribute(
                "text-anchor",
                "end"
            );

            label.setAttribute(
                "class",
                "chart-axis-text"
            );

            label.textContent =
                value;

            svg.appendChild(
                label
            );
        }


        const points =
            valid.map(
                (
                    row,
                    index
                ) => (
                    {
                        x:
                            xPosition(
                                index
                            ),

                        y:
                            yPosition(
                                Number(
                                    row.avg_velocity
                                )
                            ),

                        season:
                            row.season,

                        velocity:
                            Number(
                                row.avg_velocity
                            )
                    }
                )
            );


        const path =
            document.createElementNS(
                "http://www.w3.org/2000/svg",
                "path"
            );

        path.setAttribute(

            "d",

            points
                .map(
                    (
                        point,
                        index
                    ) =>

                        `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`

                )
                .join(
                    " "
                )

        );

        path.setAttribute(
            "class",
            "chart-line"
        );

        svg.appendChild(
            path
        );


        points.forEach(
            point => {

                const circle =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "circle"
                    );

                circle.setAttribute(
                    "cx",
                    point.x
                );

                circle.setAttribute(
                    "cy",
                    point.y
                );

                circle.setAttribute(
                    "r",
                    4
                );

                circle.setAttribute(
                    "class",
                    "chart-point"
                );

                svg.appendChild(
                    circle
                );


                const valueLabel =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "text"
                    );

                valueLabel.setAttribute(
                    "x",
                    point.x
                );

                valueLabel.setAttribute(
                    "y",
                    point.y
                    -
                    14
                );

                valueLabel.setAttribute(
                    "text-anchor",
                    "middle"
                );

                valueLabel.setAttribute(
                    "class",
                    "chart-value"
                );

                valueLabel.textContent =
                    point.velocity.toFixed(
                        2
                    );

                svg.appendChild(
                    valueLabel
                );


                const seasonLabel =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "text"
                    );

                seasonLabel.setAttribute(
                    "x",
                    point.x
                );

                seasonLabel.setAttribute(
                    "y",
                    height
                    -
                    12
                );

                seasonLabel.setAttribute(
                    "text-anchor",
                    "middle"
                );

                seasonLabel.setAttribute(
                    "class",
                    "chart-axis-text"
                );

                seasonLabel.textContent =
                    point.season;

                svg.appendChild(
                    seasonLabel
                );

            }
        );
    }


    // ==================================================
    // Change Detection Summary
    // ==================================================

    function updateDeviationList(
        data
    ) {
        const container =
            document.getElementById(
                "deviation-list"
            );

        if (
            !container
        ) {
            return;
        }

        container.innerHTML =
            "";

        const major =
            data

                .filter(
                    row =>
                        numberOrNull(
                            row.z_score
                        )
                        !==
                        null

                        &&

                        Math.abs(
                            Number(
                                row.z_score
                            )
                        )
                        >=
                        2
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

        if (
            !major.length
        ) {

            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "deviation-row";

            item.innerHTML = `

                <div class="deviation-rank">
                    --
                </div>

                <div class="deviation-info">

                    <div class="deviation-name">
                        No ≥2 SD departures detected
                    </div>

                    <div class="deviation-detail">
                        No current metric crossed the dashboard's large-deviation screening threshold.
                    </div>

                </div>

                <div class="deviation-score">
                    Screening result
                </div>

            `;

            container.appendChild(
                item
            );

            return;
        }


        major.forEach(
            (
                row,
                index
            ) => {

                const dateText =
                    row.first_sustained_change

                        ?

                        ` • sustained flag ${formatDate(
                            row.first_sustained_change
                        )}`

                        :

                        "";

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "deviation-row";

                item.innerHTML = `

                    <div class="deviation-rank">

                        ${String(
                            index
                            +
                            1
                        ).padStart(
                            2,
                            "0"
                        )}

                    </div>


                    <div class="deviation-info">

                        <div class="deviation-name">
                            ${row.metric}
                        </div>

                        <div class="deviation-detail">

                            ${row.baseline_mean}

                            →

                            ${row.current_mean}

                            ${row.unit ?? ""}

                            ${dateText}

                        </div>

                    </div>


                    <div class="deviation-score">

                        ${describeZScore(
                            row.z_score
                        )}

                    </div>

                `;

                container.appendChild(
                    item
                );

            }
        );
    }


    function updateChangeTimeline(
        data
    ) {
        const container =
            document.getElementById(
                "change-timeline"
            );

        if (
            !container
        ) {
            return;
        }

        container.innerHTML =
            "";

        const events =
            data

                .filter(
                    row =>
                        row.first_sustained_change
                )

                .sort(
                    (
                        a,
                        b
                    ) =>

                        new Date(
                            a.first_sustained_change
                        )

                        -

                        new Date(
                            b.first_sustained_change
                        )
                );

        if (
            !events.length
        ) {

            container.innerHTML = `

                <div class="timeline-event">

                    <div class="timeline-date">
                        --
                    </div>

                    <div class="timeline-marker">
                    </div>

                    <div class="timeline-content">

                        <div class="timeline-metric">
                            No sustained flags
                        </div>

                        <div class="timeline-direction">
                            No metric currently meets the sustained-change rule.
                        </div>

                    </div>

                </div>

            `;

            return;
        }

        events.forEach(
            row => {

                const event =
                    document.createElement(
                        "div"
                    );

                event.className =
                    "timeline-event";

                const arrow =
                    row.direction
                    ===
                    "Above baseline"

                        ?

                        "↑"

                        :

                        "↓";

                event.innerHTML = `

                    <div class="timeline-date">

                        ${formatDate(
                            row.first_sustained_change
                        )}

                    </div>

                    <div class="timeline-marker">
                    </div>

                    <div class="timeline-content">

                        <div class="timeline-metric">
                            ${row.metric}
                        </div>

                        <div class="timeline-direction">

                            ${arrow}

                            ${row.direction}

                        </div>

                    </div>

                `;

                container.appendChild(
                    event
                );

            }
        );
    }


    // ==================================================
    // Investigation Timeline
    // ==================================================

    const investigationMetricSettings = {

        avg_velocity: {
            label:
                "Velocity",
            unit:
                "mph",
            decimals:
                1
        },

        avg_spin: {
            label:
                "Spin Rate",
            unit:
                "rpm",
            decimals:
                0
        },

        avg_extension: {
            label:
                "Extension",
            unit:
                "ft",
            decimals:
                2
        },

        avg_vertical_movement: {
            label:
                "Vertical Movement",
            unit:
                "in",
            decimals:
                1
        },

        avg_horizontal_movement: {
            label:
                "Horizontal Movement",
            unit:
                "in",
            decimals:
                1
        },

        usage_pct: {
            label:
                "Usage",
            unit:
                "%",
            decimals:
                1
        },

        whiff_pct: {
            label:
                "Whiff Rate",
            unit:
                "%",
            decimals:
                1
        },

        run_value_per_100: {
            label:
                "Pitch Value (RV/100)",
            unit:
                "runs / 100",
            decimals:
                2
        }

    };


    function findInvestigationChange(
        pitchType,
        metricKey
    ) {
        return (

            changeData.find(
                row => {

                    if (
                        row.pitch_type
                        !==
                        pitchType

                        ||

                        !row.first_sustained_change
                    ) {
                        return false;
                    }

                    const metric =
                        String(
                            row.metric
                            ??
                            ""
                        );

                    if (
                        metricKey
                        ===
                        "avg_velocity"
                    ) {
                        return metric.includes(
                            "Velocity"
                        );
                    }

                    if (
                        metricKey
                        ===
                        "avg_spin"
                    ) {
                        return metric.includes(
                            "Spin Rate"
                        );
                    }

                    if (
                        metricKey
                        ===
                        "avg_extension"
                    ) {
                        return metric.includes(
                            "Extension"
                        );
                    }

                    if (
                        metricKey
                        ===
                        "avg_vertical_movement"
                    ) {
                        return metric.includes(
                            "Vertical Movement"
                        );
                    }

                    if (
                        metricKey
                        ===
                        "avg_horizontal_movement"
                    ) {
                        return metric.includes(
                            "Horizontal Movement"
                        );
                    }

                    return false;

                }
            )

            ??

            null

        );
    }


    function addSvgText(
        svg,
        x,
        y,
        className,
        text,
        anchor = null
    ) {
        const element =
            document.createElementNS(
                "http://www.w3.org/2000/svg",
                "text"
            );

        element.setAttribute(
            "x",
            x
        );

        element.setAttribute(
            "y",
            y
        );

        element.setAttribute(
            "class",
            className
        );

        if (
            anchor
        ) {
            element.setAttribute(
                "text-anchor",
                anchor
            );
        }

        element.textContent =
            text;

        svg.appendChild(
            element
        );

        return element;
    }


    function drawInvestigationChart() {
        const pitchSelector =
            document.getElementById(
                "investigation-pitch"
            );

        const metricSelector =
            document.getElementById(
                "investigation-metric"
            );

        const svg =
            document.getElementById(
                "investigation-chart"
            );

        if (
            !pitchSelector
            ||
            !metricSelector
            ||
            !svg
        ) {
            return;
        }


        const pitchType =
            pitchSelector.value;

        const metricKey =
            metricSelector.value;

        const metric =
            investigationMetricSettings[
                metricKey
            ];

        if (
            !metric
        ) {
            return;
        }

        svg.innerHTML =
            "";


        let rows =
            timelineData.filter(
                row =>
                    row.pitch_type
                    ===
                    pitchType
            );


        const note =
            document.getElementById(
                "investigation-sample-note"
            );


        if (
            metricKey
            ===
            "whiff_pct"
        ) {

            rows =
                rows.filter(
                    row =>
                        Number(
                            row.swings
                            ??
                            0
                        )
                        >=
                        3
                );

            if (
                note
            ) {
                note.textContent =
                    "Outings with 3+ swings";
            }

        }

        else if (
            metricKey
            ===
            "usage_pct"
        ) {

            if (
                note
            ) {
                note.textContent =
                    "All outings";
            }

        }

        else {

            rows =
                rows.filter(
                    row =>
                        Number(
                            row.pitch_count
                            ??
                            0
                        )
                        >=
                        5
                );

            if (
                note
            ) {
                note.textContent =
                    "Outings with 5+ pitches";
            }

        }


        rows =
            rows

                .filter(
                    row =>
                        numberOrNull(
                            row[
                                metricKey
                            ]
                        )
                        !==
                        null
                )

                .sort(
                    (
                        a,
                        b
                    ) =>
                        new Date(
                            a.game_date
                        )
                        -
                        new Date(
                            b.game_date
                        )
                );


        const title =
            document.getElementById(
                "investigation-title"
            );

        if (
            title
        ) {

            title.textContent =
                `${pitchNames[pitchType] ?? pitchType} — ${metric.label}`;

        }


        if (
            !rows.length
        ) {

            addSvgText(
                svg,
                450,
                160,
                "investigation-empty",
                "This selection does not meet the current sample-size requirement.",
                "middle"
            );

            return;
        }


        const width =
            900;

        const height =
            320;

        const margin = {
            left:
                58,
            right:
                28,
            top:
                28,
            bottom:
                45
        };


        const values =
            rows.map(
                row =>
                    Number(
                        row[
                            metricKey
                        ]
                    )
            );


        let minValue =
            Math.min(
                ...values
            );

        let maxValue =
            Math.max(
                ...values
            );

        let range =
            maxValue
            -
            minValue;


        if (
            range === 0
        ) {
            range =
                1;
        }


        const pad =
            range
            *
            0.12;


        minValue -=
            pad;

        maxValue +=
            pad;


        const dates =
            rows.map(
                row =>
                    new Date(
                        `${row.game_date}T00:00:00`
                    )
            );


        const minDate =
            Math.min(
                ...dates.map(
                    date =>
                        date.getTime()
                )
            );


        const maxDate =
            Math.max(
                ...dates.map(
                    date =>
                        date.getTime()
                )
            );


        function xPosition(
            time
        ) {
            if (
                minDate
                ===
                maxDate
            ) {
                return (
                    width
                    /
                    2
                );
            }

            return (
                margin.left

                +

                (
                    (
                        time
                        -
                        minDate
                    )
                    /
                    (
                        maxDate
                        -
                        minDate
                    )
                )

                *

                (
                    width
                    -
                    margin.left
                    -
                    margin.right
                )
            );
        }


        function yPosition(
            value
        ) {
            return (
                margin.top

                +

                (
                    (
                        maxValue
                        -
                        value
                    )
                    /
                    (
                        maxValue
                        -
                        minValue
                    )
                )

                *

                (
                    height
                    -
                    margin.top
                    -
                    margin.bottom
                )
            );
        }


        // ----------------------------------------------
        // Y grid and labels
        // ----------------------------------------------

        const yTicks =
            5;


        for (
            let i = 0;
            i <= yTicks;
            i++
        ) {

            const value =
                minValue

                +

                (
                    (
                        maxValue
                        -
                        minValue
                    )
                    *
                    i
                    /
                    yTicks
                );


            const y =
                yPosition(
                    value
                );


            const line =
                document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "line"
                );


            line.setAttribute(
                "x1",
                margin.left
            );


            line.setAttribute(
                "x2",
                width
                -
                margin.right
            );


            line.setAttribute(
                "y1",
                y
            );


            line.setAttribute(
                "y2",
                y
            );


            line.setAttribute(
                "class",
                "investigation-grid-line"
            );


            svg.appendChild(
                line
            );


            addSvgText(

                svg,

                margin.left
                -
                10,

                y
                +
                4,

                "investigation-axis-text",

                value.toFixed(
                    metric.decimals
                ),

                "end"

            );

        }


        // ----------------------------------------------
        // Date labels
        // ----------------------------------------------

        const xTicks =
            5;


        for (
            let i = 0;
            i <= xTicks;
            i++
        ) {

            const time =
                minDate

                +

                (
                    (
                        maxDate
                        -
                        minDate
                    )
                    *
                    i
                    /
                    xTicks
                );


            const date =
                new Date(
                    time
                );


            addSvgText(

                svg,

                xPosition(
                    time
                ),

                height
                -
                12,

                "investigation-axis-text",

                date.toLocaleDateString(
                    "en-US",
                    {
                        month:
                            "short",

                        day:
                            "numeric"
                    }
                ),

                "middle"

            );

        }


        const points =
            rows.map(
                row => {

                    const date =
                        new Date(
                            `${row.game_date}T00:00:00`
                        );

                    return {
                        row:

                            row,

                        x:

                            xPosition(
                                date.getTime()
                            ),

                        y:

                            yPosition(
                                Number(
                                    row[
                                        metricKey
                                    ]
                                )
                            )
                    };

                }
            );


        const path =
            document.createElementNS(
                "http://www.w3.org/2000/svg",
                "path"
            );


        path.setAttribute(

            "d",

            points

                .map(
                    (
                        point,
                        index
                    ) =>

                        `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`

                )

                .join(
                    " "
                )

        );


        path.setAttribute(
            "class",
            "investigation-line"
        );


        svg.appendChild(
            path
        );


        points.forEach(
            point => {

                const circle =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "circle"
                    );


                circle.setAttribute(
                    "cx",
                    point.x
                );


                circle.setAttribute(
                    "cy",
                    point.y
                );


                circle.setAttribute(
                    "r",
                    4
                );


                circle.setAttribute(
                    "class",
                    "investigation-point"
                );


                const tooltip =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "title"
                    );


                const outingDate =
                    new Date(
                        `${point.row.game_date}T00:00:00`
                    )
                    .toLocaleDateString(
                        "en-US",
                        {
                            month:
                                "short",

                            day:
                                "numeric",

                            year:
                                "numeric"
                        }
                    );


                tooltip.textContent =

                    `${outingDate}\n`

                    +

                    `${metric.label}: ${Number(
                        point.row[
                            metricKey
                        ]
                    ).toFixed(
                        metric.decimals
                    )} ${metric.unit}\n`

                    +

                    `Pitches: ${point.row.pitch_count ?? "--"}\n`

                    +

                    `Swings: ${point.row.swings ?? "--"}`;


                circle.appendChild(
                    tooltip
                );


                svg.appendChild(
                    circle
                );

            }
        );


        const detected =
            findInvestigationChange(
                pitchType,
                metricKey
            );


        if (
            detected
        ) {

            const changeDate =
                new Date(
                    `${detected.first_sustained_change}T00:00:00`
                );


            const changeTime =
                changeDate.getTime();


            if (
                changeTime >= minDate
                &&
                changeTime <= maxDate
            ) {

                const x =
                    xPosition(
                        changeTime
                    );


                const marker =
                    document.createElementNS(
                        "http://www.w3.org/2000/svg",
                        "line"
                    );


                marker.setAttribute(
                    "x1",
                    x
                );


                marker.setAttribute(
                    "x2",
                    x
                );


                marker.setAttribute(
                    "y1",
                    margin.top
                );


                marker.setAttribute(
                    "y2",
                    height
                    -
                    margin.bottom
                );


                marker.setAttribute(
                    "class",
                    "investigation-change-line"
                );


                svg.appendChild(
                    marker
                );


                addSvgText(

                    svg,

                    x
                    +
                    7,

                    margin.top
                    +
                    10,

                    "investigation-change-text",

                    `Detected change — ${formatDate(
                        detected.first_sustained_change
                    )}`

                );

            }

        }
    }


    // ==================================================
    // Initialize Dashboard
    // ==================================================

    async function initializeDashboard() {
        try {
            if (window.pitcherResearchLab?.ready) {
                await window.pitcherResearchLab.ready;
            }
            if (!window.pitcherResearchLab?.pitcherId) return;

            const [
                changesResponse,
                timelineResponse
            ] =

                await Promise.all(
                    [

                        fetch(
                            window.pitcherResearchLab.apiUrl("changes")
                        ),

                        fetch(
                            window.pitcherResearchLab.apiUrl("timeline")
                        )

                    ]
                );


            if (
                !changesResponse.ok
                ||
                !timelineResponse.ok
            ) {
                throw new Error(
                    "Research API request failed."
                );
            }


            const changesPayload =
                await changesResponse.json();


            const timelinePayload =
                await timelineResponse.json();


            changeData =
                Array.isArray(
                    changesPayload
                )
                    ?
                    changesPayload
                    :
                    [];


            timelineData =
                Array.isArray(
                    timelinePayload
                )
                    ?
                    timelinePayload
                    :
                    [];


            updateDeviationList(
                changeData
            );


            updateChangeTimeline(
                changeData
            );


            const startingPitch =
                pitchSelect?.value
                ||
                "FF";


            await loadPitchData(
                startingPitch
            );


            drawInvestigationChart();

        }

        catch (
            error
        ) {

            console.error(
                "Dashboard initialization error:",
                error
            );

            const deviations = document.getElementById("deviation-list");
            const timeline = document.getElementById("change-timeline");
            if (deviations) {
                deviations.innerHTML = '<div class="deviation-row"><div class="deviation-info"><div class="deviation-name">Change screening unavailable</div><div class="deviation-detail">This pitcher does not currently have enough usable data for this comparison.</div></div></div>';
            }
            if (timeline) {
                timeline.innerHTML = '<div class="timeline-event"><div class="timeline-content"><div class="timeline-metric">Timeline unavailable</div><div class="timeline-direction">More regular-season tracking data is needed.</div></div></div>';
            }

        }
    }


    // ==================================================
    // Events
    // ==================================================

    if (
        pitchSelect
    ) {

        pitchSelect.addEventListener(

            "change",

            () => {

                loadPitchData(
                    pitchSelect.value
                );

            }

        );

    }


    const investigationPitch =
        document.getElementById(
            "investigation-pitch"
        );


    const investigationMetric =
        document.getElementById(
            "investigation-metric"
        );


    if (
        investigationPitch
    ) {

        investigationPitch.addEventListener(
            "change",
            drawInvestigationChart
        );

    }


    if (
        investigationMetric
    ) {

        investigationMetric.addEventListener(
            "change",
            drawInvestigationChart
        );

    }


    initializeDashboard();

})();
