// Pitcher Research Lab — Full Career Timeline / Baseline Audit

(() => {

    "use strict";


    const pitchNames = {

        FF: "Four-Seam Fastball",
        SI: "Sinker",
        SL: "Slider",
        ST: "Sweeper",
        FS: "Splitter",
        CH: "Changeup",
        CU: "Curveball"

    };


    const pitchMetrics = {

        avg_velocity: {

            label: "Velocity",
            unit: "mph",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_spin: {

            label: "Spin Rate",
            unit: "rpm",
            decimals: 0,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_extension: {

            label: "Extension",
            unit: "ft",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_release_x: {

            label: "Release X",
            unit: "ft",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_release_z: {

            label: "Release Z",
            unit: "ft",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_horizontal_movement: {

            label: "Horizontal Movement",
            unit: "in",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        avg_vertical_movement: {

            label: "Vertical Movement",
            unit: "in",
            decimals: 2,

            min:
                row =>
                    row.pitch_count >= 5

        },


        usage_pct: {

            label: "Usage",
            unit: "%",
            decimals: 1,

            min:
                row =>
                    row.pitch_count >= 1

        },


        whiff_pct: {

            label: "Whiff Rate",
            unit: "%",
            decimals: 1,

            min:
                row =>
                    row.swings >= 5

        },


        pitch_value_per_100: {

            label:
                "Pitch Value (RV/100)",

            unit:
                "runs / 100",

            decimals:
                2,

            min:
                row =>
                    row.pitch_count >= 5

        }

    };


    const overallMetrics = {

        earned_runs: {

            label:
                "Earned Runs",

            unit:
                "ER",

            decimals:
                0,

            source:
                "official"

        },


        k_minus_bb_pct: {

            label:
                "K-BB%",

            unit:
                "%",

            decimals:
                1,

            source:
                "official"

        },


        whiff_pct: {

            label:
                "Whiff Rate",

            unit:
                "%",

            decimals:
                1,

            source:
                "process"

        },


        chase_pct: {

            label:
                "Chase Rate",

            unit:
                "%",

            decimals:
                1,

            source:
                "process"

        },


        hard_hit_pct: {

            label:
                "Hard-Hit Rate",

            unit:
                "%",

            decimals:
                1,

            source:
                "process"

        },


        avg_exit_velocity: {

            label:
                "Average Exit Velocity",

            unit:
                "mph",

            decimals:
                1,

            source:
                "process"

        },


        xwoba_allowed: {

            label:
                "xwOBA Allowed",

            unit:
                "xwOBA",

            decimals:
                3,

            source:
                "process"

        },


        pitch_value_per_100: {

            label:
                "Pitch Value (RV/100)",

            unit:
                "runs / 100",

            decimals:
                2,

            source:
                "process"

        }

    };


    let careerData = null;

    let overallByGame =
        new Map();


    function availableSeasons() {
        return [...new Set(
            (careerData?.overall_outings || [])
                .map(row => Number(row.season))
                .filter(Number.isFinite)
        )].sort((a, b) => a - b);
    }


    function populateSeasonOptions() {
        const select = document.getElementById("career-season");
        if (!select) return;
        const current = select.value || "ALL";
        select.innerHTML = '<option value="ALL">All MLB Outings</option>';
        availableSeasons().forEach(season => {
            const option = document.createElement("option");
            option.value = String(season);
            option.textContent = String(season);
            select.appendChild(option);
        });
        select.value = [...select.options].some(option => option.value === current) ? current : "ALL";
    }


    const $ =
        id =>
            document.getElementById(
                id
            );


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


        const number =
            Number(
                value
            );


        return Number.isFinite(
            number
        )
            ? number
            : null;

    }


    function formatValue(
        value,
        metric
    ) {

        const number =
            numberOrNull(
                value
            );


        if (
            number === null
        ) {

            return "--";

        }


        return number.toFixed(
            metric.decimals
        );

    }


    function formatDate(
        value,
        includeYear = false
    ) {

        if (!value) {
            return "--";
        }


        const date =
            new Date(
                `${value}T00:00:00`
            );


        return date.toLocaleDateString(
            "en-US",

            includeYear

                ? {
                    month:
                        "short",

                    day:
                        "numeric",

                    year:
                        "numeric"
                }

                : {
                    month:
                        "short",

                    day:
                        "numeric"
                }

        );

    }


    function loadStyles() {

        if (
            $("career-styles")
        ) {

            return;

        }


        const link =
            document.createElement(
                "link"
            );


        link.id =
            "career-styles";


        link.rel =
            "stylesheet";


        link.href =
            "/static/career.css";


        document.head.appendChild(
            link
        );

    }


    function buildShell() {

        const view =
            document.querySelector(
                '[data-view-panel="changes"]'
            );


        const header =
            view?.querySelector(
                ".view-page-header"
            );


        if (
            !view
            ||
            !header
        ) {

            return false;

        }


        if (
            $("career-audit-panel")
        ) {

            return true;

        }


        const section =
            document.createElement(
                "section"
            );


        section.id =
            "career-audit-panel";


        section.className =
            "career-audit";


        section.innerHTML = `

            <div class="career-audit-header">

                <div>

                    <div class="eyebrow">
                        FULL MLB CAREER
                    </div>

                    <h3>
                        Career Timeline & Baseline Audit
                    </h3>

                    <p>
                        Inspect every regular-season outing before
                        deciding which periods deserve closer study.
                        The current screening window is a comparison
                        aid, not an assumed starting point.
                    </p>

                </div>


                <div
                    class="career-range"
                    id="career-range"
                >
                    Loading career data...
                </div>

            </div>


            <div class="career-controls">


                <div class="career-control">

                    <label for="career-season">
                        Time Range
                    </label>

                    <select id="career-season">

                        <option
                            value="ALL"
                            selected
                        >
                            All MLB Outings
                        </option>


                    </select>

                </div>


                <div class="career-control">

                    <label for="career-scope">
                        Analysis
                    </label>

                    <select id="career-scope">

                        <option
                            value="PITCH"
                            selected
                        >
                            Pitch Profile
                        </option>

                        <option value="OVERALL">
                            Overall Performance
                        </option>

                    </select>

                </div>


                <div
                    class="career-control"
                    id="career-pitch-control"
                >

                    <label for="career-pitch">
                        Pitch
                    </label>

                    <select id="career-pitch">

                        <option
                            value="FF"
                            selected
                        >
                            Four-Seam Fastball
                        </option>

                        <option value="SI">
                            Sinker
                        </option>

                        <option value="CH">
                            Changeup
                        </option>

                        <option value="FS">
                            Splitter
                        </option>

                        <option value="SL">
                            Slider
                        </option>

                        <option value="ST">
                            Sweeper
                        </option>

                        <option value="CU">
                            Curveball
                        </option>

                    </select>

                </div>


                <div
                    class="
                        career-control
                        career-control-wide
                    "
                >

                    <label for="career-metric">
                        Metric
                    </label>

                    <select id="career-metric">
                    </select>

                </div>


                <label class="career-check">

                    <input
                        id="career-rolling"
                        type="checkbox"
                        checked
                    >

                    <span>
                        Show 3-outing rolling average
                    </span>

                </label>

            </div>


            <div
                class="career-chart-card"
                id="career-chart-card"
            >

                <div class="career-chart-heading">

                    <div>

                        <div
                            class="career-chart-title"
                            id="career-chart-title"
                        >
                            Four-Seam Fastball — Velocity
                        </div>

                        <div
                            class="career-chart-subtitle"
                            id="career-chart-subtitle"
                        >
                        </div>

                    </div>


                    <div
                        class="career-chart-sample"
                        id="career-chart-sample"
                    >
                        --
                    </div>

                </div>


                <div class="career-chart-wrap">

                    <svg
                        id="career-chart"
                        viewBox="0 0 1100 390"
                        preserveAspectRatio="xMidYMid meet"
                    >
                    </svg>

                </div>

            </div>


            <div class="career-audit-heading">

                <div>

                    <div class="eyebrow">
                        DESCRIPTIVE DRIFT CHECK
                    </div>

                    <h4>
                        Season Audit
                    </h4>

                </div>


                <div class="career-audit-note">

                    First 5 vs last 5 usable outings —
                    descriptive, not a formal change point

                </div>

            </div>


            <div
                class="career-season-grid"
                id="career-season-grid"
            >
            </div>


            <div class="career-table-card">

                <div class="career-table-heading">

                    <div>

                        <div class="eyebrow">
                            OUTING LOG
                        </div>

                        <h4 id="career-table-title">
                            All MLB outings
                        </h4>

                    </div>


                    <div
                        class="career-table-note"
                        id="career-table-note"
                    >
                        --
                    </div>

                </div>


                <div
                    class="
                        table-wrapper
                        career-table-wrap
                    "
                >

                    <table class="career-table">

                        <thead>

                            <tr>

                                <th>
                                    Date
                                </th>

                                <th>
                                    Season
                                </th>

                                <th>
                                    Opponent
                                </th>

                                <th
                                    id="career-table-metric-label"
                                >
                                    Velocity
                                </th>

                                <th>
                                    Sample
                                </th>

                            </tr>

                        </thead>


                        <tbody id="career-table-body">
                        </tbody>

                    </table>

                </div>

            </div>


            <div class="career-method-note">

                <strong>
                    Why this exists:
                </strong>

                the current-season detector is useful for screening,
                but this view keeps the analysis from assuming
                that an important shift began in one specific
                window. The complete career trajectory provides
                context before deeper analysis.

            </div>

        `;


        header.insertAdjacentElement(
            "afterend",
            section
        );


        return true;

    }


    function currentMetricMap() {

        return (
            $("career-scope")?.value
            ===
            "OVERALL"
        )

            ? overallMetrics

            : pitchMetrics;

    }


    function populateMetricOptions() {

        const select =
            $("career-metric");


        if (!select) {
            return;
        }


        const scope =
            $("career-scope")?.value
            ||
            "PITCH";


        const map =
            currentMetricMap();


        const preferred =

            scope === "PITCH"

                ? "avg_velocity"

                : "xwoba_allowed";


        const existing =
            select.value;


        select.innerHTML =

            Object.entries(
                map
            )
            .map(
                (
                    [
                        key,
                        metric
                    ]
                ) =>
                    `
                        <option value="${key}">
                            ${metric.label}
                        </option>
                    `
            )
            .join("");


        select.value =

            map[
                existing
            ]

                ? existing

                : preferred;


        const pitchControl =
            $("career-pitch-control");


        if (
            pitchControl
        ) {

            pitchControl.classList.toggle(
                "career-hidden",
                scope === "OVERALL"
            );

        }

    }


    function getMetricValue(
        row,
        key,
        scope
    ) {

        if (
            scope === "PITCH"
        ) {

            return numberOrNull(
                row[
                    key
                ]
            );

        }


        const metric =
            overallMetrics[
                key
            ];


        return numberOrNull(

            row?.[
                metric.source
            ]?.[
                key
            ]

        );

    }


    function allCareerGameIndex() {

        const map =
            new Map();


        (
            careerData?.overall_outings
            ||
            []
        )
        .forEach(
            (
                row,
                index
            ) => {

                map.set(
                    Number(
                        row.game_pk
                    ),
                    index
                );

            }
        );


        return map;

    }


    function selectedRows() {

        const scope =
            $("career-scope")?.value
            ||
            "PITCH";


        const season =
            $("career-season")?.value
            ||
            "ALL";


        const metricKey =
            $("career-metric")?.value;


        const metric =
            currentMetricMap()[
                metricKey
            ];


        const pitchType =
            $("career-pitch")?.value
            ||
            "FF";


        const gameIndex =
            allCareerGameIndex();


        let rows;


        if (
            scope === "PITCH"
        ) {

            rows = (

                careerData?.pitch_outings
                ||
                []

            )
            .filter(
                row =>
                    row.pitch_type
                    ===
                    pitchType
            )
            .filter(
                row =>
                    metric?.min

                        ? metric.min(
                            row
                        )

                        : true
            );

        }

        else {

            rows = [
                ...(
                    careerData?.overall_outings
                    ||
                    []
                )
            ];

        }


        if (
            season !== "ALL"
        ) {

            rows = rows.filter(
                row =>
                    Number(
                        row.season
                    )
                    ===
                    Number(
                        season
                    )
            );

        }


        return rows

            .map(
                row => ({

                    ...row,

                    career_index:
                        gameIndex.get(
                            Number(
                                row.game_pk
                            )
                        ),

                    value:
                        getMetricValue(
                            row,
                            metricKey,
                            scope
                        )

                })
            )

            .filter(
                row =>
                    row.value
                    !==
                    null

                    &&

                    row.career_index
                    !==
                    undefined
            )

            .sort(
                (
                    a,
                    b
                ) =>
                    a.career_index
                    -
                    b.career_index
            );

    }


    function selectedMetric() {

        return currentMetricMap()[
            $("career-metric")?.value
        ];

    }


    function contextForRow(
        row
    ) {

        return (

            overallByGame.get(
                Number(
                    row.game_pk
                )
            )

            ||

            row

        );

    }


    function opponentText(
        row
    ) {

        const context =
            contextForRow(
                row
            );


        const official =
            context?.official;


        if (
            !official?.opponent
        ) {

            return "--";

        }


        return (

            `${

                official.home_away
                ===
                "Away"

                    ? "@"

                    : "vs"

            } ${official.opponent}`

        );

    }


    function sampleText(
        row
    ) {

        const scope =
            $("career-scope")?.value
            ||
            "PITCH";


        const key =
            $("career-metric")?.value;


        if (
            scope === "OVERALL"
        ) {

            if (
                [
                    "hard_hit_pct",
                    "avg_exit_velocity"
                ]
                .includes(
                    key
                )
            ) {

                return (
                    `${

                        row.process
                        ?.tracked_batted_balls
                        ??
                        "--"

                    } BIP`
                );

            }


            if (
                key === "xwoba_allowed"
            ) {

                return (
                    `${

                        row.official
                        ?.batters_faced
                        ??
                        "--"

                    } BF`
                );

            }


            return (
                `${

                    row.process
                    ?.pitches
                    ??
                    "--"

                } pitches`
            );

        }


        if (
            key === "whiff_pct"
        ) {

            return (
                `${

                    row.swings
                    ??
                    "--"

                } swings`
            );

        }


        return (
            `${

                row.pitch_count
                ??
                "--"

            } pitches`
        );

    }


    function rollingValues(
        rows
    ) {

        return rows.map(
            (
                row,
                index
            ) => {

                if (
                    index < 2
                ) {

                    return {
                        ...row,
                        rolling:
                            null
                    };

                }


                const window =
                    rows.slice(
                        index - 2,
                        index + 1
                    );


                const avg =

                    window.reduce(
                        (
                            sum,
                            item
                        ) =>
                            sum
                            +
                            item.value,
                        0
                    )

                    /

                    3;


                return {

                    ...row,

                    rolling:
                        avg

                };

            }
        );

    }


    function svgElement(
        name,
        attrs = {}
    ) {

        const el =
            document.createElementNS(
                "http://www.w3.org/2000/svg",
                name
            );


        Object.entries(
            attrs
        )
        .forEach(
            (
                [
                    key,
                    value
                ]
            ) => {

                el.setAttribute(
                    key,
                    value
                );

            }
        );


        return el;

    }


    function svgText(
        svg,
        x,
        y,
        className,
        value,
        anchor = "start"
    ) {

        const el =
            svgElement(
                "text",
                {
                    x,
                    y,
                    class:
                        className,

                    "text-anchor":
                        anchor
                }
            );


        el.textContent =
            value;


        svg.appendChild(
            el
        );

    }


    function drawChart() {

        const svg =
            $("career-chart");


        if (!svg) {
            return;
        }


        const rows =
            selectedRows();


        const metric =
            selectedMetric();


        const metricKey =
            $("career-metric")?.value;


        const scope =
            $("career-scope")?.value
            ||
            "PITCH";


        const pitchType =
            $("career-pitch")?.value
            ||
            "FF";


        const season =
            $("career-season")?.value
            ||
            "ALL";


        const showRolling =
            $("career-rolling")?.checked
            ??
            true;


        svg.innerHTML =
            "";


        const title =

            scope === "PITCH"

                ? `${

                    pitchNames[
                        pitchType
                    ]
                    ??
                    pitchType

                } — ${metric.label}`

                : `Overall Performance — ${metric.label}`;


        $(
            "career-chart-title"
        ).textContent =
            title;


        $(
            "career-chart-subtitle"
        ).textContent =

            season === "ALL"

                ? (
                    "Complete MLB outing sequence; "
                    +
                    "offseasons are separated by "
                    +
                    "season markers rather than calendar gaps."
                )

                : `${season} regular-season outings.`;


        $(
            "career-chart-sample"
        ).textContent =
            `${rows.length} usable outings`;


        if (
            !rows.length
        ) {

            svgText(
                svg,
                550,
                195,
                "career-empty",
                "No usable outings for this selection.",
                "middle"
            );

            return;

        }


        const W =
            1100;


        const H =
            390;


        const M = {

            l:
                76,

            r:
                28,

            t:
                42,

            b:
                58

        };


        const fullOutings =

            careerData?.overall_outings
            ||
            [];


        const xDomainRows =

            season === "ALL"

                ? fullOutings

                : fullOutings.filter(
                    row =>
                        Number(
                            row.season
                        )
                        ===
                        Number(
                            season
                        )
                );


        const gameIndex =
            allCareerGameIndex();


        const firstIndex =
            Math.min(
                ...xDomainRows.map(
                    row =>
                        gameIndex.get(
                            Number(
                                row.game_pk
                            )
                        )
                )
            );


        const lastIndex =
            Math.max(
                ...xDomainRows.map(
                    row =>
                        gameIndex.get(
                            Number(
                                row.game_pk
                            )
                        )
                )
            );


        const values =
            rows.map(
                row =>
                    row.value
            );


        let min =
            Math.min(
                ...values
            );


        let max =
            Math.max(
                ...values
            );


        let spread =
            max - min;


        if (
            spread === 0
        ) {

            spread =
                Math.max(
                    Math.abs(
                        max
                    ),
                    1
                );

        }


        const padding =
            spread
            *
            0.16;


        min -=
            padding;


        max +=
            padding;


        if (
            metricKey
            ===
            "earned_runs"
        ) {

            min =
                Math.min(
                    0,
                    min
                );

        }


        const x =
            index => {

                if (
                    lastIndex
                    ===
                    firstIndex
                ) {

                    return W / 2;

                }


                return (

                    M.l

                    +

                    (
                        (
                            index
                            -
                            firstIndex
                        )

                        /

                        (
                            lastIndex
                            -
                            firstIndex
                        )
                    )

                    *

                    (
                        W
                        -
                        M.l
                        -
                        M.r
                    )

                );

            };


        const y =
            value => (

                M.t

                +

                (
                    (
                        max
                        -
                        value
                    )

                    /

                    (
                        max
                        -
                        min
                    )
                )

                *

                (
                    H
                    -
                    M.t
                    -
                    M.b
                )

            );


        for (
            let i = 0;
            i <= 5;
            i += 1
        ) {

            const value =

                min

                +

                (
                    (
                        max
                        -
                        min
                    )

                    *
                    i

                    /
                    5
                );


            const yy =
                y(
                    value
                );


            svg.appendChild(
                svgElement(
                    "line",
                    {

                        x1:
                            M.l,

                        x2:
                            W - M.r,

                        y1:
                            yy,

                        y2:
                            yy,

                        class:
                            "career-grid-line"

                    }
                )
            );


            svgText(
                svg,
                M.l - 12,
                yy + 4,
                "career-axis-text",

                value.toFixed(
                    metric.decimals
                ),

                "end"
            );

        }


        const seasonsToMark =

            season === "ALL"

                ? availableSeasons()

                : [
                    Number(
                        season
                    )
                ];


        seasonsToMark.forEach(
            seasonValue => {

                const seasonRows =
                    fullOutings.filter(
                        row =>
                            Number(
                                row.season
                            )
                            ===
                            seasonValue
                    );


                if (
                    !seasonRows.length
                ) {

                    return;

                }


                const startIndex =
                    gameIndex.get(
                        Number(
                            seasonRows[
                                0
                            ].game_pk
                        )
                    );


                const endIndex =
                    gameIndex.get(
                        Number(
                            seasonRows[
                                seasonRows.length
                                -
                                1
                            ].game_pk
                        )
                    );


                if (
                    startIndex
                    <
                    firstIndex

                    ||

                    startIndex
                    >
                    lastIndex
                ) {

                    return;

                }


                svg.appendChild(
                    svgElement(
                        "line",
                        {

                            x1:
                                x(
                                    startIndex
                                ),

                            x2:
                                x(
                                    startIndex
                                ),

                            y1:
                                M.t,

                            y2:
                                H - M.b,

                            class:
                                "career-season-line"

                        }
                    )
                );


                svgText(
                    svg,

                    x(
                        (
                            startIndex
                            +
                            endIndex
                        )
                        /
                        2
                    ),

                    H - 19,

                    "career-season-label",

                    String(
                        seasonValue
                    ),

                    "middle"
                );

            }
        );


        const screen =
            careerData?.current_screen_window;


        if (
            screen

            &&

            (
                season === "ALL"

                ||

                season === String(
                    new Date(`${screen.start}T00:00:00`).getFullYear()
                )
            )
        ) {

            const inWindow =

                fullOutings.filter(
                    row =>
                        row.game_date
                        >=
                        screen.start

                        &&

                        row.game_date
                        <=
                        screen.end
                );


            if (
                inWindow.length
            ) {

                const startIndex =
                    gameIndex.get(
                        Number(
                            inWindow[
                                0
                            ].game_pk
                        )
                    );


                const endIndex =
                    gameIndex.get(
                        Number(
                            inWindow[
                                inWindow.length
                                -
                                1
                            ].game_pk
                        )
                    );


                const left =
                    Math.max(
                        x(
                            startIndex
                        )
                        -
                        8,
                        M.l
                    );


                const right =
                    Math.min(
                        x(
                            endIndex
                        )
                        +
                        8,
                        W - M.r
                    );


                svg.appendChild(
                    svgElement(
                        "rect",
                        {

                            x:
                                left,

                            y:
                                M.t,

                            width:
                                Math.max(
                                    right
                                    -
                                    left,
                                    1
                                ),

                            height:
                                H
                                -
                                M.t
                                -
                                M.b,

                            class:
                                "career-screen-window"

                        }
                    )
                );


                svgText(
                    svg,

                    (
                        left
                        +
                        right
                    )
                    /
                    2,

                    M.t + 15,

                    "career-window-label",

                    "Current screen window",

                    "middle"
                );

            }

        }


        const plotted =
            rollingValues(
                rows
            );


        plotted.forEach(
            row => {

                const circle =
                    svgElement(
                        "circle",
                        {

                            cx:
                                x(
                                    row.career_index
                                ),

                            cy:
                                y(
                                    row.value
                                ),

                            r:
                                4,

                            class:
                                "career-raw-point"

                        }
                    );


                const titleElement =
                    svgElement(
                        "title"
                    );


                titleElement.textContent =
                    `${

                        formatDate(
                            row.game_date,
                            true
                        )

                    } ${

                        opponentText(
                            row
                        )

                    }\n${

                        metric.label

                    }: ${

                        formatValue(
                            row.value,
                            metric
                        )

                    } ${

                        metric.unit

                    }\n${

                        sampleText(
                            row
                        )

                    }`;


                circle.appendChild(
                    titleElement
                );


                svg.appendChild(
                    circle
                );

            }
        );


        if (
            showRolling
        ) {

            const rollingRows =

                plotted.filter(
                    row =>
                        row.rolling
                        !==
                        null
                );


            if (
                rollingRows.length
            ) {

                const path =
                    svgElement(
                        "path",
                        {

                            d:
                                rollingRows
                                .map(
                                    (
                                        row,
                                        index
                                    ) =>
                                        `${

                                            index
                                                ? "L"
                                                : "M"

                                        } ${

                                            x(
                                                row.career_index
                                            )

                                        } ${

                                            y(
                                                row.rolling
                                            )

                                        }`
                                )
                                .join(
                                    " "
                                ),

                            class:
                                "career-rolling-line"

                        }
                    );


                svg.appendChild(
                    path
                );

            }

        }


        svgText(
            svg,
            M.l,
            23,
            "career-unit-label",
            metric.unit
        );

    }


    function mean(
        rows
    ) {

        if (
            !rows.length
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
                    row.value,
                0
            )

            /

            rows.length

        );

    }


    function selectedRowsIgnoringSeason() {

        const saved =
            $("career-season").value;


        $("career-season").value =
            "ALL";


        const rows =
            selectedRows();


        $("career-season").value =
            saved;


        return rows;

    }


    function renderSeasonAudit() {

        const container =
            $("career-season-grid");


        if (!container) {
            return;
        }


        const allRows =
            selectedRowsIgnoringSeason();


        const metric =
            selectedMetric();


        container.innerHTML =
            "";


        availableSeasons()
        .forEach(
            season => {

                const rows =
                    allRows.filter(
                        row =>
                            Number(
                                row.season
                            )
                            ===
                            season
                    );


                const seasonMean =
                    mean(
                        rows
                    );


                const firstFive =
                    mean(
                        rows.slice(
                            0,
                            5
                        )
                    );


                const lastFive =
                    mean(
                        rows.slice(
                            -5
                        )
                    );


                const delta =

                    (
                        firstFive
                        !==
                        null

                        &&

                        lastFive
                        !==
                        null
                    )

                        ? (
                            lastFive
                            -
                            firstFive
                        )

                        : null;


                const card =
                    document.createElement(
                        "article"
                    );


                card.className =
                    "career-season-card";


                card.innerHTML = `

                    <div class="career-season-card-top">

                        <strong>
                            ${season}
                        </strong>

                        <span>
                            ${rows.length} usable outings
                        </span>

                    </div>


                    <div class="career-season-main-value">

                        ${formatValue(
                            seasonMean,
                            metric
                        )}

                        <small>
                            ${metric.unit}
                        </small>

                    </div>


                    <div class="career-season-main-label">
                        Season average
                    </div>


                    <div class="career-season-detail-grid">

                        <div>

                            <span>
                                First 5
                            </span>

                            <strong>
                                ${formatValue(
                                    firstFive,
                                    metric
                                )}
                            </strong>

                        </div>


                        <div>

                            <span>
                                Last 5
                            </span>

                            <strong>
                                ${formatValue(
                                    lastFive,
                                    metric
                                )}
                            </strong>

                        </div>


                        <div>

                            <span>
                                Last 5 − First 5
                            </span>

                            <strong>

                                ${
                                    delta
                                    ===
                                    null

                                        ? "--"

                                        : `${

                                            delta
                                            >
                                            0

                                                ? "+"

                                                : ""

                                        }${

                                            delta.toFixed(
                                                metric.decimals
                                            )

                                        }`
                                }

                            </strong>

                        </div>

                    </div>

                `;


                container.appendChild(
                    card
                );

            }
        );

    }


    function renderTable() {

        const body =
            $("career-table-body");


        if (!body) {
            return;
        }


        const rows =
            selectedRows();


        const metric =
            selectedMetric();


        const season =
            $("career-season")?.value
            ||
            "ALL";


        $(
            "career-table-title"
        ).textContent =

            season === "ALL"

                ? "All MLB outings"

                : `${season} outings`;


        $(
            "career-table-note"
        ).textContent =
            `${rows.length} usable outings`;


        $(
            "career-table-metric-label"
        ).textContent =
            metric.label;


        body.innerHTML =

            rows.map(
                row => `

                    <tr>

                        <td>
                            ${formatDate(
                                row.game_date,
                                true
                            )}
                        </td>

                        <td>
                            ${row.season}
                        </td>

                        <td>
                            ${opponentText(
                                row
                            )}
                        </td>

                        <td>

                            ${formatValue(
                                row.value,
                                metric
                            )}

                            ${metric.unit}

                        </td>

                        <td>
                            ${sampleText(
                                row
                            )}
                        </td>

                    </tr>

                `
            )
            .join("");

    }


    function renderAll() {

        drawChart();

        renderSeasonAudit();

        renderTable();

    }


    function bindControls() {

        $(
            "career-scope"
        )?.addEventListener(
            "change",
            () => {

                populateMetricOptions();

                renderAll();

            }
        );


        [

            "career-season",
            "career-pitch",
            "career-metric",
            "career-rolling"

        ]
        .forEach(
            id => {

                $(
                    id
                )?.addEventListener(
                    "change",
                    renderAll
                );

            }
        );

    }


    async function init() {
        if (window.pitcherResearchLab?.ready) {
            await window.pitcherResearchLab.ready;
        }
        if (!window.pitcherResearchLab?.pitcherId) return;

        loadStyles();


        if (
            !buildShell()
        ) {

            return;

        }


        try {

            const response =
                await fetch(
                    window.pitcherResearchLab.apiUrl("career")
                );


            if (
                !response.ok
            ) {

                throw new Error(
                    `Career API ${response.status}`
                );

            }


            careerData =
                await response.json();


            overallByGame =
                new Map(

                    (
                        careerData.overall_outings
                        ||
                        []
                    )
                    .map(
                        row => [

                            Number(
                                row.game_pk
                            ),

                            row

                        ]
                    )

                );


            $(
                "career-range"
            ).textContent =

                `${

                    formatDate(
                        careerData.career_start,
                        true
                    )

                } → ${

                    formatDate(
                        careerData.career_end,
                        true
                    )

                }`;


            populateSeasonOptions();

            populateMetricOptions();

            bindControls();

            renderAll();

        }

        catch (
            error
        ) {

            console.error(
                "Career Timeline error:",
                error
            );


            $(
                "career-range"
            ).textContent =
                "Career data could not be loaded";


            const chart =
                $("career-chart-card");


            if (
                chart
            ) {

                chart.innerHTML = `

                    <div class="career-error">

                        Career Timeline could not load.
                        Check the Flask terminal and
                        browser console.

                    </div>

                `;

            }

        }

    }


    init();

})();
