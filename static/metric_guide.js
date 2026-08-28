// ==================================================
// Pitcher Research Lab
// Global Metric Guide
// ==================================================


// ==================================================
// Load CSS
// ==================================================

function loadMetricGuideStyles() {

    if (
        document.getElementById(
            "metric-guide-styles"
        )
    ) {
        return;
    }


    const link =
        document.createElement(
            "link"
        );


    link.id =
        "metric-guide-styles";


    link.rel =
        "stylesheet";


    link.href =
        "/static/metric_guide.css";


    document.head.appendChild(
        link
    );

}


// ==================================================
// Metric Definitions
// ==================================================

const metricGuideDefinitions = [

    {
        key: "percentage_points",
        category: "Reading Changes",
        name: "Percentage Points",
        short: "The direct difference between two percentages.",
        why: "Use percentage points when comparing rates. If whiff rate rises from 24% to 28%, that is an increase of 4 percentage points.",
        note: "This is different from saying the rate increased by 4 percent."
    },

    {
        key: "pitch_value",
        category: "Pitch Results",
        name: "Pitch Value (RV/100)",
        short: "Estimated run impact per 100 pitches from the pitcher's perspective.",
        why: "It summarizes how much the outcomes of a pitch helped or hurt the pitcher after scaling to 100 pitches.",
        note: "In Pitcher Research Lab, positive is better for the pitcher because Statcast delta run expectancy is sign-flipped. This is descriptive, not causal."
    },

    {
        key: "whiff_rate",
        category: "Pitch Results",
        name: "Whiff Rate",
        short: "Swinging misses divided by total swings.",
        why: "Higher whiff rate means hitters are missing the pitch more frequently when they decide to swing.",
        note: "The denominator is swings, not total pitches."
    },

    {
        key: "chase_rate",
        category: "Pitch Results",
        name: "Chase Rate",
        short: "How often hitters swing at pitches outside the strike zone.",
        why: "A pitch can create value by convincing hitters to offer at pitches they normally should take.",
        note: "The exact rate depends on how the strike zone is defined."
    },

    {
        key: "hard_hit_rate",
        category: "Contact",
        name: "Hard-Hit Rate",
        short: "Percentage of tracked batted balls hit at least 95 mph.",
        why: "It helps describe the damage hitters create when they put the pitch into play.",
        note: "Small batted-ball samples can move this rate substantially."
    },

    {
        key: "exit_velocity",
        category: "Contact",
        name: "Average Exit Velocity",
        short: "Average speed of tracked batted balls off the bat.",
        why: "Lower exit velocity generally means weaker contact from the pitcher's perspective.",
        note: "It only describes balls with tracked contact."
    },

    {
        key: "velocity",
        category: "Pitch Characteristics",
        name: "Velocity",
        short: "Pitch speed measured near release.",
        why: "Velocity can affect reaction time and pitch interaction, but velocity alone does not determine pitch quality.",
        note: "A lower-velocity pitch can still perform well if shape, location or sequencing improve."
    },

    {
        key: "spin",
        category: "Pitch Characteristics",
        name: "Spin Rate",
        short: "How quickly the baseball rotates, measured in revolutions per minute.",
        why: "Spin contributes to pitch movement, but more spin is not automatically better.",
        note: "Spin direction, efficiency, seam effects and pitch design also matter."
    },

    {
        key: "vertical_movement",
        category: "Pitch Characteristics",
        name: "Vertical Movement",
        short: "Vertical movement relative to the pitch's gravity-only trajectory.",
        why: "It helps describe pitch shape. For a four-seam fastball, more induced vertical movement can help the pitch stay above a hitter's expected bat path.",
        note: "Movement values should be interpreted by pitch type."
    },

    {
        key: "horizontal_movement",
        category: "Pitch Characteristics",
        name: "Horizontal Movement",
        short: "Side-to-side movement of the pitch.",
        why: "Horizontal movement helps define pitch shape and separation between pitches.",
        note: "Positive and negative direction depend on the tracking coordinate system."
    },

    {
        key: "extension",
        category: "Release",
        name: "Extension",
        short: "How far toward home plate the pitcher releases the baseball.",
        why: "More extension can reduce the effective distance the ball travels and alter perceived velocity and pitch shape.",
        note: "More extension is not automatically better."
    },

    {
        key: "release_x",
        category: "Release",
        name: "Release X",
        short: "Horizontal release position.",
        why: "Changes can indicate a different release location or delivery pattern.",
        note: "A change does not prove that the pitcher's mechanics changed."
    },

    {
        key: "release_z",
        category: "Release",
        name: "Release Z",
        short: "Vertical height of the release point.",
        why: "Tracking changes can help identify whether the delivery/release profile moved over time.",
        note: "Pitch type and classification should be considered."
    },

    {
        key: "usage",
        category: "Strategy",
        name: "Pitch Usage",
        short: "Percentage of all pitches that are a particular pitch type.",
        why: "A large usage change can indicate a meaningful arsenal or game-planning shift.",
        note: "A usage change does not tell us why the strategy changed."
    },

    {
        key: "zone_rate",
        category: "Location",
        name: "Zone Rate",
        short: "Percentage of pitches inside the normalized strike zone.",
        why: "It helps describe how frequently the pitcher is attacking the strike zone.",
        note: "Zone rate is not automatically command quality."
    },

    {
        key: "zone_whiff",
        category: "Location",
        name: "Zone Whiff Rate",
        short: "Whiffs divided by swings on pitches located inside the strike zone.",
        why: "It helps distinguish pitches that miss bats even when hitters receive a strike.",
        note: "It can be especially useful when paired with chase rate."
    },

    {
        key: "heart_rate",
        category: "Location",
        name: "Heart Rate",
        short: "Share of pitches located in the central portion of our normalized strike zone.",
        why: "It provides a simple measure of how often pitches are located near the middle of the zone.",
        note: "This is an internal Pitcher Research Lab definition, not an official MLB Statcast metric."
    },

    {
        key: "edge_rate",
        category: "Location",
        name: "Edge Rate",
        short: "Share of pitches located in the outer portion of our normalized strike zone.",
        why: "It helps describe how frequently pitches are placed near the zone boundary.",
        note: "This is an internal Pitcher Research Lab definition, not an official MLB Statcast metric."
    },

    {
        key: "normalized_zone",
        category: "Location",
        name: "Normalized Strike Zone",
        short: "Pitch location rescaled to the batter-specific strike-zone dimensions.",
        why: "Normalization lets us compare locations across hitters with different strike-zone heights.",
        note: "The current location analysis is intentionally focused within the selected target season."
    },

    {
        key: "baseline",
        category: "Statistics",
        name: "Historical Baseline",
        short: "The pitcher's own prior performance used as the comparison reference.",
        why: "Rather than comparing every pitcher to one league-wide standard, the system asks whether a pitcher has changed relative to himself.",
        note: "The baseline uses the selected pitcher’s available prior-season data when possible."
    },

    {
        key: "standard_deviation",
        category: "Statistics",
        name: "Standard Deviation (SD)",
        short: "A measure of how much a metric normally varies around its average.",
        why: "It gives context to the size of a change. A 1 mph velocity shift matters differently for a pitcher who normally varies 0.3 mph versus 1.5 mph.",
        note: "The current detector uses outing-level historical variation."
    },

    {
        key: "z_score",
        category: "Statistics",
        name: "Z-Score",
        short: "How many standard deviations a value sits above or below its historical baseline.",
        why: "Large absolute z-scores identify changes that are unusual relative to the pitcher's normal variation.",
        note: "A large z-score means unusual. It does not automatically mean harmful, statistically causal or mechanically important."
    },

    {
        key: "rolling_average",
        category: "Statistics",
        name: "3-Outing Rolling Average",
        short: "The average of the most recent three outings.",
        why: "It reduces the influence of one unusually good or bad game and makes sustained trends easier to see.",
        note: "Three outings is the documented screening threshold used by the current sustained-deviation method."
    },

    {
        key: "sustained_change",
        category: "Statistics",
        name: "Sustained Change Flag",
        short: "A screening signal that a metric remained unusually far from baseline across multiple rolling windows.",
        why: "It helps identify where an analyst should begin investigating instead of manually searching every date and metric.",
        note: "A flag is not proof of injury, fatigue, mechanical change or organizational intent."
    },

    {
        key: "transition_window",
        category: "Statistics",
        name: "Transition Window",
        short: "The date range used to focus a comparison around sustained pitch changes when they are present.",
        why: "It gives the investigation a focused period for comparing early, middle and later performance.",
        note: "The window is descriptive. It does not prove that every change shares one cause."
    },

    {
        key: "sample_size",
        category: "Statistics",
        name: "Sample Size",
        short: "How many pitches, swings or batted balls support a statistic.",
        why: "Metrics become less stable when they are based on very few observations.",
        note: "Pitcher Research Lab intentionally mutes some small-sample heatmap cells."
    }

];


// ==================================================
// HTML Creation
// ==================================================

function buildMetricGuide() {

    if (
        document.getElementById(
            "metric-guide-modal"
        )
    ) {
        return;
    }


    const launcher =
        document.createElement(
            "button"
        );


    launcher.id =
        "metric-guide-launcher";


    launcher.className =
        "metric-guide-launcher";


    launcher.innerHTML = `
        <span class="metric-guide-icon">i</span>
        Metric Guide
    `;


    document.body.appendChild(
        launcher
    );


    const overlay =
        document.createElement(
            "div"
        );


    overlay.id =
        "metric-guide-modal";


    overlay.className =
        "metric-guide-overlay";


    overlay.innerHTML = `

        <div
            class="metric-guide-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="metric-guide-title"
        >

            <div class="metric-guide-header">

                <div>

                    <div class="metric-guide-eyebrow">
                        PITCHER RESEARCH LAB
                    </div>

                    <h2 id="metric-guide-title">
                        Metric Guide
                    </h2>

                    <p>
                        Professional baseball metrics explained
                        without assuming prior analytics knowledge.
                    </p>

                </div>

                <button
                    id="metric-guide-close"
                    class="metric-guide-close"
                    aria-label="Close metric guide"
                >
                    ×
                </button>

            </div>


            <div class="metric-guide-intro">

                <div>
                    <strong>
                        How to use the dashboard
                    </strong>

                    <span>
                        Start with what changed, then use the deeper
                        metrics to understand the evidence.
                    </span>
                </div>

                <div>
                    <strong>
                        Important
                    </strong>

                    <span>
                        A statistical flag identifies something worth
                        investigating. It does not prove the cause.
                    </span>
                </div>

            </div>


            <div class="metric-color-guide">

                <div>
                    <span class="metric-color-dot frequency"></span>
                    Blue = frequency / location density
                </div>

                <div>
                    <span class="metric-color-dot favorable"></span>
                    Teal = favorable pitcher result
                </div>

                <div>
                    <span class="metric-color-dot unfavorable"></span>
                    Red = unfavorable pitcher result
                </div>

                <div>
                    <span class="metric-color-dot neutral"></span>
                    Gray = neutral or insufficient sample
                </div>

                <div>
                    <span class="metric-color-dot accent"></span>
                    Gold = interface emphasis, not performance
                </div>

            </div>


            <div class="metric-guide-search-wrap">

                <input
                    id="metric-guide-search"
                    class="metric-guide-search"
                    type="text"
                    placeholder="Search metrics — whiff, RV/100, z-score, movement..."
                    autocomplete="off"
                >

            </div>


            <div
                id="metric-guide-list"
                class="metric-guide-list"
            >
            </div>

        </div>

    `;


    document.body.appendChild(
        overlay
    );


    renderMetricDefinitions();

    setupMetricGuideEvents();

    decorateOverviewMetrics();

}


// ==================================================
// Render Definitions
// ==================================================

function renderMetricDefinitions(
    searchTerm = ""
) {

    const list =
        document.getElementById(
            "metric-guide-list"
        );


    if (!list) {
        return;
    }


    const query =
        searchTerm
        .trim()
        .toLowerCase();


    const filtered =
        metricGuideDefinitions.filter(
            metric => {

                const searchable =

                    `${metric.name}
                     ${metric.category}
                     ${metric.short}
                     ${metric.why}
                     ${metric.note}`
                    .toLowerCase();


                return (
                    query === ""
                    ||
                    searchable.includes(
                        query
                    )
                );

            }
        );


    const categories =
        [...new Set(
            filtered.map(
                metric =>
                    metric.category
            )
        )];


    list.innerHTML =
        "";


    categories.forEach(
        category => {

            const section =
                document.createElement(
                    "section"
                );


            section.className =
                "metric-guide-category";


            section.innerHTML = `

                <div class="metric-guide-category-title">
                    ${category}
                </div>

                <div class="metric-guide-category-items">
                </div>

            `;


            const items =
                section.querySelector(
                    ".metric-guide-category-items"
                );


            filtered
                .filter(
                    metric =>
                        metric.category === category
                )
                .forEach(
                    metric => {

                        const card =
                            document.createElement(
                                "article"
                            );


                        card.className =
                            "metric-definition";


                        card.dataset.metricKey =
                            metric.key;


                        card.innerHTML = `

                            <div class="metric-definition-name">
                                ${metric.name}
                            </div>

                            <div class="metric-definition-short">
                                ${metric.short}
                            </div>

                            <div class="metric-definition-why">

                                <strong>
                                    Why it matters
                                </strong>

                                ${metric.why}

                            </div>

                            <div class="metric-definition-note">

                                <strong>
                                    Important context
                                </strong>

                                ${metric.note}

                            </div>

                        `;


                        items.appendChild(
                            card
                        );

                    }
                );


            list.appendChild(
                section
            );

        }
    );


    if (
        filtered.length === 0
    ) {

        list.innerHTML = `

            <div class="metric-guide-empty">
                No matching metric found.
            </div>

        `;

    }

}


// ==================================================
// Open / Close
// ==================================================

function openMetricGuide(
    metricKey = null
) {

    const overlay =
        document.getElementById(
            "metric-guide-modal"
        );


    if (!overlay) {
        return;
    }


    overlay.classList.add(
        "open"
    );


    document.body.classList.add(
        "metric-guide-open"
    );


    const search =
        document.getElementById(
            "metric-guide-search"
        );


    if (
        search
        &&
        !metricKey
    ) {

        setTimeout(
            () =>
                search.focus(),
            60
        );

    }


    if (
        metricKey
    ) {

        const metric =
            metricGuideDefinitions.find(
                item =>
                    item.key === metricKey
            );


        if (
            metric
            &&
            search
        ) {

            search.value =
                metric.name;


            renderMetricDefinitions(
                metric.name
            );


            setTimeout(
                () => {

                    const card =
                        document.querySelector(
                            `[data-metric-key="${metricKey}"]`
                        );


                    if (card) {

                        card.scrollIntoView(
                            {
                                block: "center"
                            }
                        );


                        card.classList.add(
                            "metric-definition-highlight"
                        );

                    }

                },
                80
            );

        }

    }

}


function closeMetricGuide() {

    const overlay =
        document.getElementById(
            "metric-guide-modal"
        );


    if (!overlay) {
        return;
    }


    overlay.classList.remove(
        "open"
    );


    document.body.classList.remove(
        "metric-guide-open"
    );


    const search =
        document.getElementById(
            "metric-guide-search"
        );


    if (search) {

        search.value =
            "";

        renderMetricDefinitions();

    }

}


// Make function available globally for inline metric buttons.
window.openMetricGuide =
    openMetricGuide;


// ==================================================
// Events
// ==================================================

function setupMetricGuideEvents() {

    const launcher =
        document.getElementById(
            "metric-guide-launcher"
        );


    const close =
        document.getElementById(
            "metric-guide-close"
        );


    const overlay =
        document.getElementById(
            "metric-guide-modal"
        );


    const search =
        document.getElementById(
            "metric-guide-search"
        );


    if (launcher) {

        launcher.addEventListener(
            "click",
            () =>
                openMetricGuide()
        );

    }


    if (close) {

        close.addEventListener(
            "click",
            closeMetricGuide
        );

    }


    if (overlay) {

        overlay.addEventListener(
            "click",
            event => {

                if (
                    event.target === overlay
                ) {

                    closeMetricGuide();

                }

            }
        );

    }


    if (search) {

        search.addEventListener(
            "input",
            event => {

                renderMetricDefinitions(
                    event.target.value
                );

            }
        );

    }


    document.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Escape"
            ) {

                closeMetricGuide();

            }

        }
    );

}


// ==================================================
// Add Info Buttons to Overview Metrics
// ==================================================

function decorateMetricLabel(
    valueId,
    newLabel,
    metricKey
) {

    const value =
        document.getElementById(
            valueId
        );


    if (!value) {
        return;
    }


    const card =
        value.closest(
            ".overview-kpi"
        );


    if (!card) {
        return;
    }


    const label =
        card.querySelector(
            "span"
        );


    if (!label) {
        return;
    }


    label.innerHTML =
        "";


    const text =
        document.createElement(
            "span"
        );


    text.textContent =
        newLabel;


    const button =
        document.createElement(
            "button"
        );


    button.type =
        "button";


    button.className =
        "metric-inline-info";


    button.textContent =
        "i";


    button.title =
        `Explain ${newLabel}`;


    button.addEventListener(
        "click",
        event => {

            event.stopPropagation();

            openMetricGuide(
                metricKey
            );

        }
    );


    label.appendChild(
        text
    );


    label.appendChild(
        button
    );

}


function decorateOverviewMetrics() {

    decorateMetricLabel(

        "overview-whiff-delta",

        "Whiff Rate",

        "whiff_rate"

    );


    decorateMetricLabel(

        "overview-hard-hit-delta",

        "Hard-Hit Rate",

        "hard_hit_rate"

    );


    decorateMetricLabel(

        "overview-ev-delta",

        "Average Exit Velocity",

        "exit_velocity"

    );


    decorateMetricLabel(

        "overview-rv-delta",

        "Pitch Value (RV/100)",

        "pitch_value"

    );

}


// ==================================================
// Initialize
// ==================================================

loadMetricGuideStyles();

buildMetricGuide();
