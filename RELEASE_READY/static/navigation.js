// ==================================================
// Pitcher Research Lab
// Application Navigation
// ==================================================


const viewInformation = {

    overview: {

        label:
            "OVERVIEW",

        description:
            "Executive research summary"

    },

    arsenal: {

        label:
            "ARSENAL",

        description:
            "Pitch characteristics and arsenal evolution"

    },

    changes: {

        label:
            "CHANGE DETECTION",

        description:
            "Statistical departures and transition timing"

    },

    release: {

        label:
            "RELEASE PROFILE",

        description:
            "Delivery and release-point investigation"

    },

    performance: {

        label:
            "PERFORMANCE",

        description:
            "Command, contact quality and pitch results"

    },

    location: {

        label:
            "COMMAND & LOCATION",

        description:
            "Pitch location, hitter response and contact results",

        panel:
            "performance",

        target:
            ".location-lab-v2"

    },

    career: {

        label:
            "CAREER / TIMELINE",

        description:
            "Full-career outing and season context",

        panel:
            "changes",

        target:
            "#career-audit-panel"

    }

};


// ==================================================
// Open View
// ==================================================

function openApplicationView(
    viewName
) {

    const targetInformation =
        viewInformation[
            viewName
        ];


    if (
        !targetInformation
    ) {

        return;

    }


    // --------------------------------------------------
    // Hide all views
    // --------------------------------------------------

    document
        .querySelectorAll(
            "[data-view-panel]"
        )
        .forEach(
            panel => {

                panel.classList.remove(
                    "active"
                );

            }
        );


    // --------------------------------------------------
    // Show requested view
    // --------------------------------------------------

    const panelName =
        targetInformation.panel
        ||
        viewName;


    const targetPanel =
        document.querySelector(
            `[data-view-panel="${panelName}"]`
        );


    if (
        targetPanel
    ) {

        targetPanel.classList.add(
            "active"
        );

    }


    // --------------------------------------------------
    // Sidebar active state
    // --------------------------------------------------

    document
        .querySelectorAll(
            ".nav-item[data-view]"
        )
        .forEach(
            button => {

                button.classList.remove(
                    "active"
                );

            }
        );


    const targetButton =
        document.querySelector(
            `.nav-item[data-view="${viewName}"]`
        );


    if (
        targetButton
    ) {

        targetButton.classList.add(
            "active"
        );

    }


    // --------------------------------------------------
    // Topbar
    // --------------------------------------------------

    const label =
        document.getElementById(
            "active-view-label"
        );


    const description =
        document.getElementById(
            "active-view-description"
        );


    if (
        label
    ) {

        label.textContent =
            targetInformation.label;

    }


    if (
        description
    ) {

        description.textContent =
            targetInformation.description;

    }


    // --------------------------------------------------
    // Save current view
    // --------------------------------------------------

    sessionStorage.setItem(
        "pitcherResearchView",
        viewName
    );


    const targetSection =
        targetInformation.target
            ? document.querySelector(targetInformation.target)
            : null;


    if (targetSection) {

        targetSection.scrollIntoView(
            {
                block: "start",
                behavior: "instant"
            }
        );

        return;

    }


    // --------------------------------------------------
    // Scroll back to top
    // --------------------------------------------------

    window.scrollTo(
        {
            top: 0,
            behavior: "instant"
        }
    );

}


// ==================================================
// Sidebar Buttons
// ==================================================

document
    .querySelectorAll(
        ".nav-item[data-view]"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",
                function () {

                    openApplicationView(
                        button.dataset.view
                    );

                }
            );

        }
    );


// ==================================================
// In-page Navigation
// ==================================================

document
    .querySelectorAll(
        "[data-view-link]"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",
                function () {

                    openApplicationView(
                        button.dataset.viewLink
                    );

                }
            );

        }
    );


// ==================================================
// Restore Last View
// ==================================================

const savedView =
    sessionStorage.getItem(
        "pitcherResearchView"
    );


if (
    savedView
    &&
    viewInformation[
        savedView
    ]
) {

    openApplicationView(
        savedView
    );

}

else {

    openApplicationView(
        "overview"
    );

}
