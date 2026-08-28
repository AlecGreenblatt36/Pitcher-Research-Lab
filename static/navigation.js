// Pitcher Research Lab navigation

const viewInformation = {
    overview: {
        label: "OVERVIEW",
        description: "Executive research summary",
    },
    arsenal: {
        label: "ARSENAL",
        description: "Pitch characteristics and arsenal evolution",
    },
    changes: {
        label: "CHANGE DETECTION",
        description: "Statistical departures and transition timing",
    },
    release: {
        label: "RELEASE PROFILE",
        description: "Delivery and release-point investigation",
    },
    performance: {
        label: "PERFORMANCE",
        description: "Game results, hitter response and underlying process",
    },
    location: {
        label: "COMMAND & LOCATION",
        description: "Pitch location, hitter response and contact results",
    },
    career: {
        label: "CAREER / TIMELINE",
        description: "Full-career outing and season context",
    },
};

function ensureViewPanel(viewName) {
    let panel = document.querySelector(`[data-view-panel="${viewName}"]`);

    if (panel) {
        return panel;
    }

    const main = document.querySelector(".main-content");

    if (!main) {
        return null;
    }

    panel = document.createElement("div");
    panel.className = "app-view";
    panel.dataset.viewPanel = viewName;
    main.appendChild(panel);

    return panel;
}

function prepareStandaloneViews() {
    const locationPanel = ensureViewPanel("location");
    const locationLab = document.querySelector(".location-lab-v2");

    if (
        locationPanel &&
        locationLab &&
        locationLab.parentElement !== locationPanel
    ) {
        locationPanel.appendChild(locationLab);
    }

    const careerPanel = ensureViewPanel("career");

    if (!careerPanel) {
        return;
    }

    const moveCareerAudit = () => {
        const careerAudit = document.getElementById("career-audit-panel");

        if (
            careerAudit &&
            careerAudit.parentElement !== careerPanel
        ) {
            careerPanel.appendChild(careerAudit);
        }
    };

    moveCareerAudit();

    const changesPanel =
        document.querySelector('[data-view-panel="changes"]');

    if (changesPanel) {
        const observer = new MutationObserver(moveCareerAudit);
        observer.observe(changesPanel, {
            childList: true,
            subtree: true,
        });
    }
}

function openApplicationView(viewName) {
    const targetInformation = viewInformation[viewName];

    if (!targetInformation) {
        return;
    }

    document
        .querySelectorAll("[data-view-panel]")
        .forEach(panel => panel.classList.remove("active"));

    const targetPanel =
        document.querySelector(`[data-view-panel="${viewName}"]`);

    if (!targetPanel) {
        return;
    }

    targetPanel.classList.add("active");

    document
        .querySelectorAll(".nav-item[data-view]")
        .forEach(button => button.classList.remove("active"));

    const targetButton =
        document.querySelector(`.nav-item[data-view="${viewName}"]`);

    if (targetButton) {
        targetButton.classList.add("active");
    }

    const label = document.getElementById("active-view-label");
    const description =
        document.getElementById("active-view-description");

    if (label) {
        label.textContent = targetInformation.label;
    }

    if (description) {
        description.textContent = targetInformation.description;
    }

    sessionStorage.setItem("pitcherResearchView", viewName);

    window.scrollTo({
        top: 0,
        behavior: "instant",
    });
}

prepareStandaloneViews();

document
    .querySelectorAll(".nav-item[data-view]")
    .forEach(button => {
        button.addEventListener("click", () => {
            openApplicationView(button.dataset.view);
        });
    });

document
    .querySelectorAll("[data-view-link]")
    .forEach(button => {
        button.addEventListener("click", () => {
            openApplicationView(button.dataset.viewLink);
        });
    });

const savedView = sessionStorage.getItem("pitcherResearchView");

if (savedView && viewInformation[savedView]) {
    openApplicationView(savedView);
} else {
    openApplicationView("overview");
}
