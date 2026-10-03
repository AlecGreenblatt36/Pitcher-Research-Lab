from pathlib import Path


performance = Path("static/performance.js")
performance_text = performance.read_text(encoding="utf-8")
old_performance = '''      if(controls){

        // The primary-pitch selector is a global research control. Keeping it
        // visible prevents a reload or view change from leaving the selected
        // pitch inaccessible and removes a timing-dependent browser failure.
        controls.classList.remove(
          "context-hidden"
        );

      }
'''
new_performance = '''      if(
        controls &&
        controls.classList.contains("context-hidden")
      ){

        // Guard the mutation. The observer watches class attributes, so
        // repeatedly removing an absent class can create a self-sustaining
        // MutationObserver loop in Chromium and freeze every navigation click.
        controls.classList.remove(
          "context-hidden"
        );

      }
'''
if old_performance not in performance_text:
    raise SystemExit("performance observer patch target not found")
performance.write_text(
    performance_text.replace(old_performance, new_performance, 1),
    encoding="utf-8",
)

navigation = Path("static/navigation.js")
navigation_text = navigation.read_text(encoding="utf-8")
old_navigation = '''document
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
'''
new_navigation = '''document
    .querySelectorAll(".nav-item[data-view]")
    .forEach(button => {
        button.type = "button";
        button.addEventListener("click", event => {
            event.preventDefault();
            openApplicationView(button.dataset.view);
        });
    });

document
    .querySelectorAll("[data-view-link]")
    .forEach(button => {
        button.type = "button";
        button.addEventListener("click", event => {
            event.preventDefault();
            openApplicationView(button.dataset.viewLink);
        });
    });
'''
if old_navigation not in navigation_text:
    raise SystemExit("navigation patch target not found")
navigation.write_text(
    navigation_text.replace(old_navigation, new_navigation, 1),
    encoding="utf-8",
)
