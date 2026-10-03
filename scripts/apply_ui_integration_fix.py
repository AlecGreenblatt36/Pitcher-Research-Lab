from pathlib import Path

performance = Path("static/performance.js")
text = performance.read_text(encoding="utf-8")
old = '''      if(controls){

        controls.classList.toggle(
          "context-hidden",
          active!=="arsenal" && active!=="release"
        );

      }
'''
new = '''      if(controls){

        // The primary-pitch selector is a global research control. Keeping it
        // visible prevents a reload or view change from leaving the selected
        // pitch inaccessible and removes a timing-dependent browser failure.
        controls.classList.remove(
          "context-hidden"
        );

      }
'''
if old not in text:
    raise SystemExit("performance.js patch target not found")
performance.write_text(text.replace(old, new), encoding="utf-8")

context = Path("static/pitcher_context.js")
text = context.read_text(encoding="utf-8")
old = '''            if (!Array.isArray(arsenal) || !arsenal.length) {
                const option = document.createElement("option");
                option.value = "";
                option.textContent = "No pitch data available";
                select.appendChild(option);
                return;
            }
'''
new = '''            if (!Array.isArray(arsenal) || !arsenal.length) {
                const option = document.createElement("option");
                option.value = "";
                option.textContent = "No pitch data available";
                select.appendChild(option);
                select.disabled = true;
                return;
            }

            // Neutral state disables these controls. Explicitly re-enable them
            // when a valid pitcher arsenal is loaded.
            select.disabled = false;
'''
if old not in text:
    raise SystemExit("pitcher_context.js patch target not found")
context.write_text(text.replace(old, new), encoding="utf-8")
