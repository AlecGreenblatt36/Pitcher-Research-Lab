// Pitcher Research Lab — Performance & Outcomes + UI layout fixes
(() => {
  "use strict";

  const pitchNames={FF:"Four-Seam",SI:"Sinker",SL:"Slider",ST:"Sweeper",FS:"Splitter",CH:"Changeup",CU:"Curveball",FC:"Cutter",KC:"Knuckle Curve",SV:"Slurve",FO:"Forkball",SC:"Screwball",KN:"Knuckleball",EP:"Eephus",CS:"Slow Curve"};

  const metrics={
    earned_runs:{label:"Earned Runs",source:"official",key:"earned_runs",d:0},
    k_minus_bb_pct:{label:"K-BB%",source:"official",key:"k_minus_bb_pct",d:1},
    woba_allowed:{label:"wOBA Allowed",source:"process",key:"woba_allowed",d:3},
    xwoba_allowed:{label:"xwOBA Allowed",source:"process",key:"xwoba_allowed",d:3},
    whiff_pct:{label:"Whiff Rate",source:"process",key:"whiff_pct",d:1},
    chase_pct:{label:"Chase Rate",source:"process",key:"chase_pct",d:1},
    hard_hit_pct:{label:"Hard-Hit Rate",source:"process",key:"hard_hit_pct",d:1},
    pitch_value_per_100:{label:"Pitch Value (RV/100)",source:"process",key:"pitch_value_per_100",d:2}
  };

  const explain={
    earned_runs:"Official earned runs allowed. Useful as the box-score result, but noisier than the pitch-level process underneath it.",
    k_minus_bb_pct:"Strikeout rate minus walk rate from the official game line. Higher is better for the pitcher.",
    woba_allowed:"Actual weighted offensive value allowed across plate appearances. Lower is better for the pitcher.",
    xwoba_allowed:"Expected offensive value allowed using Statcast contact quality plus actual non-contact outcomes. Lower is better.",
    whiff_pct:"Whiffs divided by swings. Higher means hitters missed more often when they offered.",
    chase_pct:"Swings at pitches outside the normalized strike zone divided by located pitches outside the zone. Higher is generally better.",
    hard_hit_pct:"Share of tracked balls in play hit 95+ mph. Lower means less hard contact.",
    pitch_value_per_100:"Pitch-level run expectancy scaled to 100 pitches from the pitcher perspective. Positive is better."
  };

  let data=null;
  let selectedGamePk=null;

  const $=id=>document.getElementById(id);

  const num=v=>
    v===null||
    v===undefined||
    v===""||
    !Number.isFinite(Number(v))
      ? null
      : Number(v);

  const fmt=(v,d=1)=>
    num(v)===null
      ? "--"
      : Number(v).toFixed(d);

  const signed=(v,d=2)=>
    num(v)===null
      ? "--"
      : `${Number(v)>0?"+":""}${Number(v).toFixed(d)}`;

  const dateFmt=(v,year=false)=>{
    if(!v)return"--";

    const d=new Date(`${v}T00:00:00`);

    return d.toLocaleDateString(
      "en-US",
      year
        ? {month:"short",day:"numeric",year:"numeric"}
        : {month:"short",day:"numeric"}
    );
  };

  const periodName=p=>
    p==="early"
      ? "Early Period"
      : p==="transition"
        ? "Middle Period"
        : "Later Period";

  const opp=o=>
    `${o?.home_away==="Away"?"@":"vs"} ${o?.opponent??"--"}`;

  const text=(id,value)=>{
    const el=$(id);

    if(el){
      el.textContent=value;
    }
  };

  const targetSeason=()=>
    Number(data?.player?.target_season) ||
    Math.max(...(data?.seasons||[]).map(r=>Number(r.season)).filter(Number.isFinite));

  const currentSeasonSummary=()=>
    data?.seasons?.find(
      r=>Number(r.season)===targetSeason()
    )??null;


  // ==================================================
  // Load Performance CSS
  // ==================================================

  function loadStyles(){

    if($("performance-styles")){
      return;
    }

    const link=
      document.createElement("link");

    link.id=
      "performance-styles";

    link.rel=
      "stylesheet";

    link.href=
      "/static/performance.css";

    document.head.appendChild(
      link
    );
  }


  // ==================================================
  // Global Layout Fixes
  // ==================================================

  function applyUiFixes(){

    [
      "velocity-chart",
      "investigation-chart"
    ].forEach(
      id=>
        $(id)?.setAttribute(
          "preserveAspectRatio",
          "xMidYMid meet"
        )
    );


    const update=()=>{

      const controls=
        document.querySelector(
          ".pitcher-header .header-controls"
        );

      const active=
        document.querySelector(
          ".app-view.active"
        )?.dataset?.viewPanel;


      if(controls){

        controls.classList.toggle(
          "context-hidden",
          active!=="arsenal"
        );

      }

    };


    update();


    document.querySelectorAll(
      ".nav-item[data-view], [data-view-link]"
    )
    .forEach(
      button=>
        button.addEventListener(
          "click",
          ()=>
            setTimeout(
              update,
              0
            )
        )
    );


    const main=
      document.querySelector(
        ".main-content"
      );


    if(main){

      new MutationObserver(
        update
      )
      .observe(
        main,
        {
          subtree:true,
          attributes:true,
          attributeFilter:[
            "class"
          ]
        }
      );

    }

  }


  // ==================================================
  // Build New Performance Page
  // ==================================================

  function buildShell(){

    const panel=
      document.querySelector(
        '[data-view-panel="performance"]'
      );


    if(!panel){
      return null;
    }


    /*
      Important:

      Keep the EXISTING location lab DOM node.

      location.js already loaded and attached its
      event listeners to this node.

      Moving the node keeps those listeners alive.
    */

    const location=
      panel.querySelector(
        ".location-lab-v2"
      );


    if(location){
      location.remove();
    }


    panel.innerHTML=`

      <section
        class="view-page-header performance-page-header"
      >

        <div>

          <div class="eyebrow">
            GAME RESULTS + UNDERLYING PROCESS
          </div>

          <h2>
            Performance & Outcomes
          </h2>

          <p>
            Compare the box-score result with the quality
            of the pitches underneath it, then test whether
            those patterns moved across the selected research window.
          </p>

        </div>


        <div
          class="performance-data-tag"
          id="performance-data-tag"
        >
          Loading pitcher outings...
        </div>

      </section>



      <section
        class="performance-latest"
        id="performance-latest"
      >

        <div class="performance-latest-main">

          <div class="eyebrow">
            LATEST OUTING
          </div>


          <div class="performance-latest-title-row">

            <div>

              <h3 id="performance-latest-title">
                Loading...
              </h3>

              <div
                class="performance-latest-meta"
                id="performance-latest-meta"
              >
                --
              </div>

            </div>


            <button
              class="performance-open-latest"
              id="performance-open-latest"
            >
              View outing details
            </button>

          </div>


          <div class="performance-boxscore-grid">

            ${["IP","ER","K","BB"]
              .map(
                x=>`
                  <div class="performance-boxscore-stat">

                    <span>
                      ${x}
                    </span>

                    <strong id="latest-${x.toLowerCase()}">
                      --
                    </strong>

                  </div>
                `
              )
              .join("")
            }

          </div>

        </div>



        <div class="performance-latest-process">

          <div class="performance-section-kicker">
            UNDER THE SURFACE
          </div>


          <div class="performance-process-grid">

            ${processCard(
              "Whiff Rate",
              "latest-whiff",
              "Misses / swings"
            )}

            ${processCard(
              "Chase Rate",
              "latest-chase",
              "Swings outside zone"
            )}

            ${processCard(
              "Hard-Hit Rate",
              "latest-hard-hit",
              "95+ mph contact"
            )}

            ${processCard(
              "xwOBA Allowed",
              "latest-xwoba",
              "Expected PA quality"
            )}

            ${processCard(
              "Pitch Value",
              "latest-rv",
              "Runs / 100 pitches"
            )}

          </div>


          <div
            class="performance-interpretation"
            id="performance-latest-interpretation"
          >
            Loading outing context...
          </div>

        </div>

      </section>



      <section class="performance-period-section">

        <div class="performance-section-heading">

          <div>

            <div class="eyebrow">
              TARGET-SEASON PERIOD COMPARISON
            </div>

            <h3>
              How did performance move across the selected periods?
            </h3>

          </div>


          <div class="performance-section-note">
            Official outcomes + Statcast process metrics
          </div>

        </div>


        <div
          class="performance-period-grid"
          id="performance-period-grid"
        >
        </div>

      </section>



      <section class="panel performance-timeline-panel">

        <div class="panel-header performance-timeline-header">

          <div>

            <div class="eyebrow">
              OUTING-BY-OUTING
            </div>

            <h3>
              Performance Timeline
            </h3>

          </div>


          <div class="performance-chart-control">

            <label for="performance-metric">
              Metric
            </label>


            <select id="performance-metric">

              <optgroup label="Baseball Outcomes">

                <option value="earned_runs">
                  Earned Runs
                </option>

                <option value="k_minus_bb_pct">
                  K-BB%
                </option>

                <option value="woba_allowed">
                  wOBA Allowed
                </option>

                <option
                  value="xwoba_allowed"
                  selected
                >
                  xwOBA Allowed
                </option>

              </optgroup>


              <optgroup label="Underlying Process">

                <option value="whiff_pct">
                  Whiff Rate
                </option>

                <option value="chase_pct">
                  Chase Rate
                </option>

                <option value="hard_hit_pct">
                  Hard-Hit Rate
                </option>

                <option value="pitch_value_per_100">
                  Pitch Value (RV/100)
                </option>

              </optgroup>

            </select>

          </div>

        </div>


        <div
          class="performance-chart-explainer"
          id="performance-chart-explainer"
        >
        </div>


        <div class="performance-chart-wrap">

          <svg
            id="performance-chart"
            viewBox="0 0 1000 340"
            preserveAspectRatio="xMidYMid meet"
          >
          </svg>

        </div>

      </section>



      <section
        class="performance-detail-section"
        id="performance-detail-section"
      >

        <div class="performance-section-heading">

          <div>

            <div class="eyebrow">
              OUTING DETAIL
            </div>

            <h3 id="performance-detail-title">
              Select an outing
            </h3>

          </div>


          <div
            class="performance-section-note"
            id="performance-detail-note"
          >
            Click any row or chart point
          </div>

        </div>


        <div class="performance-detail-grid">

          <div class="performance-detail-card">

            <div class="performance-detail-card-title">
              Official Line
            </div>

            <div
              class="performance-detail-stat-grid"
              id="performance-detail-official"
            >
            </div>

          </div>


          <div class="performance-detail-card">

            <div class="performance-detail-card-title">
              Underlying Process
            </div>

            <div
              class="performance-detail-stat-grid"
              id="performance-detail-process"
            >
            </div>

          </div>


          <div
            class="performance-detail-card performance-arsenal-card"
          >

            <div class="performance-detail-card-title">
              Arsenal That Day
            </div>

            <div
              id="performance-detail-arsenal"
              class="performance-arsenal-list"
            >
            </div>

          </div>

        </div>

      </section>



      <section class="panel performance-log-panel">

        <div class="panel-header">

          <div>

            <div class="eyebrow">
              TARGET-SEASON OUTING LOG
            </div>

            <h3>
              Game Results and Process
            </h3>

          </div>


          <div class="panel-note">
            Click an outing to inspect it
          </div>

        </div>


        <div class="table-wrapper performance-table-wrap">

          <table class="performance-table">

            <thead>

              <tr>

                <th>Date</th>
                <th>Opp</th>

                <th>IP</th>
                <th>ER</th>

                <th>K</th>
                <th>BB</th>

                <th>Whiff</th>
                <th>Chase</th>

                <th>Hard Hit</th>

                <th>xwOBA</th>

                <th>Pitch Value</th>

              </tr>

            </thead>


            <tbody id="performance-outing-body">
            </tbody>

          </table>

        </div>

      </section>

    `;


    /*
      Put our existing Command & Location Lab underneath
      the actual game-performance analysis.
    */

    if(location){

      const intro=
        document.createElement(
          "div"
        );


      intro.className=
        "performance-deeper-analysis";


      intro.innerHTML=`

        <div class="eyebrow">
          DEEPER ANALYSIS
        </div>

        <h3>
          Explain the performance shift
        </h3>

        <p>
          Once the game-level performance pattern is
          established, use the command and location views
          below to investigate where those results may
          be coming from.
        </p>

      `;


      panel.append(
        intro,
        location
      );

    }


    return panel;
  }


  function processCard(
    label,
    id,
    note
  ){

    return `

      <div class="performance-process-stat">

        <span>
          ${label}
        </span>

        <strong id="${id}">
          --
        </strong>

        <small>
          ${note}
        </small>

      </div>

    `;

  }


  // ==================================================
  // Latest Outing Interpretation
  // ==================================================

  function latestInterpretation(o){

    const p=
      o?.process??{};

    const s=
      currentSeasonSummary()?.process??{};

    const notes=[];


    const w=
      num(
        p.whiff_pct
      );

    const sw=
      num(
        s.whiff_pct
      );


    const x=
      num(
        p.xwoba_allowed
      );

    const sx=
      num(
        s.xwoba_allowed
      );


    const h=
      num(
        p.hard_hit_pct
      );

    const sh=
      num(
        s.hard_hit_pct
      );


    if(
      w!==null
      &&
      sw!==null
      &&
      Math.abs(
        w-sw
      )>=3
    ){

      notes.push(

        `Whiff rate was ${Math.abs(
          w-sw
        ).toFixed(
          1
        )} percentage points ${w>sw?"above":"below"} his target-season average.`

      );

    }


    if(
      x!==null
      &&
      sx!==null
      &&
      Math.abs(
        x-sx
      )>=0.020
    ){

      notes.push(

        `xwOBA allowed was ${Math.abs(
          x-sx
        ).toFixed(
          3
        )} ${x<sx?"lower":"higher"} than his target-season average.`

      );

    }


    if(
      !notes.length
      &&
      h!==null
      &&
      sh!==null
    ){

      notes.push(

        `Hard-hit rate was ${Math.abs(
          h-sh
        ).toFixed(
          1
        )} percentage points ${h<sh?"lower":"higher"} than his target-season average.`

      );

    }


    return (
      notes.length
        ? notes
        : [
            "The underlying process metrics were close to his target-season averages."
          ]
    )
    .slice(
      0,
      2
    )
    .join(
      " "
    );
  }


  // ==================================================
  // Latest Outing
  // ==================================================

  function renderLatest(){

    const o=
      data?.outings?.at(
        -1
      );


    if(!o){
      return;
    }


    const p=
      o.process??{};

    const g=
      o.official??{};


    text(
      "performance-latest-title",
      `${dateFmt(
        o.game_date,
        true
      )} ${opp(
        o
      )}`
    );


    text(
      "performance-latest-meta",
      `${periodName(
        o.period
      )} • ${p.pitches??g.pitches??"--"} pitches`
    );


    text(
      "latest-ip",
      g.innings_pitched??"--"
    );

    text(
      "latest-er",
      g.earned_runs??"--"
    );

    text(
      "latest-k",
      g.strikeouts??p.strikeouts_statcast??"--"
    );

    text(
      "latest-bb",
      g.walks??p.walks_statcast??"--"
    );


    text(
      "latest-whiff",
      `${fmt(
        p.whiff_pct
      )}%`
    );


    text(
      "latest-chase",
      `${fmt(
        p.chase_pct
      )}%`
    );


    text(
      "latest-hard-hit",
      `${fmt(
        p.hard_hit_pct
      )}%`
    );


    text(
      "latest-xwoba",
      fmt(
        p.xwoba_allowed,
        3
      )
    );


    text(
      "latest-rv",
      signed(
        p.pitch_value_per_100
      )
    );


    text(
      "performance-latest-interpretation",
      latestInterpretation(
        o
      )
    );


    $("performance-open-latest").onclick=
      ()=>
        selectOuting(
          o.game_pk,
          true
        );
  }


  // ==================================================
  // Period Comparison
  // ==================================================

  function renderPeriods(){

    const c=
      $("performance-period-grid");


    if(!c){
      return;
    }


    c.innerHTML="";


    [
      "early",
      "transition",
      "post"
    ]
    .forEach(
      name=>{

        const r=
          data?.periods?.find(
            x=>
              x.period===name
          );


        if(!r){
          return;
        }


        const g=
          r.official??{};

        const p=
          r.process??{};


        const el=
          document.createElement(
            "article"
          );


        el.className=
          `performance-period-card period-${name}`;


        el.innerHTML=`

          <div class="performance-period-top">

            <div>

              <span>
                ${periodName(name)}
              </span>

              <strong>
                ${r.outing_count??0} outings
              </strong>

            </div>


            <div class="performance-period-era">

              <small>
                ERA
              </small>

              <strong>
                ${fmt(
                  g.era,
                  2
                )}
              </strong>

            </div>

          </div>


          <div class="performance-period-metrics">

            ${periodMetric(
              "K-BB%",
              `${fmt(
                g.k_minus_bb_pct
              )}%`
            )}

            ${periodMetric(
              "Whiff",
              `${fmt(
                p.whiff_pct
              )}%`
            )}

            ${periodMetric(
              "Chase",
              `${fmt(
                p.chase_pct
              )}%`
            )}

            ${periodMetric(
              "Hard Hit",
              `${fmt(
                p.hard_hit_pct
              )}%`
            )}

            ${periodMetric(
              "xwOBA",
              fmt(
                p.xwoba_allowed,
                3
              )
            )}

            ${periodMetric(
              "Pitch Value",
              signed(
                p.pitch_value_per_100
              )
            )}

          </div>

        `;


        c.appendChild(
          el
        );

      }
    );
  }


  const periodMetric=
    (
      label,
      value
    )=>
      `
        <div>

          <span>
            ${label}
          </span>

          <strong>
            ${value}
          </strong>

        </div>
      `;


  // ==================================================
  // Performance Timeline
  // ==================================================

  function metricValue(
    o,
    m
  ){

    return num(
      o?.[m.source]?.[m.key]
    );

  }


  function svgText(
    svg,
    x,
    y,
    cls,
    value,
    anchor
  ){

    const e=
      document.createElementNS(
        "http://www.w3.org/2000/svg",
        "text"
      );


    e.setAttribute(
      "x",
      x
    );

    e.setAttribute(
      "y",
      y
    );

    e.setAttribute(
      "class",
      cls
    );


    if(anchor){

      e.setAttribute(
        "text-anchor",
        anchor
      );

    }


    e.textContent=
      value;


    svg.appendChild(
      e
    );
  }


  function drawChart(){

    const svg=
      $("performance-chart");

    const sel=
      $("performance-metric");


    if(
      !svg
      ||
      !sel
    ){
      return;
    }


    const key=
      sel.value;

    const m=
      metrics[key];


    svg.innerHTML="";


    text(
      "performance-chart-explainer",
      explain[key]
    );


    const rows=
      (
        data?.outings??[]
      )
      .map(
        o=>({
          o,
          v:metricValue(
            o,
            m
          )
        })
      )
      .filter(
        r=>
          r.v!==null
      );


    if(!rows.length){

      svgText(
        svg,
        500,
        170,
        "performance-chart-empty",
        "No usable data for this metric.",
        "middle"
      );

      return;
    }


    const W=1000;
    const H=340;

    const M={
      l:70,
      r:30,
      t:34,
      b:54
    };


    const vals=
      rows.map(
        r=>r.v
      );


    const times=
      rows.map(
        r=>
          new Date(
            `${r.o.game_date}T00:00:00`
          ).getTime()
      );


    let lo=
      Math.min(
        ...vals
      );

    let hi=
      Math.max(
        ...vals
      );


    let range=
      hi-lo
      ||
      Math.max(
        Math.abs(
          hi
        ),
        1
      );


    const pad=
      range*0.14;


    lo-=pad;
    hi+=pad;


    if(
      key==="earned_runs"
    ){

      lo=
        Math.min(
          0,
          lo
        );

    }


    const t0=
      Math.min(
        ...times
      );

    const t1=
      Math.max(
        ...times
      );


    const xp=
      t=>
        t0===t1
          ? W/2
          : M.l+
            (
              (
                t-t0
              )
              /
              (
                t1-t0
              )
            )
            *
            (
              W-M.l-M.r
            );


    const yp=
      v=>
        M.t+
        (
          (
            hi-v
          )
          /
          (
            hi-lo
          )
        )
        *
        (
          H-M.t-M.b
        );


    // Research-window shading

    const start=
      new Date(
        `${data.transition_window.start}T00:00:00`
      ).getTime();


    const end=
      new Date(
        `${data.transition_window.end}T00:00:00`
      ).getTime();


    if(
      end>=t0
      &&
      start<=t1
    ){

      const x1=
        xp(
          Math.max(
            start,
            t0
          )
        );

      const x2=
        xp(
          Math.min(
            end,
            t1
          )
        );


      const rect=
        document.createElementNS(
          "http://www.w3.org/2000/svg",
          "rect"
        );


      rect.setAttribute(
        "x",
        x1
      );

      rect.setAttribute(
        "y",
        M.t
      );

      rect.setAttribute(
        "width",
        Math.max(
          x2-x1,
          1
        )
      );

      rect.setAttribute(
        "height",
        H-M.t-M.b
      );

      rect.setAttribute(
        "class",
        "performance-transition-shade"
      );


      svg.appendChild(
        rect
      );


      svgText(
        svg,
        (
          x1+x2
        )/2,
        M.t+15,
        "performance-transition-label",
        "Research window",
        "middle"
      );

    }


    // Y grid

    for(
      let i=0;
      i<=5;
      i++
    ){

      const v=
        lo+
        (
          hi-lo
        )
        *
        i
        /
        5;


      const y=
        yp(
          v
        );


      const line=
        document.createElementNS(
          "http://www.w3.org/2000/svg",
          "line"
        );


      line.setAttribute(
        "x1",
        M.l
      );

      line.setAttribute(
        "x2",
        W-M.r
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
        "performance-chart-grid"
      );


      svg.appendChild(
        line
      );


      svgText(
        svg,
        M.l-12,
        y+4,
        "performance-chart-axis",
        m.d===0
          ? `${Math.round(v)}`
          : v.toFixed(m.d),
        "end"
      );

    }


    // Dates

    for(
      let i=0;
      i<=5;
      i++
    ){

      const t=
        t0+
        (
          t1-t0
        )
        *
        i
        /
        5;


      svgText(
        svg,
        xp(t),
        H-18,
        "performance-chart-axis",
        new Date(
          t
        ).toLocaleDateString(
          "en-US",
          {
            month:"short",
            day:"numeric"
          }
        ),
        "middle"
      );

    }


    const pts=
      rows.map(
        r=>({
          ...r,

          x:
            xp(
              new Date(
                `${r.o.game_date}T00:00:00`
              ).getTime()
            ),

          y:
            yp(
              r.v
            )
        })
      );


    const path=
      document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path"
      );


    path.setAttribute(
      "d",
      pts
        .map(
          (
            p,
            i
          )=>
            `${i?"L":"M"} ${p.x} ${p.y}`
        )
        .join(
          " "
        )
    );


    path.setAttribute(
      "class",
      "performance-chart-line"
    );


    svg.appendChild(
      path
    );


    pts.forEach(
      p=>{

        const c=
          document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle"
          );


        c.setAttribute(
          "cx",
          p.x
        );

        c.setAttribute(
          "cy",
          p.y
        );


        c.setAttribute(
          "r",
          Number(
            p.o.game_pk
          )
          ===
          Number(
            selectedGamePk
          )
            ? 6
            : 4.5
        );


        c.setAttribute(
          "class",
          `performance-chart-point ${
            Number(
              p.o.game_pk
            )
            ===
            Number(
              selectedGamePk
            )
              ? "selected"
              : ""
          }`
        );


        const title=
          document.createElementNS(
            "http://www.w3.org/2000/svg",
            "title"
          );


        title.textContent=
          `${dateFmt(
            p.o.game_date,
            true
          )} ${opp(
            p.o
          )}\n${m.label}: ${
            m.d===0
              ? Math.round(
                  p.v
                )
              : p.v.toFixed(
                  m.d
                )
          }`;


        c.appendChild(
          title
        );


        c.addEventListener(
          "click",
          ()=>
            selectOuting(
              p.o.game_pk,
              true
            )
        );


        svg.appendChild(
          c
        );

      }
    );
  }


  // ==================================================
  // Outing Table
  // ==================================================

  function renderTable(){

    const b=
      $("performance-outing-body");


    if(!b){
      return;
    }


    b.innerHTML="";


    [
      ...(
        data?.outings??[]
      )
    ]
    .reverse()
    .forEach(
      o=>{

        const g=
          o.official??{};

        const p=
          o.process??{};


        const tr=
          document.createElement(
            "tr"
          );


        if(
          Number(
            o.game_pk
          )
          ===
          Number(
            selectedGamePk
          )
        ){

          tr.className=
            "selected";

        }


        tr.innerHTML=`

          <td>
            ${dateFmt(o.game_date)}
          </td>

          <td>
            ${opp(o)}
          </td>

          <td>
            ${g.innings_pitched??"--"}
          </td>

          <td>
            ${g.earned_runs??"--"}
          </td>

          <td>
            ${g.strikeouts??p.strikeouts_statcast??"--"}
          </td>

          <td>
            ${g.walks??p.walks_statcast??"--"}
          </td>

          <td>
            ${fmt(p.whiff_pct)}%
          </td>

          <td>
            ${fmt(p.chase_pct)}%
          </td>

          <td>
            ${fmt(p.hard_hit_pct)}%
          </td>

          <td>
            ${fmt(
              p.xwoba_allowed,
              3
            )}
          </td>

          <td>
            ${signed(
              p.pitch_value_per_100
            )}
          </td>

        `;


        tr.addEventListener(
          "click",
          ()=>
            selectOuting(
              o.game_pk,
              true
            )
        );


        b.appendChild(
          tr
        );

      }
    );
  }


  // ==================================================
  // Selected Outing
  // ==================================================

  const detail=
    (
      label,
      value
    )=>
      `

        <div class="performance-detail-stat">

          <span>
            ${label}
          </span>

          <strong>
            ${value}
          </strong>

        </div>

      `;


  function renderDetail(o){

    const g=
      o.official??{};

    const p=
      o.process??{};


    text(
      "performance-detail-title",
      `${dateFmt(
        o.game_date,
        true
      )} ${opp(o)}`
    );


    text(
      "performance-detail-note",
      `${periodName(
        o.period
      )} • ${p.pitches??g.pitches??"--"} pitches`
    );


    $("performance-detail-official").innerHTML=

      [
        [
          "IP",
          g.innings_pitched??"--"
        ],
        [
          "ER",
          g.earned_runs??"--"
        ],
        [
          "H",
          g.hits??"--"
        ],
        [
          "K",
          g.strikeouts??p.strikeouts_statcast??"--"
        ],
        [
          "BB",
          g.walks??p.walks_statcast??"--"
        ],
        [
          "HR",
          g.home_runs??p.home_runs_statcast??"--"
        ]
      ]
      .map(
        x=>
          detail(
            ...x
          )
      )
      .join("");


    $("performance-detail-process").innerHTML=

      [
        [
          "Whiff",
          `${fmt(
            p.whiff_pct
          )}%`
        ],
        [
          "Chase",
          `${fmt(
            p.chase_pct
          )}%`
        ],
        [
          "Zone",
          `${fmt(
            p.zone_pct
          )}%`
        ],
        [
          "Hard Hit",
          `${fmt(
            p.hard_hit_pct
          )}%`
        ],
        [
          "xwOBA",
          fmt(
            p.xwoba_allowed,
            3
          )
        ],
        [
          "Pitch Value",
          signed(
            p.pitch_value_per_100
          )
        ]
      ]
      .map(
        x=>
          detail(
            ...x
          )
      )
      .join("");


    const usage=
      (
        data?.pitch_usage??[]
      )
      .filter(
        r=>
          Number(
            r.game_pk
          )
          ===
          Number(
            o.game_pk
          )
      )
      .sort(
        (
          a,
          b
        )=>
          Number(
            b.usage_pct
          )
          -
          Number(
            a.usage_pct
          )
      );


    $("performance-detail-arsenal").innerHTML=

      usage.length

        ?

        usage.map(
          r=>`

            <div class="performance-arsenal-row">

              <div class="performance-arsenal-label">

                <strong>
                  ${pitchNames[r.pitch_type]??r.pitch_type}
                </strong>

                <span>
                  ${r.pitch_count} pitches
                </span>

              </div>


              <div class="performance-arsenal-bar-track">

                <div
                  class="performance-arsenal-bar"
                  style="width:${Math.min(
                    Number(
                      r.usage_pct
                    ),
                    100
                  )}%"
                >
                </div>

              </div>


              <div class="performance-arsenal-value">
                ${fmt(
                  r.usage_pct
                )}%
              </div>

            </div>

          `
        )
        .join("")

        :

        `
          <div class="performance-empty">
            No pitch-usage rows available.
          </div>
        `;
  }


  function selectOuting(
    pk,
    scroll=false
  ){

    const o=
      data?.outings?.find(
        r=>
          Number(
            r.game_pk
          )
          ===
          Number(
            pk
          )
      );


    if(!o){
      return;
    }


    selectedGamePk=
      o.game_pk;


    renderDetail(
      o
    );

    renderTable();

    drawChart();


    if(scroll){

      $("performance-detail-section")
        ?.scrollIntoView(
          {
            behavior:"smooth",
            block:"start"
          }
        );

    }
  }


  // ==================================================
  // Data Freshness
  // ==================================================

  function updateStatus(){

    const latest=
      data?.data_status?.latest_game_date;

    const count=
      data?.data_status?.official_outings_cached;


    text(
      "performance-data-tag",

      latest

        ?

        `Data through ${dateFmt(
          latest,
          true
        )} • ${count??0} official outings cached`

        :

        "Performance data loaded"
    );


    const status=
      document.querySelector(
        ".topbar .status"
      );


    if(
      status
      &&
      latest
    ){

      status.textContent=
        `Data through ${dateFmt(
          latest
        )}`;


      status.title=
        "Latest selected-pitcher outing currently stored in Pitcher Research Lab";

    }
  }


  // ==================================================
  // Initialize
  // ==================================================

  async function init(){
    if (window.pitcherResearchLab?.ready) {
      await window.pitcherResearchLab.ready;
    }
    if (!window.pitcherResearchLab?.pitcherId) return;

    loadStyles();

    applyUiFixes();


    if(!buildShell()){
      return;
    }


    try{

      const changeResponse =
        await fetch(
          window.pitcherResearchLab.apiUrl("changes")
        );

      const changes = changeResponse.ok
        ? await changeResponse.json()
        : [];

      const windowRange =
        window.pitcherResearchLab.researchWindow(changes);

      const res=
        await fetch(
          window.pitcherResearchLab.apiUrl("performance", {
            start: windowRange.start,
            end: windowRange.end,
          })
        );


      if(!res.ok){

        throw new Error(
          `Performance API ${res.status}`
        );

      }


      data=
        await res.json();


      updateStatus();

      renderLatest();

      renderPeriods();


      const latest=
        data?.outings?.at(
          -1
        );


      if(latest){

        selectedGamePk=
          latest.game_pk;

        renderDetail(
          latest
        );

      }


      renderTable();

      drawChart();


      $("performance-metric")
        ?.addEventListener(
          "change",
          drawChart
        );

    }

    catch(err){

      console.error(
        "Performance & Outcomes error:",
        err
      );


      text(
        "performance-data-tag",
        "Performance data could not be loaded"
      );


      const p=
        $("performance-latest");


      if(p){

        p.innerHTML=`

          <div class="performance-error">
            Performance & Outcomes could not load.
            Check the Flask terminal and browser console.
          </div>

        `;

      }

    }
  }


  init();

})();
