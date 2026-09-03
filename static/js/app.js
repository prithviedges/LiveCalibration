/* ==========================================================================
   Calibration Assistant — tablet client
   No frameworks. Connects to /ws, translates the enum instruction into a
   readable label + pictogram, and updates the DOM in place with smooth
   CSS-driven transitions. Auto-reconnects on drop (Wi-Fi is never perfectly
   reliable in a control room).
   ========================================================================== */

(function () {
  "use strict";

  const RING_RADIUS = 60;
  const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
  const GRID_ROWS = 5;
  const GRID_COLS = 8;

  // Enum value -> readable label. Falls back to a generic humanizer for
  // any value not listed here, so an engine update never renders blank.
  const LABELS = {
    MOVE_LEFT: "MOVE LEFT",
    MOVE_RIGHT: "MOVE RIGHT",
    MOVE_UP: "MOVE UP",
    MOVE_DOWN: "MOVE DOWN",
    MOVE_TOP_LEFT: "TOP LEFT",
    MOVE_TOP_RIGHT: "TOP RIGHT",
    MOVE_BOTTOM_LEFT: "BOTTOM LEFT",
    MOVE_BOTTOM_RIGHT: "BOTTOM RIGHT",
    MOVE_CLOSER: "MOVE CLOSER",
    MOVE_FARTHER: "MOVE FARTHER",
    TILT_LEFT: "TILT LEFT",
    TILT_RIGHT: "TILT RIGHT",
    TILT_FORWARD: "TILT FORWARD",
    TILT_BACK: "TILT BACK",
    ROTATE_CLOCKWISE: "ROTATE CW",
    ROTATE_COUNTER_CLOCKWISE: "ROTATE CCW",
    HOLD_POSITION: "HOLD POSITION",
    CALIBRATION_COMPLETE: "COMPLETE",
  };

  const STATUS_WORD = {
    move: "MOVE CAMERA",
    almost: "ALMOST THERE",
    hold: "HOLD STILL",
    complete: "CALIBRATED",
  };

  function humanize(value) {
    if (!value) return "\u2014";
    return String(value).replace(/_/g, " ");
  }

  // ── pictograms (inline SVG, no external assets — works fully offline) ──
  function arrowIcon(rotationDeg) {
    return (
      '<svg viewBox="0 0 100 100" style="transform:rotate(' + rotationDeg + 'deg)">' +
      '<path d="M50 8 L90 58 L66 58 L66 92 L34 92 L34 58 L10 58 Z"/>' +
      "</svg>"
    );
  }

  function zoomIcon(inward) {
    const chevrons = [0, 90, 180, 270]
      .map(function (deg) {
        const d = inward ? "M50 30 L65 45 L35 45 Z" : "M50 20 L65 35 L35 35 Z";
        return '<path d="' + d + '" transform="rotate(' + deg + ' 50 50)"/>';
      })
      .join("");
    return '<svg viewBox="0 0 100 100">' + chevrons + "</svg>";
  }

  function rotateIcon(counterClockwise) {
    const scaleX = counterClockwise ? -1 : 1;
    return (
      '<svg viewBox="0 0 100 100" style="transform:scaleX(' + scaleX + ')">' +
      '<path d="M50 14 A36 36 0 1 1 17 50" fill="none" stroke-width="10" stroke-linecap="round"/>' +
      '<path d="M50 0 L66 16 L50 32 Z"/>' +
      "</svg>"
    );
  }

  function tiltIcon(axis) {
    let arrow;
    switch (axis) {
      case "left":
        arrow = '<path d="M30 30 A30 30 0 0 0 30 70" fill="none" stroke-width="8"/><path d="M20 68 L30 78 L38 66 Z"/>';
        break;
      case "right":
        arrow = '<path d="M70 30 A30 30 0 0 1 70 70" fill="none" stroke-width="8"/><path d="M80 68 L70 78 L62 66 Z"/>';
        break;
      case "forward":
        arrow = '<path d="M30 25 A30 30 0 0 1 70 25" fill="none" stroke-width="8"/><path d="M68 15 L80 25 L66 33 Z"/>';
        break;
      default: // back
        arrow = '<path d="M30 75 A30 30 0 0 1 70 75" fill="none" stroke-width="8"/><path d="M68 85 L80 75 L66 67 Z"/>';
    }
    return (
      '<svg viewBox="0 0 100 100">' +
      '<rect x="28" y="40" width="44" height="30" rx="4" fill="none" stroke-width="8"/>' +
      '<circle cx="50" cy="55" r="9" fill="none" stroke-width="7"/>' +
      arrow +
      "</svg>"
    );
  }

  function holdIcon() {
    return (
      '<svg viewBox="0 0 100 100">' +
      '<circle cx="50" cy="50" r="42" fill="none" stroke-width="8"/>' +
      '<rect x="38" y="32" width="9" height="36" rx="2"/>' +
      '<rect x="53" y="32" width="9" height="36" rx="2"/>' +
      "</svg>"
    );
  }

  function completeIcon() {
    return (
      '<svg viewBox="0 0 100 100">' +
      '<circle cx="50" cy="50" r="42" fill="none" stroke-width="8"/>' +
      '<path d="M30 52 L44 66 L72 34" fill="none" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>"
    );
  }

  const ICON_BUILDERS = {
    MOVE_LEFT: function () { return arrowIcon(-90); },
    MOVE_RIGHT: function () { return arrowIcon(90); },
    MOVE_UP: function () { return arrowIcon(0); },
    MOVE_DOWN: function () { return arrowIcon(180); },
    MOVE_TOP_LEFT: function () { return arrowIcon(-45); },
    MOVE_TOP_RIGHT: function () { return arrowIcon(45); },
    MOVE_BOTTOM_LEFT: function () { return arrowIcon(-135); },
    MOVE_BOTTOM_RIGHT: function () { return arrowIcon(135); },
    MOVE_CLOSER: function () { return zoomIcon(true); },
    MOVE_FARTHER: function () { return zoomIcon(false); },
    TILT_LEFT: function () { return tiltIcon("left"); },
    TILT_RIGHT: function () { return tiltIcon("right"); },
    TILT_FORWARD: function () { return tiltIcon("forward"); },
    TILT_BACK: function () { return tiltIcon("back"); },
    ROTATE_CLOCKWISE: function () { return rotateIcon(false); },
    ROTATE_COUNTER_CLOCKWISE: function () { return rotateIcon(true); },
    HOLD_POSITION: holdIcon,
    CALIBRATION_COMPLETE: completeIcon,
  };

  function statusFor(instruction, progress) {
    if (instruction === "CALIBRATION_COMPLETE") return "complete";
    if (instruction === "HOLD_POSITION") return "hold";
    if (progress >= 90) return "almost";
    return "move";
  }

  function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
  }

  // ── DOM refs ──
  const rig = document.getElementById("rig");
  const gradeLetter = document.getElementById("gradeLetter");
  const progressPct = document.getElementById("progressPct");
  const progressRing = document.getElementById("progressRing");
  const statusWord = document.getElementById("statusWord");
  const taskIcon = document.getElementById("taskIcon");
  const taskLabel = document.getElementById("taskLabel");
  const taskReason = document.getElementById("taskReason");
  const barSpatial = document.getElementById("barSpatial");
  const barPose = document.getElementById("barPose");
  const barDepth = document.getElementById("barDepth");
  const barRadial = document.getElementById("barRadial");
  const connLabel = document.getElementById("connLabel");
  const coverageCellsEl = document.getElementById("coverageCells");
  const coverageLabelEl = document.getElementById("coverageLabel");
  let cellEls = [];

  progressRing.style.strokeDasharray = String(RING_CIRCUMFERENCE);
  progressRing.style.strokeDashoffset = String(RING_CIRCUMFERENCE);

  function setBar(el, value) {
    const pct = clamp((Number(value) || 0) * 100, 0, 100);
    el.style.width = pct + "%";
  }

  function buildCoverageGrid() {
    coverageCellsEl.innerHTML = "";
    cellEls = [];
    for (let r = 0; r < GRID_ROWS; r++) {
      const row = [];
      for (let c = 0; c < GRID_COLS; c++) {
        const div = document.createElement("div");
        div.className = "cov-cell";
        coverageCellsEl.appendChild(div);
        row.push(div);
      }
      cellEls.push(row);
    }
  }
  buildCoverageGrid();

  function updateCoverageGrid(data) {
    const occupancy = data.occupancy;
    const activeMask = data.active_mask;
    if (!occupancy || !activeMask) return; // no grid data yet

    let covered = 0;
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const el = cellEls[r][c];
        const isActive = !!activeMask[r][c];
        const isOccupied = !!occupancy[r][c];
        const isEdge = r === 0 || r === GRID_ROWS - 1 || c === 0 || c === GRID_COLS - 1;

        el.classList.toggle("cov-cell--inactive", !isActive);
        el.classList.toggle("cov-cell--occupied", isActive && isOccupied);
        el.classList.toggle("cov-cell--edge", isActive && !isOccupied && isEdge);

        if (isActive && isOccupied) covered++;
      }
    }
    const total = data.total_active_cells || GRID_ROWS * GRID_COLS;
    coverageLabelEl.textContent = "COVERAGE — " + covered + "/" + total;
  }

  function render(data) {
    const progress = clamp(Number(data.progress) || 0, 0, 100);
    const instruction = data.instruction || "HOLD_POSITION";
    const status = statusFor(instruction, progress);

    rig.dataset.status = status;
    statusWord.textContent = STATUS_WORD[status] || status.toUpperCase();

    gradeLetter.textContent = data.grade || "\u2013";
    progressPct.textContent = Math.round(progress) + "%";
    progressRing.style.strokeDashoffset = String(
      RING_CIRCUMFERENCE * (1 - progress / 100)
    );

    taskLabel.textContent = LABELS[instruction] || humanize(instruction);
    taskReason.textContent = data.reason || "";

    const buildIcon = ICON_BUILDERS[instruction];
    taskIcon.innerHTML = buildIcon ? buildIcon() : "";

    setBar(barSpatial, data.spatial);
    setBar(barPose, data.pose);
    setBar(barDepth, data.depth);
    setBar(barRadial, data.radial);
    updateCoverageGrid(data);
  }

  function setConnected(isConnected) {
    rig.dataset.connected = isConnected ? "true" : "false";
    connLabel.textContent = isConnected ? "LINK UP" : "LINK DOWN";
    if (!isConnected) statusWord.textContent = "RECONNECTING";
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(proto + "://" + location.host + "/ws");

    ws.addEventListener("open", function () {
      setConnected(true);
    });
    ws.addEventListener("close", function () {
      setConnected(false);
      setTimeout(connect, 1500); // auto-reconnect on flaky Wi-Fi
    });
    ws.addEventListener("error", function () {
      ws.close();
    });
    ws.addEventListener("message", function (event) {
      try {
        render(JSON.parse(event.data));
      } catch (err) {
        console.error("Bad payload from /ws:", err);
      }
    });
  }

  connect();

  // Keep the tablet awake-ish and full-bleed on iOS/Android home-screen apps.
  document.addEventListener("touchmove", function (e) { e.preventDefault(); }, { passive: false });
})();