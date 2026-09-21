// module.js for "sarween" — Computer Vision for TTRPG
import {CAPTURE_MINIS, generateCaptureTargets} from "./capture_logic.mjs";
import {
  addMovementPoint,
  createMovementState,
  fallbackGridDistance,
  movementColor,
  movementOverage,
  remainingMovement,
  resetMovementAt,
  undoMovementPoint,
} from "./movement_logic.mjs";

const MODULE_ID = "sarween";
const MIN_CANVAS_ZOOM = 0.1;
const DEFAULT_DISPLAY_DIAGONAL_INCHES = 55;
const DEFAULT_GRID_INCHES = 1;
const VIEWPORT_MARKER_IDS = [10, 11, 12, 13];
const MARKER_QUIET_ZONE_FRACTION = 1 / 6;

let ws = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let heartbeatTimer = null;
let referenceCaptureTimer = null;
let manualDisconnect = false;
let markerOverlayEl = null;
let testTargetOverlayEl = null;
let testSequenceTargets = [];
let viewTransformTimer = null;
let testTargetRenderTimer = null;
let guidedCapture = null;
let capturePanelEl = null;
let captureLastStatus = {state: "idle"};
let movementState = null;
let movementOverlayEl = null;
let miniTokenBindings = new Map();
const ignoredMovementUpdates = new Set();

// UI
let statusEl = null;

function getSetting(key) {
  return game.settings.get(MODULE_ID, key);
}

function log(...args) {
  if (getSetting("debug")) console.log("sarween |", ...args);
}
function warn(...args) {
  console.warn("sarween |", ...args);
}
function error(...args) {
  console.error("sarween |", ...args);
}

function calculatePhysicalGridScale() {
  if (!canvas?.ready || !canvas.scene) return null;

  const gridSize = Number(canvas.scene.grid?.size ?? 0);
  const diagonalInches = Number(getSetting("displayDiagonalInches"));
  const targetInches = Number(getSetting("targetGridInches"));
  const calibration = Number(getSetting("physicalGridCalibration"));
  const screenWidth = Number(window.screen?.width ?? 0);
  const screenHeight = Number(window.screen?.height ?? 0);

  if (![gridSize, diagonalInches, targetInches, calibration, screenWidth, screenHeight]
    .every(value => Number.isFinite(value) && value > 0)) return null;

  const cssPixelsPerInch = Math.hypot(screenWidth, screenHeight) / diagonalInches;
  const desiredPixelsPerSquare = cssPixelsPerInch * targetInches * calibration;
  return Math.clamp(desiredPixelsPerSquare / gridSize, MIN_CANVAS_ZOOM, 6);
}

async function applyPhysicalGridScale({animate = true, notify = false} = {}) {
  const scale = calculatePhysicalGridScale();
  if (scale === null) return false;

  const position = {
    x: canvas.stage.pivot.x,
    y: canvas.stage.pivot.y,
    scale,
  };
  if (animate) await canvas.animatePan({...position, duration: 250});
  else canvas.pan(position);

  log(`Applied physical grid scale ${scale.toFixed(4)}`);
  if (notify) {
    const inches = Number(getSetting("targetGridInches"));
    ui.notifications?.info(`Sarween: grid set to ${inches.toFixed(2)} inches per square.`);
  }
  return true;
}

function updateGridSizeInput() {
  const input = statusEl?.querySelector("#sarween-grid-inches");
  if (input) input.value = Number(getSetting("targetGridInches")).toFixed(2);
}

function removeViewportMarkers() {
  markerOverlayEl?.remove();
  markerOverlayEl = null;
}

function renderViewportMarkers() {
  removeViewportMarkers();
  if (!getSetting("viewportMarkersEnabled")) return;

  const size = Number(getSetting("viewportMarkerSize"));
  const inset = Number(getSetting("viewportMarkerInset"));
  markerOverlayEl = document.createElement("div");
  markerOverlayEl.id = "sarween-viewport-markers";
  markerOverlayEl.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 10001;
    pointer-events: none;
    user-select: none;
  `;

  const corners = [
    ["top", "left"],
    ["top", "right"],
    ["bottom", "right"],
    ["bottom", "left"],
  ];
  VIEWPORT_MARKER_IDS.forEach((id, index) => {
    const image = document.createElement("img");
    image.src = `modules/${MODULE_ID}/viewport_markers/marker_${id}.png`;
    image.alt = "";
    image.dataset.markerId = String(id);
    image.style.cssText = `
      position: absolute;
      width: ${size}px;
      height: ${size}px;
      ${corners[index][0]}: ${inset}px;
      ${corners[index][1]}: ${inset}px;
      image-rendering: pixelated;
      background: #fff;
    `;
    markerOverlayEl.appendChild(image);
  });
  document.body.appendChild(markerOverlayEl);
}

function updateMarkerToggleButton() {
  const button = statusEl?.querySelector("#sarween-marker-toggle-btn");
  if (!button) return;
  const enabled = Boolean(getSetting("viewportMarkersEnabled"));
  button.setAttribute("aria-pressed", String(enabled));
  button.style.background = enabled
    ? "rgba(80, 180, 120, 0.42)"
    : "rgba(255,255,255,0.10)";
}

async function toggleViewportMarkers() {
  await game.settings.set(
    MODULE_ID,
    "viewportMarkersEnabled",
    !getSetting("viewportMarkersEnabled"),
  );
}

function readStatusPanelPosition() {
  try {
    const value = JSON.parse(String(getSetting("statusPanelPosition") || ""));
    if (Number.isFinite(value?.x) && Number.isFinite(value?.y)) return value;
  } catch (_error) {
    // An empty or outdated saved position simply uses the default placement.
  }
  return null;
}

function clampStatusPanelPosition(x, y) {
  const margin = 8;
  const width = statusEl?.offsetWidth ?? 0;
  const height = statusEl?.offsetHeight ?? 0;
  return {
    x: Math.clamp(Number(x) || margin, margin, Math.max(margin, window.innerWidth - width - margin)),
    y: Math.clamp(Number(y) || margin, margin, Math.max(margin, window.innerHeight - height - margin)),
  };
}

function defaultStatusPanelPosition() {
  const markerSize = Number(getSetting("viewportMarkerSize")) || 120;
  const markerInset = Number(getSetting("viewportMarkerInset")) || 8;
  if (window.innerWidth < markerInset + markerSize + 12 + (statusEl?.offsetWidth ?? 0)) {
    return clampStatusPanelPosition(8, window.innerHeight - markerInset - markerSize - 12 - (statusEl?.offsetHeight ?? 0));
  }
  return clampStatusPanelPosition(
    markerInset + markerSize + 12,
    window.innerHeight - (statusEl?.offsetHeight ?? 0) - 10,
  );
}

function applyStatusPanelPosition() {
  if (!statusEl) return;
  const position = readStatusPanelPosition() ?? defaultStatusPanelPosition();
  const clamped = clampStatusPanelPosition(position.x, position.y);
  statusEl.style.left = `${clamped.x}px`;
  statusEl.style.top = `${clamped.y}px`;
  statusEl.style.right = "auto";
  statusEl.style.bottom = "auto";
  statusEl.style.transform = "none";
}

async function saveStatusPanelPosition() {
  if (!statusEl) return;
  const rect = statusEl.getBoundingClientRect();
  const position = clampStatusPanelPosition(rect.left, rect.top);
  await game.settings.set(
    MODULE_ID,
    "statusPanelPosition",
    JSON.stringify(position),
  );
}

function enableStatusPanelDragging() {
  const handle = statusEl?.querySelector("#sarween-drag-handle");
  if (!handle) return;
  let drag = null;

  handle.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    const rect = statusEl.getBoundingClientRect();
    drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      panelX: rect.left,
      panelY: rect.top,
    };
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const position = clampStatusPanelPosition(
      drag.panelX + event.clientX - drag.startX,
      drag.panelY + event.clientY - drag.startY,
    );
    statusEl.style.left = `${position.x}px`;
    statusEl.style.top = `${position.y}px`;
  });

  const finishDrag = event => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    drag = null;
    try {
      handle.releasePointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture may already be released if the window lost focus.
    }
    void saveStatusPanelPosition();
  };
  handle.addEventListener("pointerup", finishDrag);
  handle.addEventListener("pointercancel", finishDrag);
}

function columnName(index) {
  let value = Number(index) + 1;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

function canvasPointToClient(x, y) {
  const transform = canvas.stage.worldTransform;
  return {
    x: transform.a * x + transform.c * y + transform.tx,
    y: transform.b * x + transform.d * y + transform.ty,
  };
}

function visibleGridBounds({safeForCapture = false} = {}) {
  if (!canvas?.ready || !canvas.scene) return null;
  const gridSize = Number(canvas.scene.grid?.size ?? 0);
  const inverse = getClientToCanvasTransform();
  if (!Number.isFinite(gridSize) || gridSize <= 0 || !inverse) return null;

  const registration = getViewportRegistration();
  const markerMargin = Number(getSetting("viewportMarkerSize")) + Number(getSetting("viewportMarkerInset")) + 12;
  const area = safeForCapture ? {
    left: Math.max(registration.left, markerMargin),
    top: Math.max(registration.top, markerMargin),
    right: Math.min(registration.right, window.innerWidth - markerMargin),
    bottom: Math.min(registration.bottom, window.innerHeight - markerMargin),
  } : registration;
  const clientCorners = [
    [area.left, area.top],
    [area.right, area.top],
    [area.right, area.bottom],
    [area.left, area.bottom],
  ];
  const canvasCorners = clientCorners.map(([x, y]) => ({
    x: inverse.a * x + inverse.c * y + inverse.tx,
    y: inverse.b * x + inverse.d * y + inverse.ty,
  }));
  const dimensions = canvas.dimensions;
  const originX = Number(dimensions?.sceneX ?? 0)
    + Number(canvas.scene.grid?.shiftX ?? 0);
  const originY = Number(dimensions?.sceneY ?? 0)
    + Number(canvas.scene.grid?.shiftY ?? 0);
  const columnCount = Math.ceil(Number(canvas.scene.width ?? 0) / gridSize);
  const rowCount = Math.ceil(Number(canvas.scene.height ?? 0) / gridSize);
  const xs = canvasCorners.map(point => point.x);
  const ys = canvasCorners.map(point => point.y);

  return {
    registration,
    gridSize,
    originX,
    originY,
    firstColumn: Math.max(0, Math.ceil((Math.min(...xs) - originX) / gridSize)),
    lastColumn: Math.min(
      columnCount - 1,
      Math.floor((Math.max(...xs) - originX) / gridSize) - 1,
    ),
    firstRow: Math.max(0, Math.ceil((Math.min(...ys) - originY) / gridSize)),
    lastRow: Math.min(
      rowCount - 1,
      Math.floor((Math.max(...ys) - originY) / gridSize) - 1,
    ),
  };
}

function generateTwoMiniTestSequence() {
  const bounds = visibleGridBounds();
  if (!bounds || bounds.lastColumn < bounds.firstColumn || bounds.lastRow < bounds.firstRow) {
    return [];
  }

  const routes = {
    A: {
      ringColor: "red",
      pattern: [
        [0.18, 0.20],
        [0.52, 0.50],
        [0.80, 0.76],
        [0.78, 0.22],
        [0.20, 0.78],
      ],
    },
    B: {
      ringColor: "blue",
      pattern: [
        [0.77, 0.73],
        [0.54, 0.50],
        [0.15, 0.20],
        [0.80, 0.69],
        [0.23, 0.78],
      ],
    },
  };
  const routeTargets = {};
  for (const [route, definition] of Object.entries(routes)) {
    routeTargets[route] = definition.pattern.map(([xFraction, yFraction], index) => {
      const column = Math.round(
        bounds.firstColumn
        + (bounds.lastColumn - bounds.firstColumn) * xFraction,
      );
      const row = Math.round(
        bounds.firstRow
        + (bounds.lastRow - bounds.firstRow) * yFraction,
      );
      return {
        route,
        ringColor: definition.ringColor,
        step: index + 1,
        label: `${route}${index + 1}`,
        row,
        column,
        cell: `${columnName(column)}${row + 1}`,
      };
    });
  }

  const targets = [];
  for (let index = 0; index < 5; index += 1) {
    targets.push(routeTargets.A[index], routeTargets.B[index]);
  }
  return targets;
}

function removeTestTargets() {
  testTargetOverlayEl?.remove();
  testTargetOverlayEl = null;
}

function renderTestTargets() {
  removeTestTargets();
  if (!testSequenceTargets.length || !canvas?.ready || !canvas.scene) return;
  const bounds = visibleGridBounds();
  if (!bounds) return;

  testTargetOverlayEl = document.createElement("div");
  testTargetOverlayEl.id = "sarween-test-targets";
  testTargetOverlayEl.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
    user-select: none;
    font-family: sans-serif;
    font-weight: 700;
  `;

  for (const target of testSequenceTargets) {
    const left = bounds.originX + target.column * bounds.gridSize;
    const top = bounds.originY + target.row * bounds.gridSize;
    const corners = [
      canvasPointToClient(left, top),
      canvasPointToClient(left + bounds.gridSize, top),
      canvasPointToClient(left + bounds.gridSize, top + bounds.gridSize),
      canvasPointToClient(left, top + bounds.gridSize),
    ];
    const xs = corners.map(point => point.x);
    const ys = corners.map(point => point.y);
    const clientLeft = Math.min(...xs);
    const clientTop = Math.min(...ys);
    const width = Math.max(...xs) - clientLeft;
    const height = Math.max(...ys) - clientTop;
    if (
      clientLeft < bounds.registration.left
      || clientTop < bounds.registration.top
      || clientLeft + width > bounds.registration.right
      || clientTop + height > bounds.registration.bottom
    ) continue;

    const marker = document.createElement("div");
    const label = target.label ?? String(target.step);
    const isMultiMiniTarget = Boolean(target.route);
    marker.textContent = label;
    marker.title = `${label}: ${target.cell}`;
    marker.style.cssText = `
      position: absolute;
      left: ${clientLeft}px;
      top: ${clientTop}px;
      width: ${width}px;
      height: ${height}px;
      box-sizing: border-box;
      display: grid;
      place-items: center;
      border: 2px ${target.route === "B" ? "dashed" : "solid"} ${
        isMultiMiniTarget ? "rgba(217, 70, 239, 0.88)" : "rgba(255, 214, 64, 0.72)"
      };
      background: ${
        isMultiMiniTarget ? "rgba(217, 70, 239, 0.07)" : "rgba(255, 214, 64, 0.16)"
      };
      color: rgba(255, 255, 255, 0.92);
      font-size: ${Math.clamp(Math.min(width, height) * 0.36, 4, 21)}px;
      line-height: 1;
      letter-spacing: 0;
      text-shadow: 0 1px 3px #000, 0 0 2px #000;
    `;
    testTargetOverlayEl.appendChild(marker);
  }
  document.body.appendChild(testTargetOverlayEl);
}

function sendTestSequence() {
  sendToPython({
    type: "testSequence",
    sceneId: canvas?.scene?.id ?? null,
    targets: testSequenceTargets,
  });
}

function updateTestTargetButton() {
  const button = statusEl?.querySelector("#sarween-two-mini-test-btn");
  if (!button) return;
  const enabled = testSequenceTargets.length > 0;
  button.setAttribute("aria-pressed", String(enabled));
  button.style.background = enabled
    ? "rgba(217, 70, 239, 0.42)"
    : "rgba(255,255,255,0.10)";
}

function toggleTestTargets() {
  testSequenceTargets = testSequenceTargets.length
    ? []
    : generateTwoMiniTestSequence();
  renderTestTargets();
  updateTestTargetButton();
  sendTestSequence();
  if (testSequenceTargets.length) {
    const sequence = testSequenceTargets
      .map(target => `${target.label ?? target.step}:${target.cell}`)
      .join("  ");
    ui.notifications?.info(`Sarween test sequence: ${sequence}`);
  }
}

function scheduleTestTargetRender() {
  if (testTargetRenderTimer) clearTimeout(testTargetRenderTimer);
  testTargetRenderTimer = setTimeout(() => {
    testTargetRenderTimer = null;
    renderTestTargets();
  }, 100);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function captureControl(action, values = {}) {
  return sendToPython({type: "captureControl", action, sessionId: guidedCapture?.sessionId, ...values});
}

function captureDelay(seconds, label, callback) {
  clearInterval(guidedCapture?.timer);
  const end = Date.now() + seconds * 1000;
  guidedCapture.waiting = true;
  guidedCapture.phase = label;
  guidedCapture.seconds = seconds;
  renderCapturePanel();
  guidedCapture.timer = setInterval(() => {
    guidedCapture.seconds = Math.max(0, Math.ceil((end - Date.now()) / 1000));
    renderCapturePanel();
    if (Date.now() >= end) {
      clearInterval(guidedCapture.timer);
      guidedCapture.timer = null;
      callback();
    }
  }, 250);
}

function promptCaptureTarget() {
  if (!guidedCapture?.recording) return;
  const target = guidedCapture.targets[guidedCapture.index];
  if (!target) {
    captureDelay(15, "Stationary · all minis", () => stopGuidedCapture("completed"));
    return;
  }
  guidedCapture.phase = "placement";
  guidedCapture.seconds = 0;
  guidedCapture.waiting = false;
  testSequenceTargets = [target];
  renderTestTargets();
  sendTestSequence();
  captureControl("prompt", {index: target.index});
  renderCapturePanel();
}

function stopGuidedCapture(reason = "userStopped") {
  if (!guidedCapture) return;
  clearInterval(guidedCapture.timer);
  guidedCapture.waiting = true;
  guidedCapture.phase = "Saving";
  guidedCapture.seconds = 0;
  captureControl("stop", {reason});
  renderCapturePanel();
}

function renderCapturePanel() {
  if (!capturePanelEl) {
    capturePanelEl = document.createElement("section");
    capturePanelEl.id = "sarween-capture";
    capturePanelEl.style.cssText = `position:fixed;left:144px;bottom:64px;z-index:10000;
      width:340px;max-width:calc(100vw - 16px);box-sizing:border-box;padding:12px;
      background:#202020;color:#fff;border:1px solid #666;border-radius:6px;
      font:13px sans-serif;line-height:1.4;`;
    document.body.appendChild(capturePanelEl);
    const bar = statusEl?.getBoundingClientRect();
    if (bar) {
      capturePanelEl.style.left = `${bar.left}px`;
      capturePanelEl.style.bottom = `${window.innerHeight - bar.top + 12}px`;
    }
  }
  const target = guidedCapture?.targets[guidedCapture.index];
  const active = Boolean(guidedCapture?.recording);
  const saved = captureLastStatus.state === "saved";
  const heading = guidedCapture?.phase === "placement" && target
    ? `${target.label} · ${target.tokenName} · ${target.cell}`
    : `${guidedCapture?.phase ?? (saved ? "Recording saved" : "Capture paused")}${guidedCapture?.seconds ? ` · ${guidedCapture.seconds}s` : ""}`;
  capturePanelEl.innerHTML = `
    <header style="display:flex;align-items:center;gap:8px;margin-bottom:10px;cursor:move">
      <i class="fas fa-grip-vertical" aria-hidden="true"></i><strong style="flex:1">Dataset capture</strong>
      <button data-action="close" title="Hide capture panel" style="width:26px;padding:2px"><i class="fas fa-times"></i></button>
    </header>
    <div style="font-size:18px;line-height:1.3;overflow-wrap:anywhere">${escapeHtml(heading)}</div>
    <div style="margin:6px 0;color:#ccc">${guidedCapture ? `${Math.min(guidedCapture.index, guidedCapture.targets.length)}/${guidedCapture.targets.length} placements confirmed` : ""}</div>
    <div style="color:#e5b4df;overflow-wrap:anywhere">${escapeHtml(guidedCapture?.message ?? (saved ? "Tracking predictions remain paused" : ""))}</div>
    <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
      <button data-action="placed" ${!active || guidedCapture.waiting ? "disabled" : ""}><i class="fas fa-check"></i> Placed</button>
      <button data-action="stop" ${!active ? "disabled" : ""} title="Save a partial recording"><i class="fas fa-stop"></i> Finish</button>
      ${!active ? '<button data-action="new"><i class="fas fa-video"></i> New capture</button><button data-action="resume"><i class="fas fa-play"></i> Resume tracking</button>' : ""}
    </div>`;
  const panelRect = capturePanelEl.getBoundingClientRect();
  capturePanelEl.style.left = `${Math.clamp(panelRect.left, 8, Math.max(8, window.innerWidth - panelRect.width - 8))}px`;
  if (panelRect.top < 8 || panelRect.bottom > window.innerHeight - 8) {
    capturePanelEl.style.top = `${Math.clamp(panelRect.top, 8, Math.max(8, window.innerHeight - panelRect.height - 8))}px`;
    capturePanelEl.style.bottom = "auto";
  }
  capturePanelEl.querySelector('[data-action="placed"]').onclick = () => {
    if (!target || guidedCapture.waiting) return;
    guidedCapture.waiting = true;
    guidedCapture.message = "Confirming placement…";
    if (!captureControl("confirm", {index: target.index})) {
      guidedCapture.message = "Connection lost; recording will be saved as partial";
    }
    renderCapturePanel();
  };
  capturePanelEl.querySelector('[data-action="stop"]').onclick = () => stopGuidedCapture();
  capturePanelEl.querySelector('[data-action="close"]').onclick = () => {
    capturePanelEl.remove();
    capturePanelEl = null;
  };
  capturePanelEl.querySelector('[data-action="new"]')?.addEventListener("click", () => openCaptureSetup());
  capturePanelEl.querySelector('[data-action="resume"]')?.addEventListener("click", () => captureControl("resume"));
  // Keep the compact tool draggable without moving Foundry's canvas.
  const header = capturePanelEl.querySelector("header");
  header.onpointerdown = event => {
    if (event.target.closest("button")) return;
    const panel = capturePanelEl;
    const rect = panel.getBoundingClientRect();
    const startX = event.clientX, startY = event.clientY;
    header.setPointerCapture(event.pointerId);
    header.onpointermove = move => {
      panel.style.left = `${Math.clamp(rect.left + move.clientX - startX, 8, Math.max(8, window.innerWidth - panel.offsetWidth - 8))}px`;
      panel.style.top = `${Math.clamp(rect.top + move.clientY - startY, 8, Math.max(8, window.innerHeight - panel.offsetHeight - 8))}px`;
      panel.style.bottom = "auto";
    };
    header.onpointerup = () => { header.onpointermove = null; };
    event.preventDefault();
  };
}

function openCaptureSetup() {
  if (guidedCapture?.recording || guidedCapture?.phase === "Starting") {
    renderCapturePanel();
    return;
  }
  if (!canvas?.ready || !ws || ws.readyState !== WebSocket.OPEN) {
    ui.notifications?.warn("Connect Sarween and open a scene first.");
    return;
  }
  const choices = listSceneTokenChoices(canvas.scene);
  const rows = CAPTURE_MINIS.map(mini => {
    const selected = choices.find(choice => choice.tokenName.toLowerCase() === mini.tokenName.toLowerCase());
    return `<tr data-route="${mini.route}"><td><input type="checkbox" ${selected ? "checked" : ""}></td>
      <td>${mini.route} · ${mini.tokenName}</td><td><select style="width:100%"><option value="">Choose token</option>
      ${choices.map(choice => `<option value="${escapeHtml(choice.tokenId)}" ${choice === selected ? "selected" : ""}>${escapeHtml(choice.tokenName)}</option>`).join("")}</select></td></tr>`;
  }).join("");
  new Dialog({
    title: "Sarween: Dataset capture",
    content: `<form><table><thead><tr><th></th><th>Mini ring</th><th>Foundry token</th></tr></thead><tbody>${rows}</tbody></table>
      <label>Fog / lighting label</label><select name="scenario"><option value="fog-on">Fog on</option><option value="fog-off">Fog off</option></select>
      <label>Notes</label><input name="notes" type="text" maxlength="2000">
      </form>`,
    buttons: {
      start: {label: "Start camera recording", icon: '<i class="fas fa-video"></i>', callback: html => {
        try {
          if (!getSetting("viewportMarkersEnabled")) throw new Error("Turn on the ArUco markers first");
          if (Number(canvas.scene.grid?.type) !== 1) throw new Error("Dataset capture currently requires a square Foundry grid");
          const root = html[0];
          const participants = CAPTURE_MINIS.flatMap(mini => {
            const row = root.querySelector(`[data-route="${mini.route}"]`);
            if (!row.querySelector("input").checked) return [];
            const tokenId = row.querySelector("select").value;
            if (!tokenId) throw new Error(`Choose a token for ${mini.tokenName}`);
            const token = canvas.scene.tokens.get(tokenId);
            if (Number(token.width) !== 1 || Number(token.height) !== 1) throw new Error(`${mini.tokenName} must be a one-square token for this capture`);
            return [{...mini, tokenId, sight: token.sight?.toObject?.() ?? token.sight}];
          });
          if (!participants.length) throw new Error("Choose at least one mini");
          if (new Set(participants.map(mini => mini.tokenId)).size !== participants.length) throw new Error("Choose a different token for every mini");
          const scenario = root.querySelector('[name="scenario"]').value;
          if (scenario === "fog-on" && !canvas.scene.tokenVision) throw new Error("Enable Token Vision in the scene before a fog-on capture");
          if (scenario === "fog-on") {
            const blind = participants.find(mini => !mini.sight?.enabled || !mini.sight?.range);
            if (blind) throw new Error(`Enable vision and a sight range for ${blind.tokenName}`);
          }
          const targets = generateCaptureTargets(visibleGridBounds({safeForCapture: true}), participants);
          guidedCapture = {
            sessionId: crypto.randomUUID(), sceneId: canvas.scene.id, participants, targets,
            index: 0, recording: false, waiting: true, phase: "Starting", timer: null,
            viewSignature: JSON.stringify(getClientToCanvasTransform()),
            originalTokens: participants.map(mini => {
              const token = canvas.scene.tokens.get(mini.tokenId);
              return {id: token.id, x: token.x, y: token.y};
            }),
            originalControlIds: (canvas.tokens.controlled ?? []).map(token => token.id),
          };
          captureLastStatus = {state: "starting"};
          canvas.tokens.releaseAll();
          for (const mini of participants) canvas.tokens.get(mini.tokenId)?.control({releaseOthers: false});
          sendSceneAndView();
          if (!captureControl("start", {
            ...guidedCapture, timer: undefined,
            scenario, notes: root.querySelector('[name="notes"]').value,
            moduleVersion: game.modules.get(MODULE_ID)?.version,
            sceneVision: {tokenVision: canvas.scene.tokenVision, fogExploration: canvas.scene.fog?.exploration},
            markerSettings: {size: getSetting("viewportMarkerSize"), inset: getSetting("viewportMarkerInset")},
          })) throw new Error("Could not start capture: connection lost");
          renderCapturePanel();
        } catch (exc) {
          guidedCapture = null;
          ui.notifications?.error(`Sarween: ${exc.message}`);
        }
      }},
      cancel: {label: "Cancel"},
    },
    default: "cancel",
  }, {width: 460}).render(true);
}

async function handleCaptureStatus(data) {
  if (data.state === "confirmed") {
    if (!guidedCapture?.recording || data.sessionId !== guidedCapture.sessionId || data.index !== guidedCapture.index) return;
    const target = guidedCapture.targets[guidedCapture.index];
    try {
      if (canvas.scene?.id !== guidedCapture.sceneId) throw new Error("Scene changed");
      const token = canvas.scene.tokens.get(target.tokenId);
      if (!token) throw new Error(`${target.tokenName} token was deleted`);
      const bounds = visibleGridBounds();
      await token.update({x: bounds.originX + target.column * bounds.gridSize, y: bounds.originY + target.row * bounds.gridSize}, {animate: getSetting("animateTokenMovement")});
      for (const mini of guidedCapture.participants) canvas.tokens.get(mini.tokenId)?.control({releaseOthers: false});
      sendToPython({type: "sceneVisualChanged", reason: "guidedTokenMove"});
      guidedCapture.index += 1;
      guidedCapture.message = "";
      testSequenceTargets = [];
      renderTestTargets();
      sendTestSequence();
      const initialDone = target.step === 0 && guidedCapture.index === guidedCapture.participants.length;
      captureDelay(initialDone ? 15 : 3, initialDone ? "Stationary · all minis" : "Settling", promptCaptureTarget);
    } catch (exc) {
      guidedCapture.message = exc.message;
      stopGuidedCapture("tokenMoveFailed");
    }
    return;
  }
  captureLastStatus = data;
  if (data.state === "recording") {
    if (data.sessionId !== guidedCapture?.sessionId) {
      sendToPython({type: "captureControl", action: "stop", sessionId: data.sessionId, reason: "browserReloaded"});
      return;
    }
    guidedCapture.recording = true;
    guidedCapture.message = "";
    captureDelay(5, "Baseline · no physical minis", promptCaptureTarget);
  } else if (data.state === "saved") {
    clearInterval(guidedCapture?.timer);
    if (guidedCapture) {
      guidedCapture.recording = false;
      guidedCapture.seconds = 0;
      guidedCapture.phase = data.reason === "completed" ? "Recording saved" : "Partial recording saved";
      guidedCapture.message = "Tracking predictions remain paused";
      const originalScene = game.scenes?.get(guidedCapture.sceneId)
        ?? (canvas.scene?.id === guidedCapture.sceneId ? canvas.scene : null);
      for (const original of guidedCapture.originalTokens ?? []) {
        const token = originalScene?.tokens.get(original.id);
        if (token) {
          try {
            await token.update({x: original.x, y: original.y}, {animate: false});
          } catch (exc) {
            warn("Could not restore token after capture", original.id, exc);
          }
        }
      }
      if (canvas.scene?.id === guidedCapture.sceneId) {
        canvas.tokens.releaseAll();
        for (const id of guidedCapture.originalControlIds ?? []) canvas.tokens.get(id)?.control({releaseOthers: false});
      }
    }
    testSequenceTargets = [];
    renderTestTargets();
    sendTestSequence();
    renderCapturePanel();
  } else if (data.state === "error") {
    if (guidedCapture) {
      guidedCapture.message = data.message;
      guidedCapture.waiting = false;
      if (!guidedCapture.recording) guidedCapture.phase = "Capture not started";
    }
    ui.notifications?.warn(`Sarween capture: ${data.message}`);
    renderCapturePanel();
  } else if (data.state === "idle") {
    if (guidedCapture?.phase === "Starting") return;
    guidedCapture = null;
    capturePanelEl?.remove();
    capturePanelEl = null;
  }
}

function getViewportRegistration() {
  const size = Number(getSetting("viewportMarkerSize"));
  const inset = Number(getSetting("viewportMarkerInset"));
  const quietZone = size * MARKER_QUIET_ZONE_FRACTION;
  return {
    left: inset + quietZone,
    top: inset + quietZone,
    right: window.innerWidth - inset - quietZone,
    bottom: window.innerHeight - inset - quietZone,
  };
}

function stopReferenceCapture() {
  if (referenceCaptureTimer) clearInterval(referenceCaptureTimer);
  referenceCaptureTimer = null;
}

function captureRenderedReference() {
  if (!canvas?.ready || !canvas.scene || ws?.readyState !== WebSocket.OPEN) return;
  try {
    const viewTransform = buildViewTransformPayload();
    if (!viewTransform) return;
    // Extract the Foundry canvas, not the camera or DOM marker overlay. The
    // registration rectangle makes its pixels correspond to the camera warp.
    const source = canvas.app.renderer.extract.canvas();
    const registration = getViewportRegistration();
    const scaleX = source.width / window.innerWidth;
    const scaleY = source.height / window.innerHeight;
    const snapshot = document.createElement("canvas");
    snapshot.width = 640;
    snapshot.height = 360;
    snapshot.getContext("2d").drawImage(
      source,
      registration.left * scaleX, registration.top * scaleY,
      (registration.right - registration.left) * scaleX,
      (registration.bottom - registration.top) * scaleY,
      0, 0, snapshot.width, snapshot.height,
    );
    sendToPython({
      type: "renderedReference",
      sceneId: canvas.scene.id,
      viewTransform,
      image: snapshot.toDataURL("image/jpeg", 0.7),
    });
  } catch (err) {
    stopReferenceCapture();
    warn("Could not capture a rendered reference; video recording will continue:", err);
  }
}

function setReferenceCapture(enabled) {
  stopReferenceCapture();
  if (enabled) {
    captureRenderedReference();
    referenceCaptureTimer = setInterval(captureRenderedReference, 2000);
  }
}

function getClientToCanvasTransform() {
  if (typeof canvas?.canvasCoordinatesFromClient === "function") {
    const sample = 100;
    const origin = canvas.canvasCoordinatesFromClient({x: 0, y: 0});
    const alongX = canvas.canvasCoordinatesFromClient({x: sample, y: 0});
    const alongY = canvas.canvasCoordinatesFromClient({x: 0, y: sample});
    if (origin && alongX && alongY) {
      return {
        a: (alongX.x - origin.x) / sample,
        b: (alongX.y - origin.y) / sample,
        c: (alongY.x - origin.x) / sample,
        d: (alongY.y - origin.y) / sample,
        tx: origin.x,
        ty: origin.y,
      };
    }
  }

  const transform = canvas.stage.worldTransform;
  const determinant = transform.a * transform.d - transform.b * transform.c;
  if (Math.abs(determinant) < 1e-9) return null;
  return {
    a: transform.d / determinant,
    b: -transform.b / determinant,
    c: -transform.c / determinant,
    d: transform.a / determinant,
    tx: (transform.c * transform.ty - transform.d * transform.tx) / determinant,
    ty: (transform.b * transform.tx - transform.a * transform.ty) / determinant,
  };
}

function buildViewTransformPayload(type = "viewTransform") {
  if (!canvas?.ready || !canvas.scene || !getSetting("viewportMarkersEnabled")) return null;
  const transform = canvas.stage.worldTransform;
  const dimensions = canvas.dimensions;
  return {
    type,
    sceneId: canvas.scene.id,
    viewportMarkerMode: true,
    markerIds: VIEWPORT_MARKER_IDS,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    registration: getViewportRegistration(),
    canvasTransform: {
      a: transform.a,
      b: transform.b,
      c: transform.c,
      d: transform.d,
      tx: transform.tx,
      ty: transform.ty,
    },
    clientToCanvasTransform: getClientToCanvasTransform(),
    gridOriginX: Number(dimensions?.sceneX ?? 0) + Number(canvas.scene.grid?.shiftX ?? 0),
    gridOriginY: Number(dimensions?.sceneY ?? 0) + Number(canvas.scene.grid?.shiftY ?? 0),
    gridSize: Number(canvas.scene.grid?.size ?? 0),
    sentAt: Date.now(),
  };
}

function sendViewTransform() {
  const payload = buildViewTransformPayload();
  if (payload) sendToPython(payload);
}

function sendSceneAndView() {
  const scene = buildSceneInfoPayload(canvas?.scene);
  if (scene) sendToPython(scene);
  sendViewTransform();
}

function scheduleViewTransformSend() {
  if (viewTransformTimer) clearTimeout(viewTransformTimer);
  viewTransformTimer = setTimeout(() => {
    viewTransformTimer = null;
    sendViewTransform();
  }, 100);
}

// ──────────────────────────────────────────────────────────────────────────────
// Movement selection and budget overlay
// ──────────────────────────────────────────────────────────────────────────────

function tokenDocument(token) {
  return token?.document ?? token ?? null;
}

function miniForToken(token) {
  const document = tokenDocument(token);
  if (!document) return null;
  for (const [miniId, tokenId] of miniTokenBindings.entries()) {
    if (String(tokenId) === String(document.id)) {
      return CAPTURE_MINIS.find(mini => mini.miniId === miniId) ?? {
        miniId,
        tokenName: document.name || miniId,
      };
    }
  }
  const name = String(document.name || "").trim().toLowerCase();
  return CAPTURE_MINIS.find(mini => mini.tokenName.toLowerCase() === name) ?? null;
}

function tokenForMini(miniId) {
  const boundId = miniTokenBindings.get(String(miniId));
  if (boundId) {
    const placeable = canvas.tokens?.get?.(boundId);
    if (placeable) return placeable;
  }
  const spec = CAPTURE_MINIS.find(mini => mini.miniId === String(miniId));
  if (!spec) return null;
  const document = (canvas.scene?.tokens ?? []).find(
    token => String(token.name || "").toLowerCase() === spec.tokenName.toLowerCase()
  );
  return document ? canvas.tokens?.get?.(document.id) ?? document : null;
}

function movementPointForToken(token, changes = {}) {
  const document = tokenDocument(token);
  const gridSize = Number(canvas.scene?.grid?.size ?? 0);
  const originX = Number(canvas.dimensions?.sceneX ?? 0) + Number(canvas.scene?.grid?.shiftX ?? 0);
  const originY = Number(canvas.dimensions?.sceneY ?? 0) + Number(canvas.scene?.grid?.shiftY ?? 0);
  const x = Number(changes.x ?? document?.x ?? 0);
  const y = Number(changes.y ?? document?.y ?? 0);
  return {
    x,
    y,
    column: gridSize > 0 ? Math.round((x - originX) / gridSize) : 0,
    row: gridSize > 0 ? Math.round((y - originY) / gridSize) : 0,
  };
}

function tokenMovementBudget(token) {
  const document = tokenDocument(token);
  const movement = document?.actor?.system?.attributes?.movement;
  const walk = Number(movement?.walk?.value ?? movement?.walk);
  if (Number.isFinite(walk) && walk > 0) return walk;
  return Math.max(0, Number(getSetting("defaultMovementSpeed")) || 30);
}

function measureMovementFeet(from, to) {
  const gridSize = Number(canvas.scene?.grid?.size ?? 0);
  const gridDistance = Number(canvas.scene?.grid?.distance ?? 5) || 5;
  if (gridSize > 0 && canvas.grid?.measurePath) {
    const half = gridSize / 2;
    try {
      const measured = canvas.grid.measurePath([
        {x: from.x + half, y: from.y + half},
        {x: to.x + half, y: to.y + half},
      ]);
      const distance = Number(measured?.distance);
      if (Number.isFinite(distance)) return distance;
    } catch (error) {
      warn("Could not use Foundry path measurement; using square-grid fallback", error);
    }
  }
  return fallbackGridDistance(from, to, gridDistance);
}

function sceneToClient(point) {
  const transform = canvas.stage?.worldTransform;
  if (!transform) return {x: point.x, y: point.y};
  return {
    x: transform.a * point.x + transform.c * point.y + transform.tx,
    y: transform.b * point.x + transform.d * point.y + transform.ty,
  };
}

function ensureMovementOverlay() {
  if (movementOverlayEl) return movementOverlayEl;
  movementOverlayEl = document.createElement("div");
  movementOverlayEl.id = "sarween-movement-overlay";
  movementOverlayEl.style.cssText = `position:fixed;inset:0;z-index:9998;pointer-events:none;overflow:hidden;`;
  document.body.appendChild(movementOverlayEl);
  return movementOverlayEl;
}

function updateMovementStatusUI() {
  const status = statusEl?.querySelector("#sarween-movement-status");
  const undo = statusEl?.querySelector("#sarween-movement-undo");
  const reset = statusEl?.querySelector("#sarween-movement-reset");
  if (!status || !undo || !reset) return;
  if (!movementState) {
    status.hidden = true;
    undo.hidden = true;
    reset.hidden = true;
    return;
  }
  const overage = movementOverage(movementState);
  status.hidden = false;
  undo.hidden = false;
  reset.hidden = false;
  status.textContent = overage > 0
    ? `${movementState.label}: ${overage} ft over`
    : `${movementState.label}: ${movementState.usedFeet}/${movementState.budgetFeet} ft`;
  status.style.color = movementColor(movementState);
  undo.disabled = movementState.segments.length === 0;
}

function renderMovementOverlay() {
  updateMovementStatusUI();
  if (!movementState || !canvas?.ready) {
    movementOverlayEl?.remove();
    movementOverlayEl = null;
    return;
  }
  const overlay = ensureMovementOverlay();
  const gridSize = Number(canvas.scene?.grid?.size ?? 0);
  const gridDistance = Number(canvas.scene?.grid?.distance ?? 5) || 5;
  const half = gridSize / 2;
  const color = movementColor(movementState);
  const current = movementState.points.at(-1);
  const currentClient = sceneToClient({x: current.x + half, y: current.y + half});
  const transform = canvas.stage?.worldTransform;
  const screenScale = transform
    ? (Math.hypot(transform.a, transform.b) + Math.hypot(transform.c, transform.d)) / 2
    : 1;
  const remaining = remainingMovement(movementState);
  const rangeRadius = gridDistance > 0 ? remaining / gridDistance * gridSize * screenScale : 0;
  const tokenRadius = Math.max(10, gridSize * screenScale * 0.58);
  const path = movementState.points.map(point => {
    const client = sceneToClient({x: point.x + half, y: point.y + half});
    return `${client.x},${client.y}`;
  }).join(" ");
  const overage = movementOverage(movementState);
  overlay.innerHTML = `<svg width="100%" height="100%" viewBox="0 0 ${window.innerWidth} ${window.innerHeight}" aria-hidden="true">
    ${remaining > 0 ? `<circle cx="${currentClient.x}" cy="${currentClient.y}" r="${rangeRadius}" fill="${color}" fill-opacity="0.055" stroke="${color}" stroke-width="3" stroke-dasharray="10 7"/>` : ""}
    ${path ? `<polyline points="${path}" fill="none" stroke="${color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>` : ""}
    <circle cx="${currentClient.x}" cy="${currentClient.y}" r="${tokenRadius}" fill="none" stroke="${color}" stroke-width="4"/>
    <text x="${currentClient.x}" y="${currentClient.y - tokenRadius - 8}" text-anchor="middle" fill="${color}" stroke="#000" stroke-width="3" paint-order="stroke" font-size="16" font-weight="700">${overage > 0 ? `${overage} ft over` : `${remaining} ft left`}</text>
  </svg>`;
}

function clearMovementSelection({notifyPython = true} = {}) {
  if (movementState && notifyPython) {
    sendToPython({
      type: "movementSelection",
      miniId: movementState.miniId,
      tokenId: movementState.tokenId,
      selected: false,
    });
  }
  movementState = null;
  renderMovementOverlay();
}

function selectMovementToken(token, {notifyPython = true} = {}) {
  const document = tokenDocument(token);
  const mini = miniForToken(document);
  if (!document || !mini) return false;
  miniTokenBindings.set(mini.miniId, document.id);
  movementState = createMovementState({
    miniId: mini.miniId,
    tokenId: document.id,
    label: document.name || mini.tokenName,
    budgetFeet: tokenMovementBudget(document),
    point: movementPointForToken(document),
  });
  if (notifyPython) {
    sendToPython({
      type: "movementSelection",
      miniId: mini.miniId,
      tokenId: document.id,
      selected: true,
    });
  }
  renderMovementOverlay();
  return true;
}

function togglePhysicalMiniSelection(miniId) {
  if (!getSetting("physicalTapSelection")) return;
  const token = tokenForMini(miniId);
  if (!token) {
    ui.notifications?.warn(`Sarween: No Foundry token is assigned to ${miniId}.`);
    return;
  }
  const document = tokenDocument(token);
  if (movementState?.miniId === String(miniId)) {
    token.release?.();
    if (movementState) clearMovementSelection();
    return;
  }
  token.control?.({releaseOthers: true});
  if (movementState?.tokenId !== document.id) selectMovementToken(document);
}

function recordSelectedMovement(token, changes = {}) {
  const document = tokenDocument(token);
  if (!movementState || document?.id !== movementState.tokenId) return;
  if (ignoredMovementUpdates.delete(document.id)) return;
  if (changes.x === undefined && changes.y === undefined) return;
  const next = movementPointForToken(document, changes);
  const previous = movementState.points.at(-1);
  movementState = addMovementPoint(
    movementState,
    next,
    measureMovementFeet(previous, next),
  );
  renderMovementOverlay();
}

async function undoSelectedMovement() {
  if (!movementState) return;
  const undone = undoMovementPoint(movementState);
  if (!undone.point) return;
  movementState = undone.state;
  ignoredMovementUpdates.add(movementState.tokenId);
  try {
    await canvas.scene.updateEmbeddedDocuments(
      "Token",
      [{_id: movementState.tokenId, x: undone.point.x, y: undone.point.y}],
      {animate: getSetting("animateTokenMovement")},
    );
  } finally {
    setTimeout(() => ignoredMovementUpdates.delete(movementState?.tokenId), 1000);
  }
  renderMovementOverlay();
}

function resetSelectedMovement() {
  if (!movementState) return;
  const token = canvas.scene?.tokens?.get?.(movementState.tokenId);
  if (!token) return;
  movementState = resetMovementAt(movementState, movementPointForToken(token));
  renderMovementOverlay();
}

// ──────────────────────────────────────────────────────────────────────────────
// Settings
// ──────────────────────────────────────────────────────────────────────────────

Hooks.once("init", () => {
  // Foundry 13 normally clamps zoom-out to a fitted-scene view. Physical-TV
  // play needs extra room so a scene with fewer columns can still use 1-inch
  // grid squares without changing scene dimensions or grid alignment.
  CONFIG.Canvas.minZoom = Math.min(
    Number(CONFIG.Canvas.minZoom ?? MIN_CANVAS_ZOOM),
    MIN_CANVAS_ZOOM
  );

  game.settings.register(MODULE_ID, "wsUrl", {
    name: "Python WebSocket URL",
    hint: "Where to connect for Sarween output. Default: ws://127.0.0.1:8765",
    scope: "world",
    config: true,
    type: String,
    default: "ws://127.0.0.1:8765"
  });

  game.settings.register(MODULE_ID, "autoConnect", {
    name: "Auto-connect to Python",
    hint: "Automatically connect when the world loads. Reconnect automatically if the connection drops.",
    scope: "world",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "debug", {
    name: "Enable debug logging",
    hint: "Log Sarween WebSocket events to the browser console.",
    scope: "client",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "physicalGridAutoApply", {
    name: "Automatically apply physical grid size",
    hint: "Reapply Sarween's physical grid scale when a scene opens or the Foundry window changes size.",
    scope: "client",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "displayDiagonalInches", {
    name: "Display diagonal (inches)",
    hint: "Physical diagonal of the TV or display showing the battle map.",
    scope: "client",
    config: true,
    type: Number,
    range: {min: 10, max: 150, step: 0.5},
    default: DEFAULT_DISPLAY_DIAGONAL_INCHES,
    onChange: () => applyPhysicalGridScale({animate: false})
  });

  game.settings.register(MODULE_ID, "targetGridInches", {
    name: "Grid square size (inches)",
    hint: "Desired physical size of each grid square on the display.",
    scope: "client",
    config: true,
    type: Number,
    range: {min: 0.25, max: 3, step: 0.05},
    default: DEFAULT_GRID_INCHES,
    onChange: () => {
      updateGridSizeInput();
      applyPhysicalGridScale({animate: false});
    }
  });

  game.settings.register(MODULE_ID, "physicalGridCalibration", {
    name: "Physical grid calibration",
    hint: "Fine adjustment after measuring a square with a ruler. Increase this value if squares are too small.",
    scope: "client",
    config: true,
    type: Number,
    range: {min: 0.5, max: 1.5, step: 0.005},
    default: 1,
    onChange: () => applyPhysicalGridScale({animate: false})
  });

  game.settings.register(MODULE_ID, "viewportMarkersEnabled", {
    name: "Fixed viewport ArUco markers",
    hint: "Show four high-contrast registration markers above fog and keep them fixed while the map pans.",
    scope: "client",
    config: true,
    type: Boolean,
    default: true,
    onChange: () => {
      renderViewportMarkers();
      updateMarkerToggleButton();
      applyStatusPanelPosition();
      scheduleViewTransformSend();
    }
  });

  game.settings.register(MODULE_ID, "viewportMarkerSize", {
    name: "Viewport marker size (pixels)",
    hint: "Displayed size including the white border. Increase this if the far markers are difficult for the camera to read.",
    scope: "client",
    config: true,
    type: Number,
    range: {min: 72, max: 240, step: 4},
    default: 120,
    onChange: () => {
      renderViewportMarkers();
      applyStatusPanelPosition();
      scheduleViewTransformSend();
    }
  });

  game.settings.register(MODULE_ID, "viewportMarkerInset", {
    name: "Viewport marker inset (pixels)",
    hint: "Distance between each marker plate and the edge of the Foundry window.",
    scope: "client",
    config: true,
    type: Number,
    range: {min: 0, max: 80, step: 2},
    default: 8,
    onChange: () => {
      renderViewportMarkers();
      applyStatusPanelPosition();
      scheduleViewTransformSend();
    }
  });

  game.settings.register(MODULE_ID, "animateTokenMovement", {
    name: "Animate tracked token movement",
    hint: "Slide tracked tokens between squares instead of updating their position immediately.",
    scope: "client",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "physicalTapSelection", {
    name: "Physical tap selects a mini",
    hint: "Let the overhead camera select or deselect a player mini after a short physical tap.",
    scope: "world",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "defaultMovementSpeed", {
    name: "Fallback movement speed (feet)",
    hint: "Used when the selected Foundry actor does not provide a walking speed.",
    scope: "world",
    config: true,
    type: Number,
    range: {min: 0, max: 120, step: 5},
    default: 30
  });

  game.settings.register(MODULE_ID, "statusPanelPosition", {
    name: "Sarween panel position",
    scope: "client",
    config: false,
    type: String,
    default: ""
  });

  console.log("sarween | Settings registered.");
});

// ──────────────────────────────────────────────────────────────────────────────
// Status UI
// ──────────────────────────────────────────────────────────────────────────────

function ensureStatusUI() {
  if (statusEl) return;

  statusEl = document.createElement("div");
  statusEl.id = "sarween-status";
  statusEl.style.cssText = `
    position: fixed;
    top: 8px;
    left: 8px;
    z-index: 10000;
    font-family: sans-serif;
    font-size: 12px;
    background: rgba(0,0,0,0.75);
    color: white;
    padding: 8px 10px;
    border-radius: 8px;
    display: flex;
    flex-wrap: wrap;
    max-width: calc(100vw - 16px);
    box-sizing: border-box;
    align-items: center;
    gap: 8px;
    user-select: none;
  `;

  statusEl.innerHTML = `
    <span id="sarween-drag-handle" title="Drag Sarween panel" style="
      width: 14px;
      height: 26px;
      display: grid;
      place-items: center;
      color: rgba(255,255,255,0.72);
      cursor: grab;
      touch-action: none;
    "><i class="fas fa-grip-vertical"></i></span>
    <span id="sarween-dot" style="width:10px;height:10px;border-radius:50%;display:inline-block;background:#aa0000;"></span>
    <span id="sarween-text">Sarween: Disconnected</span>
    <label for="sarween-grid-inches" style="margin-left:6px;white-space:nowrap;">Grid</label>
    <input id="sarween-grid-inches" type="number" min="0.25" max="3" step="0.05" style="
      width: 52px;
      padding: 3px 4px;
      border: 1px solid rgba(255,255,255,0.25);
      border-radius: 4px;
      background: rgba(255,255,255,0.10);
      color: white;
    ">
    <span>in</span>
    <button id="sarween-grid-btn" type="button" title="Apply physical grid size" style="
      width: 28px;
      height: 26px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.10);
      color: white;
      cursor: pointer;
    "><i class="fas fa-ruler-horizontal"></i></button>
    <button id="sarween-two-mini-test-btn" type="button" title="Guided one-to-five-mini dataset capture" aria-pressed="false" style="
      width: 28px;
      height: 26px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.10);
      color: white;
      cursor: pointer;
    "><i class="fas fa-users"></i></button>
    <button id="sarween-marker-toggle-btn" type="button" title="Toggle ArUco corner markers" aria-pressed="true" style="
      width: 28px;
      height: 26px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(80, 180, 120, 0.42);
      color: white;
      cursor: pointer;
    "><i class="fas fa-qrcode"></i></button>
    <span id="sarween-movement-status" hidden style="font-weight:700;white-space:nowrap;"></span>
    <button id="sarween-movement-undo" type="button" hidden title="Undo last tracked movement" style="
      width: 28px;
      height: 26px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.10);
      color: white;
      cursor: pointer;
    "><i class="fas fa-rotate-left"></i></button>
    <button id="sarween-movement-reset" type="button" hidden title="Reset movement from the current square" style="
      width: 28px;
      height: 26px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.10);
      color: white;
      cursor: pointer;
    "><i class="fas fa-arrows-rotate"></i></button>
    <button id="sarween-btn" style="
      margin-left: 6px;
      padding: 3px 8px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.25);
      background: rgba(255,255,255,0.10);
      color: white;
      cursor: pointer;
    ">Connect</button>
  `;

  document.body.appendChild(statusEl);
  updateGridSizeInput();
  updateTestTargetButton();
  updateMarkerToggleButton();
  applyStatusPanelPosition();
  enableStatusPanelDragging();

  const gridInput = statusEl.querySelector("#sarween-grid-inches");
  const saveAndApplyGridSize = async () => {
    const value = Math.clamp(Number(gridInput.value) || DEFAULT_GRID_INCHES, 0.25, 3);
    await game.settings.set(MODULE_ID, "targetGridInches", value);
  };
  gridInput.addEventListener("change", saveAndApplyGridSize);
  gridInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      saveAndApplyGridSize();
      gridInput.blur();
    }
  });
  statusEl.querySelector("#sarween-grid-btn").addEventListener("click", () => {
    applyPhysicalGridScale({animate: true, notify: true});
  });
  statusEl.querySelector("#sarween-two-mini-test-btn").addEventListener("click", () => {
    openCaptureSetup();
  });
  statusEl.querySelector("#sarween-marker-toggle-btn").addEventListener("click", () => {
    void toggleViewportMarkers();
  });
  statusEl.querySelector("#sarween-movement-undo").addEventListener("click", () => {
    void undoSelectedMovement();
  });
  statusEl.querySelector("#sarween-movement-reset").addEventListener("click", () => {
    resetSelectedMovement();
  });

  statusEl.querySelector("#sarween-btn").addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      disconnectFromPython("Manual disconnect");
    } else {
      manualDisconnect = false;
      connectToPython();
    }
  });
}

function setStatus(state, extraText = "") {
  ensureStatusUI();

  const dot = statusEl.querySelector("#sarween-dot");
  const text = statusEl.querySelector("#sarween-text");
  const btn  = statusEl.querySelector("#sarween-btn");

  const suffix = extraText ? ` (${extraText})` : "";

  if (state === "connected") {
    dot.style.background = "#00aa44";
    text.textContent = "Sarween: Connected" + suffix;
    btn.textContent = "Disconnect";
  } else if (state === "connecting") {
    dot.style.background = "#d4aa00";
    text.textContent = "Sarween: Connecting" + suffix;
    btn.textContent = "Disconnect";
  } else if (state === "disconnected") {
    dot.style.background = "#aa0000";
    text.textContent = "Sarween: Disconnected" + suffix;
    btn.textContent = "Connect";
  } else if (state === "error") {
    dot.style.background = "#aa0000";
    text.textContent = "Sarween: Error" + suffix;
    btn.textContent = "Connect";
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Send helper + assignment dialog
// ──────────────────────────────────────────────────────────────────────────────

function sendToPython(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    warn("Cannot send to Python (ws not open):", obj);
    return false;
  }
  try {
    ws.send(JSON.stringify(obj));
    return true;
  } catch (err) {
    error("Failed sending message to Python:", err);
    return false;
  }
}

function listSceneTokenChoices(scene) {
  const tokens = (scene?.tokens ?? []).map(t => ({
    tokenId: t.id,
    tokenName: t.name || `(token ${t.id})`,
    actorId: t.actorId || null
  }));
  tokens.sort((a, b) => (a.tokenName || "").localeCompare(b.tokenName || ""));
  return tokens;
}

async function promptAssignMini(miniId) {
  const scene = canvas.scene;
  if (!scene) {
    ui.notifications?.warn("Sarween: No active scene to select tokens from.");
    return;
  }

  const choices = listSceneTokenChoices(scene);
  if (!choices.length) {
    ui.notifications?.warn("Sarween: No tokens found on the current scene.");
    return;
  }

  const optionsHtml = choices
    .map(c => `<option value="${c.tokenId}">${c.tokenName}</option>`)
    .join("\n");

  return new Promise((resolve) => {
    const content = `
      <p><b>Assign scanned mini</b></p>
      <p>Mini ID: <code>${miniId}</code></p>
      <div style="margin-top:8px">
        <label>Move which token?</label>
        <select id="sarween-token-select" style="width:100%; margin-top:6px">
          ${optionsHtml}
        </select>
      </div>
      <p style="opacity:0.8; margin-top:10px">
        Tip: put the token you want on the scene first.
      </p>
    `;

    new Dialog({
      title: "Sarween: Assign Mini → Token",
      content,
      buttons: {
        assign: {
          label: "Assign",
          callback: (html) => {
            const tokenId = html.find("#sarween-token-select").val();
            const chosen = choices.find(c => c.tokenId === tokenId);

            if (!tokenId || !chosen) {
              ui.notifications?.warn("Sarween: No token selected.");
              resolve(null);
              return;
            }

            sendToPython({
              type: "assignMiniResult",
              miniId: String(miniId),
              tokenId: chosen.tokenId,
              actorId: chosen.actorId
            });
            miniTokenBindings.set(String(miniId), String(chosen.tokenId));

            ui.notifications?.info(`Sarween: Assigned mini ${miniId} → ${chosen.tokenName}`);
            resolve(chosen);
          }
        },
        cancel: {
          label: "Cancel",
          callback: () => {
            sendToPython({
              type: "assignMiniResult",
              miniId: String(miniId),
              tokenId: null,
              actorId: null,
              cancelled: true
            });
            resolve(null);
          }
        }
      },
      default: "assign"
    }).render(true);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Scene info payload builder
// ──────────────────────────────────────────────────────────────────────────────

function buildSceneInfoPayload(scene) {
  if (!scene) return null;

  const width = scene.width;
  const height = scene.height;

  const gridSize =
    scene.grid?.size ??
    scene.grid ??
    scene.data?.grid ??
    scene.data?.gridSize ??
    null;

  const shiftX =
    scene.grid?.shiftX ??
    scene.shiftX ??
    scene.data?.shiftX ??
    0;

  const shiftY =
    scene.grid?.shiftY ??
    scene.shiftY ??
    scene.data?.shiftY ??
    0;

  const gridType =
    scene.grid?.type ??
    scene.gridType ??
    scene.data?.gridType ??
    null;

  const background =
    scene.background?.src ??
    scene.img ??
    scene.data?.img ??
    null;

  const miniBindings = Object.fromEntries(CAPTURE_MINIS.flatMap(mini => {
    const matches = (scene.tokens ?? []).filter(token => token.name?.toLowerCase() === mini.tokenName.toLowerCase());
    return matches.length === 1 ? [[mini.miniId, matches[0].id]] : [];
  }));
  for (const [miniId, tokenId] of Object.entries(miniBindings)) {
    miniTokenBindings.set(miniId, tokenId);
  }

  return {
    type: "sceneInfo",
    sceneId: scene.id,
    width,
    height,
    gridSize,
    shiftX,
    shiftY,
    gridType,
    background,
    tokens: (scene.tokens ?? []).map(token => ({id: token.id, name: token.name, actorId: token.actorId})),
    miniBindings,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Reliability helpers
// ──────────────────────────────────────────────────────────────────────────────

function clearTimers() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function scheduleReconnect(reason = "") {
  clearTimers();

  if (manualDisconnect) {
    log("Manual disconnect active; not scheduling reconnect.");
    setStatus("disconnected", "manual");
    return;
  }

  if (!getSetting("autoConnect")) {
    log("Auto-connect disabled; not scheduling reconnect.");
    setStatus("disconnected", "auto-connect off");
    return;
  }

  reconnectAttempts += 1;
  const base = 2000;
  const max  = 30000;
  const delay = Math.min(max, base * Math.pow(2, reconnectAttempts - 1));

  warn(`Connection lost. Reconnecting in ${delay} ms (attempt ${reconnectAttempts})…`, reason);
  setStatus("connecting", `retry in ${Math.round(delay / 1000)}s`);

  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectToPython();
  }, delay);
}

// ──────────────────────────────────────────────────────────────────────────────
// Message handling
// ──────────────────────────────────────────────────────────────────────────────

async function handlePythonMessage(data) {
  if (data?.type !== "renderedReference") log("Received message from Python:", data);

  const { sceneId, tokenId, x, y, type } = data || {};

  if (type === "ping") return;
  if (type === "referenceCapture") {
    setReferenceCapture(Boolean(data.enabled));
    return;
  }
  if (type === "captureStatus") {
    await handleCaptureStatus(data);
    return;
  }
  if (type === "miniTap") {
    const miniId = String(data?.miniId || "").trim();
    if (miniId) togglePhysicalMiniSelection(miniId);
    return;
  }

  // NEW: Python requests scene info
  if (type === "getSceneInfo") {
    const scene = canvas.scene;
    if (!scene) {
      warn("getSceneInfo requested but no active canvas.scene");
      sendToPython({ type: "sceneInfo", error: "no active scene" });
      return;
    }

    const payload = buildSceneInfoPayload(scene);
    if (!payload || !payload.width || !payload.height || !payload.gridSize) {
      warn("sceneInfo payload missing required fields:", payload);
      sendToPython({ type: "sceneInfo", error: "missing fields", ...payload });
      return;
    }

    log("Sending sceneInfo to Python:", payload);
    sendToPython(payload);
    sendViewTransform();
    return;
  }

  // Assign request
  if (type === "assignMini") {
    const miniId = data?.miniId;
    if (!miniId) {
      warn("assignMini missing miniId:", data);
      return;
    }
    await promptAssignMini(String(miniId));
    return;
  }

  // Default behavior: move token
  if (guidedCapture?.recording || guidedCapture?.phase === "Starting") return;
  if (!sceneId || !tokenId || typeof x !== "number" || typeof y !== "number") {
    warn("Invalid payload from Python:", data);
    return;
  }

  const scene = game.scenes.get(sceneId) ?? canvas.scene;
  if (!scene) {
    warn("Scene not found:", sceneId);
    return;
  }

  if (!scene.tokens.get(tokenId)) {
    warn(`Mapped token ${tokenId} no longer exists in scene ${scene.id}`);
    sendToPython({
      type: "tokenMissing",
      sceneId: scene.id,
      tokenId
    });
    return;
  }

  try {
    await scene.updateEmbeddedDocuments(
      "Token",
      [{ _id: tokenId, x, y }],
      { animate: getSetting("animateTokenMovement") }
    );
  } catch (error) {
    warn(`Failed to move token ${tokenId}:`, error);
    sendToPython({
      type: "tokenMoveError",
      sceneId: scene.id,
      tokenId,
      error: String(error?.message ?? error)
    });
    return;
  }
  log(`Moved token ${tokenId} to (${x}, ${y}) in scene ${scene.id}`);
  setTimeout(() => {
    sendToPython({
      type: "sceneVisualChanged",
      sceneId: scene.id,
      tokenId,
      reason: "sarweenTokenMove",
      sentAt: Date.now(),
    });
  }, 150);
}

// ──────────────────────────────────────────────────────────────────────────────
// Connect / Disconnect
// ──────────────────────────────────────────────────────────────────────────────

function connectToPython() {
  ensureStatusUI();

  if (!game.user.isGM) {
    console.log("sarween | Not GM; WebSocket relay disabled.");
    setStatus("disconnected", "not GM");
    return;
  }

  const url = getSetting("wsUrl");
  if (!url) {
    warn("No WebSocket URL configured.");
    setStatus("error", "no wsUrl");
    return;
  }

  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    try { ws.close(1000, "Reconnecting"); } catch {}
  }
  ws = null;

  setStatus("connecting", "opening…");
  log("Connecting to Python WebSocket at", url);

  try {
    ws = new WebSocket(url);
  } catch (err) {
    error("Failed to create WebSocket:", err);
    setStatus("error", "create failed");
    scheduleReconnect("create failed");
    return;
  }

  ws.onopen = () => {
    reconnectAttempts = 0;
    clearTimers();
    manualDisconnect = false;

    console.log(`sarween | WebSocket connected to Python at ${url}.`);
    setStatus("connected");

    try {
      ws.send(JSON.stringify({ type: "hello", source: "foundry-sarween" }));
      if (testSequenceTargets.length) sendTestSequence();
    } catch (err) {
      error("Failed to send hello:", err);
    }

    // Proactively send sceneInfo on connect (helps Foundry mode feel instant)
    try {
      const payload = buildSceneInfoPayload(canvas.scene);
      if (payload) ws.send(JSON.stringify(payload));
      const viewPayload = buildViewTransformPayload();
      if (viewPayload) ws.send(JSON.stringify(viewPayload));
    } catch (err) {
      error("Failed to send initial scene/viewport info:", err);
    }

    heartbeatTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
        } catch (err) {
          error("Failed to send ping:", err);
        }
      }
    }, 10000);
  };

  ws.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      await handlePythonMessage(data);
    } catch (err) {
      error("Error parsing/handling message:", err);
    }
  };

  ws.onclose = (event) => {
    clearTimers();
    stopReferenceCapture();
    ws = null;
    warn("WebSocket closed:", event.code, event.reason || "(no reason)");
    setStatus("disconnected", event.reason || "closed");
    scheduleReconnect(event.reason || "closed");
  };

  ws.onerror = (errEvent) => {
    error("WebSocket error:", errEvent);
    setStatus("error", "socket error");
  };
}

function disconnectFromPython(reason = "Manual disconnect") {
  manualDisconnect = true;
  clearTimers();

  if (ws) {
    try { ws.close(1000, reason); } catch {}
  }
  ws = null;

  warn("Disconnected from Python:", reason);
  setStatus("disconnected", "manual");
}

// ──────────────────────────────────────────────────────────────────────────────
// Start-up
// ──────────────────────────────────────────────────────────────────────────────

Hooks.once("ready", () => {
  if (!game.user.isGM) {
    console.log("sarween | Not GM; WebSocket relay will not start.");
    return;
  }

  ensureStatusUI();
  renderViewportMarkers();

  game.sarween = {
    ...(game.sarween ?? {}),
    applyPhysicalGridScale,
    calculatePhysicalGridScale,
    sendViewTransform,
    toggleTestTargets,
    openCaptureSetup,
    selectMovementToken,
    clearMovementSelection,
    resetSelectedMovement,
  };

  if (getSetting("autoConnect")) {
    console.log("sarween | World ready, starting Python WebSocket connection");
    connectToPython();
  } else {
    setStatus("disconnected", "auto-connect off");
    console.log("sarween | Auto-connect disabled. Use the Sarween UI button to connect.");
  }
});

Hooks.on("canvasReady", () => {
  clearMovementSelection();
  if (guidedCapture?.recording) stopGuidedCapture("sceneChanged");
  testSequenceTargets = [];
  renderTestTargets();
  sendTestSequence();
  renderViewportMarkers();
  setTimeout(async () => {
    if (game.settings.get(MODULE_ID, "physicalGridAutoApply")) {
      await applyPhysicalGridScale({animate: true});
    }
    sendSceneAndView();
  }, 50);
});

Hooks.on("updateScene", (scene, changes) => {
  if (scene.id !== canvas?.scene?.id) return;
  const relevant = Object.keys(changes).some(key => /^(width|height|grid|background|padding)(\.|$)/.test(key));
  if (!relevant) return;
  if (guidedCapture?.recording) stopGuidedCapture("geometryChanged");
  sendSceneAndView();
  scheduleViewTransformSend();
});

Hooks.on("canvasPan", () => {
  if (guidedCapture?.recording && JSON.stringify(getClientToCanvasTransform()) !== guidedCapture.viewSignature) {
    stopGuidedCapture("viewportChanged");
  }
  scheduleViewTransformSend();
  scheduleTestTargetRender();
  renderMovementOverlay();
});

Hooks.on("controlToken", (token, controlled) => {
  if (guidedCapture?.recording) return;
  const document = tokenDocument(token);
  if (controlled) {
    selectMovementToken(document);
  } else if (movementState?.tokenId === document?.id) {
    clearMovementSelection();
  }
});

Hooks.on("updateToken", (document, changes) => {
  recordSelectedMovement(document, changes);
});

Hooks.on("deleteToken", document => {
  if (movementState?.tokenId === document?.id) clearMovementSelection();
});

window.addEventListener("resize", () => {
  renderViewportMarkers();
  renderTestTargets();
  renderMovementOverlay();
  applyStatusPanelPosition();
  if (game.settings.get(MODULE_ID, "physicalGridAutoApply")) {
    applyPhysicalGridScale({animate: false});
  }
  scheduleViewTransformSend();
});
