// module.js for "sarween" — Computer Vision for TTRPG
import {CAPTURE_MINIS, generateCaptureTargets} from "./capture_logic.mjs";
import {
  addMovementPoint,
  createMovementState,
  fallbackGridDistance,
  movementColor,
  movementOverage,
  remainingMovement,
  rebaseMovementAt,
  resetMovementAt,
  undoMovementPoint,
} from "./movement_logic.mjs";

const MODULE_ID = "sarween";
const MIN_CANVAS_ZOOM = 0.1;
const DEFAULT_DISPLAY_DIAGONAL_INCHES = 55;
const DEFAULT_GRID_INCHES = 1;
const VIEWPORT_MARKER_IDS = [10, 11, 12, 13];
const MAX_GRID_LABELS = 3000;
const VISUAL_REFRESH_DELAY_MS = 100;
const VISUAL_FALLBACK_MS = 350;

let ws = null;
let commandMessageChain = Promise.resolve();
let physicalGridSettingsBatch = false;
let physicalGridBatchSettled = Promise.resolve();
let reconnectAttempts = 0;
let reconnectTimer = null;
let heartbeatTimer = null;
let referenceCaptureTimer = null;
let referenceCaptureBusy = false;
let referenceCaptureErrors = 0;
let referenceCaptureSequence = 0;
let referenceCaptureSessionId = null;
let markerOverlayEl = null;
let gridLabelOverlayEl = null;
let gridLabelRenderTimer = null;
let testTargetOverlayEl = null;
let testSequenceTargets = [];
let viewTransformTimer = null;
let testTargetRenderTimer = null;
let guidedCapture = null;
let captureLastStatus = {state: "idle"};
let movementState = null;
let movementOverlayEl = null;
let miniTokenBindings = new Map();
const ignoredMovementUpdates = new Set();
const desktopTokenCreations = new Map();

let connectionStatus = {state: "disconnected", message: ""};
let assignmentRequests = [];
let vttControlsRequested = false;
let lastVttStateSocket = null;
let lastVttStateSignature = "";
let pendingSceneVisualChange = null;

Hooks.on("pauseGame", () => sendVttState());
Hooks.on("createActor", () => sendVttState());
Hooks.on("updateActor", () => sendVttState());
Hooks.on("deleteActor", () => sendVttState());

function getSetting(key) {
  return game.settings.get(MODULE_ID, key);
}

function viewportMarkerGeometry() {
  // Older clients saved the whole tile size with a fixed one-sixth margin.
  // Internal sentinels let existing preferences migrate without guessing whether
  // a newly registered Foundry default was explicitly selected by the user.
  const legacy = Number(getSetting("viewportMarkerSize"));
  const savedPattern = Number(getSetting("viewportMarkerPatternSize"));
  const savedMargin = Number(getSetting("viewportMarkerMargin"));
  const legacySize = Number.isFinite(legacy) && legacy > 0 ? legacy : 120;
  const patternSize = Number.isFinite(savedPattern) && savedPattern >= 1
    ? Math.min(240, savedPattern) : Math.min(240, Math.max(1, legacySize * 2 / 3));
  const margin = Number.isFinite(savedMargin) && savedMargin >= 0
    ? Math.min(12, savedMargin) : Math.min(12, legacySize / 6);
  return {patternSize, margin, size: patternSize + 2 * margin,
          inset: Number(getSetting("viewportMarkerInset"))};
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

const PHYSICAL_GRID_LIMITS = {
  targetGridInches: [0.25, 3],
  displayDiagonalInches: [10, 150],
  physicalGridCalibration: [0.5, 1.5],
};

const VTT_SETTING_RULES = {
  ...PHYSICAL_GRID_LIMITS,
  debug: "boolean", physicalGridAutoApply: "boolean",
  viewportMarkersEnabled: "boolean", viewportMarkerSize: [1, 240], viewportMarkerInset: [0, 80],
  viewportMarkerPatternSize: [1, 240], viewportMarkerMargin: [0, 12],
  animateTokenMovement: "boolean", physicalTapSelection: "boolean", defaultMovementSpeed: [0, 120],
  foundryUIHidden: "boolean", gridLabelsVisible: "boolean",
};
const VTT_ACTIONS = [
  "settings.set", "markers.set", "capture.start", "capture.confirm", "capture.stop", "capture.new",
  "capture.resume", "movement.select", "movement.undo", "movement.reset", "assignment.save",
  "assignment.cancel", "ui.setHidden", "lifecycle.reload", "state.sync",
  "lifecycle.setPaused", "token.create",
];

function captureIsActive() {
  return Boolean(guidedCapture?.recording || ["Starting", "Saving", "Restoring"].includes(guidedCapture?.phase));
}

function vttState() {
  const scene = canvas?.scene;
  const tokens = listSceneTokenChoices(scene);
  const target = guidedCapture?.phase === "placement" ? guidedCapture.targets?.[guidedCapture.index] ?? null : null;
  return {
    type: "vttState", supported: true, ready: Boolean(canvas?.ready && scene),
    sceneId: scene?.id ?? null, capabilities: VTT_ACTIONS,
    connection: {...connectionStatus},
    world: {id: game.world?.id ?? null, name: game.world?.title ?? game.world?.name ?? ""},
    user: {id: game.user?.id ?? null, name: game.user?.name ?? "", isGM: Boolean(game.user?.isGM)},
    paused: Boolean(game.paused),
    settings: {...Object.fromEntries(Object.keys(VTT_SETTING_RULES).map(key => [key, getSetting(key)])),
      viewportMarkerSize: viewportMarkerGeometry().size,
      viewportMarkerPatternSize: viewportMarkerGeometry().patternSize,
      viewportMarkerMargin: viewportMarkerGeometry().margin},
    scene: scene ? {id: scene.id, name: scene.name ?? "", gridSize: Number(scene.grid?.size ?? 0),
      gridType: Number(scene.grid?.type ?? 0), tokenVision: Boolean(scene.tokenVision)} : null,
    tokens,
    actors: listWorldActorChoices(),
    miniBindings: Object.fromEntries([...miniTokenBindings].filter(([_mini, tokenId]) =>
      tokens.some(token => token.tokenId === tokenId))),
    minis: CAPTURE_MINIS.map(mini => {
      const matches = tokens.filter(token => token.tokenName.toLowerCase() === mini.tokenName.toLowerCase());
      const assigned = miniTokenBindings.get(mini.miniId);
      return {...mini, tokenId: tokens.some(token => token.tokenId === assigned) ? assigned
        : matches.length === 1 ? matches[0].tokenId : null};
    }),
    assignment: assignmentRequests.length ? {...assignmentRequests[0], pendingCount: assignmentRequests.length} : null,
    capture: {
      state: guidedCapture?.phase === "Starting" ? "starting" : captureLastStatus.state,
      sessionId: guidedCapture?.sessionId ?? null, recording: Boolean(guidedCapture?.recording), active: captureIsActive(),
      waiting: Boolean(guidedCapture?.waiting), phase: guidedCapture?.phase ?? "Idle",
      seconds: Number(guidedCapture?.seconds ?? 0), index: Number(guidedCapture?.index ?? 0),
      total: guidedCapture?.targets?.length ?? 0, target,
      message: guidedCapture?.message ?? captureLastStatus.message ?? "",
      reason: captureLastStatus.reason ?? null,
    },
    movement: movementState ? {
      tokenId: movementState.tokenId, miniId: movementState.miniId, label: movementState.label,
      budgetFeet: movementState.budgetFeet, usedFeet: movementState.usedFeet,
      remainingFeet: remainingMovement(movementState), overageFeet: movementOverage(movementState),
      canUndo: movementState.segments.length > 0,
    } : null,
  };
}

function sendVttState({force = false} = {}) {
  if (!vttControlsRequested || physicalGridSettingsBatch || ws?.readyState !== WebSocket.OPEN) return;
  const state = vttState();
  const signature = JSON.stringify(state);
  if (!force && ws === lastVttStateSocket && signature === lastVttStateSignature) return;
  if (sendToPython(state)) {lastVttStateSocket = ws; lastVttStateSignature = signature;}
}

function registerVttSetting(key, registration) {
  const onChange = registration.onChange;
  game.settings.register(MODULE_ID, key, {...registration, onChange: async value => {
    if (physicalGridSettingsBatch) return;
    try {await onChange?.(value);}
    finally {sendVttState();}
  }});
}

function applyFoundryUIVisibility() {
  let style = document.getElementById?.("sarween-hide-foundry-ui");
  if (!getSetting("foundryUIHidden") || ws?.readyState !== WebSocket.OPEN
      || connectionStatus.state !== "connected") {style?.remove(); return;}
  if (style) return;
  style = document.createElement("style");
  style.id = "sarween-hide-foundry-ui";
  style.textContent = `#ui-left,#ui-top,#ui-bottom,#sidebar,#navigation,#players,#hotbar,#controls,#logo {
    visibility:hidden!important;pointer-events:none!important;
  }`;
  document.head.appendChild(style);
}

function requireKeys(values, allowed) {
  if (!values || typeof values !== "object" || Array.isArray(values)
      || Object.keys(values).some(key => !allowed.includes(key))) throw new Error("Invalid VTT command values.");
}

function requireIdleCapture() {
  if (captureIsActive()) throw new Error("Finish guided capture before changing the VTT display.");
}

function validateVttSettings(settings) {
  requireKeys(settings, Object.keys(VTT_SETTING_RULES));
  for (const [key, value] of Object.entries(settings)) {
    const rule = VTT_SETTING_RULES[key];
    if (Array.isArray(rule)) {
      if (typeof value !== "number" || !Number.isFinite(value) || value < rule[0] || value > rule[1]) {
        throw new Error(`Invalid value for ${key}.`);
      }
    } else if (rule === "boolean") {
      if (typeof value !== "boolean") throw new Error(`Invalid value for ${key}.`);
    }
  }
}

async function setVttSettings(settings, requireCurrent) {
  validateVttSettings(settings);
  requireIdleCapture();
  settings = {...settings};
  if (Object.hasOwn(settings, "viewportMarkerSize")) {
    if (!Object.hasOwn(settings, "viewportMarkerPatternSize")) {
      settings.viewportMarkerPatternSize = Math.max(1, settings.viewportMarkerSize * 2 / 3);
    }
    if (!Object.hasOwn(settings, "viewportMarkerMargin")) {
      settings.viewportMarkerMargin = Math.min(12, settings.viewportMarkerSize / 6);
    }
  }
  const changed = Object.entries(settings).filter(([key, value]) => getSetting(key) !== value);
  const previous = Object.fromEntries(changed.map(([key]) => [key, getSetting(key)]));
  // Applying a saved display profile also restores the physical zoom after a
  // manual zoom, even when its persisted grid settings have not changed.
  const geometry = Object.keys(settings).some(key => Object.hasOwn(PHYSICAL_GRID_LIMITS, key));
  const markers = changed.some(([key]) => key.startsWith("viewportMarker"));
  const labels = changed.some(([key]) => key === "gridLabelsVisible");
  const sceneId = canvas?.scene?.id;
  let attemptedApply = false;
  let finishBatch;
  physicalGridBatchSettled = new Promise(resolve => {finishBatch = resolve;});
  physicalGridSettingsBatch = true;
  try {
    for (const [key, value] of changed) {
      requireCurrent();
      await game.settings.set(MODULE_ID, key, value);
    }
    requireCurrent();
    if (geometry) {
      attemptedApply = true;
      if (!await applyPhysicalGridScale({animate: false})) throw new Error("Open a scene before resizing its grid.");
      requireCurrent();
    }
    if (markers) renderViewportMarkers();
    if (labels) gridLabelsChanged();
    else if (geometry) renderGridLabels();
    applyFoundryUIVisibility();
    sendSceneAndView();
  } catch (err) {
    let restored = true;
    for (const [key, value] of Object.entries(previous)) {
      try {await game.settings.set(MODULE_ID, key, value);} catch {restored = false;}
    }
    try {
      if (markers) renderViewportMarkers();
      if (labels) renderGridLabels();
      applyFoundryUIVisibility();
      if (attemptedApply && canvas?.ready && canvas.scene?.id === sceneId) {
        if (!await applyPhysicalGridScale({animate: false})) restored = false;
        sendSceneAndView();
      }
    } catch {restored = false;}
    throw new Error(restored ? `${err.message} Previous settings restored.`
      : "VTT update failed. Check the current Foundry settings.");
  } finally {
    physicalGridSettingsBatch = false;
    finishBatch();
    sendPhysicalGridState();
    sendVttState();
  }
}

function startGuidedCapture(values) {
  requireKeys(values, ["participants", "scenario", "notes"]);
  requireIdleCapture();
  if (physicalGridSettingsBatch) throw new Error("Wait for physical grid resizing to finish before starting capture.");
  if (!getSetting("viewportMarkersEnabled")) throw new Error("Turn on the ArUco markers first.");
  if (Number(canvas.scene.grid?.type) !== 1) throw new Error("Dataset capture requires a square Foundry grid.");
  if (!Array.isArray(values.participants) || !values.participants.length
      || values.participants.length > CAPTURE_MINIS.length) throw new Error("Choose at least one mini.");
  if (!["fog-on", "fog-off"].includes(values.scenario)) throw new Error("Choose a fog / lighting label.");
  if (values.notes !== undefined && (typeof values.notes !== "string" || values.notes.length > 2000)) throw new Error("Capture notes must be under 2000 characters.");
  const participants = values.participants.map(value => {
    requireKeys(value, ["miniId", "tokenId"]);
    const mini = CAPTURE_MINIS.find(item => item.miniId === value.miniId);
    const token = canvas.scene.tokens.get(value.tokenId);
    if (!mini || !token) throw new Error("Choose a current token for every mini.");
    if (Number(token.width) !== 1 || Number(token.height) !== 1) throw new Error(`${mini.tokenName} must be a one-square token.`);
    return {...mini, tokenId: token.id, sight: token.sight?.toObject?.() ?? token.sight};
  });
  if (new Set(participants.map(mini => mini.tokenId)).size !== participants.length
      || new Set(participants.map(mini => mini.miniId)).size !== participants.length) throw new Error("Choose a different token and ring for each mini.");
  if (values.scenario === "fog-on") {
    if (!canvas.scene.tokenVision) throw new Error("Enable Token Vision in the scene before a fog-on capture.");
    const blind = participants.find(mini => !mini.sight?.enabled || !mini.sight?.range);
    if (blind) throw new Error(`Enable vision and a sight range for ${blind.tokenName}.`);
  }
  const targets = generateCaptureTargets(visibleGridBounds({safeForCapture: true}), participants);
  cancelPendingSceneVisualChange();
  clearMovementSelection();
  guidedCapture = {
    sessionId: crypto.randomUUID(), sceneId: canvas.scene.id, participants, targets,
    index: 0, recording: false, waiting: true, phase: "Starting", seconds: 0, timer: null,
    viewSignature: JSON.stringify(getClientToCanvasTransform()),
    originalTokens: participants.map(mini => {const token = canvas.scene.tokens.get(mini.tokenId); return {id: token.id, x: token.x, y: token.y};}),
    originalControlIds: (canvas.tokens.controlled ?? []).map(token => token.id),
  };
  captureLastStatus = {state: "starting"};
  canvas.tokens.releaseAll();
  for (const mini of participants) canvas.tokens.get(mini.tokenId)?.control({releaseOthers: false});
  sendSceneAndView();
  if (!captureControl("start", {
    ...guidedCapture, timer: undefined, scenario: values.scenario, notes: values.notes ?? "",
    moduleVersion: game.modules.get(MODULE_ID)?.version,
    sceneVision: {tokenVision: canvas.scene.tokenVision, fogExploration: canvas.scene.fog?.exploration},
    markerSettings: viewportMarkerGeometry(),
  })) {guidedCapture = null; throw new Error("Could not start capture: connection lost.");}
  publishCaptureState();
}

function assignMiniFromDesktop(values) {
  requireKeys(values, ["miniId", "tokenId"]);
  if (typeof values.miniId !== "string" || !values.miniId.trim() || values.miniId.length > 128) throw new Error("Choose a mini to assign.");
  const chosen = listSceneTokenChoices(canvas.scene).find(token => token.tokenId === values.tokenId);
  if (!chosen) throw new Error("The selected token is no longer on this scene.");
  sendToPython({type: "assignMiniResult", sceneId: canvas.scene.id, miniId: values.miniId,
    tokenId: chosen.tokenId, actorId: chosen.actorId});
  miniTokenBindings.set(values.miniId, chosen.tokenId);
  assignmentRequests = assignmentRequests.filter(request => request.miniId !== values.miniId);
  sendSceneAndView();
}

function actorCanCreateToken(actor) {
  return Boolean(game.user?.isGM && actor && typeof actor.getTokenDocument === "function"
    && game.user.can?.("TOKEN_CREATE") !== false
    && actor.testUserPermission?.(game.user, "OBSERVER") !== false
    && canvas.scene?.canUserModify?.(game.user, "update") !== false);
}

function listWorldActorChoices() {
  const actors = game.actors?.contents ?? (game.actors?.values ? [...game.actors.values()] : []);
  return actors.filter(actorCanCreateToken).map(actor => ({actorId: actor.id,
    actorName: actor.name || `(actor ${actor.id})`, type: actor.type || ""}))
    .sort((a, b) => a.actorName.localeCompare(b.actorName) || a.actorId.localeCompare(b.actorId));
}

function validateTokenCreation(values) {
  requireKeys(values, ["actorId", "name", "placement", "row", "col", "creationId"]);
  if (typeof values.actorId !== "string" || !values.actorId || values.actorId.length > 128
      || typeof values.name !== "string" || values.name.length > 100 || /[\x00-\x1f]/.test(values.name)
      || typeof values.creationId !== "string" || !/^[a-zA-Z0-9_-]{16,64}$/.test(values.creationId)) {
    throw new Error("Choose an actor and a valid token name.");
  }
  if (!["cell", "viewport"].includes(values.placement)) throw new Error("Choose a token placement.");
  if (values.placement === "cell") {
    if (![values.row, values.col].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 100000)) {
      throw new Error("Choose a valid grid cell.");
    }
  } else if (values.row !== undefined || values.col !== undefined) throw new Error("Viewport placement has no grid cell.");
}

function desktopTokenPosition(values, data) {
  const scene = canvas.scene, grid = Number(scene.grid?.size);
  const left = Number(canvas.dimensions?.sceneX ?? 0), top = Number(canvas.dimensions?.sceneY ?? 0);
  const width = Number(scene.width), height = Number(scene.height);
  const tokenWidth = Number(data.width) * grid, tokenHeight = Number(data.height) * grid;
  if (![grid, width, height, tokenWidth, tokenHeight].every(value => Number.isFinite(value) && value > 0)
      || tokenWidth > width || tokenHeight > height) throw new Error("The prototype token does not fit this scene.");
  const originX = left + Number(scene.grid?.shiftX ?? 0), originY = top + Number(scene.grid?.shiftY ?? 0);
  let x, y;
  if (values.placement === "cell") {
    if (Number(scene.grid?.type) !== 1) throw new Error("Cell placement requires a square grid.");
    x = originX + values.col * grid; y = originY + values.row * grid;
  } else {
    const inverse = getClientToCanvasTransform();
    if (!inverse) throw new Error("The current viewport is unavailable.");
    const cx = Number(window.innerWidth) / 2, cy = Number(window.innerHeight) / 2;
    const centerX = inverse.a * cx + inverse.c * cy + inverse.tx;
    const centerY = inverse.b * cx + inverse.d * cy + inverse.ty;
    x = originX + Math.round((centerX - tokenWidth / 2 - originX) / grid) * grid;
    y = originY + Math.round((centerY - tokenHeight / 2 - originY) / grid) * grid;
    x = Math.min(left + width - tokenWidth, Math.max(left, x));
    y = Math.min(top + height - tokenHeight, Math.max(top, y));
  }
  if (![x, y].every(Number.isFinite) || x < left || y < top
      || x + tokenWidth > left + width || y + tokenHeight > top + height) {
    throw new Error("The token would be outside this scene. Choose another cell.");
  }
  return {x, y};
}

async function createDesktopToken(values, requireCurrent) {
  validateTokenCreation(values);
  requireIdleCapture();
  requireCurrent();
  const scene = canvas.scene, actor = game.actors?.get?.(values.actorId);
  if (!actorCanCreateToken(actor)) throw new Error("The actor is unavailable or you cannot create its token.");
  const signature = JSON.stringify({actorId: values.actorId, name: values.name.trim(), placement: values.placement,
    ...(values.placement === "cell" ? {row: values.row, col: values.col} : {})});
  const existing = (scene.tokens ?? []).find(token => token.flags?.[MODULE_ID]?.desktopCreation?.id === values.creationId);
  if (existing) {
    if (existing.flags[MODULE_ID].desktopCreation.signature !== signature) throw new Error("This creation ID belongs to a different token request.");
    return {tokenId: existing.id, creationId: values.creationId, name: existing.name};
  }
  const key = `${game.world?.id ?? ""}:${scene.id}:${values.creationId}`;
  if (desktopTokenCreations.has(key)) {
    const pending = desktopTokenCreations.get(key);
    if (pending.signature !== signature) throw new Error("This creation ID belongs to a different token request.");
    return pending.promise;
  }
  const create = async () => {
    // getTokenDocument preserves system-specific prototype texture, sight and
    // linking. It creates an unsaved document; only createEmbeddedDocuments writes.
    const prototype = await actor.getTokenDocument({name: values.name.trim() || actor.name}, {parent: scene});
    requireCurrent(); requireIdleCapture();
    if (!actorCanCreateToken(actor)) throw new Error("Token creation permission changed.");
    const data = prototype.toObject();
    delete data._id;
    Object.assign(data, desktopTokenPosition(values, data), {actorId: actor.id, name: values.name.trim() || actor.name});
    data.flags = {...data.flags, [MODULE_ID]: {...data.flags?.[MODULE_ID],
      desktopCreation: {id: values.creationId, signature}}};
    const created = await scene.createEmbeddedDocuments("Token", [data], {sarweenSource: "desktopTokenCreate"});
    if (!created?.[0]?.id) throw new Error("Foundry did not confirm token creation.");
    requireCurrent();
    sendSceneAndView();
    return {tokenId: created[0].id, creationId: values.creationId, name: created[0].name};
  };
  const pending = {signature, promise: create()};
  desktopTokenCreations.set(key, pending);
  try {return await pending.promise;}
  finally {if (desktopTokenCreations.get(key) === pending) desktopTokenCreations.delete(key);}
}

async function handleVttCommand(data, socket) {
  const current = () => socket === ws && socket?.readyState === WebSocket.OPEN;
  const initialSceneId = canvas?.scene?.id ?? null;
  const requireCurrent = () => {
    if (!current()) throw new Error("The Sarween connection changed. Try again.");
    if (initialSceneId !== (canvas?.scene?.id ?? null)
        || (data.sceneId !== undefined && data.sceneId !== initialSceneId)) throw new Error("The Foundry scene changed. Refresh and try again.");
  };
  let reload = false;
  let completed = false;
  let actionResult = null;
  try {
    requireCurrent();
    if (!game.user?.isGM) throw new Error("VTT controls require a Foundry GM connection.");
    if (typeof data.requestId !== "string" || !data.requestId || !VTT_ACTIONS.includes(data.action)) throw new Error("Unsupported VTT command.");
    const values = data.values ?? {};
    requireKeys(values, {
      "settings.set": ["settings"], "markers.set": ["enabled"], "capture.start": ["participants", "scenario", "notes"],
      "movement.select": ["tokenId"], "assignment.save": ["miniId", "tokenId"], "ui.setHidden": ["hidden"],
      "lifecycle.setPaused": ["paused"],
      "token.create": ["actorId", "name", "placement", "row", "col", "creationId"],
    }[data.action] ?? []);
    if (!["settings.set", "ui.setHidden", "lifecycle.reload", "lifecycle.setPaused", "state.sync"].includes(data.action)
        && (!canvas?.ready || !canvas.scene)) throw new Error("Open a Foundry scene first.");
    switch (data.action) {
      case "settings.set": await setVttSettings(values.settings, requireCurrent); break;
      case "markers.set": await setVttSettings({viewportMarkersEnabled: values.enabled}, requireCurrent); break;
      case "ui.setHidden": await setVttSettings({foundryUIHidden: values.hidden}, requireCurrent); break;
      case "capture.start": startGuidedCapture(values); break;
      case "capture.confirm": {
        const target = guidedCapture?.targets?.[guidedCapture.index];
        if (!guidedCapture?.recording || guidedCapture.waiting || !target || guidedCapture.phase !== "placement") throw new Error("Wait for the next capture placement prompt.");
        guidedCapture.waiting = true;
        guidedCapture.message = "Confirming placement…";
        if (!captureControl("confirm", {index: target.index})) throw new Error("Capture connection lost.");
        break;
      }
      case "capture.stop":
        if (!captureIsActive()) throw new Error("There is no active capture to finish.");
        if (guidedCapture.phase === "Saving") throw new Error("The capture is already being saved.");
        stopGuidedCapture(); break;
      case "capture.new":
        requireIdleCapture(); clearInterval(guidedCapture?.timer); guidedCapture = null; captureLastStatus = {state: "idle"};
        testSequenceTargets = []; renderTestTargets(); sendTestSequence(); break;
      case "capture.resume":
        requireIdleCapture();
        if (!captureControl("resume")) throw new Error("Could not resume tracking.");
        break;
      case "movement.select": {
        requireIdleCapture();
        if (values.tokenId === null) {canvas.tokens.releaseAll(); clearMovementSelection(); break;}
        const token = canvas.scene.tokens.get(values.tokenId);
        if (!token || !miniForToken(token)) throw new Error("Choose a token assigned to a mini.");
        if (movementState?.tokenId === values.tokenId) break;
        canvas.tokens.get(values.tokenId)?.control?.({releaseOthers: true});
        if (movementState?.tokenId !== values.tokenId) selectMovementToken(token);
        break;
      }
      case "movement.undo":
        requireIdleCapture(); if (!movementState?.segments.length) throw new Error("There is no movement to undo.");
        await undoSelectedMovement(); break;
      case "movement.reset":
        requireIdleCapture(); if (!movementState) throw new Error("Select a mini first.");
        resetSelectedMovement(); break;
      case "assignment.save": requireIdleCapture(); assignMiniFromDesktop(values); break;
      case "token.create": actionResult = await createDesktopToken(values, requireCurrent); break;
      case "assignment.cancel":
        if (assignmentRequests.length) sendToPython({type: "assignMiniResult", ...assignmentRequests.shift(), tokenId: null, actorId: null, cancelled: true});
        break;
      case "lifecycle.reload": requireIdleCapture(); reload = true; break;
      case "lifecycle.setPaused":
        requireIdleCapture();
        if (typeof values.paused !== "boolean") throw new Error("Choose an absolute pause state.");
        if (typeof game.togglePause !== "function") throw new Error("This Foundry version cannot change pause state.");
        await game.togglePause(values.paused, {broadcast: true});
        requireCurrent();
        if (Boolean(game.paused) !== values.paused) throw new Error("Foundry did not confirm the requested pause state.");
        break;
    }
    requireCurrent();
    sendToPython({type: "vttResult", requestId: data.requestId, success: true,
      message: actionResult ? `Created ${actionResult.name}.` : "VTT controls updated.",
      ...(actionResult ? {result: actionResult} : {}), state: vttState()});
    completed = true;
  } catch (err) {
    if (current()) sendToPython({type: "vttResult", requestId: data.requestId, success: false,
      message: String(err?.message ?? err), state: vttState()});
  } finally {sendVttState();}
  if (completed && current() && reload) setTimeout(() => {if (current() && !captureIsActive()) window.location.reload();}, 100);
}

function physicalGridSettings() {
  return Object.fromEntries(Object.keys(PHYSICAL_GRID_LIMITS)
    .map(key => [key, Number(getSetting(key))]));
}

function physicalGridState() {
  return {
    type: "physicalGridState",
    supported: true,
    ready: calculatePhysicalGridScale() !== null,
    sceneId: canvas?.scene?.id ?? null,
    settings: physicalGridSettings(),
  };
}

function sendPhysicalGridState() {
  // A get/reconnect during persistence must not publish half of a desktop edit.
  if (physicalGridSettingsBatch) return;
  sendToPython(physicalGridState());
}

async function physicalGridSettingsChanged() {
  if (physicalGridSettingsBatch) return;
  try {
    await applyPhysicalGridScale({animate: false});
  } finally {
    sendPhysicalGridState();
    sendVttState();
  }
}

async function setPhysicalGridFromPython(data, socket = ws) {
  const connectionIsCurrent = () => socket === ws && socket?.readyState === WebSocket.OPEN;
  const reply = (success, message) => connectionIsCurrent() && sendToPython({
    ...physicalGridState(), type: "physicalGridResult", requestId: data.requestId,
    success, message,
  });
  if (!connectionIsCurrent()) return;
  const settings = data.settings;
  if (typeof data.requestId !== "string" || !data.requestId
      || !settings || typeof settings !== "object" || Array.isArray(settings)
      || (data.apply !== undefined && typeof data.apply !== "boolean")) {
    reply(false, "Invalid physical grid request.");
    return;
  }
  const entries = Object.entries(settings);
  if (entries.some(([key, value]) => {
    const limits = PHYSICAL_GRID_LIMITS[key];
    return !Object.hasOwn(PHYSICAL_GRID_LIMITS, key) || typeof value !== "number"
      || !Number.isFinite(value) || value < limits[0] || value > limits[1];
  })) {
    reply(false, "Physical grid settings are outside the supported range.");
    return;
  }
  if (captureIsActive()) {
    reply(false, "Finish guided capture before resizing the physical grid.");
    return;
  }
  if (!canvas?.ready || !canvas.scene || calculatePhysicalGridScale() === null) {
    reply(false, "Open a Foundry scene before resizing the physical grid.");
    return;
  }
  if (data.sceneId !== undefined && data.sceneId !== canvas.scene.id) {
    reply(false, "The Foundry scene changed. Refresh and try again.");
    return;
  }

  const sceneId = canvas.scene.id;
  const previous = physicalGridSettings();
  const changed = entries.filter(([key, value]) => previous[key] !== value);
  let attemptedApply = false;
  let finishBatch;
  physicalGridBatchSettled = new Promise(resolve => {finishBatch = resolve;});
  const requireCurrentContext = () => {
    if (!connectionIsCurrent()) {
      throw new Error("The Sarween connection changed before grid resizing finished.");
    }
    if (!canvas?.ready || canvas.scene?.id !== sceneId) {
      throw new Error("The Foundry scene changed before grid resizing finished.");
    }
    if (captureIsActive()) {
      throw new Error("Guided capture started before grid resizing finished.");
    }
  };
  physicalGridSettingsBatch = true;
  try {
    for (const [key, value] of changed) {
      requireCurrentContext();
      await game.settings.set(MODULE_ID, key, value);
    }
    requireCurrentContext();
    if (data.apply !== false) {
      attemptedApply = true;
      if (!await applyPhysicalGridScale({animate: false})) {
        throw new Error("Foundry could not apply the physical grid size.");
      }
      requireCurrentContext();
      sendSceneAndView();
    }
    reply(true, data.apply === false ? "Physical grid settings saved." : "Physical grid size applied.");
  } catch (err) {
    // A failed settings write must not leave a partly applied desktop edit.
    let restored = true;
    for (const [key] of changed) {
      try { await game.settings.set(MODULE_ID, key, previous[key]); }
      catch { restored = false; }
    }
    if (attemptedApply && canvas?.ready && canvas.scene?.id === sceneId) {
      try {
        if (!await applyPhysicalGridScale({animate: false})) restored = false;
        sendSceneAndView();
      }
      catch { restored = false; }
    }
    reply(false, restored ? "Physical grid update failed. Previous settings restored."
      : "Physical grid update failed. Check the current Foundry grid settings.");
    warn("Physical grid update failed:", err);
  } finally {
    physicalGridSettingsBatch = false;
    finishBatch();
    sendPhysicalGridState();
    sendVttState();
  }
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
  else await canvas.pan(position);

  log(`Applied physical grid scale ${scale.toFixed(4)}`);
  if (notify) {
    const inches = Number(getSetting("targetGridInches"));
    ui.notifications?.info(`Sarween: grid set to ${inches.toFixed(2)} inches per square.`);
  }
  return true;
}

function removeViewportMarkers() {
  markerOverlayEl?.remove();
  markerOverlayEl = null;
}

function renderViewportMarkers() {
  removeViewportMarkers();
  if (!getSetting("viewportMarkersEnabled")) return;

  const {size, patternSize, margin, inset} = viewportMarkerGeometry();
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
    const plate = document.createElement("div");
    plate.dataset.markerId = String(id);
    plate.style.cssText = `
      position: absolute;
      width: ${size}px;
      height: ${size}px;
      ${corners[index][0]}: ${inset}px;
      ${corners[index][1]}: ${inset}px;
      background: #fff;
    `;
    const pattern = document.createElement("div");
    pattern.style.cssText = `
      position: absolute;
      left: ${margin}px;
      top: ${margin}px;
      width: ${patternSize}px;
      height: ${patternSize}px;
      overflow: hidden;
    `;
    const image = document.createElement("img");
    image.src = `modules/${MODULE_ID}/viewport_markers/marker_${id}.png`;
    image.alt = "";
    image.dataset.markerId = String(id);
    image.style.cssText = `
      position: absolute;
      width: ${patternSize * 1.5}px;
      height: ${patternSize * 1.5}px;
      max-width: none;
      left: ${-patternSize / 4}px;
      top: ${-patternSize / 4}px;
      image-rendering: pixelated;
      background: #fff;
    `;
    pattern.appendChild(image);
    plate.appendChild(pattern);
    markerOverlayEl.appendChild(plate);
  });
  document.body.appendChild(markerOverlayEl);
}

function canvasPointToClient(x, y, transform = canvas.stage.worldTransform) {
  return {
    x: transform.a * x + transform.c * y + transform.tx,
    y: transform.b * x + transform.d * y + transform.ty,
  };
}

function gridColumnName(column) {
  let name = "";
  for (let value = column + 1; value > 0; value = Math.floor((value - 1) / 26)) {
    name = String.fromCharCode(65 + (value - 1) % 26) + name;
  }
  return name;
}

function gridLabelLayout() {
  if (!getSetting("gridLabelsVisible") || !canvas?.ready || !canvas.scene
      || Number(canvas.scene.grid?.type) !== 1) return null;
  const width = Number(window.innerWidth), height = Number(window.innerHeight);
  const grid = Number(canvas.scene.grid?.size);
  const sceneWidth = Number(canvas.scene.width), sceneHeight = Number(canvas.scene.height);
  const originX = Number(canvas.dimensions?.sceneX ?? 0) + Number(canvas.scene.grid?.shiftX ?? 0);
  const originY = Number(canvas.dimensions?.sceneY ?? 0) + Number(canvas.scene.grid?.shiftY ?? 0);
  if (![width, height, grid, sceneWidth, sceneHeight].every(value => Number.isFinite(value) && value > 0)
      || ![originX, originY].every(Number.isFinite)) return null;
  const inverse = getClientToCanvasTransform();
  if (!inverse || !Object.values(inverse).every(Number.isFinite)) return null;
  const determinant = inverse.a * inverse.d - inverse.b * inverse.c;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;
  // Invert the same client mapping used by tracking, including Foundry's native
  // client-coordinate conversion when the canvas has a CSS offset or scale.
  const forward = {
    a: inverse.d / determinant, b: -inverse.b / determinant,
    c: -inverse.c / determinant, d: inverse.a / determinant,
    tx: (inverse.c * inverse.ty - inverse.d * inverse.tx) / determinant,
    ty: (inverse.b * inverse.tx - inverse.a * inverse.ty) / determinant,
  };
  const corners = [[0, 0], [width, 0], [width, height], [0, height]].map(([x, y]) => ({
    x: inverse.a * x + inverse.c * y + inverse.tx,
    y: inverse.b * x + inverse.d * y + inverse.ty,
  }));
  const firstColumn = Math.max(0, Math.floor((Math.min(...corners.map(point => point.x)) - originX) / grid));
  const lastColumn = Math.min(Math.ceil(sceneWidth / grid) - 1,
    Math.floor((Math.max(...corners.map(point => point.x)) - originX) / grid));
  const firstRow = Math.max(0, Math.floor((Math.min(...corners.map(point => point.y)) - originY) / grid));
  const lastRow = Math.min(Math.ceil(sceneHeight / grid) - 1,
    Math.floor((Math.max(...corners.map(point => point.y)) - originY) / grid));
  if (firstColumn > lastColumn || firstRow > lastRow) return null;
  const columns = lastColumn - firstColumn + 1, rows = lastRow - firstRow + 1;
  const count = columns * rows;
  const stride = Math.max(1, Math.ceil(Math.sqrt(count / MAX_GRID_LABELS)),
    Math.ceil(columns / MAX_GRID_LABELS), Math.ceil(rows / MAX_GRID_LABELS));
  const labels = [];
  // Keep a consistent sampling lattice when an extreme zoom-out needs fewer
  // labels. Normal tabletop views label every visible square.
  for (let row = Math.ceil(firstRow / stride) * stride; row <= lastRow; row += stride) {
    for (let column = Math.ceil(firstColumn / stride) * stride; column <= lastColumn; column += stride) {
      const point = canvasPointToClient(originX + (column + .04) * grid,
        originY + (row + .04) * grid, forward);
      if (point.x < 0 || point.y < 0 || point.x > width - 8 || point.y > height - 12) continue;
      labels.push({text: `${gridColumnName(column)}${row + 1}`, x: point.x, y: point.y});
      if (labels.length >= MAX_GRID_LABELS) return {width, height, labels};
    }
  }
  return {width, height, labels};
}

function removeGridLabels() {
  gridLabelOverlayEl?.remove();
  gridLabelOverlayEl = null;
}

function renderGridLabels() {
  const layout = gridLabelLayout();
  if (!layout || !layout.labels.length) {removeGridLabels(); return;}
  if (!gridLabelOverlayEl) {
    gridLabelOverlayEl = document.createElement("canvas");
    gridLabelOverlayEl.id = "sarween-grid-labels";
    gridLabelOverlayEl.setAttribute("aria-hidden", "true");
    gridLabelOverlayEl.style.cssText = "position:fixed;inset:0;z-index:9997;pointer-events:none;user-select:none;";
    document.body.appendChild(gridLabelOverlayEl);
  }
  const ratio = Math.min(2, Math.max(1, Number(window.devicePixelRatio) || 1));
  const pixelWidth = Math.ceil(layout.width * ratio), pixelHeight = Math.ceil(layout.height * ratio);
  if (gridLabelOverlayEl.width !== pixelWidth) gridLabelOverlayEl.width = pixelWidth;
  if (gridLabelOverlayEl.height !== pixelHeight) gridLabelOverlayEl.height = pixelHeight;
  gridLabelOverlayEl.style.width = `${layout.width}px`;
  gridLabelOverlayEl.style.height = `${layout.height}px`;
  const context = gridLabelOverlayEl.getContext("2d");
  if (!context) {removeGridLabels(); return;}
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, layout.width, layout.height);
  context.font = "10px sans-serif";
  context.textBaseline = "top";
  context.lineWidth = 2;
  context.strokeStyle = "rgba(0,0,0,.7)";
  context.fillStyle = "rgba(255,255,255,.72)";
  for (const label of layout.labels) {
    context.strokeText(label.text, label.x, label.y);
    context.fillText(label.text, label.x, label.y);
  }
}

function gridLabelsChanged() {
  renderGridLabels();
  sendToPython({type: "sceneVisualChanged", reason: "gridLabelsChanged"});
}

function scheduleGridLabelRender() {
  if (gridLabelRenderTimer) return;
  gridLabelRenderTimer = setTimeout(() => {
    gridLabelRenderTimer = null;
    renderGridLabels();
  }, 50);
}

function visibleGridBounds({safeForCapture = false} = {}) {
  if (!canvas?.ready || !canvas.scene) return null;
  const gridSize = Number(canvas.scene.grid?.size ?? 0);
  const inverse = getClientToCanvasTransform();
  if (!Number.isFinite(gridSize) || gridSize <= 0 || !inverse) return null;

  const registration = getViewportRegistration();
  const markerMargin = viewportMarkerGeometry().size + Number(getSetting("viewportMarkerInset")) + 12;
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

function scheduleTestTargetRender() {
  if (testTargetRenderTimer) clearTimeout(testTargetRenderTimer);
  testTargetRenderTimer = setTimeout(() => {
    testTargetRenderTimer = null;
    renderTestTargets();
  }, 100);
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
  publishCaptureState();
  guidedCapture.timer = setInterval(() => {
    guidedCapture.seconds = Math.max(0, Math.ceil((end - Date.now()) / 1000));
    publishCaptureState();
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
  publishCaptureState();
}

function stopGuidedCapture(reason = "userStopped") {
  if (!guidedCapture) return;
  clearInterval(guidedCapture.timer);
  guidedCapture.waiting = true;
  guidedCapture.phase = "Saving";
  guidedCapture.seconds = 0;
  captureControl("stop", {reason});
  publishCaptureState();
}

function publishCaptureState() {
  sendVttState();
}

async function handleCaptureStatus(data) {
  if (data.state !== "recording" && data.sessionId && guidedCapture
      && data.sessionId !== guidedCapture.sessionId) return;
  if (data.state === "confirmed") {
    if (!guidedCapture?.recording || guidedCapture.phase !== "placement" || !guidedCapture.waiting
        || data.sessionId !== guidedCapture.sessionId || data.index !== guidedCapture.index) return;
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
    if (guidedCapture.phase === "Saving") {publishCaptureState(); return;}
    captureDelay(5, "Baseline · no physical minis", promptCaptureTarget);
  } else if (data.state === "saved") {
    clearInterval(guidedCapture?.timer);
    if (guidedCapture) {
      guidedCapture.seconds = 0;
      guidedCapture.phase = "Restoring";
      guidedCapture.message = "Tracking predictions remain paused";
      publishCaptureState();
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
      guidedCapture.recording = false;
      guidedCapture.phase = data.reason === "completed" ? "Recording saved" : "Partial recording saved";
      if (canvas.scene?.id === guidedCapture.sceneId) {
        canvas.tokens.releaseAll();
        for (const id of guidedCapture.originalControlIds ?? []) canvas.tokens.get(id)?.control({releaseOthers: false});
      }
    }
    testSequenceTargets = [];
    renderTestTargets();
    sendTestSequence();
    publishCaptureState();
  } else if (data.state === "error") {
    if (guidedCapture) {
      guidedCapture.message = data.message;
      guidedCapture.waiting = false;
      if (!guidedCapture.recording) guidedCapture.phase = "Capture not started";
    }
    publishCaptureState();
  } else if (data.state === "idle") {
    if (guidedCapture?.phase === "Starting") return;
    guidedCapture = null;
    sendVttState();
  }
}

function getViewportRegistration() {
  const {inset, margin: quietZone} = viewportMarkerGeometry();
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
  if (referenceCaptureBusy || !canvas?.ready || !canvas.scene || ws?.readyState !== WebSocket.OPEN) return false;
  let target = null;
  referenceCaptureBusy = true;
  try {
    const renderer = canvas.app?.renderer;
    const stage = canvas.app?.stage;
    const width = Number(renderer?.screen?.width);
    const height = Number(renderer?.screen?.height);
    const resolution = Number(renderer?.resolution ?? 1);
    if (!renderer?.render || !renderer?.extract?.canvas || !stage
        || !globalThis.PIXI?.RenderTexture?.create
        || ![width, height, resolution].every(Number.isFinite)
        || width <= 0 || height <= 0 || resolution <= 0
        || width * height * resolution * resolution > 33_554_432) {
      throw new Error("The Foundry renderer cannot create a bounded viewport snapshot.");
    }
    const sceneId = canvas.scene.id;
    const renderStartedAtEpochMs = Date.now();
    // Render the actual application stage into an owned, screen-sized target.
    // extract.canvas() without a target can read a browser-cleared framebuffer
    // between animation frames. Extracting a DisplayObject directly instead
    // uses its world bounds, losing the viewport crop and camera pan/zoom.
    // The explicit texture keeps the stage transform, lighting and fog layers.
    target = PIXI.RenderTexture.create({width, height, resolution});
    renderer.render(stage, {renderTexture: target, clear: true});
    const capturedAtEpochMs = Date.now();
    const viewTransform = buildViewTransformPayload();
    if (!viewTransform || canvas.scene.id !== sceneId) return false;
    const source = renderer.extract.canvas(target);
    const registration = viewTransform.registration;
    const scaleX = source.width / viewTransform.viewportWidth;
    const scaleY = source.height / viewTransform.viewportHeight;
    if (!source.width || !source.height || ![scaleX, scaleY].every(Number.isFinite)) {
      throw new Error("Foundry returned an empty viewport snapshot.");
    }
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
      sceneId,
      viewTransform,
      capture: {
        schemaVersion: 1,
        method: "pixi-stage-render-texture-v1",
        sessionId: referenceCaptureSessionId ??= crypto.randomUUID(),
        sequence: ++referenceCaptureSequence,
        renderStartedAtEpochMs,
        capturedAtEpochMs,
        durationMs: Date.now() - renderStartedAtEpochMs,
        sourceWidth: source.width,
        sourceHeight: source.height,
        rendererResolution: resolution,
        viewportWidth: width,
        viewportHeight: height,
      },
      image: snapshot.toDataURL("image/jpeg", 0.7),
    });
    referenceCaptureErrors = 0;
    return true;
  } catch (err) {
    referenceCaptureErrors++;
    if (referenceCaptureErrors >= 3) stopReferenceCapture();
    sendToPython({type: "renderedReferenceStatus", state: "error",
      message: String(err?.message ?? err), stopped: referenceCaptureErrors >= 3});
    warn("Could not capture a rendered reference; video recording will continue:", err);
    return false;
  } finally {
    try { target?.destroy(true); }
    finally { referenceCaptureBusy = false; }
  }
}

function setReferenceCapture(enabled) {
  stopReferenceCapture();
  if (enabled) {
    referenceCaptureErrors = 0;
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
  // Paper markers still need the viewport-to-scene transform when LCD marker
  // drawings are hidden. Drawing visibility must not stop registration data.
  if (!canvas?.ready || !canvas.scene) return null;
  const transform = canvas.stage.worldTransform;
  const dimensions = canvas.dimensions;
  return {
    type,
    sceneId: canvas.scene.id,
    viewportMarkerMode: true,
    viewportMarkersVisible: Boolean(getSetting("viewportMarkersEnabled")),
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

function cancelPendingSceneVisualChange() {
  if (!pendingSceneVisualChange) return;
  clearTimeout(pendingSceneVisualChange.fallbackTimer);
  clearTimeout(pendingSceneVisualChange.refreshTimer);
  pendingSceneVisualChange = null;
}

function visualChangeIsCurrent(change) {
  return change.socket === ws && ws?.readyState === WebSocket.OPEN
    && canvas?.ready && canvas.scene?.id === change.sceneId;
}

function flushSceneVisualChange(change) {
  if (pendingSceneVisualChange !== change) return;
  cancelPendingSceneVisualChange();
  if (!visualChangeIsCurrent(change) || captureIsActive()) return;
  const reasons = [...change.reasons], tokenIds = [...change.tokenIds], commandIds = [...change.commandIds];
  sendToPython({
    type: "sceneVisualChanged", sceneId: change.sceneId,
    reason: reasons.includes("sarweenTokenMove") ? "sarweenTokenMove"
      : reasons.length === 1 ? reasons[0] : "tokenVisualChanged",
    reasons, tokenIds, commandIds, ...(tokenIds.length === 1 ? {tokenId: tokenIds[0]} : {}),
    sentAt: Date.now(),
  });
}

function queueSceneVisualChange(reason, {sceneId = canvas?.scene?.id, tokenId = null, commandId = null, socket = ws} = {}) {
  if (captureIsActive() || !sceneId || socket !== ws || socket?.readyState !== WebSocket.OPEN
      || !canvas?.ready || canvas.scene?.id !== sceneId) return;
  if (pendingSceneVisualChange && !visualChangeIsCurrent(pendingSceneVisualChange)) {
    cancelPendingSceneVisualChange();
  }
  let created = false;
  if (!pendingSceneVisualChange) {
    const change = {sceneId, socket, reasons: new Set(), tokenIds: new Set(), commandIds: new Set(),
      fallbackTimer: null, refreshTimer: null};
    pendingSceneVisualChange = change;
    created = true;
  }
  const change = pendingSceneVisualChange;
  change.reasons.add(reason);
  if (tokenId) change.tokenIds.add(tokenId);
  if (typeof commandId === "string" && commandId && commandId.length <= 128) change.commandIds.add(commandId);
  // A later mutation needs its own perception refresh; an earlier refresh is
  // no longer evidence that the final canvas state has been drawn.
  clearTimeout(change.refreshTimer);
  change.refreshTimer = null;
  // A fixed deadline prevents an animated light's repeated refreshes from
  // indefinitely postponing the notification for this known mutation.
  if (created) change.fallbackTimer = setTimeout(() => flushSceneVisualChange(change), VISUAL_FALLBACK_MS);
}

function tokenVisualChanged(token, reason, {commandId = null} = {}) {
  const document = tokenDocument(token);
  if (!document?.id) return;
  const sceneId = document.parent?.id ?? document.scene?.id ?? canvas?.scene?.id;
  queueSceneVisualChange(reason, {sceneId, tokenId: document.id, commandId});
}

function perceptionRefreshed() {
  const change = pendingSceneVisualChange;
  if (!change) return;
  if (!visualChangeIsCurrent(change)) {cancelPendingSceneVisualChange(); return;}
  clearTimeout(change.refreshTimer);
  change.refreshTimer = setTimeout(() => flushSceneVisualChange(change), VISUAL_REFRESH_DELAY_MS);
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
  const fallback = Number(getSetting("defaultMovementSpeed"));
  return Number.isFinite(fallback) && fallback >= 0 ? fallback : 30;
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

function publishMovementState() {sendVttState();}

function renderMovementOverlay() {
  publishMovementState();
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
    void promptAssignMini(miniId);
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

function recordSelectedMovement(token, changes = {}, options = {}) {
  const document = tokenDocument(token);
  if (!movementState || document?.id !== movementState.tokenId) return;
  if (ignoredMovementUpdates.delete(document.id)) return;
  if (changes.x === undefined && changes.y === undefined) return;
  const next = movementPointForToken(document, changes);
  if (options.sarweenSource === "viewportTransform") {
    movementState = rebaseMovementAt(movementState, next);
    renderMovementOverlay();
    return;
  }
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
  const tokenId = movementState.tokenId;
  ignoredMovementUpdates.add(tokenId);
  try {
    await canvas.scene.updateEmbeddedDocuments(
      "Token",
      [{_id: tokenId, x: undone.point.x, y: undone.point.y}],
      {animate: getSetting("animateTokenMovement")},
    );
    movementState = undone.state;
  } finally {setTimeout(() => ignoredMovementUpdates.delete(tokenId), 1000);}
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

  registerVttSetting("wsUrl", {
    name: "Python WebSocket URL",
    hint: "Where to connect for Sarween output. Default: ws://127.0.0.1:8765",
    scope: "world",
    config: false,
    type: String,
    default: "ws://127.0.0.1:8765"
  });

  registerVttSetting("autoConnect", {
    name: "Auto-connect to Python",
    hint: "Automatically connect when the world loads. Reconnect automatically if the connection drops.",
    scope: "world",
    config: false,
    type: Boolean,
    default: true
  });

  registerVttSetting("debug", {
    name: "Enable debug logging",
    hint: "Log Sarween WebSocket events to the browser console.",
    scope: "client",
    config: false,
    type: Boolean,
    default: true
  });

  registerVttSetting("physicalGridAutoApply", {
    name: "Automatically apply physical grid size",
    hint: "Reapply Sarween's physical grid scale when a scene opens or the Foundry window changes size.",
    scope: "client",
    config: false,
    type: Boolean,
    default: true
  });

  registerVttSetting("gridLabelsVisible", {
    name: "Show grid cell labels on the map",
    hint: "Show small A1-style labels in the top-left corner of visible grid squares.",
    scope: "client",
    config: false,
    type: Boolean,
    default: false,
    onChange: gridLabelsChanged
  });

  registerVttSetting("displayDiagonalInches", {
    name: "Display diagonal (inches)",
    hint: "Physical diagonal of the TV or display showing the battle map.",
    scope: "client",
    config: false,
    type: Number,
    range: {min: 10, max: 150, step: 0.5},
    default: DEFAULT_DISPLAY_DIAGONAL_INCHES,
    onChange: physicalGridSettingsChanged
  });

  registerVttSetting("targetGridInches", {
    name: "Grid square size (inches)",
    hint: "Desired physical size of each grid square on the display.",
    scope: "client",
    config: false,
    type: Number,
    range: {min: 0.25, max: 3, step: 0.05},
    default: DEFAULT_GRID_INCHES,
    onChange: physicalGridSettingsChanged
  });

  registerVttSetting("physicalGridCalibration", {
    name: "Physical grid calibration",
    hint: "Fine adjustment after measuring a square with a ruler. Increase this value if squares are too small.",
    scope: "client",
    config: false,
    type: Number,
    range: {min: 0.5, max: 1.5, step: 0.005},
    default: 1,
    onChange: physicalGridSettingsChanged
  });

  registerVttSetting("viewportMarkersEnabled", {
    name: "Fixed viewport ArUco markers",
    hint: "Show four high-contrast registration markers above fog and keep them fixed while the map pans.",
    scope: "client",
    config: false,
    type: Boolean,
    default: true,
    onChange: () => {
      renderViewportMarkers();
      scheduleViewTransformSend();
    }
  });

  registerVttSetting("viewportMarkerSize", {
    name: "Legacy viewport marker tile size (pixels)",
    hint: "Compatibility setting for older display profiles. Use pattern width and white margin in Sarween.",
    scope: "client",
    config: false,
    type: Number,
    range: {min: 1, max: 240, step: 1},
    default: 120,
    onChange: () => {
      renderViewportMarkers();
      scheduleViewTransformSend();
    }
  });

  registerVttSetting("viewportMarkerPatternSize", {
    name: "ArUco pattern width (pixels)",
    hint: "Width of the ArUco pattern, excluding the white margin.",
    scope: "client", config: false, type: Number,
    range: {min: 1, max: 240, step: 1},
    default: 0,
    onChange: () => {renderViewportMarkers(); scheduleViewTransformSend();},
  });

  registerVttSetting("viewportMarkerMargin", {
    name: "White margin per side (pixels)",
    hint: "White space around each ArUco pattern. Total width is pattern width plus twice this margin.",
    scope: "client", config: false, type: Number,
    range: {min: 0, max: 12, step: 1},
    default: -1,
    onChange: () => {renderViewportMarkers(); scheduleViewTransformSend();},
  });

  registerVttSetting("viewportMarkerInset", {
    name: "Viewport marker inset (pixels)",
    hint: "Distance between each marker plate and the edge of the Foundry window.",
    scope: "client",
    config: false,
    type: Number,
    range: {min: 0, max: 80, step: 2},
    default: 8,
    onChange: () => {
      renderViewportMarkers();
      scheduleViewTransformSend();
    }
  });

  registerVttSetting("animateTokenMovement", {
    name: "Animate tracked token movement",
    hint: "Slide tracked tokens between squares instead of updating their position immediately.",
    scope: "client",
    config: false,
    type: Boolean,
    default: true
  });

  registerVttSetting("physicalTapSelection", {
    name: "Physical tap selects a mini",
    hint: "Let the overhead camera select or deselect a player mini after a short physical tap.",
    scope: "world",
    config: false,
    type: Boolean,
    default: false
  });

  registerVttSetting("defaultMovementSpeed", {
    name: "Fallback movement speed (feet)",
    hint: "Used when the selected Foundry actor does not provide a walking speed.",
    scope: "world",
    config: false,
    type: Number,
    range: {min: 0, max: 120, step: 5},
    default: 30
  });

  registerVttSetting("foundryUIHidden", {
    name: "Hide Foundry interface on the battle map",
    scope: "client", config: false, type: Boolean, default: false,
    onChange: applyFoundryUIVisibility,
  });

  console.log("sarween | Settings registered.");
});

// ──────────────────────────────────────────────────────────────────────────────
// Connection state
// ──────────────────────────────────────────────────────────────────────────────

function setStatus(state, extraText = "") {
  connectionStatus = {state, message: extraText};
  applyFoundryUIVisibility();
  sendVttState();
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
    actorId: t.actorId || null,
    width: Number(t.width ?? 1), height: Number(t.height ?? 1),
    sightEnabled: Boolean(t.sight?.enabled), sightRange: Number(t.sight?.range ?? 0),
  }));
  tokens.sort((a, b) => (a.tokenName || "").localeCompare(b.tokenName || ""));
  return tokens;
}

async function promptAssignMini(miniId) {
  if (!assignmentRequests.some(request => request.miniId === String(miniId))) {
    assignmentRequests.push({miniId: String(miniId), sceneId: canvas?.scene?.id ?? null});
  }
  sendVttState();
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

  const miniBindings = Object.fromEntries([...miniTokenBindings].filter(([_miniId, tokenId]) =>
    (scene.tokens ?? []).some(token => token.id === tokenId)));
  for (const mini of CAPTURE_MINIS) {
    if (miniBindings[mini.miniId]) continue;
    const matches = (scene.tokens ?? []).filter(token => token.name?.toLowerCase() === mini.tokenName.toLowerCase());
    if (matches.length === 1) miniBindings[mini.miniId] = matches[0].id;
  }
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
  cancelPendingSceneVisualChange();
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

function persistedTokenPosition(token) {
  let source;
  if (token && "_source" in token) source = token._source;
  else if (typeof token?.toObject === "function") source = token.toObject(true);
  // Lightweight document stand-ins expose their committed fields directly.
  // Foundry Documents always provide source data; never substitute their
  // prepared x/y, which Token animation changes on every rendered frame.
  else if (token && (token.constructor?.name === "Object" || Object.getPrototypeOf(token) === null)) source = token;
  if (!source || typeof source !== "object" || Array.isArray(source)
      || !Number.isFinite(source.x) || !Number.isFinite(source.y)) {
    throw new Error("Foundry token has no valid persisted position");
  }
  return {x: source.x, y: source.y};
}

async function handlePythonMessage(data, socket = ws) {
  if (data?.type !== "renderedReference") log("Received message from Python:", data);

  const { sceneId, tokenId, x, y, type } = data || {};

  if (type === "ping") return;
  if (type === "vttGet") {
    vttControlsRequested = true;
    sendVttState({force: true});
    return;
  }
  if (type === "vttCommand") {
    vttControlsRequested = true;
    await handleVttCommand(data, socket);
    return;
  }
  if (type === "physicalGridGet") {
    sendPhysicalGridState();
    return;
  }
  if (type === "physicalGridSet") {
    await setPhysicalGridFromPython(data, socket);
    return;
  }
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
    if (data.sceneId !== canvas.scene?.id) return;
    const miniId = data?.miniId;
    if (!miniId) {
      warn("assignMini missing miniId:", data);
      return;
    }
    await promptAssignMini(String(miniId));
    return;
  }

  if (type !== "moveToken") return;
  const reply = (replyType, values = {}) => {
    // The document update can finish after reconnect. Its old command must
    // never acknowledge through the replacement connection.
    if (socket !== ws || socket?.readyState !== WebSocket.OPEN) return false;
    return sendToPython({
      type: replyType, commandId: data.commandId, miniId: data.miniId,
      sceneId, tokenId, ...values,
    });
  };
  if (!data.commandId || !sceneId || !tokenId || !Number.isFinite(x) || !Number.isFinite(y)) {
    reply("tokenMoveError", {error: "Invalid move command"});
    return;
  }
  if (captureIsActive()) {
    reply("tokenMoveError", {error: "Tracking is paused for guided capture"});
    return;
  }
  const scene = canvas.scene;
  if (!canvas.ready || scene?.id !== sceneId) {
    reply("tokenMoveError", {error: "Requested scene is not the active display scene"});
    return;
  }

  if (!scene.tokens.get(tokenId)) {
    warn(`Mapped token ${tokenId} no longer exists in scene ${scene.id}`);
    reply("tokenMoveError", {code: "tokenMissing", error: "Assigned token no longer exists"});
    return;
  }

  let moved = false;
  try {
    const token = scene.tokens.get(tokenId);
    const current = persistedTokenPosition(token);
    // A retry after a lost acknowledgement must not replay animation or budget.
    if (current.x !== x || current.y !== y) {
      await scene.updateEmbeddedDocuments(
        "Token",
        [{ _id: tokenId, x, y }],
        { animate: getSetting("animateTokenMovement"), sarweenSource: data.source, sarweenCommandId: data.commandId }
      );
      moved = true;
    }
    const applied = persistedTokenPosition(scene.tokens.get(tokenId));
    if (applied.x !== x || applied.y !== y) {
      throw new Error("Foundry did not apply the requested position");
    }
    reply("tokenMoveApplied", {x: applied.x, y: applied.y, changed: moved});
  } catch (error) {
    warn(`Failed to move token ${tokenId}:`, error);
    reply("tokenMoveError", {error: String(error?.message ?? error)});
    return;
  }
  log(`Moved token ${tokenId} to (${x}, ${y}) in scene ${scene.id}`);
  if (moved) queueSceneVisualChange("sarweenTokenMove", {sceneId: scene.id, tokenId, commandId: data.commandId, socket});
}

// ──────────────────────────────────────────────────────────────────────────────
// Connect / Disconnect
// ──────────────────────────────────────────────────────────────────────────────

function dispatchPythonMessage(data, socket) {
  if (["physicalGridSet", "vttCommand", "moveToken", "captureStatus"].includes(data.type)) {
    commandMessageChain = commandMessageChain.then(async () => {
      if (socket === ws && ws.readyState === WebSocket.OPEN) {
        await handlePythonMessage(data, socket);
      }
    }).catch(err => error("VTT command handling failed:", err));
    return commandMessageChain;
  }
  return handlePythonMessage(data, socket);
}

function connectToPython() {
  if (!game.user.isGM) return;
  cancelPendingSceneVisualChange();
  const url = "ws://127.0.0.1:8765";
  const previous = ws;
  ws = null;
  assignmentRequests = [];
  vttControlsRequested = false;
  if (previous) {
    try {previous.close(1000, "Reconnecting");} catch {}
  }
  setStatus("connecting", "opening…");
  let socket;
  try {socket = new WebSocket(url);}
  catch (err) {
    error("Failed to create WebSocket:", err);
    setStatus("error", "create failed");
    scheduleReconnect("create failed");
    return;
  }
  ws = socket;
  socket.onopen = () => {
    if (socket !== ws) return;
    reconnectAttempts = 0;
    clearTimers();
    setStatus("connected");
    socket.send(JSON.stringify({type: "hello", source: "foundry-sarween", protocolVersion: 2,
      physicalGridControls: true, vttControls: true}));
    vttControlsRequested = true;
    if (testSequenceTargets.length) sendTestSequence();
    sendSceneAndView();
    sendPhysicalGridState();
    sendVttState();
    heartbeatTimer = setInterval(() => {
      if (socket === ws && socket.readyState === WebSocket.OPEN) {
        sendToPython({type: "ping", ts: Date.now()});
      }
    }, 10000);
  };
  socket.onmessage = async event => {
    if (socket !== ws || socket.readyState !== WebSocket.OPEN) return;
    try {await dispatchPythonMessage(JSON.parse(event.data), socket);}
    catch (err) {error("Error parsing/handling message:", err);}
  };
  socket.onclose = event => {
    if (socket !== ws) return;
    clearTimers();
    stopReferenceCapture();
    ws = null;
    assignmentRequests = [];
    vttControlsRequested = false;
    setStatus("disconnected", event.reason || "closed");
    scheduleReconnect(event.reason || "closed");
  };
  socket.onerror = err => {
    if (socket !== ws) return;
    error("WebSocket error:", err);
    setStatus("error", "socket error");
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Start-up
// ──────────────────────────────────────────────────────────────────────────────

Hooks.once("ready", () => {
  if (!game.user.isGM) {
    console.log("sarween | Not GM; WebSocket relay will not start.");
    return;
  }

  renderViewportMarkers();
  renderGridLabels();

  game.sarween = {
    ...(game.sarween ?? {}),
    applyPhysicalGridScale,
    calculatePhysicalGridScale,
    sendViewTransform,
    selectMovementToken,
    clearMovementSelection,
    resetSelectedMovement,
  };

  applyFoundryUIVisibility();
  connectToPython();
});

Hooks.on("canvasReady", () => {
  cancelPendingSceneVisualChange();
  clearMovementSelection();
  if (captureIsActive()) stopGuidedCapture("sceneChanged");
  assignmentRequests = assignmentRequests.filter(request => request.sceneId === canvas?.scene?.id);
  testSequenceTargets = [];
  renderTestTargets();
  sendTestSequence();
  renderViewportMarkers();
  renderGridLabels();
  setTimeout(async () => {
    while (physicalGridSettingsBatch) await physicalGridBatchSettled;
    if (game.settings.get(MODULE_ID, "physicalGridAutoApply")) {
      await applyPhysicalGridScale({animate: true});
    }
    sendSceneAndView();
    sendPhysicalGridState();
    sendVttState();
  }, 50);
});

Hooks.on("updateScene", (scene, changes) => {
  if (scene.id !== canvas?.scene?.id) return;
  const relevant = Object.keys(changes).some(key => /^(width|height|grid|background|padding)(\.|$)/.test(key));
  const visual = relevant || Object.keys(changes).some(key =>
    /^(tokenVision|environment|darkness|darknessLevel|globalLight|globalLightThreshold|fog|fogExploration|fogOverlay|fogExploredColor|fogUnexploredColor|backgroundColor)(\.|$)/.test(key));
  if (relevant) {
    if (captureIsActive()) stopGuidedCapture("geometryChanged");
    sendSceneAndView();
    sendPhysicalGridState();
    sendVttState();
    scheduleViewTransformSend();
    scheduleGridLabelRender();
  } else sendVttState();
  if (visual) queueSceneVisualChange("sceneSettingsChanged", {sceneId: scene.id});
});

Hooks.on("canvasPan", () => {
  if (captureIsActive() && JSON.stringify(getClientToCanvasTransform()) !== guidedCapture.viewSignature) {
    stopGuidedCapture("viewportChanged");
  }
  scheduleViewTransformSend();
  scheduleTestTargetRender();
  scheduleGridLabelRender();
  renderMovementOverlay();
});

Hooks.on("canvasTearDown", () => {
  cancelPendingSceneVisualChange();
  if (gridLabelRenderTimer) clearTimeout(gridLabelRenderTimer);
  gridLabelRenderTimer = null;
  removeGridLabels();
});

Hooks.on("controlToken", (token, controlled) => {
  if (captureIsActive()) return;
  const document = tokenDocument(token);
  if (controlled) {
    selectMovementToken(document);
  } else if (movementState?.tokenId === document?.id) {
    clearMovementSelection();
  }
  tokenVisualChanged(token, controlled ? "tokenControlled" : "tokenReleased");
});

Hooks.on("updateToken", (document, changes, options = {}) => {
  recordSelectedMovement(document, changes, options);
  if (Object.keys(changes).some(key => /^(x|y|rotation|elevation|hidden|width|height|sight|light)(\.|$)/.test(key))) {
    tokenVisualChanged(document, options.sarweenSource ? "sarweenTokenMove" : "tokenUpdated", {commandId: options.sarweenCommandId});
  }
  if (Object.keys(changes).some(key => /^(name|actorId|width|height|sight)(\.|$)/.test(key))) {
    sendSceneAndView(); sendVttState();
  }
});

// These hooks only finish a queued, known scene/token mutation. Animated lighting
// refreshes by themselves must never cause a stream of background reseeds.
Hooks.on("sightRefresh", perceptionRefreshed);
Hooks.on("lightingRefresh", perceptionRefreshed);

Hooks.on("deleteToken", document => {
  if (movementState?.tokenId === document?.id) clearMovementSelection();
  sendSceneAndView(); sendVttState();
  tokenVisualChanged(document, "tokenDeleted");
});

Hooks.on("createToken", document => {
  sendSceneAndView(); sendVttState();
  tokenVisualChanged(document, "tokenCreated");
});

window.addEventListener("resize", () => {
  renderViewportMarkers();
  renderTestTargets();
  renderMovementOverlay();
  renderGridLabels();
  if (game.settings.get(MODULE_ID, "physicalGridAutoApply")) {
    applyPhysicalGridScale({animate: false});
  }
  scheduleViewTransformSend();
});
