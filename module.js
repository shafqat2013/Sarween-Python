// module.js for "sarween" — Computer Vision for TTRPG

const MODULE_ID = "sarween";

let ws = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let heartbeatTimer = null;
let manualDisconnect = false;

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

// ──────────────────────────────────────────────────────────────────────────────
// Settings
// ──────────────────────────────────────────────────────────────────────────────

Hooks.once("init", () => {
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

  console.log("sarween | Settings registered.");
});

// ──────────────────────────────────────────────────────────────────────────────
// Status UI (moved to bottom-right to avoid covering top markers)
// ──────────────────────────────────────────────────────────────────────────────

function ensureStatusUI() {
  if (statusEl) return;

  statusEl = document.createElement("div");
  statusEl.id = "sarween-status";
  statusEl.style.cssText = `
    position: fixed;
    bottom: 10px;
    right: 10px;
    z-index: 10000;
    font-family: sans-serif;
    font-size: 12px;
    background: rgba(0,0,0,0.75);
    color: white;
    padding: 8px 10px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
  `;

  statusEl.innerHTML = `
    <span id="sarween-dot" style="width:10px;height:10px;border-radius:50%;display:inline-block;background:#aa0000;"></span>
    <span id="sarween-text">Sarween: Disconnected</span>
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

  return {
    type: "sceneInfo",
    sceneId: scene.id,
    width,
    height,
    gridSize,
    shiftX,
    shiftY,
    gridType
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
  log("Received message from Python:", data);

  const { sceneId, tokenId, x, y, type } = data || {};

  if (type === "ping") return;

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
  if (!sceneId || !tokenId || typeof x !== "number" || typeof y !== "number") {
    warn("Invalid payload from Python:", data);
    return;
  }

  const scene = game.scenes.get(sceneId) ?? canvas.scene;
  if (!scene) {
    warn("Scene not found:", sceneId);
    return;
  }

  await scene.updateEmbeddedDocuments("Token", [{ _id: tokenId, x, y }]);
  log(`Moved token ${tokenId} to (${x}, ${y}) in scene ${scene.id}`);
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
    } catch (err) {
      error("Failed to send hello:", err);
    }

    // Proactively send sceneInfo on connect (helps Foundry mode feel instant)
    try {
      const payload = buildSceneInfoPayload(canvas.scene);
      if (payload) ws.send(JSON.stringify(payload));
    } catch (err) {
      error("Failed to send initial sceneInfo:", err);
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

  if (getSetting("autoConnect")) {
    console.log("sarween | World ready, starting Python WebSocket connection");
    connectToPython();
  } else {
    setStatus("disconnected", "auto-connect off");
    console.log("sarween | Auto-connect disabled. Use the Sarween UI button to connect.");
  }
});
