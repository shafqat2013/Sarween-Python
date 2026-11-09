import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness as harness} from "./module_harness.mjs";

const nextTurn = () => new Promise(resolve => setImmediate(resolve));
const captureValues = {participants: [{miniId: "red10", tokenId: "red"}], scenario: "fog-off", notes: "USB camera test"};
const lastState = h => h.sent.findLast(message => message.type === "vttState");

test("desktop controls bootstrap automatically despite legacy connection settings and create only map overlays", async () => {
  const h = await harness();
  h.settings.set("autoConnect", false);
  h.settings.set("wsUrl", "ws://old-host.invalid:1234");
  h.once.get("ready")();
  const socket = h.sockets.at(-1);
  assert.equal(socket.url, "ws://127.0.0.1:8765");
  socket.onopen();
  assert.equal(h.sent[0].type, "hello");
  assert.equal(h.sent[0].vttControls, true);
  assert.equal(h.sent[0].physicalGridControls, true);
  assert.ok(h.sent.findIndex(message => message.type === "sceneInfo") < h.sent.findIndex(message => message.type === "vttState"));
  assert.equal(lastState(h).connection.state, "connected");
  assert.equal(h.created.some(element => ["sarween-status", "sarween-capture"].includes(element.id)), false);
  assert.ok(h.elements.has("sarween-viewport-markers"));
  assert.equal(h.registrations.get("wsUrl").config, false);
  assert.equal(h.registrations.get("autoConnect").config, false);
});

test("obsolete socket callbacks cannot close, reopen, or deliver commands through a replacement connection", async () => {
  const h = await harness();
  h.api.connectToPython();
  const old = h.sockets.at(-1);
  old.onopen();
  h.api.connectToPython();
  const replacement = h.sockets.at(-1);
  replacement.onopen();
  h.sent.length = 0;
  old.onclose({code: 1000, reason: "old close"});
  old.onopen();
  old.onerror(new Error("old error"));
  await old.onmessage({data: JSON.stringify({type: "vttCommand", requestId: "old", action: "lifecycle.reload", values: {}})});
  assert.equal(h.api.getSocket(), replacement);
  assert.equal(h.sent.length, 0);
  assert.equal(h.timers.size, 0);
  replacement.readyState = 3;
  replacement.onclose({code: 1006, reason: "relay stopped"});
  assert.equal(h.api.getSocket(), null);
  assert.equal(h.timers.size, 1);
  await h.flushTimers();
  assert.equal(h.sockets.at(-1).url, "ws://127.0.0.1:8765");
});

test("state describes all settings, token choices, ring defaults, and extensible command capabilities", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red", actorId: "actor-red"});
  await h.api.handlePythonMessage({type: "vttGet"});
  const state = lastState(h);
  assert.equal(state.ready, true);
  assert.equal(state.sceneId, "scene-1");
  assert.equal(state.tokens[0].actorId, "actor-red");
  assert.equal(state.tokens[0].width, 1);
  assert.equal(state.tokens[0].sightEnabled, true);
  assert.deepEqual(state.minis[0], {route: "A", ringColor: "red", tokenName: "Red", miniId: "red10", tokenId: "red"});
  assert.equal(state.capture.target, null);
  assert.equal(state.capture.recording, false);
  assert.equal(state.movement, null);
  assert.equal(state.assignment, null);
  assert.ok(state.capabilities.includes("lifecycle.reload"));
  assert.ok(state.capabilities.includes("state.sync"));
  assert.ok(state.capabilities.includes("capture.confirm"));
  assert.equal(Object.hasOwn(state.settings, "wsUrl"), false);
  assert.equal(Object.hasOwn(state.settings, "autoConnect"), false);
  assert.equal(state.settings.defaultMovementSpeed, 30);
  assert.equal(state.settings.foundryUIHidden, false);
  h.sent.length = 0;
  h.hooks.get("canvasPan")();
  h.hooks.get("canvasPan")();
  assert.equal(h.sent.some(message => message.type === "vttState"), false, "panning must not repeatedly transmit unchanged desktop controls");
  await h.api.handlePythonMessage({type: "vttGet"});
  assert.equal(h.sent.filter(message => message.type === "vttState").length, 1, "explicit refresh always responds");
});

test("settings validate the whole edit before writing, batch marker rendering and grid resizing, and return applied state", async () => {
  const h = await harness();
  for (const invalid of [{viewportMarkerSize: 0}, {viewportMarkerPatternSize: 0}, {viewportMarkerMargin: -1},
    {viewportMarkerMargin: 13}, {viewportMarkerPatternSize: NaN}, {viewportMarkerInset: 81}, {defaultMovementSpeed: 121},
    {animateTokenMovement: 1}, {physicalTapSelection: "false"}, {foundryUIHidden: null}, {wsUrl: "ws://elsewhere"},
    {autoConnect: false}, {unknown: 1}, {targetGridInches: Infinity}]) {
    const result = await h.command("settings.set", {settings: {debug: true, ...invalid}});
    assert.equal(result.success, false);
    assert.equal(h.writes.length, 0);
  }
  const originalGrid = {...h.scene.grid};
  const result = await h.command("settings.set", {settings: {targetGridInches: 1.1, displayDiagonalInches: 60,
    physicalGridCalibration: 1.05, viewportMarkerSize: 160, viewportMarkerInset: 20,
    physicalGridAutoApply: false, animateTokenMovement: false, physicalTapSelection: false, defaultMovementSpeed: 40}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerPatternSize, 160 * 2 / 3);
  assert.equal(result.state.settings.viewportMarkerMargin, 12);
  assert.equal(result.state.settings.viewportMarkerSize, 160 * 2 / 3 + 24);
  assert.equal(result.state.settings.defaultMovementSpeed, 40);
  assert.equal(h.pans.length, 1);
  assert.equal(h.created.filter(element => element.id === "sarween-viewport-markers").length, 1);
  assert.deepEqual(h.scene.grid, originalGrid);
  assert.equal(h.pans[0].x, 789);
  assert.equal(h.pans[0].y, 456);
});

test("applying an unchanged display profile restores physical zoom after a manual zoom", async () => {
  const h = await harness();
  h.canvas.stage.worldTransform.a = h.canvas.stage.worldTransform.d = 0.2;
  const settings = Object.fromEntries(["targetGridInches", "displayDiagonalInches", "physicalGridCalibration"]
    .map(key => [key, h.settings.get(key)]));
  const result = await h.command("settings.set", {settings});
  assert.equal(result.success, true);
  assert.equal(h.writes.length, 0);
  assert.equal(h.pans.length, 1);
  const expected = Math.hypot(1920, 1080) / 55 / h.scene.grid.size;
  assert.ok(Math.abs(h.canvas.stage.worldTransform.a - expected) < 1e-9);
  assert.equal(h.pans[0].x, 789);
  assert.equal(h.pans[0].y, 456);
});

test("settings failure restores persisted values, while reconnect during a batch cannot receive an obsolete result", async () => {
  const h = await harness();
  let reject = true;
  h.setBehavior(key => {if (key === "viewportMarkerSize" && reject) {reject = false; throw new Error("Storage failure");}});
  const failed = await h.command("settings.set", {settings: {targetGridInches: 1.2, viewportMarkerSize: 160}});
  assert.equal(failed.success, false);
  assert.equal(h.settings.get("targetGridInches"), 1);
  assert.equal(h.settings.get("viewportMarkerSize"), 120);
  assert.equal(h.pans.length, 0);
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.setBehavior(() => gate);
  h.sent.length = 0;
  const pending = h.command("settings.set", {settings: {defaultMovementSpeed: 45}});
  await nextTurn();
  await h.api.handlePythonMessage({type: "vttGet"});
  assert.equal(h.sent.length, 0);
  h.api.setSocket({readyState: 1, send(raw) {h.sent.push(JSON.parse(raw));}});
  release();
  assert.equal(await pending, undefined);
  assert.equal(h.settings.get("defaultMovementSpeed"), 30);
  assert.equal(h.sent.some(message => message.type === "vttResult"), false);
  assert.equal(lastState(h).settings.defaultMovementSpeed, 30);
});

test("capture validation refuses invalid rings, duplicate tokens, missing vision, notes, and stale scenes before starting", async () => {
  const h = await harness();
  const red = h.addToken({id: "red", name: "Red"});
  for (const values of [
    {...captureValues, participants: []},
    {...captureValues, participants: [{miniId: "unknown", tokenId: "red"}]},
    {...captureValues, participants: [{miniId: "red10", tokenId: "missing"}]},
    {...captureValues, participants: [{miniId: "red10", tokenId: "red"}, {miniId: "blue", tokenId: "red"}]},
    {...captureValues, scenario: "unknown"}, {...captureValues, notes: "x".repeat(2001)},
  ]) assert.equal((await h.command("capture.start", values)).success, false);
  red.width = 2;
  assert.equal((await h.command("capture.start", captureValues)).success, false);
  red.width = 1; red.sight.enabled = false;
  assert.equal((await h.command("capture.start", {...captureValues, scenario: "fog-on"})).success, false);
  assert.equal((await h.command("capture.start", captureValues, {sceneId: "stale"})).success, false);
  assert.equal(h.api.getCapture(), null);
  assert.equal(h.sent.some(message => message.type === "captureControl"), false);
});

test("desktop drives capture start, countdown, destination, confirm, finish, restoration, new, and resume without map panels", async () => {
  const h = await harness();
  const red = h.addToken({id: "red", name: "Red", x: 100, y: 200});
  h.canvas.tokens.get("red").control();
  const started = await h.command("capture.start", captureValues);
  assert.equal(started.success, true);
  assert.equal(started.state.capture.phase, "Starting");
  assert.equal(started.state.movement, null);
  const start = h.sent.find(message => message.type === "captureControl" && message.action === "start");
  assert.equal(start.participants[0].route, "A");
  assert.equal(start.notes, captureValues.notes);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "recording", sessionId: start.sessionId}, h.socket);
  assert.equal(lastState(h).capture.seconds, 5);
  assert.equal(lastState(h).capture.target, null);
  await h.advanceTime(1000);
  assert.equal(lastState(h).capture.seconds, 4);
  await h.advanceTime(4000);
  assert.equal(lastState(h).capture.phase, "placement");
  assert.equal(lastState(h).capture.target.label, "A0");
  assert.equal(lastState(h).capture.waiting, false);
  assert.ok(h.elements.has("sarween-test-targets"));
  const confirmed = await h.command("capture.confirm");
  assert.equal(confirmed.success, true);
  assert.equal(confirmed.state.capture.waiting, true);
  assert.equal((await h.command("capture.confirm")).success, false);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "confirmed", sessionId: start.sessionId, index: 0}, h.socket);
  assert.equal(lastState(h).capture.index, 1);
  assert.equal(lastState(h).capture.seconds, 15);
  assert.equal(lastState(h).capture.target, null);
  await h.advanceTime(15000);
  assert.equal(lastState(h).capture.target.label, "A1");
  assert.equal((await h.command("capture.stop")).success, true);
  assert.equal(lastState(h).capture.phase, "Saving");
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "saved", sessionId: start.sessionId, reason: "userStopped"}, h.socket);
  assert.equal(red.x, 100); assert.equal(red.y, 200);
  assert.equal(lastState(h).capture.recording, false);
  assert.equal(lastState(h).capture.phase, "Partial recording saved");
  assert.equal(h.elements.has("sarween-test-targets"), false);
  assert.equal((await h.command("capture.new")).state.capture.state, "idle");
  assert.equal((await h.command("capture.resume")).success, true);
  assert.ok(h.sent.some(message => message.type === "captureControl" && message.action === "resume"));
  assert.equal(h.created.some(element => ["sarween-status", "sarween-capture"].includes(element.id)), false);
});

test("capture protects geometry, markers, assignment, movement and reload actions", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red"});
  await h.command("capture.start", captureValues);
  for (const [action, values] of [["settings.set", {settings: {viewportMarkerSize: 160}}], ["markers.set", {enabled: false}],
    ["ui.setHidden", {hidden: true}], ["movement.select", {tokenId: "red"}], ["assignment.save", {miniId: "red10", tokenId: "red"}],
    ["lifecycle.reload", {}], ["capture.new", {}], ["capture.resume", {}]]) {
    assert.equal((await h.command(action, values)).success, false, action);
  }
  assert.equal(h.writes.length, 0);
  assert.equal(h.reloads, 0);
});

test("stopping a slow camera start keeps the capture live through Saving and ignores late startup or placement acknowledgements", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red"});
  const start = await h.command("capture.start", captureValues);
  const sessionId = start.state.capture.sessionId;
  const stopped = await h.command("capture.stop");
  assert.equal(stopped.state.capture.recording, false);
  assert.equal(stopped.state.capture.phase, "Saving");
  assert.equal(stopped.state.capture.active, true);
  assert.equal((await h.command("capture.new")).success, false);
  assert.equal((await h.command("capture.start", captureValues)).success, false);
  assert.equal((await h.command("settings.set", {settings: {viewportMarkerSize: 160}})).success, false);
  await h.request({targetGridInches: 1.2});
  assert.equal(h.sent.findLast(message => message.type === "physicalGridResult").success, false);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "recording", sessionId}, h.socket);
  assert.equal(lastState(h).capture.phase, "Saving");
  assert.equal(h.intervals.size, 0);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "confirmed", sessionId, index: 0}, h.socket);
  assert.equal(h.api.getCapture().index, 0);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "saved", sessionId, reason: "userStopped"}, h.socket);
  assert.equal(lastState(h).capture.active, false);
  assert.equal((await h.command("capture.new")).success, true);
  const next = await h.command("capture.start", captureValues);
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "saved", sessionId, reason: "completed"}, h.socket);
  assert.equal(h.api.getCapture().sessionId, next.state.capture.sessionId);
  assert.equal(h.api.getCapture().phase, "Starting");
});

test("capture commands queue behind physical grid persistence instead of starting midway through an edit", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red"});
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.setBehavior(() => gate);
  const grid = h.request({targetGridInches: 1.2});
  const capture = h.command("capture.start", captureValues);
  await nextTurn();
  assert.equal(h.api.getCapture(), null);
  assert.equal(h.sent.some(message => message.type === "captureControl"), false);
  release();
  await grid;
  assert.equal((await capture).success, true);
  assert.equal(h.pans.length, 1);
});

test("assignment requests queue and deduplicate; save/cancel advance choices and preserve an explicit ring mapping", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red"});
  h.addToken({id: "custom", name: "Paladin"});
  await h.api.handlePythonMessage({type: "vttGet"});
  for (const miniId of ["red10", "blue", "red10"]) await h.api.handlePythonMessage({type: "assignMini", sceneId: h.scene.id, miniId});
  assert.equal(lastState(h).assignment.miniId, "red10");
  assert.equal(lastState(h).assignment.pendingCount, 2);
  const saved = await h.command("assignment.save", {miniId: "red10", tokenId: "custom"});
  assert.equal(saved.success, true);
  assert.equal(saved.state.assignment.miniId, "blue");
  assert.equal(saved.state.minis[0].tokenId, "custom");
  assert.equal(h.sent.findLast(message => message.type === "sceneInfo").miniBindings.red10, "custom");
  const cancelled = await h.command("assignment.cancel");
  assert.equal(cancelled.state.assignment, null);
  assert.equal(h.sent.findLast(message => message.type === "assignMiniResult").miniId, "blue");
  assert.equal(h.sent.findLast(message => message.type === "assignMiniResult").cancelled, true);
  assert.equal((await h.command("assignment.save", {miniId: "red10", tokenId: "deleted"})).success, false);
});

test("player mini assignments are acknowledged without replacing the legacy capture roster", async () => {
  const h = await harness();
  h.addToken({id: "player", name: "Paladin"});
  const saved = await h.command("assignment.save", {miniId: "mini1", tokenId: "player"});
  assert.equal(saved.success, true);
  assert.equal(saved.state.miniBindings.mini1, "player");
  assert.ok(saved.state.minis.every(mini => mini.miniId !== "mini1"));
  const reply = h.sent.findLast(message => message.type === "assignMiniResult");
  assert.equal(reply.miniId, "mini1");
  assert.equal(reply.tokenId, "player");
  h.scene.tokens.splice(0, 1);
  assert.equal(Object.hasOwn(h.api.vttState().miniBindings, "mini1"), false);
});

test("movement selection, budget changes, undo and reset synchronize desktop and preserve state if undo fails", async () => {
  const h = await harness();
  const red = h.addToken({id: "red", name: "Red"});
  assert.equal((await h.command("movement.select", {tokenId: "red"})).state.movement.remainingFeet, 30);
  await red.update({x: 200});
  assert.equal(lastState(h).movement.usedFeet, 5);
  assert.equal(lastState(h).movement.canUndo, true);
  await h.command("movement.select", {tokenId: "red"});
  assert.equal(lastState(h).movement.usedFeet, 5, "selecting the current token must not reset the turn");
  h.updateBehavior(() => {throw new Error("Document rejected");});
  assert.equal((await h.command("movement.undo")).success, false);
  assert.equal(lastState(h).movement.usedFeet, 5);
  h.updateBehavior(async () => {});
  const undone = await h.command("movement.undo");
  assert.equal(undone.success, true);
  assert.equal(undone.state.movement.usedFeet, 0);
  assert.equal(red.x, 100);
  await red.update({x: 300});
  assert.equal((await h.command("movement.reset")).state.movement.usedFeet, 0);
  assert.equal((await h.command("movement.select", {tokenId: null})).state.movement, null);
  assert.equal(h.elements.has("sarween-movement-overlay"), false);
});

test("hiding native Foundry UI preserves overlays and recovers automatically when the relay disconnects", async () => {
  const h = await harness();
  h.api.connectToPython();
  const socket = h.sockets.at(-1); socket.onopen();
  assert.equal((await h.command("ui.setHidden", {hidden: true})).success, true);
  const style = h.elements.get("sarween-hide-foundry-ui");
  assert.ok(style.textContent.includes("#sidebar"));
  assert.equal(style.textContent.includes("sarween-viewport-markers"), false);
  socket.readyState = 3; socket.onclose({code: 1006, reason: "offline"});
  assert.equal(h.elements.has("sarween-hide-foundry-ui"), false);
  assert.equal(h.settings.get("foundryUIHidden"), true);
  await h.flushTimers();
  h.sockets.at(-1).onopen();
  assert.equal(h.elements.has("sarween-hide-foundry-ui"), true);
  assert.equal((await h.command("ui.setHidden", {hidden: false})).success, true);
  assert.equal(h.elements.has("sarween-hide-foundry-ui"), false);
});

test("reload is acknowledged before reloading and stale, unknown, non-GM, or malformed commands fail visibly", async () => {
  const h = await harness();
  const result = await h.command("lifecycle.reload");
  assert.equal(result.success, true);
  assert.equal(h.reloads, 0);
  await h.flushTimers();
  assert.equal(h.reloads, 1);
  for (const [action, values, extra] of [["future.unknown", {}, {}], ["markers.set", {enabled: true, extra: 1}, {}],
    ["movement.reset", {}, {sceneId: "old-scene"}], ["state.sync", [], {}]]) {
    assert.equal((await h.command(action, values, extra)).success, false);
  }
  h.game.user.isGM = false;
  assert.equal((await h.command("state.sync")).success, false);
});
