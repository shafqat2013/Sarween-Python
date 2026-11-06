import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness as harness} from "./module_harness.mjs";

test("desktop edit batches settings, preserves grid geometry and pivot, then acknowledges applied view", async () => {
  const h = await harness();
  const originalGrid = {...h.scene.grid};
  await h.request({targetGridInches: 1.2, displayDiagonalInches: 65, physicalGridCalibration: 1.1});
  assert.equal(h.writes.length, 3);
  assert.equal(h.pans.length, 1);
  assert.equal(h.pans[0].x, 789);
  assert.equal(h.pans[0].y, 456);
  const expectedScale = Math.hypot(1920, 1080) / 65 * 1.2 * 1.1 / 100;
  assert.ok(Math.abs(h.pans[0].scale - expectedScale) < 1e-12);
  assert.deepEqual(h.scene.grid, originalGrid);
  assert.deepEqual(h.sent.map(message => message.type), ["sceneInfo", "viewTransform", "physicalGridResult", "physicalGridState"]);
  const result = h.sent.find(message => message.type === "physicalGridResult");
  assert.equal(result.success, true);
  assert.equal(result.requestId, "grid-1");
  assert.equal(result.ready, true);
  assert.equal(result.sceneId, "scene-1");
  assert.deepEqual(result.settings, {targetGridInches: 1.2, displayDiagonalInches: 65, physicalGridCalibration: 1.1});
});

test("every invalid setting is rejected before any write or pan", async () => {
  for (const invalid of [
    {targetGridInches: 0.24}, {targetGridInches: 3.01}, {displayDiagonalInches: 9},
    {displayDiagonalInches: 151}, {physicalGridCalibration: 0.49}, {physicalGridCalibration: 1.51},
    {targetGridInches: "1"}, {targetGridInches: true}, {targetGridInches: NaN},
    {targetGridInches: Infinity}, {unknownSetting: 1}, {constructor: 1},
  ]) {
    const h = await harness();
    await h.request({displayDiagonalInches: 65, ...invalid});
    assert.deepEqual(h.writes, [], JSON.stringify(invalid));
    assert.deepEqual(h.pans, []);
    assert.equal(h.sent[0].success, false);
  }
  for (const extra of [{settings: []}, {settings: null}, {apply: "yes"}, {requestId: ""}]) {
    const h = await harness();
    await h.request({targetGridInches: 1.2}, extra);
    assert.deepEqual(h.writes, []);
    assert.equal(h.sent[0].success, false);
  }
});

test("capture, an unavailable canvas, and stale scenes refuse edits without mutations", async () => {
  for (const capture of [{recording: true}, {phase: "Starting"}]) {
    const h = await harness();
    h.api.setCapture(capture);
    await h.request({targetGridInches: 1.2});
    assert.match(h.sent[0].message, /capture/);
    assert.deepEqual(h.writes, []);
    assert.deepEqual(h.pans, []);
  }
  const h = await harness();
  await h.request({targetGridInches: 1.2}, {sceneId: "old-scene"});
  assert.match(h.sent[0].message, /scene changed/);
  h.canvas.ready = false;
  await h.request({targetGridInches: 1.2});
  assert.equal(h.sent.at(-1).ready, false);
  assert.deepEqual(h.writes, []);
  assert.deepEqual(h.pans, []);
});

test("acknowledgement waits for settings persistence and asynchronous pan", async () => {
  const h = await harness();
  let releaseWrite;
  let releasePan;
  const writeGate = new Promise(resolve => {releaseWrite = resolve;});
  const panGate = new Promise(resolve => {releasePan = resolve;});
  h.setBehavior(() => writeGate);
  h.panBehavior(() => panGate);
  const pending = h.request({targetGridInches: 1.2});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.sent.length, 0);
  assert.equal(h.pans.length, 0);
  releaseWrite();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.pans.length, 1);
  assert.equal(h.sent.length, 0);
  releasePan();
  await pending;
  assert.equal(h.sent.find(message => message.type === "physicalGridResult").success, true);
});

test("failed writes restore all previous values without moving the map", async () => {
  const h = await harness();
  let reject = true;
  h.setBehavior(key => {
    if (key === "displayDiagonalInches" && reject) {reject = false; throw new Error("Storage failure");}
  });
  await h.request({targetGridInches: 1.2, displayDiagonalInches: 65});
  assert.equal(h.settings.get("targetGridInches"), 1);
  assert.equal(h.settings.get("displayDiagonalInches"), 55);
  assert.deepEqual(h.pans, []);
  assert.equal(h.sent[0].success, false);
  assert.match(h.sent[0].message, /restored/);
  assert.equal(h.sent.at(-1).settings.targetGridInches, 1);
});

test("failed pan restores values and synchronizes the restored viewport before error acknowledgement", async () => {
  const h = await harness();
  let reject = true;
  h.panBehavior(() => {if (reject) {reject = false; throw new Error("Pan failure");}});
  await h.request({targetGridInches: 1.2});
  assert.equal(h.settings.get("targetGridInches"), 1);
  assert.equal(h.pans.length, 2);
  assert.deepEqual(h.sent.map(message => message.type), ["sceneInfo", "viewTransform", "physicalGridResult", "physicalGridState"]);
  assert.equal(h.sent[2].success, false);
});

test("capture or scene changes while settings persist prevent applying the stale viewport", async () => {
  for (const invalidate of [h => h.api.setCapture({phase: "Starting"}), h => {h.scene.id = "scene-2";}]) {
    const h = await harness();
    h.setBehavior(() => invalidate(h));
    await h.request({targetGridInches: 1.2});
    assert.equal(h.settings.get("targetGridInches"), 1);
    assert.deepEqual(h.pans, []);
    assert.equal(h.sent[0].success, false);
  }
});

test("save-only updates do not apply a viewport; empty apply reapplies saved settings", async () => {
  const h = await harness();
  await h.request({targetGridInches: 1.3}, {apply: false});
  assert.equal(h.settings.get("targetGridInches"), 1.3);
  assert.deepEqual(h.pans, []);
  assert.equal(h.sent[0].success, true);
  h.sent.length = 0;
  h.writes.length = 0;
  await h.request();
  assert.deepEqual(h.writes, []);
  assert.equal(h.pans.length, 1);
  assert.equal(h.sent.find(message => message.type === "physicalGridResult").success, true);
});

test("local Foundry changes, get requests, and scene readiness synchronize desktop state", async () => {
  const h = await harness();
  await h.game.settings.set("sarween", "targetGridInches", 1.25);
  assert.equal(h.pans.length, 1);
  assert.equal(h.sent.at(-1).type, "physicalGridState");
  assert.equal(h.sent.at(-1).settings.targetGridInches, 1.25);
  h.sent.length = 0;
  await h.api.handlePythonMessage({type: "physicalGridGet"});
  assert.equal(h.sent[0].supported, true);
  assert.equal(h.sent[0].ready, true);
  h.settings.set("viewportMarkersEnabled", false);
  h.settings.set("physicalGridAutoApply", false);
  h.sent.length = 0;
  await h.hooks.get("canvasReady")();
  await h.flushTimers();
  assert.equal(h.sent.at(-1).type, "physicalGridState");
  h.scene.grid.size = 0;
  h.hooks.get("updateScene")(h.scene, {grid: {size: 0}});
  assert.equal(h.sent.findLast(message => message.type === "physicalGridState").ready, false);
});

test("connection advertises support and sends current state after the normal handshake", async () => {
  const h = await harness();
  h.api.connectToPython();
  h.sockets.at(-1).onopen();
  assert.deepEqual(h.sent[0], {type: "hello", source: "foundry-sarween", protocolVersion: 2, physicalGridControls: true, vttControls: true});
  assert.equal(h.sent.at(-1).type, "vttState");
  assert.equal(h.sent.at(-1).settings.displayDiagonalInches, 55);
});

test("queued edits retain wire order and commands from an obsolete connection are discarded", async () => {
  const h = await harness();
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.setBehavior(() => gate);
  const first = h.request({targetGridInches: 1.1}, {requestId: "first"});
  const second = h.request({targetGridInches: 1.2}, {requestId: "second"});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.writes.length, 1);
  release();
  await Promise.all([first, second]);
  assert.equal(h.settings.get("targetGridInches"), 1.2);
  assert.deepEqual(h.sent.filter(message => message.type === "physicalGridResult").map(message => message.requestId), ["first", "second"]);
  const count = h.writes.length;
  await h.api.dispatchPythonMessage({type: "physicalGridSet", requestId: "obsolete",
    settings: {targetGridInches: 1.3}}, {});
  assert.equal(h.writes.length, count);
});

test("disconnect or reconnect during persistence restores settings without applying or replying to the new connection", async () => {
  for (const reconnect of [false, true]) {
    const h = await harness();
    let release;
    const gate = new Promise(resolve => {release = resolve;});
    h.setBehavior(() => gate);
    const pending = h.request({targetGridInches: 1.2, displayDiagonalInches: 65});
    await new Promise(resolve => setImmediate(resolve));
    h.socket.readyState = 3;
    h.api.setSocket(reconnect ? {readyState: 1, send(raw) {h.sent.push(JSON.parse(raw));}} : null);
    release();
    await pending;
    assert.equal(h.settings.get("targetGridInches"), 1);
    assert.equal(h.settings.get("displayDiagonalInches"), 55);
    assert.deepEqual(h.pans, []);
    assert.equal(h.sent.some(message => message.type === "physicalGridResult"), false);
    if (reconnect) {
      assert.deepEqual(h.sent.map(message => message.type), ["physicalGridState"]);
      assert.equal(h.sent[0].settings.targetGridInches, 1);
    }
  }
});

test("a reconnect during asynchronous pan restores the viewport and suppresses the obsolete result", async () => {
  const h = await harness();
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.panBehavior(() => gate);
  const pending = h.request({targetGridInches: 1.2});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.pans.length, 1);
  h.api.setSocket({readyState: 1, send(raw) {h.sent.push(JSON.parse(raw));}});
  release();
  await pending;
  assert.equal(h.settings.get("targetGridInches"), 1);
  assert.equal(h.pans.length, 2);
  assert.equal(h.sent.some(message => message.type === "physicalGridResult"), false);
  assert.equal(h.sent.at(-1).settings.targetGridInches, 1);
});

test("get requests during a settings batch expose only settled values", async () => {
  const h = await harness();
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.setBehavior(() => gate);
  const pending = h.request({targetGridInches: 1.2});
  await new Promise(resolve => setImmediate(resolve));
  await h.api.handlePythonMessage({type: "physicalGridGet"});
  assert.deepEqual(h.sent, []);
  release();
  await pending;
  assert.equal(h.sent.filter(message => message.type === "physicalGridState").length, 1);
  assert.equal(h.sent.at(-1).settings.targetGridInches, 1.2);
});

test("a new scene waits for a pending settings batch before automatically applying its grid", async () => {
  const h = await harness();
  h.settings.set("viewportMarkersEnabled", false);
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  h.setBehavior(() => gate);
  const pending = h.request({targetGridInches: 1.2});
  await new Promise(resolve => setImmediate(resolve));
  h.scene.id = "scene-2";
  h.hooks.get("canvasReady")();
  const sceneReady = h.flushTimers();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.pans.length, 0);
  release();
  await Promise.all([pending, sceneReady]);
  assert.equal(h.settings.get("targetGridInches"), 1);
  assert.equal(h.pans.length, 1);
  assert.ok(Math.abs(h.pans[0].scale - Math.hypot(1920, 1080) / 55 / 100) < 1e-12);
  assert.equal(h.sent.find(message => message.type === "physicalGridResult").success, false);
  assert.equal(h.sent.at(-1).sceneId, "scene-2");
});
