import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness as harness} from "./module_harness.mjs";

const changes = h => h.sent.filter(message => message.type === "sceneVisualChanged");
const select = (h, id) => h.canvas.tokens.get(id).control();
const refresh = h => {h.hooks.get("sightRefresh")(); h.hooks.get("lightingRefresh")();};

test("manual selection and release notify after perception settles, including unmapped tokens", async () => {
  for (const name of ["Red", "Unassigned NPC"]) {
    const h = await harness();
    h.addToken({id: "token", name});
    select(h, "token");
    assert.equal(changes(h).length, 0);
    refresh(h);
    await h.advanceTimers(99);
    assert.equal(changes(h).length, 0);
    await h.advanceTimers(1);
    assert.equal(changes(h).length, 1);
    assert.equal(changes(h)[0].reason, "tokenControlled");
    assert.equal(changes(h)[0].sceneId, h.scene.id);
    assert.deepEqual(changes(h)[0].tokenIds, ["token"]);
    h.canvas.tokens.releaseAll();
    refresh(h);
    await h.advanceTimers(100);
    assert.equal(changes(h).length, 2);
    assert.equal(changes(h)[1].reason, "tokenReleased");
  }
});

test("control and perception bursts coalesce, while refresh animation alone is inert", async () => {
  const h = await harness();
  h.addToken({id: "red", name: "Red"});
  h.addToken({id: "blue", name: "Blue"});
  select(h, "red");
  select(h, "blue"); // Release Red then control Blue in the same render burst.
  for (let i = 0; i < 4; i++) {refresh(h); await h.advanceTimers(20);}
  assert.equal(changes(h).length, 0);
  await h.advanceTimers(80);
  assert.equal(changes(h).length, 1);
  assert.deepEqual(new Set(changes(h)[0].tokenIds), new Set(["red", "blue"]));
  assert.deepEqual(new Set(changes(h)[0].reasons), new Set(["tokenControlled", "tokenReleased"]));
  for (let i = 0; i < 20; i++) {refresh(h); await h.advanceTimers(50);}
  assert.equal(changes(h).length, 1, "animated refreshes without a new mutation must not reseed");
});

test("fallback has a fixed deadline even when refresh never arrives or keeps animating", async () => {
  for (const animated of [false, true]) {
    const h = await harness();
    h.addToken({id: "red"}); select(h, "red");
    if (animated) for (let i = 0; i < 6; i++) {refresh(h); await h.advanceTimers(50);}
    else await h.advanceTimers(300);
    assert.equal(changes(h).length, 0);
    await h.advanceTimers(49);
    assert.equal(changes(h).length, 0);
    await h.advanceTimers(1);
    assert.equal(changes(h).length, 1);
    await h.advanceTimers(500);
    assert.equal(changes(h).length, 1, "the other timer must not send a duplicate");
  }
});

test("manual coordinates, sight and light changes notify; unrelated and off-scene edits do not", async () => {
  for (const edit of [{x: 300, y: 500}, {sight: {range: 90}}, {"light.bright": 20},
    {light: {dim: 30}}, {rotation: 90}, {hidden: true}]) {
    const h = await harness(); const token = h.addToken({id: "red"});
    await token.update(edit); refresh(h); await h.advanceTimers(100);
    assert.equal(changes(h).length, 1, JSON.stringify(edit));
    assert.equal(changes(h)[0].reason, "tokenUpdated");
  }
  const h = await harness(); const token = h.addToken({id: "red"});
  await token.update({name: "Renamed", "flags.example.note": "text"});
  refresh(h); await h.advanceTimers(500);
  assert.equal(changes(h).length, 0);
  h.hooks.get("updateToken")({id: "elsewhere", parent: {id: "other-scene"}}, {x: 500}, {});
  refresh(h); await h.advanceTimers(500);
  assert.equal(changes(h).length, 0);
});

test("token creation and deletion invalidate only the active scene, including unmapped light sources", async () => {
  for (const hook of ["createToken", "deleteToken"]) {
    const h = await harness();
    const token = {id: "npc", name: "Unmapped light source", parent: h.scene};
    h.hooks.get(hook)(token);
    refresh(h); await h.advanceTimers(100);
    assert.equal(changes(h).length, 1);
    assert.equal(changes(h)[0].reason, hook === "createToken" ? "tokenCreated" : "tokenDeleted");
    assert.deepEqual(changes(h)[0].tokenIds, ["npc"]);
    h.hooks.get(hook)({...token, parent: {id: "other-scene"}});
    refresh(h); await h.advanceTimers(500);
    assert.equal(changes(h).length, 1, "documents on other scenes must not reseed the active camera background");
  }
});

test("scene vision, environment, global light and fog edits invalidate only the active scene", async () => {
  for (const edit of [{tokenVision: false}, {environment: {darknessLevel: .5}},
    {"environment.darknessLevel": .7}, {"environment.globalLight.enabled": true},
    {globalLight: true}, {darkness: .5}, {fogExploration: false}, {fogOverlay: "fog.png"},
    {"fog.exploration": false}, {background: {src: "new-map.png"}}]) {
    const h = await harness();
    h.hooks.get("updateScene")({...h.scene, id: "other-scene"}, edit);
    refresh(h); await h.advanceTimers(500);
    assert.equal(changes(h).length, 0);
    h.hooks.get("updateScene")(h.scene, edit);
    refresh(h); await h.advanceTimers(100);
    assert.equal(changes(h).length, 1, JSON.stringify(edit));
    assert.equal(changes(h)[0].reason, "sceneSettingsChanged");
    assert.equal(changes(h)[0].sceneId, h.scene.id);
  }
  const h = await harness();
  h.hooks.get("updateScene")(h.scene, {name: "Renamed", "flags.example.note": "text"});
  for (let i = 0; i < 10; i++) {refresh(h); await h.advanceTimers(50);}
  assert.equal(changes(h).length, 0, "metadata edits and refresh animation are not visual mutations");
});

test("a new mutation after refresh waits for its own draw or the bounded fallback", async () => {
  const h = await harness(); const token = h.addToken({id: "red"});
  select(h, "red"); refresh(h); await h.advanceTimers(50);
  await token.update({x: 250});
  await h.advanceTimers(100);
  assert.equal(changes(h).length, 0, "the earlier selection refresh cannot flush the later move");
  refresh(h); await h.advanceTimers(100);
  assert.equal(changes(h).length, 1);
});

test("stale scene, closed socket, reconnect and teardown callbacks cannot invalidate a new context", async () => {
  for (const invalidate of [
    h => {h.canvas.scene = {...h.scene, id: "scene-2"};},
    h => {h.socket.readyState = 3;},
    h => h.api.setSocket({readyState: 1, send(raw) {h.sent.push(JSON.parse(raw));}}),
    h => h.api.connectToPython(),
    h => h.hooks.get("canvasTearDown")(),
  ]) {
    const h = await harness(); h.addToken({id: "red"}); select(h, "red"); refresh(h);
    const staleCallbacks = [...h.timers.values()];
    invalidate(h);
    for (const callback of staleCallbacks) await callback();
    await h.advanceTimers(500);
    assert.equal(changes(h).length, 0);
  }
});

test("move ACK remains immediate, visual changes coalesce, and idempotent retries do not reseed", async () => {
  const h = await harness(); const token = h.addToken({id: "red"});
  const request = {type: "moveToken", commandId: "move-1", sceneId: h.scene.id,
    tokenId: token.id, miniId: "red10", x: 400, y: 500, source: "detection"};
  await h.api.dispatchPythonMessage(request, h.socket);
  assert.equal(h.sent.find(message => message.commandId === "move-1").type, "tokenMoveApplied");
  assert.equal(changes(h).length, 0);
  refresh(h); await h.advanceTimers(100);
  assert.equal(changes(h).length, 1);
  assert.equal(changes(h)[0].reason, "sarweenTokenMove");
  await h.api.dispatchPythonMessage({...request, commandId: "retry"}, h.socket);
  assert.equal(h.sent.find(message => message.commandId === "retry").type, "tokenMoveApplied");
  refresh(h); await h.advanceTimers(500);
  assert.equal(changes(h).length, 1);
});

test("async replies stay bound to their originating socket without undoing a completed move", async () => {
  for (const reject of [false, true]) {
    const h = await harness(); const token = h.addToken({id: "red"});
    let release, entered;
    const hold = new Promise(resolve => {release = resolve;});
    const started = new Promise(resolve => {entered = resolve;});
    h.updateBehavior(async () => {entered(); await hold; if (reject) throw new Error("Move rejected");});
    const request = {type: "moveToken", commandId: "old-command", sceneId: h.scene.id,
      tokenId: token.id, miniId: "red10", x: 400, y: 500, source: "detection"};
    const pending = h.api.dispatchPythonMessage(request, h.socket);
    await started;
    const replacementMessages = [];
    h.api.setSocket({readyState: 1, send(raw) {replacementMessages.push(JSON.parse(raw));}});
    release(); await pending;
    assert.equal(h.sent.some(message => message.commandId === "old-command"), false);
    assert.equal(replacementMessages.some(message => message.commandId === "old-command"), false);
    assert.deepEqual([token.x, token.y], reject ? [100, 200] : [400, 500]);
    refresh(h); await h.advanceTimers(100);
    assert.equal(replacementMessages.filter(message => message.type === "sceneVisualChanged").length,
      reject ? 0 : 1, "the actual same-scene mutation may still notify the new connection");
  }
});

test("guided capture keeps its existing explicit visual notifications", async () => {
  const h = await harness(); const token = h.addToken({id: "red", name: "Red"});
  select(h, "red");
  const result = await h.command("capture.start", {participants: [{miniId: "red10", tokenId: "red"}],
    scenario: "fog-off", notes: ""});
  assert.equal(result.success, true);
  await h.advanceTimers(500);
  h.canvas.tokens.releaseAll();
  await token.update({x: 500}); refresh(h);
  await h.advanceTimers(500);
  assert.equal(changes(h).length, 0, "capture must not receive the new generic invalidations");
  const sessionId = h.api.getCapture().sessionId;
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "recording", sessionId}, h.socket);
  await h.advanceTime(5000);
  await h.command("capture.confirm");
  await h.api.dispatchPythonMessage({type: "captureStatus", state: "confirmed", sessionId, index: 0}, h.socket);
  assert.equal(changes(h).length, 1);
  assert.equal(changes(h)[0].reason, "guidedTokenMove");
});
