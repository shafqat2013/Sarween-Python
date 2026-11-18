import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness} from "./module_harness.mjs";

const sceneValues = {tokenVision: true, fogExploration: true, globalIllumination: false};
const sightValues = {tokenId: "wizard", enabled: true, range: 30};
const turn = () => new Promise(resolve => setImmediate(resolve));

// Simulate the public Document.update contract: dotted paths are merged into
// documents and update hooks receive the original changes, as Foundry 13 does.
function applyUpdate(document, changes) {
  for (const [path, value] of Object.entries(changes)) {
    const parts = path.split(".");
    let target = document;
    for (const part of parts.slice(0, -1)) target = target[part] ??= {};
    target[parts.at(-1)] = value;
  }
}

async function harness() {
  const h = await createModuleHarness();
  h.scene.grid.units = "ft";
  h.scene.fog = {exploration: false, overlay: "fog.webp", colors: {explored: "#555555"}};
  h.scene.environment = {globalLight: {enabled: true, darkness: {min: 0, max: 0.7}}};
  h.sceneWrites = [];
  h.tokenWrites = [];
  h.scene.canUserModify = () => true;
  h.scene.update = async (changes, options) => {
    h.sceneWrites.push({changes: {...changes}, options});
    await h.beforeSceneUpdate?.();
    if (!h.ignoreSceneUpdate) applyUpdate(h.scene, changes);
    h.hooks.get("updateScene")?.(h.scene, changes, options);
    return h.scene;
  };
  h.token = h.addToken({id: "wizard", name: "Wizard", sight: {enabled: false, range: 60, angle: 270, visionMode: "darkvision"}});
  h.token.canUserModify = () => true;
  h.token.update = async (changes, options) => {
    h.tokenWrites.push({changes: {...changes}, options});
    await h.beforeTokenUpdate?.();
    if (!h.ignoreTokenUpdate) applyUpdate(h.token, changes);
    h.hooks.get("updateToken")?.(h.token, changes, options);
    return h.token;
  };
  return h;
}

test("vision state reports real scene settings, scene units, token sight and current selection", async () => {
  const h = await harness();
  const state = h.api.vttState();
  assert.deepEqual(JSON.parse(JSON.stringify(state.vision)), {tokenVision: true, fogExploration: false,
    globalIllumination: true, selectedTokenId: null, distanceUnits: "ft"});
  assert.deepEqual(JSON.parse(JSON.stringify(state.tokens[0].vision)), {enabled: false, range: 60});
  for (const action of ["vision.scene.set", "vision.token.set", "vision.token.select"]) {
    assert.ok(state.capabilities.includes(action));
  }
  await h.api.handlePythonMessage({type: "vttGet"});
  h.canvas.tokens.get("wizard").control();
  assert.equal(h.sent.findLast(message => message.type === "vttState").vision.selectedTokenId, "wizard");
  h.canvas.tokens.releaseAll();
  assert.equal(h.sent.findLast(message => message.type === "vttState").vision.selectedTokenId, null);
});

test("scene controls use Foundry 13 paths, preserve lighting and fog details and return confirmed settings", async () => {
  const h = await harness();
  const result = await h.command("vision.scene.set", sceneValues);
  assert.equal(result.success, true);
  assert.deepEqual(h.sceneWrites[0].changes,
    {tokenVision: true, "fog.exploration": true, "environment.globalLight.enabled": false});
  assert.equal(result.state.vision.fogExploration, true);
  assert.equal(result.state.vision.globalIllumination, false);
  assert.equal(h.scene.fog.overlay, "fog.webp");
  assert.equal(h.scene.fog.colors.explored, "#555555");
  assert.equal(h.scene.environment.globalLight.darkness.max, 0.7);
  await h.flushTimers();
  assert.ok(h.sent.some(message => message.type === "sceneVisualChanged" && message.reason === "sceneSettingsChanged"));
});

test("token controls edit the placed token, preserve other sight fields and allow zero and fractional range", async () => {
  const h = await harness();
  const result = await h.command("vision.token.set", sightValues);
  assert.equal(result.success, true);
  assert.deepEqual(h.tokenWrites[0].changes, {"sight.enabled": true, "sight.range": 30});
  assert.deepEqual(result.state.tokens[0].vision, {enabled: true, range: 30});
  assert.equal(h.token.sight.angle, 270);
  assert.equal(h.token.sight.visionMode, "darkvision");
  assert.equal((await h.command("vision.token.set", {...sightValues, enabled: false, range: 0})).success, true);
  assert.equal(h.token.sight.enabled, false);
  assert.equal(h.token.sight.range, 0);
  assert.equal((await h.command("vision.token.set", {...sightValues, range: 30.5})).success, true);
  await h.flushTimers();
  assert.ok(h.sent.some(message => message.type === "sceneVisualChanged" && message.tokenIds.includes("wizard")));
});

test("the TV view can select an unassigned token and clear selection without changing mini links or sight", async () => {
  const h = await harness();
  h.addToken({id: "other", name: "Other"});
  h.canvas.tokens.get("other").control();
  const selected = await h.command("vision.token.select", {tokenId: "wizard"});
  assert.equal(selected.success, true);
  assert.equal(selected.state.vision.selectedTokenId, "wizard");
  assert.deepEqual(selected.state.miniBindings, {});
  assert.equal(selected.state.movement, null);
  assert.equal(h.token.sight.enabled, false);
  assert.equal(h.tokenWrites.length, 0);
  const cleared = await h.command("vision.token.select", {tokenId: null});
  assert.equal(cleared.success, true);
  assert.equal(cleared.state.vision.selectedTokenId, null);
  assert.equal(h.canvas.tokens.controlled.length, 0);
});

test("invalid input, non-GM, stale or absent scene, capture and permissions cause no document writes", async () => {
  const h = await harness();
  for (const values of [{...sceneValues, tokenVision: 1}, {...sceneValues, fogExploration: undefined},
    {...sceneValues, globalIllumination: null}, {...sceneValues, extra: true}]) {
    assert.equal((await h.command("vision.scene.set", values)).success, false);
  }
  for (const values of [{...sightValues, tokenId: "missing"}, {...sightValues, tokenId: null},
    {...sightValues, enabled: 1}, {...sightValues, range: -1}, {...sightValues, range: Infinity},
    {...sightValues, range: NaN}, {...sightValues, range: "30"}, {...sightValues, range: true},
    {...sightValues, extra: true}]) {
    assert.equal((await h.command("vision.token.set", values)).success, false);
  }
  assert.equal((await h.command("vision.token.select", {tokenId: "missing"})).success, false);
  for (const [action, values] of [["vision.scene.set", sceneValues], ["vision.token.set", sightValues],
    ["vision.token.select", {tokenId: "wizard"}]]) {
    assert.equal((await h.command(action, values, {sceneId: "old"})).success, false);
    assert.equal((await h.command(action, values, {sceneId: undefined})).success, false);
    h.game.user.isGM = false;
    assert.equal((await h.command(action, values)).success, false);
    h.game.user.isGM = true;
    h.api.setCapture({recording: true, phase: "placement"});
    assert.equal((await h.command(action, values)).success, false);
    h.api.setCapture(null);
    h.scene.canUserModify = () => false;
    assert.equal((await h.command(action, values)).success, false);
    h.scene.canUserModify = () => true;
    h.canvas.ready = false;
    assert.equal((await h.command(action, values)).success, false);
    h.canvas.ready = true;
  }
  h.token.canUserModify = () => false;
  assert.equal((await h.command("vision.token.set", sightValues)).success, false);
  assert.equal(h.sceneWrites.length, 0);
  assert.equal(h.tokenWrites.length, 0);
});

test("rejected or unconfirmed updates fail visibly with actual state, without claiming the requested settings", async () => {
  for (const kind of ["scene", "token"]) {
    const h = await harness();
    const action = kind === "scene" ? "vision.scene.set" : "vision.token.set";
    const values = kind === "scene" ? sceneValues : sightValues;
    h[kind === "scene" ? "beforeSceneUpdate" : "beforeTokenUpdate"] = async () => {throw new Error("Document rejected");};
    const failed = await h.command(action, values);
    assert.equal(failed.success, false);
    assert.match(failed.message, /Document rejected/);
    h[kind === "scene" ? "beforeSceneUpdate" : "beforeTokenUpdate"] = null;
    h[kind === "scene" ? "ignoreSceneUpdate" : "ignoreTokenUpdate"] = true;
    const ignored = await h.command(action, values);
    assert.equal(ignored.success, false);
    assert.match(ignored.message, /did not confirm/);
    assert.equal(ignored.state.vision.fogExploration, false);
    assert.equal(ignored.state.tokens[0].vision.enabled, false);
  }
});

test("scene changes during persistence produce no successful stale-scene acknowledgement", async () => {
  for (const [action, values, before] of [["vision.scene.set", sceneValues, "beforeSceneUpdate"],
    ["vision.token.set", sightValues, "beforeTokenUpdate"]]) {
    const h = await harness();
    h[before] = async () => {h.canvas.scene = {...h.scene, id: "new-scene"};};
    const result = await h.command(action, values);
    assert.equal(result.success, false);
    assert.match(result.message, /scene changed/);
    assert.equal(result.state.sceneId, "new-scene");
  }
});

test("connection replacement during persistence cannot deliver a successful result to the replacement receiver", async () => {
  const h = await harness();
  let release;
  h.beforeSceneUpdate = () => new Promise(resolve => {release = resolve;});
  const pending = h.command("vision.scene.set", sceneValues);
  await turn();
  const replacement = new h.socket.constructor("replacement");
  h.api.setSocket(replacement);
  release();
  assert.equal(await pending, undefined);
  assert.equal(h.scene.fog.exploration, true, "the completed write is retained and the next refresh reports it");
  assert.equal(h.sent.some(message => message.type === "vttResult"), false);
});

test("selection refuses a token not rendered on the TV or a denied control change", async () => {
  const h = await harness();
  const placeable = h.canvas.tokens.get("wizard");
  placeable.control = () => false;
  assert.equal((await h.command("vision.token.select", {tokenId: "wizard"})).success, false);
  const original = h.canvas.tokens.get;
  h.canvas.tokens.get = () => undefined;
  assert.equal((await h.command("vision.token.select", {tokenId: "wizard"})).success, false);
  h.canvas.tokens.get = original;
  assert.equal(h.canvas.tokens.controlled.length, 0);
});
