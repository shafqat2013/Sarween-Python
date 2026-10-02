import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {CAPTURE_MINIS, generateCaptureTargets} from "../capture_logic.mjs";
import * as movementLogic from "../movement_logic.mjs";

const participants = CAPTURE_MINIS.map(mini => ({...mini, tokenId: mini.route}));
const bounds = {firstColumn: 0, lastColumn: 47, firstRow: 0, lastRow: 26};
const targets = generateCaptureTargets(bounds, participants);
assert.equal(targets.length, 30);
assert.equal(new Set(targets.map(target => target.cell)).size, 30);
assert.deepEqual(targets.slice(0, 5).map(target => target.label), ["A0", "B0", "C0", "D0", "E0"]);
assert.deepEqual(targets.slice(-5).map(target => target.label), ["A5", "B5", "C5", "D5", "E5"]);
assert.equal(generateCaptureTargets(bounds, [participants[1]]).length, 6);
assert.throws(() => generateCaptureTargets({firstColumn: 0, lastColumn: 1, firstRow: 0, lastRow: 1}, participants));

const hooks = new Map();
const sent = [];
const snapshotDraws = [];
const sceneTokens = participants.map((mini, index) => ({
  id: mini.tokenId,
  name: mini.tokenName,
  x: index * 100,
  y: 0,
  actor: {system: {attributes: {movement: {walk: 30}}}},
}));
sceneTokens.get = id => sceneTokens.find(token => token.id === id);
let updateCount = 0;
let rejectUpdate = false;
let updateGate = null;
const scene = {id: "new-scene", width: 1200, height: 800, grid: {type: 1, size: 100, distance: 5}, tokens: sceneTokens,
  async updateEmbeddedDocuments(_type, updates, options) {
    if (updateGate) await updateGate;
    if (rejectUpdate) throw new Error("Simulated rejected update");
    updateCount += 1;
    for (const update of updates) {
      const token = sceneTokens.get(update._id);
      Object.assign(token, {x: update.x, y: update.y});
      hooks.get("updateToken")(token, {x: update.x, y: update.y}, options);
    }
  }};
const context = vm.createContext({
  console, crypto: {randomUUID: () => "test"},
  Hooks: {once() {}, on(name, callback) {hooks.set(name, callback);}},
  window: {innerWidth: 1200, innerHeight: 800, addEventListener() {}},
  document: {createElement(tag) {return tag === "canvas"
    ? {width: 0, height: 0, getContext() {return {drawImage(...args) {snapshotDraws.push(args);}};}, toDataURL() {return "data:image/jpeg;base64,abc";}}
    : {style: {}, dataset: {}, appendChild() {}, remove() {}};}, body: {appendChild() {}}},
  game: {settings: {get(_module, key) {return {viewportMarkersEnabled: true, viewportMarkerSize: 120, viewportMarkerInset: 8, physicalTapSelection: true, defaultMovementSpeed: 30}[key];}}},
  canvas: {ready: true, scene, dimensions: {}, app: {renderer: {extract: {canvas() {return {width: 1200, height: 800};}}}}, grid: {measurePath: () => ({distance: 5})}, tokens: {get(id) {return sceneTokens.get(id);}}, stage: {worldTransform: {a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0}}},
  WebSocket: {OPEN: 1},
  setTimeout(callback) {callback(); return 1;}, clearTimeout() {},
});
const source = fs.readFileSync(new URL("../module.js", import.meta.url), "utf8");
const module = new vm.SourceTextModule(source + '\n globalThis.moduleTest = {setSocket(value) {ws = value;}, getMovementState() {return movementState;}, captureRenderedReference, handlePythonMessage, dispatchPythonMessage};', {context});
await module.link(specifier => {
  if (specifier.endsWith("capture_logic.mjs")) {
    return new vm.SyntheticModule(["CAPTURE_MINIS", "generateCaptureTargets"], function () {
      this.setExport("CAPTURE_MINIS", CAPTURE_MINIS);
      this.setExport("generateCaptureTargets", generateCaptureTargets);
    }, {context});
  }
  if (specifier.endsWith("movement_logic.mjs")) {
    const names = Object.keys(movementLogic);
    return new vm.SyntheticModule(names, function () {
      for (const name of names) this.setExport(name, movementLogic[name]);
    }, {context});
  }
  throw new Error(`Unexpected module import: ${specifier}`);
});
await module.evaluate();
// The test exercises real hooks, not a string search for their implementation.
context.moduleTest.setSocket({readyState: 1, send(raw) {sent.push(JSON.parse(raw));}});
await hooks.get("canvasReady")();
assert.equal(sent.find(message => message.type === "sceneInfo").sceneId, "new-scene");
assert.deepEqual(sent.filter(message => ["sceneInfo", "viewTransform"].includes(message.type)).map(message => message.type), ["sceneInfo", "viewTransform"]);
sent.length = 0;
hooks.get("updateScene")(scene, {grid: {size: 100}});
assert.equal(sent[0].type, "sceneInfo");
assert.equal(sent[0].miniBindings.blue, "B");
sent.length = 0;
hooks.get("controlToken")(sceneTokens[0], true);
assert.equal(sent[0].type, "movementSelection");
assert.equal(sent[0].miniId, "red10");
hooks.get("updateToken")(sceneTokens[0], {x: 100, y: 0});
assert.equal(context.moduleTest.getMovementState().usedFeet, 5);
hooks.get("controlToken")(sceneTokens[0], false);
assert.equal(context.moduleTest.getMovementState(), null);
sent.length = 0;
context.moduleTest.captureRenderedReference();
assert.equal(sent[0].type, "renderedReference");
assert.equal(sent[0].sceneId, "new-scene");
assert.equal(snapshotDraws.length, 1);
assert.deepEqual(snapshotDraws[0].slice(1), [28, 28, 1144, 744, 0, 0, 640, 360]);
console.log("PASS: five-mini routes and scene-switch hooks");

const socket = {readyState: 1, send(raw) {sent.push(JSON.parse(raw));}};
context.moduleTest.setSocket(socket);
hooks.get("controlToken")(sceneTokens[0], true);
const command = {type: "moveToken", commandId: "move-1", sceneId: scene.id,
  tokenId: "A", miniId: "red10", x: 100, y: 0};
sent.length = 0;
await context.moduleTest.handlePythonMessage(command);
assert.equal(updateCount, 1);
assert.equal(sent[0].type, "tokenMoveApplied");
assert.equal(sent[0].commandId, "move-1");
assert.equal(context.moduleTest.getMovementState().usedFeet, 5);
await context.moduleTest.handlePythonMessage(command);
assert.equal(updateCount, 1, "Retry must not repeat the update or animation");
assert.equal(context.moduleTest.getMovementState().usedFeet, 5);
rejectUpdate = true;
sent.length = 0;
const second = {...command, commandId: "move-2", x: 200};
await context.moduleTest.handlePythonMessage(second);
assert.equal(sent[0].type, "tokenMoveError");
assert.equal(sent[0].commandId, second.commandId);
rejectUpdate = false;
sent.length = 0;
await context.moduleTest.handlePythonMessage(second);
assert.equal(sent[0].type, "tokenMoveApplied");
assert.equal(sceneTokens[0].x, 200);
sent.length = 0;
await context.moduleTest.handlePythonMessage({...command, commandId: "stale-scene", sceneId: "old-scene"});
assert.equal(sent[0].type, "tokenMoveError");
assert.equal(sceneTokens[0].x, 200);
sent.length = 0;
await context.moduleTest.handlePythonMessage({...command, commandId: "deleted", tokenId: "missing"});
assert.equal(sent[0].code, "tokenMissing");

let release;
updateGate = new Promise(resolve => {release = resolve;});
const firstPending = context.moduleTest.dispatchPythonMessage({...command, commandId: "ordered-1", x: 300}, socket);
const lastPending = context.moduleTest.dispatchPythonMessage({...command, commandId: "ordered-2", x: 400}, socket);
await Promise.resolve();
assert.equal(sceneTokens[0].x, 200);
release();
await Promise.all([firstPending, lastPending]);
assert.equal(sceneTokens[0].x, 400, "Delayed updates must finish in wire order");
await context.moduleTest.dispatchPythonMessage({...command, x: 500}, {});
assert.equal(sceneTokens[0].x, 400, "Discard commands queued on an obsolete connection");
console.log("PASS: move acknowledgements, idempotent retry, rejection, scene safety and serialized delivery");

updateGate = null;
const budgetBeforePan = context.moduleTest.getMovementState().usedFeet;
const segmentsBeforePan = context.moduleTest.getMovementState().segments.length;
const pan = {...command, commandId: "pan", x: 700, y: 200, source: "viewportTransform"};
await context.moduleTest.handlePythonMessage(pan);
assert.equal(context.moduleTest.getMovementState().usedFeet, budgetBeforePan);
assert.equal(context.moduleTest.getMovementState().segments.length, segmentsBeforePan);
assert.equal(context.moduleTest.getMovementState().points.at(-1).x, 700);
assert.equal(context.moduleTest.getMovementState().points[0].x, 300);
const updatesAfterPan = updateCount;
await context.moduleTest.handlePythonMessage(pan);
assert.equal(updateCount, updatesAfterPan);
assert.equal(context.moduleTest.getMovementState().usedFeet, budgetBeforePan);
await context.moduleTest.handlePythonMessage({...pan, commandId: "after-pan", x: 800, source: "detection"});
assert.equal(context.moduleTest.getMovementState().usedFeet, budgetBeforePan + 5);
const stateBeforeRejection = context.moduleTest.getMovementState();
rejectUpdate = true;
await context.moduleTest.handlePythonMessage({...pan, commandId: "rejected-pan", x: 900});
assert.equal(context.moduleTest.getMovementState(), stateBeforeRejection);
rejectUpdate = false;
console.log("PASS: viewport remaps preserve budget/history, retries are harmless, real moves still count");
