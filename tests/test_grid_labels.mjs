import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness as harness} from "./module_harness.mjs";

const overlay = h => h.elements.get("sarween-grid-labels");
const labels = h => overlay(h)?.draws.filter(draw => draw.kind === "fill") ?? [];
const find = (h, text) => labels(h).find(label => label.text === text);
const enable = h => h.command("settings.set", {settings: {gridLabelsVisible: true}});

test("grid labels default off and use the existing desktop boolean settings contract", async () => {
  const h = await harness();
  assert.equal(h.registrations.get("gridLabelsVisible").default, false);
  assert.equal(h.registrations.get("gridLabelsVisible").scope, "client");
  assert.equal(h.registrations.get("gridLabelsVisible").config, false);
  assert.equal(h.api.vttState().settings.gridLabelsVisible, false);
  assert.equal(overlay(h), undefined);
  assert.equal((await h.command("settings.set", {settings: {gridLabelsVisible: "true"}})).success, false);
  assert.equal(h.writes.length, 0);
  const result = await enable(h);
  assert.equal(result.success, true);
  assert.equal(result.state.settings.gridLabelsVisible, true);
  assert.match(overlay(h).style.cssText, /pointer-events:none/);
  assert.equal(overlay(h)["aria-hidden"], "true");
  assert.ok(h.sent.some(message => message.type === "sceneVisualChanged" && message.reason === "gridLabelsChanged"));
  assert.equal(h.pans.length, 0);
  assert.deepEqual(h.scene.grid, {type: 1, size: 100, distance: 5, shiftX: 12, shiftY: 34});
  assert.equal((await h.command("settings.set", {settings: {gridLabelsVisible: false}})).success, true);
  assert.equal(overlay(h), undefined);
});

test("A1 labels follow scene origins, grid shifts, pan and zoom rather than world pixels", async () => {
  const h = await harness();
  await enable(h);
  assert.deepEqual(find(h, "A1"), {kind: "fill", text: "A1", x: 26, y: 58});
  Object.assign(h.canvas.stage.worldTransform, {a: 2, d: 2, tx: 31, ty: 47});
  h.hooks.get("canvasPan")();
  await h.flushTimers();
  assert.deepEqual(find(h, "A1"), {kind: "fill", text: "A1", x: 83, y: 163});
  assert.deepEqual(find(h, "B1"), {kind: "fill", text: "B1", x: 283, y: 163});
  assert.equal(h.created.filter(element => element.id === "sarween-grid-labels").length, 1,
    "panning should reuse the single canvas overlay");
});

test("labels use Foundry native client conversion when canvas CSS coordinates differ from the stage", async () => {
  const h = await harness();
  h.canvas.canvasCoordinatesFromClient = ({x, y}) => ({x: (x - 30) / 2, y: (y - 40) / 2});
  await enable(h);
  assert.deepEqual(find(h, "A1"), {kind: "fill", text: "A1", x: 82, y: 156});
});

test("rotated and sheared viewports project label anchors through the affine mapping", async () => {
  for (const transform of [
    {a: 0, b: 1, c: -1, d: 0, tx: 800, ty: 0},
    {a: 1, b: .2, c: .4, d: 1, tx: 30, ty: 40},
  ]) {
    const h = await harness();
    Object.assign(h.canvas.stage.worldTransform, transform);
    await enable(h);
    const label = find(h, "A1");
    assert.ok(Math.abs(label.x - (transform.a * 26 + transform.c * 58 + transform.tx)) < 1e-8);
    assert.ok(Math.abs(label.y - (transform.b * 26 + transform.d * 58 + transform.ty)) < 1e-8);
  }
});

test("offscreen labels are culled and extreme zoom-out has bounded drawing work", async () => {
  const h = await harness();
  Object.assign(h.canvas.stage.worldTransform, {tx: -2022, ty: -1054});
  await enable(h);
  assert.equal(find(h, "A1"), undefined);
  assert.ok(find(h, "U11"));
  assert.ok(labels(h).every(label => label.x >= 0 && label.y >= 0
    && label.x <= h.window.innerWidth - 8 && label.y <= h.window.innerHeight - 12));
  h.scene.width = h.scene.height = 1e7;
  Object.assign(h.canvas.stage.worldTransform, {a: .001, d: .001, tx: 0, ty: 0});
  h.hooks.get("canvasPan")();
  await h.flushTimers();
  assert.ok(labels(h).length > 0 && labels(h).length <= 3000);
  h.scene.height = 100;
  h.hooks.get("updateScene")(h.scene, {height: 100});
  await h.flushTimers();
  assert.ok(labels(h).length > 0 && labels(h).length <= 3000,
    "a very long single-row scene must also respect the drawing limit");
});

test("scene geometry, resize, column names and teardown update or remove the overlay", async () => {
  const h = await harness();
  h.canvas.stage.worldTransform.a = h.canvas.stage.worldTransform.d = .5;
  await enable(h);
  assert.ok(find(h, "AA1"));
  h.settings.set("physicalGridAutoApply", false);
  h.window.innerWidth = 800;
  h.window.innerHeight = 600;
  h.window.devicePixelRatio = 2;
  h.windowListeners.get("resize")();
  assert.equal(overlay(h).width, 1600);
  assert.equal(overlay(h).height, 1200);
  assert.deepEqual(overlay(h).transform, [2, 0, 0, 2, 0, 0]);
  h.scene.grid.shiftX = 42;
  h.hooks.get("updateScene")(h.scene, {grid: {shiftX: 42}});
  await h.flushTimers();
  assert.equal(find(h, "A1").x, 28);
  h.hooks.get("canvasTearDown")();
  assert.equal(overlay(h), undefined);
  h.canvas.scene = null;
  h.hooks.get("canvasReady")();
  await h.flushTimers();
  assert.equal(overlay(h), undefined);
});

test("unsupported or singular grids render no stale labels and capture rejects label edits", async () => {
  const h = await harness();
  await enable(h);
  h.scene.grid.type = 2;
  h.hooks.get("updateScene")(h.scene, {grid: {type: 2}});
  await h.flushTimers();
  assert.equal(overlay(h), undefined);
  h.scene.grid.type = 1;
  h.canvas.stage.worldTransform.a = h.canvas.stage.worldTransform.d = 0;
  h.hooks.get("canvasPan")();
  await h.flushTimers();
  assert.equal(overlay(h), undefined);
  h.api.setCapture({phase: "Starting"});
  const before = h.writes.length;
  assert.equal((await h.command("settings.set", {settings: {gridLabelsVisible: false}})).success, false);
  assert.equal(h.writes.length, before);
});
