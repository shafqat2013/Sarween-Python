import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness} from "./module_harness.mjs";

test("hiding LCD markers keeps the current viewport geometry available for paper registration", async () => {
  const h = await createModuleHarness();
  const before = h.api.buildViewTransformPayload();
  assert.equal(before.viewportMarkersVisible, true);
  const result = await h.command("markers.set", {enabled: false});
  assert.equal(result.success, true);
  const hidden = h.api.buildViewTransformPayload();
  assert.equal(hidden.viewportMarkersVisible, false);
  assert.deepEqual(hidden.registration, before.registration);
  assert.deepEqual(hidden.clientToCanvasTransform, before.clientToCanvasTransform);
  h.canvas.stage.worldTransform.tx = 125;
  h.canvas.stage.worldTransform.ty = -60;
  h.api.sendViewTransform();
  const sent = h.sent.filter(message => message.type === "viewTransform").at(-1);
  assert.equal(sent.canvasTransform.tx, 125);
  assert.equal(sent.canvasTransform.ty, -60);
  assert.equal(sent.viewportMarkersVisible, false);
});

test("paper registration receives resized viewport dimensions while LCD markers are hidden", async () => {
  const h = await createModuleHarness();
  h.settings.set("viewportMarkersEnabled", false);
  h.window.innerWidth = 1280;
  h.window.innerHeight = 720;
  h.api.sendViewTransform();
  const sent = h.sent.at(-1);
  assert.equal(sent.viewportWidth, 1280);
  assert.equal(sent.viewportHeight, 720);
  assert.equal(sent.registration.right, 1260);
  assert.equal(sent.registration.bottom, 700);
  h.canvas.ready = false;
  assert.equal(h.api.buildViewTransformPayload(), null);
});

test("margin changes preserve the pattern and update rendered plate, registration and metadata together", async () => {
  const h = await createModuleHarness();
  let result = await h.command("settings.set", {settings: {viewportMarkerPatternSize: 48, viewportMarkerMargin: 0}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerSize, 48);
  const marker = () => h.elements.get("sarween-viewport-markers").children[0];
  const pattern = () => marker().children[0];
  assert.match(marker().style.cssText, /width: 48px;/);
  assert.match(pattern().style.cssText, /width: 48px;/);
  assert.match(pattern().style.cssText, /left: 0px;/);
  assert.match(pattern().style.cssText, /overflow: hidden;/);
  assert.match(pattern().children[0].style.cssText, /width: 72px;/);
  assert.match(pattern().children[0].style.cssText, /left: -12px;/);
  const noMargin = h.api.buildViewTransformPayload();
  assert.equal(noMargin.registration.left, 8);
  assert.equal(noMargin.registration.right, 1912);
  result = await h.command("settings.set", {settings: {viewportMarkerMargin: 12}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerPatternSize, 48);
  assert.equal(result.state.settings.viewportMarkerSize, 72);
  assert.match(marker().style.cssText, /width: 72px;/);
  assert.match(pattern().style.cssText, /width: 48px;/);
  assert.match(pattern().style.cssText, /left: 12px;/);
  const withMargin = h.api.buildViewTransformPayload();
  assert.equal(withMargin.registration.left, 20);
  assert.equal(withMargin.registration.right, 1900);
  h.addToken({id: "red", name: "Red"});
  assert.equal((await h.command("capture.start", {
    participants: [{miniId: "red10", tokenId: "red"}], scenario: "fog-off"})).success, true);
  const start = h.sent.find(message => message.type === "captureControl" && message.action === "start");
  assert.deepEqual(start.markerSettings, {size: 72, patternSize: 48, margin: 12, inset: 8});
});

test("legacy size commands and explicit new fields migrate without the 72 pixel floor", async () => {
  const h = await createModuleHarness();
  let result = await h.command("settings.set", {settings: {viewportMarkerSize: 60}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerPatternSize, 40);
  assert.equal(result.state.settings.viewportMarkerMargin, 10);
  assert.equal(result.state.settings.viewportMarkerSize, 60);
  result = await h.command("settings.set", {settings: {viewportMarkerSize: 120,
    viewportMarkerPatternSize: 24, viewportMarkerMargin: 2}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerPatternSize, 24);
  assert.equal(result.state.settings.viewportMarkerMargin, 2);
  assert.equal(result.state.settings.viewportMarkerSize, 28);
  result = await h.command("settings.set", {settings: {viewportMarkerPatternSize: 1, viewportMarkerMargin: 0}});
  assert.equal(result.success, true);
  assert.equal(result.state.settings.viewportMarkerSize, 1);
});

test("failed marker edit rolls back pattern, margin and geometry atomically", async () => {
  const h = await createModuleHarness();
  const before = h.api.buildViewTransformPayload();
  let fail = true;
  h.setBehavior(key => {if (key === "viewportMarkerMargin" && fail) {fail = false; throw new Error("Store unavailable");}});
  const result = await h.command("settings.set", {settings: {viewportMarkerPatternSize: 30, viewportMarkerMargin: 0}});
  assert.equal(result.success, false);
  assert.equal(h.settings.get("viewportMarkerPatternSize"), 0);
  assert.equal(h.settings.get("viewportMarkerMargin"), -1);
  assert.deepEqual(h.api.buildViewTransformPayload().registration, before.registration);
});
