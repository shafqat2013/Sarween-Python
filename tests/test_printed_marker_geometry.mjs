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
  assert.equal(sent.registration.right, 1252);
  assert.equal(sent.registration.bottom, 692);
  h.canvas.ready = false;
  assert.equal(h.api.buildViewTransformPayload(), null);
});
